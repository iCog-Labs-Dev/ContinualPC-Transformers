"""Tests for the online per-step gates (tiny two-block BERT, offline)."""

import json
import os
import shutil
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from fabricpc.core.types import GraphParams, NodeParams
from fabricpc.experiments.domain_forgetting import runner
from fabricpc.experiments.domain_forgetting.artifacts import file_sha256, load_npz
from fabricpc.experiments.domain_forgetting.data import load_domain_cache
from fabricpc.experiments.domain_forgetting.frontier import describe_gate
from fabricpc.experiments.domain_forgetting.gates import (
    GATE_MAP_BATCH,
    SCORE_FLOOR,
    ema_update,
    gate_periods,
    map_batches,
    map_multipliers,
    online_multipliers,
    permute_multipliers,
    probe_offset,
    probe_seed,
    shuffle_rng,
    stream_windows,
    surprise_multipliers,
    type_scores,
)
from fabricpc.experiments.domain_forgetting.runner import (
    RUN_ARTIFACT_PATTERNS,
    build_structure,
    build_inference,
    fixed_mlm_batches,
    load_base_model,
    run_domain_forgetting,
)
from fabricpc.training.plasticity_gates import (
    module_multipliers,
    scale_module_updates_traced,
    set_module_multipliers,
)
from tests.test_domain_forgetting_data import build_tiny_caches, tiny_tokenizer
from tests.test_domain_forgetting_runner import (
    DOMAINS,
    tiny_config,
    write_tiny_bert_checkpoint,
)

MAP_WINDOWS = 12
UPDATES = 3
ONLINE_KEYS = ("gate_interval", "gate_ema", "gate_variant")
BLOCKS = ("L0_mha", "L0_mlp1", "L0_mlp2", "L1_mha", "L1_mlp1", "L1_mlp2")


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def _hashes(result):
    return result["summary"]["stage_params_sha256"]


def _schedule(config, stage):
    return _read_jsonl(Path(config.output_dir) / f"gate_schedule_stage_{stage}.jsonl")


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    directory = tmp_path_factory.mktemp("online_gates")
    tokenizer = tiny_tokenizer(directory)
    checkpoint = write_tiny_bert_checkpoint(directory, tokenizer, depth=2)
    build_tiny_caches(directory, tokenizer)
    return checkpoint, directory / "data"


def _online(setup, output, gate="online_surprise", **overrides):
    checkpoint, data_dir = setup
    values = {"gate": gate, "gate_map_windows": MAP_WINDOWS, **overrides}
    return tiny_config(checkpoint, data_dir, output, **values)


@pytest.fixture(scope="module")
def surprise_run(setup, tmp_path_factory):
    """``online_surprise`` refreshed after every update, with the applied state.

    Records the multipliers each update's optimizer state held and counts
    the traces of the jitted update.
    """
    config = _online(
        setup, tmp_path_factory.mktemp("surprise") / "run", gate_interval=1
    )
    held: dict[tuple[int, int], dict[str, float]] = {}
    traces = []
    original_train = runner.train_stage
    original_update = runner._apply_optimizer_update_when_masked

    def capturing(*args, stage, gate_fn=None, **kwargs):
        wrapped = None
        if gate_fn is not None:

            def wrapped(update, params, opt_state):
                new_state = gate_fn(update, params, opt_state)
                held[(stage, update + 1)] = module_multipliers(new_state)
                return new_state

        return original_train(*args, stage=stage, gate_fn=wrapped, **kwargs)

    def counting(**kwargs):
        traces.append(1)
        return original_update(**kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(runner, "train_stage", capturing)
        patch.setattr(runner, "_apply_optimizer_update_when_masked", counting)
        result = run_domain_forgetting(config, verbose=False)
    return config, result, held, len(traces)


def _graph_params():
    return GraphParams(
        nodes={
            "a": NodeParams(
                weights={"W": jnp.array([[1.5, -2.0], [0.25, 3.0]])}, biases={}
            ),
            "b": NodeParams(weights={"W": jnp.array([4.0, -1.0])}, biases={}),
            "c": NodeParams(weights={}, biases={}),
        }
    )


def test_traced_transform_scales_by_state_without_retracing():
    grads = _graph_params()
    optimizer = optax.chain(optax.sgd(1.0), scale_module_updates_traced({"a": 0.5}))
    state = optimizer.init(grads)
    assert module_multipliers(state) == {"a": 0.5, "b": 1.0, "c": 1.0}
    traces = []

    def step(g, s):
        traces.append(1)
        return optimizer.update(g, s, g)

    jitted = jax.jit(step)
    updates, state = jitted(grads, state)
    np.testing.assert_array_equal(
        updates.nodes["a"].weights["W"], -0.5 * grads.nodes["a"].weights["W"]
    )
    np.testing.assert_array_equal(
        updates.nodes["b"].weights["W"], -grads.nodes["b"].weights["W"]
    )
    state = set_module_multipliers(state, {"a": 0.25, "b": 2.0})
    assert module_multipliers(state) == {"a": 0.25, "b": 2.0, "c": 1.0}
    updates, state = jitted(grads, state)
    np.testing.assert_array_equal(
        updates.nodes["a"].weights["W"], -0.25 * grads.nodes["a"].weights["W"]
    )
    np.testing.assert_array_equal(
        updates.nodes["b"].weights["W"], -2.0 * grads.nodes["b"].weights["W"]
    )
    assert len(traces) == 1
    with pytest.raises(ValueError, match="finite and positive"):
        set_module_multipliers(state, {"a": 0.0})
    with pytest.raises(ValueError, match="finite and positive"):
        scale_module_updates_traced({"a": -1.0})
    with pytest.raises(ValueError, match="unknown nodes \\['z'\\]"):
        set_module_multipliers(state, {"z": 1.0})
    with pytest.raises(ValueError, match="unknown nodes"):
        scale_module_updates_traced({"z": 1.0}).init(grads)
    with pytest.raises(ValueError, match="no ModuleMultiplierState"):
        set_module_multipliers(optax.sgd(1.0).init(grads), {"a": 1.0})


def test_online_gate_known_answers():
    smoothed = {"L0_mha": 2.0, "L0_mlp1": 4.0, "L1_mha": 1.0, "L1_mlp1": 8.0}
    raw = {"L0_mha": 4.0, "L0_mlp1": 0.0, "L1_mha": 1.0, "L1_mlp1": 2.0}
    assert ema_update(smoothed, raw, 0.25) == {
        "L0_mha": 2.5,
        "L0_mlp1": 3.0,
        "L1_mha": 1.0,
        "L1_mlp1": 6.5,
    }
    assert ema_update(smoothed, raw, 1.0) == raw
    tau = {"L0_mha": 2.0, "L0_mlp1": 1.0, "L1_mha": 3.0, "L1_mlp1": 8.0}
    assert surprise_multipliers(smoothed, tau) == {
        "L0_mha": 0.5,
        "L0_mlp1": 0.8,
        "L1_mha": 0.25,
        "L1_mlp1": 0.5,
    }
    assert surprise_multipliers(smoothed, tau, inverse=True) == {
        "L0_mha": 0.5,
        "L0_mlp1": 0.2,
        "L1_mha": 0.75,
        "L1_mlp1": 0.5,
    }
    kwargs = dict(beta=2.0, rng=None)
    protect, permutation = online_multipliers(
        smoothed, tau, gate="online_protect", variant="live", **kwargs
    )
    assert protect == map_multipliers(smoothed, 2.0)
    assert permutation is None
    inverse, _ = online_multipliers(
        smoothed, tau, gate="online_protect", variant="inverse", **kwargs
    )
    assert inverse == map_multipliers(smoothed, 2.0, inverse=True)
    surprise, _ = online_multipliers(
        smoothed, tau, gate="online_surprise", variant="live", **kwargs
    )
    assert surprise == surprise_multipliers(smoothed, tau)
    onset, _ = online_multipliers(
        tau, tau, gate="online_surprise", variant="live", **kwargs
    )
    assert set(onset.values()) == {0.5}
    typed, _ = online_multipliers(
        smoothed, tau, gate="online_protect", variant="type", **kwargs
    )
    assert typed == map_multipliers(type_scores(smoothed), 2.0)
    assert typed["L0_mha"] == typed["L1_mha"]
    typed, _ = online_multipliers(
        smoothed, tau, gate="online_surprise", variant="type", **kwargs
    )
    assert typed == surprise_multipliers(type_scores(smoothed), type_scores(tau))
    typed_onset, _ = online_multipliers(
        tau, tau, gate="online_surprise", variant="type", **kwargs
    )
    assert set(typed_onset.values()) == {0.5}
    shuffled, permutation = online_multipliers(
        smoothed,
        tau,
        gate="online_surprise",
        variant="shuffle",
        beta=2.0,
        rng=shuffle_rng(3, 0, 2, 1),
    )
    expected = permute_multipliers(
        surprise_multipliers(smoothed, tau),
        list(smoothed),
        np.random.default_rng([3, 0, 2, 1]),
    )
    assert (shuffled, permutation) == expected
    assert sorted(shuffled.values()) == sorted(surprise.values())
    onset_rng = shuffle_rng(3, 0, 2, -1)
    assert onset_rng.permutation(6).tolist() == (
        np.random.default_rng([3, 0, 2]).permutation(6).tolist()
    )
    floored, _ = online_multipliers(
        {**smoothed, "L1_mha": 0.0},
        tau,
        gate="online_surprise",
        variant="live",
        **kwargs,
    )
    assert floored["L1_mha"] == SCORE_FLOOR / (SCORE_FLOOR + 3.0)
    assert all(v > 0 for v in floored.values())
    with pytest.raises(ValueError, match="needs an rng"):
        online_multipliers(
            smoothed, tau, gate="online_surprise", variant="shuffle", beta=1.0
        )


def test_periods_probe_offsets_and_seeds():
    assert gate_periods(7, 3) == [(-1, 1, 3), (0, 4, 6), (1, 7, 7)]
    assert gate_periods(6, 3) == [(-1, 1, 3), (0, 4, 6)]
    assert gate_periods(3, 50) == [(-1, 1, 3)]
    assert gate_periods(3, 1) == [(-1, 1, 1), (0, 2, 2), (1, 3, 3)]
    assert [probe_offset(256, r, 1000) for r in range(3)] == [256, 264, 272]
    assert probe_offset(12, 2, 20) == (12 + 16) % 20
    seeds = [probe_seed(0, stage, r) for stage in (2, 3) for r in range(4)]
    assert len(set(seeds)) == len(seeds)
    assert all(0 <= s < 2**31 for s in seeds)
    assert probe_seed(0, 2, 1) == probe_seed(0, 2, 1)
    assert probe_seed(1, 2, 1) != probe_seed(0, 2, 1)


def test_probe_windows_are_the_stream_windows_from_the_offset(setup, tmp_path):
    checkpoint, data_dir = setup
    config = tiny_config(checkpoint, data_dir, tmp_path)
    base = load_base_model(config)
    seq_len = int(base.model_config["seq_len"])
    stream = load_domain_cache(
        data_dir,
        DOMAINS[1],
        tokenizer_json=base.tokenizer_json,
        seq_len=seq_len,
        tokens_per_domain=config.tokens_per_domain,
    ).train
    run = replace(config, mode="backprop", dropout_rate=0.0)
    structure = build_structure(
        base.model_config, run, build_inference(run, base.model_config)[0]
    )
    total = stream_windows(stream, seq_len)
    windows, _ = fixed_mlm_batches(
        stream,
        structure,
        base.special_ids,
        num_batches=total,
        batch_size=1,
        mask_prob=0.0,
        seed=0,
    )
    offset = total - 3
    batches = map_batches(
        stream,
        structure,
        base.special_ids,
        num_windows=GATE_MAP_BATCH,
        mask_prob=0.0,
        seed=0,
        offset=offset,
    )
    assert len(batches) == 1
    for key in ("x", "attention_mask"):
        expected = np.concatenate(
            [
                np.asarray(windows[(offset + k) % total][key])
                for k in range(GATE_MAP_BATCH)
            ]
        )
        np.testing.assert_array_equal(np.asarray(batches[0][key]), expected)
    first = map_batches(
        stream,
        structure,
        base.special_ids,
        num_windows=GATE_MAP_BATCH,
        mask_prob=0.3,
        seed=5,
    )
    shifted = map_batches(
        stream,
        structure,
        base.special_ids,
        num_windows=GATE_MAP_BATCH,
        mask_prob=0.3,
        seed=5,
        offset=total,
    )
    for key in first[0]:
        np.testing.assert_array_equal(
            np.asarray(first[0][key]), np.asarray(shifted[0][key])
        )


def test_protect_without_refresh_is_the_map_gate(setup, tmp_path):
    checkpoint, data_dir = setup
    static = tiny_config(
        checkpoint,
        data_dir,
        tmp_path / "map",
        gate="map",
        gate_beta=2.0,
        gate_map_windows=MAP_WINDOWS,
    )
    online = _online(
        setup,
        tmp_path / "online",
        gate="online_protect",
        gate_beta=2.0,
        gate_interval=UPDATES,
    )
    online_result = run_domain_forgetting(online, verbose=False)
    static_records = _read_jsonl(tmp_path / "online" / "gates.jsonl")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(runner, "scale_module_updates", scale_module_updates_traced)
        traced_result = run_domain_forgetting(
            replace(static, output_dir=str(tmp_path / "map_traced")), verbose=False
        )
    assert _hashes(online_result) == _hashes(traced_result)
    map_result = run_domain_forgetting(static, verbose=False)
    map_records = _read_jsonl(tmp_path / "map" / "gates.jsonl")
    assert static_records[1]["map"] == map_records[1]["map"]
    assert static_records[1]["multipliers"] == map_records[1]["multipliers"]
    assert _hashes(online_result)[:2] == _hashes(map_result)[:2]
    for stage in (2, 3):
        left = load_npz(tmp_path / "map" / f"stage_{stage}.npz")
        right = load_npz(tmp_path / "online" / f"stage_{stage}.npz")
        for key in left:
            np.testing.assert_allclose(left[key], right[key], rtol=1e-4, atol=1e-5)
        schedule = _schedule(online, stage)
        assert len(schedule) == 1
        assert schedule[0]["refresh"] == -1
        assert schedule[0]["multipliers"] == {
            m: float(np.float32(v))
            for m, v in static_records[stage - 1]["multipliers"].items()
        }


def test_small_interval_changes_multipliers_every_period(surprise_run):
    config, _, _, _ = surprise_run
    output = Path(config.output_dir)
    base = load_base_model(config)
    seq_len = int(base.model_config["seq_len"])
    streams = {
        name: load_domain_cache(
            config.data_dir,
            name,
            tokenizer_json=base.tokenizer_json,
            seq_len=seq_len,
            tokens_per_domain=config.tokens_per_domain,
        ).train
        for name in config.domains
    }
    assert not (output / "gate_schedule_stage_1.jsonl").exists()
    records = _read_jsonl(output / "gates.jsonl")
    for stage in (2, 3):
        schedule = _schedule(config, stage)
        assert [
            (r["refresh"], r["first_update"], r["last_update"]) for r in schedule
        ] == (gate_periods(UPDATES, 1))
        assert set(schedule[0]["multipliers"][m] for m in BLOCKS) == {0.5}
        assert schedule[0]["energies"] is None
        assert schedule[0]["smoothed"] == records[stage - 1]["map"]
        blocks = [tuple(r["multipliers"][m] for m in BLOCKS) for r in schedule]
        assert len(set(blocks)) == len(blocks)
        for record in schedule:
            assert all(
                record["multipliers"][m] == 1.0
                for m in record["multipliers"]
                if m not in BLOCKS
            )
        previous = schedule[0]["smoothed"]
        windows = stream_windows(streams[config.domains[stage - 1]], seq_len)
        for refresh, record in enumerate(schedule[1:]):
            assert record["probe_offset"] == probe_offset(MAP_WINDOWS, refresh, windows)
            assert record["probe_seed"] == probe_seed(config.seed, stage, refresh)
            expected = ema_update(previous, record["energies"], config.gate_ema)
            assert record["smoothed"] == expected
            multipliers = surprise_multipliers(expected, records[stage - 1]["map"])
            for m in BLOCKS:
                assert record["multipliers"][m] == float(np.float32(multipliers[m]))
            previous = expected
        summary = records[stage - 1]["schedule"]
        assert summary["periods"] == len(schedule)
        path = output / summary["file"]
        assert summary["sha256"] == file_sha256(path)
    assert "gate_schedule_stage_*.jsonl" in RUN_ARTIFACT_PATTERNS


def test_schedule_holds_the_multipliers_the_optimizer_applied(surprise_run):
    config, _, held, _ = surprise_run
    for stage in (2, 3):
        for record in _schedule(config, stage):
            for update in range(record["first_update"], record["last_update"] + 1):
                state = held[(stage, update)]
                assert {m: state[m] for m in record["multipliers"]} == (
                    record["multipliers"]
                )


def test_jitted_update_compiles_once_per_stage(surprise_run):
    config, _, held, traces = surprise_run
    assert len(_schedule(config, 2)) == UPDATES
    assert traces == len(DOMAINS)


def test_replay_reproduces_the_source_run(surprise_run, setup, tmp_path):
    config, result, _, _ = surprise_run
    replay = _online(
        setup,
        tmp_path / "replay",
        gate_interval=1,
        gate_variant="replay",
        gate_source=(config.output_dir,),
    )
    replayed = run_domain_forgetting(replay, verbose=False)
    assert _hashes(replayed) == _hashes(result)
    for stage in (2, 3):
        source = _schedule(config, stage)
        mine = _schedule(replay, stage)
        assert [r["multipliers"] for r in mine] == [r["multipliers"] for r in source]
        assert all(r["energies"] is None and r["smoothed"] is None for r in mine)
    run_config = json.loads((tmp_path / "replay" / "run_config.json").read_text())
    assert "map" not in run_config["gate"]
    assert run_config["gate"]["sources"][config.output_dir]["schedule_sha256"] == {
        str(s): file_sha256(Path(config.output_dir) / f"gate_schedule_stage_{s}.jsonl")
        for s in (2, 3)
    }


def test_replay_refuses_mismatched_sources(surprise_run, setup, tmp_path):
    config, _, _, _ = surprise_run
    base = dict(gate_interval=1, gate_variant="replay")
    for overrides, match in (
        ({"gate_interval": 2}, "gate_interval 1 differs from 2"),
        ({"gate": "online_protect"}, "replay needs a 'online_protect' run"),
        ({"domains": ("reviews", "news", "wiki")}, "domain order"),
        ({"batch_size": 2, "eval_batch_size": 2}, "updates per stage"),
        ({"gate_ema": 0.5}, "gate_ema 0.3 differs from 0.5"),
        ({"gate_map_windows": MAP_WINDOWS + 1}, "gate_map_windows"),
    ):
        bad = _online(
            setup,
            tmp_path / "bad",
            **{**base, **overrides, "gate_source": (config.output_dir,)},
        )
        with pytest.raises(ValueError, match=match):
            run_domain_forgetting(bad, verbose=False, overwrite=True)
    shuffled = tmp_path / "shuffled"
    shutil.copytree(config.output_dir, shuffled)
    run_config = json.loads((shuffled / "run_config.json").read_text())
    run_config["config"]["gate_variant"] = "shuffle"
    (shuffled / "run_config.json").write_text(json.dumps(run_config))
    bad = _online(setup, tmp_path / "bad", **base, gate_source=(str(shuffled),))
    with pytest.raises(ValueError, match="needs a live source"):
        run_domain_forgetting(bad, verbose=False, overwrite=True)
    run_config["config"]["gate_variant"] = "live"
    run_config["source"]["params_sha256"] = "0" * 64
    (shuffled / "run_config.json").write_text(json.dumps(run_config))
    with pytest.raises(ValueError, match="not trained from the run's checkpoint"):
        run_domain_forgetting(bad, verbose=False, overwrite=True)
    incomplete = tmp_path / "incomplete"
    shutil.copytree(config.output_dir, incomplete)
    lines = (incomplete / "gate_schedule_stage_3.jsonl").read_text().splitlines()
    (incomplete / "gate_schedule_stage_3.jsonl").write_text(lines[0] + "\n")
    bad = _online(setup, tmp_path / "bad", **base, gate_source=(str(incomplete),))
    with pytest.raises(ValueError, match="SHA-256 differs"):
        run_domain_forgetting(bad, verbose=False, overwrite=True)
    records = _read_jsonl(incomplete / "gates.jsonl")
    records[2]["schedule"]["sha256"] = file_sha256(
        incomplete / "gate_schedule_stage_3.jsonl"
    )
    (incomplete / "gates.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records)
    )
    with pytest.raises(ValueError, match="periods"):
        run_domain_forgetting(bad, verbose=False, overwrite=True)
    (incomplete / "gate_schedule_stage_3.jsonl").unlink()
    with pytest.raises(ValueError, match="no complete gate schedule"):
        run_domain_forgetting(bad, verbose=False, overwrite=True)


def test_resume_checks_the_stage_schedules(surprise_run, tmp_path, monkeypatch):
    config, result, _, _ = surprise_run
    output = tmp_path / "run"
    shutil.copytree(config.output_dir, output)
    copied = replace(config, output_dir=str(output))
    trained = []
    original = runner.train_stage

    def counting(*args, stage, **kwargs):
        trained.append(stage)
        return original(*args, stage=stage, **kwargs)

    monkeypatch.setattr(runner, "train_stage", counting)
    run_domain_forgetting(copied, resume=True, verbose=False)
    assert trained == []
    path = output / "gate_schedule_stage_2.jsonl"
    path.write_text(path.read_text() + "\n")
    resumed = run_domain_forgetting(copied, resume=True, verbose=False)
    assert trained == [2, 3]
    assert _hashes(resumed) == _hashes(result)
    assert [
        r["schedule"]["sha256"] for r in _read_jsonl(output / "gates.jsonl")[1:]
    ] == [file_sha256(output / f"gate_schedule_stage_{s}.jsonl") for s in (2, 3)]


def test_online_gate_validation_and_identity(setup, tmp_path):
    checkpoint, data_dir = setup
    config = tiny_config(checkpoint, data_dir, tmp_path)
    online = replace(config, gate="online_protect")
    online.validate()
    for overrides, match in (
        ({"gate_variant": "bogus"}, "gate_variant must be one of"),
        ({"gate_seed": 1}, "gate_seed applies only to gate_variant 'shuffle'"),
        ({"gate_source": ("x",)}, "gate_source applies only to gate_variant"),
        ({"gate_variant": "replay"}, "exactly one gate_source"),
        ({"gate_variant": "replay", "gate_source": ("a", "b")}, "exactly one"),
        ({"gate_interval": 0}, "gate_interval must be at least 1"),
        ({"gate_ema": 0.0}, "gate_ema must be in \\(0, 1\\]"),
        ({"gate_ema": 1.5}, "gate_ema must be in"),
        ({"gate_ema": float("nan")}, "gate_ema must be in"),
        ({"gate": "online_surprise", "gate_beta": 2.0}, "gate_beta do not apply"),
    ):
        with pytest.raises(ValueError, match=match):
            replace(online, **overrides).validate()
    for gate, overrides in (
        ("map", {"gate_interval": 10}),
        ("uniform", {"gate_ema": 0.5}),
        ("none", {"gate_variant": "type"}),
    ):
        with pytest.raises(ValueError, match="do not apply"):
            replace(config, gate=gate, **overrides).validate()
    for overrides in (
        {"gate_variant": "shuffle", "gate_seed": 2},
        {"gate_variant": "replay", "gate_source": ("a",)},
        {"gate_variant": "type", "gate_interval": 1, "gate_ema": 1.0},
        {"gate": "online_surprise", "gate_variant": "inverse"},
    ):
        replace(online, **overrides).validate()
    for gate in ("map", "uniform", "freeze", "oracle"):
        identity = replace(config, gate=gate).identity()
        assert not set(ONLINE_KEYS) & set(identity)
    assert not set(ONLINE_KEYS) & set(config.identity())
    assert "gate_scale" not in online.identity()
    assert {k: online.identity()[k] for k in ONLINE_KEYS} == {
        "gate_interval": 50,
        "gate_ema": 0.3,
        "gate_variant": "live",
    }


def test_frontier_labels_online_gates():
    def label(**config):
        return describe_gate({"lr": 1e-4, **config})[2]

    assert label(gate="online_protect", gate_beta=2.0, gate_variant="live") == (
        "online_protect_b2"
    )
    assert label(gate="online_surprise", gate_beta=1.0, gate_variant="live") == (
        "online_surprise"
    )
    assert label(gate="online_protect", gate_beta=2.0, gate_variant="replay") == (
        "online_protect_b2_replay"
    )
    assert label(gate="online_surprise", gate_variant="shuffle") == (
        "online_surprise_shuffle"
    )
    assert describe_gate(
        {"lr": 5e-5, "gate": "online_surprise", "gate_variant": "type"}
    )[2] == ("online_surprise_type@lr5e-05")
    assert label(gate="map", gate_beta=2.0) == "map_b2"


def test_cli_parses_online_gate_flags(tmp_path):
    from examples import mlm_domain_forgetting as cli

    common = ["--checkpoint", "c", "--output_dir", str(tmp_path)]
    args = cli.build_parser().parse_args(
        common
        + [
            "--gate",
            "online_surprise",
            "--gate_interval",
            "25",
            "--gate_ema",
            "0.5",
            "--gate_variant",
            "replay",
            "--gate_source",
            "runs/a",
        ]
    )
    config = cli.config_from_args(args)
    assert (config.gate_interval, config.gate_ema, config.gate_variant) == (
        25,
        0.5,
        "replay",
    )
    config.validate()
    default = cli.config_from_args(cli.build_parser().parse_args(common))
    assert not set(ONLINE_KEYS) & set(default.identity())
