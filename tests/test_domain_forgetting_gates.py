"""Tests for stage-level per-module update gates (tiny two-block BERT, offline)."""

import json
import os
import shutil
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fabricpc.core.types import GraphParams, NodeParams
from fabricpc.experiments.domain_forgetting import runner
from fabricpc.experiments.domain_forgetting.artifacts import (
    atomic_write_json,
    load_npz,
)
from fabricpc.experiments.domain_forgetting.data import load_domain_cache
from fabricpc.experiments.domain_forgetting.error_maps import (
    ErrorMapConfig,
    aggregate_map,
    build_map_structure,
    node_energies,
    SPIKE_ONLY,
)
from fabricpc.experiments.domain_forgetting.gates import (
    GATE_MAP_BATCH,
    SCORE_FLOOR,
    freeze_groups,
    map_multipliers,
    resolve_modules,
)
from fabricpc.experiments.domain_forgetting.localization import (
    compute_rollback,
    load_run,
    module_names,
)
from fabricpc.experiments.domain_forgetting.runner import (
    RUN_ARTIFACT_FILES,
    build_stage_optimizer,
    fixed_mlm_batches,
    load_base_model,
    make_gradient_fn,
    run_domain_forgetting,
    train_stage,
)
from fabricpc.training.plasticity_gates import scale_module_updates
from tests.test_domain_forgetting_data import build_tiny_caches, tiny_tokenizer
from tests.test_domain_forgetting_runner import (
    DOMAINS,
    tiny_config,
    write_tiny_bert_checkpoint,
)

MAP_WINDOWS = 12
GATE_KEYS = (
    "gate",
    "gate_beta",
    "gate_modules",
    "gate_source",
    "gate_map_windows",
    "gate_seed",
)


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _bits_equal(left, right):
    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    assert len(left_leaves) == len(right_leaves)
    return all(
        np.asarray(a).dtype == np.asarray(b).dtype
        and np.asarray(a).tobytes() == np.asarray(b).tobytes()
        for a, b in zip(left_leaves, right_leaves)
    )


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    directory = tmp_path_factory.mktemp("gates")
    tokenizer = tiny_tokenizer(directory)
    checkpoint = write_tiny_bert_checkpoint(directory, tokenizer, depth=2)
    build_tiny_caches(directory, tokenizer)
    return checkpoint, directory / "data"


@pytest.fixture(scope="module")
def runs(setup, tmp_path_factory):
    checkpoint, data_dir = setup
    root = tmp_path_factory.mktemp("gate_runs")
    configs = {
        "none": tiny_config(checkpoint, data_dir, root / "none"),
        "map": tiny_config(
            checkpoint,
            data_dir,
            root / "map",
            gate="map",
            gate_map_windows=MAP_WINDOWS,
        ),
        "random": tiny_config(
            checkpoint,
            data_dir,
            root / "random",
            gate="random",
            gate_map_windows=MAP_WINDOWS,
            gate_seed=3,
        ),
        "inverse": tiny_config(
            checkpoint,
            data_dir,
            root / "inverse",
            gate="inverse",
            gate_map_windows=MAP_WINDOWS,
        ),
        "type": tiny_config(
            checkpoint,
            data_dir,
            root / "type",
            gate="type",
            gate_map_windows=MAP_WINDOWS,
            gate_beta=2.0,
        ),
        "freeze": tiny_config(
            checkpoint,
            data_dir,
            root / "freeze",
            gate="freeze",
            gate_modules=("mlp", "token_embeddings"),
        ),
    }
    return {
        name: (config, run_domain_forgetting(config, verbose=False))
        for name, config in configs.items()
    }


def _graph_params():
    return GraphParams(
        nodes={
            "a": NodeParams(
                weights={"W": jnp.array([[1.5, -2.0], [0.25, 3.0]])},
                biases={"b": jnp.array([0.5, -0.125])},
            ),
            "b": NodeParams(weights={"W": jnp.array([4.0, -1.0])}, biases={}),
            "c": NodeParams(weights={}, biases={}),
        }
    )


def test_scale_module_updates_scales_named_nodes_only():
    updates = _graph_params()
    ones = scale_module_updates({"a": 1.0, "b": 1.0})
    state = ones.init(updates)
    scaled, _ = ones.update(updates, state, updates)
    assert _bits_equal(scaled, updates)
    half = scale_module_updates({"a": 0.5, "b": 0.0})
    scaled, _ = half.update(updates, half.init(updates), updates)
    np.testing.assert_array_equal(
        scaled.nodes["a"].weights["W"], 0.5 * updates.nodes["a"].weights["W"]
    )
    np.testing.assert_array_equal(scaled.nodes["b"].weights["W"], [0.0, 0.0])
    untouched = scale_module_updates({"a": 0.0})
    scaled, _ = untouched.update(updates, untouched.init(updates), updates)
    assert _bits_equal(scaled.nodes["b"], updates.nodes["b"])
    unknown = scale_module_updates({"a": 1.0, "missing": 0.5})
    with pytest.raises(ValueError, match="unknown nodes \\['missing'\\]"):
        unknown.init(updates)
    with pytest.raises(ValueError, match="unknown nodes"):
        unknown.update(updates, None, updates)


def test_all_ones_stage_optimizer_is_bit_identical_to_ungated(setup, tmp_path):
    checkpoint, data_dir = setup
    config = tiny_config(checkpoint, data_dir, tmp_path / "run")
    params = load_base_model(config).params
    ones = {name: 1.0 for name in params.nodes}
    plain, _ = build_stage_optimizer(config, 5)
    gated, info = build_stage_optimizer(config, 5, ones)
    assert info["module_multipliers"] == ones
    rng = np.random.default_rng(0)
    left, right = params, params
    left_state, right_state = plain.init(left), gated.init(right)
    for _ in range(4):
        grads = jax.tree_util.tree_map(
            lambda x: jnp.asarray(rng.standard_normal(x.shape), x.dtype), params
        )
        updates, left_state = plain.update(grads, left_state, left)
        left = jax.tree_util.tree_map(jnp.add, left, updates)
        updates, right_state = gated.update(grads, right_state, right)
        right = jax.tree_util.tree_map(jnp.add, right, updates)
    assert _bits_equal(left, right)


def test_train_stage_with_all_ones_equals_ungated_bit_for_bit(setup, tmp_path):
    checkpoint, data_dir = setup
    config = tiny_config(checkpoint, data_dir, tmp_path / "run")
    base = load_base_model(config)
    stream = load_domain_cache(
        data_dir,
        DOMAINS[1],
        tokenizer_json=base.tokenizer_json,
        seq_len=int(base.model_config["seq_len"]),
        tokens_per_domain=config.tokens_per_domain,
    ).train
    gradient_fn = make_gradient_fn("backprop", base.structure)
    kwargs = dict(
        config=config,
        stage=2,
        special_ids=base.special_ids,
        gradient_fn=gradient_fn,
        verbose=False,
    )
    plain, plain_exposure, _ = train_stage(
        base.params, base.structure, stream, **kwargs
    )
    ones = {name: 1.0 for name in base.params.nodes}
    gated, exposure, _ = train_stage(
        base.params, base.structure, stream, multipliers=ones, **kwargs
    )
    assert _bits_equal(plain, gated)
    assert "module_multipliers" not in plain_exposure["optimizer"]
    assert exposure["optimizer"]["module_multipliers"] == ones


def test_map_multipliers_are_capped_and_decrease_with_the_map():
    scores = {"a": 1.0, "b": 2.0, "c": 4.0, "d": 8.0, "e": 16.0}
    gates = map_multipliers(scores, 1.0)
    assert gates == {"a": 1.0, "b": 1.0, "c": 1.0, "d": 0.5, "e": 0.25}
    steep = map_multipliers(scores, 2.0)
    assert steep["d"] == pytest.approx(0.25) and steep["e"] == pytest.approx(1 / 16)
    inverse = map_multipliers(scores, 1.0, inverse=True)
    assert inverse == {"a": 0.25, "b": 0.5, "c": 1.0, "d": 1.0, "e": 1.0}
    even = map_multipliers({"a": 1.0, "b": 3.0}, 1.0)
    assert even == {"a": 1.0, "b": pytest.approx(2.0 / 3.0)}
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite and positive; not at \\['b'\\]"):
            map_multipliers({"a": 1.0, "b": bad}, 1.0)


def test_freeze_groups_and_module_resolution():
    modules = [
        "token_embeddings",
        "position_embeddings",
        "L0_mha",
        "L0_mlp1",
        "L0_mlp2",
        "L1_mha",
        "L1_mlp1",
        "L1_mlp2",
        "mlm_transform",
        "mlm_head",
    ]
    groups = freeze_groups(modules)
    assert groups["mlp"] == ["L0_mlp1", "L0_mlp2", "L1_mlp1", "L1_mlp2"]
    assert groups["blocks_0-0"] == ["L0_mha", "L0_mlp1", "L0_mlp2"]
    assert len(groups["blocks"]) == 6
    assert resolve_modules(modules, ["L1_mha", "embeddings"]) == [
        "token_embeddings",
        "position_embeddings",
        "L1_mha",
    ]
    with pytest.raises(ValueError, match="unknown module or group 'L9_mha'"):
        resolve_modules(modules, ["L9_mha"])


def test_stage_one_is_ungated_and_matches_the_ungated_run(runs):
    reference = runs["none"][1]["summary"]["stage_params_sha256"]
    for name in ("map", "random", "freeze"):
        config, result = runs[name]
        records = _read_jsonl(Path(config.output_dir) / "gates.jsonl")
        assert [r["stage"] for r in records] == [1, 2, 3]
        assert [r["domain"] for r in records] == list(DOMAINS)
        first = records[0]
        assert not first["gated"]
        assert set(first["multipliers"].values()) == {1.0}
        assert first["map"] is None and first["permutation"] is None
        assert all(r["gated"] for r in records[1:])
        assert all(r["gate"] == name for r in records)
        assert result["summary"]["stage_params_sha256"][:2] == reference[:2]
        assert result["summary"]["stage_params_sha256"][2] != reference[2]
        assert "module_multipliers" not in result["exposure"][DOMAINS[0]]["optimizer"]
        assert result["exposure"][DOMAINS[1]]["optimizer"]["module_multipliers"] == (
            records[1]["multipliers"]
        )


def test_map_gate_multipliers_follow_the_onset_map(runs):
    config, _ = runs["map"]
    records = _read_jsonl(Path(config.output_dir) / "gates.jsonl")
    for record in records[1:]:
        values = record["map"]
        blocks = sorted(values)
        assert blocks == sorted(m for m in record["multipliers"] if m.startswith("L"))
        assert len(blocks) == 6
        assert record["map_windows"] == MAP_WINDOWS
        assert record["map_seed"] == config.seed + record["stage"]
        median = float(np.median(list(values.values())))
        for module, multiplier in record["multipliers"].items():
            assert 0.0 < multiplier <= 1.0
            if module not in values:
                assert multiplier == 1.0
            elif values[module] <= median:
                assert multiplier == 1.0
            else:
                assert multiplier == pytest.approx(median / values[module])
        order = sorted(blocks, key=values.get)
        gates = [record["multipliers"][m] for m in order]
        assert all(a >= b for a, b in zip(gates, gates[1:]))
        assert gates[-1] < 1.0


def test_onset_map_equals_the_spike_only_error_map(runs, setup):
    config, _ = runs["map"]
    _, data_dir = setup
    record = _read_jsonl(Path(config.output_dir) / "gates.jsonl")[1]
    context = load_run(config.output_dir)
    structure = build_map_structure(
        context, SPIKE_ONLY, ErrorMapConfig(output_energy_scale=1.0)
    )
    stream = load_domain_cache(
        data_dir,
        record["domain"],
        tokenizer_json=load_base_model(config).tokenizer_json,
        seq_len=int(context.model_config["seq_len"]),
        tokens_per_domain=config.tokens_per_domain,
    ).train
    windows, _ = fixed_mlm_batches(
        stream,
        structure,
        load_base_model(config).special_ids,
        num_batches=MAP_WINDOWS,
        batch_size=1,
        mask_prob=config.mask_prob,
        seed=record["map_seed"],
    )
    batches = [
        {
            key: jnp.concatenate([w[key] for w in windows[i : i + GATE_MAP_BATCH]])
            for key in windows[0]
        }
        for i in range(0, MAP_WINDOWS, GATE_MAP_BATCH)
    ]
    assert [len(b["x"]) for b in batches] == [GATE_MAP_BATCH, MAP_WINDOWS - 8]
    params = context.params(record["stage"] - 1)
    results = [jax.device_get(node_energies(params, structure, b)) for b in batches]
    modules = module_names(context.structure, context.snapshots[0])
    aggregated = aggregate_map(results, modules, 1.0)
    for module, value in record["map"].items():
        expected = aggregated["modules"][module]["forward_error_energy_sum_ce"]
        assert value == pytest.approx(expected, rel=1e-3)


def test_random_gate_permutes_the_map_multipliers(runs):
    map_records = _read_jsonl(Path(runs["map"][0].output_dir) / "gates.jsonl")
    config, _ = runs["random"]
    records = _read_jsonl(Path(config.output_dir) / "gates.jsonl")
    assert records[1]["map"] == pytest.approx(map_records[1]["map"])
    run_config = json.loads((Path(config.output_dir) / "run_config.json").read_text())
    blocks = run_config["gate"]["block_modules"]
    assert sorted(blocks) == sorted(records[1]["map"])
    for record in records[1:]:
        unpermuted = map_multipliers(record["map"], config.gate_beta)
        assert sorted(record["multipliers"][m] for m in blocks) == pytest.approx(
            sorted(unpermuted.values())
        )
        source = record["permutation"]
        rng = np.random.default_rng([config.gate_seed, config.seed, record["stage"]])
        assert source == [blocks[j] for j in rng.permutation(len(blocks))]
        for target, origin in zip(blocks, source):
            assert record["multipliers"][target] == unpermuted[origin]
    assert any(
        any(r["multipliers"][m] != map_multipliers(r["map"], 1.0)[m] for m in blocks)
        for r in records[1:]
    )
    assert sorted(records[1]["multipliers"].values()) == pytest.approx(
        sorted(map_records[1]["multipliers"].values())
    )


def test_inverse_gate_shrinks_modules_below_the_median(runs):
    config, _ = runs["inverse"]
    records = _read_jsonl(Path(config.output_dir) / "gates.jsonl")
    for record in records[1:]:
        values = record["map"]
        median = float(np.median(list(values.values())))
        for module, multiplier in record["multipliers"].items():
            if module not in values or values[module] >= median:
                assert multiplier == 1.0
            else:
                assert multiplier == pytest.approx(values[module] / median)
        assert min(record["multipliers"].values()) < 1.0


def test_type_gate_uses_module_type_means(runs):
    config, _ = runs["type"]
    records = _read_jsonl(Path(config.output_dir) / "gates.jsonl")
    for record in records[1:]:
        values, scores = record["map"], record["scores"]
        types = {m.split("_", 1)[1] for m in values}
        assert types == {"mha", "mlp1", "mlp2"}
        means = {
            t: np.mean([v for m, v in values.items() if m.endswith(f"_{t}")])
            for t in types
        }
        for module, score in scores.items():
            assert score == pytest.approx(means[module.split("_", 1)[1]])
        expected = map_multipliers(scores, 2.0)
        for module in values:
            assert record["multipliers"][module] == pytest.approx(expected[module])
        for t in types:
            assert len({record["multipliers"][m] for m in values if m.endswith(t)}) == 1
        order = sorted(types, key=means.get)
        gates = [record["multipliers"][f"L0_{t}"] for t in order]
        assert all(a >= b for a, b in zip(gates, gates[1:]))
        assert gates[0] == 1.0 and gates[-1] < 1.0


def test_freeze_zeroes_frozen_gradients_before_the_clip(setup, tmp_path):
    checkpoint, data_dir = setup
    config = tiny_config(checkpoint, data_dir, tmp_path / "run", grad_clip=1.0)
    params = _graph_params()
    grads = jax.tree_util.tree_map(lambda x: 100.0 * jnp.ones_like(x), params)
    grads.nodes["b"].weights["W"] = jnp.array([0.3, -0.2])
    masked = GraphParams(
        nodes={
            **grads.nodes,
            "a": jax.tree_util.tree_map(jnp.zeros_like, grads.nodes["a"]),
        }
    )
    gated, _ = build_stage_optimizer(config, 5, {"a": 0.0})
    plain, _ = build_stage_optimizer(config, 5)
    updates, _ = gated.update(grads, gated.init(params), params)
    reference, _ = plain.update(masked, plain.init(params), params)
    unmasked, _ = plain.update(grads, plain.init(params), params)
    assert _bits_equal(updates.nodes["b"], reference.nodes["b"])
    assert not _bits_equal(updates.nodes["b"], unmasked.nodes["b"])
    assert all(
        not np.any(np.asarray(leaf))
        for leaf in jax.tree_util.tree_leaves(updates.nodes["a"])
    )


def test_freeze_keeps_frozen_modules_bit_identical(runs):
    config, _ = runs["freeze"]
    output = Path(config.output_dir)
    run_config = json.loads((output / "run_config.json").read_text())
    frozen = run_config["gate"]["frozen_modules"]
    assert frozen == [
        "token_embeddings",
        "L0_mlp1",
        "L0_mlp2",
        "L1_mlp1",
        "L1_mlp2",
    ]
    snapshots = [load_npz(output / f"stage_{s}.npz") for s in range(4)]
    for key in snapshots[0]:
        module = key.split("/", 1)[0]
        if module in frozen:
            assert not np.array_equal(snapshots[0][key], snapshots[1][key]), key
            for stage in (2, 3):
                assert snapshots[stage][key].tobytes() == snapshots[1][key].tobytes()
        elif module in ("L0_mha", "mlm_head"):
            assert not np.array_equal(snapshots[1][key], snapshots[3][key]), key
    records = _read_jsonl(output / "gates.jsonl")
    for record in records[1:]:
        assert {m for m, v in record["multipliers"].items() if v == 0.0} == set(frozen)
        assert record["beta"] is None


def test_resume_reuses_gated_stages_and_their_records(runs, tmp_path, monkeypatch):
    config, result = runs["map"]
    output = tmp_path / "run"
    shutil.copytree(config.output_dir, output)
    copied = replace(config, output_dir=str(output))
    records = _read_jsonl(output / "gates.jsonl")
    (output / "stage_3.npz").unlink()
    trained = []
    original = runner.train_stage

    def counting(*args, stage, **kwargs):
        trained.append(stage)
        return original(*args, stage=stage, **kwargs)

    monkeypatch.setattr(runner, "train_stage", counting)
    resumed = run_domain_forgetting(copied, resume=True, verbose=False)
    assert trained == [3]
    assert resumed["summary"]["stage_params_sha256"] == (
        result["summary"]["stage_params_sha256"]
    )
    again = _read_jsonl(output / "gates.jsonl")
    assert again[:2] == records[:2]
    assert again[2]["multipliers"] == records[2]["multipliers"]
    trained.clear()
    run_domain_forgetting(copied, resume=True, verbose=False)
    assert trained == []
    assert _read_jsonl(output / "gates.jsonl") == again
    with pytest.raises(ValueError, match="different setup"):
        run_domain_forgetting(
            replace(copied, gate_map_windows=4), resume=True, verbose=False
        )
    (output / "gates.jsonl").unlink()
    trained.clear()
    rebuilt = run_domain_forgetting(copied, resume=True, verbose=False)
    assert trained == [1, 2, 3]
    assert rebuilt["summary"]["stage_params_sha256"] == (
        result["summary"]["stage_params_sha256"]
    )
    assert [r["stage"] for r in _read_jsonl(output / "gates.jsonl")] == [1, 2, 3]
    assert [r["multipliers"] for r in _read_jsonl(output / "gates.jsonl")] == [
        r["multipliers"] for r in records
    ]


def test_identity_and_run_config_record_gates_only_when_gated(runs):
    config, _ = runs["none"]
    identity = config.identity()
    assert not set(GATE_KEYS) & set(identity)
    run_config = json.loads((Path(config.output_dir) / "run_config.json").read_text())
    assert "gate" not in run_config
    assert not set(GATE_KEYS) & set(run_config["config"])
    gated, _ = runs["freeze"]
    values = gated.identity()
    assert values["gate"] == "freeze"
    assert values["gate_modules"] == ["mlp", "token_embeddings"]
    assert values["gate_source"] == []
    assert "gates.jsonl" in RUN_ARTIFACT_FILES


def test_analyses_load_a_gated_run(runs):
    config, result = runs["freeze"]
    context = load_run(config.output_dir)
    assert context.config.gate == "freeze"
    assert context.config.gate_modules == ("mlp", "token_embeddings")
    assert len(context.snapshots) == len(DOMAINS) + 1


def test_oracle_gate_uses_source_rollback_recovery(runs, tmp_path):
    source, _ = runs["none"]
    rollback = compute_rollback(
        load_run(source.output_dir), random_draws=0, include_groups=False
    )
    atomic_write_json(Path(source.output_dir) / "rollback.json", rollback)
    config = replace(
        runs["map"][0],
        output_dir=str(tmp_path / "oracle"),
        gate="oracle",
        gate_map_windows=256,
        gate_beta=2.0,
        gate_source=(source.output_dir,),
    )
    run_domain_forgetting(config, verbose=False)
    records = _read_jsonl(tmp_path / "oracle" / "gates.jsonl")
    by_name = {t["name"]: t for t in rollback["transitions"]}
    for record in records[1:]:
        stage = record["stage"]
        entry = by_name[f"S{stage}<-S{stage - 1}"]
        reference = entry["reference_domain"]
        assert reference == DOMAINS[stage - 2]
        assert record["map"] is None
        for module, score in record["scores"].items():
            assert score == pytest.approx(
                entry["modules"][module]["recovery"][reference]
            )
        floored = {m: max(v, SCORE_FLOOR) for m, v in record["scores"].items()}
        expected = map_multipliers(floored, 2.0)
        for module, multiplier in record["multipliers"].items():
            assert multiplier == pytest.approx(expected.get(module, 1.0))
    run_config = json.loads((tmp_path / "oracle" / "run_config.json").read_text())
    recorded = run_config["gate"]["sources"][source.output_dir]
    assert recorded["domains"] == list(DOMAINS)
    assert (recorded["mode"], recorded["seed"]) == ("backprop", 0)
    assert len(recorded["rollback_sha256"]) == 64
    with pytest.raises(FileNotFoundError, match="rollback.json"):
        run_domain_forgetting(
            replace(
                config,
                output_dir=str(tmp_path / "missing"),
                gate_source=(str(tmp_path / "oracle"),),
            ),
            verbose=False,
        )


def test_oracle_rejects_incompatible_sources(runs, tmp_path):
    source, _ = runs["none"]
    original = json.loads((Path(source.output_dir) / "run_config.json").read_text())

    def fake(name, edit):
        directory = tmp_path / name
        directory.mkdir()
        payload = json.loads(json.dumps(original))
        edit(payload)
        (directory / "run_config.json").write_text(json.dumps(payload))
        (directory / "rollback.json").write_text("{}")
        return str(directory)

    for name, edit, match in (
        ("gated", lambda p: p["config"].update(gate="map"), "must be ungated"),
        (
            "joint",
            lambda p: p["config"].update(training="joint"),
            "must be sequential",
        ),
        (
            "short",
            lambda p: p["config"].update(domains=["news", "wiki"]),
            "has 2 domains",
        ),
        (
            "checkpoint",
            lambda p: p["source"].update(params_sha256="0" * 64),
            "different checkpoint",
        ),
    ):
        config = replace(
            runs["none"][0],
            output_dir=str(tmp_path / f"run_{name}"),
            gate="oracle",
            gate_source=(fake(name, edit),),
        )
        with pytest.raises(ValueError, match=match):
            run_domain_forgetting(config, verbose=False)


def test_map_gate_needs_enough_train_windows(runs, tmp_path):
    config = replace(
        runs["map"][0], output_dir=str(tmp_path / "run"), gate_map_windows=500
    )
    with pytest.raises(ValueError, match="gate_map_windows 500 exceeds"):
        run_domain_forgetting(config, verbose=False)
    assert not (tmp_path / "run" / "stage_1.npz").exists()


def test_gate_validation(setup, tmp_path):
    checkpoint, data_dir = setup
    config = tiny_config(checkpoint, data_dir, tmp_path)
    config.validate()
    for overrides, match in (
        ({"gate": "surprise"}, "gate must be one of"),
        ({"gate_beta": 2.0}, "gate_beta do not apply to gate 'none'"),
        ({"gate_modules": ("mlp",)}, "gate_modules do not apply"),
        ({"gate_source": ("x",)}, "gate_source do not apply"),
        ({"gate_map_windows": 8}, "gate_map_windows do not apply"),
        ({"gate_seed": 1}, "gate_seed do not apply"),
        ({"gate": "map", "gate_beta": 0.0}, "gate_beta must be finite and positive"),
        ({"gate": "map", "gate_beta": float("nan")}, "gate_beta must be finite"),
        ({"gate": "map", "gate_map_windows": 0}, "gate_map_windows must be positive"),
        ({"gate": "map", "gate_seed": 1}, "gate_seed do not apply to gate 'map'"),
        ({"gate": "random", "gate_seed": -1}, "gate_seed must be non-negative"),
        ({"gate": "freeze"}, "gate 'freeze' needs gate_modules"),
        ({"gate": "freeze", "gate_modules": ("mlp", "mlp")}, "duplicates"),
        ({"gate": "freeze", "gate_modules": ("mlp",), "gate_beta": 2.0}, "gate_beta"),
        ({"gate": "oracle"}, "gate 'oracle' needs gate_source"),
        ({"gate": "map", "training": "joint"}, "gates need training 'sequential'"),
    ):
        with pytest.raises(ValueError, match=match):
            replace(config, **overrides).validate()
    for gate, overrides in (
        ("map", {"gate_beta": 0.5}),
        ("inverse", {}),
        ("type", {"gate_map_windows": 16}),
        ("random", {"gate_seed": 4}),
        ("oracle", {"gate_source": ("a",)}),
        ("freeze", {"gate_modules": ("mlp",)}),
    ):
        replace(config, gate=gate, **overrides).validate()
    with pytest.raises(ValueError, match="unknown module or group 'L9_mlp1'"):
        run_domain_forgetting(
            replace(
                config,
                output_dir=str(tmp_path / "bad"),
                gate="freeze",
                gate_modules=("L9_mlp1",),
            ),
            verbose=False,
        )


def test_cli_parses_gate_flags(tmp_path):
    from examples import mlm_domain_forgetting as cli

    common = ["--checkpoint", "c", "--output_dir", str(tmp_path)]
    args = cli.build_parser().parse_args(
        common
        + [
            "--gate",
            "freeze",
            "--gate_modules",
            "embeddings",
            "blocks_0-1",
        ]
    )
    config = cli.config_from_args(args)
    assert config.gate == "freeze"
    assert config.gate_modules == ("embeddings", "blocks_0-1")
    config.validate()
    args = cli.build_parser().parse_args(
        common
        + [
            "--gate",
            "oracle",
            "--gate_beta",
            "0.5",
            "--gate_source",
            "runs/a",
            "runs/b",
        ]
    )
    config = cli.config_from_args(args)
    assert config.gate_source == tuple(
        str(Path(p).resolve()) for p in ("runs/a", "runs/b")
    )
    assert config.gate_beta == 0.5
    args = cli.build_parser().parse_args(
        common + ["--gate", "random", "--gate_map_windows", "64", "--gate_seed", "2"]
    )
    config = cli.config_from_args(args)
    assert (config.gate_map_windows, config.gate_seed) == (64, 2)
    default = cli.config_from_args(cli.build_parser().parse_args(common))
    assert default.gate == "none"
    assert not set(GATE_KEYS) & set(default.identity())
