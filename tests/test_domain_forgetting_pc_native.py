"""Tests for learner-native error maps and gates (tiny BERT, offline)."""

import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np
import pytest

from fabricpc.core.inference import InferenceSpikingPrecision
from fabricpc.core.inference_epc import EPCInference
from fabricpc.experiments.domain_forgetting import architecture_baseline as ab
from fabricpc.experiments.domain_forgetting import gates, runner
from fabricpc.experiments.domain_forgetting.error_maps import (
    LEARNER,
    SPIKE_ONLY,
    ErrorMapConfig,
    bp_signal_energies,
    compute_error_maps,
    hidden_nodes,
    learner_energy_field,
    learner_map_structure,
    learner_signal_energies,
    spike_only_inference,
)
from fabricpc.experiments.domain_forgetting.frontier import (
    collect_runs,
    compute_frontier,
    describe_gate,
)
from fabricpc.experiments.domain_forgetting.localization import load_run
from fabricpc.experiments.domain_forgetting.runner import (
    load_base_model,
    max_free_depth,
    run_domain_forgetting,
)
from fabricpc.graph_initialization import initialize_params
from fabricpc.training.plasticity_gates import module_multipliers
from tests.test_domain_forgetting_data import build_tiny_caches, tiny_tokenizer
from tests.test_domain_forgetting_error_maps import _batch, _tiny_bert
from tests.test_domain_forgetting_runner import tiny_config, write_tiny_bert_checkpoint

MAP_WINDOWS = 12
BLOCKS = ("L0_mha", "L0_mlp1", "L0_mlp2", "L1_mha", "L1_mlp1", "L1_mlp2")
EPC = {
    "mode": "pc",
    "pc_inference": "epc",
    "pc_eta_infer": 0.01,
    "pc_infer_steps": 3,
    "pc_forward_updates": "none",
}
MAP = {"gate": "map", "gate_beta": 2.0, "gate_map_windows": MAP_WINDOWS}
PROTECT = {
    "gate": "online_protect",
    "gate_beta": 2.0,
    "gate_map_windows": MAP_WINDOWS,
    "gate_interval": 1,
}
PINNED = {
    "map_bp": [
        "5da9ad1fe24ec2af248ee78f93ed9e8aebc9f9bd503fa14c69297e8a5be97d1e",
        "93666a30890ec83db3727178c6acf71cdd1a92cf050311e91348f49bd9ea8f67",
        "e25bc0683de0db43cf592741dec9a36494676800dd5d8abcbbf3d45f7f1eae2e",
        "3afc125550b51d3e31c8bb7cb00b7fc98fc7c2dea17902f6409b8db76608d201",
    ],
    "protect_bp": [
        "5da9ad1fe24ec2af248ee78f93ed9e8aebc9f9bd503fa14c69297e8a5be97d1e",
        "93666a30890ec83db3727178c6acf71cdd1a92cf050311e91348f49bd9ea8f67",
        "c0e26c9d028b0cde11d39fbf9cf7bebc053b1cdf75b7930c55ca9bbf4c6b6921",
        "6ab22f7087426bc494a98c871fe428bbf333e66c04b0f8f6099f49bd5ee3cc64",
    ],
    "map_epc": [
        "5da9ad1fe24ec2af248ee78f93ed9e8aebc9f9bd503fa14c69297e8a5be97d1e",
        "33adcda57512953f044e50e8bd559085b12199803a0deb91f489609af41f8d79",
        "d09c47caf4d0b13700f2c82b62aa1fbaddd24cfeb902d82aac696a851a051169",
        "34fe928af14bfb2870d0260df8bba5a6f90bf380dd4377ad2bca9dee0f5b7ef9",
    ],
}


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def _hashes(result):
    return result["summary"]["stage_params_sha256"]


def _relative(got, expected):
    got, expected = np.asarray(got), np.asarray(expected)
    return float(np.max(np.abs(got - expected) / np.abs(expected)))


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    directory = tmp_path_factory.mktemp("pc_native")
    tokenizer = tiny_tokenizer(directory)
    checkpoint = write_tiny_bert_checkpoint(directory, tokenizer, depth=2)
    build_tiny_caches(directory, tokenizer)
    return checkpoint, directory / "data", directory


def _config(setup, name, **overrides):
    checkpoint, data_dir, directory = setup
    return tiny_config(checkpoint, data_dir, directory / name, **overrides)


@pytest.fixture(scope="module")
def runs(setup):
    """The pinned runs at the default source and their learner-source twins."""
    configs = {
        "map_bp": _config(setup, "map_bp", **MAP),
        "protect_bp": _config(setup, "protect_bp", **PROTECT),
        "map_epc": _config(setup, "map_epc", **MAP, **EPC),
        "map_bp_learner": _config(
            setup, "map_bp_learner", **MAP, gate_map_source="learner"
        ),
        "map_epc_learner": _config(
            setup, "map_epc_learner", **MAP, **EPC, gate_map_source="learner"
        ),
    }
    return {
        name: (config, run_domain_forgetting(config, verbose=False))
        for name, config in configs.items()
    }


@pytest.fixture(scope="module")
def online_epc(setup):
    """ePC ``online_protect`` on the learner map, refreshed after every update.

    Records the multipliers each update's optimizer state held and counts
    the traces of the learner energy function.
    """
    config = _config(setup, "online_epc", **PROTECT, **EPC, gate_map_source="learner")
    held: dict[tuple[int, int], dict[str, float]] = {}
    traces = []
    calls = []
    original_train = runner.train_stage
    original_energy = gates.learner_signal_energies
    original_onset = gates.onset_map

    def capturing(*args, stage, gate_fn=None, **kwargs):
        wrapped = None
        if gate_fn is not None:

            def wrapped(update, params, opt_state):
                new_state = gate_fn(update, params, opt_state)
                held[(stage, update + 1)] = module_multipliers(new_state)
                return new_state

        return original_train(*args, stage=stage, gate_fn=wrapped, **kwargs)

    def counting(*args, **kwargs):
        traces.append(args[2]["x"].shape)
        return original_energy(*args, **kwargs)

    def onset(*args, **kwargs):
        calls.append(kwargs["num_windows"])
        return original_onset(*args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(runner, "train_stage", capturing)
        patch.setattr(gates, "learner_signal_energies", counting)
        patch.setattr(gates, "onset_map", onset)
        result = run_domain_forgetting(config, verbose=False)
    return config, result, held, traces, calls


def test_spike_only_learner_map_equals_the_bp_map_in_float64():
    with jax.enable_x64(True):
        probe = _tiny_bert(spike_only_inference(1))
        structure = _tiny_bert(spike_only_inference(max_free_depth(probe)))
        params = initialize_params(structure, jax.random.PRNGKey(0))
        batch = _batch()
        learner = learner_signal_energies(
            params, structure, batch, "forward_error_energy"
        )
        bp = bp_signal_energies(params, structure, batch, 1.0)
    assert set(learner) == set(bp) == set(hidden_nodes(structure))
    for node in bp:
        assert np.asarray(learner[node]).dtype == np.float64
        assert _relative(learner[node], bp[node]) < 1e-6


def test_one_epc_step_map_is_eta_squared_times_the_bp_map():
    """One step from ε = 0 gives ``ε_m = −η s ∂C/∂z_m``, so ``E = η² E_BP``."""
    eta = 0.05
    with jax.enable_x64(True):
        structure = _tiny_bert(EPCInference(eta_infer=eta, infer_steps=1))
        params = initialize_params(structure, jax.random.PRNGKey(0))
        batch = _batch()
        learner = learner_signal_energies(params, structure, batch, "final_energy")
        bp = bp_signal_energies(params, structure, batch, 1.0)
    for node in bp:
        assert _relative(learner[node], eta**2 * np.asarray(bp[node])) < 1e-6


def test_epc_map_at_small_rate_approaches_eta_t_squared_times_the_bp_map():
    """After ``T`` steps ``E → (ηT)² E_BP`` with a first-order error in ``η``."""
    steps = 5

    def worst(eta):
        structure = _tiny_bert(EPCInference(eta_infer=eta, infer_steps=steps))
        learner = learner_signal_energies(params, structure, batch, "final_energy")
        return max(
            _relative(learner[n], (eta * steps) ** 2 * np.asarray(bp[n])) for n in bp
        )

    with jax.enable_x64(True):
        probe = _tiny_bert(EPCInference(eta_infer=1e-5, infer_steps=steps))
        params = initialize_params(probe, jax.random.PRNGKey(0))
        batch = _batch()
        bp = bp_signal_energies(params, probe, batch, 1.0)
        coarse, fine, relaxed = worst(1e-5), worst(1e-6), worst(0.3)
    assert fine < 2e-4
    assert 8.0 < coarse / fine < 12.0
    assert relaxed > 0.1


def test_learner_structure_and_field_per_learner(setup):
    model_config = load_base_model(_config(setup, "probe")).model_config
    backprop = _config(setup, "probe")
    structure = learner_map_structure(model_config, backprop)
    inference = structure.config["inference"]
    assert isinstance(inference, InferenceSpikingPrecision)
    assert inference.config["eta_infer"] == 0.0
    assert learner_energy_field(backprop) == "forward_error_energy"
    epc = _config(setup, "probe", **EPC, pc_output_energy_scale=2.0)
    structure = learner_map_structure(model_config, epc)
    inference = structure.config["inference"]
    assert isinstance(inference, EPCInference)
    assert inference.config["eta_infer"] == EPC["pc_eta_infer"]
    assert inference.config["infer_steps"] == EPC["pc_infer_steps"]
    assert learner_energy_field(epc) == "final_energy"
    output = structure.task_map["y"]
    node_config = structure.nodes[output].node_info.node_config
    assert node_config["output_energy_scale"] == 2.0
    spiking = _config(setup, "probe", mode="pc")
    assert learner_energy_field(spiking) == "forward_error_energy"


def test_bp_source_keeps_existing_runs_bit_identical(runs):
    for name, expected in PINNED.items():
        config, result = runs[name]
        assert config.gate_map_source == "bp"
        assert _hashes(result) == expected
        assert "gate_map_source" not in config.identity()
        output = Path(config.output_dir)
        run_config = json.loads((output / "run_config.json").read_text())
        assert "gate_map_source" not in run_config["config"]
        assert "source" not in run_config["gate"]["map"]
        assert run_config["gate"]["map"]["energy"].endswith("(spike-only map)")
        records = _read_jsonl(output / "gates.jsonl")
        assert all("map_source" not in r for r in records)
    config, _ = runs["protect_bp"]
    for stage in (2, 3):
        path = Path(config.output_dir) / f"gate_schedule_stage_{stage}.jsonl"
        assert all("map_source" not in r for r in _read_jsonl(path))
    config, _ = runs["map_bp"]
    resumed = run_domain_forgetting(config, resume=True, verbose=False)
    assert _hashes(resumed) == PINNED["map_bp"]


def test_backprop_learner_source_equals_the_bp_source(runs):
    config, result = runs["map_bp_learner"]
    _, reference = runs["map_bp"]
    assert _hashes(result) == _hashes(reference)
    assert config.identity()["gate_map_source"] == "learner"
    records = _read_jsonl(Path(config.output_dir) / "gates.jsonl")
    assert [r["map_source"] for r in records] == ["learner"] * 3
    reference_records = _read_jsonl(Path(runs["map_bp"][0].output_dir) / "gates.jsonl")
    assert [r["map"] for r in records] == [r["map"] for r in reference_records]


def test_epc_learner_source_changes_and_logs_the_applied_multipliers(runs):
    config, result = runs["map_epc_learner"]
    reference_config, reference = runs["map_epc"]
    output = Path(config.output_dir)
    run_config = json.loads((output / "run_config.json").read_text())
    assert run_config["config"]["gate_map_source"] == "learner"
    described = run_config["gate"]["map"]
    assert described["source"] == "learner"
    assert described["inference"]["class"] == "EPCInference"
    assert described["inference"]["eta_infer"] == EPC["pc_eta_infer"]
    assert "final_energy" in described["energy"]
    records = _read_jsonl(output / "gates.jsonl")
    bp_records = _read_jsonl(Path(reference_config.output_dir) / "gates.jsonl")
    assert _hashes(result)[:2] == _hashes(reference)[:2]
    assert _hashes(result)[2] != _hashes(reference)[2]
    learner_map, bp_map = records[1]["map"], bp_records[1]["map"]
    ratios = [learner_map[m] / bp_map[m] for m in BLOCKS]
    assert max(ratios) / min(ratios) > 1.0 + 1e-6
    assert records[1]["multipliers"] != bp_records[1]["multipliers"]
    exposure = json.loads((output / "exposure.json").read_text())
    for record in records[1:]:
        assert record["map_source"] == "learner"
        applied = exposure[record["domain"]]["optimizer"]["module_multipliers"]
        assert applied == record["multipliers"]


def test_gate_onset_map_equals_a_direct_learner_onset_map(runs):
    """The recorded gate map is ``onset_map`` of ``learner_signal_energies``."""
    config, _ = runs["map_epc_learner"]
    context = load_run(config.output_dir)
    record = _read_jsonl(Path(config.output_dir) / "gates.jsonl")[1]
    structure = learner_map_structure(context.model_config, context.config)
    base = load_base_model(config)
    from fabricpc.experiments.domain_forgetting.data import load_domain_cache

    stream = load_domain_cache(
        config.data_dir,
        record["domain"],
        tokenizer_json=base.tokenizer_json,
        seq_len=int(context.model_config["seq_len"]),
        tokens_per_domain=config.tokens_per_domain,
    ).train
    direct = gates.onset_map(
        context.params(record["stage"] - 1),
        structure,
        stream,
        base.special_ids,
        modules=list(BLOCKS),
        num_windows=MAP_WINDOWS,
        mask_prob=config.mask_prob,
        seed=record["map_seed"],
        energy_fn=jax.jit(
            lambda p, b: learner_signal_energies(p, structure, b, "final_energy")
        ),
    )
    for module in BLOCKS:
        assert direct[module] == pytest.approx(record["map"][module], rel=1e-6)


def test_online_learner_gate_logs_applied_multipliers_and_compiles_once(online_epc):
    config, _, held, traces, calls = online_epc
    output = Path(config.output_dir)
    records = _read_jsonl(output / "gates.jsonl")
    assert len(calls) == 2 * (1 + 2)
    assert len(traces) == len({tuple(s) for s in traces}) == 2
    for stage in (2, 3):
        assert records[stage - 1]["map_source"] == "learner"
        schedule = _read_jsonl(output / f"gate_schedule_stage_{stage}.jsonl")
        assert all(r["map_source"] == "learner" for r in schedule)
        assert all(r["energies"] is not None for r in schedule[1:])
        for record in schedule:
            for update in range(record["first_update"], record["last_update"] + 1):
                state = held[(stage, update)]
                assert {m: state[m] for m in record["multipliers"]} == (
                    record["multipliers"]
                )


def test_replay_refuses_a_source_of_another_map_source(online_epc, setup):
    config, result, _, _, _ = online_epc
    base = {**PROTECT, **EPC, "gate_variant": "replay"}
    bad = _config(setup, "replay_bad", **base, gate_source=(config.output_dir,))
    with pytest.raises(ValueError, match="gate_map_source 'learner' differs"):
        run_domain_forgetting(bad, verbose=False, overwrite=True)
    replay = _config(
        setup,
        "replay_ok",
        **base,
        gate_map_source="learner",
        gate_source=(config.output_dir,),
    )
    replayed = run_domain_forgetting(replay, verbose=False)
    assert _hashes(replayed) == _hashes(result)


def test_gate_map_source_validation(setup):
    for overrides, match in (
        ({**MAP, "gate_map_source": "epc"}, "gate_map_source must be one of"),
        ({"gate_map_source": "learner"}, "do not apply to gate 'none'"),
        (
            {"gate": "uniform", "gate_scale": 0.5, "gate_map_source": "learner"},
            "do not apply to gate 'uniform'",
        ),
        (
            {"gate": "freeze", "gate_modules": ("mlp",), "gate_map_source": "learner"},
            "do not apply to gate 'freeze'",
        ),
    ):
        with pytest.raises(ValueError, match=match):
            _config(setup, "invalid", **overrides).validate()
    for gate in ("map", "random", "inverse", "type", "online_surprise"):
        overrides = {"gate": gate, "gate_map_source": "learner"}
        _config(setup, "valid", **overrides).validate()


def test_cli_parses_the_gate_map_source(tmp_path):
    from examples.mlm_domain_forgetting import build_parser, config_from_args

    args = build_parser().parse_args(
        [
            "--checkpoint",
            "c.json",
            "--domains",
            "news",
            "wiki",
            "--output_dir",
            str(tmp_path),
            *("--mode", "pc", "--pc_inference", "epc"),
            *("--pc_eta_infer", "1e-4", "--pc_infer_steps", "5"),
            *("--gate", "map", "--gate_beta", "2", "--gate_map_source", "learner"),
        ]
    )
    config = config_from_args(args)
    config.validate()
    assert config.gate_map_source == "learner"
    default = config_from_args(
        build_parser().parse_args(
            ["--checkpoint", "c.json", "--domains", "news", "--output_dir", "o"]
        )
    )
    assert default.gate_map_source == "bp"


def test_error_maps_default_is_unchanged_and_learner_adds_its_setting(runs, tmp_path):
    from examples import domain_forgetting_analysis as cli

    config, _ = runs["map_epc_learner"]
    context = load_run(config.output_dir)
    expected = compute_error_maps(context, ErrorMapConfig(settings=(SPIKE_ONLY,)))
    default = cli.main(
        ["error_maps", "--run_dir", config.output_dir, "--settings", SPIKE_ONLY]
    )
    assert default["maps"] == expected["maps"]
    assert default["settings"] == expected["settings"]
    learner = cli.main(
        [
            "error_maps",
            "--run_dir",
            config.output_dir,
            "--settings",
            SPIKE_ONLY,
            "--map_source",
            "learner",
            "--quiet",
        ]
    )
    spike = [m for m in learner["maps"] if m["setting"] == SPIKE_ONLY]
    assert spike == expected["maps"]
    settings = learner["settings"][LEARNER]
    assert settings["inference"] == "EPCInference"
    assert settings["eta_infer"] == EPC["pc_eta_infer"]
    assert settings["primary_energy"] == "final_energy"
    assert settings["pc_inference"] == "epc"
    native = [m for m in learner["maps"] if m["setting"] == LEARNER]
    assert [(m["kind"], m["domain"], m["stage"]) for m in native] == [
        (m["kind"], m["domain"], m["stage"]) for m in spike
    ]
    for ours, theirs in zip(native, spike):
        for module in BLOCKS:
            values = ours["modules"][module]
            assert values["energy_sum_ce"] == values["final_energy_sum_ce"]
            ratio = values["energy_sum_ce"] / theirs["modules"][module]["energy_sum_ce"]
            assert 0.0 < ratio < 1.0


def test_backprop_learner_error_map_is_the_spike_only_map(runs):
    config, _ = runs["map_bp"]
    context = load_run(config.output_dir)
    result = compute_error_maps(context, ErrorMapConfig(settings=(SPIKE_ONLY, LEARNER)))
    by_setting = {
        setting: [m for m in result["maps"] if m["setting"] == setting]
        for setting in (SPIKE_ONLY, LEARNER)
    }
    for ours, theirs in zip(by_setting[LEARNER], by_setting[SPIKE_ONLY]):
        assert ours["modules"] == theirs["modules"]


def _learner_table(table):
    return {
        name: {
            **entry,
            "predictors": {
                key.replace("spike_only_", "learner_"): value
                for key, value in entry["predictors"].items()
            },
        }
        for name, entry in table.items()
    }


def test_architecture_map_source_selects_the_learner_onset(tmp_path, monkeypatch):
    from examples import domain_forgetting_analysis as cli
    from tests.test_domain_forgetting_architecture import energy_runs

    tables = {}
    run_dirs = []
    for index, entry in enumerate(energy_runs()):
        path = tmp_path / f"run{index}"
        path.mkdir()
        config = {"config": {"domains": list(entry["order"]), "seed": 0}}
        (path / "run_config.json").write_text(json.dumps(config))
        tables[str(path)] = _learner_table(entry["table"])
        run_dirs.append(str(path))
    monkeypatch.setattr(ab, "run_predictor_table", lambda path: tables[str(path)])
    common = ["architecture", "--run_dirs", *run_dirs, "--permutations", "20"]
    with pytest.raises(ValueError, match="spike_only_onset"):
        cli.main([*common, "--output_dir", str(tmp_path / "bp"), "--quiet"])
    learner = cli.main(
        [*common, "--output_dir", str(tmp_path / "learner"), "--map_source", "learner"]
    )
    reference = ab.analyse(energy_runs(), permutations=20)
    assert learner["config"]["setting"] == LEARNER
    assert learner["primary"] == reference["primary"]
    assert learner["looo"] == reference["looo"]


def _write_point(root, name, *, x, y, seed, lr, mode="pc", inference="epc", **gate):
    path = root / name
    path.mkdir()
    domains = ("news", "reviews", "wiki")
    config = {
        "lr": lr,
        "seed": seed,
        "domains": list(domains),
        "mode": mode,
        "pc_inference": inference,
        **gate,
    }
    (path / "run_config.json").write_text(json.dumps({"config": config}))
    summary = {
        "domains": list(domains),
        "mode": mode,
        "final_ce": {d: y for d in domains},
        "learning_gain_ce": {d: x for d in domains},
        "bwt_ce": -x,
    }
    (path / "summary.json").write_text(json.dumps(summary))
    return path


def test_frontier_selects_an_epc_reference_curve(tmp_path, capsys):
    from examples import domain_forgetting_analysis as cli

    curve = {1e-5: (0.2, 3.3), 3e-5: (0.4, 3.1), 1e-4: (0.6, 3.2)}
    paths = []
    for seed in (0, 1):
        for lr, (x, y) in curve.items():
            paths.append(
                _write_point(tmp_path, f"epc{lr:g}_{seed}", x=x, y=y, seed=seed, lr=lr)
            )
            paths.append(
                _write_point(
                    tmp_path,
                    f"bp{lr:g}_{seed}",
                    x=x,
                    y=y + 1.0,
                    seed=seed,
                    lr=lr,
                    mode="backprop",
                    inference="spiking",
                )
            )
            paths.append(
                _write_point(
                    tmp_path,
                    f"spk{lr:g}_{seed}",
                    x=x,
                    y=y - 1.0,
                    seed=seed,
                    lr=lr,
                    inference="spiking",
                )
            )
        gate = {"gate": "map", "gate_beta": 2.0}
        paths.append(
            _write_point(
                tmp_path,
                f"map_learner_{seed}",
                x=0.5,
                y=3.0,
                seed=seed,
                lr=3e-5,
                **gate,
                gate_map_source="learner",
            )
        )
        paths.append(
            _write_point(
                tmp_path, f"map_bp_{seed}", x=0.5, y=3.4, seed=seed, lr=3e-5, **gate
            )
        )
    assert (
        describe_gate(
            {"gate": "map", "gate_beta": 2.0, "lr": 1e-4, "gate_map_source": "learner"}
        )[2]
        == "map_b2_learner"
    )
    assert (
        describe_gate(
            {
                "gate": "online_protect",
                "gate_beta": 2.0,
                "lr": 1e-4,
                "gate_variant": "replay",
                "gate_map_source": "learner",
            }
        )[2]
        == "online_protect_b2_learner_replay"
    )
    backprop = collect_runs(paths)
    assert {p["mode"] for p in backprop} == {"backprop"}
    epc = collect_runs(paths, mode="pc", pc_inference="epc")
    assert {p["pc_inference"] for p in epc} == {"epc"} and len(epc) == 10
    assert len(collect_runs(paths, mode="pc")) == 16
    result = compute_frontier(epc, seq_lr=3e-5, map_source="learner")
    order = result["orders"]["news>reviews>wiki"]
    assert order["curves"]["0"]["y"] == pytest.approx([3.3, 3.1, 3.2])
    assert set(order["conditions"]) >= {"map_b2_learner@lr3e-05", "map_b2@lr3e-05"}
    assert order["primary_test"]["2"]["d"] == pytest.approx([3.0 - 3.15] * 2)
    bp_source = compute_frontier(epc, seq_lr=3e-5)
    assert bp_source["orders"]["news>reviews>wiki"]["primary_test"]["2"]["d"] == (
        pytest.approx([3.4 - 3.15] * 2)
    )
    out = tmp_path / "out"
    written = cli.main(
        [
            "frontier",
            "--run_dirs",
            *map(str, paths),
            "--reference_mode",
            "pc",
            "--pc_inference",
            "epc",
            "--map_source",
            "learner",
            "--seq_lr",
            "3e-5",
            "--output_dir",
            str(out),
            "--no_figure",
            "--quiet",
        ]
    )
    assert written["reference"] == {
        "mode": "pc",
        "pc_inference": "epc",
        "map_source": "learner",
    }
    on_disk = json.loads((out / "frontier.json").read_text())
    assert on_disk["primary_test_both_orders"]["2"]["per_order"] == {
        "news>reviews>wiki": order["primary_test"]["2"]["pass"]
    }
    default = cli.main(
        [
            "frontier",
            "--run_dirs",
            *map(str, paths),
            "--output_dir",
            str(tmp_path / "bp_out"),
            "--no_figure",
            "--quiet",
        ]
    )
    assert "reference" not in default
    curve_y = default["orders"]["news>reviews>wiki"]["curves"]["0"]["y"]
    assert curve_y == pytest.approx([4.3, 4.1, 4.2])
    with pytest.raises(SystemExit):
        cli.main(["frontier", "--run_dirs", str(paths[0]), "--pc_inference", "epc"])


def _epc_frontier_runs(root, *, diverged_ref=None, diverged_gate=None, seeds=(0, 1)):
    root.mkdir(parents=True, exist_ok=True)
    curve = {1e-5: (0.2, 3.3), 3e-5: (0.4, 3.1), 1e-4: (0.6, 3.2)}
    paths = []
    for seed in (0, 1):
        for lr, (x, y) in curve.items():
            if (lr, seed) == diverged_ref:
                y = float("nan")
            paths.append(
                _write_point(root, f"ref{lr:g}_{seed}", x=x, y=y, seed=seed, lr=lr)
            )
    for seed in seeds:
        y = float("nan") if seed == diverged_gate else 3.0
        paths.append(
            _write_point(
                root,
                f"map_{seed}",
                x=0.5,
                y=y,
                seed=seed,
                lr=3e-5,
                gate="map",
                gate_beta=2.0,
                gate_map_source="learner",
            )
        )
    return collect_runs(paths, mode="pc", pc_inference="epc")


def test_frontier_records_diverged_reference_runs_and_interpolates_the_rest(
    tmp_path,
):
    points = _epc_frontier_runs(tmp_path, diverged_ref=(1e-4, 1))
    failed = [p for p in points if p["diverged"]]
    assert len(failed) == 1 and failed[0]["x"] is None and failed[0]["y"] is None
    result = compute_frontier(points, seq_lr=3e-5, map_source="learner")
    order = result["orders"]["news>reviews>wiki"]
    assert order["failed_reference_runs"] == [
        {"run_dir": failed[0]["run_dir"], "lr": 1e-4, "seed": 1}
    ]
    assert order["curves"]["1"]["lr"] == [1e-5, 3e-5]
    assert order["curves"]["0"]["lr"] == [1e-5, 3e-5, 1e-4]
    test = order["primary_test"]["2"]
    assert test["expected_seeds"] == [0, 1] and test["missing_seeds"] == []
    assert test["d"][0] == pytest.approx(3.0 - 3.15)
    assert test["d"][1] is None and test["pass"] is False
    from fabricpc.experiments.domain_forgetting.frontier import format_table

    table = format_table(result)
    assert "failed reference run (diverged" in table and "seed 1" in table


def test_frontier_primary_test_fails_on_diverged_or_missing_gate_seeds(tmp_path):
    passing = compute_frontier(
        _epc_frontier_runs(tmp_path / "ok"), seq_lr=3e-5, map_source="learner"
    )
    ok = passing["orders"]["news>reviews>wiki"]["primary_test"]["2"]
    assert ok["pass"] is True and ok["diverged_seeds"] == []
    diverged = compute_frontier(
        _epc_frontier_runs(tmp_path / "div", diverged_gate=1),
        seq_lr=3e-5,
        map_source="learner",
    )
    test = diverged["orders"]["news>reviews>wiki"]["primary_test"]["2"]
    assert test["diverged_seeds"] == [1] and test["pass"] is False
    assert test["all_seeds_negative"] is False
    missing = compute_frontier(
        _epc_frontier_runs(tmp_path / "miss", seeds=(0,)),
        seq_lr=3e-5,
        map_source="learner",
    )
    test = missing["orders"]["news>reviews>wiki"]["primary_test"]["2"]
    assert test["missing_seeds"] == [1] and test["pass"] is False
    explicit = compute_frontier(
        _epc_frontier_runs(tmp_path / "miss2", seeds=(0,)),
        seq_lr=3e-5,
        map_source="learner",
        expected_seeds=[0],
    )
    assert explicit["orders"]["news>reviews>wiki"]["primary_test"]["2"]["pass"]
    from fabricpc.experiments.domain_forgetting.frontier import format_table

    assert "missing seeds [1]" in format_table(missing)
    assert "contrasts: none" in format_table(missing)


def test_error_maps_merge_by_setting_and_refuse_changes(runs, tmp_path):
    from examples import domain_forgetting_analysis as cli
    from fabricpc.experiments.domain_forgetting.error_maps import merge_error_maps

    config, _ = runs["map_bp_learner"]
    path = Path(config.output_dir) / "error_maps.json"
    assert not path.exists()
    first = cli.main(
        ["error_maps", "--run_dir", config.output_dir, "--settings", SPIKE_ONLY]
    )
    before = path.read_text()
    kept = cli.main(
        ["error_maps", "--run_dir", config.output_dir, "--settings", SPIKE_ONLY]
    )
    assert path.read_text() == before and kept["maps"] == first["maps"]
    merged = cli.main(
        [
            "error_maps",
            "--run_dir",
            config.output_dir,
            "--settings",
            SPIKE_ONLY,
            "--map_source",
            "learner",
        ]
    )
    on_disk = json.loads(path.read_text())
    assert sorted(on_disk["settings"]) == sorted([SPIKE_ONLY, LEARNER])
    assert on_disk["config"]["settings"] == [SPIKE_ONLY, LEARNER]
    spike = [m for m in on_disk["maps"] if m["setting"] == SPIKE_ONLY]
    assert spike == json.loads(before)["maps"]
    assert merged["settings"] == on_disk["settings"]
    with pytest.raises(SystemExit, match="config"):
        cli.main(
            [
                "error_maps",
                "--run_dir",
                config.output_dir,
                "--settings",
                SPIKE_ONLY,
                "--map_windows",
                "128",
            ]
        )
    tampered = json.loads(json.dumps(on_disk))
    tampered["settings"][SPIKE_ONLY]["infer_steps"] += 1
    path.write_text(json.dumps(tampered))
    with pytest.raises(SystemExit, match="hold setting 'spike_only'"):
        cli.main(
            ["error_maps", "--run_dir", config.output_dir, "--settings", SPIKE_ONLY]
        )
    assert json.loads(path.read_text()) == tampered
    path.write_text(json.dumps(on_disk))
    with pytest.raises(ValueError, match="already exist"):
        merge_error_maps(on_disk, on_disk)
    other = dict(on_disk, head_energy="mean_masked_ce")
    with pytest.raises(ValueError, match="head_energy"):
        merge_error_maps(on_disk, other)


def test_predictors_are_kept_or_refused_never_overwritten(runs, monkeypatch):
    from examples import domain_forgetting_analysis as cli

    config, _ = runs["map_bp_learner"]
    path = Path(config.output_dir) / "predictors.json"
    common = ["predictors", "--run_dir", config.output_dir, "--quiet"]
    small = ["--predictor_windows", "8", "--predictor_batch_size", "4"]
    cli.main([*common, *small])
    before = path.read_text()

    def fail(*args, **kwargs):
        raise AssertionError("recomputed")

    monkeypatch.setattr(cli, "compute_predictors", fail)
    cli.main([*common, *small])
    assert path.read_text() == before
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        cli.main(common)
    assert path.read_text() == before
