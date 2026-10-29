"""Tests for ePC runs, energy/gradient logging and the regime probe (tiny BERT)."""

import json
import math
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

from fabricpc.experiments.domain_forgetting.correlations import (
    compute_correlations,
    standard_map_convention,
)
from fabricpc.experiments.domain_forgetting.data import load_domain_cache
from fabricpc.experiments.domain_forgetting.error_maps import (
    SPIKE_ONLY,
    STANDARD_PC,
    ErrorMapConfig,
    aggregate_map,
    bp_signal_energies,
    node_energies,
    spike_only_inference,
)
from fabricpc.experiments.domain_forgetting.probe import probe_alignment
from fabricpc.experiments.domain_forgetting.runner import (
    HEAD_ENERGY,
    RUN_ARTIFACT_FILES,
    load_base_model,
    make_gradient_fn,
    max_free_depth,
    run_domain_forgetting,
    train_stage,
)
from fabricpc.graph_initialization import initialize_params
from tests.test_domain_forgetting_error_maps import SCALE, _batch, _tiny_bert
from tests.test_domain_forgetting_runner import DOMAINS, make_setup, tiny_config

EPC = {
    "mode": "pc",
    "pc_inference": "epc",
    "pc_eta_infer": 0.01,
    "pc_infer_steps": 3,
    "pc_forward_updates": "none",
}
REGIME = {"pc_regime_interval": 2, "pc_regime_eta": 0.01, "pc_regime_steps": 3}


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    return make_setup(tmp_path_factory.mktemp("epc"))


@pytest.fixture(scope="module")
def epc_run(setup, tmp_path_factory):
    checkpoint, data_dir = setup
    output = tmp_path_factory.mktemp("epc_run") / "run"
    config = tiny_config(
        checkpoint, data_dir, output, pc_regime_interval=2, pc_regime_iters=8, **EPC
    )
    return config, run_domain_forgetting(config, verbose=False)


@pytest.fixture(scope="module")
def bp_runs(setup, tmp_path_factory):
    checkpoint, data_dir = setup
    root = tmp_path_factory.mktemp("bp_regime")
    plain = tiny_config(checkpoint, data_dir, root / "plain")
    probed = tiny_config(checkpoint, data_dir, root / "probed", **REGIME)
    return (
        (plain, run_domain_forgetting(plain, verbose=False)),
        (probed, run_domain_forgetting(probed, verbose=False)),
    )


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def test_epc_run_logs_energy_statistics(epc_run):
    config, result = epc_run
    output = Path(config.output_dir)
    run_config = json.loads((output / "run_config.json").read_text())
    assert run_config["pc"]["inference"] == "epc"
    assert run_config["pc"]["infer_steps"] == 3
    assert run_config["pc"]["head_energy"] == HEAD_ENERGY == "sum_masked_ce"
    assert "pc_regime_interval" not in run_config["config"]
    ce = np.asarray(result["matrices"]["masked_ce"])
    assert ce.shape == (4, 3) and np.all(np.isfinite(ce))
    for stage in range(1, 4):
        log = _read_jsonl(output / f"training_log_stage_{stage}.jsonl")
        assert len(log) == 3
        for entry in log:
            for key in (
                "pc_energy_start",
                "pc_energy_end",
                "pc_energy_rel_decrease",
                "pc_total_energy",
                "grad_norm",
            ):
                assert np.isfinite(entry[key]), key
            assert entry["pc_energy_rises"] >= 0
            assert entry["pc_energy_increased"] is False
            assert entry["pc_energy_end"] < entry["pc_energy_start"]
            assert entry["pc_total_energy"] == pytest.approx(
                entry["pc_energy_end"], rel=1e-4
            )
            assert entry["clipped"] == (entry["grad_norm"] > config.clip_norm)


def test_epc_divergence_is_flagged_once(setup, tmp_path, capsys):
    checkpoint, data_dir = setup
    config = tiny_config(
        checkpoint,
        data_dir,
        tmp_path / "run",
        **{**EPC, "pc_eta_infer": 50.0, "pc_infer_steps": 5},
    )
    run_domain_forgetting(config, verbose=False)
    warnings = [
        line for line in capsys.readouterr().out.splitlines() if "WARNING" in line
    ]
    assert len(warnings) == 3
    assert all("epc PC inference raised the energy" in w for w in warnings)
    log = _read_jsonl(tmp_path / "run" / "training_log_stage_1.jsonl")
    assert any(entry["pc_energy_increased"] for entry in log)
    assert any(entry["pc_energy_rises"] > 0 for entry in log)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"pc_eta_infer": 0.0}, "ePC needs pc_eta_infer > 0"),
        ({"pc_infer_steps": None}, "ePC needs pc_infer_steps"),
        ({"pc_forward_updates": "initial"}, "pc_forward_updates 'none'"),
        ({"pc_spike_eta": 0.5}, "pc_spike_eta"),
        ({"pc_spike_depth": "shortest"}, "pc_spike_depth"),
    ],
)
def test_epc_validation(overrides, message, tmp_path):
    config = tiny_config("ck", "data", tmp_path, **{**EPC, **overrides})
    with pytest.raises(ValueError, match=message):
        config.validate()


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"pc_regime_interval": -1}, "non-negative"),
        ({"pc_regime_eta": 0.1}, "need pc_regime_interval > 0"),
        ({"pc_regime_iters": 5}, "need pc_regime_interval > 0"),
        ({"pc_regime_interval": 2}, "pc_regime_eta and pc_regime_steps"),
        ({**REGIME, "pc_regime_eta": 0.0}, "pc_regime_eta must be"),
        ({**REGIME, "pc_regime_steps": 0}, "pc_regime_steps must be"),
        ({**REGIME, "pc_regime_iters": 0}, "pc_regime_iters must be"),
    ],
)
def test_regime_validation(overrides, message, tmp_path):
    config = tiny_config("ck", "data", tmp_path, **overrides)
    with pytest.raises(ValueError, match=message):
        config.validate()


def test_regime_defaults_to_the_runs_epc_setting(tmp_path):
    config = tiny_config("ck", "data", tmp_path, pc_regime_interval=1, **EPC)
    config.validate()
    assert config.regime_setting == (0.01, 3)


def test_regime_records_cadence_and_content(epc_run):
    config, _ = epc_run
    output = Path(config.output_dir)
    assert "regime.jsonl" in RUN_ARTIFACT_FILES
    records = _read_jsonl(output / "regime.jsonl")
    assert [(r["stage"], r["update"], r["when"]) for r in records] == [
        (stage, update, when)
        for stage in range(1, 4)
        for update, when in ((0, "start"), (2, "interval"), (3, "end"))
    ]
    summary = json.loads((output / "summary.json").read_text())
    hashes = summary["stage_params_sha256"]
    modules = None
    for record in records:
        assert record["domain"] == DOMAINS[record["stage"] - 1]
        assert record["eta_infer"] == 0.01 and record["infer_steps"] == 3
        assert record["lanczos_iters"] == 8 and record["num_updates"] == 3
        assert record["batch_size"] == config.batch_size
        assert np.isfinite(record["lambda_max"]) and record["lambda_max"] > 0
        assert record["lambda_min"] <= record["lambda_max"]
        assert 0.0 <= record["negative_weight"] <= 1.0
        assert record["band"] in (
            "backprop-like",
            "partially relaxed",
            "near PC equilibrium",
        )
        assert isinstance(record["regime"], str) and record["regime"]
        assert record["unstable"] is False
        assert record["f_valid"] is True and 0.0 <= record["f_weighted"] <= 1.0
        assert "eta_controlled" not in record
        modules = modules or list(record["weight_norms"])
        assert list(record["weight_norms"]) == modules
        assert all(v > 0 for v in record["weight_norms"].values())
        if record["when"] == "start":
            assert record["params_sha256"] == hashes[record["stage"] - 1]
        if record["when"] == "end":
            assert record["params_sha256"] == hashes[record["stage"]]
    assert "mlm_head" in modules
    runtime = json.loads((output / "runtime.json").read_text())
    assert runtime["stages"]["1"]["regime_seconds"] > 0


def test_regime_probe_does_not_change_backprop_training(bp_runs):
    (plain, plain_result), (probed, probed_result) = bp_runs
    assert probed_result["summary"]["stage_params_sha256"] == (
        plain_result["summary"]["stage_params_sha256"]
    )
    assert not (Path(plain.output_dir) / "regime.jsonl").exists()
    records = _read_jsonl(Path(probed.output_dir) / "regime.jsonl")
    assert len(records) == 9
    plain_config = json.loads((Path(plain.output_dir) / "run_config.json").read_text())
    probed_config = json.loads(
        (Path(probed.output_dir) / "run_config.json").read_text()
    )
    assert plain_config == {**probed_config, "source": plain_config["source"]}
    assert plain_config["pc"] is None
    for stage in range(1, 4):
        log = _read_jsonl(Path(plain.output_dir) / f"training_log_stage_{stage}.jsonl")
        for entry in log:
            assert entry["grad_norm"] > 0 and isinstance(entry["clipped"], bool)
            assert "pc_total_energy" not in entry


def test_resume_keeps_regime_records(bp_runs):
    _, (probed, _) = bp_runs
    path = Path(probed.output_dir) / "regime.jsonl"
    before = path.read_text()
    run_domain_forgetting(probed, resume=True, verbose=False)
    assert path.read_text() == before
    runtime = json.loads((Path(probed.output_dir) / "runtime.json").read_text())
    assert all(stage["reused"] for stage in runtime["stages"].values())


def test_probe_epc_grid_reports_regime(setup, capsys):
    checkpoint, data_dir = setup
    report = probe_alignment(
        domain="news",
        checkpoint=str(checkpoint),
        data_dir=str(data_dir),
        num_batches=2,
        batch_size=4,
        mask_prob=0.3,
        pc_inference="epc",
        settings=[(1e-4, 1), (0.05, 5), (50.0, 2)],
        lanczos_iters=8,
    )
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0].split()[:3] == ["eta", "T", "cos_hid"]
    assert len(lines) == 4
    assert report["forward_updates"] == "none"
    assert report["lanczos_iters"] == 8 and len(report["spectrum"]) == 2
    lam = max(row["lambda_max"] for row in report["spectrum"])
    small, _middle, large = report["settings"]
    for setting, line in zip(report["settings"], lines[1:]):
        regime = setting["regime"]
        assert regime["lambda_max"] == pytest.approx(lam)
        assert len(regime["batches"]) == 2
        assert regime["regime"] in line
    assert small["regime"]["bands"] == ["backprop-like"]
    assert small["modules"]["mlm_head"]["cosine"] > 0.999
    assert small["modules"]["mlm_head"]["norm_ratio"] == pytest.approx(1.0, rel=1e-3)
    assert large["regime"]["unstable"] and "unstable" in lines[3]


def test_spike_only_maps_keep_the_mean_ce_numbers():
    """Map values are BP's mean-CE signal and its summed-CE variant, as before
    the head energy became ``s · Σ_masked CE``."""
    with jax.enable_x64(True):
        probe = _tiny_bert(spike_only_inference(1))
        structure = _tiny_bert(spike_only_inference(max_free_depth(probe)))
        params = initialize_params(structure, jax.random.PRNGKey(0))
        batch = _batch()
        result = jax.device_get(node_energies(params, structure, batch))
        bp = jax.device_get(bp_signal_energies(params, structure, batch, SCALE))
    modules = [n for n in bp if n.endswith(("mha", "mlp1", "mlp2"))] + ["mlm_head"]
    aggregated = aggregate_map([result], modules, SCALE)
    count = float(np.sum(np.asarray(batch["loss_mask"])))
    valid = np.asarray(batch["loss_mask"]).sum(axis=-1) > 0
    for module in modules[:-1]:
        expected = float(np.asarray(bp[module])[valid].mean())
        values = aggregated["modules"][module]
        assert values["forward_error_energy"] == pytest.approx(expected, rel=1e-8)
        assert values["forward_error_energy_sum_ce"] == pytest.approx(
            expected * (count / SCALE) ** 2, rel=1e-8
        )


def test_spectrum_is_invariant_to_tiling_the_batch():
    from tests.test_inference_epc import bert_batch, tiny_bert

    from fabricpc.core import EPCInference
    from fabricpc.experiments.domain_forgetting.runner import make_regime_probe

    with jax.enable_x64(True):
        structure = tiny_bert(EPCInference(), dropout_rate=0.0)
        params = initialize_params(structure, jax.random.PRNGKey(3))
        probe = make_regime_probe(structure, iters=30)
        batch = bert_batch()
        tiled = {k: jnp.concatenate([v] * 4) for k, v in batch.items()}
        one = probe(params, batch, jax.random.PRNGKey(0))[0].host()
        four = probe(params, tiled, jax.random.PRNGKey(0))[0].host()
    assert four.lambda_max == pytest.approx(one.lambda_max, rel=1e-6)
    assert four.lambda_min == pytest.approx(one.lambda_min, rel=1e-6)
    assert four.negative_weight == pytest.approx(one.negative_weight, abs=1e-6)


def test_tiny_clip_marks_every_update_clipped(setup, tmp_path):
    checkpoint, data_dir = setup
    config = tiny_config(checkpoint, data_dir, tmp_path / "run", grad_clip=1e-6)
    run_domain_forgetting(config, verbose=False)
    log = _read_jsonl(tmp_path / "run" / "training_log_stage_1.jsonl")
    assert all(entry["clipped"] for entry in log)
    assert all(entry["grad_norm"] > 1e-6 for entry in log)


def test_grad_norm_excludes_frozen_modules(setup, tmp_path):
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
    backprop = make_gradient_fn("backprop", base.structure)
    first = []

    def recording(params, batch, key):
        result = backprop(params, batch, key)
        if not first:
            first.append(result)
        return result

    kwargs = dict(
        config=config,
        stage=2,
        special_ids=base.special_ids,
        gradient_fn=recording,
        verbose=False,
    )
    _, _, plain = train_stage(base.params, base.structure, stream, **kwargs)
    frozen = {name: 1.0 for name in base.params.nodes}
    frozen["mlm_head"] = 0.0
    _, _, gated = train_stage(
        base.params, base.structure, stream, multipliers=frozen, **kwargs
    )
    grads = first[0][0]
    full = float(optax.tree.norm(grads))
    head = float(optax.tree.norm(grads.nodes["mlm_head"]))
    assert plain[0]["grad_norm"] == pytest.approx(full, rel=1e-5)
    assert gated[0]["grad_norm"] == pytest.approx(
        math.sqrt(full**2 - head**2), rel=1e-5
    )


def test_regime_probe_does_not_change_pc_or_epc_training(setup, epc_run, tmp_path):
    checkpoint, data_dir = setup
    config, result = epc_run
    plain = replace(
        config,
        output_dir=str(tmp_path / "epc"),
        pc_regime_interval=0,
        pc_regime_iters=20,
    )
    again = run_domain_forgetting(plain, verbose=False)
    assert again["summary"]["stage_params_sha256"] == (
        result["summary"]["stage_params_sha256"]
    )
    spiking = tiny_config(
        checkpoint, data_dir, tmp_path / "spiking", mode="pc", pc_alignment_batches=1
    )
    probed = replace(spiking, output_dir=str(tmp_path / "spiking_probed"), **REGIME)
    left = run_domain_forgetting(spiking, verbose=False)
    right = run_domain_forgetting(probed, verbose=False)
    assert left["summary"]["stage_params_sha256"] == (
        right["summary"]["stage_params_sha256"]
    )
    alignment = _read_jsonl(tmp_path / "spiking" / "alignment.jsonl")
    assert alignment and all(r["head_energy"] == HEAD_ENERGY for r in alignment)
    assert all(r["eta_infer"] == 0.0 and "eta_controlled" not in r for r in alignment)
    assert len(_read_jsonl(tmp_path / "spiking_probed" / "regime.jsonl")) == 9


def test_pc_run_without_head_energy_refuses_resume(setup, tmp_path):
    checkpoint, data_dir = setup
    config = tiny_config(checkpoint, data_dir, tmp_path / "run", mode="pc")
    run_domain_forgetting(config, verbose=False)
    path = tmp_path / "run" / "run_config.json"
    legacy = json.loads(path.read_text())
    del legacy["pc"]["head_energy"]
    path.write_text(json.dumps(legacy))
    with pytest.raises(ValueError, match="cannot resume with a different setup"):
        run_domain_forgetting(config, resume=True, verbose=False)


def test_resumed_stage_is_probed_with_the_current_regime_setting(bp_runs, tmp_path):
    _, (probed, _) = bp_runs
    copy = tmp_path / "copy"
    shutil.copytree(probed.output_dir, copy)
    (copy / "stage_3.npz").unlink()
    resumed = replace(
        probed, output_dir=str(copy), pc_regime_eta=0.02, pc_regime_final=True
    )
    run_domain_forgetting(resumed, resume=True, verbose=False)
    records = _read_jsonl(copy / "regime.jsonl")
    assert [(r["stage"], r["update"]) for r in records] == [
        (stage, update) for stage in range(1, 4) for update in (0, 2, 3)
    ]
    for record in records:
        retrained = record["stage"] == 3
        assert record["eta_infer"] == (0.02 if retrained else 0.01)
        assert record["regime_final"] is retrained
        assert record["regime_interval"] == 2
        assert ("lambda_max_final" in record) is retrained


def test_probe_reports_head_energy_and_final_spectrum(setup, capsys):
    checkpoint, data_dir = setup
    report = probe_alignment(
        domain="news",
        checkpoint=str(checkpoint),
        data_dir=str(data_dir),
        num_batches=1,
        batch_size=4,
        mask_prob=0.3,
        pc_inference="epc",
        settings=[(1e-3, 3)],
        lanczos_iters=8,
        final_spectrum=True,
    )
    assert report["head_energy"] == HEAD_ENERGY and report["final_spectrum"]
    regime = report["settings"][0]["regime"]
    assert np.isfinite(regime["lambda_max_final"])
    assert regime["unstable_final"] is False
    assert f"{regime['lambda_max_final']:.4g}" in capsys.readouterr().out
    spiking = probe_alignment(
        domain="news",
        checkpoint=str(checkpoint),
        data_dir=str(data_dir),
        num_batches=1,
        batch_size=4,
        verbose=False,
    )
    assert spiking["head_energy"] == HEAD_ENERGY and "spectrum" not in spiking


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({**EPC, "pc_max_norm": 1.0}, "pc_max_norm does not apply to ePC"),
        ({"pc_regime_final": True}, "need pc_regime_interval > 0"),
    ],
)
def test_more_validation(overrides, message, tmp_path):
    with pytest.raises(ValueError, match=message):
        tiny_config("ck", "data", tmp_path, **overrides).validate()


def test_standard_pc_error_maps_need_an_explicit_rate():
    with pytest.raises(ValueError, match="explicit pc_eta_infer"):
        ErrorMapConfig().validate()
    ErrorMapConfig(settings=(SPIKE_ONLY,)).validate()
    ErrorMapConfig(pc_eta_infer=1e-4).validate()


def test_correlations_refuse_mixed_head_conventions(tmp_path):
    standard = {"setting": STANDARD_PC}
    spike = {"setting": SPIKE_ONLY}
    assert standard_map_convention({"maps": [spike]}) is None
    assert standard_map_convention(
        {"head_energy": HEAD_ENERGY, "maps": [spike, standard]}
    ) == (HEAD_ENERGY)
    with pytest.raises(ValueError, match="predates"):
        standard_map_convention({"maps": [spike, standard]})
    for name, convention in (("a", HEAD_ENERGY), ("b", "other")):
        (tmp_path / name).mkdir()
        (tmp_path / name / "error_maps.json").write_text(
            json.dumps({"head_energy": convention, "maps": [standard]})
        )
    with pytest.raises(ValueError, match="different head energies"):
        compute_correlations([str(tmp_path / "a"), str(tmp_path / "b")])
