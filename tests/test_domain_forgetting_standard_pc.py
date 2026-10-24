"""Tests for standard PC, gradient alignment and the probe (tiny BERT, offline)."""

import json
import os
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np
import pytest

from fabricpc.experiments.domain_forgetting import runner
from fabricpc.experiments.domain_forgetting.probe import parse_grid, probe_alignment
from fabricpc.experiments.domain_forgetting.runner import (
    compare_runs,
    load_base_model,
    run_domain_forgetting,
)
from tests.test_domain_forgetting_runner import DOMAINS, make_setup, tiny_config

STANDARD = {
    "mode": "pc",
    "pc_inference": "standard",
    "pc_eta_infer": 0.1,
    "pc_infer_steps": 20,
    "pc_forward_updates": "none",
}


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    return make_setup(tmp_path_factory.mktemp("standard_pc"))


@pytest.fixture(scope="module")
def runs(setup, tmp_path_factory):
    checkpoint, data_dir = setup
    root = tmp_path_factory.mktemp("standard_runs")
    bp = tiny_config(checkpoint, data_dir, root / "bp")
    standard = tiny_config(
        checkpoint, data_dir, root / "standard", pc_alignment_batches=2, **STANDARD
    )
    return (
        (bp, run_domain_forgetting(bp, verbose=False)),
        (standard, run_domain_forgetting(standard, verbose=False)),
    )


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_standard_pc_runs_all_stages_and_is_not_backprop(runs):
    (bp_config, bp), (config, result) = runs
    output = Path(config.output_dir)
    for name in (
        "run_config.json",
        "checkpoint_manifest.json",
        "evaluations.jsonl",
        "masked_ce_matrix.json",
        "forgetting_matrices.json",
        "summary.json",
        "alignment.jsonl",
        *(f"stage_{s}.npz" for s in range(4)),
        *(f"training_log_stage_{s}.jsonl" for s in range(1, 4)),
    ):
        assert (output / name).exists(), name
    ce = np.asarray(result["matrices"]["masked_ce"])
    assert ce.shape == (4, 3) and np.all(np.isfinite(ce))
    for index, name in enumerate(DOMAINS):
        assert ce[index + 1, index] < ce[index, index], name
    run_config = json.loads((output / "run_config.json").read_text())
    assert run_config["pc"]["inference"] == "standard"
    assert run_config["pc"]["infer_steps"] == 20
    assert run_config["pc"]["forward_updates"] == "none"
    assert "pc_alignment_batches" not in run_config["config"]
    agreement = compare_runs(output, bp_config.output_dir)
    assert agreement["stage_param_rel_distance"][0] == 0.0
    assert agreement["max_stage_param_rel_distance"] > 1e-3
    for stage in range(1, 4):
        log = _read_jsonl(output / f"training_log_stage_{stage}.jsonl")
        assert len(log) == 3
        for entry in log:
            assert np.isfinite(entry["pc_total_energy"])
            assert entry["pc_energy_rises"] >= 0
            assert entry["pc_energy_increased"] is False
            assert entry["pc_energy_end"] < entry["pc_energy_start"]
            assert entry["pc_energy_rel_decrease"] == pytest.approx(
                1 - entry["pc_energy_end"] / entry["pc_energy_start"]
            )


def test_alignment_records_bracket_each_stage(runs):
    _, (config, result) = runs
    output = Path(config.output_dir)
    records = _read_jsonl(output / "alignment.jsonl")
    hashes = result["summary"]["stage_params_sha256"]
    assert [(r["stage"], r["when"]) for r in records] == [
        (s, w) for s in range(1, 4) for w in ("start", "end")
    ]
    base = load_base_model(config)
    modules = [
        name
        for name in base.structure.node_order
        if jax.tree_util.tree_leaves(base.params.nodes[name])
    ]
    for record in records:
        stage = record["stage"]
        assert record["domain"] == DOMAINS[stage - 1]
        expected = hashes[stage - 1] if record["when"] == "start" else hashes[stage]
        assert record["params_sha256"] == expected
        assert record["num_batches"] == 2
        assert record["pc_inference"] == "standard"
        assert record["module_order"] == modules
        assert set(record["modules"]) == set(modules)
        head = record["modules"]["mlm_head"]
        assert head["cosine"] > 0.99
        assert head["rel_norm_ratio"] == pytest.approx(1.0)
        for row in record["modules"].values():
            assert set(row) == {
                "cosine",
                "cosine_min",
                "pc_norm",
                "bp_norm",
                "norm_ratio",
                "rel_norm_ratio",
            }
            assert -1.0 <= row["cosine_min"] <= row["cosine"] <= 1.0 + 1e-6
        assert record["min_cosine"] == min(
            r["cosine"] for r in record["modules"].values()
        )


def test_alignment_does_not_change_training(runs, tmp_path):
    _, (config, result) = runs
    quiet = replace(config, output_dir=str(tmp_path / "run"), pc_alignment_batches=0)
    again = run_domain_forgetting(quiet, verbose=False)
    assert again["summary"]["stage_params_sha256"] == (
        result["summary"]["stage_params_sha256"]
    )
    assert not (tmp_path / "run" / "alignment.jsonl").exists()


def test_overwrite_removes_alignment(tmp_path):
    (tmp_path / "alignment.jsonl").write_text("stale")
    (tmp_path / "notes.txt").write_text("keep")
    assert runner.remove_run_artifacts(tmp_path) == ["alignment.jsonl"]
    assert (tmp_path / "notes.txt").exists()


def test_standard_validation(setup, tmp_path):
    checkpoint, data_dir = setup
    config = tiny_config(checkpoint, data_dir, tmp_path / "run", **STANDARD)
    config.validate()
    for overrides, match in (
        ({"pc_eta_infer": 0.0}, "pc_eta_infer > 0"),
        ({"pc_infer_steps": None}, "pc_infer_steps >= 1"),
        ({"pc_infer_steps": 0}, "pc_infer_steps >= 1"),
        ({"pc_forward_updates": "initial"}, "pc_forward_updates 'none'"),
        ({"pc_spike_eta": 0.5}, "pc_spike_eta"),
        ({"pc_spike_depth": "shortest"}, "pc_spike_depth"),
        ({"pc_alignment_batches": -1}, "non-negative"),
        ({"mode": "backprop", "pc_alignment_batches": 2}, "needs mode 'pc'"),
    ):
        with pytest.raises(ValueError, match=match):
            replace(config, **overrides).validate()


def test_cli_resolves_forward_updates_by_inference(setup, tmp_path):
    from examples import mlm_domain_forgetting as cli

    checkpoint, data_dir = setup
    common = [
        "--checkpoint",
        str(checkpoint),
        "--data_dir",
        str(data_dir),
        "--output_dir",
        str(tmp_path),
        "--mode",
        "pc",
    ]
    parser = cli.build_parser()
    standard = cli.config_from_args(
        parser.parse_args(
            [
                *common,
                "--pc_inference",
                "standard",
                "--pc_eta_infer",
                "0.1",
                "--pc_infer_steps",
                "20",
                "--pc_alignment_batches",
                "3",
            ]
        )
    )
    assert standard.pc_forward_updates == "none"
    assert standard.pc_alignment_batches == 3
    standard.validate()
    spiking = cli.config_from_args(parser.parse_args(common))
    assert spiking.pc_forward_updates == "initial"
    assert spiking.pc_inference == "spiking"


def test_probe_spiking_is_exact_and_small_eta_standard_matches_at_the_head(setup):
    checkpoint, data_dir = setup
    common = {
        "domain": "news",
        "checkpoint": str(checkpoint),
        "data_dir": str(data_dir),
        "num_batches": 2,
        "batch_size": 4,
        "mask_prob": 0.3,
        "verbose": False,
    }
    spiking = probe_alignment(**common)
    (setting,) = spiking["settings"]
    assert spiking["forward_updates"] == "initial"
    assert setting["min_cosine"] >= 0.999
    for row in setting["modules"].values():
        assert row["cosine_min"] >= 0.999
        assert row["norm_ratio"] == pytest.approx(1.0, rel=1e-3)
    standard = probe_alignment(
        **common, pc_inference="standard", settings=[(0.01, 200), (0.1, 20)]
    )
    assert standard["forward_updates"] == "none"
    assert [s["infer_steps"] for s in standard["settings"]] == [200, 20]
    for result in standard["settings"]:
        assert result["modules"]["mlm_head"]["cosine"] > 0.99
        assert result["modules"]["mlm_transform"]["cosine"] > 0.99
        assert result["min_cosine"] < 0.999
    with pytest.raises(ValueError, match="pc_eta_infer > 0"):
        probe_alignment(**common, pc_inference="standard")


def test_probe_cli_grid_and_stage_snapshot(setup, runs, tmp_path, capsys):
    from examples import mlm_domain_forgetting as cli

    checkpoint, data_dir = setup
    _, (config, result) = runs
    out = tmp_path / "probe.json"
    report = cli.main(
        [
            "probe",
            "--run_dir",
            config.output_dir,
            "--stage",
            "2",
            "--domain",
            "reviews",
            "--batches",
            "1",
            "--batch_size",
            "4",
            "--pc_inference",
            "standard",
            "--grid",
            "0.1:5, 0.05:10",
            "--skip_shape_check",
            "--out",
            str(out),
        ]
    )
    assert report["params_sha256"] == result["summary"]["stage_params_sha256"][2]
    assert [(s["eta_infer"], s["infer_steps"]) for s in report["settings"]] == [
        (0.1, 5),
        (0.05, 10),
    ]
    assert json.loads(out.read_text())["settings"][1]["infer_steps"] == 10
    lines = capsys.readouterr().out.strip().splitlines()
    modules = len(report["settings"][0]["modules"])
    assert len(lines) == 2 * (modules + 1)
    assert lines[modules].startswith("eta=0.1 steps=5 mean")
    assert parse_grid("0.1:20,0.2:50") == [(0.1, 20), (0.2, 50)]
    for bad in ("0.1", "0.1:x", "0.1:0", ""):
        with pytest.raises(ValueError):
            parse_grid(bad)
    with pytest.raises(ValueError, match="together"):
        probe_alignment(domain="news", checkpoint=str(checkpoint), stage=1)


def test_standard_pc_flags_diverging_inference(setup, tmp_path, capsys):
    checkpoint, data_dir = setup
    config = tiny_config(
        checkpoint,
        data_dir,
        tmp_path / "diverging",
        **{**STANDARD, "pc_eta_infer": 0.5, "pc_infer_steps": 50},
    )
    run_domain_forgetting(config, verbose=True)

    entries = [
        entry
        for stage in range(1, 4)
        for entry in _read_jsonl(
            tmp_path / "diverging" / f"training_log_stage_{stage}.jsonl"
        )
    ]
    increased = [entry for entry in entries if entry["pc_energy_increased"]]
    assert increased
    assert all(entry["pc_energy_rises"] > 0 for entry in increased)
    assert "inference is diverging" in capsys.readouterr().out
