"""Tests for the joint-training upper bound (tiny BERT, offline)."""

import json
import os
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

from fabricpc.experiments.domain_forgetting.localization import load_run
from fabricpc.experiments.domain_forgetting.runner import (
    num_stage_updates,
    run_domain_forgetting,
)
from tests.test_domain_forgetting_runner import DOMAINS, make_setup, tiny_config


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    return make_setup(tmp_path_factory.mktemp("joint"))


@pytest.fixture(scope="module")
def joint(setup, tmp_path_factory):
    checkpoint, data_dir = setup
    config = tiny_config(
        checkpoint, data_dir, tmp_path_factory.mktemp("joint_run"), training="joint"
    )
    return config, run_domain_forgetting(config, verbose=False)


def test_joint_run_trains_one_stage_on_all_domains(joint):
    config, result = joint
    output = Path(config.output_dir)
    for name in (
        "run_config.json",
        "checkpoint_manifest.json",
        "evaluations.jsonl",
        "masked_ce_matrix.json",
        "summary.json",
        "exposure.json",
        "stage_0.npz",
        "stage_1.npz",
        "training_log_stage_1.jsonl",
    ):
        assert (output / name).exists(), name
    assert not (output / "forgetting_matrices.json").exists()
    assert not (output / "stage_2.npz").exists()

    ce = np.asarray(result["matrices"]["masked_ce"])
    assert ce.shape == (2, len(DOMAINS))
    assert np.all(np.isfinite(ce))
    summary = result["summary"]
    assert summary["training"] == "joint"
    assert summary["domains"] == list(DOMAINS)
    for index, name in enumerate(DOMAINS):
        assert summary["initial_ce"][name] == pytest.approx(ce[0, index])
        assert summary["final_ce"][name] == pytest.approx(ce[1, index])
        assert summary["learning_gain_ce"][name] == pytest.approx(
            ce[0, index] - ce[1, index]
        )
    assert np.mean(list(summary["learning_gain_ce"].values())) > 0
    assert len(summary["stage_params_sha256"]) == 2

    manifest = json.loads((output / "checkpoint_manifest.json").read_text())
    assert [entry["domain"] for entry in manifest["stages"]] == [None, "joint"]
    matrix = json.loads((output / "masked_ce_matrix.json").read_text())
    assert matrix["stages"] == ["S0", "S1"]


def test_joint_exposure_covers_every_domain_budget(joint):
    config, result = joint
    exposure = result["exposure"]["joint"]
    budget = config.tokens_per_domain * len(DOMAINS)
    assert config.stage_tokens == budget
    assert exposure["stage_tokens"] == budget
    assert exposure["within_one_update"]
    assert abs(exposure["stream_tokens"] - budget) <= (
        exposure["max_stream_tokens_per_update"]
    )
    run_config = json.loads((Path(config.output_dir) / "run_config.json").read_text())
    assert run_config["config"]["training"] == "joint"
    seq_len = exposure["max_stream_tokens_per_update"] // config.batch_size + 1
    assert run_config["updates_per_stage"] == exposure["updates"]
    assert exposure["updates"] == num_stage_updates(budget, config.batch_size, seq_len)


def test_joint_resume_reuses_the_stage(joint):
    config, result = joint
    again = run_domain_forgetting(config, resume=True, verbose=False)
    assert again["summary"]["stage_params_sha256"] == (
        result["summary"]["stage_params_sha256"]
    )


def test_sequential_identity_has_no_training_key(setup, tmp_path):
    checkpoint, data_dir = setup
    sequential = tiny_config(checkpoint, data_dir, tmp_path)
    assert "training" not in sequential.identity()
    assert sequential.stage_domains == tuple(DOMAINS)
    assert sequential.stage_tokens == sequential.tokens_per_domain
    joint = replace(sequential, training="joint")
    assert joint.identity()["training"] == "joint"
    assert joint.stage_domains == ("joint",)


def test_joint_validation(setup, tmp_path):
    checkpoint, data_dir = setup
    config = tiny_config(checkpoint, data_dir, tmp_path, training="joint")
    config.validate()
    for overrides, match in (
        ({"training": "mixed"}, "training must be one of"),
        ({"domains": ("news",)}, "at least two domains"),
        (
            {"mode": "pc", "pc_alignment_batches": 2},
            "pc_alignment_batches needs training 'sequential'",
        ),
    ):
        with pytest.raises(ValueError, match=match):
            replace(config, **overrides).validate()


def test_analyses_refuse_joint_runs(joint):
    config, _ = joint
    with pytest.raises(ValueError, match="need a sequential run"):
        load_run(config.output_dir)


def test_cli_parses_training(setup, tmp_path):
    from examples import mlm_domain_forgetting as cli

    checkpoint, data_dir = setup
    args = cli.build_parser().parse_args(
        [
            "--checkpoint",
            str(checkpoint),
            "--data_dir",
            str(data_dir),
            "--output_dir",
            str(tmp_path),
            "--training",
            "joint",
        ]
    )
    assert cli.config_from_args(args).training == "joint"
    default = cli.build_parser().parse_args(
        ["--checkpoint", "c", "--output_dir", str(tmp_path)]
    )
    assert cli.config_from_args(default).training == "sequential"
