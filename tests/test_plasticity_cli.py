"""End-to-end checks of the plasticity benchmark CLI on offline synthetic tasks."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import json

import pytest

from fabricpc.experiments.forgetting_benchmark.run_plasticity_benchmark import (
    main,
)


def _argv(tmp_path, command, *extra):
    return [
        command,
        "--preset",
        "smoke",
        "--data-dir",
        str(tmp_path / "data"),
        "--output-dir",
        str(tmp_path / "out"),
        "--max-chars-per-task",
        "1500",
        "--max-samples",
        "160",
        "--epochs-per-task",
        "1",
        "--diag-every",
        "5",
        "--warmup-steps",
        "3",
        "--calib-batches",
        "2",
        "--infer-steps",
        "2",
        *extra,
    ]


def test_compare_runs_resumes_and_refuses_changed_hyperparameters(tmp_path):
    extra = (
        "--seeds",
        "0",
        "1",
        "--conditions",
        "surprise",
        "none",
        "replay_gates",
        "replay_gates_cross",
    )
    main(_argv(tmp_path, "compare", *extra))

    summary = json.loads((tmp_path / "out" / "comparison.json").read_text())
    runs = summary["per_run"]
    assert runs["replay_gates"]["0"] == runs["surprise"]["0"] | {
        key: runs["replay_gates"]["0"][key]
        for key in ("elapsed_seconds", "diagnostic_seconds")
    }
    assert "surprise - none" in summary["contrasts"]
    assert summary["contrasts"]["surprise - none"]["average_forgetting"]["n"] == 2

    main(_argv(tmp_path, "compare", *extra))

    with pytest.raises(ValueError):
        main(_argv(tmp_path, "compare", *extra, "--lr", "0.02"))


def test_lr_match_requires_a_match_seed_outside_the_comparison_seeds(tmp_path):
    with pytest.raises(ValueError):
        main(_argv(tmp_path, "lr-match", "--seeds", "0"))
    with pytest.raises(AssertionError):
        main(_argv(tmp_path, "lr-match", "--seeds", "0", "--match-seed", "0"))


def test_compare_matches_path_length_per_seed_and_validates_early(tmp_path):
    main(
        _argv(
            tmp_path,
            "compare",
            "--seeds",
            "0",
            "1",
            "--conditions",
            "influence",
            "none",
            "lr_matched",
            "lr_matched_influence",
        )
    )

    summary = json.loads((tmp_path / "out" / "comparison.json").read_text())
    assert set(summary["matches"]) == {"lr_matched", "lr_matched_influence"}
    for per_seed in summary["matches"].values():
        assert set(per_seed) == {"0", "1"}
        for match in per_seed.values():
            assert match["criterion"] == "path_length"
            assert abs(match["ratio"] - 1.0) < 0.2
    assert "surprise" in summary["per_run"]
    assert "influence - lr_matched_influence" in summary["contrasts"]
    assert "influence - surprise" not in summary["contrasts"]
    assert (
        summary["contrasts"]["influence - lr_matched"]["average_forgetting"]["p_value"]
        is None
    )


def test_commutator_reports_distance_and_noise_floor(tmp_path):
    main(
        _argv(
            tmp_path,
            "commutator",
            "--steps-per-task",
            "6",
            "--match-seed",
            "50",
            "--conditions",
            "surprise",
            "none",
            "replay_gates",
            "replay_gates_cross",
            "lr_matched",
        )
    )

    results = json.loads((tmp_path / "out" / "commutator_results.json").read_text())
    conditions = results["conditions"]
    assert set(conditions) == {
        "surprise",
        "none",
        "replay_gates",
        "replay_gates_cross",
        "lr_matched",
    }
    assert "path_ratio" in conditions["lr_matched"]["match"]
    for report in conditions.values():
        assert report["normalized_distance"] > 0
        assert report["noise_floor"]["normalized_distance"] > 0
    assert (
        conditions["replay_gates"]["normalized_distance"]
        == conditions["surprise"]["normalized_distance"]
    )
