"""Tests for per-run metrics, paired contrasts and lr matching."""

import math

import pytest

from fabricpc.experiments.forgetting_benchmark.plasticity_analysis import (
    CONTRAST_NOTE,
    match_lr_scale,
    paired_contrast,
    run_metrics,
    summarize_contrasts,
)

PERFORMANCE = {
    "A": {"A": {"perplexity": 2.0}},
    "B": {"A": {"perplexity": 4.0}, "B": {"perplexity": 3.0}},
}


def test_run_metrics_reports_retention_and_learning():
    metrics = run_metrics(PERFORMANCE, ["A", "B"], {"path_length": 7.0})

    assert metrics["average_forgetting"] == pytest.approx(2.0)
    assert metrics["new_task_log_ppl"] == pytest.approx(
        (math.log(2.0) + math.log(3.0)) / 2
    )
    assert metrics["path_length"] == 7.0


def test_paired_contrast_reports_statistics_only_with_enough_seeds():
    few = paired_contrast([1.0, 2.0, 3.0, 4.0], [0.5, 1.0, 2.0, 3.5])
    many = paired_contrast([1, 2, 3, 4, 5], [0.5, 1, 2, 2, 4])

    assert few["signed_differences"] == [0.5, 1.0, 1.0, 0.5]
    assert few["mean_difference"] == pytest.approx(0.75)
    assert few["p_value"] is None and few["ci95"] is None and few["cohens_d"] is None
    assert many["ci95"][0] < many["mean_difference"] < many["ci95"][1]
    assert many["p_value"] is not None


def test_zero_variance_differences_are_exact_without_nan():
    same = paired_contrast([1.0] * 5, [1.0] * 5)
    constant = paired_contrast([2.0] * 5, [1.0] * 5)

    for out in (same, constant):
        assert out["zero_variance"] and out["p_value"] is None
        assert out["ci95"] == [out["mean_difference"]] * 2
    assert same["mean_difference"] == 0.0


def test_summarize_contrasts_pairs_shared_seeds_only():
    per_run = {
        "surprise": {0: {"bwt": -1.0}, 1: {"bwt": -2.0}},
        "none": {1: {"bwt": -3.0}, 2: {"bwt": -4.0}},
    }
    out = summarize_contrasts(per_run, metrics=("bwt",))
    assert "influence - surprise" not in summarize_contrasts(
        {"influence": {0: {"bwt": 1.0}}, "surprise": {0: {"bwt": 0.0}}},
        metrics=("bwt",),
    )

    assert out["surprise - none"]["seeds"] == [1]
    assert out["surprise - none"]["bwt"]["signed_differences"] == [1.0]
    assert out["surprise - none"]["primary"] is True
    assert "descriptive" in CONTRAST_NOTE


def test_match_lr_scale_finds_root_of_monotone_measure():
    result = match_lr_scale(lambda s: 10.0 * s**0.9, target=5.0)

    assert result["relative_error"] <= 0.02
    assert result["scale"] == pytest.approx((0.5) ** (1 / 0.9), rel=0.05)
    assert len(result["search"]) >= 2
