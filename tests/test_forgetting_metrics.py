"""Tests for baseline forgetting metrics."""

import numpy as np
import pytest

from fabricpc.experiments.forgetting_benchmark.forgetting_metrics import (
    ForgettingResult,
    compute_backward_transfer,
    compute_final_retention,
    compute_forgetting_matrix,
    compute_forgetting_scores,
)


def test_forgetting_matrix_and_summary_metrics():
    performance = {
        "A": {"A": {"perplexity": 10.0}},
        "B": {"A": {"perplexity": 15.0}, "B": {"perplexity": 12.0}},
        "C": {
            "A": {"perplexity": 20.0},
            "B": {"perplexity": 14.0},
            "C": {"perplexity": 11.0},
        },
    }
    matrix = compute_forgetting_matrix(performance)

    assert matrix.shape == (3, 3)
    assert matrix[0, 2] == 20.0
    assert compute_backward_transfer(matrix) == pytest.approx(-6.0)
    assert compute_forgetting_scores(matrix) == {0: 10.0, 1: 2.0}
    assert compute_final_retention(matrix) == pytest.approx((10 / 20 + 12 / 14) / 2)

    result = ForgettingResult.from_performance_matrix(performance, ["A", "B", "C"])
    assert result.average_forgetting == pytest.approx(6.0)
    assert "Conclusion: forgetting" in result.summary()


def test_log_perplexity_forgetting_matches_known_matrix():
    performance = {
        "A": {"A": {"perplexity": 2.0}},
        "B": {"A": {"perplexity": 4.0}, "B": {"perplexity": 3.0}},
        "C": {
            "A": {"perplexity": 8.0},
            "B": {"perplexity": 6.0},
            "C": {"perplexity": 5.0},
        },
    }
    result = ForgettingResult.from_performance_matrix(performance, ["A", "B", "C"])

    assert result.log_matrix[0, 2] == pytest.approx(np.log(8.0))
    assert result.log_average_forgetting == pytest.approx(
        (np.log(8.0 / 2.0) + np.log(6.0 / 3.0)) / 2
    )
    assert result.log_bwt == pytest.approx(
        ((np.log(2.0) - np.log(8.0)) + (np.log(3.0) - np.log(6.0))) / 2
    )
