import numpy as np
import pytest

from fabricpc.experiments.forgetting_benchmark.classification_metrics import (
    compute_forgetting_metrics,
    validate_accuracy_matrix,
)


def test_forgetting_metric_formulas_include_per_task_values():
    matrix = np.array(
        [
            [0.50, 0.40, 0.30],
            [0.80, 0.45, 0.35],
            [0.70, 0.75, 0.40],
            [0.60, 0.65, 0.90],
        ]
    )

    metrics = compute_forgetting_metrics(matrix, ["sst2", "imdb", "yelp_polarity"])

    assert metrics["ACC"] == pytest.approx((0.60 + 0.65 + 0.90) / 3)
    assert metrics["BWT"] == pytest.approx(((0.60 - 0.80) + (0.65 - 0.75)) / 2)
    assert metrics["FWT"] == pytest.approx(((0.45 - 0.40) + (0.40 - 0.30)) / 2)
    assert metrics["per_task"]["sst2"]["final_accuracy"] == pytest.approx(0.60)
    assert metrics["per_task"]["imdb"]["backward_transfer"] == pytest.approx(-0.10)
    assert metrics["per_task"]["yelp_polarity"]["forward_transfer"] == pytest.approx(
        0.10
    )


def test_forgetting_metric_single_task_uses_zero_transfer_means():
    metrics = compute_forgetting_metrics(np.array([[0.25], [0.75]]), ["sst2"])

    assert metrics["ACC"] == pytest.approx(0.75)
    assert metrics["BWT"] == pytest.approx(0.0)
    assert metrics["FWT"] == pytest.approx(0.0)


def test_accuracy_matrix_validation_rejects_bad_shapes_and_values():
    validate_accuracy_matrix([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], 2)

    with pytest.raises(ValueError, match="shape"):
        validate_accuracy_matrix([[0.1, 0.2]], 2)
    with pytest.raises(ValueError, match="finite"):
        validate_accuracy_matrix([[0.1], [float("nan")]], 1)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        validate_accuracy_matrix([[0.1], [1.1]], 1)
