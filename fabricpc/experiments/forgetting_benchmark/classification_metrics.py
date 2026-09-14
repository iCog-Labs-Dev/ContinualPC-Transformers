"""Classification forgetting metrics for sequential task benchmarks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def validate_accuracy_matrix(
    matrix: Sequence[Sequence[float]], num_tasks: int
) -> np.ndarray:
    """Validate and return an (N+1, N) accuracy matrix."""
    values = np.asarray(matrix, dtype=float)
    expected_shape = (num_tasks + 1, num_tasks)
    if values.shape != expected_shape:
        raise ValueError(
            f"accuracy matrix must have shape {expected_shape}, got {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("accuracy matrix values must be finite")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("accuracy matrix values must be in [0, 1]")
    return values


def compute_forgetting_metrics(
    matrix: Sequence[Sequence[float]], task_names: Sequence[str]
) -> dict[str, Any]:
    """Compute ACC, BWT, FWT, and per-task final metrics.

    Rows are stages: row 0 is before fine-tuning, row i+1 is after task i.
    Columns are tasks.
    """
    num_tasks = len(task_names)
    if num_tasks == 0:
        raise ValueError("at least one task is required")
    if len(set(task_names)) != num_tasks:
        raise ValueError("task names must be unique")

    values = validate_accuracy_matrix(matrix, num_tasks)
    final_row = values[num_tasks, :]
    bwt_values = [final_row[i] - values[i + 1, i] for i in range(num_tasks - 1)]
    fwt_values = [values[i, i] - values[0, i] for i in range(1, num_tasks)]

    per_task: dict[str, dict[str, float | None]] = {}
    for i, task_name in enumerate(task_names):
        per_task[task_name] = {
            "final_accuracy": float(final_row[i]),
            "initial_accuracy": float(values[0, i]),
            "learned_accuracy": float(values[i + 1, i]),
            "backward_transfer": (
                float(final_row[i] - values[i + 1, i]) if i < num_tasks - 1 else None
            ),
            "forward_transfer": (float(values[i, i] - values[0, i]) if i > 0 else None),
        }

    return {
        "ACC": float(np.mean(final_row)),
        "BWT": float(np.mean(bwt_values)) if bwt_values else 0.0,
        "FWT": float(np.mean(fwt_values)) if fwt_values else 0.0,
        "per_task": per_task,
        "BWT_per_task": {
            task_names[i]: float(bwt_values[i]) for i in range(len(bwt_values))
        },
        "FWT_per_task": {
            task_names[i]: float(values[i, i] - values[0, i])
            for i in range(1, num_tasks)
        },
    }
