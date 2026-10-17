"""Result export helpers for the catastrophic forgetting benchmark."""

from __future__ import annotations

import csv
import json
from io import StringIO
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def _jsonable(value: Any) -> Any:
    """Convert benchmark result values into JSON-compatible objects."""
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def export_results_json(results: dict[str, Any], path: str | Path) -> Path:
    """Write benchmark results as machine-readable JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_jsonable(results), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_path


def export_per_stage_performance_csv(
    performance_matrix: dict[str, dict[str, dict[str, float]]],
    path: str | Path,
) -> Path:
    """Write CE, perplexity, and accuracy for every evaluated task/stage.

    The long format avoids hiding the triangular nature of a sequential
    benchmark and can be imported directly by a tracker or spreadsheet.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stream = StringIO()
    writer = csv.DictWriter(
        stream,
        fieldnames=[
            "after_task",
            "evaluated_task",
            "cross_entropy",
            "perplexity",
            "accuracy",
        ],
    )
    writer.writeheader()
    for after_task, task_metrics in performance_matrix.items():
        for evaluated_task, metrics in task_metrics.items():
            writer.writerow(
                {
                    "after_task": after_task,
                    "evaluated_task": evaluated_task,
                    "cross_entropy": metrics.get("loss"),
                    "perplexity": metrics.get("perplexity"),
                    "accuracy": metrics.get("accuracy"),
                }
            )
    output_path.write_text(stream.getvalue(), encoding="utf-8")
    return output_path


def generate_latex_table(
    forgetting_matrix: np.ndarray,
    task_names: Sequence[str],
    metric_name: str = "Perplexity",
) -> str:
    """Return a LaTeX table for a task-by-training-stage metric matrix."""
    matrix = np.asarray(forgetting_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("forgetting_matrix must be a square two-dimensional array")
    if len(task_names) != matrix.shape[0]:
        raise ValueError("task_names must match forgetting_matrix dimensions")

    escaped_names = [name.replace("_", r"\_") for name in task_names]
    columns = "l" + "r" * len(task_names)
    lines = [
        rf"\begin{{tabular}}{{{columns}}}",
        r"\toprule",
        "Evaluated task & " + " & ".join(escaped_names) + r" \\",
        r"\midrule",
    ]
    for name, row in zip(escaped_names, matrix):
        cells = ["--" if np.isnan(value) else f"{value:.2f}" for value in row]
        lines.append(name + " & " + " & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            rf"% Metric: {metric_name}",
        ]
    )
    return "\n".join(lines)


def generate_summary_report(results: dict[str, Any]) -> str:
    """Return a compact human-readable benchmark summary."""
    lines = [
        "Objective 2: Catastrophic Forgetting Benchmark",
        f"Tasks: {results.get('num_tasks', 'unknown')}",
        f"Backward transfer: {float(results.get('bwt', 0.0)):.4f}",
        f"Average forgetting: {float(results.get('average_forgetting', 0.0)):.4f}",
        f"Final retention: {float(results.get('final_retention', 1.0)):.2%}",
        f"Average final perplexity: {float(results.get('avg_final_ppl', 0.0)):.4f}",
        f"Conclusion: {results.get('conclusion', 'not available')}",
        f"Next diagnostic: {results.get('next_diagnostic', 'not available')}",
    ]
    return "\n".join(lines) + "\n"
