"""Metrics for the sequential catastrophic-forgetting baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def compute_forgetting_matrix(
    performance_matrix: dict[str, dict[str, dict[str, float]]],
    metric: str = "perplexity",
) -> np.ndarray:
    """Return task performance after every observed training stage."""
    tasks = list(performance_matrix)
    matrix = np.full((len(tasks), len(tasks)), np.nan)
    for stage_idx, stage_name in enumerate(tasks):
        for task_idx, task_name in enumerate(tasks[: stage_idx + 1]):
            matrix[task_idx, stage_idx] = performance_matrix[stage_name][task_name][
                metric
            ]
    return matrix


def compute_backward_transfer(matrix: np.ndarray) -> float:
    """Mean initial-minus-final perplexity; negative means forgetting."""
    if len(matrix) < 2:
        return 0.0
    changes = [
        matrix[index, index] - matrix[index, -1] for index in range(len(matrix) - 1)
    ]
    return float(np.mean(changes))


def compute_forgetting_scores(matrix: np.ndarray) -> dict[int, float]:
    """Worst later perplexity increase for each completed task."""
    return {
        index: float(np.nanmax(matrix[index, index + 1 :]) - matrix[index, index])
        for index in range(len(matrix) - 1)
    }


def compute_final_retention(matrix: np.ndarray) -> float:
    """Mean ratio of initial to final perplexity; one means no change."""
    if len(matrix) < 2:
        return 1.0
    return float(
        np.mean(
            [
                matrix[index, index] / matrix[index, -1]
                for index in range(len(matrix) - 1)
            ]
        )
    )


@dataclass
class ForgettingResult:
    matrix: np.ndarray
    task_names: list[str]
    bwt: float
    forgetting_scores: dict[int, float]
    avg_final: float
    average_forgetting: float
    final_retention: float
    log_matrix: Optional[np.ndarray] = None
    log_bwt: float = 0.0
    log_average_forgetting: float = 0.0

    @classmethod
    def from_performance_matrix(
        cls,
        performance_matrix: dict[str, dict[str, dict[str, float]]],
        task_names: list[str],
        metric: str = "perplexity",
    ) -> "ForgettingResult":
        if metric != "perplexity":
            raise ValueError("The baseline forgetting summary uses perplexity")
        matrix = compute_forgetting_matrix(performance_matrix, metric)
        scores = compute_forgetting_scores(matrix)
        log_matrix = np.log(matrix)
        log_scores = compute_forgetting_scores(log_matrix)
        return cls(
            matrix=matrix,
            task_names=task_names,
            bwt=compute_backward_transfer(matrix),
            forgetting_scores=scores,
            avg_final=float(np.nanmean(matrix[:, -1])),
            average_forgetting=float(np.mean(list(scores.values()))) if scores else 0.0,
            final_retention=compute_final_retention(matrix),
            log_matrix=log_matrix,
            log_bwt=compute_backward_transfer(log_matrix),
            log_average_forgetting=(
                float(np.mean(list(log_scores.values()))) if log_scores else 0.0
            ),
        )

    def summary(self) -> str:
        conclusion, next_diagnostic = self.conclusion()
        return (
            f"Forgetting Analysis ({len(self.task_names)} tasks)\n"
            f"  Avg final perplexity: {self.avg_final:.2f}\n"
            f"  Backward transfer: {self.bwt:.4f}\n"
            f"  Average forgetting: {self.average_forgetting:.4f}\n"
            f"  Final retention: {self.final_retention:.2%}\n"
            f"  Conclusion: {conclusion}\n"
            f"  Next diagnostic: {next_diagnostic}\n"
        )

    def conclusion(self) -> tuple[str, str]:
        """Return the baseline conclusion and its next reversible diagnostic."""
        if self.bwt < -0.1:
            return (
                "forgetting appeared",
                "inspect the largest earlier-task perplexity increase before selecting a mitigation",
            )
        return (
            "no material forgetting appeared",
            "increase the training budget before deciding that mitigation is unnecessary",
        )
