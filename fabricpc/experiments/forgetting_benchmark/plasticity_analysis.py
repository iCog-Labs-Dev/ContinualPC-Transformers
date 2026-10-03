"""Per-run metrics, paired seed contrasts and learning-rate matching."""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy import stats

from fabricpc.experiments.forgetting_benchmark.forgetting_metrics import (
    ForgettingResult,
)
from fabricpc.experiments.statistics import cohens_d, paired_ttest

# influence - surprise is deliberately absent: the influence gate also shrinks
# the effective step, so only the lr-matched contrasts isolate the influence.
CONTRASTS = (
    ("surprise", "none"),
    ("surprise", "lr_matched"),
    ("surprise", "replay_gates_cross"),
    ("surprise", "replay_gates"),
    ("influence", "lr_matched_influence"),
    ("influence", "lr_matched"),
    ("surprise", "lr_matched_learning"),
)
PRIMARY_CONTRAST = ("surprise", "none")
PRIMARY_METRIC = "average_forgetting"
CONTRAST_NOTE = (
    "Primary: surprise - none on average_forgetting. All other contrasts and "
    "metrics are descriptive and uncorrected for multiple comparisons. "
    "influence - surprise is confounded with a smaller effective step; the "
    "influence effect is read from influence - lr_matched_influence."
)
METRICS = (
    "average_forgetting",
    "log_average_forgetting",
    "bwt",
    "final_retention",
    "avg_final_ppl",
    "new_task_log_ppl",
    "immediate_train_loss",
    "path_length",
)
MIN_SEEDS_FOR_INTERVAL = 5


def run_metrics(
    performance_matrix: Mapping[str, Any],
    task_names: Sequence[str],
    run_record: Mapping[str, Any],
) -> dict[str, Any]:
    """Retention, learning and cost summary of one sequential run."""
    forgetting = ForgettingResult.from_performance_matrix(
        dict(performance_matrix), list(task_names)
    )
    diagonal = [
        math.log(performance_matrix[name][name]["perplexity"]) for name in task_names
    ]
    return {
        "average_forgetting": forgetting.average_forgetting,
        "log_average_forgetting": forgetting.log_average_forgetting,
        "bwt": forgetting.bwt,
        "log_bwt": forgetting.log_bwt,
        "final_retention": forgetting.final_retention,
        "avg_final_ppl": forgetting.avg_final,
        "new_task_log_ppl": float(np.mean(diagonal)),
        "immediate_train_loss": (
            float(np.mean(run_record["immediate_train_loss"]))
            if run_record.get("immediate_train_loss")
            else None
        ),
        "final_ppl_per_task": {
            name: float(forgetting.matrix[i, -1]) for i, name in enumerate(task_names)
        },
        "path_length": run_record.get("path_length"),
        "elapsed_seconds": run_record.get("elapsed_seconds"),
        "diagnostic_seconds": run_record.get("diagnostic_seconds"),
        "num_updates": run_record.get("num_updates"),
        "num_tokens": run_record.get("num_tokens"),
    }


def paired_contrast(a: Sequence[float], b: Sequence[float]) -> dict[str, Any]:
    """Signed per-seed differences a - b.

    t, p, Cohen's d and a t interval appear only for n >= MIN_SEEDS_FOR_INTERVAL;
    with fewer seeds only the signed differences are given. Identical
    differences (zero variance) are reported as such, without test statistics.
    """
    a_arr, b_arr = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    diffs = a_arr - b_arr
    n = int(diffs.size)
    mean = float(diffs.mean()) if n else None
    out: dict[str, Any] = {
        "n": n,
        "signed_differences": diffs.tolist(),
        "mean_difference": mean,
        "t_statistic": None,
        "p_value": None,
        "cohens_d": None,
        "ci95": None,
        "zero_variance": bool(n >= 2 and np.ptp(diffs) == 0),
    }
    if n < MIN_SEEDS_FOR_INTERVAL:
        return out
    if out["zero_variance"]:
        out["ci95"] = [mean, mean]
        return out
    test = paired_ttest(a_arr, b_arr)
    half = float(stats.t.ppf(0.975, n - 1) * diffs.std(ddof=1) / math.sqrt(n))
    out.update(
        {
            "t_statistic": test.t_statistic,
            "p_value": test.p_value,
            "cohens_d": cohens_d(a_arr, b_arr).d,
            "ci95": [mean - half, mean + half],
        }
    )
    return out


def summarize_contrasts(
    per_run: Mapping[str, Mapping[int, Mapping[str, Any]]],
    contrasts: Sequence[tuple[str, str]] = CONTRASTS,
    metrics: Sequence[str] = METRICS,
) -> dict[str, Any]:
    """Paired contrasts over the seeds shared by both conditions."""
    summary: dict[str, Any] = {}
    for first, second in contrasts:
        if first not in per_run or second not in per_run:
            continue
        seeds = sorted(set(per_run[first]) & set(per_run[second]))
        summary[f"{first} - {second}"] = {
            "seeds": seeds,
            "primary": (first, second) == PRIMARY_CONTRAST,
            **{
                metric: paired_contrast(
                    [per_run[first][s][metric] for s in seeds],
                    [per_run[second][s][metric] for s in seeds],
                )
                for metric in metrics
            },
        }
    return summary


def match_lr_scale(
    measure: Callable[[float], float],
    target: float,
    *,
    tolerance: float = 0.02,
    max_iterations: int = 8,
    bracket: tuple[float, float] = (0.1, 2.0),
) -> dict[str, Any]:
    """Find an lr scale whose measured quantity equals `target`.

    `measure(scale)` runs the plain-PC condition on the smoke budget and
    returns path length (or another monotone quantity). Secant steps in
    log-scale from the bracket ends; every evaluation is logged.
    """
    log: list[dict[str, float]] = []

    def evaluate(scale: float) -> float:
        value = float(measure(scale))
        log.append({"scale": scale, "value": value, "target": target})
        return value

    lo, hi = bracket
    f_lo, f_hi = evaluate(lo) - target, evaluate(hi) - target
    best = min(log, key=lambda r: abs(r["value"] - target))
    for _ in range(max_iterations):
        if abs(best["value"] - target) <= tolerance * abs(target):
            break
        if f_hi == f_lo:
            break
        # Secant on log(scale); clamp inside the bracket.
        log_scale = math.log(lo) - f_lo * (math.log(hi) - math.log(lo)) / (f_hi - f_lo)
        scale = float(np.clip(math.exp(log_scale), min(lo, hi) * 0.5, max(lo, hi) * 2))
        value = evaluate(scale)
        lo, f_lo, hi, f_hi = hi, f_hi, scale, value - target
        best = min(log, key=lambda r: abs(r["value"] - target))
    return {
        "scale": best["scale"],
        "value": best["value"],
        "target": target,
        "relative_error": abs(best["value"] - target) / abs(target),
        "search": log,
    }
