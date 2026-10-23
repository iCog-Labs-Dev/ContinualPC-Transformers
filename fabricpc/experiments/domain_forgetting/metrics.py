"""Forgetting, transfer and PC = BP metrics on stage-by-domain matrices.

Every matrix here has one row per stage (S0 = the loaded model, S_k = after
training on the k-th domain) and one column per domain in training order, so
domain ``d`` (0-based) is learned at stage ``s_d = d + 1``.

``forgetting_metrics.py`` works on a square task-by-stage matrix of a
lower-is-better metric (its docstrings use perplexity): backward transfer is
``initial - final`` (negative = forgetting) and forgetting scores are
``max later - initial`` (positive = forgetting). CE and perplexity are passed
to it as they are; accuracy is negated first, so the same functions give
accuracy's sign conventions.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np

from fabricpc.experiments.forgetting_benchmark.forgetting_metrics import (
    compute_backward_transfer,
    compute_final_retention,
    compute_forgetting_scores,
)


def _as_stage_matrix(matrix: Any) -> np.ndarray:
    """Return a float64 ``(num_domains + 1, num_domains)`` stage matrix."""
    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != array.shape[1] + 1:
        raise ValueError(
            f"expected a (domains + 1, domains) stage matrix, got {array.shape}"
        )
    return array


def task_stage_matrix(matrix: Any) -> np.ndarray:
    """Return ``M[d, k] = matrix[k + 1, d]``, the square matrix of ``forgetting_metrics``."""
    return _as_stage_matrix(matrix)[1:].T.copy()


def degradation_matrix(matrix: Any, *, higher_is_better: bool) -> np.ndarray:
    """Return ``F[s, d]`` = degradation of domain ``d`` at stage ``s`` since ``s_d``.

    Defined (non-NaN) only for ``s > s_d``; positive means the model got worse.
    """
    array = _as_stage_matrix(matrix)
    sign = -1.0 if higher_is_better else 1.0
    out = np.full(array.shape, np.nan)
    for d in range(array.shape[1]):
        learned = d + 1
        out[learned + 1 :, d] = sign * (array[learned + 1 :, d] - array[learned, d])
    return out


def ce_forgetting_matrix(ce: Any) -> np.ndarray:
    """``CE[s, d] - CE[s_d, d]`` for ``s > s_d`` (positive = degradation)."""
    return degradation_matrix(ce, higher_is_better=False)


def perplexity_forgetting_matrix(ppl: Any) -> np.ndarray:
    """``PPL[s, d] - PPL[s_d, d]`` for ``s > s_d`` (positive = degradation)."""
    return degradation_matrix(ppl, higher_is_better=False)


def accuracy_forgetting_matrix(acc: Any) -> np.ndarray:
    """``acc[s_d, d] - acc[s, d]`` for ``s > s_d`` (positive = degradation)."""
    return degradation_matrix(acc, higher_is_better=True)


def bwt_ce(ce: Any) -> float:
    """Mean over ``d`` before the last of ``CE[s_d, d] - CE[last, d]``.

    Negative means forgetting, the same sign convention as ``bwt_accuracy``.
    """
    return compute_backward_transfer(task_stage_matrix(ce))


def bwt_accuracy(acc: Any) -> float:
    """Mean over ``d`` before the last of ``acc[last, d] - acc[s_d, d]``.

    Negative means forgetting.
    """
    return -compute_backward_transfer(task_stage_matrix(acc))


def max_forgetting(matrix: Any, *, higher_is_better: bool) -> list[float]:
    """Worst later degradation of each domain before the last (positive = worse)."""
    tasks = task_stage_matrix(matrix)
    if higher_is_better:
        tasks = -tasks
    scores = compute_forgetting_scores(tasks)
    return [scores[index] for index in range(len(tasks) - 1)]


def ce_final_retention(ce: Any) -> float:
    """Mean of ``CE[s_d, d] / CE[last, d]`` over earlier domains; one = unchanged."""
    return compute_final_retention(task_stage_matrix(ce))


def learning_gain(ce: Any) -> list[float]:
    """``CE[s_d - 1, d] - CE[s_d, d]`` per domain (positive = learned)."""
    array = _as_stage_matrix(ce)
    return [float(array[d, d] - array[d + 1, d]) for d in range(array.shape[1])]


def forward_transfer(ce: Any) -> list[float]:
    """``CE[0, d] - CE[s_d - 1, d]`` per domain (positive = earlier stages helped)."""
    array = _as_stage_matrix(ce)
    return [float(array[0, d] - array[d, d]) for d in range(array.shape[1])]


def _nan_to_none(matrix: np.ndarray) -> list[list[float | None]]:
    return [
        [None if not np.isfinite(value) else float(value) for value in row]
        for row in np.asarray(matrix)
    ]


def forgetting_matrices(ce: Any, ppl: Any, acc: Any) -> dict[str, Any]:
    """Return the three degradation matrices with undefined cells as ``None``."""
    return {
        "ce_forgetting": _nan_to_none(ce_forgetting_matrix(ce)),
        "perplexity_forgetting": _nan_to_none(perplexity_forgetting_matrix(ppl)),
        "accuracy_forgetting": _nan_to_none(accuracy_forgetting_matrix(acc)),
        "definition": {
            "ce_forgetting": "CE[s, d] - CE[s_d, d] for s > s_d",
            "perplexity_forgetting": "PPL[s, d] - PPL[s_d, d] for s > s_d",
            "accuracy_forgetting": "acc[s_d, d] - acc[s, d] for s > s_d",
            "sign": "positive = degradation",
        },
    }


def summarize_matrices(
    domains: Sequence[str], ce: Any, ppl: Any, acc: Any
) -> dict[str, Any]:
    """Return BWT, forgetting, learning gain and forward transfer per domain."""
    ce_array = _as_stage_matrix(ce)
    if ce_array.shape[1] != len(domains):
        raise ValueError("matrix columns must match the domains")
    earlier = list(domains[:-1])
    return {
        "domains": list(domains),
        "bwt_ce": bwt_ce(ce),
        "bwt_accuracy": bwt_accuracy(acc),
        "ce_final_retention": ce_final_retention(ce),
        "max_ce_forgetting": dict(
            zip(earlier, max_forgetting(ce, higher_is_better=False))
        ),
        "max_accuracy_forgetting": dict(
            zip(earlier, max_forgetting(acc, higher_is_better=True))
        ),
        "final_ce_forgetting": {
            name: float(ce_array[-1, d] - ce_array[d + 1, d])
            for d, name in enumerate(earlier)
        },
        "learning_gain_ce": dict(zip(domains, learning_gain(ce))),
        "forward_transfer_ce": dict(zip(domains, forward_transfer(ce))),
        "final_ce": dict(zip(domains, ce_array[-1].tolist())),
        "final_perplexity": dict(zip(domains, _as_stage_matrix(ppl)[-1].tolist())),
        "final_accuracy": dict(zip(domains, _as_stage_matrix(acc)[-1].tolist())),
        "sign_conventions": {
            "bwt_ce": "mean CE[s_d, d] - CE[last, d]; negative = forgetting",
            "bwt_accuracy": "mean acc[last, d] - acc[s_d, d]; negative = forgetting",
            "max_*_forgetting": "worst later degradation; positive = forgetting",
            "learning_gain_ce": "CE[s_d - 1, d] - CE[s_d, d]; positive = learned",
            "forward_transfer_ce": "CE[0, d] - CE[s_d - 1, d]; positive = helped",
        },
    }


def is_attention_key_bias(name: str) -> bool:
    """Whether a flat key is an attention key bias (``*_mha/biases/b_k``).

    Softmax attention is invariant to a key bias, so its true gradient is
    exactly zero and any change is floating-point noise.
    """
    node, _, leaf = name.partition("/")
    return node.endswith("_mha") and leaf.rsplit("/", 1)[-1] == "b_k"


def relative_param_distance(
    params: Mapping[str, np.ndarray],
    reference: Mapping[str, np.ndarray],
    *,
    exclude: Callable[[str], bool] | None = None,
) -> float:
    """Return ``||params - reference|| / ||reference||`` over the named leaves.

    Leaves for which ``exclude(name)`` is true are left out of both norms.
    """
    if set(params) != set(reference):
        raise ValueError("parameter sets have different leaves")
    diff = 0.0
    norm = 0.0
    for name in sorted(reference):
        if exclude is not None and exclude(name):
            continue
        left = np.asarray(params[name], dtype=np.float64)
        right = np.asarray(reference[name], dtype=np.float64)
        if left.shape != right.shape:
            raise ValueError(f"{name}: shape {left.shape} != {right.shape}")
        diff += float(np.sum((left - right) ** 2))
        norm += float(np.sum(right**2))
    return float(np.sqrt(diff / norm)) if norm > 0 else float(np.sqrt(diff))


def pc_bp_agreement(
    ce_pc: Any,
    ce_bp: Any,
    stage_params_pc: Sequence[Mapping[str, np.ndarray]],
    stage_params_bp: Sequence[Mapping[str, np.ndarray]],
) -> dict[str, Any]:
    """Compare a run (PC, or perturbed BP) with a reference BP run.

    Returns ``max_abs_ce_diff`` over all cells, ``max_rel_ce_diff``
    (``|ΔCE| / |CE_BP|``) and the per-stage ``||θ - θ_BP|| / ||θ_BP||`` over
    all parameters (``stage_param_rel_distance``) and without the attention
    key biases (``stage_param_rel_distance_no_key_bias``), whose true
    gradient is zero.
    """
    pc = _as_stage_matrix(ce_pc)
    bp = _as_stage_matrix(ce_bp)
    if pc.shape != bp.shape:
        raise ValueError("CE matrices have different shapes")
    if len(stage_params_pc) != len(stage_params_bp):
        raise ValueError("runs have different numbers of stages")
    abs_diff = np.abs(pc - bp)
    pairs = list(zip(stage_params_pc, stage_params_bp))
    distances = [relative_param_distance(left, right) for left, right in pairs]
    no_key_bias = [
        relative_param_distance(left, right, exclude=is_attention_key_bias)
        for left, right in pairs
    ]
    return {
        "max_abs_ce_diff": float(abs_diff.max()),
        "max_rel_ce_diff": float((abs_diff / np.abs(bp)).max()),
        "stage_param_rel_distance": distances,
        "max_stage_param_rel_distance": float(max(distances)) if distances else 0.0,
        "stage_param_rel_distance_no_key_bias": no_key_bias,
        "max_stage_param_rel_distance_no_key_bias": (
            float(max(no_key_bias)) if no_key_bias else 0.0
        ),
        "definition": {
            "stage_param_rel_distance": "||theta - theta_ref|| / ||theta_ref||, "
            "all parameters",
            "stage_param_rel_distance_no_key_bias": "the same without the "
            "attention key biases (*_mha/biases/b_k), whose true gradient is 0",
        },
    }
