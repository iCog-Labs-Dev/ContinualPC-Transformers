"""Omitted-update influence of per-node PC updates on a probe loss.

I_i = L(theta + Delta_without_i) - L(theta + Delta_all): positive when node i's
update lowers the probe loss. For plain SGD the first-order estimate is
-<grad_i L, Delta_i> (Pruthi et al. 2020, Eq. 1 specialised to one step).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax
from scipy import stats

from fabricpc.core.types import GraphParams, GraphStructure
from fabricpc.training.plasticity_gates import weighted_gates
from fabricpc.training.train_autoregressive import _eval_step_autoregressive

MULTIPLIER_MAX = 2.0

LossFn = Callable[[GraphParams, Any], jax.Array]

__all__ = [
    "make_probe_loss_fn",
    "measure_update_influence",
    "first_order_influence",
    "influence_multipliers",
    "rank_agreement",
    "weighted_gates",
]


def make_probe_loss_fn(
    structure: GraphStructure, use_causal_mask: bool, key: jax.Array
) -> LossFn:
    """Jitted output cross-entropy in evaluation mode (no dropout) on one batch."""

    @jax.jit
    def loss_fn(params: GraphParams, batch: Any) -> jax.Array:
        loss, _ = _eval_step_autoregressive(
            params, structure, batch, key, use_causal_mask
        )
        return loss

    return loss_fn


def _mean_loss(loss_fn: LossFn, params: GraphParams, batches: Sequence[Any]) -> float:
    return float(np.mean([float(loss_fn(params, batch)) for batch in batches]))


def _updated_nodes(updates: GraphParams) -> list[str]:
    return [
        name for name, node in updates.nodes.items() if jax.tree_util.tree_leaves(node)
    ]


def _without_node(updates: GraphParams, name: str) -> GraphParams:
    nodes = dict(updates.nodes)
    nodes[name] = jax.tree_util.tree_map(jnp.zeros_like, updates.nodes[name])
    return GraphParams(nodes=nodes)


def measure_update_influence(
    params: GraphParams,
    candidate_updates: GraphParams,
    probe_batches: Sequence[Any],
    loss_fn: LossFn,
) -> Dict[str, float]:
    """Signed influence per updated node; zeros when the probe is empty.

    `params` is never mutated. loss_fn(params, batch) should be jitted so the
    N + 1 parameter trees share one compilation.
    """
    names = _updated_nodes(candidate_updates)
    if not probe_batches:
        return {name: 0.0 for name in names}
    loss_all = _mean_loss(
        loss_fn, optax.apply_updates(params, candidate_updates), probe_batches
    )
    influence: Dict[str, float] = {}
    for name in names:
        reduced = optax.apply_updates(params, _without_node(candidate_updates, name))
        influence[name] = _mean_loss(loss_fn, reduced, probe_batches) - loss_all
    return influence


def first_order_influence(
    params: GraphParams,
    candidate_updates: GraphParams,
    probe_batches: Sequence[Any],
    loss_fn: LossFn,
) -> Dict[str, float]:
    """-<grad_i L_probe, Delta_i> from one probe gradient; zeros if no probe."""
    names = _updated_nodes(candidate_updates)
    if not probe_batches:
        return {name: 0.0 for name in names}

    def mean_loss(p: GraphParams) -> jax.Array:
        return jnp.mean(jnp.stack([loss_fn(p, batch) for batch in probe_batches]))

    grads = jax.grad(mean_loss)(params)
    estimate: Dict[str, float] = {}
    for name in names:
        dots = jax.tree_util.tree_map(
            lambda g, u: jnp.sum(g * u),
            grads.nodes[name],
            candidate_updates.nodes[name],
        )
        estimate[name] = -float(sum(jax.tree_util.tree_leaves(dots)))
    return estimate


def influence_multipliers(influence: Mapping[str, float]) -> Dict[str, float]:
    """Positive influence over the mean positive influence, clipped to [0, 2].

    Non-positive entries map to 0; all ones when no entry is positive.
    """
    values = np.asarray(list(influence.values()), dtype=np.float64)
    if values.size and not np.all(np.isfinite(values)):
        raise ValueError("non-finite influence")
    positive = values[values > 0]
    if positive.size == 0:
        return {name: 1.0 for name in influence}
    mean_positive = float(positive.mean())
    return {
        name: (
            float(np.clip(value / mean_positive, 0.0, MULTIPLIER_MAX))
            if value > 0
            else 0.0
        )
        for name, value in influence.items()
    }


def rank_agreement(
    surprise: Mapping[str, float], influence: Mapping[str, float]
) -> tuple[float, list[str]]:
    """Spearman correlation of surprise and influence, plus the nodes whose
    surprise is above the median while influence is non-positive."""
    names = [name for name in surprise if name in influence]
    if len(names) < 2:
        return float("nan"), []
    s = np.asarray([surprise[n] for n in names], dtype=np.float64)
    i = np.asarray([influence[n] for n in names], dtype=np.float64)
    if np.ptp(s) == 0 or np.ptp(i) == 0:
        rho = float("nan")
    else:
        rho = float(stats.spearmanr(s, i)[0])
    median = float(np.median(s))
    flagged = [n for n, sv, iv in zip(names, s, i) if sv > median and iv <= 0]
    return rho, flagged
