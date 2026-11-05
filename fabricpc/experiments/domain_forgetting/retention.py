"""Diagonal Fisher of the EWC retention baseline (Kirkpatrick et al. 2017).

``F_k`` is computed from the weights at the end of stage ``k`` on the backprop
graph with dropout off, whichever learner produced them. It is the mean over
batches of the squared gradient of the batch's mean masked CE, on
``FISHER_WINDOWS`` evenly spaced train windows of the stage's domain
(``gates.spaced_batches``) in batches of ``GATE_MAP_BATCH``, with masks fixed by
``SeedSequence([seed, stage, FISHER_TAG])``. This is a batch approximation: a
true per-example diagonal Fisher would average squared per-example gradients,
so ``F_k`` is smaller by about the batch's token count where those gradients
are uncorrelated. The pass reads no optimizer or random state, so ``F_k``
depends only on the stage's snapshot, its domain and the run's seed.
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from fabricpc.core.types import GraphParams, GraphStructure
from fabricpc.experiments.domain_forgetting.gates import (
    GATE_MAP_BATCH,
    probe_seed,
    spaced_batches,
)
from fabricpc.experiments.domain_forgetting.runner import (
    RunConfig,
    build_inference,
    build_structure,
)
from fabricpc.experiments.forgetting_benchmark.sequential_trainer import params_hash
from fabricpc.training.train_backprop import _loss_and_grad_backprop_mlm_no_update

FISHER_WINDOWS = 256
FISHER_TAG = 0x666973


def fisher_structure(
    model_config: Mapping[str, Any], config: RunConfig
) -> GraphStructure:
    """The run's graph as backprop with dropout off."""
    run = replace(config, mode="backprop", dropout_rate=0.0)
    return build_structure(model_config, run, build_inference(run, model_config)[0])


def make_squared_gradient_fn(structure: GraphStructure) -> Any:
    """Return the jitted ``(params, batch) -> (squared mean-CE gradient, count)``."""

    @jax.jit
    def squared_gradient(params, batch):
        _, grads, count = _loss_and_grad_backprop_mlm_no_update(
            params, batch, structure, jax.random.PRNGKey(0)
        )
        return jax.tree_util.tree_map(jnp.square, grads), count

    return squared_gradient


def diagonal_fisher(
    params: GraphParams,
    batches: Sequence[Mapping[str, jax.Array]],
    squared_gradient: Any,
) -> tuple[GraphParams, int]:
    """Mean over batches with a masked token of the squared batch gradient."""
    total = None
    used = 0
    for batch in batches:
        squares, count = squared_gradient(params, batch)
        if float(count) == 0.0:
            continue
        total = (
            squares
            if total is None
            else jax.tree_util.tree_map(jnp.add, total, squares)
        )
        used += 1
    if total is None:
        raise ValueError("no batch has a masked token for the Fisher")
    return jax.tree_util.tree_map(lambda v: v / used, total), used


def stage_fisher(
    params: GraphParams,
    stream: np.ndarray,
    structure: GraphStructure,
    squared_gradient: Any,
    special_ids: Mapping[str, int],
    *,
    config: RunConfig,
    stage: int,
    domain: str,
) -> tuple[GraphParams, dict[str, Any]]:
    """Return ``F_stage`` and its provenance record."""
    started = time.time()
    seed = probe_seed(config.seed, stage, FISHER_TAG)
    batches, windows = spaced_batches(
        stream,
        structure,
        special_ids,
        num_windows=FISHER_WINDOWS,
        mask_prob=config.mask_prob,
        seed=seed,
    )
    fisher, used = diagonal_fisher(params, batches, squared_gradient)
    return fisher, {
        "stage": stage,
        "domain": domain,
        "params_sha256": params_hash(params),
        "fisher_sha256": params_hash(fisher),
        "windows": FISHER_WINDOWS,
        "first_window": int(windows[0]),
        "last_window": int(windows[-1]),
        "batch_size": GATE_MAP_BATCH,
        "batches": used,
        "mask_seed": seed,
        "module_traces": {
            name: float(sum(jnp.sum(leaf) for leaf in jax.tree_util.tree_leaves(node)))
            for name, node in fisher.nodes.items()
            if jax.tree_util.tree_leaves(node)
        },
        "seconds": time.time() - started,
    }
