"""Order-dependence (commutator) diagnostics for A->B versus B->A training.

Both orders start from one theta_0 and consume the same per-task batches and
RNG keys, so task order is the only difference. The distance between final
parameters is normalized by the summed path lengths and judged against the
distance produced by re-drawing the batch stream (seed noise).
"""

from __future__ import annotations

import itertools
import zlib
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax

from fabricpc.core.types import GraphParams, GraphStructure
from fabricpc.experiments.forgetting_benchmark.gated_trainer import (
    MODES,
    GateEngine,
    GatedConfig,
    RunContext,
    assert_records_comparable,
    contract_fingerprint,
)
from fabricpc.experiments.forgetting_benchmark.sequential_trainer import params_hash
from fabricpc.training.plasticity_gates import GateSchedule

BatchFn = Callable[[str, np.ndarray], dict[str, Any]]
EvalFn = Callable[[GraphParams, Any], tuple[Any, Any]]
_REPLAY_KEYS = ("theta0_hash", "num_tokens", "num_updates")


def _task_code(name: str) -> int:
    return zlib.crc32(name.encode("utf-8"))


@dataclass(frozen=True)
class Streams:
    """Batch indices and RNG keys indexed by (task name, step)."""

    seed: int
    batch_size: int
    steps_per_task: int
    indices: dict[str, np.ndarray]
    keys: dict[str, jax.Array]
    probe_indices: dict[str, np.ndarray]
    diag_indices: dict[str, np.ndarray]
    diag_keys: dict[str, jax.Array]


def precompute_streams(
    seed: int,
    task_names: Sequence[str],
    steps_per_task: int,
    *,
    batch_size: int,
    num_sequences: int | Mapping[str, int],
    probe_batches: int = 2,
) -> Streams:
    """Draw every task's batches and keys up front, keyed by task name.

    Streams depend on (seed, task name) only, never on task position, so
    reversing the order reuses identical samples and dropout draws.
    """
    indices, keys, probes, diags, diag_keys = {}, {}, {}, {}, {}
    base = jax.random.PRNGKey(seed)
    for name in task_names:
        size = num_sequences if isinstance(num_sequences, int) else num_sequences[name]
        rng = np.random.default_rng([seed, _task_code(name)])
        indices[name] = rng.integers(0, size, size=(steps_per_task, batch_size))
        probes[name] = rng.integers(0, size, size=(probe_batches, batch_size))
        diags[name] = rng.integers(0, size, size=(1, batch_size))
        task_key = jax.random.fold_in(base, _task_code(name))
        step_key, diag_key = jax.random.split(task_key)
        keys[name] = jax.random.split(step_key, steps_per_task)
        diag_keys[name] = diag_key
    return Streams(
        seed=seed,
        batch_size=batch_size,
        steps_per_task=steps_per_task,
        indices=indices,
        keys=keys,
        probe_indices=probes,
        diag_indices=diags,
        diag_keys=diag_keys,
    )


def make_batch_fn(benchmark: Any, task_names: Sequence[str]) -> BatchFn:
    """Index next-character windows of each task's training data."""
    index_of = {benchmark.get_task_name(i): i for i in range(len(benchmark))}
    loaders = {name: benchmark.get_train_loader(index_of[name]) for name in task_names}

    def batch_fn(task: str, idx: np.ndarray) -> dict[str, Any]:
        loader = loaders[task]
        offsets = np.asarray(idx)[:, None] + np.arange(loader.seq_len)[None, :]
        return {
            "x": jnp.asarray(loader.data[offsets], dtype=jnp.int32),
            "y": jnp.asarray(loader.data[offsets + 1], dtype=jnp.int32),
        }

    return batch_fn


def num_sequences_of(benchmark: Any, task_names: Sequence[str]) -> dict[str, int]:
    index_of = {benchmark.get_task_name(i): i for i in range(len(benchmark))}
    return {
        name: benchmark.get_train_loader(index_of[name]).num_sequences
        for name in task_names
    }


def run_order(
    theta_0: GraphParams,
    order: Sequence[str],
    streams: Streams,
    mode: str,
    *,
    structure: GraphStructure,
    config: GatedConfig,
    batch_fn: BatchFn,
    tau: Optional[Mapping[str, float]] = None,
    replay_schedule: Optional[GateSchedule] = None,
    engine: Optional[GateEngine] = None,
    meta: Optional[dict[str, Any]] = None,
    recorder_out: Optional[dict[str, GateSchedule]] = None,
    data_identity: Any = None,
    settings: Any = None,
) -> tuple[GraphParams, float]:
    """Train the tasks in `order` from theta_0; return (theta, path_length).

    The optimizer is reset at every task boundary. `tau` is required for every
    mode except "none" and must not depend on the order. `meta`, if given, is
    filled with the run-comparability record.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode '{mode}'; expected one of {MODES}")
    if config.warmup_steps:
        raise ValueError("the commutator runner calibrates tau at theta_0 only")
    if tau is None and mode != "none":
        raise ValueError("tau is required for gated modes")
    engine = engine or GateEngine(structure, optax.sgd(config.lr), config)
    ctx = RunContext(
        mode=mode,
        config=config,
        calib_batches=[],
        calib_key=jax.random.PRNGKey(0),
        replay_schedule=replay_schedule,
        tau=dict(tau) if tau is not None else {n: 1.0 for n in engine.trainable},
    )
    params = theta_0
    completed: list[str] = []
    for position, task in enumerate(order):
        ctx.task_idx = position
        ctx.h = {}
        ctx.diag_batch = batch_fn(task, streams.diag_indices[task][0])
        ctx.diag_key = streams.diag_keys[task]
        sources = completed if config.probe_source == "past" else [task]
        ctx.probe_batches = [
            batch_fn(source, idx)
            for source in sources
            for idx in streams.probe_indices[source]
        ]
        steps = (
            (batch_fn(task, idx), key)
            for idx, key in zip(streams.indices[task], streams.keys[task])
        )
        params = engine.train_stage(params, steps, ctx)
        completed.append(task)
    if recorder_out is not None:
        recorder_out["+".join(order)] = ctx.recorder
    if meta is not None:
        meta.update(
            {
                "order": list(order),
                "mode": mode,
                "contract_fingerprint": contract_fingerprint(
                    config, streams.seed, sorted(order), data_identity, settings
                ),
                "theta0_hash": params_hash(theta_0),
                "num_tokens": ctx.num_tokens,
                "num_updates": ctx.num_updates,
                "path_length": ctx.path_length,
                "diagnostic_seconds": ctx.diagnostic_seconds,
                "train_seconds": ctx.train_seconds,
            }
        )
    return params, ctx.path_length


def shared_warmup(
    theta_init: GraphParams,
    tasks: Sequence[str],
    streams: Streams,
    steps: int,
    batch_fn: BatchFn,
    engine: GateEngine,
    config: GatedConfig,
) -> GraphParams:
    """Ungated steps alternating over the tasks; the result is the shared theta_0.

    Uses the same per-task batches and keys as the ordered runs, so no stream
    outside `streams` is consumed.
    """
    if steps <= 0:
        return theta_init
    per_task = -(-steps // len(tasks))
    if per_task > streams.steps_per_task:
        raise ValueError("warmup is longer than the per-task stream")

    def interleaved():
        for index in range(per_task):
            for task in tasks:
                yield batch_fn(task, streams.indices[task][index]), streams.keys[task][
                    index
                ]

    ctx = RunContext(
        mode="none",
        config=config,
        calib_batches=[],
        calib_key=jax.random.PRNGKey(0),
        tau={n: 1.0 for n in engine.trainable},
    )
    return engine.train_stage(theta_init, itertools.islice(interleaved(), steps), ctx)


def _l2_distance(a: GraphParams, b: GraphParams) -> float:
    total = 0.0
    for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
        diff = np.asarray(x, dtype=np.float64) - np.asarray(y, dtype=np.float64)
        total += float(np.sum(diff * diff))
    return float(np.sqrt(total))


def commutator_report(
    theta_ab: GraphParams,
    theta_ba: GraphParams,
    path_ab: float,
    path_ba: float,
    probe: Mapping[str, Any],
    eval_fn: EvalFn,
    eps: float = 1e-12,
    *,
    theta_0: Optional[GraphParams] = None,
    meta_ab: Optional[dict[str, Any]] = None,
    meta_ba: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Distance and function-space disagreement between the two orders.

    eval_fn(theta, batch) -> (loss, probabilities). With theta_0 given, the
    learning achieved is the probe-loss decrease from theta_0 per task and order.
    """
    if meta_ab is not None and meta_ba is not None:
        assert_records_comparable([meta_ab, meta_ba])
    distance = _l2_distance(theta_ab, theta_ba)
    report: dict[str, Any] = {
        "distance": distance,
        "path_ab": float(path_ab),
        "path_ba": float(path_ba),
        "normalized_distance": distance / (path_ab + path_ba + eps),
        "tasks": {},
    }
    for task, batch in probe.items():
        loss_ab, p_ab = eval_fn(theta_ab, batch)
        loss_ba, p_ba = eval_fn(theta_ba, batch)
        p_ab, p_ba = np.asarray(p_ab, np.float64), np.asarray(p_ba, np.float64)
        log_ab, log_ba = np.log(p_ab + 1e-10), np.log(p_ba + 1e-10)
        entry = {
            "loss_ab": float(loss_ab),
            "loss_ba": float(loss_ba),
            "loss_difference": float(loss_ab) - float(loss_ba),
            "logprob_disagreement": float(np.mean(np.abs(log_ab - log_ba))),
            "tv_disagreement": float(0.5 * np.mean(np.sum(np.abs(p_ab - p_ba), -1))),
            "argmax_disagreement": float(
                np.mean(np.argmax(p_ab, -1) != np.argmax(p_ba, -1))
            ),
        }
        if theta_0 is not None:
            start = float(eval_fn(theta_0, batch)[0])
            entry["learning_ab"] = start - float(loss_ab)
            entry["learning_ba"] = start - float(loss_ba)
        report["tasks"][task] = entry
    return report


def noise_floor(
    theta_0: GraphParams,
    order: Sequence[str],
    streams: Streams,
    alt_streams: Streams,
    mode: str,
    eps: float = 1e-12,
    **run_kwargs: Any,
) -> dict[str, Any]:
    """Distance between two runs of one order that differ only in batch stream."""
    meta_a: dict[str, Any] = {}
    meta_b: dict[str, Any] = {}
    theta_a, path_a = run_order(
        theta_0, order, streams, mode, meta=meta_a, **run_kwargs
    )
    theta_b, path_b = run_order(
        theta_0, order, alt_streams, mode, meta=meta_b, **run_kwargs
    )
    assert_records_comparable([meta_a, meta_b], keys=_REPLAY_KEYS)
    distance = _l2_distance(theta_a, theta_b)
    return {
        "distance": distance,
        "path_a": path_a,
        "path_b": path_b,
        "normalized_distance": distance / (path_a + path_b + eps),
    }
