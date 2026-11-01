"""Node-local surprise gates for predictive-coding weight updates.

The gate for node i is g_i = s_i / (s_i + tau_i), where s_i is the mean squared
prediction error of the settled node and tau_i its calibrated median score.
Gates scale a node's plain-SGD update and never change inference.
"""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Mapping, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from fabricpc.core.types import GraphParams, GraphState, GraphStructure, NodeParams

TAU_FLOOR = 1e-12


def _is_concrete(value: jax.Array) -> bool:
    return not isinstance(value, jax.core.Tracer)


def _require_finite(name: str, value: jax.Array, what: str) -> None:
    if _is_concrete(value) and not bool(jnp.all(jnp.isfinite(value))):
        raise ValueError(f"non-finite {what} for node '{name}'")


def local_surprise(
    final_state: GraphState, structure: GraphStructure
) -> Dict[str, jax.Array]:
    """Mean of error**2 over every axis, one scalar per node with in_degree > 0."""
    scores: Dict[str, jax.Array] = {}
    for name, node in structure.nodes.items():
        if node.node_info.in_degree <= 0:
            continue
        score = jnp.mean(jnp.square(final_state.nodes[name].error))
        _require_finite(name, score, "surprise score")
        scores[name] = score
    return scores


def calibrate_tau(scores: Mapping[str, jax.Array]) -> Dict[str, float]:
    """Per-node median over calibration batches, floored at a positive epsilon."""
    tau: Dict[str, float] = {}
    for name, values in scores.items():
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError(f"cannot calibrate tau for node '{name}'")
        tau[name] = float(max(np.median(values), TAU_FLOOR))
    return tau


def surprise_gates(
    scores: Mapping[str, jax.Array], calibration: Mapping[str, jax.Array]
) -> Dict[str, jax.Array]:
    """clip(s / (s + tau), 0, 1); zero score gives zero gate."""
    gates: Dict[str, jax.Array] = {}
    for name, score in scores.items():
        score = jnp.asarray(score)
        tau = jnp.asarray(calibration[name], dtype=score.dtype)
        _require_finite(name, score, "surprise score")
        gates[name] = jnp.clip(score / (score + tau), 0.0, 1.0)
    return gates


def scale_node_updates(
    updates: GraphParams, gates: Mapping[str, jax.Array | float]
) -> GraphParams:
    """Multiply every weight and bias leaf of node i by gates[i].

    Nodes without parameters may lack a gate; a parameter-bearing node without
    one is an error.
    """
    nodes: Dict[str, NodeParams] = {}
    for name, node_updates in updates.nodes.items():
        if name not in gates:
            if jax.tree_util.tree_leaves(node_updates):
                raise ValueError(f"no gate for parameter-bearing node '{name}'")
            nodes[name] = node_updates
            continue
        gate = gates[name]
        nodes[name] = jax.tree_util.tree_map(lambda x: x * gate, node_updates)
    return GraphParams(nodes=nodes)


def scale_module_updates(
    multipliers: Mapping[str, float],
) -> optax.GradientTransformation:
    """Stateless transform multiplying each node's update leaves by its multiplier.

    Nodes absent from ``multipliers`` get 1.0; a multiplier of 0 zeroes the
    whole update, so chained after AdamW the node is frozen exactly. Names
    that are not nodes of the params raise at ``init`` and at ``update``.
    """
    values = {str(name): float(value) for name, value in multipliers.items()}

    def check(tree: GraphParams) -> None:
        unknown = sorted(set(values) - set(tree.nodes))
        if unknown:
            raise ValueError(f"multipliers for unknown nodes {unknown}")

    def init_fn(params: GraphParams) -> optax.EmptyState:
        check(params)
        return optax.EmptyState()

    def update_fn(
        updates: GraphParams,
        state: optax.EmptyState,
        params: GraphParams | None = None,
    ) -> tuple[GraphParams, optax.EmptyState]:
        check(updates)
        gates = {name: values.get(name, 1.0) for name in updates.nodes}
        return scale_node_updates(updates, gates), state

    return optax.GradientTransformation(init_fn, update_fn)


class ModuleMultiplierState(NamedTuple):
    """Per-node float32 scalar multipliers of ``scale_module_updates_traced``."""

    multipliers: Dict[str, jax.Array]


def _positive_multipliers(multipliers: Mapping[str, float]) -> Dict[str, float]:
    values = {str(name): float(value) for name, value in multipliers.items()}
    bad = sorted(n for n, v in values.items() if not (np.isfinite(v) and v > 0))
    if bad:
        raise ValueError(f"multipliers must be finite and positive; not at {bad}")
    return values


def scale_module_updates_traced(
    multipliers: Mapping[str, float],
) -> optax.GradientTransformation:
    """``scale_module_updates`` with the multipliers held in the optimizer state.

    The state holds one float32 scalar per node of the params (1.0 for nodes
    absent from ``multipliers``), so ``set_module_multipliers`` changes them
    between updates without retracing a jitted update. Multipliers must be
    finite and positive: no node is frozen, so the global-norm clip before
    AdamW needs no zeroed gradients. Each update leaf keeps its dtype. Names
    that are not nodes raise at ``init``.
    """
    values = _positive_multipliers(multipliers)

    def init_fn(params: GraphParams) -> ModuleMultiplierState:
        unknown = sorted(set(values) - set(params.nodes))
        if unknown:
            raise ValueError(f"multipliers for unknown nodes {unknown}")
        return ModuleMultiplierState(
            {
                name: jnp.asarray(values.get(name, 1.0), dtype=jnp.float32)
                for name in params.nodes
            }
        )

    def update_fn(
        updates: GraphParams,
        state: ModuleMultiplierState,
        params: GraphParams | None = None,
    ) -> tuple[GraphParams, ModuleMultiplierState]:
        nodes = {
            name: jax.tree_util.tree_map(
                lambda x, g=state.multipliers[name]: x * g.astype(x.dtype), node
            )
            for name, node in updates.nodes.items()
        }
        return GraphParams(nodes=nodes), state

    return optax.GradientTransformation(init_fn, update_fn)


def _is_multiplier_state(node: Any) -> bool:
    return isinstance(node, ModuleMultiplierState)


def set_module_multipliers(opt_state: Any, multipliers: Mapping[str, float]) -> Any:
    """Return ``opt_state`` with every ``ModuleMultiplierState`` set to ``multipliers``.

    Nodes absent from ``multipliers`` get 1.0; values must be finite and
    positive and names must be nodes of the state. The new scalars are
    float32, as at ``init``, so a jitted update is not retraced.
    """
    values = _positive_multipliers(multipliers)
    found = []

    def replace_state(node: Any) -> Any:
        if not _is_multiplier_state(node):
            return node
        unknown = sorted(set(values) - set(node.multipliers))
        if unknown:
            raise ValueError(f"multipliers for unknown nodes {unknown}")
        found.append(node)
        return ModuleMultiplierState(
            {
                name: jnp.asarray(values.get(name, 1.0), dtype=jnp.float32)
                for name in node.multipliers
            }
        )

    new_state = jax.tree_util.tree_map(
        replace_state, opt_state, is_leaf=_is_multiplier_state
    )
    if not found:
        raise ValueError("opt_state has no ModuleMultiplierState")
    return new_state


def module_multipliers(opt_state: Any) -> Dict[str, float]:
    """The multipliers the single ``ModuleMultiplierState`` of ``opt_state`` holds."""
    states = [
        node
        for node in jax.tree_util.tree_leaves(opt_state, is_leaf=_is_multiplier_state)
        if _is_multiplier_state(node)
    ]
    if len(states) != 1:
        raise ValueError(f"opt_state has {len(states)} ModuleMultiplierStates, not 1")
    return {name: float(value) for name, value in states[0].multipliers.items()}


class GateSchedule:
    """Per-step, per-node gate trajectory for open-loop replay."""

    def __init__(self) -> None:
        self._gates: Dict[int, Dict[str, float]] = {}

    def __len__(self) -> int:
        return len(self._gates)

    def __contains__(self, step: int) -> bool:
        return step in self._gates

    def record(self, step: int, gates: Mapping[str, float]) -> None:
        self._gates[int(step)] = {name: float(g) for name, g in gates.items()}

    def gates_at(self, step: int) -> Dict[str, float]:
        return dict(self._gates[int(step)])

    def save(self, path: Path) -> None:
        payload = {str(step): gates for step, gates in sorted(self._gates.items())}
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "GateSchedule":
        schedule = cls()
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        for step, gates in payload.items():
            schedule.record(int(step), gates)
        return schedule


GATE_MODES = ("none", "surprise", "influence", "replay")


@dataclass(frozen=True)
class GateFnState:
    """Gate inputs as a pytree: array leaves are traced, only `mode` is static.

    tau and h hold one scalar per node with in_degree > 0; `override` carries
    replayed gates and is ignored unless mode is "replay".
    """

    mode: str
    tau: Dict[str, jax.Array]
    h: Dict[str, jax.Array]
    alpha: jax.Array
    override: Dict[str, jax.Array]


jax.tree_util.register_pytree_node(
    GateFnState,
    lambda s: ((s.tau, s.h, s.alpha, s.override), s.mode),
    lambda mode, children: GateFnState(mode, *children),
)


def make_gate_state(
    mode: str,
    tau: Mapping[str, float],
    *,
    h: Mapping[str, float] | None = None,
    alpha: float = 0.5,
    override: Mapping[str, float] | None = None,
) -> GateFnState:
    """Build a GateFnState with float32 scalars; h and override default to ones."""
    if mode not in GATE_MODES:
        raise ValueError(f"unknown gate mode '{mode}'; expected one of {GATE_MODES}")

    def as_scalars(values: Mapping[str, float] | None) -> Dict[str, jax.Array]:
        values = values or {}
        return {
            name: jnp.asarray(values.get(name, 1.0), dtype=jnp.float32) for name in tau
        }

    return GateFnState(
        mode=mode,
        tau={
            name: jnp.asarray(value, dtype=jnp.float32) for name, value in tau.items()
        },
        h=as_scalars(h),
        alpha=jnp.asarray(alpha, dtype=jnp.float32),
        override=as_scalars(override),
    )


def compute_gates(
    scores: Mapping[str, jax.Array], state: GateFnState
) -> Dict[str, jax.Array]:
    """Per-node gates for the state's mode."""
    if state.mode == "none":
        return {name: jnp.ones((), dtype=jnp.float32) for name in scores}
    if state.mode == "replay":
        return {name: state.override[name] for name in scores}
    gates = surprise_gates(scores, state.tau)
    if state.mode == "influence":
        return weighted_gates(gates, state.h, state.alpha)
    return gates


def weighted_gates(
    surprise_g: Mapping[str, jax.Array],
    h: Mapping[str, jax.Array],
    alpha: jax.Array | float = 0.5,
) -> Dict[str, jax.Array]:
    """clip(g * ((1 - alpha) + alpha * h), 0, 1); alpha = 0 returns g exactly."""
    return {
        name: jnp.clip(g * ((1.0 - alpha) + alpha * h[name]), 0.0, 1.0)
        for name, g in surprise_g.items()
    }
