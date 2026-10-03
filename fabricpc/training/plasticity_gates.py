"""Node-local surprise gates for predictive-coding weight updates.

The gate for node i is g_i = s_i / (s_i + tau_i), where s_i is the mean squared
prediction error of the settled node and tau_i its calibrated median score.
Gates scale a node's plain-SGD update and never change inference.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping

import jax
import jax.numpy as jnp
import numpy as np

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
    updates: GraphParams, gates: Mapping[str, jax.Array]
) -> GraphParams:
    """Multiply every weight and bias leaf of node i by gates[i]; others unchanged."""
    nodes: Dict[str, NodeParams] = {}
    for name, node_updates in updates.nodes.items():
        if name not in gates:
            nodes[name] = node_updates
            continue
        gate = gates[name]
        nodes[name] = jax.tree_util.tree_map(lambda x: x * gate, node_updates)
    return GraphParams(nodes=nodes)


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
