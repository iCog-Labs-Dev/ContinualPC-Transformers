"""Tests for node-local surprise gates and gated updates."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from fabricpc.core.types import GraphParams, GraphState, NodeParams, NodeState
from fabricpc.training.plasticity_gates import (
    GateSchedule,
    calibrate_tau,
    local_surprise,
    scale_node_updates,
    surprise_gates,
)


def _node_state(error: jnp.ndarray) -> NodeState:
    zeros = jnp.zeros_like(error)
    return NodeState(
        z_latent=zeros,
        z_mu=zeros,
        error=error,
        energy=jnp.zeros(error.shape[0]),
        pre_activation=zeros,
        latent_grad=zeros,
    )


def _structure(in_degrees: dict[str, int]) -> SimpleNamespace:
    return SimpleNamespace(
        nodes={
            name: SimpleNamespace(node_info=SimpleNamespace(in_degree=degree))
            for name, degree in in_degrees.items()
        }
    )


def _state(errors: dict[str, jnp.ndarray]) -> GraphState:
    return GraphState(
        nodes={name: _node_state(err) for name, err in errors.items()}, batch_size=2
    )


def _params() -> GraphParams:
    return GraphParams(
        nodes={
            "a": NodeParams(
                weights={"w": jnp.arange(6.0).reshape(2, 3)},
                biases={"b": jnp.ones(3)},
            ),
            "b": NodeParams(
                weights={"w": -jnp.arange(4.0).reshape(2, 2)},
                biases={"b": jnp.full(2, 2.0)},
            ),
            "src": NodeParams(weights={}, biases={}),
        }
    )


def test_local_surprise_is_mean_squared_error_and_ignores_width_and_batch():
    structure = _structure({"a": 1})
    small = _state({"a": jnp.full((2, 3, 4), 0.5)})
    wide_batch = _state({"a": jnp.full((8, 3, 16), 0.5)})

    assert float(local_surprise(small, structure)["a"]) == pytest.approx(0.25)
    assert float(local_surprise(wide_batch, structure)["a"]) == pytest.approx(0.25)


def test_local_surprise_omits_source_nodes():
    structure = _structure({"src": 0, "a": 2})
    state = _state({"src": jnp.ones((2, 3)), "a": jnp.ones((2, 3))})

    assert set(local_surprise(state, structure)) == {"a"}


def test_local_surprise_rejects_nonfinite():
    structure = _structure({"a": 1})
    state = _state({"a": jnp.array([[1.0, jnp.nan]])})

    with pytest.raises(ValueError):
        local_surprise(state, structure)
    with pytest.raises(ValueError):
        surprise_gates({"a": jnp.array(jnp.inf)}, {"a": 1.0})


def test_gate_is_bounded_and_finite_for_zero_and_huge_scores():
    scores = {"a": jnp.array(0.0), "b": jnp.array(1e30), "c": jnp.array(1e-30)}
    gates = surprise_gates(scores, {"a": 1.0, "b": 1.0, "c": 1.0})

    values = np.array([float(gates[k]) for k in "abc"])
    assert np.all(np.isfinite(values))
    assert np.all((values >= 0.0) & (values <= 1.0))
    assert values[0] == 0.0


def test_zero_score_gives_zero_gate_and_finite_gradient():
    def gate_of_error(error):
        score = {"a": jnp.mean(error**2)}
        return surprise_gates(score, {"a": 0.3})["a"]

    error = jnp.zeros((2, 3))
    assert float(gate_of_error(error)) == 0.0
    assert np.all(np.isfinite(np.asarray(jax.grad(gate_of_error)(error))))


def test_changing_one_node_error_changes_only_its_gate():
    structure = _structure({"a": 1, "b": 1})
    calibration = {"a": 0.5, "b": 0.5}
    base = _state({"a": jnp.full((2, 3), 0.4), "b": jnp.full((2, 3), 0.4)})
    changed = _state({"a": jnp.full((2, 3), 0.4), "b": jnp.full((2, 3), 0.9)})

    gate_base = surprise_gates(local_surprise(base, structure), calibration)
    gate_changed = surprise_gates(local_surprise(changed, structure), calibration)

    assert float(gate_base["a"]) == float(gate_changed["a"])
    assert float(gate_base["b"]) < float(gate_changed["b"])


def test_calibrate_tau_is_median_with_positive_floor():
    tau = calibrate_tau(
        {"a": jnp.array([1.0, 3.0, 2.0]), "b": jnp.array([0.0, 0.0, 0.0])}
    )

    assert tau["a"] == pytest.approx(2.0)
    assert tau["b"] > 0.0


def test_all_one_gates_reproduce_ungated_sgd_exactly():
    params = _params()
    updates = jax.tree_util.tree_map(lambda x: -0.1 * (x + 1.0), params)
    gates = {"a": jnp.array(1.0), "b": jnp.array(1.0)}

    scaled = scale_node_updates(updates, gates)

    for got, want in zip(
        jax.tree_util.tree_leaves(scaled), jax.tree_util.tree_leaves(updates)
    ):
        np.testing.assert_array_equal(got, want)


def test_zero_gate_leaves_node_params_bitwise_unchanged():
    params = _params()
    updates = jax.tree_util.tree_map(lambda x: -0.1 * (x + 1.0), params)
    gates = {"a": jnp.array(0.0), "b": jnp.array(1.0)}

    new = optax.apply_updates(params, scale_node_updates(updates, gates))

    for got, want in zip(
        jax.tree_util.tree_leaves(new.nodes["a"]),
        jax.tree_util.tree_leaves(params.nodes["a"]),
    ):
        np.testing.assert_array_equal(got, want)
    assert not np.array_equal(
        new.nodes["b"].weights["w"], params.nodes["b"].weights["w"]
    )


def test_weights_and_biases_of_a_node_share_one_scalar():
    params = _params()
    updates = jax.tree_util.tree_map(jnp.ones_like, params)
    scaled = scale_node_updates(updates, {"a": jnp.array(0.25), "b": jnp.array(0.5)})

    np.testing.assert_array_equal(scaled.nodes["a"].weights["w"], 0.25)
    np.testing.assert_array_equal(scaled.nodes["a"].biases["b"], 0.25)
    np.testing.assert_array_equal(scaled.nodes["b"].biases["b"], 0.5)


def test_gate_schedule_roundtrip(tmp_path):
    schedule = GateSchedule()
    schedule.record(0, {"a": 0.25, "b": 1.0})
    schedule.record(3, {"a": 0.5, "b": 0.125})

    path = tmp_path / "gates.json"
    schedule.save(path)
    loaded = GateSchedule.load(path)

    assert len(loaded) == 2
    assert loaded.gates_at(3) == {"a": 0.5, "b": 0.125}
    with pytest.raises(KeyError):
        loaded.gates_at(1)
