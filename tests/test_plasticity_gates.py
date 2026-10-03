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


# --- gated autoregressive step -------------------------------------------------

from fabricpc.core.inference import InferenceSGDNormClip
from fabricpc.graph_initialization import initialize_params
from fabricpc.nodes.transformer_v2 import create_deep_transformer
from fabricpc.training.plasticity_gates import (
    compute_gates,
    make_gate_state,
    weighted_gates,
)
from fabricpc.training.train_autoregressive import (
    train_step_autoregressive,
    train_step_autoregressive_gated,
    validate_gated_config,
)

VOCAB, SEQ, BATCH = 16, 8, 4


@pytest.fixture(scope="module")
def tiny():
    structure = create_deep_transformer(
        depth=1,
        embed_dim=16,
        num_heads=2,
        mlp_dim=32,
        seq_len=SEQ,
        vocab_size=VOCAB,
        inference=InferenceSGDNormClip(eta_infer=0.05, infer_steps=2, max_norm=5.0),
    )
    params = initialize_params(structure, jax.random.PRNGKey(0))
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.integers(0, VOCAB, size=(BATCH, SEQ)), dtype=jnp.int32)
    batch = {"x": x, "y": (x + 1) % VOCAB}
    trainable = [n for n, nd in structure.nodes.items() if nd.node_info.in_degree > 0]
    return structure, params, batch, trainable


def _leaves(node_params):
    return jax.tree_util.tree_leaves(node_params)


def _gate_state(trainable, mode="replay", override=None, tau=1.0):
    return make_gate_state(mode, {n: tau for n in trainable}, override=override or {})


def test_gated_step_with_all_one_gates_matches_plain_step_exactly(tiny):
    structure, params, batch, trainable = tiny
    optimizer = optax.sgd(0.1)
    opt_state = optimizer.init(params)
    key = jax.random.PRNGKey(5)

    plain = jax.jit(
        lambda p, o, b, k: train_step_autoregressive(p, o, b, structure, optimizer, k)
    )(params, opt_state, batch, key)
    gated = jax.jit(
        lambda p, o, b, k, g: train_step_autoregressive_gated(
            p, o, b, structure, optimizer, k, g
        )
    )(params, opt_state, batch, key, _gate_state(trainable))

    for got, want in zip(
        jax.tree_util.tree_leaves(gated[0]), jax.tree_util.tree_leaves(plain[0])
    ):
        np.testing.assert_array_equal(got, want)
    assert float(gated[2]) == float(plain[2])
    assert float(gated[3]) == float(plain[3])


def test_zero_gate_after_nonzero_grad_step_freezes_only_that_node(tiny):
    structure, params, batch, trainable = tiny
    optimizer = optax.sgd(0.1)
    opt_state = optimizer.init(params)
    key = jax.random.PRNGKey(5)
    plain_params = train_step_autoregressive(
        params, opt_state, batch, structure, optimizer, key
    )[0]
    moved = [
        n
        for n in trainable
        if any(
            not np.array_equal(a, b)
            for a, b in zip(_leaves(plain_params.nodes[n]), _leaves(params.nodes[n]))
        )
    ]
    assert len(moved) >= 2
    frozen = moved[0]

    state = _gate_state(trainable, override={frozen: 0.0})
    new_params = train_step_autoregressive_gated(
        params, opt_state, batch, structure, optimizer, key, state
    )[0]

    for a, b in zip(_leaves(new_params.nodes[frozen]), _leaves(params.nodes[frozen])):
        np.testing.assert_array_equal(a, b)
    for name in moved[1:]:
        for a, b in zip(
            _leaves(new_params.nodes[name]), _leaves(plain_params.nodes[name])
        ):
            np.testing.assert_array_equal(a, b)


def test_perturbing_node_i_error_does_not_scale_other_nodes_update(tiny):
    structure, params, batch, trainable = tiny
    state = make_gate_state("surprise", {n: 0.5 for n in trainable})
    base = {n: jnp.asarray(0.4) for n in trainable}
    bumped = dict(base)
    bumped[trainable[0]] = jnp.asarray(2.0)
    updates = jax.tree_util.tree_map(jnp.ones_like, params)

    scaled_base = scale_node_updates(updates, compute_gates(base, state))
    scaled_bumped = scale_node_updates(updates, compute_gates(bumped, state))

    for name in trainable[1:]:
        for a, b in zip(
            _leaves(scaled_base.nodes[name]), _leaves(scaled_bumped.nodes[name])
        ):
            np.testing.assert_array_equal(a, b)
    first = trainable[0]
    assert not np.array_equal(
        _leaves(scaled_base.nodes[first])[0], _leaves(scaled_bumped.nodes[first])[0]
    )


def test_gated_step_does_not_recompile_when_gate_values_change(tiny):
    structure, params, batch, trainable = tiny
    optimizer = optax.sgd(0.1)
    opt_state = optimizer.init(params)
    traces = []

    @jax.jit
    def step(p, o, b, k, g):
        traces.append(1)
        return train_step_autoregressive_gated(p, o, b, structure, optimizer, k, g)

    key = jax.random.PRNGKey(1)
    for value in (0.1, 0.5, 0.9):
        state = _gate_state(trainable, override={n: value for n in trainable})
        step(params, opt_state, batch, key, state)
    for tau in (0.2, 3.0):
        step(params, opt_state, batch, key, _gate_state(trainable, "replay", tau=tau))

    assert len(traces) == 1


def test_validate_gated_config_rejects_unsupported_combinations():
    validate_gated_config({"name": "sgd"})
    validate_gated_config({"name": "sgd", "momentum": 0.0, "weight_decay": 0.0})
    for bad in (
        {"name": "adam"},
        {"name": "sgd", "momentum": 0.9},
        {"name": "sgd", "weight_decay": 1e-4},
        {"name": "sgd", "gradient_accumulation_steps": 4},
        {"name": "sgd", "pmap": True},
    ):
        with pytest.raises(ValueError):
            validate_gated_config(bad)


def test_weighted_gates_alpha_zero_is_identity_and_bounded():
    g = {"a": jnp.asarray(0.3), "b": jnp.asarray(0.9)}
    h = {"a": jnp.asarray(2.0), "b": jnp.asarray(2.0)}

    np.testing.assert_array_equal(weighted_gates(g, h, 0.0)["a"], g["a"])
    assert float(weighted_gates(g, h, 0.5)["b"]) == 1.0


def test_parameter_bearing_node_without_gate_is_rejected():
    updates = jax.tree_util.tree_map(jnp.ones_like, _params())

    with pytest.raises(ValueError):
        scale_node_updates(updates, {"a": jnp.array(1.0)})
