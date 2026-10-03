"""Tests for omitted-update influence and the influence-weighted gate."""

import importlib
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from fabricpc.core.inference import InferenceSGDNormClip
from fabricpc.core.types import GraphParams, NodeParams
from fabricpc.experiments.forgetting_benchmark.influence import (
    first_order_influence,
    influence_multipliers,
    make_probe_loss_fn,
    measure_update_influence,
    rank_agreement,
    weighted_gates,
)
from fabricpc.graph_initialization import initialize_params
from fabricpc.nodes.transformer_v2 import create_deep_transformer
from fabricpc.training.plasticity_gates import surprise_gates

TARGET = {"a": jnp.array([1.0, -2.0, 0.5]), "b": jnp.array([0.3, 0.7])}


def _params(offset: float = 0.0) -> GraphParams:
    return GraphParams(
        nodes={
            name: NodeParams(weights={"w": jnp.full_like(t, offset)}, biases={})
            for name, t in TARGET.items()
        }
    )


def quadratic_loss(params: GraphParams, _batch) -> jax.Array:
    return sum(
        jnp.sum((params.nodes[n].weights["w"] - t) ** 2) for n, t in TARGET.items()
    )


def _toward_target(params: GraphParams, scale: float) -> GraphParams:
    return GraphParams(
        nodes={
            n: NodeParams(
                weights={"w": scale * (TARGET[n] - params.nodes[n].weights["w"])},
                biases={},
            )
            for n in TARGET
        }
    )


def test_helpful_update_has_positive_influence_on_tiny_model():
    params = _params()
    influence = measure_update_influence(
        params, _toward_target(params, 0.1), [None], quadratic_loss
    )
    assert all(v > 0 for v in influence.values())


def test_harmful_update_has_negative_influence():
    params = _params()
    influence = measure_update_influence(
        params, _toward_target(params, -0.1), [None], quadratic_loss
    )
    assert all(v < 0 for v in influence.values())


def test_zero_update_has_zero_influence():
    params = _params()
    zero = jax.tree_util.tree_map(jnp.zeros_like, params)
    influence = measure_update_influence(params, zero, [None], quadratic_loss)
    assert influence == {"a": 0.0, "b": 0.0}


def test_intervention_changes_only_chosen_node_and_leaves_params_unmutated():
    params = _params(0.2)
    before = jax.tree_util.tree_map(np.asarray, params)
    updates = _toward_target(params, 0.1)
    seen = []

    def recording_loss(p, batch):
        seen.append(p)
        return quadratic_loss(p, batch)

    measure_update_influence(params, updates, [None], recording_loss)

    full = optax.apply_updates(params, updates)
    assert len(seen) == 3
    for tree in seen[1:]:
        differing = [
            n
            for n in TARGET
            if not np.array_equal(
                tree.nodes[n].weights["w"], full.nodes[n].weights["w"]
            )
        ]
        assert len(differing) == 1
        untouched = [n for n in TARGET if n != differing[0]][0]
        np.testing.assert_array_equal(
            tree.nodes[differing[0]].weights["w"],
            params.nodes[differing[0]].weights["w"],
        )
    for got, want in zip(
        jax.tree_util.tree_leaves(params), jax.tree_util.tree_leaves(before)
    ):
        np.testing.assert_array_equal(got, want)


def test_first_order_estimate_matches_exact_for_small_lr():
    def loss(params, _batch):
        return sum(jnp.sum(jnp.sin(params.nodes[n].weights["w"])) for n in TARGET)

    params = _params(0.4)
    grads = jax.grad(lambda p: loss(p, None))(params)
    updates = jax.tree_util.tree_map(lambda g: -1e-3 * g, grads)

    exact = measure_update_influence(params, updates, [None], loss)
    estimate = first_order_influence(params, updates, [None], loss)

    for name in TARGET:
        assert estimate[name] == pytest.approx(exact[name], rel=1e-2, abs=1e-9)


def test_multipliers_fallback_to_ones_without_positive_influence():
    assert influence_multipliers({"a": -1.0, "b": 0.0}) == {"a": 1.0, "b": 1.0}
    assert influence_multipliers({"a": 0.0, "b": 0.0}) == {"a": 1.0, "b": 1.0}


def test_multipliers_clip_at_two_and_are_finite():
    h = influence_multipliers({"a": 1.0, "b": 1.0, "c": 10.0, "d": -3.0})

    assert h["c"] == 2.0
    assert h["d"] == 0.0
    assert all(np.isfinite(v) for v in h.values())
    with pytest.raises(ValueError):
        influence_multipliers({"a": float("nan"), "b": 1.0})


def test_alpha_zero_equals_surprise_gate_exactly():
    g = surprise_gates({"a": jnp.asarray(0.37)}, {"a": 0.5})
    out = weighted_gates(g, {"a": jnp.asarray(1.7)}, alpha=0.0)
    np.testing.assert_array_equal(out["a"], g["a"])


def test_more_helpful_node_gets_larger_gate_at_equal_surprise():
    g = surprise_gates(
        {"a": jnp.asarray(0.2), "b": jnp.asarray(0.2)}, {"a": 0.5, "b": 0.5}
    )
    h = influence_multipliers({"a": 4.0, "b": 1.0})
    out = weighted_gates(g, h, alpha=0.5)

    assert float(out["a"]) > float(out["b"])


def test_rank_agreement_flags_high_surprise_nonpositive_influence():
    rho, flagged = rank_agreement(
        {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0},
        {"a": 0.1, "b": 0.2, "c": -0.1, "d": 0.4},
    )
    assert -1.0 <= rho <= 1.0
    assert flagged == ["c"]


# --- real PC model ---------------------------------------------------------------

VOCAB, SEQ = 16, 8
train_ar = importlib.import_module("fabricpc.training.train_autoregressive")


@pytest.fixture(scope="module")
def tiny_pc():
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
    x = jnp.asarray(
        np.random.default_rng(0).integers(0, VOCAB, size=(4, SEQ)), dtype=jnp.int32
    )
    return structure, params, {"x": x, "y": (x + 1) % VOCAB}


def test_empty_past_probe_gives_all_ones(tiny_pc):
    structure, params, _ = tiny_pc
    calls = []

    def loss_fn(p, b):
        calls.append(1)
        return jnp.asarray(0.0)

    updates = jax.tree_util.tree_map(jnp.ones_like, params)
    influence = measure_update_influence(params, updates, [], loss_fn)
    h = influence_multipliers(influence)

    assert not calls
    assert influence and all(v == 1.0 for v in h.values())


def test_probe_uses_eval_mode_dropout(tiny_pc, monkeypatch):
    structure, params, batch = tiny_pc
    seen = []
    original = train_ar.create_dropout_clamps

    def spy(structure, batch_size, rng_key, training):
        seen.append(training)
        return original(structure, batch_size, rng_key, training)

    monkeypatch.setattr(train_ar, "create_dropout_clamps", spy)
    loss_fn = make_probe_loss_fn(structure, True, jax.random.PRNGKey(3))
    first = float(loss_fn(params, batch))
    second = float(loss_fn(params, batch))

    assert seen and not any(seen)
    assert first == second


def test_zero_update_control_on_real_model_is_zero(tiny_pc):
    structure, params, batch = tiny_pc
    loss_fn = make_probe_loss_fn(structure, True, jax.random.PRNGKey(3))
    zero = jax.tree_util.tree_map(jnp.zeros_like, params)

    influence = measure_update_influence(params, zero, [batch], loss_fn)

    assert all(abs(v) < 1e-7 for v in influence.values())
