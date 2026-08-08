"""Tests for the additive encoder-only sentiment transformer builder."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp

from fabricpc.core.inference import InferenceSGDNormClip
from fabricpc.graph_initialization import initialize_params
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.models.bert import (
    AddPositionEmbeddingNode,
    FirstTokenPoolNode,
    create_encoder_sentiment_classifier,
)


def build_small_encoder():
    return create_encoder_sentiment_classifier(
        depth=1,
        embed_dim=16,
        num_heads=4,
        mlp_dim=32,
        seq_len=6,
        vocab_size=20,
        num_classes=2,
        inference=InferenceSGDNormClip(eta_infer=0.1, infer_steps=3, max_norm=5.0),
    )


def test_encoder_topology_is_bidirectional():
    structure = build_small_encoder()

    assert structure.nodes["L0_mha"].node_info.node_config["is_causal"] is False
    assert structure.nodes["L0_mha"].node_info.node_config["use_rope"] is False
    assert structure.nodes["position_embeddings"].node_info.node_class is AddPositionEmbeddingNode
    assert structure.nodes["pooler"].node_info.node_class is FirstTokenPoolNode
    assert structure.task_map["x"] == "input_ids"
    assert structure.task_map["y"] == "classifier"


def test_encoder_forward_shape_smoke():
    rng_key = jax.random.PRNGKey(0)
    structure = build_small_encoder()
    params = initialize_params(structure, rng_key)

    x = jnp.array([[1, 2, 3, 4, 0, 0], [1, 5, 6, 7, 8, 0]], dtype=jnp.int32)
    state = initialize_graph_state(
        structure,
        batch_size=2,
        rng_key=rng_key,
        clamps={"input_ids": x},
        params=params,
    )

    probs = state.nodes["classifier"].z_mu
    assert probs.shape == (2, 2)
    assert jnp.all(jnp.isfinite(probs))
    assert jnp.allclose(jnp.sum(probs, axis=-1), 1.0, atol=1e-5)


def test_future_token_can_affect_first_position_attention():
    rng_key = jax.random.PRNGKey(1)
    structure = build_small_encoder()
    params = initialize_params(structure, rng_key)

    base = jnp.array([[1, 2, 3, 4, 0, 0]], dtype=jnp.int32)
    changed_future = base.at[:, 4].set(9)

    state_base = initialize_graph_state(
        structure,
        batch_size=1,
        rng_key=rng_key,
        clamps={"input_ids": base},
        params=params,
    )
    state_changed = initialize_graph_state(
        structure,
        batch_size=1,
        rng_key=rng_key,
        clamps={"input_ids": changed_future},
        params=params,
    )

    first_position_delta = jnp.max(
        jnp.abs(state_base.nodes["L0_mha"].z_mu[:, 0, :] - state_changed.nodes["L0_mha"].z_mu[:, 0, :])
    )
    assert first_position_delta > 1e-6
