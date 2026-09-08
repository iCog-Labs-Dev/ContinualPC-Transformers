"""Tests for the additive encoder-only sentiment transformer builder."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import optax
import pytest

from examples.bert_demo import build_optimizer
from fabricpc.training.train_backprop import compute_forward_pass

from fabricpc.core.inference import InferenceSGDNormClip
from fabricpc.graph_initialization import initialize_params
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.nodes.encoder_transformer import (
    AddPositionEmbeddingNode,
    FinalLayerNormNode,
    FirstTokenPoolNode,
    create_encoder_mlm_model,
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
    assert (
        structure.nodes["position_embeddings"].node_info.node_class
        is AddPositionEmbeddingNode
    )
    assert structure.nodes["final_layernorm"].node_info.node_class is FinalLayerNormNode
    assert structure.nodes["pooler"].node_info.node_class is FirstTokenPoolNode
    assert structure.nodes["classifier"].node_info.weight_init.config["std"] == 0.02
    assert structure.task_map["x"] == "input_ids"
    assert structure.task_map["y"] == "classifier"


def test_encoder_dropout_tasks_connect_to_residual_slots():
    structure = create_encoder_mlm_model(
        depth=1,
        embed_dim=16,
        num_heads=4,
        mlp_dim=32,
        seq_len=6,
        vocab_size=20,
        dropout_rate=0.1,
        inference=InferenceSGDNormClip(eta_infer=0.1, infer_steps=3, max_norm=5.0),
    )

    assert structure.config["dropout_rate"] == 0.1
    assert structure.task_map["dropout_L0_mha"] == "dropout_mask_L0_mha"
    assert structure.task_map["dropout_L0_mlp2"] == "dropout_mask_L0_mlp2"
    assert "dropout_mask_L0_mha->L0_mha:dropout_mask" in structure.edges
    assert "dropout_mask_L0_mlp2->L0_mlp2:dropout_mask" in structure.edges


def test_encoder_weightless_transform_slots_are_not_mupc_scaled():
    inference = InferenceSGDNormClip(eta_infer=0.1, infer_steps=3, max_norm=5.0)
    mlm = create_encoder_mlm_model(
        depth=1,
        embed_dim=16,
        num_heads=4,
        mlp_dim=32,
        seq_len=6,
        vocab_size=20,
        inference=inference,
    )
    classifier = create_encoder_sentiment_classifier(
        depth=1,
        embed_dim=16,
        num_heads=4,
        mlp_dim=32,
        seq_len=6,
        vocab_size=20,
        num_classes=2,
        inference=inference,
    )

    for structure in (mlm, classifier):
        for node_name in ("position_embeddings", "final_layernorm", "pooler"):
            if node_name not in structure.nodes:
                continue
            scaling = structure.nodes[node_name].node_info.scaling_config
            forward_scale = {} if scaling is None else scaling.forward_scale
            assert not any(key.endswith(f"->{node_name}:in") for key in forward_scale)


@pytest.mark.parametrize(
    "overrides",
    [
        {"depth": 0},
        {"embed_dim": 0},
        {"num_heads": 0},
        {"mlp_dim": 0},
        {"seq_len": 0},
        {"vocab_size": 0},
        {"embed_dim": 10, "num_heads": 4},
    ],
)
def test_encoder_mlm_builder_rejects_invalid_config(overrides):
    kwargs = dict(
        depth=1,
        embed_dim=16,
        num_heads=4,
        mlp_dim=32,
        seq_len=6,
        vocab_size=20,
        inference=InferenceSGDNormClip(eta_infer=0.1, infer_steps=3, max_norm=5.0),
    )
    kwargs.update(overrides)

    with pytest.raises(ValueError):
        create_encoder_mlm_model(**kwargs)


def test_encoder_classifier_builder_rejects_invalid_num_classes():
    with pytest.raises(ValueError, match="num_classes"):
        create_encoder_sentiment_classifier(
            depth=1,
            embed_dim=16,
            num_heads=4,
            mlp_dim=32,
            seq_len=6,
            vocab_size=20,
            num_classes=0,
            inference=InferenceSGDNormClip(eta_infer=0.1, infer_steps=3, max_norm=5.0),
        )


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


def test_encoder_dropout_train_eval_smoke():
    rng_key = jax.random.PRNGKey(4)
    structure = create_encoder_mlm_model(
        depth=1,
        embed_dim=16,
        num_heads=4,
        mlp_dim=32,
        seq_len=6,
        vocab_size=20,
        dropout_rate=0.5,
        inference=InferenceSGDNormClip(eta_infer=0.1, infer_steps=3, max_norm=5.0),
    )
    params = initialize_params(structure, rng_key)
    batch = {"x": jnp.array([[1, 2, 3, 4, 0, 0]], dtype=jnp.int32)}

    eval_a = compute_forward_pass(params, structure, batch, rng_key, training=False)
    eval_b = compute_forward_pass(params, structure, batch, rng_key, training=False)
    train_a = compute_forward_pass(params, structure, batch, rng_key, training=True)
    train_b = compute_forward_pass(
        params, structure, batch, jax.random.PRNGKey(5), training=True
    )

    assert jnp.allclose(eval_a.nodes["mlm_head"].z_mu, eval_b.nodes["mlm_head"].z_mu)
    assert not jnp.allclose(train_a.nodes["mlm_head"].z_mu, train_b.nodes["mlm_head"].z_mu)


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
        jnp.abs(
            state_base.nodes["L0_mha"].z_mu[:, 0, :]
            - state_changed.nodes["L0_mha"].z_mu[:, 0, :]
        )
    )
    assert first_position_delta > 1e-6


def test_sentiment_demo_optimizer_supports_warmup_and_clipping():
    optimizer = build_optimizer(
        learning_rate=1e-2,
        weight_decay=1e-4,
        grad_clip=1.0,
        warmup_steps=2,
        total_steps=4,
    )
    params = {"w": jnp.ones((2,))}
    grads = {"w": jnp.array([10.0, -10.0])}

    opt_state = optimizer.init(params)
    updates, _ = optimizer.update(grads, opt_state, params)
    updated_params = optax.apply_updates(params, updates)

    assert updated_params["w"].shape == params["w"].shape
    assert jnp.all(jnp.isfinite(updated_params["w"]))
