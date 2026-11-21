"""Tests for the additive encoder-only sentiment transformer builder."""

import os
import importlib.util
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import optax
import pytest

from fabricpc.core.inference import InferenceSGDNormClip
from fabricpc.graph_initialization import initialize_params
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.models.bert import (
    AddPositionEmbeddingNode,
    FinalLayerNormNode,
    FirstTokenPoolNode,
    create_encoder_sentiment_classifier,
)


def load_demo_module():
    demo_path = (
        Path(__file__).resolve().parents[1] / "examples" / "bert_sentiment_demo.py"
    )
    spec = importlib.util.spec_from_file_location(
        "bert_sentiment_demo_for_test", demo_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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
        jnp.abs(
            state_base.nodes["L0_mha"].z_mu[:, 0, :]
            - state_changed.nodes["L0_mha"].z_mu[:, 0, :]
        )
    )
    assert first_position_delta > 1e-6


def test_sentiment_demo_optimizer_supports_warmup_and_clipping():
    build_optimizer = load_demo_module().build_optimizer
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


def test_sentiment_demo_uses_explicit_splits_and_train_vocabulary(tmp_path, capsys):
    demo = load_demo_module()
    train_csv = tmp_path / "train.csv"
    eval_csv = tmp_path / "eval.csv"
    train_csv.write_text(
        "text,label\npleasant film,1\nbright story,1\ndull film,0\nbad story,0\n"
    )
    eval_csv.write_text("text,label\nheldout pleasant,1\nheldout dull,0\n")

    train_data = demo.load_examples(train_csv)
    assert "heldout" not in demo.build_vocab(train_data)

    args = demo.parse_args(
        [
            "--train_csv",
            str(train_csv),
            "--eval_csv",
            str(eval_csv),
            "--num_epochs",
            "1",
            "--batch_size",
            "2",
        ]
    )
    demo.main(args)
    assert "epoch=1" in capsys.readouterr().out


def test_sentiment_demo_rejects_invalid_or_shared_data(tmp_path):
    demo = load_demo_module()
    train_csv = tmp_path / "train.csv"
    train_csv.write_text("text,label\nexample,other\n")
    with pytest.raises(ValueError, match="0/1 label"):
        demo.load_examples(train_csv)

    args = demo.parse_args(
        ["--train_csv", str(train_csv), "--eval_csv", str(train_csv)]
    )
    with pytest.raises(ValueError, match="different files"):
        demo.main(args)


def test_sentiment_evaluation_weights_uneven_batches(monkeypatch):
    demo = load_demo_module()
    monkeypatch.setattr(
        demo,
        "compute_forward_pass",
        lambda params, structure, batch, rng_key: SimpleNamespace(
            nodes={"classifier": SimpleNamespace(z_mu=batch["probs"])}
        ),
    )
    loader = [
        {
            "y": jnp.array([0, 1]),
            "probs": jnp.array([[0.9, 0.1], [0.2, 0.8]]),
        },
        {"y": jnp.array([0]), "probs": jnp.array([[0.25, 0.75]])},
    ]

    metrics = demo.evaluate(
        None, SimpleNamespace(task_map={"y": "classifier"}), loader, None
    )

    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["loss"] == pytest.approx(
        (-jnp.log(0.9) - jnp.log(0.8) - jnp.log(0.25)) / 3
    )
