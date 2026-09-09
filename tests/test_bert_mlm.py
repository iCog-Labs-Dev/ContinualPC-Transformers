"""Tests for the minimal BERT-style masked-language-model path."""

import os
import json

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from examples.bert_demo import load_checkpoint, save_checkpoint

from fabricpc.core.inference import InferenceSGDNormClip
from fabricpc.core.types import GraphParams, GraphState, NodeState
from fabricpc.graph_initialization import initialize_params
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.nodes.encoder_transformer import (
    copy_encoder_params,
    create_encoder_mlm_model,
    create_encoder_sentiment_classifier,
)
import fabricpc.training as training
from fabricpc.training.train_backprop import (
    compute_loss,
    evaluate_mlm,
    load_mlm_training_checkpoint,
    train_backprop_mlm,
    train_step_backprop_mlm,
)
from fabricpc.utils.data import MlmTextBatcher


def build_small_mlm():
    return create_encoder_mlm_model(
        depth=1,
        embed_dim=16,
        num_heads=4,
        mlp_dim=32,
        seq_len=6,
        vocab_size=20,
        inference=InferenceSGDNormClip(eta_infer=0.1, infer_steps=3, max_norm=5.0),
    )


def tree_any_changed(before, after):
    before_leaves = jax.tree_util.tree_leaves(before)
    after_leaves = jax.tree_util.tree_leaves(after)
    assert len(before_leaves) == len(after_leaves)
    return any(not jnp.allclose(a, b) for a, b in zip(before_leaves, after_leaves))


def tree_allclose(left, right):
    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    assert len(left_leaves) == len(right_leaves)
    return all(jnp.allclose(a, b) for a, b in zip(left_leaves, right_leaves))


def test_mlm_model_forward_shape_smoke(rng_key):
    structure = build_small_mlm()
    params = initialize_params(structure, rng_key)
    x = jnp.array([[1, 2, 3, 4, 0, 0], [1, 5, 6, 7, 8, 0]], dtype=jnp.int32)

    state = initialize_graph_state(
        structure,
        batch_size=2,
        rng_key=rng_key,
        clamps={"input_ids": x},
        params=params,
    )

    probs = state.nodes["mlm_head"].z_mu
    assert probs.shape == (2, 6, 20)
    assert jnp.all(jnp.isfinite(probs))
    assert jnp.allclose(jnp.sum(probs, axis=-1), 1.0, atol=1e-5)


def test_mlm_head_uses_masked_vocab_projection_node():
    structure = build_small_mlm()

    from fabricpc.nodes.encoder_transformer import MaskedVocabProjectionNode

    assert structure.nodes["mlm_head"].node_info.node_class is MaskedVocabProjectionNode


def test_mlm_training_helpers_are_exported_from_training_package():
    assert training.train_step_backprop_mlm is train_step_backprop_mlm
    assert training.evaluate_mlm is evaluate_mlm


def test_compute_loss_with_mask_ignores_unmasked_positions():
    predictions = jnp.array(
        [
            [
                [0.9, 0.1],
                [0.2, 0.8],
                [0.7, 0.3],
            ]
        ]
    )
    dummy = jnp.zeros_like(predictions)
    final_state = GraphState(
        nodes={
            "mlm": NodeState(
                z_latent=dummy,
                pre_activation=predictions,
                z_mu=predictions,
                error=dummy,
                energy=jnp.zeros((1,)),
                latent_grad=dummy,
            )
        },
        batch_size=1,
    )
    loss_mask = jnp.array([[True, False, False]])

    original_targets = jnp.array([[0, 1, 0]])
    changed_unmasked_targets = jnp.array([[0, 0, 1]])

    original_loss = compute_loss(
        final_state,
        original_targets,
        "mlm",
        loss_type="cross_entropy",
        loss_mask=loss_mask,
    )
    changed_loss = compute_loss(
        final_state,
        changed_unmasked_targets,
        "mlm",
        loss_type="cross_entropy",
        loss_mask=loss_mask,
    )

    assert jnp.allclose(original_loss, changed_loss)


def test_compute_loss_masked_mse_averages_over_masked_features():
    predictions = jnp.array([[[1.0, 3.0], [10.0, 10.0]]])
    targets = jnp.array([[[0.0, 1.0], [0.0, 0.0]]])
    dummy = jnp.zeros_like(predictions)
    final_state = GraphState(
        nodes={
            "out": NodeState(
                z_latent=dummy,
                pre_activation=predictions,
                z_mu=predictions,
                error=dummy,
                energy=jnp.zeros((1,)),
                latent_grad=dummy,
            )
        },
        batch_size=1,
    )

    loss = compute_loss(
        final_state,
        targets,
        "out",
        loss_type="mse",
        loss_mask=jnp.array([[True, False]]),
    )

    assert jnp.allclose(loss, 2.5)


def test_compute_loss_cross_entropy_rejects_broadcastable_mask():
    predictions = jnp.array([[[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]]])
    dummy = jnp.zeros_like(predictions)
    final_state = GraphState(
        nodes={
            "mlm": NodeState(
                z_latent=dummy,
                pre_activation=predictions,
                z_mu=predictions,
                error=dummy,
                energy=jnp.zeros((1,)),
                latent_grad=dummy,
            )
        },
        batch_size=1,
    )

    try:
        compute_loss(
            final_state,
            jnp.array([[0, 1, 0]]),
            "mlm",
            loss_type="cross_entropy",
            loss_mask=jnp.array([[True], [False], [True]]),
        )
    except ValueError as exc:
        assert "loss_mask shape" in str(exc)
        assert "must match" in str(exc)
    else:
        raise AssertionError("Expected cross-entropy loss_mask shape validation")


def test_demo_checkpoint_roundtrip_json_npz_and_encoder_copy(tmp_path, rng_key):
    source_structure = build_small_mlm()
    source_params = initialize_params(source_structure, rng_key)
    checkpoint_path = tmp_path / "mlm_checkpoint.json"
    vocab = {"[PAD]": 0, "[MASK]": 1, "great": 2}
    config = {"depth": 1, "embed_dim": 16, "num_heads": 4, "mlp_dim": 32, "seq_len": 6}

    save_checkpoint(str(checkpoint_path), source_params, vocab, config)
    loaded = load_checkpoint(str(checkpoint_path))

    assert checkpoint_path.exists()
    assert checkpoint_path.suffix == ".json"
    assert checkpoint_path.with_suffix(".json.npz").exists()
    assert isinstance(loaded["params"], GraphParams)
    assert loaded["vocab"] == vocab
    assert loaded["config"] == config
    assert tree_allclose(
        source_params.nodes["token_embeddings"],
        loaded["params"].nodes["token_embeddings"],
    )
    assert tree_allclose(
        source_params.nodes["L0_mha"], loaded["params"].nodes["L0_mha"]
    )

    target_structure = create_encoder_sentiment_classifier(
        depth=1,
        embed_dim=16,
        num_heads=4,
        mlp_dim=32,
        seq_len=6,
        vocab_size=20,
        num_classes=2,
        inference=InferenceSGDNormClip(eta_infer=0.1, infer_steps=3, max_norm=5.0),
    )
    _, target_key = jax.random.split(rng_key)
    target_params = initialize_params(target_structure, target_key)
    original_classifier = target_params.nodes["classifier"]
    copied_params = copy_encoder_params(loaded["params"], target_params)

    assert tree_allclose(
        loaded["params"].nodes["token_embeddings"],
        copied_params.nodes["token_embeddings"],
    )
    assert tree_allclose(
        loaded["params"].nodes["L0_mha"], copied_params.nodes["L0_mha"]
    )
    assert tree_allclose(original_classifier, copied_params.nodes["classifier"])
    with np.load(
        checkpoint_path.with_suffix(".json.npz"), allow_pickle=False
    ) as tensors:
        assert any(key.startswith("token_embeddings/weights/") for key in tensors)


def test_train_step_backprop_mlm_updates_params(rng_key):
    structure = build_small_mlm()
    params = initialize_params(structure, rng_key)
    optimizer = optax.adam(1e-2)
    opt_state = optimizer.init(params)
    batch = {
        "x": jnp.array([[1, 2, 3, 4, 0, 0], [1, 5, 6, 7, 8, 0]], dtype=jnp.int32),
        "y": jnp.array([[1, 9, 3, 4, 0, 0], [1, 5, 10, 7, 8, 0]], dtype=jnp.int32),
        "loss_mask": jnp.array(
            [
                [False, True, False, False, False, False],
                [False, False, True, False, False, False],
            ]
        ),
    }

    new_params, _, loss = train_step_backprop_mlm(
        params, opt_state, batch, structure, optimizer, rng_key
    )

    assert jnp.isfinite(loss)
    assert tree_any_changed(params, new_params)


def test_train_step_backprop_mlm_allows_empty_loss_mask_as_noop(rng_key):
    structure = build_small_mlm()
    params = initialize_params(structure, rng_key)
    optimizer = optax.adamw(1e-2, weight_decay=1e-1)
    opt_state = optimizer.init(params)
    batch = {
        "x": jnp.array([[1, 2, 3, 4, 0, 0]], dtype=jnp.int32),
        "y": jnp.array([[1, 2, 3, 4, 0, 0]], dtype=jnp.int32),
        "loss_mask": jnp.array([[False, False, False, False, False, False]]),
    }

    new_params, new_opt_state, loss = train_step_backprop_mlm(
        params, opt_state, batch, structure, optimizer, rng_key
    )

    assert float(loss) == 0.0
    assert tree_allclose(params, new_params)
    assert tree_allclose(opt_state, new_opt_state)


def test_train_backprop_mlm_rejects_invalid_intervals(rng_key):
    structure = build_small_mlm()
    params = initialize_params(structure, rng_key)
    optimizer = optax.adamw(1e-2)
    loader = MlmTextBatcher(
        token_ids=np.array([1, 5, 6, 7, 8, 2, 1, 9, 10, 11, 2], dtype=np.int32),
        seq_len=6,
        batch_size=2,
        pad_token_id=0,
        mask_token_id=3,
        vocab_size=20,
        special_token_ids={0, 1, 2, 3},
        seed=2,
        mask_prob=0.5,
        stride=3,
        shuffle=False,
    )

    with pytest.raises(ValueError, match="max_steps"):
        train_backprop_mlm(
            params,
            structure,
            loader,
            optimizer,
            {"max_steps": 0},
            rng_key,
            verbose=False,
        )

    with pytest.raises(ValueError, match="checkpoint_interval"):
        train_backprop_mlm(
            params,
            structure,
            loader,
            optimizer,
            {"max_steps": 1, "checkpoint_interval": 0},
            rng_key,
            verbose=False,
        )


def test_train_step_backprop_mlm_requires_loss_mask(rng_key):
    structure = build_small_mlm()
    params = initialize_params(structure, rng_key)
    optimizer = optax.adam(1e-2)
    opt_state = optimizer.init(params)
    batch = {
        "x": jnp.array([[1, 2, 3, 4, 0, 0]], dtype=jnp.int32),
        "y": jnp.array([[1, 2, 3, 4, 0, 0]], dtype=jnp.int32),
    }

    try:
        train_step_backprop_mlm(params, opt_state, batch, structure, optimizer, rng_key)
    except ValueError as exc:
        assert "loss_mask" in str(exc)
        assert "requires" in str(exc)
    else:
        raise AssertionError("Expected missing MLM loss_mask to be rejected")


def test_evaluate_mlm_counts_only_masked_tokens(rng_key):
    structure = build_small_mlm()
    params = initialize_params(structure, rng_key)
    batch = {
        "x": jnp.array([[1, 2, 3, 4, 0, 0]], dtype=jnp.int32),
        "y": jnp.array([[1, 2, 3, 4, 0, 0]], dtype=jnp.int32),
        "loss_mask": jnp.array([[False, True, False, True, False, False]]),
    }

    metrics = evaluate_mlm(params, structure, [batch], rng_key)

    assert metrics["num_masked_tokens"] == 2
    assert 0.0 <= metrics["masked_accuracy"] <= 1.0
    assert jnp.isfinite(metrics["masked_loss"])


def test_evaluate_mlm_rejects_malformed_loss_mask(rng_key):
    structure = build_small_mlm()
    params = initialize_params(structure, rng_key)
    batch = {
        "x": jnp.array([[1, 2, 3, 4, 0, 0]], dtype=jnp.int32),
        "y": jnp.array([[1, 2, 3, 4, 0, 0]], dtype=jnp.int32),
        "loss_mask": jnp.array([[True], [False], [True]]),
    }

    try:
        evaluate_mlm(params, structure, [batch], rng_key)
    except ValueError as exc:
        assert "loss_mask shape" in str(exc)
        assert "must match" in str(exc)
    else:
        raise AssertionError("Expected malformed eval loss_mask to be rejected")


def test_evaluate_mlm_restores_loader_state(rng_key):
    structure = build_small_mlm()
    params = initialize_params(structure, rng_key)
    loader = MlmTextBatcher(
        token_ids=np.array([1, 5, 6, 7, 8, 2, 1, 9, 10, 11, 2], dtype=np.int32),
        seq_len=6,
        batch_size=2,
        pad_token_id=0,
        mask_token_id=3,
        vocab_size=20,
        special_token_ids={0, 1, 2, 3},
        seed=2,
        mask_prob=0.5,
        stride=3,
        shuffle=False,
    )
    before = loader.state_dict()

    evaluate_mlm(params, structure, loader, rng_key)

    assert loader.state_dict() == before


def test_train_backprop_mlm_records_validation_steps(tmp_path, rng_key):
    structure = build_small_mlm()
    params = initialize_params(structure, rng_key)
    optimizer = optax.adamw(1e-2)

    def make_loader(seed):
        return MlmTextBatcher(
            token_ids=np.array([1, 5, 6, 7, 8, 2, 1, 9, 10, 11, 2], dtype=np.int32),
            seq_len=6,
            batch_size=2,
            pad_token_id=0,
            mask_token_id=3,
            vocab_size=20,
            special_token_ids={0, 1, 2, 3},
            seed=seed,
            mask_prob=0.5,
            stride=3,
            shuffle=False,
        )

    _, state = train_backprop_mlm(
        params,
        structure,
        make_loader(2),
        optimizer,
        {
            "max_steps": 2,
            "validation_interval": 1,
            "checkpoint_path": str(tmp_path / "mlm.json"),
        },
        rng_key,
        val_loader=make_loader(3),
        verbose=False,
    )

    assert [record["step"] for record in state["validation"]] == [1, 2]


def test_train_backprop_mlm_records_exposure_and_jsonl(tmp_path, rng_key):
    structure = build_small_mlm()
    params = initialize_params(structure, rng_key)
    optimizer = optax.adamw(1e-2)
    metrics_path = tmp_path / "metrics.jsonl"
    loader = MlmTextBatcher(
        token_ids=np.array([1, 5, 6, 7, 0, 0, 1, 8, 9, 2], dtype=np.int32),
        seq_len=6,
        batch_size=2,
        pad_token_id=0,
        mask_token_id=3,
        vocab_size=20,
        special_token_ids={0, 1, 2, 3},
        seed=2,
        mask_prob=1.0,
        stride=6,
        shuffle=False,
    )

    _, state = train_backprop_mlm(
        params,
        structure,
        loader,
        optimizer,
        {"max_steps": 1, "metrics_jsonl": str(metrics_path)},
        rng_key,
        verbose=False,
    )

    assert state["real_tokens_seen"] == 8
    assert state["real_tokens_seen_estimated"] is False
    assert state["max_token_slots_seen"] == 12
    assert state["masked_positions_seen"] == 5
    assert state["epochs_equivalent"] == 1.0
    records = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    assert records[-1]["event"] == "final"
    assert records[-1]["real_tokens_seen"] == 8


def test_train_backprop_mlm_counts_max_steps_as_optimizer_updates(rng_key):
    structure = build_small_mlm()
    params = initialize_params(structure, rng_key)
    optimizer = optax.adamw(1e-2)
    loader = MlmTextBatcher(
        token_ids=np.array(
            [
                1,
                5,
                6,
                7,
                8,
                2,
                1,
                9,
                10,
                11,
                12,
                2,
                1,
                13,
                14,
                15,
                16,
                2,
                1,
                17,
                18,
                5,
                6,
                2,
            ],
            dtype=np.int32,
        ),
        seq_len=6,
        batch_size=1,
        pad_token_id=0,
        mask_token_id=3,
        vocab_size=20,
        special_token_ids={0, 1, 2, 3},
        seed=2,
        mask_prob=1.0,
        stride=6,
        shuffle=False,
    )

    _, state = train_backprop_mlm(
        params,
        structure,
        loader,
        optimizer,
        {"max_steps": 2, "gradient_accumulation_steps": 2},
        rng_key,
        verbose=False,
    )

    assert state["step"] == 2
    assert state["update_steps"] == 2
    assert state["micro_steps"] == 4
    assert len(state["losses"]) == 2


def test_train_backprop_mlm_does_not_count_trailing_partial_accumulation(rng_key):
    structure = build_small_mlm()
    params = initialize_params(structure, rng_key)
    optimizer = optax.adamw(1e-2)
    loader = MlmTextBatcher(
        token_ids=np.array(
            [1, 5, 6, 7, 8, 2, 1, 9, 10, 11, 12, 2, 1, 13, 14, 15, 16, 2],
            dtype=np.int32,
        ),
        seq_len=6,
        batch_size=1,
        pad_token_id=0,
        mask_token_id=3,
        vocab_size=20,
        special_token_ids={0, 1, 2, 3},
        seed=2,
        mask_prob=1.0,
        stride=6,
        shuffle=False,
    )

    _, state = train_backprop_mlm(
        params,
        structure,
        loader,
        optimizer,
        {"max_steps": 1, "gradient_accumulation_steps": 2},
        rng_key,
        verbose=False,
    )

    assert state["step"] == 1
    assert state["micro_steps"] == 2
    assert state["real_tokens_seen"] == 12


def test_train_backprop_mlm_uses_validation_fn_for_interval_average(rng_key):
    structure = build_small_mlm()
    params = initialize_params(structure, rng_key)
    optimizer = optax.adamw(1e-2)
    loader = MlmTextBatcher(
        token_ids=np.array([1, 5, 6, 7, 8, 2, 1, 9, 10, 11, 12, 2], dtype=np.int32),
        seq_len=6,
        batch_size=1,
        pad_token_id=0,
        mask_token_id=3,
        vocab_size=20,
        special_token_ids={0, 1, 2, 3},
        seed=2,
        mask_prob=1.0,
        stride=6,
        shuffle=False,
    )

    def validation_fn(_params, _rng_key):
        return {"masked_loss": 1.5, "masked_accuracy": 0.25, "num_masked_tokens": 4}

    _, state = train_backprop_mlm(
        params,
        structure,
        loader,
        optimizer,
        {"max_steps": 2, "validation_interval": 1, "validation_fn": validation_fn},
        rng_key,
        verbose=False,
    )

    assert [record["masked_loss"] for record in state["validation"]] == [1.5, 1.5]


def test_train_backprop_mlm_resume_rejects_accumulation_mismatch(tmp_path, rng_key):
    structure = build_small_mlm()
    params = initialize_params(structure, rng_key)
    optimizer = optax.adamw(1e-2)
    checkpoint_path = tmp_path / "mlm.json"
    loader_kwargs = dict(
        token_ids=np.array([1, 5, 6, 7, 8, 2, 1, 9, 10, 11, 12, 2], dtype=np.int32),
        seq_len=6,
        batch_size=1,
        pad_token_id=0,
        mask_token_id=3,
        vocab_size=20,
        special_token_ids={0, 1, 2, 3},
        seed=2,
        mask_prob=1.0,
        stride=6,
        shuffle=False,
    )
    train_backprop_mlm(
        params,
        structure,
        MlmTextBatcher(**loader_kwargs),
        optimizer,
        {
            "max_steps": 1,
            "gradient_accumulation_steps": 2,
            "checkpoint_path": str(checkpoint_path),
        },
        rng_key,
        verbose=False,
    )

    with pytest.raises(ValueError, match="gradient_accumulation_steps"):
        train_backprop_mlm(
            params,
            structure,
            MlmTextBatcher(**loader_kwargs),
            optimizer,
            {
                "max_steps": 2,
                "gradient_accumulation_steps": 1,
                "resume_from": str(checkpoint_path),
            },
            rng_key,
            verbose=False,
        )


def test_load_mlm_training_checkpoint_rejects_unknown_version(tmp_path, rng_key):
    structure = build_small_mlm()
    params = initialize_params(structure, rng_key)
    optimizer = optax.adamw(1e-2)
    checkpoint_path = tmp_path / "mlm.json"
    train_backprop_mlm(
        params,
        structure,
        MlmTextBatcher(
            token_ids=np.array([1, 5, 6, 7, 8, 2, 1, 9, 10, 11, 2], dtype=np.int32),
            seq_len=6,
            batch_size=2,
            pad_token_id=0,
            mask_token_id=3,
            vocab_size=20,
            special_token_ids={0, 1, 2, 3},
            seed=2,
            mask_prob=0.5,
            stride=3,
            shuffle=False,
        ),
        optimizer,
        {"max_steps": 1, "checkpoint_path": str(checkpoint_path)},
        rng_key,
        verbose=False,
    )
    metadata = checkpoint_path.read_text()
    checkpoint_path.write_text(metadata.replace('"version": 1', '"version": 2'))

    with pytest.raises(ValueError, match="checkpoint version"):
        load_mlm_training_checkpoint(
            str(checkpoint_path), opt_state_template=optimizer.init(params)
        )


def test_copy_encoder_params_copies_encoder_and_leaves_head(rng_key):
    source_structure = build_small_mlm()
    target_structure = create_encoder_sentiment_classifier(
        depth=1,
        embed_dim=16,
        num_heads=4,
        mlp_dim=32,
        seq_len=6,
        vocab_size=20,
        num_classes=2,
        inference=InferenceSGDNormClip(eta_infer=0.1, infer_steps=3, max_norm=5.0),
    )
    source_params = initialize_params(source_structure, rng_key)
    _, target_key = jax.random.split(rng_key)
    target_params = initialize_params(target_structure, target_key)
    original_classifier = target_params.nodes["classifier"]

    copied_params = copy_encoder_params(source_params, target_params)

    assert tree_allclose(
        source_params.nodes["token_embeddings"], copied_params.nodes["token_embeddings"]
    )
    assert tree_allclose(source_params.nodes["L0_mha"], copied_params.nodes["L0_mha"])
    assert tree_allclose(original_classifier, copied_params.nodes["classifier"])
