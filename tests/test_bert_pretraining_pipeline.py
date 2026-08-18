"""Tests for BERT-style MLM pretraining infrastructure."""

import os
import json

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from fabricpc.core.inference import InferenceSGDNormClip
from fabricpc.graph_initialization import initialize_params
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.models import create_encoder_mlm_model
from fabricpc.training.train_backprop import (
    create_pmap_train_step_backprop_mlm,
    train_backprop_mlm,
)
from fabricpc.utils.data.mlm_loader import (
    MlmTextBatcher,
    apply_mlm_mask,
    prepare_bpe_tokenized_texts,
)


def build_masked_encoder():
    return create_encoder_mlm_model(
        depth=1,
        embed_dim=16,
        num_heads=4,
        mlp_dim=32,
        seq_len=6,
        vocab_size=24,
        inference=InferenceSGDNormClip(eta_infer=0.1, infer_steps=3, max_norm=5.0),
        use_attention_mask=True,
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


def test_apply_mlm_mask_masks_only_valid_tokens():
    tokens = np.array([[1, 5, 6, 0, 0]], dtype=np.int32)
    masked, targets, loss_mask = apply_mlm_mask(
        tokens,
        mask_token_id=3,
        vocab_size=10,
        special_token_ids={0, 1, 2, 3},
        seed=0,
        mask_prob=1.0,
    )

    assert np.array_equal(targets, tokens)
    assert np.array_equal(loss_mask, np.array([[False, True, True, False, False]]))
    assert np.all(masked[loss_mask] != targets[loss_mask])
    assert np.array_equal(masked[~loss_mask], tokens[~loss_mask])


def test_mlm_text_batcher_emits_attention_mask_and_mlm_fields():
    batcher = MlmTextBatcher(
        token_ids=np.array([1, 5, 6, 2, 1, 7, 2], dtype=np.int32),
        seq_len=5,
        batch_size=2,
        pad_token_id=0,
        mask_token_id=3,
        vocab_size=10,
        special_token_ids={0, 1, 2, 3},
        seed=0,
        mask_prob=1.0,
        stride=3,
        shuffle=False,
    )

    batch = next(iter(batcher))

    assert set(batch) == {"x", "y", "loss_mask", "attention_mask"}
    assert batch["x"].shape == (2, 5)
    assert batch["y"].shape == (2, 5)
    assert batch["loss_mask"].shape == (2, 5)
    assert batch["attention_mask"].shape == (2, 5)
    assert batch["attention_mask"].dtype == bool
    assert np.any(batch["loss_mask"])


def test_bpe_cache_uses_separate_token_files_per_split(tmp_path):
    pytest.importorskip("tokenizers")

    train_tokens, _ = prepare_bpe_tokenized_texts(
        texts=["warm happy movie"],
        cache_dir=tmp_path,
        vocab_size=32,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
        token_stem="tokens_train",
    )
    val_tokens, _ = prepare_bpe_tokenized_texts(
        texts=["cold sad movie"],
        cache_dir=tmp_path,
        vocab_size=32,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
        token_stem="tokens_validation",
    )
    reloaded_train_tokens, _ = prepare_bpe_tokenized_texts(
        texts=["warm happy movie"],
        cache_dir=tmp_path,
        vocab_size=32,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
        token_stem="tokens_train",
    )

    assert (tmp_path / "tokens_train.npy").exists()
    assert (tmp_path / "tokens_validation.npy").exists()
    assert not np.array_equal(train_tokens, val_tokens)
    assert np.array_equal(train_tokens, reloaded_train_tokens)

    prepare_bpe_tokenized_texts(
        texts=["new warm corpus"],
        cache_dir=tmp_path,
        vocab_size=32,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
        token_stem="tokens_train",
        overwrite=True,
    )
    assert not (tmp_path / "tokens_validation.npy").exists()


def test_encoder_attention_mask_prevents_pad_token_attention():
    rng_key = jax.random.PRNGKey(0)
    structure = build_masked_encoder()
    params = initialize_params(structure, rng_key)
    base = jnp.array([[1, 5, 6, 0, 0, 0]], dtype=jnp.int32)
    changed_pad = base.at[:, 4].set(9)
    attention_mask = jnp.array([[True, True, True, False, False, False]])

    base_state = initialize_graph_state(
        structure,
        batch_size=1,
        rng_key=rng_key,
        clamps={"input_ids": base, "attention_mask": attention_mask},
        params=params,
    )
    changed_state = initialize_graph_state(
        structure,
        batch_size=1,
        rng_key=rng_key,
        clamps={"input_ids": changed_pad, "attention_mask": attention_mask},
        params=params,
    )

    real_token_delta = jnp.max(
        jnp.abs(
            base_state.nodes["L0_mha"].z_mu[:, :3, :]
            - changed_state.nodes["L0_mha"].z_mu[:, :3, :]
        )
    )
    assert real_token_delta < 1e-6


def test_train_backprop_mlm_saves_and_resumes_checkpoint(tmp_path):
    rng_key = jax.random.PRNGKey(1)
    structure = build_masked_encoder()
    params = initialize_params(structure, rng_key)
    optimizer = optax.adamw(1e-2)
    loader = MlmTextBatcher(
        token_ids=np.array([1, 5, 6, 7, 8, 2, 1, 9, 10, 11, 2], dtype=np.int32),
        seq_len=6,
        batch_size=2,
        pad_token_id=0,
        mask_token_id=3,
        vocab_size=24,
        special_token_ids={0, 1, 2, 3},
        seed=2,
        mask_prob=0.5,
        stride=3,
        shuffle=False,
    )
    checkpoint_path = tmp_path / "mlm_checkpoint.json"

    trained, state = train_backprop_mlm(
        params,
        structure,
        loader,
        optimizer,
        {"max_steps": 1, "checkpoint_path": str(checkpoint_path)},
        rng_key,
        verbose=False,
    )
    resumed, resumed_state = train_backprop_mlm(
        params,
        structure,
        loader,
        optimizer,
        {"max_steps": 2, "resume_from": str(checkpoint_path)},
        rng_key,
        verbose=False,
    )

    assert checkpoint_path.exists()
    assert checkpoint_path.with_suffix(".json.npz").exists()
    metadata = json.loads(checkpoint_path.read_text())
    assert metadata["rng_key"] is not None
    assert "batch_in_epoch" in metadata["loader_state"]
    assert state["step"] == 1
    assert resumed_state["step"] == 2
    assert tree_any_changed(params, trained)
    assert tree_any_changed(trained, resumed)


def test_train_backprop_mlm_resume_matches_uninterrupted_run(tmp_path, rng_key):
    structure = build_masked_encoder()
    params = initialize_params(structure, rng_key)
    optimizer = optax.adamw(1e-2)

    def make_loader():
        return MlmTextBatcher(
            token_ids=np.array([1, 5, 6, 7, 8, 2, 1, 9, 10, 11, 2], dtype=np.int32),
            seq_len=6,
            batch_size=2,
            pad_token_id=0,
            mask_token_id=3,
            vocab_size=24,
            special_token_ids={0, 1, 2, 3},
            seed=2,
            mask_prob=0.5,
            stride=3,
            shuffle=False,
        )

    uninterrupted, _ = train_backprop_mlm(
        params,
        structure,
        make_loader(),
        optimizer,
        {"max_steps": 2},
        rng_key,
        verbose=False,
    )
    checkpoint_path = tmp_path / "resume_exact.json"
    train_backprop_mlm(
        params,
        structure,
        make_loader(),
        optimizer,
        {"max_steps": 1, "checkpoint_path": str(checkpoint_path)},
        rng_key,
        verbose=False,
    )
    resumed, _ = train_backprop_mlm(
        params,
        structure,
        make_loader(),
        optimizer,
        {"max_steps": 2, "resume_from": str(checkpoint_path)},
        rng_key,
        verbose=False,
    )

    assert tree_allclose(uninterrupted, resumed)


def test_train_backprop_mlm_rejects_empty_loader(rng_key):
    structure = build_masked_encoder()
    params = initialize_params(structure, rng_key)
    optimizer = optax.adamw(1e-2)
    loader = MlmTextBatcher(
        token_ids=np.array([1, 2, 3], dtype=np.int32),
        seq_len=6,
        batch_size=2,
        pad_token_id=0,
        mask_token_id=3,
        vocab_size=24,
        special_token_ids={0, 1, 2, 3},
        max_sequences=1,
    )

    with pytest.raises(ValueError, match="at least one full MLM batch"):
        train_backprop_mlm(
            params,
            structure,
            loader,
            optimizer,
            {"max_steps": 1},
            rng_key,
            verbose=False,
        )


def test_pmap_mlm_step_can_run_on_single_device_path():
    rng_key = jax.random.PRNGKey(3)
    structure = build_masked_encoder()
    params = initialize_params(structure, rng_key)
    optimizer = optax.adam(1e-2)
    opt_state = optimizer.init(params)
    batch = {
        "x": jnp.array([[1, 3, 6, 2, 0, 0]], dtype=jnp.int32),
        "y": jnp.array([[1, 5, 6, 2, 0, 0]], dtype=jnp.int32),
        "loss_mask": jnp.array([[False, True, False, False, False, False]]),
        "attention_mask": jnp.array([[True, True, True, True, False, False]]),
    }

    step_fn = create_pmap_train_step_backprop_mlm(structure, optimizer)
    replicated_params = jax.tree_util.tree_map(lambda x: x[None, ...], params)
    replicated_opt_state = jax.tree_util.tree_map(lambda x: x[None, ...], opt_state)
    sharded_batch = jax.tree_util.tree_map(lambda x: x[None, ...], batch)
    device_keys = jax.random.split(rng_key, 1)

    new_params, _, losses = step_fn(
        replicated_params, replicated_opt_state, sharded_batch, device_keys
    )

    assert losses.shape == (1,)
    assert jnp.isfinite(losses[0])
    assert tree_any_changed(replicated_params, new_params)
