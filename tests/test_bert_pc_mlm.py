"""Tests for BERT-style predictive-coding MLM training support."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from fabricpc.core.inference import InferenceSGDNormClip
from fabricpc.graph_initialization import initialize_params
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.nodes.encoder_transformer import (
    MaskedVocabProjectionNode,
    create_encoder_mlm_model,
)
from fabricpc.training.train_mlm import (
    create_pc_bp_gradient_comparison,
    pc_node_gradient_norms,
    train_mlm,
    train_step_pcn_mlm,
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
        inference=InferenceSGDNormClip(eta_infer=0.01, infer_steps=3, max_norm=5.0),
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


def tiny_batch(loss_mask=None):
    mask = jnp.array([[1, 0, 0, 0, 0, 0]], dtype=jnp.float32)
    return {
        "x": jnp.array([[1, 2, 3, 4, 0, 0]], dtype=jnp.int32),
        "y": jnp.array([[1, 2, 3, 4, 0, 0]], dtype=jnp.int32),
        "attention_mask": jnp.array([[1, 1, 1, 1, 0, 0]], dtype=jnp.float32),
        "loss_mask": mask if loss_mask is None else loss_mask,
    }


def tiny_loader(mask_prob: float = 1.0):
    return MlmTextBatcher(
        token_ids=np.array([1, 5, 6, 7, 8, 2, 1, 9, 10, 11, 12, 2], dtype=np.int32),
        seq_len=6,
        batch_size=1,
        pad_token_id=0,
        mask_token_id=3,
        vocab_size=20,
        special_token_ids={0, 1, 2, 3},
        seed=2,
        mask_prob=mask_prob,
        stride=6,
        shuffle=False,
    )


def tiny_optimizer_identity(lr: float = 1e-2):
    return {"type": "adam", "lr": lr}


def test_mlm_builder_uses_masked_output_head():
    structure = build_small_mlm()

    assert structure.task_map["loss_mask"] == "mlm_loss_mask"
    assert structure.nodes["mlm_head"].node_info.node_class is MaskedVocabProjectionNode
    assert "mlm_loss_mask->mlm_head:loss_mask" in structure.edges


def test_masked_output_energy_ignores_unmasked_labels(rng_key):
    structure = build_small_mlm()
    params = initialize_params(structure, rng_key)
    batch = tiny_batch()
    vocab_size = structure.nodes[structure.task_map["y"]].node_info.shape[-1]

    targets = jax.nn.one_hot(batch["y"], vocab_size, dtype=jnp.float32)
    changed_unmasked_targets = targets.at[:, 1:, :].set(
        jax.nn.one_hot(
            jnp.array([[5, 6, 7, 8, 9]], dtype=jnp.int32),
            vocab_size,
            dtype=jnp.float32,
        )
    )
    clamps = {
        structure.task_map["x"]: batch["x"],
        structure.task_map["attention_mask"]: batch["attention_mask"],
        structure.task_map["loss_mask"]: batch["loss_mask"],
        structure.task_map["y"]: targets,
    }
    changed_clamps = {**clamps, structure.task_map["y"]: changed_unmasked_targets}

    original = initialize_graph_state(
        structure, 1, rng_key, clamps=clamps, params=params
    )
    changed = initialize_graph_state(
        structure, 1, rng_key, clamps=changed_clamps, params=params
    )

    assert jnp.allclose(
        original.nodes["mlm_head"].energy,
        changed.nodes["mlm_head"].energy,
        atol=1e-6,
    )


def test_masked_output_energy_is_zero_for_empty_mask(rng_key):
    structure = build_small_mlm()
    params = initialize_params(structure, rng_key)
    batch = tiny_batch(loss_mask=jnp.zeros((1, 6), dtype=jnp.float32))
    vocab_size = structure.nodes[structure.task_map["y"]].node_info.shape[-1]
    targets = jax.nn.one_hot(batch["y"], vocab_size, dtype=jnp.float32)

    state = initialize_graph_state(
        structure,
        1,
        rng_key,
        clamps={
            structure.task_map["x"]: batch["x"],
            structure.task_map["attention_mask"]: batch["attention_mask"],
            structure.task_map["loss_mask"]: batch["loss_mask"],
            structure.task_map["y"]: targets,
        },
        params=params,
    )

    assert jnp.allclose(state.nodes["mlm_head"].energy, 0.0)
    assert jnp.allclose(state.nodes["mlm_head"].error, 0.0)


def test_train_mlm_backprop_dispatch_runs_one_step(rng_key):
    structure = build_small_mlm()
    params = initialize_params(structure, rng_key)
    optimizer = optax.adam(1e-2)

    updated_params, state = train_mlm(
        params,
        structure,
        [tiny_batch()],
        optimizer,
        {"max_steps": 1, "log_interval": 0},
        rng_key,
        mode="backprop",
        verbose=False,
    )

    assert state["mode"] == "backprop"
    assert state["step"] == 1
    assert tree_any_changed(params, updated_params)


def test_train_step_pcn_mlm_empty_mask_preserves_params_and_optimizer(rng_key):
    structure = build_small_mlm()
    params = initialize_params(structure, rng_key)
    optimizer = optax.adam(1e-2)
    opt_state = optimizer.init(params)
    batch = tiny_batch(loss_mask=jnp.zeros((1, 6), dtype=jnp.float32))

    updated_params, updated_opt_state, metrics, _initial, _free, _final = (
        train_step_pcn_mlm(
            params,
            opt_state,
            batch,
            structure,
            optimizer,
            rng_key,
        )
    )

    assert metrics["num_masked_tokens"] == 0
    assert tree_allclose(params, updated_params)
    assert tree_allclose(opt_state, updated_opt_state)


def test_train_mlm_pc_saves_and_resumes_checkpoint(tmp_path, rng_key):
    structure = build_small_mlm()
    params = initialize_params(structure, rng_key)
    optimizer = optax.adam(1e-2)
    checkpoint_path = tmp_path / "pc_mlm.json"

    _first_params, first_state = train_mlm(
        params,
        structure,
        [tiny_batch()],
        optimizer,
        {
            "max_steps": 1,
            "checkpoint_path": str(checkpoint_path),
            "optimizer_identity": tiny_optimizer_identity(),
        },
        rng_key,
        mode="pc",
        verbose=False,
    )
    resumed_params, resumed_state = train_mlm(
        params,
        structure,
        [tiny_batch()],
        optimizer,
        {
            "max_steps": 2,
            "checkpoint_path": str(checkpoint_path),
            "resume_from": str(checkpoint_path),
            "optimizer_identity": tiny_optimizer_identity(),
        },
        rng_key,
        mode="pc",
        verbose=False,
    )

    assert checkpoint_path.exists()
    assert checkpoint_path.with_suffix(".json.npz").exists()
    assert first_state["step"] == 1
    assert resumed_state["step"] == 2
    assert tree_any_changed(params, resumed_params)


def test_train_mlm_pc_supports_gradient_accumulation(rng_key):
    structure = build_small_mlm()
    params = initialize_params(structure, rng_key)
    optimizer = optax.adam(1e-2)

    updated_params, state = train_mlm(
        params,
        structure,
        [tiny_batch(), tiny_batch()],
        optimizer,
        {
            "max_steps": 1,
            "gradient_accumulation_steps": 2,
        },
        rng_key,
        mode="pc",
        verbose=False,
    )

    assert state["step"] == 1
    assert state["micro_steps"] == 2
    assert state["gradient_accumulation_steps"] == 2
    assert tree_any_changed(params, updated_params)


def test_train_mlm_pc_derives_accumulation_from_tokens_per_update(rng_key):
    structure = build_small_mlm()
    params = initialize_params(structure, rng_key)
    optimizer = optax.adam(1e-2)

    _updated_params, state = train_mlm(
        params,
        structure,
        tiny_loader(mask_prob=1.0),
        optimizer,
        {"max_steps": 1, "tokens_per_update": 12},
        rng_key,
        mode="pc",
        verbose=False,
    )

    assert state["step"] == 1
    assert state["micro_steps"] == 2
    assert state["gradient_accumulation_steps"] == 2


def test_train_mlm_pc_empty_mask_batches_do_not_count_toward_accumulation(rng_key):
    structure = build_small_mlm()
    params = initialize_params(structure, rng_key)
    optimizer = optax.adam(1e-2)
    empty_batch = tiny_batch(loss_mask=jnp.zeros((1, 6), dtype=jnp.float32))

    updated_params, state = train_mlm(
        params,
        structure,
        [empty_batch, tiny_batch(), tiny_batch()],
        optimizer,
        {"max_steps": 1, "gradient_accumulation_steps": 2},
        rng_key,
        mode="pc",
        verbose=False,
    )

    assert state["step"] == 1
    assert state["micro_steps"] == 2
    assert tree_any_changed(params, updated_params)


def test_train_mlm_pc_all_empty_masks_fail_clearly(rng_key):
    structure = build_small_mlm()
    params = initialize_params(structure, rng_key)
    optimizer = optax.adam(1e-2)
    empty_batch = tiny_batch(loss_mask=jnp.zeros((1, 6), dtype=jnp.float32))

    with pytest.raises(ValueError, match="supervised MLM batches"):
        train_mlm(
            params,
            structure,
            [empty_batch],
            optimizer,
            {"max_steps": 1},
            rng_key,
            mode="pc",
            verbose=False,
        )


def test_train_mlm_pc_resume_rejects_training_identity_mismatch(tmp_path, rng_key):
    structure = build_small_mlm()
    params = initialize_params(structure, rng_key)
    optimizer = optax.adam(1e-2)
    checkpoint_path = tmp_path / "pc_mlm.json"

    train_mlm(
        params,
        structure,
        [tiny_batch()],
        optimizer,
        {
            "max_steps": 1,
            "checkpoint_path": str(checkpoint_path),
            "optimizer_identity": tiny_optimizer_identity(),
        },
        rng_key,
        mode="pc",
        verbose=False,
    )
    changed_structure = create_encoder_mlm_model(
        depth=1,
        embed_dim=16,
        num_heads=4,
        mlp_dim=32,
        seq_len=6,
        vocab_size=20,
        inference=InferenceSGDNormClip(eta_infer=0.02, infer_steps=3, max_norm=5.0),
        use_attention_mask=True,
    )

    with pytest.raises(ValueError, match="training identity"):
        train_mlm(
            params,
            changed_structure,
            [tiny_batch()],
            optimizer,
            {
                "max_steps": 2,
                "resume_from": str(checkpoint_path),
                "optimizer_identity": tiny_optimizer_identity(),
            },
            rng_key,
            mode="pc",
            verbose=False,
        )


def test_train_mlm_pc_resume_rejects_loader_identity_mismatch(tmp_path, rng_key):
    structure = build_small_mlm()
    params = initialize_params(structure, rng_key)
    optimizer = optax.adam(1e-2)
    checkpoint_path = tmp_path / "pc_mlm.json"

    train_mlm(
        params,
        structure,
        tiny_loader(mask_prob=1.0),
        optimizer,
        {
            "max_steps": 1,
            "checkpoint_path": str(checkpoint_path),
            "optimizer_identity": tiny_optimizer_identity(),
        },
        rng_key,
        mode="pc",
        verbose=False,
    )

    with pytest.raises(ValueError, match="training identity"):
        train_mlm(
            params,
            structure,
            tiny_loader(mask_prob=0.5),
            optimizer,
            {
                "max_steps": 2,
                "resume_from": str(checkpoint_path),
                "optimizer_identity": tiny_optimizer_identity(),
            },
            rng_key,
            mode="pc",
            verbose=False,
        )


def test_train_mlm_pc_resume_rejects_training_config_mismatch(tmp_path, rng_key):
    structure = build_small_mlm()
    params = initialize_params(structure, rng_key)
    optimizer = optax.adam(1e-2)
    checkpoint_path = tmp_path / "pc_mlm.json"

    train_mlm(
        params,
        structure,
        [tiny_batch()],
        optimizer,
        {
            "max_steps": 1,
            "checkpoint_path": str(checkpoint_path),
            "optimizer_identity": tiny_optimizer_identity(lr=1e-2),
        },
        rng_key,
        mode="pc",
        verbose=False,
    )

    with pytest.raises(ValueError, match="training config"):
        train_mlm(
            params,
            structure,
            [tiny_batch()],
            optimizer,
            {
                "max_steps": 2,
                "resume_from": str(checkpoint_path),
                "optimizer_identity": tiny_optimizer_identity(lr=1e-3),
            },
            rng_key,
            mode="pc",
            verbose=False,
        )


def test_train_mlm_pc_checkpoint_requires_optimizer_identity(tmp_path, rng_key):
    structure = build_small_mlm()
    params = initialize_params(structure, rng_key)
    optimizer = optax.adam(1e-2)

    with pytest.raises(ValueError, match="optimizer_identity"):
        train_mlm(
            params,
            structure,
            [tiny_batch()],
            optimizer,
            {"max_steps": 1, "checkpoint_path": str(tmp_path / "pc_mlm.json")},
            rng_key,
            mode="pc",
            verbose=False,
        )


def _small_mlm_with_inference(eta_infer: float, infer_steps: int):
    return create_encoder_mlm_model(
        depth=1,
        embed_dim=16,
        num_heads=4,
        mlp_dim=32,
        seq_len=6,
        vocab_size=20,
        inference=InferenceSGDNormClip(
            eta_infer=eta_infer, infer_steps=infer_steps, max_norm=5.0
        ),
        use_attention_mask=True,
    )


def test_pc_node_gradient_norms_zero_for_unreached_embeddings(rng_key):
    structure = _small_mlm_with_inference(eta_infer=0.01, infer_steps=1)
    params = initialize_params(structure, rng_key)

    norms = pc_node_gradient_norms(params, tiny_batch(), structure, rng_key)

    assert set(norms) == {
        name for name in params.nodes if jax.tree_util.tree_leaves(params.nodes[name])
    }
    assert norms["token_embeddings"] == 0.0
    assert norms["mlm_head"] > 0.0


def test_pc_node_gradient_norms_reach_embeddings_with_enough_inference(rng_key):
    structure = _small_mlm_with_inference(eta_infer=0.3, infer_steps=20)
    params = initialize_params(structure, rng_key)

    norms = pc_node_gradient_norms(params, tiny_batch(), structure, rng_key)

    assert all(np.isfinite(list(norms.values())))
    assert all(norm > 0.0 for norm in norms.values())


def test_pc_bp_gradient_comparison_matches_backprop_at_output_head(rng_key):
    structure = _small_mlm_with_inference(eta_infer=0.01, infer_steps=1)
    params = initialize_params(structure, rng_key)

    result = create_pc_bp_gradient_comparison(structure)(params, tiny_batch(), rng_key)

    head = result["mlm_head"]
    assert float(head["cosine"]) == pytest.approx(1.0, abs=1e-4)
    assert float(head["pc_norm"]) == pytest.approx(float(head["bp_norm"]), rel=1e-2)
    assert all(-1.0001 <= float(row["cosine"]) <= 1.0001 for row in result.values())
