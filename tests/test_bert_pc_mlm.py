"""Tests for BERT-style predictive-coding MLM training support."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from fabricpc.core.inference import InferenceSGDNormClip, InferenceSGDNormClipDecay
from fabricpc.graph_initialization import initialize_params
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.nodes.encoder_transformer import (
    MaskedVocabProjectionNode,
    create_encoder_mlm_model,
)
from fabricpc.core.learning import compute_local_weight_gradients
from fabricpc.training.optimizers import build_warmup_cosine_adamw
from fabricpc.training.train_mlm import (
    _pc_training_identity,
    _prepare_pcn_mlm_batch,
    _train_step_pcn_mlm_incremental_with_state,
    create_jit_train_step_pcn_mlm_incremental,
    get_graph_param_gradient_pcn_mlm,
    create_pc_bp_gradient_comparison,
    pc_node_gradient_norms,
    train_mlm,
    train_step_pcn_mlm,
    train_step_pcn_mlm_incremental,
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


def _small_mlm_with_guard(eta_infer: float, infer_steps: int, rise_decay: float):
    return create_encoder_mlm_model(
        depth=1,
        embed_dim=16,
        num_heads=4,
        mlp_dim=32,
        seq_len=6,
        vocab_size=20,
        inference=InferenceSGDNormClipDecay(
            eta_infer=eta_infer,
            infer_steps=infer_steps,
            max_norm=5.0,
            rise_decay=rise_decay,
        ),
        use_attention_mask=True,
    )


def test_guard_without_decay_matches_plain_inference(rng_key):
    plain = _small_mlm_with_inference(eta_infer=0.2, infer_steps=10)
    guarded = _small_mlm_with_guard(eta_infer=0.2, infer_steps=10, rise_decay=1.0)
    params = initialize_params(plain, rng_key)

    plain_grads, *_ = get_graph_param_gradient_pcn_mlm(
        params, tiny_batch(), plain, rng_key
    )
    guarded_grads, *_ = get_graph_param_gradient_pcn_mlm(
        params, tiny_batch(), guarded, rng_key
    )

    assert tree_allclose(plain_grads, guarded_grads)


def _guard_inputs(structure, params, batch, rng_key):
    output_node = structure.task_map["y"]
    clamps = {
        structure.task_map["x"]: batch["x"],
        structure.task_map["loss_mask"]: batch["loss_mask"],
        structure.task_map["attention_mask"]: batch["attention_mask"],
        output_node: jax.nn.one_hot(batch["y"], 20),
    }
    state = initialize_graph_state(
        structure, batch["x"].shape[0], rng_key, clamps=clamps, params=params
    )
    return state, clamps


def _two_sample_batch():
    single = tiny_batch()
    other = {
        "x": jnp.array([[5, 6, 7, 8, 9, 10]], dtype=jnp.int32),
        "y": jnp.array([[5, 6, 7, 8, 9, 10]], dtype=jnp.int32),
        "attention_mask": jnp.ones((1, 6), dtype=jnp.float32),
        "loss_mask": jnp.array([[0, 1, 0, 1, 0, 1]], dtype=jnp.float32),
    }
    return {key: jnp.concatenate([single[key], other[key]]) for key in single}


def test_guard_rate_follows_per_sample_rise_count(rng_key):
    structure = _small_mlm_with_guard(eta_infer=5.0, infer_steps=20, rise_decay=0.5)
    params = initialize_params(structure, rng_key)
    state, clamps = _guard_inputs(structure, params, _two_sample_batch(), rng_key)

    _final, stats = InferenceSGDNormClipDecay.run_inference_with_stats(
        params, state, clamps, structure
    )

    rises = int(stats["energy_rises"])
    final_eta = np.asarray(stats["final_eta"])
    per_sample_rises = np.round(np.log(final_eta / 5.0) / np.log(0.5))
    assert rises > 0
    assert final_eta.shape == (2,)
    assert int(per_sample_rises.sum()) == rises


def test_guard_runs_under_x64(rng_key):
    with jax.enable_x64(True):
        structure = _small_mlm_with_guard(eta_infer=0.2, infer_steps=5, rise_decay=0.9)
        params = initialize_params(structure, rng_key)
        state, clamps = _guard_inputs(structure, params, tiny_batch(), rng_key)

        final_state, stats = InferenceSGDNormClipDecay.run_inference_with_stats(
            params, state, clamps, structure
        )

    assert stats["energy_rises"].dtype == jnp.int32
    assert all(
        np.all(np.isfinite(np.asarray(node.z_latent)))
        for node in final_state.nodes.values()
    )


def test_guard_rejects_invalid_decay():
    with pytest.raises(ValueError, match="rise_decay"):
        InferenceSGDNormClipDecay(rise_decay=0.0)


def test_train_mlm_pc_logs_energy_rises_and_validation(rng_key, capsys):
    structure = _small_mlm_with_guard(eta_infer=5.0, infer_steps=5, rise_decay=0.5)
    params = initialize_params(structure, rng_key)

    train_mlm(
        params,
        structure,
        [tiny_batch()],
        optax.adam(1e-2),
        {"max_steps": 1, "log_interval": 1, "validation_interval": 1},
        rng_key,
        mode="pc",
        val_loader=[tiny_batch()],
        verbose=True,
    )

    out = capsys.readouterr().out
    assert "energy_rises=" in out
    assert "step=1 validation_masked_loss=" in out


def _assert_states_close(left, right):
    for name in left.nodes:
        assert jnp.allclose(
            left.nodes[name].z_latent, right.nodes[name].z_latent, atol=1e-6
        ), name


@pytest.mark.parametrize("guarded", [False, True])
def test_incremental_zero_lr_matches_standard_pc_latents(rng_key, guarded):
    structure = (
        _small_mlm_with_guard(eta_infer=0.5, infer_steps=6, rise_decay=0.5)
        if guarded
        else _small_mlm_with_inference(eta_infer=0.2, infer_steps=6)
    )
    params = initialize_params(structure, rng_key)
    optimizer = optax.sgd(0.0)
    batch = _two_sample_batch()

    new_params, _opt_state, metrics, final_state = (
        _train_step_pcn_mlm_incremental_with_state(
            params, optimizer.init(params), batch, structure, optimizer, rng_key
        )
    )
    _grads, pc_metrics, _initial, _free, pc_final = get_graph_param_gradient_pcn_mlm(
        params, batch, structure, rng_key
    )

    assert tree_allclose(params, new_params)
    _assert_states_close(final_state, pc_final)
    assert int(metrics["pc_energy_rises"]) == int(pc_metrics["pc_energy_rises"])
    assert jnp.allclose(metrics["masked_loss"], pc_metrics["masked_loss"])
    assert jnp.allclose(metrics["masked_accuracy"], pc_metrics["masked_accuracy"])
    assert metrics["num_masked_tokens"] == pc_metrics["num_masked_tokens"]


def test_incremental_single_step_matches_manual_update(rng_key):
    structure = _small_mlm_with_inference(eta_infer=0.2, infer_steps=1)
    params = initialize_params(structure, rng_key)
    optimizer = optax.sgd(0.1, momentum=0.9)
    opt_state = optimizer.init(params)
    batch = _two_sample_batch()

    new_params, new_opt_state, metrics = train_step_pcn_mlm_incremental(
        params, opt_state, batch, structure, optimizer, rng_key
    )
    _grads, _metrics, initial_state, _free, _final = get_graph_param_gradient_pcn_mlm(
        params, batch, structure, rng_key
    )
    expected_grads = compute_local_weight_gradients(params, initial_state, structure)
    updates, expected_opt_state = optimizer.update(expected_grads, opt_state, params)
    expected_params = optax.apply_updates(params, updates)

    assert tree_any_changed(params, new_params)
    assert tree_allclose(expected_params, new_params)
    assert all(
        jnp.allclose(left, right, atol=1e-6)
        for left, right in zip(
            jax.tree_util.tree_leaves(expected_opt_state),
            jax.tree_util.tree_leaves(new_opt_state),
        )
    )
    assert jnp.allclose(metrics["grad_norm_mean"], optax.tree.norm(expected_grads))
    assert metrics["pc_energy_rises"].dtype == jnp.int32
    assert int(metrics["pc_energy_rises"]) == 0


def test_incremental_three_steps_match_manual_unroll(rng_key):
    structure = _small_mlm_with_inference(eta_infer=0.2, infer_steps=3)
    params = initialize_params(structure, rng_key)
    optimizer = optax.sgd(0.1, momentum=0.9)
    opt_state = optimizer.init(params)
    batch = _two_sample_batch()

    new_params, _new_opt_state, _metrics = train_step_pcn_mlm_incremental(
        params, opt_state, batch, structure, optimizer, rng_key
    )

    _shared, _free_metrics, _free, state, clamps = _prepare_pcn_mlm_batch(
        params, batch, structure, rng_key
    )
    inference = structure.config["inference"]
    expected_params, expected_opt_state = params, opt_state
    for _ in range(3):
        grads = compute_local_weight_gradients(expected_params, state, structure)
        state = type(inference).inference_step(
            expected_params, state, clamps, structure, dict(inference.config)
        )
        updates, expected_opt_state = optimizer.update(
            grads, expected_opt_state, expected_params
        )
        expected_params = optax.apply_updates(expected_params, updates)

    assert all(
        jnp.allclose(left, right, atol=1e-5)
        for left, right in zip(
            jax.tree_util.tree_leaves(expected_params),
            jax.tree_util.tree_leaves(new_params),
        )
    )


def test_incremental_empty_mask_preserves_params_and_optimizer(rng_key):
    structure = _small_mlm_with_inference(eta_infer=0.2, infer_steps=3)
    params = initialize_params(structure, rng_key)
    optimizer = optax.adam(1e-2)
    opt_state = optimizer.init(params)
    batch = tiny_batch(loss_mask=jnp.zeros((1, 6), dtype=jnp.float32))

    new_params, new_opt_state, metrics = train_step_pcn_mlm_incremental(
        params, opt_state, batch, structure, optimizer, rng_key
    )

    assert metrics["num_masked_tokens"] == 0
    assert tree_allclose(params, new_params)
    assert tree_allclose(opt_state, new_opt_state)


def test_incremental_optimizer_steps_once_per_inference_step(rng_key):
    infer_steps = 4
    structure = _small_mlm_with_inference(eta_infer=0.2, infer_steps=infer_steps)
    params = initialize_params(structure, rng_key)
    optimizer = build_warmup_cosine_adamw(
        learning_rate=1e-2,
        total_steps=10,
        warmup_steps=3,
        schedule_step_divisor=infer_steps,
    )
    step_fn = create_jit_train_step_pcn_mlm_incremental(structure, optimizer)
    opt_state = optimizer.init(params)

    for batch_index in range(1, 3):
        params, opt_state, _metrics = step_fn(params, opt_state, tiny_batch(), rng_key)
        counts = {
            int(count)
            for _path, count in optax.tree_utils.tree_get_all_with_path(
                opt_state, "count"
            )
        }
        assert counts == {batch_index * infer_steps}


def test_incremental_guard_counts_rises_on_two_sample_batch(rng_key):
    structure = _small_mlm_with_guard(eta_infer=5.0, infer_steps=10, rise_decay=0.5)
    params = initialize_params(structure, rng_key)
    optimizer = optax.adam(1e-3)

    new_params, _opt_state, metrics = train_step_pcn_mlm_incremental(
        params,
        optimizer.init(params),
        _two_sample_batch(),
        structure,
        optimizer,
        rng_key,
    )

    assert metrics["pc_energy_rises"].dtype == jnp.int32
    assert int(metrics["pc_energy_rises"]) > 0
    assert np.isfinite(float(metrics["grad_norm_mean"]))
    assert tree_any_changed(params, new_params)


def test_incremental_guard_runs_under_x64(rng_key):
    with jax.enable_x64(True):
        structure = _small_mlm_with_guard(eta_infer=0.2, infer_steps=4, rise_decay=0.9)
        params = initialize_params(structure, rng_key)
        optimizer = optax.adam(1e-3)
        step_fn = create_jit_train_step_pcn_mlm_incremental(structure, optimizer)

        new_params, _opt_state, metrics = step_fn(
            params, optimizer.init(params), _two_sample_batch(), rng_key
        )

    assert metrics["pc_energy_rises"].dtype == jnp.int32
    assert np.isfinite(float(metrics["grad_norm_mean"]))
    assert all(
        np.all(np.isfinite(np.asarray(leaf)))
        for leaf in jax.tree_util.tree_leaves(new_params)
    )


def test_train_mlm_pc_incremental_logs_and_validates(rng_key, capsys):
    structure = _small_mlm_with_guard(eta_infer=5.0, infer_steps=5, rise_decay=0.5)
    params = initialize_params(structure, rng_key)

    updated_params, state = train_mlm(
        params,
        structure,
        [tiny_batch(), tiny_batch()],
        optax.adam(1e-2),
        {
            "max_steps": 2,
            "log_interval": 1,
            "validation_interval": 1,
            "pc_incremental": True,
        },
        rng_key,
        mode="pc",
        val_loader=[tiny_batch()],
        verbose=True,
    )

    out = capsys.readouterr().out
    assert state["step"] == 2
    assert state["pc_incremental"] is True
    assert "step=2 masked_loss=" in out
    assert "grad_norm=" in out
    assert "energy_rises=" in out
    assert "step=1 validation_masked_loss=" in out
    assert tree_any_changed(params, updated_params)


@pytest.mark.parametrize(
    "extra",
    [{"gradient_accumulation_steps": 2}, {"tokens_per_update": 12}],
)
def test_train_mlm_pc_incremental_rejects_accumulation(rng_key, extra):
    structure = _small_mlm_with_inference(eta_infer=0.2, infer_steps=2)
    params = initialize_params(structure, rng_key)

    with pytest.raises(ValueError, match="gradient accumulation"):
        train_mlm(
            params,
            structure,
            tiny_loader(mask_prob=1.0),
            optax.adam(1e-2),
            {"max_steps": 1, "pc_incremental": True, **extra},
            rng_key,
            mode="pc",
            verbose=False,
        )


def test_train_mlm_pc_incremental_resume_and_mode_mismatch(tmp_path, rng_key):
    structure = _small_mlm_with_inference(eta_infer=0.2, infer_steps=2)
    params = initialize_params(structure, rng_key)
    optimizer = optax.adam(1e-2)
    checkpoint_path = tmp_path / "ipc_mlm.json"
    base = {
        "checkpoint_path": str(checkpoint_path),
        "optimizer_identity": tiny_optimizer_identity(),
    }

    train_mlm(
        params,
        structure,
        [tiny_batch()],
        optimizer,
        {**base, "max_steps": 1, "pc_incremental": True},
        rng_key,
        mode="pc",
        verbose=False,
    )
    _resumed, resumed_state = train_mlm(
        params,
        structure,
        [tiny_batch()],
        optimizer,
        {
            **base,
            "max_steps": 2,
            "resume_from": str(checkpoint_path),
            "pc_incremental": True,
        },
        rng_key,
        mode="pc",
        verbose=False,
    )
    assert resumed_state["step"] == 2
    assert _pc_training_identity(
        structure, [tiny_batch()], True
    ) != _pc_training_identity(structure, [tiny_batch()], False)

    with pytest.raises(ValueError, match="training (config|identity)"):
        train_mlm(
            params,
            structure,
            [tiny_batch()],
            optimizer,
            {
                "optimizer_identity": tiny_optimizer_identity(),
                "max_steps": 3,
                "resume_from": str(checkpoint_path),
            },
            rng_key,
            mode="pc",
            verbose=False,
        )
