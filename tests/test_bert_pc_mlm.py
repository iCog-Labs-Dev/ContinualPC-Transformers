"""Tests for BERT-style predictive-coding MLM training support."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from fabricpc.core.activations import TanhActivation
from fabricpc.core.depth import node_output_depths
from fabricpc.core.inference import (
    InferenceSGDNormClip,
    InferenceSGDNormClipDecay,
    InferenceSpikingPrecision,
)
from fabricpc.core.initializers import NormalInitializer
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import (
    FeedforwardStateInit,
    GlobalStateInit,
    initialize_params,
)
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.nodes.encoder_transformer import (
    MaskedVocabProjectionNode,
    create_encoder_mlm_model,
)
from fabricpc.core.learning import (
    compute_local_weight_gradients,
    compute_local_weight_gradients_forward_update,
)
from fabricpc.nodes.identity import IdentityNode
from fabricpc.nodes.linear import Linear
from fabricpc.nodes.linear_residual import LinearResidual
from fabricpc.training.train_backprop import _loss_and_grad_backprop_mlm_no_update
from fabricpc.training.optimizers import build_warmup_cosine_adamw
from fabricpc.training.train_mlm import (
    _pc_training_identity,
    _prepare_pcn_mlm_batch,
    _scale_spike_step_grads,
    ipc_weight_spike_steps,
    _train_step_pcn_mlm_incremental_with_state,
    create_jit_gradient_step_pcn_mlm,
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


def _spiking_mlm(depth: int = 1, transformer_init_std: float = 0.02, **inference):
    config = {"eta_infer": 0.2, "infer_steps": 6, "max_norm": 5.0, **inference}
    return create_encoder_mlm_model(
        depth=depth,
        embed_dim=16,
        num_heads=4,
        mlp_dim=32,
        seq_len=6,
        vocab_size=20,
        inference=InferenceSpikingPrecision(**config),
        use_attention_mask=True,
        transformer_init_std=transformer_init_std,
    )


def _max_free_depth(structure, mode: str = "longest") -> int:
    depths = node_output_depths(structure, mode)
    return max(
        depth
        for name, depth in depths.items()
        if structure.nodes[name].node_info.in_degree > 0
    )


def _relative_errors(left, right):
    errors = {}
    for name, left_node in left.nodes.items():
        left_leaves = jax.tree_util.tree_leaves(left_node)
        if not left_leaves:
            continue
        right_leaves = jax.tree_util.tree_leaves(right.nodes[name])
        diff = sum(jnp.sum((a - b) ** 2) for a, b in zip(left_leaves, right_leaves))
        norm = sum(jnp.sum(b**2) for b in right_leaves)
        errors[name] = float(jnp.sqrt(diff / norm))
    return errors


def _cosines(left, right):
    cosines = {}
    for name, left_node in left.nodes.items():
        left_leaves = jax.tree_util.tree_leaves(left_node)
        if not left_leaves:
            continue
        right_leaves = jax.tree_util.tree_leaves(right.nodes[name])
        dot = sum(jnp.vdot(a, b) for a, b in zip(left_leaves, right_leaves))
        cosines[name] = float(
            dot / (optax.tree.norm(left_leaves) * optax.tree.norm(right_leaves))
        )
    return cosines


def _zil_encoder_errors(depth_mode: str, transformer_init_std: float = 0.02):
    """Return PC-vs-BP relative errors and cosines under the Z-IL schedule."""
    structure = _spiking_mlm(
        depth=2,
        transformer_init_std=transformer_init_std,
        eta_infer=0.0,
        infer_steps=1,
        depth_mode=depth_mode,
    )
    steps = _max_free_depth(structure, "longest")
    structure = _spiking_mlm(
        depth=2,
        transformer_init_std=transformer_init_std,
        eta_infer=0.0,
        infer_steps=steps,
        depth_mode=depth_mode,
    )
    key = jax.random.PRNGKey(0)
    params = initialize_params(structure, key)
    batch = _two_sample_batch()
    pc_grads, *_ = get_graph_param_gradient_pcn_mlm(
        params, batch, structure, key, forward_updates="initial"
    )
    _loss, bp_grads, _count = _loss_and_grad_backprop_mlm_no_update(
        params, batch, structure, key
    )
    assert jax.tree_util.tree_leaves(pc_grads)[0].dtype == jnp.float64
    return _relative_errors(pc_grads, bp_grads), _cosines(pc_grads, bp_grads)


def test_spiking_rejects_invalid_config():
    with pytest.raises(ValueError, match="spike_eta"):
        InferenceSpikingPrecision(spike_eta=0.0)
    with pytest.raises(ValueError, match="depth_mode"):
        InferenceSpikingPrecision(depth_mode="median")
    with pytest.raises(ValueError, match="spike_offset"):
        InferenceSpikingPrecision(spike_offset=1.5)
    with pytest.raises(ValueError, match="spike_offset must be non-negative"):
        InferenceSpikingPrecision(spike_offset=-1)
    with pytest.raises(ValueError, match="eta_infer"):
        InferenceSpikingPrecision(eta_infer=-0.1)
    assert InferenceSpikingPrecision(eta_infer=0.0).config["eta_infer"] == 0.0


def test_spiking_with_equal_rates_matches_norm_clip(rng_key):
    plain = _small_mlm_with_inference(eta_infer=0.2, infer_steps=6)
    spiking = _spiking_mlm(spike_eta=0.2, spike_clip=True)
    params = initialize_params(plain, rng_key)
    batch = _two_sample_batch()

    plain_grads, _m, _i, _f, plain_final = get_graph_param_gradient_pcn_mlm(
        params, batch, plain, rng_key
    )
    spiking_grads, _m, _i, _f, spiking_final = get_graph_param_gradient_pcn_mlm(
        params, batch, spiking, rng_key
    )

    for name in plain_final.nodes:
        assert jnp.allclose(
            plain_final.nodes[name].z_latent,
            spiking_final.nodes[name].z_latent,
            rtol=0.0,
            atol=0.0,
        ), name
    assert all(
        jnp.array_equal(a, b)
        for a, b in zip(
            jax.tree_util.tree_leaves(plain_grads),
            jax.tree_util.tree_leaves(spiking_grads),
        )
    )


def test_spike_step_is_unclipped_and_other_nodes_are_clipped(rng_key):
    structure = _spiking_mlm(eta_infer=0.1, max_norm=1e-3, spike_eta=0.7)
    params = initialize_params(structure, rng_key)
    state, clamps = _guard_inputs(structure, params, _two_sample_batch(), rng_key)
    noise_keys = jax.random.split(rng_key, len(structure.nodes))
    for key, name in zip(noise_keys, structure.nodes):
        if name not in clamps:
            node = state.nodes[name]
            state = state._replace(
                nodes={
                    **state.nodes,
                    name: node._replace(
                        z_latent=node.z_latent
                        + 0.1 * jax.random.normal(key, node.z_latent.shape)
                    ),
                }
            )
    config = dict(structure.config["inference"].config)
    spike_steps = InferenceSpikingPrecision.spike_steps(structure, config)
    t = spike_steps["L0_mlp1"]
    assert t == 2
    grad_state = InferenceSpikingPrecision.zero_grads(params, state, clamps, structure)
    grad_state = InferenceSpikingPrecision.forward_value_and_grad(
        params, grad_state, clamps, structure
    )

    new_state = InferenceSpikingPrecision.update_latents_at_step(
        params, grad_state, jnp.array(t), clamps, structure, config
    )

    spikers = {name for name, step in spike_steps.items() if step == t}
    assert spikers == {"L0_mlp1"}
    for name in structure.nodes:
        if name in clamps:
            continue
        before = grad_state.nodes[name]
        grad = before.latent_grad
        norms = jnp.sqrt(jnp.sum(grad**2, axis=tuple(range(1, grad.ndim))))
        assert float(jnp.min(norms)) > 1e-3, name
        if name in spikers:
            expected = before.z_latent - 0.7 * grad
        else:
            shape = (-1,) + (1,) * (grad.ndim - 1)
            clipped = grad * jnp.minimum(1.0, 1e-3 / (norms.reshape(shape) + 1e-8))
            expected = before.z_latent - 0.1 * clipped
        assert jnp.allclose(new_state.nodes[name].z_latent, expected, atol=1e-6), name


def test_spiking_zil_matches_backprop_downstream_of_skip_merges():
    with jax.enable_x64(True):
        errors, cosines = _zil_encoder_errors("longest")

    for name in ("mlm_head", "final_layernorm", "L1_mlp2", "L1_mlp1"):
        assert errors[name] < 1e-6, (name, errors[name])
        assert cosines[name] == pytest.approx(1.0, abs=1e-10)


@pytest.mark.xfail(
    strict=True,
    reason="two causes: (1) residual-merge desync, mha reaches mlp2 through "
    "mlp1 and the residual edge, and mlp1's spike changes mlp2's error before "
    "mha reads it, so no one-spike-per-node schedule reproduces backprop at "
    "mha and below; (2) muPC scales the top-down gradient on the GELU "
    "mlp1 input edge by jacobian_gain 1.168, which is not part of the chain rule",
)
def test_spiking_zil_matches_backprop_for_every_parameter():
    with jax.enable_x64(True):
        errors, cosines = _zil_encoder_errors("longest")

    assert max(errors.values()) < 1e-6, errors
    assert min(cosines.values()) == pytest.approx(1.0, abs=1e-10)


def test_spiking_zil_stays_close_to_backprop_at_default_init():
    with jax.enable_x64(True):
        errors, cosines = _zil_encoder_errors("longest")

    assert set(errors) == set(cosines)
    assert all(error < 1e-3 for error in errors.values()), errors
    assert all(cosine > 0.9999 for cosine in cosines.values()), cosines


def _zil_encoder_errors_with_dropout(dropout_rate: float):
    """Return PC-vs-BP relative errors and cosines with dropout masks active."""

    def build(infer_steps: int):
        return create_encoder_mlm_model(
            depth=2,
            embed_dim=16,
            num_heads=4,
            mlp_dim=32,
            seq_len=6,
            vocab_size=20,
            inference=InferenceSpikingPrecision(
                eta_infer=0.0, infer_steps=infer_steps, max_norm=5.0
            ),
            use_attention_mask=True,
            dropout_rate=dropout_rate,
        )

    structure = build(_max_free_depth(build(1), "longest"))
    key = jax.random.PRNGKey(0)
    params = initialize_params(structure, key)
    batch = _two_sample_batch()
    pc_grads, *_ = get_graph_param_gradient_pcn_mlm(
        params, batch, structure, key, forward_updates="initial"
    )
    _loss, bp_grads, _count = _loss_and_grad_backprop_mlm_no_update(
        params, batch, structure, key
    )
    return _relative_errors(pc_grads, bp_grads), _cosines(pc_grads, bp_grads)


def test_spiking_zil_matches_backprop_with_dropout():
    with jax.enable_x64(True):
        errors, cosines = _zil_encoder_errors_with_dropout(0.3)

    assert all(error < 1e-3 for error in errors.values()), errors
    assert all(cosine > 0.9999 for cosine in cosines.values()), cosines


def test_spiking_zil_shortest_depth_misaligns_block_inputs():
    with jax.enable_x64(True):
        longest, _ = _zil_encoder_errors("longest", transformer_init_std=0.3)
        shortest, _ = _zil_encoder_errors("shortest", transformer_init_std=0.3)

    block_nodes = ["L0_mha", "L0_mlp1", "L1_mha"]
    assert max(shortest[name] for name in block_nodes) > 1e-2
    assert all(shortest[name] > longest[name] for name in block_nodes)
    assert shortest["L1_mlp1"] < 1e-6


def _small_graph(kind: str, depth_mode: str = "longest", **inference):
    weight_init = NormalInitializer(std=0.5)
    x = IdentityNode(shape=(4,), name="x")
    h1 = Linear(
        shape=(5,), name="h1", activation=TanhActivation(), weight_init=weight_init
    )
    h2 = Linear(
        shape=(5,), name="h2", activation=TanhActivation(), weight_init=weight_init
    )
    y = Linear(shape=(3,), name="y", weight_init=weight_init)
    nodes = [x, h1, h2]
    edges = [
        Edge(source=x, target=h1.slot("in")),
        Edge(source=h1, target=h2.slot("in")),
    ]
    if kind == "chain":
        h3 = Linear(
            shape=(5,), name="h3", activation=TanhActivation(), weight_init=weight_init
        )
    else:
        h3 = LinearResidual(
            shape=(5,), name="h3", activation=TanhActivation(), weight_init=weight_init
        )
    edges.append(Edge(source=h2, target=h3.slot("in")))
    if kind == "residual":
        edges.append(Edge(source=h1, target=h3.slot("skip")))
    if kind == "auxiliary":
        aux = IdentityNode(shape=(5,), name="aux")
        nodes.append(aux)
        edges.extend(
            [
                Edge(source=h1, target=aux.slot("in")),
                Edge(source=aux, target=h3.slot("skip")),
            ]
        )
    nodes.extend([h3, y])
    edges.append(Edge(source=h3, target=y.slot("in")))

    def build(infer_steps: int):
        config = {
            "eta_infer": 0.0,
            "infer_steps": infer_steps,
            "depth_mode": depth_mode,
            **inference,
        }
        return graph(
            nodes=nodes,
            edges=edges,
            task_map=TaskMap(x=x, y=y),
            inference=InferenceSpikingPrecision(**config),
            graph_state_initializer=FeedforwardStateInit(),
        )

    steps = max(node_output_depths(build(1), depth_mode).values())
    return build(inference.pop("infer_steps", steps))


def _small_graph_pc_and_bp(structure, mode: str = "initial"):
    key = jax.random.PRNGKey(1)
    params = initialize_params(structure, key)
    clamps = {
        "x": jax.random.normal(key, (2, 4)),
        "y": jax.random.normal(jax.random.PRNGKey(2), (2, 3)),
    }
    initial = initialize_graph_state(structure, 2, key, clamps=clamps, params=params)
    final = InferenceSpikingPrecision.run_inference(params, initial, clamps, structure)
    pc_grads = compute_local_weight_gradients_forward_update(
        params, initial, final, structure, mode
    )

    def output_loss(p):
        state = initialize_graph_state(structure, 2, key, clamps=clamps, params=p)
        return jnp.sum(state.nodes["y"].energy)

    return pc_grads, jax.grad(output_loss)(params), params, initial, final


@pytest.mark.parametrize("kind", ["chain", "auxiliary"])
@pytest.mark.parametrize("depth_mode", ["longest", "shortest"])
def test_spiking_zil_matches_backprop_on_small_graphs(kind, depth_mode):
    with jax.enable_x64(True):
        pc_grads, bp_grads, *_ = _small_graph_pc_and_bp(_small_graph(kind, depth_mode))
        errors = _relative_errors(pc_grads, bp_grads)

    assert max(errors.values()) < 1e-12, errors


def test_spiking_zil_misses_backprop_across_a_skip_merge():
    with jax.enable_x64(True):
        pc_grads, bp_grads, *_ = _small_graph_pc_and_bp(_small_graph("residual"))
        errors = _relative_errors(pc_grads, bp_grads)

    assert max(errors[name] for name in ("y", "h3", "h2")) < 1e-12
    assert errors["h1"] > 0.1


def test_forward_update_final_uses_feedforward_error_with_final_inputs():
    with jax.enable_x64(True):
        structure = _small_graph("chain", eta_infer=0.3, infer_steps=6)
        pc_grads, _bp, params, initial, final = _small_graph_pc_and_bp(
            structure, mode="final"
        )
        node = params.nodes["h2"]
        x_final = final.nodes["h1"].z_latent
        error = final.nodes["h2"].z_latent - initial.nodes["h2"].z_mu
        pre = x_final @ node.weights["h1->h2:in"] + node.biases["b"]
        delta = error * (1.0 - jnp.tanh(pre) ** 2)
        normal = compute_local_weight_gradients(params, final, structure)

        assert jnp.allclose(
            pc_grads.nodes["h2"].weights["h1->h2:in"], -x_final.T @ delta, atol=1e-12
        )
        assert jnp.allclose(
            pc_grads.nodes["h2"].biases["b"], -delta.sum(axis=0), atol=1e-12
        )
        assert not jnp.allclose(
            normal.nodes["h2"].weights["h1->h2:in"],
            pc_grads.nodes["h2"].weights["h1->h2:in"],
        )


def test_forward_update_final_keeps_cross_entropy_head_gradient(rng_key):
    structure = _spiking_mlm()
    params = initialize_params(structure, rng_key)
    batch = _two_sample_batch()
    _g, _m, initial, _free, final = get_graph_param_gradient_pcn_mlm(
        params, batch, structure, rng_key
    )

    forward = compute_local_weight_gradients_forward_update(
        params, initial, final, structure, "final"
    )
    normal = compute_local_weight_gradients(params, final, structure)

    assert tree_allclose(forward.nodes["mlm_head"], normal.nodes["mlm_head"])
    assert not tree_allclose(
        forward.nodes["final_layernorm"], normal.nodes["final_layernorm"]
    )


def test_forward_update_rejects_unknown_mode(rng_key):
    structure = _spiking_mlm()
    params = initialize_params(structure, rng_key)
    with pytest.raises(ValueError, match="pc_forward_updates"):
        get_graph_param_gradient_pcn_mlm(
            params, tiny_batch(), structure, rng_key, forward_updates="late"
        )
    with pytest.raises(ValueError, match="pc_forward_updates"):
        create_pc_bp_gradient_comparison(structure, "late")


def test_pc_bp_probe_uses_forward_update_rule(rng_key):
    structure = _spiking_mlm(infer_steps=8)
    params = initialize_params(structure, rng_key)
    batch = _two_sample_batch()

    result = create_pc_bp_gradient_comparison(structure, "final")(
        params, batch, rng_key
    )
    pc_grads, *_ = get_graph_param_gradient_pcn_mlm(
        params, batch, structure, rng_key, forward_updates="final"
    )

    for name, row in result.items():
        expected = optax.tree.norm(jax.tree_util.tree_leaves(pc_grads.nodes[name]))
        assert float(row["pc_norm"]) == pytest.approx(float(expected), rel=1e-5)


def test_spiking_gradient_step_compiles_and_counts_rises(rng_key):
    structure = _spiking_mlm(eta_infer=0.5, spike_eta=3.0, infer_steps=6)
    params = initialize_params(structure, rng_key)

    grads, loss_numerator, count, rises = create_jit_gradient_step_pcn_mlm(
        structure, "final"
    )(params, _two_sample_batch(), rng_key)

    assert rises.dtype == jnp.int32
    assert np.isfinite(float(loss_numerator)) and float(count) > 0
    assert all(
        np.all(np.isfinite(np.asarray(leaf)))
        for leaf in jax.tree_util.tree_leaves(grads)
    )


def test_spiking_runs_under_x64(rng_key):
    with jax.enable_x64(True):
        structure = _spiking_mlm()
        params = initialize_params(structure, rng_key)
        state, clamps = _guard_inputs(structure, params, tiny_batch(), rng_key)

        final_state, stats = jax.jit(
            lambda p, s, c: InferenceSpikingPrecision.run_inference_with_stats(
                p, s, c, structure
            )
        )(params, state, clamps)

    assert stats["energy_rises"].dtype == jnp.int32
    assert stats["final_eta"].dtype == jnp.float64
    assert final_state.nodes["L0_mha"].z_latent.dtype == jnp.float64
    assert all(
        np.all(np.isfinite(np.asarray(node.z_latent)))
        for node in final_state.nodes.values()
    )


def _manual_spiking_ipc(params, opt_state, batch, structure, optimizer, rng_key, scale):
    _shared, _metrics, _free, state, clamps = _prepare_pcn_mlm_batch(
        params, batch, structure, rng_key
    )
    config = dict(structure.config["inference"].config)
    spike_steps = InferenceSpikingPrecision.spike_steps(structure, config)
    weight_steps = ipc_weight_spike_steps(structure, clamps)
    shortest = node_output_depths(structure, "shortest")
    single_path = {
        name
        for name, depth in node_output_depths(structure).items()
        if shortest[name] == depth and jax.tree_util.tree_leaves(params.nodes[name])
    }
    prev_energy = jnp.full((batch["x"].shape[0],), jnp.inf)
    for t in range(config["infer_steps"]):
        grads = compute_local_weight_gradients(params, state, structure)
        for name in single_path:
            if spike_steps[name] == t:
                assert float(optax.tree.norm(grads.nodes[name])) == 0.0, name
        state, prev_energy, _rises = InferenceSpikingPrecision.spiking_step(
            params, state, jnp.array(t), prev_energy, clamps, structure
        )
        if scale:
            grads = _scale_spike_step_grads(
                grads,
                weight_steps,
                config["spike_eta"] / config["eta_infer"],
                jnp.array(t),
            )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
    return params


def test_incremental_spiking_scales_weight_step_after_spike(rng_key):
    structure = _spiking_mlm(eta_infer=0.2, spike_eta=1.0, infer_steps=6)
    params = initialize_params(structure, rng_key)
    optimizer = optax.sgd(1e-2)
    opt_state = optimizer.init(params)
    batch = _two_sample_batch()

    new_params, _opt_state, metrics = train_step_pcn_mlm_incremental(
        params, opt_state, batch, structure, optimizer, rng_key
    )
    expected = _manual_spiking_ipc(
        params, opt_state, batch, structure, optimizer, rng_key, scale=True
    )
    unscaled = _manual_spiking_ipc(
        params, opt_state, batch, structure, optimizer, rng_key, scale=False
    )

    assert metrics["pc_energy_rises"].dtype == jnp.int32
    assert all(
        jnp.allclose(a, b, atol=1e-6)
        for a, b in zip(
            jax.tree_util.tree_leaves(expected), jax.tree_util.tree_leaves(new_params)
        )
    )
    for name in ("final_layernorm", "L0_mlp2", "L0_mha"):
        assert not tree_allclose(unscaled.nodes[name], expected.nodes[name]), name


def test_spike_step_grad_scaling_targets_only_spiking_nodes():
    grads = initialize_params(_spiking_mlm(), jax.random.PRNGKey(0))
    spike_steps = InferenceSpikingPrecision.spike_steps(
        _spiking_mlm(), dict(_spiking_mlm().config["inference"].config)
    )

    scaled = _scale_spike_step_grads(grads, spike_steps, 5.0, jnp.array(1))

    for name in grads.nodes:
        factor = 5.0 if spike_steps.get(name) == 1 else 1.0
        assert tree_allclose(
            scaled.nodes[name],
            jax.tree_util.tree_map(lambda g: g * factor, grads.nodes[name]),
        ), name


def test_incremental_spiking_rejects_zero_eta(rng_key):
    structure = _spiking_mlm(eta_infer=0.0, infer_steps=2)
    params = initialize_params(structure, rng_key)
    optimizer = optax.sgd(1e-2)
    with pytest.raises(ValueError, match="eta_infer"):
        train_step_pcn_mlm_incremental(
            params, optimizer.init(params), tiny_batch(), structure, optimizer, rng_key
        )


def test_train_mlm_pc_incremental_rejects_forward_updates(rng_key):
    structure = _spiking_mlm(infer_steps=2)
    params = initialize_params(structure, rng_key)

    with pytest.raises(ValueError, match="pc_forward_updates"):
        train_mlm(
            params,
            structure,
            [tiny_batch()],
            optax.adam(1e-2),
            {"max_steps": 1, "pc_incremental": True, "pc_forward_updates": "final"},
            rng_key,
            mode="pc",
            verbose=False,
        )


@pytest.mark.parametrize("incremental", [False, True])
def test_train_mlm_pc_spiking_logs_energy_rises(rng_key, capsys, incremental):
    structure = _spiking_mlm(eta_infer=0.2, spike_eta=1.0, infer_steps=6)
    params = initialize_params(structure, rng_key)
    extra = {"pc_incremental": True} if incremental else {"pc_forward_updates": "final"}

    updated, state = train_mlm(
        params,
        structure,
        [tiny_batch()],
        optax.adam(1e-2),
        {"max_steps": 1, "log_interval": 1, **extra},
        rng_key,
        mode="pc",
        verbose=True,
    )

    out = capsys.readouterr().out
    assert state["step"] == 1
    assert "energy_rises=" in out
    assert tree_any_changed(params, updated)


def test_ipc_weight_spike_steps_skip_outputs_and_clamps(rng_key):
    structure = _spiking_mlm(eta_infer=0.2, infer_steps=6)
    params = initialize_params(structure, rng_key)
    _shared, _metrics, _free, _state, clamps = _prepare_pcn_mlm_batch(
        params, tiny_batch(), structure, rng_key
    )

    steps = ipc_weight_spike_steps(structure, clamps)

    assert "mlm_head" not in steps
    assert not set(clamps) & set(steps)
    assert steps == {
        "final_layernorm": 1,
        "L0_mlp2": 2,
        "L0_mlp1": 3,
        "L0_mha": 4,
        "position_embeddings": 5,
        "token_embeddings": 6,
    }


def test_incremental_spiking_first_iteration_is_unscaled(rng_key):
    structure = _spiking_mlm(eta_infer=0.01, spike_eta=1.0, infer_steps=1)
    params = initialize_params(structure, rng_key)
    optimizer = optax.sgd(1e-2)
    opt_state = optimizer.init(params)
    batch = _two_sample_batch()

    new_params, _opt_state, _metrics = train_step_pcn_mlm_incremental(
        params, opt_state, batch, structure, optimizer, rng_key
    )
    _shared, _free_metrics, _free, state, _clamps = _prepare_pcn_mlm_batch(
        params, batch, structure, rng_key
    )
    grads = compute_local_weight_gradients(params, state, structure)
    updates, _ = optimizer.update(grads, opt_state, params)
    expected = optax.apply_updates(params, updates)

    assert float(optax.tree.norm(grads.nodes["mlm_head"])) > 0.0
    assert tree_allclose(expected.nodes["mlm_head"], new_params.nodes["mlm_head"])
    assert tree_allclose(expected, new_params)


def test_incremental_spiking_rejects_shortest_depth(rng_key):
    structure = _spiking_mlm(infer_steps=2, depth_mode="shortest")
    params = initialize_params(structure, rng_key)
    optimizer = optax.sgd(1e-2)
    with pytest.raises(ValueError, match="depth_mode"):
        train_step_pcn_mlm_incremental(
            params, optimizer.init(params), tiny_batch(), structure, optimizer, rng_key
        )


def _with_global_init(structure):
    return structure._replace(
        config={**structure.config, "graph_state_initializer": GlobalStateInit()}
    )


def test_spiking_and_forward_updates_require_feedforward_init(rng_key):
    spiking = _with_global_init(_spiking_mlm(infer_steps=2))
    plain = _with_global_init(_small_mlm_with_inference(eta_infer=0.1, infer_steps=2))
    params = initialize_params(spiking, rng_key)
    optimizer = optax.sgd(1e-2)

    with pytest.raises(ValueError, match="FeedforwardStateInit"):
        get_graph_param_gradient_pcn_mlm(params, tiny_batch(), spiking, rng_key)
    with pytest.raises(ValueError, match="FeedforwardStateInit"):
        get_graph_param_gradient_pcn_mlm(
            params, tiny_batch(), plain, rng_key, forward_updates="initial"
        )
    with pytest.raises(ValueError, match="FeedforwardStateInit"):
        create_pc_bp_gradient_comparison(plain, "final")
    with pytest.raises(ValueError, match="FeedforwardStateInit"):
        create_jit_gradient_step_pcn_mlm(spiking)(params, tiny_batch(), rng_key)
    with pytest.raises(ValueError, match="FeedforwardStateInit"):
        train_step_pcn_mlm_incremental(
            params, optimizer.init(params), tiny_batch(), spiking, optimizer, rng_key
        )
    with pytest.raises(ValueError, match="FeedforwardStateInit"):
        train_mlm(
            params,
            plain,
            [tiny_batch()],
            optimizer,
            {"max_steps": 1, "pc_forward_updates": "final"},
            rng_key,
            mode="pc",
            verbose=False,
        )
    get_graph_param_gradient_pcn_mlm(params, tiny_batch(), plain, rng_key)


def test_train_step_pcn_mlm_threads_forward_updates(rng_key):
    structure = _spiking_mlm(infer_steps=6)
    params = initialize_params(structure, rng_key)
    optimizer = optax.sgd(1e-2)
    opt_state = optimizer.init(params)
    batch = _two_sample_batch()

    new_params, *_ = train_step_pcn_mlm(
        params, opt_state, batch, structure, optimizer, rng_key, "final"
    )
    grads, *_ = get_graph_param_gradient_pcn_mlm(
        params, batch, structure, rng_key, forward_updates="final"
    )
    updates, _ = optimizer.update(grads, opt_state, params)

    assert tree_allclose(optax.apply_updates(params, updates), new_params)


def test_spike_step_skips_latent_decay(rng_key):
    structure = _spiking_mlm(eta_infer=0.1, latent_decay=0.5, spike_eta=0.7)
    params = initialize_params(structure, rng_key)
    state, clamps = _guard_inputs(structure, params, _two_sample_batch(), rng_key)
    config = dict(structure.config["inference"].config)
    state = InferenceSpikingPrecision.zero_grads(params, state, clamps, structure)
    state = InferenceSpikingPrecision.forward_value_and_grad(
        params, state, clamps, structure
    )

    new_state = InferenceSpikingPrecision.update_latents_at_step(
        params, state, jnp.array(0), clamps, structure, config
    )

    before = state.nodes["final_layernorm"]
    assert float(jnp.max(jnp.abs(before.latent_grad))) > 0.0
    assert jnp.allclose(
        new_state.nodes["final_layernorm"].z_latent,
        before.z_latent - 0.7 * before.latent_grad,
        atol=1e-7,
    )
    other = state.nodes["L0_mha"]
    assert jnp.allclose(
        new_state.nodes["L0_mha"].z_latent,
        other.z_latent * (1.0 - 0.1 * 0.5) - 0.1 * other.latent_grad,
        atol=1e-7,
    )


def test_inference_history_rejects_spiking(rng_key):
    from fabricpc.utils.dashboarding.inference_tracking import (
        run_inference_with_full_history,
        run_inference_with_history,
    )

    structure = _spiking_mlm(infer_steps=2)
    params = initialize_params(structure, rng_key)
    state, clamps = _guard_inputs(structure, params, tiny_batch(), rng_key)

    with pytest.raises(NotImplementedError, match="InferenceSpikingPrecision"):
        run_inference_with_history(params, state, clamps, structure)
    with pytest.raises(NotImplementedError, match="InferenceSpikingPrecision"):
        run_inference_with_full_history(params, state, clamps, structure)
    with pytest.raises(NotImplementedError, match="step index"):
        InferenceSpikingPrecision.inference_step(
            params, state, clamps, structure, structure.config["inference"].config
        )
