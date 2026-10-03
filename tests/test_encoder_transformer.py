"""Tests for the bidirectional encoder transformer and its masked-LM head."""

import jax
import jax.numpy as jnp
import pytest

from fabricpc.core.inference import InferenceSGDNormClip
from fabricpc.graph_initialization import initialize_params
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.nodes import (
    AddPositionEmbeddingNode,
    FinalLayerNormNode,
    MaskedVocabProjectionNode,
    VocabProjectionNode,
)
from fabricpc.nodes.encoder_transformer import create_encoder_mlm_model
from fabricpc.training.train_backprop import compute_forward_pass

SEQ_LEN = 6
VOCAB_SIZE = 20


def build_mlm(**overrides):
    kwargs = dict(
        depth=1,
        embed_dim=16,
        num_heads=4,
        mlp_dim=32,
        seq_len=SEQ_LEN,
        vocab_size=VOCAB_SIZE,
        inference=InferenceSGDNormClip(eta_infer=0.1, infer_steps=3, max_norm=5.0),
    )
    kwargs.update(overrides)
    return create_encoder_mlm_model(**kwargs)


def mlm_batch(loss_mask=None):
    return {
        "x": jnp.array([[1, 2, 3, 4, 0, 0]], dtype=jnp.int32),
        "y": jnp.array([[1, 2, 3, 4, 0, 0]], dtype=jnp.int32),
        "attention_mask": jnp.array([[1, 1, 1, 1, 0, 0]], dtype=jnp.float32),
        "loss_mask": (
            jnp.array([[1, 0, 0, 0, 0, 0]], dtype=jnp.float32)
            if loss_mask is None
            else loss_mask
        ),
    }


def head_clamps(structure, batch, targets):
    task_map = structure.task_map
    return {
        task_map["x"]: batch["x"],
        task_map["attention_mask"]: batch["attention_mask"],
        task_map["loss_mask"]: batch["loss_mask"],
        task_map["y"]: targets,
    }


class TestEncoderTopology:
    def test_attention_is_bidirectional_without_rope(self):
        structure = build_mlm()
        mha_config = structure.nodes["L0_mha"].node_info.node_config

        assert mha_config["is_causal"] is False
        assert mha_config["use_rope"] is False
        assert (
            structure.nodes["position_embeddings"].node_info.node_class
            is AddPositionEmbeddingNode
        )
        assert (
            structure.nodes["final_layernorm"].node_info.node_class
            is FinalLayerNormNode
        )

    def test_task_map_exposes_mlm_inputs(self):
        structure = build_mlm()

        assert structure.task_map["x"] == "input_ids"
        assert structure.task_map["y"] == "mlm_head"
        assert structure.task_map["loss_mask"] == "mlm_loss_mask"
        assert structure.task_map["attention_mask"] == "attention_mask"
        assert "attention_mask->L0_mha:attention_mask" in structure.edges
        assert "mlm_loss_mask->mlm_head:loss_mask" in structure.edges

    def test_attention_mask_is_optional(self):
        structure = build_mlm(use_attention_mask=False)

        assert "attention_mask" not in structure.task_map
        assert "attention_mask" not in structure.nodes

    def test_dropout_tasks_follow_transformer_v2_convention(self):
        structure = build_mlm(dropout_rate=0.1)

        assert structure.config["dropout_rate"] == 0.1
        for task in ("dropout_L0_mha", "dropout_L0_mlp2"):
            assert structure.task_map[task] == task
        assert "dropout_L0_mha->L0_mha:dropout_mask" in structure.edges
        assert "dropout_L0_mlp2->L0_mlp2:dropout_mask" in structure.edges

    def test_pass_through_nodes_are_not_mupc_scaled(self):
        structure = build_mlm()

        for node_name in ("position_embeddings", "final_layernorm"):
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
            {"dropout_rate": 1.0},
        ],
    )
    def test_builder_rejects_invalid_config(self, overrides):
        with pytest.raises(ValueError):
            build_mlm(**overrides)


class TestEncoderForward:
    def test_mlm_head_outputs_token_distributions(self, rng_key):
        structure = build_mlm()
        params = initialize_params(structure, rng_key)

        state = compute_forward_pass(params, structure, mlm_batch(), rng_key)

        probs = state.nodes["mlm_head"].z_mu
        assert probs.shape == (1, SEQ_LEN, VOCAB_SIZE)
        assert jnp.allclose(jnp.sum(probs, axis=-1), 1.0, atol=1e-5)

    def test_future_token_affects_first_position(self, rng_key):
        structure = build_mlm(use_attention_mask=False)
        params = initialize_params(structure, rng_key)
        batch = mlm_batch()
        changed = {**batch, "x": batch["x"].at[:, 3].set(9)}

        base = compute_forward_pass(params, structure, batch, rng_key)
        moved = compute_forward_pass(params, structure, changed, rng_key)

        delta = jnp.abs(
            base.nodes["L0_mha"].z_mu[:, 0] - moved.nodes["L0_mha"].z_mu[:, 0]
        )
        assert jnp.max(delta) > 1e-6

    def test_unclamped_attention_mask_attends_to_all_tokens(self, rng_key):
        structure = build_mlm()
        params = initialize_params(structure, rng_key)
        batch = mlm_batch()
        without_mask = {k: v for k, v in batch.items() if k != "attention_mask"}
        all_ones = {**batch, "attention_mask": jnp.ones((1, SEQ_LEN))}

        unclamped = compute_forward_pass(params, structure, without_mask, rng_key)
        ones = compute_forward_pass(params, structure, all_ones, rng_key)

        assert jnp.allclose(
            unclamped.nodes["mlm_head"].z_mu, ones.nodes["mlm_head"].z_mu, atol=1e-6
        )

    def test_dropout_applies_only_in_training(self, rng_key):
        structure = build_mlm(dropout_rate=0.5)
        params = initialize_params(structure, rng_key)
        batch = mlm_batch()

        def head(key, training):
            state = compute_forward_pass(params, structure, batch, key, training)
            return state.nodes["mlm_head"].z_mu

        other_key = jax.random.PRNGKey(5)
        assert jnp.allclose(head(rng_key, False), head(other_key, False))
        assert not jnp.allclose(head(rng_key, True), head(other_key, True))


class TestMaskedVocabProjection:
    def test_reuses_vocab_projection_params(self):
        structure = build_mlm()

        assert issubclass(MaskedVocabProjectionNode, VocabProjectionNode)
        assert (
            structure.nodes["mlm_head"].node_info.node_class
            is MaskedVocabProjectionNode
        )

    def test_energy_ignores_unmasked_labels(self, rng_key):
        structure = build_mlm()
        params = initialize_params(structure, rng_key)
        batch = mlm_batch()
        targets = jax.nn.one_hot(batch["y"], VOCAB_SIZE)
        changed_targets = targets.at[:, 1:].set(
            jax.nn.one_hot(jnp.array([[5, 6, 7, 8, 9]]), VOCAB_SIZE)
        )

        original = initialize_graph_state(
            structure, 1, rng_key, head_clamps(structure, batch, targets), params=params
        )
        changed = initialize_graph_state(
            structure,
            1,
            rng_key,
            head_clamps(structure, batch, changed_targets),
            params=params,
        )

        assert jnp.allclose(
            original.nodes["mlm_head"].energy,
            changed.nodes["mlm_head"].energy,
            atol=1e-6,
        )

    def test_energy_and_error_are_zero_for_empty_mask(self, rng_key):
        structure = build_mlm()
        params = initialize_params(structure, rng_key)
        batch = mlm_batch(loss_mask=jnp.zeros((1, SEQ_LEN)))
        targets = jax.nn.one_hot(batch["y"], VOCAB_SIZE)

        state = initialize_graph_state(
            structure, 1, rng_key, head_clamps(structure, batch, targets), params=params
        )

        assert jnp.allclose(state.nodes["mlm_head"].energy, 0.0)
        assert jnp.allclose(state.nodes["mlm_head"].error, 0.0)

    def test_output_energy_scale_multiplies_energy(self, rng_key):
        batch = mlm_batch()
        targets = jax.nn.one_hot(batch["y"], VOCAB_SIZE)
        energies = []
        for scale in (1.0, 3.0):
            structure = build_mlm(pc_output_energy_scale=scale)
            params = initialize_params(structure, rng_key)
            state = initialize_graph_state(
                structure,
                1,
                rng_key,
                head_clamps(structure, batch, targets),
                params=params,
            )
            energies.append(state.nodes["mlm_head"].energy)

        assert jnp.allclose(energies[1], 3.0 * energies[0], rtol=1e-5)
