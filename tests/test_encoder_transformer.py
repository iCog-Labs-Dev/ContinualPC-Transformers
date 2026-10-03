"""Tests for the bidirectional encoder transformer and its masked-LM head."""

import jax
import jax.numpy as jnp
import pytest

from fabricpc.core.activations import SoftmaxActivation
from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.inference import InferenceSGDNormClip
from fabricpc.graph_initialization import initialize_params
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.nodes import (
    AddPositionEmbeddingNode,
    FinalLayerNormNode,
    MaskedVocabProjectionNode,
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
        batch = {k: v for k, v in mlm_batch().items() if k != "attention_mask"}
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


def reference_head_energy(params, state, batch, targets, scale=1.0):
    """Masked CE summed per sample, from the head logits."""
    head = params.nodes["mlm_head"]
    hidden = state.nodes["final_layernorm"].z_mu
    logits = jnp.dot(hidden, head.weights["W_out"]) + head.biases["b_out"]
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    mask = batch["loss_mask"]
    per_token = -jnp.sum(targets * log_probs, axis=-1)
    return scale * jnp.sum(per_token * mask, axis=-1)


class TestMaskedVocabProjection:
    def test_has_vocab_projection_params(self, rng_key):
        structure = build_mlm()
        params = initialize_params(structure, rng_key)
        head = params.nodes["mlm_head"]

        assert (
            structure.nodes["mlm_head"].node_info.node_class
            is MaskedVocabProjectionNode
        )
        assert set(head.weights) == {"W_out"} and set(head.biases) == {"b_out"}
        assert head.weights["W_out"].shape == (16, VOCAB_SIZE)
        assert head.biases["b_out"].shape == (VOCAB_SIZE,)

    @pytest.mark.parametrize("fixed", ["activation", "energy"])
    def test_rejects_activation_and_energy_overrides(self, fixed):
        override = {"activation": SoftmaxActivation(), "energy": GaussianEnergy()}
        with pytest.raises(ValueError, match=fixed):
            MaskedVocabProjectionNode(
                shape=(SEQ_LEN, VOCAB_SIZE),
                name="head",
                vocab_size=VOCAB_SIZE,
                embed_dim=16,
                **{fixed: override[fixed]},
            )

    @pytest.mark.parametrize(
        "loss_mask",
        [
            jnp.array([[1, 0, 0, 0, 0, 0]], dtype=jnp.float32),
            jnp.array([[1, 1, 0, 1, 0, 0]], dtype=jnp.float32),
        ],
    )
    def test_energy_matches_masked_cross_entropy(self, rng_key, loss_mask):
        structure = build_mlm()
        params = initialize_params(structure, rng_key)
        batch = mlm_batch(loss_mask=loss_mask)
        targets = jax.nn.one_hot(batch["y"], VOCAB_SIZE)

        state = initialize_graph_state(
            structure, 1, rng_key, head_clamps(structure, batch, targets), params=params
        )

        expected = reference_head_energy(params, state, batch, targets)
        assert jnp.allclose(state.nodes["mlm_head"].energy, expected, rtol=1e-6)

    def test_energy_functional_without_mask_selects_every_position(self, rng_key):
        structure = build_mlm()
        params = initialize_params(structure, rng_key)
        batch = mlm_batch(loss_mask=jnp.ones((1, SEQ_LEN), jnp.float32))
        targets = jax.nn.one_hot(batch["y"], VOCAB_SIZE)
        state = initialize_graph_state(
            structure, 1, rng_key, head_clamps(structure, batch, targets), params=params
        )
        node_info = structure.nodes["mlm_head"].node_info
        head_params = params.nodes["mlm_head"]
        logits = (
            jnp.dot(state.nodes["final_layernorm"].z_mu, head_params.weights["W_out"])
            + head_params.biases["b_out"]
        )
        head = state.nodes["mlm_head"]._replace(pre_activation=logits)

        out = node_info.node_class.energy_functional(head, node_info)

        expected = reference_head_energy(params, state, batch, targets)
        assert jnp.allclose(out.energy, expected, rtol=1e-6)

    def test_error_is_raw_at_every_position(self, rng_key):
        structure = build_mlm()
        params = initialize_params(structure, rng_key)
        batch = mlm_batch()
        targets = jax.nn.one_hot(batch["y"], VOCAB_SIZE)

        head = initialize_graph_state(
            structure, 1, rng_key, head_clamps(structure, batch, targets), params=params
        ).nodes["mlm_head"]

        assert jnp.allclose(head.error, head.z_latent - head.z_mu)
        assert jnp.max(jnp.abs(head.error[:, 1:])) > 1e-3

    def test_forward_records_logits_and_matches_reference_gradients(self, rng_key):
        structure = build_mlm()
        params = initialize_params(structure, rng_key)
        batch = mlm_batch(loss_mask=jnp.array([[1, 0, 1, 0, 0, 0]], jnp.float32))
        targets = jax.nn.one_hot(batch["y"], VOCAB_SIZE)
        state = initialize_graph_state(
            structure, 1, rng_key, head_clamps(structure, batch, targets), params=params
        )
        node = structure.nodes["mlm_head"]
        node_info = node.node_info
        inputs = {
            "final_layernorm->mlm_head:in": state.nodes["final_layernorm"].z_mu,
            "mlm_loss_mask->mlm_head:loss_mask": batch["loss_mask"],
        }
        head_state = state.nodes["mlm_head"]

        def node_energy(head_params):
            energy, _ = node_info.node_class.forward(
                head_params, inputs, head_state, node_info
            )
            return energy

        def reference(head_params):
            full = params._replace(nodes={**params.nodes, "mlm_head": head_params})
            return jnp.sum(reference_head_energy(full, state, batch, targets))

        _, out = node_info.node_class.forward(
            params.nodes["mlm_head"], inputs, head_state, node_info
        )
        assert jnp.allclose(out.z_mu, jax.nn.softmax(out.pre_activation, axis=-1))

        grads = jax.grad(node_energy)(params.nodes["mlm_head"])
        expected = jax.grad(reference)(params.nodes["mlm_head"])
        for got, want in zip(
            jax.tree_util.tree_leaves(grads), jax.tree_util.tree_leaves(expected)
        ):
            assert jnp.allclose(got, want, atol=1e-6)

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

    def test_energy_is_cross_entropy_summed_over_masked_tokens(self, rng_key):
        structure = build_mlm()
        params = initialize_params(structure, rng_key)
        batch = mlm_batch(loss_mask=jnp.array([[1, 0, 1, 1, 0, 0]], dtype=jnp.float32))
        targets = jax.nn.one_hot(batch["y"], VOCAB_SIZE)

        state = initialize_graph_state(
            structure, 1, rng_key, head_clamps(structure, batch, targets), params=params
        )

        head = state.nodes["mlm_head"]
        per_token = -jnp.sum(targets * jnp.log(head.z_mu), axis=-1)
        expected = jnp.sum(per_token * batch["loss_mask"], axis=-1)
        assert jnp.allclose(head.energy, expected, rtol=1e-5)

    def test_energy_of_a_sample_ignores_other_samples_masks(self, rng_key):
        structure = build_mlm()
        params = initialize_params(structure, rng_key)
        single = mlm_batch()
        pair = {
            key: jnp.concatenate([value, value], axis=0)
            for key, value in single.items()
        }
        pair["loss_mask"] = pair["loss_mask"].at[1].set(1.0)

        def head_energy(batch, batch_size):
            targets = jax.nn.one_hot(batch["y"], VOCAB_SIZE)
            state = initialize_graph_state(
                structure,
                batch_size,
                rng_key,
                head_clamps(structure, batch, targets),
                params=params,
            )
            return state.nodes["mlm_head"].energy

        assert jnp.allclose(
            head_energy(pair, 2)[0], head_energy(single, 1)[0], rtol=1e-5
        )

    def test_energy_is_zero_and_error_raw_for_empty_mask(self, rng_key):
        structure = build_mlm()
        params = initialize_params(structure, rng_key)
        batch = mlm_batch(loss_mask=jnp.zeros((1, SEQ_LEN)))
        targets = jax.nn.one_hot(batch["y"], VOCAB_SIZE)

        state = initialize_graph_state(
            structure, 1, rng_key, head_clamps(structure, batch, targets), params=params
        )

        head = state.nodes["mlm_head"]
        assert jnp.allclose(head.energy, 0.0)
        assert jnp.allclose(head.error, head.z_latent - head.z_mu)

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
