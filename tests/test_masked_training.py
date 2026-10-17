"""Tests for the generic masking primitives used by masked-LM training.

Covers the MhaResidualNode padding mask, ``compute_loss(loss_mask=...)``,
auxiliary-task clamping in the backprop forward/loss paths, dropout in the
generic backprop step, integer-target evaluation, and the pmap guard for
masked batches.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from fabricpc.core.activations import (
    GeluActivation,
    IdentityActivation,
    SoftmaxActivation,
)
from fabricpc.core.inference import InferenceSGDNormClip
from fabricpc.core.topology import Edge
from fabricpc.core.types import GraphState, NodeParams, NodeState
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.graph_initialization.state_initializer import FeedforwardStateInit
from fabricpc.nodes import IdentityNode, Linear
from fabricpc.nodes.encoder_transformer import create_encoder_mlm_model
from fabricpc.nodes.transformer import TransformerBlock
from fabricpc.nodes.transformer_v2 import MhaResidualNode, create_deep_transformer
from fabricpc.training import evaluate_mlm
from fabricpc.training.train_autoregressive import compute_loss, create_dropout_clamps
from fabricpc.training.train_backprop import (
    _target_labels,
    compute_forward_pass,
    compute_loss_autoregressive,
    eval_step_backprop,
    evaluate_backprop_autoregressive,
    train_backprop_autoregressive,
    train_step_backprop,
)
from fabricpc.utils.data import MlmTextBatcher
from fabricpc.utils.helpers import layernorm

SEQ_LEN = 6
EMBED_DIM = 8


@pytest.fixture
def masked_attention_graph(rng_key):
    """input -> bidirectional MHA (with padding mask) -> output."""
    input_node = Linear(shape=(SEQ_LEN, EMBED_DIM), name="input")
    mask_node = IdentityNode(shape=(SEQ_LEN,), name="attention_mask")
    mha = MhaResidualNode(
        shape=(SEQ_LEN, EMBED_DIM),
        name="mha",
        embed_dim=EMBED_DIM,
        num_heads=2,
        use_rope=False,
        is_causal=False,
    )
    output = Linear(shape=(SEQ_LEN, EMBED_DIM), name="output")
    structure = graph(
        nodes=[input_node, mask_node, mha, output],
        edges=[
            Edge(source=input_node, target=mha.slot("in")),
            Edge(source=mask_node, target=mha.slot("attention_mask")),
            Edge(source=mha, target=output.slot("in")),
        ],
        task_map=TaskMap(x=input_node, attention_mask=mask_node, y=output),
        inference=InferenceSGDNormClip(eta_infer=0.1, infer_steps=2, max_norm=5.0),
        graph_state_initializer=FeedforwardStateInit(),
    )
    return initialize_params(structure, rng_key), structure


def _forward(params, structure, x, mask, rng_key):
    batch = {"x": x, "attention_mask": mask, "y": jnp.zeros_like(x)}
    return compute_forward_pass(params, structure, batch, rng_key)


def _padded_mask(valid_len):
    return (jnp.arange(SEQ_LEN) < valid_len).astype(jnp.float32)[None, :]


class TestAttentionMask:
    def test_padded_keys_do_not_affect_valid_positions(
        self, masked_attention_graph, rng_key
    ):
        params, structure = masked_attention_graph
        x = jax.random.normal(rng_key, (1, SEQ_LEN, EMBED_DIM))
        x_changed_pad = x.at[:, 4:, :].add(10.0)
        mask = _padded_mask(4)

        base = _forward(params, structure, x, mask, rng_key).nodes["mha"].z_mu
        changed = (
            _forward(params, structure, x_changed_pad, mask, rng_key).nodes["mha"].z_mu
        )

        assert jnp.max(jnp.abs(base[:, :4] - changed[:, :4])) < 1e-5

    def test_padded_queries_get_no_attention_output(
        self, masked_attention_graph, rng_key
    ):
        params, structure = masked_attention_graph
        params = _with_output_bias(params, 1.0)
        x = jax.random.normal(rng_key, (1, SEQ_LEN, EMBED_DIM))
        z_mu = (
            _forward(params, structure, x, _padded_mask(4), rng_key).nodes["mha"].z_mu
        )

        assert jnp.array_equal(z_mu[:, 4:], x[:, 4:])

    def test_valid_queries_match_key_masked_attention(
        self, masked_attention_graph, rng_key
    ):
        params, structure = masked_attention_graph
        params = _with_output_bias(params, 1.0)
        x = jax.random.normal(rng_key, (1, SEQ_LEN, EMBED_DIM))
        mask = _padded_mask(4)
        z_mu = _forward(params, structure, x, mask, rng_key).nodes["mha"].z_mu

        expected = _key_masked_attention(params.nodes["mha"], x, mask, num_heads=2)
        assert jnp.allclose(z_mu[:, :4], expected[:, :4], atol=1e-5)

    def test_forward_pass_clamps_attention_mask(self, masked_attention_graph, rng_key):
        """Without clamping, the mask node gets a random latent and is ignored."""
        params, structure = masked_attention_graph
        x = jax.random.normal(rng_key, (1, SEQ_LEN, EMBED_DIM))
        full = _forward(params, structure, x, _padded_mask(SEQ_LEN), rng_key)
        padded = _forward(params, structure, x, _padded_mask(4), rng_key)

        assert jnp.allclose(padded.nodes["attention_mask"].z_latent, _padded_mask(4))
        delta = jnp.abs(full.nodes["mha"].z_mu - padded.nodes["mha"].z_mu)
        assert jnp.max(delta[:, :4]) > 1e-4


def _with_output_bias(params, value):
    mha = params.nodes["mha"]
    biases = {**mha.biases, "b_o": jnp.full_like(mha.biases["b_o"], value)}
    return params._replace(
        nodes={**params.nodes, "mha": NodeParams(mha.weights, biases)}
    )


def _key_masked_attention(mha_params, x, mask, num_heads):
    """Bidirectional MHA residual with padded keys excluded, no RoPE."""
    w, b = mha_params.weights, mha_params.biases
    batch, length, dim = x.shape
    head_dim = dim // num_heads
    h = layernorm(x, w["ln_gamma"], b["ln_beta"])

    def heads(name):
        out = jnp.dot(h, w[f"W_{name}"]) + b[f"b_{name}"]
        return out.reshape(batch, length, num_heads, head_dim).transpose(0, 2, 1, 3)

    scores = heads("q") @ heads("k").swapaxes(-1, -2) / jnp.sqrt(head_dim)
    scores = jnp.where(mask[:, None, None, :] > 0, scores, -1e9)
    attn = jax.nn.softmax(scores, axis=-1) @ heads("v")
    attn = attn.transpose(0, 2, 1, 3).reshape(batch, length, dim)
    return x + jnp.dot(attn, w["W_o"]) + b["b_o"]


def _state_with_predictions(predictions):
    node = NodeState(
        z_latent=predictions,
        z_mu=predictions,
        error=jnp.zeros_like(predictions),
        energy=jnp.zeros(predictions.shape[0]),
        pre_activation=predictions,
        latent_grad=jnp.zeros_like(predictions),
    )
    return GraphState(nodes={"out": node}, batch_size=predictions.shape[0])


class TestComputeLossMask:
    def test_cross_entropy_ignores_unmasked_positions(self):
        probs = jax.nn.softmax(jax.random.normal(jax.random.PRNGKey(0), (2, 3, 5)))
        targets = jnp.array([[1, 2, 3], [0, 4, 1]])
        mask = jnp.array([[1, 0, 0], [0, 1, 0]], dtype=jnp.float32)
        changed = probs.at[0, 1].set(jnp.full((5,), 0.2))

        loss = compute_loss(
            _state_with_predictions(probs), targets, "out", loss_mask=mask
        )
        loss_changed = compute_loss(
            _state_with_predictions(changed), targets, "out", loss_mask=mask
        )

        expected = (
            -(jnp.log(probs[0, 0, 1] + 1e-10) + jnp.log(probs[1, 1, 4] + 1e-10)) / 2
        )
        assert loss == pytest.approx(float(expected), rel=1e-5)
        assert loss_changed == pytest.approx(float(loss), rel=1e-6)

    def test_cross_entropy_rejects_broadcastable_mask(self):
        probs = jnp.full((2, 3, 5), 0.2)
        targets = jnp.zeros((2, 3), dtype=jnp.int32)
        with pytest.raises(ValueError, match="loss_mask shape"):
            compute_loss(
                _state_with_predictions(probs), targets, "out", loss_mask=jnp.ones((3,))
            )

    def test_mse_mask_broadcasts_over_features(self):
        preds = jnp.zeros((2, 3, 4))
        targets = jnp.zeros((2, 3, 4)).at[0, 0].set(2.0).at[1, 2].set(100.0)
        mask = jnp.array([[1, 0, 0], [0, 0, 0]])

        loss = compute_loss(
            _state_with_predictions(preds), targets, "out", "mse", loss_mask=mask
        )

        assert loss == pytest.approx(4.0)

    def test_no_mask_matches_previous_mean(self):
        probs = jax.nn.softmax(jax.random.normal(jax.random.PRNGKey(1), (2, 3, 5)))
        targets = jnp.array([[1, 2, 3], [0, 4, 1]])
        one_hot = jax.nn.one_hot(targets, 5)
        expected = -jnp.mean(jnp.sum(one_hot * jnp.log(probs + 1e-10), axis=-1))

        loss = compute_loss(_state_with_predictions(probs), targets, "out")

        assert loss == pytest.approx(float(expected), rel=1e-6)


@pytest.fixture
def tiny_lm(rng_key):
    structure = create_deep_transformer(
        depth=1,
        embed_dim=EMBED_DIM,
        num_heads=2,
        mlp_dim=16,
        seq_len=SEQ_LEN,
        vocab_size=11,
        inference=InferenceSGDNormClip(eta_infer=0.1, infer_steps=2, max_norm=5.0),
    )
    return initialize_params(structure, rng_key), structure


def _token_batch(rng_key, batch_size=4, vocab_size=11):
    x_key, y_key = jax.random.split(rng_key)
    return {
        "x": jax.random.randint(x_key, (batch_size, SEQ_LEN), 0, vocab_size),
        "y": jax.random.randint(y_key, (batch_size, SEQ_LEN), 0, vocab_size),
    }


class TestComputeLossAutoregressiveMask:
    def test_loss_mask_restricts_cross_entropy(self, tiny_lm, rng_key):
        params, structure = tiny_lm
        batch = _token_batch(rng_key)
        mask = jnp.zeros((4, SEQ_LEN)).at[:, 2].set(1.0)

        unmasked, predictions = compute_loss_autoregressive(
            params, structure, batch, rng_key, training=False
        )
        masked, _ = compute_loss_autoregressive(
            params, structure, {**batch, "loss_mask": mask}, rng_key, training=False
        )

        one_hot = jax.nn.one_hot(batch["y"], predictions.shape[-1])
        per_token = -jnp.sum(one_hot * jnp.log(predictions + 1e-10), axis=-1)
        assert masked == pytest.approx(float(jnp.mean(per_token[:, 2])), rel=1e-5)
        assert unmasked == pytest.approx(float(jnp.mean(per_token)), rel=1e-5)


class TestStandardBackpropMask:
    """train_step_backprop / eval_step_backprop honour batch['loss_mask']."""

    def test_eval_step_scores_only_masked_positions(self, tiny_lm, rng_key):
        params, structure = tiny_lm
        batch = _token_batch(rng_key)
        mask = jnp.zeros((4, SEQ_LEN)).at[:, :2].set(1.0)

        expected_loss, _ = compute_loss_autoregressive(
            params, structure, {**batch, "loss_mask": mask}, rng_key, training=False
        )
        one_hot_batch = {
            **batch,
            "y": jax.nn.one_hot(batch["y"], 11),
            "loss_mask": mask,
        }
        loss, correct, total = eval_step_backprop(
            params, one_hot_batch, structure, rng_key
        )

        assert total == 8
        assert 0 <= correct <= 8
        assert loss == pytest.approx(float(expected_loss), rel=1e-5)

    def test_train_step_ignores_unmasked_targets(self, tiny_lm, rng_key):
        params, structure = tiny_lm
        batch = _token_batch(rng_key)
        mask = jnp.zeros((4, SEQ_LEN)).at[:, 0].set(1.0)
        changed_targets = batch["y"].at[:, 1:].set((batch["y"][:, 1:] + 1) % 11)
        optimizer = optax.sgd(1e-2)

        def step(targets):
            masked = {**batch, "y": targets, "loss_mask": mask}
            new_params, _, _ = train_step_backprop(
                params, optimizer.init(params), masked, structure, optimizer, rng_key
            )
            return new_params

        a, b = step(batch["y"]), step(changed_targets)
        assert all(
            jnp.allclose(x, y)
            for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b))
        )


class _ListLoader:
    def __init__(self, batches):
        self.batches = batches

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        return iter(self.batches)


class TestAutoregressiveEvaluatorRejectsMaskedBatches:
    def test_points_to_evaluate_mlm(self, tiny_lm, rng_key):
        params, structure = tiny_lm
        batch = {**_token_batch(rng_key), "loss_mask": jnp.ones((4, SEQ_LEN))}

        for pmap_single_device in (False, True):
            with pytest.raises(ValueError, match="evaluate_mlm"):
                evaluate_backprop_autoregressive(
                    params,
                    structure,
                    _ListLoader([batch]),
                    {},
                    rng_key,
                    pmap_single_device=pmap_single_device,
                )


class TestMaskedPmapGuard:
    def test_masked_batches_rejected_on_pmap_path(self, tiny_lm, rng_key):
        params, structure = tiny_lm
        batch = {**_token_batch(rng_key), "loss_mask": jnp.ones((4, SEQ_LEN))}

        with pytest.raises(NotImplementedError, match="loss_mask"):
            train_backprop_autoregressive(
                params,
                structure,
                _ListLoader([batch]),
                optax.sgd(1e-3),
                {"num_epochs": 1},
                rng_key,
                verbose=False,
                pmap_single_device=True,
            )


MLM_VOCAB = 16
MLM_SPECIALS = {0, 1, 2, 3}


def _encoder(rng_key, dropout_rate=0.0):
    structure = create_encoder_mlm_model(
        depth=1,
        embed_dim=EMBED_DIM,
        num_heads=2,
        mlp_dim=16,
        seq_len=SEQ_LEN,
        vocab_size=MLM_VOCAB,
        dropout_rate=dropout_rate,
        inference=InferenceSGDNormClip(eta_infer=0.1, infer_steps=2, max_norm=5.0),
    )
    return initialize_params(structure, rng_key), structure


def _mlm_batch(batch_size, seed=0):
    loader = MlmTextBatcher(
        token_ids=np.tile(np.arange(4, MLM_VOCAB, dtype=np.int32), 6),
        seq_len=SEQ_LEN,
        batch_size=batch_size,
        pad_token_id=0,
        mask_token_id=3,
        vocab_size=MLM_VOCAB,
        special_token_ids=MLM_SPECIALS,
        seed=seed,
        mask_prob=0.4,
    )
    return next(iter(loader))


def _trees_equal(a, b):
    return all(
        bool(jnp.array_equal(x, y))
        for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b))
    )


class TestBackpropDropout:
    def test_train_step_samples_dropout(self, rng_key):
        params, structure = _encoder(rng_key, dropout_rate=0.5)
        batch = {k: jnp.asarray(v) for k, v in _mlm_batch(4).items()}
        optimizer = optax.sgd(1e-1)

        def step(key):
            new_params, _, _ = train_step_backprop(
                params, optimizer.init(params), batch, structure, optimizer, key
            )
            return new_params

        assert not _trees_equal(step(rng_key), step(jax.random.PRNGKey(7)))

    def test_eval_step_is_deterministic(self, rng_key):
        params, structure = _encoder(rng_key, dropout_rate=0.5)
        batch = {k: jnp.asarray(v) for k, v in _mlm_batch(4).items()}

        first = eval_step_backprop(params, batch, structure, rng_key)
        second = eval_step_backprop(params, batch, structure, jax.random.PRNGKey(7))

        assert all(float(a) == float(b) for a, b in zip(first, second))


class TestEvalStepIntegerTargets:
    @pytest.mark.parametrize("batch_size", [4, 1])
    def test_masked_accuracy_matches_evaluate_mlm(self, rng_key, batch_size):
        params, structure = _encoder(rng_key)
        batch = {k: jnp.asarray(v) for k, v in _mlm_batch(batch_size).items()}
        state = compute_forward_pass(params, structure, batch, rng_key)
        predicted = jnp.argmax(state.nodes["mlm_head"].z_mu, axis=-1)
        batch["y"] = predicted.at[:, -1].set((predicted[:, -1] + 1) % MLM_VOCAB)

        loss, correct, total = eval_step_backprop(params, batch, structure, rng_key)
        expected = evaluate_mlm(params, structure, _ListLoader([batch]), rng_key)

        assert 0.0 < expected["masked_accuracy"]
        assert int(total) == expected["num_masked_tokens"]
        assert float(correct) / float(total) == pytest.approx(
            expected["masked_accuracy"]
        )
        assert float(loss) == pytest.approx(expected["masked_loss"], rel=1e-5)

    def test_one_hot_targets_are_reduced_to_labels(self):
        predictions = jnp.zeros((3, 5))
        labels = jnp.array([4, 0, 2])

        result = _target_labels(jax.nn.one_hot(labels, 5), predictions)

        assert jnp.array_equal(result, labels)

    @pytest.mark.parametrize(
        "target_shape, prediction_shape",
        [((1, 6), (1, 6, 6)), ((4, 6), (4, 6, 11)), ((2, 3, 1), (2, 3, 5))],
    )
    def test_other_targets_are_used_as_labels(self, target_shape, prediction_shape):
        targets = jnp.arange(np.prod(target_shape)).reshape(target_shape)

        result = _target_labels(targets, jnp.zeros(prediction_shape))

        assert result is targets


def _reference_dropout_clamps(structure, batch_size, rng_key):
    """Training-mode dropout masks: one key per task in sorted task order."""
    tasks = sorted(k for k in structure.task_map if k.startswith("dropout_"))
    keep = 1.0 - structure.config["dropout_rate"]
    clamps = {}
    for task, key in zip(tasks, jax.random.split(rng_key, len(tasks))):
        node = structure.task_map[task]
        shape = (batch_size, *structure.nodes[node].node_info.shape)
        clamps[node] = jax.random.bernoulli(key, keep, shape).astype(jnp.float32) / keep
    return clamps


@pytest.fixture
def causal_mask_graph(rng_key):
    """one-hot input -> embed -> TransformerBlock (external mask) -> softmax."""
    vocab = 7
    input_node = Linear(
        shape=(SEQ_LEN, vocab), activation=IdentityActivation(), name="input"
    )
    embed = Linear(
        shape=(SEQ_LEN, EMBED_DIM), activation=IdentityActivation(), name="embed"
    )
    mask_node = Linear(
        shape=(1, SEQ_LEN, SEQ_LEN), activation=IdentityActivation(), name="mask"
    )
    block = TransformerBlock(
        shape=(SEQ_LEN, EMBED_DIM),
        num_heads=2,
        ff_dim=16,
        internal_activation=GeluActivation(),
        rope_theta=100.0,
        name="block",
    )
    output = Linear(
        shape=(SEQ_LEN, vocab), activation=SoftmaxActivation(), name="output"
    )
    structure = graph(
        nodes=[input_node, embed, mask_node, block, output],
        edges=[
            Edge(source=input_node, target=embed.slot("in")),
            Edge(source=embed, target=block.slot("in")),
            Edge(source=mask_node, target=block.slot("mask")),
            Edge(source=block, target=output.slot("in")),
        ],
        task_map=TaskMap(x=input_node, y=output, causal_mask=mask_node),
        inference=InferenceSGDNormClip(eta_infer=0.1, infer_steps=2, max_norm=5.0),
        graph_state_initializer=FeedforwardStateInit(),
    )
    tokens = jax.random.randint(rng_key, (2, SEQ_LEN), 0, vocab)
    batch = {"x": jax.nn.one_hot(tokens, vocab), "y": tokens}
    return initialize_params(structure, rng_key), structure, batch


class TestSuppliedInputsTakePrecedence:
    def test_supplied_zero_dropout_mask_is_kept(self, rng_key):
        params, structure = _encoder(rng_key, dropout_rate=0.5)
        batch = {k: jnp.asarray(v) for k, v in _mlm_batch(2).items()}
        batch["dropout_L0_mha"] = jnp.zeros((2, SEQ_LEN, EMBED_DIM))

        for training in (False, True):
            state = compute_forward_pass(params, structure, batch, rng_key, training)
            assert jnp.all(state.nodes["dropout_L0_mha"].z_latent == 0.0)

    def test_supplied_dropout_masks_fix_the_loss_path(self, rng_key):
        params, structure = _encoder(rng_key, dropout_rate=0.5)
        batch = {k: jnp.asarray(v) for k, v in _mlm_batch(2).items()}
        batch["dropout_L0_mha"] = jnp.zeros((2, SEQ_LEN, EMBED_DIM))
        batch["dropout_L0_mlp2"] = jnp.ones((2, SEQ_LEN, EMBED_DIM))

        _, first = compute_loss_autoregressive(params, structure, batch, rng_key)
        _, second = compute_loss_autoregressive(
            params, structure, batch, jax.random.PRNGKey(7)
        )
        state = compute_forward_pass(params, structure, batch, rng_key)

        assert jnp.array_equal(first, second)
        assert jnp.allclose(first, state.nodes["mlm_head"].z_mu, atol=1e-6)

    def test_generated_masks_are_reproducible(self, rng_key):
        _, structure = _encoder(rng_key, dropout_rate=0.5)

        clamps = create_dropout_clamps(structure, 2, rng_key, training=True)
        expected = _reference_dropout_clamps(structure, 2, rng_key)

        assert clamps.keys() == expected.keys()
        assert all(jnp.array_equal(clamps[k], expected[k]) for k in expected)

    def test_excluding_one_mask_keeps_the_others(self, rng_key):
        _, structure = _encoder(rng_key, dropout_rate=0.5)

        full = create_dropout_clamps(structure, 2, rng_key, training=True)
        partial = create_dropout_clamps(
            structure, 2, rng_key, training=True, exclude={"dropout_L0_mha"}
        )

        assert set(partial) == set(full) - {"dropout_L0_mha"}
        assert all(jnp.array_equal(partial[k], full[k]) for k in partial)

    def test_supplying_one_mask_keeps_the_others_in_forward_pass(self, rng_key):
        params, structure = _encoder(rng_key, dropout_rate=0.5)
        batch = {k: jnp.asarray(v) for k, v in _mlm_batch(2).items()}
        supplied = {**batch, "dropout_L0_mha": jnp.zeros((2, SEQ_LEN, EMBED_DIM))}

        base = compute_forward_pass(params, structure, batch, rng_key, True)
        partial = compute_forward_pass(params, structure, supplied, rng_key, True)

        assert jnp.array_equal(
            base.nodes["dropout_L0_mlp2"].z_latent,
            partial.nodes["dropout_L0_mlp2"].z_latent,
        )

    def test_supplied_causal_mask_is_kept(self, causal_mask_graph, rng_key):
        params, structure, batch = causal_mask_graph
        full = jnp.ones((2, 1, SEQ_LEN, SEQ_LEN))
        causal = jnp.broadcast_to(
            jnp.tril(jnp.ones((SEQ_LEN, SEQ_LEN))), (2, 1, SEQ_LEN, SEQ_LEN)
        )

        def predictions(extra):
            _, preds = compute_loss_autoregressive(
                params, structure, {**batch, **extra}, rng_key, training=False
            )
            return preds

        default = predictions({})
        assert jnp.allclose(predictions({"causal_mask": causal}), default)
        assert not jnp.allclose(predictions({"causal_mask": full}), default)


_PMAP_SCRIPT = textwrap.dedent("""
    import jax
    import jax.numpy as jnp
    import optax

    from fabricpc.core.inference import InferenceSGDNormClip
    from fabricpc.graph_initialization import initialize_params
    from fabricpc.nodes.transformer_v2 import create_deep_transformer
    from fabricpc.training.train import (
        replicate_opt_state,
        replicate_params,
        shard_batch,
    )
    from fabricpc.training.train_backprop import (
        create_pmap_eval_step_backprop_autoregressive,
        create_pmap_train_step_backprop_autoregressive,
        train_step_backprop_autoregressive,
    )

    assert jax.local_device_count() == 2
    key = jax.random.PRNGKey(0)
    structure = create_deep_transformer(
        depth=1, embed_dim=8, num_heads=2, mlp_dim=16, seq_len=6, vocab_size=11,
        inference=InferenceSGDNormClip(eta_infer=0.1, infer_steps=2, max_norm=5.0),
    )
    params = initialize_params(structure, key)
    optimizer = optax.sgd(0.1)
    x_key, y_key = jax.random.split(key)
    batch = {
        "x": jax.random.randint(x_key, (4, 6), 0, 11),
        "y": jax.random.randint(y_key, (4, 6), 0, 11),
    }
    loss_mask = jnp.zeros((4, 6)).at[0, 0].set(1.0).at[2:, :2].set(1.0)
    masked = {**batch, "loss_mask": loss_mask}
    device_keys = jax.random.split(key, 2)
    rep_params = replicate_params(params, 2)
    rep_opt = replicate_opt_state(optimizer.init(params), 2)

    step = create_pmap_train_step_backprop_autoregressive(structure, optimizer)
    try:
        step(rep_params, rep_opt, shard_batch(masked, 2), device_keys)
    except NotImplementedError as error:
        assert "loss_mask" in str(error)
    else:
        raise AssertionError("masked train batch was accepted")

    new_params, _, _, _ = step(rep_params, rep_opt, shard_batch(batch, 2), device_keys)
    single, _, _, _ = train_step_backprop_autoregressive(
        params, optimizer.init(params), batch, structure, optimizer, key
    )
    diffs = jax.tree_util.tree_map(
        lambda a, b: jnp.max(jnp.abs(a[0] - b)), new_params, single
    )
    assert max(float(d) for d in jax.tree_util.tree_leaves(diffs)) < 1e-5

    eval_step = create_pmap_eval_step_backprop_autoregressive(structure, True)
    try:
        eval_step(rep_params, shard_batch(masked, 2), device_keys)
    except NotImplementedError as error:
        assert "loss_mask" in str(error)
    else:
        raise AssertionError("masked eval batch was accepted")
    print("ok")
    """)


class TestPmapStepFactoriesRejectMaskedBatches:
    def test_two_device_factories(self):
        """Runs on two forced CPU devices, which must be set before JAX starts."""
        repo_root = Path(__file__).resolve().parents[1]
        env = {
            **os.environ,
            "XLA_FLAGS": "--xla_force_host_platform_device_count=2",
            "JAX_PLATFORMS": "cpu",
            "PYTHONPATH": os.pathsep.join(
                p for p in (str(repo_root), os.environ.get("PYTHONPATH")) if p
            ),
        }
        result = subprocess.run(
            [sys.executable, "-c", _PMAP_SCRIPT],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )

        assert result.returncode == 0, result.stderr[-4000:]
        assert result.stdout.strip().endswith("ok")
