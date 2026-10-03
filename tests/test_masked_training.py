"""Tests for the generic masking primitives used by masked-LM training.

Covers the MhaResidualNode padding mask, ``compute_loss(loss_mask=...)``,
auxiliary-task clamping in the backprop forward/loss paths, and the pmap guard
for masked batches.
"""

import jax
import jax.numpy as jnp
import optax
import pytest

from fabricpc.core.inference import InferenceSGDNormClip
from fabricpc.core.topology import Edge
from fabricpc.core.types import GraphState, NodeState
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.graph_initialization.state_initializer import FeedforwardStateInit
from fabricpc.nodes import IdentityNode, Linear
from fabricpc.nodes.transformer_v2 import MhaResidualNode, create_deep_transformer
from fabricpc.training.train_autoregressive import compute_loss
from fabricpc.training.train_backprop import (
    compute_forward_pass,
    compute_loss_autoregressive,
    eval_step_backprop,
    evaluate_backprop_autoregressive,
    train_backprop_autoregressive,
    train_step_backprop,
)

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
        x = jax.random.normal(rng_key, (1, SEQ_LEN, EMBED_DIM))
        z_mu = (
            _forward(params, structure, x, _padded_mask(4), rng_key).nodes["mha"].z_mu
        )

        # Residual only: zero attention output plus the (zero-initialised) b_o.
        b_o = params.nodes["mha"].biases["b_o"]
        assert jnp.allclose(z_mu[:, 4:], x[:, 4:] + b_o, atol=1e-6)

    def test_forward_pass_clamps_attention_mask(self, masked_attention_graph, rng_key):
        """Without clamping, the mask node gets a random latent and is ignored."""
        params, structure = masked_attention_graph
        x = jax.random.normal(rng_key, (1, SEQ_LEN, EMBED_DIM))
        full = _forward(params, structure, x, _padded_mask(SEQ_LEN), rng_key)
        padded = _forward(params, structure, x, _padded_mask(4), rng_key)

        assert jnp.allclose(padded.nodes["attention_mask"].z_latent, _padded_mask(4))
        delta = jnp.abs(full.nodes["mha"].z_mu - padded.nodes["mha"].z_mu)
        assert jnp.max(delta[:, :4]) > 1e-4


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
        # eval_step_backprop takes one-hot targets (it argmaxes them).
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
