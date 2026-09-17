"""Tests for learned absolute Vision Transformer position embeddings."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import pytest

from fabricpc.core.activations import IdentityActivation
from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.inference import InferenceSGD
from fabricpc.core.initializers import NormalInitializer, OnesInitializer
from fabricpc.core.mupc import MuPCConfig
from fabricpc.core.topology import Edge
from fabricpc.core.types import NodeInfo, NodeState
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.nodes import (
    IdentityNode,
    LearnedPositionEmbeddingNode,
    Linear,
    PatchEmbeddingNode,
    interpolate_position_embeddings,
)

GRID_SIZE = (2, 3)
NUM_PREFIX_TOKENS = 1
EMBED_DIM = 8
NUM_PATCHES = GRID_SIZE[0] * GRID_SIZE[1]
SEQUENCE_LENGTH = NUM_PREFIX_TOKENS + NUM_PATCHES
PATCH_SHAPE = (NUM_PATCHES, EMBED_DIM)
OUTPUT_SHAPE = (SEQUENCE_LENGTH, EMBED_DIM)


def _make_node_info(
    shape=OUTPUT_SHAPE,
    grid_size=GRID_SIZE,
    num_prefix_tokens=NUM_PREFIX_TOKENS,
):
    return NodeInfo(
        name="position",
        shape=shape,
        node_type=LearnedPositionEmbeddingNode.__name__,
        node_class=LearnedPositionEmbeddingNode,
        node_config={
            "grid_size": grid_size,
            "num_prefix_tokens": num_prefix_tokens,
        },
        activation=IdentityActivation(),
        energy=GaussianEnergy(),
        latent_init=NormalInitializer(),
        weight_init=NormalInitializer(std=0.02),
        slots={},
        in_degree=1,
        out_degree=0,
        in_edges=("tokens->position:in",),
        out_edges=(),
    )


def _make_state(key, batch_size, shape=OUTPUT_SHAPE):
    z_latent = jax.random.normal(key, (batch_size, *shape))
    zeros = jnp.zeros_like(z_latent)
    return NodeState(
        z_latent=z_latent,
        z_mu=zeros,
        error=zeros,
        energy=jnp.zeros(batch_size),
        pre_activation=zeros,
        latent_grad=zeros,
    )


def _make_params(
    key=jax.random.PRNGKey(0),
    shape=OUTPUT_SHAPE,
    input_shape=PATCH_SHAPE,
    initializer=NormalInitializer(std=0.02),
    num_prefix_tokens=NUM_PREFIX_TOKENS,
    grid_size=GRID_SIZE,
):
    return LearnedPositionEmbeddingNode.initialize_params(
        key=key,
        node_shape=shape,
        input_shapes={"tokens->position:in": input_shape},
        weight_init=initializer,
        config={
            "grid_size": grid_size,
            "num_prefix_tokens": num_prefix_tokens,
        },
    )


class TestConstruction:
    def test_valid_construction(self):
        node = LearnedPositionEmbeddingNode(
            shape=(SEQUENCE_LENGTH, EMBED_DIM),
            name="position",
            grid_size=GRID_SIZE,
            num_prefix_tokens=NUM_PREFIX_TOKENS,
        )
        assert node.shape == (SEQUENCE_LENGTH, EMBED_DIM)

    def test_supports_no_prefix_tokens(self):
        node = LearnedPositionEmbeddingNode(
            shape=(6, EMBED_DIM),
            name="position",
            grid_size=GRID_SIZE,
            num_prefix_tokens=0,
        )
        assert node.shape == (6, EMBED_DIM)

    def test_shape_must_be_two_dimensional(self):
        with pytest.raises(ValueError, match="shape must be"):
            LearnedPositionEmbeddingNode(
                shape=(SEQUENCE_LENGTH,),
                name="bad",
                grid_size=GRID_SIZE,
            )

    @pytest.mark.parametrize("shape", [(0, 8), (-1, 8), (7, 0), (True, 8)])
    def test_shape_dimensions_must_be_positive_integers(self, shape):
        with pytest.raises(ValueError, match="positive integer"):
            LearnedPositionEmbeddingNode(
                shape=shape,
                name="bad",
                grid_size=GRID_SIZE,
            )

    @pytest.mark.parametrize(
        "grid_size",
        [[2, 3], (2,), (0, 3), (2, -1), (True, 3), (2.0, 3)],
    )
    def test_grid_size_validation(self, grid_size):
        with pytest.raises(ValueError, match="grid_size"):
            LearnedPositionEmbeddingNode(
                shape=(SEQUENCE_LENGTH, EMBED_DIM),
                name="bad",
                grid_size=grid_size,
            )

    @pytest.mark.parametrize("num_prefix_tokens", [-1, True, 1.5, "1"])
    def test_prefix_count_validation(self, num_prefix_tokens):
        with pytest.raises(ValueError, match="non-negative integer"):
            LearnedPositionEmbeddingNode(
                shape=(SEQUENCE_LENGTH, EMBED_DIM),
                name="bad",
                grid_size=GRID_SIZE,
                num_prefix_tokens=num_prefix_tokens,
            )

    def test_sequence_length_must_match_grid_and_prefix(self):
        with pytest.raises(ValueError, match="must equal num_prefix_tokens"):
            LearnedPositionEmbeddingNode(
                shape=(99, EMBED_DIM),
                name="bad",
                grid_size=GRID_SIZE,
                num_prefix_tokens=1,
            )


class TestSlotsAndParams:
    def test_slot_is_single_input_and_unscaled(self):
        slots = LearnedPositionEmbeddingNode.get_slots()
        assert set(slots) == {"in"}
        assert not slots["in"].is_multi_input
        assert not slots["in"].is_variance_scalable
        assert LearnedPositionEmbeddingNode.get_weight_fan_in(PATCH_SHAPE, {}) == 1

    def test_parameter_shape(self):
        params = _make_params()
        assert params.weights["position_embeddings"].shape == OUTPUT_SHAPE
        assert params.weights["cls_token"].shape == (NUM_PREFIX_TOKENS, EMBED_DIM)
        assert params.biases == {}

    def test_no_prefix_omits_cls_token_parameter(self):
        params = _make_params(
            shape=PATCH_SHAPE,
            input_shape=PATCH_SHAPE,
            num_prefix_tokens=0,
        )
        assert params.weights["position_embeddings"].shape == PATCH_SHAPE
        assert "cls_token" not in params.weights

    def test_initializer_is_used(self):
        params = _make_params(initializer=OnesInitializer())
        assert jnp.all(params.weights["position_embeddings"] == 1)
        assert jnp.all(params.weights["cls_token"] == 1)

    def test_weight_init_none_falls_back_to_default(self):
        params = _make_params(initializer=None)
        assert params.weights["position_embeddings"].shape == OUTPUT_SHAPE
        assert params.weights["cls_token"].shape == (NUM_PREFIX_TOKENS, EMBED_DIM)

    @pytest.mark.parametrize(
        "input_shapes",
        [
            {},
            {
                "first": (SEQUENCE_LENGTH, EMBED_DIM),
                "second": (SEQUENCE_LENGTH, EMBED_DIM),
            },
        ],
    )
    def test_requires_exactly_one_input(self, input_shapes):
        with pytest.raises(ValueError, match="exactly one input shape"):
            LearnedPositionEmbeddingNode.initialize_params(
                key=jax.random.PRNGKey(0),
                node_shape=(SEQUENCE_LENGTH, EMBED_DIM),
                input_shapes=input_shapes,
                weight_init=NormalInitializer(std=0.02),
                config={},
            )

    def test_input_must_be_patch_sequence(self):
        with pytest.raises(ValueError, match="expected"):
            LearnedPositionEmbeddingNode.initialize_params(
                key=jax.random.PRNGKey(0),
                node_shape=OUTPUT_SHAPE,
                input_shapes={"in": OUTPUT_SHAPE},
                weight_init=NormalInitializer(std=0.02),
                config={"num_prefix_tokens": NUM_PREFIX_TOKENS},
            )

    def test_embed_dim_must_match(self):
        with pytest.raises(ValueError, match="embed_dim mismatch"):
            LearnedPositionEmbeddingNode.initialize_params(
                key=jax.random.PRNGKey(0),
                node_shape=OUTPUT_SHAPE,
                input_shapes={"in": (NUM_PATCHES, EMBED_DIM + 1)},
                weight_init=NormalInitializer(std=0.02),
                config={"num_prefix_tokens": NUM_PREFIX_TOKENS},
            )

    def test_input_must_be_2d(self):
        with pytest.raises(ValueError, match="2-D patch input"):
            LearnedPositionEmbeddingNode.initialize_params(
                key=jax.random.PRNGKey(0),
                node_shape=OUTPUT_SHAPE,
                input_shapes={"in": (NUM_PATCHES,)},
                weight_init=NormalInitializer(std=0.02),
                config={"num_prefix_tokens": NUM_PREFIX_TOKENS},
            )


class TestForwardAndGradients:
    def test_forward_prepends_cls_and_adds_positions(self):
        batch_size = 3
        params = _make_params(initializer=OnesInitializer())
        node_info = _make_node_info()
        x = jnp.full((batch_size, NUM_PATCHES, EMBED_DIM), 2.0)
        state = _make_state(jax.random.PRNGKey(1), batch_size)

        _, new_state = LearnedPositionEmbeddingNode.forward(
            params,
            {node_info.in_edges[0]: x},
            state,
            node_info,
        )

        assert new_state.z_mu.shape == (batch_size, SEQUENCE_LENGTH, EMBED_DIM)
        # μ_CLS = c + P_0 = 1 + 1; μ_patch = z_patch + P_patch = 2 + 1
        assert jnp.allclose(new_state.z_mu[:, 0, :], 2.0)
        assert jnp.allclose(new_state.z_mu[:, 1:, :], 3.0)
        assert jnp.all(new_state.energy >= 0)

    def test_cls_is_at_index_zero_and_patch_order_is_preserved(self):
        params = _make_params()
        node_info = _make_node_info()
        x = jnp.arange(NUM_PATCHES * EMBED_DIM, dtype=jnp.float32).reshape(
            1, NUM_PATCHES, EMBED_DIM
        )
        state = _make_state(jax.random.PRNGKey(1), 1)
        _, new_state = LearnedPositionEmbeddingNode.forward(
            params, {node_info.in_edges[0]: x}, state, node_info
        )
        cls = params.weights["cls_token"]
        pos = params.weights["position_embeddings"]
        expected = jnp.concatenate([cls[jnp.newaxis], x], axis=1) + pos
        assert jnp.allclose(new_state.z_mu, expected)
        assert jnp.allclose(
            new_state.z_mu[0, 1:] - pos[1:],
            x[0],
        )

    def test_jit_forward(self):
        params = _make_params()
        node_info = _make_node_info()
        x = jnp.zeros((2, NUM_PATCHES, EMBED_DIM))
        state = _make_state(jax.random.PRNGKey(2), 2)

        @jax.jit
        def run(p, inputs, node_state):
            _, result = LearnedPositionEmbeddingNode.forward(
                p, inputs, node_state, node_info
            )
            return result

        result = run(params, {node_info.in_edges[0]: x}, state)
        assert result.z_mu.shape == (2, SEQUENCE_LENGTH, EMBED_DIM)

    def test_raw_parameter_and_input_gradients(self):
        params = _make_params()
        node_info = _make_node_info()
        x = jnp.zeros((2, NUM_PATCHES, EMBED_DIM))
        state = _make_state(jax.random.PRNGKey(3), 2)

        def parameter_loss(p):
            _, result = LearnedPositionEmbeddingNode.forward(
                p, {node_info.in_edges[0]: x}, state, node_info
            )
            return jnp.sum(result.energy)

        def input_loss(value):
            _, result = LearnedPositionEmbeddingNode.forward(
                params, {node_info.in_edges[0]: value}, state, node_info
            )
            return jnp.sum(result.energy)

        parameter_grads = jax.grad(parameter_loss)(params)
        input_grads = jax.grad(input_loss)(x)
        assert jnp.any(parameter_grads.weights["position_embeddings"] != 0)
        assert jnp.any(parameter_grads.weights["cls_token"] != 0)
        assert input_grads.shape == x.shape
        assert jnp.any(input_grads != 0)

    def test_cls_gradient_comes_from_cls_error_slice(self):
        batch_size = 4
        params = _make_params(initializer=OnesInitializer())
        params = params._replace(
            weights={
                "position_embeddings": jnp.zeros(OUTPUT_SHAPE),
                "cls_token": jnp.zeros((NUM_PREFIX_TOKENS, EMBED_DIM)),
            }
        )
        node_info = _make_node_info()
        x = jnp.zeros((batch_size, NUM_PATCHES, EMBED_DIM))
        z_latent = jnp.zeros((batch_size, SEQUENCE_LENGTH, EMBED_DIM))
        z_latent = z_latent.at[:, 0, :].set(1.0)
        state = _make_state(jax.random.PRNGKey(5), batch_size)._replace(
            z_latent=z_latent
        )

        def loss(p):
            _, result = LearnedPositionEmbeddingNode.forward(
                p, {node_info.in_edges[0]: x}, state, node_info
            )
            return jnp.sum(result.energy)

        grads = jax.grad(loss)(params)
        assert jnp.allclose(
            grads.weights["cls_token"],
            -batch_size * jnp.ones((NUM_PREFIX_TOKENS, EMBED_DIM)),
        )
        pos_grad = grads.weights["position_embeddings"]
        assert jnp.allclose(pos_grad[0], -batch_size * jnp.ones((EMBED_DIM,)))
        assert jnp.allclose(pos_grad[1:], 0.0)

    def test_patch_input_gradient_excludes_cls_rows(self):
        params = _make_params(initializer=OnesInitializer())
        params = params._replace(
            weights={
                "position_embeddings": jnp.zeros(OUTPUT_SHAPE),
                "cls_token": jnp.zeros((NUM_PREFIX_TOKENS, EMBED_DIM)),
            }
        )
        node_info = _make_node_info()
        x = jnp.zeros((2, NUM_PATCHES, EMBED_DIM))
        z_latent = jnp.zeros((2, SEQUENCE_LENGTH, EMBED_DIM))
        z_latent = z_latent.at[:, 0, :].set(1.0)
        state = _make_state(jax.random.PRNGKey(6), 2)._replace(z_latent=z_latent)

        def input_loss(value):
            _, result = LearnedPositionEmbeddingNode.forward(
                params, {node_info.in_edges[0]: value}, state, node_info
            )
            return jnp.sum(result.energy)

        input_grads = jax.grad(input_loss)(x)
        assert jnp.allclose(input_grads, 0.0)

    def test_fabricpc_local_gradient_apis(self):
        params = _make_params()
        node_info = replace(
            _make_node_info(),
            out_degree=1,
            out_edges=("position->next:in",),
        )
        edge_key = node_info.in_edges[0]
        x = jnp.zeros((2, NUM_PATCHES, EMBED_DIM))
        state = _make_state(jax.random.PRNGKey(4), 2)

        _, input_grads, self_grad = (
            LearnedPositionEmbeddingNode.forward_and_latent_grads(
                params,
                {edge_key: x},
                state,
                node_info,
                is_clamped=False,
            )
        )
        _, parameter_grads = LearnedPositionEmbeddingNode.forward_and_weight_grads(
            params,
            {edge_key: x},
            state,
            node_info,
        )

        assert input_grads[edge_key].shape == x.shape
        assert jnp.any(input_grads[edge_key] != 0)
        assert jnp.any(self_grad != 0)
        assert jnp.any(parameter_grads.weights["position_embeddings"] != 0)
        assert jnp.any(parameter_grads.weights["cls_token"] != 0)

    def test_shared_position_gradient_sums_across_batch(self):
        batch_size = 4
        params = _make_params(initializer=OnesInitializer())
        params = params._replace(
            weights={
                "position_embeddings": jnp.zeros(OUTPUT_SHAPE),
                "cls_token": jnp.zeros((NUM_PREFIX_TOKENS, EMBED_DIM)),
            }
        )
        node_info = _make_node_info()
        x = jnp.zeros((batch_size, NUM_PATCHES, EMBED_DIM))
        state = _make_state(jax.random.PRNGKey(5), batch_size)._replace(
            z_latent=jnp.ones((batch_size, SEQUENCE_LENGTH, EMBED_DIM))
        )

        def loss(p):
            _, result = LearnedPositionEmbeddingNode.forward(
                p, {node_info.in_edges[0]: x}, state, node_info
            )
            return jnp.sum(result.energy)

        gradient = jax.grad(loss)(params).weights["position_embeddings"]
        assert jnp.allclose(
            gradient,
            -batch_size * jnp.ones(OUTPUT_SHAPE),
        )

    def test_multiple_prefix_tokens_are_all_prepended(self):
        """K=2 prepends both learned tokens in row order ahead of the patches."""
        num_prefix = 2
        sequence_length = num_prefix + NUM_PATCHES
        shape = (sequence_length, EMBED_DIM)
        params = _make_params(shape=shape, num_prefix_tokens=num_prefix)._replace(
            weights={
                "position_embeddings": jnp.zeros(shape),
                # Distinct rows so a swapped or dropped prefix token is visible.
                "cls_token": jnp.array([[1.0] * EMBED_DIM, [2.0] * EMBED_DIM]),
            }
        )
        node_info = _make_node_info(shape=shape, num_prefix_tokens=num_prefix)
        x = jnp.full((2, NUM_PATCHES, EMBED_DIM), 7.0)
        state = _make_state(jax.random.PRNGKey(11), 2, shape)

        _, new_state = LearnedPositionEmbeddingNode.forward(
            params, {node_info.in_edges[0]: x}, state, node_info
        )

        assert new_state.z_mu.shape == (2, sequence_length, EMBED_DIM)
        assert jnp.allclose(new_state.z_mu[:, 0, :], 1.0)
        assert jnp.allclose(new_state.z_mu[:, 1, :], 2.0)
        assert jnp.allclose(new_state.z_mu[:, 2:, :], 7.0)

    def test_multiple_prefix_token_gradients_are_row_isolated(self):
        """With K=2, each prefix row learns only from its own error slice."""
        num_prefix = 2
        sequence_length = num_prefix + NUM_PATCHES
        shape = (sequence_length, EMBED_DIM)
        batch_size = 3
        params = _make_params(shape=shape, num_prefix_tokens=num_prefix)._replace(
            weights={
                "position_embeddings": jnp.zeros(shape),
                "cls_token": jnp.zeros((num_prefix, EMBED_DIM)),
            }
        )
        node_info = _make_node_info(shape=shape, num_prefix_tokens=num_prefix)
        x = jnp.zeros((batch_size, NUM_PATCHES, EMBED_DIM))
        # Error only on the second prefix row.
        z_latent = (
            jnp.zeros((batch_size, sequence_length, EMBED_DIM)).at[:, 1, :].set(1.0)
        )
        state = _make_state(jax.random.PRNGKey(12), batch_size, shape)._replace(
            z_latent=z_latent
        )

        def loss(p):
            _, result = LearnedPositionEmbeddingNode.forward(
                p, {node_info.in_edges[0]: x}, state, node_info
            )
            return jnp.sum(result.energy)

        cls_grad = jax.grad(loss)(params).weights["cls_token"]
        assert jnp.allclose(cls_grad[0], 0.0)
        assert jnp.allclose(cls_grad[1], -batch_size * jnp.ones((EMBED_DIM,)))

    def test_pre_activation_is_the_activation_input(self):
        """NodeBase contract: with no activation, pre_activation == z_mu."""
        params = _make_params()
        node_info = _make_node_info()
        x = jnp.full((2, NUM_PATCHES, EMBED_DIM), 0.5)
        state = _make_state(jax.random.PRNGKey(13), 2)

        _, new_state = LearnedPositionEmbeddingNode.forward(
            params, {node_info.in_edges[0]: x}, state, node_info
        )

        assert jnp.allclose(new_state.pre_activation, new_state.z_mu)

    def test_forward_rejects_input_inconsistent_with_declared_length(self):
        """A single-token input must raise instead of broadcasting to (T, D)."""
        params = _make_params()
        node_info = _make_node_info(num_prefix_tokens=0)
        x = jnp.ones((2, 1, EMBED_DIM))
        state = _make_state(jax.random.PRNGKey(14), 2)

        with pytest.raises(ValueError, match="declares sequence length"):
            LearnedPositionEmbeddingNode.forward(
                params, {node_info.in_edges[0]: x}, state, node_info
            )

    def test_position_table_and_cls_token_use_independent_keys(self):
        """The two parameters must not be drawn from the same PRNG key."""
        params = _make_params(initializer=NormalInitializer(std=1.0))
        # Same key would make cls_token equal the table's first K rows.
        assert not jnp.allclose(
            params.weights["cls_token"],
            params.weights["position_embeddings"][:NUM_PREFIX_TOKENS],
        )

    def test_no_prefix_is_pure_addition(self):
        params = _make_params(
            shape=PATCH_SHAPE,
            input_shape=PATCH_SHAPE,
            num_prefix_tokens=0,
            initializer=OnesInitializer(),
        )
        node_info = _make_node_info(shape=PATCH_SHAPE, num_prefix_tokens=0)
        x = jnp.full((2, NUM_PATCHES, EMBED_DIM), 2.0)
        state = _make_state(jax.random.PRNGKey(7), 2, PATCH_SHAPE)
        _, new_state = LearnedPositionEmbeddingNode.forward(
            params, {node_info.in_edges[0]: x}, state, node_info
        )
        assert new_state.z_mu.shape == x.shape
        assert jnp.allclose(new_state.z_mu, 3.0)


class TestInterpolation:
    def test_upsampling_preserves_prefix_and_changes_shape(self):
        positions = jnp.arange(7 * 2, dtype=jnp.float32).reshape(7, 2)
        resized = interpolate_position_embeddings(
            positions,
            (2, 3),
            (4, 5),
            num_prefix_tokens=1,
        )
        assert resized.shape == (21, 2)
        assert jnp.array_equal(resized[:1], positions[:1])
        assert jnp.all(jnp.isfinite(resized))

    def test_downsampling_preserves_multiple_prefix_tokens(self):
        positions = jnp.arange(18 * 3, dtype=jnp.float32).reshape(18, 3)
        resized = interpolate_position_embeddings(
            positions,
            (4, 4),
            (2, 2),
            num_prefix_tokens=2,
        )
        assert resized.shape == (6, 3)
        assert jnp.array_equal(resized[:2], positions[:2])

    def test_constant_patch_grid_remains_constant(self):
        prefix = jnp.array([[9.0, 10.0]])
        patches = jnp.full((6, 2), 3.5)
        positions = jnp.concatenate([prefix, patches], axis=0)
        resized = interpolate_position_embeddings(
            positions,
            (2, 3),
            (5, 4),
            num_prefix_tokens=1,
        )
        assert jnp.array_equal(resized[:1], prefix)
        assert jnp.allclose(resized[1:], 3.5, atol=1e-5)

    def test_same_grid_returns_unchanged_values(self):
        positions = jax.random.normal(jax.random.PRNGKey(6), (7, 4))
        resized = interpolate_position_embeddings(
            positions,
            (2, 3),
            (2, 3),
            num_prefix_tokens=1,
        )
        assert jnp.array_equal(resized, positions)

    def test_no_prefix_tokens(self):
        positions = jnp.ones((6, 4))
        resized = interpolate_position_embeddings(
            positions,
            (2, 3),
            (3, 3),
            num_prefix_tokens=0,
        )
        assert resized.shape == (9, 4)
        assert jnp.allclose(resized, 1.0)

    def test_interpolate_params_preserves_other_parameters(self):
        params = _make_params()
        params = params._replace(
            weights={
                **params.weights,
                "other": jnp.array([1.0]),
            }
        )
        resized = LearnedPositionEmbeddingNode.interpolate_params(
            params,
            (2, 3),
            (4, 4),
            num_prefix_tokens=1,
        )
        assert resized.weights["position_embeddings"].shape == (17, EMBED_DIM)
        assert jnp.array_equal(
            resized.weights["cls_token"], params.weights["cls_token"]
        )
        assert jnp.array_equal(resized.weights["other"], params.weights["other"])

    @pytest.mark.parametrize(
        "positions,source_grid,prefix,match",
        [
            (jnp.zeros((1, 7, 4)), (2, 3), 1, "must have shape"),
            (jnp.zeros((8, 4)), (2, 3), 1, "does not match"),
            (jnp.zeros((7, 4)), (0, 3), 1, "positive integer"),
            (jnp.zeros((7, 4)), (2, 3), -1, "non-negative integer"),
        ],
    )
    def test_interpolation_validation(self, positions, source_grid, prefix, match):
        with pytest.raises(ValueError, match=match):
            interpolate_position_embeddings(
                positions,
                source_grid,
                (3, 3),
                num_prefix_tokens=prefix,
            )


def _build_position_graph(with_mupc=False):
    tokens = IdentityNode(shape=PATCH_SHAPE, name="tokens")
    position = LearnedPositionEmbeddingNode(
        shape=OUTPUT_SHAPE,
        name="position",
        grid_size=GRID_SIZE,
        num_prefix_tokens=NUM_PREFIX_TOKENS,
    )
    output = Linear(shape=(2,), name="output", flatten_input=True)
    return graph(
        nodes=[tokens, position, output],
        edges=[
            Edge(source=tokens, target=position.slot("in")),
            Edge(source=position, target=output.slot("in")),
        ],
        task_map=TaskMap(x=tokens, y=output),
        inference=InferenceSGD(eta_infer=0.05, infer_steps=3),
        scaling=MuPCConfig() if with_mupc else None,
    )


def test_real_graph_initialization_and_feedforward_state(rng_key):
    structure = _build_position_graph()
    params = initialize_params(structure, rng_key)
    x = jax.random.normal(rng_key, (2, NUM_PATCHES, EMBED_DIM))
    state = initialize_graph_state(
        structure,
        batch_size=2,
        rng_key=rng_key,
        clamps={"tokens": x, "output": jnp.zeros((2, 2))},
        params=params,
    )

    position_info = structure.nodes["position"].node_info
    assert position_info.node_config["grid_size"] == GRID_SIZE
    assert position_info.node_config["num_prefix_tokens"] == NUM_PREFIX_TOKENS

    positions = params.nodes["position"].weights["position_embeddings"]
    cls_token = params.nodes["position"].weights["cls_token"]
    position_mu = state.nodes["position"].z_mu
    expected = (
        jnp.concatenate(
            [jnp.broadcast_to(cls_token[jnp.newaxis], (2, 1, EMBED_DIM)), x],
            axis=1,
        )
        + positions[jnp.newaxis]
    )
    assert positions.shape == OUTPUT_SHAPE
    assert cls_token.shape == (NUM_PREFIX_TOKENS, EMBED_DIM)
    assert position_mu.shape == (2, SEQUENCE_LENGTH, EMBED_DIM)
    assert jnp.allclose(position_mu, expected)


def test_wired_after_patch_embedding_node(rng_key):
    """image → PatchEmbeddingNode (N, D) → this node → (N+1, D).

    Exercises the real ViT stem wiring: the asymmetric (N, D) → (N+K, D) shape
    contract has to survive graph construction against a genuine patch source,
    not just an IdentityNode standing in for one.
    """
    patch_size, embed_dim, channels = 4, EMBED_DIM, 1
    height = width = 8
    grid_size = (height // patch_size, width // patch_size)  # (2, 2)
    num_patches = grid_size[0] * grid_size[1]  # 4
    sequence_length = num_patches + 1

    image = IdentityNode(shape=(height, width, channels), name="image")
    patch = PatchEmbeddingNode(
        shape=(num_patches, embed_dim),
        name="patch",
        patch_size=patch_size,
        embed_dim=embed_dim,
        in_channels=channels,
    )
    position = LearnedPositionEmbeddingNode(
        shape=(sequence_length, embed_dim),
        name="position",
        grid_size=grid_size,
        num_prefix_tokens=1,
    )
    output = Linear(shape=(2,), name="output", flatten_input=True)
    structure = graph(
        nodes=[image, patch, position, output],
        edges=[
            Edge(source=image, target=patch.slot("in")),
            Edge(source=patch, target=position.slot("in")),
            Edge(source=position, target=output.slot("in")),
        ],
        task_map=TaskMap(x=image, y=output),
        inference=InferenceSGD(eta_infer=0.05, infer_steps=3),
    )

    params = initialize_params(structure, rng_key)
    x = jax.random.normal(rng_key, (2, height, width, channels))
    state = initialize_graph_state(
        structure,
        batch_size=2,
        rng_key=rng_key,
        clamps={"image": x, "output": jnp.zeros((2, 2))},
        params=params,
    )

    patch_mu = state.nodes["patch"].z_mu
    position_mu = state.nodes["position"].z_mu
    assert patch_mu.shape == (2, num_patches, embed_dim)
    assert position_mu.shape == (2, sequence_length, embed_dim)

    # The patch rows of z_0 are the patch prediction plus the patch positions,
    # and row 0 is the CLS parameter plus the CLS position.
    weights = params.nodes["position"].weights
    positions = weights["position_embeddings"]
    assert jnp.allclose(
        position_mu[:, 1:, :], patch_mu + positions[jnp.newaxis, 1:, :], atol=1e-5
    )
    assert jnp.allclose(
        position_mu[:, 0, :], weights["cls_token"][0] + positions[0], atol=1e-5
    )


def test_mupc_preserves_additive_input_scale():
    structure = _build_position_graph(with_mupc=True)
    position_info = structure.nodes["position"].node_info
    edge_key = position_info.in_edges[0]
    scaling = position_info.scaling_config

    assert scaling is not None
    assert edge_key not in scaling.forward_scale
    assert edge_key not in scaling.topdown_grad_scale
    assert edge_key not in scaling.weight_grad_scale
