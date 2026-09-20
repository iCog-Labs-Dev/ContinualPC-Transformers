"""
Unit tests for PatchEmbeddingNode.

Tests cover:
  - Construction-time validation (shape, patch_size, embed_dim, in_channels)
  - Parameter initialization (kernel shape, bias, input shape validation)
  - Forward pass output shape and correctness
  - Non-square images and raster-scan patch order
  - Hypothesis property test: conv projection == explicit x_p @ E
  - Integration with the FabricPC graph builder
  - muPC fan-in calculation

Run with:
    pytest tests/test_patch_embed.py -v
"""

import jax
import jax.numpy as jnp
import pytest
from hypothesis import given, settings, strategies as st

from fabricpc.nodes import IdentityNode, Linear, PatchEmbeddingNode
from fabricpc.core.types import NodeParams, NodeState, NodeInfo
from fabricpc.core.activations import IdentityActivation, ReLUActivation
from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.initializers import (
    KaimingInitializer,
    NormalInitializer,
    ZerosInitializer,
    OnesInitializer,
)
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.core.inference import InferenceSGD
from fabricpc.core.mupc import MuPCConfig


def _make_node_info(
    node_cls,
    node_shape,
    config,
    activation,
    energy_obj,
    latent_init,
    weight_init,
    in_edges=("src->dst:in",),
):
    return NodeInfo(
        name="test_patch_embed",
        shape=node_shape,
        node_type=node_cls.__name__,
        node_class=node_cls,
        node_config=config,
        activation=activation,
        energy=energy_obj,
        latent_init=latent_init,
        weight_init=weight_init,
        slots={},
        in_degree=len(in_edges),
        out_degree=0,
        in_edges=in_edges,
        out_edges=(),
    )


def _make_state(key, batch_size, node_shape):
    z_latent = jax.random.normal(key, (batch_size, *node_shape))
    zeros = jnp.zeros_like(z_latent)
    return NodeState(
        z_latent=z_latent,
        z_mu=zeros,
        error=zeros,
        energy=jnp.zeros(batch_size),
        pre_activation=zeros,
        latent_grad=zeros,
    )


# Standard ViT-Base/16 config for convenience
PATCH_SIZE = 16
EMBED_DIM = 768
IN_CHANNELS = 3
IMAGE_H, IMAGE_W = 224, 224
NUM_PATCHES = (IMAGE_H // PATCH_SIZE) * (IMAGE_W // PATCH_SIZE)  # 196
GRID_SIZE = (IMAGE_H // PATCH_SIZE, IMAGE_W // PATCH_SIZE)


class TestPatchEmbeddingConstruction:
    """Tests for constructor argument validation."""

    def test_valid_construction(self):
        """Standard ViT-Base/16 construction succeeds."""
        node = PatchEmbeddingNode(
            shape=(NUM_PATCHES, EMBED_DIM),
            name="patch_embed",
            patch_size=PATCH_SIZE,
            grid_size=GRID_SIZE,
            embed_dim=EMBED_DIM,
            in_channels=IN_CHANNELS,
        )
        assert node.shape == (NUM_PATCHES, EMBED_DIM)
        assert node.name == "patch_embed"

    def test_shape_must_be_2d(self):
        """Output shape must be exactly (N, D)."""
        with pytest.raises(ValueError, match="exactly 2 dimensions"):
            PatchEmbeddingNode(
                shape=(NUM_PATCHES,),  # 1D — wrong
                name="bad",
                patch_size=PATCH_SIZE,
                grid_size=GRID_SIZE,
                embed_dim=EMBED_DIM,
                in_channels=IN_CHANNELS,
            )

    def test_shape_must_be_2d_3d(self):
        """3D shape is also invalid."""
        with pytest.raises(ValueError, match="exactly 2 dimensions"):
            PatchEmbeddingNode(
                shape=(14, 14, EMBED_DIM),  # 3D — wrong
                name="bad",
                patch_size=PATCH_SIZE,
                grid_size=GRID_SIZE,
                embed_dim=EMBED_DIM,
                in_channels=IN_CHANNELS,
            )

    def test_embed_dim_mismatch(self):
        """shape[-1] must equal embed_dim."""
        with pytest.raises(ValueError, match="must equal embed_dim"):
            PatchEmbeddingNode(
                shape=(NUM_PATCHES, 512),  # D=512 ≠ embed_dim=768
                name="bad",
                patch_size=PATCH_SIZE,
                grid_size=GRID_SIZE,
                embed_dim=EMBED_DIM,
                in_channels=IN_CHANNELS,
            )

    def test_custom_activation(self):
        """Non-default activation is accepted."""
        node = PatchEmbeddingNode(
            shape=(NUM_PATCHES, EMBED_DIM),
            name="patch",
            patch_size=PATCH_SIZE,
            grid_size=GRID_SIZE,
            embed_dim=EMBED_DIM,
            in_channels=IN_CHANNELS,
            activation=ReLUActivation(),
        )
        assert node._activation is not None

    def test_small_patch_size(self):
        """Smaller patch sizes work (ViT-Huge/14 uses P=14)."""
        P = 14
        N = (IMAGE_H // P) * (IMAGE_W // P)  # 256
        node = PatchEmbeddingNode(
            shape=(N, EMBED_DIM),
            name="patch14",
            patch_size=P,
            grid_size=(IMAGE_H // P, IMAGE_W // P),
            embed_dim=EMBED_DIM,
            in_channels=IN_CHANNELS,
        )
        assert node.shape == (N, EMBED_DIM)


class TestPatchEmbeddingSlots:
    """Tests for slot specification and muPC fan-in."""

    def test_single_input_slot(self):
        """Node has exactly one single-input slot named 'in'."""
        slots = PatchEmbeddingNode.get_slots()
        assert len(slots) == 1
        assert "in" in slots
        assert not slots["in"].is_multi_input

    def test_fan_in(self):
        """fan_in = C_in × P × P."""
        config = {
            "patch_size": PATCH_SIZE,
            "in_channels": IN_CHANNELS,
        }
        fan_in = PatchEmbeddingNode.get_weight_fan_in(
            source_shape=(IMAGE_H, IMAGE_W, IN_CHANNELS), config=config
        )
        assert fan_in == IN_CHANNELS * PATCH_SIZE * PATCH_SIZE  # 3 * 16 * 16 = 768

    def test_fan_in_different_config(self):
        """fan_in adapts to different patch sizes and channels."""
        config = {"patch_size": 8, "in_channels": 1}
        fan_in = PatchEmbeddingNode.get_weight_fan_in(
            source_shape=(32, 32, 1), config=config
        )
        assert fan_in == 1 * 8 * 8  # 64


class TestPatchEmbeddingParams:
    """Tests for initialize_params."""

    def test_kernel_shape(self):
        """Projection kernel has shape (P, P, C_in, D)."""
        key = jax.random.PRNGKey(0)
        config = {
            "patch_size": PATCH_SIZE,
            "embed_dim": EMBED_DIM,
            "in_channels": IN_CHANNELS,
            "use_bias": True,
            "bias_init": ZerosInitializer(),
        }
        input_shapes = {"src->dst:in": (IMAGE_H, IMAGE_W, IN_CHANNELS)}

        params = PatchEmbeddingNode.initialize_params(
            key=key,
            node_shape=(NUM_PATCHES, EMBED_DIM),
            input_shapes=input_shapes,
            weight_init=KaimingInitializer(),
            config=config,
        )
        assert params.weights["proj_kernel"].shape == (
            PATCH_SIZE,
            PATCH_SIZE,
            IN_CHANNELS,
            EMBED_DIM,
        )

    def test_bias_shape(self):
        """Bias has shape (D,) when use_bias=True."""
        key = jax.random.PRNGKey(0)
        config = {
            "patch_size": PATCH_SIZE,
            "embed_dim": EMBED_DIM,
            "in_channels": IN_CHANNELS,
            "use_bias": True,
            "bias_init": ZerosInitializer(),
        }
        input_shapes = {"src->dst:in": (IMAGE_H, IMAGE_W, IN_CHANNELS)}

        params = PatchEmbeddingNode.initialize_params(
            key=key,
            node_shape=(NUM_PATCHES, EMBED_DIM),
            input_shapes=input_shapes,
            weight_init=KaimingInitializer(),
            config=config,
        )
        assert "proj_bias" in params.biases
        assert params.biases["proj_bias"].shape == (EMBED_DIM,)

    def test_no_bias(self):
        """No bias created when use_bias=False."""
        key = jax.random.PRNGKey(0)
        config = {
            "patch_size": PATCH_SIZE,
            "embed_dim": EMBED_DIM,
            "in_channels": IN_CHANNELS,
            "use_bias": False,
        }
        input_shapes = {"src->dst:in": (IMAGE_H, IMAGE_W, IN_CHANNELS)}

        params = PatchEmbeddingNode.initialize_params(
            key=key,
            node_shape=(NUM_PATCHES, EMBED_DIM),
            input_shapes=input_shapes,
            weight_init=KaimingInitializer(),
            config=config,
        )
        assert "proj_bias" not in params.biases

    def test_input_not_divisible_by_patch(self):
        """Raises if image dimensions are not divisible by patch_size."""
        key = jax.random.PRNGKey(0)
        config = {
            "patch_size": PATCH_SIZE,
            "embed_dim": EMBED_DIM,
            "in_channels": IN_CHANNELS,
            "use_bias": False,
        }
        # 225 is not divisible by 16
        input_shapes = {"src->dst:in": (225, 225, IN_CHANNELS)}

        with pytest.raises(ValueError, match="not evenly divisible"):
            PatchEmbeddingNode.initialize_params(
                key=key,
                node_shape=(NUM_PATCHES, EMBED_DIM),
                input_shapes=input_shapes,
                weight_init=KaimingInitializer(),
                config=config,
            )

    def test_wrong_input_channels(self):
        """Raises if input channels don't match declared in_channels."""
        key = jax.random.PRNGKey(0)
        config = {
            "patch_size": PATCH_SIZE,
            "embed_dim": EMBED_DIM,
            "in_channels": IN_CHANNELS,  # expects 3
            "use_bias": False,
        }
        input_shapes = {"src->dst:in": (IMAGE_H, IMAGE_W, 1)}  # but got 1

        with pytest.raises(ValueError, match="input channels"):
            PatchEmbeddingNode.initialize_params(
                key=key,
                node_shape=(NUM_PATCHES, EMBED_DIM),
                input_shapes=input_shapes,
                weight_init=KaimingInitializer(),
                config=config,
            )

    def test_wrong_num_patches(self):
        """Raises if declared N doesn't match H*W/P²."""
        key = jax.random.PRNGKey(0)
        config = {
            "patch_size": PATCH_SIZE,
            "embed_dim": EMBED_DIM,
            "in_channels": IN_CHANNELS,
            "use_bias": False,
        }
        input_shapes = {"src->dst:in": (IMAGE_H, IMAGE_W, IN_CHANNELS)}

        with pytest.raises(ValueError, match="declared N="):
            PatchEmbeddingNode.initialize_params(
                key=key,
                node_shape=(100, EMBED_DIM),  # wrong: should be 196
                input_shapes=input_shapes,
                weight_init=KaimingInitializer(),
                config=config,
            )

    def test_rejects_different_grid_with_same_patch_count(self):
        """Equal patch counts do not hide a different spatial geometry."""
        config = {
            "patch_size": 4,
            "grid_size": (2, 6),
            "embed_dim": 8,
            "in_channels": 1,
            "use_bias": False,
        }

        with pytest.raises(ValueError, match="produces patch grid \\(3, 4\\)"):
            PatchEmbeddingNode.initialize_params(
                key=jax.random.PRNGKey(0),
                node_shape=(12, 8),
                input_shapes={"src->dst:in": (12, 16, 1)},
                weight_init=KaimingInitializer(),
                config=config,
            )

    def test_input_must_be_3d(self):
        """Input shape must be (H, W, C), not 1D or 2D."""
        key = jax.random.PRNGKey(0)
        config = {
            "patch_size": PATCH_SIZE,
            "embed_dim": EMBED_DIM,
            "in_channels": IN_CHANNELS,
            "use_bias": False,
        }
        input_shapes = {"src->dst:in": (784,)}  # 1D — wrong

        with pytest.raises(ValueError, match="expected 3-D input"):
            PatchEmbeddingNode.initialize_params(
                key=key,
                node_shape=(NUM_PATCHES, EMBED_DIM),
                input_shapes=input_shapes,
                weight_init=KaimingInitializer(),
                config=config,
            )

    def test_bias_init_none_with_use_bias_true(self):
        """Raises if use_bias=True but bias_init is None."""
        key = jax.random.PRNGKey(0)
        config = {
            "patch_size": PATCH_SIZE,
            "embed_dim": EMBED_DIM,
            "in_channels": IN_CHANNELS,
            "use_bias": True,
            "bias_init": None,
        }
        input_shapes = {"src->dst:in": (IMAGE_H, IMAGE_W, IN_CHANNELS)}

        with pytest.raises(ValueError, match="bias_init is None"):
            PatchEmbeddingNode.initialize_params(
                key=key,
                node_shape=(NUM_PATCHES, EMBED_DIM),
                input_shapes=input_shapes,
                weight_init=KaimingInitializer(),
                config=config,
            )


class TestPatchEmbeddingForward:
    """Tests for the forward pass."""

    @pytest.fixture
    def setup_vit_base(self):
        """Set up a ViT-Base/16 PatchEmbeddingNode with parameters."""
        key = jax.random.PRNGKey(42)
        node_shape = (NUM_PATCHES, EMBED_DIM)
        config = {
            "patch_size": PATCH_SIZE,
            "embed_dim": EMBED_DIM,
            "in_channels": IN_CHANNELS,
            "use_bias": True,
            "bias_init": ZerosInitializer(),
        }
        input_shapes = {"img->patch_embed:in": (IMAGE_H, IMAGE_W, IN_CHANNELS)}
        activation = IdentityActivation()
        energy = GaussianEnergy()
        latent_init = NormalInitializer()
        weight_init = KaimingInitializer()

        k1, k2 = jax.random.split(key)
        params = PatchEmbeddingNode.initialize_params(
            key=k1,
            node_shape=node_shape,
            input_shapes=input_shapes,
            weight_init=weight_init,
            config=config,
        )
        node_info = _make_node_info(
            PatchEmbeddingNode,
            node_shape,
            config,
            activation,
            energy,
            latent_init,
            weight_init,
        )
        return params, node_info, k2

    def test_output_shape(self, setup_vit_base):
        """Forward pass produces (B, N, D) output."""
        params, node_info, key = setup_vit_base
        B = 2
        k1, k2 = jax.random.split(key)
        x = jax.random.normal(k1, (B, IMAGE_H, IMAGE_W, IN_CHANNELS))
        state = _make_state(k2, B, (NUM_PATCHES, EMBED_DIM))
        inputs = {"img->patch_embed:in": x}

        _, new_state = PatchEmbeddingNode.forward(params, inputs, state, node_info)

        assert new_state.z_mu.shape == (B, NUM_PATCHES, EMBED_DIM)
        assert new_state.error.shape == (B, NUM_PATCHES, EMBED_DIM)
        assert new_state.energy.shape == (B,)

    def test_output_shape_small(self):
        """Forward pass on small images (32×32, P=8)."""
        H, W, C, P, D = 32, 32, 3, 8, 64
        N = (H // P) * (W // P)  # 16
        B = 4

        key = jax.random.PRNGKey(0)
        node_shape = (N, D)
        config = {
            "patch_size": P,
            "embed_dim": D,
            "in_channels": C,
            "use_bias": True,
            "bias_init": ZerosInitializer(),
        }
        input_shapes = {"src->dst:in": (H, W, C)}

        k1, k2, k3 = jax.random.split(key, 3)
        params = PatchEmbeddingNode.initialize_params(
            key=k1,
            node_shape=node_shape,
            input_shapes=input_shapes,
            weight_init=KaimingInitializer(),
            config=config,
        )
        node_info = _make_node_info(
            PatchEmbeddingNode,
            node_shape,
            config,
            IdentityActivation(),
            GaussianEnergy(),
            NormalInitializer(),
            KaimingInitializer(),
        )
        x = jax.random.normal(k2, (B, H, W, C))
        state = _make_state(k3, B, node_shape)
        inputs = {"src->dst:in": x}

        _, new_state = PatchEmbeddingNode.forward(params, inputs, state, node_info)

        assert new_state.z_mu.shape == (B, N, D)

    def test_energy_is_nonnegative(self, setup_vit_base):
        """Energy should be non-negative (Gaussian energy)."""
        params, node_info, key = setup_vit_base
        B = 2
        k1, k2 = jax.random.split(key)
        x = jax.random.normal(k1, (B, IMAGE_H, IMAGE_W, IN_CHANNELS))
        state = _make_state(k2, B, (NUM_PATCHES, EMBED_DIM))
        inputs = {"img->patch_embed:in": x}

        _, new_state = PatchEmbeddingNode.forward(params, inputs, state, node_info)

        assert jnp.all(new_state.energy >= 0)

    def test_zero_input_produces_bias_only(self):
        """Zero image input should produce z_mu = bias (broadcast)."""
        H, W, C, P, D = 32, 32, 1, 8, 16
        N = (H // P) * (W // P)
        B = 1

        key = jax.random.PRNGKey(99)
        node_shape = (N, D)
        bias_val = 2.5
        config = {
            "patch_size": P,
            "embed_dim": D,
            "in_channels": C,
            "use_bias": True,
            "bias_init": OnesInitializer(gain=bias_val),
        }
        input_shapes = {"src->dst:in": (H, W, C)}

        k1, k2 = jax.random.split(key)
        params = PatchEmbeddingNode.initialize_params(
            key=k1,
            node_shape=node_shape,
            input_shapes=input_shapes,
            weight_init=KaimingInitializer(),
            config=config,
        )
        node_info = _make_node_info(
            PatchEmbeddingNode,
            node_shape,
            config,
            IdentityActivation(),
            GaussianEnergy(),
            NormalInitializer(),
            KaimingInitializer(),
        )

        x = jnp.zeros((B, H, W, C))
        state = _make_state(k2, B, node_shape)
        inputs = {"src->dst:in": x}

        _, new_state = PatchEmbeddingNode.forward(params, inputs, state, node_info)

        # With zero input, conv output is zero, so z_mu = 0 + bias
        expected_bias = params.biases["proj_bias"]
        # Each patch in the sequence should get the same bias vector
        assert jnp.allclose(new_state.z_mu[0, 0, :], expected_bias, atol=1e-5)
        assert jnp.allclose(new_state.z_mu[0, -1, :], expected_bias, atol=1e-5)

    def test_no_bias_forward(self):
        """Forward pass works without bias."""
        H, W, C, P, D = 32, 32, 3, 8, 32
        N = (H // P) * (W // P)
        B = 2

        key = jax.random.PRNGKey(7)
        node_shape = (N, D)
        config = {
            "patch_size": P,
            "embed_dim": D,
            "in_channels": C,
            "use_bias": False,
        }
        input_shapes = {"src->dst:in": (H, W, C)}

        k1, k2, k3 = jax.random.split(key, 3)
        params = PatchEmbeddingNode.initialize_params(
            key=k1,
            node_shape=node_shape,
            input_shapes=input_shapes,
            weight_init=KaimingInitializer(),
            config=config,
        )
        node_info = _make_node_info(
            PatchEmbeddingNode,
            node_shape,
            config,
            IdentityActivation(),
            GaussianEnergy(),
            NormalInitializer(),
            KaimingInitializer(),
        )
        x = jax.random.normal(k2, (B, H, W, C))
        state = _make_state(k3, B, node_shape)
        inputs = {"src->dst:in": x}

        _, new_state = PatchEmbeddingNode.forward(params, inputs, state, node_info)

        assert new_state.z_mu.shape == (B, N, D)
        assert "proj_bias" not in params.biases

    def test_sequence_order_matches_raster_scan(self):
        """Patches are ordered in raster-scan (row-major) order.

        For a 4×4 image with P=2, the 4 patches should correspond to:
        patch 0 = top-left, patch 1 = top-right,
        patch 2 = bottom-left, patch 3 = bottom-right.
        """
        H, W, C, P, D = 4, 4, 1, 2, 1
        N = (H // P) * (W // P)  # 4
        B = 1

        key = jax.random.PRNGKey(0)
        node_shape = (N, D)
        config = {
            "patch_size": P,
            "embed_dim": D,
            "in_channels": C,
            "use_bias": False,
        }
        input_shapes = {"src->dst:in": (H, W, C)}

        # Use identity-like kernel: sum all pixels in each patch
        # kernel shape = (P, P, C, D) = (2, 2, 1, 1)
        kernel = jnp.ones((P, P, C, D))
        params = NodeParams(
            weights={"proj_kernel": kernel},
            biases={},
        )

        node_info = _make_node_info(
            PatchEmbeddingNode,
            node_shape,
            config,
            IdentityActivation(),
            GaussianEnergy(),
            NormalInitializer(),
            KaimingInitializer(),
        )

        # Create a known image: each 2×2 quadrant has a distinct value
        # Top-left=1, Top-right=2, Bottom-left=3, Bottom-right=4
        img = jnp.array(
            [
                [
                    [[1.0], [1.0], [2.0], [2.0]],
                    [[1.0], [1.0], [2.0], [2.0]],
                    [[3.0], [3.0], [4.0], [4.0]],
                    [[3.0], [3.0], [4.0], [4.0]],
                ]
            ]
        )  # (1, 4, 4, 1)

        state = _make_state(key, B, node_shape)
        inputs = {"src->dst:in": img}

        _, new_state = PatchEmbeddingNode.forward(params, inputs, state, node_info)

        # Each patch sums 4 pixels (kernel=ones), so:
        # patch 0 (top-left):     4 × 1.0 = 4.0
        # patch 1 (top-right):    4 × 2.0 = 8.0
        # patch 2 (bottom-left):  4 × 3.0 = 12.0
        # patch 3 (bottom-right): 4 × 4.0 = 16.0
        z_mu = new_state.z_mu[0, :, 0]  # (N,)
        expected = jnp.array([4.0, 8.0, 12.0, 16.0])
        assert jnp.allclose(z_mu, expected, atol=1e-5), (
            f"Raster order mismatch: got {z_mu}, expected {expected}"
        )


class TestPatchEmbeddingAutodiff:
    """Verify JAX differentiability for PC inference."""

    def test_jit_forward(self):
        """Forward pass works under jax.jit."""
        H, W, C, P, D = 32, 32, 3, 8, 32
        N = (H // P) * (W // P)
        B = 2

        key = jax.random.PRNGKey(0)
        node_shape = (N, D)
        config = {
            "patch_size": P,
            "embed_dim": D,
            "in_channels": C,
            "use_bias": True,
            "bias_init": ZerosInitializer(),
        }
        input_shapes = {"src->dst:in": (H, W, C)}

        k1, k2, k3 = jax.random.split(key, 3)
        params = PatchEmbeddingNode.initialize_params(
            key=k1,
            node_shape=node_shape,
            input_shapes=input_shapes,
            weight_init=KaimingInitializer(),
            config=config,
        )
        node_info = _make_node_info(
            PatchEmbeddingNode,
            node_shape,
            config,
            IdentityActivation(),
            GaussianEnergy(),
            NormalInitializer(),
            KaimingInitializer(),
        )
        x = jax.random.normal(k2, (B, H, W, C))
        state = _make_state(k3, B, node_shape)
        inputs = {"src->dst:in": x}

        def forward_jit(params, inputs, state):
            _, new_state = PatchEmbeddingNode.forward(params, inputs, state, node_info)
            return new_state

        forward_jit = jax.jit(forward_jit)
        new_state = forward_jit(params, inputs, state)

        assert new_state.z_mu.shape == (B, N, D)

    def test_grad_wrt_params(self):
        """Gradients flow through the projection kernel."""
        H, W, C, P, D = 8, 8, 1, 4, 4
        N = (H // P) * (W // P)  # 4
        B = 1

        key = jax.random.PRNGKey(0)
        node_shape = (N, D)
        config = {
            "patch_size": P,
            "embed_dim": D,
            "in_channels": C,
            "use_bias": True,
            "bias_init": ZerosInitializer(),
        }
        input_shapes = {"src->dst:in": (H, W, C)}

        k1, k2, k3 = jax.random.split(key, 3)
        params = PatchEmbeddingNode.initialize_params(
            key=k1,
            node_shape=node_shape,
            input_shapes=input_shapes,
            weight_init=KaimingInitializer(),
            config=config,
        )
        node_info = _make_node_info(
            PatchEmbeddingNode,
            node_shape,
            config,
            IdentityActivation(),
            GaussianEnergy(),
            NormalInitializer(),
            KaimingInitializer(),
        )
        x = jax.random.normal(k2, (B, H, W, C))
        state = _make_state(k3, B, node_shape)
        inputs = {"src->dst:in": x}

        def loss_fn(p):
            energy, _s = PatchEmbeddingNode.forward(p, inputs, state, node_info)
            return energy

        grads = jax.grad(loss_fn)(params)

        # Kernel gradient should be non-zero (non-trivial input)
        assert grads.weights["proj_kernel"].shape == (P, P, C, D)
        assert jnp.any(grads.weights["proj_kernel"] != 0)

    def test_grad_wrt_input(self):
        """Gradients flow back through the input (for PC inference)."""
        H, W, C, P, D = 8, 8, 1, 4, 4
        N = (H // P) * (W // P)
        B = 1

        key = jax.random.PRNGKey(0)
        node_shape = (N, D)
        config = {
            "patch_size": P,
            "embed_dim": D,
            "in_channels": C,
            "use_bias": False,
        }
        input_shapes = {"src->dst:in": (H, W, C)}

        k1, k2, k3 = jax.random.split(key, 3)
        params = PatchEmbeddingNode.initialize_params(
            key=k1,
            node_shape=node_shape,
            input_shapes=input_shapes,
            weight_init=KaimingInitializer(),
            config=config,
        )
        node_info = _make_node_info(
            PatchEmbeddingNode,
            node_shape,
            config,
            IdentityActivation(),
            GaussianEnergy(),
            NormalInitializer(),
            KaimingInitializer(),
        )
        x = jax.random.normal(k2, (B, H, W, C))
        state = _make_state(k3, B, node_shape)

        def loss_fn(img):
            inp = {"src->dst:in": img}
            energy, _s = PatchEmbeddingNode.forward(params, inp, state, node_info)
            return energy

        grad_x = jax.grad(loss_fn)(x)

        assert grad_x.shape == x.shape
        assert jnp.any(grad_x != 0)


class TestPatchEmbeddingVariants:
    """Test different ViT model configurations."""

    @pytest.mark.parametrize(
        "img_h,img_w,patch_size,embed_dim,in_channels",
        [
            (224, 224, 16, 768, 3),  # ViT-Base/16
            (224, 224, 16, 1024, 3),  # ViT-Large/16
            (224, 224, 14, 1280, 3),  # ViT-Huge/14
            (384, 384, 16, 768, 3),  # ViT-Base/16 @ 384 resolution
            (32, 32, 4, 128, 3),  # Small ViT for CIFAR
            (28, 28, 7, 64, 1),  # Grayscale (MNIST-like)
        ],
    )
    def test_variant_shapes(self, img_h, img_w, patch_size, embed_dim, in_channels):
        """All standard ViT configurations produce correct output shapes."""
        N = (img_h // patch_size) * (img_w // patch_size)
        B = 1

        key = jax.random.PRNGKey(0)
        node_shape = (N, embed_dim)
        config = {
            "patch_size": patch_size,
            "embed_dim": embed_dim,
            "in_channels": in_channels,
            "use_bias": True,
            "bias_init": ZerosInitializer(),
        }
        input_shapes = {"src->dst:in": (img_h, img_w, in_channels)}

        k1, k2, k3 = jax.random.split(key, 3)
        params = PatchEmbeddingNode.initialize_params(
            key=k1,
            node_shape=node_shape,
            input_shapes=input_shapes,
            weight_init=KaimingInitializer(),
            config=config,
        )
        node_info = _make_node_info(
            PatchEmbeddingNode,
            node_shape,
            config,
            IdentityActivation(),
            GaussianEnergy(),
            NormalInitializer(),
            KaimingInitializer(),
        )
        x = jax.random.normal(k2, (B, img_h, img_w, in_channels))
        state = _make_state(k3, B, node_shape)
        inputs = {"src->dst:in": x}

        _, new_state = PatchEmbeddingNode.forward(params, inputs, state, node_info)

        assert new_state.z_mu.shape == (B, N, embed_dim)


# validation and integration coverage


@pytest.mark.parametrize(
    "arg_name,bad_value",
    [
        ("patch_size", 0),
        ("patch_size", -1),
        ("patch_size", 2.5),
        ("patch_size", True),
        ("embed_dim", 0),
        ("embed_dim", -1),
        ("in_channels", 0),
        ("in_channels", -1),
    ],
)
def test_dimension_arguments_must_be_positive_integers(arg_name, bad_value):
    """Public dimensions fail early instead of producing JAX shape errors."""
    kwargs = {
        "shape": (NUM_PATCHES, EMBED_DIM),
        "name": "bad",
        "patch_size": PATCH_SIZE,
        "grid_size": GRID_SIZE,
        "embed_dim": EMBED_DIM,
        "in_channels": IN_CHANNELS,
    }
    kwargs[arg_name] = bad_value
    if arg_name == "embed_dim":
        kwargs["shape"] = (NUM_PATCHES, bad_value)

    with pytest.raises(ValueError, match=f"{arg_name} must be a positive integer"):
        PatchEmbeddingNode(**kwargs)


@pytest.mark.parametrize("num_patches", [0, -1, 1.5, True])
def test_num_patches_must_be_a_positive_integer(num_patches):
    with pytest.raises(ValueError, match="number of patches.*positive integer"):
        PatchEmbeddingNode(
            shape=(num_patches, EMBED_DIM),
            name="bad",
            patch_size=PATCH_SIZE,
            grid_size=GRID_SIZE,
            embed_dim=EMBED_DIM,
            in_channels=IN_CHANNELS,
        )


def test_default_initializer_preserves_linear_fan_in_variance():
    """The identity-activation default uses std=1/sqrt(fan_in)."""
    node = PatchEmbeddingNode(
        shape=(4, 32),
        name="patch",
        patch_size=4,
        grid_size=(2, 2),
        embed_dim=32,
        in_channels=3,
    )
    params = PatchEmbeddingNode.initialize_params(
        key=jax.random.PRNGKey(0),
        node_shape=node.shape,
        input_shapes={"image->patch:in": (8, 8, 3)},
        weight_init=node._weight_init,
        config=node._extra_config,
    )
    kernel = params.weights["proj_kernel"]
    expected_std = 1.0 / jnp.sqrt(4 * 4 * 3)
    assert jnp.isclose(jnp.std(kernel), expected_std, rtol=0.2)


def test_matches_explicit_patch_projection_equation():
    """Conv implementation is equal to explicit flattened-patch x_p @ E."""
    batch_size, height, width, channels = 2, 6, 4, 2
    patch_size, embed_dim = 2, 3
    num_patches = (height // patch_size) * (width // patch_size)
    key_x, key_w, key_state = jax.random.split(jax.random.PRNGKey(12), 3)
    x = jax.random.normal(key_x, (batch_size, height, width, channels))
    kernel = jax.random.normal(key_w, (patch_size, patch_size, channels, embed_dim))
    bias = jnp.array([0.25, -0.5, 1.0])
    params = NodeParams(weights={"proj_kernel": kernel}, biases={"proj_bias": bias})
    config = {
        "patch_size": patch_size,
        "embed_dim": embed_dim,
        "in_channels": channels,
        "use_bias": True,
        "bias_init": ZerosInitializer(),
    }
    node_info = _make_node_info(
        PatchEmbeddingNode,
        (num_patches, embed_dim),
        config,
        IdentityActivation(),
        GaussianEnergy(),
        NormalInitializer(),
        KaimingInitializer(),
    )
    state = _make_state(key_state, batch_size, (num_patches, embed_dim))
    _, result = PatchEmbeddingNode.forward(
        params, {"image->patch:in": x}, state, node_info
    )
    actual = result.z_mu

    patches = []
    for row in range(0, height, patch_size):
        for col in range(0, width, patch_size):
            patch = x[:, row : row + patch_size, col : col + patch_size, :]
            patches.append(patch.reshape(batch_size, -1))
    patch_matrix = jnp.stack(patches, axis=1)
    expected = patch_matrix @ kernel.reshape(-1, embed_dim) + bias

    assert jnp.allclose(actual, expected, atol=1e-5)


# Non-square images, square-patch-only API, and property-based equivalence


def _explicit_patch_projection(x, kernel, patch_size):
    """Reference implementation of ViT Eq. 1: raster-order x_p^i @ E."""
    batch_size, height, width, _ = x.shape
    embed_dim = kernel.shape[-1]
    patches = [
        x[:, row : row + patch_size, col : col + patch_size, :].reshape(batch_size, -1)
        for row in range(0, height, patch_size)
        for col in range(0, width, patch_size)
    ]
    patch_matrix = jnp.stack(patches, axis=1)
    return patch_matrix @ kernel.reshape(-1, embed_dim)


def _forward_with_kernel(kernel, x, patch_size, embed_dim, in_channels, num_patches):
    """Run PatchEmbeddingNode.forward with a caller-supplied kernel, no bias."""
    config = {
        "patch_size": patch_size,
        "embed_dim": embed_dim,
        "in_channels": in_channels,
        "use_bias": False,
    }
    node_info = _make_node_info(
        PatchEmbeddingNode,
        (num_patches, embed_dim),
        config,
        IdentityActivation(),
        GaussianEnergy(),
        NormalInitializer(),
        KaimingInitializer(),
    )
    state = _make_state(jax.random.PRNGKey(0), x.shape[0], (num_patches, embed_dim))
    params = NodeParams(weights={"proj_kernel": kernel}, biases={})
    _, new_state = PatchEmbeddingNode.forward(
        params, {"src->dst:in": x}, state, node_info
    )
    return new_state.z_mu


def test_non_square_image_output_shape():
    """Non-square images (224x112) give N = (H/P)*(W/P) patches."""
    H, W, P, D, C, B = 224, 112, 16, 64, 3, 1
    N = (H // P) * (W // P)  # 14 * 7 = 98
    assert N == 98

    k1, k2 = jax.random.split(jax.random.PRNGKey(3))
    params = PatchEmbeddingNode.initialize_params(
        key=k1,
        node_shape=(N, D),
        input_shapes={"src->dst:in": (H, W, C)},
        weight_init=KaimingInitializer(),
        config={
            "patch_size": P,
            "embed_dim": D,
            "in_channels": C,
            "use_bias": False,
        },
    )
    x = jax.random.normal(k2, (B, H, W, C))
    z_mu = _forward_with_kernel(params.weights["proj_kernel"], x, P, D, C, N)

    assert z_mu.shape == (B, N, D)
    expected = _explicit_patch_projection(x, params.weights["proj_kernel"], P)
    assert jnp.allclose(z_mu, expected, atol=1e-4)


def test_non_square_image_raster_order_is_width_major():
    """On a tall image the sequence advances along width first, then height.

    A 6x4 image with P=2 has a 3x2 patch grid. Labelling each patch with its
    raster index means patch i must read back as exactly i.
    """
    H, W, P, C, D = 6, 4, 2, 1, 1
    grid_h, grid_w = H // P, W // P
    N = grid_h * grid_w  # 6

    # Pixel value = raster index of the patch it belongs to.
    img = jnp.array(
        [[[float((r // P) * grid_w + (c // P))] for c in range(W)] for r in range(H)]
    )[None]  # (1, 6, 4, 1)

    # Kernel of ones averages -> sum = P*P * patch_index.
    kernel = jnp.ones((P, P, C, D))
    z_mu = _forward_with_kernel(kernel, img, P, D, C, N)

    expected = jnp.arange(N, dtype=jnp.float32) * (P * P)
    assert jnp.allclose(z_mu[0, :, 0], expected), (
        f"Non-square raster order mismatch: got {z_mu[0, :, 0]}, expected {expected}"
    )


def test_non_square_patch_size_is_rejected():
    """Only square patches are supported: a (Ph, Pw) tuple must fail early."""
    with pytest.raises(ValueError, match="patch_size must be a positive integer"):
        PatchEmbeddingNode(
            shape=(NUM_PATCHES, EMBED_DIM),
            name="bad",
            patch_size=(16, 8),  # non-square — unsupported
            grid_size=GRID_SIZE,
            embed_dim=EMBED_DIM,
            in_channels=IN_CHANNELS,
        )


def test_forward_rejects_image_inconsistent_with_declared_shape():
    """A wrong-sized image raises instead of silently broadcasting.

    Declared N=1 with a 4x4 image (P=2, actual N=4) would broadcast to N=4
    at the ``z_latent - z_mu`` step if the reshape used ``-1``.
    """
    P, C, D = 2, 1, 1
    x = jnp.ones((1, 4, 4, C))
    kernel = jnp.ones((P, P, C, D))

    with pytest.raises(TypeError, match="reshape"):
        _forward_with_kernel(kernel, x, P, D, C, num_patches=1)


@settings(deadline=None, max_examples=25)
@given(
    patch_size=st.integers(min_value=1, max_value=4),
    grid_h=st.integers(min_value=1, max_value=4),
    grid_w=st.integers(min_value=1, max_value=4),
    in_channels=st.integers(min_value=1, max_value=3),
    embed_dim=st.integers(min_value=1, max_value=6),
    batch_size=st.integers(min_value=1, max_value=3),
    seed=st.integers(min_value=0, max_value=2**16),
)
def test_conv_projection_equals_explicit_equation_property(
    patch_size, grid_h, grid_w, in_channels, embed_dim, batch_size, seed
):
    """For any valid config, the conv equals raster-order x_p^i @ E (ViT Eq. 1)."""
    height, width = grid_h * patch_size, grid_w * patch_size
    num_patches = grid_h * grid_w

    key_x, key_w = jax.random.split(jax.random.PRNGKey(seed))
    x = jax.random.normal(key_x, (batch_size, height, width, in_channels))
    kernel = jax.random.normal(key_w, (patch_size, patch_size, in_channels, embed_dim))

    z_mu = _forward_with_kernel(
        kernel, x, patch_size, embed_dim, in_channels, num_patches
    )

    assert z_mu.shape == (batch_size, num_patches, embed_dim)
    expected = _explicit_patch_projection(x, kernel, patch_size)
    assert jnp.allclose(z_mu, expected, atol=1e-4)


@settings(deadline=None, max_examples=25)
@given(
    patch_size=st.integers(min_value=1, max_value=4),
    grid_h=st.integers(min_value=1, max_value=4),
    grid_w=st.integers(min_value=1, max_value=4),
    in_channels=st.integers(min_value=1, max_value=3),
    embed_dim=st.integers(min_value=1, max_value=6),
)
def test_initialize_params_accepts_any_valid_config_property(
    patch_size, grid_h, grid_w, in_channels, embed_dim
):
    """Any divisible image/patch pairing initializes to kernel (P, P, C, D)."""
    height, width = grid_h * patch_size, grid_w * patch_size
    num_patches = grid_h * grid_w

    node = PatchEmbeddingNode(
        shape=(num_patches, embed_dim),
        name="patch",
        patch_size=patch_size,
        grid_size=(grid_h, grid_w),
        embed_dim=embed_dim,
        in_channels=in_channels,
    )
    params = PatchEmbeddingNode.initialize_params(
        key=jax.random.PRNGKey(0),
        node_shape=node.shape,
        input_shapes={"src->dst:in": (height, width, in_channels)},
        weight_init=node._weight_init,
        config=node._extra_config,
    )

    assert params.weights["proj_kernel"].shape == (
        patch_size,
        patch_size,
        in_channels,
        embed_dim,
    )
    assert params.biases["proj_bias"].shape == (embed_dim,)
    assert jnp.all(jnp.isfinite(params.weights["proj_kernel"]))


def test_kernel_and_bias_use_independent_prng_keys():
    """The kernel and bias must not be drawn from the same key."""
    config = {
        "patch_size": 2,
        "embed_dim": 8,
        "in_channels": 2,
        "use_bias": True,
        # A non-zero bias init is required to observe key reuse at all.
        "bias_init": NormalInitializer(std=1.0),
    }
    params = PatchEmbeddingNode.initialize_params(
        key=jax.random.PRNGKey(0),
        node_shape=(4, 8),
        input_shapes={"src->dst:in": (4, 4, 2)},
        weight_init=NormalInitializer(std=1.0),
        config=config,
    )
    # If one key fed both draws, bias would equal the first 8 kernel elements.
    kernel_head = params.weights["proj_kernel"].reshape(-1)[:8]
    assert not jnp.allclose(kernel_head, params.biases["proj_bias"])


def _build_patch_graph(with_mupc=False):
    image = IdentityNode(shape=(8, 8, 1), name="image")
    patch = PatchEmbeddingNode(
        shape=(4, 8),
        name="patch",
        patch_size=4,
        grid_size=(2, 2),
        embed_dim=8,
        in_channels=1,
    )
    output = Linear(shape=(2,), name="output", flatten_input=True)
    return graph(
        nodes=[image, patch, output],
        edges=[
            Edge(source=image, target=patch.slot("in")),
            Edge(source=patch, target=output.slot("in")),
        ],
        task_map=TaskMap(x=image, y=output),
        inference=InferenceSGD(eta_infer=0.05, infer_steps=3),
        scaling=MuPCConfig() if with_mupc else None,
    )


def test_real_graph_initialization_and_feedforward_state(rng_key):
    """The public node works through graph, parameter, and state initialization."""
    structure = _build_patch_graph()
    params = initialize_params(structure, rng_key)
    x = jax.random.normal(rng_key, (2, 8, 8, 1))
    y = jnp.zeros((2, 2))
    state = initialize_graph_state(
        structure,
        batch_size=2,
        rng_key=rng_key,
        clamps={"image": x, "output": y},
        params=params,
    )

    assert params.nodes["patch"].weights["proj_kernel"].shape == (4, 4, 1, 8)
    assert state.nodes["patch"].z_mu.shape == (2, 4, 8)
    assert jnp.all(jnp.isfinite(state.nodes["patch"].z_mu))


def test_fabricpc_local_gradient_apis(rng_key):
    """Base local-gradient APIs produce finite input, self, and parameter grads."""
    structure = _build_patch_graph()
    params = initialize_params(structure, rng_key)
    x = jax.random.normal(rng_key, (2, 8, 8, 1))
    state = initialize_graph_state(
        structure,
        batch_size=2,
        rng_key=rng_key,
        clamps={"image": x, "output": jnp.zeros((2, 2))},
        params=params,
    )
    patch_info = structure.nodes["patch"].node_info
    edge_key = patch_info.in_edges[0]
    inputs = {edge_key: x}
    patch_state = state.nodes["patch"]._replace(
        z_latent=state.nodes["patch"].z_latent + 0.25
    )

    _, input_grads, self_grad = PatchEmbeddingNode.forward_and_latent_grads(
        params.nodes["patch"],
        inputs,
        patch_state,
        patch_info,
        is_clamped=False,
    )
    _, param_grads = PatchEmbeddingNode.forward_and_weight_grads(
        params.nodes["patch"], inputs, patch_state, patch_info
    )

    assert input_grads[edge_key].shape == x.shape
    assert jnp.any(input_grads[edge_key] != 0)
    assert jnp.any(self_grad != 0)
    assert jnp.any(param_grads.weights["proj_kernel"] != 0)
    assert jnp.all(jnp.isfinite(param_grads.weights["proj_kernel"]))


def test_mupc_scaling_uses_patch_receptive_field():
    """Graph muPC attaches a=1/sqrt(P*P*C) to the patch input edge."""
    structure = _build_patch_graph(with_mupc=True)
    patch_info = structure.nodes["patch"].node_info
    edge_key = patch_info.in_edges[0]
    scaling = patch_info.scaling_config

    assert scaling is not None
    assert scaling.forward_scale[edge_key] == pytest.approx(1.0 / 4.0)


def test_weight_init_none_falls_back_to_default():
    "weight_init=None uses the node default instaead of crashing."
    node = PatchEmbeddingNode(
        shape=(4, 8),
        name="patch",
        patch_size=4,
        grid_size=(2, 2),
        embed_dim=8,
        in_channels=1,
        weight_init=None,
    )
    params = PatchEmbeddingNode.initialize_params(
        key=jax.random.PRNGKey(0),
        node_shape=node.shape,
        input_shapes={"src->dst:in": (8, 8, 1)},
        weight_init=None,
        config=node._extra_config,
    )

    assert params.weights["proj_kernel"].shape == (4,4,1,8)
