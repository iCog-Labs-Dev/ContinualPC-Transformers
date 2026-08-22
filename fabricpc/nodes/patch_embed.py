"""
Patch embedding node for Vision Transformer (ViT) predictive coding graphs.

Implements the patch-projection term of ViT Equation 1

    z_0 = [x_class; x_p^1 E; x_p^2 E; …; x_p^N E] + E_pos

    x_p^i ∈ ℝ^{P²·C},   E ∈ ℝ^{(P²·C) × D},   x_p^i · E ∈ ℝ^D

The node extracts non-overlapping P×P patches from an input image, projects
each patch into an embedding dimension D, and flattens the 2-D patch grid
into a 1-D sequence:

    (B, H, W, C) → extract P×P patches → project to D → (B, N, D)

where N = (H·W) / P² = (H/P)·(W/P).

Internally the linear projection is performed with a 2-D convolution whose
kernel size and stride both equal the patch size P.  This is mathematically
equivalent to flattening each patch and multiplying by a dense matrix E, but
avoids the explicit reshape and is the standard implementation in all major
ViT code-bases (google/vit, timm, etc.).  The equivalence holds because a
stride-P, VALID-padded P×P kernel sees exactly the disjoint patch grid, and
the HWIO kernel layout ``(P, P, C, D)`` reshaped to ``(P²·C, D)`` is E.
``tests/test_patch_embed.py::test_matches_explicit_patch_projection_equation``
asserts this against an explicit ``x_p @ E`` reference.

Patch order is raster-scan (row-major): patch index i = row·(W/P) + col.
NHWC convolution output ``(B, H/P, W/P, D)`` is already row-major over
(row, col), so flattening the two spatial axes needs no transpose — the
reshape is a metadata-only view of contiguous memory.  Note that P is a
single integer: only square patches are supported, matching ViT.

Usage
-----
    patch_node = PatchEmbeddingNode(
        shape=(num_patches, embed_dim),   # (N, D) = (H*W/P², D)
        name="patch_embed",
        patch_size=16,
        embed_dim=768,
        in_channels=3,
    )
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple, TYPE_CHECKING
import jax
import jax.lax as lax
import jax.numpy as jnp

from fabricpc.nodes.base import NodeBase, SlotSpec
from fabricpc.core.types import NodeParams, NodeState, NodeInfo
from fabricpc.core.activations import IdentityActivation
from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.initializers import (
    KaimingInitializer,
    ZerosInitializer,
    NormalInitializer,
    InitializerBase,
    initialize,
)

if TYPE_CHECKING:
    from fabricpc.core.activations import ActivationBase
    from fabricpc.core.energy import EnergyFunctional


class PatchEmbeddingNode(NodeBase):
    """Extract non-overlapping image patches and project to embedding space.

    Implements the patch-projection step of ViT Equation 1:

        x_p^i · E,   i = 1 … N

    via a Conv2D with ``kernel_size = stride = (P, P)`` and
    ``out_channels = D``, followed by a spatial-flatten to sequence order.

    Input shape:  ``(B, H, W, C)``  — batch of images in NHWC format.
    Output shape: ``(B, N, D)``     — flat sequence of patch embeddings,
                                      where ``N = (H * W) / P²``.

    Args:
        shape:       ``(N, D)`` — output shape excluding batch.
                     Must satisfy ``N = (H * W) / patch_size²`` where H, W
                     are the spatial dimensions of the input image.
        name:        Node name.
        patch_size:  Side length P of each square patch. Only square patches
                     are supported (ViT uses square patches); a non-integer
                     value such as a ``(Ph, Pw)`` tuple is rejected.
        embed_dim:   Embedding dimension D.
        in_channels: Number of input image channels C (e.g. 3 for RGB).
        activation:  Default: IdentityActivation. ViT Eq. 1 applies no
                     nonlinearity to the patch projection, so identity is the
                     paper-faithful choice; a different activation is accepted
                     but departs from the paper.
        energy:      Default: GaussianEnergy.
        use_bias:    Whether to include a learnable bias. Default: True,
                     matching the ``Conv`` patch projection in google/vit,
                     which is bias-enabled.
        weight_init: Initializer for the projection kernel. The default gives
                     ``std = 1 / sqrt(fan_in)`` with
                     ``fan_in = P² · C_in``, i.e. LeCun-normal scaling, which
                     is the variance-preserving choice for the default
                     identity activation and matches google/vit's
                     ``lecun_normal`` patch-projection init. It is spelled as
                     ``KaimingInitializer(gain=2**-0.5)`` because
                     ``KaimingInitializer`` bakes in the ReLU gain ``sqrt(2)``;
                     the ``2**-0.5`` cancels it exactly.
        bias_init:   Initializer for the bias. Default: ZerosInitializer —
                     zero bias is the Flax ``Conv`` default used by google/vit,
                     and is correct here because the bias is a free additive
                     offset with no symmetry to break.
        latent_init: Initializer for latent states.
                     Default: NormalInitializer.
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        name: str,
        patch_size: int,
        embed_dim: int,
        in_channels: int,
        activation: "ActivationBase" = IdentityActivation(),
        energy: "EnergyFunctional" = GaussianEnergy(),
        use_bias: bool = True,
        weight_init: InitializerBase = KaimingInitializer(gain=2**-0.5),
        bias_init: InitializerBase = ZerosInitializer(),
        latent_init: InitializerBase = NormalInitializer(),
    ):
        for arg_name, value in (
            ("patch_size", patch_size),
            ("embed_dim", embed_dim),
            ("in_channels", in_channels),
        ):
            # bool is an int subclass, but accepting True as a dimension leads
            # to opaque shape failures later during graph initialization.
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(
                    f"PatchEmbeddingNode {name}: {arg_name} must be a "
                    f"positive integer, got {value!r}"
                )

        # Validate output shape: must be (N, D).
        if len(shape) != 2:
            raise ValueError(
                f"PatchEmbeddingNode '{name}': output shape must be (N, D) "
                f"with exactly 2 dimensions, got shape={shape} "
                f"({len(shape)} dims)"
            )
        if shape[1] != embed_dim:
            raise ValueError(
                f"PatchEmbeddingNode '{name}': shape[-1]={shape[1]} must "
                f"equal embed_dim={embed_dim}"
            )
        if isinstance(shape[0], bool) or not isinstance(shape[0], int) or shape[0] <= 0:
            raise ValueError(
                f"PatchEmbeddingNode {name}: shape[0] (number of patches) "
                f"must be a positive integer, got {shape[0]!r}"
            )

        super().__init__(
            shape=shape,
            name=name,
            activation=activation,
            energy=energy,
            latent_init=latent_init,
            weight_init=weight_init,
            patch_size=patch_size,
            embed_dim=embed_dim,
            in_channels=in_channels,
            use_bias=use_bias,
            bias_init=bias_init,
        )

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        """Single input slot for the image tensor."""
        return {"in": SlotSpec(name="in", is_multi_input=False)}

    @staticmethod
    def get_weight_fan_in(source_shape: Tuple[int, ...], config: Dict[str, Any]) -> int:
        """fan_in = C_in × P × P (receptive-field size of each output unit)."""
        patch_size = config["patch_size"]
        return source_shape[-1] * patch_size * patch_size

    @staticmethod
    def initialize_params(
        key: jax.Array,
        node_shape: Tuple[int, ...],
        input_shapes: Dict[str, Tuple[int, ...]],
        weight_init: Optional[InitializerBase],
        config: Optional[Dict[str, Any]] = None,
    ) -> NodeParams:
        """Initialise the projection kernel and optional bias.

        Kernel shape: ``(P, P, C_in, D)``  — HWIO layout for
        ``lax.conv_general_dilated`` with dimension numbers ``("NHWC","HWIO","NHWC")``.

        Bias shape: ``(D,)`` when ``use_bias=True``.
        """
        if config is None:
            config = {}

        patch_size = config["patch_size"]
        embed_dim = config["embed_dim"]
        in_channels = config["in_channels"]
        use_bias = config.get("use_bias", True)
        bias_init = config.get("bias_init")

        if use_bias and bias_init is None:
            raise ValueError(
                "PatchEmbeddingNode: use_bias=True but bias_init is None. "
                "Pass a bias initializer (e.g. ZerosInitializer()) or set "
                "use_bias=False."
            )

        # Validate that the input image dimensions are compatible with the
        # declared output sequence length N.
        for edge_key, in_shape in input_shapes.items():
            if len(in_shape) != 3:
                raise ValueError(
                    f"PatchEmbeddingNode: expected 3-D input shape (H, W, C) "
                    f"excluding batch, got {in_shape} for edge '{edge_key}'"
                )
            H, W, C = in_shape
            if C != in_channels:
                raise ValueError(
                    f"PatchEmbeddingNode: input channels {C} from edge "
                    f"'{edge_key}' != declared in_channels={in_channels}"
                )
            if H % patch_size != 0 or W % patch_size != 0:
                raise ValueError(
                    f"PatchEmbeddingNode: image size ({H}, {W}) is not evenly "
                    f"divisible by patch_size={patch_size}"
                )
            expected_num_patches = (H // patch_size) * (W // patch_size)
            declared_num_patches = node_shape[0]
            if expected_num_patches != declared_num_patches:
                raise ValueError(
                    f"PatchEmbeddingNode: declared N={declared_num_patches} "
                    f"but image ({H}, {W}) with patch_size={patch_size} "
                    f"yields N={expected_num_patches}"
                )

        n_keys = 2 if use_bias else 1
        keys = jax.random.split(key, n_keys)

        # Conv kernel: (P, P, C_in, D)
        kernel_shape = (patch_size, patch_size, in_channels, embed_dim)
        weights = {"proj_kernel": initialize(keys[0], kernel_shape, weight_init)}

        biases = {}
        if use_bias:
            biases["proj_bias"] = initialize(keys[1], (embed_dim,), bias_init)

        return NodeParams(weights=weights, biases=biases)

    @staticmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[jax.Array, NodeState]:
        """Patch-project and flatten to sequence.

        Steps:
        1. Conv2D with kernel=P, stride=P  →  (B, H/P, W/P, D)
        2. Reshape spatial grid to flat sequence  →  (B, N, D)
        3. Optional bias addition
        4. Activation (default identity — linear projection)
        5. Compute prediction error and energy
        """
        config = node_info.node_config
        patch_size = config["patch_size"]

        # Get the single input image
        x = inputs[next(iter(inputs))]  # (B, H, W, C)

        # 1. Conv2D projection (non-overlapping patches)
        proj = lax.conv_general_dilated(
            lhs=x,
            rhs=params.weights["proj_kernel"],
            window_strides=(patch_size, patch_size),
            padding="VALID",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )  # (B, H/P, W/P, D)

        # 2. Flatten spatial grid → sequence (raster order, no transpose)
        # Reshaping to the declared (N, D) rather than (-1, D) keeps the node
        # shape authoritative: a wrong-sized image raises here instead of
        # silently broadcasting against z_latent at the error step below
        B = proj.shape[0]
        z = proj.reshape(B, *node_info.shape)  # (B, N, D)

        # 3. Bias
        if "proj_bias" in params.biases:
            z = z + params.biases["proj_bias"]  # broadcast (D,) over (B, N, D)

        # 4. Activation (identity by default for linear projection)
        act_obj = node_info.activation
        z_mu = type(act_obj).forward(z, act_obj.config)

        # 5. Error and energy (NodeBase steps 3–6)
        error = state.z_latent - z_mu
        state = state._replace(pre_activation=z, z_mu=z_mu, error=error)
        state = node_info.node_class.energy_functional(state, node_info)
        return jnp.sum(state.energy), state