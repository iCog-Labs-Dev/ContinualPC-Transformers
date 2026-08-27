"""Position-aware initial ViT tokens.

Implements the positional embedding term:

    z_0 = [x_class; x_p^1 E; …; x_p^N E] + E_pos

    μ_0 = concat(c, z_patch) + P_pos

    c ∈ ℝ^{K×D},   z_patch ∈ ℝ^{N×D},   P_pos ∈ ℝ^{(N+K)×D}

K is ``num_prefix_tokens`` (normally 1). The independent latent is
z_0 ∈ ℝ^{(N+K)×D}. Patchify/concat stay inside this prediction.

Position conditioning earns its own PC latent because it changes every
token from content-only to content-plus-location. One local error
ε_0 = z_0 - μ_0 drives all three local learning rules:

    ∂E_0/∂P_pos  = -π_0 · ε_0
    ∂E_0/∂c      = -π_0 · ε_0[CLS rows]
    ∂E_0/∂z_patch = -π_0 · ε_0[patch rows]

so the CLS parameter learns from the CLS slice of the error alone, with no
separate node. These are obtained by autodiff here, not written out; the
tests assert them numerically.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

import jax
import jax.numpy as jnp

from fabricpc.core.activations import IdentityActivation
from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.initializers import (
    InitializerBase,
    NormalInitializer,
    initialize,
)
from fabricpc.core.types import NodeInfo, NodeParams, NodeState
from fabricpc.nodes.base import NodeBase, SlotSpec

if TYPE_CHECKING:
    from fabricpc.core.energy import EnergyFunctional


def _validate_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def _validate_grid_size(name: str, grid_size: Tuple[int, int]) -> None:
    if not isinstance(grid_size, tuple) or len(grid_size) != 2:
        raise ValueError(f"{name} must be a (height, width) tuple, got {grid_size!r}")
    _validate_positive_int(f"{name}[0]", grid_size[0])
    _validate_positive_int(f"{name}[1]", grid_size[1])


def interpolate_position_embeddings(
    position_embeddings: jnp.ndarray,
    source_grid_size: Tuple[int, int],
    target_grid_size: Tuple[int, int],
    *,
    num_prefix_tokens: int = 1,
    method: str = "bicubic",
) -> jnp.ndarray:
    """Resize patch-position embeddings while preserving prefix positions.

    Args:
        position_embeddings: Learned positions with shape (T, D), where
            T = num_prefix_tokens + source_height * source_width.
        source_grid_size: Source patch grid (height, width).
        target_grid_size: Target patch grid (height, width).
        num_prefix_tokens: Leading non-spatial positions, normally one CLS
            position. These values are copied without interpolation.
        method: Interpolation method accepted by jax.image.resize.

    Returns:
        Position embeddings with shape
        (num_prefix_tokens + target_height * target_width, D).
    """
    _validate_grid_size("source_grid_size", source_grid_size)
    _validate_grid_size("target_grid_size", target_grid_size)
    if (
        isinstance(num_prefix_tokens, bool)
        or not isinstance(num_prefix_tokens, int)
        or num_prefix_tokens < 0
    ):
        raise ValueError(
            "num_prefix_tokens must be a non-negative integer, "
            f"got {num_prefix_tokens!r}"
        )
    if position_embeddings.ndim != 2:
        raise ValueError(
            "position_embeddings must have shape (sequence_length, embed_dim), "
            f"got {position_embeddings.shape}"
        )
    if not isinstance(method, str) or not method:
        raise ValueError(f"method must be a non-empty string, got {method!r}")

    source_height, source_width = source_grid_size
    target_height, target_width = target_grid_size
    expected_tokens = num_prefix_tokens + source_height * source_width
    if position_embeddings.shape[0] != expected_tokens:
        raise ValueError(
            "position_embeddings sequence length does not match the source "
            f"grid: got {position_embeddings.shape[0]}, expected "
            f"{num_prefix_tokens} + {source_height} * {source_width} = "
            f"{expected_tokens}"
        )

    if source_grid_size == target_grid_size:
        return position_embeddings

    prefix_positions = position_embeddings[:num_prefix_tokens]
    patch_positions = position_embeddings[num_prefix_tokens:]
    embed_dim = patch_positions.shape[-1]
    patch_grid = patch_positions.reshape(source_height, source_width, embed_dim)
    resized_grid = jax.image.resize(
        patch_grid,
        (target_height, target_width, embed_dim),
        method=method,
    )
    resized_patches = resized_grid.reshape(target_height * target_width, embed_dim)
    return jnp.concatenate([prefix_positions, resized_patches], axis=0)


class LearnedPositionEmbeddingNode(NodeBase):
    """Predict the position-aware ViT sequence from content-only patch tokens.

    Input shape excluding batch: ``(N, D)`` — patch embeddings from
    ``PatchEmbeddingNode``. Output shape: ``(N + K, D)`` where
    ``K = num_prefix_tokens``.

    Parameters
    cls_token:
        Learned prefix tokens ``c``, shape ``(K, D)``. Absent when ``K = 0``.
    position_embeddings:
        Learned 1-D table ``P_pos``, shape ``(N + K, D)``.
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        name: str,
        grid_size: Tuple[int, int],
        num_prefix_tokens: int = 1,
        energy: "EnergyFunctional" = GaussianEnergy(),
        weight_init: InitializerBase = NormalInitializer(std=0.02),
        latent_init: InitializerBase = NormalInitializer(),
    ):
        if len(shape) != 2:
            raise ValueError(
                f"LearnedPositionEmbeddingNode {name!r}: shape must be "
                f"(sequence_length, embed_dim), got {shape}"
            )
        for axis, value in enumerate(shape):
            _validate_positive_int(f"shape[{axis}]", value)
        _validate_grid_size("grid_size", grid_size)
        if (
            isinstance(num_prefix_tokens, bool)
            or not isinstance(num_prefix_tokens, int)
            or num_prefix_tokens < 0
        ):
            raise ValueError(
                "num_prefix_tokens must be a non-negative integer, "
                f"got {num_prefix_tokens!r}"
            )

        expected_sequence_length = num_prefix_tokens + grid_size[0] * grid_size[1]
        if shape[0] != expected_sequence_length:
            raise ValueError(
                f"LearnedPositionEmbeddingNode {name!r}: shape[0]={shape[0]} "
                "must equal num_prefix_tokens + grid height * grid width = "
                f"{num_prefix_tokens} + {grid_size[0]} * {grid_size[1]} = "
                f"{expected_sequence_length}"
            )

        super().__init__(
            shape=shape,
            name=name,
            activation=IdentityActivation(),
            energy=energy,
            latent_init=latent_init,
            weight_init=weight_init,
            grid_size=grid_size,
            num_prefix_tokens=num_prefix_tokens,
        )

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        """Unscaled slot: concat(c, z_patch) + P must not become concat(c, a z_patch) + P."""
        return {
            "in": SlotSpec(
                name="in",
                is_multi_input=False,
                is_variance_scalable=False,
            )
        }

    @staticmethod
    def get_weight_fan_in(source_shape: Tuple[int, ...], config: Dict[str, Any]) -> int:
        """Return one: patch tokens are copied, not projected by a weight matrix."""
        return 1

    @staticmethod
    def initialize_params(
        key: jax.Array,
        node_shape: Tuple[int, ...],
        input_shapes: Dict[str, Tuple[int, ...]],
        weight_init: Optional[InitializerBase],
        config: Optional[Dict[str, Any]] = None,
    ) -> NodeParams:
        if len(input_shapes) != 1:
            raise ValueError(
                "LearnedPositionEmbeddingNode requires exactly one input shape, "
                f"got {len(input_shapes)}"
            )
        if config is None:
            config = {}

        num_prefix_tokens = config.get("num_prefix_tokens", 1)
        input_shape = next(iter(input_shapes.values()))
        if len(input_shape) != 2:
            raise ValueError(
                "LearnedPositionEmbeddingNode expected 2-D patch input "
                f"(num_patches, embed_dim), got {input_shape}"
            )
        if input_shape[-1] != node_shape[-1]:
            raise ValueError(
                "LearnedPositionEmbeddingNode embed_dim mismatch: "
                f"input D={input_shape[-1]}, output D={node_shape[-1]}"
            )
        expected_patches = node_shape[0] - num_prefix_tokens
        if expected_patches <= 0:
            raise ValueError(
                "LearnedPositionEmbeddingNode output length "
                f"{node_shape[0]} must exceed num_prefix_tokens="
                f"{num_prefix_tokens}"
            )
        if input_shape[0] != expected_patches:
            raise ValueError(
                "LearnedPositionEmbeddingNode expected "
                f"(N, D)=({expected_patches}, {node_shape[-1]}) patch input "
                f"(output length {node_shape[0]} minus "
                f"num_prefix_tokens={num_prefix_tokens}), got {input_shape}"
            )

        if num_prefix_tokens > 0:
            key_pos, key_cls = jax.random.split(key)
            weights = {
                "position_embeddings": initialize(key_pos, node_shape, weight_init),
                "cls_token": initialize(
                    key_cls,
                    (num_prefix_tokens, node_shape[-1]),
                    weight_init,
                ),
            }
        else:
            weights = {
                "position_embeddings": initialize(key, node_shape, weight_init),
            }
        return NodeParams(weights=weights, biases={})

    @staticmethod
    def interpolate_params(
        params: NodeParams,
        source_grid_size: Tuple[int, int],
        target_grid_size: Tuple[int, int],
        *,
        num_prefix_tokens: int = 1,
        method: str = "bicubic",
    ) -> NodeParams:
        """Resize P_pos for a new patch grid; leave cls_token unchanged.

        Returns new params only — it cannot change the node's declared shape.
        Resolution transfer therefore takes two steps: rebuild the graph with a
        node whose ``grid_size``/``shape`` match ``target_grid_size``, then load
        these resized params into it. Feeding resized params to a node still
        declaring the source grid raises in ``forward``.

        ``num_prefix_tokens`` must match the value the table was built with; a
        wrong value is caught by the sequence-length check in
        ``interpolate_position_embeddings`` rather than silently mis-slicing the
        prefix rows.
        """
        resized = interpolate_position_embeddings(
            params.weights["position_embeddings"],
            source_grid_size,
            target_grid_size,
            num_prefix_tokens=num_prefix_tokens,
            method=method,
        )
        return params._replace(
            weights={**params.weights, "position_embeddings": resized}
        )

    @staticmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[jax.Array, NodeState]:
        """ViT Eq. 1: μ_0 = concat(c, z_patch) + P_pos.

        ``pre_activation`` is set to ``z_mu``, per the ``NodeBase.forward``
        contract for nodes that apply no activation. It must stay the *input to
        the activation function* — that is the sole functional consumer
        (``LinearExplicitGrad`` evaluates ``activation.derivative`` at it). The
        pre-position value ``concat(c, z_patch)`` would be the wrong point to
        evaluate f' at, so it is deliberately not recorded here.
        """
        z_patch = inputs[next(iter(inputs))]
        num_prefix_tokens = node_info.node_config["num_prefix_tokens"]
        position_embeddings = params.weights["position_embeddings"]
        if num_prefix_tokens > 0:
            cls_token = params.weights["cls_token"]
            batch_size = z_patch.shape[0]
            # A broadcast view, not a tile: autodiff sums the CLS cotangent
            # over the batch, which is exactly the shared-parameter rule.
            cls_tokens = jnp.broadcast_to(
                cls_token[jnp.newaxis, :, :],
                (batch_size, num_prefix_tokens, z_patch.shape[-1]),
            )
            tokens = jnp.concatenate([cls_tokens, z_patch], axis=1)
        else:
            tokens = z_patch

        # The declared sequence length is authoritative. Without this check a
        # single-token input would broadcast silently against the (T, D) table
        # instead of failing, yielding a wrong-length z_mu.
        if tokens.shape[1] != node_info.shape[0]:
            raise ValueError(
                f"LearnedPositionEmbeddingNode {node_info.name!r}: patch input "
                f"of length {z_patch.shape[1]} plus num_prefix_tokens="
                f"{num_prefix_tokens} gives {tokens.shape[1]} tokens, but the "
                f"node declares sequence length {node_info.shape[0]}"
            )

        z_mu_pre = tokens + position_embeddings[jnp.newaxis, :, :]
        activation = node_info.activation
        z_mu = type(activation).forward(z_mu_pre, activation.config)
        error = state.z_latent - z_mu
        state = state._replace(pre_activation=z_mu_pre, z_mu=z_mu, error=error)
        state = node_info.node_class.energy_functional(state, node_info)
        return jnp.sum(state.energy), state
