"""Class-token readout for Vision Transformer predictive coding graphs.

Implements ViT Equation:

    y = LN(z_L^0)

The final encoder latent is a sequence ``z_L ∈ ℝ^{S×D}``. This node selects
the CLS token at position ``cls_index`` (default 0) and applies LayerNorm,
producing the image-level representation ``z_r ∈ ℝ^D``.

Token selection and LayerNorm share one node because LayerNorm is
channel-wise per token, so ``[LN(Z)]_0 = LN(Z_0)``. Applying LN to the
selected CLS token avoids an unused full-sequence latent
``z_norm ∈ ℝ^{S×D}``.

Top-down error from the classification head enters ``z_r``. This node's
local input gradient is supported only on the CLS position; redistribution
across patch tokens happens through encoder attention, not here.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

import jax
import jax.numpy as jnp

from fabricpc.core.activations import IdentityActivation
from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.initializers import (
    InitializerBase,
    NormalInitializer,
    OnesInitializer,
    ZerosInitializer,
    initialize,
)
from fabricpc.core.types import NodeInfo, NodeParams, NodeState
from fabricpc.nodes.base import NodeBase, SlotSpec
from fabricpc.utils.helpers import layernorm

if TYPE_CHECKING:
    from fabricpc.core.energy import EnergyFunctional


def _validate_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def _validate_nonneg_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")


def _validate_positive_float(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive float, got {value!r}")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be a positive float, got {value!r}")
    return parsed


class ClassTokenReadoutNode(NodeBase):
    """Select the CLS token and apply the final ViT LayerNorm.

    Input shape excluding batch: ``(S, D)`` — final encoder sequence ``z_L``.
    Output shape excluding batch: ``(D,)`` — image-level representation.

    Args:
        shape: ``(D,)`` — CLS representation dimension.
        name: Node name.
        embed_dim: Embedding dimension D. Must equal ``shape[0]``.
        cls_index: Sequence index of the CLS token. Default: 0.
        ln_eps: LayerNorm epsilon. Default: ``1e-6`` (Flax / Google ViT).
        energy: Default: GaussianEnergy.
        weight_init: Initializer for LayerNorm ``gamma``. Default: ones.
        latent_init: Initializer for latent states.
    """

    DEFAULT_ENERGY = GaussianEnergy
    DEFAULT_ACTIVATION = IdentityActivation

    def __init__(
        self,
        shape: Tuple[int, ...],
        name: str,
        embed_dim: int,
        cls_index: int = 0,
        ln_eps: float = 1e-6,
        energy: "EnergyFunctional" = GaussianEnergy(),
        weight_init: InitializerBase = OnesInitializer(),
        latent_init: InitializerBase = NormalInitializer(),
    ):
        if len(shape) != 1:
            raise ValueError(
                f"ClassTokenReadoutNode {name!r}: output shape must be (D,) "
                f"with exactly 1 dimension, got shape={shape} "
                f"({len(shape)} dims)"
            )
        _validate_positive_int("embed_dim", embed_dim)
        if isinstance(shape[0], bool) or not isinstance(shape[0], int) or shape[0] <= 0:
            raise ValueError(
                f"ClassTokenReadoutNode {name!r}: shape[0] (embed_dim) "
                f"must be a positive integer, got {shape[0]!r}"
            )
        if shape[0] != embed_dim:
            raise ValueError(
                f"ClassTokenReadoutNode {name!r}: shape[0]={shape[0]} must "
                f"equal embed_dim={embed_dim}"
            )
        _validate_nonneg_int("cls_index", cls_index)
        ln_eps = _validate_positive_float("ln_eps", ln_eps)

        super().__init__(
            shape=shape,
            name=name,
            activation=IdentityActivation(),
            energy=energy,
            latent_init=latent_init,
            weight_init=weight_init,
            embed_dim=embed_dim,
            cls_index=cls_index,
            ln_eps=ln_eps,
        )

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        """Single unscaled sequence input from the final encoder block."""
        return {
            "in": SlotSpec(
                name="in",
                is_multi_input=False,
                is_variance_scalable=False,
            )
        }

    @staticmethod
    def get_weight_fan_in(source_shape: Tuple[int, ...], config: Dict[str, Any]) -> int:
        """Return 1 because token selection and LayerNorm have no mixing weight."""
        return 1

    @staticmethod
    def initialize_params(
        key: jax.Array,
        node_shape: Tuple[int, ...],
        input_shapes: Dict[str, Tuple[int, ...]],
        weight_init: Optional[InitializerBase],
        config: Optional[Dict[str, Any]] = None,
    ) -> NodeParams:
        """Initialise LayerNorm affine parameters ``gamma`` and ``beta``."""
        if config is None:
            config = {}
        if len(input_shapes) != 1:
            raise ValueError(
                "ClassTokenReadoutNode requires exactly one input edge, "
                f"got {len(input_shapes)}"
            )

        input_shape = next(iter(input_shapes.values()))
        if len(input_shape) != 2:
            raise ValueError(
                "ClassTokenReadoutNode expected 2-D sequence input "
                f"(sequence_length, embed_dim), got {input_shape}"
            )

        seq_len, input_dim = input_shape
        output_dim = node_shape[0]
        if input_dim != output_dim:
            raise ValueError(
                "ClassTokenReadoutNode embed_dim mismatch: input D="
                f"{input_dim}, output D={output_dim}"
            )

        cls_index = config.get("cls_index", 0)
        if cls_index >= seq_len:
            raise ValueError(
                f"ClassTokenReadoutNode cls_index={cls_index} is out of range "
                f"for sequence length {seq_len}"
            )

        weights = {"ln_gamma": initialize(key, (output_dim,), weight_init)}
        biases = {
            "ln_beta": initialize(key, (output_dim,), ZerosInitializer()),
        }
        return NodeParams(weights=weights, biases=biases)

    @staticmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[jax.Array, NodeState]:
        """Select CLS and LayerNorm: ``z_mu = LN(z_L[:, cls_index, :])``.

        Steps:
        1. Read the encoder sequence ``(B, S, D)``.
        2. Select the CLS token ``(B, D)``.
        3. Apply LayerNorm with configurable ``ln_eps``.
        4. Activation (identity by default).
        5. Compute prediction error and energy.
        """
        cfg = node_info.node_config
        x = inputs[next(iter(inputs))]
        cls_index = cfg["cls_index"]
        cls_token = x[:, cls_index, :]

        z_ln = layernorm(
            cls_token,
            params.weights["ln_gamma"],
            params.biases["ln_beta"],
            eps=cfg["ln_eps"],
        )

        act_obj = node_info.activation
        z_mu = type(act_obj).forward(z_ln, act_obj.config)

        error = state.z_latent - z_mu
        state = state._replace(
            pre_activation=z_ln,
            z_mu=z_mu,
            error=error,
        )
        state = node_info.node_class.energy_functional(state, node_info)
        return jnp.sum(state.energy), state
