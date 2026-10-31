"""Encoder-only transformer builders for classification experiments.

This module intentionally keeps the encoder experiment additive: it reuses the
existing decomposed transformer v2 nodes and does not modify
``fabricpc.nodes.transformer_v2``. The builder below is BERT-style in the sense
that attention is bidirectional and sequence representations are pooled for a
classification head.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp

from fabricpc.core.activations import IdentityActivation, SoftmaxActivation
from fabricpc.core.energy import CrossEntropyEnergy, GaussianEnergy
from fabricpc.core.inference import InferenceBase
from fabricpc.core.initializers import InitializerBase, NormalInitializer, initialize
from fabricpc.core.mupc import MuPCConfig, jacobian_only_scalings
from fabricpc.core.topology import Edge
from fabricpc.core.types import (
    GraphParams,
    GraphStructure,
    NodeInfo,
    NodeParams,
    NodeState,
)
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import FeedforwardStateInit
from fabricpc.nodes.base import NodeBase, SlotSpec
from fabricpc.nodes.identity import IdentityNode
from fabricpc.nodes.linear import Linear
from fabricpc.nodes.transformer_v2 import (
    EmbeddingNode,
    LnMlp1Node,
    MhaResidualNode,
    Mlp2ResidualNode,
    _non_default_options,
)
from fabricpc.utils.helpers import layernorm

ARCHITECTURES = ("encoder", "bert")
BERT_LN_EPS = 1e-12
MUPC_MODES = ("off", "jacobian", "full")


class AddPositionEmbeddingNode(NodeBase):
    """Add learned absolute position embeddings to token embeddings.

    With ``layer_norm=True`` the sum is normalised, ``LN(x + P, ln_eps)``, as
    in the BERT embedding layer.
    """

    DEFAULT_ENERGY = GaussianEnergy
    DEFAULT_ACTIVATION = IdentityActivation

    def __init__(
        self,
        shape: Tuple[int, int],
        name: str,
        seq_len: int,
        embed_dim: int,
        layer_norm: bool = False,
        ln_eps: float = 1e-5,
        weight_init: InitializerBase = NormalInitializer(std=0.02),
        latent_init: InitializerBase = NormalInitializer(),
        energy: GaussianEnergy = GaussianEnergy(),
        **kwargs,
    ):
        super().__init__(
            shape=shape,
            name=name,
            seq_len=seq_len,
            embed_dim=embed_dim,
            weight_init=weight_init,
            latent_init=latent_init,
            energy=energy,
            **_non_default_options(
                layer_norm=(layer_norm, False), ln_eps=(ln_eps, 1e-5)
            ),
            **kwargs,
        )

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        return {"in": SlotSpec("in", is_multi_input=False, is_variance_scalable=False)}

    @staticmethod
    def initialize_params(
        key: jax.Array,
        node_shape: Tuple[int, ...],
        input_shapes: Dict[str, Tuple[int, ...]],
        weight_init: InitializerBase,
        config: Optional[Dict[str, Any]] = None,
    ) -> NodeParams:
        if config is None:
            config = {}
        seq_len = config.get("seq_len", node_shape[0])
        embed_dim = config.get("embed_dim", node_shape[-1])
        weights = {
            "position_embeddings": initialize(key, (seq_len, embed_dim), weight_init)
        }
        biases = {}
        if config.get("layer_norm", False):
            weights["ln_gamma"] = jnp.ones((embed_dim,))
            biases["ln_beta"] = jnp.zeros((embed_dim,))
        return NodeParams(weights=weights, biases=biases)

    @staticmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[jax.Array, NodeState]:
        x = inputs[next(iter(inputs))]
        seq_len = x.shape[1]
        z_mu = x + params.weights["position_embeddings"][:seq_len][None, :, :]
        if node_info.node_config.get("layer_norm", False):
            z_mu = layernorm(
                z_mu,
                params.weights["ln_gamma"],
                params.biases["ln_beta"],
                node_info.node_config.get("ln_eps", 1e-5),
            )
        error = state.z_latent - z_mu
        state = state._replace(pre_activation=z_mu, z_mu=z_mu, error=error)
        state = node_info.node_class.energy_functional(state, node_info)
        return jnp.sum(state.energy), state


class FirstTokenPoolNode(NodeBase):
    """Pool a sequence by selecting the first token representation."""

    DEFAULT_ENERGY = GaussianEnergy
    DEFAULT_ACTIVATION = IdentityActivation

    def __init__(
        self,
        shape: Tuple[int, ...],
        name: str,
        token_index: int = 0,
        latent_init: InitializerBase = NormalInitializer(),
        energy: GaussianEnergy = GaussianEnergy(),
        **kwargs,
    ):
        super().__init__(
            shape=shape,
            name=name,
            token_index=token_index,
            latent_init=latent_init,
            energy=energy,
            **kwargs,
        )

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        return {"in": SlotSpec("in", is_multi_input=False, is_variance_scalable=False)}

    @staticmethod
    def initialize_params(
        key: jax.Array,
        node_shape: Tuple[int, ...],
        input_shapes: Dict[str, Tuple[int, ...]],
        weight_init: InitializerBase,
        config: Optional[Dict[str, Any]] = None,
    ) -> NodeParams:
        return NodeParams(weights={}, biases={})

    @staticmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[jax.Array, NodeState]:
        x = inputs[next(iter(inputs))]
        token_index = node_info.node_config.get("token_index", 0)
        z_mu = x[:, token_index, :]
        error = state.z_latent - z_mu
        state = state._replace(pre_activation=z_mu, z_mu=z_mu, error=error)
        state = node_info.node_class.energy_functional(state, node_info)
        return jnp.sum(state.energy), state


class FinalLayerNormNode(NodeBase):
    """Normalize final encoder token representations before pooling."""

    DEFAULT_ENERGY = GaussianEnergy
    DEFAULT_ACTIVATION = IdentityActivation

    def __init__(
        self,
        shape: Tuple[int, int],
        name: str,
        embed_dim: int,
        latent_init: InitializerBase = NormalInitializer(),
        energy: GaussianEnergy = GaussianEnergy(),
        **kwargs,
    ):
        super().__init__(
            shape=shape,
            name=name,
            embed_dim=embed_dim,
            latent_init=latent_init,
            energy=energy,
            **kwargs,
        )

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        return {"in": SlotSpec("in", is_multi_input=False, is_variance_scalable=False)}

    @staticmethod
    def initialize_params(
        key: jax.Array,
        node_shape: Tuple[int, ...],
        input_shapes: Dict[str, Tuple[int, ...]],
        weight_init: InitializerBase,
        config: Optional[Dict[str, Any]] = None,
    ) -> NodeParams:
        embed_dim = node_shape[-1]
        return NodeParams(
            weights={"ln_gamma": jnp.ones((embed_dim,))},
            biases={"ln_beta": jnp.zeros((embed_dim,))},
        )

    @staticmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[jax.Array, NodeState]:
        x = inputs[next(iter(inputs))]
        z_mu = layernorm(x, params.weights["ln_gamma"], params.biases["ln_beta"])
        error = state.z_latent - z_mu
        state = state._replace(pre_activation=z_mu, z_mu=z_mu, error=error)
        state = node_info.node_class.energy_functional(state, node_info)
        return jnp.sum(state.energy), state


class MlmTransformNode(NodeBase):
    """BERT MLM head transform, ``LN(gelu_erf(W_t x + b_t), ln_eps)``."""

    DEFAULT_ENERGY = GaussianEnergy
    DEFAULT_ACTIVATION = IdentityActivation

    def __init__(
        self,
        shape: Tuple[int, int],
        name: str,
        embed_dim: int,
        ln_eps: float = BERT_LN_EPS,
        weight_init: InitializerBase = NormalInitializer(std=0.02),
        latent_init: InitializerBase = NormalInitializer(),
        energy: GaussianEnergy = GaussianEnergy(),
        **kwargs,
    ):
        super().__init__(
            shape=shape,
            name=name,
            embed_dim=embed_dim,
            ln_eps=ln_eps,
            weight_init=weight_init,
            latent_init=latent_init,
            energy=energy,
            **kwargs,
        )

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        return {"in": SlotSpec("in", is_multi_input=False)}

    @staticmethod
    def initialize_params(
        key: jax.Array,
        node_shape: Tuple[int, ...],
        input_shapes: Dict[str, Tuple[int, ...]],
        weight_init: InitializerBase,
        config: Optional[Dict[str, Any]] = None,
    ) -> NodeParams:
        embed_dim = node_shape[-1]
        return NodeParams(
            weights={
                "W_t": initialize(key, (embed_dim, embed_dim), weight_init),
                "ln_gamma": jnp.ones((embed_dim,)),
            },
            biases={
                "b_t": jnp.zeros((embed_dim,)),
                "ln_beta": jnp.zeros((embed_dim,)),
            },
        )

    @staticmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[jax.Array, NodeState]:
        x = inputs[next(iter(inputs))]
        h = jnp.dot(x, params.weights["W_t"]) + params.biases["b_t"]
        z_mu = layernorm(
            jax.nn.gelu(h, approximate=False),
            params.weights["ln_gamma"],
            params.biases["ln_beta"],
            node_info.node_config.get("ln_eps", BERT_LN_EPS),
        )
        error = state.z_latent - z_mu
        state = state._replace(pre_activation=z_mu, z_mu=z_mu, error=error)
        state = node_info.node_class.energy_functional(state, node_info)
        return jnp.sum(state.energy), state


def _input_by_slot(inputs: Dict[str, jnp.ndarray], slot_name: str) -> jnp.ndarray:
    matches = [value for key, value in inputs.items() if key.endswith(f":{slot_name}")]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one input for slot {slot_name!r}")
    return matches[0]


class MaskedVocabProjectionNode(NodeBase):
    """Vocabulary projection whose PC output energy is masked by MLM positions.

    The per-sample energy is ``output_energy_scale * Σ_masked CE``: a fixed
    per-token precision, independent of the batch size and of how many
    tokens the batch masks. PC weight gradients are therefore sums over the
    masked tokens; callers divide by the masked-token count of the update.
    """

    DEFAULT_ENERGY = CrossEntropyEnergy
    DEFAULT_ACTIVATION = SoftmaxActivation

    def __init__(
        self,
        shape,
        name,
        vocab_size,
        embed_dim,
        output_energy_scale: float = 1.0,
        activation=None,
        weight_init=None,
        latent_init=None,
        energy=None,
        **kwargs,
    ):
        super().__init__(
            shape=shape,
            name=name,
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            output_energy_scale=output_energy_scale,
            activation=activation or SoftmaxActivation(),
            weight_init=weight_init or NormalInitializer(),
            latent_init=latent_init or NormalInitializer(),
            energy=energy or CrossEntropyEnergy(),
            **kwargs,
        )

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        return {
            "in": SlotSpec("in", is_multi_input=False),
            "loss_mask": SlotSpec(
                "loss_mask",
                is_multi_input=False,
                is_variance_scalable=False,
            ),
        }

    @staticmethod
    def initialize_params(
        key: jax.Array,
        node_shape: Tuple[int, ...],
        input_shapes: Dict[str, Tuple[int, ...]],
        weight_init: InitializerBase,
        config: Optional[Dict[str, Any]] = None,
    ) -> NodeParams:
        if config is None:
            config = {}
        vocab, dim = config["vocab_size"], config["embed_dim"]
        return NodeParams(
            weights={"W_out": initialize(key, (dim, vocab), weight_init)},
            biases={"b_out": jnp.zeros((vocab,))},
        )

    @staticmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[jax.Array, NodeState]:
        hidden = _input_by_slot(inputs, "in")
        loss_mask = _input_by_slot(inputs, "loss_mask")
        if loss_mask.shape != hidden.shape[:2]:
            raise ValueError(
                "MLM loss_mask shape must match hidden token shape exactly; "
                f"got {loss_mask.shape}, expected {hidden.shape[:2]}"
            )

        logits = jnp.dot(hidden, params.weights["W_out"]) + params.biases["b_out"]
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        probs = jnp.exp(log_probs)
        mask = jax.lax.stop_gradient(loss_mask.astype(logits.dtype))
        target = state.z_latent
        per_token_ce = -jnp.sum(target * log_probs, axis=-1)
        masked_ce = per_token_ce * mask
        output_energy_scale = float(
            node_info.node_config.get("output_energy_scale", 1.0)
        )
        energy = output_energy_scale * jnp.sum(masked_ce, axis=-1)
        error = (target - probs) * mask[..., None]
        state = state._replace(
            pre_activation=logits,
            z_mu=probs,
            error=error,
            energy=energy,
        )
        return jnp.sum(energy), state


def _build_encoder_backbone(
    *,
    depth: int,
    embed_dim: int,
    num_heads: int,
    mlp_dim: int,
    seq_len: int,
    vocab_size: int,
    weight_init: InitializerBase,
    embedding_init: InitializerBase,
    position_init: InitializerBase,
    dropout_rate: float = 0.0,
    use_attention_mask: bool = False,
    architecture: str = "encoder",
    ln_eps: float = 1e-5,
    residual_identity: bool = False,
) -> tuple[IdentityNode, Optional[IdentityNode], NodeBase, list[NodeBase], list[Edge]]:
    """Build the encoder nodes and edges and return the last encoder node.

    ``"encoder"`` is pre-LN with a final LayerNorm node; ``"bert"`` is post-LN
    with an embedding LayerNorm, erf GELU and no final LayerNorm node. With
    ``residual_identity`` the attention output reaches ``L{i}_mlp2``'s residual
    slot through an identity node ``L{i}_residual``, so under a one-spike-per-node
    schedule the attention node reads the residual error before ``L{i}_mlp1``'s
    spike changes it, which keeps PC weight gradients equal to backprop
    (Salvatori et al. 2022). The node adds a latent and lengthens each block's
    shortest path to the output by one edge, which changes standard PC dynamics.
    """
    bert = architecture == "bert"
    input_node = IdentityNode(shape=(seq_len,), name="input_ids")
    attention_mask_node = (
        IdentityNode(shape=(seq_len,), name="attention_mask")
        if use_attention_mask
        else None
    )
    embed = EmbeddingNode(
        name="token_embeddings",
        shape=(seq_len, embed_dim),
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        weight_init=embedding_init,
    )
    positions = AddPositionEmbeddingNode(
        name="position_embeddings",
        shape=(seq_len, embed_dim),
        seq_len=seq_len,
        embed_dim=embed_dim,
        layer_norm=bert,
        ln_eps=ln_eps,
        weight_init=position_init,
    )

    nodes: list[NodeBase] = [input_node, embed, positions]
    if attention_mask_node is not None:
        nodes.append(attention_mask_node)
    edges = [
        Edge(source=input_node, target=embed.slot("in")),
        Edge(source=embed, target=positions.slot("in")),
    ]

    previous_residual: NodeBase = positions
    for i in range(depth):
        mha = MhaResidualNode(
            name=f"L{i}_mha",
            shape=(seq_len, embed_dim),
            embed_dim=embed_dim,
            num_heads=num_heads,
            is_causal=False,
            use_rope=False,
            norm_position="post" if bert else "pre",
            ln_eps=ln_eps,
            weight_init=weight_init,
        )
        mlp1 = LnMlp1Node(
            name=f"L{i}_mlp1",
            shape=(seq_len, mlp_dim),
            embed_dim=embed_dim,
            ff_dim=mlp_dim,
            pre_ln=not bert,
            exact_gelu=bert,
            weight_init=weight_init,
        )
        mlp2 = Mlp2ResidualNode(
            name=f"L{i}_mlp2",
            shape=(seq_len, embed_dim),
            embed_dim=embed_dim,
            ff_dim=mlp_dim,
            post_ln=bert,
            ln_eps=ln_eps,
            weight_init=weight_init,
        )
        nodes.extend([mha, mlp1, mlp2])
        residual: NodeBase = mha
        if residual_identity:
            residual = IdentityNode(shape=(seq_len, embed_dim), name=f"L{i}_residual")
            nodes.append(residual)
            edges.append(Edge(source=mha, target=residual.slot("in")))
        mha_dropout = None
        mlp2_dropout = None
        if dropout_rate > 0.0:
            mha_dropout = IdentityNode(
                shape=(seq_len, embed_dim), name=f"dropout_mask_L{i}_mha"
            )
            mlp2_dropout = IdentityNode(
                shape=(seq_len, embed_dim), name=f"dropout_mask_L{i}_mlp2"
            )
            nodes.extend([mha_dropout, mlp2_dropout])
        edges.extend(
            [
                Edge(source=previous_residual, target=mha.slot("in")),
                Edge(source=previous_residual, target=mha.slot("skip")),
                Edge(source=mha, target=mlp1.slot("in")),
                Edge(source=mlp1, target=mlp2.slot("in")),
                Edge(source=residual, target=mlp2.slot("residual")),
            ]
        )
        if attention_mask_node is not None:
            edges.append(
                Edge(source=attention_mask_node, target=mha.slot("attention_mask"))
            )
        if mha_dropout is not None and mlp2_dropout is not None:
            edges.extend(
                [
                    Edge(source=mha_dropout, target=mha.slot("dropout_mask")),
                    Edge(source=mlp2_dropout, target=mlp2.slot("dropout_mask")),
                ]
            )
        previous_residual = mlp2

    if bert:
        return input_node, attention_mask_node, previous_residual, nodes, edges
    final_ln = FinalLayerNormNode(
        shape=(seq_len, embed_dim),
        name="final_layernorm",
        embed_dim=embed_dim,
    )
    nodes.append(final_ln)
    edges.append(Edge(source=previous_residual, target=final_ln.slot("in")))
    return input_node, attention_mask_node, final_ln, nodes, edges


def _validate_encoder_config(
    *,
    depth: int,
    embed_dim: int,
    num_heads: int,
    mlp_dim: int,
    seq_len: int,
    vocab_size: int,
    dropout_rate: float,
) -> None:
    values = {
        "depth": depth,
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "mlp_dim": mlp_dim,
        "seq_len": seq_len,
        "vocab_size": vocab_size,
    }
    for name, value in values.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if embed_dim % num_heads != 0:
        raise ValueError("embed_dim must be divisible by num_heads")
    if not 0.0 <= dropout_rate < 1.0:
        raise ValueError("dropout_rate must be in [0, 1)")


def _resolve_ln_eps(architecture: str, ln_eps: Optional[float]) -> float:
    if architecture not in ARCHITECTURES:
        raise ValueError(
            f"architecture must be one of {ARCHITECTURES}, got {architecture!r}"
        )
    if architecture == "encoder":
        if ln_eps is not None:
            raise ValueError("ln_eps is only configurable for architecture='bert'")
        return 1e-5
    return BERT_LN_EPS if ln_eps is None else float(ln_eps)


def _resolve_residual_identity(
    architecture: str, residual_identity: Optional[bool]
) -> bool:
    if residual_identity is None:
        return architecture == "bert"
    if residual_identity and architecture != "bert":
        raise ValueError("residual_identity is only supported for architecture='bert'")
    return bool(residual_identity)


def _resolve_mupc(architecture: str, mupc: Optional[str]) -> Optional[str]:
    """Return the BERT μPC mode (``"off"`` by default), or None for the encoder.

    The pre-LN encoder always keeps its own μPC scaling; it accepts only
    ``None`` or ``"off"``, meaning no BERT μPC mode.
    """
    if architecture != "bert":
        if mupc not in (None, "off"):
            raise ValueError("mupc is only configurable for architecture='bert'")
        return None
    if mupc is None:
        return "off"
    if mupc not in MUPC_MODES:
        raise ValueError(f"mupc must be one of {MUPC_MODES}, got {mupc!r}")
    return mupc


def _graph_scaling(
    architecture: str, mupc: Optional[str] = None
) -> Optional[MuPCConfig]:
    if architecture != "bert":
        return MuPCConfig(include_output=False)
    return None if mupc in (None, "off") else MuPCConfig(include_output=False)


def _apply_bert_mupc(structure: GraphStructure, mupc: Optional[str]) -> GraphStructure:
    """Adapt ``compute_mupc_scalings``' factors to a BERT μPC mode.

    Edges into parameterless ``IdentityNode`` s (the ``L{i}_residual`` nodes)
    stay unscaled in both modes, so the factors equal μPC of the
    ``residual_identity=False`` graph, where the attention output reaches
    ``L{i}_mlp2``'s non-scalable residual slot directly: the identity node
    only relays the residual stream, has no weight that could absorb ``a``,
    and its ``jacobian_gain`` is 1. Residual depth ``L`` counts skip slots
    and is the same for both graphs.
    ``"jacobian"`` replaces every remaining node's factors by
    ``jacobian_only_scalings``, so only the edges into GELU nodes keep a
    factor. Nodes left without a factor get none. A μPC mode is recorded as
    ``structure.config["mupc"]``; ``"off"`` and the encoder return
    ``structure`` unchanged.
    """
    if mupc in (None, "off"):
        return structure
    nodes = {}
    for name, node in structure.nodes.items():
        info = node.node_info
        factors = info.scaling_config
        if factors is not None and info.node_class is IdentityNode:
            factors = None
        if factors is not None and mupc == "jacobian":
            factors = jacobian_only_scalings(factors)
        if factors is not None and not (
            factors.forward_scale or factors.topdown_grad_scale
        ):
            factors = None
        if factors is not info.scaling_config:
            node = node._with_graph_info(replace(info, scaling_config=factors))
        nodes[name] = node
    return structure._replace(nodes=nodes, config={**structure.config, "mupc": mupc})


@dataclass(frozen=True)
class LinearInputRule:
    """Weights that a node applies linearly first to one slot's input.

    ``applies`` checks the node's config and wiring; ``requirement`` names
    what it checks, for the error message.
    """

    weights: Tuple[str, ...]
    applies: Callable[[NodeInfo], bool]
    requirement: str


def _has_slot_edge(info: NodeInfo, slot: str) -> bool:
    return bool(info.slots[slot].in_neighbors) if slot in info.slots else False


MUPC_LINEAR_INPUTS: Dict[Tuple[type, str], LinearInputRule] = {
    (MhaResidualNode, "in"): LinearInputRule(
        ("W_q", "W_k", "W_v"),
        lambda info: info.node_config.get("norm_position", "pre") == "post"
        and _has_slot_edge(info, "skip"),
        "norm_position='post' and a skip edge",
    ),
    (LnMlp1Node, "in"): LinearInputRule(
        ("W_ff1",),
        lambda info: not info.node_config.get("pre_ln", True),
        "pre_ln=False",
    ),
    (Mlp2ResidualNode, "in"): LinearInputRule(
        ("W_ff2",),
        lambda info: _has_slot_edge(info, "residual"),
        "a residual edge",
    ),
    (MlmTransformNode, "in"): LinearInputRule(("W_t",), lambda info: True, "none"),
}


def mupc_weight_scales(structure: GraphStructure) -> Dict[str, Dict[str, float]]:
    """Return ``{node: {weight: a}}`` for every weight behind a scaled edge.

    Each scaled edge must match an entry of ``MUPC_LINEAR_INPUTS`` (exact node
    class and slot, with its requirement met), so its input reaches only
    those weights, linearly, before anything else. Raises ``ValueError`` for
    any other scaled edge, and when two scaled edges of a node would give a
    weight different factors.
    """
    scales: Dict[str, Dict[str, float]] = {}
    for name in structure.node_order:
        info = structure.nodes[name].node_info
        factors = info.scaling_config
        if factors is None:
            continue
        node_scales: Dict[str, float] = {}
        for edge_key, a in factors.forward_scale.items():
            slot = structure.edges[edge_key].slot
            rule = MUPC_LINEAR_INPUTS.get((info.node_class, slot))
            if rule is None or not rule.applies(info):
                needs = "" if rule is None else f" (needs {rule.requirement})"
                raise ValueError(
                    f"{edge_key}: no function-preserving μPC load for a scaled "
                    f"input into {info.node_type} slot {slot!r}{needs}"
                )
            for weight in rule.weights:
                if weight in node_scales and node_scales[weight] != a:
                    raise ValueError(
                        f"{name}.{weight} is behind scaled edges with different a"
                    )
                node_scales[weight] = float(a)
        if node_scales:
            scales[name] = node_scales
    return scales


def is_mupc_full(structure: GraphStructure) -> bool:
    """Whether ``structure`` was built with ``mupc="full"``."""
    return structure.config.get("mupc") == "full"


def _rescale_weights(
    structure: GraphStructure, params: GraphParams, divide: bool
) -> GraphParams:
    if not is_mupc_full(structure):
        raise ValueError("the W / a conversion needs a graph built with mupc='full'")
    nodes = dict(params.nodes)
    for name, node_scales in mupc_weight_scales(structure).items():
        node = params.nodes[name]
        weights = dict(node.weights)
        for weight, a in node_scales.items():
            value = weights[weight]
            weights[weight] = (value / a if divide else value * a).astype(value.dtype)
        nodes[name] = NodeParams(weights=weights, biases=dict(node.biases))
    return GraphParams(nodes=nodes)


def mupc_function_preserving_params(
    structure: GraphStructure, params: GraphParams
) -> GraphParams:
    """Store each weight behind a scaled edge of a ``mupc="full"`` graph as ``W / a``.

    ``params`` are in the unscaled parameterisation (e.g. converted Hugging
    Face weights). Because each scaled input ``a x`` meets only these
    weights first, ``(W / a)(a x) = W x`` and the returned params compute
    the same function under ``structure``. Biases, LayerNorm parameters,
    embeddings and the output head are unchanged. Raises ``ValueError``
    unless ``structure`` is a ``mupc="full"`` graph that
    ``mupc_weight_scales`` accepts. ``mupc_plain_params`` is the inverse.
    """
    return _rescale_weights(structure, params, divide=True)


def mupc_plain_params(structure: GraphStructure, params: GraphParams) -> GraphParams:
    """Return ``W`` from params stored as ``W / a`` under a ``mupc="full"`` graph.

    The inverse of ``mupc_function_preserving_params`` (to rounding): the
    result computes the same function under the ``"off"`` or ``"jacobian"``
    graph of the same shape.
    """
    return _rescale_weights(structure, params, divide=False)


def mupc_plain_grads(structure: GraphStructure, grads: GraphParams) -> GraphParams:
    """Map gradients with respect to ``W / a`` to gradients with respect to ``W``.

    ``∂E/∂W = (∂E/∂(W / a)) / a`` for each rescaled weight; every other leaf
    is unchanged.
    """
    return _rescale_weights(structure, grads, divide=True)


def create_encoder_sentiment_classifier(
    *,
    depth: int,
    embed_dim: int,
    num_heads: int,
    mlp_dim: int,
    seq_len: int,
    vocab_size: int,
    num_classes: int,
    inference: InferenceBase,
    weight_init: Optional[InitializerBase] = None,
    dropout_rate: float = 0.0,
    embedding_init_std: float = 1.0,
    position_init_std: float = 0.02,
    transformer_init_std: float = 0.02,
    use_attention_mask: bool = False,
    architecture: str = "encoder",
    ln_eps: Optional[float] = None,
    residual_identity: Optional[bool] = None,
    mupc: Optional[str] = None,
) -> GraphStructure:
    """Create a bidirectional encoder transformer for sequence classification.

    ``architecture``, ``ln_eps`` and ``mupc`` select the backbone as in
    ``create_encoder_mlm_model``.
    """
    resolved_mupc = _resolve_mupc(architecture, mupc)
    _validate_encoder_config(
        depth=depth,
        embed_dim=embed_dim,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        seq_len=seq_len,
        vocab_size=vocab_size,
        dropout_rate=dropout_rate,
    )
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    resolved_ln_eps = _resolve_ln_eps(architecture, ln_eps)
    if weight_init is None:
        weight_init = NormalInitializer(std=transformer_init_std)

    input_node, attention_mask_node, final_ln, nodes, edges = _build_encoder_backbone(
        depth=depth,
        embed_dim=embed_dim,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        seq_len=seq_len,
        vocab_size=vocab_size,
        weight_init=weight_init,
        embedding_init=NormalInitializer(std=embedding_init_std),
        position_init=NormalInitializer(std=position_init_std),
        dropout_rate=dropout_rate,
        use_attention_mask=use_attention_mask,
        architecture=architecture,
        ln_eps=resolved_ln_eps,
        residual_identity=_resolve_residual_identity(architecture, residual_identity),
    )
    pool = FirstTokenPoolNode(shape=(embed_dim,), name="pooler")
    classifier = Linear(
        shape=(num_classes,),
        name="classifier",
        activation=SoftmaxActivation(),
        energy=CrossEntropyEnergy(),
        weight_init=NormalInitializer(std=0.02),
    )
    nodes.extend([pool, classifier])
    edges.extend(
        [
            Edge(source=final_ln, target=pool.slot("in")),
            Edge(source=pool, target=classifier.slot("in")),
        ]
    )

    task_kwargs = {
        **({"attention_mask": attention_mask_node} if attention_mask_node else {}),
        **{
            f"dropout_L{i}_mha": f"dropout_mask_L{i}_mha"
            for i in range(depth)
            if dropout_rate > 0.0
        },
        **{
            f"dropout_L{i}_mlp2": f"dropout_mask_L{i}_mlp2"
            for i in range(depth)
            if dropout_rate > 0.0
        },
    }
    structure = graph(
        nodes=nodes,
        edges=edges,
        task_map=TaskMap(
            x=input_node,
            y=classifier,
            **task_kwargs,
        ),
        inference=inference,
        scaling=_graph_scaling(architecture, resolved_mupc),
        graph_state_initializer=FeedforwardStateInit(),
    )
    structure = _apply_bert_mupc(structure, resolved_mupc)
    return structure._replace(
        config={**structure.config, "dropout_rate": float(dropout_rate)}
    )


def create_encoder_mlm_model(
    *,
    depth: int,
    embed_dim: int,
    num_heads: int,
    mlp_dim: int,
    seq_len: int,
    vocab_size: int,
    inference: InferenceBase,
    weight_init: Optional[InitializerBase] = None,
    dropout_rate: float = 0.0,
    embedding_init_std: float = 1.0,
    position_init_std: float = 0.02,
    transformer_init_std: float = 0.02,
    use_attention_mask: bool = False,
    pc_output_energy_scale: float = 1.0,
    architecture: str = "encoder",
    ln_eps: Optional[float] = None,
    residual_identity: Optional[bool] = None,
    mupc: Optional[str] = None,
) -> GraphStructure:
    """Create a bidirectional encoder transformer for masked LM pretraining.

    ``architecture="encoder"`` is the pre-LN encoder with a final LayerNorm
    and µPC input scaling. ``architecture="bert"`` matches Hugging Face
    ``BertForMaskedLM``: post-LN blocks, an embedding LayerNorm, erf GELU, an
    ``mlm_transform`` node before ``mlm_head``, no µPC scaling and
    ``ln_eps=1e-12`` unless given. The MLM decoder is untied from the token
    embeddings, and dropout masks act only before the two residual adds.
    ``residual_identity`` (default on for ``"bert"``) adds the parameterless
    ``L{i}_residual`` nodes that keep spike-only PC exact; turn it off for
    standard PC runs that should keep BERT's direct residual edges.

    ``mupc`` (``"bert"`` only) selects μPC for BERT: ``"off"`` (default) has
    no scaling; ``"full"`` attaches ``compute_mupc_scalings``' factors
    (``include_output=False``) and needs its weights loaded through
    ``mupc_function_preserving_params``; ``"jacobian"`` keeps only the
    top-down Jacobian gain on the same edges (``jacobian_only_scalings``)
    and uses the weights as they are. Under standard PC inference the two
    give the same latent dynamics; they differ in what the optimiser sees.
    """
    resolved_ln_eps = _resolve_ln_eps(architecture, ln_eps)
    resolved_mupc = _resolve_mupc(architecture, mupc)
    _validate_encoder_config(
        depth=depth,
        embed_dim=embed_dim,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        seq_len=seq_len,
        vocab_size=vocab_size,
        dropout_rate=dropout_rate,
    )
    if weight_init is None:
        weight_init = NormalInitializer(std=transformer_init_std)

    input_node, attention_mask_node, final_ln, nodes, edges = _build_encoder_backbone(
        depth=depth,
        embed_dim=embed_dim,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        seq_len=seq_len,
        vocab_size=vocab_size,
        weight_init=weight_init,
        embedding_init=NormalInitializer(std=embedding_init_std),
        position_init=NormalInitializer(std=position_init_std),
        dropout_rate=dropout_rate,
        use_attention_mask=use_attention_mask,
        architecture=architecture,
        ln_eps=resolved_ln_eps,
        residual_identity=_resolve_residual_identity(architecture, residual_identity),
    )
    if architecture == "bert":
        transform = MlmTransformNode(
            shape=(seq_len, embed_dim),
            name="mlm_transform",
            embed_dim=embed_dim,
            ln_eps=resolved_ln_eps,
            weight_init=weight_init,
        )
        nodes.append(transform)
        edges.append(Edge(source=final_ln, target=transform.slot("in")))
        final_ln = transform
    loss_mask_node = IdentityNode(shape=(seq_len,), name="mlm_loss_mask")
    mlm_head = MaskedVocabProjectionNode(
        shape=(seq_len, vocab_size),
        name="mlm_head",
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        output_energy_scale=pc_output_energy_scale,
        weight_init=NormalInitializer(std=float(jnp.sqrt(1.0 / embed_dim))),
    )
    nodes.extend([loss_mask_node, mlm_head])
    edges.extend(
        [
            Edge(source=final_ln, target=mlm_head.slot("in")),
            Edge(source=loss_mask_node, target=mlm_head.slot("loss_mask")),
        ]
    )

    task_kwargs = {
        **({"attention_mask": attention_mask_node} if attention_mask_node else {}),
        **{
            f"dropout_L{i}_mha": f"dropout_mask_L{i}_mha"
            for i in range(depth)
            if dropout_rate > 0.0
        },
        **{
            f"dropout_L{i}_mlp2": f"dropout_mask_L{i}_mlp2"
            for i in range(depth)
            if dropout_rate > 0.0
        },
    }
    structure = graph(
        nodes=nodes,
        edges=edges,
        task_map=TaskMap(
            x=input_node,
            y=mlm_head,
            loss_mask=loss_mask_node,
            **task_kwargs,
        ),
        inference=inference,
        scaling=_graph_scaling(architecture, resolved_mupc),
        graph_state_initializer=FeedforwardStateInit(),
    )
    structure = _apply_bert_mupc(structure, resolved_mupc)
    return structure._replace(
        config={**structure.config, "dropout_rate": float(dropout_rate)}
    )


def _default_encoder_node_names(source_params: GraphParams) -> list[str]:
    names = ["token_embeddings", "position_embeddings"]
    layer = 0
    while f"L{layer}_mha" in source_params.nodes:
        names.extend([f"L{layer}_mha", f"L{layer}_mlp1", f"L{layer}_mlp2"])
        layer += 1
    if "final_layernorm" in source_params.nodes:
        names.append("final_layernorm")
    return names


def copy_encoder_params(
    source_params: GraphParams,
    target_params: GraphParams,
    node_names: Optional[Sequence[str]] = None,
) -> GraphParams:
    """Copy matching encoder node params into another BERT-style graph.

    By default the embeddings, every block and ``final_layernorm`` (when the
    source has one) are copied; head nodes such as ``mlm_transform`` are not.
    """
    if node_names is None:
        node_names = _default_encoder_node_names(source_params)

    copied_nodes = {
        name: NodeParams(weights=dict(params.weights), biases=dict(params.biases))
        for name, params in target_params.nodes.items()
    }

    for node_name in node_names:
        if node_name not in source_params.nodes:
            raise ValueError(f"source params missing encoder node {node_name!r}")
        if node_name not in copied_nodes:
            raise ValueError(f"target params missing encoder node {node_name!r}")

        source_node = source_params.nodes[node_name]
        target_node = copied_nodes[node_name]
        for collection_name in ("weights", "biases"):
            source_collection = getattr(source_node, collection_name)
            target_collection = getattr(target_node, collection_name)
            if source_collection.keys() != target_collection.keys():
                raise ValueError(
                    f"{node_name}.{collection_name} keys differ: "
                    f"source={sorted(source_collection)} target={sorted(target_collection)}"
                )
            for key, source_value in source_collection.items():
                target_value = target_collection[key]
                if source_value.shape != target_value.shape:
                    raise ValueError(
                        f"shape mismatch for {node_name}.{collection_name}[{key!r}]: "
                        f"source={source_value.shape} target={target_value.shape}"
                    )

        copied_nodes[node_name] = NodeParams(
            weights=dict(source_node.weights), biases=dict(source_node.biases)
        )

    return GraphParams(nodes=copied_nodes)
