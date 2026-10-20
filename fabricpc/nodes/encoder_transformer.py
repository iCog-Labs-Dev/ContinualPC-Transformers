"""Bidirectional (BERT-style) encoder transformer nodes and builders.

The encoder reuses the decomposed transformer-v2 blocks from
``fabricpc.nodes.transformer_v2`` with ``is_causal=False`` and ``use_rope=False``,
and adds the pieces an encoder needs on top of them: learned absolute position
embeddings, a final LayerNorm, and a masked-language-model (MLM) output head
whose loss and PC energy only count the masked token positions.
"""

from __future__ import annotations

import math
from typing import Any, Dict, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp

from fabricpc.core.activations import IdentityActivation, SoftmaxActivation
from fabricpc.core.energy import CrossEntropyEnergy, GaussianEnergy
from fabricpc.core.inference import InferenceBase
from fabricpc.core.initializers import (
    InitializerBase,
    NormalInitializer,
    OnesInitializer,
    XavierInitializer,
    initialize,
)
from fabricpc.core.mupc import MuPCConfig
from fabricpc.core.topology import Edge
from fabricpc.core.types import GraphStructure, NodeInfo, NodeParams, NodeState
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import FeedforwardStateInit
from fabricpc.nodes.base import NodeBase, SlotSpec
from fabricpc.nodes.identity import IdentityNode
from fabricpc.nodes.transformer_v2 import (
    EmbeddingNode,
    LnMlp1Node,
    MhaResidualNode,
    Mlp2ResidualNode,
    VocabProjectionNode,
)
from fabricpc.utils.helpers import layernorm


class AddPositionEmbeddingNode(NodeBase):
    """Add learned absolute position embeddings to token embeddings."""

    DEFAULT_ENERGY = GaussianEnergy
    DEFAULT_ACTIVATION = IdentityActivation

    def __init__(
        self,
        shape: Tuple[int, int],
        name: str,
        seq_len: int,
        embed_dim: int,
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
        config = config or {}
        seq_len = config.get("seq_len", node_shape[0])
        embed_dim = config.get("embed_dim", node_shape[-1])
        return NodeParams(
            weights={
                "position_embeddings": initialize(
                    key, (seq_len, embed_dim), weight_init
                )
            },
            biases={},
        )

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
        error = state.z_latent - z_mu
        state = state._replace(pre_activation=z_mu, z_mu=z_mu, error=error)
        state = node_info.node_class.energy_functional(state, node_info)
        return jnp.sum(state.energy), state


class FinalLayerNormNode(NodeBase):
    """LayerNorm applied to the last encoder block's token representations."""

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


def _input_by_slot(inputs: Dict[str, jnp.ndarray], slot_name: str) -> jnp.ndarray:
    matches = [value for key, value in inputs.items() if key.endswith(f":{slot_name}")]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one input for slot {slot_name!r}")
    return matches[0]


class MaskedVocabProjectionNode(NodeBase):
    """Vocabulary projection whose cross-entropy energy only counts masked tokens.

    Takes the encoder output on ``in`` and a ``(batch, seq_len)`` MLM mask on
    ``loss_mask``. The prediction is ``softmax(logits)`` with the parameters of
    ``VocabProjectionNode``. The per-sample energy is the cross-entropy summed
    over that sample's masked tokens and scaled by ``output_energy_scale``,
    matching ``CrossEntropyEnergy`` restricted to masked positions, so in PC
    training unmasked positions exert no pull on the latents. The activation
    and energy are fixed; passing ``activation`` or ``energy`` raises
    ``ValueError``.

    The energy depends on the ``loss_mask`` input, so ``energy_functional``
    takes it as an extra argument; ``error`` stays the raw
    ``z_latent - z_mu`` at every position.

    Weights, logits and the softmax output are the same as
    ``VocabProjectionNode``'s, so backprop training gives the same results with
    either head: the backprop loss applies ``loss_mask`` itself in
    ``compute_loss`` and never reads this energy. The masked energy only
    affects PC training.
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
        weight_init: Optional[InitializerBase] = None,
        latent_init: Optional[InitializerBase] = None,
        **kwargs,
    ):
        for fixed in ("activation", "energy"):
            if kwargs.get(fixed) is not None:
                raise ValueError(
                    f"MaskedVocabProjectionNode does not accept {fixed}=; it "
                    "always uses softmax with masked cross-entropy"
                )
            kwargs.pop(fixed, None)
        super().__init__(
            shape=shape,
            name=name,
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            output_energy_scale=float(output_energy_scale),
            activation=SoftmaxActivation(),
            energy=CrossEntropyEnergy(),
            weight_init=weight_init or XavierInitializer(),
            latent_init=latent_init or NormalInitializer(),
            **kwargs,
        )

    @staticmethod
    def get_slots():
        return {
            "in": SlotSpec("in", is_multi_input=False),
            "loss_mask": SlotSpec(
                "loss_mask", is_multi_input=False, is_variance_scalable=False
            ),
        }

    @staticmethod
    def initialize_params(key, node_shape, input_shapes, weight_init, config):
        return VocabProjectionNode.initialize_params(
            key, node_shape, input_shapes, weight_init, config
        )

    @staticmethod
    def forward(params, inputs, state, node_info):
        hidden = _input_by_slot(inputs, "in")
        loss_mask = _input_by_slot(inputs, "loss_mask")
        if loss_mask.shape != hidden.shape[:2]:
            raise ValueError(
                "MLM loss_mask shape must match hidden token shape exactly; "
                f"got {loss_mask.shape}, expected {hidden.shape[:2]}"
            )

        logits = jnp.dot(hidden, params.weights["W_out"]) + params.biases["b_out"]
        z_mu = jax.nn.softmax(logits, axis=-1)
        error = state.z_latent - z_mu
        state = state._replace(z_mu=z_mu, pre_activation=logits, error=error)
        state = node_info.node_class.energy_functional(state, node_info, loss_mask)
        return jnp.sum(state.energy), state

    @staticmethod
    def energy_functional(
        state: NodeState,
        node_info: NodeInfo,
        loss_mask: Optional[jnp.ndarray] = None,
    ) -> NodeState:
        """Masked cross-entropy of ``z_latent`` under ``softmax(pre_activation)``.

        Args:
            state: NodeState with ``pre_activation`` holding the logits.
            node_info: NodeInfo carrying ``output_energy_scale``.
            loss_mask: ``(batch, seq_len)`` selection of supervised positions;
                ``None`` selects every position.

        Returns:
            NodeState with a per-sample ``energy`` of shape ``(batch,)``.
        """
        log_probs = jax.nn.log_softmax(state.pre_activation, axis=-1)
        if loss_mask is None:
            loss_mask = jnp.ones(log_probs.shape[:2])
        mask = jax.lax.stop_gradient(loss_mask.astype(log_probs.dtype))
        per_token_ce = -jnp.sum(state.z_latent * log_probs, axis=-1)
        scale = node_info.node_config.get("output_energy_scale", 1.0)
        energy = scale * jnp.sum(per_token_ce * mask, axis=-1)
        return state._replace(energy=energy)


class EncoderBackbone(NamedTuple):
    """Nodes, edges and task entries of an encoder up to its final LayerNorm."""

    input_node: IdentityNode
    output_node: FinalLayerNormNode
    nodes: list
    edges: list
    tasks: Dict[str, NodeBase]


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
    sizes = {
        "depth": depth,
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "mlp_dim": mlp_dim,
        "seq_len": seq_len,
        "vocab_size": vocab_size,
    }
    for name, value in sizes.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if embed_dim % num_heads != 0:
        raise ValueError("embed_dim must be divisible by num_heads")
    if not 0.0 <= dropout_rate < 1.0:
        raise ValueError(f"dropout_rate must be in [0, 1), got {dropout_rate}")


def _build_encoder_backbone(
    *,
    depth: int,
    embed_dim: int,
    num_heads: int,
    mlp_dim: int,
    seq_len: int,
    vocab_size: int,
    dropout_rate: float,
    use_attention_mask: bool,
    embedding_init_std: float,
    position_init_std: float,
    transformer_init_std: float,
) -> EncoderBackbone:
    """input_ids -> embeddings -> positions -> [MHA -> MLP] x depth -> LayerNorm."""
    weight_init = NormalInitializer(std=transformer_init_std)
    input_node = IdentityNode(shape=(seq_len,), name="input_ids")
    embed = EmbeddingNode(
        name="token_embeddings",
        shape=(seq_len, embed_dim),
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        weight_init=NormalInitializer(std=embedding_init_std),
    )
    positions = AddPositionEmbeddingNode(
        name="position_embeddings",
        shape=(seq_len, embed_dim),
        seq_len=seq_len,
        embed_dim=embed_dim,
        weight_init=NormalInitializer(std=position_init_std),
    )
    nodes: list = [input_node, embed, positions]
    edges = [
        Edge(source=input_node, target=embed.slot("in")),
        Edge(source=embed, target=positions.slot("in")),
    ]
    tasks: Dict[str, NodeBase] = {}

    attention_mask_node = None
    if use_attention_mask:
        attention_mask_node = IdentityNode(
            shape=(seq_len,), name="attention_mask", latent_init=OnesInitializer()
        )
        nodes.append(attention_mask_node)
        tasks["attention_mask"] = attention_mask_node

    def add_dropout_mask(task_name: str, target: NodeBase) -> None:
        if dropout_rate <= 0.0:
            return
        mask_node = IdentityNode(
            shape=(seq_len, embed_dim), name=task_name, latent_init=OnesInitializer()
        )
        nodes.append(mask_node)
        edges.append(Edge(source=mask_node, target=target.slot("dropout_mask")))
        tasks[task_name] = mask_node

    previous: NodeBase = positions
    for i in range(depth):
        mha = MhaResidualNode(
            name=f"L{i}_mha",
            shape=(seq_len, embed_dim),
            embed_dim=embed_dim,
            num_heads=num_heads,
            is_causal=False,
            use_rope=False,
            weight_init=weight_init,
        )
        mlp1 = LnMlp1Node(
            name=f"L{i}_mlp1",
            shape=(seq_len, mlp_dim),
            embed_dim=embed_dim,
            ff_dim=mlp_dim,
            weight_init=weight_init,
        )
        mlp2 = Mlp2ResidualNode(
            name=f"L{i}_mlp2",
            shape=(seq_len, embed_dim),
            embed_dim=embed_dim,
            ff_dim=mlp_dim,
            weight_init=weight_init,
        )
        nodes.extend([mha, mlp1, mlp2])
        edges.extend(
            [
                Edge(source=previous, target=mha.slot("in")),
                Edge(source=previous, target=mha.slot("skip")),
                Edge(source=mha, target=mlp1.slot("in")),
                Edge(source=mlp1, target=mlp2.slot("in")),
                Edge(source=mha, target=mlp2.slot("residual")),
            ]
        )
        if attention_mask_node is not None:
            edges.append(
                Edge(source=attention_mask_node, target=mha.slot("attention_mask"))
            )
        add_dropout_mask(f"dropout_L{i}_mha", mha)
        add_dropout_mask(f"dropout_L{i}_mlp2", mlp2)
        previous = mlp2

    final_ln = FinalLayerNormNode(
        shape=(seq_len, embed_dim), name="final_layernorm", embed_dim=embed_dim
    )
    nodes.append(final_ln)
    edges.append(Edge(source=previous, target=final_ln.slot("in")))
    return EncoderBackbone(input_node, final_ln, nodes, edges, tasks)


def create_encoder_mlm_model(
    *,
    depth: int,
    embed_dim: int,
    num_heads: int,
    mlp_dim: int,
    seq_len: int,
    vocab_size: int,
    inference: InferenceBase,
    dropout_rate: float = 0.0,
    use_attention_mask: bool = True,
    embedding_init_std: float = 1.0,
    position_init_std: float = 0.02,
    transformer_init_std: float = 0.02,
    pc_output_energy_scale: float = 1.0,
) -> GraphStructure:
    """Create a bidirectional encoder transformer for masked-LM pretraining.

    Task map: ``x`` (token ids), ``y`` (MLM head, targets are the original
    token ids), ``loss_mask`` (1 at masked positions), plus ``attention_mask``
    (1 at real tokens, 0 at padding) when ``use_attention_mask`` is set, and
    ``dropout_*`` mask sources when ``dropout_rate > 0``. Batches must provide
    ``loss_mask``; the training paths raise without it. An omitted
    ``attention_mask`` attends to every position, which is only correct for
    unpadded batches. ``fabricpc.utils.data.MlmTextBatcher`` yields both.
    """
    _validate_encoder_config(
        depth=depth,
        embed_dim=embed_dim,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        seq_len=seq_len,
        vocab_size=vocab_size,
        dropout_rate=dropout_rate,
    )
    backbone = _build_encoder_backbone(
        depth=depth,
        embed_dim=embed_dim,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        seq_len=seq_len,
        vocab_size=vocab_size,
        dropout_rate=dropout_rate,
        use_attention_mask=use_attention_mask,
        embedding_init_std=embedding_init_std,
        position_init_std=position_init_std,
        transformer_init_std=transformer_init_std,
    )
    loss_mask_node = IdentityNode(shape=(seq_len,), name="mlm_loss_mask")
    mlm_head = MaskedVocabProjectionNode(
        shape=(seq_len, vocab_size),
        name="mlm_head",
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        output_energy_scale=pc_output_energy_scale,
        weight_init=NormalInitializer(std=math.sqrt(1.0 / embed_dim)),
    )
    edges = backbone.edges + [
        Edge(source=backbone.output_node, target=mlm_head.slot("in")),
        Edge(source=loss_mask_node, target=mlm_head.slot("loss_mask")),
    ]
    structure = graph(
        nodes=backbone.nodes + [loss_mask_node, mlm_head],
        edges=edges,
        task_map=TaskMap(
            x=backbone.input_node,
            y=mlm_head,
            loss_mask=loss_mask_node,
            **backbone.tasks,
        ),
        inference=inference,
        scaling=MuPCConfig(include_output=False),
        graph_state_initializer=FeedforwardStateInit(),
    )
    return structure._replace(
        config={**structure.config, "dropout_rate": float(dropout_rate)}
    )
