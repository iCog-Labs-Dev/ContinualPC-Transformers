"""Encoder-only transformer builders for classification experiments.

This module intentionally keeps the encoder experiment additive: it reuses the
existing decomposed transformer v2 nodes and does not modify
``fabricpc.nodes.transformer_v2``. The builder below is BERT-style in the sense
that attention is bidirectional and sequence representations are pooled for a
classification head.
"""

from __future__ import annotations

from typing import Dict, Tuple, Any, Optional, Sequence

import jax
import jax.numpy as jnp

from fabricpc.core.activations import IdentityActivation, SoftmaxActivation
from fabricpc.core.energy import CrossEntropyEnergy, GaussianEnergy
from fabricpc.core.inference import InferenceBase
from fabricpc.core.initializers import InitializerBase, NormalInitializer, initialize
from fabricpc.core.mupc import MuPCConfig
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
from fabricpc.nodes import (
    EmbeddingNode,
    IdentityNode,
    Linear,
    LnMlp1Node,
    MhaResidualNode,
    Mlp2ResidualNode,
    NodeBase,
    SlotSpec,
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
        if config is None:
            config = {}
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
) -> tuple[IdentityNode, Optional[IdentityNode], NodeBase, list[NodeBase], list[Edge]]:
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
                Edge(source=mha, target=mlp2.slot("residual")),
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
) -> GraphStructure:
    """Create a bidirectional encoder transformer for sequence classification."""
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
        scaling=MuPCConfig(include_output=False),
        graph_state_initializer=FeedforwardStateInit(),
    )
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
) -> GraphStructure:
    """Create a bidirectional encoder transformer for masked LM pretraining."""
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
    )
    mlm_head = VocabProjectionNode(
        shape=(seq_len, vocab_size),
        name="mlm_head",
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        weight_init=NormalInitializer(std=float(jnp.sqrt(1.0 / embed_dim))),
    )
    nodes.append(mlm_head)
    edges.append(Edge(source=final_ln, target=mlm_head.slot("in")))

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
            **task_kwargs,
        ),
        inference=inference,
        scaling=MuPCConfig(include_output=False),
        graph_state_initializer=FeedforwardStateInit(),
    )
    return structure._replace(
        config={**structure.config, "dropout_rate": float(dropout_rate)}
    )


def _default_encoder_node_names(source_params: GraphParams) -> list[str]:
    names = ["token_embeddings", "position_embeddings"]
    layer = 0
    while f"L{layer}_mha" in source_params.nodes:
        names.extend([f"L{layer}_mha", f"L{layer}_mlp1", f"L{layer}_mlp2"])
        layer += 1
    names.append("final_layernorm")
    return names


def copy_encoder_params(
    source_params: GraphParams,
    target_params: GraphParams,
    node_names: Optional[Sequence[str]] = None,
) -> GraphParams:
    """Copy matching encoder node params into another BERT-style graph."""
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
