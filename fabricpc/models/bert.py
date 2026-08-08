"""Encoder-only transformer builders for classification experiments.

This module intentionally keeps the encoder experiment additive: it reuses the
existing decomposed transformer v2 nodes and does not modify
``fabricpc.nodes.transformer_v2``. The builder below is BERT-style in the sense
that attention is bidirectional and sequence representations are pooled for a
classification head.
"""

from __future__ import annotations

from typing import Dict, Tuple, Any, Optional

import jax
import jax.numpy as jnp

from fabricpc.core.activations import IdentityActivation, SoftmaxActivation
from fabricpc.core.energy import CrossEntropyEnergy, GaussianEnergy
from fabricpc.core.inference import InferenceBase
from fabricpc.core.initializers import InitializerBase, NormalInitializer, initialize
from fabricpc.core.mupc import MuPCConfig
from fabricpc.core.topology import Edge
from fabricpc.core.types import NodeInfo, NodeParams, NodeState
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
)


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
        energy: GaussianEnergy = GaussianEnergy(),
        **kwargs,
    ):
        super().__init__(
            shape=shape,
            name=name,
            seq_len=seq_len,
            embed_dim=embed_dim,
            weight_init=weight_init,
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
        if config is None:
            config = {}
        seq_len = config.get("seq_len", node_shape[0])
        embed_dim = config.get("embed_dim", node_shape[-1])
        return NodeParams(
            weights={"position_embeddings": initialize(key, (seq_len, embed_dim), weight_init)},
            biases={},
        )

    @staticmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> NodeState:
        x = inputs[next(iter(inputs))]
        seq_len = x.shape[1]
        z_mu = x + params.weights["position_embeddings"][:seq_len][None, :, :]
        error = state.z_latent - z_mu
        state = state._replace(z_mu=z_mu, error=error)
        return node_info.node_class.energy_functional(state, node_info)


class FirstTokenPoolNode(NodeBase):
    """Pool a sequence by selecting the first token representation."""

    DEFAULT_ENERGY = GaussianEnergy
    DEFAULT_ACTIVATION = IdentityActivation

    def __init__(
        self,
        shape: Tuple[int, ...],
        name: str,
        token_index: int = 0,
        energy: GaussianEnergy = GaussianEnergy(),
        **kwargs,
    ):
        super().__init__(
            shape=shape,
            name=name,
            token_index=token_index,
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
        return NodeParams(weights={}, biases={})

    @staticmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> NodeState:
        x = inputs[next(iter(inputs))]
        token_index = node_info.node_config.get("token_index", 0)
        z_mu = x[:, token_index, :]
        error = state.z_latent - z_mu
        state = state._replace(z_mu=z_mu, error=error)
        return node_info.node_class.energy_functional(state, node_info)


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
):
    """Create a bidirectional encoder transformer for sequence classification."""
    if weight_init is None:
        weight_init = NormalInitializer(std=0.02)

    input_node = IdentityNode(shape=(seq_len,), name="input_ids")
    embed = EmbeddingNode(
        name="token_embeddings",
        shape=(seq_len, embed_dim),
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        weight_init=NormalInitializer(std=1.0),
    )
    positions = AddPositionEmbeddingNode(
        name="position_embeddings",
        shape=(seq_len, embed_dim),
        seq_len=seq_len,
        embed_dim=embed_dim,
        weight_init=weight_init,
    )

    nodes = [input_node, embed, positions]
    edges = [
        Edge(source=input_node, target=embed.slot("in")),
        Edge(source=embed, target=positions.slot("in")),
    ]

    previous_residual = positions
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
                Edge(source=previous_residual, target=mha.slot("in")),
                Edge(source=previous_residual, target=mha.slot("skip")),
                Edge(source=mha, target=mlp1.slot("in")),
                Edge(source=mlp1, target=mlp2.slot("in")),
                Edge(source=mha, target=mlp2.slot("residual")),
            ]
        )
        previous_residual = mlp2

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
            Edge(source=previous_residual, target=pool.slot("in")),
            Edge(source=pool, target=classifier.slot("in")),
        ]
    )

    return graph(
        nodes=nodes,
        edges=edges,
        task_map=TaskMap(x=input_node, y=classifier),
        inference=inference,
        scaling=MuPCConfig(include_output=False),
        graph_state_initializer=FeedforwardStateInit(),
    )
