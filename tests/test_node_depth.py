"""Tests for static node-to-output depths."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest

from fabricpc.core.depth import node_output_depths, output_node_names
from fabricpc.core.inference import InferenceSGDNormClip
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.nodes.identity import IdentityNode
from fabricpc.nodes.linear import Linear
from fabricpc.nodes.linear_residual import LinearResidual
from fabricpc.nodes.encoder_transformer import create_encoder_mlm_model


def _encoder(depth: int, dropout_rate: float = 0.0):
    return create_encoder_mlm_model(
        depth=depth,
        embed_dim=8,
        num_heads=2,
        mlp_dim=8,
        seq_len=4,
        vocab_size=10,
        inference=InferenceSGDNormClip(),
        use_attention_mask=True,
        dropout_rate=dropout_rate,
    )


def test_encoder_longest_depths_match_layer_schedule():
    depths = node_output_depths(_encoder(4, dropout_rate=0.1))

    expected = {"mlm_head": 0, "final_layernorm": 1}
    for layer in range(4):
        base = 2 + 3 * (3 - layer)
        expected[f"L{layer}_mlp2"] = base
        expected[f"L{layer}_mlp1"] = base + 1
        expected[f"L{layer}_mha"] = base + 2
    expected["position_embeddings"] = 14
    expected["token_embeddings"] = 15
    assert {name: depths[name] for name in expected} == expected
    assert expected["L0_mha"] == 13


def test_encoder_shortest_depth_follows_skip_edges():
    depths = node_output_depths(_encoder(4), mode="shortest")

    assert depths["token_embeddings"] == 11
    assert depths["L3_mha"] == 3
    assert depths["L3_mlp1"] == 3


def test_output_nodes_are_sinks_with_inputs():
    assert output_node_names(_encoder(1)) == ("mlm_head",)


def test_depths_on_residual_graph():
    x = IdentityNode(shape=(3,), name="x")
    h1 = Linear(shape=(3,), name="h1")
    h2 = Linear(shape=(3,), name="h2")
    h3 = LinearResidual(shape=(3,), name="h3")
    structure = graph(
        nodes=[x, h1, h2, h3],
        edges=[
            Edge(source=x, target=h1.slot("in")),
            Edge(source=h1, target=h2.slot("in")),
            Edge(source=h2, target=h3.slot("in")),
            Edge(source=h1, target=h3.slot("skip")),
        ],
        task_map=TaskMap(x=x, y=h3),
        inference=InferenceSGDNormClip(),
    )

    assert node_output_depths(structure) == {"h3": 0, "h2": 1, "h1": 2, "x": 3}
    assert node_output_depths(structure, "shortest")["h1"] == 1


def test_depths_reject_unknown_mode():
    with pytest.raises(ValueError, match="depth mode"):
        node_output_depths(_encoder(1), mode="median")


def test_outputs_are_graph_sinks_not_task_map_targets():
    x = IdentityNode(shape=(3,), name="x")
    h1 = Linear(shape=(3,), name="h1")
    y = Linear(shape=(3,), name="y")
    side = Linear(shape=(3,), name="side")
    structure = graph(
        nodes=[x, h1, y, side],
        edges=[
            Edge(source=x, target=h1.slot("in")),
            Edge(source=h1, target=y.slot("in")),
            Edge(source=x, target=side.slot("in")),
        ],
        task_map=TaskMap(x=x, y=y),
        inference=InferenceSGDNormClip(),
    )

    assert output_node_names(structure) == ("y", "side")
    assert node_output_depths(structure) == {"y": 0, "side": 0, "h1": 1, "x": 2}
