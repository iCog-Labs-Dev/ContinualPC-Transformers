"""Static graph depth of each node, measured in edges to the output nodes."""

from typing import Dict, Tuple

from fabricpc.core.types import GraphStructure

DEPTH_MODES = ("longest", "shortest")


def output_node_names(structure: GraphStructure) -> Tuple[str, ...]:
    """Return the graph sinks with inputs, in graph order.

    Outputs are found from the topology, not ``task_map``: every node with
    ``in_degree > 0`` and ``out_degree == 0``.
    """
    return tuple(
        name
        for name, node in structure.nodes.items()
        if node.node_info.out_degree == 0 and node.node_info.in_degree > 0
    )


def node_output_depths(
    structure: GraphStructure, mode: str = "longest"
) -> Dict[str, int]:
    """Return each node's path length in edges to the output nodes.

    Output nodes (graph sinks with inputs, see ``output_node_names``; not read
    from ``task_map``) have depth 0. With
    ``mode="longest"`` a node's depth is one more than the largest depth of its
    children, with ``mode="shortest"`` one more than the smallest. Nodes from
    which no output is reachable are left out. Clamped source nodes are
    included; inference never updates them.

    Raises:
        ValueError: on an unknown mode, a graph without output nodes, or a
            graph whose ``node_order`` is not a topological order.
    """
    if mode not in DEPTH_MODES:
        raise ValueError(f"depth mode must be one of {DEPTH_MODES}, got {mode!r}")
    outputs = output_node_names(structure)
    if not outputs:
        raise ValueError("graph has no output nodes")
    position = {name: index for index, name in enumerate(structure.node_order)}
    for edge in structure.edges.values():
        if position[edge.source] >= position[edge.target]:
            raise ValueError("node depths require an acyclic graph")
    pick = max if mode == "longest" else min
    depths = {name: 0 for name in outputs}
    for name in reversed(structure.node_order):
        if name in depths:
            continue
        node_info = structure.nodes[name].node_info
        child_depths = [
            depths[structure.edges[edge_key].target] + 1
            for edge_key in node_info.out_edges
            if structure.edges[edge_key].target in depths
        ]
        if child_depths:
            depths[name] = pick(child_depths)
    return depths
