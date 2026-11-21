"""GraphState manipulation helpers.

These operate on the immutable ``GraphState`` defined in ``core.types``,
returning new states rather than mutating in place. They live in ``core``
because they are foundational state operations used across inference,
training, and initialization.
"""

from typing import Dict

from jax import numpy as jnp

from fabricpc.core.types import GraphState


def update_node_in_state(state: GraphState, node_name: str, **updates) -> GraphState:
    """
    Update a node's fields in GraphState.
    Args:
        state: Current GraphState
        node_name: Name of the node to update
        **updates: kv args fields to update in the NodeState
    Returns:
        New GraphState with updated node
    Usage:
    state = update_node_in_state(state, "hidden", latent_grad=grad)
    """
    updated_node = state.nodes[node_name]._replace(**updates)
    return state._replace(nodes={**state.nodes, node_name: updated_node})


def set_latents_to_clamps(
    state: GraphState,
    clamps: Dict[str, jnp.ndarray],
) -> GraphState:
    """
    Set the latent states of specified nodes to their clamped values.

    The clamp's dtype is preserved (e.g. integer token indices stay int).
    Callers that need a specific dtype should cast values before applying clamps.

    Args:
        state: Current graph state
        clamps: Dictionary of clamped values, keyed on node names

    Returns:
        Updated GraphState with latents set to clamped values
    """
    for node_name, clamp_value in clamps.items():
        if node_name in state.nodes:
            state = update_node_in_state(
                state, node_name, z_latent=jnp.asarray(clamp_value)
            )
    return state


def zero_duals(state: GraphState) -> GraphState:
    """
    Set every node's ``dual`` to zeros shaped and typed like ``latent_grad``.

    ``latent_grad`` is the dtype source because a clamped source node's
    ``z_latent`` may be integer.
    """
    return state._replace(
        nodes={
            name: node_state._replace(dual=jnp.zeros_like(node_state.latent_grad))
            for name, node_state in state.nodes.items()
        }
    )


def drop_duals(state: GraphState) -> GraphState:
    """Set every node's ``dual`` to None."""
    if all(node_state.dual is None for node_state in state.nodes.values()):
        return state
    return state._replace(
        nodes={
            name: node_state._replace(dual=None)
            for name, node_state in state.nodes.items()
        }
    )
