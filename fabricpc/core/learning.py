import jax

from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.types import GraphParams, GraphState, GraphStructure, NodeParams
from fabricpc.core.inference import gather_inputs
from fabricpc.core.scaling import scale_inputs, scale_weight_grads


def compute_local_weight_gradients(
    params: GraphParams,
    final_state: GraphState,
    structure: GraphStructure,
) -> GraphParams:
    """
    Compute local weight gradients for each node using its own error signal.

    This implements the local Hebbian learning rule for predictive coding.
    muPC scaling is applied here (pre-scale inputs, post-scale weight
    gradients), keeping node methods (forward_and_weight_grads) scaling-unaware.

    Args:
        params: Current model parameters
        final_state: Converged state after inference
        structure: Graph structure

    Returns:
        GraphParams containing gradients for the parameters
    """
    gradients = {}

    for node_name, node in structure.nodes.items():
        node_info = node.node_info
        # Source nodes have no weights, but need empty gradient dict for consistency
        if node_info.in_degree == 0:
            gradients[node_name] = NodeParams(weights={}, biases={})
            continue

        in_edges_data = gather_inputs(node_info, structure, final_state)

        node_class = node_info.node_class
        sc = node_info.scaling_config

        # Pre-scale inputs by muPC forward scaling factors
        scaled_inputs = scale_inputs(in_edges_data, sc)

        # Compute local gradients using node's method (pure autodiff)
        node_state, grad_params = node_class.forward_and_weight_grads(
            params.nodes[node_name],
            scaled_inputs,
            final_state.nodes[node_name],
            node_info,
        )

        # Post-scale weight gradients by muPC factors
        grad_params = scale_weight_grads(grad_params, sc)

        # Store gradients
        gradients[node_name] = grad_params

    # convert to GraphParams
    params_gradients = GraphParams(nodes=gradients)

    return params_gradients


FORWARD_UPDATE_MODES = ("initial", "final")


def compute_local_weight_gradients_forward_update(
    params: GraphParams,
    initial_state: GraphState,
    final_state: GraphState,
    structure: GraphStructure,
    mode: str,
) -> GraphParams:
    """
    Compute local weight gradients from the error against the feedforward prediction.

    Forward updates, Eq. 5, Qi et al. 2025 (arXiv 2506.23800): each node's
    weight gradient uses the error ``z_T - mu_0``, where ``z_T`` is the final
    latent and ``mu_0`` the prediction stored in ``initial_state`` (the
    feedforward-initialised state).

    With ``mode="initial"`` the node's inputs are gathered from
    ``initial_state`` and its target is ``z_T``, so both the error and the
    presynaptic activity are those of the feedforward pass; this applies to
    every node, the cross-entropy output head included.

    With ``mode="final"`` the inputs are gathered from ``final_state``. Nodes
    with ``GaussianEnergy`` get the target ``z_T - mu_0 + mu(x_T)``, with
    ``mu(x_T)`` the prediction from the final inputs held constant, so the
    error entering the weight gradient is ``z_T - mu_0``. Nodes with any other
    energy (e.g. the cross-entropy MLM head) keep the gradient of
    ``compute_local_weight_gradients``.

    muPC scaling is applied as in ``compute_local_weight_gradients``.

    Args:
        params: Current model parameters
        initial_state: Feedforward-initialised state before inference
        final_state: State after inference
        structure: Graph structure
        mode: "initial" or "final"

    Returns:
        GraphParams containing gradients for the parameters
    """
    if mode not in FORWARD_UPDATE_MODES:
        raise ValueError(
            f"forward update mode must be one of {FORWARD_UPDATE_MODES}, got {mode!r}"
        )
    input_state = initial_state if mode == "initial" else final_state
    gradients = {}

    for node_name, node in structure.nodes.items():
        node_info = node.node_info
        if node_info.in_degree == 0:
            gradients[node_name] = NodeParams(weights={}, biases={})
            continue

        node_class = node_info.node_class
        node_params = params.nodes[node_name]
        sc = node_info.scaling_config
        in_edges_data = gather_inputs(node_info, structure, input_state)
        scaled_inputs = scale_inputs(in_edges_data, sc)
        node_state = final_state.nodes[node_name]

        if mode == "final" and isinstance(node_info.energy, GaussianEnergy):
            _energy, predicted = node_class.forward(
                node_params, scaled_inputs, node_state, node_info
            )
            target = (
                node_state.z_latent
                - initial_state.nodes[node_name].z_mu
                + jax.lax.stop_gradient(predicted.z_mu)
            )
            node_state = node_state._replace(z_latent=target)

        _node_state, grad_params = node_class.forward_and_weight_grads(
            node_params,
            scaled_inputs,
            node_state,
            node_info,
        )
        gradients[node_name] = scale_weight_grads(grad_params, sc)

    return GraphParams(nodes=gradients)
