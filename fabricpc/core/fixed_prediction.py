"""Fixed-prediction latent inference for Transformer V2.

The feed-forward graph state is cached once. During settling, every node keeps
using the cached feed-forward inputs, so both its prediction and its local
Jacobian remain fixed while latent errors are transported. Local parameter
gradients are also read using those same cached inputs.

This is a distinct training rule from ordinary predictive coding. At
convergence it approaches the backpropagation gradient on a feed-forward DAG.
"""

from typing import Dict, Tuple

import jax
import jax.numpy as jnp

from fabricpc.core.inference import (
    InferenceBase,
    InferenceSGDNormClip,
    gather_inputs,
)
from fabricpc.core.scaling import (
    scale_input_grads,
    scale_inputs,
    scale_self_grad,
    scale_weight_grads,
)
from fabricpc.core.state_ops import update_node_in_state
from fabricpc.core.types import (
    GraphParams,
    GraphState,
    GraphStructure,
    NodeParams,
)


class InferenceSGDFixedPrediction(InferenceSGDNormClip):
    """Marker/configuration class for norm-clipped fixed-prediction inference.

    Evaluation and generation are deliberately feed-forward: if the graph's
    target node is not clamped, ``run_inference`` returns the initialized
    feed-forward state without applying latent updates.
    """

    def __init__(
        self,
        eta_infer=0.3864,
        infer_steps=20,
        latent_decay=0.0,
        max_norm=5.0,
        eps=1e-8,
    ):
        if eta_infer <= 0.0:
            raise ValueError(f"eta_infer must be positive, got {eta_infer}")
        if infer_steps < 0:
            raise ValueError(f"infer_steps must be non-negative, got {infer_steps}")
        if max_norm <= 0.0:
            raise ValueError(f"max_norm must be positive, got {max_norm}")
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}")

        super().__init__(
            eta_infer=float(eta_infer),
            infer_steps=int(infer_steps),
            latent_decay=float(latent_decay),
            max_norm=float(max_norm),
            eps=float(eps),
        )

    @staticmethod
    def run_inference(
        params: GraphParams,
        initial_state: GraphState,
        clamps: Dict[str, jnp.ndarray],
        structure: GraphStructure,
    ) -> GraphState:
        # Evaluation/generation clamp only inputs and dropout masks. Their
        # initialized state is already the desired feed-forward prediction.
        output_node = structure.task_map.get("y")
        if output_node is None or output_node not in clamps:
            return initial_state

        final_state, _ = run_fixed_prediction_inference(
            params,
            initial_state,
            clamps,
            structure,
        )
        return final_state


def fixed_prediction_gradient_pass(
    params: GraphParams,
    state: GraphState,
    feedforward_state: GraphState,
    clamps: Dict[str, jnp.ndarray],
    structure: GraphStructure,
) -> GraphState:
    """Refresh local errors and latent gradients at cached feed-forward inputs."""
    state = InferenceBase.zero_grads(params, state, clamps, structure)

    for node_name in structure.node_order:
        node = structure.nodes[node_name]
        node_info = node.node_info
        scaling = node_info.scaling_config

        # Defining operation: predictions and Jacobians see feed-forward inputs,
        # while the postsynaptic latent/error comes from the current state.
        inputs = gather_inputs(node_info, structure, feedforward_state)
        inputs = scale_inputs(inputs, scaling)

        node_state, input_grads, self_grad = (
            node_info.node_class.forward_and_latent_grads(
                params.nodes[node_name],
                inputs,
                state.nodes[node_name],
                node_info,
                is_clamped=(node_name in clamps),
            )
        )

        input_grads = scale_input_grads(input_grads, scaling)
        self_grad = scale_self_grad(self_grad, scaling)
        node_state = node_state._replace(
            latent_grad=node_state.latent_grad + self_grad
        )
        state = state._replace(nodes={**state.nodes, node_name: node_state})

        for edge_key, gradient in input_grads.items():
            source_name = structure.edges[edge_key].source
            accumulated = state.nodes[source_name].latent_grad + gradient
            state = update_node_in_state(
                state,
                source_name,
                latent_grad=accumulated,
            )

    return state


def run_fixed_prediction_inference(
    params: GraphParams,
    initial_state: GraphState,
    clamps: Dict[str, jnp.ndarray],
    structure: GraphStructure,
) -> Tuple[GraphState, GraphState]:
    """Settle latents using feed-forward predictions and Jacobians."""
    feedforward_state = initial_state
    inference = structure.config["inference"]
    config = inference.config
    infer_steps = config["infer_steps"]

    def body_fn(_step, state):
        state = fixed_prediction_gradient_pass(
            params,
            state,
            feedforward_state,
            clamps,
            structure,
        )

        for node_name in structure.node_order:
            if node_name in clamps:
                continue
            new_latent = type(inference).compute_new_latent(
                node_name,
                state.nodes[node_name],
                config,
            )
            state = update_node_in_state(
                state,
                node_name,
                z_latent=new_latent,
            )
        return state

    final_state = jax.lax.fori_loop(
        0,
        infer_steps,
        body_fn,
        initial_state,
    )

    # Refresh errors, energies, and latent gradients at the final latents.
    final_state = fixed_prediction_gradient_pass(
        params,
        final_state,
        feedforward_state,
        clamps,
        structure,
    )
    return final_state, feedforward_state


def compute_fixed_prediction_weight_gradients(
    params: GraphParams,
    settled_state: GraphState,
    feedforward_state: GraphState,
    structure: GraphStructure,
) -> GraphParams:
    """Read local parameter gradients using cached feed-forward inputs."""
    gradients = {}

    for node_name in structure.node_order:
        node_info = structure.nodes[node_name].node_info
        if node_info.in_degree == 0:
            gradients[node_name] = NodeParams(weights={}, biases={})
            continue

        scaling = node_info.scaling_config
        inputs = gather_inputs(node_info, structure, feedforward_state)
        inputs = scale_inputs(inputs, scaling)

        _, parameter_gradient = node_info.node_class.forward_and_weight_grads(
            params.nodes[node_name],
            inputs,
            settled_state.nodes[node_name],
            node_info,
        )
        gradients[node_name] = scale_weight_grads(
            parameter_gradient,
            scaling,
        )

    return GraphParams(nodes=gradients)


def fixed_prediction_inference_and_gradients(
    params: GraphParams,
    initial_state: GraphState,
    clamps: Dict[str, jnp.ndarray],
    structure: GraphStructure,
) -> Tuple[GraphParams, GraphState, GraphState]:
    """Run fixed-prediction inference and perform the matching gradient readout."""
    final_state, feedforward_state = run_fixed_prediction_inference(
        params,
        initial_state,
        clamps,
        structure,
    )
    gradients = compute_fixed_prediction_weight_gradients(
        params,
        final_state,
        feedforward_state,
        structure,
    )
    return gradients, final_state, feedforward_state