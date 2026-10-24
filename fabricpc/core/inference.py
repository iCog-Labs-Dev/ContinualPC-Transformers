"""
Core inference dynamics for JAX predictive coding networks.

This module provides:
- InferenceBase: Abstract base class for inference algorithms
- InferenceSGD: Default SGD-based inference (z -= eta * grad)
- InferenceSGDNormClip: SGD inference with per-node gradient norm clipping
- InferencePCALM: Augmented Lagrangian PC with per-node dual ascent
- run_inference: Convenience function wrapping the class-based API

Inference algorithms control how latent states are updated during the
inference loop. The primary extension point is `latent_update()`.

Usage:
    from fabricpc.core.inference import InferenceSGD

    structure = graph(
        nodes=[...], edges=[...], task_map=...,
        inference=InferenceSGD(eta_infer=0.05, infer_steps=20),
    )
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
import types

import jax
import jax.numpy as jnp

from fabricpc.core.scaling import (
    scale_input_grads,
    scale_inputs,
    scale_self_grad,
)
from fabricpc.core.types import (
    GraphParams,
    GraphState,
    GraphStructure,
    NodeInfo,
    NodeState,
)
from fabricpc.core.state_ops import drop_duals, update_node_in_state, zero_duals

# =============================================================================
# Utility Functions
# =============================================================================


def gather_inputs(
    node_info: NodeInfo,
    structure: GraphStructure,
    state: GraphState,
) -> Dict[str, jax.Array]:
    """
    Gather inputs for a node from the graph structure.
    """
    in_edges_data = {}
    for edge_key in node_info.in_edges:
        edge_info = structure.edges[edge_key]  # get the edge object
        node = edge_info.source
        in_edges_data[edge_key] = state.nodes[
            node
        ].z_latent  # get the data sent along this edge

    return in_edges_data


# =============================================================================
# Inference Base Class
# =============================================================================


class InferenceBase(ABC):
    """
    Abstract base class for inference algorithms.

    Inference algorithms control how latent states are updated during the
    inference loop. The primary extension point is `latent_update()`.

    Custom inference algorithms extend this class:

    All computation methods are static for JAX compatibility (pure functions, no state).

    Set ``uses_dual = True`` in algorithms that carry a ``NodeState.dual``;
    ``initialize_graph_state`` then allocates it as zeros.
    """

    uses_dual: bool = False

    def __init__(self, **config):
        self.config = types.MappingProxyType(config)  # Immutable dictionary

    @staticmethod
    def inference_step(
        params: GraphParams,
        state: GraphState,
        clamps: Dict[str, jnp.ndarray],
        structure: GraphStructure,
        config: Dict[str, Any],
    ) -> GraphState:
        """
        Single inference step: forward phase -> latent update.

        Override for algorithms that need a different phase structure
        (e.g., momentum that accumulates across steps).
        """
        inference_obj = structure.config["inference"]
        cls = type(inference_obj)

        # Phase 1: Zero the latent gradients
        state = cls.zero_grads(params, state, clamps, structure)

        # Phase 2: Forward pass to predict expected state (z_mu) & accumulate latent grads
        state = cls.forward_value_and_grad(params, state, clamps, structure)

        # Phase 3: Update the latent (z_state) with gradient
        state = cls.update_latents(params, state, clamps, structure, config)

        return state

    @staticmethod
    def prepare_state(state: GraphState, structure: GraphStructure) -> GraphState:
        """
        Prepare the state once before the inference loop.

        Drops any ``dual`` so leftover multipliers cannot enter the energy of
        an algorithm without duals. Dual-carrying algorithms override this.
        """
        return drop_duals(state)

    @staticmethod
    def final_inference_step(
        params: GraphParams,
        state: GraphState,
        clamps: Dict[str, jnp.ndarray],
        structure: GraphStructure,
        config: Dict[str, Any],
    ) -> GraphState:
        """Last step of the inference loop; ``inference_step`` by default."""
        cls = type(structure.config["inference"])
        return cls.inference_step(params, state, clamps, structure, config)

    @staticmethod
    def zero_grads(
        params: GraphParams,
        state: GraphState,
        clamps: Dict[str, jnp.ndarray],
        structure: GraphStructure,
    ) -> GraphState:
        """
        Phase 1: Zero the latent gradients
        """
        # Phase 1: Zero latent gradients
        # Use latent_grad (not z_latent) as the dtype/shape source: z_latent may
        # carry an integer clamp dtype on source nodes, but gradients are float.
        for node_name in structure.nodes:
            node_state = state.nodes[node_name]
            zero_grad = jnp.zeros_like(node_state.latent_grad)
            state = update_node_in_state(state, node_name, latent_grad=zero_grad)

        return state

    @staticmethod
    def forward_value_and_grad(
        params: GraphParams,
        state: GraphState,
        clamps: Dict[str, jnp.ndarray],
        structure: GraphStructure,
    ) -> GraphState:
        """
        Phase 2: Forward pass value & grad; accumulate latent grads.

        This is the universal predictive coding mechanics shared by all inference algorithms.
        Phase 3 (latent update) is handled by the inference algorithm's latent_update() method.

        muPC scaling is applied here (pre-scale inputs, post-scale gradients),
        keeping node methods (forward_and_latent_grads) scaling-unaware. The
        self-grad contribution is scaled and *added* to ``latent_grad``
        without re-scaling pre-existing accumulations from previously
        iterated downstream successors — those were already scaled by
        their own ``topdown_grad_scale``.
        """
        for node_name in structure.nodes:
            # Get node and its info
            node = structure.nodes[node_name]
            node_info = node.node_info
            node_class = node_info.node_class
            node_state = state.nodes[node_name]
            node_params = params.nodes[node_name]

            # Gather inputs for each slot
            in_edges_data = gather_inputs(node_info, structure, state)

            # Pre-scale inputs by muPC forward scaling factors
            sc = node_info.scaling_config
            scaled_inputs = scale_inputs(in_edges_data, sc)

            # Compute predictions, error, input gradients, and the self-latent
            # gradient contribution (dE/dz_latent for this node only).
            node_state, inedge_grads, self_grad = node_class.forward_and_latent_grads(
                node_params,
                scaled_inputs,
                node_state,
                node_info,
                is_clamped=(node_name in clamps),
            )

            # Scale the new self-grad and add it to the accumulator. Any
            # contributions added earlier in this iteration by downstream
            # successors are preserved unchanged.
            inedge_grads = scale_input_grads(inedge_grads, sc)
            self_grad = scale_self_grad(self_grad, sc)
            node_state = node_state._replace(
                latent_grad=node_state.latent_grad + self_grad
            )

            # Update the graph state with node state containing errors and energy
            state = state._replace(nodes={**state.nodes, node_name: node_state})

            # Accumulate gradient contributions to pre-synaptic nodes (local backward pass to in-neighbors)
            for edge_key, grad in inedge_grads.items():
                source_name = structure.edges[edge_key].source
                latent_grad = state.nodes[source_name].latent_grad + grad
                state = update_node_in_state(
                    state, source_name, latent_grad=latent_grad
                )

        return state

    @staticmethod
    def update_latents(
        params: GraphParams,
        state: GraphState,
        clamps: Dict[str, jnp.ndarray],
        structure: GraphStructure,
        config: Dict[str, Any],
    ) -> GraphState:
        """
        Update latent states for each node based on the accumulated latent gradients.
        """
        inference_obj = structure.config["inference"]
        cls = type(inference_obj)

        for node_name in structure.nodes:
            node_state = state.nodes[node_name]

            # Use the inference algorithm's compute_new_latent() method to get the updated z_latent
            if node_name not in clamps:
                new_z_latent = cls.compute_new_latent(
                    node_name,
                    node_state,
                    config,
                )
                state = update_node_in_state(state, node_name, z_latent=new_z_latent)

        return state

    @staticmethod
    @abstractmethod
    def compute_new_latent(
        node_name: str,
        node_state: NodeState,
        config: Dict[str, Any],
    ) -> jnp.ndarray:
        """
        Compute the updated z_latent for a single node.
        This is the primary extension point for custom inference algorithms.

        Returns:
            Updated z_latent array
        """
        pass

    @staticmethod
    def run_inference(
        params: GraphParams,
        initial_state: GraphState,
        clamps: Dict[str, jnp.ndarray],
        structure: GraphStructure,
    ) -> GraphState:
        """
        Outer inference loop using lax.fori_loop.

        Override for scan-based tracking, adaptive stopping, etc.
        infer_steps is read from self.config['infer_steps']. Runs
        ``prepare_state``, ``infer_steps - 1`` calls of ``inference_step``,
        then ``final_inference_step``.
        """
        inference_obj = structure.config["inference"]
        inference_cls = type(inference_obj)
        config = inference_obj.config
        infer_steps = config["infer_steps"]

        state = inference_cls.prepare_state(initial_state, structure)
        if infer_steps < 1:
            return state

        def body_fn(t, state):
            return inference_cls.inference_step(
                params, state, clamps, structure, config
            )

        # A single fori_loop keeps the compiled program unchanged for algorithms
        # without a distinct final step.
        if inference_cls.final_inference_step is InferenceBase.final_inference_step:
            return jax.lax.fori_loop(0, infer_steps, body_fn, state)
        state = jax.lax.fori_loop(0, infer_steps - 1, body_fn, state)
        return inference_cls.final_inference_step(
            params, state, clamps, structure, config
        )


# =============================================================================
# Built-in Inference Algorithms
# =============================================================================


class InferenceSGD(InferenceBase):
    """
    Standard SGD inference: z -= eta * grad.

    This is the default inference algorithm for predictive coding networks.

    Args:
        eta_infer: Inference rate (default: 0.1)
        infer_steps: Number of inference iterations (default: 20)
    """

    def __init__(self, eta_infer=0.1, infer_steps=20, latent_decay=0.0):
        super().__init__(
            eta_infer=eta_infer, infer_steps=infer_steps, latent_decay=latent_decay
        )

    @staticmethod
    def compute_new_latent(node_name, node_state, config):
        # Precondition: node_state.z_latent is float here.
        # update_latents() gates calls with `node_name not in clamps`, so this
        # only runs on non-clamped nodes — and non-clamped z_latent is always
        # float (state initializers only propagate the clamp dtype to nodes
        # that are clamped). A clamped node's int z_latent (e.g. token
        # indices) never reaches this update path.
        eta_infer = config["eta_infer"]
        latent_decay = config["latent_decay"]

        new_latent = (
            node_state.z_latent * (1.0 - eta_infer * latent_decay)
            - eta_infer * node_state.latent_grad
        )
        return new_latent


class InferenceSGDNormClip(InferenceBase):
    """
    SGD inference with per-node gradient norm clipping: z -= eta * clip(grad).

    Clips the L2 norm of each node's latent gradient independently per sample.
    If ||grad|| > max_norm, scales grad down to max_norm.
    Uses safe division with epsilon to handle zero gradients.

    Args:
        eta_infer: Inference rate (default: 0.1)
        infer_steps: Number of inference iterations (default: 20)
        max_norm: Maximum gradient norm per node (default: 1.0)
        eps: Small constant for numerical stability (default: 1e-8)
    """

    def __init__(
        self, eta_infer=0.1, infer_steps=20, latent_decay=0.0, max_norm=1.0, eps=1e-8
    ):
        super().__init__(
            eta_infer=eta_infer,
            latent_decay=latent_decay,
            infer_steps=infer_steps,
            max_norm=max_norm,
            eps=eps,
        )

    @staticmethod
    def compute_new_latent(node_name, node_state, config):
        eta_infer = config["eta_infer"]
        latent_decay = config["latent_decay"]
        max_norm = config["max_norm"]
        eps = config["eps"]

        grad = node_state.latent_grad
        # Per-sample L2 norm (sum over all non-batch dims)
        grad_norm = jnp.sqrt(
            jnp.sum(grad.conj() * grad, axis=tuple(range(1, grad.ndim)), keepdims=True)
        )
        clip_factor = jnp.minimum(1.0, max_norm / (grad_norm + eps))
        clipped_grad = grad * clip_factor

        return (
            node_state.z_latent * (1.0 - eta_infer * latent_decay)
            - eta_infer * clipped_grad
        )


class InferencePCALM(InferenceBase):
    """
    Augmented Lagrangian predictive coding (PC-ALM; Seely & Gould 2026,
    arXiv 2605.31022).

    Each free node (not clamped, in_degree > 0, out_degree > 0) carries a dual
    ``lambda`` on the constraint ``z_latent = z_mu`` and its energy becomes
    ``E(z, mu) + sum(lambda * (z - mu))``. Each step takes a primal SGD step on
    the augmented energy, then a dual ascent step ``lambda += alpha * (z - mu)``.
    Duals start at zero for every inference run. The final step is primal-only,
    so the learning phase sees the duals used in the last primal step.

    With ``alpha = 0`` this is ``InferenceSGD`` (``max_norm=None``) or
    ``InferenceSGDNormClip`` (``max_norm`` set).

    At a fixed point ``z = mu`` on free nodes and the duals are the negative
    backprop adjoints, so weight gradients equal backprop's. This needs
    hidden-node energies with dE/dz = 0 at z = mu (Gaussian, Huber; not
    cross-entropy, Bernoulli or KL on free hidden nodes). Under muPC the dual
    path picks up ``topdown_grad_scale``, so each node's adjoint and weight
    gradient is scaled by the product of the ``jacobian_gain`` factors of the
    nodes above it (direction unchanged; exact for identity/ReLU activations
    or without muPC).

    Args:
        eta_infer: Inference rate (default: 0.1)
        infer_steps: Number of inference iterations, >= 1 (default: 20)
        alpha: Dual ascent rate, >= 0 (default: 1.0)
        dual_timing: ``"sequential"`` evaluates the residual after the primal
            step (one extra forward pass per step); ``"simultaneous"``
            (Arrow-Hurwicz) reuses the residual from the step's forward pass.
        latent_decay: Weight decay on latent states (default: 0.0)
        max_norm: Per-node per-sample L2 clip on the primal gradient, or None
            for no clipping (default: None)
        eps: Small constant for numerical stability in clipping (default: 1e-8)
    """

    uses_dual = True
    DUAL_TIMINGS = ("sequential", "simultaneous")

    def __init__(
        self,
        eta_infer=0.1,
        infer_steps=20,
        alpha=1.0,
        dual_timing="sequential",
        latent_decay=0.0,
        max_norm=None,
        eps=1e-8,
    ):
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")
        if int(infer_steps) != infer_steps or infer_steps < 1:
            raise ValueError(f"infer_steps must be an integer >= 1, got {infer_steps}")
        if dual_timing not in InferencePCALM.DUAL_TIMINGS:
            raise ValueError(
                f"dual_timing must be one of {InferencePCALM.DUAL_TIMINGS}, "
                f"got {dual_timing!r}"
            )
        if max_norm is not None and max_norm <= 0:
            raise ValueError(f"max_norm must be > 0 or None, got {max_norm}")
        super().__init__(
            eta_infer=eta_infer,
            infer_steps=int(infer_steps),
            alpha=alpha,
            dual_timing=dual_timing,
            latent_decay=latent_decay,
            max_norm=max_norm,
            eps=eps,
        )

    @staticmethod
    def compute_new_latent(node_name, node_state, config):
        if config["max_norm"] is None:
            return InferenceSGD.compute_new_latent(node_name, node_state, config)
        return InferenceSGDNormClip.compute_new_latent(node_name, node_state, config)

    @staticmethod
    def dual_nodes(
        clamps: Dict[str, jnp.ndarray], structure: GraphStructure
    ) -> Tuple[str, ...]:
        """Nodes whose dual is updated: unclamped, with a parent and a child."""
        return tuple(
            name
            for name, node in structure.nodes.items()
            if name not in clamps
            and node.node_info.in_degree > 0
            and node.node_info.out_degree > 0
        )

    @staticmethod
    def update_duals(
        params: GraphParams,
        state: GraphState,
        clamps: Dict[str, jnp.ndarray],
        structure: GraphStructure,
        config: Dict[str, Any],
    ) -> GraphState:
        """Dual ascent ``lambda += alpha * (z_latent - z_mu)`` on free nodes."""
        alpha = config["alpha"]
        if alpha == 0:
            return state
        sequential = config["dual_timing"] == "sequential"

        for node_name in InferencePCALM.dual_nodes(clamps, structure):
            node_state = state.nodes[node_name]
            if sequential:
                node_info = structure.nodes[node_name].node_info
                in_edges_data = gather_inputs(node_info, structure, state)
                scaled_inputs = scale_inputs(in_edges_data, node_info.scaling_config)
                _, projected = node_info.node_class.forward(
                    params.nodes[node_name], scaled_inputs, node_state, node_info
                )
                residual = node_state.z_latent - projected.z_mu
            else:
                residual = node_state.error
            state = update_node_in_state(
                state, node_name, dual=node_state.dual + alpha * residual
            )

        return state

    @staticmethod
    def prepare_state(state: GraphState, structure: GraphStructure) -> GraphState:
        """Start every inference run with zero duals."""
        return zero_duals(state)

    @staticmethod
    def primal_step(
        params: GraphParams,
        state: GraphState,
        clamps: Dict[str, jnp.ndarray],
        structure: GraphStructure,
        config: Dict[str, Any],
    ) -> GraphState:
        """SGD step on the augmented energy (zero grads, forward, latent update)."""
        return InferenceBase.inference_step(params, state, clamps, structure, config)

    @staticmethod
    def inference_step(
        params: GraphParams,
        state: GraphState,
        clamps: Dict[str, jnp.ndarray],
        structure: GraphStructure,
        config: Dict[str, Any],
    ) -> GraphState:
        """Primal step on the augmented energy, then dual ascent."""
        cls = type(structure.config["inference"])
        state = cls.primal_step(params, state, clamps, structure, config)
        return cls.update_duals(params, state, clamps, structure, config)

    @staticmethod
    def final_inference_step(
        params: GraphParams,
        state: GraphState,
        clamps: Dict[str, jnp.ndarray],
        structure: GraphStructure,
        config: Dict[str, Any],
    ) -> GraphState:
        """Primal-only, so learning sees the duals of the last primal step."""
        cls = type(structure.config["inference"])
        return cls.primal_step(params, state, clamps, structure, config)


# =============================================================================
# Convenience Function
# =============================================================================


def run_inference(
    params: GraphParams,
    initial_state: GraphState,
    clamps: Dict[str, jnp.ndarray],
    structure: GraphStructure,
) -> GraphState:
    """
    Run inference using the algorithm object stored in the graph structure.

    Convenience wrapper that extracts the inference object from
    ``structure.config["inference"]`` and delegates to its class's
    static ``run_inference`` method.

    Args:
        params: Graph parameters.
        initial_state: Initial graph state.
        clamps: Clamped node values.
        structure: Graph structure containing the inference object.

    Returns:
        Converged graph state after inference.
    """
    inference_object = structure.config["inference"]
    return type(inference_object).run_inference(
        params, initial_state, clamps, structure
    )
