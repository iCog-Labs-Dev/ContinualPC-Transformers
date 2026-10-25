"""
Error-parameterized predictive coding (ePC) inference.

State-based PC (``InferenceSGD`` and variants) relaxes the latents z_latent
by local gradient descent, so the output-loss signal attenuates by the
inference rate per layer per step and deep graphs need many steps. ePC
(Goemaere et al., arXiv 2505.20137) reparameterizes the same energy over the
prediction errors: the error ε is the relaxed variable and each z_latent is
derived by a forward pass in topological order (``structure.node_order``),
``z_latent := z_mu + ε``. Because every node's z_mu depends on all upstream
latents, one ``jax.value_and_grad`` over the ε pytree through the whole
derived forward delivers the loss signal to every layer unattenuated. The
ε ↔ z_latent map is a bijection with unit-determinant triangular Jacobian
(paper Appendix C): identical energies, identical equilibria, and the final
derived state feeds the existing local weight-gradient path unchanged.

Relaxed set. Every unclamped node is relaxed except unclamped readouts
(out_degree == 0): the state-based path holds those at z_latent = z_mu with
zero energy, and ``NodeBase.forward_from_error`` does the same, so derived
energies equal the state-based energies at the same latents.

muPC note: ``scale_inputs`` applies inside the differentiated forward
exactly where the state-based loop applies it, and the global reverse pass
supplies the chain-rule factors automatically. The per-hop gradient
preconditioners (``topdown_grad_scale``'s jacobian gain and
``self_grad_scale``) condition one-hop updates and are not replicated here.
``scale_weight_grads`` at learning time is untouched.

Memory: each step's single ``value_and_grad`` stores activations for the
whole derived forward, reverse-mode memory at backprop scale, versus the
state-based loop's per-node closures.
"""

import math
from typing import Any, Callable, Dict, NamedTuple, Tuple

import jax
import jax.numpy as jnp

from fabricpc.core.epsilon_spectrum import EpsilonSpectrum, weighted_relaxed_fraction
from fabricpc.core.inference import InferenceBase, gather_inputs
from fabricpc.core.scaling import scale_inputs
from fabricpc.core.state_ops import update_node_in_state
from fabricpc.core.types import (
    GraphParams,
    GraphState,
    GraphStructure,
    NodeState,
)


class Regime(NamedTuple):
    """Verdict of ``EPCInference.regime`` on one measured spectrum.

    Every flag is the local quadratic model's prediction at ε = 0 for the
    solver's (eta_infer, infer_steps) = (η, T) on the spectrum
    :class:`~fabricpc.core.epsilon_spectrum.EpsilonSpectrum` describes; on
    a linear graph it is exact.

    Attributes:
        eta, steps: η and T.
        eta_lambda_max, eta_T_lambda_max: η·λ_max and η·T·λ_max.
        unstable: η·λ_max > 2, the top mode's distance grows every step.
        output_gradient_reverses: at unit precision the output residual
            after T steps is r_T = (r/λ)·[1 + (λ − 1)(1 − ηλ)^T] along a mode
            with eigenvalue λ, so the output layer's weight gradient has the
            wrong sign along the top mode when (1 − ηλ_max)^T < −1/(λ_max − 1)
            (T = 1: η(λ_max − 1) > 1). Possible only at odd T, since
            (1 − ηλ)^T ≥ 0 at even T; the hidden errors keep their sign for
            every ηλ < 2.
        f_max: f(λ_max) = 1 − (1 − ηλ_max)^T, the fastest mode's relaxed
            fraction.
        f_weighted: f̄, the gradient-weighted relaxed fraction over the
            positive-curvature modes (``weighted_relaxed_fraction``).
        band: on f̄: "backprop-like" below 0.1, "near PC equilibrium" above
            0.9, "partially relaxed" between; "no positive curvature" when no
            positive mode carries gradient weight.
        negative_weight: fraction of ‖g0‖² on negative-curvature modes.
        growth_min: (1 + η·max(0, −λ_min))^T, the growth of the most negative
            mode over T steps (1.0 when λ_min ≥ 0).
        lambda_max, lambda_min: the spectrum's extremes.

    ``str(regime)`` is the one-line label with precedence: ``unstable``,
    then ``growth_min > 1.1`` (indefinite), then the band with f̄, f_max, and
    the reversal note. Negative-curvature modes have no equilibrium to relax
    toward, so the band never reads them; ``growth_min`` does.
    """

    eta: float
    steps: int
    eta_lambda_max: float
    eta_T_lambda_max: float
    unstable: bool
    output_gradient_reverses: bool
    f_max: float
    f_weighted: float
    band: str
    negative_weight: float
    growth_min: float
    lambda_max: float
    lambda_min: float

    def __str__(self) -> str:
        if self.unstable:
            return f"unstable: eta*lambda_max = {self.eta_lambda_max:.3g} > 2"
        if self.growth_min > 1.1:
            return (
                f"indefinite: negative curvature carrying "
                f"{100 * self.negative_weight:.0f}% of the gradient grows "
                f"{self.growth_min:.3g}x over {self.steps} steps "
                f"(lambda_min = {self.lambda_min:.3g})"
            )
        text = (
            f"eta*T*lambda_max = {self.eta_T_lambda_max:.3g} (gradient-weighted "
            f"relaxed fraction {self.f_weighted:.2f}, fastest mode "
            f"{self.f_max:.2f}): {self.band}"
        )
        if self.output_gradient_reverses:
            text += (
                f"; output-layer gradient reverses on the top mode "
                f"(eta*(lambda_max - 1) = {self.eta * (self.lambda_max - 1.0):.3g})"
            )
        return text


class EPCInference(InferenceBase):
    """
    ePC inference: relax the prediction errors, derive the latents.

    Args:
        eta_infer: Inference rate on ε (default: 1e-3). The ε gradient is
            taken through the full network's transfer function — a change in
            one node's ε moves every downstream derived latent — so tune it
            like a weight learning rate, not like the local per-node rate of
            state-based PC (``InferenceSGD``).
            Gradient descent on the error-coordinate energy is stable only
            for eta_infer < 2/λ_max(H_ε), the top excited eigenvalue of that
            energy's Hessian; ``fabricpc.core.epsilon_spectrum.epsilon_spectrum``
            measures it on any graph, and ``regime`` reads the verdict.
        infer_steps: Number of inference iterations (default: 5). One
            reverse pass per step reaches every layer, so a few steps replace
            the hundreds state-based PC needs on deep DAGs.
        latent_decay: Decay factor on ε in the update (default: 0.0). It
            pulls ε toward 0, i.e. the latents toward the feedforward
            prediction, whereas ``InferenceSGD``'s decay pulls z toward 0, so
            the two solvers share an equilibrium only at zero decay.

    Backprop regime. One step from ε = 0 leaves ε_t = −eta_infer·∂L/∂z_t
    exactly, the backprop activation gradient at the feedforward point. The
    local weight gradients are then taken at the re-derived latents
    (``finalize_state``), so they match backprop's to first order in
    eta_infer·λ_max(H_ε): hidden layers scaled by eta_infer, the output
    layer unscaled (Goemaere et al., Theorem C.9, Case 1). The remainder
    comes from a node's input latent being re-derived at the perturbed
    upstream state, so a layer fed only by clamped nodes matches exactly.
    After T steps each excited error mode with Hessian eigenvalue λ has
    relaxed toward equilibrium by f(λ) = 1 − (1 − eta_infer·λ)^T. The
    regime is read on the modes that carry the starting gradient, weighted
    by the fraction of ‖∇_ε E‖² each carries (f̄, ``Regime.f_weighted``):
    f̄ ≪ 0.1 is backprop-like (Case 2); f̄ > 0.9 is the PC equilibrium, which
    needs eta_infer·T·λ ≳ 3 on those modes, and on a linear graph they are
    the eig(S) modes of Innocenti et al.'s Theorem 1 (S = I + JJᵀ, J the map
    from the hidden errors to the output prediction), so the slowest of them
    sets T. eta_infer·λ_max < 2 is required for stability at every T. At odd
    T the output layer is damaged earlier: its weight gradient is
    proportional to the output residual after T steps, which along the top
    mode is (1 − eta_infer(λ_max − 1))·r at T = 1 and reverses sign once
    eta_infer·(λ_max − 1) > 1, while the hidden errors keep their sign for
    every eta_infer·λ < 2 (``Regime.output_gradient_reverses``).

    Optimizer. Under Adam the eta_infer scaling of the hidden-layer gradients
    is normalized away while eta_infer·|g| ≫ Adam's ε (1e-8), so 1-step ePC
    with Adam trains as backprop with Adam; below that the ε term damps the
    hidden layers. Without Adam the hidden layers learn eta_infer times
    slower than the output layer, a 1000× disparity at the default.

    Weight scale. On a chain λ_max = 1 + σ_max(J)² grows with the product of
    the downstream weights' actions, so it grows during training and a fixed
    eta_infer can cross 2/λ_max late in a run. The defaults have no regime
    meaning independent of the graph, so set both arguments explicitly and
    read ``regime`` on a spectrum re-measured during training.
    """

    def __init__(
        self,
        eta_infer: float = 1e-3,
        infer_steps: int = 5,
        latent_decay: float = 0.0,
    ):
        super().__init__(
            eta_infer=eta_infer, infer_steps=infer_steps, latent_decay=latent_decay
        )

    def regime(self, spectrum: EpsilonSpectrum) -> Regime:
        """The verdict of this solver's (eta_infer, infer_steps) on a measured
        excited spectrum (``epsilon_spectrum`` on any graph, or
        ``EpsilonSpectrum.from_modes`` on known eigenvalues). λ_max grows with
        the weights during training, so a verdict at init describes init.
        """
        eta = float(self.config["eta_infer"])
        steps = int(self.config["infer_steps"])
        lam_max = float(spectrum.lambda_max)
        lam_min = float(spectrum.lambda_min)
        x = eta * lam_max
        contraction = (1.0 - x) ** steps
        f_max = 1.0 - contraction
        f_weighted = weighted_relaxed_fraction(spectrum, eta, steps)
        if math.isnan(f_weighted):
            band = "no positive curvature"
        elif f_weighted < 0.1:
            band = "backprop-like"
        elif f_weighted > 0.9:
            band = "near PC equilibrium"
        else:
            band = "partially relaxed"
        reverses = lam_max > 1.0 and contraction < -1.0 / (lam_max - 1.0)
        growth_min = (1.0 + eta * max(0.0, -lam_min)) ** steps
        return Regime(
            eta=eta,
            steps=steps,
            eta_lambda_max=x,
            eta_T_lambda_max=x * steps,
            unstable=x > 2.0,
            output_gradient_reverses=bool(reverses),
            f_max=f_max,
            f_weighted=f_weighted,
            band=band,
            negative_weight=float(spectrum.negative_weight),
            growth_min=growth_min,
            lambda_max=lam_max,
            lambda_min=lam_min,
        )

    @staticmethod
    def _relaxed_errors(
        structure: GraphStructure, clamps: Dict[str, jnp.ndarray]
    ) -> Tuple[str, ...]:
        """The relaxed node names: every unclamped node except readouts.

        Trace-time Python over static structure. Clamped nodes are never
        relaxed (their z_latent stays the clamp and their error is derived),
        which also keeps int-dtype token sources out of the AD pytree.
        """
        return tuple(
            name
            for name in structure.node_order
            if name not in clamps
            and not (
                structure.nodes[name].node_info.out_degree == 0
                and structure.nodes[name].node_info.in_degree > 0
            )
        )

    @staticmethod
    def _check_acyclic(structure: GraphStructure) -> None:
        if len(structure.node_order) != len(structure.nodes):
            raise ValueError(
                "EPCInference derives latents along structure.node_order, which "
                "omits nodes on cycles; the graph must be acyclic"
            )

    @classmethod
    def derive_states(
        cls,
        params: GraphParams,
        state: GraphState,
        clamps: Dict[str, jnp.ndarray],
        structure: GraphStructure,
    ) -> GraphState:
        """
        Forward-from-errors pass: iterate ``structure.node_order`` and derive
        each node's state from the carried ε via the node's
        ``forward_from_error`` (z_mu from ``forward`` at the latest source
        latents; z_latent = z_mu + ε for relaxed nodes; clamped nodes keep
        the clamp and derive ε).
        """
        cls._check_acyclic(structure)
        for node_name in structure.node_order:
            node_info = structure.nodes[node_name].node_info
            in_edges_data = gather_inputs(node_info, structure, state)
            scaled_inputs = scale_inputs(in_edges_data, node_info.scaling_config)
            new_node_state = node_info.node_class.forward_from_error(
                params.nodes[node_name],
                scaled_inputs,
                state.nodes[node_name],
                node_info,
                is_clamped=(node_name in clamps),
            )
            state = state._replace(nodes={**state.nodes, node_name: new_node_state})
        return state

    @classmethod
    def error_energy(
        cls,
        params: GraphParams,
        state: GraphState,
        clamps: Dict[str, jnp.ndarray],
        structure: GraphStructure,
    ) -> Tuple[
        Callable[[Dict[str, jnp.ndarray]], Tuple[jnp.ndarray, GraphState]],
        Dict[str, jnp.ndarray],
    ]:
        """
        The total energy as a function of the relaxed errors.

        Returns ``(energy_of, errors)``: ``errors`` is the relaxed pytree
        {node name: ε} read from ``state``, and ``energy_of(errors)`` writes
        those ε into the state, derives all latents in topological order, and
        returns ``(total, derived_state)`` with ``total`` the sum of the
        per-sample energies of every ``in_degree > 0`` node (the set the
        training loop sums, so equilibria match state-based PC; a source's ε
        gradient arrives purely through downstream z_mu). One owner of the
        ε-energy for the solver's gradient, Hessian-vector products, and the
        Lanczos spectrum estimator (``fabricpc.core.epsilon_spectrum``).
        """
        relaxed = cls._relaxed_errors(structure, clamps)

        def energy_of(errors):
            inner = state
            for name in relaxed:
                inner = update_node_in_state(inner, name, error=errors[name])
            inner = cls.derive_states(params, inner, clamps, structure)
            total = jnp.asarray(0.0)
            for name in structure.nodes:
                if structure.nodes[name].node_info.in_degree > 0:
                    total = total + jnp.sum(inner.nodes[name].energy)
            return total, inner

        errors = {name: state.nodes[name].error for name in relaxed}
        return energy_of, errors

    @classmethod
    def forward_value_and_grad(
        cls,
        params: GraphParams,
        state: GraphState,
        clamps: Dict[str, jnp.ndarray],
        structure: GraphStructure,
    ) -> GraphState:
        """
        One global energy gradient with respect to the relaxed errors.

        Differentiates ``error_energy``'s scalar with respect to the relaxed
        pytree; gradients accumulate into ``latent_grad`` (never replace it).
        """
        relaxed = cls._relaxed_errors(structure, clamps)
        energy_of, errors = cls.error_energy(params, state, clamps, structure)
        (_, new_state), grads = jax.value_and_grad(energy_of, has_aux=True)(errors)

        for name in relaxed:
            latent_grad = new_state.nodes[name].latent_grad + grads[name]
            new_state = update_node_in_state(new_state, name, latent_grad=latent_grad)
        return new_state

    @classmethod
    def update_latents(
        cls,
        params: GraphParams,
        state: GraphState,
        clamps: Dict[str, jnp.ndarray],
        structure: GraphStructure,
        config: Dict[str, Any],
    ) -> GraphState:
        """Step every relaxed node's ε down the accumulated gradient."""
        for node_name in cls._relaxed_errors(structure, clamps):
            node_state = state.nodes[node_name]
            new_error = cls.compute_new_error(node_name, node_state, config)
            state = update_node_in_state(state, node_name, error=new_error)
        return state

    @staticmethod
    def compute_new_error(
        node_name: str,
        node_state: NodeState,
        config: Dict[str, Any],
    ) -> jnp.ndarray:
        """ε update: ε * (1 - eta * decay) - eta * latent_grad."""
        eta_infer = config["eta_infer"]
        latent_decay = config["latent_decay"]
        return (
            node_state.error * (1.0 - eta_infer * latent_decay)
            - eta_infer * node_state.latent_grad
        )

    @staticmethod
    def compute_new_latent(
        node_name: str, node_state: NodeState, config: Dict[str, Any]
    ) -> jnp.ndarray:
        raise NotImplementedError(
            "EPCInference relaxes errors, not latents; the per-step update is "
            "compute_new_error()."
        )

    @classmethod
    def inference_step(
        cls,
        params: GraphParams,
        state: GraphState,
        clamps: Dict[str, jnp.ndarray],
        structure: GraphStructure,
        config: Dict[str, Any],
    ) -> GraphState:
        """Zero the gradients, take one ε gradient, step ε."""
        state = cls.zero_grads(params, state, clamps, structure)
        state = cls.forward_value_and_grad(params, state, clamps, structure)
        return cls.update_latents(params, state, clamps, structure, config)

    @classmethod
    def begin_segment(
        cls,
        params: GraphParams,
        state: GraphState,
        clamps: Dict[str, jnp.ndarray],
        structure: GraphStructure,
    ) -> GraphState:
        """
        Resync ε to the incoming latents before the first ε update.

        One state-based forward pass at the carried z_latents: each node's
        state is recomputed from its sources' carried latents by the node's
        ``forward_and_latent_grads`` primal (gradients discarded), so
        ε := z_latent - z_mu with the state-based conventions for sources and
        readouts. The first ``derive_states`` then reproduces the incoming
        z_latent node by node, so a preceding solver's latents or a
        distribution initializer's random latents survive the handoff. From
        a feedforward state (z_latent = z_mu) it gives ε = 0 exactly.
        """
        cls._check_acyclic(structure)
        for node_name in structure.node_order:
            node_info = structure.nodes[node_name].node_info
            in_edges_data = gather_inputs(node_info, structure, state)
            scaled_inputs = scale_inputs(in_edges_data, node_info.scaling_config)
            new_node_state, _, _ = node_info.node_class.forward_and_latent_grads(
                params.nodes[node_name],
                scaled_inputs,
                state.nodes[node_name],
                node_info,
                is_clamped=(node_name in clamps),
            )
            state = state._replace(nodes={**state.nodes, node_name: new_node_state})
        return state

    @classmethod
    def finalize_state(
        cls,
        params: GraphParams,
        state: GraphState,
        clamps: Dict[str, jnp.ndarray],
        structure: GraphStructure,
    ) -> GraphState:
        """
        One ``derive_states`` rebuild after the last ε update, so the
        returned state satisfies z_latent = z_mu + ε with energies at the
        final point, the paper's weight rule. The local weight-gradient path,
        the train-loop energy, and eval readouts of z_mu then work unchanged.
        """
        return cls.derive_states(params, state, clamps, structure)

    @classmethod
    def run_inference(
        cls,
        params: GraphParams,
        initial_state: GraphState,
        clamps: Dict[str, jnp.ndarray],
        structure: GraphStructure,
    ) -> GraphState:
        """``begin_segment``, ``infer_steps`` ε steps under ``lax.fori_loop``,
        then ``finalize_state``. The config is read from
        ``structure.config["inference"]``."""
        config = structure.config["inference"].config
        state = cls.begin_segment(params, initial_state, clamps, structure)

        def body_fn(t, state):
            return cls.inference_step(params, state, clamps, structure, config)

        state = jax.lax.fori_loop(0, config["infer_steps"], body_fn, state)
        return cls.finalize_state(params, state, clamps, structure)
