"""Tests for EPCInference (Goemaere et al., arXiv 2505.20137) and the ε spectrum."""

import math
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jax.flatten_util import ravel_pytree

from fabricpc.core import EPCInference, InferenceSGD, run_inference
from fabricpc.core.activations import (
    IdentityActivation,
    SoftmaxActivation,
    TanhActivation,
)
from fabricpc.core.energy import CrossEntropyEnergy
from fabricpc.core.epsilon_spectrum import (
    EpsilonSpectrum,
    epsilon_spectrum,
    weighted_relaxed_fraction,
)
from fabricpc.core.initializers import NormalInitializer
from fabricpc.core.learning import compute_local_weight_gradients
from fabricpc.core.topology import Edge
from fabricpc.experiments.domain_forgetting.runner import make_gradient_fn
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.nodes import Linear, StorkeyHopfield
from fabricpc.nodes.encoder_transformer import create_encoder_mlm_model
from fabricpc.nodes.identity import IdentityNode
from fabricpc.training.train_mlm import (
    _prepare_pcn_mlm_batch,
    _train_step_pcn_mlm_incremental_with_state,
    get_graph_param_gradient_pcn_mlm,
)
from fabricpc.utils.dashboarding.inference_tracking import (
    run_inference_with_full_history,
    run_inference_with_history,
)


def with_inference(structure, inference):
    return structure._replace(config={**structure.config, "inference": inference})


def mlp(output="gaussian", std=0.5, activation=TanhActivation, inference=None):
    """x(4) -> h1(3) -> h2(3) -> y(2) with a Gaussian or softmax + CE output."""
    w_init = NormalInitializer(std=std)
    x = IdentityNode(shape=(4,), name="x")
    h1 = Linear(shape=(3,), name="h1", activation=activation(), weight_init=w_init)
    h2 = Linear(shape=(3,), name="h2", activation=activation(), weight_init=w_init)
    if output == "gaussian":
        y = Linear(
            shape=(2,), name="y", activation=IdentityActivation(), weight_init=w_init
        )
    else:
        y = Linear(
            shape=(2,),
            name="y",
            activation=SoftmaxActivation(),
            energy=CrossEntropyEnergy(),
            weight_init=w_init,
        )
    return graph(
        nodes=[x, h1, h2, y],
        edges=[
            Edge(source=x, target=h1.slot("in")),
            Edge(source=h1, target=h2.slot("in")),
            Edge(source=h2, target=y.slot("in")),
        ],
        task_map=TaskMap(x=x, y=y),
        inference=inference or EPCInference(eta_infer=0.05, infer_steps=1),
    )


def mlp_inputs(structure, output, batch=3, clamp_output=True, seed=0):
    key = jax.random.PRNGKey(seed)
    params = initialize_params(structure, key)
    params = params._replace(
        nodes={
            name: node._replace(
                biases={
                    k: 0.3 * jax.random.normal(jax.random.fold_in(key, i), b.shape)
                    for i, (k, b) in enumerate(node.biases.items())
                }
            )
            for name, node in params.nodes.items()
        }
    )
    clamps = {"x": jax.random.normal(jax.random.PRNGKey(seed + 1), (batch, 4))}
    if clamp_output:
        if output == "gaussian":
            clamps["y"] = jax.random.normal(jax.random.PRNGKey(seed + 2), (batch, 2))
        else:
            labels = jax.random.randint(jax.random.PRNGKey(seed + 2), (batch,), 0, 2)
            clamps["y"] = jax.nn.one_hot(labels, 2)
    state = initialize_graph_state(structure, batch, key, clamps, params=params)
    return params, clamps, state


def tiny_bert(inference, dropout_rate=0.1):
    return create_encoder_mlm_model(
        depth=1,
        embed_dim=8,
        num_heads=2,
        mlp_dim=16,
        seq_len=6,
        vocab_size=20,
        inference=inference,
        use_attention_mask=True,
        dropout_rate=dropout_rate,
        architecture="bert",
        residual_identity=True,
    )


def bert_batch():
    return {
        "x": jnp.array([[1, 3, 5, 3, 0, 0], [1, 7, 3, 9, 4, 2]], dtype=jnp.int32),
        "y": jnp.array([[1, 4, 5, 6, 0, 0], [1, 7, 8, 9, 4, 2]], dtype=jnp.int32),
        "attention_mask": jnp.array(
            [[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=jnp.float32
        ),
        "loss_mask": jnp.array(
            [[0, 1, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0]], dtype=jnp.float32
        ),
    }


def bert_inputs(structure, clamp_output=True):
    key = jax.random.PRNGKey(3)
    params = initialize_params(structure, key)
    shared, _metrics, free_state, initial_state, clamps = _prepare_pcn_mlm_batch(
        params, bert_batch(), structure, key
    )
    if clamp_output:
        return params, clamps, initial_state
    return params, shared, free_state


def standard_state(params, state, clamps, structure):
    """The state-based path's node states (energies) at ``state``'s latents."""
    state = InferenceSGD.zero_grads(params, state, clamps, structure)
    return InferenceSGD.forward_value_and_grad(params, state, clamps, structure)


def perturb_free_latents(state, clamps, structure, scale=0.1, seed=5):
    keys = jax.random.split(jax.random.PRNGKey(seed), len(structure.nodes))
    for key, name in zip(keys, structure.nodes):
        info = structure.nodes[name].node_info
        if name in clamps or info.in_degree == 0 or info.out_degree == 0:
            continue
        node = state.nodes[name]
        noise = scale * jax.random.normal(key, node.z_latent.shape, node.z_latent.dtype)
        state = state._replace(
            nodes={**state.nodes, name: node._replace(z_latent=node.z_latent + noise)}
        )
    return state


def assert_states_match(derived, reference, structure, atol):
    for name in structure.node_order:
        got, ref = derived.nodes[name], reference.nodes[name]
        np.testing.assert_allclose(got.z_latent, ref.z_latent, atol=atol, err_msg=name)
        np.testing.assert_allclose(got.z_mu, ref.z_mu, atol=atol, err_msg=name)
        np.testing.assert_allclose(got.energy, ref.energy, atol=atol, err_msg=name)


def total_energy(state):
    return sum(float(jnp.sum(node.energy)) for node in state.nodes.values())


def rel(a, b):
    return float(jnp.linalg.norm(a - b) / jnp.linalg.norm(b))


class TestZeroErrorIsFeedforward:
    @pytest.mark.parametrize("output", ["gaussian", "ce"])
    @pytest.mark.parametrize("clamp_output", [True, False])
    def test_mlp(self, output, clamp_output):
        structure = mlp(output)
        params, clamps, state = mlp_inputs(structure, output, clamp_output=clamp_output)
        synced = EPCInference.begin_segment(params, state, clamps, structure)
        for name in EPCInference._relaxed_errors(structure, clamps):
            assert jnp.all(synced.nodes[name].error == 0.0), name
        derived = EPCInference.derive_states(params, synced, clamps, structure)
        for name in structure.nodes:
            assert jnp.array_equal(
                derived.nodes[name].z_latent, state.nodes[name].z_latent
            ), name
        assert_states_match(
            derived, standard_state(params, state, clamps, structure), structure, 0.0
        )

    @pytest.mark.parametrize("clamp_output", [True, False])
    def test_tiny_bert(self, clamp_output):
        structure = tiny_bert(EPCInference(eta_infer=1e-3, infer_steps=2))
        params, clamps, state = bert_inputs(structure, clamp_output)
        synced = EPCInference.begin_segment(params, state, clamps, structure)
        relaxed = EPCInference._relaxed_errors(structure, clamps)
        assert "L0_mha" in relaxed and "mlm_head" not in relaxed
        for name in relaxed:
            assert jnp.all(synced.nodes[name].error == 0.0), name
        derived = EPCInference.derive_states(params, synced, clamps, structure)
        reference = standard_state(params, state, clamps, structure)
        for name in structure.nodes:
            assert jnp.array_equal(
                derived.nodes[name].z_latent, state.nodes[name].z_latent
            ), name
        assert_states_match(derived, reference, structure, 1e-6)
        head = float(jnp.sum(derived.nodes["mlm_head"].energy))
        assert (head > 0.0) == clamp_output


class TestEnergyConsistency:
    def test_tiny_bert_perturbed_latents(self):
        """At arbitrary latents, the resync + derive reproduces the latents and
        every node's energy (masked MLM CE included) to rounding."""
        with jax.enable_x64(True):
            structure = tiny_bert(EPCInference(), dropout_rate=0.1)
            params, clamps, state = bert_inputs(structure)
            state = perturb_free_latents(state, clamps, structure)
            synced = EPCInference.begin_segment(params, state, clamps, structure)
            derived = EPCInference.derive_states(params, synced, clamps, structure)
            reference = standard_state(params, state, clamps, structure)
            assert_states_match(derived, reference, structure, 1e-10)
            assert float(jnp.sum(reference.nodes["L0_mlp1"].energy)) > 1e-3

    def test_storkey_hopfield_internal_node(self, rng_key):
        x = IdentityNode(shape=(6,), name="x")
        hop = StorkeyHopfield(shape=(6,), name="hop", hopfield_strength=2.0)
        y = Linear(shape=(3,), name="y", activation=IdentityActivation())
        structure = graph(
            nodes=[x, hop, y],
            edges=[
                Edge(source=x, target=hop.slot("in")),
                Edge(source=hop, target=y.slot("in")),
            ],
            task_map=TaskMap(x=x, y=y),
            inference=EPCInference(eta_infer=0.01, infer_steps=3),
        )
        params = initialize_params(structure, rng_key)
        clamps = {
            "x": jax.random.normal(rng_key, (4, 6)),
            "y": jax.random.normal(jax.random.PRNGKey(1), (4, 3)),
        }
        state = initialize_graph_state(structure, 4, rng_key, clamps, params=params)
        state = perturb_free_latents(state, clamps, structure)
        synced = EPCInference.begin_segment(params, state, clamps, structure)
        derived = EPCInference.derive_states(params, synced, clamps, structure)
        reference = standard_state(params, state, clamps, structure)
        assert_states_match(derived, reference, structure, 1e-5)


def backprop_activation_grads(params, x, y, output):
    W1, b1 = params.nodes["h1"].weights["x->h1:in"], params.nodes["h1"].biases["b"]
    W2, b2 = params.nodes["h2"].weights["h1->h2:in"], params.nodes["h2"].biases["b"]
    Wy, by = params.nodes["y"].weights["h2->y:in"], params.nodes["y"].biases["b"]
    a1 = jnp.tanh(x @ W1 + b1)
    a2 = jnp.tanh(a1 @ W2 + b2)
    pre_y = a2 @ Wy + by
    delta_y = pre_y - y if output == "gaussian" else jax.nn.softmax(pre_y) - y
    g_h2 = delta_y @ Wy.T
    g_h1 = (g_h2 * (1.0 - a2**2)) @ W2.T
    return {"h1": g_h1, "h2": g_h2}


class TestBackpropRegime:
    """One step from ε = 0 (Goemaere et al., Theorem C.9, Case 1)."""

    @pytest.mark.parametrize("output", ["gaussian", "ce"])
    def test_mlp_one_step(self, output):
        eta = 1e-4
        with jax.enable_x64(True):
            structure = mlp(
                output, inference=EPCInference(eta_infer=eta, infer_steps=1)
            )
            params, clamps, state = mlp_inputs(structure, output)
            final = run_inference(params, state, clamps, structure)
            g = backprop_activation_grads(params, clamps["x"], clamps["y"], output)
            for name in ("h1", "h2"):
                np.testing.assert_allclose(
                    final.nodes[name].error, -eta * g[name], rtol=1e-12, atol=1e-16
                )
            for name in structure.nodes:
                node = final.nodes[name]
                if name not in clamps:
                    np.testing.assert_allclose(
                        node.z_latent, node.z_mu + node.error, atol=1e-15
                    )

            def loss(p):
                ff = initialize_graph_state(
                    structure, 3, jax.random.PRNGKey(0), clamps, params=p
                )
                ff = standard_state(p, ff, clamps, structure)
                return jnp.sum(ff.nodes["y"].energy)

            g_bp = jax.grad(loss)(params)
            g_pc = compute_local_weight_gradients(params, final, structure)
            for name in ("h1", "h2", "y"):
                scale = 1.0 if name == "y" else eta
                for kind in ("weights", "biases"):
                    for key, ref in getattr(g_bp.nodes[name], kind).items():
                        got = getattr(g_pc.nodes[name], kind)[key] / scale
                        assert rel(got, ref) < 1e-3, (name, key, rel(got, ref))

    def test_tiny_bert_through_domain_runner_gradient_fn(self):
        """The runner's PC gradient path with ePC: the MLM head gradient is
        backprop's and every hidden module's is eta times backprop's."""
        eta = 1e-4
        with jax.enable_x64(True):
            structure = tiny_bert(EPCInference(eta_infer=eta, infer_steps=1))
            params = initialize_params(structure, jax.random.PRNGKey(3))
            key = jax.random.PRNGKey(11)
            pc_fn = make_gradient_fn("pc", structure, "none", "standard")
            bp_fn = make_gradient_fn("backprop", structure)
            pc_grads, pc_loss, pc_count, extras = pc_fn(params, bert_batch(), key)
            bp_grads, bp_loss, bp_count, _ = bp_fn(params, bert_batch(), key)
            assert float(pc_count) == float(bp_count) == 3.0
            np.testing.assert_allclose(pc_loss, bp_loss, rtol=1e-10)
            assert float(extras["pc_energy_end"]) < float(extras["pc_energy_start"])
            checked = 0
            for name in structure.node_order:
                pc_leaves = jax.tree_util.tree_leaves(pc_grads.nodes[name])
                if not pc_leaves:
                    continue
                bp_leaves = jax.tree_util.tree_leaves(bp_grads.nodes[name])
                scale = 1.0 if name == "mlm_head" else eta
                got = ravel_pytree(pc_leaves)[0] / scale
                ref = ravel_pytree(bp_leaves)[0]
                assert rel(got, ref) < 1e-3, (name, rel(got, ref))
                checked += 1
            assert checked >= 7


class TestEquilibrium:
    def test_matches_state_based_pc(self):
        """Run to convergence, ePC and InferenceSGD reach the same stationary
        point: latents, energies and local weight gradients."""
        with jax.enable_x64(True):
            finals = {}
            for key, inference in (
                ("spc", InferenceSGD(eta_infer=0.1, infer_steps=3000)),
                ("epc", EPCInference(eta_infer=0.1, infer_steps=3000)),
            ):
                structure = mlp("gaussian", std=0.5, inference=inference)
                params, clamps, state = mlp_inputs(structure, "gaussian")
                finals[key] = run_inference(params, state, clamps, structure)
            spc, epc = finals["spc"], finals["epc"]
            assert_states_match(epc, spc, structure, 1e-8)
            reference = standard_state(params, epc, clamps, structure)
            for name in ("h1", "h2"):
                assert float(jnp.max(jnp.abs(reference.nodes[name].latent_grad))) < 1e-9
            g_spc = compute_local_weight_gradients(params, spc, structure)
            g_epc = compute_local_weight_gradients(params, epc, structure)
            for a, b in zip(
                jax.tree_util.tree_leaves(g_spc), jax.tree_util.tree_leaves(g_epc)
            ):
                np.testing.assert_allclose(a, b, atol=1e-8)
            ff = standard_state(params, state, clamps, structure)

            def total(s):
                return sum(float(jnp.sum(s.nodes[n].energy)) for n in ("h1", "h2", "y"))

            assert total(epc) < total(ff)

    def test_jit_and_energy_descent(self):
        structure = mlp("ce", inference=EPCInference(eta_infer=0.05, infer_steps=1))
        params, clamps, state = mlp_inputs(structure, "ce")
        step = jax.jit(lambda p, s: run_inference(p, s, clamps, structure))
        energies = []
        for _ in range(5):
            state = step(params, state)
            energies.append(
                sum(float(jnp.sum(state.nodes[n].energy)) for n in ("h1", "h2", "y"))
            )
        assert all(b < a for a, b in zip(energies, energies[1:])), energies


def linear_chain():
    return mlp("gaussian", std=0.8, activation=IdentityActivation)


class TestEpsilonSpectrum:
    def test_linear_chain_matches_exact_hessian(self):
        with jax.enable_x64(True):
            structure = linear_chain()
            params, clamps, state = mlp_inputs(structure, "gaussian", batch=2)
            synced = EPCInference.begin_segment(params, state, clamps, structure)
            energy_of, errors = EPCInference.error_energy(
                params, synced, clamps, structure
            )
            flat, unflat = ravel_pytree(errors)
            energy = lambda v: energy_of(unflat(v))[0]  # noqa: E731
            H = np.asarray(jax.hessian(energy)(flat))
            g0 = np.asarray(jax.grad(energy)(flat))
            values, vectors = np.linalg.eigh(H)
            weights = (vectors.T @ g0) ** 2 / np.dot(g0, g0)
            excited = values[weights > 1e-12]

            spectrum = epsilon_spectrum(params, state, clamps, structure, iters=12)
            assert not spectrum.random_start
            assert spectrum.lambda_max == pytest.approx(excited.max(), rel=1e-8)
            assert spectrum.lambda_min == pytest.approx(excited.min(), rel=1e-8)
            assert np.all(values > 0.0) and spectrum.negative_weight == 0.0
            exact = EpsilonSpectrum.from_modes(values, weights)
            for eta, steps in ((1e-3, 1), (0.05, 10), (0.2, 50)):
                assert weighted_relaxed_fraction(spectrum, eta, steps) == pytest.approx(
                    weighted_relaxed_fraction(exact, eta, steps), abs=1e-8
                )

    def test_one_step_relaxation_matches_quadratic_model(self):
        """On a linear graph one ε step from 0 moves the energy by exactly the
        quadratic model built from g0 and H_ε."""
        eta = 0.02
        with jax.enable_x64(True):
            structure = with_inference(
                linear_chain(), EPCInference(eta_infer=eta, infer_steps=1)
            )
            params, clamps, state = mlp_inputs(structure, "gaussian", batch=2)
            synced = EPCInference.begin_segment(params, state, clamps, structure)
            energy_of, errors = EPCInference.error_energy(
                params, synced, clamps, structure
            )
            flat, unflat = ravel_pytree(errors)
            energy = lambda v: energy_of(unflat(v))[0]  # noqa: E731
            g0 = jax.grad(energy)(flat)
            H = jax.hessian(energy)(flat)
            predicted = energy(flat) - eta * g0 @ g0 + 0.5 * eta**2 * g0 @ H @ g0
            final = run_inference(params, state, clamps, structure)
            got = sum(float(jnp.sum(final.nodes[n].energy)) for n in ("h1", "h2", "y"))
            assert got == pytest.approx(float(predicted), rel=1e-10)

    def test_tiny_bert_regime_predicts_relaxation(self):
        """The measured spectrum predicts the one-step energy change to
        second order, and ``regime``'s stability verdict matches whether
        many ε steps lower or raise the energy."""
        with jax.enable_x64(True):
            structure = tiny_bert(EPCInference(), dropout_rate=0.0)
            params, clamps, state = bert_inputs(structure)
            spectrum = epsilon_spectrum(params, state, clamps, structure, iters=30)
            lam = spectrum.lambda_max
            g2 = spectrum.gradient_norm**2
            curvature = float(np.sum(spectrum.ritz_weights * spectrum.ritz_values))
            start = total_energy(standard_state(params, state, clamps, structure))

            def run(eta, steps):
                inference = EPCInference(eta_infer=eta, infer_steps=steps)
                s = with_inference(structure, inference)
                final = run_inference(params, state, clamps, s)
                return inference.regime(spectrum), total_energy(final)

            eta = 0.05 / lam
            regime, end = run(eta, 1)
            predicted = -eta * g2 + 0.5 * eta**2 * g2 * curvature
            assert regime.band == "backprop-like"
            assert end - start == pytest.approx(predicted, rel=1e-3)

            stable, stable_end = run(1.0 / lam, 40)
            assert not stable.unstable and stable_end < start
            unstable, unstable_end = run(2.5 / lam, 40)
            assert unstable.unstable and unstable_end > start


class TestRegime:
    def test_bands_on_a_compact_spectrum(self):
        compact = EpsilonSpectrum.from_modes([10.0, 12.0, 16.4], [0.3, 0.3, 0.4])
        r = EPCInference().regime(compact)
        assert r.band == "backprop-like" and not r.unstable
        assert r.f_max == pytest.approx(1.0 - (1.0 - 1e-3 * 16.4) ** 5)
        assert 0.0 < r.f_weighted < r.f_max
        r = EPCInference(eta_infer=0.05, infer_steps=100).regime(compact)
        assert r.band == "near PC equilibrium"
        assert EPCInference(eta_infer=0.02, infer_steps=5).regime(compact).band == (
            "partially relaxed"
        )

    def test_output_gradient_reversal(self):
        spectrum = EpsilonSpectrum.from_modes([10.0], [1.0])
        assert not (
            EPCInference(eta_infer=0.1, infer_steps=1)
            .regime(spectrum)
            .output_gradient_reverses
        )
        one_step = EPCInference(eta_infer=0.12, infer_steps=1).regime(spectrum)
        assert one_step.output_gradient_reverses and not one_step.unstable
        assert "reverses" in str(one_step)
        for eta in np.linspace(0.01, 0.199, 20):
            assert (
                not EPCInference(eta_infer=float(eta), infer_steps=2)
                .regime(spectrum)
                .output_gradient_reverses
            )

    def test_indefinite_and_no_positive_curvature(self):
        mild = EpsilonSpectrum.from_modes([-15.0, 1.1, 1.3], [0.2, 0.4, 0.4])
        strong = EPCInference(eta_infer=0.03, infer_steps=3).regime(mild)
        assert strong.growth_min == pytest.approx(1.45**3)
        assert str(strong).startswith("indefinite") and "20%" in str(strong)
        r = EPCInference().regime(EpsilonSpectrum.from_modes([-2.0, -1.0], [0.5, 0.5]))
        assert r.band == "no positive curvature" and math.isnan(r.f_weighted)


class PenaltyLinear(Linear):
    """Linear node whose ``forward`` adds a latent penalty to the energy."""

    ENERGY_FROM_FORWARD = True

    @staticmethod
    def forward(params, inputs, state, node_info):
        _, state = Linear.forward(params, inputs, state, node_info)
        penalty = 0.25 * jnp.sum(state.z_latent**2, axis=-1)
        state = state._replace(energy=state.energy + penalty)
        return jnp.sum(state.energy), state


class PenaltyLinearFastPath(PenaltyLinear):
    ENERGY_FROM_FORWARD = False


def penalty_chain(node_class):
    x = IdentityNode(shape=(4,), name="x")
    h = node_class(shape=(3,), name="h", activation=TanhActivation())
    y = Linear(shape=(2,), name="y")
    return graph(
        nodes=[x, h, y],
        edges=[
            Edge(source=x, target=h.slot("in")),
            Edge(source=h, target=y.slot("in")),
        ],
        task_map=TaskMap(x=x, y=y),
        inference=EPCInference(),
    )


class TestEnergyFromForward:
    def test_flag_keeps_extra_energy_terms(self, rng_key):
        assert StorkeyHopfield.ENERGY_FROM_FORWARD and not Linear.ENERGY_FROM_FORWARD
        clamps = {
            "x": jax.random.normal(rng_key, (3, 4)),
            "y": jax.random.normal(jax.random.PRNGKey(1), (3, 2)),
        }
        for node_class, matches in (
            (PenaltyLinear, True),
            (PenaltyLinearFastPath, False),
        ):
            structure = penalty_chain(node_class)
            params = initialize_params(structure, rng_key)
            state = initialize_graph_state(structure, 3, rng_key, clamps, params=params)
            state = perturb_free_latents(state, clamps, structure)
            synced = EPCInference.begin_segment(params, state, clamps, structure)
            derived = EPCInference.derive_states(params, synced, clamps, structure)
            reference = standard_state(params, state, clamps, structure)
            same = jnp.allclose(
                derived.nodes["h"].energy, reference.nodes["h"].energy, atol=1e-5
            )
            assert bool(same) == matches, node_class.__name__
            np.testing.assert_allclose(
                derived.nodes["h"].error, synced.nodes["h"].error, atol=0.0
            )


class TestNodeConventions:
    def test_linear_explicit_grad_readout_matches(self, rng_key):
        """An unclamped LinearExplicitGrad readout has zero energy on both
        paths, as the base forward_and_latent_grads convention requires."""
        from fabricpc.nodes.linear_explicit_grad import LinearExplicitGrad

        x = IdentityNode(shape=(4,), name="x")
        h = LinearExplicitGrad(shape=(3,), name="h")
        y = LinearExplicitGrad(shape=(2,), name="y")
        structure = graph(
            nodes=[x, h, y],
            edges=[
                Edge(source=x, target=h.slot("in")),
                Edge(source=h, target=y.slot("in")),
            ],
            task_map=TaskMap(x=x, y=y),
            inference=EPCInference(),
        )
        params = initialize_params(structure, rng_key)
        clamps = {"x": jax.random.normal(rng_key, (3, 4))}
        state = initialize_graph_state(structure, 3, rng_key, clamps, params=params)
        state = perturb_free_latents(state, clamps, structure)
        reference = standard_state(params, state, clamps, structure)
        assert jnp.all(reference.nodes["y"].energy == 0.0)
        synced = EPCInference.begin_segment(params, state, clamps, structure)
        derived = EPCInference.derive_states(params, synced, clamps, structure)
        assert_states_match(derived, reference, structure, 1e-5)


class TestStepMechanics:
    def test_free_source_relaxes(self, rng_key):
        """An unclamped source keeps its z_mu, takes z_latent = z_mu + ε, and
        its ε gradient is the state-based gradient of its latent."""
        eta = 0.05
        prior = IdentityNode(shape=(3,), name="prior")
        x = IdentityNode(shape=(4,), name="x")
        h = Linear(shape=(3,), name="h", activation=TanhActivation())
        y = Linear(shape=(2,), name="y")
        structure = graph(
            nodes=[prior, x, h, y],
            edges=[
                Edge(source=x, target=h.slot("in")),
                Edge(source=prior, target=h.slot("in")),
                Edge(source=h, target=y.slot("in")),
            ],
            task_map=TaskMap(x=x, y=y),
            inference=EPCInference(eta_infer=eta, infer_steps=1),
        )
        params = initialize_params(structure, rng_key)
        clamps = {
            "x": jax.random.normal(rng_key, (3, 4)),
            "y": jax.random.normal(jax.random.PRNGKey(1), (3, 2)),
        }
        state = initialize_graph_state(structure, 3, rng_key, clamps, params=params)
        state = perturb_free_latents(state, clamps, structure)
        assert "prior" in EPCInference._relaxed_errors(structure, clamps)
        synced = EPCInference.begin_segment(params, state, clamps, structure)
        eps_h = synced.nodes["h"].error
        z0 = state.nodes["prior"].z_latent
        h_params, y_params = params.nodes["h"], params.nodes["y"]

        def energy(z_prior):
            mu_h = jnp.tanh(
                clamps["x"] @ h_params.weights["x->h:in"]
                + z_prior @ h_params.weights["prior->h:in"]
                + h_params.biases["b"]
            )
            z_h = mu_h + eps_h
            mu_y = z_h @ y_params.weights["h->y:in"] + y_params.biases["b"]
            return 0.5 * jnp.sum(eps_h**2) + 0.5 * jnp.sum((clamps["y"] - mu_y) ** 2)

        expected_grad = jax.grad(energy)(z0)
        assert float(jnp.max(jnp.abs(expected_grad))) > 1e-3
        synced = EPCInference.zero_grads(params, synced, clamps, structure)
        with_grads = EPCInference.forward_value_and_grad(
            params, synced, clamps, structure
        )
        np.testing.assert_allclose(
            with_grads.nodes["prior"].latent_grad, expected_grad, atol=1e-5
        )
        final = run_inference(params, state, clamps, structure)
        np.testing.assert_allclose(final.nodes["prior"].z_mu, z0, atol=0.0)
        np.testing.assert_allclose(
            final.nodes["prior"].z_latent, z0 - eta * expected_grad, atol=1e-5
        )

    def test_latent_grad_accumulates_and_step_zeroes(self):
        structure = mlp("gaussian")
        params, clamps, state = mlp_inputs(structure, "gaussian")
        state = perturb_free_latents(state, clamps, structure)
        state = EPCInference.begin_segment(params, state, clamps, structure)
        sentinel = jnp.full_like(state.nodes["h1"].latent_grad, 1.75)
        dirty = state._replace(
            nodes={
                **state.nodes,
                "h1": state.nodes["h1"]._replace(latent_grad=sentinel),
            }
        )
        clean = EPCInference.zero_grads(params, state, clamps, structure)
        a = EPCInference.forward_value_and_grad(params, dirty, clamps, structure)
        b = EPCInference.forward_value_and_grad(params, clean, clamps, structure)
        np.testing.assert_allclose(
            a.nodes["h1"].latent_grad, b.nodes["h1"].latent_grad + 1.75, atol=1e-6
        )
        config = dict(structure.config["inference"].config)
        step_dirty = EPCInference.inference_step(
            params, dirty, clamps, structure, config
        )
        step_clean = EPCInference.inference_step(
            params, clean, clamps, structure, config
        )
        for name in ("h1", "h2"):
            np.testing.assert_allclose(
                step_dirty.nodes[name].error, step_clean.nodes[name].error, atol=0.0
            )

    def test_compute_new_error_decays_toward_feedforward(self):
        """ε·(1 − η·decay) − η·grad: decay pulls ε (not z) toward zero."""
        node = jnp.zeros((2, 3))
        eps = jnp.arange(6.0).reshape(2, 3)
        grad = jnp.ones((2, 3))
        from fabricpc.core.types import NodeState

        state = NodeState(
            z_latent=eps + 10.0,
            z_mu=node + 10.0,
            error=eps,
            energy=jnp.zeros(2),
            pre_activation=node,
            latent_grad=grad,
        )
        new = EPCInference.compute_new_error(
            "h", state, {"eta_infer": 0.1, "latent_decay": 0.5}
        )
        np.testing.assert_allclose(new, eps * 0.95 - 0.1 * grad)
        decayed = EPCInference.compute_new_error(
            "h",
            state._replace(latent_grad=node),
            {"eta_infer": 0.1, "latent_decay": 0.5},
        )
        assert float(jnp.max(jnp.abs(decayed))) < float(jnp.max(jnp.abs(eps)))


class TestTinyBertEquilibrium:
    def test_matches_state_based_pc(self):
        with jax.enable_x64(True):
            finals = {}
            for key, inference in (
                ("spc", InferenceSGD(eta_infer=0.2, infer_steps=5000)),
                ("epc", EPCInference(eta_infer=0.5, infer_steps=300)),
            ):
                structure = tiny_bert(inference, dropout_rate=0.0)
                params, clamps, state = bert_inputs(structure)
                finals[key] = jax.jit(
                    lambda p, s, st=structure: run_inference(p, s, clamps, st)
                )(params, state)
            spc, epc = finals["spc"], finals["epc"]
            assert_states_match(epc, spc, structure, 1e-7)
            reference = standard_state(params, epc, clamps, structure)
            for name in EPCInference._relaxed_errors(structure, clamps):
                grad = reference.nodes[name].latent_grad
                assert float(jnp.max(jnp.abs(grad))) < 1e-10, name
            g_spc = compute_local_weight_gradients(params, spc, structure)
            g_epc = compute_local_weight_gradients(params, epc, structure)
            for a, b in zip(
                jax.tree_util.tree_leaves(g_spc), jax.tree_util.tree_leaves(g_epc)
            ):
                np.testing.assert_allclose(a, b, atol=1e-7)


class TestRejections:
    def test_forward_updates_rejected(self):
        structure = tiny_bert(EPCInference(), dropout_rate=0.0)
        params = initialize_params(structure, jax.random.PRNGKey(3))
        for mode in ("initial", "final"):
            with pytest.raises(ValueError, match="EPCInference"):
                make_gradient_fn("pc", structure, mode, "standard")
            with pytest.raises(ValueError, match="EPCInference"):
                make_gradient_fn("pc", structure, mode, "spiking")
            with pytest.raises(ValueError, match="EPCInference"):
                get_graph_param_gradient_pcn_mlm(
                    params, bert_batch(), structure, jax.random.PRNGKey(0), mode
                )
        make_gradient_fn("pc", structure, "none", "standard")

    def test_incremental_pc_rejected(self):
        structure = tiny_bert(EPCInference(), dropout_rate=0.0)
        params = initialize_params(structure, jax.random.PRNGKey(3))
        optimizer = optax.sgd(1e-3)
        with pytest.raises(NotImplementedError, match="EPCInference"):
            _train_step_pcn_mlm_incremental_with_state(
                params,
                optimizer.init(params),
                bert_batch(),
                structure,
                optimizer,
                jax.random.PRNGKey(0),
            )

    def test_history_tracking_rejected(self):
        structure = mlp("gaussian")
        params, clamps, state = mlp_inputs(structure, "gaussian")
        for fn in (run_inference_with_history, run_inference_with_full_history):
            with pytest.raises(NotImplementedError, match="EPCInference"):
                fn(params, state, clamps, structure)
