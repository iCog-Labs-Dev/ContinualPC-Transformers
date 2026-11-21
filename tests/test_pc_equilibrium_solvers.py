"""Tests for the equilibrium solvers, Lanczos, escape test and acceptance gates."""

import math
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

from diagnostics.pc_equilibrium_solvers import (
    AcceptanceThresholds,
    Budget,
    LbfgsConfig,
    Objective,
    TimeAllocator,
    TrustRegionConfig,
    Work,
    _Counted,
    _truncated_cg,
    assess_reference,
    batch_status,
    compare_module_gradients,
    energy_margin,
    escape_test,
    lanczos_full_reorth,
    lanczos_memory_bytes,
    lbfgs,
    make_energy_functions,
    module_gradients,
    tree_spectrum,
    trust_region_newton_cg,
)
from fabricpc.core import EPCInference, InferenceSGD
from fabricpc.core.activations import IdentityActivation, TanhActivation
from fabricpc.core.initializers import NormalInitializer
from fabricpc.core.learning import compute_local_weight_gradients
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.nodes import Linear
from fabricpc.nodes.encoder_transformer import create_encoder_mlm_model
from fabricpc.nodes.identity import IdentityNode
from fabricpc.training.train_mlm import _prepare_pcn_mlm_batch

THRESHOLDS = AcceptanceThresholds()


def objective_of(energy):
    """Build an ``Objective`` from a scalar energy over a pytree."""
    grad = jax.grad(energy)

    def hvp(x, v):
        return jax.jvp(grad, (x,), (v,))[1]

    return Objective(
        value=jax.jit(energy),
        value_and_grad=jax.jit(jax.value_and_grad(energy)),
        hvp=jax.jit(hvp),
    )


def double_well(x):
    """Minimiser (±1, 0); negative curvature in the first coordinate for |x0| < 1/√3."""
    return 0.25 * (x["a"] ** 2 - 1.0) ** 2 + 0.5 * x["b"] ** 2


def tiny_bert(energy_scale=1.0):
    return create_encoder_mlm_model(
        depth=1,
        embed_dim=8,
        num_heads=2,
        mlp_dim=16,
        seq_len=6,
        vocab_size=20,
        inference=EPCInference(eta_infer=1e-3, infer_steps=1),
        use_attention_mask=True,
        dropout_rate=0.0,
        pc_output_energy_scale=energy_scale,
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


def bert_problem(energy_scale=1.0):
    structure = tiny_bert(energy_scale)
    key = jax.random.PRNGKey(3)
    params = initialize_params(structure, key)
    _, _, _, initial, clamps = _prepare_pcn_mlm_batch(
        params, bert_batch(), structure, key
    )
    state = EPCInference.begin_segment(params, initial, clamps, structure)
    return structure, params, state, clamps


def mlp_problem(seed=0, activation=TanhActivation):
    w_init = NormalInitializer(std=0.5)
    x = IdentityNode(shape=(4,), name="x")
    h1 = Linear(shape=(3,), name="h1", activation=activation(), weight_init=w_init)
    h2 = Linear(shape=(3,), name="h2", activation=activation(), weight_init=w_init)
    y = Linear(
        shape=(2,), name="y", activation=IdentityActivation(), weight_init=w_init
    )
    structure = graph(
        nodes=[x, h1, h2, y],
        edges=[
            Edge(source=x, target=h1.slot("in")),
            Edge(source=h1, target=h2.slot("in")),
            Edge(source=h2, target=y.slot("in")),
        ],
        task_map=TaskMap(x=x, y=y),
        inference=EPCInference(eta_infer=0.05, infer_steps=1),
    )
    key = jax.random.PRNGKey(seed)
    params = initialize_params(structure, key)
    clamps = {
        "x": jax.random.normal(jax.random.PRNGKey(seed + 1), (3, 4)),
        "y": jax.random.normal(jax.random.PRNGKey(seed + 2), (3, 2)),
    }
    state = initialize_graph_state(structure, 3, key, clamps, params=params)
    state = EPCInference.begin_segment(params, state, clamps, structure)
    return structure, params, state, clamps


def zero_errors(structure, params, state, clamps):
    return jax.tree_util.tree_map(
        jnp.zeros_like,
        EPCInference.error_energy(params, state, clamps, structure)[1],
    )


@pytest.mark.parametrize("solver", [trust_region_newton_cg, lbfgs])
def test_known_answer_quadratic(solver):
    with jax.enable_x64(True):
        matrix = jnp.array([[4.0, 1.0, 0.0], [1.0, 3.0, 0.5], [0.0, 0.5, 2.0]])
        target = jnp.array([1.0, -2.0, 0.5])

        def energy(x):
            return 0.5 * x["v"] @ matrix @ x["v"] - target @ x["v"]

        result = solver(objective_of(energy), {"v": jnp.zeros(3)})
        assert result.status == "converged"
        assert result.rel_stationarity <= 1e-6
        np.testing.assert_allclose(
            result.x["v"], np.linalg.solve(matrix, target), atol=1e-5
        )


def test_trust_region_exits_on_negative_curvature_and_reaches_minimiser():
    with jax.enable_x64(True):
        x0 = {"a": jnp.array(0.1), "b": jnp.array(0.2)}
        hessian = jax.hessian(lambda a: double_well({"a": a, "b": 0.0}))(x0["a"])
        assert float(hessian) < 0.0
        result = trust_region_newton_cg(
            objective_of(double_well), x0, TrustRegionConfig(initial_radius=0.25)
        )
        assert result.status == "converged"
        assert result.work.negative_curvature_exits >= 1
        assert float(result.x["a"]) == pytest.approx(1.0, abs=1e-6)
        assert float(result.x["b"]) == pytest.approx(0.0, abs=1e-6)


def test_truncated_cg_on_an_indefinite_model_stops_on_the_boundary():
    with jax.enable_x64(True):
        hessian = jnp.array([[1.0, 0.0], [0.0, -2.0]])
        g = {"v": jnp.array([1.0, 1.0])}

        def energy(x):
            return g["v"] @ x["v"] + 0.5 * x["v"] @ hessian @ x["v"]

        work = Work()
        step, predicted, on_boundary, finite = _truncated_cg(
            _Counted(objective_of(energy), work),
            {"v": jnp.zeros(2)},
            g,
            float(jnp.linalg.norm(g["v"])),
            float(jnp.linalg.norm(g["v"])),
            0.75,
            TrustRegionConfig(),
            Budget(),
        )
        z = step["v"]
        assert finite and on_boundary
        assert work.negative_curvature_exits == 1
        assert float(jnp.linalg.norm(z)) == pytest.approx(0.75, rel=1e-12)
        direct = -(g["v"] @ z + 0.5 * z @ hessian @ z)
        assert predicted == pytest.approx(float(direct), rel=1e-12)
        assert float(z[0]) == pytest.approx(float(z[1]))
        assert float(z[0]) < 0.0


def test_lbfgs_descends_from_negative_curvature():
    with jax.enable_x64(True):
        x0 = {"a": jnp.array(0.1), "b": jnp.array(0.2)}
        result = lbfgs(objective_of(double_well), x0)
        assert result.status == "converged"
        assert float(result.x["a"]) == pytest.approx(1.0, abs=1e-5)
        values = [row["value"] for row in result.trace]
        assert all(b <= a + 1e-12 for a, b in zip(values, values[1:]))


def test_exact_counters_on_a_toy():
    with jax.enable_x64(True):

        def energy(x):
            return 0.5 * jnp.sum(x["v"] ** 2) - jnp.sum(x["v"]) * 0.1

        objective = objective_of(energy)
        start = {"v": jnp.zeros(4)}
        trust = trust_region_newton_cg(objective, start)
        assert trust.status == "converged"
        assert (trust.work.gradient_evals, trust.work.hvps) == (2, 1)
        assert trust.work.energy_evals == 3
        assert trust.work.passes == 4
        assert (trust.work.accepted_steps, trust.work.rejected_steps) == (1, 0)
        quasi = lbfgs(objective, start)
        assert quasi.status == "converged"
        assert quasi.work.gradient_evals == 2
        assert quasi.work.energy_evals == 2
        assert quasi.work.passes == 2
        assert quasi.work.hvps == 0
        assert (quasi.work.accepted_steps, quasi.work.rejected_steps) == (1, 0)


def test_lbfgs_counts_each_rejected_trial_once():
    with jax.enable_x64(True):

        def energy(x):
            return 0.5 * jnp.sum(x["v"] ** 2)

        result = lbfgs(
            objective_of(energy),
            {"v": jnp.array([0.5])},
            LbfgsConfig(curvature=0.1, max_line_search=3),
        )
        accepted_passes = result.work.accepted_steps
        trials = result.work.gradient_evals - 1
        assert result.work.rejected_steps == trials - accepted_passes


def test_lbfgs_skips_a_pair_that_fails_the_curvature_condition():
    with jax.enable_x64(True):
        result = lbfgs(
            objective_of(double_well),
            {"a": jnp.array(0.1), "b": jnp.array(0.0)},
            LbfgsConfig(max_line_search=1),
            Budget(max_passes=50),
        )
        assert result.work.skipped_pairs >= 1


@pytest.mark.parametrize("solver", [trust_region_newton_cg, lbfgs])
def test_budget_censors_and_counts_work(solver):
    with jax.enable_x64(True):
        result = solver(
            objective_of(double_well),
            {"a": jnp.array(0.1), "b": jnp.array(0.2)},
            budget=Budget(max_passes=3),
        )
        assert result.status == "censored_budget"
        assert result.work.passes >= 3
        assert result.work.gradient_evals >= 1
        assert result.work.energy_evals >= 1


def test_expired_deadline_censors_immediately():
    with jax.enable_x64(True):
        result = lbfgs(
            objective_of(double_well),
            {"a": jnp.array(0.1), "b": jnp.array(0.2)},
            budget=Budget(deadline=0.0),
        )
        assert result.status == "censored_budget"


def test_trust_region_counts_rejected_steps():
    with jax.enable_x64(True):
        result = trust_region_newton_cg(
            objective_of(double_well),
            {"a": jnp.array(0.1), "b": jnp.array(0.2)},
            TrustRegionConfig(initial_radius=50.0, max_radius=50.0),
        )
        assert result.status == "converged"
        assert result.work.rejected_steps >= 1


@pytest.mark.parametrize("solver", [trust_region_newton_cg, lbfgs])
def test_energy_noise_is_reported_as_a_stall_not_as_convergence(solver):
    with jax.enable_x64(True):

        def smooth(x):
            return 0.5 * jnp.sum(x["v"] ** 2) - jnp.sum(x["v"])

        grad = jax.grad(smooth)

        def noisy_value(x):
            return float(np.round(float(smooth(x)), 2))

        def value_and_grad(x):
            return noisy_value(x), grad(x)

        def hvp(x, v):
            return jax.jvp(grad, (x,), (v,))[1]

        objective = Objective(noisy_value, value_and_grad, hvp)
        result = solver(
            objective,
            {"v": jnp.full((4,), 5.0)},
            budget=Budget(max_passes=2000),
        )
        if result.status == "converged":
            assert result.rel_stationarity <= 1e-6
        else:
            assert result.status in ("stalled", "censored_budget")
            summary = solve_summary(
                status=result.status, stationarity=result.rel_stationarity
            )
            assert (
                assess_reference(summary, summary, True, THRESHOLDS, "passed")["status"]
                != "accepted"
            )


def test_energy_margin_scales_with_the_energy_and_not_with_one():
    with jax.enable_x64(True):
        assert energy_margin(1e-6) < 1e-20
        assert energy_margin(1e6) == pytest.approx(10.0 * 2.220446e-16 * 1e6, rel=1e-3)
        assert energy_margin(0.0) == 0.0


def test_time_allocator_computes_each_deadline_when_the_unit_starts():
    now = [0.0]
    allocator = TimeAllocator(100.0, 4, clock=lambda: now[0])
    assert allocator.next_deadline() == pytest.approx(25.0)
    now[0] = 1.0
    second = allocator.next_deadline()
    assert second == pytest.approx(1.0 + 99.0 / 3.0)
    now[0] = 80.0
    third = allocator.next_deadline()
    assert third == pytest.approx(80.0 + 20.0 / 2.0)
    assert not allocator.expired()
    now[0] = 100.0
    assert allocator.expired()
    assert allocator.next_deadline() == pytest.approx(100.0)


def test_traced_scale_matches_the_rebuilt_structure():
    with jax.enable_x64(True):
        structure, params, state, clamps = bert_problem()
        scaled, scaled_params, scaled_state, scaled_clamps = bert_problem(0.25)
        errors = jax.tree_util.tree_map(
            lambda e: 0.01 * jnp.ones_like(e),
            EPCInference.error_energy(params, state, clamps, structure)[1],
        )
        fns = make_energy_functions(structure)
        energy_of, _ = EPCInference.error_energy(
            scaled_params, scaled_state, scaled_clamps, scaled
        )
        assert float(fns.value(errors, params, state, clamps, 0.25)) == pytest.approx(
            float(energy_of(errors)[0]), rel=1e-12
        )
        derived = fns.derived(errors, params, state, clamps, 0.25)
        expected = compute_local_weight_gradients(
            scaled_params, energy_of(errors)[1], scaled
        )
        actual = fns.weight_gradients(params, derived, 0.25)
        for name in structure.nodes:
            for got, want in zip(
                jax.tree_util.tree_leaves(actual.nodes[name]),
                jax.tree_util.tree_leaves(expected.nodes[name]),
            ):
                np.testing.assert_allclose(got, want, atol=1e-12)


def test_hvp_matches_dense_hessian_on_tiny_encoder():
    with jax.enable_x64(True):
        structure, params, state, clamps = bert_problem()
        fns = make_energy_functions(structure)
        errors = zero_errors(structure, params, state, clamps)
        flat, unravel = ravel_pytree(errors)
        dense = jax.hessian(
            lambda v: fns.value(unravel(v), params, state, clamps, 1.0)
        )(flat)
        np.testing.assert_allclose(dense, dense.T, atol=1e-8)
        direction = jax.random.normal(jax.random.PRNGKey(0), flat.shape, flat.dtype)
        hvp = fns.hvp(errors, unravel(direction), params, state, clamps, 1.0)
        np.testing.assert_allclose(ravel_pytree(hvp)[0], dense @ direction, atol=1e-8)


def test_energy_at_small_scale_is_free_of_cancellation_noise():
    with jax.enable_x64(True):
        structure, params, state, clamps = bert_problem()
        fns = make_energy_functions(structure)
        errors = zero_errors(structure, params, state, clamps)
        s = 1e-8
        value = float(fns.value(errors, params, state, clamps, s))
        derived = fns.derived(errors, params, state, clamps, s)
        head = float(jnp.sum(derived.nodes[structure.task_map["y"]].energy))
        assert value == pytest.approx(s * head, rel=1e-13)


def test_derivation_leaves_clamps_unchanged():
    with jax.enable_x64(True):
        structure, params, state, clamps = mlp_problem()
        fns = make_energy_functions(structure)
        errors = jax.tree_util.tree_map(
            lambda e: 0.1 * jnp.ones_like(e),
            zero_errors(structure, params, state, clamps),
        )
        derived = fns.derived(errors, params, state, clamps, 0.5)
        assert set(errors) == {"h1", "h2"}
        for name, clamp in clamps.items():
            np.testing.assert_array_equal(derived.nodes[name].z_latent, clamp)
            assert name not in errors


@pytest.mark.parametrize("solver", [trust_region_newton_cg, lbfgs])
def test_iterates_respect_the_energy_bound(solver):
    with jax.enable_x64(True):
        structure, params, state, clamps = bert_problem()
        fns = make_energy_functions(structure)
        errors = zero_errors(structure, params, state, clamps)
        objective = fns.objective(params, state, clamps, 1.0)
        energy_start = float(objective.value(errors))
        result = solver(objective, errors, budget=Budget(max_passes=400))
        assert result.max_norm_sq <= 2.0 * energy_start * (1.0 + 1e-9)
        assert result.value <= energy_start


def test_equilibrium_is_stationary_in_state_coordinates():
    with jax.enable_x64(True):
        structure, params, state, clamps = mlp_problem()
        fns = make_energy_functions(structure)
        errors = zero_errors(structure, params, state, clamps)
        result = trust_region_newton_cg(
            fns.objective(params, state, clamps, 1.0),
            errors,
            TrustRegionConfig(tolerance=1e-10),
        )
        assert result.status == "converged"
        derived = fns.derived(result.x, params, state, clamps, 1.0)
        zeroed = InferenceSGD.zero_grads(params, derived, clamps, structure)
        graded = InferenceSGD.forward_value_and_grad(params, zeroed, clamps, structure)
        for name in result.x:
            np.testing.assert_allclose(graded.nodes[name].latent_grad, 0.0, atol=1e-7)


def test_linear_graph_matches_the_closed_form():
    with jax.enable_x64(True):
        structure, params, state, clamps = mlp_problem(activation=IdentityActivation)
        fns = make_energy_functions(structure)
        errors = zero_errors(structure, params, state, clamps)
        result = trust_region_newton_cg(
            fns.objective(params, state, clamps, 1.0),
            errors,
            TrustRegionConfig(tolerance=1e-12),
        )
        assert result.status == "converged"
        w2 = next(iter(params.nodes["h2"].weights.values()))
        w3 = next(iter(params.nodes["y"].weights.values()))
        j1 = np.asarray((w2 @ w3).T)
        j2 = np.asarray(w3.T)
        jacobian = np.concatenate([j1, j2], axis=1)
        residual = np.asarray(clamps["y"] - state.nodes["y"].z_mu)
        s_matrix = np.eye(2) + jacobian @ jacobian.T
        output = np.linalg.solve(s_matrix, residual.T).T
        np.testing.assert_allclose(result.x["h1"], output @ j1, atol=1e-9)
        np.testing.assert_allclose(result.x["h2"], output @ j2, atol=1e-9)

        derived = fns.derived(result.x, params, state, clamps, 1.0)
        grads = fns.weight_gradients(params, derived, 1.0)
        z1 = np.asarray(derived.nodes["h1"].z_latent)
        z2 = np.asarray(derived.nodes["h2"].z_latent)
        expected = {
            "h1": -np.asarray(clamps["x"]).T @ np.asarray(result.x["h1"]),
            "h2": -z1.T @ np.asarray(result.x["h2"]),
            "y": -z2.T @ output,
        }
        for name, want in expected.items():
            got = next(iter(grads.nodes[name].weights.values()))
            np.testing.assert_allclose(got, want, atol=1e-8)


def gradients(**modules):
    return {name: np.asarray(value, dtype=float) for name, value in modules.items()}


def test_compare_gates_on_cosine_and_norm():
    ref = gradients(a=[1.0, 0.0], b=[0.0, 2.0])
    same = compare_module_gradients(ref, ref, THRESHOLDS)
    assert same["passed"] and same["modules"]["a"]["cosine"] == pytest.approx(1.0)
    rotated = compare_module_gradients(
        ref, gradients(a=[0.0, 1.0], b=[0.0, 2.0]), THRESHOLDS
    )
    assert not rotated["passed"]
    longer = compare_module_gradients(
        ref, gradients(a=[1.2, 0.0], b=[0.0, 2.0]), THRESHOLDS
    )
    assert not longer["passed"]
    assert longer["modules"]["a"]["rel_norm_error"] == pytest.approx(0.2)


def test_zero_norm_modules_get_null_cosine_and_are_listed_as_excluded():
    ref = gradients(a=[1.0, 0.0], zero=[0.0, 0.0], small=[1e-9, 0.0])
    other = gradients(a=[1.0, 0.0], zero=[0.0, 0.0], small=[0.0, 1e-9])
    result = compare_module_gradients(ref, other, THRESHOLDS)
    assert result["modules"]["zero"]["cosine"] is None
    assert result["modules"]["zero"]["gated"] is False
    assert result["modules"]["small"]["gated"] is False
    assert result["excluded"] == ["zero", "small"]
    assert result["passed"]


def test_module_floor_is_symmetric():
    first = gradients(a=[1.0, 0.0], b=[1e-9, 0.0])
    second = gradients(a=[1.0, 0.0], b=[0.0, 5.0])
    forward = compare_module_gradients(first, second, THRESHOLDS)
    backward = compare_module_gradients(second, first, THRESHOLDS)
    assert forward["modules"]["b"]["gated"] and backward["modules"]["b"]["gated"]
    assert not forward["passed"]
    assert not backward["passed"]


def test_gated_module_with_zero_gradient_on_either_side_fails():
    full = gradients(a=[1.0, 0.0])
    zero = gradients(a=[0.0, 0.0])
    for ref, other in ((full, zero), (zero, full)):
        result = compare_module_gradients(ref, other, THRESHOLDS)
        assert result["modules"]["a"]["cosine"] is None
        assert not result["passed"]


def test_all_zero_reference_does_not_pass():
    ref = gradients(a=[0.0, 0.0])
    assert not compare_module_gradients(ref, ref, THRESHOLDS)["passed"]


def test_module_gradients_skip_nodes_without_parameters():
    with jax.enable_x64(True):
        structure, params, state, clamps = mlp_problem()
        fns = make_energy_functions(structure)
        derived = fns.derived(
            zero_errors(structure, params, state, clamps), params, state, clamps, 1.0
        )
        modules = module_gradients(
            fns.weight_gradients(params, derived, 1.0), structure
        )
        assert set(modules) == {"h1", "h2", "y"}
        assert all(value.ndim == 1 for value in modules.values())


def solve_summary(
    status="converged",
    stationarity=1e-8,
    lam_min=1.0,
    lam_max=100.0,
    residual=0.0,
    energy=1.0,
    energy_start=2.0,
    max_norm_sq=1.0,
    finite=True,
    censored=False,
):
    return {
        "status": status,
        "rel_stationarity": stationarity,
        "energy": energy,
        "energy_start": energy_start,
        "max_norm_sq": max_norm_sq,
        "lambda_min": lam_min,
        "lambda_max": lam_max,
        "residual_min": residual,
        "residual_max": residual,
        "spectrum_finite": finite,
        "spectrum_censored": censored,
    }


@pytest.mark.parametrize(
    ("first", "second", "agree", "escape", "expected"),
    [
        (solve_summary(), solve_summary(), True, "passed", "accepted"),
        (solve_summary(), solve_summary(), False, "passed", "solver_disagreement"),
        (
            solve_summary(energy=1.0),
            solve_summary(energy=1.0 + 2e-6),
            True,
            "passed",
            "solver_disagreement",
        ),
        (
            solve_summary(status="censored_budget"),
            solve_summary(),
            True,
            "passed",
            "censored_budget",
        ),
        (
            solve_summary(status="nonfinite"),
            solve_summary(status="censored_budget"),
            True,
            None,
            "nonfinite",
        ),
        (
            solve_summary(status="stalled"),
            solve_summary(),
            True,
            "passed",
            "not_stationary",
        ),
        (
            solve_summary(stationarity=1e-3),
            solve_summary(),
            True,
            "passed",
            "not_stationary",
        ),
        (
            solve_summary(stationarity=float("nan")),
            solve_summary(),
            True,
            "passed",
            "not_stationary",
        ),
        (
            solve_summary(max_norm_sq=4.1),
            solve_summary(),
            True,
            "passed",
            "bound_violation",
        ),
        (solve_summary(lam_min=-1.0), solve_summary(), True, "passed", "saddle"),
        (solve_summary(), solve_summary(lam_min=-1.0), True, "passed", "saddle"),
        (
            solve_summary(lam_min=-0.02, lam_max=1e6),
            solve_summary(),
            True,
            "passed",
            "saddle",
        ),
        (solve_summary(lam_min=-0.005), solve_summary(), True, "passed", "accepted"),
        (solve_summary(residual=50.0), solve_summary(), True, "passed", "accepted"),
        (
            solve_summary(lam_min=float("nan")),
            solve_summary(),
            True,
            "passed",
            "curvature_unresolved",
        ),
        (
            solve_summary(lam_max=float("inf")),
            solve_summary(),
            True,
            "passed",
            "curvature_unresolved",
        ),
        (
            solve_summary(residual=float("nan")),
            solve_summary(),
            True,
            "passed",
            "curvature_unresolved",
        ),
        (
            solve_summary(lam_min=None),
            solve_summary(),
            True,
            "passed",
            "curvature_unresolved",
        ),
        (
            solve_summary(finite=False),
            solve_summary(),
            True,
            "passed",
            "curvature_unresolved",
        ),
        (
            solve_summary(censored=True),
            solve_summary(),
            True,
            "passed",
            "censored_budget",
        ),
        (solve_summary(), solve_summary(), True, "saddle", "saddle"),
        (solve_summary(), solve_summary(), True, "censored", "censored_budget"),
        (solve_summary(), solve_summary(), True, "unresolved", "curvature_unresolved"),
        (solve_summary(), solve_summary(), True, None, "curvature_unresolved"),
    ],
)
def test_reference_status(first, second, agree, escape, expected):
    verdict = assess_reference(first, second, agree, THRESHOLDS, escape)
    assert verdict["status"] == expected


def test_curvature_evidence_separates_lanczos_from_the_escape_test():
    summary = solve_summary()
    without = assess_reference(summary, summary, True, THRESHOLDS, None)
    with_escape = assess_reference(summary, summary, True, THRESHOLDS, "passed")
    assert without["curvature_evidence"] == "no_negative_mode_found"
    assert without["status"] != "accepted"
    assert with_escape["curvature_evidence"] == "escape_test_passed"


def test_one_flipped_sample_of_64_fails_though_the_sum_passes():
    rng = np.random.default_rng(0)
    samples = [rng.normal(size=1000) + 3.0 for _ in range(64)]
    flipped = [*samples[:-1], -samples[-1]]
    summed = compare_module_gradients(
        {"m": sum(samples)}, {"m": sum(flipped)}, THRESHOLDS
    )
    assert summed["modules"]["m"]["cosine"] > 0.99
    statuses = []
    for reference, other in zip(samples, flipped):
        agree = compare_module_gradients({"m": reference}, {"m": other}, THRESHOLDS)
        summary = solve_summary()
        statuses.append(
            assess_reference(summary, summary, agree["passed"], THRESHOLDS, "passed")[
                "status"
            ]
        )
    assert statuses.count("solver_disagreement") == 1
    assert batch_status(statuses, True) == "solver_disagreement"


@pytest.mark.parametrize(
    ("statuses", "summed", "expected"),
    [
        (["accepted", "accepted"], True, "accepted"),
        (["accepted", "zero_signal"], True, "accepted"),
        (["zero_signal", "zero_signal"], True, "zero_signal"),
        (["accepted", "accepted"], False, "solver_disagreement"),
        (["accepted", "accepted"], None, "curvature_unresolved"),
        (["accepted", "saddle"], True, "saddle"),
        (["censored_budget", "saddle"], True, "saddle"),
        (["censored_budget", "accepted"], True, "censored_budget"),
        (["not_stationary", "censored_budget"], True, "not_stationary"),
        (["bound_violation", "accepted"], True, "bound_violation"),
    ],
)
def test_batch_status(statuses, summed, expected):
    assert batch_status(statuses, summed) == expected


def test_lanczos_matches_dense_eigenvalues_and_reports_the_lowest_vector():
    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.normal(size=(60, 60)))
    values = np.concatenate([[-3.0], np.linspace(0.5, 40.0, 59)])
    matrix = q @ np.diag(values) @ q.T
    with jax.enable_x64(True):
        result = lanczos_full_reorth(lambda v: matrix @ v, rng.normal(size=60), 60)
    assert result.finite and not result.censored
    assert result.lambda_min == pytest.approx(-3.0, abs=1e-8)
    assert result.lambda_max == pytest.approx(40.0, abs=1e-8)
    np.testing.assert_allclose(result.ritz_values, values, atol=1e-7)
    assert result.ritz_weights.sum() == pytest.approx(1.0)
    vector = result.lowest_vector
    np.testing.assert_allclose(matrix @ vector, -3.0 * vector, atol=1e-7)
    assert lanczos_memory_bytes(60, 200) == 8 * 60 * 60


def test_lanczos_keeps_the_basis_orthogonal_past_where_plain_lanczos_fails():
    values = np.concatenate([np.geomspace(1.0, 1e6, 40), np.linspace(0.8, 1.2, 1000)])
    start = np.random.default_rng(0).normal(size=values.size)
    with jax.enable_x64(True):
        result = lanczos_full_reorth(lambda v: values * v, start, 150)
    distinct = np.sum(np.diff(result.ritz_values) < 1e-8)
    assert distinct == 0
    assert result.lambda_max == pytest.approx(1e6, rel=1e-8)


def test_lanczos_reports_nonfinite_breakdown_and_deadline():
    start = np.ones(5)
    nan = lanczos_full_reorth(lambda v: np.full(5, np.nan), start, 5)
    assert not nan.finite and nan.lambda_min is None
    identity = lanczos_full_reorth(lambda v: v, start, 5)
    assert identity.breakdown and identity.iterations == 1
    assert identity.lambda_min == pytest.approx(1.0)
    late = lanczos_full_reorth(lambda v: v, start, 5, deadline=0.0)
    assert late.censored
    assert not lanczos_full_reorth(lambda v: v, np.zeros(5), 5).finite


def embedded_saddle(dimension=20000, negative=-20.0):
    """A quadratic with one negative mode that the gradient never excites."""
    rng = np.random.default_rng(0)
    spectrum = rng.uniform(0.8, 1.5, dimension)
    spectrum[:50] = np.geomspace(2.0, 7e3, 50)
    spectrum[-1] = negative
    target = rng.normal(size=dimension)
    target[-1] = 0.0
    spectrum_j, target_j = jnp.asarray(spectrum), jnp.asarray(target)

    def energy(x):
        v = x["v"]
        return (
            0.5 * jnp.sum(spectrum_j * v * v) - target_j @ v + 25.0 * v[-1] ** 4 + 1e5
        )

    return objective_of(energy), {"v": jnp.zeros(dimension)}


def solve_and_measure(objective, start, iterations):
    summaries = []
    solved = {}
    for name, solver, config in (
        (
            "trust_region",
            trust_region_newton_cg,
            TrustRegionConfig(max_radius=1e4),
        ),
        ("lbfgs", lbfgs, LbfgsConfig()),
    ):
        result = solver(objective, start, config, Budget(max_passes=3000))
        assert result.status == "converged"
        assert float(result.x["v"][-1]) == 0.0
        spectrum = tree_spectrum(
            objective,
            result.x,
            {"v": jax.random.normal(jax.random.PRNGKey(1), (start["v"].size,))},
            iterations,
        )
        solved[name] = (result, spectrum)
        summaries.append(
            solve_summary(
                stationarity=result.rel_stationarity,
                lam_min=spectrum.lambda_min,
                lam_max=spectrum.lambda_max,
                residual=spectrum.residual_min,
                energy=result.value,
                energy_start=result.value + 1e9,
                max_norm_sq=result.max_norm_sq,
            )
        )
    return summaries, solved


def test_embedded_saddle_is_found_by_lanczos_end_to_end():
    with jax.enable_x64(True):
        objective, start = embedded_saddle()
        summaries, solved = solve_and_measure(objective, start, 200)
        assert solved["trust_region"][1].lambda_min == pytest.approx(-20.0, abs=1e-3)
        verdict = assess_reference(*summaries, True, THRESHOLDS, "passed")
        assert verdict["status"] == "saddle"


def test_escape_test_catches_a_saddle_that_lanczos_missed():
    with jax.enable_x64(True):
        objective, start = embedded_saddle()
        summaries, solved = solve_and_measure(objective, start, 3)
        result, spectrum = solved["trust_region"]
        assert spectrum.lambda_min > 0.0
        assert (
            assess_reference(*summaries, True, THRESHOLDS, "passed")["status"]
            == "accepted"
        )
        escape = escape_test(
            objective,
            result.x,
            result.value,
            spectrum.lowest_vector,
            1e-6 * result.initial_grad_norm,
            lambda point: True,
            THRESHOLDS,
            1e4,
            Budget(max_passes=3000),
            0,
        )
        assert escape.outcome == "saddle"
        assert escape.energy < result.value
        assert (
            assess_reference(*summaries, True, THRESHOLDS, escape.outcome)["status"]
            == "saddle"
        )


def test_escape_test_passes_at_a_true_minimum():
    with jax.enable_x64(True):
        objective = objective_of(double_well)
        start = {"a": jnp.array(0.1), "b": jnp.array(0.2)}
        result = trust_region_newton_cg(objective, start, TrustRegionConfig())
        spectrum = tree_spectrum(
            objective, result.x, {"a": jnp.array(1.0), "b": jnp.array(1.0)}, 2
        )
        escape = escape_test(
            objective,
            result.x,
            result.value,
            spectrum.lowest_vector,
            1e-6 * result.initial_grad_norm,
            lambda point: True,
            THRESHOLDS,
            10.0,
            Budget(max_passes=500),
            0,
        )
        assert escape.outcome == "passed"
        assert escape.perturbation_norm == pytest.approx(0.05 * 1.0, rel=1e-3)
        differing = escape_test(
            objective,
            result.x,
            result.value,
            spectrum.lowest_vector,
            1e-6 * result.initial_grad_norm,
            lambda point: False,
            THRESHOLDS,
            10.0,
            Budget(max_passes=500),
            0,
        )
        assert differing.outcome == "saddle"


def test_escape_test_with_a_perturbation_below_the_target_is_unresolved():
    with jax.enable_x64(True):
        objective = objective_of(double_well)
        escape = escape_test(
            objective,
            {"a": jnp.array(1.0), "b": jnp.array(0.0)},
            0.0,
            None,
            math.inf,
            lambda point: True,
            THRESHOLDS,
            10.0,
            Budget(),
            0,
        )
        assert escape.outcome == "unresolved"
