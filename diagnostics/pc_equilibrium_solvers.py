"""Second-order descent solvers and acceptance gates for the PC error energy.

The energy is E_s(ε) = ½Σ‖ε‖² + s·L(z(ε)), with the latents z derived from the
errors by ``EPCInference``. Samples are decoupled in E_s, so each is solved on
its own with batch size 1, and every gate applies per sample.

The solvers take any scalar objective over a pytree: trust-region Newton-CG with
a negative-curvature exit to the boundary, and L-BFGS with a weak Wolfe line
search. Curvature at a solution is measured by fully reorthogonalised Lanczos.
Lanczos can miss a mode, so an accepted point must also pass an escape test: a
perturbation along the lowest Ritz vector plus random noise, then a re-solve.

The head scale s is a traced argument on a graph built at s = 1: the head energy
and head weight gradient are multiplied by s, which matches a graph built with
``pc_output_energy_scale=s`` and shares one compilation across scales.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from fabricpc.core.inference_epc import EPCInference
from fabricpc.core.learning import compute_local_weight_gradients
from fabricpc.core.types import GraphParams, GraphState, GraphStructure, NodeParams

Pytree = Any

CONVERGED = "converged"
CENSORED = "censored_budget"
STALLED = "stalled"
NONFINITE = "nonfinite"

ACCEPTED = "accepted"
ZERO_SIGNAL = "zero_signal"
NOT_STATIONARY = "not_stationary"
BOUND_VIOLATION = "bound_violation"
SADDLE = "saddle"
CURVATURE_UNRESOLVED = "curvature_unresolved"
SOLVER_DISAGREEMENT = "solver_disagreement"

EVIDENCE_NONE = "not_measured"
EVIDENCE_LANCZOS = "no_negative_mode_found"
EVIDENCE_ESCAPE = "escape_test_passed"

ESCAPE_PASSED = "passed"
ESCAPE_SADDLE = "saddle"
ESCAPE_UNRESOLVED = "unresolved"
ESCAPE_CENSORED = "censored"

REFUTING_STATUSES = (SADDLE, BOUND_VIOLATION, SOLVER_DISAGREEMENT)
UNRESOLVED_STATUSES = (NONFINITE, NOT_STATIONARY, CURVATURE_UNRESOLVED, CENSORED)
SATISFIED_STATUSES = (ACCEPTED, ZERO_SIGNAL)


def tree_dot(a: Pytree, b: Pytree) -> float:
    """Inner product of two pytrees."""
    return float(
        sum(
            jnp.sum(x * y)
            for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b))
        )
    )


def tree_norm(a: Pytree) -> float:
    """Euclidean norm of a pytree; NaN if any entry is non-finite."""
    dot = tree_dot(a, a)
    return math.sqrt(dot) if dot >= 0.0 else math.nan


def tree_axpy(alpha: float, x: Pytree, y: Pytree) -> Pytree:
    """Return alpha·x + y."""
    return jax.tree_util.tree_map(lambda xi, yi: alpha * xi + yi, x, y)


def tree_scale(alpha: float, x: Pytree) -> Pytree:
    """Return alpha·x."""
    return jax.tree_util.tree_map(lambda xi: alpha * xi, x)


@dataclass(frozen=True)
class Objective:
    """A scalar objective over a pytree and its derivatives."""

    value: Callable[[Pytree], jax.Array]
    value_and_grad: Callable[[Pytree], tuple[jax.Array, Pytree]]
    hvp: Callable[[Pytree, Pytree], Pytree]


@dataclass
class Work:
    """Solver work. A pass is one compiled value, value-and-gradient or HVP call.

    ``rejected_steps`` counts trial points not taken as the next iterate, for
    both solvers. ``skipped_pairs`` counts L-BFGS pairs dropped for failing the
    curvature condition.
    """

    energy_evals: int = 0
    gradient_evals: int = 0
    hvps: int = 0
    rejected_steps: int = 0
    accepted_steps: int = 0
    negative_curvature_exits: int = 0
    history_resets: int = 0
    skipped_pairs: int = 0
    passes: int = 0


@dataclass(frozen=True)
class Budget:
    """Stop limits: compiled passes, and a ``time.monotonic`` deadline."""

    max_passes: int | None = None
    deadline: float | None = None

    def exhausted(self, work: Work) -> bool:
        """Whether either limit has been reached."""
        if self.max_passes is not None and work.passes >= self.max_passes:
            return True
        return self.deadline is not None and time.monotonic() >= self.deadline


class TimeAllocator:
    """Shares a wall-clock end time over a known count of remaining work units.

    Each deadline is computed when its unit starts, as an equal share of what
    is left, so time a unit doesn't use rolls over and a slow unit is charged to
    its own window.
    """

    def __init__(
        self,
        end: float,
        units: int,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.end = end
        self.remaining = max(units, 1)
        self.clock = clock

    def expired(self) -> bool:
        """Whether the end time has passed."""
        return self.clock() >= self.end

    def next_deadline(self) -> float:
        """Start the next unit and return its deadline."""
        now = self.clock()
        share = max(self.end - now, 0.0) / self.remaining
        self.remaining = max(self.remaining - 1, 1)
        return now + share


@dataclass
class SolveResult:
    """Final point and the record of one solve.

    ``max_norm_sq`` is the largest ‖x‖² over accepted iterates; ``trace`` has
    one row per accepted iterate.
    """

    x: Pytree
    value: float
    grad_norm: float
    initial_grad_norm: float
    rel_stationarity: float
    status: str
    work: Work
    max_norm_sq: float
    trace: list[dict[str, float]] = field(default_factory=list)
    seconds: float = 0.0


class _Counted:
    """An ``Objective`` that records its calls in a ``Work``."""

    def __init__(self, objective: Objective, work: Work):
        self.objective = objective
        self.work = work

    def value(self, x: Pytree) -> float:
        self.work.energy_evals += 1
        self.work.passes += 1
        return float(self.objective.value(x))

    def value_and_grad(self, x: Pytree) -> tuple[float, Pytree]:
        self.work.energy_evals += 1
        self.work.gradient_evals += 1
        self.work.passes += 1
        value, grad = self.objective.value_and_grad(x)
        return float(value), grad

    def hvp(self, x: Pytree, v: Pytree) -> Pytree:
        self.work.hvps += 1
        self.work.passes += 1
        return self.objective.hvp(x, v)


def energy_margin(value: float) -> float:
    """Rounding margin on an energy comparison, 10·eps·|E|.

    It scales with the energy alone, so a small energy can't hide a real rise.
    Below that resolution a comparison carries no information, and a solve that
    can't reach its tolerance ends ``stalled``, never ``converged``.
    """
    eps = float(jnp.finfo(jnp.result_type(float)).eps)
    return 10.0 * eps * abs(value)


def _boundary_step(z: Pytree, d: Pytree, radius: float) -> float:
    """The positive τ with ‖z + τd‖ = radius."""
    a = tree_dot(d, d)
    b = 2.0 * tree_dot(z, d)
    c = tree_dot(z, z) - radius**2
    return (-b + math.sqrt(max(b * b - 4.0 * a * c, 0.0))) / (2.0 * a)


@dataclass(frozen=True)
class TrustRegionConfig:
    """Trust-region Newton-CG settings.

    ``absolute_tolerance``, when given, replaces ``tolerance``·‖g₀‖ as the
    gradient-norm target.
    """

    tolerance: float = 1e-6
    absolute_tolerance: float | None = None
    initial_radius: float = 1.0
    max_radius: float = 1e6
    min_radius: float = 1e-14
    accept_ratio: float = 1e-4
    shrink_ratio: float = 0.25
    grow_ratio: float = 0.75
    max_cg_iters: int = 500
    max_iterations: int = 100000


def _truncated_cg(
    counted: _Counted,
    x: Pytree,
    g: Pytree,
    g_norm: float,
    g0_norm: float,
    radius: float,
    config: TrustRegionConfig,
    budget: Budget,
) -> tuple[Pytree, float, bool, bool]:
    """Truncated CG on the local quadratic model.

    Returns ``(step, predicted_reduction, on_boundary, finite)``. The residual
    r = g + Hz gives the model reduction −½z·(g + r) without a further HVP.
    The forcing tolerance is min(½, √(‖g‖/‖g₀‖))·‖g‖.
    """
    z = jax.tree_util.tree_map(jnp.zeros_like, g)
    r = g
    d = tree_scale(-1.0, g)
    rr = g_norm**2
    tol = min(0.5, math.sqrt(g_norm / g0_norm)) * g_norm if g0_norm > 0.0 else 0.0

    def done(z_final: Pytree, r_final: Pytree, boundary: bool):
        reduction = -0.5 * (tree_dot(z_final, g) + tree_dot(z_final, r_final))
        return z_final, reduction, boundary, True

    for _ in range(config.max_cg_iters):
        if budget.exhausted(counted.work):
            return done(z, r, False)
        hd = counted.hvp(x, d)
        dhd = tree_dot(d, hd)
        dd = tree_dot(d, d)
        if not math.isfinite(dhd):
            return z, 0.0, False, False
        if dhd <= 1e-12 * dd:
            counted.work.negative_curvature_exits += 1
            tau = _boundary_step(z, d, radius)
            return done(tree_axpy(tau, d, z), tree_axpy(tau, hd, r), True)
        alpha = rr / dhd
        z_next = tree_axpy(alpha, d, z)
        if tree_norm(z_next) >= radius:
            tau = _boundary_step(z, d, radius)
            return done(tree_axpy(tau, d, z), tree_axpy(tau, hd, r), True)
        r_next = tree_axpy(alpha, hd, r)
        rr_next = tree_dot(r_next, r_next)
        z, r = z_next, r_next
        if math.sqrt(max(rr_next, 0.0)) <= tol:
            return done(z, r, False)
        d = tree_axpy(rr_next / rr, d, tree_scale(-1.0, r))
        rr = rr_next
    return done(z, r, False)


def trust_region_newton_cg(
    objective: Objective,
    x0: Pytree,
    config: TrustRegionConfig = TrustRegionConfig(),
    budget: Budget = Budget(),
) -> SolveResult:
    """Minimise ``objective`` by trust-region Newton-CG from ``x0``.

    The ratio test adds ``energy_margin`` to both reductions, so it stays
    meaningful once reductions reach the energy's resolution.
    """
    started = time.monotonic()
    work = Work()
    counted = _Counted(objective, work)
    x = x0
    value, g = counted.value_and_grad(x)
    g_norm = tree_norm(g)
    g0_norm = g_norm
    max_norm_sq = tree_dot(x, x)
    trace = [{"passes": work.passes, "value": value, "grad_norm": g_norm}]
    radius = min(config.initial_radius, config.max_radius)
    target = (
        config.absolute_tolerance
        if config.absolute_tolerance is not None
        else config.tolerance * g0_norm
    )

    def finish(status: str) -> SolveResult:
        result = _result(x, value, g_norm, g0_norm, status, work, max_norm_sq, trace)
        result.seconds = time.monotonic() - started
        return result

    if not (math.isfinite(value) and math.isfinite(g_norm)):
        return finish(NONFINITE)
    for _ in range(config.max_iterations):
        if g_norm <= target:
            return finish(CONVERGED)
        if budget.exhausted(work):
            return finish(CENSORED)
        step, predicted, on_boundary, finite = _truncated_cg(
            counted, x, g, g_norm, g0_norm, radius, config, budget
        )
        if not finite:
            return finish(NONFINITE)
        step_norm = tree_norm(step)
        if step_norm == 0.0:
            return finish(CENSORED if budget.exhausted(work) else STALLED)
        trial = tree_axpy(1.0, step, x)
        trial_value = counted.value(trial)
        margin = energy_margin(value)
        if math.isfinite(trial_value):
            ratio = (value - trial_value + margin) / (predicted + margin)
        else:
            ratio = -math.inf
        if ratio < config.shrink_ratio:
            radius = config.shrink_ratio * step_norm
        elif ratio > config.grow_ratio and on_boundary:
            radius = min(2.0 * radius, config.max_radius)
        if ratio > config.accept_ratio and trial_value <= value + margin:
            x = trial
            value, g = counted.value_and_grad(x)
            g_norm = tree_norm(g)
            work.accepted_steps += 1
            max_norm_sq = max(max_norm_sq, tree_dot(x, x))
            trace.append({"passes": work.passes, "value": value, "grad_norm": g_norm})
            if not math.isfinite(g_norm):
                return finish(NONFINITE)
        else:
            work.rejected_steps += 1
        if radius < config.min_radius:
            return finish(STALLED)
    return finish(STALLED)


def _result(
    x: Pytree,
    value: float,
    grad_norm: float,
    initial: float,
    status: str,
    work: Work,
    max_norm_sq: float,
    trace: list[dict[str, float]],
) -> SolveResult:
    return SolveResult(
        x=x,
        value=value,
        grad_norm=grad_norm,
        initial_grad_norm=initial,
        rel_stationarity=grad_norm / initial if initial > 0.0 else 0.0,
        status=status,
        work=work,
        max_norm_sq=max_norm_sq,
        trace=trace,
    )


@dataclass(frozen=True)
class LbfgsConfig:
    """L-BFGS settings; ``armijo`` and ``curvature`` are the weak Wolfe constants."""

    tolerance: float = 1e-6
    history: int = 20
    armijo: float = 1e-4
    curvature: float = 0.9
    max_line_search: int = 30
    max_iterations: int = 100000


def _two_loop(g: Pytree, history: deque[tuple[Pytree, Pytree, float]]) -> Pytree:
    """Return −H·g by the L-BFGS two-loop recursion."""
    q = g
    alphas = []
    for s, y, rho in reversed(history):
        alpha = rho * tree_dot(s, q)
        alphas.append(alpha)
        q = tree_axpy(-alpha, y, q)
    if history:
        s, y, rho = history[-1]
        q = tree_scale(1.0 / (rho * tree_dot(y, y)), q)
    for (s, y, rho), alpha in zip(history, reversed(alphas)):
        beta = rho * tree_dot(y, q)
        q = tree_axpy(alpha - beta, s, q)
    return tree_scale(-1.0, q)


def _weak_wolfe(
    counted: _Counted,
    x: Pytree,
    value: float,
    slope: float,
    d: Pytree,
    first_step: float,
    config: LbfgsConfig,
    budget: Budget,
) -> tuple[Pytree, float, Pytree] | None:
    """Bisection for a weak Wolfe point.

    A non-finite trial counts as an Armijo failure. Returns the point, or the
    best Armijo point if the search stops early, or None when there is neither.
    Every trial other than the returned point counts as rejected.
    """
    low, high = 0.0, math.inf
    step = first_step
    best = None
    trials = 0
    margin = energy_margin(value)

    def settle(found: tuple[Pytree, float, Pytree] | None):
        counted.work.rejected_steps += trials - (0 if found is None else 1)
        return found

    for _ in range(config.max_line_search):
        if budget.exhausted(counted.work):
            break
        trial = tree_axpy(step, d, x)
        trial_value, trial_grad = counted.value_and_grad(trial)
        trials += 1
        if not math.isfinite(trial_value) or (
            trial_value > value + config.armijo * step * slope + margin
        ):
            high = step
        else:
            best = (trial, trial_value, trial_grad)
            if tree_dot(trial_grad, d) >= config.curvature * slope:
                return settle(best)
            low = step
        step = 2.0 * low if math.isinf(high) else 0.5 * (low + high)
    return settle(best)


def lbfgs(
    objective: Objective,
    x0: Pytree,
    config: LbfgsConfig = LbfgsConfig(),
    budget: Budget = Budget(),
) -> SolveResult:
    """Minimise ``objective`` by L-BFGS with a weak Wolfe line search from ``x0``."""
    started = time.monotonic()
    work = Work()
    counted = _Counted(objective, work)
    x = x0
    value, g = counted.value_and_grad(x)
    g_norm = tree_norm(g)
    g0_norm = g_norm
    max_norm_sq = tree_dot(x, x)
    trace = [{"passes": work.passes, "value": value, "grad_norm": g_norm}]
    history: deque[tuple[Pytree, Pytree, float]] = deque(maxlen=config.history)

    def finish(status: str) -> SolveResult:
        result = _result(x, value, g_norm, g0_norm, status, work, max_norm_sq, trace)
        result.seconds = time.monotonic() - started
        return result

    if not (math.isfinite(value) and math.isfinite(g_norm)):
        return finish(NONFINITE)
    for _ in range(config.max_iterations):
        if g_norm <= config.tolerance * g0_norm:
            return finish(CONVERGED)
        if budget.exhausted(work):
            return finish(CENSORED)
        d = _two_loop(g, history)
        slope = tree_dot(g, d)
        if not (math.isfinite(slope) and slope < 0.0):
            history.clear()
            work.history_resets += 1
            d = tree_scale(-1.0, g)
            slope = -(g_norm**2)
        first_step = 1.0 if history else min(1.0, 1.0 / g_norm)
        found = _weak_wolfe(counted, x, value, slope, d, first_step, config, budget)
        if found is None:
            if budget.exhausted(work):
                return finish(CENSORED)
            if not history:
                return finish(STALLED)
            history.clear()
            work.history_resets += 1
            continue
        new_x, new_value, new_g = found
        s = tree_axpy(-1.0, x, new_x)
        y = tree_axpy(-1.0, g, new_g)
        sy = tree_dot(s, y)
        if sy > 1e-10 * tree_norm(s) * tree_norm(y):
            history.append((s, y, 1.0 / sy))
        else:
            work.skipped_pairs += 1
        x, value, g = new_x, new_value, new_g
        g_norm = tree_norm(g)
        work.accepted_steps += 1
        max_norm_sq = max(max_norm_sq, tree_dot(x, x))
        trace.append({"passes": work.passes, "value": value, "grad_norm": g_norm})
        if not math.isfinite(g_norm):
            return finish(NONFINITE)
    return finish(STALLED)


@dataclass
class SpectrumResult:
    """Lanczos estimates at one point.

    ``finite`` is false if any operator output or Ritz value was non-finite;
    ``censored`` if the deadline stopped it before ``iterations`` reached the
    request. ``lowest_vector`` is the lowest Ritz vector, flattened.
    """

    ritz_values: np.ndarray
    ritz_weights: np.ndarray
    lambda_min: float | None
    lambda_max: float | None
    residual_min: float | None
    residual_max: float | None
    iterations: int
    requested_iterations: int
    finite: bool
    censored: bool
    breakdown: bool
    lowest_vector: np.ndarray | None
    seconds: float = 0.0


def lanczos_memory_bytes(dimension: int, iterations: int) -> int:
    """Bytes held by the Lanczos basis at float64."""
    return 8 * dimension * min(iterations, dimension)


def lanczos_full_reorth(
    operator: Callable[[np.ndarray], np.ndarray],
    start: np.ndarray,
    iterations: int,
    deadline: float | None = None,
) -> SpectrumResult:
    """Lanczos with two passes of full reorthogonalisation against the basis.

    Holds the whole basis, ``lanczos_memory_bytes``. Stops early when the Krylov
    space is exhausted, or at the deadline.
    """
    started = time.monotonic()
    dimension = start.size
    iterations = min(iterations, dimension)
    norm = float(np.linalg.norm(start))

    def result(alphas, betas, basis, finite, censored, breakdown) -> SpectrumResult:
        count = len(alphas)
        empty = np.zeros(0)
        if count == 0 or not finite:
            return SpectrumResult(
                empty,
                empty,
                None,
                None,
                None,
                None,
                count,
                iterations,
                finite,
                censored,
                breakdown,
                None,
                time.monotonic() - started,
            )
        off = np.asarray(betas[:-1])
        tridiagonal = np.diag(alphas) + np.diag(off, k=1) + np.diag(off, k=-1)
        values, vectors = np.linalg.eigh(tridiagonal)
        residuals = abs(betas[-1]) * np.abs(vectors[-1, :])
        return SpectrumResult(
            values,
            vectors[0, :] ** 2,
            float(values[0]),
            float(values[-1]),
            float(residuals[0]),
            float(residuals[-1]),
            count,
            iterations,
            bool(np.all(np.isfinite(values))),
            censored,
            breakdown,
            basis[:count].T @ vectors[:, 0],
            time.monotonic() - started,
        )

    if not math.isfinite(norm) or norm == 0.0:
        return result([], [], None, False, False, False)
    basis = np.zeros((iterations, dimension), dtype=np.float64)
    tolerance = math.sqrt(float(np.finfo(np.float64).eps))
    v = start / norm
    alphas: list[float] = []
    betas: list[float] = []
    for j in range(iterations):
        if deadline is not None and time.monotonic() >= deadline:
            return result(alphas, betas, basis, True, True, False)
        basis[j] = v
        w = np.asarray(operator(v), dtype=np.float64)
        if not np.all(np.isfinite(w)):
            return result(alphas, betas, basis, False, False, False)
        alpha = float(v @ w)
        active = basis[: j + 1]
        for _ in range(2):
            w = w - active.T @ (active @ w)
        beta = float(np.linalg.norm(w))
        alphas.append(alpha)
        betas.append(beta)
        scale = max(max(abs(a) for a in alphas), max(betas))
        if beta <= tolerance * scale:
            betas[-1] = 0.0
            return result(alphas, betas, basis, True, False, True)
        v = w / beta
    return result(alphas, betas, basis, True, False, False)


def tree_spectrum(
    objective: Objective,
    x: Pytree,
    start: Pytree,
    iterations: int,
    deadline: float | None = None,
) -> SpectrumResult:
    """``lanczos_full_reorth`` of the objective's Hessian at ``x``."""
    flat_start, unravel = ravel_pytree(start)

    def operator(v: np.ndarray) -> np.ndarray:
        return np.asarray(ravel_pytree(objective.hvp(x, unravel(jnp.asarray(v))))[0])

    return lanczos_full_reorth(
        operator, np.asarray(flat_start, dtype=np.float64), iterations, deadline
    )


@dataclass(frozen=True)
class AcceptanceThresholds:
    """Reference-acceptance thresholds, frozen in each run manifest.

    ``curvature_floor`` is absolute, in units of the identity floor of H_ε.
    """

    rel_stationarity: float = 1e-6
    curvature_floor: float = 1e-2
    energy_gap: float = 1e-6
    bound_slack: float = 1e-9
    module_cosine: float = 0.99
    module_rel_norm_error: float = 0.05
    module_norm_floor: float = 1e-6
    bp_distinct_cosine: float = 0.95
    bp_like_cosine: float = 0.99
    min_accepted_fraction: float = 0.75
    escape_fraction: float = 0.05
    escape_min_norm: float = 1e-2
    escape_noise_weight: float = 0.5

    def to_dict(self) -> dict[str, float]:
        """The thresholds as a plain dict."""
        return asdict(self)


@dataclass
class EscapeResult:
    """Outcome of one escape test."""

    outcome: str
    perturbation_norm: float
    start_grad_norm: float | None
    energy: float | None
    energy_gap: float | None
    gradients_agree: bool | None
    solve_status: str | None
    work: Work | None
    used_ritz_vector: bool
    seconds: float


def relative_gap(a: float, b: float) -> float:
    """|a − b| / max(|a|, |b|), and 0 when both are 0."""
    scale = max(abs(a), abs(b))
    return abs(a - b) / scale if scale > 0.0 else 0.0


def escape_test(
    objective: Objective,
    x_star: Pytree,
    value_star: float,
    lowest_vector: np.ndarray | None,
    absolute_tolerance: float,
    gradient_check: Callable[[Pytree], bool],
    thresholds: AcceptanceThresholds,
    max_radius: float,
    budget: Budget,
    seed: int,
) -> EscapeResult:
    """Perturb along the lowest Ritz vector plus noise, re-solve, compare.

    The re-solve stops at the original absolute gradient target. An energy lower
    by more than ``energy_gap`` relative or the target squared, the resolution that
    target allows, or weight gradients that differ, mean ``x_star`` was a saddle.
    A perturbation
    too small to raise the gradient above the target tests nothing, so it is
    ``unresolved``, and so is a re-solve that didn't converge.
    """
    started = time.monotonic()
    flat, unravel = ravel_pytree(x_star)
    flat = np.asarray(flat, dtype=np.float64)
    noise = np.random.default_rng(seed).normal(size=flat.size)
    noise /= np.linalg.norm(noise)
    usable = lowest_vector is not None and np.linalg.norm(lowest_vector) > 0.0
    if usable:
        direction = (
            1.0 - thresholds.escape_noise_weight
        ) * lowest_vector / np.linalg.norm(
            lowest_vector
        ) + thresholds.escape_noise_weight * noise
    else:
        direction = noise
    direction /= np.linalg.norm(direction)
    size = thresholds.escape_fraction * max(
        float(np.linalg.norm(flat)), thresholds.escape_min_norm
    )
    start = unravel(jnp.asarray(flat + size * direction))
    start_grad_norm = tree_norm(objective.value_and_grad(start)[1])

    def finish(outcome, energy=None, gap=None, agree=None, status=None, work=None):
        return EscapeResult(
            outcome,
            size,
            start_grad_norm,
            energy,
            gap,
            agree,
            status,
            work,
            usable,
            time.monotonic() - started,
        )

    if not math.isfinite(start_grad_norm) or start_grad_norm <= absolute_tolerance:
        return finish(ESCAPE_UNRESOLVED)
    solve = trust_region_newton_cg(
        objective,
        start,
        TrustRegionConfig(
            absolute_tolerance=absolute_tolerance,
            initial_radius=min(2.0 * size, max_radius),
            max_radius=max_radius,
        ),
        budget,
    )
    if solve.status == CENSORED:
        return finish(ESCAPE_CENSORED, status=solve.status, work=solve.work)
    if solve.status != CONVERGED:
        return finish(ESCAPE_UNRESOLVED, status=solve.status, work=solve.work)
    gap = relative_gap(solve.value, value_star)
    slack = max(
        thresholds.energy_gap * abs(value_star),
        absolute_tolerance**2,
    )
    lower = value_star - solve.value > slack
    agree = bool(gradient_check(solve.x))
    outcome = ESCAPE_SADDLE if lower or not agree else ESCAPE_PASSED
    return finish(outcome, solve.value, gap, agree, solve.status, solve.work)


class EnergyFunctions:
    """Compiled E_s(ε) and its derivatives for one graph at s = 1.

    Every function takes ``(params, state, clamps, s)`` as arguments, so one
    compilation serves all samples and head scales of equal shape. ``state``
    is a state after ``EPCInference.begin_segment``. The energy sums s·head with
    the other nodes directly: subtracting (1 − s)·head from a total at s = 1 would
    leave rounding noise of order eps·head, far above E_s when s is small.
    """

    def __init__(self, structure: GraphStructure):
        self.structure = structure
        self.head = structure.task_map["y"]

        def energy(errors, params, state, clamps, s):
            energy_of, _ = EPCInference.error_energy(params, state, clamps, structure)
            _, derived = energy_of(errors)
            total = s * jnp.sum(derived.nodes[self.head].energy)
            for name, node in structure.nodes.items():
                if node.node_info.in_degree > 0 and name != self.head:
                    total = total + jnp.sum(derived.nodes[name].energy)
            return total, derived

        def value(errors, params, state, clamps, s):
            return energy(errors, params, state, clamps, s)[0]

        def derived(errors, params, state, clamps, s):
            return energy(errors, params, state, clamps, s)[1]

        grad = jax.grad(value)

        def hvp(errors, v, params, state, clamps, s):
            return jax.jvp(
                lambda e: grad(e, params, state, clamps, s), (errors,), (v,)
            )[1]

        def weight_gradients(params, derived_state, s):
            grads = compute_local_weight_gradients(params, derived_state, structure)
            head = grads.nodes[self.head]
            scaled = NodeParams(
                weights={k: s * g for k, g in head.weights.items()},
                biases={k: s * g for k, g in head.biases.items()},
            )
            return grads._replace(nodes={**grads.nodes, self.head: scaled})

        self.value = jax.jit(value)
        self.value_and_grad = jax.jit(jax.value_and_grad(value))
        self.hvp = jax.jit(hvp)
        self.derived = jax.jit(derived)
        self.weight_gradients = jax.jit(weight_gradients)

    def objective(
        self,
        params: GraphParams,
        state: GraphState,
        clamps: Mapping[str, jax.Array],
        s: float,
    ) -> Objective:
        """Bind one sample and head scale."""

        def value(errors):
            return self.value(errors, params, state, clamps, s)

        def value_and_grad(errors):
            return self.value_and_grad(errors, params, state, clamps, s)

        def hvp(errors, v):
            return self.hvp(errors, v, params, state, clamps, s)

        return Objective(value=value, value_and_grad=value_and_grad, hvp=hvp)


def make_energy_functions(structure: GraphStructure) -> EnergyFunctions:
    """Compile the energy functions of a graph built at output scale 1."""
    return EnergyFunctions(structure)


def module_gradients(
    grads: GraphParams, structure: GraphStructure
) -> dict[str, np.ndarray]:
    """Flatten each parameterised node's gradient, in graph order, as float64."""
    modules = {}
    for name in structure.node_order:
        leaves = jax.tree_util.tree_leaves(grads.nodes[name])
        if leaves:
            modules[name] = np.concatenate(
                [np.asarray(leaf, dtype=np.float64).ravel() for leaf in leaves]
            )
    return modules


def compare_module_gradients(
    reference: Mapping[str, np.ndarray],
    other: Mapping[str, np.ndarray],
    thresholds: AcceptanceThresholds,
) -> dict[str, Any]:
    """Gate ``other`` against ``reference`` per module.

    A module is gated if either norm is at least ``module_norm_floor`` of the
    largest norm over both vectors; the rest are listed in ``excluded``. A cosine
    involving a zero vector is None, and a gated module with an undefined cosine
    or norm error fails. With no gated module the comparison does not pass.
    """
    ref_norms = {name: float(np.linalg.norm(g)) for name, g in reference.items()}
    other_norms = {name: float(np.linalg.norm(other[name])) for name in reference}
    largest = max([*ref_norms.values(), *other_norms.values()], default=0.0)
    floor = thresholds.module_norm_floor * largest
    modules: dict[str, dict[str, Any]] = {}
    excluded = []
    passed = largest > 0.0
    for name, ref in reference.items():
        ref_norm, other_norm = ref_norms[name], other_norms[name]
        cosine = None
        if ref_norm > 0.0 and other_norm > 0.0:
            cosine = float(np.dot(ref, other[name]) / (ref_norm * other_norm))
        rel_error = (
            float(np.linalg.norm(other[name] - ref) / ref_norm)
            if ref_norm > 0.0
            else None
        )
        gated = largest > 0.0 and max(ref_norm, other_norm) >= floor
        ok = None
        if gated:
            ok = (
                cosine is not None
                and rel_error is not None
                and cosine >= thresholds.module_cosine
                and rel_error <= thresholds.module_rel_norm_error
            )
            passed = passed and ok
        else:
            excluded.append(name)
        modules[name] = {
            "cosine": cosine,
            "rel_norm_error": rel_error,
            "reference_norm": ref_norm,
            "other_norm": other_norm,
            "gated": gated,
            "passed": ok,
        }
    return {"passed": bool(passed), "modules": modules, "excluded": excluded}


def _is_finite_number(value: Any) -> bool:
    return value is not None and math.isfinite(value)


def assess_reference(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    gradients_agree: bool,
    thresholds: AcceptanceThresholds,
    escape: str | None = None,
) -> dict[str, Any]:
    """Decide whether two solves of one sample form an accepted reference.

    Each summary holds ``status``, ``rel_stationarity``, ``energy``,
    ``max_norm_sq``, ``energy_start`` and ``lambda_min``, ``lambda_max``,
    ``residual_min``, ``residual_max``, ``spectrum_finite``, ``spectrum_censored``.
    Non-finite takes priority over censoring, and a stall is never accepted.
    ``escape`` is the escape outcome, or None if the test didn't run.
    ``curvature_evidence`` separates finding no negative mode from passing the
    escape test.
    """
    solves = (first, second)

    def verdict(status: str, reason: str, evidence: str = EVIDENCE_NONE):
        return {"status": status, "reason": reason, "curvature_evidence": evidence}

    if any(s["status"] == NONFINITE for s in solves) or not all(
        _is_finite_number(s["energy"]) for s in solves
    ):
        return verdict(NONFINITE, "a solve produced non-finite values")
    if any(s["status"] == CENSORED for s in solves):
        return verdict(CENSORED, "a solve hit its budget")
    for solve in solves:
        stationarity = solve["rel_stationarity"]
        if solve["status"] != CONVERGED or not (
            _is_finite_number(stationarity)
            and stationarity <= thresholds.rel_stationarity
        ):
            return verdict(NOT_STATIONARY, f"solve ended {solve['status']}")
    for solve in solves:
        limit = 2.0 * solve["energy_start"] * (1.0 + thresholds.bound_slack)
        if solve["max_norm_sq"] > limit:
            return verdict(BOUND_VIOLATION, "‖ε‖² exceeds 2E_s(0)")
    if any(s.get("spectrum_censored") for s in solves):
        return verdict(CENSORED, "Lanczos hit its budget")
    for solve in solves:
        values = (
            solve["lambda_min"],
            solve["lambda_max"],
            solve["residual_min"],
            solve["residual_max"],
        )
        if not solve.get("spectrum_finite", True) or not all(
            _is_finite_number(v) for v in values
        ):
            return verdict(CURVATURE_UNRESOLVED, "non-finite or missing spectrum")
    if any(s["lambda_min"] < -thresholds.curvature_floor for s in solves):
        return verdict(SADDLE, "Lanczos found a negative mode")
    if relative_gap(first["energy"], second["energy"]) > thresholds.energy_gap:
        return verdict(
            SOLVER_DISAGREEMENT, "energies differ", evidence=EVIDENCE_LANCZOS
        )
    if not gradients_agree:
        return verdict(
            SOLVER_DISAGREEMENT, "weight gradients differ", evidence=EVIDENCE_LANCZOS
        )
    if escape == ESCAPE_SADDLE:
        return verdict(SADDLE, "the escape test left the point", EVIDENCE_LANCZOS)
    if escape == ESCAPE_CENSORED:
        return verdict(CENSORED, "the escape test hit its budget", EVIDENCE_LANCZOS)
    if escape != ESCAPE_PASSED:
        return verdict(
            CURVATURE_UNRESOLVED, "the escape test did not resolve", EVIDENCE_LANCZOS
        )
    return verdict(ACCEPTED, "all gates passed", EVIDENCE_ESCAPE)


def batch_status(sample_statuses: Sequence[str], summed_agree: bool | None) -> str:
    """Combine sample statuses: accepted only if every sample is.

    Refuting statuses outrank unresolved ones. A batch whose samples were all
    without signal is ``zero_signal``.
    """
    for status in (*REFUTING_STATUSES, *UNRESOLVED_STATUSES):
        if status in sample_statuses:
            return status
    if sample_statuses and all(s == ZERO_SIGNAL for s in sample_statuses):
        return ZERO_SIGNAL
    if summed_agree is False:
        return SOLVER_DISAGREEMENT
    if summed_agree is None:
        return CURVATURE_UNRESOLVED
    return ACCEPTED


def lbfgs_memory_bytes(dimension: int, config: LbfgsConfig = LbfgsConfig()) -> int:
    """Bytes held by L-BFGS history and working vectors at float64."""
    return 8 * dimension * (2 * config.history + 8)
