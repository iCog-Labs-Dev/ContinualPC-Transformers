"""Baseline predictors and the error-map correlation test.

Per rollback transition ``S_t <- S_r`` (old domain ``r``, first new domain
``r + 1``) and per module the predictors are:

- ``update_norm`` and ``relative_update_norm``: ``‖θ_t − θ_r‖`` and that
  over ``‖θ_r‖`` (the drift of the whole interval);
- ``grad_norm``: ``‖∇ L_{r+1}(θ_r)‖``, the new-domain loss gradient at the
  start of stage ``r + 1`` (mean masked CE over the fixed windows, dropout
  off);
- ``fisher``: the empirical Fisher (per-window mean CE, true labels) of
  the old-domain loss at ``θ_r``, traced over the module: the mean over
  windows of the squared norm of each window's mean-masked-CE gradient
  with the observed labels;
- ``param_count``;
- the error maps of ``error_maps.json``: ``<setting>_onset`` (domain
  ``r + 1`` at ``S_r``) and ``<setting>_retrospective`` (domain ``r`` at
  ``S_t``), using each map's batch-invariant ``energy_sum_ce``, and the
  same maps per output dimension (``<setting>_<kind>_per_dim``).

The test is Spearman's ρ over modules between each predictor and rollback
recovery of the old domain. Each ρ gets a percentile bootstrap 95% CI over
modules (descriptive) and a seeded two-sided permutation p-value; each map
predictor also gets a paired bootstrap of ``ρ_map − ρ_update_norm`` on the
same resampled modules. Where both settings are present, the Spearman ρ
between the standard-PC and spike-only maps of each kind is reported per
transition (``map_agreement``).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats

from fabricpc.core.types import GraphParams
from fabricpc.experiments.domain_forgetting.error_maps import (
    SPIKE_ONLY,
    STANDARD_PC,
    window_batches,
)
from fabricpc.experiments.domain_forgetting.localization import (
    RunContext,
    check_provenance,
    drift_stats,
    module_keys,
    module_names,
    num_scalars,
    rollback_transitions,
    run_provenance,
    transition_name,
)
from fabricpc.training.train_autoregressive import compute_loss
from fabricpc.training.train_backprop import compute_forward_pass

BOOTSTRAP_RESAMPLES = 2000
PERMUTATIONS = 2000
FISHER_LABEL = "empirical Fisher (per-window mean CE, true labels)"
MAP_KINDS = ("onset", "retrospective")
BASELINE_PREDICTORS = (
    "update_norm",
    "relative_update_norm",
    "grad_norm",
    "fisher",
    "param_count",
)


def _module_square_sums(grads: GraphParams, modules: Sequence[str]) -> jax.Array:
    return jnp.stack(
        [
            sum(jnp.sum(leaf**2) for leaf in jax.tree_util.tree_leaves(grads.nodes[m]))
            for m in modules
        ]
    )


def make_gradient_statistics(context: RunContext):
    """Return ``(modules, sum_ce_grad, per_window_sq)`` for the predictors.

    ``sum_ce_grad(params, batch)`` is the gradient of the batch's summed
    masked CE; ``per_window_sq(params, batch)`` returns, per module, the sum
    over windows of the squared norm of each window's mean-CE gradient and
    the number of windows with a masked token. Both are jitted, dropout off.
    """
    structure = context.structure
    modules = module_names(structure, context.snapshots[0])
    output = structure.task_map["y"]

    def mean_ce(params, batch):
        state = compute_forward_pass(
            params, structure, batch, jax.random.PRNGKey(0), training=False
        )
        return compute_loss(
            state, batch["y"], output, "cross_entropy", loss_mask=batch["loss_mask"]
        )

    def sum_ce_grad(params, batch):
        count = jnp.sum(batch["loss_mask"].astype(jnp.result_type(float)))
        return jax.grad(lambda p: mean_ce(p, batch) * count)(params)

    def per_window_sq(params, batch):
        def one(window):
            single = {k: v[None] for k, v in window.items()}
            return _module_square_sums(jax.grad(mean_ce)(params, single), modules)

        squares = jax.vmap(one)(batch)
        valid = jnp.sum(batch["loss_mask"], axis=-1) > 0
        return jnp.sum(squares * valid[:, None], axis=0), jnp.sum(valid)

    return modules, jax.jit(sum_ce_grad), jax.jit(per_window_sq)


def _gradient_norms(
    params: GraphParams, batches, modules, sum_ce_grad
) -> dict[str, float]:
    total = None
    count = 0.0
    for batch in batches:
        grads = sum_ce_grad(params, batch)
        total = (
            grads if total is None else jax.tree_util.tree_map(jnp.add, total, grads)
        )
        count += float(jnp.sum(batch["loss_mask"]))
    if count == 0:
        raise ValueError("no masked tokens for the gradient norm")
    squares = np.asarray(_module_square_sums(total, modules), np.float64)
    return {m: float(math.sqrt(v) / count) for m, v in zip(modules, squares)}


def _fisher_traces(
    params: GraphParams, batches, modules, per_window_sq
) -> tuple[dict[str, float], int]:
    total = np.zeros(len(modules))
    windows = 0
    for batch in batches:
        squares, valid = per_window_sq(params, batch)
        total += np.asarray(squares, np.float64)
        windows += int(valid)
    if windows == 0:
        raise ValueError("no window has a masked token for the Fisher")
    return {m: float(v / windows) for m, v in zip(modules, total)}, windows


def compute_predictors(
    context: RunContext,
    *,
    num_windows: int = 256,
    batch_size: int = 8,
    verbose: bool = False,
) -> dict[str, Any]:
    """Compute the baseline predictors for every rollback transition.

    Gradient norms and Fisher traces use the first ``num_windows`` fixed
    validation windows of the relevant domain.
    """
    modules, sum_ce_grad, per_window_sq = make_gradient_statistics(context)
    snapshots = context.snapshots
    keys = {m: module_keys(snapshots[0], m) for m in modules}
    counts = {m: num_scalars(snapshots[0], keys[m]) for m in modules}
    cache: dict[tuple[str, int, str], Any] = {}

    def statistic(kind: str, stage: int, domain: str) -> Any:
        key = (kind, stage, domain)
        if key not in cache:
            batches = window_batches(
                context.valid_sets[domain], num_windows, batch_size
            )
            params = context.params(stage)
            if kind == "grad_norm":
                cache[key] = _gradient_norms(params, batches, modules, sum_ce_grad)
            else:
                cache[key] = _fisher_traces(params, batches, modules, per_window_sq)
        return cache[key]

    transitions = []
    for target, reference in rollback_transitions(len(context.domains)):
        old = context.domains[reference - 1]
        new = context.domains[reference]
        drift = {
            m: drift_stats(snapshots[reference], snapshots[target], keys[m])
            for m in modules
        }
        fisher, fisher_windows = statistic("fisher", reference, old)
        grad_norm = statistic("grad_norm", reference, new)
        transitions.append(
            {
                "name": transition_name(target, reference),
                "target_stage": target,
                "reference_stage": reference,
                "old_domain": old,
                "new_domain": new,
                "fisher_windows": fisher_windows,
                "predictors": {
                    "update_norm": {m: drift[m]["update_norm"] for m in modules},
                    "relative_update_norm": {
                        m: drift[m]["relative_update_norm"] for m in modules
                    },
                    "grad_norm": grad_norm,
                    "fisher": fisher,
                    "param_count": {m: float(counts[m]) for m in modules},
                },
            }
        )
        if verbose:
            print(f"predictors {transitions[-1]['name']} done")
    return {
        "domains": list(context.domains),
        "provenance": run_provenance(context),
        "modules": modules,
        "num_windows": num_windows,
        "batch_size": batch_size,
        "labels": {"fisher": FISHER_LABEL},
        "definition": {
            "update_norm": "||theta_t - theta_r|| per module",
            "relative_update_norm": "||theta_t - theta_r|| / ||theta_r||",
            "grad_norm": "||grad of mean masked CE of domain r+1 at S_r||",
            "fisher": f"{FISHER_LABEL}: mean over windows of ||grad of the "
            "window's mean masked CE (domain r, observed labels) at S_r||^2, "
            "the module trace of the diagonal empirical Fisher",
            "param_count": "number of scalars",
        },
        "transitions": transitions,
    }


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman's ρ with average ranks for ties; NaN if either input is constant."""
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("spearman needs two 1-D sequences of equal length")
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return math.nan
    return float(stats.spearmanr(a, b).statistic)


def bootstrap_ci(
    x: Sequence[float],
    y: Sequence[float],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 0,
    level: float = 0.95,
) -> dict[str, Any]:
    """Percentile bootstrap CI of Spearman's ρ, resampling modules.

    Resamples where either side is constant (ρ undefined) are dropped and
    counted in ``num_valid``.
    """
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(resamples):
        index = rng.integers(0, len(a), len(a))
        rho = spearman(a[index], b[index])
        if not math.isnan(rho):
            values.append(rho)
    if not values:
        return {"low": math.nan, "high": math.nan, "num_valid": 0}
    tail = 100.0 * (1.0 - level) / 2.0
    low, high = np.percentile(values, [tail, 100.0 - tail])
    return {"low": float(low), "high": float(high), "num_valid": len(values)}


def paired_bootstrap_difference(
    x: Sequence[float],
    z: Sequence[float],
    y: Sequence[float],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 0,
    level: float = 0.95,
) -> dict[str, Any]:
    """Paired percentile bootstrap of ``ρ(x, y) − ρ(z, y)`` over modules.

    Both correlations are computed on the same resampled module indices;
    resamples where either ρ is undefined are dropped. ``fraction_positive``
    is the share of valid resamples with a positive difference.
    """
    a = np.asarray(x, dtype=np.float64)
    c = np.asarray(z, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(resamples):
        index = rng.integers(0, len(a), len(a))
        first = spearman(a[index], b[index])
        second = spearman(c[index], b[index])
        if not (math.isnan(first) or math.isnan(second)):
            values.append(first - second)
    estimate = spearman(a, b) - spearman(c, b)
    if not values:
        return {
            "estimate": estimate,
            "low": math.nan,
            "high": math.nan,
            "fraction_positive": math.nan,
            "num_valid": 0,
        }
    tail = 100.0 * (1.0 - level) / 2.0
    low, high = np.percentile(values, [tail, 100.0 - tail])
    return {
        "estimate": estimate,
        "low": float(low),
        "high": float(high),
        "fraction_positive": float(np.mean(np.asarray(values) > 0)),
        "num_valid": len(values),
    }


def permutation_p_value(
    x: Sequence[float],
    y: Sequence[float],
    *,
    permutations: int = PERMUTATIONS,
    seed: int = 0,
) -> float:
    """Two-sided permutation p-value of Spearman's ρ (``y`` permuted).

    ``(1 + #{|ρ_perm| ≥ |ρ_obs|}) / (1 + permutations)``; NaN when ρ is
    undefined.
    """
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    observed = spearman(a, b)
    if math.isnan(observed):
        return math.nan
    rank_a = stats.rankdata(a)
    rank_b = stats.rankdata(b)
    rng = np.random.default_rng(seed)
    centred_a = rank_a - rank_a.mean()
    centred_b = rank_b - rank_b.mean()
    denominator = math.sqrt(float(centred_a @ centred_a) * float(centred_b @ centred_b))
    extreme = 0
    for _ in range(permutations):
        rho = float(centred_a @ rng.permutation(centred_b)) / denominator
        if abs(rho) >= abs(observed) - 1e-12:
            extreme += 1
    return (1 + extreme) / (1 + permutations)


def _read(path: Path) -> Any:
    return json.loads(path.read_text()) if path.exists() else None


def standard_map_convention(error_maps: Mapping[str, Any] | None) -> str | None:
    """The ``head_energy`` of an ``error_maps.json`` that holds standard-PC maps.

    Returns None when there are no standard-PC maps. Raises for a legacy file
    (no ``head_energy``, written under the head energy ``s · Σ CE / N``) with
    standard-PC maps: those values are not comparable with current ones,
    while its spike-only ``energy_sum_ce`` values are.
    """
    maps = (error_maps or {}).get("maps", [])
    if not any(m["setting"] != SPIKE_ONLY for m in maps):
        return None
    convention = error_maps.get("head_energy")
    if convention is None:
        raise ValueError(
            "error_maps.json predates the recorded head_energy convention; its "
            "standard_pc maps are not comparable with current ones. Recompute "
            "the error maps, or keep only the spike_only setting"
        )
    return convention


def run_predictor_table(run_dir: str | Path) -> dict[str, dict[str, Any]]:
    """Collect recovery and every available predictor per transition.

    Reads ``rollback.json`` (required), ``predictors.json`` and
    ``error_maps.json`` (optional) from ``run_dir``, refusing any whose
    recorded stage and validation SHA-256s differ from the run's current
    ones. Map predictors use ``energy_sum_ce`` and, as ``*_per_dim``,
    ``energy_sum_ce_per_dim``. Standard-PC maps of a legacy ``error_maps.json``
    are refused (``standard_map_convention``).
    """
    path = Path(run_dir)
    rollback = _read(path / "rollback.json")
    if rollback is None:
        raise FileNotFoundError(f"{path / 'rollback.json'} is missing")
    predictors = _read(path / "predictors.json")
    error_maps = _read(path / "error_maps.json")
    for name, payload in (
        ("rollback.json", rollback),
        ("predictors.json", predictors),
        ("error_maps.json", error_maps),
    ):
        if payload is not None:
            check_provenance(path, payload, name)
    standard_map_convention(error_maps)
    domains = rollback["domains"]
    modules = rollback["modules"]
    by_name = {t["name"]: t for t in (predictors or {}).get("transitions", [])}
    maps = {
        (m["setting"], m["kind"], m["domain"], m["stage"]): m
        for m in (error_maps or {}).get("maps", [])
    }
    settings = sorted({key[0] for key in maps})
    table = {}
    for entry in rollback["transitions"]:
        target, reference = entry["target_stage"], entry["reference_stage"]
        old = domains[reference - 1]
        values: dict[str, dict[str, float]] = {}
        if entry["name"] in by_name:
            values.update(by_name[entry["name"]]["predictors"])
        for setting in settings:
            for kind, domain, stage in (
                ("onset", domains[reference], reference),
                ("retrospective", old, target),
            ):
                found = maps.get((setting, kind, domain, stage))
                if found is not None:
                    values[f"{setting}_{kind}"] = {
                        m: found["modules"][m]["energy_sum_ce"] for m in modules
                    }
                    values[f"{setting}_{kind}_per_dim"] = {
                        m: found["modules"][m]["energy_sum_ce_per_dim"] for m in modules
                    }
        table[entry["name"]] = {
            "old_domain": old,
            "recovery": {m: entry["modules"][m]["recovery"][old] for m in modules},
            "predictors": values,
        }
    return table


def _is_map_predictor(name: str) -> bool:
    return name.startswith(f"{SPIKE_ONLY}_") or name.startswith(f"{STANDARD_PC}_")


def correlate_run(
    run_dir: str | Path,
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    permutations: int = PERMUTATIONS,
    seed: int = 0,
) -> dict[str, Any]:
    """Spearman ρ, bootstrap CI and permutation p per transition and predictor.

    Map predictors also get ``vs_update_norm``, the paired bootstrap of
    ``ρ_map − ρ_update_norm``; ``map_agreement`` holds the Spearman ρ between
    the standard-PC and spike-only maps of each kind (and per dimension).
    """
    table = run_predictor_table(run_dir)
    result = {}
    for name, entry in table.items():
        modules = list(entry["recovery"])
        recovery = [entry["recovery"][m] for m in modules]
        columns = {
            predictor: [values[m] for m in modules]
            for predictor, values in entry["predictors"].items()
        }
        correlations = {}
        for predictor, column in columns.items():
            record = {
                "rho": spearman(column, recovery),
                "ci95": bootstrap_ci(column, recovery, resamples=resamples, seed=seed),
                "permutation_p": permutation_p_value(
                    column, recovery, permutations=permutations, seed=seed
                ),
            }
            if _is_map_predictor(predictor) and "update_norm" in columns:
                record["vs_update_norm"] = paired_bootstrap_difference(
                    column,
                    columns["update_norm"],
                    recovery,
                    resamples=resamples,
                    seed=seed,
                )
            correlations[predictor] = record
        agreement = {}
        for kind in MAP_KINDS:
            for suffix in ("", "_per_dim"):
                spike = columns.get(f"{SPIKE_ONLY}_{kind}{suffix}")
                standard = columns.get(f"{STANDARD_PC}_{kind}{suffix}")
                if spike is not None and standard is not None:
                    agreement[f"{kind}{suffix}"] = spearman(standard, spike)
        result[name] = {
            "old_domain": entry["old_domain"],
            "num_modules": len(modules),
            "predictors": correlations,
            "map_agreement": agreement,
        }
    return result


def compute_correlations(
    run_dirs: Sequence[str | Path],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    permutations: int = PERMUTATIONS,
    seed: int = 0,
) -> dict[str, Any]:
    """Correlate every run and aggregate ρ over runs (seeds) as mean ± sd.

    Runs whose error maps hold standard-PC maps must share one head-energy
    convention; spike-only maps aggregate across conventions.
    """
    conventions = {
        convention
        for path in run_dirs
        if (
            convention := standard_map_convention(_read(Path(path) / "error_maps.json"))
        )
        is not None
    }
    if len(conventions) > 1:
        raise ValueError(
            f"standard_pc error maps use different head energies {sorted(conventions)}"
        )
    runs = {
        str(path): correlate_run(
            path, resamples=resamples, permutations=permutations, seed=seed
        )
        for path in run_dirs
    }
    provenance = {}
    for path in run_dirs:
        rollback = _read(Path(path) / "rollback.json") or {}
        provenance[str(path)] = rollback.get("provenance")
    aggregate: dict[str, dict[str, Any]] = {}
    for per_run in runs.values():
        for name, entry in per_run.items():
            for predictor, value in entry["predictors"].items():
                aggregate.setdefault(name, {}).setdefault(predictor, []).append(
                    value["rho"]
                )
    summary = {}
    for name, predictors in aggregate.items():
        summary[name] = {}
        for predictor, rhos in predictors.items():
            finite = [r for r in rhos if not math.isnan(r)]
            summary[name][predictor] = {
                "rhos": rhos,
                "num_runs": len(rhos),
                "mean": float(np.mean(finite)) if finite else math.nan,
                "sd": float(np.std(finite, ddof=1)) if len(finite) > 1 else math.nan,
            }
    return {
        "run_dirs": [str(p) for p in run_dirs],
        "resamples": resamples,
        "permutations": permutations,
        "seed": seed,
        "provenance": provenance,
        "labels": {"fisher": FISHER_LABEL},
        "statistics": {
            "ci95": "percentile bootstrap over modules (descriptive)",
            "permutation_p": "two-sided, recovery permuted over modules",
            "vs_update_norm": "paired bootstrap of rho_map - rho_update_norm on "
            "the same resampled modules",
            "map_agreement": "Spearman rho between standard-PC and spike-only "
            "maps over modules",
        },
        "target": "rollback recovery of the old domain, over modules",
        "runs": runs,
        "aggregate": summary,
    }


def plot_correlations(result: Mapping[str, Any], path: str | Path) -> Path | None:
    """Plot ρ per predictor (runs as dots, mean as a bar) per transition.

    Returns ``None`` without plotting when matplotlib is not installed.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping the correlation figure")
        return None
    names = list(result["aggregate"])
    figure, axes = plt.subplots(
        1, len(names), figsize=(4.5 * len(names), 4.0), squeeze=False, sharey=True
    )
    for axis, name in zip(axes[0], names):
        predictors = list(result["aggregate"][name])
        for index, predictor in enumerate(predictors):
            entry = result["aggregate"][name][predictor]
            axis.scatter([index] * len(entry["rhos"]), entry["rhos"], s=18, zorder=3)
            if not math.isnan(entry["mean"]):
                axis.bar(index, entry["mean"], color="0.85", zorder=1)
        axis.axhline(0.0, color="0.4", linewidth=0.8)
        axis.set_xticks(range(len(predictors)))
        axis.set_xticklabels(predictors, rotation=60, ha="right", fontsize=7)
        axis.set_title(name)
        axis.set_ylim(-1.05, 1.05)
    axes[0][0].set_ylabel("Spearman rho vs rollback recovery")
    figure.tight_layout()
    output = Path(path)
    figure.savefig(output, dpi=150)
    plt.close(figure)
    return output
