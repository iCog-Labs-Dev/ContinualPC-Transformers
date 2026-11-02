"""Does onset energy predict rollback recovery beyond architecture?

The unit of replication is the domain order. Runs are grouped by their
``config.domains`` tuple and recovery and predictors are averaged over the
seeds of an order (per transition and block module) before any test.

Models predict the rank of recovery within each (order, transition) over the
12 block modules ``L{i}_{mha,mlp1,mlp2}``:

- ``arch``: module type (intercept + 2 treatment dummies, reference ``mha``)
  and block depth ``i``;
- ``energy``: log onset energy; ``arch+energy``: both;
- ``perdim``: log onset energy per output dimension; ``arch+perdim``: both;
- ``update``: log update norm (post hoc reference).

Primary fit: leave one order out. OLS on the recovery ranks of the other
orders, prediction of the held-out order, scored by Spearman's ρ between
prediction and the held-out order's recovery. The primary test is
``Δ = ρ(arch+energy) − ρ(arch)`` on the primary transition over the held-out
orders; it is supported iff ``Δ > 0`` in every held-out order and the mean
``Δ`` exceeds 0.05.

Secondary: within-order partial rank correlation of each predictor with
recovery given type + depth (ranks residualised on the arch design, Pearson
of the residuals), with a two-sided permutation p-value permuting recovery
within module type.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import stats

from fabricpc.experiments.domain_forgetting.correlations import (
    run_predictor_table,
    spearman,
)
from fabricpc.experiments.domain_forgetting.localization import (
    BLOCK_PATTERN,
    block_index,
)

Order = tuple[str, ...]
FORWARD: Order = ("news", "reviews", "wiki")
REVERSE: Order = ("wiki", "reviews", "news")
DEVELOPMENT_ORDERS: tuple[Order, ...] = (FORWARD, REVERSE)
MODULE_TYPES = ("mha", "mlp1", "mlp2")
NUM_BLOCK_MODULES = 12
PRIMARY_TRANSITION = "S3<-S1"
TRANSITIONS = ("S2<-S1", "S3<-S1", "S3<-S2")
MEAN_DELTA_THRESHOLD = 0.05
PERMUTATIONS = 5000
SETTING = "spike_only"
MODELS = ("arch", "energy", "arch+energy", "perdim", "arch+perdim", "update")
COMPARISONS = {
    "arch+energy_vs_arch": ("arch+energy", "arch"),
    "arch+perdim_vs_arch": ("arch+perdim", "arch"),
    "perdim_vs_arch": ("perdim", "arch"),
    "energy_vs_arch": ("energy", "arch"),
    "update_vs_arch": ("update", "arch"),
}
FEATURES = ("energy", "perdim", "update")
NOT_EVALUATED = "not evaluated: no held-out orders"


def predictor_names(setting: str = SETTING) -> dict[str, str]:
    """Map each feature to its ``run_predictor_table`` predictor name."""
    return {
        "energy": f"{setting}_onset",
        "perdim": f"{setting}_onset_per_dim",
        "update": "update_norm",
    }


def order_key(order: Sequence[str]) -> str:
    """Return the comma-joined name of a domain order."""
    return ",".join(order)


def parse_order(text: str) -> Order:
    """Parse ``"news,reviews,wiki"`` into a domain tuple."""
    order = tuple(part.strip() for part in text.split(",") if part.strip())
    if len(order) < 2:
        raise ValueError(f"order {text!r} needs at least two domains")
    return order


def module_type(module: str) -> str:
    """Return the type suffix of an ``L{i}_*`` block module."""
    match = BLOCK_PATTERN.match(module)
    if match is None:
        raise ValueError(f"{module!r} is not a block module")
    kind = module[match.end() :]
    if kind not in MODULE_TYPES:
        raise ValueError(f"unknown module type {kind!r} of {module!r}")
    return kind


def module_depth(module: str) -> int:
    """Return the block index ``i`` of an ``L{i}_*`` module."""
    depth = block_index(module)
    if depth is None:
        raise ValueError(f"{module!r} is not a block module")
    return depth


def block_modules(modules: Sequence[str]) -> list[str]:
    """Return the block modules sorted by depth then type."""
    blocks = [m for m in modules if block_index(m) is not None]
    return sorted(
        blocks, key=lambda m: (module_depth(m), MODULE_TYPES.index(module_type(m)))
    )


@dataclass
class OrderData:
    """Seed-averaged and per-seed recovery and features of one order.

    ``recovery[t]`` and ``features[t][f]`` are arrays over ``modules``;
    ``per_seed`` holds the same per run, in ``run_dirs`` order.
    """

    order: Order
    modules: list[str]
    run_dirs: list[str]
    recovery: dict[str, np.ndarray] = field(default_factory=dict)
    features: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    per_seed: list[dict[str, Any]] = field(default_factory=list)


def load_tables(run_dirs: Sequence[str | Path]) -> list[dict[str, Any]]:
    """Read each run's domain order and ``run_predictor_table``."""
    runs = []
    for path in run_dirs:
        config = json.loads((Path(path) / "run_config.json").read_text())
        runs.append(
            {
                "run_dir": str(path),
                "order": tuple(config["config"]["domains"]),
                "table": run_predictor_table(path),
            }
        )
    return runs


def _seed_values(
    table: Mapping[str, Any],
    modules: Sequence[str],
    names: Mapping[str, str],
    transitions: Sequence[str],
    label: str,
) -> dict[str, Any]:
    recovery: dict[str, np.ndarray] = {}
    features: dict[str, dict[str, np.ndarray]] = {}
    for transition in transitions:
        if transition not in table:
            raise ValueError(f"{label}: transition {transition} is missing")
        entry = table[transition]
        found = block_modules(entry["recovery"])
        if found != list(modules):
            raise ValueError(
                f"{label} {transition}: block modules {found} differ from {list(modules)}"
            )
        recovery[transition] = np.array(
            [entry["recovery"][m] for m in modules], np.float64
        )
        features[transition] = {}
        for feature, name in names.items():
            values = entry["predictors"].get(name)
            if values is None:
                raise ValueError(f"{label} {transition}: predictor {name} is missing")
            missing = [m for m in modules if m not in values]
            if missing:
                raise ValueError(f"{label} {transition}: {name} lacks {missing}")
            features[transition][feature] = np.array(
                [values[m] for m in modules], np.float64
            )
    return {"recovery": recovery, "features": features}


def group_orders(
    runs: Sequence[Mapping[str, Any]],
    *,
    setting: str = SETTING,
    transitions: Sequence[str] = TRANSITIONS,
    num_modules: int = NUM_BLOCK_MODULES,
) -> dict[Order, OrderData]:
    """Group runs by domain order and average over seeds.

    Each run is ``{"run_dir", "order", "table"}`` with ``table`` as returned by
    ``run_predictor_table``. Raw values are averaged arithmetically; logs are
    taken later, of the averages.
    """
    names = predictor_names(setting)
    grouped: dict[Order, list[Mapping[str, Any]]] = {}
    for run in runs:
        grouped.setdefault(tuple(run["order"]), []).append(run)
    reference: list[str] | None = None
    orders = {}
    for order, members in grouped.items():
        first = members[0]["table"]
        modules = block_modules(first[transitions[0]]["recovery"])
        if len(modules) != num_modules:
            raise ValueError(
                f"order {order_key(order)}: {len(modules)} block modules, "
                f"expected {num_modules}"
            )
        if reference is None:
            reference = modules
        elif modules != reference:
            raise ValueError(
                f"order {order_key(order)}: modules {modules} differ from {reference}"
            )
        data = OrderData(order, modules, [str(r["run_dir"]) for r in members])
        for run in members:
            values = _seed_values(
                run["table"], modules, names, transitions, str(run["run_dir"])
            )
            data.per_seed.append({"run_dir": str(run["run_dir"]), **values})
        for transition in transitions:
            data.recovery[transition] = np.mean(
                [s["recovery"][transition] for s in data.per_seed], axis=0
            )
            data.features[transition] = {
                f: np.mean(
                    [s["features"][transition][f] for s in data.per_seed], axis=0
                )
                for f in names
            }
        orders[order] = data
    return orders


def safe_log(values: np.ndarray, label: str) -> np.ndarray:
    """Natural log, raising if any value is not strictly positive and finite."""
    array = np.asarray(values, np.float64)
    bad = ~np.isfinite(array) | (array <= 0)
    if bad.any():
        raise ValueError(
            f"{label}: log needs positive finite values, got {array[bad].tolist()}"
        )
    return np.log(array)


def arch_design(modules: Sequence[str]) -> np.ndarray:
    """Return ``[mlp1, mlp2, depth]`` columns (``mha`` is the reference type)."""
    types = [module_type(m) for m in modules]
    return np.column_stack(
        [
            [float(t == "mlp1") for t in types],
            [float(t == "mlp2") for t in types],
            [float(module_depth(m)) for m in modules],
        ]
    )


def model_features(
    model: str, modules: Sequence[str], features: Mapping[str, np.ndarray], label: str
) -> np.ndarray:
    """Return the feature matrix of ``model`` without the intercept."""
    columns = []
    if model.startswith("arch"):
        columns.append(arch_design(modules))
    for feature in FEATURES:
        if model.endswith(feature):
            columns.append(safe_log(features[feature], f"{label} {feature}")[:, None])
    if not columns:
        raise ValueError(f"unknown model {model!r}")
    return np.hstack(columns)


def with_intercept(matrix: np.ndarray) -> np.ndarray:
    """Prepend a column of ones."""
    return np.hstack([np.ones((len(matrix), 1)), matrix])


def ols_fit(matrix: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Least-squares coefficients of ``target`` on ``matrix`` (intercept included)."""
    coefficients, *_ = np.linalg.lstsq(matrix, target, rcond=None)
    return coefficients


def leave_one_order_out(
    orders: Mapping[Order, OrderData],
    *,
    transitions: Sequence[str] = TRANSITIONS,
    models: Sequence[str] = MODELS,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Fit each model on all other orders and score it on each held-out order.

    Returns ``{order: {transition: {model: {"rho", "prediction",
    "coefficients"}}}}``; the target is the within-order rank of recovery.
    """
    if len(orders) < 2:
        raise ValueError("leave-one-order-out needs at least two orders")
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for held in orders:
        result[order_key(held)] = {}
        for transition in transitions:
            entry = {}
            for model in models:
                rows, targets = [], []
                for order, data in orders.items():
                    if order == held:
                        continue
                    label = f"{order_key(order)} {transition}"
                    rows.append(
                        model_features(
                            model, data.modules, data.features[transition], label
                        )
                    )
                    targets.append(stats.rankdata(data.recovery[transition]))
                coefficients = ols_fit(
                    with_intercept(np.vstack(rows)), np.concatenate(targets)
                )
                data = orders[held]
                test = with_intercept(
                    model_features(
                        model,
                        data.modules,
                        data.features[transition],
                        f"{order_key(held)} {transition}",
                    )
                )
                prediction = test @ coefficients
                entry[model] = {
                    "rho": spearman(
                        prediction.tolist(), data.recovery[transition].tolist()
                    ),
                    "prediction": prediction.tolist(),
                    "coefficients": coefficients.tolist(),
                }
            result[order_key(held)][transition] = entry
    return result


def compare(
    looo: Mapping[str, Mapping[str, Mapping[str, Any]]],
    scope: Sequence[str],
    transition: str,
    first: str,
    second: str,
) -> dict[str, Any]:
    """``Δ = ρ(first) − ρ(second)`` per order in ``scope`` with sign counts."""
    deltas = {
        key: looo[key][transition][first]["rho"] - looo[key][transition][second]["rho"]
        for key in scope
    }
    values = np.array(list(deltas.values()), np.float64)
    finite = values[np.isfinite(values)]
    return {
        "deltas": deltas,
        "mean": (
            float(np.mean(finite))
            if len(finite) == len(values) and len(values)
            else math.nan
        ),
        "positive": int(np.sum(finite > 0)),
        "negative": int(np.sum(finite < 0)),
        "zero": int(np.sum(finite == 0)),
        "undefined": int(len(values) - len(finite)),
        "num_orders": len(values),
    }


def primary_decision(
    comparison: Mapping[str, Any], threshold: float = MEAN_DELTA_THRESHOLD
) -> dict[str, Any]:
    """Supported iff every held-out ``Δ > 0`` and the mean ``Δ`` exceeds ``threshold``."""
    if comparison["num_orders"] == 0:
        return {"status": NOT_EVALUATED, "supported": None}
    all_positive = comparison["positive"] == comparison["num_orders"]
    mean = comparison["mean"]
    supported = bool(all_positive and math.isfinite(mean) and mean > threshold)
    return {
        "status": "supported" if supported else "not supported",
        "supported": supported,
        "all_positive": all_positive,
        "mean_delta": mean,
        "threshold": threshold,
    }


def _residuals(design: np.ndarray, target: np.ndarray) -> np.ndarray:
    return target - design @ ols_fit(design, target)


def partial_rank_correlation(
    predictor: np.ndarray, recovery: np.ndarray, modules: Sequence[str]
) -> float:
    """Pearson ρ of rank residuals after OLS on intercept + type + depth."""
    design = with_intercept(arch_design(modules))
    a = _residuals(design, stats.rankdata(predictor))
    b = _residuals(design, stats.rankdata(recovery))
    denominator = math.sqrt(float(a @ a) * float(b @ b))
    if denominator < 1e-12:
        return math.nan
    return float(a @ b) / denominator


def permute_within(groups: Sequence[str], rng: np.random.Generator) -> np.ndarray:
    """Return an index permutation that only swaps positions of equal group."""
    labels = np.asarray(groups)
    index = np.arange(len(labels))
    for label in np.unique(labels):
        positions = np.flatnonzero(labels == label)
        index[positions] = rng.permutation(positions)
    return index


def partial_permutation_test(
    predictor: np.ndarray,
    recovery: np.ndarray,
    modules: Sequence[str],
    *,
    permutations: int = PERMUTATIONS,
    seed: int = 0,
) -> dict[str, float]:
    """Partial rank correlation and its two-sided within-type permutation p."""
    observed = partial_rank_correlation(predictor, recovery, modules)
    if math.isnan(observed):
        return {"partial": math.nan, "p": math.nan}
    types = [module_type(m) for m in modules]
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(permutations):
        index = permute_within(types, rng)
        value = partial_rank_correlation(predictor, recovery[index], modules)
        if not math.isnan(value) and abs(value) >= abs(observed) - 1e-12:
            extreme += 1
    return {"partial": observed, "p": (1 + extreme) / (1 + permutations)}


def secondary_tests(
    orders: Mapping[Order, OrderData],
    *,
    transitions: Sequence[str] = TRANSITIONS,
    permutations: int = PERMUTATIONS,
    seed: int = 0,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Within-order partial correlations on seed means and per seed."""
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for order, data in orders.items():
        result[order_key(order)] = {}
        for transition in transitions:
            entry = {}
            for feature in FEATURES:
                seed_mean = partial_permutation_test(
                    data.features[transition][feature],
                    data.recovery[transition],
                    data.modules,
                    permutations=permutations,
                    seed=seed,
                )
                per_seed = [
                    {
                        "run_dir": s["run_dir"],
                        **partial_permutation_test(
                            s["features"][transition][feature],
                            s["recovery"][transition],
                            data.modules,
                            permutations=permutations,
                            seed=seed,
                        ),
                    }
                    for s in data.per_seed
                ]
                entry[feature] = {"seed_mean": seed_mean, "per_seed": per_seed}
            result[order_key(order)][transition] = entry
    return result


def git_commit() -> str | None:
    """Return ``git rev-parse HEAD`` of this repository, or ``None``."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def _order_record(data: OrderData, transitions: Sequence[str]) -> dict[str, Any]:
    return {
        "order": list(data.order),
        "run_dirs": data.run_dirs,
        "num_seeds": len(data.per_seed),
        "modules": data.modules,
        "seed_mean": {
            t: {
                "recovery": data.recovery[t].tolist(),
                **{f: v.tolist() for f, v in data.features[t].items()},
            }
            for t in transitions
        },
        "per_seed": [
            {
                "run_dir": s["run_dir"],
                **{
                    t: {
                        "recovery": s["recovery"][t].tolist(),
                        **{f: v.tolist() for f, v in s["features"][t].items()},
                    }
                    for t in transitions
                },
            }
            for s in data.per_seed
        ],
    }


def analyse(
    runs: Sequence[Mapping[str, Any]],
    *,
    development: Sequence[Order] = DEVELOPMENT_ORDERS,
    held_out: Sequence[Order] | None = None,
    setting: str = SETTING,
    transitions: Sequence[str] = TRANSITIONS,
    primary_transition: str = PRIMARY_TRANSITION,
    permutations: int = PERMUTATIONS,
    seed: int = 0,
    threshold: float = MEAN_DELTA_THRESHOLD,
) -> dict[str, Any]:
    """Run the full architecture-baseline analysis on loaded runs.

    ``held_out`` defaults to every order present that is not in
    ``development``; each held-out order must be present in ``runs``.
    """
    if primary_transition not in transitions:
        raise ValueError(f"primary transition {primary_transition} not analysed")
    orders = group_orders(runs, setting=setting, transitions=transitions)
    development = [tuple(o) for o in development]
    if held_out is None:
        held = [o for o in orders if o not in development]
    else:
        held = [tuple(o) for o in held_out]
        absent = [order_key(o) for o in held if o not in orders]
        if absent:
            raise ValueError(f"held-out orders {absent} have no runs")
    looo = leave_one_order_out(orders, transitions=transitions)
    scopes = {
        "held_out": [order_key(o) for o in held],
        "all": [order_key(o) for o in orders],
    }
    comparisons = {
        scope: {
            transition: {
                name: compare(looo, keys, transition, first, second)
                for name, (first, second) in COMPARISONS.items()
            }
            for transition in transitions
        }
        for scope, keys in scopes.items()
    }
    primary = comparisons["held_out"][primary_transition]["arch+energy_vs_arch"]
    return {
        "git_commit": git_commit(),
        "config": {
            "development_orders": [order_key(o) for o in development],
            "held_out_orders": scopes["held_out"],
            "all_orders": scopes["all"],
            "setting": setting,
            "predictors": predictor_names(setting),
            "transitions": list(transitions),
            "primary_transition": primary_transition,
            "permutations": permutations,
            "seed": seed,
            "threshold": threshold,
        },
        "definitions": {
            "replication_unit": "domain order; seeds averaged (arithmetic mean "
            "of raw values) within an order before any test",
            "target": "rank of seed-mean rollback recovery of the old domain "
            "within each (order, transition) over the block modules",
            "models": {
                "arch": "intercept + type dummies (mlp1, mlp2; mha reference) "
                "+ block depth",
                "energy": "intercept + log seed-mean onset energy",
                "arch+energy": "arch + log onset energy",
                "perdim": "intercept + log onset energy per output dimension",
                "arch+perdim": "arch + log per-dim onset energy",
                "update": "intercept + log update norm (post hoc reference)",
            },
            "fit": "OLS on the stacked rows of all other orders; Spearman rho "
            "between the held-out prediction and its seed-mean recovery",
            "primary": f"delta = rho(arch+energy) - rho(arch) on "
            f"{primary_transition}; supported iff delta > 0 in every held-out "
            f"order and mean delta > {threshold}",
            "secondary": "Pearson rho of rank residuals of predictor and "
            "recovery after OLS on intercept + type + depth; two-sided "
            "permutation p with recovery permuted within module type, "
            "(1 + #extreme) / (1 + permutations)",
            "power": "with 4-6 orders only large, consistent effects are detectable",
        },
        "orders": {
            order_key(o): _order_record(d, transitions) for o, d in orders.items()
        },
        "looo": looo,
        "comparisons": comparisons,
        "primary": {
            "transition": primary_transition,
            "comparison": "arch+energy_vs_arch",
            "held_out_orders": scopes["held_out"],
            **primary,
            **primary_decision(primary, threshold),
        },
        "secondary": secondary_tests(
            orders, transitions=transitions, permutations=permutations, seed=seed
        ),
    }


def run_architecture_baseline(
    run_dirs: Sequence[str | Path], **kwargs: Any
) -> dict[str, Any]:
    """Load ``run_dirs`` and run ``analyse`` on them."""
    result = analyse(load_tables(run_dirs), **kwargs)
    result["run_dirs"] = [str(p) for p in run_dirs]
    return result


def _fmt(value: float) -> str:
    return "  nan" if value is None or math.isnan(value) else f"{value:+.2f}"


def format_table(result: Mapping[str, Any]) -> str:
    """Return a concise text table of the LOOO ρ, Δ and secondary partials."""
    transitions = result["config"]["transitions"]
    lines = ["leave-one-order-out Spearman rho (prediction vs recovery)"]
    header = f"{'order':<20} {'transition':<8} " + " ".join(f"{m:>11}" for m in MODELS)
    lines += [header + f" {'d_energy':>9} {'d_perdim':>9}"]
    for key, per_transition in result["looo"].items():
        for transition in transitions:
            entry = per_transition[transition]
            rhos = " ".join(f"{_fmt(entry[m]['rho']):>11}" for m in MODELS)
            d_energy = entry["arch+energy"]["rho"] - entry["arch"]["rho"]
            d_perdim = entry["arch+perdim"]["rho"] - entry["arch"]["rho"]
            lines.append(
                f"{key:<20} {transition:<8} {rhos} {_fmt(d_energy):>9} {_fmt(d_perdim):>9}"
            )
    primary = result["primary"]
    lines.append("")
    lines.append(
        f"primary ({primary['transition']}, arch+energy vs arch, held out "
        f"{primary['held_out_orders'] or 'none'}): {primary['status']}"
    )
    for scope in ("held_out", "all"):
        comparison = result["comparisons"][scope][primary["transition"]][
            "arch+energy_vs_arch"
        ]
        lines.append(
            f"  {scope}: mean delta {_fmt(comparison['mean'])}, "
            f"+{comparison['positive']} -{comparison['negative']} "
            f"of {comparison['num_orders']}"
        )
    lines.append("")
    lines.append("within-order partial rank correlation | type + depth (p within type)")
    for key, per_transition in result["secondary"].items():
        for transition in transitions:
            cells = []
            for feature in FEATURES:
                entry = per_transition[transition][feature]
                seeds = ",".join(_fmt(s["partial"]) for s in entry["per_seed"])
                cells.append(
                    f"{feature} {_fmt(entry['seed_mean']['partial'])} "
                    f"p={entry['seed_mean']['p']:.3f} [{seeds}]"
                )
            lines.append(f"{key:<20} {transition:<8} " + " | ".join(cells))
    return "\n".join(lines)
