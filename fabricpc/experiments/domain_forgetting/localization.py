"""Per-module localisation of forgetting: modules, drift and rollback.

A module is the parameter subtree of one parameterised graph node (16 for
BERT-Mini); parameterless nodes such as ``L{i}_residual``, the clamped
inputs and the dropout masks are not modules. Groups are a readable summary:
the embeddings, the lower half of the blocks, the upper half and the head.

Drift (observational) compares consecutive stage snapshots. Rollback
(causal) replaces one module of a later snapshot with its value in an earlier
snapshot and re-evaluates every domain. Two random controls replace scalars
drawn across the whole model: the count-matched control draws as many
scalars as the module has, the norm-matched control draws scalars until
their squared update ``Σ (θ_ref − θ_cur)²`` reaches the module's
``‖Δθ_m‖²``.

Every analysis records the run's stage ``params_sha256`` values and
validation SHA-256s (``run_provenance``) so stale results can be refused.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from fabricpc.core.types import GraphParams, GraphStructure
from fabricpc.experiments.domain_forgetting.artifacts import file_sha256, load_npz
from fabricpc.experiments.domain_forgetting.data import load_domain_cache
from fabricpc.experiments.domain_forgetting.runner import (
    RunConfig,
    load_base_model,
    params_from_flat,
)
from fabricpc.experiments.domain_forgetting.validation import MultiDomainEvaluator

FlatParams = Mapping[str, np.ndarray]
BLOCK_PATTERN = re.compile(r"^L(\d+)_")
RANDOM_DRAWS = 3


def module_names(structure: GraphStructure, flat: FlatParams) -> list[str]:
    """Return the parameterised nodes in graph order."""
    owners = {key.split("/", 1)[0] for key in flat}
    unknown = owners - set(structure.nodes)
    if unknown:
        raise ValueError(f"parameters for unknown nodes {sorted(unknown)}")
    return [name for name in structure.node_order if name in owners]


def module_keys(flat: FlatParams, module: str) -> list[str]:
    """Return the sorted flat keys of one module."""
    prefix = f"{module}/"
    keys = sorted(key for key in flat if key.startswith(prefix))
    if not keys:
        raise ValueError(f"module {module!r} has no parameters")
    return keys


def block_index(module: str) -> int | None:
    """Return ``i`` for an ``L{i}_*`` block module, else ``None``."""
    match = BLOCK_PATTERN.match(module)
    return int(match.group(1)) if match else None


def module_groups(modules: Sequence[str]) -> dict[str, list[str]]:
    """Split modules into embeddings, lower blocks, upper blocks and head.

    With ``d`` blocks the lower group is blocks ``0..floor(d/2)-1`` and the
    upper group the rest. Non-block modules before the first block are the
    embeddings and those after the last block the head. Empty groups are
    left out.
    """
    blocks = [block_index(name) for name in modules]
    positions = [i for i, b in enumerate(blocks) if b is not None]
    if not positions:
        raise ValueError("no L{i}_* block modules found")
    depth = max(b for b in blocks if b is not None) + 1
    half = depth // 2
    first, last = positions[0], positions[-1]
    for index, name in enumerate(modules):
        if blocks[index] is None and first < index < last:
            raise ValueError(f"non-block module {name!r} between blocks")
    groups = {
        "embeddings": list(modules[:first]),
        f"blocks_0-{half - 1}": [
            n for n, b in zip(modules, blocks) if b is not None and b < half
        ],
        f"blocks_{half}-{depth - 1}": [
            n for n, b in zip(modules, blocks) if b is not None and b >= half
        ],
        "head": list(modules[last + 1 :]),
    }
    return {name: members for name, members in groups.items() if members}


def group_keys(flat: FlatParams, members: Sequence[str]) -> list[str]:
    """Return the flat keys of every module in ``members``."""
    return [key for module in members for key in module_keys(flat, module)]


def num_scalars(flat: FlatParams, keys: Sequence[str]) -> int:
    """Return the number of scalars in ``keys``."""
    return int(sum(np.asarray(flat[key]).size for key in keys))


def _vector(flat: FlatParams, keys: Sequence[str]) -> np.ndarray:
    return np.concatenate(
        [np.asarray(flat[key], dtype=np.float64).ravel() for key in keys]
    )


def drift_stats(
    previous: FlatParams, current: FlatParams, keys: Sequence[str]
) -> dict[str, float]:
    """Return ``‖Δθ‖₂``, ``‖Δθ‖ / ‖θ_prev‖`` and ``cos(θ_prev, θ_next)``.

    Computed in float64 over the concatenation of ``keys``.
    """
    before = _vector(previous, keys)
    after = _vector(current, keys)
    norm_before = float(np.linalg.norm(before))
    norm_after = float(np.linalg.norm(after))
    update = float(np.linalg.norm(after - before))
    denominator = norm_before * norm_after
    return {
        "update_norm": update,
        "relative_update_norm": update / norm_before if norm_before else math.nan,
        "cosine": float(before @ after) / denominator if denominator else math.nan,
        "norm_previous": norm_before,
        "norm_next": norm_after,
    }


def stage_name(stage: int) -> str:
    """Return ``"S<stage>"``."""
    return f"S{stage}"


def compute_drift(
    snapshots: Sequence[FlatParams],
    structure: GraphStructure,
    domains: Sequence[str],
    *,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Per-module, per-group and whole-model drift over consecutive stages.

    For the transition ``S{s-1} -> S{s}`` each entry holds the
    ``drift_stats`` of that transition plus the cumulative drift from S0
    (``cumulative_norm`` = ``‖θ_s − θ_0‖`` and ``cumulative_relative`` =
    ``‖θ_s − θ_0‖ / ‖θ_0‖``). ``provenance`` (``run_provenance``) is
    stored as given.
    """
    if len(snapshots) != len(domains) + 1:
        raise ValueError("need one snapshot per stage, S0 included")
    base = snapshots[0]
    modules = module_names(structure, base)
    groups = module_groups(modules)
    units: dict[str, dict[str, list[str]]] = {
        "modules": {m: module_keys(base, m) for m in modules},
        "groups": {g: group_keys(base, members) for g, members in groups.items()},
        "model": {"model": sorted(base)},
    }
    transitions = []
    for stage in range(1, len(snapshots)):
        entry: dict[str, Any] = {
            "name": f"{stage_name(stage - 1)}->{stage_name(stage)}",
            "from_stage": stage - 1,
            "to_stage": stage,
            "domain": domains[stage - 1],
        }
        for kind, named_keys in units.items():
            values = {}
            for name, keys in named_keys.items():
                stats = drift_stats(snapshots[stage - 1], snapshots[stage], keys)
                cumulative = drift_stats(base, snapshots[stage], keys)
                stats["cumulative_norm"] = cumulative["update_norm"]
                stats["cumulative_relative"] = cumulative["relative_update_norm"]
                values[name] = stats
            entry[kind] = values if kind != "model" else values["model"]
        transitions.append(entry)
    return {
        "domains": list(domains),
        "provenance": dict(provenance) if provenance is not None else None,
        "modules": modules,
        "groups": groups,
        "num_scalars": {
            m: num_scalars(base, keys) for m, keys in units["modules"].items()
        },
        "transitions": transitions,
    }


def rollback_flat(
    target: FlatParams, reference: FlatParams, keys: Sequence[str]
) -> dict[str, np.ndarray]:
    """Return ``target`` with the leaves ``keys`` taken from ``reference``."""
    if set(target) != set(reference):
        raise ValueError("snapshots have different parameter keys")
    rolled = dict(target)
    for key in keys:
        if np.shape(reference[key]) != np.shape(target[key]):
            raise ValueError(f"shape mismatch for {key!r}")
        rolled[key] = reference[key]
    return rolled


def rollback_positions(
    target: FlatParams, reference: FlatParams, positions: np.ndarray
) -> dict[str, np.ndarray]:
    """Return ``target`` with the scalars at flat ``positions`` from ``reference``.

    Positions index the concatenation of every leaf, in sorted key order and
    flattened.
    """
    if set(target) != set(reference):
        raise ValueError("snapshots have different parameter keys")
    keys = sorted(target)
    sizes = np.array([np.asarray(target[key]).size for key in keys], dtype=np.int64)
    chosen = np.sort(np.asarray(positions, dtype=np.int64))
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    rolled = dict(target)
    for index, key in enumerate(keys):
        lo, hi = np.searchsorted(chosen, [offsets[index], offsets[index + 1]])
        if lo == hi:
            continue
        local = chosen[lo:hi] - offsets[index]
        leaf = np.array(target[key], copy=True)
        flat_leaf = leaf.reshape(-1)
        flat_leaf[local] = np.asarray(reference[key]).reshape(-1)[local]
        rolled[key] = leaf
    return rolled


def random_rollback_flat(
    target: FlatParams,
    reference: FlatParams,
    count: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Return ``target`` with ``count`` random scalars taken from ``reference``.

    Scalar positions are drawn uniformly without replacement over every
    parameter of the model (leaves in sorted key order, flattened).
    """
    if set(target) != set(reference):
        raise ValueError("snapshots have different parameter keys")
    total = int(sum(np.asarray(target[key]).size for key in target))
    if not 0 <= count <= total:
        raise ValueError(f"count must be in [0, {total}], got {count}")
    return rollback_positions(
        target, reference, rng.choice(total, size=count, replace=False)
    )


def squared_updates(target: FlatParams, reference: FlatParams) -> np.ndarray:
    """Return ``(θ_ref − θ_target)²`` per scalar in float64, sorted key order."""
    keys = sorted(target)
    return np.concatenate(
        [
            (
                np.asarray(reference[key], np.float64)
                - np.asarray(target[key], np.float64)
            ).ravel()
            ** 2
            for key in keys
        ]
    )


def norm_matched_positions(
    squares: np.ndarray, budget: float, rng: np.random.Generator
) -> np.ndarray:
    """Draw scalars uniformly without replacement until ``Σ squares ≥ budget``.

    Returns the drawn flat positions (the shortest prefix of a uniform
    random permutation whose squared update reaches ``budget``; empty when
    ``budget`` is not positive).
    """
    if budget <= 0:
        return np.zeros(0, dtype=np.int64)
    order = rng.permutation(len(squares))
    cumulative = np.cumsum(squares[order])
    count = int(np.searchsorted(cumulative, budget * (1.0 - 1e-12))) + 1
    return order[: min(count, len(order))]


@dataclass
class RunContext:
    """A finished run: its config, graph, stage snapshots and fixed sets."""

    run_dir: Path
    config: RunConfig
    run_config: dict[str, Any]
    structure: GraphStructure
    model_config: dict[str, Any]
    domains: list[str]
    snapshots: list[dict[str, np.ndarray]]
    valid_sets: dict[str, dict[str, np.ndarray]]

    def params(self, stage_or_flat: int | FlatParams) -> GraphParams:
        """Return graph params for a stage index or a flat parameter dict."""
        flat = (
            self.snapshots[stage_or_flat]
            if isinstance(stage_or_flat, int)
            else stage_or_flat
        )
        return params_from_flat(flat, self.structure)


def load_run(
    run_dir: str | Path,
    *,
    checkpoint: str | None = None,
    data_dir: str | None = None,
    max_valid_windows: int | None = None,
) -> RunContext:
    """Load a finished run directory written by ``run_domain_forgetting``.

    Every ``stage_<s>.npz`` is checked against ``checkpoint_manifest.json``
    and every domain cache against the validation SHA-256 recorded in
    ``run_config.json``. ``checkpoint`` and ``data_dir`` override the
    recorded paths; ``max_valid_windows`` defaults to the run's setting.
    """
    path = Path(run_dir)
    run_config = json.loads((path / "run_config.json").read_text())
    values = dict(run_config["config"])
    values["domains"] = tuple(values["domains"])
    config = RunConfig(**values, output_dir=str(path))
    if config.training != "sequential":
        raise ValueError(
            f"{path}: analyses need a sequential run, not {config.training}"
        )
    overrides: dict[str, Any] = {}
    if checkpoint is not None:
        overrides["checkpoint"] = checkpoint
    if data_dir is not None:
        overrides["data_dir"] = data_dir
    if max_valid_windows is not None:
        overrides["max_valid_windows"] = max_valid_windows
    config = replace(config, **overrides)
    base = load_base_model(config)
    if base.source["params_sha256"] != run_config["source"]["params_sha256"]:
        raise ValueError(f"{config.checkpoint}: not the checkpoint this run used")
    manifest = json.loads((path / "checkpoint_manifest.json").read_text())
    stages = sorted(manifest["stages"], key=lambda entry: int(entry["stage"]))
    domains = list(config.domains)
    if [int(entry["stage"]) for entry in stages] != list(range(len(domains) + 1)):
        raise ValueError(f"{path}: run is incomplete")
    snapshots = []
    for entry in stages:
        snapshot_path = path / entry["file"]
        if file_sha256(snapshot_path) != entry["file_sha256"]:
            raise ValueError(f"{snapshot_path}: SHA-256 does not match the manifest")
        snapshots.append(load_npz(snapshot_path))
    seq_len = int(base.model_config["seq_len"])
    valid_sets = {}
    for name in domains:
        cache = load_domain_cache(
            config.data_dir,
            name,
            tokenizer_json=base.tokenizer_json,
            seq_len=seq_len,
            max_valid_windows=config.max_valid_windows,
        )
        if cache.manifest["valid"]["arrays_sha256"] != run_config["valid_sha256"][name]:
            raise ValueError(f"{name}: validation set differs from the run's")
        valid_sets[name] = cache.valid
    return RunContext(
        run_dir=path,
        config=config,
        run_config=run_config,
        structure=base.structure,
        model_config=base.model_config,
        domains=domains,
        snapshots=snapshots,
        valid_sets=valid_sets,
    )


def run_provenance(context: RunContext) -> dict[str, Any]:
    """Return the stage ``params_sha256`` values and validation SHA-256s used."""
    return {
        "stage_params_sha256": stage_params_sha256(context.run_dir),
        "valid_sha256": dict(context.run_config["valid_sha256"]),
        "num_valid_windows": {
            name: int(len(arrays["input_ids"]))
            for name, arrays in context.valid_sets.items()
        },
    }


def stage_params_sha256(run_dir: str | Path) -> list[str]:
    """Return a run's stage ``params_sha256`` values from its manifest."""
    manifest = json.loads((Path(run_dir) / "checkpoint_manifest.json").read_text())
    stages = sorted(manifest["stages"], key=lambda entry: int(entry["stage"]))
    return [entry["params_sha256"] for entry in stages]


def check_provenance(
    run_dir: str | Path, payload: Mapping[str, Any] | None, name: str
) -> None:
    """Raise unless ``payload`` was computed from the run's current stages.

    The stored stage ``params_sha256`` values and validation SHA-256s must
    equal those of ``checkpoint_manifest.json`` and ``run_config.json``.
    """
    path = Path(run_dir)
    provenance = (payload or {}).get("provenance")
    if not provenance:
        raise ValueError(f"{path / name}: no provenance; recompute it")
    run_config = json.loads((path / "run_config.json").read_text())
    if provenance.get("stage_params_sha256") != stage_params_sha256(path):
        raise ValueError(f"{path / name}: stale (stage params changed); recompute")
    if provenance.get("valid_sha256") != run_config["valid_sha256"]:
        raise ValueError(f"{path / name}: stale (validation sets changed); recompute")


def rollback_transitions(num_domains: int) -> list[tuple[int, int]]:
    """Return ``(target, reference)`` stage pairs, latest target first.

    For three domains: ``(3, 1)``, ``(3, 2)``, ``(2, 1)``.
    """
    return [
        (target, reference)
        for target in range(num_domains, 1, -1)
        for reference in range(1, target)
    ]


def transition_name(target: int, reference: int) -> str:
    """Return ``"S<target><-S<reference>"``."""
    return f"{stage_name(target)}<-{stage_name(reference)}"


def _effects(
    rolled: Mapping[str, Mapping[str, Any]],
    baseline: Mapping[str, Mapping[str, Any]],
    current: str,
) -> dict[str, Any]:
    return {
        "masked_ce": {d: rolled[d]["masked_ce"] for d in baseline},
        "masked_accuracy": {d: rolled[d]["masked_accuracy"] for d in baseline},
        "recovery": {
            d: baseline[d]["masked_ce"] - rolled[d]["masked_ce"] for d in baseline
        },
        "cost": rolled[current]["masked_ce"] - baseline[current]["masked_ce"],
    }


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    std = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    return float(array.mean()), std


def _control_summary(
    draws: Sequence[Mapping[str, Any]], domains: Sequence[str]
) -> dict[str, Any]:
    """Mean and sample sd of a control's recovery and cost over its draws."""
    stats = {d: _mean_std([item["recovery"][d] for item in draws]) for d in domains}
    cost_mean, cost_std = _mean_std([item["cost"] for item in draws])
    return {
        "draws": [dict(item) for item in draws],
        "recovery_mean": {d: stats[d][0] for d in domains},
        "recovery_std": {d: stats[d][1] for d in domains},
        "cost_mean": cost_mean,
        "cost_std": cost_std,
    }


def compute_rollback(
    context: RunContext,
    *,
    random_draws: int = RANDOM_DRAWS,
    seed: int = 0,
    include_groups: bool = True,
    eval_batch_size: int | None = None,
    evaluator: Callable[[GraphParams], dict[str, dict[str, Any]]] | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Roll modules and groups back to earlier snapshots and evaluate.

    For every transition ``S_t <- S_r`` and every module (and group) the
    rolled-back model ``θ' = θ_t`` with that unit from ``θ_r`` is evaluated
    on all domains in one pass. ``recovery[d] = CE[S_t, d] − CE[θ', d]``
    (positive = rollback recovers ``d``) and ``cost = CE[θ', c] − CE[S_t, c]``
    with ``c`` the domain trained at stage ``t``. Per module two random
    controls draw scalars uniformly across the whole model, ``random_draws``
    draws each: ``random`` takes as many scalars as the module has
    (generator seeded by ``(seed, t, r, module index, draw)``) and
    ``random_norm_matched`` takes scalars until their ``Σ (θ_r − θ_t)²``
    reaches the module's ``‖θ_r − θ_t‖²`` (seeded by ``(seed, t, r, module
    index, draw, 1)``). Each reports the mean and sample sd of its recovery
    and cost.
    """
    if evaluator is None:
        evaluator = MultiDomainEvaluator(
            context.structure,
            context.valid_sets,
            eval_batch_size or context.config.eval_batch_size,
        )
    snapshots = context.snapshots
    modules = module_names(context.structure, snapshots[0])
    groups = module_groups(modules)
    keys = {m: module_keys(snapshots[0], m) for m in modules}
    transitions = []
    for target, reference in rollback_transitions(len(context.domains)):
        current = context.domains[target - 1]
        baseline = evaluator(context.params(target))
        reference_eval = evaluator(context.params(reference))
        squares = squared_updates(snapshots[target], snapshots[reference])
        entry: dict[str, Any] = {
            "name": transition_name(target, reference),
            "target_stage": target,
            "reference_stage": reference,
            "reference_domain": context.domains[reference - 1],
            "current_domain": current,
            "baseline": baseline,
            "reference": reference_eval,
            "full_recovery": {
                d: baseline[d]["masked_ce"] - reference_eval[d]["masked_ce"]
                for d in baseline
            },
            "modules": {},
            "groups": {},
        }
        for index, module in enumerate(modules):
            rolled = rollback_flat(
                snapshots[target], snapshots[reference], keys[module]
            )
            budget = float(
                drift_stats(snapshots[reference], snapshots[target], keys[module])[
                    "update_norm"
                ]
                ** 2
            )
            record = {
                "num_scalars": num_scalars(snapshots[0], keys[module]),
                "squared_update": budget,
                **_effects(evaluator(context.params(rolled)), baseline, current),
            }
            count_draws = []
            norm_draws = []
            for draw in range(random_draws):
                rng = np.random.default_rng([seed, target, reference, index, draw])
                random_flat = random_rollback_flat(
                    snapshots[target], snapshots[reference], record["num_scalars"], rng
                )
                effects = _effects(
                    evaluator(context.params(random_flat)), baseline, current
                )
                count_draws.append(
                    {"recovery": effects["recovery"], "cost": effects["cost"]}
                )
                rng = np.random.default_rng([seed, target, reference, index, draw, 1])
                positions = norm_matched_positions(squares, budget, rng)
                matched_flat = rollback_positions(
                    snapshots[target], snapshots[reference], positions
                )
                effects = _effects(
                    evaluator(context.params(matched_flat)), baseline, current
                )
                norm_draws.append(
                    {
                        "recovery": effects["recovery"],
                        "cost": effects["cost"],
                        "num_scalars": int(len(positions)),
                        "squared_update": float(squares[positions].sum()),
                    }
                )
            if count_draws:
                for name, draws in (
                    ("random", count_draws),
                    ("random_norm_matched", norm_draws),
                ):
                    summary = _control_summary(draws, list(baseline))
                    record[name] = summary
                    record[f"recovery_minus_{name}"] = {
                        d: record["recovery"][d] - summary["recovery_mean"][d]
                        for d in baseline
                    }
            entry["modules"][module] = record
            if verbose:
                print(
                    f"{entry['name']} {module} recovery="
                    + " ".join(f"{d}:{v:.4g}" for d, v in record["recovery"].items())
                )
        if include_groups:
            for group, members in groups.items():
                group_key_list = [k for m in members for k in keys[m]]
                rolled = rollback_flat(
                    snapshots[target], snapshots[reference], group_key_list
                )
                entry["groups"][group] = {
                    "modules": list(members),
                    "num_scalars": num_scalars(snapshots[0], group_key_list),
                    **_effects(evaluator(context.params(rolled)), baseline, current),
                }
        transitions.append(entry)
    return {
        "domains": list(context.domains),
        "provenance": run_provenance(context),
        "modules": modules,
        "groups": groups,
        "random_draws": random_draws,
        "seed": seed,
        "definition": {
            "recovery": "CE[S_t, d] - CE[rolled back, d]; positive = recovers d",
            "cost": "CE[rolled back, current] - CE[S_t, current]",
            "random": "same scalar count drawn uniformly over all parameters",
            "random_norm_matched": "scalars drawn uniformly over all parameters "
            "until sum (theta_r - theta_t)^2 reaches the module's ||dtheta_m||^2",
            "full_recovery": "CE[S_t, d] - CE[S_r, d] (whole model rolled back)",
        },
        "transitions": transitions,
    }
