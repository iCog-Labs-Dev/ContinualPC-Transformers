"""Commutator test of domain-order dependence for the domain-forgetting runs.

For a domain pair ``{A, B}``, a condition (``frontier.describe_gate`` label)
and a seed, ``θ_AB`` and ``θ_BA`` are the end params of the ``A→B`` and
``B→A`` runs from the same checkpoint. A run's **path** is the sum of its
logged per-update ``update_norm`` over all stages (per module: of
``module_update_norms``).

- **Order distance** ``D = ‖θ_AB − θ_BA‖ / (path_AB + path_BA)``, globally
  and per module (each module's distance over its own paths).
- **Noise floor** ``N``: the same ratio between two seeds of one order and
  condition, over every seed pair of each order.
- **Chord-to-path** per run and stage: ``‖θ_s − θ_{s−1}‖ / path_s``.
- **Function-space distance** (``divergence=True``): the mean over masked
  tokens of the symmetric KL ``½(KL(p_AB‖p_BA) + KL(p_BA‖p_AB))`` between the
  two models' masked-token predictions on each domain's fixed validation
  set (dropout off), and the per-domain masked-CE difference
  ``CE(θ_AB) − CE(θ_BA)``.

``D`` and ``N`` are reported separately; ``D / N`` is not a test statistic.
The **commutator test** of a gate against a control passes iff
``D_gate < D_control`` in every shared seed and the mean gap exceeds the
seed SD of ``D_control``. Means are over seeds; SDs are sample SDs.
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from fabricpc.core.types import GraphParams, GraphStructure
from fabricpc.experiments.domain_forgetting.frontier import describe_gate, mean_sd
from fabricpc.experiments.domain_forgetting.localization import RunContext, load_run
from fabricpc.experiments.domain_forgetting.runner import evaluate_domains
from fabricpc.experiments.domain_forgetting.validation import head_input
from fabricpc.training.train_backprop import compute_forward_pass

FlatParams = Mapping[str, np.ndarray]


def module_of(key: str) -> str:
    """Return the module (node) name of a flat parameter key."""
    return key.split("/", 1)[0]


def flat_distance(a: FlatParams, b: FlatParams) -> tuple[float, dict[str, float]]:
    """Return ``‖a − b‖`` and its per-module parts (float64 accumulation)."""
    if set(a) != set(b):
        raise ValueError("params have different keys")
    squares: dict[str, float] = {}
    for key in sorted(a):
        diff = np.asarray(a[key], np.float64) - np.asarray(b[key], np.float64)
        module = module_of(key)
        squares[module] = squares.get(module, 0.0) + float(np.sum(diff * diff))
    modules = {name: math.sqrt(value) for name, value in squares.items()}
    return math.sqrt(sum(squares.values())), modules


def read_paths(run_dir: str | Path, num_stages: int) -> list[dict[str, Any]]:
    """Return each stage's ``{"total", "modules"}`` path from its training log."""
    stages = []
    for stage in range(1, num_stages + 1):
        path = Path(run_dir) / f"training_log_stage_{stage}.jsonl"
        lines = [json.loads(line) for line in path.read_text().splitlines() if line]
        if not lines or any("update_norm" not in entry for entry in lines):
            raise ValueError(f"{path}: no per-update update_norm logged")
        modules: dict[str, float] = {}
        for entry in lines:
            for name, value in entry["module_update_norms"].items():
                modules[name] = modules.get(name, 0.0) + float(value)
        total = float(sum(float(entry["update_norm"]) for entry in lines))
        stages.append({"total": total, "modules": modules})
    return stages


def run_path(stages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Sum stage paths into a run path."""
    modules: dict[str, float] = {}
    for stage in stages:
        for name, value in stage["modules"].items():
            modules[name] = modules.get(name, 0.0) + value
    return {"total": float(sum(s["total"] for s in stages)), "modules": modules}


def path_ratio(
    a: FlatParams,
    b: FlatParams,
    path_a: Mapping[str, Any],
    path_b: Mapping[str, Any],
) -> dict[str, Any]:
    """Return ``‖a − b‖ / (path_a + path_b)`` globally and per module."""
    distance, modules = flat_distance(a, b)
    ratio = {}
    for name, value in modules.items():
        denominator = path_a["modules"].get(name, 0.0) + path_b["modules"].get(
            name, 0.0
        )
        ratio[name] = value / denominator if denominator > 0 else None
    return {
        "distance": distance,
        "path": path_a["total"] + path_b["total"],
        "ratio": distance / (path_a["total"] + path_b["total"]),
        "module_distance": modules,
        "module_ratio": ratio,
    }


def chord_to_path(
    snapshots: Sequence[FlatParams], stages: Sequence[Mapping[str, Any]]
) -> list[float]:
    """Return ``‖θ_s − θ_{s−1}‖ / path_s`` per stage."""
    return [
        flat_distance(snapshots[s], snapshots[s - 1])[0] / stages[s - 1]["total"]
        for s in range(1, len(snapshots))
    ]


def divergence_step(structure: GraphStructure):
    """Return the jitted ``(params_a, params_b, batch, index, valid)`` KL sums."""
    output = structure.task_map["y"]

    def log_probs(params, batch, index):
        state = compute_forward_pass(
            params, structure, batch, jax.random.PRNGKey(0), training=False
        )
        hidden = head_input(structure, state, output)
        gathered = hidden.reshape(-1, hidden.shape[-1])[index]
        head = params.nodes[output]
        logits = gathered @ head.weights["W_out"] + head.biases["b_out"]
        return jax.nn.log_softmax(logits, axis=-1)

    def step(params_a, params_b, batch, index, valid):
        la = log_probs(params_a, batch, index)
        lb = log_probs(params_b, batch, index)
        sym = 0.5 * jnp.sum((jnp.exp(la) - jnp.exp(lb)) * (la - lb), axis=-1)
        weight = valid.astype(sym.dtype)
        return jnp.sum(sym * weight), jnp.sum(weight)

    return jax.jit(step)


def prediction_divergence(
    structure: GraphStructure,
    valid_set: Mapping[str, np.ndarray],
    params_a: GraphParams,
    params_b: GraphParams,
    *,
    batch_size: int = 64,
    step: Any = None,
) -> float:
    """Mean symmetric KL over the masked tokens of one fixed validation set.

    The last batch is padded with windows whose loss mask is empty, so every
    masked token counts and one jitted step serves every batch.
    """
    step = step or divergence_step(structure)
    names = ("masked_input_ids", "labels", "loss_mask", "attention_mask")
    arrays = {name: np.asarray(valid_set[name]) for name in names}
    pad = -len(arrays["loss_mask"]) % batch_size
    if pad:
        for name, value in arrays.items():
            filler = np.repeat(value[:1], pad, axis=0)
            if name == "loss_mask":
                filler = np.zeros_like(filler)
            arrays[name] = np.concatenate([value, filler])
    masks = arrays["loss_mask"].astype(bool)
    max_masked = max(
        1,
        max(
            int(masks[i : i + batch_size].sum())
            for i in range(0, len(masks), batch_size)
        ),
    )
    total, count = 0.0, 0.0
    for start in range(0, len(masks), batch_size):
        stop = start + batch_size
        flat = np.flatnonzero(masks[start:stop].reshape(-1))
        index = np.zeros(max_masked, dtype=np.int32)
        valid = np.zeros(max_masked, dtype=bool)
        index[: len(flat)] = flat
        valid[: len(flat)] = True
        batch = {
            "x": jnp.asarray(arrays["masked_input_ids"][start:stop]),
            "y": jnp.asarray(arrays["labels"][start:stop]),
            "loss_mask": jnp.asarray(arrays["loss_mask"][start:stop]),
            "attention_mask": jnp.asarray(arrays["attention_mask"][start:stop]),
        }
        s, c = step(params_a, params_b, batch, jnp.asarray(index), jnp.asarray(valid))
        total += float(s)
        count += float(c)
    if count == 0:
        raise ValueError("validation set has no masked tokens")
    return total / count


def run_key(context: RunContext) -> tuple[str, tuple[str, ...], int]:
    """Return ``(condition label, domain order, seed)`` of a run."""
    label = describe_gate(context.run_config["config"])[2]
    return label, tuple(context.domains), int(context.config.seed)


def compute_commutator(
    run_dirs: Iterable[str | Path],
    *,
    divergence: bool = True,
    batch_size: int = 64,
    checkpoint: str | None = None,
    data_dir: str | None = None,
) -> dict[str, Any]:
    """Return order distances, noise floors and chord ratios per pair and condition.

    Every run needs two or more domains, the same checkpoint params and no
    init perturbation; a pair needs both orders of each seed it reports.
    """
    runs: dict[tuple[str, tuple[str, ...], int], dict[str, Any]] = {}
    start_sha: str | None = None
    for run_dir in run_dirs:
        context = load_run(run_dir, checkpoint=checkpoint, data_dir=data_dir)
        if context.config.init_perturbation != 0.0:
            raise ValueError(f"{run_dir}: perturbed start")
        sha = context.run_config["source"]["params_sha256"]
        if start_sha is None:
            start_sha = sha
        elif sha != start_sha:
            raise ValueError(f"{run_dir}: different start params")
        key = run_key(context)
        if key in runs:
            raise ValueError(f"{run_dir}: duplicate run for {key}")
        stages = read_paths(context.run_dir, len(context.domains))
        chords = chord_to_path(context.snapshots, stages)
        context.snapshots = [context.snapshots[-1]]
        runs[key] = {
            "context": context,
            "stages": stages,
            "chords": chords,
            "path": run_path(stages),
            "run_dir": str(context.run_dir),
        }
    groups: dict[tuple[str, frozenset[str]], dict[tuple[str, ...], dict[int, Any]]] = {}
    for (label, order, seed), run in runs.items():
        groups.setdefault((label, frozenset(order)), {}).setdefault(order, {})[
            seed
        ] = run
    steps: dict[int, Any] = {}
    results = []
    for (label, pair), orders in sorted(
        groups.items(), key=lambda kv: (sorted(kv[0][1]), kv[0][0])
    ):
        if len(orders) != 2:
            raise ValueError(f"{label} {sorted(pair)}: needs both domain orders")
        first, second = sorted(orders)
        if first != tuple(reversed(second)):
            raise ValueError(f"{label}: orders {first} and {second} are not reverses")
        seeds = sorted(set(orders[first]) & set(orders[second]))
        per_seed = []
        for seed in seeds:
            a, b = orders[first][seed], orders[second][seed]
            entry: dict[str, Any] = {
                "seed": seed,
                "runs": [a["run_dir"], b["run_dir"]],
                **path_ratio(
                    a["context"].snapshots[-1],
                    b["context"].snapshots[-1],
                    a["path"],
                    b["path"],
                ),
            }
            if divergence:
                params_a = a["context"].params(-1)
                params_b = b["context"].params(-1)
                structure = a["context"].structure
                valid_sets = a["context"].valid_sets
                step = steps.setdefault(id(structure), divergence_step(structure))
                entry["symmetric_kl"] = {
                    name: prediction_divergence(
                        structure,
                        valid_sets[name],
                        params_a,
                        params_b,
                        batch_size=batch_size,
                        step=step,
                    )
                    for name in sorted(valid_sets)
                }
                ce_a = evaluate_domains(
                    params_a, structure, valid_sets, batch_size=batch_size
                )
                ce_b = evaluate_domains(
                    params_b, structure, valid_sets, batch_size=batch_size
                )
                entry["ce_difference"] = {
                    name: ce_a[name]["masked_ce"] - ce_b[name]["masked_ce"]
                    for name in sorted(valid_sets)
                }
            per_seed.append(entry)
        noise: dict[str, list[dict[str, Any]]] = {}
        chord_ratios: dict[str, dict[str, list[float]]] = {}
        for order in (first, second):
            members = orders[order]
            noise["->".join(order)] = [
                {
                    "seeds": [s, t],
                    **path_ratio(
                        members[s]["context"].snapshots[-1],
                        members[t]["context"].snapshots[-1],
                        members[s]["path"],
                        members[t]["path"],
                    ),
                }
                for s, t in itertools.combinations(sorted(members), 2)
            ]
            chord_ratios["->".join(order)] = {
                str(seed): run["chords"] for seed, run in sorted(members.items())
            }
        noise_ratios = [n["ratio"] for values in noise.values() for n in values]
        results.append(
            {
                "condition": label,
                "pair": sorted(pair),
                "orders": ["->".join(first), "->".join(second)],
                "seeds": seeds,
                "per_seed": per_seed,
                "D": mean_sd([e["ratio"] for e in per_seed]),
                "noise": noise,
                "N": mean_sd(noise_ratios),
                "chord_to_path": chord_ratios,
            }
        )
    return {"start_params_sha256": start_sha, "results": results}


def commutator_test(
    result: Mapping[str, Any], gate: str, control: str, pair: Sequence[str]
) -> dict[str, Any]:
    """Test ``D_gate < D_control`` on one pair (seeds paired by seed)."""
    rows = {r["condition"]: r for r in result["results"] if r["pair"] == sorted(pair)}
    if gate not in rows or control not in rows:
        raise ValueError(f"{sorted(pair)}: needs conditions {gate} and {control}")
    gate_d = {e["seed"]: e["ratio"] for e in rows[gate]["per_seed"]}
    control_d = {e["seed"]: e["ratio"] for e in rows[control]["per_seed"]}
    seeds = sorted(set(gate_d) & set(control_d))
    gaps = [control_d[s] - gate_d[s] for s in seeds]
    control_sd = mean_sd([control_d[s] for s in seeds])["sd"]
    mean_gap = float(np.mean(gaps)) if gaps else None
    passed = bool(
        seeds
        and all(g > 0 for g in gaps)
        and control_sd is not None
        and mean_gap is not None
        and mean_gap > control_sd
    )
    return {
        "gate": gate,
        "control": control,
        "pair": sorted(pair),
        "seeds": seeds,
        "gaps": gaps,
        "mean_gap": mean_gap,
        "control_sd": control_sd,
        "passed": passed,
    }


def format_table(result: Mapping[str, Any]) -> str:
    """Return a plain-text table of D and N per pair and condition."""

    def fmt(stats: Mapping[str, Any]) -> str:
        if stats["mean"] is None:
            return "-"
        sd = "" if stats["sd"] is None else f" ± {stats['sd']:.4g}"
        return f"{stats['mean']:.4g}{sd} (n={stats['n']})"

    lines = ["pair | condition | D | N"]
    for row in result["results"]:
        lines.append(
            f"{'/'.join(row['pair'])} | {row['condition']} | {fmt(row['D'])} | {fmt(row['N'])}"
        )
    return "\n".join(lines)
