"""Stage-level per-module update gates for the domain-forgetting runner.

A gate fixes one multiplier per module for a whole stage; the runner chains
``scale_module_updates`` after AdamW with them, so a multiplier scales the
module's final update and 0 freezes it. Modules with multiplier exactly 0
also have their gradients zeroed before the global-norm clip, so a frozen
module does not shrink the clipped step of the others; other multipliers
leave the clip unchanged (it sees the full gradient). Stage 1 is always
ungated. From
stage 2 on, with ``B`` the block modules (``L{i}_*``) and ``s_i`` a score
per block module, ``m_i = min(1, (median_B s / s_i)^β)`` (``map``,
``oracle``) or ``m_i = min(1, (s_i / median_B s)^β)`` (``inverse``); every
other module keeps 1.

- ``map``: ``s_i`` is the onset error map ``energy_sum_ce`` =
  ``½‖∂(Σ_masked CE_b)/∂z_i‖²`` of BP's error signal at node ``i`` (the
  spike-only PC map, ``error_maps.bp_signal_energies``), averaged over the
  windows with a masked token among the first ``gate_map_windows``
  unshuffled windows of the new domain's train stream, masked with seed
  ``seed + stage``, at the stage's start params with dropout off.
- ``random``: the ``map`` multipliers permuted across the block modules by
  ``numpy.random.default_rng([gate_seed, seed, stage])``.
- ``inverse``: the reversed rule on the onset map.
- ``type``: a module-type control; ``s_i`` is the mean onset map over the
  block modules of ``i``'s type (the name after ``L{j}_``: ``mha``,
  ``mlp1``, ``mlp2``) at that stage.
- ``oracle``: ``s_i`` is the mean over the ``gate_source`` runs of module
  ``i``'s rollback recovery of the reference domain for the transition
  ``S{stage}<-S{stage-1}``, floored at ``SCORE_FLOOR`` because a recovery
  can be zero or negative. Sources must be ungated sequential runs from the
  same checkpoint with as many domains; their domain order may differ, so
  the scores are a module-level ranking across orders.
- ``freeze``: the ``gate_modules`` modules or groups get 0, all others 1.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from fabricpc.core.types import GraphParams, GraphStructure
from fabricpc.experiments.domain_forgetting.artifacts import file_sha256
from fabricpc.experiments.domain_forgetting.error_maps import (
    bp_signal_energies,
    hidden_nodes,
)
from fabricpc.experiments.domain_forgetting.localization import (
    block_index,
    check_provenance,
    module_groups,
    module_names,
    transition_name,
)
from fabricpc.experiments.domain_forgetting.runner import (
    GATE_SETTINGS,
    BaseModel,
    RunConfig,
    build_inference,
    build_structure,
    fixed_mlm_batches,
    flatten_params,
)
from fabricpc.experiments.forgetting_benchmark.sequential_trainer import params_hash

GATE_MAP_BATCH = 8
SCORE_FLOOR = 1e-6
MAP_GATES = ("map", "random", "inverse", "type")
MLP_PATTERN = re.compile(r"^L\d+_mlp\d*$")
EnergyFn = Callable[[GraphParams, dict[str, jax.Array]], dict[str, jax.Array]]


def block_modules(modules: Sequence[str]) -> list[str]:
    """Return the ``L{i}_*`` block modules in order."""
    return [m for m in modules if block_index(m) is not None]


def freeze_groups(modules: Sequence[str]) -> dict[str, list[str]]:
    """Return ``module_groups`` plus ``blocks`` (every block) and ``mlp``."""
    groups = dict(module_groups(modules))
    groups["blocks"] = block_modules(modules)
    groups["mlp"] = [m for m in modules if MLP_PATTERN.match(m)]
    return {name: members for name, members in groups.items() if members}


def resolve_modules(modules: Sequence[str], names: Sequence[str]) -> list[str]:
    """Expand module and group names into modules, in graph order."""
    groups = freeze_groups(modules)
    chosen: set[str] = set()
    for name in names:
        if name in groups:
            chosen.update(groups[name])
        elif name in modules:
            chosen.add(name)
        else:
            raise ValueError(
                f"unknown module or group {name!r}; modules {list(modules)}, "
                f"groups {sorted(groups)}"
            )
    return [m for m in modules if m in chosen]


def map_multipliers(
    scores: Mapping[str, float], beta: float, *, inverse: bool = False
) -> dict[str, float]:
    """``min(1, (median / s_i)^β)`` per module, ``min(1, (s_i / median)^β)`` inverse."""
    if not scores:
        raise ValueError("gate scores are empty")
    bad = [m for m, v in scores.items() if not (math.isfinite(v) and v > 0)]
    if bad:
        raise ValueError(f"gate scores must be finite and positive; not at {bad}")
    median = float(np.median(np.asarray(list(scores.values()), np.float64)))
    return {
        m: min(1.0, float(((v / median) if inverse else (median / v)) ** beta))
        for m, v in scores.items()
    }


def module_type(module: str) -> str:
    """Return the name after the ``L{i}_`` prefix of a block module."""
    if block_index(module) is None:
        raise ValueError(f"{module!r} is not a block module")
    return module.split("_", 1)[1]


def type_scores(scores: Mapping[str, float]) -> dict[str, float]:
    """Replace each block module's score by the mean over its module type."""
    groups: dict[str, list[float]] = {}
    for module, value in scores.items():
        groups.setdefault(module_type(module), []).append(float(value))
    means = {name: float(np.mean(values)) for name, values in groups.items()}
    return {module: means[module_type(module)] for module in scores}


def permute_multipliers(
    multipliers: Mapping[str, float],
    order: Sequence[str],
    rng: np.random.Generator,
) -> tuple[dict[str, float], list[str]]:
    """Give ``order[i]`` the multiplier of ``source[i]`` for a random permutation.

    Returns the permuted multipliers and ``source``.
    """
    source = [order[j] for j in rng.permutation(len(order))]
    return {m: multipliers[s] for m, s in zip(order, source)}, source


def map_batches(
    stream: np.ndarray,
    structure: GraphStructure,
    special_ids: Mapping[str, int],
    *,
    num_windows: int,
    mask_prob: float,
    seed: int,
) -> list[dict[str, jax.Array]]:
    """The first ``num_windows`` unshuffled windows in batches of ``GATE_MAP_BATCH``.

    Windows are drawn one at a time (``fixed_mlm_batches`` with batch size 1,
    masks from ``seed``), so the window set and masks do not depend on the
    batching; the last batch may be smaller.
    """
    try:
        windows, _ = fixed_mlm_batches(
            stream,
            structure,
            special_ids,
            num_batches=num_windows,
            batch_size=1,
            mask_prob=mask_prob,
            seed=seed,
        )
    except ValueError as error:
        raise ValueError(
            f"gate_map_windows {num_windows} exceeds the train stream: {error}"
        ) from error
    return [
        {
            key: jnp.concatenate(
                [w[key] for w in windows[start : start + GATE_MAP_BATCH]]
            )
            for key in windows[0]
        }
        for start in range(0, num_windows, GATE_MAP_BATCH)
    ]


def onset_map(
    params: GraphParams,
    structure: GraphStructure,
    stream: np.ndarray,
    special_ids: Mapping[str, int],
    *,
    modules: Sequence[str],
    num_windows: int,
    mask_prob: float,
    seed: int,
    energy_fn: EnergyFn,
) -> dict[str, float]:
    """Mean ``energy_sum_ce`` per module over the first ``num_windows`` windows.

    ``energy_fn`` returns ``bp_signal_energies`` at output scale 1; each
    window's value times the batch's squared masked-token count is the
    batch-invariant ``½‖∂(Σ_masked CE_b)/∂z‖²``. Windows without a masked
    token are left out.
    """
    missing = sorted(set(modules) - set(hidden_nodes(structure)))
    if missing:
        raise ValueError(f"modules {missing} are not hidden nodes of the graph")
    batches = map_batches(
        stream,
        structure,
        special_ids,
        num_windows=num_windows,
        mask_prob=mask_prob,
        seed=seed,
    )
    sums = {m: 0.0 for m in modules}
    windows = 0
    for batch in batches:
        energies = jax.device_get(energy_fn(params, batch))
        mask = np.asarray(batch["loss_mask"], np.float64)
        valid = mask.sum(axis=-1) > 0
        factor = float(mask.sum()) ** 2
        windows += int(valid.sum())
        for m in modules:
            sums[m] += float(np.asarray(energies[m], np.float64)[valid].sum()) * factor
    if windows == 0:
        raise ValueError("no gate-map window has a masked token")
    return {m: total / windows for m, total in sums.items()}


def load_rollbacks(
    sources: Sequence[str], *, num_domains: int, params_sha256: str
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Read each source run's ``rollback.json``, checking source and provenance.

    A source must be an ungated sequential run of ``num_domains`` domains
    (in any order) from the checkpoint with ``params_sha256``. Returns the
    payloads and, per source, its ``rollback.json`` SHA-256, domains, mode
    and seed.
    """
    payloads: dict[str, dict[str, Any]] = {}
    info: dict[str, dict[str, Any]] = {}
    for source in sources:
        path = Path(source) / "rollback.json"
        if not path.is_file():
            raise FileNotFoundError(f"{path} is missing")
        run_config_path = Path(source) / "run_config.json"
        if not run_config_path.is_file():
            raise FileNotFoundError(f"{run_config_path} is missing")
        run_config = json.loads(run_config_path.read_text())
        values = run_config["config"]
        if values.get("training", "sequential") != "sequential":
            raise ValueError(f"{source}: oracle sources must be sequential runs")
        if values.get("gate", "none") != "none":
            raise ValueError(
                f"{source}: oracle sources must be ungated, not gate "
                f"{values['gate']!r}"
            )
        if len(values["domains"]) != num_domains:
            raise ValueError(
                f"{source}: has {len(values['domains'])} domains, the run has "
                f"{num_domains}"
            )
        if run_config["source"]["params_sha256"] != params_sha256:
            raise ValueError(f"{source}: trained from a different checkpoint")
        payload = json.loads(path.read_text())
        check_provenance(source, payload, "rollback.json")
        payloads[source] = payload
        info[source] = {
            "rollback_sha256": file_sha256(path),
            "domains": list(values["domains"]),
            "mode": values.get("mode"),
            "seed": values.get("seed"),
        }
    return payloads, info


def oracle_scores(
    rollbacks: Mapping[str, Mapping[str, Any]],
    stage: int,
    modules: Sequence[str],
) -> dict[str, float]:
    """Mean recovery of the reference domain for ``S{stage}<-S{stage-1}``.

    Averaged over the sources per module; not floored.
    """
    name = transition_name(stage, stage - 1)
    rows = []
    for source, payload in rollbacks.items():
        entry = next((t for t in payload["transitions"] if t["name"] == name), None)
        if entry is None:
            raise ValueError(f"{source}: rollback.json has no transition {name}")
        missing = [m for m in modules if m not in entry["modules"]]
        if missing:
            raise ValueError(f"{source}: rollback.json lacks modules {missing}")
        reference = entry["reference_domain"]
        rows.append(
            [float(entry["modules"][m]["recovery"][reference]) for m in modules]
        )
    mean = np.mean(np.asarray(rows, np.float64), axis=0)
    return {m: float(v) for m, v in zip(modules, mean)}


class GatePlan:
    """One run's gate: resolves its modules and sources, records each stage."""

    def __init__(
        self,
        config: RunConfig,
        base: BaseModel,
        streams: Mapping[str, np.ndarray],
    ) -> None:
        """``streams`` are the domains' train streams, used by the map gates."""
        self.config = config
        self.special_ids = dict(base.special_ids)
        self.modules = module_names(base.structure, flatten_params(base.params))
        self.blocks = block_modules(self.modules)
        if config.gate != "freeze" and not self.blocks:
            raise ValueError(f"gate {config.gate!r} needs L{{i}}_* block modules")
        self.frozen = (
            resolve_modules(self.modules, config.gate_modules)
            if config.gate == "freeze"
            else []
        )
        self.rollbacks: dict[str, dict[str, Any]] = {}
        self.sources: dict[str, dict[str, Any]] = {}
        if config.gate == "oracle":
            self.rollbacks, self.sources = load_rollbacks(
                config.gate_source,
                num_domains=len(config.domains),
                params_sha256=base.source["params_sha256"],
            )
            for stage in range(2, len(config.domains) + 1):
                oracle_scores(self.rollbacks, stage, self.blocks)
        self.structure: GraphStructure | None = None
        self.energy_fn: EnergyFn | None = None
        if config.gate in MAP_GATES:
            run = replace(config, mode="backprop", dropout_rate=0.0)
            structure = build_structure(
                base.model_config, run, build_inference(run, base.model_config)[0]
            )
            self.structure = structure
            self.energy_fn = jax.jit(
                lambda p, b: bp_signal_energies(p, structure, b, 1.0)
            )
            for stage in range(2, len(config.domains) + 1):
                map_batches(
                    streams[config.domains[stage - 1]],
                    structure,
                    self.special_ids,
                    num_windows=config.gate_map_windows,
                    mask_prob=config.mask_prob,
                    seed=config.seed + stage,
                )

    def describe(self) -> dict[str, Any]:
        """The resolved gate setup recorded in ``run_config.json``."""
        info: dict[str, Any] = {
            "gate": self.config.gate,
            "modules": list(self.modules),
            "block_modules": list(self.blocks),
            "placement": "after AdamW (scales the final update, weight decay "
            "included); stage 1 ungated",
            "clip": "gradients of modules with multiplier 0 are zeroed before "
            "the global-norm clip; other multipliers act after it",
        }
        if self.config.gate == "freeze":
            info["frozen_modules"] = list(self.frozen)
        if self.config.gate == "oracle":
            info["sources"] = {k: dict(v) for k, v in self.sources.items()}
            info["score_floor"] = SCORE_FLOOR
        if self.config.gate in MAP_GATES:
            info["map"] = {
                "energy": "energy_sum_ce of bp_signal_energies (spike-only map)",
                "stream": "train stream of the stage's domain, unshuffled",
                "windows": self.config.gate_map_windows,
                "batch_size": GATE_MAP_BATCH,
                "windows_masked": "one at a time (batch size 1)",
                "mask_seed": "seed + stage",
            }
        return info

    def record(
        self, stage: int, domain: str, params: GraphParams, stream: np.ndarray
    ) -> dict[str, Any]:
        """Return the stage's gate record, with ``multipliers`` for every module.

        ``params`` are the stage's start params and ``stream`` its domain's
        train stream; stage 1 is ungated (``gated`` false, multipliers 1).
        """
        started = time.time()
        config = self.config
        uses_beta = "gate_beta" in GATE_SETTINGS[config.gate]
        record: dict[str, Any] = {
            "stage": stage,
            "domain": domain,
            "gate": config.gate,
            "beta": config.gate_beta if uses_beta else None,
            "gated": stage >= 2,
            "params_sha256": params_hash(params),
            "map": None,
            "map_windows": None,
            "map_seed": None,
            "scores": None,
            "permutation": None,
        }
        multipliers = {m: 1.0 for m in self.modules}
        if stage >= 2:
            if config.gate == "freeze":
                multipliers.update({m: 0.0 for m in self.frozen})
            elif config.gate == "oracle":
                raw = oracle_scores(self.rollbacks, stage, self.blocks)
                scores = {m: max(v, SCORE_FLOOR) for m, v in raw.items()}
                record["scores"] = raw
                multipliers.update(map_multipliers(scores, config.gate_beta))
            else:
                if self.structure is None or self.energy_fn is None:
                    raise ValueError(f"gate {config.gate!r} has no onset map")
                values = onset_map(
                    params,
                    self.structure,
                    stream,
                    self.special_ids,
                    modules=self.blocks,
                    num_windows=config.gate_map_windows,
                    mask_prob=config.mask_prob,
                    seed=config.seed + stage,
                    energy_fn=self.energy_fn,
                )
                record.update(
                    map=values,
                    map_windows=config.gate_map_windows,
                    map_seed=config.seed + stage,
                )
                scores = values
                if config.gate == "type":
                    scores = type_scores(values)
                    record["scores"] = scores
                try:
                    block = map_multipliers(
                        scores, config.gate_beta, inverse=config.gate == "inverse"
                    )
                except ValueError as error:
                    raise ValueError(
                        f"stage {stage} ({domain}) onset map: {error}"
                    ) from error
                if config.gate == "random":
                    rng = np.random.default_rng([config.gate_seed, config.seed, stage])
                    block, record["permutation"] = permute_multipliers(
                        block, self.blocks, rng
                    )
                multipliers.update(block)
        record["multipliers"] = multipliers
        record["seconds"] = time.time() - started
        return record
