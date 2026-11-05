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
- ``uniform``: every block module gets ``gate_scale``, all others 1; the
  step-size control matched to a gate's update path.
- ``uniform_all``: every parameterised module gets ``gate_scale``; the control
  for methods that also slow the embeddings and the head.

The online gates (``online_protect``, ``online_surprise``) refresh their
block multipliers inside a stage (``OnlineGate``). The smoothed energy
``ŝ_i`` starts at the ``map`` gate's onset map; after every update ``u``
with ``u % gate_interval == 0`` and ``u`` before the stage's last, the raw
``s_i`` of ``onset_map`` on a probe of ``GATE_MAP_BATCH`` windows at the
live params (``probe_offset``, ``probe_seed``) updates
``ŝ_i ← (1 − α) ŝ_i + α s_i`` with ``α = gate_ema``, and updates
``u + 1 … u + gate_interval`` use the multipliers of the new ``ŝ``
(``online_multipliers``): ``online_protect`` is ``map_multipliers(ŝ, β)``,
so with ``gate_interval`` at least the stage length it is the ``map``
gate, and ``online_surprise`` is ``ŝ_i / (ŝ_i + τ_i)`` with ``τ_i`` the
onset map, 0.5 at onset. ``gate_variant`` ``type`` uses module-type means
of ``ŝ`` (and of ``τ``), ``shuffle`` permutes the block multipliers at
every refresh, ``inverse`` reverses the rule and ``replay`` applies the
recorded schedule of the ``gate_source`` run without computing energies.
Each gated stage's periods are written to ``gate_schedule_stage_<s>.jsonl``
with the multipliers the optimizer state held.

With ``gate_map_source="learner"`` every map-derived and online gate reads
the run's own learner errors instead of BP's signal: ``s_i`` is
``energy_sum_ce`` of ``error_maps.learner_signal_energies`` on the graph of
``error_maps.learner_map_structure`` (ePC at its starting rate
``pc_eta_infer``, fixed for the whole run), on the same windows and with the
same normalisation; backprop runs keep ``bp_signal_energies``, so their
multipliers are those of ``"bp"``. The gate records and schedules of such a
run carry ``map_source``, and a replay needs a source run of the same map
source.

``online_influence`` multiplies the ``online_surprise`` multiplier ``g_i`` of
each refresh by ``(1 − α) + α h_i`` and clips the product to ``[0, 1]``
(``plasticity_gates.weighted_gates``, α = ``gate_influence_alpha``), the bound
every other gate's multipliers already satisfy. From a stage's second refresh
on, ``h_i`` comes from ``I_i = L(θ with module i at the previous refresh's
params) − L(θ)``, ``L`` the masked CE of the validation evaluator on a fixed
probe of ``INFLUENCE_PROBE_WINDOWS`` windows (``influence_weights``,
``InfluenceProbe``); the onset and the stage's first refresh have no window
and use ``h = 1``. The probe is windows evenly spaced over the stage's train
stream (variant ``past``: the previous domain's), masked once, since nearby
windows share a few documents and give noisier influence. Train streams share
no document with any validation set (``data.DEDUP_POLICY``), so the probe is
disjoint from the reported metrics; every train window is also trained on once
per stage. It skips the first ``INFLUENCE_PROBE_SKIP`` windows, past the map and
refresh windows, when the stream is long enough, as in the benchmark. Each
refresh logs the split-half Spearman of ``I`` over alternate probe windows, the
floored and positive fractions, and ``I_i`` over the multiplier held during its
window, since the realised update already carries it. A stage starts with no
window, so nothing is differenced across stages, and the window is not stored:
a stage is resumed whole or retrained whole. Stage 1 is ungated in both
variants, so a stage without a previous domain never evaluates a probe.
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
from scipy import stats

from fabricpc.core.types import GraphParams, GraphStructure
from fabricpc.experiments.domain_forgetting.artifacts import (
    arrays_sha256,
    file_sha256,
)
from fabricpc.experiments.domain_forgetting.error_maps import (
    bp_signal_energies,
    hidden_nodes,
    learner_energy_field,
    learner_map_structure,
    learner_signal_energies,
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
    ONLINE_GATES,
    BaseModel,
    RunConfig,
    build_inference,
    build_structure,
    fixed_mlm_batches,
    flatten_params,
    num_stage_updates,
)
from fabricpc.experiments.domain_forgetting.validation import MultiDomainEvaluator
from fabricpc.experiments.forgetting_benchmark.sequential_trainer import params_hash
from fabricpc.training.plasticity_gates import (
    module_multipliers,
    set_module_multipliers,
    weighted_gates,
)

GATE_MAP_BATCH = 8
SCORE_FLOOR = 1e-6
INFLUENCE_PROBE_WINDOWS = 256
INFLUENCE_PROBE_BATCH = 64
INFLUENCE_PROBE_SKIP = 1024
INFLUENCE_REL_FLOOR = 1e-6
INFLUENCE_WEIGHT_MAX = 2.0
INFLUENCE_PROBE_TAG = 0x696E66
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


def stream_windows(stream: np.ndarray, seq_len: int) -> int:
    """Number of unshuffled windows (``seq_len - 1`` tokens after ``[CLS]``)."""
    return math.ceil(len(stream) / (seq_len - 1))


def window_stream(
    stream: np.ndarray, seq_len: int, *, offset: int, count: int, pad_id: int
) -> np.ndarray:
    """The tokens of windows ``offset … offset + count - 1`` modulo the window count.

    A short last window is padded with ``pad_id``, so each window of the
    result is the stream's window, padding and attention included.
    """
    stride = seq_len - 1
    total = stream_windows(stream, seq_len)
    segments = []
    for k in range(count):
        start = ((offset + k) % total) * stride
        segment = np.asarray(stream[start : start + stride], dtype=np.int32)
        segments.append(
            np.pad(segment, (0, stride - len(segment)), constant_values=pad_id)
        )
    return np.concatenate(segments)


def map_batches(
    stream: np.ndarray,
    structure: GraphStructure,
    special_ids: Mapping[str, int],
    *,
    num_windows: int,
    mask_prob: float,
    seed: int,
    offset: int = 0,
) -> list[dict[str, jax.Array]]:
    """The first ``num_windows`` unshuffled windows in batches of ``GATE_MAP_BATCH``.

    Windows are drawn one at a time (``fixed_mlm_batches`` with batch size 1,
    masks from ``seed``), so the window set and masks do not depend on the
    batching; the last batch may be smaller. With ``offset`` > 0 the windows
    start at window ``offset`` and wrap modulo the stream's window count
    (``window_stream``).
    """
    if offset:
        seq_len = int(structure.nodes[structure.task_map["y"]].node_info.shape[0])
        stream = window_stream(
            stream,
            seq_len,
            offset=offset,
            count=num_windows,
            pad_id=special_ids["[PAD]"],
        )
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
    offset: int = 0,
) -> dict[str, float]:
    """Mean ``energy_sum_ce`` per module over ``num_windows`` windows from ``offset``.

    ``energy_fn`` returns ``bp_signal_energies`` at output scale 1 (or
    ``learner_signal_energies``, in the same units); each window's value times
    the batch's squared masked-token count is the batch-invariant
    ``½‖∂(Σ_masked CE_b)/∂z‖²`` (the learner's ``½‖e_b‖² / s²``). Windows
    without a masked token are left out. The windows are ``map_batches``'.
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
        offset=offset,
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


def gate_periods(num_updates: int, interval: int) -> list[tuple[int, int, int]]:
    """``(refresh, first_update, last_update)`` of each online gate period.

    The onset period is refresh -1; refresh ``r`` follows update
    ``(r + 1) * interval`` and lasts to the next refresh or the stage's end.
    """
    starts = [0, *range(interval, num_updates, interval)]
    return [
        (index - 1, start + 1, min(start + interval, num_updates))
        for index, start in enumerate(starts)
    ]


def probe_offset(map_windows: int, refresh: int, num_windows: int) -> int:
    """First probe window of ``refresh``: ``map_windows + refresh * GATE_MAP_BATCH``.

    Taken modulo the stream's ``num_windows``.
    """
    return (map_windows + refresh * GATE_MAP_BATCH) % num_windows


def probe_seed(seed: int, stage: int, refresh: int) -> int:
    """Mask seed of a probe, from ``SeedSequence([seed, stage, refresh])``, < 2^31."""
    state = np.random.SeedSequence([seed, stage, refresh]).generate_state(1)
    return int(state[0] >> 1)


def shuffle_rng(
    gate_seed: int, seed: int, stage: int, refresh: int
) -> np.random.Generator:
    """``default_rng([gate_seed, seed, stage, refresh])``; the onset (-1) drops ``refresh``.

    The onset generator is the ``random`` gate's, so a stage without
    refreshes permutes as that gate does.
    """
    entropy = [gate_seed, seed, stage] + ([refresh] if refresh >= 0 else [])
    return np.random.default_rng(entropy)


def ema_update(
    smoothed: Mapping[str, float], raw: Mapping[str, float], alpha: float
) -> dict[str, float]:
    """``(1 − α) ŝ_i + α s_i`` per module."""
    return {m: (1.0 - alpha) * v + alpha * float(raw[m]) for m, v in smoothed.items()}


def surprise_multipliers(
    scores: Mapping[str, float],
    tau: Mapping[str, float],
    *,
    inverse: bool = False,
) -> dict[str, float]:
    """``s_i / (s_i + τ_i)`` per module, ``τ_i / (s_i + τ_i)`` inverse."""
    return {
        m: (float(tau[m]) if inverse else v) / (v + float(tau[m]))
        for m, v in scores.items()
    }


def online_multipliers(
    smoothed: Mapping[str, float],
    onset: Mapping[str, float],
    *,
    gate: str,
    variant: str,
    beta: float,
    rng: np.random.Generator | None = None,
) -> tuple[dict[str, float], list[str] | None]:
    """The block multipliers of an online gate at ``smoothed``, and the permutation.

    Scores and ``τ = onset`` are floored at ``SCORE_FLOOR``; ``type`` replaces
    both by their module-type means, so the surprise gate still starts at
    0.5; ``shuffle`` permutes the multipliers with ``rng``
    (``permute_multipliers``). Every multiplier is positive.
    """
    scores = {m: max(float(v), SCORE_FLOOR) for m, v in smoothed.items()}
    tau = {m: max(float(onset[m]), SCORE_FLOOR) for m in scores}
    if variant == "type":
        scores, tau = type_scores(scores), type_scores(tau)
    inverse = variant == "inverse"
    if gate == "online_protect":
        block = map_multipliers(scores, beta, inverse=inverse)
    elif gate in ("online_surprise", "online_influence"):
        block = surprise_multipliers(scores, tau, inverse=inverse)
    else:
        raise ValueError(f"{gate!r} is not an online gate")
    permutation = None
    if variant == "shuffle":
        if rng is None:
            raise ValueError("variant 'shuffle' needs an rng")
        block, permutation = permute_multipliers(block, list(scores), rng)
    return block, permutation


def influence_weights(
    influence: Mapping[str, float], floor: float = 0.0
) -> dict[str, float]:
    """Positive influence over the mean positive influence, clipped to [0, 2].

    Entries with ``|I| < floor`` carry no signal and get the neutral weight 1.
    Among the rest, non-positive entries get 0; all ones when no entry at or
    above the floor is positive.
    """
    values = np.asarray(list(influence.values()), dtype=np.float64)
    if values.size and not np.all(np.isfinite(values)):
        raise ValueError("non-finite influence")
    positive = values[(np.abs(values) >= floor) & (values > 0)]
    if positive.size == 0:
        return {m: 1.0 for m in influence}
    mean_positive = float(positive.mean())
    weights = {}
    for module, value in influence.items():
        if abs(value) < floor:
            weights[module] = 1.0
        elif value > 0:
            weights[module] = float(
                np.clip(value / mean_positive, 0.0, INFLUENCE_WEIGHT_MAX)
            )
        else:
            weights[module] = 0.0
    return weights


def influence_multipliers(
    surprise: Mapping[str, float], h: Mapping[str, float], alpha: float
) -> dict[str, float]:
    """``clip(g_i ((1 − α) + α h_i), 0, 1)`` per module, as Python floats."""
    weighted = weighted_gates(
        {m: jnp.asarray(v) for m, v in surprise.items()},
        {m: jnp.asarray(h[m]) for m in surprise},
        alpha,
    )
    return {m: float(v) for m, v in weighted.items()}


def spaced_windows(stream: np.ndarray, seq_len: int, count: int) -> np.ndarray:
    """``count`` evenly spaced full windows of ``stream``, skipping the first ones.

    The skip applies only when ``count`` windows remain after it; a stream with
    fewer than ``count`` full windows repeats some.
    """
    total = len(stream) // (seq_len - 1)
    first = INFLUENCE_PROBE_SKIP if total - INFLUENCE_PROBE_SKIP >= count else 0
    return np.linspace(first, total - 1, count).astype(np.int64)


def spaced_batches(
    stream: np.ndarray,
    structure: GraphStructure,
    special_ids: Mapping[str, int],
    *,
    num_windows: int,
    mask_prob: float,
    seed: int,
) -> tuple[list[dict[str, jax.Array]], np.ndarray]:
    """``spaced_windows`` of the stream as ``map_batches``, and the window indices."""
    seq_len = int(structure.nodes[structure.task_map["y"]].node_info.shape[0])
    windows = spaced_windows(stream, seq_len, num_windows)
    tokens = np.concatenate(
        [
            window_stream(
                stream,
                seq_len,
                offset=int(window),
                count=1,
                pad_id=special_ids["[PAD]"],
            )
            for window in windows
        ]
    )
    batches = map_batches(
        tokens,
        structure,
        special_ids,
        num_windows=num_windows,
        mask_prob=mask_prob,
        seed=seed,
    )
    return batches, windows


def spaced_probe(
    stream: np.ndarray,
    structure: GraphStructure,
    special_ids: Mapping[str, int],
    *,
    num_windows: int,
    mask_prob: float,
    seed: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """``spaced_batches`` of the stream as evaluator arrays, and the window indices.

    Masks are drawn with ``seed`` (``map_batches``) and no other random state
    is read, so the probe depends only on its arguments.
    """
    batches, windows = spaced_batches(
        stream,
        structure,
        special_ids,
        num_windows=num_windows,
        mask_prob=mask_prob,
        seed=seed,
    )
    joined = {
        key: np.concatenate([np.asarray(b[key]) for b in batches]) for key in batches[0]
    }
    arrays = {
        "input_ids": joined["x"],
        "masked_input_ids": joined["x"],
        "labels": joined["y"],
        "loss_mask": joined["loss_mask"],
        "attention_mask": joined["attention_mask"],
    }
    return arrays, windows


def with_module_from(
    params: GraphParams, source: GraphParams, module: str
) -> GraphParams:
    """``params`` with ``module``'s parameters taken from ``source``."""
    return GraphParams(nodes={**params.nodes, module: source.nodes[module]})


class InfluenceProbe:
    """One stage's leave-one-module-out influence on a fixed probe.

    ``measure`` reads the params it is given and holds a reference to them
    (JAX arrays are immutable) as the start of the next window. The loss is
    the jitted forward pass of ``MultiDomainEvaluator``, dropout off with its
    own fixed key, so no training state or random stream is read or written.
    Restoring module ``i`` to the window's start params is ``θ − Δθ_i``
    without the rounding of a subtraction.
    """

    def __init__(
        self,
        structure: GraphStructure,
        arrays: Mapping[str, np.ndarray],
        modules: Sequence[str],
    ) -> None:
        """``arrays`` is the probe (``spaced_probe``), ``modules`` the block modules."""
        self.evaluator = MultiDomainEvaluator(
            structure,
            {
                "probe": arrays,
                "half_a": {k: v[::2] for k, v in arrays.items()},
                "half_b": {k: v[1::2] for k, v in arrays.items()},
            },
            min(INFLUENCE_PROBE_BATCH, len(arrays["input_ids"])),
        )
        self.modules = list(modules)
        self.start: GraphParams | None = None

    def losses(self, params: GraphParams) -> dict[str, float]:
        """Masked CE of ``params`` on the probe and on each half of it."""
        return {
            name: float(result["masked_ce"])
            for name, result in self.evaluator(params).items()
        }

    def loss(self, params: GraphParams) -> float:
        """Masked CE of ``params`` on the probe."""
        return self.losses(params)["probe"]

    def measure(self, params: GraphParams) -> dict[str, Any]:
        """``influence``, ``influence_h``, floor and probe loss of one refresh.

        The stage's first call has no window: ``influence`` is ``None`` and
        ``h`` is 1, with no forward pass.
        """
        start, self.start = self.start, params
        if start is None:
            return {
                "influence": None,
                "influence_h": {m: 1.0 for m in self.modules},
                "influence_floor": None,
                "influence_probe_loss": None,
                "influence_split_half_spearman": None,
                "influence_floored_fraction": None,
                "influence_positive_fraction": None,
            }
        base_losses = self.losses(params)
        base = base_losses["probe"]
        removed = {
            m: self.losses(with_module_from(params, start, m)) for m in self.modules
        }
        influence = {m: removed[m]["probe"] - base for m in self.modules}
        floor = INFLUENCE_REL_FLOOR * abs(base)
        values = np.asarray(list(influence.values()))
        halves = [
            [removed[m][half] - base_losses[half] for m in self.modules]
            for half in ("half_a", "half_b")
        ]
        agreement = (
            float(stats.spearmanr(*halves).statistic)
            if len(self.modules) > 2 and np.isfinite(halves).all()
            else math.nan
        )
        return {
            "influence_split_half_spearman": (
                agreement if math.isfinite(agreement) else None
            ),
            "influence_floored_fraction": float(np.mean(np.abs(values) < floor)),
            "influence_positive_fraction": float(np.mean(values >= floor)),
            "influence": influence,
            "influence_h": influence_weights(influence, floor),
            "influence_floor": floor,
            "influence_probe_loss": base,
        }


def signal_energy_fn(
    config: RunConfig, model_config: Mapping[str, Any], *, learner: bool
) -> tuple[GraphStructure, EnergyFn, str | None]:
    """The map's graph, jitted energies and field: the learner's errors or BP's signal."""
    if learner:
        structure = learner_map_structure(model_config, config)
        field = learner_energy_field(config)
        return (
            structure,
            jax.jit(lambda p, b: learner_signal_energies(p, structure, b, field)),
            field,
        )
    run = replace(config, mode="backprop", dropout_rate=0.0)
    structure = build_structure(
        model_config, run, build_inference(run, model_config)[0]
    )
    return (
        structure,
        jax.jit(lambda p, b: bp_signal_energies(p, structure, b, 1.0)),
        None,
    )


def schedule_name(stage: int) -> str:
    """The file of stage ``stage``'s online gate schedule."""
    return f"gate_schedule_stage_{stage}.jsonl"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def load_replay_schedules(
    source: str,
    config: RunConfig,
    *,
    modules: Sequence[str],
    num_updates: int,
    params_sha256: str,
) -> tuple[dict[int, list[dict[str, Any]]], dict[str, Any]]:
    """Read the source run's gated-stage schedules, checking it can be replayed.

    The source must be a ``live`` run of the same online gate from the same
    checkpoint params, with the same domain order, mode, ``gate_beta``,
    ``gate_ema``, ``gate_map_windows``, ``gate_map_source``, per-stage update
    count and
    ``gate_interval``, and a complete schedule
    for every gated stage whose SHA-256 its ``gates.jsonl`` records. Returns
    the schedules by stage and the source's description.
    """
    root = Path(source)
    run_config_path = root / "run_config.json"
    if not run_config_path.is_file():
        raise FileNotFoundError(f"{run_config_path} is missing")
    run_config = json.loads(run_config_path.read_text())
    values = run_config["config"]
    if values.get("gate") != config.gate:
        raise ValueError(
            f"{source}: replay needs a {config.gate!r} run, not gate "
            f"{values.get('gate', 'none')!r}"
        )
    if list(values["domains"]) != list(config.domains):
        raise ValueError(
            f"{source}: domain order {values['domains']} differs from "
            f"{list(config.domains)}"
        )
    if run_config.get("updates_per_stage") != num_updates:
        raise ValueError(
            f"{source}: has {run_config.get('updates_per_stage')} updates per "
            f"stage, the run has {num_updates}"
        )
    if values.get("gate_variant", "live") != "live":
        raise ValueError(
            f"{source}: replay needs a live source, not variant "
            f"{values.get('gate_variant')!r}"
        )
    names = ["mode", "gate_beta", "gate_ema", "gate_map_windows"]
    if config.gate == "online_influence":
        names.append("gate_influence_alpha")
    for name in names:
        if values.get(name) != getattr(config, name):
            raise ValueError(
                f"{source}: {name} {values.get(name)!r} differs from "
                f"{getattr(config, name)!r}"
            )
    map_source = values.get("gate_map_source", "bp")
    if map_source != config.gate_map_source:
        raise ValueError(
            f"{source}: gate_map_source {map_source!r} differs from "
            f"{config.gate_map_source!r}"
        )
    if run_config["source"]["params_sha256"] != params_sha256:
        raise ValueError(f"{source}: not trained from the run's checkpoint")
    if values.get("gate_interval") != config.gate_interval:
        raise ValueError(
            f"{source}: gate_interval {values.get('gate_interval')} differs from "
            f"{config.gate_interval}"
        )
    gates_path = root / "gates.jsonl"
    records = (
        {int(r["stage"]): r for r in _read_jsonl(gates_path)}
        if (gates_path.is_file())
        else {}
    )
    periods = gate_periods(num_updates, config.gate_interval)
    schedules: dict[int, list[dict[str, Any]]] = {}
    digests: dict[str, str] = {}
    for stage in range(2, len(config.domains) + 1):
        path = root / schedule_name(stage)
        summary = (records.get(stage) or {}).get("schedule")
        if summary is None or not path.is_file():
            raise ValueError(f"{source}: stage {stage} has no complete gate schedule")
        digest = file_sha256(path)
        if digest != summary["sha256"]:
            raise ValueError(f"{path}: SHA-256 differs from gates.jsonl")
        schedule = _read_jsonl(path)
        found = [
            (int(r["refresh"]), int(r["first_update"]), int(r["last_update"]))
            for r in schedule
        ]
        if found != periods:
            raise ValueError(
                f"{path}: periods {found} are not the {len(periods)} expected"
            )
        for entry in schedule:
            if sorted(entry["multipliers"]) != sorted(modules):
                raise ValueError(f"{path}: multipliers do not cover the modules")
        schedules[stage] = schedule
        digests[str(stage)] = digest
    info = {
        "run_dir": source,
        "gate": values["gate"],
        "variant": values.get("gate_variant"),
        "seed": values.get("seed"),
        "schedule_sha256": digests,
    }
    return schedules, info


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
        self.base_structure = base.structure
        self.streams = streams
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
        self.base_seq_len = int(base.model_config["seq_len"])
        self.num_updates = num_stage_updates(
            config.stage_tokens, config.batch_size, self.base_seq_len
        )
        self.replay: dict[int, list[dict[str, Any]]] = {}
        if config.gate in ONLINE_GATES and config.gate_variant == "replay":
            self.replay, self.sources[config.gate_source[0]] = load_replay_schedules(
                config.gate_source[0],
                config,
                modules=self.modules,
                num_updates=self.num_updates,
                params_sha256=base.source["params_sha256"],
            )
        self.structure: GraphStructure | None = None
        self.energy_fn: EnergyFn | None = None
        self.energy_field: str | None = None
        if self.uses_map:
            self.structure, self.energy_fn, self.energy_field = signal_energy_fn(
                config, base.model_config, learner=self.learner_map
            )
            for stage in range(2, len(config.domains) + 1):
                map_batches(
                    streams[config.domains[stage - 1]],
                    self.structure,
                    self.special_ids,
                    num_windows=config.gate_map_windows,
                    mask_prob=config.mask_prob,
                    seed=config.seed + stage,
                )

    @property
    def learner_map(self) -> bool:
        """True when the energies are a PC learner's own errors."""
        return self.config.gate_map_source == "learner" and self.config.mode == "pc"

    @property
    def uses_map(self) -> bool:
        """True when the gate computes the onset map and energies."""
        if self.config.gate in ONLINE_GATES:
            return self.config.gate_variant != "replay"
        return self.config.gate in MAP_GATES

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
        if self.uses_map:
            info["map"] = {
                "energy": "energy_sum_ce of bp_signal_energies (spike-only map)",
                "stream": "train stream of the stage's domain, unshuffled",
                "windows": self.config.gate_map_windows,
                "batch_size": GATE_MAP_BATCH,
                "windows_masked": "one at a time (batch size 1)",
                "mask_seed": "seed + stage",
            }
            if self.config.gate_map_source != "bp":
                info["map"]["source"] = self.config.gate_map_source
            if self.learner_map and self.structure is not None:
                inference = self.structure.config["inference"]
                info["map"]["energy"] = (
                    f"energy_sum_ce of learner_signal_energies ({self.energy_field} "
                    "of the run's own learner errors)"
                )
                info["map"]["inference"] = {
                    "class": type(inference).__name__,
                    **dict(inference.config),
                }
        if self.config.gate in ONLINE_GATES:
            info["online"] = {
                "variant": self.config.gate_variant,
                "interval": self.config.gate_interval,
                "ema": self.config.gate_ema,
                "updates_per_stage": self.num_updates,
                "probe": {
                    "windows": GATE_MAP_BATCH,
                    "offset": "gate_map_windows + refresh * windows, modulo the "
                    "stream's window count",
                    "mask_seed": "SeedSequence([seed, stage, refresh]) >> 1",
                },
                "score_floor": SCORE_FLOOR,
                "multipliers": "traced optimizer state, set at each refresh",
            }
            if self.config.gate_variant == "shuffle":
                info["online"]["shuffle_rng"] = (
                    "default_rng([gate_seed, seed, stage, refresh]); onset "
                    "default_rng([gate_seed, seed, stage])"
                )
            if self.config.gate_variant == "replay":
                info["sources"] = {k: dict(v) for k, v in self.sources.items()}
            if self.config.gate == "online_influence":
                info["online"]["influence"] = {
                    "alpha": self.config.gate_influence_alpha,
                    "rule": "clip(g * ((1 - alpha) + alpha * h), 0, 1)",
                    "relative_floor": INFLUENCE_REL_FLOOR,
                    "probe_windows": INFLUENCE_PROBE_WINDOWS,
                    "probe": "evenly spaced windows of the "
                    + (
                        "previous domain's"
                        if self.config.gate_variant == "past"
                        else "stage's"
                    )
                    + " train stream, masked once per stage",
                    "mask_seed": f"SeedSequence([seed, stage, {INFLUENCE_PROBE_TAG}]) "
                    ">> 1",
                }
        return info

    def influence_probe(self, stage: int) -> tuple[InfluenceProbe, dict[str, Any]]:
        """The stage's probe and its record: domain, windows, mask seed and SHA-256."""
        config = self.config
        index = stage - (2 if config.gate_variant == "past" else 1)
        if index < 0:
            raise ValueError(f"stage {stage} has no previous domain to probe")
        domain = config.domains[index]
        seed = probe_seed(config.seed, stage, INFLUENCE_PROBE_TAG)
        arrays, windows = spaced_probe(
            self.streams[domain],
            self.base_structure,
            self.special_ids,
            num_windows=INFLUENCE_PROBE_WINDOWS,
            mask_prob=config.mask_prob,
            seed=seed,
        )
        return InfluenceProbe(self.base_structure, arrays, self.blocks), {
            "domain": domain,
            "first_window": int(windows[0]),
            "last_window": int(windows[-1]),
            "windows": INFLUENCE_PROBE_WINDOWS,
            "mask_seed": seed,
            "masked_tokens": int(np.asarray(arrays["loss_mask"]).sum()),
            "sha256": arrays_sha256(dict(arrays)),
        }

    def _onset_map(
        self,
        record: dict[str, Any],
        stage: int,
        domain: str,
        params: GraphParams,
        stream: np.ndarray,
    ) -> dict[str, float]:
        """Compute the stage's onset map and store it in ``record``."""
        if self.structure is None or self.energy_fn is None:
            raise ValueError(f"gate {self.config.gate!r} has no onset map")
        config = self.config
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
        return values

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
        if config.gate_map_source != "bp":
            record["map_source"] = config.gate_map_source
        multipliers = {m: 1.0 for m in self.modules}
        if stage >= 2:
            if config.gate == "freeze":
                multipliers.update({m: 0.0 for m in self.frozen})
            elif config.gate == "uniform":
                multipliers.update({m: config.gate_scale for m in self.blocks})
            elif config.gate == "uniform_all":
                multipliers.update({m: config.gate_scale for m in self.modules})
            elif config.gate == "oracle":
                raw = oracle_scores(self.rollbacks, stage, self.blocks)
                scores = {m: max(v, SCORE_FLOOR) for m, v in raw.items()}
                record["scores"] = raw
                multipliers.update(map_multipliers(scores, config.gate_beta))
            elif config.gate in ONLINE_GATES:
                if config.gate_variant == "replay":
                    multipliers.update(self.replay[stage][0]["multipliers"])
                else:
                    values = self._onset_map(record, stage, domain, params, stream)
                    block, record["permutation"] = online_multipliers(
                        values,
                        values,
                        gate=config.gate,
                        variant=config.gate_variant,
                        beta=config.gate_beta,
                        rng=shuffle_rng(config.gate_seed, config.seed, stage, -1),
                    )
                    multipliers.update(block)
            else:
                values = self._onset_map(record, stage, domain, params, stream)
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
        if config.gate in ONLINE_GATES:
            record.update(
                variant=config.gate_variant,
                interval=config.gate_interval,
                ema=config.gate_ema,
                schedule=None,
            )
        record["multipliers"] = multipliers
        record["seconds"] = time.time() - started
        return record

    def online(self, record: Mapping[str, Any], stream: np.ndarray) -> "OnlineGate":
        """The ``train_stage`` gate callback of a gated stage's ``record``."""
        if self.config.gate not in ONLINE_GATES or not record.get("gated"):
            raise ValueError("only gated stages of online gates refresh")
        return OnlineGate(self, record, stream)


class OnlineGate:
    """One stage's online gate: refreshes the optimizer's block multipliers.

    Called by ``train_stage`` with the update count, the params and the
    optimizer state before the first update and after every update; it
    returns the state, with new multipliers after a refresh. ``schedule``
    holds one record per period with the multipliers read back from the
    optimizer state, so they are the ones applied; ``seconds`` is the time
    spent refreshing.
    """

    def __init__(
        self, plan: GatePlan, record: Mapping[str, Any], stream: np.ndarray
    ) -> None:
        """``record`` is the stage's ``GatePlan.record`` and ``stream`` its domain's."""
        self.plan = plan
        self.config = plan.config
        self.stage = int(record["stage"])
        self.stream = stream
        self.periods = gate_periods(plan.num_updates, self.config.gate_interval)
        self.replay = plan.replay.get(self.stage)
        self.onset = dict(record["map"]) if record.get("map") is not None else None
        self.smoothed = dict(self.onset) if self.onset is not None else None
        self.permutation = record.get("permutation")
        self.num_windows = stream_windows(stream, plan.base_seq_len)
        self.schedule: list[dict[str, Any]] = []
        self.seconds = 0.0
        self.influence: InfluenceProbe | None = None
        self.influence_record: dict[str, Any] | None = None
        if self.config.gate == "online_influence" and self.replay is None:
            self.influence, self.influence_record = plan.influence_probe(self.stage)

    def __call__(self, update: int, params: GraphParams, opt_state: Any) -> Any:
        """Record the onset at update 0; refresh after each ``gate_interval`` updates."""
        config = self.config
        interval = config.gate_interval
        if update == 0:
            onset_audit = None
            if self.influence is not None:
                onset_audit = {
                    "influence_probe": self.influence_record,
                    "influence": None,
                    "influence_h": {m: 1.0 for m in self.plan.blocks},
                    "influence_floor": None,
                    "influence_probe_loss": None,
                    "influence_split_half_spearman": None,
                    "influence_floored_fraction": None,
                    "influence_positive_fraction": None,
                    "influence_per_multiplier": None,
                    "surprise_multipliers": None,
                }
            self._append(-1, None, None, None, opt_state, onset_audit)
            return opt_state
        if update % interval or update >= self.plan.num_updates:
            return opt_state
        started = time.time()
        refresh = update // interval - 1
        raw = None
        offset = seed = None
        permutation = None
        audit: dict[str, Any] | None = None
        if self.replay is not None:
            multipliers = self.replay[refresh + 1]["multipliers"]
        else:
            if self.smoothed is None or self.onset is None:
                raise ValueError(f"stage {self.stage} has no onset map")
            if self.plan.structure is None or self.plan.energy_fn is None:
                raise ValueError(f"gate {config.gate!r} has no energy function")
            offset = probe_offset(config.gate_map_windows, refresh, self.num_windows)
            seed = probe_seed(config.seed, self.stage, refresh)
            raw = onset_map(
                params,
                self.plan.structure,
                self.stream,
                self.plan.special_ids,
                modules=self.plan.blocks,
                num_windows=GATE_MAP_BATCH,
                mask_prob=config.mask_prob,
                seed=seed,
                energy_fn=self.plan.energy_fn,
                offset=offset,
            )
            self.smoothed = ema_update(self.smoothed, raw, config.gate_ema)
            block, permutation = online_multipliers(
                self.smoothed,
                self.onset,
                gate=config.gate,
                variant=config.gate_variant,
                beta=config.gate_beta,
                rng=shuffle_rng(config.gate_seed, config.seed, self.stage, refresh),
            )
            if self.influence is not None:
                audit = self.influence.measure(params)
                held = self.schedule[-1]["multipliers"]
                audit["influence_per_multiplier"] = (
                    None
                    if audit["influence"] is None
                    else {m: v / held[m] for m, v in audit["influence"].items()}
                )
                audit["surprise_multipliers"] = dict(block)
                block = influence_multipliers(
                    block, audit["influence_h"], config.gate_influence_alpha
                )
            multipliers = {**{m: 1.0 for m in self.plan.modules}, **block}
        self.permutation = permutation
        opt_state = set_module_multipliers(opt_state, multipliers)
        self._append(refresh, raw, offset, seed, opt_state, audit)
        self.seconds += time.time() - started
        return opt_state

    def _append(
        self,
        refresh: int,
        raw: Mapping[str, float] | None,
        offset: int | None,
        seed: int | None,
        opt_state: Any,
        audit: Mapping[str, Any] | None = None,
    ) -> None:
        held = module_multipliers(opt_state)
        _, first, last = self.periods[refresh + 1]
        source = (
            {"map_source": self.config.gate_map_source}
            if self.config.gate_map_source != "bp"
            else {}
        )
        self.schedule.append(
            {
                **source,
                "stage": self.stage,
                "refresh": refresh,
                "first_update": first,
                "last_update": last,
                "energies": None if raw is None else dict(raw),
                "smoothed": None if self.smoothed is None else dict(self.smoothed),
                "multipliers": {m: held[m] for m in self.plan.modules},
                "permutation": self.permutation,
                "probe_offset": offset,
                "probe_seed": seed,
                **(audit or {}),
            }
        )
