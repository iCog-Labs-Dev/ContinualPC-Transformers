"""Ablation and perturbation influence of each module at each stage onset.

No training. At each onset ``S_s`` of a finished sequential run (the snapshot
before stage ``s + 1``), for every parameterised module:

- **Ablation** removes the module's contribution. Attention and ``mlp2``:
  the output added to the residual is multiplied by zero through the node's
  dropout mask, so the node returns its residual path after LayerNorm.
  ``mlp1``: its output is set to zero, so ``mlp2`` returns its bias. Position
  embeddings: zeroed. Token embeddings: every row replaced by the mean row.
  Head modules (``module_groups``' ``head``) have no ablation.
- **Perturbation** adds Gaussian noise of ``scale ×`` RMS to the module's node
  output, the RMS taken per batch over the output (the residual path included
  for attention and ``mlp2``). ``mlm_head``'s output is its logits at the masked
  tokens, since the full logits are never formed. ``NOISE_DRAWS`` draws, keys
  from ``SeedSequence([seed, s, module, scale, draw, domain])``.
- **Score**: Δ masked CE (intervened − baseline; positive = the module helps) on
  fixed probes of ``PROBE_WINDOWS`` windows per domain, evenly spaced over each
  train stream (``gates.spaced_probe``, so disjoint from the validation windows
  of the reported metrics), masked once per seed and domain. ``current``
  is the next domain, ``old`` the mean over the previous domains; at ``S_0`` the
  ``old`` key is absent.

Spearman's ρ over the block modules is reported per onset against the gate's
surprise map at that onset (``gates.onset_map`` on the next domain; the learner's
map for PC runs, BP's for backprop) and, from ``S_1`` on, against the recovery of
the transition ``S_{s+1} <- S_s`` in ``rollback.json`` (``gates.oracle_scores``).
``S_0`` has no such transition. Descriptive only; there is no pass rule.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from fabricpc.core.types import GraphParams, GraphStructure, NodeParams
from fabricpc.experiments.domain_forgetting.artifacts import atomic_write_json
from fabricpc.experiments.domain_forgetting.correlations import spearman
from fabricpc.experiments.domain_forgetting.data import load_domain_cache
from fabricpc.experiments.domain_forgetting.frontier import mean_sd
from fabricpc.experiments.domain_forgetting.gates import (
    EnergyFn,
    block_modules,
    load_rollbacks,
    onset_map,
    oracle_scores,
    probe_seed,
    signal_energy_fn,
    spaced_probe,
)
from fabricpc.experiments.domain_forgetting.localization import (
    RunContext,
    check_provenance,
    load_run,
    module_groups,
    module_names,
    run_provenance,
    stage_name,
)
from fabricpc.experiments.domain_forgetting.runner import (
    RunConfig,
    build_inference,
    build_structure,
)
from fabricpc.experiments.domain_forgetting.validation import (
    MultiDomainEvaluator,
    head_input,
)
from fabricpc.training.train_backprop import compute_forward_pass

PROBE_WINDOWS = 256
PROBE_BATCH = 64
MAP_WINDOWS = 256
NOISE_SCALES = (0.01, 0.1)
NOISE_DRAWS = 8
PROBE_TAG = 0x6F6E73
RESULT_FILE = "influence.json"
SCORES = ("current", "old")
CeStep = Callable[..., tuple[jax.Array, jax.Array]]


def _masked_ce_sums(
    logits: jax.Array, labels: jax.Array, valid: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Summed masked CE and token count, the per-token loss of the evaluator."""
    probs = jnp.exp(jax.nn.log_softmax(logits, axis=-1))
    target = jnp.take_along_axis(probs, labels[:, None], axis=-1)[:, 0]
    weight = valid.astype(probs.dtype)
    return jnp.sum(-jnp.log(target + 1e-10) * weight), jnp.sum(weight)


def _logits(
    structure: GraphStructure,
    params: GraphParams,
    batch: Mapping[str, jax.Array],
    index: jax.Array,
    clamps: Mapping[str, jax.Array] | None = None,
) -> jax.Array:
    """The head's logits at the positions ``index`` after a forward pass."""
    output = structure.task_map["y"]
    state = compute_forward_pass(
        params,
        structure,
        dict(batch),
        jax.random.PRNGKey(0),
        training=False,
        extra_clamps=dict(clamps) if clamps else None,
    )
    hidden = head_input(structure, state, output)
    gathered = hidden.reshape(-1, hidden.shape[-1])[index]
    head = params.nodes[output]
    return gathered @ head.weights["W_out"] + head.biases["b_out"]


def ablation_clamps(
    structure: GraphStructure, module: str, batch_size: int
) -> dict[str, jax.Array]:
    """The zero clamp that ablates a block module."""
    mask = f"dropout_mask_{module}"
    if mask in structure.nodes:
        shape = structure.nodes[mask].node_info.shape
        return {mask: jnp.zeros((batch_size, *shape))}
    if module.endswith("_mlp1"):
        shape = structure.nodes[module].node_info.shape
        return {module: jnp.zeros((batch_size, *shape))}
    raise ValueError(
        f"{module}: no dropout mask node to ablate through; the graph needs "
        "dropout_rate > 0"
    )


def ablated_params(params: GraphParams, module: str) -> GraphParams:
    """``params`` with an embedding module's contribution removed."""
    node = params.nodes[module]
    if module == "token_embeddings":
        table = node.weights["embeddings"]
        replaced = jnp.broadcast_to(jnp.mean(table, axis=0), table.shape)
        weights = {**node.weights, "embeddings": replaced}
    elif module == "position_embeddings":
        table = node.weights["position_embeddings"]
        weights = {**node.weights, "position_embeddings": jnp.zeros_like(table)}
    else:
        raise ValueError(f"{module} is not an embedding module")
    new = NodeParams(weights=weights, biases=node.biases)
    return GraphParams(nodes={**params.nodes, module: new})


class InterventionSteps:
    """Jitted masked-CE steps of one graph, compiled once per module and kind.

    Every step takes ``(params, batch, index, valid)``, the ``perturb`` steps
    also ``(key, scale)``, and returns the summed CE and masked-token count.
    """

    def __init__(self, structure: GraphStructure) -> None:
        self.structure = structure
        self.output = structure.task_map["y"]
        self._steps: dict[tuple[str, str | None], CeStep] = {}

    def _cached(self, kind: str, module: str | None, build: Callable[[], CeStep]):
        key = (kind, module)
        if key not in self._steps:
            self._steps[key] = jax.jit(build())
        return self._steps[key]

    def clean(self) -> CeStep:
        """No intervention beyond the params it is given."""

        def build() -> CeStep:
            def step(params, batch, index, valid):
                logits = _logits(self.structure, params, batch, index)
                return _masked_ce_sums(logits, batch["y"].reshape(-1)[index], valid)

            return step

        return self._cached("clean", None, build)

    def ablate(self, module: str) -> CeStep:
        """A block module's contribution removed (``ablation_clamps``)."""

        def build() -> CeStep:
            def step(params, batch, index, valid):
                clamps = ablation_clamps(self.structure, module, batch["x"].shape[0])
                logits = _logits(self.structure, params, batch, index, clamps)
                return _masked_ce_sums(logits, batch["y"].reshape(-1)[index], valid)

            return step

        return self._cached("ablate", module, build)

    def perturb(self, module: str) -> CeStep:
        """Noise of ``scale ×`` RMS on the module's output."""

        def build() -> CeStep:
            if module == self.output:

                def logit_step(params, batch, index, valid, key, scale):
                    logits = _logits(self.structure, params, batch, index)
                    weight = valid.astype(logits.dtype)[:, None]
                    count = jnp.maximum(jnp.sum(weight) * logits.shape[-1], 1.0)
                    rms = jnp.sqrt(jnp.sum(jnp.square(logits) * weight) / count)
                    noise = jax.random.normal(key, logits.shape, logits.dtype)
                    return _masked_ce_sums(
                        logits + scale * rms * noise,
                        batch["y"].reshape(-1)[index],
                        valid,
                    )

                return logit_step

            def step(params, batch, index, valid, key, scale):
                state = compute_forward_pass(
                    params,
                    self.structure,
                    dict(batch),
                    jax.random.PRNGKey(0),
                    training=False,
                )
                value = state.nodes[module].z_latent
                rms = jnp.sqrt(jnp.mean(jnp.square(value)))
                noise = jax.random.normal(key, value.shape, value.dtype)
                clamps = {module: value + scale * rms * noise}
                logits = _logits(self.structure, params, batch, index, clamps)
                return _masked_ce_sums(logits, batch["y"].reshape(-1)[index], valid)

            return step

        return self._cached("perturb", module, build)


def probe_ce(
    evaluator: MultiDomainEvaluator,
    step: CeStep,
    params: GraphParams,
    key: jax.Array | None = None,
    scale: float | None = None,
) -> float:
    """Mean masked CE of one step over a probe's batches (one domain per evaluator).

    A batch's noise key is ``fold_in(key, batch index)``.
    """
    total, count = 0.0, 0.0
    size = evaluator.batch_size
    for position, start in enumerate(range(0, len(evaluator.arrays["labels"]), size)):
        arrays = evaluator.arrays
        batch = {
            "x": jnp.asarray(arrays["masked_input_ids"][start : start + size]),
            "y": jnp.asarray(arrays["labels"][start : start + size]),
            "loss_mask": jnp.asarray(arrays["loss_mask"][start : start + size]),
            "attention_mask": jnp.asarray(
                arrays["attention_mask"][start : start + size]
            ),
        }
        index, valid = evaluator.positions[position]
        args: tuple[Any, ...] = (params, batch, jnp.asarray(index), jnp.asarray(valid))
        if key is not None:
            args += (jax.random.fold_in(key, position), jnp.asarray(scale))
        ce_sum, tokens = step(*args)
        total += float(ce_sum)
        count += float(tokens)
    if count == 0:
        raise ValueError("probe has no masked tokens")
    return total / count


def noise_key(seed: int, onset: int, module: int, scale: int, draw: int, domain: int):
    """The fixed key of one perturbation draw on one domain's probe."""
    state = np.random.SeedSequence([seed, onset, module, scale, draw, domain])
    return jax.random.PRNGKey(int(state.generate_state(1)[0] >> 1))


def _split_scores(
    deltas: Mapping[str, float], current: str, old: Sequence[str]
) -> dict[str, Any]:
    """``current``, the mean ``old`` when there are old domains, and ``per_domain``."""
    scores: dict[str, Any] = {"current": deltas[current], "per_domain": dict(deltas)}
    if old:
        scores["old"] = float(np.mean([deltas[name] for name in old]))
    return scores


def rank_correlations(
    modules: Sequence[str],
    records: Mapping[str, Mapping[str, Any]],
    targets: Mapping[str, Mapping[str, float] | None],
) -> dict[str, dict[str, float | None]]:
    """Spearman's ρ of every score over ``modules`` against every available target.

    Keys ``ablation_current``, ``perturbation_0.1_old`` etc. A score absent at
    this onset is left out; ρ undefined on constant input is ``None``.
    """
    series: dict[str, list[float]] = {}
    for name in SCORES:
        if name in records[modules[0]]["ablation"]:
            series[f"ablation_{name}"] = [records[m]["ablation"][name] for m in modules]
        for scale in NOISE_SCALES:
            entry = records[modules[0]]["perturbation"][f"{scale:g}"]
            if name in entry:
                series[f"perturbation_{scale:g}_{name}"] = [
                    records[m]["perturbation"][f"{scale:g}"][name] for m in modules
                ]
    out: dict[str, dict[str, float | None]] = {}
    for target, values in targets.items():
        if values is None:
            continue
        reference = [values[m] for m in modules]
        out[target] = {}
        for name, scores in series.items():
            rho = spearman(scores, reference)
            out[target][name] = None if math.isnan(rho) else rho
    return out


class DiagnosticCache:
    """What several runs share: compiled steps, energy functions and train streams."""

    def __init__(self) -> None:
        self.steps: dict[str, InterventionSteps] = {}
        self.energies: dict[str, tuple[GraphStructure, EnergyFn]] = {}
        self.caches: dict[tuple[str, str, int], Any] = {}

    def interventions(self, context: RunContext) -> InterventionSteps:
        """Steps on the run's graph in backprop mode, which computes the same forward."""
        config = context.config
        if config.pc_mupc != "off":
            raise ValueError(f"{context.run_dir}: influence needs pc_mupc 'off'")
        if config.dropout_rate <= 0.0:
            raise ValueError(
                f"{context.run_dir}: ablation acts through dropout mask nodes; "
                "the run must have dropout_rate > 0"
            )
        key = json.dumps([context.model_config, config.dropout_rate], sort_keys=True)
        if key not in self.steps:
            run = replace(config, mode="backprop")
            inference = build_inference(run, context.model_config)[0]
            self.steps[key] = InterventionSteps(
                build_structure(context.model_config, run, inference)
            )
        return self.steps[key]

    def energy(self, context: RunContext) -> tuple[GraphStructure, EnergyFn]:
        """The surprise map's graph and energies: the learner's for PC runs."""
        settings = {
            k: v
            for k, v in asdict(context.config).items()
            if k not in ("seed", "domains", "output_dir", "lr")
        }
        key = json.dumps([context.model_config, settings], sort_keys=True, default=str)
        if key not in self.energies:
            structure, energy_fn, _ = signal_energy_fn(
                context.config,
                context.model_config,
                learner=context.config.mode == "pc",
            )
            self.energies[key] = (structure, energy_fn)
        return self.energies[key]

    def domain(self, config: RunConfig, name: str, tokenizer: str, seq_len: int) -> Any:
        """The verified domain cache of ``name``, cut to the run's stream length."""
        key = (str(config.data_dir), name, config.tokens_per_domain)
        if key not in self.caches:
            self.caches[key] = load_domain_cache(
                config.data_dir,
                name,
                tokenizer_json=tokenizer,
                seq_len=seq_len,
                tokens_per_domain=config.tokens_per_domain,
                max_valid_windows=config.max_valid_windows,
            )
        return self.caches[key]


def compute_influence(
    context: RunContext,
    *,
    cache: DiagnosticCache | None = None,
    probe_windows: int = PROBE_WINDOWS,
    probe_batch: int = PROBE_BATCH,
    map_windows: int = MAP_WINDOWS,
    verbose: bool = False,
) -> dict[str, Any]:
    """Score every module at every onset of one finished run; needs ``rollback.json``."""
    cache = cache or DiagnosticCache()
    config = context.config
    domains = list(context.domains)
    if len(domains) < 2:
        raise ValueError(f"{context.run_dir}: influence needs two or more domains")
    run_dir = str(context.run_dir)
    rollbacks, _ = load_rollbacks(
        [run_dir],
        num_domains=len(domains),
        params_sha256=context.run_config["source"]["params_sha256"],
    )
    steps = cache.interventions(context)
    modules = module_names(context.structure, context.snapshots[0])
    blocks = block_modules(modules)
    head = module_groups(modules).get("head", [])
    if context.tokenizer_json is None:
        raise ValueError(f"{run_dir}: the run context holds no tokenizer")
    tokenizer = context.tokenizer_json
    seq_len = int(context.model_config["seq_len"])
    domain_caches = {
        name: cache.domain(config, name, tokenizer, seq_len) for name in domains
    }
    special_ids = dict(domain_caches[domains[0]].manifest["special_token_ids"])
    evaluators = {}
    for position, name in enumerate(sorted(domains)):
        arrays, _ = spaced_probe(
            domain_caches[name].train,
            steps.structure,
            special_ids,
            num_windows=probe_windows,
            mask_prob=config.mask_prob,
            seed=probe_seed(config.seed, position, PROBE_TAG),
        )
        evaluators[name] = MultiDomainEvaluator(
            steps.structure, {name: arrays}, probe_batch
        )
    map_structure, energy_fn = cache.energy(context)
    domain_index = {name: i for i, name in enumerate(sorted(domains))}
    onsets = []
    for onset in range(len(domains)):
        params = context.params(onset)
        current = domains[onset]
        old = domains[:onset]
        probes = [current, *old]
        baseline = {
            name: probe_ce(evaluators[name], steps.clean(), params) for name in probes
        }
        records: dict[str, dict[str, Any]] = {}
        for position, module in enumerate(modules):
            ablation: dict[str, Any] | None = None
            if module not in head:
                if module in blocks:
                    deltas = {
                        name: probe_ce(evaluators[name], steps.ablate(module), params)
                        - baseline[name]
                        for name in probes
                    }
                else:
                    removed = ablated_params(params, module)
                    deltas = {
                        name: probe_ce(evaluators[name], steps.clean(), removed)
                        - baseline[name]
                        for name in probes
                    }
                ablation = _split_scores(deltas, current, old)
            perturbation = {}
            for scale_index, scale in enumerate(NOISE_SCALES):
                draws: dict[str, list[float]] = {name: [] for name in probes}
                for draw in range(NOISE_DRAWS):
                    for name in probes:
                        key = noise_key(
                            config.seed,
                            onset,
                            position,
                            scale_index,
                            draw,
                            domain_index[name],
                        )
                        draws[name].append(
                            probe_ce(
                                evaluators[name],
                                steps.perturb(module),
                                params,
                                key,
                                scale,
                            )
                            - baseline[name]
                        )
                entry = _split_scores(
                    {name: float(np.mean(v)) for name, v in draws.items()},
                    current,
                    old,
                )
                entry["draw_sd"] = {
                    name: mean_sd(values)["sd"] for name, values in draws.items()
                }
                perturbation[f"{scale:g}"] = entry
            records[module] = {"ablation": ablation, "perturbation": perturbation}
        surprise = onset_map(
            params,
            map_structure,
            domain_caches[current].train,
            special_ids,
            modules=blocks,
            num_windows=map_windows,
            mask_prob=config.mask_prob,
            seed=config.seed + onset + 1,
            energy_fn=energy_fn,
        )
        recovery = oracle_scores(rollbacks, onset + 1, blocks) if onset >= 1 else None
        onsets.append(
            {
                "onset": stage_name(onset),
                "stage": onset,
                "current_domain": current,
                "old_domains": old,
                "baseline_ce": baseline,
                "modules": records,
                "surprise_map": surprise,
                "recovery": recovery,
                "spearman": rank_correlations(
                    blocks,
                    records,
                    {"surprise_map": surprise, "recovery": recovery},
                ),
            }
        )
        if verbose:
            print(f"{run_dir} {stage_name(onset)} done", flush=True)
    return {
        "run_dir": run_dir,
        "mode": config.mode,
        "seed": config.seed,
        "domains": domains,
        "modules": modules,
        "block_modules": blocks,
        "head_modules": head,
        "map_source": "learner" if config.mode == "pc" else "bp",
        "provenance": run_provenance(context),
        "settings": {
            "probe_windows": probe_windows,
            "probe": "evenly spaced windows of each domain's train stream",
            "map_windows": map_windows,
            "noise_scales": list(NOISE_SCALES),
            "noise_draws": NOISE_DRAWS,
        },
        "onsets": onsets,
    }


def compute_influence_runs(
    run_dirs: Iterable[str | Path],
    *,
    checkpoint: str | None = None,
    data_dir: str | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Write ``influence.json`` per run and summarise ρ over seeds.

    A run whose file holds this run's provenance is reused. A missing snapshot
    raises ``FileNotFoundError`` before any run is scored.
    """
    paths = [Path(p) for p in run_dirs]
    missing = []
    for path in paths:
        manifest = json.loads((path / "checkpoint_manifest.json").read_text())
        missing += [
            str(path / entry["file"])
            for entry in manifest["stages"]
            if not (path / entry["file"]).is_file()
        ]
    if missing:
        raise FileNotFoundError(f"missing stage snapshots: {missing}")
    cache = DiagnosticCache()
    runs = []
    for path in paths:
        existing = path / RESULT_FILE
        if existing.is_file():
            payload = json.loads(existing.read_text())
            check_provenance(path, payload, RESULT_FILE)
            runs.append(payload)
            continue
        context = load_run(path, checkpoint=checkpoint, data_dir=data_dir)
        payload = compute_influence(context, cache=cache, verbose=verbose)
        atomic_write_json(existing, payload)
        runs.append(payload)
    groups: dict[tuple[str, str, str, str, str], list[float | None]] = {}
    for payload in runs:
        order = ">".join(payload["domains"])
        for entry in payload["onsets"]:
            for target, values in entry["spearman"].items():
                for score, rho in values.items():
                    key = (payload["mode"], order, entry["onset"], target, score)
                    groups.setdefault(key, []).append(rho)
    summary = [
        {
            "mode": mode,
            "order": order,
            "onset": onset,
            "target": target,
            "score": score,
            "per_seed": values,
            **mean_sd(values),
        }
        for (mode, order, onset, target, score), values in sorted(groups.items())
    ]
    return {"runs": [str(p) for p in paths], "spearman": summary}


def format_influence_table(result: Mapping[str, Any]) -> str:
    """Return a table of mean ± SD ρ per learner, order, onset, target and score."""
    lines = [f"{'mode / order / onset':<36}{'target':<14}{'score':<24}{'rho':>16}  n"]
    for row in result["spearman"]:
        name = f"{row['mode']} {row['order']} {row['onset']}"
        if row["mean"] is None:
            shown = "-"
        elif row["sd"] is None:
            shown = f"{row['mean']:+.3f}"
        else:
            shown = f"{row['mean']:+.3f}±{row['sd']:.3f}"
        lines.append(
            f"{name:<36}{row['target']:<14}{row['score']:<24}{shown:>16}  {row['n']}"
        )
    return "\n".join(lines)
