"""Frozen-weight probe: is there a reproducible PC equilibrium distinct from BP?

Each sample of a batch is solved on its own in float64, from the feedforward
state, by two solvers. Per sample the record holds the energy drop, ‖ε*‖ against
the bound √(2E_s(0)), weight-gradient agreement between the solvers, the spectrum
at ε = 0 and at each solution, an escape test, solver work and a float32 rounding
test. Weight gradients are summed over samples and compared with batch backprop.

Finished samples and cases are flushed as they complete, so a killed run keeps
them. An existing output directory is never overwritten.

"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from diagnostics.pc_equilibrium_solvers import (
    ACCEPTED,
    CENSORED,
    CONVERGED,
    CURVATURE_UNRESOLVED,
    EVIDENCE_LANCZOS,
    SATISFIED_STATUSES,
    UNRESOLVED_STATUSES,
    ZERO_SIGNAL,
    AcceptanceThresholds,
    Budget,
    EnergyFunctions,
    EscapeResult,
    LbfgsConfig,
    SolveResult,
    SpectrumResult,
    TimeAllocator,
    TrustRegionConfig,
    assess_reference,
    batch_status,
    compare_module_gradients,
    escape_test,
    lanczos_memory_bytes,
    lbfgs,
    lbfgs_memory_bytes,
    make_energy_functions,
    module_gradients,
    relative_gap,
    tree_norm,
    tree_spectrum,
    trust_region_newton_cg,
)
from fabricpc.core.inference_epc import EPCInference
from fabricpc.core.learning import compute_local_weight_gradients
from fabricpc.core.types import GraphParams, GraphState, GraphStructure
from fabricpc.experiments.domain_forgetting.data import load_domain_cache
from fabricpc.experiments.domain_forgetting.probe import _stage_params
from fabricpc.experiments.domain_forgetting.runner import (
    HEAD_ENERGY,
    RunConfig,
    fixed_mlm_batches,
    load_base_model,
    make_gradient_fn,
)
from fabricpc.experiments.forgetting_benchmark.sequential_trainer import params_hash
from fabricpc.graph_initialization import initialize_params
from fabricpc.nodes.encoder_transformer import create_encoder_mlm_model
from fabricpc.training.train_mlm import _prepare_pcn_mlm_batch

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    "diagnostics/pc_equilibrium_probe.py",
    "diagnostics/pc_equilibrium_solvers.py",
    "fabricpc/core/inference_epc.py",
    "fabricpc/core/learning.py",
    "fabricpc/nodes/encoder_transformer.py",
    "fabricpc/experiments/domain_forgetting/runner.py",
    "fabricpc/training/train_mlm.py",
)
DEPENDENCIES = ("jax", "jaxlib", "numpy", "tokenizers")
SOLVERS = ("trust_region", "lbfgs")
ROUNDING_VARIANTS = ("rounded_latents", "rounded_errors")
TINY_MASK_TOKEN = 3
UNITS_PER_SAMPLE = 6


def json_ready(value: Any, path: str = "") -> tuple[Any, list[str]]:
    """Return ``value`` as strict JSON data and the paths of nulled non-finites."""
    if isinstance(value, Mapping):
        clean, nulled = {}, []
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            clean[str(key)], sub = json_ready(item, child)
            nulled.extend(sub)
        return clean, nulled
    if isinstance(value, (list, tuple)):
        clean_list, nulled = [], []
        for index, item in enumerate(value):
            cleaned, sub = json_ready(item, f"{path}[{index}]")
            clean_list.append(cleaned)
            nulled.extend(sub)
        return clean_list, nulled
    if isinstance(value, (np.ndarray, jax.Array)):
        array = np.asarray(value)
        return json_ready(array.item() if array.ndim == 0 else array.tolist(), path)
    if isinstance(value, np.generic):
        return json_ready(value.item(), path)
    if isinstance(value, float) and not math.isfinite(value):
        return None, [path]
    return value, []


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write strict JSON atomically, listing nulled non-finites in ``null_fields``."""
    clean, nulled = json_ready(payload)
    if nulled:
        clean["null_fields"] = nulled
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(clean, indent=1, allow_nan=False) + "\n")
    os.replace(temporary, path)


def prepare_output_dir(path: Path) -> None:
    """Create a run directory, refusing any path that already exists."""
    if path.exists():
        raise FileExistsError(f"{path} exists; a run directory is never overwritten")
    (path / "records").mkdir(parents=True)


def parse_head_scales(text: str) -> list[str]:
    """Parse comma-separated positive numbers and the literals mean and midpoint.

    A repeated scale, including the same number written differently, is an error.
    """
    specs = [item.strip() for item in text.split(",") if item.strip()]
    if not specs:
        raise ValueError("no head scales given")
    seen = set()
    for spec in specs:
        key: Any = spec
        if spec not in ("mean", "midpoint"):
            number = float(spec)
            if not (math.isfinite(number) and number > 0.0):
                raise ValueError(f"head scale {spec!r} must be positive and finite")
            key = number
        if key in seen:
            raise ValueError(f"head scale {spec!r} is repeated")
        seen.add(key)
    return specs


def resolve_head_scale(spec: str, num_masked: float) -> float:
    """Return s: a number, ``mean`` = 1/N_masked, or ``midpoint`` = 1/√N_masked.

    N_masked is the batch's count. With no masked tokens the head energy is zero
    at every s, so s is 1.
    """
    if spec in ("mean", "midpoint") and num_masked <= 0.0:
        return 1.0
    if spec == "mean":
        return 1.0 / num_masked
    if spec == "midpoint":
        return 1.0 / math.sqrt(num_masked)
    return float(spec)


def case_seed(seed: int, batch_index: int) -> int:
    """The base random seed of one case."""
    return seed * 100003 + batch_index


def _to_double(tree: Any) -> Any:
    """Cast floating leaves to float64; requires x64 mode."""

    def cast(leaf: Any) -> Any:
        array = jnp.asarray(leaf)
        return (
            array.astype(jnp.float64)
            if jnp.issubdtype(array.dtype, jnp.floating)
            else leaf
        )

    return jax.tree_util.tree_map(cast, tree)


def measured_float_dtypes(tree: Any) -> list[str]:
    """The sorted set of floating dtypes among a pytree's leaves."""
    dtypes = {
        str(np.asarray(leaf).dtype)
        for leaf in jax.tree_util.tree_leaves(tree)
        if np.issubdtype(np.asarray(leaf).dtype, np.floating)
    }
    return sorted(dtypes)


def _assert_double(params: GraphParams) -> None:
    dtypes = measured_float_dtypes(params)
    if dtypes != ["float64"]:
        raise ValueError(f"reference parameters must be float64, found {dtypes}")


def _to_single(tree: Any) -> Any:
    """Round floating leaves to float32; call with x64 disabled."""

    def cast(leaf: Any) -> Any:
        if not isinstance(leaf, (np.ndarray, jax.Array, np.generic)):
            return leaf
        array = np.asarray(leaf)
        if np.issubdtype(array.dtype, np.floating):
            return jnp.asarray(array.astype(np.float32))
        if np.issubdtype(array.dtype, np.integer):
            return jnp.asarray(array.astype(np.int32))
        return jnp.asarray(array)

    return jax.tree_util.tree_map(cast, tree)


_SINGLE_FUNCTIONS: dict[int, tuple[Any, Any]] = {}


def _single_functions(structure: GraphStructure) -> tuple[Any, Any]:
    """Compiled local weight gradients and error-derivation for one graph."""
    key = id(structure)
    if key not in _SINGLE_FUNCTIONS:

        def gradients(params, state):
            return compute_local_weight_gradients(params, state, structure)

        def derive(params, state, clamps):
            return EPCInference.derive_states(params, state, clamps, structure)

        _SINGLE_FUNCTIONS[key] = (jax.jit(gradients), jax.jit(derive))
    return _SINGLE_FUNCTIONS[key]


def float32_gradients(
    structure: GraphStructure,
    params: GraphParams,
    derived: Any,
    clamps: Mapping[str, jax.Array],
    s: float,
) -> dict[str, tuple[dict[str, np.ndarray], str]]:
    """Round a float64 solution to float32 and recompute weight gradients there.

    ``rounded_latents`` rounds the latents, as a float32 state-coordinate solver
    would hold them. ``rounded_errors`` rounds ε and re-derives the latents in
    float32, as a float32 error-coordinate solver would. Both run with x64
    disabled, so every operation is single precision, and parameters are rounded
    too. This tests representability at the solution, not whether a float32
    solver could reach it.
    """
    grads_fn, derive_fn = _single_functions(structure)
    head = structure.task_map["y"]
    results = {}
    with jax.enable_x64(False):
        params_single = _to_single(params)
        state_single = _to_single(derived)
        clamps_single = _to_single(dict(clamps))
        states = {
            "rounded_latents": state_single,
            "rounded_errors": derive_fn(params_single, state_single, clamps_single),
        }
        for name, state in states.items():
            grads = grads_fn(params_single, state)
            dtype = str(jax.tree_util.tree_leaves(grads.nodes[head])[0].dtype)
            modules = module_gradients(grads, structure)
            results[name] = (
                {
                    module: s * values if module == head else values
                    for module, values in modules.items()
                },
                dtype,
            )
    return results


def float32_rounding_test(
    structure: GraphStructure,
    params: GraphParams,
    derived: Any,
    clamps: Mapping[str, jax.Array],
    s: float,
    reference: Mapping[str, np.ndarray],
    thresholds: AcceptanceThresholds,
) -> dict[str, Any]:
    """Compare ``float32_gradients`` with the float64 reference through the gate."""
    return {
        name: {
            "dtype": dtype,
            "comparison": compare_module_gradients(reference, modules, thresholds),
        }
        for name, (modules, dtype) in float32_gradients(
            structure, params, derived, clamps, s
        ).items()
    }


@dataclass
class Sample:
    """One sample of a batch, cut from the batch's prepared state."""

    index: int
    state: Any
    clamps: Mapping[str, jax.Array]
    zero_errors: dict[str, jax.Array]
    num_masked: float


@dataclass
class Case:
    """One fixed batch, prepared at the feedforward state, and its samples."""

    seed: int
    batch_index: int
    batch: Mapping[str, jax.Array]
    key: jax.Array
    state: Any
    clamps: Mapping[str, jax.Array]
    zero_errors: dict[str, jax.Array]
    num_masked: float
    samples: list[Sample]


@dataclass
class Fixture:
    """A graph at output scale 1, float64 parameters, and its prepared cases."""

    name: str
    structure: GraphStructure
    params: GraphParams
    cases: list[Case]
    source: dict[str, Any]
    backprop: Any


def _slice_tree(tree: Any, index: int, batch_size: int) -> Any:
    """Keep sample ``index`` in every leaf whose leading axis is the batch."""

    def cut(leaf: Any) -> Any:
        if hasattr(leaf, "ndim") and leaf.ndim > 0 and leaf.shape[0] == batch_size:
            return leaf[index : index + 1]
        return leaf

    return jax.tree_util.tree_map(cut, tree)


def _slice_samples(
    state: Any,
    clamps: Mapping[str, jax.Array],
    zero_errors: dict[str, jax.Array],
    loss_mask: jax.Array,
) -> list[Sample]:
    """Cut a prepared batch into batch-size-1 samples sharing its masks."""
    batch_size = int(loss_mask.shape[0])
    samples = []
    for index in range(batch_size):
        nodes = {
            name: _slice_tree(node, index, batch_size)
            for name, node in state.nodes.items()
        }
        samples.append(
            Sample(
                index=index,
                state=GraphState(nodes=nodes, batch_size=1),
                clamps=_slice_tree(dict(clamps), index, batch_size),
                zero_errors=_slice_tree(zero_errors, index, batch_size),
                num_masked=float(jnp.sum(loss_mask[index])),
            )
        )
    return samples


def _prepare_cases(
    structure: GraphStructure,
    params: GraphParams,
    batches: Sequence[Mapping[str, jax.Array]],
    keys: Sequence[jax.Array],
    seed: int,
    first_index: int = 0,
) -> list[Case]:
    """Build the feedforward state of each batch, with ε set exactly to zero."""
    cases = []
    for offset, (batch, key) in enumerate(zip(batches, keys, strict=True)):
        _, _, _, initial, clamps = _prepare_pcn_mlm_batch(params, batch, structure, key)
        state = EPCInference.begin_segment(params, initial, clamps, structure)
        _, errors = EPCInference.error_energy(params, state, clamps, structure)
        zero_errors = jax.tree_util.tree_map(jnp.zeros_like, errors)
        cases.append(
            Case(
                seed=seed,
                batch_index=first_index + offset,
                batch=batch,
                key=key,
                state=state,
                clamps=clamps,
                zero_errors=zero_errors,
                num_masked=float(jnp.sum(batch["loss_mask"])),
                samples=_slice_samples(state, clamps, zero_errors, batch["loss_mask"]),
            )
        )
    return cases


def _tiny_batch(seed: int, index: int, batch_size: int, seq_len: int, vocab: int):
    rng = np.random.RandomState(seed * 1000 + index)
    labels = rng.randint(4, vocab, size=(batch_size, seq_len))
    labels[:, 0] = 1
    loss_mask = (rng.uniform(size=labels.shape) < 0.3).astype(np.float32)
    loss_mask[:, 0] = 0.0
    loss_mask[:, 1] = 1.0
    inputs = np.where(loss_mask > 0, TINY_MASK_TOKEN, labels)
    return {
        "x": jnp.asarray(inputs, dtype=jnp.int32),
        "y": jnp.asarray(labels, dtype=jnp.int32),
        "attention_mask": jnp.ones((batch_size, seq_len), dtype=jnp.float32),
        "loss_mask": jnp.asarray(loss_mask, dtype=jnp.float32),
    }


def make_tiny_fixture(
    seed: int,
    batch_size: int,
    num_batches: int,
    seeds: Sequence[int] | None = None,
    depth: int = 2,
    embed_dim: int = 64,
    mlp_dim: int = 128,
    seq_len: int = 16,
    vocab: int = 50,
) -> Fixture:
    """A randomly initialised encoder on synthetic batches."""
    structure = create_encoder_mlm_model(
        depth=depth,
        embed_dim=embed_dim,
        num_heads=2,
        mlp_dim=mlp_dim,
        seq_len=seq_len,
        vocab_size=vocab,
        inference=EPCInference(eta_infer=1e-3, infer_steps=1),
        use_attention_mask=True,
        dropout_rate=0.0,
        architecture="bert",
        residual_identity=True,
    )
    params = initialize_params(structure, jax.random.PRNGKey(seed))
    source = {"params_sha256": params_hash(params), "kind": "random_init"}
    params = _to_double(params)
    _assert_double(params)
    cases = []
    for case_id in seeds if seeds is not None else [seed]:
        batches = [
            _tiny_batch(case_id, index, batch_size, seq_len, vocab)
            for index in range(num_batches)
        ]
        keys = list(jax.random.split(jax.random.PRNGKey(case_id), num_batches))
        cases.extend(_prepare_cases(structure, params, batches, keys, case_id))
    return Fixture(
        "tiny",
        structure,
        params,
        cases,
        source,
        make_gradient_fn("backprop", structure),
    )


def make_bert_mini_fixture(args: argparse.Namespace, seeds: Sequence[int]) -> Fixture:
    """A BERT checkpoint, optionally at a run's stage, on fixed domain batches."""
    config = RunConfig(
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        domains=(args.domain,),
        output_dir="",
        mode="pc",
        batch_size=args.batch_size,
        dropout_rate=0.0,
        mask_prob=args.mask_prob,
        seed=seeds[0],
        expected_shape=None,
        pc_inference="epc",
        pc_eta_infer=1e-4,
        pc_infer_steps=5,
        pc_forward_updates="none",
        pc_output_energy_scale=1.0,
    )
    base = load_base_model(config)
    params = base.params
    source = dict(base.source)
    if args.run_dir is not None:
        params, sha = _stage_params(Path(args.run_dir), args.stage, base.structure)
        source["params_sha256"] = sha
        source["run_dir"] = args.run_dir
        source["stage"] = args.stage
    params = _to_double(params)
    _assert_double(params)
    cache = load_domain_cache(
        config.data_dir,
        args.domain,
        tokenizer_json=base.tokenizer_json,
        seq_len=int(base.model_config["seq_len"]),
    )
    source["domain"] = args.domain
    source["train_array_sha256"] = cache.manifest["train"]["array_sha256"]
    source["head_energy"] = HEAD_ENERGY
    cases = []
    for seed in seeds:
        batches, keys = fixed_mlm_batches(
            cache.train,
            base.structure,
            base.special_ids,
            num_batches=args.num_batches,
            batch_size=args.batch_size,
            mask_prob=args.mask_prob,
            seed=seed,
        )
        cases.extend(_prepare_cases(base.structure, params, batches, keys, seed))
    return Fixture(
        "bert_mini",
        base.structure,
        params,
        cases,
        source,
        make_gradient_fn("backprop", base.structure),
    )


def _file_sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def build_manifest(
    arguments: Mapping[str, Any],
    fixture: Fixture,
    thresholds: AcceptanceThresholds,
    lanczos_iters: int,
) -> dict[str, Any]:
    """Everything needed to identify a run, written before its first case.

    ``case_settings`` starts empty and gains the settings each case actually used.
    """
    status = _git("status", "--porcelain", "--", "fabricpc", "diagnostics", "tests")
    diff = _git("diff", "HEAD", "--", "fabricpc", "diagnostics")
    error_dtypes = (
        measured_float_dtypes(fixture.cases[0].zero_errors) if fixture.cases else []
    )
    return {
        "arguments": dict(arguments),
        "git_revision": _git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "git_diff_sha256": (
            None if diff is None else hashlib.sha256(diff.encode()).hexdigest()
        ),
        "source_sha256": {
            name: _file_sha256(REPO_ROOT / name) for name in SOURCE_FILES
        },
        "dependencies": {
            **{name: _version(name) for name in DEPENDENCIES},
            "jax": jax.__version__,
            "python": platform.python_version(),
        },
        "hardware": {
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "machine": platform.machine(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
        },
        "dtypes": {
            "reference": ",".join(measured_float_dtypes(fixture.params)),
            "errors": ",".join(error_dtypes),
            "rounding_test": "float32",
        },
        "params_sha256": fixture.source.get("params_sha256"),
        "fixture": {"name": fixture.name, "source": fixture.source},
        "thresholds": thresholds.to_dict(),
        "solver_defaults": {
            "trust_region": asdict(TrustRegionConfig()),
            "lbfgs": asdict(LbfgsConfig()),
        },
        "lanczos_iters": lanczos_iters,
        "case_settings": {},
    }


def _random_like(seed: int, tree: Any) -> Any:
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    keys = jax.random.split(jax.random.PRNGKey(seed), len(leaves))
    return treedef.unflatten(
        [jax.random.normal(k, x.shape, x.dtype) for k, x in zip(keys, leaves)]
    )


def _spectrum_record(result: SpectrumResult, start_vector: str) -> dict[str, Any]:
    """The JSON scalars of a spectrum.

    The start vector differs between points, and with it which modes converge, so
    a gradient-start and a random-start spectrum are not comparable. The
    negative-mode weight is named after its start vector.
    """
    weights = result.ritz_weights[result.ritz_values < 0.0]
    weight = float(weights.sum()) if result.ritz_values.size else None
    return {
        "start_vector": start_vector,
        "lambda_max": result.lambda_max,
        "lambda_min": result.lambda_min,
        "residual_max": result.residual_max,
        "residual_min": result.residual_min,
        f"{start_vector}_negative_weight": weight,
        "iterations": result.iterations,
        "requested_iterations": result.requested_iterations,
        "finite": result.finite,
        "censored": result.censored,
        "breakdown": result.breakdown,
        "seconds": result.seconds,
    }


def alignment_with_backprop(
    pc: Mapping[str, np.ndarray],
    bp: Mapping[str, np.ndarray],
    head: str,
    thresholds: AcceptanceThresholds,
) -> dict[str, Any]:
    """Per-module cosine and norm ratio against backprop.

    PC gradients are summed and carry s while backprop's are a mean, so
    ``rel_norm_ratio`` divides by the head's ratio to cancel the units.
    ``min_cosine`` covers gated modules with a defined cosine;
    ``undefined_cosine_modules`` lists gated modules where one norm is zero, and
    ``excluded`` the modules below the floor.
    """
    comparison = compare_module_gradients(pc, bp, thresholds)
    head_ratio = None
    head_row = comparison["modules"].get(head)
    if head_row and head_row["other_norm"] > 0.0:
        head_ratio = head_row["reference_norm"] / head_row["other_norm"]
    modules = {}
    for name, row in comparison["modules"].items():
        ratio = (
            row["reference_norm"] / row["other_norm"]
            if row["other_norm"] > 0.0
            else None
        )
        modules[name] = {
            "cosine": row["cosine"],
            "pc_norm": row["reference_norm"],
            "bp_norm": row["other_norm"],
            "norm_ratio": ratio,
            "rel_norm_ratio": (
                ratio / head_ratio if ratio is not None and head_ratio else None
            ),
            "gated": row["gated"],
        }
    gated = [row for row in comparison["modules"].items() if row[1]["gated"]]
    cosines = [row["cosine"] for _, row in gated if row["cosine"] is not None]
    return {
        "modules": modules,
        "min_cosine": min(cosines) if cosines else None,
        "undefined_cosine_modules": [
            name for name, row in gated if row["cosine"] is None
        ],
        "excluded": comparison["excluded"],
    }


def _run_solver(
    name: str,
    objective: Any,
    start: Any,
    max_radius: float,
    thresholds: AcceptanceThresholds,
    budget: Budget,
) -> tuple[SolveResult, dict[str, Any]]:
    """Run one solver; return its result and the settings it used."""
    if name == "trust_region":
        config = TrustRegionConfig(
            tolerance=thresholds.rel_stationarity,
            initial_radius=min(1.0, max_radius),
            max_radius=max_radius,
        )
        return trust_region_newton_cg(objective, start, config, budget), asdict(config)
    config_lbfgs = LbfgsConfig(tolerance=thresholds.rel_stationarity)
    return lbfgs(objective, start, config_lbfgs, budget), asdict(config_lbfgs)


def _add_modules(
    total: dict[str, np.ndarray] | None, modules: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    if total is None:
        return {name: values.copy() for name, values in modules.items()}
    for name, values in modules.items():
        total[name] = total[name] + values
    return total


def _deadline(allocator: TimeAllocator | None) -> float | None:
    return None if allocator is None else allocator.next_deadline()


@dataclass
class SampleOutcome:
    """A sample's record, and the arrays folded into the batch sums."""

    record: dict[str, Any]
    modules: dict[str, dict[str, np.ndarray]] | None
    rounded: dict[str, dict[str, np.ndarray]] | None
    final_gradient_sq: dict[str, float]
    final_norm_sq: dict[str, float]


def _zero_signal_sample(
    sample: Sample,
    fns: EnergyFunctions,
    params: GraphParams,
    s: float,
    structure: GraphStructure,
    energy_start: float,
) -> SampleOutcome:
    """A sample with no masked tokens: E_s = ½‖ε‖², so ε = 0 is the exact solution."""
    derived = fns.derived(sample.zero_errors, params, sample.state, sample.clamps, s)
    grads = module_gradients(fns.weight_gradients(params, derived, s), structure)
    rounded = {
        name: modules
        for name, (modules, _) in float32_gradients(
            structure, params, derived, sample.clamps, s
        ).items()
    }
    record = {
        "status": ZERO_SIGNAL,
        "reason": "no masked tokens; ε = 0 is exact",
        "curvature_evidence": "exact",
        "sample": sample.index,
        "num_masked": sample.num_masked,
        "energy_start": energy_start,
        "epsilon_bound": 0.0,
        "solvers": {},
    }
    return SampleOutcome(
        record,
        {name: grads for name in SOLVERS},
        rounded,
        {name: 0.0 for name in SOLVERS},
        {name: 0.0 for name in SOLVERS},
    )


def run_sample(
    fixture: Fixture,
    sample: Sample,
    s: float,
    fns: EnergyFunctions,
    max_evals: int | None,
    lanczos_iters: int,
    start_lanczos_iters: int,
    thresholds: AcceptanceThresholds,
    allocator: TimeAllocator | None,
    seed: int,
) -> SampleOutcome:
    """Solve one sample with both solvers, measure curvature and run the gates."""
    structure, params = fixture.structure, fixture.params
    args = (params, sample.state, sample.clamps, s)
    objective = fns.objective(*args)
    energy_start, g0 = objective.value_and_grad(sample.zero_errors)
    energy_start = float(energy_start)
    if sample.num_masked <= 0.0:
        return _zero_signal_sample(sample, fns, params, s, structure, energy_start)
    bound = math.sqrt(max(2.0 * energy_start, 0.0))
    g0_norm = tree_norm(g0)
    start_vector = g0 if g0_norm > 0.0 else _random_like(seed, g0)
    spectrum_start = _spectrum_record(
        tree_spectrum(
            objective,
            sample.zero_errors,
            start_vector,
            start_lanczos_iters,
            _deadline(allocator),
        ),
        "gradient" if g0_norm > 0.0 else "random",
    )
    max_radius = 2.0 * bound if bound > 0.0 else 1.0

    solvers: dict[str, dict[str, Any]] = {}
    results: dict[str, SolveResult] = {}
    spectra: dict[str, SpectrumResult] = {}
    modules: dict[str, dict[str, np.ndarray]] = {}
    deriveds: dict[str, Any] = {}
    final_gradient_sq: dict[str, float] = {}
    final_norm_sq: dict[str, float] = {}
    settings: dict[str, Any] = {}
    for name in SOLVERS:
        result, settings[name] = _run_solver(
            name,
            objective,
            sample.zero_errors,
            max_radius,
            thresholds,
            Budget(max_passes=max_evals, deadline=_deadline(allocator)),
        )
        derived = fns.derived(result.x, *args)
        modules[name] = module_gradients(
            fns.weight_gradients(params, derived, s), structure
        )
        deriveds[name] = derived
        results[name] = result
        spectrum = tree_spectrum(
            objective,
            result.x,
            _random_like(seed + 1, result.x),
            lanczos_iters,
            _deadline(allocator),
        )
        spectra[name] = spectrum
        norm_sq = sum(
            float(jnp.sum(leaf**2)) for leaf in jax.tree_util.tree_leaves(result.x)
        )
        final_norm_sq[name] = norm_sq
        final_gradient_sq[name] = result.grad_norm**2
        solvers[name] = {
            "status": result.status,
            "energy_end": result.value,
            "energy_drop": (
                (energy_start - result.value) / energy_start
                if energy_start > 0.0
                else None
            ),
            "epsilon_norm": math.sqrt(norm_sq),
            "max_iterate_norm": math.sqrt(result.max_norm_sq),
            "epsilon_norm_by_node": {
                node: float(jnp.linalg.norm(error)) for node, error in result.x.items()
            },
            "grad_norm_start": result.initial_grad_norm,
            "grad_norm_end": result.grad_norm,
            "rel_stationarity": result.rel_stationarity,
            "spectrum_end": _spectrum_record(spectrum, "random"),
            "work": asdict(result.work),
            "seconds": result.seconds,
            "trace": result.trace,
        }

    between = compare_module_gradients(
        modules["trust_region"], modules["lbfgs"], thresholds
    )

    def summary(name: str) -> dict[str, Any]:
        return {
            "status": results[name].status,
            "rel_stationarity": results[name].rel_stationarity,
            "energy": results[name].value,
            "energy_start": energy_start,
            "max_norm_sq": results[name].max_norm_sq,
            "lambda_min": spectra[name].lambda_min,
            "lambda_max": spectra[name].lambda_max,
            "residual_min": spectra[name].residual_min,
            "residual_max": spectra[name].residual_max,
            "spectrum_finite": spectra[name].finite,
            "spectrum_censored": spectra[name].censored,
        }

    escape: EscapeResult | None = None
    preliminary = assess_reference(
        summary("trust_region"),
        summary("lbfgs"),
        between["passed"],
        thresholds,
        escape=None,
    )
    if (
        preliminary["status"] == CURVATURE_UNRESOLVED
        and preliminary["curvature_evidence"] == EVIDENCE_LANCZOS
    ):

        def agrees(point: Any) -> bool:
            point_modules = module_gradients(
                fns.weight_gradients(params, fns.derived(point, *args), s), structure
            )
            return compare_module_gradients(
                modules["trust_region"], point_modules, thresholds
            )["passed"]

        escape = escape_test(
            objective,
            results["trust_region"].x,
            results["trust_region"].value,
            spectra["trust_region"].lowest_vector,
            thresholds.rel_stationarity * g0_norm,
            agrees,
            thresholds,
            max_radius,
            Budget(max_passes=max_evals, deadline=_deadline(allocator)),
            seed + 2,
        )
    verdict = assess_reference(
        summary("trust_region"),
        summary("lbfgs"),
        between["passed"],
        thresholds,
        escape=None if escape is None else escape.outcome,
    )
    rounded_results = float32_gradients(
        structure, params, deriveds["trust_region"], sample.clamps, s
    )
    rounding = {
        name: {
            "dtype": dtype,
            "comparison": compare_module_gradients(
                modules["trust_region"], rounded_modules, thresholds
            ),
        }
        for name, (rounded_modules, dtype) in rounded_results.items()
    }
    record = {
        **verdict,
        "sample": sample.index,
        "num_masked": sample.num_masked,
        "energy_start": energy_start,
        "epsilon_bound": bound,
        "energy_gap": _energy_gap(results),
        "spectrum_start": spectrum_start,
        "solvers": solvers,
        "between_solvers": between,
        "escape": None if escape is None else asdict(escape),
        "float32": rounding,
        "settings": {
            **settings,
            "lanczos_iters": lanczos_iters,
            "start_lanczos_iters": start_lanczos_iters,
            "max_evals": max_evals,
        },
    }
    return SampleOutcome(
        record,
        modules,
        {name: value[0] for name, value in rounded_results.items()},
        final_gradient_sq,
        final_norm_sq,
    )


def _energy_gap(results: Mapping[str, SolveResult]) -> float | None:
    a, b = results["trust_region"].value, results["lbfgs"].value
    return relative_gap(a, b) if math.isfinite(a) and math.isfinite(b) else None


def _sample_view(record: Mapping[str, Any]) -> dict[str, Any]:
    """A sample record without its traces."""
    view = dict(record)
    view["solvers"] = {
        name: {k: v for k, v in solver.items() if k != "trace"}
        for name, solver in record.get("solvers", {}).items()
    }
    return view


def _not_started_sample(sample: Sample) -> dict[str, Any]:
    return {
        "status": CENSORED,
        "reason": "time cap reached before this sample",
        "not_started": True,
        "sample": sample.index,
        "num_masked": sample.num_masked,
        "solvers": {},
    }


def _lowest(values: Sequence[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return min(present) if present else None


def _highest(values: Sequence[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return max(present) if present else None


def _batch_solver_record(
    name: str,
    records: Sequence[Mapping[str, Any]],
    energy_start: float,
    outcomes: Sequence[SampleOutcome],
    g0_norm: float,
) -> dict[str, Any]:
    """Fold per-sample solver records into the batch's view of one solver."""
    rows = [(r, r["solvers"][name]) for r in records if name in r.get("solvers", {})]
    solves = [row for _, row in rows]
    energy_end = sum(row["energy_end"] for row in solves) if solves else None
    work: dict[str, int] = {}
    for row in solves:
        for key, count in row["work"].items():
            work[key] = work.get(key, 0) + count
    gradient_sq = sum(o.final_gradient_sq[name] for o in outcomes)
    norm_sq = sum(o.final_norm_sq[name] for o in outcomes)
    return {
        "sample_statuses": [row["status"] for row in solves],
        "energy_end": energy_end,
        "energy_drop": (
            (energy_start - energy_end) / energy_start
            if energy_end is not None and energy_start > 0.0
            else None
        ),
        "epsilon_norm": math.sqrt(norm_sq),
        "rel_stationarity": math.sqrt(gradient_sq) / g0_norm if g0_norm > 0.0 else None,
        "max_sample_rel_stationarity": _highest(
            [row["rel_stationarity"] for row in solves]
        ),
        "spectrum_end": {
            "lambda_min": _lowest(
                [row["spectrum_end"]["lambda_min"] for row in solves]
            ),
            "lambda_max": _highest(
                [row["spectrum_end"]["lambda_max"] for row in solves]
            ),
            "start_vector": "random",
        },
        "work": work,
        "seconds": sum(row["seconds"] for row in solves),
    }


def run_case(
    fixture: Fixture,
    case: Case,
    head_scale: str,
    fns: EnergyFunctions,
    max_evals: int | None,
    lanczos_iters: int,
    thresholds: AcceptanceThresholds,
    allocator: TimeAllocator | None = None,
    start_lanczos_iters: int = 40,
    sample_dir: Path | None = None,
) -> dict[str, Any]:
    """Solve every sample of one batch and head scale; combine and assess them.

    s is defined by the batch, and weight gradients are summed over samples for
    the comparison with batch backprop. The batch is accepted only if every
    sample is. With ``sample_dir``, each sample is written as it finishes.
    """
    started = time.monotonic()
    structure, params = fixture.structure, fixture.params
    head = structure.task_map["y"]
    s = resolve_head_scale(head_scale, case.num_masked)
    records: list[dict[str, Any]] = []
    outcomes: list[SampleOutcome] = []
    summed: dict[str, dict[str, np.ndarray] | None] = {n: None for n in SOLVERS}
    rounded: dict[str, dict[str, np.ndarray] | None] = {
        n: None for n in ROUNDING_VARIANTS
    }
    complete = True
    for sample in case.samples:
        if allocator is not None and allocator.expired():
            record = _not_started_sample(sample)
            complete = False
        else:
            outcome = run_sample(
                fixture,
                sample,
                s,
                fns,
                max_evals,
                lanczos_iters,
                start_lanczos_iters,
                thresholds,
                allocator,
                case_seed(case.seed, case.batch_index) * 1000 + sample.index,
            )
            record = outcome.record
            outcomes.append(outcome)
            for name in SOLVERS:
                summed[name] = _add_modules(summed[name], outcome.modules[name])
            for name in ROUNDING_VARIANTS:
                rounded[name] = _add_modules(rounded[name], outcome.rounded[name])
        if sample_dir is not None:
            sample_dir.mkdir(parents=True, exist_ok=True)
            write_json(sample_dir / f"sample{sample.index}.json", record)
        records.append(_sample_view(record))

    statuses = [record["status"] for record in records]
    between = None
    alignment: dict[str, Any] = {}
    float32: dict[str, Any] = {}
    if complete:
        between = compare_module_gradients(
            summed["trust_region"], summed["lbfgs"], thresholds
        )
        if case.num_masked > 0.0:
            bp_grads, _, _, _ = fixture.backprop(params, case.batch, case.key)
            bp_modules = module_gradients(bp_grads, structure)
            alignment = {
                name: alignment_with_backprop(
                    summed[name], bp_modules, head, thresholds
                )
                for name in SOLVERS
            }
        for name in ROUNDING_VARIANTS:
            comparison = compare_module_gradients(
                summed["trust_region"], rounded[name], thresholds
            )
            samples_passed = sum(
                record["float32"][name]["comparison"]["passed"]
                for record in records
                if "float32" in record
            )
            samples_measured = sum("float32" in record for record in records)
            float32[name] = {
                "dtype": "float32",
                "comparison": comparison,
                "samples_passed": samples_passed,
                "samples_measured": samples_measured,
                "passed": comparison["passed"] and samples_passed == samples_measured,
            }
    summed_agree = None if between is None else between["passed"]
    status = (
        ZERO_SIGNAL if case.num_masked <= 0.0 else batch_status(statuses, summed_agree)
    )
    energy_start = sum(record.get("energy_start", 0.0) for record in records)
    g0_norm = math.sqrt(
        sum(
            row["solvers"]["trust_region"]["grad_norm_start"] ** 2
            for row in records
            if "trust_region" in row.get("solvers", {})
        )
    )
    solvers = {
        name: _batch_solver_record(name, records, energy_start, outcomes, g0_norm)
        for name in SOLVERS
    }
    for name in SOLVERS:
        solvers[name]["vs_backprop"] = alignment.get(name)
    satisfied = sum(status_ in SATISFIED_STATUSES for status_ in statuses)
    return {
        "status": status,
        "seed": case.seed,
        "batch_index": case.batch_index,
        "head_scale_spec": head_scale,
        "head_scale": s,
        "num_masked": case.num_masked,
        "energy_start": energy_start,
        "epsilon_bound": math.sqrt(max(2.0 * energy_start, 0.0)),
        "sample_statuses": statuses,
        "accepted_fraction": satisfied / len(statuses) if statuses else None,
        "curvature_evidence": [r.get("curvature_evidence") for r in records],
        "spectrum_start": {
            "start_vector": "gradient",
            "lambda_min": _lowest(
                [
                    r["spectrum_start"]["lambda_min"]
                    for r in records
                    if "spectrum_start" in r
                ]
            ),
            "lambda_max": _highest(
                [
                    r["spectrum_start"]["lambda_max"]
                    for r in records
                    if "spectrum_start" in r
                ]
            ),
            "max_gradient_negative_weight": _highest(
                [
                    r["spectrum_start"].get("gradient_negative_weight")
                    for r in records
                    if "spectrum_start" in r
                ]
            ),
        },
        "solvers": solvers,
        "between_solvers": between,
        "float32": float32,
        "samples": records,
        "duration_seconds": {
            "total": time.monotonic() - started,
            **{name: solvers[name]["seconds"] for name in SOLVERS},
        },
    }


def _not_started(case: Case, head_scale: str) -> dict[str, Any]:
    return {
        "status": CENSORED,
        "not_started": True,
        "seed": case.seed,
        "batch_index": case.batch_index,
        "head_scale_spec": head_scale,
        "head_scale": resolve_head_scale(head_scale, case.num_masked),
        "num_masked": case.num_masked,
    }


def summarize(
    records: Sequence[Mapping[str, Any]], thresholds: AcceptanceThresholds
) -> dict[str, Any]:
    """Per head scale: reference statuses and the tri-state hypothesis readings.

    Cases without signal are left out. With r = ``min_accepted_fraction``:

    - ``h1_reference_reproducible``: true if accepted ≥ r of cases, false if
      accepted plus unresolved can't reach r, else null. Hitting a budget or cap
      is unresolved.
    - ``h2_distinct_from_backprop``: true if any accepted case has a module cosine
      below ``bp_distinct_cosine``; false if nothing is unresolved, at least one
      case is accepted and all accepted are at or above ``bp_like_cosine``; else
      null.
    - ``h3_float32``, per rounding variant: false if any accepted case fails the
      gate; true if nothing is unresolved, at least one is accepted and all pass;
      else null.
    """
    by_scale: dict[str, dict[str, Any]] = {}
    for spec in dict.fromkeys(r["head_scale_spec"] for r in records):
        group = [
            r
            for r in records
            if r["head_scale_spec"] == spec and r["status"] != ZERO_SIGNAL
        ]
        statuses: dict[str, int] = {}
        for record in group:
            statuses[record["status"]] = statuses.get(record["status"], 0) + 1
        accepted = [r for r in group if r["status"] == ACCEPTED]
        unresolved = sum(statuses.get(status, 0) for status in UNRESOLVED_STATUSES)
        needed = thresholds.min_accepted_fraction * len(group)
        reproducible = None
        if group:
            if len(accepted) >= needed:
                reproducible = True
            elif len(accepted) + unresolved < needed:
                reproducible = False
        cosines = [
            r["solvers"]["trust_region"]["vs_backprop"]["min_cosine"] for r in accepted
        ]
        cosines = [c for c in cosines if c is not None]
        distinct = None
        if cosines and min(cosines) < thresholds.bp_distinct_cosine:
            distinct = True
        elif cosines and unresolved == 0 and min(cosines) >= thresholds.bp_like_cosine:
            distinct = False
        float32: dict[str, Any] = {}
        for variant in ROUNDING_VARIANTS:
            passes = [r["float32"][variant]["passed"] for r in accepted]
            reading = None
            if not all(passes):
                reading = False
            elif passes and unresolved == 0:
                reading = True
            float32[variant] = {"reading": reading, "passed": sum(passes)}
        by_scale[spec] = {
            "cases": len(group),
            "statuses": statuses,
            "censored_fraction": (
                statuses.get(CENSORED, 0) / len(group) if group else None
            ),
            "unresolved_cases": unresolved,
            "h1_reference_reproducible": reproducible,
            "min_cosine_vs_backprop": min(cosines) if cosines else None,
            "h2_distinct_from_backprop": distinct,
            "h3_float32": float32,
            "h3_accepted_cases": len(accepted),
        }
    return by_scale


def run_finished(records: Sequence[Mapping[str, Any]], planned: int) -> bool:
    """Whether every planned case ran to a result that a budget didn't cut short."""
    return len(records) == planned and not any(r["status"] == CENSORED for r in records)


def _peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def measure_flop_rate(dimension: int, vectors: int = 20) -> float:
    """Measured speed of the reorthogonalisation's matrix-vector products, flop/s."""
    basis = np.ones((vectors, dimension), dtype=np.float64)
    v = np.ones(dimension, dtype=np.float64)
    started = time.monotonic()
    np.asarray(basis.T @ (basis @ v))
    return 4.0 * vectors * dimension / max(time.monotonic() - started, 1e-9)


def reorth_seconds(dimension: int, iterations: int, flop_rate: float) -> float:
    """Reorthogonalisation time of one Lanczos run.

    Two passes per iteration, each two matrix-vector products of 2·k·n flops,
    summed over k up to the iteration count: 4·n·iterations² flops.
    """
    return 4.0 * dimension * iterations**2 / flop_rate


def timing_smoke(
    fixture: Fixture,
    fns: EnergyFunctions,
    specs: Sequence[str],
    max_evals: int,
    lanczos_iters: int,
    start_lanczos_iters: int,
    cap_minutes: float,
    timing_passes: int,
    num_cases: int,
    thresholds: AcceptanceThresholds,
    batch_sizes: Sequence[int],
) -> None:
    """Time short real per-sample solves and print cost and memory per batch size.

    A solver that converges within ``timing_passes`` sets the low estimate with
    its actual passes; one that doesn't sets it with ``timing_passes``, which is
    only a floor. The high estimate assumes every solve reaches ``max_evals``.
    """
    case = fixture.cases[0]
    sample = next(
        (item for item in case.samples if item.num_masked > 0.0), case.samples[0]
    )
    dimension = sum(x.size for x in jax.tree_util.tree_leaves(sample.zero_errors))
    print(f"backend: {jax.default_backend()}, devices: {jax.devices()}")
    print(f"per-sample epsilon dimension: {dimension}")
    flop_rate = measure_flop_rate(dimension)
    low_seconds = 0.0
    high_seconds = 0.0
    for spec in specs:
        s = resolve_head_scale(spec, case.num_masked)
        objective = fns.objective(fixture.params, sample.state, sample.clamps, s)
        first = time.monotonic()
        _, g0 = objective.value_and_grad(sample.zero_errors)
        jax.block_until_ready(g0)
        first_grad = time.monotonic() - first
        first = time.monotonic()
        jax.block_until_ready(objective.hvp(sample.zero_errors, g0))
        first_hvp = time.monotonic() - first
        bound = math.sqrt(max(2.0 * float(objective.value(sample.zero_errors)), 0.0))
        max_radius = 2.0 * bound if bound > 0.0 else 1.0
        solves = {}
        for name in SOLVERS:
            solves[name], _ = _run_solver(
                name,
                objective,
                sample.zero_errors,
                max_radius,
                thresholds,
                Budget(max_passes=timing_passes),
            )
        lanczos_probe = 10
        spectrum = tree_spectrum(
            objective,
            solves["trust_region"].x,
            _random_like(0, sample.zero_errors),
            lanczos_probe,
        )
        per_iter = spectrum.seconds / max(spectrum.iterations, 1)
        reorth = 2.0 * reorth_seconds(dimension, lanczos_iters, flop_rate)
        lanczos_seconds = per_iter * (lanczos_iters * 2 + start_lanczos_iters) + reorth
        escape = escape_test(
            objective,
            solves["trust_region"].x,
            solves["trust_region"].value,
            spectrum.lowest_vector,
            thresholds.rel_stationarity * tree_norm(g0),
            _always_agrees,
            thresholds,
            max_radius,
            Budget(max_passes=timing_passes),
            0,
        )
        print(
            f"s={spec} ({s:.4g}): first call incl. compile: gradient "
            f"{first_grad:.2f}s, HVP {first_hvp:.2f}s (compile happens once)"
        )
        low = lanczos_seconds + escape.seconds
        high = lanczos_seconds
        for name in SOLVERS:
            solve = solves[name]
            per_pass = solve.seconds / max(solve.work.passes, 1)
            state = "converged" if solve.status == CONVERGED else solve.status
            print(
                f"  {name}: {solve.work.passes} passes in {solve.seconds:.1f}s, "
                f"{state}, {per_pass:.3f}s per pass"
            )
            low += solve.seconds
            high += per_pass * max_evals
        high += (escape.seconds / max(_escape_passes(escape), 1)) * max_evals
        print(
            f"  Lanczos: {per_iter:.3f}s per iteration, about "
            f"{lanczos_seconds:.0f}s per sample including reorthogonalisation "
            f"(measured {flop_rate:.1e} flop/s)"
        )
        print(f"  escape test: {escape.seconds:.1f}s, outcome {escape.outcome}")
        print(f"  per-sample cost: {low:.0f}s low, {high:.0f}s high")
        low_seconds += low
        high_seconds += high
    lanczos_bytes = lanczos_memory_bytes(dimension, lanczos_iters)
    lbfgs_bytes = lbfgs_memory_bytes(dimension)
    print(
        f"memory per sample: Lanczos basis {lanczos_bytes / 2**30:.2f} GiB, L-BFGS "
        f"{lbfgs_bytes / 2**30:.2f} GiB; both are held in turn, so peak is about "
        f"{max(lanczos_bytes, lbfgs_bytes) / 2**30:.2f} GiB plus the process"
    )
    print(f"process peak RSS so far: {_peak_rss_bytes() / 2**30:.2f} GiB")
    for batch_size in batch_sizes:
        samples = num_cases * batch_size
        low_minutes = samples * low_seconds / 60.0
        high_minutes = samples * high_seconds / 60.0
        verdict = (
            "yes"
            if high_minutes <= cap_minutes
            else ("no" if low_minutes > cap_minutes else "only if solves stop early")
        )
        print(
            f"batch size {batch_size}: {samples} sample solves per scale, estimated "
            f"total {low_minutes:.1f}-{high_minutes:.1f} minutes over all scales; "
            f"fits the {cap_minutes:g}-minute cap: {verdict}"
        )


def _always_agrees(point: Any) -> bool:
    return True


def _escape_passes(escape: EscapeResult) -> int:
    return 0 if escape.work is None else escape.work.passes


def _effective_lanczos_iters(requested: int, dimension: int, max_gib: float) -> int:
    """Limit Lanczos iterations so its basis fits in ``max_gib``."""
    allowed = int(max_gib * 2**30 / (8 * dimension))
    return max(min(requested, allowed, dimension), 1)


def build_parser() -> argparse.ArgumentParser:
    """The probe's command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fixture", choices=("tiny", "bert_mini"), required=True)
    parser.add_argument("--checkpoint", default="checkpoints/bert_mini_128.json")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--stage", type=int, default=None)
    parser.add_argument("--data-dir", default="data/domain_forgetting")
    parser.add_argument("--domain", default="news")
    parser.add_argument("--mask-prob", type=float, default=0.15)
    parser.add_argument("--head-scales", default="1,mean,midpoint")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-batches", type=int, default=4)
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--max-evals", type=int, default=3000)
    parser.add_argument("--lanczos-iters", type=int, default=200)
    parser.add_argument("--start-lanczos-iters", type=int, default=40)
    parser.add_argument("--lanczos-max-gib", type=float, default=8.0)
    parser.add_argument("--time-cap-minutes", type=float, default=180.0)
    parser.add_argument("--timing-passes", type=int, default=40)
    parser.add_argument("--output", default=None)
    parser.add_argument("--timing-smoke", action="store_true")
    return parser


def _case_settings(
    record: Mapping[str, Any], lanczos_iters: int, max_evals: int
) -> dict[str, Any]:
    samples = record.get("samples", [])
    settings = next(
        (sample["settings"] for sample in samples if "settings" in sample), None
    )
    return {
        "head_scale": record["head_scale"],
        "lanczos_iters": lanczos_iters,
        "max_evals": max_evals,
        "first_sample_settings": settings,
        "per_sample_max_radius": [
            sample["settings"]["trust_region"]["max_radius"]
            for sample in samples
            if "settings" in sample
        ],
    }


def _summary_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """A case record for the run summary, without per-sample detail."""
    return {k: v for k, v in record.items() if k != "samples"}


def main(argv: Sequence[str] | None = None) -> int:
    """Run the probe; returns 0 when every case was recorded."""
    args = build_parser().parse_args(argv)
    jax.config.update("jax_enable_x64", True)
    if (args.run_dir is None) != (args.stage is None):
        raise ValueError("--run-dir and --stage must be given together")
    if not args.timing_smoke and args.output is None:
        raise ValueError("--output is required unless --timing-smoke is set")
    output = None if args.timing_smoke else Path(args.output)
    if output is not None and output.exists():
        raise FileExistsError(f"{output} exists; a run directory is never overwritten")
    specs = parse_head_scales(args.head_scales)
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    thresholds = AcceptanceThresholds()
    print(f"backend: {jax.default_backend()}, x64: {jax.config.jax_enable_x64}")
    if args.fixture == "tiny":
        fixture = make_tiny_fixture(seeds[0], args.batch_size, args.num_batches, seeds)
    else:
        fixture = make_bert_mini_fixture(args, seeds)
    fns = make_energy_functions(fixture.structure)
    dimension = sum(
        x.size
        for x in jax.tree_util.tree_leaves(fixture.cases[0].samples[0].zero_errors)
    )
    lanczos_iters = _effective_lanczos_iters(
        args.lanczos_iters, dimension, args.lanczos_max_gib
    )
    if lanczos_iters < min(args.lanczos_iters, dimension):
        print(
            f"Lanczos limited to {lanczos_iters} iterations by "
            f"--lanczos-max-gib {args.lanczos_max_gib:g}"
        )
    if args.timing_smoke:
        timing_smoke(
            fixture,
            fns,
            specs,
            args.max_evals,
            lanczos_iters,
            args.start_lanczos_iters,
            args.time_cap_minutes,
            args.timing_passes,
            len(fixture.cases),
            thresholds,
            sorted({args.batch_size, 8, 64}),
        )
        return 0

    prepare_output_dir(output)
    manifest = build_manifest(vars(args), fixture, thresholds, lanczos_iters)
    write_json(output / "manifest.json", manifest)
    work_list = [(case, spec) for case in fixture.cases for spec in specs]
    total_samples = sum(len(case.samples) for case, _ in work_list)
    allocator = TimeAllocator(
        time.monotonic() + args.time_cap_minutes * 60.0,
        total_samples * UNITS_PER_SAMPLE,
    )
    records: list[dict[str, Any]] = []
    for case, spec in work_list:
        stem = f"seed{case.seed}_batch{case.batch_index}_s{spec}"
        if allocator.expired():
            record = _not_started(case, spec)
        else:
            record = run_case(
                fixture,
                case,
                spec,
                fns,
                args.max_evals,
                lanczos_iters,
                thresholds,
                allocator,
                args.start_lanczos_iters,
                output / "records" / stem,
            )
        write_json(output / "records" / f"{stem}.json", record)
        manifest["case_settings"][stem] = _case_settings(
            record, lanczos_iters, args.max_evals
        )
        write_json(output / "manifest.json", manifest)
        records.append(_summary_record(record))
        write_json(
            output / "summary.json",
            {
                "complete": run_finished(records, len(work_list)),
                "cases_recorded": len(records),
                "cases_planned": len(work_list),
                "by_head_scale": summarize(records, thresholds),
                "records": records,
            },
        )
        print(
            f"seed={case.seed} batch={case.batch_index} s={spec} "
            f"status={record['status']} "
            f"samples={record.get('sample_statuses')}",
            flush=True,
        )
    print(json.dumps(json_ready(summarize(records, thresholds))[0], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
