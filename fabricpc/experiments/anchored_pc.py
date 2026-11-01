"""Paired, fixed-weight anchored-PC diagnostic. No training loop or optimizer."""

from __future__ import annotations

import argparse
import json
import math
import platform
import resource
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np

from fabricpc.core.inference import InferenceBase, InferenceSGDNormClip
from fabricpc.core.learning import compute_local_weight_gradients
from fabricpc.experiments.domain_forgetting.artifacts import file_sha256
from fabricpc.experiments.domain_forgetting.data import load_domain_cache
from fabricpc.experiments.domain_forgetting.runner import (
    RunConfig,
    build_structure,
    fixed_mlm_batches,
    load_base_model,
    max_free_depth,
)
from fabricpc.experiments.domain_forgetting.validation import FixedMlmBatches
from fabricpc.experiments.forgetting_benchmark.sequential_trainer import params_hash
from fabricpc.graph_initialization.params_initializer import initialize_params
from fabricpc.nodes.encoder_transformer import create_encoder_mlm_model
from fabricpc.training.train_autoregressive import compute_loss
from fabricpc.training.train_backprop import (
    _loss_and_grad_backprop_mlm_no_update,
    compute_forward_pass,
)
from fabricpc.training.train_mlm import (
    _prepare_pcn_mlm_batch,
    get_graph_param_gradient_pcn_mlm,
)

Tree = Any
BETAS = (0.0, 0.01, 0.1, 0.5, 1.0)


def _validate_tree(tree: Tree) -> None:
    for leaf in jax.tree_util.tree_leaves(tree):
        if not np.all(np.isfinite(np.asarray(leaf))):
            raise ValueError("gradient contains non-finite values")


def _validate_pair(left: Tree, right: Tree) -> None:
    if cast(Any, jax.tree_util.tree_structure(left)) != jax.tree_util.tree_structure(
        right
    ):
        raise ValueError("gradient tree structures differ")
    for a, b in zip(jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right)):
        if a.shape != b.shape:
            raise ValueError("gradient shapes differ")
    _validate_tree(left)
    _validate_tree(right)


def _norm(tree: Tree) -> float:
    # Accumulate host diagnostics in float64, even for float32 model leaves.
    return float(
        np.sqrt(
            sum(
                np.vdot(
                    np.asarray(x, dtype=np.float64), np.asarray(x, dtype=np.float64)
                ).real
                for x in jax.tree_util.tree_leaves(tree)
            )
        )
    )


def _difference(left: Tree, right: Tree) -> Tree:
    return jax.tree_util.tree_map(lambda a, b: a - b, left, right)


def blend_gradients(bp: Tree, pc: Tree, beta: float) -> Tree:
    """Mix compatible finite gradients; preserve both exact endpoints."""
    if not math.isfinite(beta) or not 0 <= beta <= 1:
        raise ValueError("beta must be finite and in [0, 1]")
    _validate_pair(bp, pc)
    if beta == 0:
        return bp
    if beta == 1:
        return pc
    return jax.tree_util.tree_map(lambda a, b: (1 - beta) * a + beta * b, bp, pc)


def normalized_pc_gradient(grads: Tree, scale: float) -> Tree:
    """Compensate the output-energy multiplier in the weak-supervision rule."""
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("output energy scale must be finite and positive")
    _validate_tree(grads)
    return jax.tree_util.tree_map(lambda g: g / scale, grads)


def match_norm(bp: Tree, target: Tree) -> Tree | None:
    """Keep BP direction with target norm, or report an undefined direction."""
    _validate_pair(bp, target)
    norm = _norm(bp)
    if norm <= 1e-12:
        return None
    ratio = _norm(target) / norm
    return jax.tree_util.tree_map(lambda g: g * ratio, bp)


def direction_metrics(bp: Tree, candidate: Tree) -> dict[str, Any]:
    """Return norm, alignment and the correction orthogonal to BP."""
    _validate_pair(bp, candidate)
    bn, cn = _norm(bp), _norm(candidate)
    dot = float(
        sum(
            np.vdot(
                np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
            ).real
            for a, b in zip(
                jax.tree_util.tree_leaves(bp), jax.tree_util.tree_leaves(candidate)
            )
        )
    )
    perpendicular = None
    if bn > 1e-12:
        perpendicular = _norm(
            jax.tree_util.tree_map(lambda a, b: b - (dot / bn**2) * a, bp, candidate)
        )
    difference = _norm(_difference(candidate, bp))
    return {
        "bp_norm": bn,
        "candidate_norm": cn,
        "norm_ratio": cn / bn if bn > 1e-12 else None,
        "cosine": (
            float(np.clip(dot / (bn * cn), -1, 1)) if min(bn, cn) > 1e-12 else None
        ),
        "difference_norm": difference,
        "relative_error": difference / bn if bn > 1e-12 else None,
        "perpendicular_norm": perpendicular,
    }


def module_metrics(bp: Tree, candidate: Tree) -> dict[str, Any]:
    return {
        name: direction_metrics(bp.nodes[name], candidate.nodes[name])
        for name in bp.nodes
        if jax.tree_util.tree_leaves(bp.nodes[name])
    }


def anchor_passes(metrics: dict[str, Any], *, strict: bool = False) -> bool:
    relative_tol, cosine_tol, absolute_tol = (
        (1e-6, 0.999999, 1e-10) if strict else (1e-3, 0.9999, 1e-7)
    )
    return all(
        (
            m["difference_norm"] <= absolute_tol
            if m["bp_norm"] <= 1e-12
            else m["relative_error"] <= relative_tol
            and m["cosine"] is not None
            and m["cosine"] >= cosine_tol
        )
        for m in metrics.values()
    )


def convergence_passes(residual: float, initial: float, gradient_change: float) -> bool:
    """Check absolute and forcing-relative residuals plus T/2T agreement."""
    return bool(
        all(math.isfinite(v) and v >= 0 for v in (residual, initial, gradient_change))
        and residual / max(1.0, initial) <= 1e-3
        and residual / max(1e-12, initial) <= 1e-3
        and gradient_change <= 0.01
    )


def _refresh(params: Tree, state: Tree, clamps: Tree, structure: Tree) -> Tree:
    state = InferenceBase.zero_grads(params, state, clamps, structure)
    return InferenceBase.forward_value_and_grad(params, state, clamps, structure)


def make_pc_trace(structure: Tree, steps: int):
    """JIT one solve, with recomputed final-state gradients and raw residuals."""
    config = dict(structure.config["inference"].config)

    def solve(params, batch, key):
        _, _, _, initial, clamps = _prepare_pcn_mlm_batch(params, batch, structure, key)
        free = tuple(name for name in structure.nodes if name not in clamps)

        def stats(state):
            residual = jnp.sqrt(
                sum(jnp.sum(state.nodes[n].latent_grad ** 2) for n in free)
            )
            energy = (
                sum(jnp.sum(s.energy) for s in state.nodes.values())
                / batch["x"].shape[0]
            )
            clipped, count = jnp.array(0.0), 0
            for name in free:
                g = state.nodes[name].latent_grad
                norms = jnp.sqrt(jnp.sum(g**2, axis=tuple(range(1, g.ndim))))
                clipped = clipped + jnp.sum(norms > config["max_norm"])
                count += g.shape[0]
            displacement = jnp.sqrt(
                sum(
                    jnp.sum((state.nodes[n].z_latent - initial.nodes[n].z_latent) ** 2)
                    for n in free
                )
            )
            return jnp.stack([energy, residual, displacement, clipped / max(1, count)])

        initial = _refresh(params, initial, clamps, structure)

        def step(state, _):
            before = stats(state)
            state = InferenceSGDNormClip.update_latents(
                params, state, clamps, structure, config
            )
            return _refresh(params, state, clamps, structure), before

        final, trace = jax.lax.scan(step, initial, None, length=steps)
        trace = jnp.concatenate([trace, stats(final)[None, :]], axis=0)
        return compute_local_weight_gradients(params, final, structure), trace

    return jax.jit(solve)


def tiny_anchor_check() -> list[dict[str, Any]]:
    """Strict float64 endpoint gate, including perturbed weights and dropout."""
    from fabricpc.core.inference import InferenceSpikingPrecision

    results = []
    with jax.enable_x64(True):
        for dropout in (0.0, 0.1):

            def build(steps):
                return create_encoder_mlm_model(
                    depth=1,
                    embed_dim=8,
                    num_heads=2,
                    mlp_dim=16,
                    seq_len=6,
                    vocab_size=20,
                    architecture="bert",
                    residual_identity=True,
                    dropout_rate=dropout,
                    use_attention_mask=True,
                    inference=InferenceSpikingPrecision(eta_infer=0, infer_steps=steps),
                )

            structure = build(max_free_depth(build(1)))
            key = jax.random.PRNGKey(0)
            params = jax.tree_util.tree_map(
                lambda p: p.astype(jnp.float64), initialize_params(structure, key)
            )
            batch = {
                "x": jnp.array([[1, 3, 4, 5, 2, 0]]),
                "y": jnp.array([[1, 6, 4, 5, 2, 0]]),
                "loss_mask": jnp.array([[0, 1, 0, 0, 0, 0]], dtype=jnp.float32),
                "attention_mask": jnp.array([[1, 1, 1, 1, 1, 0]]),
            }
            for perturb in (0.0, 1e-4):
                p = jax.tree_util.tree_map(lambda x: x + perturb, params)
                _, bp, _ = _loss_and_grad_backprop_mlm_no_update(
                    p, batch, structure, key
                )
                pc, *_ = get_graph_param_gradient_pcn_mlm(
                    p, batch, structure, key, "initial"
                )
                metrics = module_metrics(bp, pc)
                results.append(
                    {
                        "dropout": dropout,
                        "perturbation": perturb,
                        "passed": anchor_passes(metrics, strict=True),
                        "modules": metrics,
                    }
                )
    return results


@dataclass(frozen=True)
class DiagnosticConfig:
    checkpoint: str
    data_dir: str
    output_dir: str
    domain: str = "news"
    num_batches: int = 8
    batch_size: int = 8
    seed: int = 0
    budget_seconds: float = 7200
    rates: tuple[float, ...] = (0.1, 0.05, 0.01)
    step_pairs: tuple[int, ...] = (30, 60, 120)
    scales: tuple[float, ...] = (1.0, 0.1, 0.01)
    dropout_rate: float = 0.1
    expected_shape: dict[str, int] | None = None

    def validate(self):
        if self.num_batches < 2 or self.batch_size < 1:
            raise ValueError("need >=2 batches for calibration and positive batch size")
        if not math.isfinite(self.budget_seconds) or self.budget_seconds <= 0:
            raise ValueError("budget must be finite and positive")
        if not 0 <= self.dropout_rate < 1:
            raise ValueError("dropout rate must be in [0, 1)")
        if not self.rates or any(not math.isfinite(r) or r <= 0 for r in self.rates):
            raise ValueError("inference rates must be finite and positive")
        if not self.step_pairs or any(t < 1 or t * 2 > 240 for t in self.step_pairs):
            raise ValueError("step pairs must allow T/2T within the 240-step cap")
        if self.scales != (1.0, 0.1, 0.01):
            raise ValueError("diagnostic requires the preregistered output scales")


class BudgetExceeded(RuntimeError):
    """Diagnostic reached its wall-clock budget between operations."""


def run_diagnostic(config: DiagnosticConfig) -> dict[str, Any]:
    """Run Stage A only, writing incremental evidence and a conservative decision."""
    config.validate()
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    summary: dict[str, Any] = {
        "status": "running",
        "decision": "no_go",
        "config": asdict(config),
        "jax": jax.__version__,
        "python": platform.python_version(),
        "devices": [str(d) for d in jax.devices()],
        "limitations": [
            "One-step probes are not training evidence.",
            "Budget checks occur between operations; use an outer process timeout.",
        ],
    }
    report_path = output / "summary.json"

    def save():
        summary["elapsed_seconds"] = time.monotonic() - started
        summary["host_peak_rss_raw"] = resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss
        summary["host_peak_rss_unit"] = (
            "bytes" if platform.system() == "Darwin" else "KiB"
        )
        summary["device_memory"] = [d.memory_stats() for d in jax.devices()]
        report_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")

    def emit(kind, **values):
        event = {"kind": kind, "elapsed_seconds": time.monotonic() - started, **values}
        with (output / "events.jsonl").open("a") as f:
            f.write(json.dumps(event, allow_nan=False) + "\n")
        print(f"{event['elapsed_seconds']:.1f}s {kind}", flush=True)

    def check_budget():
        if time.monotonic() - started >= config.budget_seconds:
            raise BudgetExceeded("wall-clock budget reached")

    try:
        summary["revision"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        summary["revision"] = "source bundle; see bundle manifest"
    save()
    try:
        check_budget()
        summary["tiny_anchor"] = tiny_anchor_check()
        save()
        if not all(r["passed"] for r in summary["tiny_anchor"]):
            summary.update(status="anchor_failed", reason="tiny float64 anchor failed")
            return summary
        # Tiny probes are separate correctness tests; release their compiled graphs.
        jax.clear_caches()
        run = RunConfig(
            checkpoint=config.checkpoint,
            data_dir=config.data_dir,
            output_dir="",
            domains=(config.domain,),
            mode="pc",
            batch_size=config.batch_size,
            # Keep dropout source nodes in every graph, including dropout-off
            # probes. Toggle only their mask sampling probability below.
            dropout_rate=max(config.dropout_rate, 0.1),
            expected_shape=config.expected_shape,
        )
        run.validate()
        base = load_base_model(run)
        base.structure = base.structure._replace(
            config={**base.structure.config, "dropout_rate": 0.0}
        )
        cache = load_domain_cache(
            config.data_dir,
            config.domain,
            tokenizer_json=base.tokenizer_json,
            seq_len=base.model_config["seq_len"],
            max_valid_windows=config.num_batches * config.batch_size,
        )
        batches, keys = fixed_mlm_batches(
            cache.train,
            base.structure,
            base.special_ids,
            num_batches=config.num_batches,
            batch_size=config.batch_size,
            mask_prob=0.15,
            seed=config.seed,
        )
        valid = [
            {k: jnp.asarray(v) for k, v in b.items()}
            for b in FixedMlmBatches(cache.valid, config.batch_size)
        ]
        if len(valid) < config.num_batches:
            raise ValueError("insufficient fixed validation batches")
        summary["source"] = base.source
        summary["data_manifest"] = cache.manifest
        summary["data_manifest_sha256"] = file_sha256(
            Path(config.data_dir) / config.domain / "manifest.json"
        )
        stored_batches: dict[str, Any] = {
            f"{split}_{i}_{name}": np.asarray(value)
            for split, bs in (("train", batches), ("valid", valid))
            for i, b in enumerate(bs)
            for name, value in b.items()
        }
        stored_batches["dropout_keys"] = np.asarray(keys)
        np.savez_compressed(output / "probe_batches.npz", **stored_batches)
        summary["batch_sha256"] = file_sha256(output / "probe_batches.npz")
        summary["initial_params_sha256"] = params_hash(base.params)
        # Check every batch with dropout off, then repeat with shared training masks.
        anchor_error = []
        anchors = []
        bp_grads = []
        spike_grads = []
        for dropout in (0.0, config.dropout_rate):
            structure = base.structure._replace(
                config={**base.structure.config, "dropout_rate": dropout}
            )
            bp_fn = jax.jit(
                lambda p, b, k: _loss_and_grad_backprop_mlm_no_update(
                    p, b, structure, k
                )
            )
            spike_fn = jax.jit(
                lambda p, b, k: get_graph_param_gradient_pcn_mlm(
                    p, b, structure, k, "initial"
                )[0]
            )
            for i, (batch, key) in enumerate(zip(batches, keys)):
                check_budget()
                loss, bp, _ = bp_fn(base.params, batch, key)
                pc = spike_fn(base.params, batch, key)
                metrics = module_metrics(bp, pc)
                record = {
                    "batch": i,
                    "dropout": dropout,
                    "passed": anchor_passes(metrics),
                    "modules": metrics,
                    "loss": float(loss),
                }
                anchors.append(record)
                emit("anchor", **record)
                if dropout == 0:
                    bp_grads.append(bp)
                    spike_grads.append(pc)
                    anchor_error.append(_norm(_difference(pc, bp)))
            del bp_fn, spike_fn
        summary["anchor_checks"] = anchors
        save()
        if not all(r["passed"] for r in anchors):
            summary.update(status="anchor_failed", reason="checkpoint anchor failed")
            return summary

        def pc_structure(rate, steps, scale):
            structure = build_structure(
                base.model_config,
                replace(run, pc_output_energy_scale=scale),
                InferenceSGDNormClip(eta_infer=rate, infer_steps=steps, max_norm=5),
            )
            return structure._replace(config={**structure.config, "dropout_rate": 0.0})

        def assess(fn_short, fn_long, i, scale):
            check_budget()
            g1, t1 = fn_short(base.params, batches[i], keys[i])
            g2, t2 = fn_long(base.params, batches[i], keys[i])
            g1 = normalized_pc_gradient(g1, scale)
            g2 = normalized_pc_gradient(g2, scale)
            trace = np.asarray(t2)
            if not np.all(np.isfinite(trace)):
                raise ValueError("non-finite PC inference trace")
            change = _norm(_difference(g2, g1)) / max(_norm(g2), 1e-12)
            residual, initial = float(trace[-1, 1]), float(trace[0, 1])
            return (
                g2,
                trace,
                {
                    "residual": residual,
                    "initial_residual": initial,
                    "relative_residual": residual / max(initial, 1e-12),
                    "gradient_change_T_2T": change,
                    "converged": convergence_passes(residual, initial, change),
                },
            )

        selected = None
        # Order by max step count, then rate; first passing candidate is the fixed rule.
        for steps in sorted(config.step_pairs):
            for rate in config.rates:
                check_budget()
                short = make_pc_trace(pc_structure(rate, steps, 1), steps)
                long = make_pc_trace(pc_structure(rate, 2 * steps, 1), 2 * steps)
                passed = True
                for i in range(2):
                    _, trace, stats = assess(short, long, i, 1)
                    emit("calibration", rate=rate, steps=steps, batch=i, **stats)
                    np.save(output / f"calibration_{rate}_{steps}_{i}.npy", trace)
                    passed &= stats["converged"]
                del short, long
                jax.clear_caches()
                if passed:
                    selected = (rate, steps)
                    break
            if selected:
                break
        summary["selected_solver"] = selected
        save()
        if selected is None:
            summary.update(
                status="convergence_failed",
                reason="no solver passed both calibration batches",
            )
            return summary

        rate, steps = selected
        # Verify the selected solver under a shared training-dropout mask too.
        dropout_structure = pc_structure(rate, steps * 2, 1)
        dropout_structure = dropout_structure._replace(
            config={**dropout_structure.config, "dropout_rate": config.dropout_rate}
        )
        drop_long = make_pc_trace(dropout_structure, steps * 2)
        drop_short = make_pc_trace(dropout_structure, steps)
        _, drop_trace, drop_stats = assess(drop_short, drop_long, 0, 1)
        summary["dropout_solver_check"] = drop_stats
        np.save(output / "dropout_trace.npy", drop_trace)
        emit("dropout_solver", **drop_stats)
        if not drop_stats["converged"]:
            summary.update(
                status="convergence_failed",
                reason="selected solver failed training-dropout check",
            )
            return summary
        del drop_short, drop_long
        eval_structure = base.structure

        @jax.jit
        def loss_fn(params, batch, key):
            state = compute_forward_pass(
                params, eval_structure, batch, key, training=False
            )
            return compute_loss(
                state,
                batch["y"],
                eval_structure.task_map["y"],
                loss_type="cross_entropy",
                loss_mask=batch["loss_mask"],
            )

        @jax.jit
        def stepped_loss(params, gradient, eta, batch, key):
            updated = jax.tree_util.tree_map(lambda p, g: p - eta * g, params, gradient)
            return loss_fn(updated, batch, key)

        # A norm-relative center is only a search scale, not a tuned PC learning rate.
        center = 1e-3 * _norm(base.params) / max(_norm(bp_grads[0]), 1e-12)
        lr_candidates = [float(center * m) for m in (0.01, 0.1, 1, 10, 100)]
        initial_loss = [
            float(loss_fn(base.params, b, k)) for b, k in zip(batches, keys)
        ]
        eta_ref = None
        for attempt in range(2):
            trials = []
            for eta in lr_candidates:
                check_budget()
                losses = [
                    float(
                        stepped_loss(base.params, bp_grads[i], eta, batches[i], keys[i])
                    )
                    for i in range(2)
                ]
                emit(
                    "bp_lr_calibration",
                    eta=eta,
                    losses=[v if math.isfinite(v) else None for v in losses],
                )
                if all(
                    math.isfinite(v) and v < initial_loss[i]
                    for i, v in enumerate(losses)
                ):
                    trials.append((sum(losses), eta))
            if trials:
                eta_ref = min(trials)[1]
                break
            lr_candidates = [v * 0.01 for v in lr_candidates]
        summary["eta_ref"] = eta_ref
        if eta_ref is None:
            summary.update(
                status="bp_calibration_failed",
                reason="no finite loss-reducing BP reference",
            )
            return summary
        identified = {str(beta): 0 for beta in BETAS if 0 < beta < 1}
        convergence = []
        for scale in config.scales:
            short = make_pc_trace(pc_structure(rate, steps, scale), steps)
            long = make_pc_trace(pc_structure(rate, steps * 2, scale), steps * 2)
            for i in range(config.num_batches):
                gpc, trace, stats = assess(short, long, i, scale)
                convergence.append({"scale": scale, "batch": i, **stats})
                emit("convergence", scale=scale, batch=i, **stats)
                np.save(output / f"trace_{scale}_{i}.npy", trace)
                bp = bp_grads[i]
                candidates = {"bp": bp}
                if scale == 1:
                    for beta in BETAS:
                        blend = blend_gradients(spike_grads[i], gpc, beta)
                        name = f"anchor_{beta}"
                        candidates[name] = blend
                        metrics = direction_metrics(bp, blend)
                        emit(
                            "direction",
                            batch=i,
                            beta=beta,
                            global_metrics=metrics,
                            modules=module_metrics(bp, blend),
                        )
                        if 0 < beta < 1:
                            if (metrics["perpendicular_norm"] or 0) > 10 * max(
                                anchor_error[i], 1e-12
                            ):
                                identified[str(beta)] += 1
                            matched = match_norm(bp, blend)
                            if matched is not None:
                                candidates[f"norm_matched_{beta}"] = matched
                            module_matches = {
                                n: match_norm(bp.nodes[n], blend.nodes[n])
                                for n in bp.nodes
                            }
                            if all(
                                v is not None
                                or not jax.tree_util.tree_leaves(bp.nodes[n])
                                for n, v in module_matches.items()
                            ):
                                candidates[f"module_matched_{beta}"] = bp._replace(
                                    nodes={
                                        n: (v if v is not None else bp.nodes[n])
                                        for n, v in module_matches.items()
                                    }
                                )
                else:
                    candidates[f"pc_nudge_{scale}"] = gpc
                baseline_valid = float(loss_fn(base.params, valid[i], keys[i]))
                for name, gradient in candidates.items():
                    for multiplier in (0.3, 1, 3, 10):
                        check_budget()
                        eta = eta_ref * multiplier
                        train_loss = float(
                            stepped_loss(
                                base.params, gradient, eta, batches[i], keys[i]
                            )
                        )
                        valid_loss = float(
                            stepped_loss(base.params, gradient, eta, valid[i], keys[i])
                        )
                        emit(
                            "one_step",
                            batch=i,
                            method=name,
                            eta=eta,
                            train_before=initial_loss[i],
                            valid_before=baseline_valid,
                            train_after=(
                                train_loss if math.isfinite(train_loss) else None
                            ),
                            valid_after=(
                                valid_loss if math.isfinite(valid_loss) else None
                            ),
                            finite=math.isfinite(train_loss)
                            and math.isfinite(valid_loss),
                        )
            del short, long
            jax.clear_caches()
        summary["convergence"] = convergence
        summary["identified_batches"] = identified
        required = max(1, math.ceil(0.75 * config.num_batches))
        summary["required_identified_batches"] = required
        summary["final_params_sha256"] = params_hash(base.params)
        if summary["final_params_sha256"] != summary["initial_params_sha256"]:
            raise RuntimeError("diagnostic mutated source parameters")
        full = config.num_batches >= 8
        go = (
            full
            and all(r["converged"] for r in convergence)
            and any(count >= required for count in identified.values())
        )
        summary.update(
            status="complete",
            decision="go" if go else "no_go",
            reason=(
                "all diagnostic gates passed"
                if go
                else "insufficient batches, convergence, or identifiable directional correction"
            ),
        )
        if not full:
            summary["limitations"].append(
                "Reduced-batch smoke run cannot authorize training."
            )
        return summary
    except BudgetExceeded as error:
        summary.update(status="budget_exhausted", reason=str(error))
        return summary
    except Exception as error:
        summary.update(status="error", reason=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()
        emit("finished", status=summary["status"], decision=summary["decision"])


def render_report(output: Path) -> Path:
    """Render completed or partial diagnostic evidence without rerunning a solve."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = json.loads((output / "summary.json").read_text())
    events_path = output / "events.jsonl"
    events = (
        [json.loads(line) for line in events_path.read_text().splitlines()]
        if events_path.exists()
        else []
    )
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for path in sorted(output.glob("calibration_*.npy")):
        trace = np.load(path, allow_pickle=False)
        axes[0, 0].plot(trace[:, 0], label=path.stem)
        axes[0, 1].semilogy(
            np.maximum(trace[:, 1] / max(float(trace[0, 1]), 1e-12), 1e-15)
        )
    axes[0, 0].set(
        xlabel="Inference step", ylabel="Total energy", title="Calibration solves"
    )
    axes[0, 1].axhline(1e-3, color="black", linestyle="--")
    axes[0, 1].set(
        xlabel="Inference step",
        ylabel="Residual / initial residual",
        title="Unclipped convergence residual",
    )
    directions = [e for e in events if e["kind"] == "direction"]
    for beta in BETAS:
        rows = [e for e in directions if e["beta"] == beta]
        axes[1, 0].plot(
            [e["batch"] for e in rows],
            [e["global_metrics"]["perpendicular_norm"] for e in rows],
            marker="o",
            label=str(beta),
        )
    axes[1, 0].set(
        xlabel="Training probe batch",
        ylabel="Perpendicular correction norm",
        title="Direction beyond BP scaling",
    )
    rows = [e for e in events if e["kind"] == "one_step" and e["finite"]]
    methods = sorted({e["method"] for e in rows})
    for index, method in enumerate(methods):
        values = [
            e["valid_after"] - e["valid_before"] for e in rows if e["method"] == method
        ]
        axes[1, 1].scatter([index] * len(values), values, s=10, alpha=0.5)
    axes[1, 1].axhline(0, color="black", linestyle="--")
    axes[1, 1].set(
        xticks=list(range(len(methods))),
        xticklabels=methods,
        ylabel="Validation CE change",
        title="One-step probes across learning rates",
    )
    axes[1, 1].tick_params(axis="x", rotation=60, labelsize=7)
    figure.suptitle(f"Anchored PC: {summary['status']} / {summary['decision']}")
    figure.tight_layout()
    path = output / "diagnostic.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--budget-seconds", type=float, default=7200)
    args = parser.parse_args()
    config = DiagnosticConfig(
        **vars(args),
        expected_shape={
            "depth": 4,
            "embed_dim": 256,
            "num_heads": 4,
            "mlp_dim": 1024,
            "seq_len": 128,
            "vocab_size": 30522,
        },
    )
    result = run_diagnostic(config)
    render_report(Path(config.output_dir))
    print(
        json.dumps({k: result[k] for k in ("status", "decision", "reason")}, indent=2)
    )


if __name__ == "__main__":
    main()
