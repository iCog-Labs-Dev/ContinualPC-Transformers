"""Standalone tanh-MLP anchored-PC mechanism pilot on checksum-verified MNIST.

Inference minimizes per-example quadratic hidden errors plus scaled logit CE.
Local weight gradients average over examples. Batch size therefore does not
implicitly change the strength of the output clamp. This is a new small-MLP
experiment, not a modification or replication of the BERT energy convention.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import platform
import shutil
import struct
import tempfile
import time
from functools import partial
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import jax
import jax.numpy as jnp
import numpy as np

METHODS = (
    "bp",
    "pc",
    "anchor_0.1",
    "anchor_0.5",
    "nudge_0.1",
    "nudge_0.01",
    "matched_0.1",
    "matched_0.5",
)
RESIDUAL_LIMIT = 1e-3
GRADIENT_LIMIT = 0.01
RESOURCES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}
Params = tuple[tuple[jax.Array, jax.Array], ...]


def rate_sweep_schedule() -> list[tuple[str, float, int]]:
    """All fixed rate/seed cells; no selection of a winning learning rate."""
    methods = [m for m in METHODS if not m.startswith("nudge_")]
    return [
        (m, lr, seed) for seed in (1, 2, 3) for lr in (0.3, 1.0, 3.0) for m in methods
    ]


def gradient_failure(stats: np.ndarray, passed: bool) -> str | None:
    if passed:
        return None
    if not np.isfinite(stats[:2]).all():
        return "nonfinite_inference"
    if stats[0] > RESIDUAL_LIMIT or stats[1] > GRADIENT_LIMIT:
        return "inference_failed"
    return "nonfinite_gradient"


def download_verified(url: str, destination: Path, md5: str) -> None:
    """Publish a download only after checksum verification; preserve existing data."""
    if destination.exists():
        if hashlib.md5(destination.read_bytes()).hexdigest() != md5:
            raise ValueError(f"MNIST checksum mismatch: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=destination.name + ".",
            suffix=".part",
            delete=False,
        ) as target:
            temporary = Path(target.name)
            with urlopen(url, timeout=60) as response:
                shutil.copyfileobj(response, target)
        if hashlib.md5(temporary.read_bytes()).hexdigest() != md5:
            raise ValueError(f"MNIST download checksum mismatch: {destination}")
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def init_params(seed: int, widths: tuple[int, ...] = (784, 128, 64, 10)) -> Params:
    keys = jax.random.split(jax.random.PRNGKey(seed), len(widths) - 1)
    return tuple(
        (
            jax.random.normal(k, (a, b), dtype=jnp.float64) * math.sqrt(2 / (a + b)),
            jnp.zeros(b, dtype=jnp.float64),
        )
        for k, a, b in zip(keys, widths[:-1], widths[1:])
    )


def forward(p: Params, x: jax.Array):
    h1 = jnp.tanh(x @ p[0][0] + p[0][1])
    h2 = jnp.tanh(h1 @ p[1][0] + p[1][1])
    return h1, h2, h2 @ p[2][0] + p[2][1]


def cross_entropy(logits, y):
    return -jnp.take_along_axis(jax.nn.log_softmax(logits), y[:, None], axis=1)[:, 0]


def loss(p: Params, x: jax.Array, y: jax.Array):
    return cross_entropy(forward(p, x)[2], y).mean()


def energy(p: Params, x: jax.Array, y: jax.Array, z, scale: float):
    z1, z2 = z
    f1 = jnp.tanh(x @ p[0][0] + p[0][1])
    f2 = jnp.tanh(z1 @ p[1][0] + p[1][1])
    logits = z2 @ p[2][0] + p[2][1]
    return jnp.mean(
        0.5 * jnp.sum((z1 - f1) ** 2, axis=1)
        + 0.5 * jnp.sum((z2 - f2) ** 2, axis=1)
        + scale * cross_entropy(logits, y)
    )


def latent_gradient(p, x, y, z, scale):
    """Gradient of the sum of per-example energies, before any clipping."""
    z1, z2 = z
    f1 = jnp.tanh(x @ p[0][0] + p[0][1])
    f2 = jnp.tanh(z1 @ p[1][0] + p[1][1])
    delta = jax.nn.softmax(z2 @ p[2][0] + p[2][1]) - jax.nn.one_hot(y, p[2][1].size)
    return (
        z1 - f1 - ((z2 - f2) * (1 - f2**2)) @ p[1][0].T,
        z2 - f2 + scale * delta @ p[2][0].T,
    )


def local_gradients(p, x, y, z, scale):
    z1, z2 = z
    f1 = jnp.tanh(x @ p[0][0] + p[0][1])
    f2 = jnp.tanh(z1 @ p[1][0] + p[1][1])
    d1, d2 = (f1 - z1) * (1 - f1**2), (f2 - z2) * (1 - f2**2)
    d3 = scale * (
        jax.nn.softmax(z2 @ p[2][0] + p[2][1]) - jax.nn.one_hot(y, p[2][1].size)
    )
    n = len(x)
    return (
        (x.T @ d1 / n, d1.mean(0)),
        (z1.T @ d2 / n, d2.mean(0)),
        (z2.T @ d3 / n, d3.mean(0)),
    )


def spike_gradient(p, x, y):
    """Two scheduled unit latent spikes, with initial predictions/inputs frozen.

    First the final hidden node spikes, then its predecessor. No BP oracle
    is called here; local errors produce the final parameter gradients.
    """
    h1, h2, logits = forward(p, x)
    d3 = jax.nn.softmax(logits) - jax.nn.one_hot(y, logits.shape[-1])
    z2 = h2 - d3 @ p[2][0].T
    z1 = h1 + ((z2 - h2) * (1 - h2**2)) @ p[1][0].T
    d1, d2 = (h1 - z1) * (1 - h1**2), (h2 - z2) * (1 - h2**2)
    n = len(x)
    return (
        (x.T @ d1 / n, d1.mean(0)),
        (h1.T @ d2 / n, d2.mean(0)),
        (h2.T @ d3 / n, d3.mean(0)),
    )


def norm(tree):
    return jnp.sqrt(sum(jnp.vdot(a, a).real for a in jax.tree_util.tree_leaves(tree)))


@partial(jax.jit, static_argnums=(5,))
def _relax(p, x, y, z, rate, steps, scale):
    def stats(state):
        return jnp.stack(
            [
                energy(p, x, y, state, scale),
                norm(latent_gradient(p, x, y, state, scale)),
            ]
        )

    def step(state, _):
        g = latent_gradient(p, x, y, state, scale)
        after = jax.tree_util.tree_map(lambda a, b: a - rate * b, state, g)
        return after, stats(after)

    initial = stats(z)
    final, trace = jax.lax.scan(step, z, None, length=steps)
    return final, jnp.concatenate([initial[None, :], trace])


def infer(p, x, y, rate, steps, scale):
    h1, h2, _ = forward(p, x)
    return _relax(p, x, y, (h1, h2), rate, steps, scale)


def pc_pair(p, x, y, solver, scale):
    rate, steps = solver
    z1, t1 = infer(p, x, y, rate, steps, scale)
    z2, t2 = _relax(p, x, y, z1, rate, steps, scale)
    g1 = jax.tree_util.tree_map(
        lambda g: g / scale, local_gradients(p, x, y, z1, scale)
    )
    g2 = jax.tree_util.tree_map(
        lambda g: g / scale, local_gradients(p, x, y, z2, scale)
    )
    change = norm(jax.tree_util.tree_map(lambda a, b: a - b, g1, g2)) / jnp.maximum(
        norm(g2), 1e-30
    )
    ratio = t2[-1, 1] / jnp.maximum(t1[0, 1], 1e-30)
    return g2, jnp.stack([ratio, change]), jnp.concatenate([t1, t2[1:]])


def gradient(p, x, y, method, solver, clip=2.0):
    diagnostic = jnp.zeros(2)
    if method == "bp":
        g = jax.grad(loss)(p, x, y)
    elif method == "spike":
        g = spike_gradient(p, x, y)
    elif method == "pc" or method.startswith(("anchor_", "matched_", "nudge_")):
        scale = float(method.split("_")[1]) if method.startswith("nudge_") else 1.0
        g, diagnostic, _ = pc_pair(p, x, y, solver, scale)
        if method.startswith(("anchor_", "matched_")):
            beta = float(method.split("_")[1])
            spike = spike_gradient(p, x, y)
            g = jax.tree_util.tree_map(lambda a, b: (1 - beta) * a + beta * b, spike, g)
    else:
        raise ValueError(f"Unknown method: {method}")
    size = norm(g)
    g = jax.tree_util.tree_map(
        lambda a: a * jnp.minimum(1.0, clip / jnp.maximum(size, 1e-30)), g
    )
    if method.startswith("matched_"):
        bp = jax.grad(loss)(p, x, y)
        factor = norm(g) / jnp.maximum(norm(bp), 1e-30)
        g = jax.tree_util.tree_map(lambda a: a * factor, bp)
    return g, jnp.concatenate([diagnostic, jnp.asarray([size > clip], dtype=x.dtype)])


def split_indices(total: int, train_count: int, valid_count: int, seed: int):
    if min(train_count, valid_count) < 1 or train_count + valid_count > total:
        raise ValueError(
            "Training and validation counts must fit in the training split"
        )
    order = np.random.default_rng(seed).permutation(total)
    return order[valid_count : valid_count + train_count], order[:valid_count]


def read_idx(path: Path):
    payload = gzip.decompress(path.read_bytes())
    magic = struct.unpack(">I", payload[:4])[0]
    if magic == 2051:
        n, h, w = struct.unpack(">III", payload[4:16])
        if (h, w) != (28, 28) or len(payload) != 16 + n * h * w:
            raise ValueError("Invalid MNIST image dimensions or payload")
        return np.frombuffer(payload, np.uint8, offset=16).reshape(n, h * w)
    if magic == 2049:
        n = struct.unpack(">I", payload[4:8])[0]
        if len(payload) != n + 8:
            raise ValueError("Invalid MNIST label payload")
        return np.frombuffer(payload, np.uint8, offset=8)
    raise ValueError("Invalid IDX magic")


def load_data(raw: Path, output: Path):
    provenance = {}
    for name, md5 in RESOURCES.items():
        path = raw / name
        if hashlib.md5(path.read_bytes()).hexdigest() != md5:
            raise ValueError(f"MNIST checksum mismatch: {path}")
        provenance[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    images = read_idx(raw / "train-images-idx3-ubyte.gz")
    labels = read_idx(raw / "train-labels-idx1-ubyte.gz")
    if images.shape != (60000, 784) or labels.shape != (60000,):
        raise ValueError("Expected the standard 60,000-image training split")
    train_ids, valid_ids = split_indices(60000, 10000, 5000, 20261031)
    np.savez_compressed(output / "split_indices.npz", train=train_ids, valid=valid_ids)
    return (
        (
            jnp.asarray(images[train_ids], dtype=jnp.float64) / 255,
            jnp.asarray(labels[train_ids], dtype=jnp.int32),
        ),
        (
            jnp.asarray(images[valid_ids], dtype=jnp.float64) / 255,
            jnp.asarray(labels[valid_ids], dtype=jnp.int32),
        ),
        provenance,
    )


@jax.jit
def evaluate(p, x, y):
    logits = forward(p, x)[2]
    return jnp.stack(
        [cross_entropy(logits, y).mean(), (jnp.argmax(logits, -1) == y).mean()]
    )


@partial(jax.jit, static_argnames=("method", "solver"))
def train_step(p, x, y, lr, *, method, solver):
    g, diagnostic = gradient(p, x, y, method, solver)
    finite = jnp.isfinite(norm(g)) & jnp.all(jnp.isfinite(diagnostic))
    passed = (
        finite & (diagnostic[0] <= RESIDUAL_LIMIT) & (diagnostic[1] <= GRADIENT_LIMIT)
    )
    updated = jax.tree_util.tree_map(
        lambda a, b: jnp.where(passed, a - lr * b, a), p, g
    )
    return updated, diagnostic, passed


class BudgetExceeded(RuntimeError):
    pass


def run(
    raw: Path,
    output: Path,
    *,
    steps: int = 200,
    tune_steps: int = 50,
    budget_seconds: float = 1800,
    protocol: str = "pilot",
):
    if (
        min(steps, tune_steps) < 1
        or not math.isfinite(budget_seconds)
        or budget_seconds <= 0
    ):
        raise ValueError("Positive update counts and finite positive budget required")
    if protocol not in {"pilot", "rate-sweep"}:
        raise ValueError("Unknown experiment protocol")
    jax.config.update("jax_enable_x64", True)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report: dict[str, Any] = {
        "status": "running",
        "protocol": protocol,
        "training_started": False,
        "model": [784, 128, 64, 10],
        "activation": "tanh",
        "dtype": "float64",
        "optimizer": "SGD",
        "clip": 2.0,
        "weight_decay": 0.0,
        "train_images": 10000,
        "validation_images": 5000,
        "seeds": [1, 2, 3],
        "tuning_seed": 0,
        "steps": steps,
        "tune_steps": tune_steps,
        "lr_grid": [0.03, 0.1, 0.3] if protocol == "pilot" else [0.3, 1.0, 3.0],
        "residual_limit": RESIDUAL_LIMIT,
        "gradient_change_limit": GRADIENT_LIMIT,
        "budget_seconds": budget_seconds,
        "devices": [str(d) for d in jax.devices()],
        "jax": jax.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "device_kinds": [d.device_kind for d in jax.devices()],
        "backend": jax.default_backend(),
        "test_used_for_tuning": False,
        "claim_scope": "Short small-MLP mechanism pilot, not transformer evidence",
    }

    def save():
        report["elapsed_seconds"] = time.monotonic() - started
        (output / "summary.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )

    def emit(kind, **record):
        record.update(kind=kind, elapsed_seconds=time.monotonic() - started)
        with (output / "events.jsonl").open("a") as f:
            f.write(json.dumps(record, allow_nan=False) + "\n")
        print(
            f"{record['elapsed_seconds']:.1f}s {kind} "
            + json.dumps(
                {
                    k: v
                    for k, v in record.items()
                    if k not in {"kind", "elapsed_seconds"}
                }
            ),
            flush=True,
        )

    def budget():
        if time.monotonic() - started >= budget_seconds:
            raise BudgetExceeded("Pilot time budget reached")

    save()
    try:
        shutil.copy2(Path(__file__), output / "source.py")
        train, valid, provenance = load_data(raw, output)
        report["data_sha256"] = provenance
        report["source_sha256"] = hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest()
        streams = {
            seed: np.random.default_rng(1000 + seed).integers(
                0, len(train[0]), size=(max(steps, tune_steps, 20), 64)
            )
            for seed in (0, 1, 2, 3)
        }
        stored_streams: dict[str, Any] = {f"seed_{s}": v for s, v in streams.items()}
        np.savez_compressed(output / "batch_indices.npz", **stored_streams)
        report["solver_selection_margin"] = {"residual": 1e-4, "gradient_change": 0.001}
        p0 = init_params(0)
        # Gate at initialization and after an ordinary 20-update BP trajectory.
        warm = p0
        for ids in streams[0][:20]:
            warm, _, _ = train_step(
                warm, train[0][ids], train[1][ids], 0.1, method="bp", solver=(0.1, 120)
            )
        anchor_checks = []
        for state_name, p in (("initial", p0), ("bp_20", warm)):
            for i, ids in enumerate(streams[0][:2]):
                x, y = train[0][ids], train[1][ids]
                bp, spike = jax.grad(loss)(p, x, y), spike_gradient(p, x, y)
                for layer, (a, b) in enumerate(zip(bp, spike)):
                    error = float(
                        norm(jax.tree_util.tree_map(lambda u, v: u - v, a, b)) / norm(a)
                    )
                    anchor_checks.append(error)
                    emit(
                        "anchor",
                        state=state_name,
                        batch=i,
                        layer=layer,
                        relative_error=error,
                    )
        report["max_anchor_relative_error"] = max(anchor_checks)
        if max(anchor_checks) > 1e-8:
            report["status"] = "anchor_failed"
            return report
        solver = None
        calibration = []
        for base_steps in (60, 120, 240):
            for rate in (0.2, 0.1, 0.05):
                budget()
                checks = []
                for state_name, p in (("initial", p0), ("bp_20", warm)):
                    for i, ids in enumerate(streams[0][:2]):
                        for scale in (1.0, 0.1, 0.01):
                            _, stats, trace = pc_pair(
                                p,
                                train[0][ids],
                                train[1][ids],
                                (rate, base_steps),
                                scale,
                            )
                            stats = np.asarray(stats)
                            passed = bool(
                                np.isfinite(stats).all()
                                and stats[0] <= 1e-4
                                and stats[1] <= 0.001
                            )
                            checks.append(passed)
                            row = {
                                "state": state_name,
                                "batch": i,
                                "rate": rate,
                                "T": base_steps,
                                "scale": scale,
                                "relative_residual": (
                                    float(stats[0]) if np.isfinite(stats[0]) else None
                                ),
                                "gradient_change": (
                                    float(stats[1]) if np.isfinite(stats[1]) else None
                                ),
                                "gate_passed": bool(
                                    np.isfinite(stats).all()
                                    and stats[0] <= RESIDUAL_LIMIT
                                    and stats[1] <= GRADIENT_LIMIT
                                ),
                                "selection_margin_passed": passed,
                            }
                            calibration.append(row)
                            np.save(
                                output
                                / f"trace_{state_name}_{i}_{rate}_{base_steps}_{scale}.npy",
                                trace,
                            )
                emit(
                    "calibration",
                    rate=rate,
                    T=base_steps,
                    passed=sum(checks),
                    total=len(checks),
                )
                if all(checks):
                    solver = (rate, base_steps)
                    break
            if solver:
                break
        report["calibration"] = calibration
        report["solver"] = solver
        save()
        if solver is None:
            report["status"] = "solver_selection_failed"
            report["reason"] = (
                "No common solver passed the conservative calibration margins across all scales and states"
            )
            return report

        def trial(method, lr, seed, updates, phase):
            p = init_params(seed)
            before = np.asarray(evaluate(p, *valid))
            times, clips = [], []
            status, completed = "complete", 0
            worst = [0.0, 0.0]
            failure_diagnostics = None
            for index, ids in enumerate(streams[seed][:updates]):
                budget()
                begin = time.monotonic()
                p, stats, passed = train_step(
                    p, train[0][ids], train[1][ids], lr, method=method, solver=solver
                )
                values = np.asarray(stats)
                times.append(time.monotonic() - begin)
                if not bool(passed):
                    status = gradient_failure(values, False)
                    failure_diagnostics = [
                        float(v) if np.isfinite(v) else None for v in values
                    ]
                    break
                completed += 1
                worst = [max(worst[j], float(values[j])) for j in (0, 1)]
                clips.append(float(values[2]))
            after = np.asarray(evaluate(p, *valid))
            if not np.isfinite(after).all() and status == "complete":
                status = "nonfinite_validation"
            row = {
                "method": method,
                "lr": lr,
                "seed": seed,
                "phase": phase,
                "status": status,
                "completed_updates": completed,
                "validation_loss": float(after[0]) if np.isfinite(after[0]) else None,
                "validation_accuracy": (
                    float(after[1]) if np.isfinite(after[1]) else None
                ),
                "initial_validation_loss": float(before[0]),
                "max_relative_residual": worst[0],
                "max_gradient_change": worst[1],
                "median_update_seconds": float(np.median(times[1:] or times)),
                "update_seconds_including_compile": sum(times),
                "clip_fraction": float(np.mean(clips)) if clips else None,
                "failure_diagnostics": failure_diagnostics,
            }
            emit("trial", **row)
            if phase in {"paired_pilot", "rate_sweep"}:
                tensors = {
                    f"layer_{i}_{name}": np.asarray(value)
                    for i, layer in enumerate(p)
                    for name, value in zip(("W", "b"), layer)
                }
                suffix = f"_{lr}" if phase == "rate_sweep" else ""
                np.savez_compressed(
                    output / f"params_{method}_{seed}{suffix}.npz", **tensors
                )
            return row

        report["training_started"] = True
        if protocol == "rate-sweep":
            schedule = rate_sweep_schedule()
            report.update(
                lr_grid=[0.3, 1.0, 3.0],
                tuning=[],
                tune_steps=0,
                planned_trials=len(schedule),
                results=[],
                test_evaluated=False,
                selection_rule="No tuning; report every fixed rate/seed cell, including failures",
            )
            for method, lr, seed in schedule:
                report["results"].append(trial(method, lr, seed, steps, "rate_sweep"))
                save()
            report["all_methods_completed"] = all(
                r["status"] == "complete" for r in report["results"]
            )
            report["status"] = "complete"
            report["limitations"] = [
                "Short fixed-rate validation stress test, not a held-out test or retention result.",
                "Inference failures are distinct from numerical training failures; finite poor accuracy is not classified as divergence.",
            ]
            return report
        tuning, selected = [], {}
        for method in METHODS:
            for lr in report["lr_grid"]:
                tuning.append(trial(method, lr, 0, tune_steps, "tuning"))
            candidates = [
                r for r in tuning if r["method"] == method and r["status"] == "complete"
            ]
            if candidates:
                selected[method] = min(candidates, key=lambda r: r["validation_loss"])[
                    "lr"
                ]
            report["tuning"], report["selected_lrs"] = tuning, selected
            save()
        results = []
        for seed in (1, 2, 3):
            for method, lr in selected.items():
                results.append(trial(method, lr, seed, steps, "paired_pilot"))
                report["results"] = results
                save()
        report["status"] = "complete"
        report["all_methods_completed"] = len(results) == len(METHODS) * 3 and all(
            r["status"] == "complete" for r in results
        )
        report["test_evaluated"] = False
        report["limitations"] = [
            "Three-seed short validation pilot; no held-out test result or retention claim.",
            "Float64, plain SGD and a tanh MLP; not BERT evidence.",
        ]
        return report
    except BudgetExceeded as error:
        report.update(status="budget_exhausted", reason=str(error))
        return report
    except Exception as error:
        report.update(status="error", reason=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--tune-steps", type=int, default=50)
    parser.add_argument("--budget-seconds", type=float, default=1800)
    parser.add_argument("--protocol", choices=("pilot", "rate-sweep"), default="pilot")
    parser.add_argument(
        "--require-gpu", action="store_true", help="Refuse a CPU fallback"
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download and checksum-verify public MNIST files",
    )
    args = parser.parse_args()
    if args.require_gpu and jax.default_backend() != "gpu":
        parser.error(
            "GPU required: JAX selected a non-GPU backend; check the GPU JAX installation"
        )
    if args.download:
        for name, digest in RESOURCES.items():
            download_verified(
                "https://ossci-datasets.s3.amazonaws.com/mnist/" + name,
                args.data_dir / name,
                digest,
            )
    result = run(
        args.data_dir,
        args.output_dir,
        steps=args.steps,
        tune_steps=args.tune_steps,
        budget_seconds=args.budget_seconds,
        protocol=args.protocol,
    )
    print(
        json.dumps({k: result.get(k) for k in ("status", "solver", "elapsed_seconds")}),
        flush=True,
    )
    if result["status"] != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
