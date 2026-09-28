#!/usr/bin/env python3
"""Benchmark synchronized BP, PC-FP, and feed-forward evaluation runtime.

The benchmark uses the locked character-level Transformer configuration and
the same pmap training kernels as the main experiment.  Every timed call is
explicitly synchronized.  PC-FP is measured at T=0 and T=20 so that the
incremental cost per inference sweep can be estimated without attributing
feed-forward, gradient-readout, optimizer, or collective overhead to settling.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd

from fabricpc.core.fixed_prediction import InferenceSGDFixedPrediction
from fabricpc.core.inference import InferenceSGDNormClip
from fabricpc.graph_initialization import initialize_params
from fabricpc.nodes.transformer_v2 import create_deep_transformer
from fabricpc.training.train import (
    replicate_opt_state,
    replicate_params,
    shard_batch,
)
from fabricpc.training.train_autoregressive import (
    create_pmap_eval_step_autoregressive,
    create_pmap_train_step_autoregressive,
)
from fabricpc.training.train_backprop import (
    create_pmap_train_step_backprop_autoregressive,
)
from fabricpc.utils.data import CharDataLoader


LOCKED_CONFIG = {
    "depth": 2,
    "embed_dim": 128,
    "num_heads": 8,
    "mlp_dim": 512,
    "seq_len": 64,
    "global_batch_size": 32,
    "epochs": 5,
    "lr": 4.02e-5,
    "weight_init_std": 0.0228,
    "dropout_rate": 0.125,
    "eta_infer": 0.3864,
    "infer_steps": 20,
    "max_norm": 5.0,
}


def convert_batch(batch_data) -> dict[str, jax.Array]:
    if isinstance(batch_data, dict):
        return {name: jnp.asarray(value) for name, value in batch_data.items()}
    if isinstance(batch_data, (tuple, list)) and len(batch_data) >= 2:
        return {"x": jnp.asarray(batch_data[0]), "y": jnp.asarray(batch_data[1])}
    raise TypeError(f"unsupported batch type: {type(batch_data)}")


def build_structure(vocab_size: int, infer_steps: int):
    if infer_steps < 0:
        inference = InferenceSGDNormClip(
            eta_infer=LOCKED_CONFIG["eta_infer"],
            infer_steps=0,
            latent_decay=0.0,
            max_norm=LOCKED_CONFIG["max_norm"],
        )
    else:
        inference = InferenceSGDFixedPrediction(
            eta_infer=LOCKED_CONFIG["eta_infer"],
            infer_steps=infer_steps,
            latent_decay=0.0,
            max_norm=LOCKED_CONFIG["max_norm"],
        )
    return create_deep_transformer(
        depth=LOCKED_CONFIG["depth"],
        embed_dim=LOCKED_CONFIG["embed_dim"],
        num_heads=LOCKED_CONFIG["num_heads"],
        mlp_dim=LOCKED_CONFIG["mlp_dim"],
        seq_len=LOCKED_CONFIG["seq_len"],
        vocab_size=vocab_size,
        inference=inference,
        weight_init={"type": "normal", "std": LOCKED_CONFIG["weight_init_std"]},
        dropout_rate=LOCKED_CONFIG["dropout_rate"],
    )


def summarize(milliseconds: list[float]) -> dict[str, float]:
    values = np.asarray(milliseconds, dtype=np.float64)
    return {
        "mean_ms": float(np.mean(values)),
        "std_ms": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "median_ms": float(np.median(values)),
        "p10_ms": float(np.percentile(values, 10)),
        "p90_ms": float(np.percentile(values, 90)),
        "minimum_ms": float(np.min(values)),
        "maximum_ms": float(np.max(values)),
    }


def benchmark_training_kernel(
    name: str,
    step_fn,
    params,
    optimizer,
    sharded_batch,
    *,
    n_devices: int,
    seed: int,
    warmup_steps: int,
    timed_steps: int,
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    local_params = replicate_params(params, n_devices)
    local_opt_state = replicate_opt_state(optimizer.init(params), n_devices)
    base_key = jax.random.PRNGKey(seed)

    def keys(index: int):
        return jax.random.split(jax.random.fold_in(base_key, index), n_devices)

    start = time.perf_counter()
    output = step_fn(local_params, local_opt_state, sharded_batch, keys(0))
    jax.block_until_ready(output)
    first_call_seconds = time.perf_counter() - start
    local_params, local_opt_state = output[0], output[1]

    for index in range(1, warmup_steps + 1):
        output = step_fn(local_params, local_opt_state, sharded_batch, keys(index))
        jax.block_until_ready(output)
        local_params, local_opt_state = output[0], output[1]

    rows = []
    offset = warmup_steps + 1
    for repeat in range(timed_steps):
        start = time.perf_counter()
        output = step_fn(
            local_params,
            local_opt_state,
            sharded_batch,
            keys(offset + repeat),
        )
        jax.block_until_ready(output)
        elapsed_ms = 1000.0 * (time.perf_counter() - start)
        local_params, local_opt_state = output[0], output[1]
        rows.append({"kernel": name, "repeat": repeat, "elapsed_ms": elapsed_ms})

    summary = summarize([row["elapsed_ms"] for row in rows])
    median_seconds = summary["median_ms"] / 1000.0
    return (
        {
            "kernel": name,
            "first_call_seconds": first_call_seconds,
            "warmup_steps": warmup_steps,
            "timed_steps": timed_steps,
            **summary,
            "global_examples_per_second": (
                LOCKED_CONFIG["global_batch_size"] / median_seconds
            ),
            "global_tokens_per_second": (
                LOCKED_CONFIG["global_batch_size"]
                * LOCKED_CONFIG["seq_len"]
                / median_seconds
            ),
        },
        pd.DataFrame(rows),
    )


def benchmark_eval_kernel(
    step_fn,
    replicated_params,
    sharded_batch,
    *,
    n_devices: int,
    seed: int,
    warmup_steps: int,
    timed_steps: int,
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    base_key = jax.random.PRNGKey(seed + 10_000)

    def keys(index: int):
        return jax.random.split(jax.random.fold_in(base_key, index), n_devices)

    start = time.perf_counter()
    output = step_fn(replicated_params, sharded_batch, keys(0))
    jax.block_until_ready(output)
    first_call_seconds = time.perf_counter() - start
    for index in range(1, warmup_steps + 1):
        output = step_fn(replicated_params, sharded_batch, keys(index))
        jax.block_until_ready(output)

    rows = []
    offset = warmup_steps + 1
    for repeat in range(timed_steps):
        start = time.perf_counter()
        output = step_fn(replicated_params, sharded_batch, keys(offset + repeat))
        jax.block_until_ready(output)
        elapsed_ms = 1000.0 * (time.perf_counter() - start)
        rows.append(
            {"kernel": "feedforward_eval", "repeat": repeat, "elapsed_ms": elapsed_ms}
        )
    summary = summarize([row["elapsed_ms"] for row in rows])
    median_seconds = summary["median_ms"] / 1000.0
    return (
        {
            "kernel": "feedforward_eval",
            "first_call_seconds": first_call_seconds,
            "warmup_steps": warmup_steps,
            "timed_steps": timed_steps,
            **summary,
            "global_examples_per_second": (
                LOCKED_CONFIG["global_batch_size"] / median_seconds
            ),
            "global_tokens_per_second": (
                LOCKED_CONFIG["global_batch_size"]
                * LOCKED_CONFIG["seq_len"]
                / median_seconds
            ),
        },
        pd.DataFrame(rows),
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--timed-steps", type=int, default=30)
    parser.add_argument(
        "--kernels",
        nargs="+",
        choices=(
            "backprop",
            "pc_fp_T0",
            "pc_fp_T20",
            "feedforward_eval",
        ),
        default=(
            "backprop",
            "pc_fp_T0",
            "pc_fp_T20",
            "feedforward_eval",
        ),
        help=(
            "Kernels to benchmark. Select one kernel per fresh Python "
            "process when measuring first-call compilation latency."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/runtime_benchmark/locked_char_pmap"),
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    if args.warmup_steps < 1 or args.timed_steps < 2:
        raise ValueError("use at least one warmup and two timed steps")

    n_devices = jax.local_device_count()
    if LOCKED_CONFIG["global_batch_size"] % n_devices:
        raise ValueError(
            "global batch size must be divisible by local device count"
        )

    loader = CharDataLoader(
        "train",
        seq_len=LOCKED_CONFIG["seq_len"],
        batch_size=LOCKED_CONFIG["global_batch_size"],
        shuffle=True,
        seed=args.seed,
    )
    batch = convert_batch(next(iter(loader)))
    sharded_batch = shard_batch(batch, n_devices)
    vocab_size = int(loader.vocab_size)

    pc20_structure = build_structure(
        vocab_size,
        LOCKED_CONFIG["infer_steps"],
    )
    pc0_structure = build_structure(vocab_size, 0)
    eval_structure = build_structure(vocab_size, -1)

    params = initialize_params(
        pc20_structure,
        jax.random.PRNGKey(args.seed),
    )

    schedule = optax.cosine_decay_schedule(
        init_value=LOCKED_CONFIG["lr"],
        decay_steps=LOCKED_CONFIG["epochs"] * len(loader),
        alpha=0.1,
    )
    optimizer = optax.adam(schedule)

    training_kernels = [
        (
            "backprop",
            create_pmap_train_step_backprop_autoregressive(
                pc20_structure,
                optimizer,
                use_causal_mask=True,
            ),
        ),
        (
            "pc_fp_T0",
            create_pmap_train_step_autoregressive(
                pc0_structure,
                optimizer,
                use_causal_mask=True,
            ),
        ),
        (
            "pc_fp_T20",
            create_pmap_train_step_autoregressive(
                pc20_structure,
                optimizer,
                use_causal_mask=True,
            ),
        ),
    ]

    summaries = []
    raw_frames = []
    selected = set(args.kernels)

    for name, step_fn in training_kernels:
        if name not in selected:
            continue

        print(f"Benchmarking {name}")

        summary, raw = benchmark_training_kernel(
            name,
            step_fn,
            params,
            optimizer,
            sharded_batch,
            n_devices=n_devices,
            seed=args.seed,
            warmup_steps=args.warmup_steps,
            timed_steps=args.timed_steps,
        )
        summaries.append(summary)
        raw_frames.append(raw)

        print(
            f"  first={summary['first_call_seconds']:.3f}s "
            f"median={summary['median_ms']:.3f}ms "
            f"examples/s={summary['global_examples_per_second']:.1f}"
        )

    if "feedforward_eval" in selected:
        print("Benchmarking feedforward_eval")

        eval_step = create_pmap_eval_step_autoregressive(
            eval_structure,
            use_causal_mask=True,
        )
        eval_summary, eval_raw = benchmark_eval_kernel(
            eval_step,
            replicate_params(params, n_devices),
            sharded_batch,
            n_devices=n_devices,
            seed=args.seed,
            warmup_steps=args.warmup_steps,
            timed_steps=args.timed_steps,
        )
        summaries.append(eval_summary)
        raw_frames.append(eval_raw)

    if not summaries:
        raise RuntimeError("No benchmark kernels were selected")

    summary_frame = pd.DataFrame(summaries)
    raw_frame = pd.concat(raw_frames, ignore_index=True)
    by_name = summary_frame.set_index("kernel")

    derived = {}
    required = {"backprop", "pc_fp_T0", "pc_fp_T20"}

    if required <= set(by_name.index):
        pc0_ms = float(by_name.loc["pc_fp_T0", "median_ms"])
        pc20_ms = float(by_name.loc["pc_fp_T20", "median_ms"])
        bp_ms = float(by_name.loc["backprop", "median_ms"])

        incremental_sweep_ms = (
            (pc20_ms - pc0_ms)
            / LOCKED_CONFIG["infer_steps"]
        )

        derived = {
            "pc_fp_over_bp_median_ratio": pc20_ms / bp_ms,
            "pc_fp_T0_overhead_ms": pc0_ms,
            "pc_fp_T20_median_ms": pc20_ms,
            "incremental_inference_sweep_ms": incremental_sweep_ms,
            "incremental_20_sweeps_ms": pc20_ms - pc0_ms,
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_frame.to_csv(
        args.output_dir / "runtime_summary.csv",
        index=False,
    )
    raw_frame.to_csv(
        args.output_dir / "runtime_repeats.csv",
        index=False,
    )

    manifest = {
        "study": "locked_char_pmap_runtime",
        "config": LOCKED_CONFIG,
        "seed": args.seed,
        "selected_kernels": args.kernels,
        "devices": [
            str(device) for device in jax.local_devices()
        ],
        "local_device_count": n_devices,
        "jax_version": jax.__version__,
        "python_version": platform.python_version(),
        "timing_clock": "time.perf_counter",
        "synchronization": (
            "jax.block_until_ready after every measured call"
        ),
        "inference_sweep_estimator": (
            "(median PC-FP T20 - median PC-FP T0) / 20"
        ),
        "derived": derived,
        "notes": [
            (
                "Training kernels use the same pmap functions "
                "as the main experiment."
            ),
            (
                "First-call latency includes tracing, compilation, "
                "and first execution."
            ),
            (
                "Feed-forward evaluation uses dropout disabled "
                "and zero inference steps."
            ),
            (
                "Compilation measurements may depend on benchmark "
                "order and process-level caches."
            ),
        ],
    }

    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nRuntime summary:")
    print(
        summary_frame[
            [
                "kernel",
                "first_call_seconds",
                "median_ms",
                "std_ms",
                "global_examples_per_second",
                "global_tokens_per_second",
            ]
        ].to_string(index=False)
    )

    if derived:
        print("\nDerived PC-FP costs:")
        for name, value in derived.items():
            print(f"{name}: {value:.6f}")

    print(
        f"\nWrote runtime benchmark to {args.output_dir}"
    )


if __name__ == "__main__":
    main()