#!/usr/bin/env python3
"""Audit fixed-prediction settling after every latent-inference sweep.

Sweep zero is the refreshed feed-forward state before any latent update.  Each
subsequent row is recorded after one norm-clipped update and another exact
fixed-prediction gradient/energy refresh.  The script intentionally reuses the
checkpoint, dataset, graph, and dropout conventions of
``audit_fixed_prediction_energy.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp

from diagnostics.audit_fixed_prediction_energy import (
    EPOCHS,
    METHODS,
    build_structure,
    checkpoint_path,
    load_audit_arrays,
    load_checkpoint,
)
from diagnostics.pcinterp_adapter import GRAPH_NODES
from fabricpc.core.fixed_prediction import fixed_prediction_gradient_pass
from fabricpc.core.state_ops import update_node_in_state
from fabricpc.graph_initialization import initialize_graph_state
from fabricpc.training.train_autoregressive import create_dropout_clamps


COLORS = {"BP": "#2878B5", "PC-FP": "#D95319", "Shared init": "#777777"}
LINESTYLES = {"BP": "-", "PC-FP": "--", "Shared init": "-"}


def make_trajectory_batch(structure, nodes: tuple[str, ...], vocab_size: int):
    """Return a jitted batch function emitting metrics for sweeps 0..T."""
    inference = structure.config["inference"]
    config = inference.config
    max_norm = float(config["max_norm"])
    infer_steps = int(config["infer_steps"])
    state_key = jax.random.PRNGKey(1729)
    updatable = tuple(node not in structure.task_map.values() for node in nodes)

    def metrics(state, initial_state):
        energy = jnp.stack([jnp.sum(state.nodes[node].energy) for node in nodes])
        error_sq = jnp.stack(
            [jnp.sum(jnp.square(state.nodes[node].error)) for node in nodes]
        )
        displacement_sq = jnp.stack(
            [
                jnp.sum(
                    jnp.square(
                        state.nodes[node].z_latent
                        - initial_state.nodes[node].z_latent
                    )
                )
                for node in nodes
            ]
        )
        clipped_count = []
        grad_norm_sum = []
        for node, can_update in zip(nodes, updatable, strict=True):
            gradient = state.nodes[node].latent_grad
            axes = tuple(range(1, gradient.ndim))
            norm = jnp.sqrt(jnp.sum(gradient.conj() * gradient, axis=axes))
            if can_update:
                clipped_count.append(jnp.sum(norm > max_norm))
                grad_norm_sum.append(jnp.sum(norm))
            else:
                clipped_count.append(jnp.asarray(0, dtype=jnp.int32))
                grad_norm_sum.append(jnp.asarray(0.0, dtype=jnp.float32))
        return (
            energy,
            error_sq,
            displacement_sq,
            jnp.stack(clipped_count),
            jnp.stack(grad_norm_sum),
        )

    @jax.jit
    def trajectory_batch(params, x, y_tokens, dropout_key):
        batch_size = x.shape[0]
        clamps = {
            structure.task_map["x"]: x,
            structure.task_map["y"]: jax.nn.one_hot(y_tokens, vocab_size),
        }
        clamps.update(
            create_dropout_clamps(
                structure,
                batch_size,
                dropout_key,
                training=True,
            )
        )
        initial_state = initialize_graph_state(
            structure,
            batch_size,
            state_key,
            clamps=clamps,
            params=params,
        )
        feedforward_state = initial_state
        refreshed = fixed_prediction_gradient_pass(
            params,
            initial_state,
            feedforward_state,
            clamps,
            structure,
        )
        initial_metrics = metrics(refreshed, initial_state)

        def one_sweep(state, _):
            for node_name in structure.node_order:
                if node_name in clamps:
                    continue
                new_latent = type(inference).compute_new_latent(
                    node_name,
                    state.nodes[node_name],
                    config,
                )
                state = update_node_in_state(
                    state,
                    node_name,
                    z_latent=new_latent,
                )
            state = fixed_prediction_gradient_pass(
                params,
                state,
                feedforward_state,
                clamps,
                structure,
            )
            return state, metrics(state, initial_state)

        _, later_metrics = jax.lax.scan(
            one_sweep,
            refreshed,
            xs=None,
            length=infer_steps,
        )
        return tuple(
            jnp.concatenate([first[None, ...], later], axis=0)
            for first, later in zip(initial_metrics, later_metrics, strict=True)
        )

    return trajectory_batch, updatable


def audit_checkpoint(
    params,
    *,
    trajectory_batch,
    updatable: tuple[bool, ...],
    x: np.ndarray,
    y: np.ndarray,
    nodes: tuple[str, ...],
    widths: dict[str, int],
    batch_size: int,
    dropout_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_sweeps = None
    totals = None
    n_samples = 0
    for batch_index, start in enumerate(range(0, len(x), batch_size)):
        stop = min(start + batch_size, len(x))
        key = jax.random.fold_in(jax.random.PRNGKey(dropout_seed), batch_index)
        output = trajectory_batch(
            params,
            jnp.asarray(x[start:stop]),
            jnp.asarray(y[start:stop]),
            key,
        )
        jax.block_until_ready(output)
        arrays = tuple(np.asarray(value, dtype=np.float64) for value in output)
        if totals is None:
            totals = [np.zeros_like(value, dtype=np.float64) for value in arrays]
            n_sweeps = arrays[0].shape[0]
        for total, value in zip(totals, arrays, strict=True):
            total += value
        n_samples += stop - start

    if totals is None or n_sweeps is None:
        raise ValueError("trajectory audit received no examples")
    energy, error_sq, displacement_sq, clipped_count, grad_norm_sum = totals
    seq_len = int(x.shape[1])
    logits_index = nodes.index("logits")
    hidden_indexes = [index for index, node in enumerate(nodes) if node != "logits"]
    hidden_coordinates = sum(
        n_samples * seq_len * widths[nodes[index]] for index in hidden_indexes
    )

    checkpoint_rows = []
    node_rows = []
    for sweep in range(n_sweeps):
        output_energy = energy[sweep, logits_index] / n_samples
        hidden_energy = energy[sweep, hidden_indexes].sum() / n_samples
        checkpoint_rows.append(
            {
                "sweep": sweep,
                "samples": n_samples,
                "total_energy_per_sample": energy[sweep].sum() / n_samples,
                "hidden_energy_per_sample": hidden_energy,
                "output_energy_per_sample": output_energy,
                "output_ce_per_token": output_energy / seq_len,
                "hidden_error_rms": np.sqrt(
                    error_sq[sweep, hidden_indexes].sum() / hidden_coordinates
                ),
                "hidden_displacement_rms": np.sqrt(
                    displacement_sq[sweep, hidden_indexes].sum()
                    / hidden_coordinates
                ),
                "nodes_clipped_fraction_mean": float(
                    np.mean(
                        [
                            clipped_count[sweep, index] / n_samples
                            for index, can_update in enumerate(updatable)
                            if can_update
                        ]
                    )
                ),
            }
        )
        for index, node in enumerate(nodes):
            coordinates = n_samples * seq_len * widths[node]
            node_rows.append(
                {
                    "sweep": sweep,
                    "node": node,
                    "width": widths[node],
                    "updatable": updatable[index],
                    "energy_per_sample": energy[sweep, index] / n_samples,
                    "energy_per_token": energy[sweep, index]
                    / (n_samples * seq_len),
                    "energy_per_coordinate": (
                        np.nan
                        if node == "logits"
                        else energy[sweep, index] / coordinates
                    ),
                    "error_rms": np.sqrt(error_sq[sweep, index] / coordinates),
                    "displacement_rms": np.sqrt(
                        displacement_sq[sweep, index] / coordinates
                    ),
                    "latent_gradient_norm_mean": (
                        grad_norm_sum[sweep, index] / n_samples
                        if updatable[index]
                        else np.nan
                    ),
                    "clipped_fraction": (
                        clipped_count[sweep, index] / n_samples
                        if updatable[index]
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(checkpoint_rows), pd.DataFrame(node_rows)


def summarize_trajectory(frame: pd.DataFrame) -> dict[str, float | int]:
    energy = frame["total_energy_per_sample"].to_numpy(dtype=float)
    ce = frame["output_ce_per_token"].to_numpy(dtype=float)
    tolerance = 1.0e-10
    return {
        "initial_total_energy_per_sample": energy[0],
        "final_total_energy_per_sample": energy[-1],
        "minimum_total_energy_per_sample": float(np.min(energy)),
        "minimum_total_energy_sweep": int(np.argmin(energy)),
        "energy_nonincreasing_transition_fraction": float(
            np.mean(np.diff(energy) <= tolerance)
        ),
        "initial_output_ce_per_token": ce[0],
        "final_output_ce_per_token": ce[-1],
        "minimum_output_ce_per_token": float(np.min(ce)),
        "minimum_output_ce_sweep": int(np.argmin(ce)),
        "ce_nonincreasing_transition_fraction": float(
            np.mean(np.diff(ce) <= tolerance)
        ),
        "final_hidden_error_rms": float(frame["hidden_error_rms"].iloc[-1]),
        "final_hidden_displacement_rms": float(
            frame["hidden_displacement_rms"].iloc[-1]
        ),
        "maximum_nodes_clipped_fraction_mean": float(
            frame["nodes_clipped_fraction_mean"].max()
        ),
    }


def save_figure(fig: plt.Figure, directory: Path, name: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(directory / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_checkpoint_trajectories(frame: pd.DataFrame, output_dir: Path) -> None:
    metrics = (
        ("total_energy_per_sample", "Total energy per sample"),
        ("hidden_energy_per_sample", "Hidden energy per sample"),
        ("output_ce_per_token", "Output CE per token"),
        ("hidden_displacement_rms", "Hidden displacement RMS"),
    )
    epochs = (0, *EPOCHS)
    fig, axes = plt.subplots(len(metrics), len(epochs), figsize=(14.0, 10.0), sharex=True)
    for column, epoch in enumerate(epochs):
        selected = frame[frame["epoch"] == epoch]
        for row, (metric, ylabel) in enumerate(metrics):
            ax = axes[row, column]
            for method in ("Shared init", *METHODS):
                values = selected[selected["method"] == method]
                if values.empty:
                    continue
                ax.plot(
                    values["sweep"],
                    values[metric],
                    color=COLORS[method],
                    linestyle=LINESTYLES[method],
                    label=method,
                )
            if row == 0:
                ax.set_title("Initialization" if epoch == 0 else f"Epoch {epoch}")
            if column == 0:
                ax.set_ylabel(ylabel)
            if row == len(metrics) - 1:
                ax.set_xlabel("Inference sweep")
            ax.grid(alpha=0.18)
    handles, labels = axes[0, -1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Fixed-prediction settling trajectory (seed 0)", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_figure(fig, output_dir / "figures", "checkpoint_trajectories")


def plot_node_heatmap(nodes: pd.DataFrame, output_dir: Path) -> None:
    selected = nodes[(nodes["method"] == "PC-FP") & (nodes["epoch"] == 5)]
    matrix = (
        selected.pivot(index="node", columns="sweep", values="energy_per_sample")
        .reindex(index=GRAPH_NODES)
        .to_numpy(dtype=float)
    )
    positive = matrix[np.isfinite(matrix) & (matrix > 0)]
    floor = float(positive.min()) * 0.5 if positive.size else 1.0e-12
    image_values = np.log10(np.maximum(matrix, floor))
    fig, ax = plt.subplots(figsize=(9.0, 4.3))
    image = ax.imshow(image_values, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(GRAPH_NODES)), GRAPH_NODES)
    ticks = np.arange(0, matrix.shape[1], max(1, matrix.shape[1] // 5))
    ax.set_xticks(ticks, ticks)
    ax.set(xlabel="Inference sweep", ylabel="Node")
    ax.set_title("PC-FP epoch-5 local energy during settling")
    fig.colorbar(image, ax=ax, label="log10(energy per sample)")
    fig.tight_layout()
    save_figure(fig, output_dir / "figures", "pc_fp_epoch5_node_energy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=Path("runs/checkpoints")
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("artifacts/interpretation/char_v1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/energy_audit/fixed_prediction_trajectory_seed0"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--infer-steps", type=int, default=20)
    parser.add_argument("--eta-infer", type=float, default=0.3864)
    parser.add_argument("--dropout-seed", type=int, default=20260927)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples < 1 or args.batch_size < 1:
        raise ValueError("--samples and --batch-size must be positive")
    if args.samples % args.batch_size:
        raise ValueError("--samples must be divisible by --batch-size")
    if args.infer_steps < 1:
        raise ValueError("--infer-steps must be positive")

    x, y = load_audit_arrays(args.dataset_dir, args.samples)
    reference_path = checkpoint_path(args.checkpoint_dir, args.seed, "Shared init", 0)
    reference_params, reference_config = load_checkpoint(reference_path)
    vocab_size = int(reference_params.nodes["logits"].biases["b_out"].shape[-1])
    reference_config["vocab_size"] = vocab_size
    structure = build_structure(
        reference_config,
        infer_steps=args.infer_steps,
        eta_infer=args.eta_infer,
    )
    nodes = tuple(GRAPH_NODES)
    widths = {node: int(structure.nodes[node].node_info.shape[-1]) for node in nodes}
    trajectory_batch, updatable = make_trajectory_batch(structure, nodes, vocab_size)

    checkpoint_frames = []
    node_frames = []
    summary_rows = []
    schedule = [("Shared init", 0)] + [
        (method, epoch) for epoch in EPOCHS for method in METHODS
    ]
    for method, epoch in schedule:
        path = checkpoint_path(args.checkpoint_dir, args.seed, method, epoch)
        if not path.is_file():
            raise FileNotFoundError(path)
        params, config = load_checkpoint(path)
        for field in ("depth", "embed_dim", "num_heads", "mlp_dim", "seq_len"):
            if int(config[field]) != int(reference_config[field]):
                raise ValueError(f"{path}: incompatible {field}")
        print(f"{method} epoch {epoch}: tracing sweeps 0..{args.infer_steps}")
        checkpoints, per_node = audit_checkpoint(
            params,
            trajectory_batch=trajectory_batch,
            updatable=updatable,
            x=x,
            y=y,
            nodes=nodes,
            widths=widths,
            batch_size=args.batch_size,
            dropout_seed=args.dropout_seed,
        )
        for frame in (checkpoints, per_node):
            frame.insert(0, "epoch", epoch)
            frame.insert(0, "method", method)
            frame.insert(0, "seed", args.seed)
            frame.insert(3, "checkpoint", str(path))
        checkpoint_frames.append(checkpoints)
        node_frames.append(per_node)
        summary_rows.append(
            {
                "seed": args.seed,
                "method": method,
                "epoch": epoch,
                "checkpoint": str(path),
                **summarize_trajectory(checkpoints),
            }
        )

    checkpoint_frame = pd.concat(checkpoint_frames, ignore_index=True)
    node_frame = pd.concat(node_frames, ignore_index=True)
    summary_frame = pd.DataFrame(summary_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_frame.to_csv(args.output_dir / "checkpoint_trajectory.csv", index=False)
    node_frame.to_csv(args.output_dir / "node_trajectory.csv", index=False)
    summary_frame.to_csv(args.output_dir / "trajectory_summary.csv", index=False)
    plot_checkpoint_trajectories(checkpoint_frame, args.output_dir)
    plot_node_heatmap(node_frame, args.output_dir)

    manifest = {
        "study": "fixed_prediction_settling_trajectory",
        "seed": args.seed,
        "samples": args.samples,
        "sample_split": "frozen char_v1 test rows, first N",
        "batch_size": args.batch_size,
        "infer_steps": args.infer_steps,
        "eta_infer": args.eta_infer,
        "dropout_seed": args.dropout_seed,
        "dropout_training_mode": True,
        "nodes": list(nodes),
        "sweep_zero": "refreshed feed-forward state before latent updates",
        "clipping_definition": (
            "fraction of samples whose whole-sequence node latent-gradient norm "
            "exceeds max_norm"
        ),
        "notes": [
            "Inputs, targets, and dropout masks match the endpoint energy audit.",
            "Predictions and Jacobians remain cached at feed-forward values.",
            "Clamped nodes are not included in mean clipping fractions.",
        ],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print("\nSettling trajectory summary:")
    columns = [
        "method",
        "epoch",
        "initial_total_energy_per_sample",
        "final_total_energy_per_sample",
        "energy_nonincreasing_transition_fraction",
        "initial_output_ce_per_token",
        "final_output_ce_per_token",
        "ce_nonincreasing_transition_fraction",
        "maximum_nodes_clipped_fraction_mean",
    ]
    print(summary_frame[columns].to_string(index=False))
    print(f"\nWrote trajectory audit to {args.output_dir}")


if __name__ == "__main__":
    main()
