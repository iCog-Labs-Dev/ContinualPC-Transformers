#!/usr/bin/env python3
"""Controlled per-node energy audit for fixed-prediction inference.

The audit evaluates shared initialization plus BP and PC-FP checkpoints from
epochs 1, 3, and 5 on identical frozen sequences, full next-token targets, and
dropout masks.  It separates output cross-entropy from hidden Gaussian energy
and records per-node error/activation scales.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp

from diagnostics.pcinterp_adapter import GRAPH_NODES, load_frozen_dataset
from fabricpc.core.fixed_prediction import (
    InferenceSGDFixedPrediction,
    run_fixed_prediction_inference,
)
from fabricpc.graph_initialization import initialize_graph_state
from fabricpc.nodes.transformer_v2 import create_deep_transformer
from fabricpc.training.train_autoregressive import create_dropout_clamps


EPOCHS = (1, 3, 5)
METHODS = ("BP", "PC-FP")
COLORS = {"BP": "#2878B5", "PC-FP": "#D95319"}
MARKERS = {"BP": "o", "PC-FP": "s"}
LINESTYLES = {"BP": "-", "PC-FP": "--"}


def checkpoint_path(checkpoint_dir: Path, seed: int, method: str, epoch: int) -> Path:
    if method == "Shared init":
        return checkpoint_dir / f"ckpt_bp_char_seed{seed}_init.pkl"
    if method == "BP":
        return checkpoint_dir / f"ckpt_bp_char_seed{seed}_epoch{epoch}.pkl"
    if method == "PC-FP":
        return checkpoint_dir / (
            f"ckpt_pc_fp_T20_eta0.3864_char_seed{seed}_epoch{epoch}.pkl"
        )
    raise ValueError(f"unknown method: {method}")


def load_checkpoint(path: Path):
    with path.open("rb") as handle:
        saved = pickle.load(handle)
    if not isinstance(saved, dict) or not {"params", "config"} <= set(saved):
        raise ValueError(f"invalid checkpoint: {path}")
    params = jax.tree_util.tree_map(jnp.asarray, saved["params"])
    return params, dict(saved["config"])


def build_structure(config: dict, *, infer_steps: int, eta_infer: float):
    vocab_size = int(config.get("vocab_size", 65))
    # vocab_size is not guaranteed to be serialized, so the caller repairs it
    # from the output projection when necessary.
    inference = InferenceSGDFixedPrediction(
        eta_infer=eta_infer,
        infer_steps=infer_steps,
        latent_decay=0.0,
        max_norm=float(config.get("max_norm", 5.0)),
    )
    return create_deep_transformer(
        depth=int(config["depth"]),
        embed_dim=int(config["embed_dim"]),
        num_heads=int(config["num_heads"]),
        mlp_dim=int(config["mlp_dim"]),
        seq_len=int(config["seq_len"]),
        vocab_size=vocab_size,
        inference=inference,
        weight_init={
            "type": "normal",
            "std": float(config.get("weight_init_std", 0.0228)),
        },
        dropout_rate=float(config.get("dropout_rate", 0.0)),
    )


def load_audit_arrays(dataset_dir: Path, samples: int) -> tuple[np.ndarray, np.ndarray]:
    # This verifies the frozen artifact and every stored array hash.
    load_frozen_dataset(dataset_dir)
    with np.load(dataset_dir / "dataset.npz", allow_pickle=False) as saved:
        inputs = saved["inputs"]
        final_targets = saved["targets"]
        test_rows = saved["split_test"]
        selected = test_rows[:samples]
        x = np.asarray(inputs[selected], dtype=np.int32)
        last = np.asarray(final_targets[selected], dtype=np.int32)

    # Reconstruct the full teacher-forced next-token sequence.  The frozen
    # artifact stores the final next character separately; earlier targets are
    # simply the input shifted one position to the left.
    y = np.concatenate([x[:, 1:], last[:, None]], axis=1).astype(np.int32)
    return x, y


def make_audit_batch(structure, nodes: tuple[str, ...], vocab_size: int):
    state_key = jax.random.PRNGKey(1729)

    @jax.jit
    def audit_batch(params, x, y_tokens, dropout_key):
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
        final_state, feedforward_state = run_fixed_prediction_inference(
            params,
            initial_state,
            clamps,
            structure,
        )

        final_energy = jnp.stack(
            [jnp.sum(final_state.nodes[node].energy) for node in nodes]
        )
        feedforward_energy = jnp.stack(
            [jnp.sum(feedforward_state.nodes[node].energy) for node in nodes]
        )
        error_sq = jnp.stack(
            [jnp.sum(jnp.square(final_state.nodes[node].error)) for node in nodes]
        )
        prediction_sq = jnp.stack(
            [jnp.sum(jnp.square(final_state.nodes[node].z_mu)) for node in nodes]
        )
        latent_sq = jnp.stack(
            [jnp.sum(jnp.square(final_state.nodes[node].z_latent)) for node in nodes]
        )
        return (
            final_energy,
            feedforward_energy,
            error_sq,
            prediction_sq,
            latent_sq,
        )

    return audit_batch


def audit_checkpoint(
    params,
    *,
    audit_batch,
    x: np.ndarray,
    y: np.ndarray,
    nodes: tuple[str, ...],
    widths: dict[str, int],
    batch_size: int,
    dropout_seed: int,
) -> tuple[dict[str, float], list[dict[str, float | str | int]]]:
    totals = {
        key: np.zeros(len(nodes), dtype=np.float64)
        for key in (
            "final_energy",
            "feedforward_energy",
            "error_sq",
            "prediction_sq",
            "latent_sq",
        )
    }
    n_samples = 0
    for batch_index, start in enumerate(range(0, len(x), batch_size)):
        stop = min(start + batch_size, len(x))
        piece_x = jnp.asarray(x[start:stop])
        piece_y = jnp.asarray(y[start:stop])
        key = jax.random.fold_in(jax.random.PRNGKey(dropout_seed), batch_index)
        outputs = audit_batch(params, piece_x, piece_y, key)
        jax.block_until_ready(outputs)
        for name, value in zip(totals, outputs, strict=True):
            totals[name] += np.asarray(value, dtype=np.float64)
        n_samples += stop - start

    seq_len = int(x.shape[1])
    node_rows: list[dict[str, float | str | int]] = []
    for index, node in enumerate(nodes):
        width = widths[node]
        coordinates = n_samples * seq_len * width
        final_energy = totals["final_energy"][index]
        row = {
            "node": node,
            "width": width,
            "energy_per_sample": final_energy / n_samples,
            "energy_per_token": final_energy / (n_samples * seq_len),
            "energy_per_coordinate": (
                np.nan if node == "logits" else final_energy / coordinates
            ),
            "feedforward_energy_per_sample": (
                totals["feedforward_energy"][index] / n_samples
            ),
            "error_rms": np.sqrt(totals["error_sq"][index] / coordinates),
            "prediction_rms": np.sqrt(
                totals["prediction_sq"][index] / coordinates
            ),
            "latent_rms": np.sqrt(totals["latent_sq"][index] / coordinates),
        }
        node_rows.append(row)

    final_total = float(totals["final_energy"].sum() / n_samples)
    feedforward_total = float(totals["feedforward_energy"].sum() / n_samples)
    logits_index = nodes.index("logits")
    output_per_sample = float(totals["final_energy"][logits_index] / n_samples)
    hidden_per_sample = float(
        (totals["final_energy"].sum() - totals["final_energy"][logits_index])
        / n_samples
    )
    hidden_coordinates = sum(
        n_samples * seq_len * widths[node] for node in nodes if node != "logits"
    )
    hidden_error_sq = sum(
        totals["error_sq"][index]
        for index, node in enumerate(nodes)
        if node != "logits"
    )
    summary = {
        "samples": n_samples,
        "total_energy_per_sample": final_total,
        "feedforward_energy_per_sample": feedforward_total,
        "settling_added_energy_per_sample": final_total - feedforward_total,
        "hidden_energy_per_sample": hidden_per_sample,
        "output_energy_per_sample": output_per_sample,
        "output_ce_per_token": output_per_sample / seq_len,
        "hidden_error_rms": float(np.sqrt(hidden_error_sq / hidden_coordinates)),
    }
    return summary, node_rows


def aggregate_mean_std(frame: pd.DataFrame, groups: list[str], metrics: list[str]):
    result = frame.groupby(groups)[metrics].agg(["mean", "std"]).reset_index()
    result.columns = [
        "_".join(str(part) for part in column if part)
        if isinstance(column, tuple)
        else str(column)
        for column in result.columns
    ]
    return result


def add_shared_init_for_plot(frame: pd.DataFrame) -> pd.DataFrame:
    shared = frame[frame["method"] == "Shared init"].copy()
    copies = []
    for method in METHODS:
        piece = shared.copy()
        piece["method"] = method
        copies.append(piece)
    trained = frame[frame["method"].isin(METHODS)]
    return pd.concat([trained, *copies], ignore_index=True)


def save_figure(fig: plt.Figure, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_components(checkpoints: pd.DataFrame, output_dir: Path) -> None:
    plot_frame = add_shared_init_for_plot(checkpoints)
    metrics = [
        ("total_energy_per_sample", "Settled total energy", "Energy per sample"),
        ("hidden_energy_per_sample", "Hidden Gaussian energy", "Energy per sample"),
        ("output_ce_per_token", "Output cross-entropy", "CE per token"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.1))
    for ax, (metric, title, ylabel) in zip(axes, metrics, strict=True):
        for method in METHODS:
            rows = plot_frame[plot_frame["method"] == method]
            stats = rows.groupby("epoch")[metric].agg(["mean", "std"]).sort_index()
            x = stats.index.to_numpy(dtype=float)
            mean = stats["mean"].to_numpy(dtype=float)
            std = stats["std"].fillna(0.0).to_numpy(dtype=float)
            ax.errorbar(
                x,
                mean,
                yerr=std,
                color=COLORS[method],
                marker=MARKERS[method],
                markerfacecolor="white" if method == "PC-FP" else COLORS[method],
                linestyle=LINESTYLES[method],
                capsize=3,
                label=method,
            )
        ax.set_xticks((0, 1, 3, 5))
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.18)
    axes[-1].legend(frameon=False)
    fig.suptitle("Controlled fixed-prediction energy audit (mean ± SD, five seeds)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, output_dir, "energy_components_multiseed")


def plot_node_heatmaps(nodes: pd.DataFrame, output_dir: Path, metric: str, name: str) -> None:
    plot_frame = add_shared_init_for_plot(nodes)
    means = (
        plot_frame.groupby(["method", "epoch", "node"])[metric]
        .mean()
        .reset_index()
    )
    matrices = {}
    for method in METHODS:
        matrix = (
            means[means["method"] == method]
            .pivot(index="node", columns="epoch", values=metric)
            .reindex(index=GRAPH_NODES, columns=(0, 1, 3, 5))
            .to_numpy(dtype=float)
        )
        matrices[method] = matrix

    positive = np.concatenate([value.ravel() for value in matrices.values()])
    positive = positive[np.isfinite(positive) & (positive > 0)]
    floor = float(positive.min()) * 0.5 if positive.size else 1e-12
    transformed = {
        method: np.log10(np.maximum(matrix, floor))
        for method, matrix in matrices.items()
    }
    vmin = min(float(np.nanmin(value)) for value in transformed.values())
    vmax = max(float(np.nanmax(value)) for value in transformed.values())

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 5.2), sharey=True)
    image = None
    for ax, method in zip(axes, METHODS, strict=True):
        image = ax.imshow(
            transformed[method], aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax
        )
        for row in range(len(GRAPH_NODES)):
            for column in range(4):
                value = matrices[method][row, column]
                if np.isfinite(value):
                    ax.text(
                        column,
                        row,
                        f"{value:.3g}",
                        ha="center",
                        va="center",
                        fontsize=7.5,
                        color=(
                            "white"
                            if transformed[method][row, column] < (vmin + vmax) / 2
                            else "black"
                        ),
                    )
        ax.set_xticks(range(4), ("init", "1", "3", "5"))
        ax.set_yticks(range(len(GRAPH_NODES)), GRAPH_NODES)
        ax.set_xlabel("Checkpoint epoch")
        ax.set_title(method)
    axes[0].set_ylabel("Node")
    fig.colorbar(image, ax=axes, fraction=0.025, pad=0.03, label=f"log10({metric})")
    fig.suptitle(name)
    fig.subplots_adjust(top=0.88, bottom=0.12, left=0.13, right=0.90, wspace=0.08)
    save_figure(fig, output_dir, metric)


def main() -> None:
    parser = argparse.ArgumentParser()
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
        default=Path("runs/energy_audit/fixed_prediction_multiseed"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--infer-steps", type=int, default=20)
    parser.add_argument("--eta-infer", type=float, default=0.3864)
    parser.add_argument("--dropout-seed", type=int, default=20260927)
    args = parser.parse_args()

    if args.samples < 1 or args.batch_size < 1:
        raise ValueError("--samples and --batch-size must be positive")
    if args.samples % args.batch_size:
        raise ValueError("--samples must be divisible by --batch-size")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must not contain duplicates")

    x, y = load_audit_arrays(args.dataset_dir, args.samples)
    reference_path = checkpoint_path(args.checkpoint_dir, args.seeds[0], "Shared init", 0)
    reference_params, reference_config = load_checkpoint(reference_path)
    vocab_size = int(reference_params.nodes["logits"].biases["b_out"].shape[-1])
    reference_config["vocab_size"] = vocab_size
    structure = build_structure(
        reference_config,
        infer_steps=args.infer_steps,
        eta_infer=args.eta_infer,
    )
    nodes = tuple(GRAPH_NODES)
    widths = {
        node: int(structure.nodes[node].node_info.shape[-1]) for node in nodes
    }
    audit_batch = make_audit_batch(structure, nodes, vocab_size)

    checkpoint_rows = []
    node_rows = []
    schedule = [("Shared init", 0)] + [
        (method, epoch) for method in METHODS for epoch in EPOCHS
    ]
    for seed in args.seeds:
        for method, epoch in schedule:
            path = checkpoint_path(args.checkpoint_dir, seed, method, epoch)
            if not path.is_file():
                raise FileNotFoundError(path)
            params, config = load_checkpoint(path)
            for field in ("depth", "embed_dim", "num_heads", "mlp_dim", "seq_len"):
                if int(config[field]) != int(reference_config[field]):
                    raise ValueError(f"{path}: incompatible {field}")
            print(f"seed {seed}: {method} epoch {epoch}")
            summary, per_node = audit_checkpoint(
                params,
                audit_batch=audit_batch,
                x=x,
                y=y,
                nodes=nodes,
                widths=widths,
                batch_size=args.batch_size,
                dropout_seed=args.dropout_seed,
            )
            checkpoint_rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "epoch": epoch,
                    "checkpoint": str(path),
                    **summary,
                }
            )
            for row in per_node:
                node_rows.append(
                    {
                        "seed": seed,
                        "method": method,
                        "epoch": epoch,
                        **row,
                    }
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_frame = pd.DataFrame(checkpoint_rows)
    node_frame = pd.DataFrame(node_rows)
    checkpoint_frame.to_csv(args.output_dir / "checkpoint_energy.csv", index=False)
    node_frame.to_csv(args.output_dir / "node_energy.csv", index=False)

    checkpoint_metrics = [
        "total_energy_per_sample",
        "feedforward_energy_per_sample",
        "settling_added_energy_per_sample",
        "hidden_energy_per_sample",
        "output_energy_per_sample",
        "output_ce_per_token",
        "hidden_error_rms",
    ]
    node_metrics = [
        "energy_per_sample",
        "energy_per_token",
        "energy_per_coordinate",
        "feedforward_energy_per_sample",
        "error_rms",
        "prediction_rms",
        "latent_rms",
    ]
    aggregate_mean_std(
        checkpoint_frame,
        ["method", "epoch"],
        checkpoint_metrics,
    ).to_csv(args.output_dir / "checkpoint_energy_summary.csv", index=False)
    aggregate_mean_std(
        node_frame,
        ["method", "epoch", "node"],
        node_metrics,
    ).to_csv(args.output_dir / "node_energy_summary.csv", index=False)

    plot_components(checkpoint_frame, args.output_dir)
    plot_node_heatmaps(
        node_frame,
        args.output_dir,
        "energy_per_sample",
        "Settled local-energy contributions (five-seed mean)",
    )
    plot_node_heatmaps(
        node_frame,
        args.output_dir,
        "error_rms",
        "Local prediction-error RMS (five-seed mean)",
    )

    manifest = {
        "seeds": args.seeds,
        "samples": args.samples,
        "sample_split": "frozen char_v1 test rows, first N",
        "batch_size": args.batch_size,
        "infer_steps": args.infer_steps,
        "eta_infer": args.eta_infer,
        "dropout_seed": args.dropout_seed,
        "dropout_training_mode": True,
        "nodes": list(nodes),
        "checkpoints": checkpoint_frame["checkpoint"].tolist(),
        "notes": [
            "All checkpoints use identical inputs, full next-token targets, and dropout masks.",
            "Energy per sample matches the code's batch-only normalization.",
            "Output CE is also reported per token; hidden Gaussian energy is separated.",
        ],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    summary = pd.read_csv(args.output_dir / "checkpoint_energy_summary.csv")
    print("\nPC-FP controlled energy summary:")
    pc = summary[summary["method"] == "PC-FP"].sort_values("epoch")
    for _, row in pc.iterrows():
        print(
            f"epoch {int(row['epoch'])}: total={row['total_energy_per_sample_mean']:.3f} "
            f"hidden={row['hidden_energy_per_sample_mean']:.3f} "
            f"CE/token={row['output_ce_per_token_mean']:.4f} "
            f"hidden_error_rms={row['hidden_error_rms_mean']:.5f}"
        )
    print(f"\nWrote audit to {args.output_dir}")


if __name__ == "__main__":
    main()
