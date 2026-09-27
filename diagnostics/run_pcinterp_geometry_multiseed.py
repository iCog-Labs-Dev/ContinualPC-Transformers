#!/usr/bin/env python3
"""Run and aggregate pcinterp geometry for paired BP/PC-FP seeds.

Each seed compares a shared initialization, the matched BP epoch-5 checkpoint,
and the matched PC-FP epoch-5 checkpoint on the frozen char_v1 dataset.
Completed seed directories are reused unless ``--force`` is passed.
"""

from __future__ import annotations

import argparse
import gc
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pcinterp.studies import geometry_study

from diagnostics.pcinterp_adapter import (
    GRAPH_NODES,
    FabricPCTransformerV2Adapter,
    load_frozen_dataset,
)


NODE_ORDER = list(GRAPH_NODES)
PAIR_ORDER = ["Init–BP", "Init–PC-FP", "BP–PC-FP"]
ROLE_ORDER = ["Shared init", "BP", "PC-FP"]
COLORS = {"Shared init": "#777777", "BP": "#2878B5", "PC-FP": "#D95319"}
MARKERS = {"Shared init": "o", "BP": "o", "PC-FP": "s"}
LINESTYLES = {"Shared init": "-", "BP": "-", "PC-FP": "--"}
REQUIRED_OUTPUTS = ("cka.csv", "participation_rank.csv", "spectrum.csv", "manifest.json")


def checkpoint_paths(checkpoint_dir: Path, seed: int) -> dict[str, Path]:
    return {
        "bp_init": checkpoint_dir / f"ckpt_bp_char_seed{seed}_init.pkl",
        "pc_init": checkpoint_dir
        / f"ckpt_pc_fp_T20_eta0.3864_char_seed{seed}_init.pkl",
        "bp_final": checkpoint_dir / f"ckpt_bp_char_seed{seed}_epoch5.pkl",
        "pc_final": checkpoint_dir
        / f"ckpt_pc_fp_T20_eta0.3864_char_seed{seed}_epoch5.pkl",
    }


def load_params(path: Path):
    with path.open("rb") as handle:
        saved = pickle.load(handle)
    if not isinstance(saved, dict) or "params" not in saved:
        raise ValueError(f"{path} does not contain checkpoint parameters")
    return saved["params"]


def initialization_max_abs(bp_path: Path, pc_path: Path) -> float:
    import jax

    bp_leaves, bp_tree = jax.tree_util.tree_flatten(load_params(bp_path))
    pc_leaves, pc_tree = jax.tree_util.tree_flatten(load_params(pc_path))
    if bp_tree != pc_tree or len(bp_leaves) != len(pc_leaves):
        raise AssertionError("BP and PC-FP initialization pytrees differ")
    differences = [
        float(np.max(np.abs(np.asarray(left) - np.asarray(right))))
        for left, right in zip(bp_leaves, pc_leaves, strict=True)
    ]
    return max(differences, default=0.0)


def complete(directory: Path) -> bool:
    return all((directory / name).is_file() for name in REQUIRED_OUTPUTS)


def model_role(name: str) -> str:
    if name.startswith("shared_init_seed"):
        return "Shared init"
    if name.startswith("bp_seed"):
        return "BP"
    if name.startswith("pc_fp_seed"):
        return "PC-FP"
    raise ValueError(f"Unrecognized adapter name: {name}")


def model_pair(left: str, right: str) -> str:
    roles = {model_role(left), model_role(right)}
    if roles == {"Shared init", "BP"}:
        return "Init–BP"
    if roles == {"Shared init", "PC-FP"}:
        return "Init–PC-FP"
    if roles == {"BP", "PC-FP"}:
        return "BP–PC-FP"
    raise ValueError(f"Unexpected geometry pair: {left}, {right}")


def save_figure(fig: plt.Figure, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def run_seed(
    seed: int,
    *,
    dataset,
    checkpoint_dir: Path,
    output_root: Path,
    batch_size: int,
    top_eigenvalues: int,
    force: bool,
) -> tuple[Path, float]:
    paths = checkpoint_paths(checkpoint_dir, seed)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing checkpoints:\n" + "\n".join(missing))

    init_difference = initialization_max_abs(paths["bp_init"], paths["pc_init"])
    if init_difference != 0.0:
        raise AssertionError(
            f"seed {seed}: BP/PC-FP initialization differs; max_abs={init_difference}"
        )
    print(f"seed {seed}: initialization max_abs={init_difference:.1f}")

    seed_dir = output_root / f"seed{seed}"
    if complete(seed_dir) and not force:
        print(f"seed {seed}: complete outputs already exist; reusing {seed_dir}")
        return seed_dir, init_difference

    adapters = [
        FabricPCTransformerV2Adapter(
            paths["bp_init"], name=f"shared_init_seed{seed}", batch_size=batch_size
        ),
        FabricPCTransformerV2Adapter(
            paths["bp_final"], name=f"bp_seed{seed}_epoch5", batch_size=batch_size
        ),
        FabricPCTransformerV2Adapter(
            paths["pc_final"], name=f"pc_fp_seed{seed}_epoch5", batch_size=batch_size
        ),
    ]
    print(f"seed {seed}: running geometry study")
    geometry_study.run(
        adapters,
        dataset,
        seed_dir,
        nodes=GRAPH_NODES,
        top_eigenvalues=top_eigenvalues,
    )
    print(f"seed {seed}: wrote {seed_dir}")
    del adapters
    gc.collect()
    return seed_dir, init_difference


def read_seed_outputs(seed_dirs: dict[int, Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cka_frames = []
    rank_frames = []
    spectrum_frames = []
    for seed, directory in seed_dirs.items():
        cka = pd.read_csv(directory / "cka.csv")
        cka.insert(0, "seed", seed)
        cka["pair"] = [
            model_pair(left, right)
            for left, right in zip(cka["left"], cka["right"], strict=True)
        ]
        cka_frames.append(cka)

        ranks = pd.read_csv(directory / "participation_rank.csv")
        ranks.insert(0, "seed", seed)
        ranks["role"] = ranks["model"].map(model_role)
        rank_frames.append(ranks)

        spectra = pd.read_csv(directory / "spectrum.csv")
        spectra.insert(0, "seed", seed)
        spectra["role"] = spectra["model"].map(model_role)
        spectrum_frames.append(spectra)

    return (
        pd.concat(cka_frames, ignore_index=True),
        pd.concat(rank_frames, ignore_index=True),
        pd.concat(spectrum_frames, ignore_index=True),
    )


def summarize(
    cka: pd.DataFrame,
    ranks: pd.DataFrame,
    spectra: pd.DataFrame,
    init_rows: list[dict[str, float]],
    output_root: Path,
) -> None:
    aggregate = output_root / "aggregate"
    figures = output_root / "figures"
    aggregate.mkdir(parents=True, exist_ok=True)

    cka.to_csv(aggregate / "cka_all_seeds.csv", index=False)
    ranks.to_csv(aggregate / "rank_all_seeds.csv", index=False)
    spectra.to_csv(aggregate / "spectrum_all_seeds.csv", index=False)
    pd.DataFrame(init_rows).to_csv(
        aggregate / "initialization_checks.csv", index=False
    )

    cka_summary = (
        cka.groupby(["node", "pair"])["cka"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )
    cka_summary.to_csv(aggregate / "cka_summary.csv", index=False)

    rank_metrics = ["participation_rank", "entropy_rank", "total_variance"]
    rank_summary = (
        ranks.groupby(["node", "role"], as_index=False)[rank_metrics]
        .agg(["mean", "std"])
    )
    rank_summary.columns = [
        "_".join(str(part) for part in column if part)
        if isinstance(column, tuple)
        else str(column)
        for column in rank_summary.columns
    ]
    rank_summary.to_csv(aggregate / "rank_summary.csv", index=False)

    wide = ranks.pivot(
        index=["seed", "node"], columns="role", values=rank_metrics
    )
    difference_rows = []
    for seed, node in wide.index:
        for metric in rank_metrics:
            difference_rows.append(
                {
                    "seed": seed,
                    "node": node,
                    "metric": metric,
                    "pc_fp_minus_bp": float(
                        wide.loc[(seed, node), (metric, "PC-FP")]
                        - wide.loc[(seed, node), (metric, "BP")]
                    ),
                }
            )
    differences = pd.DataFrame(difference_rows)
    differences.to_csv(aggregate / "rank_paired_differences.csv", index=False)
    difference_summary = (
        differences.groupby(["node", "metric"])["pc_fp_minus_bp"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )
    difference_summary.to_csv(
        aggregate / "rank_paired_difference_summary.csv", index=False
    )

    plot_cka(cka, figures)
    plot_rank_summaries(ranks, figures)
    plot_spectra(spectra, ranks, figures)

    print("\nBP vs PC-FP CKA across seeds:")
    report = cka_summary[cka_summary["pair"] == "BP–PC-FP"].set_index("node")
    for node in NODE_ORDER:
        row = report.loc[node]
        print(
            f"{node:8} mean={row['mean']:.4f} std={row['std']:.4f} "
            f"range=[{row['min']:.4f}, {row['max']:.4f}]"
        )
    print(f"\nAggregate tables: {aggregate}")
    print(f"Aggregate figures: {figures}")


def plot_cka(cka: pd.DataFrame, output_dir: Path) -> None:
    means = cka.pivot_table(index="node", columns="pair", values="cka", aggfunc="mean")
    stds = cka.pivot_table(index="node", columns="pair", values="cka", aggfunc="std")
    means = means.reindex(index=NODE_ORDER, columns=PAIR_ORDER)
    stds = stds.reindex(index=NODE_ORDER, columns=PAIR_ORDER)
    matrix = means.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    for row in range(len(NODE_ORDER)):
        for column in range(len(PAIR_ORDER)):
            value = matrix[row, column]
            spread = float(stds.iloc[row, column])
            ax.text(
                column,
                row,
                f"{value:.3f}\n±{spread:.3f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value < 0.70 else "black",
            )
    ax.set_xticks(range(len(PAIR_ORDER)), PAIR_ORDER, rotation=25, ha="right")
    ax.set_yticks(range(len(NODE_ORDER)), NODE_ORDER)
    ax.set_xlabel("Model pair")
    ax.set_ylabel("Node")
    ax.set_title("Linear CKA across paired seeds (mean ± SD)")
    fig.colorbar(image, ax=ax, fraction=0.045, pad=0.04, label="CKA")
    fig.tight_layout()
    save_figure(fig, output_dir, "geometry_cka_multiseed")


def plot_rank_summaries(ranks: pd.DataFrame, output_dir: Path) -> None:
    metrics = [
        ("participation_rank", "Participation rank", False),
        ("entropy_rank", "Entropy rank", False),
        ("total_variance", "Total activation variance", True),
    ]
    x = np.arange(len(NODE_ORDER), dtype=float)
    offsets = {"Shared init": -0.10, "BP": 0.0, "PC-FP": 0.10}
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))
    for ax, (metric, title, log_scale) in zip(axes, metrics, strict=True):
        for role in ROLE_ORDER:
            rows = ranks[ranks["role"] == role]
            stats = rows.groupby("node")[metric].agg(["mean", "std"]).reindex(NODE_ORDER)
            ax.errorbar(
                x + offsets[role],
                stats["mean"],
                yerr=stats["std"],
                color=COLORS[role],
                marker=MARKERS[role],
                markerfacecolor="white" if role == "PC-FP" else COLORS[role],
                linestyle=LINESTYLES[role],
                linewidth=1.4,
                capsize=2.5,
                label=role,
            )
        ax.set_xticks(x, NODE_ORDER, rotation=45, ha="right")
        ax.set_title(title)
        ax.grid(alpha=0.18)
        if log_scale:
            ax.set_yscale("log")
    axes[0].set_ylabel("Effective rank")
    axes[1].set_ylabel("Effective rank")
    axes[2].set_ylabel("Variance (log scale)")
    axes[2].legend(frameon=False, loc="upper left")
    fig.suptitle("Representation spectrum summaries across seeds (mean ± SD)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, output_dir, "geometry_effective_rank_variance_multiseed")


def plot_spectra(
    spectra: pd.DataFrame,
    ranks: pd.DataFrame,
    output_dir: Path,
) -> None:
    totals = ranks[["seed", "node", "role", "total_variance"]]
    values = spectra.merge(totals, on=["seed", "node", "role"], how="left")
    values["variance_fraction"] = values["eigenvalue"] / values["total_variance"]
    fig, axes = plt.subplots(2, 4, figsize=(13.2, 6.8), sharex=True, sharey=True)
    for ax, node in zip(axes.flat, NODE_ORDER, strict=True):
        for role in ROLE_ORDER:
            rows = values[(values["node"] == node) & (values["role"] == role)]
            stats = (
                rows.groupby("index")["variance_fraction"]
                .agg(["mean", "std"])
                .reset_index()
                .sort_values("index")
            )
            x = stats["index"].to_numpy(dtype=float) + 1.0
            mean = stats["mean"].to_numpy(dtype=float)
            std = stats["std"].fillna(0.0).to_numpy(dtype=float)
            ax.plot(
                x,
                mean,
                color=COLORS[role],
                marker=MARKERS[role],
                markerfacecolor="white" if role == "PC-FP" else COLORS[role],
                markersize=2.8,
                linewidth=1.2,
                linestyle=LINESTYLES[role],
                label=role,
            )
            ax.fill_between(
                x,
                np.maximum(mean - std, np.finfo(float).tiny),
                mean + std,
                color=COLORS[role],
                alpha=0.10,
            )
        ax.set_title(node)
        ax.set_yscale("log")
        ax.grid(alpha=0.18)
    for ax in axes[-1, :]:
        ax.set_xlabel("Eigenvalue rank")
    for ax in axes[:, 0]:
        ax.set_ylabel("Fraction of total variance")
    handles, labels = axes.flat[-1].get_legend_handles_labels()
    axes[0, 3].legend(handles, labels, frameon=False, loc="upper right")
    fig.suptitle("Normalized activation eigenspectra across seeds", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, output_dir, "geometry_normalized_spectra_multiseed")


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
        "--output-root",
        type=Path,
        default=Path("runs/pcinterp/geometry/multiseed"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--top-eigenvalues", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must not contain duplicates")

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    dataset = load_frozen_dataset(args.dataset_dir)
    seed_dirs: dict[int, Path] = {}
    init_rows: list[dict[str, float]] = []
    for seed in args.seeds:
        seed_dir, difference = run_seed(
            seed,
            dataset=dataset,
            checkpoint_dir=args.checkpoint_dir,
            output_root=args.output_root,
            batch_size=args.batch_size,
            top_eigenvalues=args.top_eigenvalues,
            force=args.force,
        )
        seed_dirs[seed] = seed_dir
        init_rows.append({"seed": seed, "max_abs_difference": difference})

    cka, ranks, spectra = read_seed_outputs(seed_dirs)
    summarize(cka, ranks, spectra, init_rows, args.output_root)


if __name__ == "__main__":
    main()
