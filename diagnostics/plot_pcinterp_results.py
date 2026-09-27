#!/usr/bin/env python3
"""Create reproducible figures from the completed seed-0 pcinterp studies.

The script only reads the CSV artifacts written by pcinterp.  It writes each
figure as both a 300-dpi PNG and a vector PDF.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NODES = [
    "embed",
    "L0_mha",
    "L0_mlp1",
    "L0_mlp2",
    "L1_mha",
    "L1_mlp1",
    "L1_mlp2",
    "logits",
]
FACTORS = ["long_range_recurrence", "strict_induction"]
FACTOR_LABELS = {
    "long_range_recurrence": "Long-range recurrence",
    "strict_induction": "Strict induction",
}
MODEL_LABELS = {
    "shared_init_seed0": "Shared init",
    "bp_seed0_epoch5": "BP",
    "pc_fp_seed0_epoch5": "PC-FP",
}
MODEL_ORDER = ["Shared init", "BP", "PC-FP"]
MODEL_COLORS = {
    "Shared init": "#777777",
    "BP": "#2878B5",
    "PC-FP": "#D95319",
}
CONTROL_LABELS = {
    "current_char": "Current character",
    "trigram_distribution": "Trigram distribution",
}


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def save_figure(fig: plt.Figure, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def annotate_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    gates: np.ndarray | None = None,
    *,
    decimals: int = 3,
) -> None:
    finite = matrix[np.isfinite(matrix)]
    midpoint = float(np.nanmedian(finite)) if finite.size else 0.0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            if not np.isfinite(value):
                continue
            suffix = ""
            if gates is not None:
                suffix = "\nPASS" if bool(gates[row, col]) else "\nFAIL"
            color = "white" if value < midpoint else "black"
            ax.text(
                col,
                row,
                f"{value:.{decimals}f}{suffix}",
                ha="center",
                va="center",
                fontsize=7.5,
                color=color,
            )


def load_probe_frames(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    directories = [
        root / "probes" / "shared_init_seed0",
        root / "probes" / "bp_seed0_epoch5",
        root / "probes" / "pc_fp_seed0_epoch5",
    ]
    summaries = []
    curves = []
    for directory in directories:
        summary_path = directory / "probe_summary.csv"
        curve_path = directory / "probe_learning_curves.csv"
        if not summary_path.exists() or not curve_path.exists():
            raise FileNotFoundError(f"Missing probe output in {directory}")
        summaries.append(pd.read_csv(summary_path))
        curves.append(pd.read_csv(curve_path))
    summary = pd.concat(summaries, ignore_index=True)
    curve = pd.concat(curves, ignore_index=True)
    summary["model_label"] = summary["model"].map(MODEL_LABELS).fillna(summary["model"])
    curve["model_label"] = curve["model"].map(MODEL_LABELS).fillna(curve["model"])
    summary["gate_passed"] = as_bool(summary["gate_passed"])
    return summary, curve


def plot_probe_heatmaps(summary: pd.DataFrame, output_dir: Path) -> None:
    for metric, title, cmap, limits in [
        ("auroc", "Probe AUROC", "viridis", (0.45, 0.75)),
        ("selectivity", "Probe selectivity over shuffled labels", "coolwarm", (-0.05, 0.25)),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(9.8, 5.2), sharey=True)
        last_image = None
        for ax, factor in zip(axes, FACTORS, strict=True):
            subset = summary[summary["factor"] == factor]
            values = subset.pivot(index="node", columns="model_label", values=metric)
            gates = subset.pivot(index="node", columns="model_label", values="gate_passed")
            values = values.reindex(index=NODES, columns=MODEL_ORDER)
            gates = gates.reindex(index=NODES, columns=MODEL_ORDER).fillna(False)
            matrix = values.to_numpy(dtype=float)
            last_image = ax.imshow(
                matrix,
                aspect="auto",
                cmap=cmap,
                vmin=limits[0],
                vmax=limits[1],
            )
            annotate_heatmap(ax, matrix, gates.to_numpy(dtype=bool))
            ax.set_title(FACTOR_LABELS[factor])
            ax.set_xticks(range(len(MODEL_ORDER)), MODEL_ORDER, rotation=25, ha="right")
            ax.set_yticks(range(len(NODES)), NODES)
            ax.set_xlabel("Checkpoint")
        axes[0].set_ylabel("Node")
        fig.suptitle(f"{title} (seed 0)\nPASS/FAIL is the complete admissibility gate")
        fig.colorbar(last_image, ax=axes, fraction=0.025, pad=0.03, label=metric)
        fig.subplots_adjust(top=0.82, bottom=0.18, left=0.10, right=0.90, wspace=0.08)
        save_figure(fig, output_dir, f"probe_{metric}_seed0")


def plot_probe_learning_curves(curves: pd.DataFrame, output_dir: Path) -> None:
    node_colors = dict(zip(NODES, plt.cm.tab10(np.linspace(0.0, 0.9, len(NODES))), strict=True))
    for model_label in MODEL_ORDER:
        model_rows = curves[curves["model_label"] == model_label]
        if model_rows.empty:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharey=True)
        for ax, factor in zip(axes, FACTORS, strict=True):
            factor_rows = model_rows[model_rows["factor"] == factor]
            for node in NODES:
                rows = factor_rows[factor_rows["node"] == node]
                if rows.empty:
                    continue
                stats = rows.groupby("budget", as_index=False)["auroc"].agg(["mean", "std"]).reset_index()
                x = stats["budget"].to_numpy(dtype=float)
                mean = stats["mean"].to_numpy(dtype=float)
                std = stats["std"].fillna(0.0).to_numpy(dtype=float)
                ax.plot(x, mean, marker="o", ms=3, lw=1.4, color=node_colors[node], label=node)
                ax.fill_between(x, mean - std, mean + std, color=node_colors[node], alpha=0.10)
            ax.axhline(0.5, color="black", ls=":", lw=1, label="Chance")
            ax.axhline(0.6, color="black", ls="--", lw=1, label="AUROC gate")
            ax.set_xscale("log")
            ax.set_ylim(0.45, 0.75)
            ax.set_title(FACTOR_LABELS[factor])
            ax.set_xlabel("Probe training examples")
            ax.grid(alpha=0.18)
        axes[0].set_ylabel("Test AUROC")
        handles, labels = axes[-1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.99, 0.5), frameon=False)
        fig.suptitle(f"Probe sample efficiency: {model_label} (seed 0)")
        fig.tight_layout(rect=(0, 0, 0.86, 0.94))
        save_figure(fig, output_dir, f"probe_learning_curves_{slug(model_label)}_seed0")


def control_name(row: pd.Series) -> str:
    if str(row.get("node", "")) in CONTROL_LABELS:
        return CONTROL_LABELS[str(row["node"])]
    if str(row.get("model", "")) in CONTROL_LABELS:
        return CONTROL_LABELS[str(row["model"])]
    return str(row.get("node", row.get("model", "control")))


def plot_fixed_controls(root: Path, output_dir: Path) -> None:
    path = root / "probes" / "fixed_char_ngram" / "probe_summary.csv"
    frame = pd.read_csv(path)
    frame["control"] = frame.apply(control_name, axis=1)
    frame["gate_passed"] = as_bool(frame["gate_passed"])
    controls = list(dict.fromkeys(frame["control"].tolist()))
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), sharey=True)
    for ax, factor in zip(axes, FACTORS, strict=True):
        rows = frame[frame["factor"] == factor].set_index("control").reindex(controls)
        x = np.arange(len(controls))
        width = 0.34
        ax.bar(x - width / 2, rows["auroc"], width, label="True labels", color="#4C78A8")
        ax.bar(x + width / 2, rows["shuffled_auroc"], width, label="Shuffled", color="#BAB0AC")
        for index, (_, row) in enumerate(rows.iterrows()):
            verdict = "PASS" if bool(row["gate_passed"]) else "FAIL"
            ax.text(
                index,
                max(float(row["auroc"]), float(row["shuffled_auroc"])) + 0.012,
                f"sel. {float(row['selectivity']):+.3f}\n{verdict}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.axhline(0.5, color="black", ls=":", lw=1)
        ax.axhline(0.6, color="black", ls="--", lw=1)
        ax.set_xticks(x, controls, rotation=15, ha="right")
        ax.set_ylim(0.45, 0.75)
        ax.set_title(FACTOR_LABELS[factor])
        ax.grid(axis="y", alpha=0.18)
    axes[0].set_ylabel("Test AUROC")
    axes[-1].legend(frameon=False, loc="upper right")
    fig.suptitle("Fixed-feature probe controls")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, output_dir, "fixed_feature_controls_seed0")


def display_model(name: str) -> str:
    return MODEL_LABELS.get(name, name)


def pair_label(left: str, right: str) -> str:
    labels = {display_model(left), display_model(right)}
    if labels == {"Shared init", "BP"}:
        return "Init–BP"
    if labels == {"Shared init", "PC-FP"}:
        return "Init–PC-FP"
    if labels == {"BP", "PC-FP"}:
        return "BP–PC-FP"
    return f"{display_model(left)}–{display_model(right)}"


def plot_geometry(root: Path, output_dir: Path) -> None:
    directory = root / "geometry" / "final_seed0"
    cka = pd.read_csv(directory / "cka.csv")
    ranks = pd.read_csv(directory / "participation_rank.csv")
    spectra = pd.read_csv(directory / "spectrum.csv")

    cka["pair"] = [pair_label(left, right) for left, right in zip(cka["left"], cka["right"], strict=True)]
    pair_order = ["Init–BP", "Init–PC-FP", "BP–PC-FP"]
    table = cka.pivot(index="node", columns="pair", values="cka").reindex(index=NODES, columns=pair_order)
    matrix = table.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(5.8, 5.3))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    annotate_heatmap(ax, matrix, decimals=3)
    ax.set_xticks(range(len(pair_order)), pair_order, rotation=25, ha="right")
    ax.set_yticks(range(len(NODES)), NODES)
    ax.set_xlabel("Model pair")
    ax.set_ylabel("Node")
    ax.set_title("Linear CKA by node (seed 0)")
    fig.colorbar(image, ax=ax, fraction=0.045, pad=0.04, label="CKA")
    fig.tight_layout()
    save_figure(fig, output_dir, "geometry_cka_seed0")

    ranks["model_label"] = ranks["model"].map(MODEL_LABELS).fillna(ranks["model"])
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2))
    for model_label in MODEL_ORDER:
        rows = ranks[ranks["model_label"] == model_label].set_index("node").reindex(NODES)
        color = MODEL_COLORS[model_label]
        axes[0].plot(NODES, rows["participation_rank"], marker="o", color=color, label=model_label)
        axes[1].plot(NODES, rows["entropy_rank"], marker="o", color=color, label=model_label)
        axes[2].plot(NODES, rows["total_variance"], marker="o", color=color, label=model_label)
    axes[0].set_title("Participation rank")
    axes[1].set_title("Entropy rank")
    axes[2].set_title("Total activation variance")
    axes[2].set_yscale("log")
    for ax in axes:
        ax.tick_params(axis="x", rotation=45)
        ax.grid(alpha=0.18)
    axes[0].set_ylabel("Effective rank")
    axes[1].set_ylabel("Effective rank")
    axes[2].set_ylabel("Variance (log scale)")
    axes[-1].legend(frameon=False)
    fig.suptitle("Representation spectrum summaries (seed 0)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, output_dir, "geometry_effective_rank_variance_seed0")

    spectra["model_label"] = spectra["model"].map(MODEL_LABELS).fillna(spectra["model"])
    totals = ranks[["node", "model_label", "total_variance"]]
    spectra = spectra.merge(totals, on=["node", "model_label"], how="left")
    spectra["variance_fraction"] = spectra["eigenvalue"] / spectra["total_variance"]
    fig, axes = plt.subplots(2, 4, figsize=(13.0, 6.6), sharex=True, sharey=True)
    for ax, node in zip(axes.flat, NODES, strict=True):
        for model_label in MODEL_ORDER:
            rows = spectra[(spectra["node"] == node) & (spectra["model_label"] == model_label)].sort_values("index")
            if rows.empty:
                continue
            ax.plot(
                rows["index"] + 1,
                rows["variance_fraction"],
                marker="o",
                ms=2.8,
                lw=1.2,
                color=MODEL_COLORS[model_label],
                label=model_label,
            )
        ax.set_title(node)
        ax.set_yscale("log")
        ax.grid(alpha=0.18)
    for ax in axes[-1, :]:
        ax.set_xlabel("Eigenvalue rank")
    for ax in axes[:, 0]:
        ax.set_ylabel("Fraction of total variance")
    handles, labels = axes.flat[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Normalized activation eigenspectra (seed 0)")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(fig, output_dir, "geometry_normalized_spectra_seed0")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("runs/pcinterp"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/pcinterp/figures/single_seed"),
    )
    args = parser.parse_args()

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    summary, curves = load_probe_frames(args.root)
    plot_probe_heatmaps(summary, args.output_dir)
    plot_probe_learning_curves(curves, args.output_dir)
    plot_fixed_controls(args.root, args.output_dir)
    plot_geometry(args.root, args.output_dir)
    print(f"Saved seed-0 figures to {args.output_dir}")


if __name__ == "__main__":
    main()