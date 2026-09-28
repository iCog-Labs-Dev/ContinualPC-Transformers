#!/usr/bin/env python3
"""Aggregate same-checkpoint fixed-prediction gradient audits.

Expected inputs are the module and combined CSV files produced by
``examples/fixed_prediction_audit.py`` for shared initialization and paired BP
and PC-FP checkpoints.  The script computes directional and material credit
fronts, saves checkpoint-level tables, and creates compact paper diagnostics.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODULE_ORDER = (
    "embed",
    "L0_mha",
    "L0_mlp1",
    "L0_mlp2",
    "L1_mha",
    "L1_mlp1",
    "L1_mlp2",
    "logits",
)
METHOD_ORDER = ("Shared init", "BP", "PC-FP")
COLORS = {"Shared init": "#777777", "BP": "#2878B5", "PC-FP": "#D95319"}
LINESTYLES = {"Shared init": "-", "BP": "-", "PC-FP": "--"}


def checkpoint_identity(path: Path) -> tuple[str, int, str]:
    stem = path.name.removesuffix("_fixed_prediction_modules.csv")
    if stem.endswith("_init"):
        return "Shared init", 0, stem
    epoch_match = re.search(r"_epoch(\d+)$", stem)
    if epoch_match is None:
        raise ValueError(f"cannot infer checkpoint epoch from {path}")
    epoch = int(epoch_match.group(1))
    if stem.startswith("ckpt_bp_"):
        return "BP", epoch, stem
    if stem.startswith("ckpt_pc_fp_"):
        return "PC-FP", epoch, stem
    raise ValueError(f"cannot infer checkpoint method from {path}")


def discover(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    module_frames = []
    combined_frames = []
    paths = sorted(root.rglob("*_fixed_prediction_modules.csv"))
    if not paths:
        raise FileNotFoundError(f"no module audit CSVs beneath {root}")
    seen: set[tuple[str, int]] = set()
    for module_path in paths:
        method, epoch, checkpoint = checkpoint_identity(module_path)
        identity = (method, epoch)
        if identity in seen:
            raise ValueError(f"duplicate audit for {method} epoch {epoch}")
        seen.add(identity)

        combined_path = module_path.with_name(
            module_path.name.replace("_modules.csv", "_combined.csv")
        )
        if not combined_path.is_file():
            raise FileNotFoundError(combined_path)
        modules = pd.read_csv(module_path)
        combined = pd.read_csv(combined_path)
        for frame in (modules, combined):
            frame.insert(0, "checkpoint", checkpoint)
            frame.insert(0, "epoch", epoch)
            frame.insert(0, "method", method)
        module_frames.append(modules)
        combined_frames.append(combined)

    required = {("Shared init", 0)} | {
        (method, epoch) for method in ("BP", "PC-FP") for epoch in (1, 3, 5)
    }
    missing = required - seen
    if missing:
        raise FileNotFoundError(f"missing expected audits: {sorted(missing)}")
    return (
        pd.concat(module_frames, ignore_index=True),
        pd.concat(combined_frames, ignore_index=True),
    )


def first_passing_step(
    frame: pd.DataFrame,
    *,
    cosine_threshold: float,
    norm_threshold: float | None,
) -> float:
    valid = frame[frame["cosine_similarity"] >= cosine_threshold]
    if norm_threshold is not None:
        valid = valid[valid["norm_ratio_pc_over_bp"] >= norm_threshold]
    if valid.empty:
        return float("nan")
    return float(valid["inference_step"].min())


def credit_fronts(
    modules: pd.DataFrame,
    *,
    cosine_threshold: float,
    material_norm_threshold: float,
) -> pd.DataFrame:
    rows = []
    for (method, epoch, checkpoint, module), group in modules.groupby(
        ["method", "epoch", "checkpoint", "module"], sort=False
    ):
        rows.append(
            {
                "method": method,
                "epoch": epoch,
                "checkpoint": checkpoint,
                "module": module,
                "directional_front_step": first_passing_step(
                    group,
                    cosine_threshold=cosine_threshold,
                    norm_threshold=None,
                ),
                "material_front_step": first_passing_step(
                    group,
                    cosine_threshold=cosine_threshold,
                    norm_threshold=material_norm_threshold,
                ),
            }
        )
    return pd.DataFrame(rows)


def condition_label(method: str, epoch: int) -> str:
    return "Init" if method == "Shared init" else f"{method} e{epoch}"


def ordered_conditions(frame: pd.DataFrame) -> list[tuple[str, int]]:
    available = set(zip(frame["method"], frame["epoch"], strict=False))
    wanted = [("Shared init", 0)] + [
        (method, epoch) for epoch in (1, 3, 5) for method in ("BP", "PC-FP")
    ]
    return [condition for condition in wanted if condition in available]


def save_figure(fig: plt.Figure, directory: Path, name: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(directory / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_global(combined: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharex=True)
    for method, epoch in ordered_conditions(combined):
        rows = combined[
            (combined["method"] == method) & (combined["epoch"] == epoch)
        ].sort_values("inference_step")
        label = condition_label(method, epoch)
        alpha = 0.8 if epoch in (0, 5) else 0.45
        linewidth = 2.0 if epoch in (0, 5) else 1.2
        axes[0].plot(
            rows["inference_step"],
            rows["global_cosine"],
            color=COLORS[method],
            linestyle=LINESTYLES[method],
            alpha=alpha,
            linewidth=linewidth,
            label=label,
        )
        axes[1].plot(
            rows["inference_step"],
            rows["global_norm_ratio_pc_over_bp"],
            color=COLORS[method],
            linestyle=LINESTYLES[method],
            alpha=alpha,
            linewidth=linewidth,
            label=label,
        )
    axes[0].set_ylabel("Global PC–BP cosine")
    axes[1].set_ylabel("Global PC/BP gradient norm")
    for ax in axes:
        ax.set_xlabel("Inference sweep")
        ax.grid(alpha=0.18)
    axes[1].axhline(1.0, color="black", linestyle=":", linewidth=1.0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle("Same-checkpoint fixed-prediction gradient convergence", y=1.03)
    fig.tight_layout()
    save_figure(fig, output_dir, "global_gradient_convergence")


def heatmap(
    frame: pd.DataFrame,
    *,
    value: str,
    title: str,
    colorbar: str,
    output_dir: Path,
    filename: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    conditions = ordered_conditions(frame)
    matrix = np.full((len(conditions), len(MODULE_ORDER)), np.nan)
    for row_index, (method, epoch) in enumerate(conditions):
        selected = frame[(frame["method"] == method) & (frame["epoch"] == epoch)]
        values = selected.set_index("module")[value]
        for column, module in enumerate(MODULE_ORDER):
            if module in values:
                matrix[row_index, column] = values[module]

    fig, ax = plt.subplots(figsize=(9.4, 4.2))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(MODULE_ORDER)), MODULE_ORDER, rotation=35, ha="right")
    ax.set_yticks(
        range(len(conditions)),
        [condition_label(method, epoch) for method, epoch in conditions],
    )
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            number = matrix[row, column]
            if np.isfinite(number):
                label = f"{number:.3f}" if "step" not in value else f"{number:.0f}"
                ax.text(column, row, label, ha="center", va="center", fontsize=7)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label=colorbar)
    fig.tight_layout()
    save_figure(fig, output_dir, filename)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("runs/fixed_prediction_gradient_audit"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/fixed_prediction_gradient_audit/aggregate"),
    )
    parser.add_argument("--cosine-threshold", type=float, default=0.5)
    parser.add_argument("--material-norm-threshold", type=float, default=0.05)
    parser.add_argument("--selected-step", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not -1.0 <= args.cosine_threshold <= 1.0:
        raise ValueError("--cosine-threshold must be in [-1, 1]")
    if args.material_norm_threshold < 0.0:
        raise ValueError("--material-norm-threshold must be non-negative")

    modules, combined = discover(args.input_dir)
    fronts = credit_fronts(
        modules,
        cosine_threshold=args.cosine_threshold,
        material_norm_threshold=args.material_norm_threshold,
    )
    selected_modules = modules[
        modules["inference_step"] == args.selected_step
    ].copy()
    selected_combined = combined[
        combined["inference_step"] == args.selected_step
    ].copy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    modules.to_csv(args.output_dir / "modules_all_checkpoints.csv", index=False)
    combined.to_csv(args.output_dir / "global_all_checkpoints.csv", index=False)
    fronts.to_csv(args.output_dir / "credit_fronts.csv", index=False)
    selected_modules.to_csv(
        args.output_dir / f"module_summary_T{args.selected_step}.csv", index=False
    )
    selected_combined.to_csv(
        args.output_dir / f"global_summary_T{args.selected_step}.csv", index=False
    )

    plot_global(combined, args.output_dir / "figures")
    heatmap(
        selected_modules,
        value="cosine_similarity",
        title=f"Module gradient cosine at T={args.selected_step}",
        colorbar="Cosine similarity",
        output_dir=args.output_dir / "figures",
        filename=f"module_cosine_T{args.selected_step}",
        vmin=0.0,
        vmax=1.0,
    )
    heatmap(
        selected_modules,
        value="norm_ratio_pc_over_bp",
        title=f"Module PC/BP gradient norm at T={args.selected_step}",
        colorbar="PC/BP norm ratio",
        output_dir=args.output_dir / "figures",
        filename=f"module_norm_ratio_T{args.selected_step}",
        vmin=0.0,
        vmax=max(1.05, float(selected_modules["norm_ratio_pc_over_bp"].max())),
    )
    heatmap(
        fronts,
        value="material_front_step",
        title=(
            f"Material credit front (cosine ≥ {args.cosine_threshold:g}, "
            f"norm ratio ≥ {args.material_norm_threshold:g})"
        ),
        colorbar="Earliest inference sweep",
        output_dir=args.output_dir / "figures",
        filename="material_credit_front",
        vmin=0.0,
        vmax=float(modules["inference_step"].max()),
    )

    manifest = {
        "study": "fixed_prediction_gradient_audit_aggregate",
        "cosine_threshold": args.cosine_threshold,
        "material_norm_threshold": args.material_norm_threshold,
        "selected_step": args.selected_step,
        "module_order": list(MODULE_ORDER),
        "conditions": [
            {"method": method, "epoch": epoch}
            for method, epoch in ordered_conditions(modules)
        ],
        "notes": [
            "Directional fronts use cosine only.",
            "Material fronts additionally require a non-negligible PC/BP norm ratio.",
            "Gradient norms use the batch-by-sequence normalization in the source audits.",
        ],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print("\nGlobal summary at selected inference step:")
    columns = [
        "method",
        "epoch",
        "global_cosine",
        "global_norm_ratio_pc_over_bp",
        "global_pc_gradient_norm_over_parameter_norm",
        "global_bp_gradient_norm_over_parameter_norm",
    ]
    available = [column for column in columns if column in selected_combined]
    print(selected_combined[available].to_string(index=False))
    print("\nMaterial credit fronts:")
    print(
        fronts.pivot(
            index="module",
            columns=["method", "epoch"],
            values="material_front_step",
        ).reindex(MODULE_ORDER).to_string()
    )
    print(f"\nWrote aggregate audit to {args.output_dir}")


if __name__ == "__main__":
    main()
