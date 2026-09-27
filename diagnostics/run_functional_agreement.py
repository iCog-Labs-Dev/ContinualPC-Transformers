#!/usr/bin/env python3
"""Measure paired predictive agreement between BP and PC-FP checkpoints.

The study evaluates the final-position next-character distribution on the
leakage-safe test split of the frozen ``char_v1`` interpretation dataset.  BP
and PC-FP always see the same examples, and per-example measurements are saved
so that later statistical analyses do not require another model evaluation.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from diagnostics.pcinterp_adapter import (
    FabricPCTransformerV2Adapter,
    load_frozen_dataset,
)


def checkpoint_paths(checkpoint_dir: Path, seed: int) -> tuple[Path, Path]:
    return (
        checkpoint_dir / f"ckpt_bp_char_seed{seed}_epoch5.pkl",
        checkpoint_dir
        / f"ckpt_pc_fp_T20_eta0.3864_char_seed{seed}_epoch5.pkl",
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def log_softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    maximum = np.max(values, axis=-1, keepdims=True)
    shifted = values - maximum
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def rankdata(values: np.ndarray) -> np.ndarray:
    """Return average ranks without requiring SciPy."""
    return pd.Series(np.asarray(values)).rank(method="average").to_numpy()


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def evaluate_pair(
    seed: int,
    bp_path: Path,
    pc_path: Path,
    inputs: np.ndarray,
    targets: np.ndarray,
    source_rows: np.ndarray,
    *,
    batch_size: int,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    bp = FabricPCTransformerV2Adapter(
        bp_path, name=f"bp_seed{seed}_epoch5", batch_size=batch_size
    )
    pc = FabricPCTransformerV2Adapter(
        pc_path, name=f"pc_fp_seed{seed}_epoch5", batch_size=batch_size
    )

    bp_logits = np.asarray(bp.logits(inputs), dtype=np.float64)
    pc_logits = np.asarray(pc.logits(inputs), dtype=np.float64)
    if bp_logits.shape != pc_logits.shape:
        raise AssertionError(
            f"seed {seed}: BP/PC-FP logit shapes differ: "
            f"{bp_logits.shape} versus {pc_logits.shape}"
        )

    bp_logp = log_softmax(bp_logits)
    pc_logp = log_softmax(pc_logits)
    bp_prob = np.exp(bp_logp)
    pc_prob = np.exp(pc_logp)
    mixture = 0.5 * (bp_prob + pc_prob)
    log_mixture = np.log(np.maximum(mixture, np.finfo(np.float64).tiny))
    js = 0.5 * (
        np.sum(bp_prob * (bp_logp - log_mixture), axis=-1)
        + np.sum(pc_prob * (pc_logp - log_mixture), axis=-1)
    )

    rows = np.arange(len(targets))
    bp_nll = -bp_logp[rows, targets]
    pc_nll = -pc_logp[rows, targets]
    bp_prediction = np.argmax(bp_logits, axis=-1)
    pc_prediction = np.argmax(pc_logits, axis=-1)
    agreement = bp_prediction == pc_prediction
    bp_correct = bp_prediction == targets
    pc_correct = pc_prediction == targets

    examples = pd.DataFrame(
        {
            "seed": seed,
            "dataset_row": source_rows,
            "target": targets,
            "bp_prediction": bp_prediction,
            "pc_fp_prediction": pc_prediction,
            "top1_agreement": agreement.astype(np.int8),
            "bp_correct": bp_correct.astype(np.int8),
            "pc_fp_correct": pc_correct.astype(np.int8),
            "bp_nll": bp_nll,
            "pc_fp_nll": pc_nll,
            "pc_fp_minus_bp_nll": pc_nll - bp_nll,
            "js_divergence_nats": js,
        }
    )

    summary: dict[str, float | int] = {
        "seed": seed,
        "n_examples": len(targets),
        "top1_agreement": float(np.mean(agreement)),
        "both_correct": float(np.mean(bp_correct & pc_correct)),
        "bp_only_correct": float(np.mean(bp_correct & ~pc_correct)),
        "pc_fp_only_correct": float(np.mean(pc_correct & ~bp_correct)),
        "both_incorrect": float(np.mean(~bp_correct & ~pc_correct)),
        "bp_accuracy": float(np.mean(bp_correct)),
        "pc_fp_accuracy": float(np.mean(pc_correct)),
        "bp_mean_nll": float(np.mean(bp_nll)),
        "pc_fp_mean_nll": float(np.mean(pc_nll)),
        "pc_fp_minus_bp_mean_nll": float(np.mean(pc_nll - bp_nll)),
        "mean_js_divergence_nats": float(np.mean(js)),
        "median_js_divergence_nats": float(np.median(js)),
        "nll_pearson": correlation(bp_nll, pc_nll),
        "nll_spearman": correlation(rankdata(bp_nll), rankdata(pc_nll)),
    }
    del bp, pc, bp_logits, pc_logits, bp_logp, pc_logp, bp_prob, pc_prob
    gc.collect()
    return summary, examples


def aggregate_seed_metrics(seed_frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [column for column in seed_frame.columns if column not in {"seed", "n_examples"}]
    rows = []
    for metric in metrics:
        values = seed_frame[metric].to_numpy(dtype=np.float64)
        count = int(np.sum(np.isfinite(values)))
        mean = float(np.nanmean(values))
        std = float(np.nanstd(values, ddof=1)) if count > 1 else float("nan")
        se = std / np.sqrt(count) if count > 1 else float("nan")
        rows.append(
            {
                "metric": metric,
                "mean": mean,
                "std": std,
                "se": se,
                "normal_95ci_low": mean - 1.96 * se,
                "normal_95ci_high": mean + 1.96 * se,
                "n_seeds": count,
            }
        )
    return pd.DataFrame(rows)


def save_figures(seed_frame: pd.DataFrame, examples: pd.DataFrame, output_dir: Path) -> None:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.7))
    axes[0].plot(seed_frame["seed"], 100.0 * seed_frame["top1_agreement"], "o-")
    axes[0].set(xlabel="Seed", ylabel="Top-1 agreement (%)")
    axes[0].grid(alpha=0.25)
    axes[1].plot(
        seed_frame["seed"], seed_frame["mean_js_divergence_nats"], "o-", color="#D95319"
    )
    axes[1].set(xlabel="Seed", ylabel="Mean JS divergence (nats)")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "agreement_by_seed.png", dpi=300, bbox_inches="tight")
    fig.savefig(figure_dir / "agreement_by_seed.pdf", bbox_inches="tight")
    plt.close(fig)

    # A deterministic subset keeps the vector figure compact while preserving
    # the full per-example table for quantitative analysis.
    scatter = examples.iloc[:: max(1, len(examples) // 10_000)]
    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    ax.scatter(scatter["bp_nll"], scatter["pc_fp_nll"], s=5, alpha=0.15)
    limit = float(max(scatter["bp_nll"].max(), scatter["pc_fp_nll"].max()))
    ax.plot([0.0, limit], [0.0, limit], "--", color="black", linewidth=1.0)
    ax.set(xlabel="BP per-example NLL", ylabel="PC-FP per-example NLL")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figure_dir / "paired_nll.png", dpi=300, bbox_inches="tight")
    fig.savefig(figure_dir / "paired_nll.pdf", bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=Path("runs/checkpoints")
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("artifacts/interpretation/char_v1"),
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/functional_agreement/final_multiseed"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # This validates the frozen artifact and supplies the canonical test rows.
    dataset = load_frozen_dataset(args.dataset_dir)
    test_rows = np.asarray(dataset.split.test, dtype=np.int64)
    with np.load(args.dataset_dir / "dataset.npz", allow_pickle=False) as saved:
        inputs = np.asarray(saved["inputs"][test_rows], dtype=np.int32)
        targets = np.asarray(saved["targets"][test_rows], dtype=np.int32)

    summaries = []
    example_frames = []
    checkpoint_records = []
    for seed in args.seeds:
        bp_path, pc_path = checkpoint_paths(args.checkpoint_dir, seed)
        missing = [str(path) for path in (bp_path, pc_path) if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing checkpoints:\n" + "\n".join(missing))
        print(f"seed {seed}: evaluating BP and PC-FP")
        summary, examples = evaluate_pair(
            seed,
            bp_path,
            pc_path,
            inputs,
            targets,
            test_rows,
            batch_size=args.batch_size,
        )
        summaries.append(summary)
        example_frames.append(examples)
        checkpoint_records.extend(
            [
                {"seed": seed, "method": "BP", "path": str(bp_path), "sha256": file_sha256(bp_path)},
                {"seed": seed, "method": "PC-FP", "path": str(pc_path), "sha256": file_sha256(pc_path)},
            ]
        )
        print(
            f"seed {seed}: agreement={summary['top1_agreement']:.4f} "
            f"JS={summary['mean_js_divergence_nats']:.6f} "
            f"NLL r={summary['nll_pearson']:.4f} "
            f"delta_NLL={summary['pc_fp_minus_bp_mean_nll']:+.6f}"
        )

    seed_frame = pd.DataFrame(summaries).sort_values("seed")
    examples = pd.concat(example_frames, ignore_index=True)
    aggregate = aggregate_seed_metrics(seed_frame)
    seed_frame.to_csv(args.output_dir / "seed_summary.csv", index=False)
    aggregate.to_csv(args.output_dir / "aggregate_summary.csv", index=False)
    examples.to_csv(args.output_dir / "per_example.csv", index=False)
    pd.DataFrame(checkpoint_records).to_csv(
        args.output_dir / "checkpoints.csv", index=False
    )
    save_figures(seed_frame, examples, args.output_dir)

    manifest = {
        "study": "paired_bp_pc_fp_functional_agreement",
        "seeds": args.seeds,
        "dataset": str(args.dataset_dir / "dataset.npz"),
        "dataset_sha256": file_sha256(args.dataset_dir / "dataset.npz"),
        "split": "test",
        "n_examples_per_seed": len(test_rows),
        "prediction_position": "final context position",
        "distribution_units": "natural-log probabilities; JS divergence in nats",
        "dropout": "disabled by deterministic feed-forward adapter",
        "outputs": [
            "seed_summary.csv",
            "aggregate_summary.csv",
            "per_example.csv",
            "checkpoints.csv",
            "figures/agreement_by_seed.{png,pdf}",
            "figures/paired_nll.{png,pdf}",
        ],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print("\nFunctional agreement across seeds:")
    for row in aggregate.itertuples(index=False):
        if row.metric in {
            "top1_agreement",
            "mean_js_divergence_nats",
            "nll_pearson",
            "nll_spearman",
            "pc_fp_minus_bp_mean_nll",
        }:
            print(
                f"{row.metric:<30} mean={row.mean:.6f} "
                f"std={row.std:.6f} n={row.n_seeds}"
            )
    print(f"Wrote study to {args.output_dir}")


if __name__ == "__main__":
    main()
