"""
Aggregate multi-seed runs into the numbers and figures a paper table needs.

Reads the per-run files written by examples/transformer_v2_demo.py:
    runs/metrics/metrics_{mode}_{tokenizer}_seed{seed}.json   final metrics
    runs/plots/history_{mode}_{tokenizer}_seed{seed}.json     per-epoch curves

and reports mean +/- sample std across seeds for each (mode, tokenizer) arm,
plus seed-band curve plots that overlay the PC and backprop arms.

Usage:
    # all arms found on disk, char only
    PYTHONPATH=. python diagnostics/aggregate_seeds.py --tokenizer char

    # every tokenizer, and skip the plots
    PYTHONPATH=. python diagnostics/aggregate_seeds.py --no_plots

    # markdown table ready to paste into the paper
    PYTHONPATH=. python diagnostics/aggregate_seeds.py --tokenizer char --markdown
"""

import argparse
import glob
import json
import os
import statistics
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

# (json key, printed label, format spec). Order controls table row order.
METRIC_ROWS = [
    ("final_train_ppl", "Final train perplexity", ".2f"),
    ("test_ppl", "Final test perplexity", ".2f"),
    ("test_accuracy_pct", "Test accuracy (%)", ".2f"),
    ("test_loss", "Test CE loss", ".4f"),
    ("final_train_energy", "Final train energy", ".2f"),
    ("train_seconds", "Training time (s)", ".0f"),
]


def mean_std(values):
    """Mean and sample std. Std is None for a single seed (undefined)."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None
    mean = statistics.fmean(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else None
    return mean, std


def fmt(mean, std, spec):
    if mean is None:
        return "n/a"
    if std is None:
        return f"{mean:{spec}}"
    return f"{mean:{spec}} ± {std:{spec}}"


def load_metrics(metrics_dir, tokenizer):
    """Group final-metric files into arms keyed by (mode, tokenizer)."""
    arms = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(metrics_dir, "metrics_*.json"))):
        with open(path) as fh:
            m = json.load(fh)
        if tokenizer and m.get("tokenizer") != tokenizer:
            continue
        # Percent accuracy is what the table reports; derive it once here.
        m["test_accuracy_pct"] = m["test_accuracy"] * 100.0
        arms[(m["mode"], m["tokenizer"])].append(m)
    return arms


def load_histories(plot_dir, mode, tokenizer, seeds):
    """Per-epoch curves for one arm, restricted to the seeds that completed."""
    histories = []
    for seed in seeds:
        path = os.path.join(plot_dir, f"history_{mode}_{tokenizer}_seed{seed}.json")
        if os.path.exists(path):
            with open(path) as fh:
                histories.append(json.load(fh))
    return histories


def curve_band(histories, key):
    """Per-epoch (epochs, mean, lo, hi) across seeds, truncated to the shortest run."""
    series = [h[key] for h in histories if h.get(key)]
    if not series:
        return None
    n = min(len(s) for s in series)
    epochs = list(range(1, n + 1))
    means, los, his = [], [], []
    for i in range(n):
        vals = [s[i] for s in series]
        means.append(statistics.fmean(vals))
        los.append(min(vals))
        his.append(max(vals))
    return epochs, means, los, his


def plot_arm_comparison(arms, plot_dir, out_dir, tokenizer):
    """Overlay PC vs backprop with a min/max band across seeds."""
    colors = {"pc": "tab:blue", "backprop": "tab:orange"}
    written = []

    for key, ylabel, fname in [
        ("test_ppl", "Test Perplexity", "seeds_test_ppl"),
        ("train_ppl", "Train Perplexity", "seeds_train_ppl"),
        ("train_energy", "Train Energy", "seeds_train_energy"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        plotted = False

        for (mode, tok), runs in sorted(arms.items()):
            if tokenizer and tok != tokenizer:
                continue
            seeds = sorted(r["seed"] for r in runs)
            histories = load_histories(plot_dir, mode, tok, seeds)
            band = curve_band(histories, key)
            if band is None:
                continue
            epochs, means, los, his = band
            color = colors.get(mode, "tab:green")
            label = f"{mode} ({len(histories)} seeds)"
            ax.plot(epochs, means, "o-", color=color, label=label)
            if len(histories) > 1:
                ax.fill_between(epochs, los, his, color=color, alpha=0.18)
            plotted = True

        if not plotted:
            plt.close(fig)
            continue

        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        suffix = f" — {tokenizer}" if tokenizer else ""
        ax.set_title(f"{ylabel} across seeds (mean, min-max band){suffix}")
        ax.legend()
        fig.tight_layout()
        out = os.path.join(out_dir, f"{fname}{'_' + tokenizer if tokenizer else ''}.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        written.append(out)

    return written


def print_table(arms, markdown):
    for (mode, tok), runs in sorted(arms.items()):
        seeds = sorted(r["seed"] for r in runs)
        n_params = runs[0]["n_params"]
        print(f"\n=== {mode} / {tok} — {len(runs)} seed(s): {seeds} ===")
        print(f"parameters: {n_params:,}")
        if len(runs) == 1:
            print("(single seed — std undefined, shown as bare value)")

        rows = []
        for key, label, spec in METRIC_ROWS:
            mean, std = mean_std([r.get(key) for r in runs])
            if mean is None:
                continue  # e.g. energy on the backprop arm
            rows.append((label, fmt(mean, std, spec)))

        if markdown:
            print(f"\n| Metric | {mode} ({len(runs)} seeds) |")
            print("|---|---|")
            for label, value in rows:
                print(f"| {label} | {value} |")
        else:
            width = max(len(label) for label, _ in rows)
            for label, value in rows:
                print(f"  {label:<{width}}  {value}")


def main():
    p = argparse.ArgumentParser(
        description="Aggregate multi-seed runs into mean +/- std and seed-band plots"
    )
    p.add_argument("--metrics_dir", default="runs/metrics")
    p.add_argument("--plot_dir", default="runs/plots")
    p.add_argument("--out_dir", default="runs/plots")
    p.add_argument(
        "--tokenizer",
        default=None,
        choices=["char", "bpe"],
        help="Restrict to one tokenizer (default: report every arm found)",
    )
    p.add_argument("--markdown", action="store_true", help="Emit markdown tables")
    p.add_argument("--no_plots", action="store_true", help="Skip the seed-band plots")
    args = p.parse_args()

    arms = load_metrics(args.metrics_dir, args.tokenizer)
    if not arms:
        raise SystemExit(
            f"No metrics_*.json found in {args.metrics_dir}"
            + (f" for tokenizer={args.tokenizer}" if args.tokenizer else "")
            + ". Run examples/transformer_v2_demo.py first."
        )

    print_table(arms, args.markdown)

    if not args.no_plots:
        os.makedirs(args.out_dir, exist_ok=True)
        written = plot_arm_comparison(arms, args.plot_dir, args.out_dir, args.tokenizer)
        for path in written:
            print(f"\nwrote {path}")


if __name__ == "__main__":
    main()