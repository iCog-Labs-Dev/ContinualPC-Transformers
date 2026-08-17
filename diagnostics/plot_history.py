"""
Re-plot training/test curves from saved history_*.json files — no retraining.

Usage:
    # one run: stacked train (ppl + energy) and a test-ppl plot
    python diagnostics/plot_history.py runs/plots/history_pc_bpe.json

    # compare several runs on one test-ppl axis (e.g. PC vs backprop)
    python diagnostics/plot_history.py runs/plots/history_pc_bpe.json runs/plots/history_backprop_bpe.json
"""

import os
import sys
import json
import argparse

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


def load(path):
    with open(path) as fh:
        h = json.load(fh)
    # tag = filename without "history_" prefix and ".json" suffix
    tag = os.path.basename(path)
    if tag.startswith("history_"):
        tag = tag[len("history_"):]
    if tag.endswith(".json"):
        tag = tag[: -len(".json")]
    return tag, h


def plot_single(tag, h, out_dir):
    epochs = h["epochs"]

    # --- Training: perplexity (top) + energy (bottom), separate panels ---
    fig, (ax_ppl, ax_en) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    ax_ppl.plot(epochs, h["train_ppl"], "o-", color="tab:blue")
    ax_ppl.set_ylabel("Train Perplexity")
    ax_ppl.set_title(f"Training ({tag})")

    if h.get("train_energy"):  # PC only; empty for backprop
        ax_en.plot(epochs, h["train_energy"], "s--", color="tab:red")
        ax_en.set_ylabel("Train Energy")
    else:
        ax_en.set_visible(False)

    ax_en.set_xlabel("Epoch")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"train_{tag}.png"), dpi=120)
    plt.close(fig)

    # --- Test perplexity ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, h["test_ppl"], "o-", color="tab:green")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Perplexity")
    ax.set_title(f"Test Perplexity ({tag})")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"test_{tag}.png"), dpi=120)
    plt.close(fig)


def plot_compare(runs, out_dir):
    """Overlay test perplexity for multiple runs on one axis."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for tag, h in runs:
        ax.plot(h["epochs"], h["test_ppl"], "o-", label=tag)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Perplexity")
    ax.set_title("Test Perplexity comparison")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(out_dir, "test_compare.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main():
    p = argparse.ArgumentParser(description="Re-plot from history_*.json (no retraining)")
    p.add_argument("history", nargs="+", help="one or more history_*.json files")
    p.add_argument("--out_dir", default="runs/plots", help="where to write PNGs")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    runs = [load(path) for path in args.history]

    for tag, h in runs:
        plot_single(tag, h, args.out_dir)
        print(f"wrote train_{tag}.png, test_{tag}.png")

    if len(runs) > 1:
        out = plot_compare(runs, args.out_dir)
        print(f"wrote {os.path.basename(out)} (overlay of {len(runs)} runs)")


if __name__ == "__main__":
    main()