"""Figures for the PC BERT diagnostics: sweeps, PC-vs-BP alignment and stability.

Reads the result files written by ``examples/pc_reach_sweep.sh``, the
``bert_demo.py --pc_bp_probe`` outputs and the training logs, and writes PNGs
to ``--out``. Missing inputs are skipped.

Usage:
    python examples/plotting/plot_pc_bert_diagnostics.py --root . --out results/figures
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e1"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
NEUTRAL = "#8a8986"

STEP_RE = re.compile(r"^step=(\d+) masked_loss=([0-9.]+)(?: grad_norm=([0-9.eE+-]+))?")
FINAL_RE = re.compile(r"final_validation_masked_loss=([0-9.]+)")
PROBE_RE = re.compile(r"^pc_bp node=(\S+) .*cosine_mean=(-?[0-9.]+)")
HEADER_RE = re.compile(r"^(steps=\S+ eta=\S+)")


def style_axes(ax) -> None:
    """Apply the recessive grid and ink colours."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(NEUTRAL)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)


def new_figure(width: float, height: float, nrows: int = 1, ncols: int = 1, **kw):
    """Create a figure on the chart surface."""
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height), **kw)
    fig.patch.set_facecolor(SURFACE)
    for ax in fig.axes:
        style_axes(ax)
    return fig, axes


def save(fig, out: Path, name: str, title: str) -> None:
    """Title the figure with its finding and write it as a PNG."""
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = out / name
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path}")


def read_log(path: Path) -> dict:
    """Parse a bert_demo training log into steps, losses, grad norms and final loss."""
    steps, losses, grads, final = [], [], [], None
    for line in path.read_text().splitlines():
        match = STEP_RE.match(line)
        if match:
            steps.append(int(match.group(1)))
            losses.append(float(match.group(2)))
            grads.append(float(match.group(3)) if match.group(3) else None)
        final_match = FINAL_RE.search(line)
        if final_match:
            final = float(final_match.group(1))
    spikes = sum(1 for g in grads if g is not None and g > 10)
    return {
        "steps": steps,
        "loss": losses,
        "grad": grads,
        "final": final,
        "spikes": spikes,
    }


def rolling(values: list[float], window: int) -> list[float]:
    """Trailing rolling mean."""
    out = []
    for i in range(len(values)):
        chunk = values[max(0, i - window + 1) : i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def read_probe(path: Path) -> dict[str, dict[str, float]]:
    """Parse probe output into {setting header: {node: cosine}}; no header -> 'all'."""
    blocks: dict[str, dict[str, float]] = {}
    current = "all"
    for line in path.read_text().splitlines():
        header = HEADER_RE.match(line)
        if header:
            current = header.group(1)
            continue
        match = PROBE_RE.match(line)
        if match:
            blocks.setdefault(current, {})[match.group(1)] = float(match.group(2))
    return blocks


def fig_sweep(root: Path, out: Path) -> None:
    """2k-update validation loss per config: seeds as dots, mean as a bar tick."""
    rows = []
    for name in (
        "results/pc_reach_sweep/summary.csv",
        "results/pc_long_inference/summary.csv",
    ):
        path = root / name
        if path.exists():
            with path.open() as handle:
                rows.extend(r for r in csv.DictReader(handle) if r["val_masked_loss"])
    if not rows:
        return
    groups: dict[str, list[float]] = {}
    for row in rows:
        label = (
            "BP"
            if row["mode"] == "backprop"
            else f"PC {row['infer_steps']} × {row['eta_infer']}"
        )
        groups.setdefault(label, [])
        if row["mode"] == "backprop" and len(groups[label]) >= 3:
            continue
        groups[label].append(float(row["val_masked_loss"]))
    order = sorted(groups, key=lambda k: statistics.mean(groups[k]), reverse=True)
    fig, ax = new_figure(8, 0.42 * len(order) + 1.4)
    bp_mean = statistics.mean(groups["BP"]) if "BP" in groups else None
    if bp_mean is not None:
        ax.axvline(bp_mean, color=SERIES[0], linewidth=1.5, linestyle="--")
        ax.text(
            bp_mean, len(order) - 0.4, " BP mean", color=INK_2, fontsize=8, va="bottom"
        )
    for y, label in enumerate(order):
        values = groups[label]
        colour = SERIES[0] if label == "BP" else SERIES[1]
        ax.scatter(
            values,
            [y] * len(values),
            s=28,
            color=colour,
            edgecolor=SURFACE,
            linewidth=1,
            zorder=3,
        )
        ax.plot(
            [statistics.mean(values)] * 2,
            [y - 0.28, y + 0.28],
            color=INK,
            linewidth=2,
            zorder=4,
        )
    ax.set_yticks(range(len(order)), order)
    ax.set_xlabel("Validation masked loss after 2,000 updates (lower is better)")
    save(
        fig,
        out,
        "fig1_sweep_2k.png",
        "No PC setting matched BP at 2k updates; long inference varied widely by seed",
    )


def fig_alignment(root: Path, out: Path) -> None:
    """PC-vs-BP cosine by layer at 12 layers, residual path and mlp1 side by side."""
    specs = [
        ("depth12_pc_bp.txt", "steps=5 eta=0.01", "5 × 0.01, float32"),
        ("depth12_long_inference.txt", "steps=120 eta=0.3", "120 × 0.3, float32"),
        ("depth12_longer.txt", "steps=300 eta=0.2", "300 × 0.2, float32"),
        ("x64_120_03.txt", "all", "120 × 0.3, float64"),
    ]
    series = []
    for filename, key, label in specs:
        path = root / filename
        if path.exists():
            block = read_probe(path).get(key)
            if block:
                series.append((label, block))
    if not series:
        return
    layers = sorted(
        {
            int(m.group(1))
            for _, b in series
            for n in b
            if (m := re.match(r"L(\d+)_", n))
        }
    )
    fig, (left, right) = new_figure(11, 4.6, 1, 2, sharey=True)
    xs_left = list(range(len(layers) + 1))
    for idx, (label, block) in enumerate(series):
        colour = SERIES[idx % len(SERIES)]
        ys = [block.get("token_embeddings")] + [block.get(f"L{i}_mlp2") for i in layers]
        left.plot(
            xs_left,
            ys,
            color=colour,
            linewidth=2,
            marker="o",
            markersize=4,
            label=label,
        )
        ys_mlp1 = [block.get(f"L{i}_mlp1") for i in layers]
        right.plot(layers, ys_mlp1, color=colour, linewidth=2, marker="o", markersize=4)
    left.set_xticks(xs_left, ["emb"] + [f"L{i}" for i in layers])
    left.set_xlim(-0.5, xs_left[-1] + 0.5)
    left.set_title(
        "Residual path (token embeddings, then each layer's mlp2)",
        color=INK,
        fontsize=10,
        loc="left",
    )
    right.set_xticks(layers, [f"L{i}" for i in layers])
    right.set_title("mlp1 in each layer", color=INK, fontsize=10, loc="left")
    left.set_ylabel("Cosine of PC vs BP weight gradient")
    left.set_ylim(-0.15, 1.05)
    left.legend(loc="lower right", frameon=False, fontsize=8)
    save(
        fig,
        out,
        "fig2_alignment_depth12.png",
        "At 12 layers PC matches BP down the network only in float64; mlp1 never aligns",
    )


def fig_stability(root: Path, out: Path) -> None:
    """Gradient norm and smoothed training loss for the collapse-diagnosis runs."""
    runs = [
        ("A_base.log", "A: lr 3e-4, 60 × 0.2"),
        ("B_lr1e-4.log", "B: lr 1e-4, 60 × 0.2"),
        ("C_eta0.1.log", "C: lr 3e-4, 120 × 0.1"),
    ]
    logs = [
        (label, read_log(root / "results/pc_collapse" / name))
        for name, label in runs
        if (root / "results/pc_collapse" / name).exists()
    ]
    if not logs:
        return
    fig, (top, bottom) = new_figure(10, 6.4, 2, 1, sharex=True)
    for idx, (label, log) in enumerate(logs):
        colour = SERIES[idx % len(SERIES)]
        points = [(s, g) for s, g in zip(log["steps"], log["grad"]) if g is not None]
        if points:
            top.plot(
                [p[0] for p in points],
                [p[1] for p in points],
                color=colour,
                linewidth=1.2,
                label=f"{label} ({log['spikes']} spikes)",
            )
        bottom.plot(
            log["steps"],
            rolling(log["loss"], 10),
            color=colour,
            linewidth=2,
            label=f"{label}, final val {log['final']:.3f}" if log["final"] else label,
        )
    top.set_yscale("log")
    top.axhline(10, color=NEUTRAL, linewidth=1, linestyle=":")
    top.set_ylabel("Gradient norm before clipping")
    top.legend(frameon=False, fontsize=8, loc="upper left")
    bottom.set_ylabel("Training masked loss (rolling mean)")
    bottom.set_xlabel("Update")
    bottom.legend(frameon=False, fontsize=8, loc="upper right")
    save(
        fig,
        out,
        "fig3_stability_runs.png",
        "Inference blow-ups appear as gradient spikes; run A collapses after they become chronic",
    )


def fig_spikes_vs_loss(root: Path, out: Path) -> None:
    """Final validation loss against spike count for every PC run with norm logging."""
    points = []
    for folder, length in (
        ("results/pc_collapse", "2k updates"),
        ("results/pc_10k", "10k updates"),
    ):
        paths = sorted((root / folder).glob("pc_*.log")) + sorted(
            (root / folder).glob("[A-D]_*.log")
        )
        for path in paths:
            if path.stem.endswith("_gpu1"):
                continue
            log = read_log(path)
            if log["final"] is not None and any(g is not None for g in log["grad"]):
                points.append((length, log["spikes"], log["final"]))
    if not points:
        return
    fig, ax = new_figure(8, 4.6)
    for idx, length in enumerate(("2k updates", "10k updates")):
        sel = [(s, l) for g, s, l in points if g == length]
        if sel:
            ax.scatter(
                [s + 1 for s, _ in sel],
                [l for _, l in sel],
                s=48,
                color=SERIES[idx + 1],
                edgecolor=SURFACE,
                linewidth=1.5,
                label=f"PC, {length}",
                zorder=3,
            )
    bp_10k = [
        read_log(p)["final"] for p in sorted((root / "results/pc_10k").glob("bp_*.log"))
    ]
    bp_10k = [v for v in bp_10k if v is not None]
    references = []
    sweep = root / "results/pc_reach_sweep/summary.csv"
    if sweep.exists():
        with sweep.open() as handle:
            bp_2k = [
                float(r["val_masked_loss"])
                for r in csv.DictReader(handle)
                if r["mode"] == "backprop" and r["val_masked_loss"]
            ]
        if bp_2k:
            references.append(("BP, 2k updates", statistics.mean(bp_2k)))
    if bp_10k:
        references.append(("BP, 10k updates", statistics.mean(bp_10k)))
    for label, value in references:
        ax.axhline(value, color=SERIES[0], linewidth=1.5, linestyle="--")
        ax.text(
            0.99,
            value,
            f"{label} ({value:.2f})",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="bottom",
            color=INK_2,
            fontsize=8,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Gradient spikes during training (grad norm > 10), plus 1")
    ax.set_ylabel("Final validation masked loss")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    save(
        fig,
        out,
        "fig4_spikes_vs_loss.png",
        "Every collapsed PC run had over 100 gradient spikes",
    )


def fig_training_10k(root: Path, out: Path) -> None:
    """Smoothed training loss over 10k updates: BP vs the two PC configs."""
    folder = root / "results/pc_10k"
    groups = [
        ("bp_seed*.log", "BP, lr 3e-4", SERIES[0]),
        ("pc_lr1e-4_s120_e0.1_seed*.log", "PC, lr 1e-4, 120 × 0.1", SERIES[1]),
        ("pc_lr2e-4_s240_e0.05_seed*.log", "PC, lr 2e-4, 240 × 0.05", SERIES[2]),
    ]
    fig, ax = new_figure(10, 4.8)
    drawn = False
    for pattern, label, colour in groups:
        for idx, path in enumerate(sorted(folder.glob(pattern))):
            log = read_log(path)
            if not log["steps"]:
                continue
            seed = re.search(r"seed(\d+)", path.name).group(1)
            final = f", final val {log['final']:.2f}" if log["final"] else ""
            ax.plot(
                log["steps"],
                rolling(log["loss"], 20),
                color=colour,
                linewidth=2,
                linestyle="-" if idx == 0 else "--",
                label=f"{label}, seed {seed}{final}",
            )
            drawn = True
    if not drawn:
        plt.close(fig)
        return
    ax.set_xlabel("Update")
    ax.set_ylabel("Training masked loss (rolling mean of 20 logs)")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    save(
        fig,
        out,
        "fig5_training_10k.png",
        "Over 10k updates BP keeps improving while PC stalls or collapses",
    )


def main() -> None:
    """Write every figure whose inputs exist."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        default=".",
        help="Directory holding results/ and the probe .txt files",
    )
    parser.add_argument("--out", default="results/figures")
    args = parser.parse_args()
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for figure in (
        fig_sweep,
        fig_alignment,
        fig_stability,
        fig_spikes_vs_loss,
        fig_training_10k,
    ):
        figure(root, out)


if __name__ == "__main__":
    main()
