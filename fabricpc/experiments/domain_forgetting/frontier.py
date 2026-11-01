"""The uniform-lr frontier test for the domain-forgetting gate runs.

Per run, ``x`` is the mean over domains of ``learning_gain_ce`` (mean
learning gain), ``y`` the mean over domains of ``final_ce`` (mean final CE)
and ``bwt`` the run's ``bwt_ce``. Runs are grouped by (domain order, seed).

- The **reference curve** of an (order, seed) is the piecewise-linear
  interpolation ``y_ref(x)`` through its ungated sequential runs (one per
  lr), sorted by ``x``; ``bwt_ref(x)`` likewise.
- The **frontier distance** of a run is ``d = y - y_ref(x)`` on the curve of
  the same order and seed (negative = lower final CE than learning more
  slowly), and ``d_bwt = bwt - bwt_ref(x)`` (positive = less forgetting).
  Outside the curve's ``x`` range both are ``None`` and ``outside_range``
  names the side and the gap to the nearest endpoint; nothing is
  extrapolated.
- **Gap closed** is ``(y_seq - y) / (y_seq - y_joint)``, with ``y_seq`` the
  same order and seed's sequential run at ``seq_lr`` and ``y_joint`` the
  mean over joint seeds (joint training is order-free).
- The **primary test** for ``(order, gate, beta)`` passes iff ``d < 0`` in
  every seed and ``|mean d|`` exceeds the seed SD of ``y`` of that order's
  ``seq_lr`` sequential runs.
- **Contrasts** ``map - random`` and ``map - type`` are paired by seed per
  order and ``beta``, on ``y`` and on ``d``.

Means are over seeds; SDs are sample SDs (``ddof=1``), ``None`` for one seed.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from fabricpc.experiments.domain_forgetting.runner import GATE_SETTINGS, ONLINE_GATES

SEQ_LR = 1e-4
PRIMARY_GATE = "map"
CONTRASTS = (("map", "random"), ("map", "type"))
REFERENCE = "none"
OKABE_ITO = {
    "none": "#000000",
    "map": "#0072B2",
    "random": "#E69F00",
    "type": "#009E73",
    "inverse": "#D55E00",
    "oracle": "#CC79A7",
    "freeze": "#56B4E9",
}
BETA_MARKERS = {0.5: "v", 1.0: "o", 2.0: "^"}
OTHER_MARKERS = ("s", "D", "P", "X", "h", "*")
DEFINITIONS = {
    "x": "mean over domains of learning_gain_ce (mean learning gain)",
    "y": "mean over domains of final_ce (mean final CE)",
    "bwt": "bwt_ce; negative = forgetting",
    "d": "y - y_ref(x) on the same order and seed's uniform-lr curve; "
    "negative = better than learning more slowly; None outside the curve",
    "d_bwt": "bwt - bwt_ref(x) on the same curve; positive = less forgetting",
    "gap_closed": "(y_seq - y) / (y_seq - y_joint); y_seq the same order and "
    "seed's sequential run at seq_lr, y_joint the mean over joint seeds",
    "primary_test": "d < 0 in every seed and |mean d| > seed SD of y of the "
    "order's seq_lr sequential runs",
    "contrast": "per seed a - b on y and on d; fraction_negative over seeds",
    "sd": "sample SD over seeds (ddof=1); None for one seed",
}


def _read(path: Path) -> Any:
    with open(path) as handle:
        return json.load(handle)


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values))


def lr_text(lr: float) -> str:
    """Return ``lr`` as ``"1e-04"`` when one significant digit is exact."""
    short = f"{lr:.0e}"
    return short if float(short) == lr else f"{lr:g}"


def describe_gate(config: Mapping[str, Any]) -> tuple[str, float | None, str]:
    """Return ``(gate, beta, label)`` of a run's ``run_config["config"]``.

    ``beta`` is ``None`` for gates that do not use it. Labels are
    ``"map_b1"``, ``"freeze[embeddings,L0]"``, ``"uniform_s0.5"``,
    ``"online_protect_b2"``, ``"online_surprise"`` (a non-live variant
    appended, as in ``"online_protect_b2_replay"``) and, for ungated runs,
    ``"lr3e-05"``.
    """
    gate = config.get("gate", REFERENCE)
    if gate == REFERENCE:
        return gate, None, f"lr{lr_text(float(config['lr']))}"
    settings = GATE_SETTINGS.get(gate, ())
    beta = float(config["gate_beta"]) if "gate_beta" in settings else None
    if gate == "freeze":
        label = f"freeze[{','.join(config.get('gate_modules', []))}]"
    elif gate == "uniform":
        label = f"uniform_s{float(config['gate_scale']):g}"
    elif beta is not None:
        label = f"{gate}_b{beta:g}"
    else:
        label = gate
    variant = config.get("gate_variant", "live")
    if gate in ONLINE_GATES and variant != "live":
        label += f"_{variant}"
    if not math.isclose(float(config["lr"]), SEQ_LR):
        label += f"@lr{lr_text(float(config['lr']))}"
    return gate, beta, label


def load_run_point(run_dir: str | Path) -> dict[str, Any] | None:
    """Read one run's frontier point; ``None`` (with a warning) if unfinished."""
    path = Path(run_dir)
    if not (path / "summary.json").exists() or not (path / "run_config.json").exists():
        print(f"warning: {path} has no summary.json or run_config.json; skipped")
        return None
    summary = _read(path / "summary.json")
    config = _read(path / "run_config.json")["config"]
    domains = tuple(config["domains"])
    if tuple(summary["domains"]) != domains:
        raise ValueError(f"{path}: summary and run_config domains differ")
    gate, beta, label = describe_gate(config)
    training = config.get("training", summary.get("training", "sequential"))
    final = summary["final_ce"]
    gains = summary["learning_gain_ce"]
    return {
        "run_dir": str(path),
        "domains": list(domains),
        "seed": int(config["seed"]),
        "lr": float(config["lr"]),
        "mode": summary.get("mode", config.get("mode")),
        "tokens_per_domain": config.get("tokens_per_domain"),
        "training": training,
        "gate": gate,
        "beta": beta,
        "label": label,
        "x": _mean(gains[d] for d in domains),
        "y": _mean(final[d] for d in domains),
        "bwt": float(summary["bwt_ce"]) if "bwt_ce" in summary else None,
    }


def collect_runs(
    run_dirs: Sequence[str | Path], mode: str = "backprop"
) -> list[dict[str, Any]]:
    """Load finished runs of ``mode`` with at least two domains or joint."""
    points = []
    for run_dir in run_dirs:
        point = load_run_point(run_dir)
        if point is None:
            continue
        if point["mode"] != mode:
            print(f"note: {run_dir} is mode {point['mode']!r}, not {mode!r}; skipped")
            continue
        if point["training"] != "joint" and len(point["domains"]) < 2:
            print(f"note: {run_dir} is a single-domain run; skipped")
            continue
        points.append(point)
    return points


def order_name(domains: Sequence[str]) -> str:
    """Return the order key used in the JSON, e.g. ``"news>reviews>wiki"``."""
    return ">".join(domains)


def reference_curve(points: Sequence[Mapping[str, Any]]) -> dict[str, list]:
    """Sort reference points by ``x``; returns parallel ``x``/``y``/``bwt``/``lr``."""
    ordered = sorted(points, key=lambda p: p["x"])
    xs = [p["x"] for p in ordered]
    if len(set(xs)) != len(xs):
        raise ValueError(f"reference curve has repeated x values: {xs}")
    return {
        "lr": [p["lr"] for p in ordered],
        "x": xs,
        "y": [p["y"] for p in ordered],
        "bwt": [p["bwt"] for p in ordered],
    }


def interpolate(curve: Mapping[str, Sequence[float]], x: float, key: str = "y"):
    """Return ``(value, outside)``: ``curve[key]`` linearly interpolated at ``x``.

    Outside ``[min x, max x]`` the value is ``None`` and ``outside`` is
    ``{"side": "below" | "above", "endpoint_x", "x_gap"}``; inside it is
    ``None``.
    """
    xs, ys = curve["x"], curve[key]
    if not xs:
        raise ValueError("empty reference curve")
    if x < xs[0] or x > xs[-1]:
        below = x < xs[0]
        endpoint = xs[0] if below else xs[-1]
        return None, {
            "side": "below" if below else "above",
            "endpoint_x": endpoint,
            "endpoint_lr": curve["lr"][0 if below else -1],
            "x_gap": x - endpoint,
        }
    return float(np.interp(x, xs, ys)), None


def frontier_distance(
    point: Mapping[str, Any], curve: Mapping[str, Sequence[float]]
) -> dict[str, Any]:
    """Return ``d``, ``d_bwt`` and ``outside_range`` of a run against its curve."""
    y_ref, outside = interpolate(curve, point["x"], "y")
    if outside is not None:
        index = 0 if outside["side"] == "below" else -1
        outside["y_minus_endpoint_y"] = point["y"] - curve["y"][index]
        return {"d": None, "d_bwt": None, "y_ref": None, "outside_range": outside}
    d_bwt = None
    if point.get("bwt") is not None and None not in curve["bwt"]:
        d_bwt = point["bwt"] - interpolate(curve, point["x"], "bwt")[0]
    return {
        "d": point["y"] - y_ref,
        "d_bwt": d_bwt,
        "y_ref": y_ref,
        "outside_range": None,
    }


def gap_closed(y: float, y_seq: float, y_joint: float) -> float:
    """Return ``(y_seq - y) / (y_seq - y_joint)``."""
    if y_seq == y_joint:
        raise ValueError("sequential and joint CE are equal; gap undefined")
    return (y_seq - y) / (y_seq - y_joint)


def mean_sd(values: Sequence[float | None]) -> dict[str, Any]:
    """Return mean, sample SD and count of the non-``None`` values."""
    kept = [float(v) for v in values if v is not None]
    if not kept:
        return {"mean": None, "sd": None, "n": 0}
    sd = float(np.std(kept, ddof=1)) if len(kept) > 1 else None
    return {"mean": float(np.mean(kept)), "sd": sd, "n": len(kept)}


def _is_reference(point: Mapping[str, Any]) -> bool:
    return point["training"] != "joint" and point["gate"] == REFERENCE


def _seed_mean_curve(points: Sequence[Mapping[str, Any]]) -> dict[str, list]:
    by_lr: dict[float, list] = {}
    for point in points:
        by_lr.setdefault(point["lr"], []).append(point)
    rows = [
        {
            "lr": lr,
            "x": _mean(p["x"] for p in group),
            "y": _mean(p["y"] for p in group),
            "x_sd": mean_sd([p["x"] for p in group])["sd"],
            "y_sd": mean_sd([p["y"] for p in group])["sd"],
            "n": len(group),
        }
        for lr, group in by_lr.items()
    ]
    rows.sort(key=lambda r: r["x"])
    return {
        key: [r[key] for r in rows] for key in ("lr", "x", "y", "x_sd", "y_sd", "n")
    }


def compute_frontier(
    points: Sequence[Mapping[str, Any]], seq_lr: float = SEQ_LR
) -> dict[str, Any]:
    """Run the frontier analysis on loaded run points (``collect_runs``)."""
    joint = [p for p in points if p["training"] == "joint"]
    sequential = [dict(p) for p in points if p["training"] != "joint"]
    y_joint = _mean(p["y"] for p in joint) if joint else None
    orders: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for point in sequential:
        orders.setdefault(tuple(point["domains"]), []).append(point)

    result_orders: dict[str, Any] = {}
    for domains, members in orders.items():
        name = order_name(domains)
        references = [p for p in members if _is_reference(p)]
        curves: dict[int, dict[str, list]] = {}
        seq_y: dict[int, float] = {}
        for seed in sorted({p["seed"] for p in references}):
            seed_refs = [p for p in references if p["seed"] == seed]
            if len({p["lr"] for p in seed_refs}) != len(seed_refs):
                raise ValueError(f"{name} seed {seed}: two reference runs share an lr")
            curves[seed] = reference_curve(seed_refs)
            for p in seed_refs:
                if math.isclose(p["lr"], seq_lr):
                    seq_y[seed] = p["y"]

        for point in members:
            curve = curves.get(point["seed"])
            if curve is None:
                point.update(d=None, d_bwt=None, y_ref=None, outside_range=None)
                point["no_reference"] = True
            else:
                point.update(frontier_distance(point, curve))
            y_seq = seq_y.get(point["seed"])
            point["gap_closed"] = (
                gap_closed(point["y"], y_seq, y_joint)
                if y_seq is not None and y_joint is not None
                else None
            )

        seq_sd = mean_sd(list(seq_y.values()))
        conditions = _aggregate(members)
        result_orders[name] = {
            "domains": list(domains),
            "curves": {str(s): c for s, c in curves.items()},
            "mean_curve": _seed_mean_curve(references),
            "seq": {"lr": seq_lr, "y": seq_sd, "per_seed": seq_y},
            "conditions": conditions,
            "runs": sorted(members, key=lambda p: (p["label"], p["seed"])),
            "primary_test": _primary_tests(members, seq_sd["sd"], seq_lr),
            "contrasts": _contrasts(members, seq_lr),
        }

    return {
        "definitions": DEFINITIONS,
        "seq_lr": seq_lr,
        "joint": {
            "y": mean_sd([p["y"] for p in joint]),
            "runs": [p["run_dir"] for p in joint],
        },
        "orders": result_orders,
        "primary_test_both_orders": _both_orders(result_orders),
    }


def _aggregate(members: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labels: dict[str, list[Mapping[str, Any]]] = {}
    for point in members:
        labels.setdefault(point["label"], []).append(point)
    out = {}
    for label, group in sorted(labels.items()):
        group = sorted(group, key=lambda p: p["seed"])
        out[label] = {
            "gate": group[0]["gate"],
            "beta": group[0]["beta"],
            "lr": group[0]["lr"],
            "seeds": [p["seed"] for p in group],
            "n": len(group),
            **{
                key: mean_sd([p[key] for p in group])
                for key in ("x", "y", "bwt", "d", "d_bwt", "gap_closed")
            },
            "outside_range": {
                str(p["seed"]): p["outside_range"]
                for p in group
                if p.get("outside_range")
            },
        }
    return out


def _by_beta(
    members: Sequence[Mapping[str, Any]], gate: str, seq_lr: float
) -> dict[float, dict[int, Mapping[str, Any]]]:
    out: dict[float, dict[int, Mapping[str, Any]]] = {}
    for p in members:
        if p["gate"] == gate and math.isclose(p["lr"], seq_lr):
            out.setdefault(p["beta"], {})[p["seed"]] = p
    return out


def _primary_tests(
    members: Sequence[Mapping[str, Any]], seq_sd: float | None, seq_lr: float
) -> dict[str, Any]:
    tests = {}
    for beta, runs in sorted(_by_beta(members, PRIMARY_GATE, seq_lr).items()):
        ds = [runs[s]["d"] for s in sorted(runs)]
        complete = all(d is not None for d in ds)
        stats = mean_sd(ds)
        all_negative = complete and all(d < 0 for d in ds)
        exceeds = (
            complete
            and seq_sd is not None
            and stats["mean"] is not None
            and abs(stats["mean"]) > seq_sd
        )
        tests[f"{beta:g}"] = {
            "beta": beta,
            "seeds": sorted(runs),
            "d": ds,
            "d_mean": stats["mean"],
            "d_sd": stats["sd"],
            "seq_y_sd": seq_sd,
            "all_seeds_negative": all_negative,
            "exceeds_seq_sd": bool(exceeds),
            "pass": bool(all_negative and exceeds),
        }
    return tests


def _both_orders(orders: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    betas = sorted({b for o in orders.values() for b in o["primary_test"]}, key=float)
    out = {}
    for beta in betas:
        per_order = {
            name: o["primary_test"][beta]["pass"] if beta in o["primary_test"] else None
            for name, o in orders.items()
        }
        out[beta] = {
            "per_order": per_order,
            "pass": len(per_order) >= 2 and all(v is True for v in per_order.values()),
        }
    return out


def _contrasts(members: Sequence[Mapping[str, Any]], seq_lr: float) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for first, second in CONTRASTS:
        a_runs = _by_beta(members, first, seq_lr)
        b_runs = _by_beta(members, second, seq_lr)
        for beta in sorted(set(a_runs) & set(b_runs)):
            seeds = sorted(set(a_runs[beta]) & set(b_runs[beta]))
            if not seeds:
                continue
            entry: dict[str, Any] = {"beta": beta, "seeds": seeds}
            for key in ("y", "d"):
                diffs = [
                    (
                        a_runs[beta][s][key] - b_runs[beta][s][key]
                        if a_runs[beta][s][key] is not None
                        and b_runs[beta][s][key] is not None
                        else None
                    )
                    for s in seeds
                ]
                kept = [v for v in diffs if v is not None]
                entry[key] = {
                    "per_seed": diffs,
                    **mean_sd(diffs),
                    "fraction_negative": (
                        sum(v < 0 for v in kept) / len(kept) if kept else None
                    ),
                }
            out[f"{first}-{second}_b{beta:g}"] = entry
    return out


def _fmt(stats: Mapping[str, Any], digits: int = 3) -> str:
    if stats["mean"] is None:
        return "-"
    if stats["sd"] is None:
        return f"{stats['mean']:+.{digits}f}"
    return f"{stats['mean']:+.{digits}f}±{stats['sd']:.{digits}f}"


def format_table(result: Mapping[str, Any]) -> str:
    """Return the per-order condition table, primary tests and contrasts."""
    lines = []
    joint = result["joint"]["y"]
    if joint["mean"] is not None:
        lines.append(f"joint: y = {_fmt(joint)} (n={joint['n']})")
    header = (
        f"{'condition':<28}{'n':>3}  {'x (gain)':>14}  {'y (final CE)':>14}  "
        f"{'bwt':>14}  {'d':>14}  {'gap closed':>14}"
    )
    for name, order in result["orders"].items():
        lines += ["", f"order {name}", header]
        for label, c in order["conditions"].items():
            flag = " *out" if c["outside_range"] else ""
            lines.append(
                f"{label:<28}{c['n']:>3}  {_fmt(c['x']):>14}  {_fmt(c['y']):>14}  "
                f"{_fmt(c['bwt']):>14}  {_fmt(c['d'], 4):>14}  "
                f"{_fmt(c['gap_closed']):>14}{flag}"
            )
        for beta, test in order["primary_test"].items():
            lines.append(
                f"primary map b{beta}: pass={test['pass']} "
                f"(all d<0: {test['all_seeds_negative']}, |mean d| > seq SD "
                f"{test['seq_y_sd']}: {test['exceeds_seq_sd']})"
            )
        for key, c in order["contrasts"].items():
            lines.append(
                f"contrast {key}: y {_fmt(c['y'], 4)} "
                f"(neg {c['y']['fraction_negative']}), d {_fmt(c['d'], 4)} "
                f"(neg {c['d']['fraction_negative']})"
            )
    for beta, test in result["primary_test_both_orders"].items():
        lines.append(f"primary map b{beta} both orders: pass={test['pass']}")
    return "\n".join(lines)


def _marker(beta: float | None, label: str, others: list[str]) -> str:
    if beta is not None:
        return BETA_MARKERS.get(beta, "p")
    if label not in others:
        others.append(label)
    return OTHER_MARKERS[others.index(label) % len(OTHER_MARKERS)]


def plot_frontier(result: Mapping[str, Any], path: str | Path) -> list[Path] | None:
    """Plot one panel per order; writes ``path`` as ``.pdf`` and ``.png``.

    Returns ``None`` without plotting when matplotlib is not installed.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping the frontier figure")
        return None
    orders = result["orders"]
    if not orders:
        return None
    figure, axes = plt.subplots(
        1, len(orders), figsize=(7.0, 3.4), squeeze=False, constrained_layout=True
    )
    joint = result["joint"]["y"]["mean"]
    others: list[str] = []
    handles: dict[str, Any] = {}
    for axis, (name, order) in zip(axes[0], orders.items()):
        curve = order["mean_curve"]
        (line,) = axis.plot(
            curve["x"],
            curve["y"],
            color=OKABE_ITO["none"],
            marker="o",
            markersize=3.5,
            linewidth=1.0,
            label="uniform lr (seed mean)",
        )
        handles.setdefault("uniform lr (seed mean)", line)
        for x, y, lr in zip(curve["x"], curve["y"], curve["lr"]):
            axis.annotate(
                lr_text(lr),
                (x, y),
                textcoords="offset points",
                xytext=(3, 3),
                fontsize=6,
                color="0.3",
            )
        if joint is not None:
            handle = axis.axhline(
                joint, color="0.45", linestyle="--", linewidth=0.9, label="joint"
            )
            handles.setdefault("joint", handle)
        for label, c in order["conditions"].items():
            if c["gate"] == REFERENCE or c["x"]["mean"] is None:
                continue
            handle = axis.errorbar(
                c["x"]["mean"],
                c["y"]["mean"],
                xerr=c["x"]["sd"] or 0.0,
                yerr=c["y"]["sd"] or 0.0,
                color=OKABE_ITO.get(c["gate"], "0.5"),
                marker=_marker(c["beta"], label, others),
                markersize=5,
                linestyle="none",
                elinewidth=0.8,
                capsize=1.5,
                label=label,
            )
            handles.setdefault(label, handle)
        axis.set_title(name.replace(">", " → "), fontsize=9)
        axis.set_xlabel("mean learning gain (CE)", fontsize=8)
        axis.tick_params(labelsize=7)
        axis.margins(x=0.1)
        axis.grid(alpha=0.25, linewidth=0.5)
    axes[0][0].set_ylabel("mean final CE", fontsize=8)
    figure.legend(
        list(handles.values()),
        list(handles),
        loc="outside lower center",
        ncol=min(4, len(handles)),
        fontsize=7,
        frameon=False,
    )
    base = Path(path).with_suffix("")
    outputs = [base.with_suffix(".pdf"), base.with_suffix(".png")]
    for output in outputs:
        figure.savefig(output, dpi=200)
    plt.close(figure)
    return outputs
