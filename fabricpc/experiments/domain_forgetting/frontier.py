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

The reference runs are those of one learner: ``collect_runs`` keeps the
runs of ``mode`` (backprop by default) and, for PC, optionally of one
``pc_inference``, so an ePC curve is its own ePC sequential runs. Map gates
with ``gate_map_source="learner"`` are labelled ``..._learner``; the primary
test and contrasts use the gate runs of one ``map_source`` (``"bp"`` by
default).

A run whose final CE, learning gain or BWT is not finite has diverged: its
``x``, ``y`` and ``bwt`` are ``None``. Diverged reference runs leave the
curve (it interpolates between the others) and are listed per order under
``failed_reference_runs``; a diverged or missing gate seed fails the primary
test, which needs every expected seed (by default every seed with a
reference run in that order).

The **lower envelope** of a curve (``lower_envelope``) drops every point
dominated by another (gain ≥ x and CE ≤ y) and interpolates along the rest,
``bwt`` included. It is a post hoc reference, added because the pre-registered
linear interpolation is not a frontier on a non-monotone curve. Every run also
reports ``d_envelope`` and ``d_bwt_envelope`` on it and every primary test an
``envelope`` entry; the pre-registered fields are unchanged. Below the first
retained point but inside the curve's range, the reference is that point: it has
the curve's lowest CE and more gain, and the measured points there all have
higher CE, so the reference is conservative. Outside the curve's range there is
no envelope distance.

An **influence verdict** (``influence_verdict``) applies one rule to a gate on
the frontier, against its matched control and against surprise only: the
per-seed value (``d``, or ``d`` of the gate minus ``d`` of the control paired by
seed) is < 0 in every expected seed and ``|mean|`` exceeds the order's sequential
seed SD, in both orders. It is reported under both computations whatever it
shows. Each comparison's ``status`` is ``pass``, ``fail`` (a measured value is
>= 0, or the mean does not exceed the SD) or ``undefined`` (no measured value
rules it out, but one is missing or off the curve). An undefined comparison is
not a pass. A verdict across learning rates, or one asked for as descriptive, is
marked ``descriptive_only``; its rows are reported but do not decide the test.

Means are over seeds; SDs are sample SDs (``ddof=1``), ``None`` for one seed.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from fabricpc.experiments.domain_forgetting.runner import (
    GATE_DEFAULTS,
    GATE_SETTINGS,
    ONLINE_GATES,
)

SEQ_LR = 1e-4
PRIMARY_GATE = "map"
CONTRASTS = (("map", "random"), ("map", "type"))
REFERENCE = "none"
ALPHA_KEY = "gate_influence_alpha"
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
    "d_envelope": "post hoc: y - y_env(x) on the curve's lower envelope (points "
    "not dominated by gain >= x and CE <= y), held at its first point below "
    "that point; None outside the curve's range",
    "d_bwt_envelope": "post hoc: bwt - bwt_env(x) on the same retained points",
    "primary_test": "d < 0 in every seed and |mean d| > seed SD of y of the "
    "order's seq_lr sequential runs",
    "contrast": "per seed a - b on y and on d; fraction_negative over seeds",
    "sd": "sample SD over seeds (ddof=1); None for one seed",
}


def _read(path: Path) -> Any:
    with open(path) as handle:
        return json.load(handle)


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values))


def lr_text(lr: float) -> str:
    """Return ``lr`` as ``"1e-04"`` when one significant digit is exact."""
    short = f"{lr:.0e}"
    return short if float(short) == lr else f"{lr:g}"


def intervention_label(config: Mapping[str, Any]) -> str:
    """Return the suffix naming a run's error-norm penalty and update scale.

    ``"_pen0.001"`` for ``pc_error_norm_penalty`` 0.001 and
    ``"_scale0.5[embeddings,L0_mha]"`` for an update scale on those modules;
    empty for a run with neither.
    """
    label = ""
    penalty = float(config.get("pc_error_norm_penalty", 0.0))
    if penalty:
        label += f"_pen{penalty:g}"
    modules = config.get("update_scale_modules", [])
    if modules:
        scale = float(config.get("update_scale", 1.0))
        label += f"_scale{scale:g}[{','.join(modules)}]"
    return label


def describe_gate(config: Mapping[str, Any]) -> tuple[str, float | None, str]:
    """Return ``(gate, beta, label)`` of a run's ``run_config["config"]``.

    ``beta`` is ``None`` for gates that do not use it. Labels are
    ``"map_b1"``, ``"freeze[embeddings,L0]"``, ``"uniform_s0.5"``,
    ``"online_protect_b2"``, ``"online_surprise"`` (``"_learner"`` appended
    for ``gate_map_source="learner"``, then a non-live variant, as in
    ``"online_protect_b2_learner_replay"``, then an ``online_influence`` α
    other than the default, as in ``"online_influence_a0"``) and, for ungated
    runs, ``"lr3e-05"``. An error-norm penalty or update scale appends
    ``intervention_label``, as in ``"lr3e-05_pen0.001"``.
    """
    gate = config.get("gate", REFERENCE)
    intervention = intervention_label(config)
    if gate == REFERENCE:
        return gate, None, f"lr{lr_text(float(config['lr']))}{intervention}"
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
    if config.get("gate_map_source", "bp") != "bp":
        label += f"_{config['gate_map_source']}"
    variant = config.get("gate_variant", "live")
    if gate in ONLINE_GATES and variant != "live":
        label += f"_{variant}"
    if gate == "online_influence":
        alpha = float(config.get("gate_influence_alpha", GATE_DEFAULTS[ALPHA_KEY]))
        if alpha != GATE_DEFAULTS[ALPHA_KEY]:
            label += f"_a{alpha:g}"
    if not math.isclose(float(config["lr"]), SEQ_LR):
        label += f"@lr{lr_text(float(config['lr']))}"
    return gate, beta, label + intervention


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
    bwt = summary.get("bwt_ce")
    values = [*(final[d] for d in domains), *(gains[d] for d in domains)]
    diverged = not all(_finite(v) for v in values) or (
        "bwt_ce" in summary and not _finite(bwt)
    )
    return {
        "run_dir": str(path),
        "domains": list(domains),
        "seed": int(config["seed"]),
        "lr": float(config["lr"]),
        "mode": summary.get("mode", config.get("mode")),
        "pc_inference": config.get("pc_inference"),
        "map_source": config.get("gate_map_source", "bp"),
        "tokens_per_domain": config.get("tokens_per_domain"),
        "training": training,
        "gate": gate,
        "beta": beta,
        "label": label,
        "intervention": bool(intervention_label(config)),
        "diverged": diverged,
        "x": None if diverged else _mean(gains[d] for d in domains),
        "y": None if diverged else _mean(final[d] for d in domains),
        "bwt": None if diverged or "bwt_ce" not in summary else float(bwt),
    }


def collect_runs(
    run_dirs: Sequence[str | Path],
    mode: str = "backprop",
    pc_inference: str | None = None,
) -> list[dict[str, Any]]:
    """Load finished runs of ``mode`` with at least two domains or joint.

    With ``mode="pc"`` and ``pc_inference`` only runs of that PC inference
    are kept.
    """
    points = []
    for run_dir in run_dirs:
        point = load_run_point(run_dir)
        if point is None:
            continue
        if point["mode"] != mode:
            print(f"note: {run_dir} is mode {point['mode']!r}, not {mode!r}; skipped")
            continue
        if (
            mode == "pc"
            and pc_inference is not None
            and point["pc_inference"] != pc_inference
        ):
            print(
                f"note: {run_dir} is pc_inference {point['pc_inference']!r}, not "
                f"{pc_inference!r}; skipped"
            )
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


def lower_envelope(curve: Mapping[str, Sequence[Any]]) -> dict[str, list]:
    """The curve's points not dominated by another with gain ≥ x and CE ≤ y.

    Sorted by ``x`` with the curve's ``lr``, ``x``, ``y`` and ``bwt``, so ``bwt``
    interpolates along the same points. Of two points with equal ``x`` only the
    lower ``y`` stays, of two with equal ``y`` only the higher ``x``, and of
    exact duplicates only the first.
    """
    rows = sorted(
        zip(curve["lr"], curve["x"], curve["y"], curve["bwt"]), key=lambda r: r[1]
    )
    kept: list[tuple[Any, ...]] = []
    seen: set[tuple[float, float]] = set()
    for row in rows:
        point = (row[1], row[2])
        if point in seen:
            continue
        seen.add(point)
        dominated = any(
            other[1] >= point[0]
            and other[2] <= point[1]
            and (other[1], other[2]) != point
            for other in rows
        )
        if not dominated:
            kept.append(row)
    return {
        key: [row[index] for row in kept]
        for index, key in enumerate(("lr", "x", "y", "bwt"))
    }


def envelope_distance(
    point: Mapping[str, Any], curve: Mapping[str, Sequence[Any]]
) -> dict[str, Any]:
    """``frontier_distance`` on the curve's lower envelope, under its own keys.

    A point below the envelope's first point and inside the curve's range is
    measured against that first point; ``below_envelope`` marks it.
    """
    envelope = lower_envelope(curve)
    below = min(curve["x"]) <= point["x"] < envelope["x"][0]
    if below:
        envelope = {
            "lr": [envelope["lr"][0]] * 2,
            "x": [point["x"], envelope["x"][0]],
            "y": [envelope["y"][0]] * 2,
            "bwt": [envelope["bwt"][0]] * 2,
        }
    distance = frontier_distance(point, envelope)
    return {
        "below_envelope": below,
        "d_envelope": distance["d"],
        "d_bwt_envelope": distance["d_bwt"],
        "y_ref_envelope": distance["y_ref"],
        "outside_range_envelope": distance["outside_range"],
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
    """Ungated sequential runs without an error-norm penalty or update scale."""
    return (
        point["training"] != "joint"
        and point["gate"] == REFERENCE
        and not point.get("intervention", False)
    )


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
    points: Sequence[Mapping[str, Any]],
    seq_lr: float = SEQ_LR,
    map_source: str = "bp",
    expected_seeds: Sequence[int] | None = None,
    influence_verdicts: Sequence[Sequence[str]] = (),
    descriptive_verdicts: Sequence[Sequence[str]] = (),
) -> dict[str, Any]:
    """Run the frontier analysis on loaded run points (``collect_runs``).

    The primary test and contrasts use the gate runs of ``map_source``; the
    primary test needs every seed of ``expected_seeds`` (default: the seeds
    with a reference run in the order). ``influence_verdicts`` holds
    ``(gate, matched, surprise)`` condition labels to judge
    (``influence_verdict``); ``descriptive_verdicts`` are reported the same way
    and marked as not deciding the test.
    """
    joint = [p for p in points if p["training"] == "joint" and not p.get("diverged")]
    sequential = [dict(p) for p in points if p["training"] != "joint"]
    y_joint = _mean(p["y"] for p in joint) if joint else None
    orders: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for point in sequential:
        orders.setdefault(tuple(point["domains"]), []).append(point)

    result_orders: dict[str, Any] = {}
    for domains, members in orders.items():
        name = order_name(domains)
        candidates = [p for p in members if _is_reference(p)]
        references = [p for p in candidates if not p.get("diverged")]
        failed = [
            {"run_dir": p["run_dir"], "lr": p["lr"], "seed": p["seed"]}
            for p in sorted(candidates, key=lambda p: (p["seed"], p["lr"]))
            if p.get("diverged")
        ]
        expected = sorted(
            {p["seed"] for p in candidates}
            if expected_seeds is None
            else set(expected_seeds)
        )
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
            point.update(
                d_envelope=None,
                d_bwt_envelope=None,
                y_ref_envelope=None,
                outside_range_envelope=None,
            )
            if point.get("diverged"):
                point.update(d=None, d_bwt=None, y_ref=None, outside_range=None)
            elif curve is None:
                point.update(d=None, d_bwt=None, y_ref=None, outside_range=None)
                point["no_reference"] = True
            else:
                point.update(frontier_distance(point, curve))
                point.update(envelope_distance(point, curve))
            y_seq = seq_y.get(point["seed"])
            point["gap_closed"] = (
                gap_closed(point["y"], y_seq, y_joint)
                if y_seq is not None and y_joint is not None and point["y"] is not None
                else None
            )

        seq_sd = mean_sd(list(seq_y.values()))
        conditions = _aggregate(members)
        result_orders[name] = {
            "domains": list(domains),
            "curves": {str(s): c for s, c in curves.items()},
            "envelope_curves": {str(s): lower_envelope(c) for s, c in curves.items()},
            "failed_reference_runs": failed,
            "expected_seeds": expected,
            "mean_curve": _seed_mean_curve(references),
            "seq": {"lr": seq_lr, "y": seq_sd, "per_seed": seq_y},
            "conditions": conditions,
            "runs": sorted(members, key=lambda p: (p["label"], p["seed"])),
            "primary_test": _primary_tests(
                members, seq_sd["sd"], seq_lr, map_source, expected
            ),
            "contrasts": _contrasts(members, seq_lr, map_source),
        }

    verdicts = [
        influence_verdict(result_orders, *labels) for labels in influence_verdicts
    ] + [
        influence_verdict(result_orders, *labels, descriptive=True)
        for labels in descriptive_verdicts
    ]
    return {
        **({"influence_verdicts": verdicts} if verdicts else {}),
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
                for key in (
                    "x",
                    "y",
                    "bwt",
                    "d",
                    "d_bwt",
                    "gap_closed",
                    "d_envelope",
                    "d_bwt_envelope",
                )
            },
            "outside_range_envelope": {
                str(p["seed"]): p["outside_range_envelope"]
                for p in group
                if p.get("outside_range_envelope")
            },
            "outside_range": {
                str(p["seed"]): p["outside_range"]
                for p in group
                if p.get("outside_range")
            },
        }
    return out


def _by_beta(
    members: Sequence[Mapping[str, Any]],
    gate: str,
    seq_lr: float,
    map_source: str = "bp",
) -> dict[float, dict[int, Mapping[str, Any]]]:
    out: dict[float, dict[int, Mapping[str, Any]]] = {}
    for p in members:
        if (
            p["gate"] == gate
            and math.isclose(p["lr"], seq_lr)
            and not p.get("intervention", False)
            and p.get("map_source", "bp") == map_source
        ):
            out.setdefault(p["beta"], {})[p["seed"]] = p
    return out


def _primary_tests(
    members: Sequence[Mapping[str, Any]],
    seq_sd: float | None,
    seq_lr: float,
    map_source: str = "bp",
    expected: Sequence[int] = (),
) -> dict[str, Any]:
    tests = {}
    for beta, runs in sorted(
        _by_beta(members, PRIMARY_GATE, seq_lr, map_source).items()
    ):
        ds = [runs[s]["d"] for s in sorted(runs)]
        missing = sorted(set(expected) - set(runs))
        diverged = sorted(s for s, p in runs.items() if p.get("diverged"))
        complete = not missing and all(d is not None for d in ds)
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
            "expected_seeds": list(expected),
            "missing_seeds": missing,
            "diverged_seeds": diverged,
            "all_seeds_negative": all_negative,
            "exceeds_seq_sd": bool(exceeds),
            "pass": bool(all_negative and exceeds),
            "envelope": {
                "post_hoc": True,
                **pass_rule(
                    [
                        runs[s].get("d_envelope") if s in runs else None
                        for s in sorted({*runs, *expected})
                    ],
                    seq_sd,
                ),
            },
        }
    return tests


def pass_rule(values: Sequence[float | None], seq_sd: float | None) -> dict[str, Any]:
    """Every value < 0 and ``|mean|`` above ``seq_sd``; a missing value never passes.

    ``status`` separates a measured failure from a comparison with values
    missing: a measured value >= 0 fails whatever the others are.
    """
    stats = mean_sd(values)
    complete = bool(values) and all(v is not None for v in values)
    all_negative = complete and all(v < 0 for v in values)  # type: ignore[operator]
    exceeds = (
        complete
        and seq_sd is not None
        and stats["mean"] is not None
        and abs(stats["mean"]) > seq_sd
    )
    passed = bool(all_negative and exceeds)
    if passed:
        status = "pass"
    elif complete or any(v is not None and v >= 0 for v in values):
        status = "fail"
    else:
        status = "undefined"
    return {
        "status": status,
        "values": list(values),
        "mean": stats["mean"],
        "sd": stats["sd"],
        "seq_y_sd": seq_sd,
        "all_negative": all_negative,
        "exceeds_seq_sd": bool(exceeds),
        "pass": passed,
    }


def _combined_status(rows: Sequence[Mapping[str, Any]]) -> str:
    """``fail`` if any order fails, ``pass`` if all pass, else ``undefined``."""
    statuses = [row["status"] for row in rows]
    if "fail" in statuses:
        return "fail"
    return "pass" if statuses and set(statuses) == {"pass"} else "undefined"


def _difference(first: float | None, second: float | None) -> float | None:
    return None if first is None or second is None else first - second


def _seed_values(
    by_label: Mapping[str, Mapping[int, Mapping[str, Any]]],
    label: str,
    key: str,
    seeds: Sequence[int],
) -> list[float | None]:
    runs = by_label.get(label, {})
    return [runs[s][key] if s in runs else None for s in seeds]


def influence_verdict(
    orders: Mapping[str, Mapping[str, Any]],
    gate: str,
    matched: str,
    surprise: str,
    *,
    descriptive: bool = False,
) -> dict[str, Any]:
    """Judge ``gate`` on the frontier, against ``matched`` and against ``surprise``.

    Applies ``pass_rule`` per order over its expected seeds, paired by seed,
    under both computations: ``pre_registered`` (``d``) and
    ``envelope_post_hoc`` (``d_envelope``). ``lr`` holds each condition's lr per
    order, and ``lr_mismatch`` is true when the conditions present differ
    anywhere: such a comparison also differs in base rate, so it is
    ``descriptive_only``. ``missing_conditions`` lists conditions with no run.
    ``paired_y`` is the per-seed final CE difference from each control, defined
    even when a run is off the curve.
    """
    out: dict[str, Any] = {
        "gate": gate,
        "matched": matched,
        "surprise": surprise,
        "rule": "value < 0 in every expected seed and |mean| > the order's "
        "sequential seed SD, in both orders; values are d of the gate, then d "
        "of the gate minus d of the control paired by seed",
        "lr": {},
    }
    for name, key, post_hoc in (
        ("pre_registered", "d", False),
        ("envelope_post_hoc", "d_envelope", True),
    ):
        per_order: dict[str, dict[str, Any]] = {}
        for order_key, order in orders.items():
            by_label: dict[str, dict[int, Mapping[str, Any]]] = {}
            for run in order["runs"]:
                by_label.setdefault(run["label"], {})[run["seed"]] = run
            seeds = order["expected_seeds"]
            seq_sd = order["seq"]["y"]["sd"]

            own = _seed_values(by_label, gate, key, seeds)
            per_order[order_key] = {
                "seeds": list(seeds),
                "frontier": pass_rule(own, seq_sd),
                "vs_matched": pass_rule(
                    [
                        _difference(a, b)
                        for a, b in zip(
                            own, _seed_values(by_label, matched, key, seeds)
                        )
                    ],
                    seq_sd,
                ),
                "vs_surprise": pass_rule(
                    [
                        _difference(a, b)
                        for a, b in zip(
                            own, _seed_values(by_label, surprise, key, seeds)
                        )
                    ],
                    seq_sd,
                ),
            }
            for comparison, control in (
                ("vs_matched", matched),
                ("vs_surprise", surprise),
            ):
                per_order[order_key][comparison]["paired_y"] = [
                    _difference(a, b)
                    for a, b in zip(
                        _seed_values(by_label, gate, "y", seeds),
                        _seed_values(by_label, control, "y", seeds),
                    )
                ]
            out["lr"][order_key] = {
                label: sorted({r["lr"] for r in by_label.get(label, {}).values()})
                for label in (gate, matched, surprise)
            }
        passes = {
            comparison: len(per_order) >= 2
            and all(entry[comparison]["pass"] for entry in per_order.values())
            for comparison in ("frontier", "vs_matched", "vs_surprise")
        }
        out[name] = {
            "post_hoc": post_hoc,
            "orders": per_order,
            "passes": passes,
            "status": {
                comparison: _combined_status(
                    [entry[comparison] for entry in per_order.values()]
                )
                for comparison in passes
            },
            "pass": all(passes.values()),
        }
    out["lr_mismatch"] = any(
        len({tuple(lrs) for lrs in entry.values() if lrs}) > 1
        for entry in out["lr"].values()
    )
    out["missing_conditions"] = sorted(
        {
            label
            for entry in out["lr"].values()
            for label, lrs in entry.items()
            if not lrs
        }
    )
    out["descriptive_only"] = descriptive or out["lr_mismatch"]
    return out


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
            "envelope_pass_post_hoc": len(per_order) >= 2
            and all(
                beta in o["primary_test"]
                and o["primary_test"][beta]["envelope"]["pass"]
                for o in orders.values()
            ),
        }
    return out


def _contrasts(
    members: Sequence[Mapping[str, Any]], seq_lr: float, map_source: str = "bp"
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for first, second in CONTRASTS:
        a_runs = _by_beta(members, first, seq_lr, map_source)
        b_runs = _by_beta(members, second, seq_lr, map_source)
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
        f"{'bwt':>14}  {'d':>14}  {'gap closed':>14}  {'d env (post hoc)':>16}"
    )
    for name, order in result["orders"].items():
        lines += ["", f"order {name}", header]
        for label, c in order["conditions"].items():
            flag = " *out" if c["outside_range"] else ""
            lines.append(
                f"{label:<28}{c['n']:>3}  {_fmt(c['x']):>14}  {_fmt(c['y']):>14}  "
                f"{_fmt(c['bwt']):>14}  {_fmt(c['d'], 4):>14}  "
                f"{_fmt(c['gap_closed']):>14}  {_fmt(c['d_envelope'], 4):>16}{flag}"
            )
        for failed in order.get("failed_reference_runs", []):
            lines.append(
                f"failed reference run (diverged, left out of the curve): lr "
                f"{lr_text(failed['lr'])} seed {failed['seed']} {failed['run_dir']}"
            )
        for beta, test in order["primary_test"].items():
            gaps = ""
            if test.get("missing_seeds"):
                gaps += f", missing seeds {test['missing_seeds']}"
            if test.get("diverged_seeds"):
                gaps += f", diverged seeds {test['diverged_seeds']}"
            lines.append(
                f"primary map b{beta}: pass={test['pass']} "
                f"(all d<0: {test['all_seeds_negative']}, |mean d| > seq SD "
                f"{test['seq_y_sd']}: {test['exceeds_seq_sd']}{gaps})"
            )
        contrasts = {
            key: c
            for key, c in order["contrasts"].items()
            if c["y"]["n"] or c["d"]["n"]
        }
        if not contrasts:
            lines.append("contrasts: none (no paired gate runs)")
        for key, c in contrasts.items():
            lines.append(
                f"contrast {key}: y {_fmt(c['y'], 4)} "
                f"(neg {c['y']['fraction_negative']}), d {_fmt(c['d'], 4)} "
                f"(neg {c['d']['fraction_negative']})"
            )
    for beta, test in result["primary_test_both_orders"].items():
        lines.append(f"primary map b{beta} both orders: pass={test['pass']}")
    for verdict in result.get("influence_verdicts", []):
        lines += [
            "",
            f"verdict {verdict['gate']} vs {verdict['matched']} and "
            f"{verdict['surprise']} (lr mismatch: {verdict['lr_mismatch']}, "
            f"descriptive only: {verdict['descriptive_only']}, "
            f"missing: {verdict['missing_conditions']})",
        ]
        for name in ("pre_registered", "envelope_post_hoc"):
            entry = verdict[name]
            lines.append(f"  {name}: pass={entry['pass']} {entry['status']}")
            for order, rows in entry["orders"].items():
                for comparison in ("frontier", "vs_matched", "vs_surprise"):
                    row = rows[comparison]
                    shown = " ".join(
                        "-" if v is None else f"{v:+.4f}" for v in row["values"]
                    )
                    paired = " ".join(
                        "-" if v is None else f"{v:+.4f}"
                        for v in row.get("paired_y", [])
                    )
                    lines.append(
                        f"    {order} {comparison}: {shown} {row['status']}"
                        + (f" (paired y: {paired})" if paired else "")
                    )
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
