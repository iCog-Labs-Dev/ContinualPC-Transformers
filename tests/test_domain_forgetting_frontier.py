"""Tests for the uniform-lr frontier analysis of domain-forgetting runs."""

import json
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest

from fabricpc.experiments.domain_forgetting.frontier import (
    collect_runs,
    compute_frontier,
    describe_gate,
    format_table,
    frontier_distance,
    gap_closed,
    interpolate,
    reference_curve,
)

FORWARD = ("news", "reviews", "wiki")
REVERSE = ("wiki", "reviews", "news")
CURVE = {1e-5: (0.2, 3.3), 3e-5: (0.4, 3.1), 1e-4: (0.6, 3.2)}
JOINT_Y = 2.9


def write_run(
    root,
    name,
    *,
    x,
    y,
    seed,
    lr=1e-4,
    domains=FORWARD,
    gate=None,
    beta=1.0,
    modules=(),
    joint=False,
    bwt=-0.3,
    finished=True,
):
    """Write a minimal run directory whose means over domains are ``x``, ``y``."""
    path = root / name
    path.mkdir()
    config = {"lr": lr, "seed": seed, "domains": list(domains), "mode": "backprop"}
    if joint:
        config["training"] = "joint"
    if gate is not None:
        config.update(
            gate=gate,
            gate_beta=beta,
            gate_modules=list(modules),
            gate_source=[],
            gate_map_windows=256,
            gate_seed=0,
        )
    (path / "run_config.json").write_text(json.dumps({"config": config}))
    if not finished:
        return path
    spread = (-0.1, 0.0, 0.1)
    summary = {
        "domains": list(domains),
        "mode": "backprop",
        "final_ce": {d: y + s for d, s in zip(domains, spread)},
        "learning_gain_ce": {d: x - s for d, s in zip(domains, spread)},
    }
    if joint:
        summary["training"] = "joint"
    else:
        summary["bwt_ce"] = bwt
        summary["final_ce_forgetting"] = {d: 0.1 for d in domains[:-1]}
    (path / "summary.json").write_text(json.dumps(summary))
    return path


def write_references(root, domains=FORWARD, tag="fwd", seeds=(0, 1, 2), shift=0.0):
    for seed in seeds:
        for lr, (x, y) in CURVE.items():
            write_run(
                root,
                f"{tag}_lr{lr:g}_s{seed}",
                x=x,
                y=y + shift + 0.001 * seed,
                seed=seed,
                lr=lr,
                domains=domains,
                bwt=-x,
            )


def write_joint(root):
    for seed in range(3):
        write_run(
            root, f"joint_s{seed}", x=0.9, y=JOINT_Y, seed=seed, joint=True, bwt=None
        )


def analyse(root):
    return compute_frontier(collect_runs(sorted(root.iterdir())))


def test_interpolation_known_answer():
    curve = reference_curve(
        [
            {"lr": lr, "x": x, "y": y, "bwt": -x}
            for lr, (x, y) in reversed(CURVE.items())
        ]
    )
    assert curve["x"] == [0.2, 0.4, 0.6]
    assert curve["lr"] == [1e-5, 3e-5, 1e-4]
    assert interpolate(curve, 0.3)[0] == pytest.approx(3.2)
    assert interpolate(curve, 0.5)[0] == pytest.approx(3.15)
    assert interpolate(curve, 0.6)[0] == pytest.approx(3.2)
    assert interpolate(curve, 0.5, "bwt")[0] == pytest.approx(-0.5)


def test_outside_range_is_flagged_not_extrapolated():
    curve = reference_curve(
        [{"lr": lr, "x": x, "y": y, "bwt": -x} for lr, (x, y) in CURVE.items()]
    )
    low = frontier_distance({"x": 0.1, "y": 3.0, "bwt": -0.1}, curve)
    assert low["d"] is None and low["d_bwt"] is None
    assert low["outside_range"]["side"] == "below"
    assert low["outside_range"]["x_gap"] == pytest.approx(-0.1)
    assert low["outside_range"]["endpoint_lr"] == 1e-5
    assert low["outside_range"]["y_minus_endpoint_y"] == pytest.approx(-0.3)
    high = frontier_distance({"x": 0.7, "y": 3.0, "bwt": -0.7}, curve)
    assert high["outside_range"]["side"] == "above"
    assert high["outside_range"]["x_gap"] == pytest.approx(0.1)


def test_gap_closed_formula():
    assert gap_closed(3.1, 3.2, 3.0) == pytest.approx(0.5)
    assert gap_closed(3.2, 3.2, 3.0) == 0.0
    assert gap_closed(3.0, 3.2, 3.0) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        gap_closed(3.1, 3.0, 3.0)


def test_describe_gate_labels():
    assert describe_gate({"lr": 3e-5}) == ("none", None, "lr3e-05")
    assert describe_gate({"lr": 1e-4, "gate": "map", "gate_beta": 0.5}) == (
        "map",
        0.5,
        "map_b0.5",
    )
    freeze = {
        "lr": 1e-4,
        "gate": "freeze",
        "gate_beta": 1.0,
        "gate_modules": ["embeddings", "L0", "L1"],
    }
    assert describe_gate(freeze) == ("freeze", None, "freeze[embeddings,L0,L1]")


def test_distance_is_paired_by_seed_and_grouped_by_order(tmp_path):
    write_references(tmp_path)
    write_references(tmp_path, REVERSE, "rev", shift=0.5)
    write_joint(tmp_path)
    # seed 1's curve is 0.001 higher than seed 0's; each is compared to its own
    write_run(tmp_path, "map_s0", x=0.5, y=3.14, seed=0, gate="map")
    write_run(tmp_path, "map_s1", x=0.5, y=3.14, seed=1, gate="map")
    write_run(
        tmp_path, "rev_map_s0", x=0.5, y=3.14, seed=0, gate="map", domains=REVERSE
    )
    result = analyse(tmp_path)
    assert set(result["orders"]) == {"news>reviews>wiki", "wiki>reviews>news"}
    forward = result["orders"]["news>reviews>wiki"]
    runs = {r["run_dir"].rsplit("/", 1)[-1]: r for r in forward["runs"]}
    assert runs["map_s0"]["d"] == pytest.approx(3.14 - 3.15)
    assert runs["map_s1"]["d"] == pytest.approx(3.14 - 3.151)
    assert runs["map_s0"]["d"] < 0
    assert runs["map_s0"]["gap_closed"] == pytest.approx((3.2 - 3.14) / (3.2 - 2.9))
    reverse = result["orders"]["wiki>reviews>news"]
    rev = [r for r in reverse["runs"] if r["gate"] == "map"][0]
    assert rev["d"] == pytest.approx(3.14 - 3.65)
    assert forward["conditions"]["map_b1"]["n"] == 2
    assert forward["conditions"]["lr1e-05"]["n"] == 3
    assert forward["conditions"]["lr1e-04"]["d"]["mean"] == pytest.approx(0.0)
    assert result["joint"]["y"]["mean"] == pytest.approx(JOINT_Y)
    assert forward["mean_curve"]["lr"] == [1e-5, 3e-5, 1e-4]
    assert forward["mean_curve"]["y"][0] == pytest.approx(3.301)


def test_primary_test_pass_and_fail(tmp_path):
    write_references(tmp_path)
    write_references(tmp_path, REVERSE, "rev")
    write_joint(tmp_path)
    for seed in range(3):
        # b1: below the curve in every seed by far more than seq's seed SD 0.001
        write_run(tmp_path, f"b1_s{seed}", x=0.5, y=3.10, seed=seed, gate="map")
        write_run(
            tmp_path,
            f"rb1_s{seed}",
            x=0.5,
            y=3.10,
            seed=seed,
            gate="map",
            domains=REVERSE,
        )
        # b2: one seed above the curve
        y = 3.2 if seed == 2 else 3.10
        write_run(tmp_path, f"b2_s{seed}", x=0.5, y=y, seed=seed, gate="map", beta=2.0)
        # b0.5: below in every seed, but by less than seq's seed SD
        write_run(
            tmp_path,
            f"b05_s{seed}",
            x=0.5,
            y=3.15 + 0.001 * seed - 0.0005,
            seed=seed,
            gate="map",
            beta=0.5,
        )
    result = analyse(tmp_path)
    tests = result["orders"]["news>reviews>wiki"]["primary_test"]
    assert tests["1"]["pass"] is True
    assert tests["1"]["seq_y_sd"] == pytest.approx(0.001)
    assert tests["2"]["all_seeds_negative"] is False and tests["2"]["pass"] is False
    assert tests["0.5"]["all_seeds_negative"] is True
    assert tests["0.5"]["exceeds_seq_sd"] is False and tests["0.5"]["pass"] is False
    both = result["primary_test_both_orders"]
    assert both["1"]["pass"] is True
    assert both["2"]["pass"] is False
    assert both["2"]["per_order"]["wiki>reviews>news"] is None


def test_paired_contrasts(tmp_path):
    write_references(tmp_path)
    write_joint(tmp_path)
    for seed in range(3):
        write_run(tmp_path, f"map_s{seed}", x=0.5, y=3.10, seed=seed, gate="map")
        y_random = 3.12 if seed < 2 else 3.05
        write_run(tmp_path, f"rnd_s{seed}", x=0.5, y=y_random, seed=seed, gate="random")
    write_run(tmp_path, "map2_s0", x=0.5, y=3.1, seed=0, gate="map", beta=2.0)
    contrasts = analyse(tmp_path)["orders"]["news>reviews>wiki"]["contrasts"]
    assert set(contrasts) == {"map-random_b1"}
    entry = contrasts["map-random_b1"]
    assert entry["seeds"] == [0, 1, 2]
    assert entry["y"]["per_seed"] == pytest.approx([-0.02, -0.02, 0.05])
    assert entry["y"]["fraction_negative"] == pytest.approx(2 / 3)
    assert entry["d"]["per_seed"] == pytest.approx([-0.02, -0.02, 0.05])
    assert entry["y"]["mean"] == pytest.approx(0.01 / 3)


def test_incomplete_and_non_bp_runs_are_skipped(tmp_path, capsys):
    write_references(tmp_path, seeds=(0,))
    write_run(tmp_path, "unfinished", x=0.5, y=3.0, seed=0, gate="map", finished=False)
    single = write_run(tmp_path, "single", x=0.5, y=3.0, seed=0, domains=("news",))
    points = collect_runs(sorted(tmp_path.iterdir()))
    names = {p["run_dir"].rsplit("/", 1)[-1] for p in points}
    assert "unfinished" not in names and "single" not in names
    assert len(points) == 3
    output = capsys.readouterr().out
    assert "unfinished" in output and "warning" in output
    assert str(single) in output


def test_cli_writes_json_and_figure(tmp_path):
    from examples import domain_forgetting_analysis as cli

    runs = tmp_path / "runs"
    runs.mkdir()
    write_references(runs)
    write_references(runs, REVERSE, "rev")
    write_joint(runs)
    for seed in range(3):
        for order, tag in ((FORWARD, "f"), (REVERSE, "r")):
            for gate in ("map", "random", "type"):
                write_run(
                    runs,
                    f"{tag}_{gate}_s{seed}",
                    x=0.45,
                    y=3.12,
                    seed=seed,
                    gate=gate,
                    domains=order,
                )
            write_run(
                runs,
                f"{tag}_freeze_s{seed}",
                x=0.1,
                y=3.0,
                seed=seed,
                gate="freeze",
                modules=("embeddings",),
                domains=order,
            )
    output = tmp_path / "out"
    result = cli.main(
        ["frontier", "--results_dir", str(runs), "--output_dir", str(output), "--quiet"]
    )
    written = json.loads((output / "frontier.json").read_text())
    assert written["primary_test_both_orders"]["1"]["pass"] is True
    assert set(written["orders"]["news>reviews>wiki"]["contrasts"]) == {
        "map-random_b1",
        "map-type_b1",
    }
    freeze = written["orders"]["news>reviews>wiki"]["conditions"]["freeze[embeddings]"]
    assert freeze["d"]["mean"] is None and set(freeze["outside_range"]) == {
        "0",
        "1",
        "2",
    }
    assert (output / "frontier.pdf").stat().st_size > 0
    assert (output / "frontier.png").stat().st_size > 0
    table = format_table(result)
    assert "map_b1" in table and "freeze[embeddings]" in table
