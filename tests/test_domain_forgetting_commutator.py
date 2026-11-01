"""Tests for update-path logging, the uniform gate and the commutator analysis."""

import json
import math
import re
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

from fabricpc.experiments.domain_forgetting.commutator import (
    commutator_test,
    compute_commutator,
    flat_distance,
    path_ratio,
    prediction_divergence,
    read_paths,
)
from fabricpc.experiments.domain_forgetting.localization import load_run
from fabricpc.experiments.domain_forgetting.runner import run_domain_forgetting
from tests.test_domain_forgetting_runner import make_setup, tiny_config

PAIR = ("news", "wiki")
SEEDS = (0, 1)


def _jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_flat_distance_known_answer():
    a = {"m1/W": np.array([1.0, 2.0]), "m2/b": np.array([0.0])}
    b = {"m1/W": np.array([4.0, 6.0]), "m2/b": np.array([12.0])}
    total, modules = flat_distance(a, b)
    assert modules == {"m1": 5.0, "m2": 12.0}
    assert total == pytest.approx(13.0)
    assert total**2 == pytest.approx(sum(v**2 for v in modules.values()))


def test_flat_distance_rejects_different_keys():
    with pytest.raises(ValueError, match="different keys"):
        flat_distance({"a/W": np.zeros(1)}, {"b/W": np.zeros(1)})


def test_path_ratio_is_zero_for_identical_params():
    a = {"m/W": np.ones(3)}
    path = {"total": 2.0, "modules": {"m": 2.0}}
    result = path_ratio(a, a, path, path)
    assert result["ratio"] == 0.0 and result["module_ratio"] == {"m": 0.0}
    assert result["path"] == 4.0


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("commutator")
    checkpoint, data_dir = make_setup(root / "setup")
    out = {}
    for label, gate in (
        ("none", {}),
        ("uniform", {"gate": "uniform", "gate_scale": 0.5}),
    ):
        for seed in SEEDS:
            for order in (PAIR, PAIR[::-1]):
                path = root / f"{label}_{'-'.join(order)}_s{seed}"
                config = tiny_config(
                    checkpoint, data_dir, path, domains=order, seed=seed, **gate
                )
                run_domain_forgetting(config, verbose=False)
                out[(label, order, seed)] = path
    one = root / "uniform_one"
    config = tiny_config(
        checkpoint, data_dir, one, domains=PAIR, gate="uniform", gate_scale=1.0
    )
    run_domain_forgetting(config, verbose=False)
    out["uniform_one"] = one
    single = root / "single_update"
    config = tiny_config(
        checkpoint, data_dir, single, domains=PAIR, tokens_per_domain=20
    )
    run_domain_forgetting(config, verbose=False)
    out["single_update"] = single
    return out


def test_update_norms_are_logged_and_bound_the_chord(runs):
    path = runs[("none", PAIR, 0)]
    context = load_run(path)
    stages = read_paths(path, len(PAIR))
    for stage, record in enumerate(stages, start=1):
        log = _jsonl(path / f"training_log_stage_{stage}.jsonl")
        for entry in log:
            parts = entry["module_update_norms"].values()
            assert entry["update_norm"] == pytest.approx(
                math.sqrt(sum(v * v for v in parts)), rel=1e-6
            )
        chord = flat_distance(context.snapshots[stage], context.snapshots[stage - 1])[0]
        assert record["total"] >= chord * (1 - 1e-5)


def test_single_update_path_equals_chord(runs):
    path = runs["single_update"]
    context = load_run(path)
    stages = read_paths(path, len(PAIR))
    for stage, record in enumerate(stages, start=1):
        assert len(_jsonl(path / f"training_log_stage_{stage}.jsonl")) == 1
        chord, modules = flat_distance(
            context.snapshots[stage], context.snapshots[stage - 1]
        )
        assert record["total"] == pytest.approx(chord, rel=1e-4)
        for name, value in modules.items():
            assert record["modules"][name] == pytest.approx(value, rel=1e-4, abs=1e-9)


def test_uniform_scale_one_matches_ungated(runs):
    gated = json.loads((runs["uniform_one"] / "summary.json").read_text())
    plain = json.loads((runs[("none", PAIR, 0)] / "summary.json").read_text())
    assert gated["stage_params_sha256"] == plain["stage_params_sha256"]


def test_uniform_gate_scales_block_modules_only(runs):
    records = _jsonl(runs[("uniform", PAIR, 0)] / "gates.jsonl")
    stage_two = next(r for r in records if r["stage"] == 2)
    for name, value in stage_two["multipliers"].items():
        expected = 0.5 if re.match(r"^L\d+_", name) else 1.0
        assert value == expected, name
    stage_one = next(r for r in records if r["stage"] == 1)
    assert set(stage_one["multipliers"].values()) == {1.0}


def test_uniform_gate_halves_block_update_norms(runs):
    gated = _jsonl(runs[("uniform", PAIR, 0)] / "training_log_stage_2.jsonl")
    plain = _jsonl(runs[("none", PAIR, 0)] / "training_log_stage_2.jsonl")
    first_gated, first_plain = (
        gated[0]["module_update_norms"],
        plain[0]["module_update_norms"],
    )
    for name in first_gated:
        if re.match(r"^L\d+_", name):
            assert first_gated[name] == pytest.approx(0.5 * first_plain[name], rel=1e-4)
        else:
            assert first_gated[name] == pytest.approx(first_plain[name], rel=1e-4)


def test_gate_scale_validation(tmp_path):
    with pytest.raises(ValueError, match="gate_scale must be finite and positive"):
        tiny_config(
            tmp_path, tmp_path, tmp_path, gate="uniform", gate_scale=0.0
        ).validate()
    with pytest.raises(ValueError, match="gate_scale do not apply"):
        tiny_config(tmp_path, tmp_path, tmp_path, gate="map", gate_scale=0.5).validate()


def test_gate_scale_is_left_out_of_other_gates_identity(tmp_path):
    config = tiny_config(tmp_path, tmp_path, tmp_path, gate="map")
    assert "gate_scale" not in config.identity()
    uniform = tiny_config(tmp_path, tmp_path, tmp_path, gate="uniform", gate_scale=0.5)
    assert uniform.identity()["gate_scale"] == 0.5


def test_prediction_divergence_is_zero_for_identical_params(runs):
    context = load_run(runs[("none", PAIR, 0)])
    params = context.params(-1)
    value = prediction_divergence(
        context.structure, context.valid_sets["news"], params, params, batch_size=3
    )
    assert value == pytest.approx(0.0, abs=1e-6)
    other = context.params(0)
    assert (
        prediction_divergence(
            context.structure, context.valid_sets["news"], params, other, batch_size=3
        )
        > 0
    )


def test_compute_commutator_pairs_orders_and_seeds(runs):
    dirs = [v for k, v in runs.items() if isinstance(k, tuple)]
    result = compute_commutator(dirs, batch_size=4)
    rows = {r["condition"]: r for r in result["results"]}
    assert set(rows) == {"lr1e-02", "uniform_s0.5@lr1e-02"}
    for row in rows.values():
        assert row["pair"] == sorted(PAIR)
        assert row["seeds"] == list(SEEDS)
        assert row["D"]["n"] == len(SEEDS)
        assert all(len(v) == 1 for v in row["noise"].values())
        assert row["N"]["n"] == 2
        for entry in row["per_seed"]:
            assert entry["ratio"] > 0
            assert set(entry["symmetric_kl"]) == set(PAIR)
            assert all(v >= 0 for v in entry["symmetric_kl"].values())
        for chords in row["chord_to_path"].values():
            for values in chords.values():
                assert all(0 < v <= 1 + 1e-5 for v in values)
    test = commutator_test(result, "uniform_s0.5@lr1e-02", "lr1e-02", PAIR)
    assert test["seeds"] == list(SEEDS) and isinstance(test["passed"], bool)


def test_compute_commutator_needs_both_orders(runs):
    dirs = [runs[("none", PAIR, 0)], runs[("none", PAIR, 1)]]
    with pytest.raises(ValueError, match="both domain orders"):
        compute_commutator(dirs, divergence=False)
