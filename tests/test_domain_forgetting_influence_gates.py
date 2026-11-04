"""Tests for influence weights, the online influence gate and its diagnostics."""

import fnmatch
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fabricpc.experiments.domain_forgetting import runner
from fabricpc.experiments.domain_forgetting.artifacts import atomic_write_json
from fabricpc.experiments.domain_forgetting.commutator import (
    commutator_test,
    compute_commutator,
)
from fabricpc.experiments.domain_forgetting.data import load_domain_cache
from fabricpc.experiments.domain_forgetting.frontier import (
    collect_runs,
    compute_frontier,
    describe_gate,
    format_table,
    frontier_distance,
    envelope_distance,
    lower_envelope,
    pass_rule,
)
from fabricpc.experiments.domain_forgetting.gates import (
    INFLUENCE_PROBE_SKIP,
    INFLUENCE_PROBE_WINDOWS,
    InfluenceProbe,
    influence_multipliers,
    influence_weights,
    spaced_probe,
    spaced_windows,
    with_module_from,
)
from fabricpc.experiments.domain_forgetting.influence import (
    InterventionSteps,
    NOISE_DRAWS,
    ablated_params,
    compute_influence_runs,
    format_influence_table,
    probe_ce,
)
from fabricpc.experiments.domain_forgetting.localization import (
    compute_rollback,
    load_run,
)
from fabricpc.experiments.domain_forgetting.runner import (
    load_base_model,
    run_domain_forgetting,
)
from fabricpc.training.train_backprop import compute_forward_pass
from fabricpc.experiments.domain_forgetting.validation import (
    MultiDomainEvaluator,
    evaluate_fixed_set,
)
from fabricpc.training.plasticity_gates import weighted_gates
from tests.test_domain_forgetting_data import build_tiny_caches, tiny_tokenizer
from tests.test_domain_forgetting_runner import (
    DOMAINS,
    tiny_config,
    write_tiny_bert_checkpoint,
)

MAP_WINDOWS = 12
BLOCKS = ("L0_mha", "L0_mlp1", "L0_mlp2", "L1_mha", "L1_mlp1", "L1_mlp2")


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    directory = tmp_path_factory.mktemp("influence_gates")
    tokenizer = tiny_tokenizer(directory)
    checkpoint = write_tiny_bert_checkpoint(directory, tokenizer, depth=2)
    build_tiny_caches(directory, tokenizer)
    return checkpoint, directory / "data"


def _online(setup, output, gate="online_influence", **overrides):
    checkpoint, data_dir = setup
    values = {
        "gate": gate,
        "gate_map_windows": MAP_WINDOWS,
        "gate_interval": 1,
        **overrides,
    }
    return tiny_config(checkpoint, data_dir, output, **values)


@pytest.fixture(scope="module")
def surprise_run(setup, tmp_path_factory):
    config = _online(
        setup, tmp_path_factory.mktemp("surprise") / "run", gate="online_surprise"
    )
    return config, run_domain_forgetting(config, verbose=False)


@pytest.fixture(scope="module")
def influence_run(setup, tmp_path_factory):
    config = _online(setup, tmp_path_factory.mktemp("influence") / "run")
    return config, run_domain_forgetting(config, verbose=False)


@pytest.mark.parametrize(
    "influence, floor, expected",
    [
        ({"a": 1.0, "b": 3.0}, 0.0, {"a": 0.5, "b": 1.5}),
        ({"a": 1.0, "b": 9.0}, 0.0, {"a": 0.2, "b": 1.8}),
        ({"a": 1.0, "b": 1.0, "c": 7.0}, 0.0, {"a": 1 / 3, "b": 1 / 3, "c": 2.0}),
        ({"a": 2.0, "b": -1.0, "c": 0.0}, 0.0, {"a": 1.0, "b": 0.0, "c": 0.0}),
        ({"a": 2.0, "b": 1e-9, "c": -1e-9}, 1e-6, {"a": 1.0, "b": 1.0, "c": 1.0}),
        ({"a": -2.0, "b": 0.0}, 0.0, {"a": 1.0, "b": 1.0}),
        ({"a": -2.0, "b": 1e-9}, 1e-6, {"a": 1.0, "b": 1.0}),
        ({"a": 1e-9, "b": -1e-9}, 1e-6, {"a": 1.0, "b": 1.0}),
    ],
)
def test_influence_weights_known_answers(influence, floor, expected):
    assert influence_weights(influence, floor) == pytest.approx(expected)


def test_influence_weights_value_at_the_floor_is_informative():
    assert influence_weights({"a": 1.0, "b": -1.0, "c": 3.0}, 1.0) == pytest.approx(
        {"a": 0.5, "b": 0.0, "c": 1.5}
    )
    with pytest.raises(ValueError, match="non-finite"):
        influence_weights({"a": float("nan")})


def test_influence_multipliers_match_the_weighted_gate():
    g = {"a": 0.6, "b": 0.5, "c": 0.25}
    h = {"a": 2.0, "b": 0.0, "c": 1.0}
    assert influence_multipliers(g, h, 0.5) == pytest.approx(
        {"a": 0.9, "b": 0.25, "c": 0.25}
    )
    assert influence_multipliers({"a": 0.9}, {"a": 2.0}, 0.5) == pytest.approx(
        {"a": 1.0}
    )
    plain = influence_multipliers(g, h, 0.0)
    assert plain == {k: float(jnp.asarray(v)) for k, v in g.items()}
    direct = weighted_gates(
        {k: jnp.asarray(v) for k, v in g.items()},
        {k: jnp.asarray(v) for k, v in h.items()},
        0.5,
    )
    assert influence_multipliers(g, h, 0.5) == {k: float(v) for k, v in direct.items()}


def test_alpha_zero_reproduces_surprise_bit_for_bit(setup, surprise_run, tmp_path):
    base, base_result = surprise_run
    config = _online(setup, tmp_path / "run", gate_influence_alpha=0.0)
    result = run_domain_forgetting(config, verbose=False)
    assert (
        result["summary"]["stage_params_sha256"]
        == base_result["summary"]["stage_params_sha256"]
    )
    for stage in (2, 3):
        name = f"gate_schedule_stage_{stage}.jsonl"
        ours = _read_jsonl(Path(config.output_dir) / name)
        theirs = _read_jsonl(Path(base.output_dir) / name)
        assert [e["multipliers"] for e in ours] == [e["multipliers"] for e in theirs]
        assert [e["energies"] for e in ours] == [e["energies"] for e in theirs]
        assert any(e["influence"] is not None for e in ours)


def test_schedule_records_influence_weights_and_final_multipliers(influence_run):
    config, _ = influence_run
    for stage in (2, 3):
        schedule = _read_jsonl(
            Path(config.output_dir) / f"gate_schedule_stage_{stage}.jsonl"
        )
        assert [e["refresh"] for e in schedule] == [-1, 0, 1]
        onset, first, later = schedule
        for entry in (onset, first):
            assert entry["influence"] is None
            assert set(entry["influence_h"].values()) == {1.0}
        assert onset["influence_probe"]["windows"] == INFLUENCE_PROBE_WINDOWS
        assert onset["influence_probe"]["domain"] == DOMAINS[stage - 1]
        assert first["multipliers"]["L0_mha"] == pytest.approx(
            first["surprise_multipliers"]["L0_mha"], rel=1e-6
        )
        assert sorted(later["influence"]) == sorted(BLOCKS)
        assert later["influence_floor"] == pytest.approx(
            1e-6 * later["influence_probe_loss"]
        )
        assert later["influence_h"] == pytest.approx(
            influence_weights(later["influence"], later["influence_floor"])
        )
        expected = influence_multipliers(
            later["surprise_multipliers"], later["influence_h"], 0.5
        )
        for module in BLOCKS:
            assert later["multipliers"][module] == pytest.approx(
                expected[module], rel=1e-6
            )
        for module in set(later["multipliers"]) - set(BLOCKS):
            assert later["multipliers"][module] == 1.0


def test_stage_one_is_ungated_and_windows_do_not_cross_stages(influence_run):
    config, _ = influence_run
    output = Path(config.output_dir)
    assert not (output / "gate_schedule_stage_1.jsonl").exists()
    records = _read_jsonl(output / "gates.jsonl")
    assert records[0]["gated"] is False
    assert set(records[0]["multipliers"].values()) == {1.0}
    for stage in (2, 3):
        schedule = _read_jsonl(output / f"gate_schedule_stage_{stage}.jsonl")
        assert [r["influence"] is None for r in schedule[:2]] == [True, True]
        assert all(r["influence"] is not None for r in schedule[2:])


def test_each_refresh_covers_one_interval_and_divides_by_the_held_multiplier(
    influence_run,
):
    config, _ = influence_run
    schedule = _read_jsonl(Path(config.output_dir) / "gate_schedule_stage_2.jsonl")
    assert len(schedule) > 2
    for previous, record in zip(schedule[1:], schedule[2:]):
        assert previous["last_update"] - previous["first_update"] + 1 == (
            config.gate_interval
        )
        for module, value in record["influence"].items():
            assert record["influence_per_multiplier"][module] == pytest.approx(
                value / previous["multipliers"][module]
            )


def test_past_variant_probes_the_previous_domain(setup, tmp_path):
    config = _online(setup, tmp_path / "run", gate_variant="past")
    run_domain_forgetting(config, verbose=False)
    output = Path(config.output_dir)
    assert not (output / "gate_schedule_stage_1.jsonl").exists()
    for stage in (2, 3):
        onset = _read_jsonl(output / f"gate_schedule_stage_{stage}.jsonl")[0]
        assert onset["influence_probe"]["domain"] == DOMAINS[stage - 2]


def _tiny_model(setup, tmp_path):
    config = _online(setup, tmp_path / "unused")
    base = load_base_model(config)
    cache = load_domain_cache(
        config.data_dir,
        "news",
        tokenizer_json=base.tokenizer_json,
        seq_len=int(base.model_config["seq_len"]),
        tokens_per_domain=config.tokens_per_domain,
    )
    return config, base, cache


def test_probe_is_spaced_fixed_and_disjoint_from_validation(setup, tmp_path):
    _, base, cache = _tiny_model(setup, tmp_path)
    kwargs = {"num_windows": 3, "mask_prob": 0.3, "seed": 5}
    first, windows = spaced_probe(
        cache.train, base.structure, base.special_ids, **kwargs
    )
    again, _ = spaced_probe(cache.train, base.structure, base.special_ids, **kwargs)
    for name in first:
        np.testing.assert_array_equal(first[name], again[name])
    seq_len = int(base.model_config["seq_len"])
    stride = seq_len - 1
    for row, tokens in enumerate(first["labels"]):
        start = int(windows[row]) * stride
        np.testing.assert_array_equal(tokens[1:], cache.train[start : start + stride])
    valid = {tuple(w) for w in np.asarray(cache.valid["input_ids"]).tolist()}
    assert not {tuple(w) for w in np.asarray(first["labels"]).tolist()} & valid


def test_spaced_windows_skip_the_map_windows_and_reach_the_last_full_window():
    stream = np.zeros(127 * 5000 + 10, dtype=np.int32)
    windows = spaced_windows(stream, 128, 256)
    assert windows[0] == INFLUENCE_PROBE_SKIP
    assert windows[-1] == 4999
    assert len(set(windows.tolist())) == 256
    gaps = np.diff(windows)
    assert gaps.max() - gaps.min() <= 1
    short = np.zeros(127 * 300, dtype=np.int32)
    assert spaced_windows(short, 128, 256)[0] == 0
    assert spaced_windows(short, 128, 256)[-1] == 299


def test_leave_one_module_out_equals_a_direct_evaluation(setup, tmp_path):
    _, base, cache = _tiny_model(setup, tmp_path)
    arrays, _ = spaced_probe(
        cache.train,
        base.structure,
        base.special_ids,
        num_windows=8,
        mask_prob=0.3,
        seed=5,
    )
    start = base.params
    params = jax.tree_util.tree_map(lambda leaf: leaf + 0.01, start)
    probe = InfluenceProbe(base.structure, arrays, BLOCKS)
    assert probe.measure(start)["influence"] is None
    audit = probe.measure(params)

    def direct(tree):
        return evaluate_fixed_set(tree, base.structure, arrays, batch_size=8)[
            "masked_ce"
        ]

    assert audit["influence_probe_loss"] == pytest.approx(direct(params), abs=1e-5)
    for module in BLOCKS:
        nodes = dict(params.nodes)
        nodes[module] = jax.tree_util.tree_map(
            lambda p, s: p - (p - s), params.nodes[module], start.nodes[module]
        )
        assert audit["influence"][module] == pytest.approx(
            direct(type(params)(nodes=nodes)) - direct(params), abs=1e-5
        )
    values = np.asarray(list(audit["influence"].values()))
    floor = audit["influence_floor"]
    assert audit["influence_floored_fraction"] == np.mean(np.abs(values) < floor)
    assert audit["influence_positive_fraction"] == np.mean(values >= floor)
    agreement = audit["influence_split_half_spearman"]
    assert agreement is None or -1.0 <= agreement <= 1.0
    assert probe.start is params
    left = jax.tree_util.tree_leaves(start)
    right = jax.tree_util.tree_leaves(base.params)
    assert all(a is b for a, b in zip(left, right))


def test_first_measure_is_neutral_without_a_forward_pass(setup, tmp_path):
    _, base, cache = _tiny_model(setup, tmp_path)
    arrays, _ = spaced_probe(
        cache.train,
        base.structure,
        base.special_ids,
        num_windows=4,
        mask_prob=0.3,
        seed=5,
    )
    probe = InfluenceProbe(base.structure, arrays, BLOCKS)
    probe.evaluator = None
    audit = probe.measure(base.params)
    assert audit["influence"] is None and set(audit["influence_h"].values()) == {1.0}


def test_resume_reuses_finished_stages_and_retrains_a_deleted_one(
    influence_run, tmp_path
):
    config, result = influence_run
    copy = tmp_path / "copy"
    shutil.copytree(config.output_dir, copy)
    resumed = runner.replace(config, output_dir=str(copy))
    run_domain_forgetting(resumed, resume=True, verbose=False)
    stages = json.loads((copy / "runtime.json").read_text())["stages"]
    assert all(stages[str(s)].get("reused") for s in range(4))
    (copy / "stage_3.npz").unlink()
    third = run_domain_forgetting(resumed, resume=True, verbose=False)
    assert (
        third["summary"]["stage_params_sha256"]
        == result["summary"]["stage_params_sha256"]
    )
    stages = json.loads((copy / "runtime.json").read_text())["stages"]
    assert stages["2"].get("reused") and not stages["3"].get("reused")


def test_validation_and_identity(setup, tmp_path):
    with pytest.raises(ValueError, match="gate_influence_alpha"):
        _online(setup, tmp_path / "a", gate_influence_alpha=1.0).validate()
    with pytest.raises(ValueError, match="takes gate_variant"):
        _online(setup, tmp_path / "b", gate_variant="shuffle", gate_seed=1).validate()
    with pytest.raises(ValueError, match="applies only"):
        _online(
            setup, tmp_path / "c", gate="online_surprise", gate_variant="past"
        ).validate()
    with pytest.raises(ValueError, match="do not apply"):
        _online(
            setup, tmp_path / "d", gate="online_surprise", gate_influence_alpha=0.1
        ).validate()
    surprise = _online(setup, tmp_path / "e", gate="online_surprise").identity()
    assert "gate_influence_alpha" not in surprise
    assert _online(setup, tmp_path / "f").identity()["gate_influence_alpha"] == 0.5


def test_labels():
    config = {"gate": "online_influence", "lr": 1e-4, "gate_influence_alpha": 0.5}
    assert describe_gate(config)[2] == "online_influence"
    assert describe_gate({**config, "gate_variant": "past"})[2] == (
        "online_influence_past"
    )
    assert describe_gate({**config, "gate_map_source": "learner", "lr": 3e-5})[2] == (
        "online_influence_learner@lr3e-05"
    )
    assert describe_gate({**config, "gate_influence_alpha": 0.0})[2] == (
        "online_influence_a0"
    )


def test_run_name_prefix_is_outside_the_existing_globs():
    globs = (
        "online_protect*_seed?",
        "online_surprise*_seed?",
        "online_matched_*_seed?",
        "g9_*_seed?",
        "epcsuite_*_seed?",
        "epc_fwd_lr3e-5_seed?",
        "commutator_*_seed*",
    )
    names = (
        "infl_bp_live_fwd_seed0",
        "infl_epc_matched_rev_seed1",
        "infl_bp_pair_matched_pair_fwd_seed0",
    )
    for name in names:
        assert not any(fnmatch.fnmatch(name, glob) for glob in globs)
    assert not fnmatch.fnmatch(
        "infl_bp_pair_matched_pair_fwd_seed0", "infl_bp_matched_*"
    )


REVERSE = {
    "lr": [1e-5, 2e-5, 3e-5, 5e-5, 1e-4],
    "x": [0.256, 0.265, 0.269, 0.295, 0.296],
    "y": [3.283, 3.840, 3.713, 3.342, 3.448],
    "bwt": [-0.1, -0.5, -0.4, -0.2, -0.3],
}


def test_envelope_flips_the_sign_on_the_non_monotone_curve():
    point = {"x": 0.277, "y": 3.42, "bwt": -0.2}
    linear = frontier_distance(point, REVERSE)
    envelope = envelope_distance(point, REVERSE)
    assert lower_envelope(REVERSE)["x"] == [0.256, 0.295, 0.296]
    assert lower_envelope(REVERSE)["lr"] == [1e-5, 5e-5, 1e-4]
    assert linear["d"] == pytest.approx(
        3.42 - np.interp(0.277, REVERSE["x"], REVERSE["y"])
    )
    assert linear["d"] < -0.17
    assert envelope["d_envelope"] == pytest.approx(0.105, abs=1e-3)
    expected_bwt = -0.2 - np.interp(0.277, [0.256, 0.295, 0.296], [-0.1, -0.2, -0.3])
    assert envelope["d_bwt_envelope"] == pytest.approx(expected_bwt)
    assert envelope["outside_range_envelope"] is None


def test_envelope_equals_linear_on_a_curve_rising_with_gain():
    curve = {
        "lr": [1e-5, 3e-5, 1e-4],
        "x": [0.2, 0.4, 0.6],
        "y": [3.0, 3.1, 3.3],
        "bwt": [-0.1, -0.2, -0.4],
    }
    assert lower_envelope(curve) == curve
    for x in (0.2, 0.33, 0.6):
        point = {"x": x, "y": 3.05, "bwt": -0.15}
        linear = frontier_distance(point, curve)
        envelope = envelope_distance(point, curve)
        assert envelope["d_envelope"] == linear["d"]
        assert envelope["d_bwt_envelope"] == linear["d_bwt"]


def test_envelope_ties_and_duplicates():
    curve = {
        "lr": [1, 2, 3, 4, 5],
        "x": [0.1, 0.2, 0.2, 0.3, 0.3],
        "y": [1.0, 2.0, 3.0, 4.0, 4.0],
        "bwt": [0.0, 0.1, 0.2, 0.3, 0.4],
    }
    kept = lower_envelope(curve)
    assert kept["lr"] == [1, 2, 4]
    same_y = {"lr": [1, 2], "x": [0.1, 0.2], "y": [1.0, 1.0], "bwt": [0.0, 0.1]}
    assert lower_envelope(same_y)["lr"] == [2]


def test_envelope_of_a_falling_curve_holds_its_last_point_inside_the_curve():
    curve = {
        "lr": [1, 2, 3],
        "x": [0.1, 0.2, 0.3],
        "y": [3.0, 2.0, 1.0],
        "bwt": [0, 0, 0],
    }
    assert lower_envelope(curve)["x"] == [0.3]
    below = envelope_distance({"x": 0.2, "y": 1.5, "bwt": 0.0}, curve)
    assert below["d_envelope"] == 0.5
    assert below["below_envelope"] is True
    assert below["outside_range_envelope"] is None
    outside = envelope_distance({"x": 0.05, "y": 0.5, "bwt": 0.0}, curve)
    assert outside["d_envelope"] is None
    assert outside["below_envelope"] is False
    assert outside["outside_range_envelope"]["side"] == "below"
    above = envelope_distance({"x": 0.4, "y": 0.5, "bwt": 0.0}, curve)
    assert above["d_envelope"] is None
    assert above["outside_range_envelope"]["side"] == "above"
    assert frontier_distance({"x": 0.2, "y": 1.5, "bwt": 0.0}, curve)["d"] == -0.5
    at = envelope_distance({"x": 0.3, "y": 1.5, "bwt": 0.0}, curve)
    assert at["d_envelope"] == 0.5


def _write_run(root, name, *, x, y, seed, lr, domains, gate=None, scale=None):
    path = root / name
    path.mkdir()
    config = {"lr": lr, "seed": seed, "domains": list(domains), "mode": "backprop"}
    if gate is not None:
        config["gate"] = gate
    if scale is not None:
        config["gate_scale"] = scale
    (path / "run_config.json").write_text(json.dumps({"config": config}))
    (path / "summary.json").write_text(
        json.dumps(
            {
                "domains": list(domains),
                "mode": "backprop",
                "final_ce": {d: y for d in domains},
                "learning_gain_ce": {d: x for d in domains},
                "bwt_ce": -0.3,
            }
        )
    )
    return path


def _frontier_runs(tmp_path, *, surprise_lr=3e-5):
    dirs = []
    orders = {"fwd": DOMAINS, "rev": tuple(reversed(DOMAINS))}
    for tag, domains in orders.items():
        for seed in range(3):
            jitter = seed * 1e-3
            for lr, (x, y) in {
                1e-5: (0.256, 3.283 + jitter),
                2e-5: (0.265, 3.840 + jitter),
                3e-5: (0.269, 3.713 + jitter),
                5e-5: (0.295, 3.342 + jitter),
                1e-4: (0.296, 3.448 + jitter),
            }.items():
                dirs.append(
                    _write_run(
                        tmp_path,
                        f"ref_{lr}_{tag}_{seed}",
                        x=x,
                        y=y,
                        seed=seed,
                        lr=lr,
                        domains=domains,
                    )
                )
            for name, gate, scale, y, lr in (
                ("infl", "online_influence", None, 3.42, 3e-5),
                ("match", "uniform", 0.48, 3.60, 3e-5),
                ("surp", "online_surprise", None, 3.55, surprise_lr),
            ):
                dirs.append(
                    _write_run(
                        tmp_path,
                        f"{name}_{tag}_{seed}",
                        x=0.277,
                        y=y + jitter,
                        seed=seed,
                        lr=lr,
                        domains=domains,
                        gate=gate,
                        scale=scale,
                    )
                )
    return dirs


def test_verdict_reports_both_computations_and_three_comparisons(tmp_path):
    points = collect_runs(_frontier_runs(tmp_path))
    labels = (
        "online_influence@lr3e-05",
        "uniform_s0.48@lr3e-05",
        "online_surprise@lr3e-05",
    )
    plain = compute_frontier(points, seq_lr=3e-5)
    result = compute_frontier(points, seq_lr=3e-5, influence_verdicts=[labels])
    assert plain["orders"].keys() == result["orders"].keys()
    for key in ("curves", "seq", "expected_seeds"):
        for order in plain["orders"]:
            assert plain["orders"][order][key] == result["orders"][order][key]
    verdict = result["influence_verdicts"][0]
    linear = verdict["pre_registered"]
    envelope = verdict["envelope_post_hoc"]
    assert linear["post_hoc"] is False and envelope["post_hoc"] is True
    assert linear["passes"]["frontier"] is True
    assert envelope["passes"]["frontier"] is False
    assert linear["passes"]["vs_matched"] and linear["passes"]["vs_surprise"]
    assert envelope["passes"]["vs_matched"] and envelope["passes"]["vs_surprise"]
    assert linear["pass"] is True and envelope["pass"] is False
    assert verdict["lr_mismatch"] is False
    row = next(iter(linear["orders"].values()))
    assert row["vs_matched"]["values"] == pytest.approx([-0.18] * 3)
    assert row["vs_surprise"]["values"] == pytest.approx([-0.13] * 3)
    run = next(
        r
        for r in result["orders"][">".join(DOMAINS)]["runs"]
        if r["label"] == labels[0]
    )
    assert run["d"] < 0 < run["d_envelope"]
    table = format_table(result)
    assert "d env (post hoc)" in table and "envelope_post_hoc" in table


def test_verdict_flags_a_learning_rate_mismatch_and_a_missing_condition(tmp_path):
    points = collect_runs(_frontier_runs(tmp_path, surprise_lr=1e-4))
    labels = ("online_influence@lr3e-05", "uniform_s0.48@lr3e-05", "online_surprise")
    verdict = compute_frontier(points, seq_lr=3e-5, influence_verdicts=[labels])[
        "influence_verdicts"
    ][0]
    assert verdict["lr_mismatch"] is True
    labels = ("online_influence@lr3e-05", "uniform_s0.48@lr3e-05", "absent")
    missing = compute_frontier(points, seq_lr=3e-5, influence_verdicts=[labels])[
        "influence_verdicts"
    ][0]
    assert missing["pre_registered"]["passes"]["vs_surprise"] is False
    assert missing["pre_registered"]["passes"]["vs_matched"] is True
    assert missing["pre_registered"]["status"]["vs_surprise"] == "undefined"
    assert missing["pre_registered"]["status"]["vs_matched"] == "pass"
    assert missing["lr_mismatch"] is False
    assert missing["missing_conditions"] == ["absent"]
    assert missing["descriptive_only"] is False
    assert verdict["descriptive_only"] is True
    rows = missing["pre_registered"]["orders"]
    for order in rows.values():
        assert all(v is None for v in order["vs_surprise"]["paired_y"])
        assert all(v is not None for v in order["vs_matched"]["paired_y"])
    marked = compute_frontier(points, seq_lr=3e-5, descriptive_verdicts=[labels])[
        "influence_verdicts"
    ][0]
    assert marked["descriptive_only"] is True


def test_pass_rule():
    assert pass_rule([-1.0, -1.1], 0.5)["pass"] is True
    assert pass_rule([-1.0, 0.1], 0.1)["pass"] is False
    assert pass_rule([-0.1, -0.1], 0.5)["pass"] is False
    assert pass_rule([-1.0, None], 0.1)["pass"] is False
    assert pass_rule([-1.0, -1.0], None)["pass"] is False
    assert pass_rule([-1.0, -1.1], 0.5)["status"] == "pass"
    assert pass_rule([-1.0, 0.1], 0.1)["status"] == "fail"
    assert pass_rule([-0.1, -0.1], 0.5)["status"] == "fail"
    assert pass_rule([-1.0, None], 0.1)["status"] == "undefined"
    assert pass_rule([None, None], 0.1)["status"] == "undefined"
    assert pass_rule([0.2, None], 0.1)["status"] == "fail"
    assert pass_rule([], 0.1)["status"] == "undefined"


def _zeroed(params, module, keys):
    node = params.nodes[module]
    weights = {
        k: jnp.zeros_like(v) if k in keys else v for k, v in node.weights.items()
    }
    biases = {k: jnp.zeros_like(v) if k in keys else v for k, v in node.biases.items()}
    return type(params)(
        nodes={**params.nodes, module: type(node)(weights=weights, biases=biases)}
    )


@pytest.fixture(scope="module")
def probe_setup(setup, tmp_path_factory):
    _, base, cache = _tiny_model(setup, tmp_path_factory.mktemp("probe"))
    arrays, _ = spaced_probe(
        cache.train,
        base.structure,
        base.special_ids,
        num_windows=8,
        mask_prob=0.3,
        seed=5,
    )
    evaluator = MultiDomainEvaluator(base.structure, {"p": arrays}, 4)
    return base, evaluator, InterventionSteps(base.structure)


def test_clean_step_equals_the_evaluator(probe_setup):
    base, evaluator, steps = probe_setup
    assert probe_ce(evaluator, steps.clean(), base.params) == pytest.approx(
        evaluator(base.params)["p"]["masked_ce"], abs=1e-6
    )


@pytest.mark.parametrize(
    "module, keys",
    [
        ("L0_mha", ("W_o", "b_o")),
        ("L1_mlp2", ("W_ff2", "b_ff2")),
        ("L0_mlp1", ("W_ff1", "b_ff1")),
    ],
)
def test_ablating_a_zero_output_module_changes_nothing(probe_setup, module, keys):
    base, evaluator, steps = probe_setup
    params = _zeroed(base.params, module, keys)
    clean = probe_ce(evaluator, steps.clean(), params)
    assert probe_ce(evaluator, steps.ablate(module), params) == pytest.approx(
        clean, abs=1e-6
    )
    full = probe_ce(evaluator, steps.clean(), base.params)
    assert probe_ce(evaluator, steps.ablate(module), base.params) != pytest.approx(
        full, abs=1e-6
    )


def test_ablating_embeddings(probe_setup):
    base, evaluator, steps = probe_setup
    params = base.params
    zeroed = _zeroed(params, "position_embeddings", ("position_embeddings",))
    assert probe_ce(
        evaluator, steps.clean(), ablated_params(zeroed, "position_embeddings")
    ) == pytest.approx(probe_ce(evaluator, steps.clean(), zeroed), abs=1e-7)
    table = params.nodes["token_embeddings"].weights["embeddings"]
    removed = ablated_params(params, "token_embeddings")
    means = removed.nodes["token_embeddings"].weights["embeddings"]
    np.testing.assert_allclose(np.asarray(means[3]), np.asarray(table.mean(axis=0)))
    assert np.ptp(np.asarray(means), axis=0).max() == 0
    with pytest.raises(ValueError, match="not an embedding"):
        ablated_params(params, "L0_mha")


@pytest.mark.parametrize(
    "module",
    [
        "token_embeddings",
        "position_embeddings",
        "L0_mha",
        "L0_mlp1",
        "mlm_transform",
        "mlm_head",
    ],
)
def test_perturbation_at_zero_noise_changes_nothing(probe_setup, module):
    base, evaluator, steps = probe_setup
    key = jax.random.PRNGKey(3)
    clean = probe_ce(evaluator, steps.clean(), base.params)
    step = steps.perturb(module)
    assert probe_ce(evaluator, step, base.params, key, 0.0) == pytest.approx(
        clean, abs=1e-6
    )
    noisy = probe_ce(evaluator, step, base.params, key, 0.5)
    assert noisy != pytest.approx(clean, abs=1e-6)
    assert noisy == probe_ce(evaluator, step, base.params, key, 0.5)


def test_perturbation_noise_is_scaled_to_the_output_rms(
    probe_setup, setup, tmp_path, monkeypatch
):
    from fabricpc.experiments.domain_forgetting import influence as module

    base, _, _ = probe_setup
    _, _, cache = _tiny_model(setup, tmp_path)
    steps = InterventionSteps(base.structure)
    seen = {}

    def capture(structure, params, batch, index, clamps=None):
        seen["clamps"] = clamps
        return jax.numpy.zeros((1, 2))

    monkeypatch.setattr(module, "_logits", capture)
    monkeypatch.setattr(module, "_masked_ce_sums", lambda *args: (0.0, 0.0))
    arrays, _ = spaced_probe(
        cache.train,
        base.structure,
        base.special_ids,
        num_windows=4,
        mask_prob=0.3,
        seed=5,
    )
    batch = {
        "x": arrays["input_ids"],
        "y": arrays["labels"],
        "loss_mask": arrays["loss_mask"],
        "attention_mask": arrays["attention_mask"],
    }
    value = (
        compute_forward_pass(
            base.params,
            base.structure,
            dict(batch),
            jax.random.PRNGKey(0),
            training=False,
        )
        .nodes["L0_mlp1"]
        .z_latent
    )
    monkeypatch.setattr(
        module.jax.random,
        "normal",
        lambda key, shape, dtype=jax.numpy.float32: jax.numpy.ones(shape, dtype),
    )
    with jax.disable_jit():
        steps.perturb("L0_mlp1")(
            base.params,
            batch,
            jax.numpy.zeros(1, int),
            jax.numpy.ones(1),
            jax.random.PRNGKey(0),
            0.1,
        )
    rms = jax.numpy.sqrt(jax.numpy.mean(jax.numpy.square(value)))
    np.testing.assert_allclose(
        seen["clamps"]["L0_mlp1"] - value, 0.1 * rms, rtol=1e-3, atol=1e-5
    )


@pytest.fixture(scope="module")
def finished_run(setup, tmp_path_factory):
    checkpoint, data_dir = setup
    output = tmp_path_factory.mktemp("finished") / "run"
    config = tiny_config(checkpoint, data_dir, output)
    run_domain_forgetting(config, verbose=False)
    context = load_run(output)
    atomic_write_json(
        output / "rollback.json",
        compute_rollback(context, random_draws=0, include_groups=False),
    )
    return output


def test_influence_diagnostics_shape_and_absent_old_score(finished_run, monkeypatch):
    from fabricpc.experiments.domain_forgetting import influence as module

    original = module.compute_influence

    def small(context, **kwargs):
        return original(
            context, probe_windows=8, probe_batch=4, map_windows=8, **kwargs
        )

    monkeypatch.setattr(module, "compute_influence", small)
    result = compute_influence_runs([finished_run])
    assert "S0" in format_influence_table(result)
    payload = json.loads((finished_run / "influence.json").read_text())
    assert len(payload["modules"]) == 10
    assert len(payload["block_modules"]) == 6
    onsets = payload["onsets"]
    assert [o["onset"] for o in onsets] == ["S0", "S1", "S2"]
    assert [o["old_domains"] for o in onsets] == [
        [],
        list(DOMAINS[:1]),
        list(DOMAINS[:2]),
    ]
    first, second, third = onsets
    block = payload["block_modules"][0]
    assert "old" not in first["modules"][block]["ablation"]
    assert "old" in second["modules"][block]["ablation"]
    assert first["recovery"] is None and "recovery" not in first["spearman"]
    assert "recovery" in second["spearman"]
    assert "ablation_old" not in first["spearman"]["surprise_map"]
    assert "ablation_old" in third["spearman"]["surprise_map"]
    for name in payload["head_modules"]:
        assert first["modules"][name]["ablation"] is None
        assert first["modules"][name]["perturbation"]["0.1"]["current"] != 0.0
    assert third["modules"][block]["ablation"]["old"] == pytest.approx(
        np.mean(
            [third["modules"][block]["ablation"]["per_domain"][d] for d in DOMAINS[:2]]
        )
    )
    assert NOISE_DRAWS == 8
    again = compute_influence_runs([finished_run])
    assert again["spearman"] == result["spearman"]


def test_influence_fails_loudly_on_a_missing_snapshot(finished_run, tmp_path):
    copy = tmp_path / "copy"
    shutil.copytree(finished_run, copy)
    (copy / "stage_2.npz").unlink()
    with pytest.raises(FileNotFoundError, match="stage_2.npz"):
        compute_influence_runs([copy])


def test_influence_gate_runs_through_the_commutator(setup, tmp_path):
    checkpoint, data_dir = setup
    pair = ("news", "wiki")
    dirs = []
    for order in (pair, tuple(reversed(pair))):
        for seed in (0, 1):
            for name, extra in (
                (
                    "gate",
                    {
                        "gate": "online_influence",
                        "gate_map_windows": MAP_WINDOWS,
                        "gate_interval": 1,
                    },
                ),
                ("match", {"gate": "uniform", "gate_scale": 0.5}),
            ):
                out = tmp_path / f"{name}_{'_'.join(order)}_{seed}"
                config = tiny_config(
                    checkpoint, data_dir, out, domains=order, seed=seed, **extra
                )
                run_domain_forgetting(config, verbose=False)
                dirs.append(out)
    result = compute_commutator(dirs, divergence=True, batch_size=4)
    conditions = {row["condition"] for row in result["results"]}
    assert conditions == {"online_influence@lr1e-02", "uniform_s0.5@lr1e-02"}
    test = commutator_test(
        result, "online_influence@lr1e-02", "uniform_s0.5@lr1e-02", pair
    )
    assert test["seeds"] == [0, 1]
    assert test["verdict"] in ("supported", "parameter-only", "null")
    assert isinstance(test["function_space_gap"], float)
    for row in result["results"]:
        assert row["N"]["n"] > 0 and all("symmetric_kl" in e for e in row["per_seed"])


def test_replay_of_an_influence_run_reproduces_it(influence_run, setup, tmp_path):
    config, result = influence_run
    replay = _online(
        setup,
        tmp_path / "replay",
        gate_variant="replay",
        gate_source=(config.output_dir,),
    )
    replayed = run_domain_forgetting(replay, verbose=False)
    assert (
        replayed["summary"]["stage_params_sha256"]
        == result["summary"]["stage_params_sha256"]
    )
    onset = _read_jsonl(tmp_path / "replay" / "gate_schedule_stage_2.jsonl")[0]
    assert "influence" not in onset


def test_online_influence_and_diagnostics_on_epc_with_the_learner_map(setup, tmp_path):
    from fabricpc.experiments.domain_forgetting import influence as module

    epc = {
        "mode": "pc",
        "pc_inference": "epc",
        "pc_eta_infer": 0.01,
        "pc_infer_steps": 3,
        "pc_forward_updates": "none",
    }
    config = _online(setup, tmp_path / "gate", gate_map_source="learner", **epc)
    run_domain_forgetting(config, verbose=False)
    later = _read_jsonl(tmp_path / "gate" / "gate_schedule_stage_2.jsonl")[-1]
    assert later["map_source"] == "learner"
    assert sorted(later["influence"]) == sorted(BLOCKS)
    checkpoint, data_dir = setup
    output = tmp_path / "plain"
    run_domain_forgetting(
        tiny_config(checkpoint, data_dir, output, **epc), verbose=False
    )
    context = load_run(output)
    atomic_write_json(
        output / "rollback.json",
        compute_rollback(context, random_draws=0, include_groups=False),
    )
    payload = module.compute_influence(
        context, probe_windows=8, probe_batch=4, map_windows=8
    )
    assert payload["map_source"] == "learner" and payload["mode"] == "pc"
    assert set(payload["onsets"][1]["spearman"]) == {"surprise_map", "recovery"}


def test_cli_flags(tmp_path):
    from examples import domain_forgetting_analysis as analysis
    from examples import mlm_domain_forgetting as cli

    common = ["--checkpoint", "c", "--output_dir", str(tmp_path)]
    args = cli.build_parser().parse_args(
        common
        + [
            "--gate",
            "online_influence",
            "--gate_variant",
            "past",
            "--gate_influence_alpha",
            "0.25",
        ]
    )
    config = cli.config_from_args(args)
    config.validate()
    assert config.gate_variant == "past" and config.gate_influence_alpha == 0.25
    parsed = analysis.build_parser().parse_args(
        ["frontier", "--influence_verdict", "a", "b", "c"]
    )
    assert parsed.influence_verdict == [["a", "b", "c"]]
    parsed = analysis.build_parser().parse_args(
        ["commutator", "--commutator_test", "a", "b"]
    )
    assert parsed.commutator_test == [["a", "b"]]
