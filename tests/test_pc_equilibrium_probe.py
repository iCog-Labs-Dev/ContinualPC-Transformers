"""Tests for the frozen-weight equilibrium probe."""

import json
import math
import os
from argparse import Namespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

import diagnostics.pc_equilibrium_probe as probe
from diagnostics.pc_equilibrium_probe import (
    alignment_with_backprop,
    build_manifest,
    case_seed,
    float32_rounding_test,
    json_ready,
    main,
    make_bert_mini_fixture,
    make_tiny_fixture,
    parse_head_scales,
    prepare_output_dir,
    resolve_head_scale,
    run_case,
    run_finished,
    summarize,
    write_json,
)
from diagnostics.pc_equilibrium_solvers import (
    AcceptanceThresholds,
    EscapeResult,
    SpectrumResult,
    TimeAllocator,
    make_energy_functions,
    module_gradients,
    trust_region_newton_cg,
)
from tests.domain_forgetting_fixtures import build_tiny_run
from tests.test_domain_forgetting_runner import make_setup

THRESHOLDS = AcceptanceThresholds()


def small_fixture(batch_size=2, num_batches=1, seed=0):
    return make_tiny_fixture(
        seed=seed,
        batch_size=batch_size,
        num_batches=num_batches,
        depth=1,
        embed_dim=8,
        mlp_dim=16,
        seq_len=6,
        vocab=20,
    )


def small_case(head_scale="1", batch_size=2, lanczos_iters=8, max_evals=400, **kwargs):
    fixture = small_fixture(batch_size)
    return fixture, run_case(
        fixture,
        fixture.cases[0],
        head_scale,
        make_energy_functions(fixture.structure),
        max_evals=max_evals,
        lanczos_iters=lanczos_iters,
        thresholds=THRESHOLDS,
        start_lanczos_iters=8,
        **kwargs,
    )


def test_json_ready_nulls_nonfinite_values_and_reports_paths():
    clean, nulled = json_ready(
        {
            "a": float("nan"),
            "b": [1.0, float("inf"), np.float64(2.0)],
            "c": np.array([1.0, np.nan]),
            "d": {"e": None, "f": np.int32(3)},
            "g": jnp.array(float("-inf")),
        }
    )
    assert clean == {
        "a": None,
        "b": [1.0, None, 2.0],
        "c": [1.0, None],
        "d": {"e": None, "f": 3},
        "g": None,
    }
    assert sorted(nulled) == ["a", "b[1]", "c[1]", "g"]
    json.dumps(clean, allow_nan=False)


def test_write_json_records_null_fields_and_is_strict(tmp_path):
    path = tmp_path / "record.json"
    write_json(path, {"status": "not_stationary", "value": math.inf})
    loaded = json.loads(path.read_text())
    assert loaded["value"] is None
    assert loaded["status"] == "not_stationary"
    assert loaded["null_fields"] == ["value"]
    assert not list(tmp_path.glob("*.tmp"))


def test_output_dir_is_never_overwritten(tmp_path):
    fresh = tmp_path / "run"
    prepare_output_dir(fresh)
    assert fresh.is_dir()
    with pytest.raises(FileExistsError):
        prepare_output_dir(fresh)


def test_head_scale_parsing_and_resolution():
    assert parse_head_scales("1, mean,midpoint,0.5") == [
        "1",
        "mean",
        "midpoint",
        "0.5",
    ]
    assert resolve_head_scale("0.5", 100.0) == 0.5
    assert resolve_head_scale("mean", 100.0) == pytest.approx(0.01)
    assert resolve_head_scale("midpoint", 100.0) == pytest.approx(0.1)
    for bad in ("0", "-1", "nope", "inf", "1,1.0", "mean,mean"):
        with pytest.raises(ValueError):
            parse_head_scales(bad)


def test_head_scale_with_no_masked_tokens_does_not_abort():
    assert resolve_head_scale("mean", 0.0) == 1.0
    assert resolve_head_scale("midpoint", 0.0) == 1.0
    assert resolve_head_scale("0.5", 0.0) == 0.5


def test_manifest_measures_dtypes_and_rejects_single_precision_parameters():
    with jax.enable_x64(True):
        fixture = small_fixture()
        manifest = build_manifest({"fixture": "tiny"}, fixture, THRESHOLDS, 8)
        with pytest.raises(ValueError):
            probe._assert_double(
                jax.tree_util.tree_map(lambda x: x.astype(jnp.float32), fixture.params)
            )
    for key in (
        "git_revision",
        "git_dirty",
        "source_sha256",
        "dependencies",
        "hardware",
        "params_sha256",
        "case_settings",
    ):
        assert key in manifest
    assert manifest["dtypes"]["reference"] == "float64"
    assert manifest["dtypes"]["errors"] == "float64"
    assert manifest["thresholds"]["rel_stationarity"] == 1e-6
    assert manifest["thresholds"]["curvature_floor"] == 1e-2
    assert manifest["source_sha256"]["diagnostics/pc_equilibrium_solvers.py"]
    assert manifest["dependencies"]["jax"] == jax.__version__


def test_samples_are_decoupled_and_share_the_batch_masks():
    with jax.enable_x64(True):
        fixture = small_fixture(batch_size=3)
        case = fixture.cases[0]
        fns = make_energy_functions(fixture.structure)
        errors = jax.tree_util.tree_map(
            lambda e: 0.01 * jax.random.normal(jax.random.PRNGKey(0), e.shape, e.dtype),
            case.zero_errors,
        )
        value, grad = fns.value_and_grad(
            errors, fixture.params, case.state, case.clamps, 0.5
        )
        total = 0.0
        for sample in case.samples:
            assert sample.state.batch_size == 1
            part = jax.tree_util.tree_map(
                lambda e: e[sample.index : sample.index + 1], errors
            )
            part_value, part_grad = fns.value_and_grad(
                part, fixture.params, sample.state, sample.clamps, 0.5
            )
            total += float(part_value)
            for name in errors:
                np.testing.assert_allclose(
                    part_grad[name][0], grad[name][sample.index], atol=1e-12
                )
            mask_name = fixture.structure.task_map["loss_mask"]
            np.testing.assert_array_equal(
                sample.clamps[mask_name][0], case.clamps[mask_name][sample.index]
            )
        assert total == pytest.approx(float(value), rel=1e-12)
        assert sum(s.num_masked for s in case.samples) == case.num_masked


def test_float32_rounding_at_a_solution_with_scaled_head():
    with jax.enable_x64(True):
        fixture = small_fixture()
        sample = fixture.cases[0].samples[0]
        s = 0.3
        fns = make_energy_functions(fixture.structure)
        solve = trust_region_newton_cg(
            fns.objective(fixture.params, sample.state, sample.clamps, s),
            sample.zero_errors,
        )
        assert solve.status == "converged"
        assert solve.grad_norm > 0.0
        derived = fns.derived(solve.x, fixture.params, sample.state, sample.clamps, s)
        reference = module_gradients(
            fns.weight_gradients(fixture.params, derived, s), fixture.structure
        )
        result = float32_rounding_test(
            fixture.structure,
            fixture.params,
            derived,
            sample.clamps,
            s,
            reference,
            THRESHOLDS,
        )
        assert set(result) == {"rounded_latents", "rounded_errors"}
        for variant in result.values():
            assert variant["dtype"] == "float32"
            assert variant["comparison"]["passed"]
            head = variant["comparison"]["modules"][fixture.structure.task_map["y"]]
            assert head["rel_norm_error"] < 1e-3


def test_run_case_records_every_required_field_and_passes_the_gates():
    with jax.enable_x64(True):
        fixture, record = small_case(lanczos_iters=30)
    assert record["status"] == "accepted"
    assert record["accepted_fraction"] == 1.0
    assert record["sample_statuses"] == ["accepted", "accepted"]
    for key in (
        "head_scale",
        "num_masked",
        "energy_start",
        "epsilon_bound",
        "spectrum_start",
        "solvers",
        "between_solvers",
        "float32",
        "duration_seconds",
        "samples",
    ):
        assert key in record
    assert record["spectrum_start"]["start_vector"] == "gradient"
    assert "max_gradient_negative_weight" in record["spectrum_start"]
    assert record["epsilon_bound"] == pytest.approx(
        math.sqrt(2.0 * record["energy_start"]), rel=1e-12
    )
    samples = record["samples"]
    for name in ("trust_region", "lbfgs"):
        solver = record["solvers"][name]
        assert solver["epsilon_norm"] <= record["epsilon_bound"]
        assert solver["seconds"] >= 0.0
        worst = max(s["solvers"][name]["rel_stationarity"] for s in samples)
        assert solver["max_sample_rel_stationarity"] == worst
        assert worst > 0.0
        assert set(solver["work"]) >= {
            "energy_evals",
            "gradient_evals",
            "hvps",
            "rejected_steps",
            "skipped_pairs",
        }
        assert solver["work"]["passes"] == sum(
            s["solvers"][name]["work"]["passes"] for s in samples
        )
    for sample in samples:
        assert sample["epsilon_bound"] == pytest.approx(
            math.sqrt(2.0 * sample["energy_start"]), rel=1e-12
        )
        assert sample["curvature_evidence"] == "escape_test_passed"
        assert sample["escape"]["outcome"] == "passed"
        assert sample["spectrum_start"]["start_vector"] == "gradient"
        assert "gradient_negative_weight" in sample["spectrum_start"]
        for solver in sample["solvers"].values():
            assert solver["spectrum_end"]["start_vector"] == "random"
            assert "random_negative_weight" in solver["spectrum_end"]
            assert "trace" not in solver
        assert sample["settings"]["trust_region"]["max_radius"] == pytest.approx(
            2.0 * sample["epsilon_bound"]
        )


def test_spectrum_end_matches_the_dense_hessian_at_each_solution():
    with jax.enable_x64(True):
        fixture = small_fixture()
        fns = make_energy_functions(fixture.structure)
        case = fixture.cases[0]
        sample = case.samples[0]
        dimension = ravel_pytree(sample.zero_errors)[0].size
        record = run_case(
            fixture,
            case,
            "1",
            fns,
            max_evals=400,
            lanczos_iters=dimension,
            thresholds=THRESHOLDS,
            start_lanczos_iters=8,
        )
        s = 1.0
        objective = fns.objective(fixture.params, sample.state, sample.clamps, s)
        solve = trust_region_newton_cg(objective, sample.zero_errors)
        flat, unravel = ravel_pytree(solve.x)

        def gradient(v):
            return ravel_pytree(objective.value_and_grad(unravel(v))[1])[0]

        eigenvalues = np.linalg.eigvalsh(np.asarray(jax.jacfwd(gradient)(flat)))
        start_flat, start_unravel = ravel_pytree(sample.zero_errors)

        def gradient_start(v):
            return ravel_pytree(objective.value_and_grad(start_unravel(v))[1])[0]

        at_start = np.linalg.eigvalsh(
            np.asarray(jax.jacfwd(gradient_start)(start_flat))
        )
    measured = record["samples"][0]["solvers"]["trust_region"]["spectrum_end"]
    assert measured["lambda_min"] == pytest.approx(eigenvalues[0], abs=1e-5)
    assert measured["lambda_max"] == pytest.approx(eigenvalues[-1], abs=1e-5)
    assert abs(at_start[-1] - eigenvalues[-1]) > 1e-3


def test_small_head_scale_recovers_backprop_in_direction_and_norm():
    with jax.enable_x64(True):
        fixture, record = small_case("1e-6")
    assert record["status"] == "accepted"
    alignment = record["solvers"]["trust_region"]["vs_backprop"]
    assert alignment["min_cosine"] >= 0.999
    expected = record["head_scale"] * record["num_masked"]
    for row in alignment["modules"].values():
        assert row["gated"]
        assert row["norm_ratio"] == pytest.approx(expected, rel=1e-3)
        assert row["rel_norm_ratio"] == pytest.approx(1.0, rel=1e-3)


def test_large_head_scale_moves_away_from_backprop():
    with jax.enable_x64(True):
        fixture, record = small_case("1", lanczos_iters=30)
    assert record["status"] == "accepted"
    assert record["solvers"]["trust_region"]["vs_backprop"]["min_cosine"] < 0.95
    assert record["solvers"]["trust_region"]["energy_drop"] > 0.05


def test_alignment_floor_uses_both_vectors():
    pc = {"a": np.array([1.0, 0.0]), "b": np.array([1e-12, 0.0])}
    bp = {"a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0])}
    result = alignment_with_backprop(pc, bp, "a", THRESHOLDS)
    assert result["modules"]["b"]["gated"]
    assert result["min_cosine"] == pytest.approx(0.0)
    zero = {"a": np.array([1.0, 0.0]), "b": np.zeros(2)}
    undefined = alignment_with_backprop(zero, bp, "a", THRESHOLDS)
    assert undefined["undefined_cosine_modules"] == ["b"]
    small = {"a": np.array([1.0, 0.0]), "b": np.array([1e-9, 0.0])}
    excluded = alignment_with_backprop(small, small, "a", THRESHOLDS)
    assert excluded["excluded"] == ["b"]


def wrap_lbfgs_changing_x(monkeypatch, factor=0.5, only_sample=None):
    real = probe.lbfgs
    calls = {"count": 0}

    def fake(objective, start, config, budget):
        result = real(objective, start, config, budget)
        index = calls["count"]
        calls["count"] += 1
        if only_sample is None or index == only_sample:
            result.x = jax.tree_util.tree_map(lambda e: factor * e, result.x)
        return result

    monkeypatch.setattr(probe, "lbfgs", fake)


def test_run_case_reports_solvers_that_disagree(monkeypatch):
    wrap_lbfgs_changing_x(monkeypatch)
    with jax.enable_x64(True):
        _, record = small_case("mean", lanczos_iters=8)
    assert record["status"] == "solver_disagreement"
    assert record["between_solvers"]["passed"] is False
    for sample in record["samples"]:
        assert sample["status"] == "solver_disagreement"
        assert sample["between_solvers"]["passed"] is False
        assert sample["reason"] == "weight gradients differ"


def test_one_disagreeing_sample_fails_the_batch(monkeypatch):
    wrap_lbfgs_changing_x(monkeypatch, only_sample=1)
    with jax.enable_x64(True):
        _, record = small_case("mean", batch_size=4, lanczos_iters=8)
    assert record["status"] == "solver_disagreement"
    assert record["sample_statuses"].count("solver_disagreement") == 1
    assert record["accepted_fraction"] == 0.75


def fake_spectrum(lambda_min):
    def spectrum(objective, x, start, iterations, deadline=None):
        size = ravel_pytree(x)[0].size
        return SpectrumResult(
            np.array([lambda_min, 2.0]),
            np.array([0.5, 0.5]),
            lambda_min,
            2.0,
            0.0,
            0.0,
            2,
            iterations,
            True,
            False,
            False,
            np.ones(size),
        )

    return spectrum


def test_run_case_reports_a_saddle_from_lanczos_and_skips_the_escape(monkeypatch):
    monkeypatch.setattr(probe, "tree_spectrum", fake_spectrum(-5.0))

    def no_escape(*args, **kwargs):
        raise AssertionError("escape test ran after a refuting gate")

    monkeypatch.setattr(probe, "escape_test", no_escape)
    with jax.enable_x64(True):
        _, record = small_case()
    assert record["status"] == "saddle"
    assert record["sample_statuses"] == ["saddle", "saddle"]


def test_run_case_reports_a_saddle_from_the_escape_test(monkeypatch):
    def escape(*args, **kwargs):
        return EscapeResult(
            "saddle", 0.1, 1.0, 0.0, 1.0, True, "converged", None, True, 0.0
        )

    monkeypatch.setattr(probe, "escape_test", escape)
    with jax.enable_x64(True):
        _, record = small_case()
    assert record["status"] == "saddle"
    assert record["samples"][0]["escape"]["outcome"] == "saddle"
    assert record["curvature_evidence"][0] == "no_negative_mode_found"


def test_nan_spectrum_is_unresolved_end_to_end(monkeypatch):
    monkeypatch.setattr(probe, "tree_spectrum", fake_spectrum(float("nan")))
    with jax.enable_x64(True):
        _, record = small_case()
    assert record["status"] == "curvature_unresolved"


def test_escape_at_the_reference_compares_real_weight_gradients(monkeypatch):
    real = probe.escape_test
    seen = {}

    def spying(objective, x, value, vector, tol, check, *rest):
        seen["same"] = check(x)
        seen["moved"] = check(jax.tree_util.tree_map(lambda e: 0.5 * e, x))
        return real(objective, x, value, vector, tol, check, *rest)

    monkeypatch.setattr(probe, "escape_test", spying)
    with jax.enable_x64(True):
        _, record = small_case("1", lanczos_iters=20)
    assert seen == {"same": True, "moved": False}
    assert record["status"] == "accepted"


def test_sample_without_masked_tokens_has_the_exact_solution():
    with jax.enable_x64(True):
        fixture = small_fixture(batch_size=2)
        case = fixture.cases[0]
        batch = {**case.batch}
        loss_mask = np.array(batch["loss_mask"])
        loss_mask[1] = 0.0
        batch["loss_mask"] = jnp.asarray(loss_mask)
        rebuilt = probe._prepare_cases(
            fixture.structure, fixture.params, [batch], [case.key], 0
        )[0]
        assert rebuilt.samples[1].num_masked == 0.0
        fns = make_energy_functions(fixture.structure)
        record = run_case(
            fixture,
            rebuilt,
            "mean",
            fns,
            max_evals=400,
            lanczos_iters=20,
            thresholds=THRESHOLDS,
            start_lanczos_iters=8,
        )
        assert record["sample_statuses"] == ["accepted", "zero_signal"]
        assert record["status"] == "accepted"
        assert record["head_scale"] == pytest.approx(1.0 / rebuilt.num_masked)

        loss_mask[:] = 0.0
        batch["loss_mask"] = jnp.asarray(loss_mask)
        empty = probe._prepare_cases(
            fixture.structure, fixture.params, [batch], [case.key], 0
        )[0]
        none = run_case(
            fixture,
            empty,
            "mean",
            fns,
            max_evals=50,
            lanczos_iters=4,
            thresholds=THRESHOLDS,
            start_lanczos_iters=4,
        )
    assert none["status"] == "zero_signal"
    assert none["head_scale"] == 1.0
    summary = summarize([none], THRESHOLDS)["mean"]
    assert summary["cases"] == 0
    assert summary["h1_reference_reproducible"] is None


def test_time_allocator_censoring_keeps_finished_samples(tmp_path):
    with jax.enable_x64(True):
        fixture = small_fixture(batch_size=2)

        class EndsAfterFirstSample(TimeAllocator):
            def __init__(self):
                super().__init__(1e12, 12)
                self.calls = 0

            def expired(self):
                self.calls += 1
                return self.calls > 1

        record = run_case(
            fixture,
            fixture.cases[0],
            "1",
            make_energy_functions(fixture.structure),
            max_evals=400,
            lanczos_iters=20,
            thresholds=THRESHOLDS,
            allocator=EndsAfterFirstSample(),
            start_lanczos_iters=8,
            sample_dir=tmp_path / "samples",
        )
    assert record["status"] == "censored_budget"
    assert record["sample_statuses"] == ["accepted", "censored_budget"]
    assert record["between_solvers"] is None
    first = json.loads((tmp_path / "samples" / "sample0.json").read_text())
    second = json.loads((tmp_path / "samples" / "sample1.json").read_text())
    assert first["status"] == "accepted"
    assert "trace" in first["solvers"]["trust_region"]
    assert second["not_started"] is True


def case_record(status, cosine=None, float32=(True, True), spec="1"):
    record = {"head_scale_spec": spec, "status": status}
    if status == "accepted":
        record["solvers"] = {"trust_region": {"vs_backprop": {"min_cosine": cosine}}}
        record["float32"] = {
            "rounded_latents": {"passed": float32[0]},
            "rounded_errors": {"passed": float32[1]},
        }
    return record


@pytest.mark.parametrize(
    ("statuses", "expected"),
    [
        (["accepted"] * 4, True),
        (["accepted"] * 3 + ["saddle"], True),
        (["accepted"] * 3 + ["censored_budget"], True),
        (["accepted", "accepted", "saddle", "saddle"], False),
        (["saddle"] * 4, False),
        (["solver_disagreement"] * 2 + ["bound_violation"] * 2, False),
        (["censored_budget"] * 4, None),
        (["accepted"] * 2 + ["censored_budget", "saddle"], None),
        (["accepted"] + ["not_stationary"] * 3, None),
        (["curvature_unresolved"] * 4, None),
        (["nonfinite"] * 3 + ["saddle"], None),
        (["censored_budget"] + ["saddle"] * 3, False),
    ],
)
def test_h1_truth_table(statuses, expected):
    records = [case_record(s, cosine=0.5) for s in statuses]
    summary = summarize(records, THRESHOLDS)["1"]
    assert summary["h1_reference_reproducible"] is expected
    assert summary["censored_fraction"] == pytest.approx(
        statuses.count("censored_budget") / len(statuses)
    )


@pytest.mark.parametrize(
    ("records", "expected"),
    [
        ([("accepted", 0.5)], True),
        ([("accepted", 0.5), ("censored_budget", None)], True),
        ([("accepted", 0.97)], None),
        ([("accepted", 0.995)], False),
        ([("accepted", 0.995), ("saddle", None)], False),
        ([("accepted", 0.995), ("censored_budget", None)], None),
        ([("censored_budget", None)], None),
        ([("saddle", None)], None),
    ],
)
def test_h2_truth_table(records, expected):
    built = [case_record(status, cosine=cosine) for status, cosine in records]
    assert summarize(built, THRESHOLDS)["1"]["h2_distinct_from_backprop"] is expected


@pytest.mark.parametrize(
    ("records", "expected"),
    [
        ([case_record("accepted", 0.5)], (True, True)),
        ([case_record("accepted", 0.5, (False, True))], (False, True)),
        ([case_record("accepted", 0.5, (True, False))], (True, False)),
        (
            [case_record("accepted", 0.5), case_record("censored_budget")],
            (None, None),
        ),
        (
            [
                case_record("accepted", 0.5, (False, True)),
                case_record("censored_budget"),
            ],
            (False, None),
        ),
        ([case_record("censored_budget")], (None, None)),
    ],
)
def test_h3_truth_table(records, expected):
    summary = summarize(records, THRESHOLDS)["1"]["h3_float32"]
    assert summary["rounded_latents"]["reading"] is expected[0]
    assert summary["rounded_errors"]["reading"] is expected[1]


def test_reorthogonalisation_estimate_counts_all_passes():
    assert probe.reorth_seconds(10, 5, 1000.0) == pytest.approx(8.0 * 10 * 25 / 1000.0)
    assert probe.measure_flop_rate(100) > 0.0


def test_run_finished_needs_every_case_and_no_censoring():
    accepted = {"status": "accepted"}
    censored = {"status": "censored_budget"}
    assert run_finished([accepted, accepted], 2)
    assert not run_finished([accepted], 2)
    assert not run_finished([accepted, censored], 2)


def test_case_seed_is_stable():
    assert case_seed(0, 1) == case_seed(0, 1)
    assert case_seed(0, 1) != case_seed(1, 1)


def smoke_arguments(path, *extra):
    return [
        "--fixture",
        "tiny",
        "--head-scales",
        "1.0",
        "--seeds",
        "0",
        "--batch-size",
        "2",
        "--num-batches",
        "1",
        "--max-evals",
        "40",
        "--lanczos-iters",
        "8",
        "--start-lanczos-iters",
        "4",
        "--time-cap-minutes",
        "5",
        "--output",
        str(path),
        *extra,
    ]


@pytest.fixture
def restore_x64():
    previous = jax.config.jax_enable_x64
    yield
    jax.config.update("jax_enable_x64", previous)


def strip_volatile(value):
    if isinstance(value, dict):
        return {
            k: strip_volatile(v)
            for k, v in value.items()
            if k not in ("duration_seconds", "seconds")
        }
    if isinstance(value, list):
        return [strip_volatile(v) for v in value]
    return value


def test_deterministic_tiny_smoke_and_refusal(tmp_path, restore_x64):
    assert main(smoke_arguments(tmp_path / "one")) == 0
    assert main(smoke_arguments(tmp_path / "two")) == 0
    one = json.loads((tmp_path / "one" / "summary.json").read_text())
    two = json.loads((tmp_path / "two" / "summary.json").read_text())
    assert strip_volatile(one["records"]) == strip_volatile(two["records"])
    manifest = json.loads((tmp_path / "one" / "manifest.json").read_text())
    settings = manifest["case_settings"]["seed0_batch0_s1.0"]
    assert settings["max_evals"] == 40
    assert len(settings["per_sample_max_radius"]) == 2
    assert (tmp_path / "one" / "records" / "seed0_batch0_s1.0.json").exists()
    assert (
        tmp_path / "one" / "records" / "seed0_batch0_s1.0" / "sample1.json"
    ).exists()
    with pytest.raises(FileExistsError):
        main(smoke_arguments(tmp_path / "one"))


def test_time_cap_zero_is_unresolved_and_incomplete(tmp_path, restore_x64):
    run = tmp_path / "run"
    assert (
        main(
            [
                "--fixture",
                "tiny",
                "--head-scales",
                "1.0,mean",
                "--seeds",
                "0",
                "--batch-size",
                "2",
                "--num-batches",
                "1",
                "--time-cap-minutes",
                "0",
                "--output",
                str(run),
            ]
        )
        == 0
    )
    summary = json.loads((run / "summary.json").read_text())
    assert len(summary["records"]) == 2
    assert all(r["status"] == "censored_budget" for r in summary["records"])
    assert summary["complete"] is False
    for reading in summary["by_head_scale"].values():
        assert reading["h1_reference_reproducible"] is None
        assert reading["h2_distinct_from_backprop"] is None
        assert reading["censored_fraction"] == 1.0


def test_timing_smoke_writes_nothing_and_reports_each_part(
    tmp_path, capsys, restore_x64
):
    target = tmp_path / "unused"
    assert main(smoke_arguments(target, "--timing-smoke", "--timing-passes", "5")) == 0
    assert not target.exists()
    output = capsys.readouterr().out
    for text in (
        "backend",
        "first call incl. compile",
        "trust_region",
        "lbfgs",
        "Lanczos",
        "escape test",
        "memory per sample",
        "peak RSS",
        "batch size 8",
        "batch size 64",
        "estimated total",
        "fits the 5-minute cap",
    ):
        assert text in output


def test_bert_mini_path_on_a_tiny_checkpoint(tmp_path, capsys, restore_x64):
    checkpoint, data_dir = make_setup(tmp_path / "setup")
    common = [
        "--fixture",
        "bert_mini",
        "--checkpoint",
        str(checkpoint),
        "--data-dir",
        str(data_dir),
        "--domain",
        "news",
        "--head-scales",
        "mean",
        "--batch-size",
        "2",
        "--num-batches",
        "1",
        "--mask-prob",
        "0.3",
        "--max-evals",
        "400",
        "--lanczos-iters",
        "16",
        "--start-lanczos-iters",
        "4",
    ]
    assert main([*common, "--timing-smoke", "--timing-passes", "5"]) == 0
    assert "estimated total" in capsys.readouterr().out
    run = tmp_path / "run"
    assert main([*common, "--output", str(run)]) == 0
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["fixture"]["source"]["train_array_sha256"]
    assert manifest["params_sha256"]
    assert manifest["dtypes"]["reference"] == "float64"
    summary = json.loads((run / "summary.json").read_text())
    assert summary["complete"]
    assert summary["records"][0]["status"] == "accepted"


def test_run_dir_and_stage_load_that_stage(tmp_path, restore_x64):
    config = build_tiny_run(tmp_path / "trained")
    arguments = Namespace(
        checkpoint=config.checkpoint,
        data_dir=config.data_dir,
        domain=config.domains[0],
        batch_size=2,
        num_batches=1,
        mask_prob=0.3,
        run_dir=None,
        stage=None,
    )
    jax.config.update("jax_enable_x64", True)
    base = make_bert_mini_fixture(arguments, [0])
    arguments.run_dir = config.output_dir
    arguments.stage = 2
    staged = make_bert_mini_fixture(arguments, [0])
    assert staged.source["stage"] == 2
    assert staged.source["params_sha256"] != base.source["params_sha256"]
    differences = [
        float(jnp.max(jnp.abs(a - b)))
        for a, b in zip(
            jax.tree_util.tree_leaves(base.params),
            jax.tree_util.tree_leaves(staged.params),
        )
    ]
    assert max(differences) > 0.0
    assert probe.measured_float_dtypes(staged.params) == ["float64"]
    with pytest.raises(ValueError):
        main(
            [
                "--fixture",
                "bert_mini",
                "--checkpoint",
                config.checkpoint,
                "--run-dir",
                config.output_dir,
                "--timing-smoke",
            ]
        )
