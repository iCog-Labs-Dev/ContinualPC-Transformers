"""Tests for the traced ePC rate and the energy-drop rate controller."""

import json
import math
import os
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fabricpc.core import EPCInference, InferenceSGD
from fabricpc.core.epsilon_spectrum import EpsilonSpectrum
from fabricpc.experiments.domain_forgetting.data import load_domain_cache
from fabricpc.experiments.domain_forgetting.probe import (
    combine_regimes,
    format_epc_setting,
)
from fabricpc.experiments.domain_forgetting.runner import (
    accumulate_gradients,
    control_eta,
    fixed_mlm_batches,
    gradient_alignment,
    load_base_model,
    load_stage_snapshot,
    make_alignment_fn,
    make_gradient_fn,
    regime_summary,
    _write_jsonl,
    run_domain_forgetting,
    train_stage,
)
from fabricpc.graph_initialization import initialize_params
from fabricpc.training.train_mlm import get_graph_param_gradient_pcn_mlm
from tests.test_domain_forgetting_runner import DOMAINS, make_setup, tiny_config
from tests.test_inference_epc import bert_batch, bert_inputs, tiny_bert

EPC = {
    "mode": "pc",
    "pc_inference": "epc",
    "pc_eta_infer": 0.01,
    "pc_infer_steps": 3,
    "pc_forward_updates": "none",
}
CONTROL = {"pc_eta_control": "energy", "pc_energy_target": 0.05}
LIMITS = {"target": 0.2, "gain": 0.5, "eta_min": 1e-3, "eta_max": 1.0, "backoff": 0.5}


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def _leaves(tree):
    return [np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(tree)]


def _assert_trees_equal(left, right):
    for a, b in zip(_leaves(left), _leaves(right), strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    return make_setup(tmp_path_factory.mktemp("eta_control"))


@pytest.fixture(scope="module")
def controlled_run(setup, tmp_path_factory):
    checkpoint, data_dir = setup
    output = tmp_path_factory.mktemp("controlled") / "run"
    config = tiny_config(
        checkpoint,
        data_dir,
        output,
        pc_regime_interval=1,
        pc_regime_iters=8,
        pc_regime_final=True,
        pc_eta_max=0.1,
        pc_alignment_batches=1,
        **{**EPC, **CONTROL},
    )
    return config, run_domain_forgetting(config, verbose=False)


def _assert_trees_close(left, right, rtol):
    """Every float leaf within ``rtol`` of ``right``'s global norm; ints equal."""
    pairs = list(zip(_leaves(left), _leaves(right), strict=True))
    scale = math.sqrt(
        sum(float(np.sum(np.square(b, dtype=np.float64))) for _, b in pairs)
    )
    for a, b in pairs:
        if np.issubdtype(b.dtype, np.integer):
            np.testing.assert_array_equal(a, b)
        else:
            np.testing.assert_allclose(a, b, rtol=0.0, atol=rtol * scale)


class TestTracedRate:
    """A traced rate replaces the configured one exactly; against a structure
    built with the rate as a compile-time constant it agrees to float32
    rounding, since XLA folds the constant into neighbouring products."""

    @pytest.mark.parametrize("eta, steps", [(3e-3, 3), (5e-2, 5)])
    def test_override_equals_construction(self, eta, steps):
        built = tiny_bert(EPCInference(eta_infer=eta, infer_steps=steps))
        first = tiny_bert(EPCInference(eta_infer=1e-2, infer_steps=steps))
        second = tiny_bert(EPCInference(eta_infer=1e-3, infer_steps=steps))
        params = initialize_params(built, jax.random.PRNGKey(3))
        key = jax.random.PRNGKey(11)
        rate = jnp.asarray(eta)
        want = make_gradient_fn("pc", built, "none", "epc")(params, bert_batch(), key)
        got = make_gradient_fn("pc", first, "none", "epc")(
            params, bert_batch(), key, eta_infer=rate
        )
        again = make_gradient_fn("pc", second, "none", "epc")(
            params, bert_batch(), key, eta_infer=rate
        )
        _assert_trees_equal(got, again)
        _assert_trees_close(got, want, 1e-6)

    def test_one_step_override_is_exact(self):
        eta = 2e-2
        built = tiny_bert(EPCInference(eta_infer=eta, infer_steps=1))
        other = tiny_bert(EPCInference(eta_infer=1e-3, infer_steps=1))
        params = initialize_params(built, jax.random.PRNGKey(3))
        key = jax.random.PRNGKey(11)
        want = make_gradient_fn("pc", built, "none", "epc")(params, bert_batch(), key)
        got = make_gradient_fn("pc", other, "none", "epc")(
            params, bert_batch(), key, eta_infer=jnp.asarray(eta)
        )
        _assert_trees_equal(got, want)

    def test_default_is_the_configured_rate(self):
        structure = tiny_bert(EPCInference(eta_infer=1e-2, infer_steps=3))
        params = initialize_params(structure, jax.random.PRNGKey(3))
        fn = make_gradient_fn("pc", structure, "none", "epc")
        key = jax.random.PRNGKey(11)
        _assert_trees_equal(
            fn(params, bert_batch(), key, eta_infer=None),
            fn(params, bert_batch(), key),
        )

    def test_run_inference_override(self):
        eta = 2e-2
        built = tiny_bert(EPCInference(eta_infer=eta, infer_steps=4))
        other = tiny_bert(EPCInference(eta_infer=1e-3, infer_steps=4))
        params, clamps, state = bert_inputs(built)
        rate = jnp.asarray(eta)
        want = EPCInference.run_inference_with_stats(params, state, clamps, built)
        got = EPCInference.run_inference_with_stats(
            params, state, clamps, other, eta_infer=rate
        )
        _assert_trees_close(got, want, 1e-6)
        final = EPCInference.run_inference(params, state, clamps, other, eta_infer=rate)
        _assert_trees_equal(final, got[0])

    def test_changing_the_rate_does_not_recompile(self):
        structure = tiny_bert(EPCInference(eta_infer=1e-2, infer_steps=3))
        params = initialize_params(structure, jax.random.PRNGKey(3))
        fn = make_gradient_fn("pc", structure, "none", "epc")
        ends = []
        for eta in (1e-3, 1e-2, 5e-2):
            _, _, _, extras = fn(
                params, bert_batch(), jax.random.PRNGKey(0), eta_infer=jnp.asarray(eta)
            )
            ends.append(float(extras["pc_energy_end"]))
        assert fn._cache_size() == 1
        assert len(set(ends)) == 3

    def test_accumulation_uses_one_rate_for_every_micro_batch(self):
        structure = tiny_bert(EPCInference(eta_infer=1e-2, infer_steps=3))
        params = initialize_params(structure, jax.random.PRNGKey(3))
        fn = make_gradient_fn("pc", structure, "none", "epc")
        seen = []

        def recording(params, batch, key, eta_infer=None):
            seen.append(eta_infer)
            return fn(params, batch, key, eta_infer=eta_infer)

        keys = list(jax.random.split(jax.random.PRNGKey(0), 2))
        eta = jnp.asarray(2e-2)
        accumulate_gradients(
            recording, params, [bert_batch()] * 2, keys, summed=True, eta_infer=eta
        )
        assert seen == [eta, eta]

        def plain(params, batch, key):
            return fn(params, batch, key)

        grads, *_ = accumulate_gradients(plain, params, [bert_batch()], keys[:1])
        assert _leaves(grads)

    def test_rate_is_refused_outside_epc(self):
        structure = tiny_bert(InferenceSGD(eta_infer=0.1, infer_steps=2))
        params = initialize_params(structure, jax.random.PRNGKey(3))
        with pytest.raises(ValueError, match="only EPCInference"):
            get_graph_param_gradient_pcn_mlm(
                params, bert_batch(), structure, jax.random.PRNGKey(0), eta_infer=0.1
            )
        for fn in (
            make_gradient_fn("backprop", structure),
            make_gradient_fn("pc", structure, "none", "standard"),
        ):
            with pytest.raises(ValueError):
                fn(params, bert_batch(), jax.random.PRNGKey(0), eta_infer=0.1)


class TestControlEta:
    def test_first_drop_seeds_the_average(self):
        _, smoothed, _ = control_eta(0.1, None, 0.3, False, **LIMITS)
        assert smoothed == 0.3
        _, smoothed, _ = control_eta(0.1, 0.3, 0.1, False, **LIMITS)
        assert smoothed == pytest.approx(0.7 * 0.3 + 0.3 * 0.1)

    def test_small_drop_raises_eta_up_to_the_cap(self):
        eta, _, backed_off = control_eta(0.1, None, 0.1, False, **LIMITS)
        assert eta == pytest.approx(0.1 * math.sqrt(2.0)) and backed_off is False
        eta, _, _ = control_eta(0.9, None, 0.1, False, **LIMITS)
        assert eta == 1.0

    def test_large_drop_lowers_eta_down_to_the_floor(self):
        eta, _, _ = control_eta(0.1, None, 0.4, False, **LIMITS)
        assert eta == pytest.approx(0.1 / math.sqrt(2.0))
        eta, _, _ = control_eta(1.2e-3, None, 0.4, False, **LIMITS)
        assert eta == 1e-3

    def test_step_factor_is_clipped(self):
        eta, _, _ = control_eta(0.1, None, 1e-4, False, **LIMITS)
        assert eta == pytest.approx(0.2)
        eta, _, _ = control_eta(0.1, None, 0.99, False, **{**LIMITS, "gain": 2.0})
        assert eta == pytest.approx(0.05)

    def test_an_energy_increase_forces_the_backoff(self):
        eta, _, backed_off = control_eta(0.1, None, 0.01, True, **LIMITS)
        assert eta == pytest.approx(0.05) and backed_off is True
        eta, _, _ = control_eta(0.1, None, 0.01, True, **{**LIMITS, "backoff": 0.25})
        assert eta == pytest.approx(0.025)

    def test_a_non_finite_drop_backs_off_as_a_full_drop(self):
        for drop in (math.nan, math.inf):
            eta, smoothed, backed_off = control_eta(0.1, 0.2, drop, False, **LIMITS)
            assert eta == pytest.approx(0.05) and backed_off
            assert smoothed == pytest.approx(0.7 * 0.2 + 0.3)
        eta, smoothed, backed_off = control_eta(0.1, None, math.nan, False, **LIMITS)
        assert eta == pytest.approx(0.05) and smoothed == 1.0 and backed_off

    def test_a_diverged_update_keeps_the_rate_from_bouncing_back(self):
        eta, smoothed, _ = control_eta(0.1, None, -3.0, True, **LIMITS)
        assert smoothed == 1.0 and eta == pytest.approx(0.05)
        after, smoothed, backed_off = control_eta(eta, smoothed, 0.01, False, **LIMITS)
        assert not backed_off and after < eta
        assert smoothed == pytest.approx(0.7 + 0.3 * 0.01)


def _stage_stream(config, base, data_dir):
    return load_domain_cache(
        data_dir,
        DOMAINS[0],
        tokenizer_json=base.tokenizer_json,
        seq_len=int(base.model_config["seq_len"]),
        tokens_per_domain=60,
    ).train


def test_closed_loop_pulls_the_drop_toward_the_target(setup, tmp_path):
    """At a rate far above the stable one, the controller restores a positive
    drop near the target, where the fixed rate keeps diverging."""
    checkpoint, data_dir = setup
    settings = {**EPC, "pc_eta_infer": 3.0, "tokens_per_domain": 600}
    fixed = tiny_config(checkpoint, data_dir, tmp_path / "fixed", **settings)
    controlled = replace(fixed, **CONTROL)
    controlled.validate()
    base = load_base_model(fixed)
    stream = _stage_stream(fixed, base, data_dir)
    gradient_fn = make_gradient_fn("pc", base.structure, "none", "epc")
    logs = {}
    for name, config in (("fixed", fixed), ("controlled", controlled)):
        _, exposure, logs[name] = train_stage(
            base.params,
            base.structure,
            stream,
            config=config,
            stage=1,
            special_ids=base.special_ids,
            gradient_fn=gradient_fn,
            verbose=False,
        )
    log = logs["controlled"]
    assert len(log) == 30
    target = controlled.pc_energy_target
    drops = np.array([e["pc_energy_rel_decrease"] for e in log])
    fixed_drops = np.array([e["pc_energy_rel_decrease"] for e in logs["fixed"]])
    assert drops[0] == fixed_drops[0] < 0
    late = drops[-10:]
    assert np.all(late > 0)
    assert abs(np.median(late) - target) < 0.5 * target
    assert np.all(fixed_drops[-10:] < -1.0)
    assert not any(e["pc_energy_increased"] for e in log[-10:])
    assert log[0]["eta_backoff"] is True
    assert [e["eta_backoff"] for e in log] == [e["pc_energy_increased"] for e in log]
    etas = [e["eta_infer"] for e in log]
    assert etas[0] == 3.0 and etas[-1] < etas[0] / 10
    assert all(e["eta_next"] == n for e, n in zip(log, etas[1:]))
    assert exposure["eta_infer"]["max"] == 3.0
    assert exposure["eta_infer"]["min"] == min(etas)
    assert exposure["eta_infer"]["last"] == log[-1]["eta_infer"]
    assert "eta_infer" not in logs["fixed"][0]


def test_rate_resets_and_regime_records_carry_it(controlled_run):
    config, _ = controlled_run
    output = Path(config.output_dir)
    records = _read_jsonl(output / "regime.jsonl")
    exposure = json.loads((output / "exposure.json").read_text())
    for stage in range(1, 4):
        log = _read_jsonl(output / f"training_log_stage_{stage}.jsonl")
        assert log[0]["eta_infer"] == config.pc_eta_infer
        assert [e["eta_infer"] for e in log[1:]] == [e["eta_next"] for e in log[:-1]]
        expected = (
            [config.pc_eta_infer]
            + [e["eta_next"] for e in log[:-1]]
            + [log[-1]["eta_infer"]]
        )
        stage_records = [r for r in records if r["stage"] == stage]
        assert [r["update"] for r in stage_records] == list(range(len(log) + 1))
        assert [r["eta_infer"] for r in stage_records] == expected
        for record in stage_records:
            assert record["eta_controlled"] is True
            assert record["eta_lambda_max_final"] == pytest.approx(
                record["eta_infer"] * record["lambda_max_final"]
            )
        stats = exposure[DOMAINS[stage - 1]]["eta_infer"]
        used = [e["eta_infer"] for e in log]
        assert stats["min"] == min(used) and stats["max"] == max(used)
        assert stats["mean"] == pytest.approx(np.mean(used))
        assert stats["last"] == used[-1]
    assert len({r["eta_infer"] for r in records}) > 1


def test_end_alignment_uses_the_last_trained_rate(controlled_run):
    config, _ = controlled_run
    output = Path(config.output_dir)
    records = _read_jsonl(output / "alignment.jsonl")
    assert [(r["stage"], r["when"]) for r in records] == [
        (stage, when) for stage in range(1, 4) for when in ("start", "end")
    ]
    base = load_base_model(config)
    for record in records:
        log = _read_jsonl(output / f"training_log_stage_{record['stage']}.jsonl")
        if record["when"] == "start":
            assert record["eta_infer"] == config.pc_eta_infer
            assert "eta_controlled" not in record
        else:
            assert record["eta_infer"] == log[-1]["eta_infer"] != config.pc_eta_infer
            assert record["eta_controlled"] is True
    end = records[1]
    stream = load_domain_cache(
        config.data_dir,
        DOMAINS[0],
        tokenizer_json=base.tokenizer_json,
        seq_len=int(base.model_config["seq_len"]),
        tokens_per_domain=config.tokens_per_domain,
    ).train
    batches, keys = fixed_mlm_batches(
        stream,
        base.structure,
        base.special_ids,
        num_batches=1,
        batch_size=config.batch_size,
        mask_prob=config.mask_prob,
        seed=config.seed + 1,
    )
    params = load_stage_snapshot(output / "stage_1.npz", base.structure)
    fn = make_alignment_fn(base.structure, "none")
    at_end = gradient_alignment(
        fn, params, batches, keys, base.structure, jnp.asarray(end["eta_infer"])
    )
    configured = gradient_alignment(fn, params, batches, keys, base.structure)
    assert end["mean_cosine"] == at_end["mean_cosine"]
    assert end["mean_cosine"] != configured["mean_cosine"]


def test_explicit_regime_rate_is_kept(setup, tmp_path):
    checkpoint, data_dir = setup
    config = tiny_config(
        checkpoint,
        data_dir,
        tmp_path / "run",
        pc_regime_interval=1,
        pc_regime_iters=4,
        pc_regime_eta=0.02,
        pc_regime_steps=3,
        domains=DOMAINS[:2],
        **{**EPC, **CONTROL},
    )
    run_domain_forgetting(config, verbose=False)
    records = _read_jsonl(tmp_path / "run" / "regime.jsonl")
    assert records and all(r["eta_infer"] == 0.02 for r in records)
    assert not any("eta_controlled" in r for r in records)


def test_run_config_holds_the_controller(controlled_run, setup, tmp_path):
    config, _ = controlled_run
    run_config = json.loads((Path(config.output_dir) / "run_config.json").read_text())
    assert run_config["config"]["pc_eta_control"] == "energy"
    assert run_config["config"]["pc_energy_target"] == 0.05
    assert run_config["pc"]["eta_control"] == {
        "rule": "energy",
        "energy_target": 0.05,
        "gain": 0.5,
        "eta_min": pytest.approx(1e-4),
        "eta_max": 0.1,
        "backoff": 0.5,
        "drop_ema": 0.7,
        "step_clip": [0.5, 2.0],
        "stage_reset": True,
    }
    plain = replace(
        config, pc_eta_control="none", pc_energy_target=0.17, pc_eta_max=None
    )
    assert "pc_eta_control" not in plain.identity()
    assert "pc_eta_min" not in plain.identity()
    with pytest.raises(ValueError, match="cannot resume with a different setup"):
        run_domain_forgetting(plain, resume=True, verbose=False)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"pc_eta_control": "pid"}, "pc_eta_control must be one of"),
        ({"pc_energy_target": 0.3}, "need pc_eta_control 'energy'"),
        ({"pc_eta_min": 1e-3}, "need pc_eta_control 'energy'"),
        ({**CONTROL, "pc_inference": "standard"}, "needs mode 'pc' and ePC"),
        ({**CONTROL, "pc_energy_target": 0.0}, r"pc_energy_target must be in \(0, 1\)"),
        ({**CONTROL, "pc_energy_target": 1.0}, r"pc_energy_target must be in \(0, 1\)"),
        ({**CONTROL, "pc_eta_gain": 0.0}, "pc_eta_gain must be"),
        ({**CONTROL, "pc_eta_backoff": 0.0}, r"pc_eta_backoff must be in \(0, 1\]"),
        ({**CONTROL, "pc_eta_backoff": 1.5}, r"pc_eta_backoff must be in \(0, 1\]"),
        ({**CONTROL, "pc_eta_min": 0.0}, "must be finite and positive"),
        ({**CONTROL, "pc_eta_min": 0.02, "pc_eta_max": 0.015}, "must not exceed"),
        ({**CONTROL, "pc_eta_max": 0.005}, "must lie in"),
    ],
)
def test_controller_validation(overrides, message, tmp_path):
    config = tiny_config("ck", "data", tmp_path, **{**EPC, **overrides})
    with pytest.raises(ValueError, match=message):
        config.validate()


def test_controller_needs_pc_mode(tmp_path):
    with pytest.raises(ValueError, match="needs mode 'pc' and ePC"):
        tiny_config("ck", "data", tmp_path, **CONTROL).validate()
    tiny_config("ck", "data", tmp_path, **{**EPC, **CONTROL}).validate()


class TestLabelling:
    def test_unstable_rate_is_invalid(self):
        spectrum = EpsilonSpectrum.from_modes([10.0, 30.0], [0.5, 0.5])
        summary = regime_summary(spectrum, 0.1, 4)
        assert summary["unstable"] is True
        assert summary["f_valid"] is False
        assert summary["f_weighted"] is None and summary["band"] == "invalid"
        assert summary["f_max"] is None
        assert summary["regime"] == "invalid: unstable, eta*lambda_max = 3"

    def test_out_of_range_fraction_is_invalid(self):
        spectrum = EpsilonSpectrum.from_modes([18.0], [1.0])
        summary = regime_summary(spectrum, 0.1, 3)
        assert summary["unstable"] is False
        assert summary["f_valid"] is False and summary["f_weighted"] is None
        assert summary["band"] == "invalid" and summary["f_max"] is None
        assert summary["regime"] == (
            "invalid: relaxed fraction outside [0, 1], eta*lambda_max = 1.8"
        )

    def test_valid_and_no_positive_curvature(self):
        valid = regime_summary(EpsilonSpectrum.from_modes([5.0], [1.0]), 0.01, 5)
        assert valid["f_valid"] is True and 0.0 < valid["f_weighted"] < 1.0
        assert valid["band"] == "partially relaxed"
        flat = regime_summary(EpsilonSpectrum.from_modes([-1.0], [1.0]), 0.01, 5)
        assert flat["f_valid"] is False and flat["f_weighted"] is None
        assert flat["band"] == "no positive curvature"

    def test_probe_prints_diverged(self):
        good = regime_summary(EpsilonSpectrum.from_modes([5.0], [1.0]), 0.01, 5)
        bad = regime_summary(
            EpsilonSpectrum.from_modes([10.0, 30.0], [0.5, 0.5]), 0.1, 4
        )
        row = {"eta_infer": 0.1, "infer_steps": 4, "min_cosine": 0.5}
        mixed = combine_regimes([good, bad])
        assert mixed["f_valid"] is False
        assert mixed["f_weighted_mean"] == pytest.approx(good["f_weighted"])
        assert "diverged" in format_epc_setting({**row, "regime": mixed})
        clean = format_epc_setting({**row, "regime": combine_regimes([good])})
        assert "diverged" not in clean and f"{good['f_weighted']:.3f}" in clean


PINNED = {
    "backprop": ([3.417818069458008, 3.4387256622314455], [0.7389356, 0.6775386]),
    "spiking": ([3.417818069458008, 3.4387256622314455], [0.7389356, 0.6775386]),
    "epc": ([3.417818069458008, 3.43857421875], [0.6274642, 0.6075130]),
}
SETTINGS = {"backprop": {}, "spiking": {"mode": "pc"}, "epc": EPC}


def _two_updates(setup, tmp_path, settings, wrap=None, **overrides):
    checkpoint, data_dir = setup
    config = tiny_config(
        checkpoint,
        data_dir,
        tmp_path / "run",
        **{"tokens_per_domain": 40, **settings, **overrides},
    )
    config.validate()
    base = load_base_model(config)
    fn = make_gradient_fn(
        config.mode, base.structure, config.pc_forward_updates, config.pc_inference
    )
    params, exposure, log = train_stage(
        base.params,
        base.structure,
        _stage_stream(config, base, data_dir),
        config=config,
        stage=1,
        special_ids=base.special_ids,
        gradient_fn=fn if wrap is None else wrap(fn),
        verbose=False,
    )
    return fn, params, exposure, log


@pytest.mark.parametrize("name", ["backprop", "spiking", "epc"])
def test_uncontrolled_path_is_unchanged(setup, tmp_path, name):
    """Without the controller, training is identical whether or not the rate
    keyword is passed, and reproduces the numbers of the code before it."""

    def explicit(fn):
        return lambda params, batch, key: fn(params, batch, key, eta_infer=None)

    _, params, _, log = _two_updates(setup, tmp_path, SETTINGS[name])
    _, again, _, explicit_log = _two_updates(setup, tmp_path, SETTINGS[name], explicit)
    _assert_trees_equal(again, params)
    assert explicit_log == log
    losses, norms = PINNED[name]
    np.testing.assert_allclose([e["masked_loss"] for e in log], losses, rtol=1e-5)
    np.testing.assert_allclose([e["grad_norm"] for e in log], norms, rtol=1e-5)
    assert not any("eta_infer" in e or "eta_backoff" in e for e in log)


def test_non_finite_start_energy_backs_off(setup, tmp_path):
    def poisoned(fn):
        def gradient_fn(params, batch, key, eta_infer=None):
            grads, loss, count, extras = fn(params, batch, key, eta_infer=eta_infer)
            return grads, loss, count, {**extras, "pc_energy_start": jnp.nan}

        return gradient_fn

    _, _, _, log = _two_updates(
        setup, tmp_path, EPC, poisoned, pc_eta_max=0.1, **CONTROL
    )
    for entry in log:
        assert math.isnan(entry["pc_energy_rel_decrease"])
        assert entry["eta_backoff"] is True
        assert entry["eta_next"] == pytest.approx(0.5 * entry["eta_infer"])
    path = tmp_path / "log.jsonl"
    _write_jsonl(path, log)
    assert _read_jsonl(path)[0]["pc_energy_rel_decrease"] is None


def test_controlled_training_compiles_once(setup, tmp_path):
    fn, _, exposure, log = _two_updates(
        setup, tmp_path, EPC, pc_eta_max=0.1, tokens_per_domain=120, **CONTROL
    )
    assert len({e["eta_infer"] for e in log}) > 1
    assert exposure["eta_infer"]["max"] > exposure["eta_infer"]["min"]
    assert fn._cache_size() == 1
