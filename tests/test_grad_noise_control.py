"""Tests for the noise-matched backprop gradient control."""

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

from fabricpc.core.types import GraphParams, NodeParams
from fabricpc.experiments.domain_forgetting import runner
from fabricpc.experiments.domain_forgetting.runner import (
    accumulate_gradients,
    build_stage_optimizer,
    compare_runs,
    grad_noise_key,
    load_base_model,
    make_gradient_fn,
    noise_gradients,
    num_stage_updates,
    run_domain_forgetting,
    train_stage,
)
from fabricpc.training.train_backprop import _apply_optimizer_update_when_masked
from tests.test_domain_forgetting_runner import DOMAINS, make_setup, tiny_config
from tests.test_epc_eta_control import PINNED, _stage_stream

NOISE_KEYS = ("grad_noise_cosine", "grad_noise_mode", "grad_noise_achieved")


def _module(rng, shapes):
    return NodeParams(
        weights={
            name: jnp.asarray(rng.normal(size=shape), jnp.float32)
            for name, shape in shapes.items()
        },
        biases={"b": jnp.asarray(rng.normal(size=(7,)), jnp.float32)},
    )


def _grads(seed=0):
    rng = np.random.default_rng(seed)
    nodes = {
        "a": _module(rng, {"W": (16, 8)}),
        "b": _module(rng, {"W": (32,), "V": (4, 4)}),
        "c": NodeParams(
            weights={"W": jnp.zeros((5, 3), jnp.float32)},
            biases={"b": jnp.zeros((3,), jnp.float32)},
        ),
        "d": NodeParams(weights={}, biases={}),
    }
    return GraphParams(nodes=nodes)


def _flat(node):
    return np.concatenate(
        [np.asarray(x, np.float64).ravel() for x in jax.tree_util.tree_leaves(node)]
    )


def _noise(grads, cosine, seed=0, stage=1, update=1, persistent=False):
    return noise_gradients(
        grads,
        jnp.asarray(cosine, jnp.float32),
        jnp.asarray(seed, jnp.uint32),
        jnp.asarray(stage, jnp.uint32),
        jnp.asarray(update, jnp.uint32),
        persistent=persistent,
    )


def _cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def _orthogonal_part(clean, noised, cosine):
    unit = clean / np.linalg.norm(clean)
    return (noised / np.linalg.norm(clean) - cosine * unit) / math.sqrt(
        1 - cosine**2
    ), unit


def _leaves(tree):
    return [np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(tree)]


def _assert_trees_equal(left, right):
    for a, b in zip(_leaves(left), _leaves(right), strict=True):
        np.testing.assert_array_equal(a, b)


class TestNoiseFunction:
    @pytest.mark.parametrize("cosine", [0.05, 0.5, 0.78, 0.89, 0.98, 0.999])
    def test_cosine_is_exact_and_norm_preserved(self, cosine):
        grads = _grads()
        noised, achieved = _noise(grads, cosine)
        for name in ("a", "b"):
            clean, out = _flat(grads.nodes[name]), _flat(noised.nodes[name])
            assert abs(_cos(out, clean) - cosine) < 1e-5
            assert abs(np.linalg.norm(out) / np.linalg.norm(clean) - 1) < 1e-5
        assert abs(float(achieved) - cosine) < 1e-5

    def test_structure_and_dtypes_are_kept(self):
        grads = _grads()
        noised, _ = _noise(grads, 0.9)
        assert jax.tree_util.tree_structure(noised) == jax.tree_util.tree_structure(
            grads
        )
        for a, b in zip(_leaves(noised), _leaves(grads), strict=True):
            assert a.shape == b.shape and a.dtype == b.dtype

    def test_zero_and_empty_modules_are_unchanged(self):
        grads = _grads()
        noised, _ = _noise(grads, 0.5)
        _assert_trees_equal(noised.nodes["c"], grads.nodes["c"])
        assert noised.nodes["d"] == grads.nodes["d"]

    def test_one_parameter_module_is_unchanged(self):
        grads = GraphParams(
            nodes={"s": NodeParams(weights={"w": jnp.asarray([3.0])}, biases={})}
        )
        noised, achieved = _noise(grads, 0.5)
        _assert_trees_equal(noised, grads)
        assert math.isnan(float(achieved))
        mixed = GraphParams(nodes={**_grads().nodes, **grads.nodes})
        _, achieved = _noise(mixed, 0.5)
        assert abs(float(achieved) - 0.5) < 1e-5

    def test_all_zero_gradient_gives_nan_achieved(self):
        grads = GraphParams(nodes={"c": _grads().nodes["c"]})
        noised, achieved = _noise(grads, 0.5)
        _assert_trees_equal(noised, grads)
        assert math.isnan(float(achieved))

    def test_noise_is_deterministic_and_keyed(self):
        grads = _grads()
        first, _ = _noise(grads, 0.7, seed=3, stage=2, update=5)
        again, _ = _noise(grads, 0.7, seed=3, stage=2, update=5)
        _assert_trees_equal(first, again)
        for other in (
            {"seed": 4, "stage": 2, "update": 5},
            {"seed": 3, "stage": 3, "update": 5},
            {"seed": 3, "stage": 2, "update": 6},
        ):
            changed, _ = _noise(grads, 0.7, **other)
            for name in ("a", "b"):
                assert not np.array_equal(
                    _flat(changed.nodes[name]), _flat(first.nodes[name])
                )

    def test_modules_get_independent_noise(self):
        rng = np.random.default_rng(1)
        shared = _module(rng, {"W": (16, 8)})
        grads = GraphParams(nodes={"x": shared, "y": shared})
        noised, _ = _noise(grads, 0.6)
        clean = _flat(shared)
        ux, _ = _orthogonal_part(clean, _flat(noised.nodes["x"]), 0.6)
        uy, _ = _orthogonal_part(clean, _flat(noised.nodes["y"]), 0.6)
        assert abs(_cos(ux, uy)) < 0.5

    def test_noise_direction_is_a_unit_vector_orthogonal_to_the_gradient(self):
        grads = _grads()
        for update in range(1, 6):
            noised, _ = _noise(grads, 0.89, update=update)
            for name in ("a", "b"):
                u, unit = _orthogonal_part(
                    _flat(grads.nodes[name]), _flat(noised.nodes[name]), 0.89
                )
                assert abs(float(u @ unit)) < 1e-5
                assert abs(np.linalg.norm(u) - 1) < 1e-5

    def test_tiny_gradients_are_handled(self):
        grads = jax.tree_util.tree_map(lambda g: g * 1e-30, _grads())
        noised, achieved = _noise(grads, 0.8)
        clean, out = _flat(grads.nodes["a"]), _flat(noised.nodes["a"])
        assert np.all(np.isfinite(out))
        assert abs(_cos(out, clean) - 0.8) < 1e-5
        assert abs(np.linalg.norm(out) / np.linalg.norm(clean) - 1) < 1e-5
        assert abs(float(achieved) - 0.8) < 1e-5

    def test_row_sparse_gradient_keeps_its_support(self):
        rng = np.random.default_rng(2)
        table = np.zeros((50, 8), np.float32)
        rows = [3, 17, 40]
        table[rows] = rng.normal(size=(3, 8))
        grads = GraphParams(
            nodes={
                "embed": NodeParams(
                    weights={"E": jnp.asarray(table)},
                    biases={"b": jnp.asarray(rng.normal(size=(8,)), jnp.float32)},
                )
            }
        )
        for cosine in (0.3, 0.89):
            noised, achieved = _noise(grads, cosine)
            out = np.asarray(noised.nodes["embed"].weights["E"])
            untouched = np.ones(50, bool)
            untouched[rows] = False
            assert np.all(out[untouched] == 0.0)
            assert np.all(out[rows] != 0.0)
            clean, flat = _flat(grads.nodes["embed"]), _flat(noised.nodes["embed"])
            assert abs(_cos(flat, clean) - cosine) < 1e-5
            assert abs(np.linalg.norm(flat) / np.linalg.norm(clean) - 1) < 1e-5
            assert abs(float(achieved) - cosine) < 1e-5

    def test_noise_keys_are_apart_from_the_training_keys(self):
        """With grad_noise_seed equal to the run seed, no noise key equals
        the update's step key, its splits or its fold-ins."""
        seed, stage = 0, 1
        _, step_key = jax.random.split(
            jax.random.fold_in(jax.random.PRNGKey(seed), stage)
        )
        training = [step_key]
        for n in (2, 3, 4, 8):
            training.extend(jax.random.split(step_key, n))
        training.extend(jax.random.fold_in(step_key, j) for j in range(8))
        training = {tuple(np.asarray(k).tolist()) for k in training}
        for persistent in (False, True):
            for index in range(8):
                key = grad_noise_key(seed, stage, 1, index, persistent)
                assert tuple(np.asarray(key).tolist()) not in training

    def test_persistent_noise_is_one_draw_reprojected(self):
        """Persistent mode reuses r = m ⊙ N(0, I) from the update-free key at
        every update and projects it against each update's gradient."""
        rng = np.random.default_rng(5)
        updates = [
            GraphParams(nodes={"a": _module(rng, {"W": (16, 8)})}) for _ in range(3)
        ]
        directions = {}
        for persistent in (False, True):
            directions[persistent] = []
            for update, grads in enumerate(updates, start=1):
                noised, achieved = _noise(
                    grads, 0.89, stage=2, update=update, persistent=persistent
                )
                clean, out = _flat(grads.nodes["a"]), _flat(noised.nodes["a"])
                assert abs(_cos(out, clean) - 0.89) < 1e-5
                assert abs(float(achieved) - 0.89) < 1e-5
                u, unit = _orthogonal_part(clean, out, 0.89)
                directions[persistent].append(u)
                key = grad_noise_key(0, 2, update, 0, persistent)
                r = np.asarray(
                    jax.random.normal(key, clean.shape, jnp.float32), np.float64
                )
                want = r - (r @ unit) * unit
                assert _cos(u, want) > 1 - 1e-5
        fresh, kept = directions[False], directions[True]
        assert abs(_cos(fresh[0], fresh[1])) < 0.5
        assert _cos(kept[0], kept[1]) > 0.5
        same, _ = _noise(updates[0], 0.89, update=1, persistent=True)
        later, _ = _noise(updates[0], 0.89, update=7, persistent=True)
        _assert_trees_equal(same, later)

    def test_changing_the_key_or_cosine_does_not_recompile(self):
        grads = _grads()
        noise_gradients.clear_cache()
        for update, cosine in ((1, 0.5), (2, 0.6), (3, 0.9)):
            _noise(grads, cosine, seed=update, stage=update, update=update)
        assert noise_gradients._cache_size() == 1


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    return make_setup(tmp_path_factory.mktemp("grad_noise"))


def _stage(setup, tmp_path, *, tokens=40, record=None, multipliers=None, **overrides):
    checkpoint, data_dir = setup
    config = tiny_config(
        checkpoint, data_dir, tmp_path / "run", tokens_per_domain=tokens, **overrides
    )
    config.validate()
    base = load_base_model(config)
    fn = make_gradient_fn(config.mode, base.structure, config.pc_forward_updates)

    def gradient_fn(params, batch, key):
        if record is not None:
            record.append((batch, key))
        return fn(params, batch, key)

    params, exposure, log = train_stage(
        base.params,
        base.structure,
        _stage_stream(config, base, data_dir),
        config=config,
        stage=1,
        special_ids=base.special_ids,
        gradient_fn=gradient_fn,
        multipliers=multipliers,
        verbose=False,
    )
    return config, base, fn, params, log


def test_off_path_matches_a_plain_optimizer_loop(setup, tmp_path):
    """Without noise, train_stage equals AdamW applied to the accumulated
    gradients of the same micro-batches, bit for bit, and logs no noise keys."""
    calls: list = []
    config, base, fn, params, log = _stage(setup, tmp_path, record=calls)
    seq_len = int(base.model_config["seq_len"])
    num_updates = num_stage_updates(config.stage_tokens, config.batch_size, seq_len)
    optimizer, _ = build_stage_optimizer(config, num_updates)
    update_fn = jax.jit(
        lambda p, o, g, c: _apply_optimizer_update_when_masked(
            params=p, opt_state=o, grads=g, optimizer=optimizer, masked_count=c
        )
    )
    want, state = base.params, optimizer.init(base.params)
    assert len(calls) == len(log) == num_updates
    for batch, key in calls:
        grads, _, count, _ = accumulate_gradients(fn, want, [batch], [key])
        want, state = update_fn(want, state, grads, jnp.asarray(count))
    _assert_trees_equal(params, want)
    losses, norms = PINNED["backprop"]
    np.testing.assert_allclose([e["masked_loss"] for e in log], losses, rtol=1e-5)
    np.testing.assert_allclose([e["grad_norm"] for e in log], norms, rtol=1e-5)
    assert not any(key in entry for entry in log for key in NOISE_KEYS)


def test_noise_keeps_batches_masks_and_keys(setup, tmp_path):
    clean_calls: list = []
    noisy_calls: list = []
    _, _, _, clean_params, clean_log = _stage(setup, tmp_path, record=clean_calls)
    _, _, _, noisy_params, noisy_log = _stage(
        setup, tmp_path, record=noisy_calls, grad_noise_cosine=0.89
    )
    assert len(clean_calls) == len(noisy_calls)
    for (clean_batch, clean_key), (noisy_batch, noisy_key) in zip(
        clean_calls, noisy_calls
    ):
        np.testing.assert_array_equal(clean_key, noisy_key)
        assert clean_batch.keys() == noisy_batch.keys()
        for name in clean_batch:
            np.testing.assert_array_equal(clean_batch[name], noisy_batch[name])
    assert noisy_log[0]["masked_loss"] == clean_log[0]["masked_loss"]
    assert noisy_log[0]["masked_tokens"] == clean_log[0]["masked_tokens"]
    assert noisy_log[1]["masked_loss"] != clean_log[1]["masked_loss"]
    assert not all(
        np.array_equal(a, b)
        for a, b in zip(_leaves(noisy_params), _leaves(clean_params))
    )


def test_logs_report_the_achieved_cosine_and_compile_once(setup, tmp_path):
    noise_gradients.clear_cache()
    _, _, _, _, log = _stage(setup, tmp_path, tokens=120, grad_noise_cosine=0.78)
    assert len(log) > 2
    assert noise_gradients._cache_size() == 1
    for entry in log:
        assert entry["grad_noise_cosine"] == 0.78
        assert entry["grad_noise_mode"] == "fresh"
        assert abs(entry["grad_noise_achieved"] - 0.78) < 1e-5
        assert math.isfinite(entry["grad_norm"]) and entry["grad_norm"] > 0


def test_noised_grad_norm_equals_the_clean_one(setup, tmp_path):
    """The rotation preserves each module's norm, so the global norm the clip
    sees on the first update is the clean one."""
    _, _, _, _, clean = _stage(setup, tmp_path)
    _, _, _, _, noisy = _stage(setup, tmp_path, grad_noise_cosine=0.5)
    assert noisy[0]["grad_norm"] == pytest.approx(clean[0]["grad_norm"], rel=1e-5)


def test_persistent_training_logs_the_mode(setup, tmp_path):
    _, _, _, _, log = _stage(
        setup,
        tmp_path,
        tokens=120,
        grad_noise_cosine=0.89,
        grad_noise_mode="persistent",
    )
    for entry in log:
        assert entry["grad_noise_mode"] == "persistent"
        assert abs(entry["grad_noise_achieved"] - 0.89) < 1e-5


def test_noise_with_a_frozen_module(setup, tmp_path):
    """A frozen module keeps its params, and its noised gradient stays out
    of the clip's global norm."""
    checkpoint, data_dir = setup
    base = load_base_model(tiny_config(checkpoint, data_dir, tmp_path))
    frozen = sorted(base.params.nodes)[0]
    multipliers = {name: 1.0 for name in base.params.nodes}
    multipliers[frozen] = 0.0
    _, _, _, params, log = _stage(
        setup, tmp_path, multipliers=multipliers, grad_noise_cosine=0.5
    )
    _assert_trees_equal(params.nodes[frozen], base.params.nodes[frozen])
    _, _, _, _, clean_frozen = _stage(setup, tmp_path, multipliers=multipliers)
    _, _, _, _, clean = _stage(setup, tmp_path)
    assert log[0]["grad_norm"] == pytest.approx(clean_frozen[0]["grad_norm"], rel=1e-5)
    assert log[0]["grad_norm"] != pytest.approx(clean[0]["grad_norm"], rel=1e-3)


def test_one_draw_per_update_with_micro_batches(setup, tmp_path, monkeypatch):
    calls: list = []
    seen: list = []

    def recording(grads, cosine, seed, stage, update, persistent=False):
        seen.append(int(update))
        return noise_gradients(grads, cosine, seed, stage, update, persistent)

    monkeypatch.setattr(runner, "noise_gradients", recording)
    _, _, _, _, log = _stage(
        setup,
        tmp_path,
        record=calls,
        micro_batch_size=2,
        grad_noise_cosine=0.89,
    )
    assert seen == list(range(1, len(log) + 1))
    assert len(calls) == 2 * len(log)
    assert all(abs(e["grad_noise_achieved"] - 0.89) < 1e-5 for e in log)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"mode": "pc", "grad_noise_cosine": 0.9}, "needs mode 'backprop'"),
        (
            {
                "mode": "pc",
                "pc_inference": "epc",
                "pc_eta_infer": 0.01,
                "pc_infer_steps": 3,
                "pc_forward_updates": "none",
                "grad_noise_cosine": 0.9,
            },
            "needs mode 'backprop'",
        ),
        ({"grad_noise_cosine": 0.0}, r"must be finite and in \(0, 1\)"),
        ({"grad_noise_cosine": -0.5}, r"must be finite and in \(0, 1\)"),
        ({"grad_noise_cosine": 1.0}, r"must be finite and in \(0, 1\)"),
        ({"grad_noise_cosine": 1.5}, r"must be finite and in \(0, 1\)"),
        ({"grad_noise_cosine": math.nan}, r"must be finite and in \(0, 1\)"),
        ({"grad_noise_cosine": math.inf}, r"must be finite and in \(0, 1\)"),
        ({"grad_noise_seed": 3}, "grad_noise_seed need grad_noise_cosine"),
        ({"grad_noise_mode": "persistent"}, "grad_noise_mode need grad_noise_cosine"),
        ({"grad_noise_mode": "sticky"}, "grad_noise_mode must be one of"),
        (
            {"grad_noise_cosine": 0.9, "grad_noise_mode": "sticky"},
            "grad_noise_mode must be one of",
        ),
        ({"grad_noise_cosine": 0.9, "grad_noise_seed": -1}, r"in \[0, 2\*\*32\)"),
        (
            {"grad_noise_cosine": 0.9, "grad_noise_seed": 2**32},
            r"in \[0, 2\*\*32\)",
        ),
    ],
)
def test_validation(overrides, message, tmp_path):
    with pytest.raises(ValueError, match=message):
        tiny_config("ck", "data", tmp_path, **overrides).validate()


def test_identity_only_holds_noise_when_enabled(tmp_path):
    clean = tiny_config("ck", "data", tmp_path)
    noisy = replace(clean, grad_noise_cosine=0.89, grad_noise_seed=2)
    assert not any(key in clean.identity() for key in NOISE_KEYS)
    assert "grad_noise_seed" not in clean.identity()
    assert noisy.identity()["grad_noise_cosine"] == 0.89
    assert noisy.identity()["grad_noise_seed"] == 2
    assert noisy.identity()["grad_noise_mode"] == "fresh"
    assert "grad_noise_mode" not in clean.identity()
    replace(noisy, grad_noise_seed=2**32 - 1, grad_noise_mode="persistent").validate()
    rest = {k: v for k, v in noisy.identity().items() if not k.startswith("grad_noise")}
    assert rest == clean.identity()


@pytest.fixture(scope="module")
def runs(setup, tmp_path_factory):
    checkpoint, data_dir = setup
    root = tmp_path_factory.mktemp("noise_runs")
    clean = tiny_config(checkpoint, data_dir, root / "clean")
    noisy = replace(
        clean, output_dir=str(root / "noisy"), grad_noise_cosine=0.89, grad_noise_seed=1
    )
    run_domain_forgetting(clean, verbose=False)
    run_domain_forgetting(noisy, verbose=False)
    return clean, noisy


def test_run_config_records_the_noise_and_resume_is_guarded(runs):
    clean, noisy = runs
    recorded = json.loads((Path(noisy.output_dir) / "run_config.json").read_text())
    assert recorded["config"]["grad_noise_cosine"] == 0.89
    assert recorded["config"]["grad_noise_seed"] == 1
    assert recorded["grad_noise"]["cosine"] == 0.89
    assert recorded["grad_noise"]["seed"] == 1
    assert recorded["grad_noise"]["mode"] == "fresh"
    assert recorded["config"]["grad_noise_mode"] == "fresh"
    plain = json.loads((Path(clean.output_dir) / "run_config.json").read_text())
    assert "grad_noise" not in plain
    assert not any(key.startswith("grad_noise") for key in plain["config"])
    logs = [
        json.loads(line)
        for line in (Path(noisy.output_dir) / "training_log_stage_1.jsonl")
        .read_text()
        .splitlines()
    ]
    assert all(abs(e["grad_noise_achieved"] - 0.89) < 1e-5 for e in logs)

    with pytest.raises(ValueError, match="cannot resume with a different setup"):
        run_domain_forgetting(
            replace(noisy, grad_noise_cosine=None, grad_noise_seed=0),
            resume=True,
            verbose=False,
        )
    with pytest.raises(ValueError, match="cannot resume with a different setup"):
        run_domain_forgetting(
            replace(noisy, grad_noise_seed=2), resume=True, verbose=False
        )
    with pytest.raises(ValueError, match="cannot resume with a different setup"):
        run_domain_forgetting(
            replace(clean, grad_noise_cosine=0.89), resume=True, verbose=False
        )
    run_domain_forgetting(noisy, resume=True, verbose=False)
    run_domain_forgetting(clean, resume=True, verbose=False)


def test_compare_accepts_a_noised_run_against_a_clean_one(runs):
    clean, noisy = runs
    report = compare_runs(noisy.output_dir, clean.output_dir)
    assert report["modes"] == ["backprop", "backprop"]
    assert report["grad_noise_cosine"] == [0.89, None]
    assert report["max_abs_ce_diff"] > 0
    assert report["stage_param_rel_distance"][0] == 0.0
    assert report["stage_param_rel_distance"][1] > 0


def test_cli_flags_reach_the_config(setup, tmp_path):
    from examples import mlm_domain_forgetting as cli

    checkpoint, data_dir = setup
    argv = [
        "--checkpoint",
        str(checkpoint),
        "--data_dir",
        str(data_dir),
        "--domains",
        *DOMAINS,
        "--output_dir",
        str(tmp_path),
        "--skip_shape_check",
    ]
    plain = cli.config_from_args(cli.build_parser().parse_args(argv))
    assert plain.grad_noise_cosine is None and plain.grad_noise_seed == 0
    noisy = cli.config_from_args(
        cli.build_parser().parse_args(
            [
                *argv,
                "--grad_noise_cosine",
                "0.89",
                "--grad_noise_seed",
                "4",
                "--grad_noise_mode",
                "persistent",
            ]
        )
    )
    assert noisy == replace(
        plain,
        grad_noise_cosine=0.89,
        grad_noise_seed=4,
        grad_noise_mode="persistent",
    )
