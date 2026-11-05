"""Tests for the retention baselines and the optimizer-restart control."""

import fnmatch
import json
import os
import re
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from examples import domain_forgetting_analysis as analysis
from examples import mlm_domain_forgetting as cli
from fabricpc.core.types import GraphParams, NodeParams
from fabricpc.experiments.domain_forgetting import runner
from fabricpc.experiments.domain_forgetting.data import load_domain_cache
from fabricpc.experiments.domain_forgetting.frontier import (
    RETENTION_GRIDS,
    _is_reference,
    collect_runs,
    compute_frontier,
    describe_gate,
    format_selection,
    format_table,
    load_run_point,
    select_retention,
)
from fabricpc.experiments.domain_forgetting.gates import spaced_batches
from fabricpc.experiments.domain_forgetting.retention import (
    FISHER_WINDOWS,
    diagonal_fisher,
    fisher_structure,
    make_squared_gradient_fn,
    stage_fisher,
)
from fabricpc.experiments.domain_forgetting.runner import (
    OptimizerCarry,
    RunConfig,
    adam_moments,
    build_stage_optimizer,
    load_adam_moments,
    load_base_model,
    make_gradient_fn,
    retention_penalty,
    run_domain_forgetting,
    save_adam_moments,
    train_stage,
    with_adam_moments,
)
from fabricpc.experiments.forgetting_benchmark.sequential_trainer import params_hash
from fabricpc.training.train_backprop import compute_forward_pass, compute_loss
from tests.test_domain_forgetting_influence_gates import _write_run
from tests.test_domain_forgetting_runner import (
    DOMAINS,
    make_setup,
    tiny_config,
)

EPC = {
    "mode": "pc",
    "pc_inference": "epc",
    "pc_eta_infer": 0.01,
    "pc_infer_steps": 3,
    "pc_forward_updates": "none",
    "pc_eta_control": "energy",
}
L2SP = {"retention": "l2sp", "retention_lambda": 0.1}
EWC = {"retention": "ewc", "retention_lambda": 10.0}


def toy_params(values):
    """Graph params of one node per entry, each leaf named ``w``."""
    return GraphParams(
        nodes={
            name: NodeParams(weights={"w": jnp.asarray(v, jnp.float32)}, biases={})
            for name, v in values.items()
        }
    )


def leaf(params, name="a"):
    return np.asarray(params.nodes[name].weights["w"])


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    return make_setup(tmp_path_factory.mktemp("retention"))


@pytest.fixture(scope="module")
def ewc_run(setup, tmp_path_factory):
    checkpoint, data_dir = setup
    output = tmp_path_factory.mktemp("ewc_run") / "run"
    config = tiny_config(checkpoint, data_dir, output, **EWC)
    return config, run_domain_forgetting(config, verbose=False)


@pytest.fixture(scope="module")
def keep_run(setup, tmp_path_factory):
    checkpoint, data_dir = setup
    output = tmp_path_factory.mktemp("keep_run") / "run"
    config = tiny_config(checkpoint, data_dir, output, keep_optimizer_state=True)
    return config, run_domain_forgetting(config, verbose=False)


@pytest.fixture(scope="module")
def plain_run(setup, tmp_path_factory):
    checkpoint, data_dir = setup
    output = tmp_path_factory.mktemp("plain_run") / "run"
    config = tiny_config(checkpoint, data_dir, output)
    return config, run_domain_forgetting(config, verbose=False)


def _hashes(result):
    return result["summary"]["stage_params_sha256"]


def _log(config, stage):
    path = Path(config.output_dir) / f"training_log_stage_{stage}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_l2sp_penalty_and_gradient_known_answer():
    params = toy_params({"a": [1.0, 2.0], "b": [3.0]})
    anchor = toy_params({"a": [0.0, 0.0], "b": [1.0]})
    value, grads = retention_penalty(params, ((anchor, None),), jnp.asarray(0.5))
    assert float(value) == pytest.approx(0.5 * 0.5 * (1.0 + 4.0 + 4.0))
    np.testing.assert_allclose(leaf(grads), [0.5, 1.0])
    np.testing.assert_allclose(leaf(grads, "b"), [1.0])


def test_penalty_is_zero_at_the_anchor():
    params = toy_params({"a": [1.0, 2.0]})
    fisher = toy_params({"a": [3.0, 4.0]})
    for term in ((params, None), (params, fisher)):
        value, grads = retention_penalty(params, (term,), jnp.asarray(7.0))
        assert float(value) == 0.0
        assert not np.any(leaf(grads))


def test_ewc_penalty_sums_separate_terms_known_answer():
    params = toy_params({"a": [1.0, 2.0]})
    terms = (
        (toy_params({"a": [0.0, 1.0]}), toy_params({"a": [2.0, 3.0]})),
        (toy_params({"a": [3.0, 2.0]}), toy_params({"a": [5.0, 7.0]})),
    )
    value, grads = retention_penalty(params, terms, jnp.asarray(0.1))
    assert float(value) == pytest.approx(0.05 * (2 * 1 + 3 * 1 + 5 * 4 + 7 * 0))
    np.testing.assert_allclose(
        leaf(grads), [0.1 * (2 * 1 + 5 * -2), 0.1 * (3 * 1 + 7 * 0)], rtol=1e-6
    )


def test_penalty_gradient_matches_autodiff_and_finite_differences():
    params = toy_params({"a": [0.5, -1.0, 2.0], "b": [[1.0], [3.0]]})
    terms = (
        (
            toy_params({"a": [0.0, 1.0, 1.0], "b": [[0.0], [2.0]]}),
            toy_params({"a": [1.0, 2.0, 3.0], "b": [[4.0], [5.0]]}),
        ),
    )
    weight = jnp.asarray(2.0)
    _, grads = retention_penalty(params, terms, weight)
    auto = jax.grad(lambda p: retention_penalty(p, terms, weight)[0])(params)
    for mine, other in zip(
        jax.tree_util.tree_leaves(grads), jax.tree_util.tree_leaves(auto), strict=True
    ):
        np.testing.assert_allclose(mine, other, rtol=1e-6)
    step = 1e-2
    shifted = toy_params({"a": [0.5 + step, -1.0, 2.0], "b": [[1.0], [3.0]]})
    below = toy_params({"a": [0.5 - step, -1.0, 2.0], "b": [[1.0], [3.0]]})
    numeric = (
        float(retention_penalty(shifted, terms, weight)[0])
        - float(retention_penalty(below, terms, weight)[0])
    ) / (2 * step)
    assert numeric == pytest.approx(float(leaf(grads)[0]), rel=1e-3)


def test_penalty_does_not_retrace_between_updates():
    params = toy_params({"a": [1.0]})
    terms = ((toy_params({"a": [0.5]}), toy_params({"a": [1.5]})),)
    retention_penalty(params, terms, jnp.asarray(1.0))
    before = retention_penalty._cache_size()
    for step in range(5):
        moved = toy_params({"a": [1.0 + step]})
        retention_penalty(moved, terms, jnp.asarray(float(step)))
    assert retention_penalty._cache_size() == before


def _tiny_base(setup, tmp_path, **overrides):
    checkpoint, data_dir = setup
    config = tiny_config(checkpoint, data_dir, tmp_path / "run", **overrides)
    return config, load_base_model(config), data_dir


def _stream(config, base, data_dir, name):
    cache = load_domain_cache(
        data_dir,
        name,
        tokenizer_json=base.tokenizer_json,
        seq_len=int(base.model_config["seq_len"]),
        tokens_per_domain=config.tokens_per_domain,
    )
    return cache.train


def test_fisher_equals_an_independent_mean_of_squared_batch_gradients(setup, tmp_path):
    config, base, data_dir = _tiny_base(setup, tmp_path, **EWC)
    stream = _stream(config, base, data_dir, "news")
    structure = fisher_structure(base.model_config, config)
    fisher, record = stage_fisher(
        base.params,
        stream,
        structure,
        make_squared_gradient_fn(structure),
        base.special_ids,
        config=config,
        stage=1,
        domain="news",
    )
    assert record["windows"] == FISHER_WINDOWS == 256
    assert record["batch_size"] == 8
    assert record["batches"] == 32

    batches, _ = spaced_batches(
        stream,
        structure,
        base.special_ids,
        num_windows=FISHER_WINDOWS,
        mask_prob=config.mask_prob,
        seed=record["mask_seed"],
    )
    assert all(b["x"].shape[0] == 8 for b in batches)

    def mean_ce(params, batch):
        state = compute_forward_pass(
            params, structure, batch, jax.random.PRNGKey(5), training=False
        )
        return compute_loss(
            state,
            batch["y"],
            structure.task_map["y"],
            "cross_entropy",
            loss_mask=batch["loss_mask"],
        )

    squares = [
        jax.tree_util.tree_map(np.square, jax.grad(mean_ce)(base.params, batch))
        for batch in batches
    ]
    expected = jax.tree_util.tree_map(
        lambda *values: np.mean(np.stack(values), axis=0), *squares
    )
    for mine, other in zip(
        jax.tree_util.tree_leaves(fisher),
        jax.tree_util.tree_leaves(expected),
        strict=True,
    ):
        np.testing.assert_allclose(mine, other, rtol=1e-4, atol=1e-10)


def test_fisher_is_non_negative_with_the_tree_of_the_params(setup, tmp_path):
    config, base, data_dir = _tiny_base(setup, tmp_path, **EWC)
    structure = fisher_structure(base.model_config, config)
    fisher, record = stage_fisher(
        base.params,
        _stream(config, base, data_dir, "wiki"),
        structure,
        make_squared_gradient_fn(structure),
        base.special_ids,
        config=config,
        stage=2,
        domain="wiki",
    )
    assert jax.tree_util.tree_structure(fisher) == jax.tree_util.tree_structure(
        base.params
    )
    leaves = jax.tree_util.tree_leaves(fisher)
    assert all(np.all(np.asarray(v) >= 0) for v in leaves)
    assert any(np.any(np.asarray(v) > 0) for v in leaves)
    assert params_hash(fisher) == record["fisher_sha256"]
    assert set(record["module_traces"]) == {
        name
        for name, node in base.params.nodes.items()
        if jax.tree_util.tree_leaves(node)
    }


def test_fisher_skips_batches_without_masked_tokens():
    params = toy_params({"a": [1.0]})
    calls = iter([(toy_params({"a": [4.0]}), 3.0), (toy_params({"a": [100.0]}), 0.0)])
    fisher, used = diagonal_fisher(params, [{}, {}], lambda p, b: next(calls))
    assert used == 1
    np.testing.assert_allclose(leaf(fisher), [4.0])
    with pytest.raises(ValueError, match="masked token"):
        diagonal_fisher(params, [{}], lambda p, b: (params, 0.0))


def test_fisher_ignores_dropout_and_the_run_random_state(setup, tmp_path):
    config, base, data_dir = _tiny_base(setup, tmp_path, **EWC)
    other = replace(config, dropout_rate=0.3, lr=5e-1, batch_size=2)
    stream = _stream(config, base, data_dir, "news")
    results = []
    for run in (config, other):
        structure = fisher_structure(base.model_config, run)
        fisher, _ = stage_fisher(
            base.params,
            stream,
            structure,
            make_squared_gradient_fn(structure),
            base.special_ids,
            config=run,
            stage=1,
            domain="news",
        )
        results.append(params_hash(fisher))
    assert results[0] == results[1]


def _zero_gradient_fn(params, batch, key, eta_infer=None):
    grads = jax.tree_util.tree_map(jnp.zeros_like, params)
    return grads, jnp.asarray(1.0), jnp.asarray(1.0), {}


@pytest.mark.parametrize("clip", [1e-3, 2.0])
def test_penalty_gradient_reaches_the_update_through_clip_and_adamw(
    setup, tmp_path, clip
):
    config, base, data_dir = _tiny_base(
        setup,
        tmp_path,
        retention="l2sp",
        retention_lambda=0.5,
        weight_decay=0.0,
        warmup_fraction=0.0,
        grad_clip=clip,
        dropout_rate=0.0,
        tokens_per_domain=4,
    )
    anchor = jax.tree_util.tree_map(lambda v: v + 1.0, base.params)
    params, _, log = train_stage(
        base.params,
        base.structure,
        _stream(replace(config, tokens_per_domain=60), base, data_dir, "news"),
        config=config,
        stage=2,
        special_ids=base.special_ids,
        gradient_fn=_zero_gradient_fn,
        retention_terms=[(anchor, None)],
        verbose=False,
    )
    assert len(log) == 1
    entry = log[0]
    assert entry["task_grad_norm"] == 0.0
    assert entry["retention_grad_norm"] == pytest.approx(entry["grad_norm"])
    count = sum(v.size for v in jax.tree_util.tree_leaves(base.params))
    assert entry["grad_norm"] == pytest.approx(0.5 * np.sqrt(count), rel=1e-3)
    assert entry["retention_penalty"] == pytest.approx(0.25 * count, rel=1e-3)
    assert entry["clipped"] == (entry["grad_norm"] > clip)
    for new, old in zip(
        jax.tree_util.tree_leaves(params),
        jax.tree_util.tree_leaves(base.params),
        strict=True,
    ):
        np.testing.assert_allclose(
            np.asarray(new) - np.asarray(old), config.lr, rtol=1e-2
        )


def test_retention_changes_the_clip_input_but_not_the_ce_gradient(setup, tmp_path):
    config, base, data_dir = _tiny_base(setup, tmp_path, **L2SP)
    stream = _stream(config, base, data_dir, "news")
    anchor = jax.tree_util.tree_map(lambda v: v + 0.1, base.params)
    gradient_fn = make_gradient_fn("backprop", base.structure)
    _, _, without = train_stage(
        base.params,
        base.structure,
        stream,
        config=config,
        stage=2,
        special_ids=base.special_ids,
        gradient_fn=gradient_fn,
        verbose=False,
    )
    _, _, with_term = train_stage(
        base.params,
        base.structure,
        stream,
        config=config,
        stage=2,
        special_ids=base.special_ids,
        gradient_fn=gradient_fn,
        retention_terms=[(anchor, None)],
        verbose=False,
    )
    assert with_term[0]["task_grad_norm"] == pytest.approx(without[0]["grad_norm"])
    assert with_term[0]["masked_loss"] == without[0]["masked_loss"]
    assert "retention_penalty" not in without[0]
    assert with_term[0]["grad_norm"] != pytest.approx(without[0]["grad_norm"])


def test_epc_inference_and_controller_do_not_see_the_penalty(setup, tmp_path):
    config, base, data_dir = _tiny_base(setup, tmp_path, **EPC, **L2SP)
    stream = _stream(config, base, data_dir, "news")
    gradient_fn = make_gradient_fn("pc", base.structure, "none", "epc")
    anchor = jax.tree_util.tree_map(lambda v: v + 1.0, base.params)
    logs = [
        train_stage(
            base.params,
            base.structure,
            stream,
            config=config,
            stage=2,
            special_ids=base.special_ids,
            gradient_fn=gradient_fn,
            retention_terms=terms,
            verbose=False,
        )
        for terms in ((), [(anchor, None)])
    ]
    plain, penalised = logs[0][2][0], logs[1][2][0]
    for key in (
        "masked_loss",
        "pc_energy_start",
        "pc_energy_end",
        "pc_energy_rel_decrease",
        "eta_infer",
        "eta_next",
    ):
        assert plain[key] == penalised[key]
    assert penalised["task_grad_norm"] == pytest.approx(plain["grad_norm"])
    assert penalised["retention_grad_norm"] > 0
    assert params_hash(logs[0][0]) != params_hash(logs[1][0])


def test_anchors_do_not_alias_the_trained_params(setup, tmp_path):
    config, base, data_dir = _tiny_base(setup, tmp_path, **L2SP)
    anchor = base.params
    before = params_hash(anchor)
    params, _, _ = train_stage(
        base.params,
        base.structure,
        _stream(config, base, data_dir, "news"),
        config=config,
        stage=2,
        special_ids=base.special_ids,
        gradient_fn=make_gradient_fn("backprop", base.structure),
        retention_terms=[(anchor, None)],
        verbose=False,
    )
    assert params_hash(anchor) == before
    assert params_hash(params) != before


def test_stage_one_has_no_penalty_and_later_stages_log_it(setup, tmp_path, plain_run):
    plain_config, plain_result = plain_run
    checkpoint, data_dir = setup
    config = tiny_config(checkpoint, data_dir, tmp_path / "run", **L2SP)
    result = run_domain_forgetting(config, verbose=False)
    assert _hashes(result)[:2] == _hashes(plain_result)[:2]
    assert _hashes(result)[2] != _hashes(plain_result)[2]
    assert all("retention_penalty" not in e for e in _log(config, 1))
    for stage in (2, 3):
        entries = _log(config, stage)
        assert entries[0]["retention_penalty"] == 0.0
        assert entries[-1]["retention_penalty"] > 0.0
        assert all({"task_grad_norm", "retention_grad_norm"} <= set(e) for e in entries)
    exposure = json.loads((tmp_path / "run" / "exposure.json").read_text())
    assert [exposure[d]["retention"]["terms"] for d in DOMAINS] == [0, 1, 1]
    recorded = json.loads((tmp_path / "run" / "run_config.json").read_text())
    assert recorded["retention"]["method"] == "l2sp"
    assert recorded["config"]["retention_lambda"] == 0.1


def test_ewc_stages_use_separate_terms_of_each_finished_stage(
    setup, tmp_path, monkeypatch, ewc_run
):
    config, result = ewc_run
    checkpoint, data_dir = setup
    seen = {}
    original = runner.train_stage

    def recording(*args, stage, retention_terms=(), **kwargs):
        seen[stage] = [
            (params_hash(anchor), params_hash(fisher))
            for anchor, fisher in retention_terms
        ]
        return original(*args, stage=stage, retention_terms=retention_terms, **kwargs)

    monkeypatch.setattr(runner, "train_stage", recording)
    again = tiny_config(checkpoint, data_dir, tmp_path / "run", **EWC)
    repeat = run_domain_forgetting(again, verbose=False)
    assert _hashes(repeat) == _hashes(result)
    records = [
        json.loads(line)
        for line in (tmp_path / "run" / "retention.jsonl").read_text().splitlines()
    ]
    assert [r["stage"] for r in records] == [1, 2]
    assert [r["domain"] for r in records] == list(DOMAINS[:2])
    hashes = _hashes(result)
    assert seen[1] == []
    assert seen[2] == [(hashes[1], records[0]["fisher_sha256"])]
    assert seen[3] == [
        (hashes[1], records[0]["fisher_sha256"]),
        (hashes[2], records[1]["fisher_sha256"]),
    ]
    assert all(r["params_sha256"] == hashes[r["stage"]] for r in records)
    assert records[0]["fisher_sha256"] != records[1]["fisher_sha256"]


def test_fisher_of_an_epc_run_uses_its_weights_on_the_backprop_graph(setup, tmp_path):
    checkpoint, data_dir = setup
    config = tiny_config(
        checkpoint, data_dir, tmp_path / "run", **EPC, **EWC, pc_alignment_batches=1
    )
    result = run_domain_forgetting(config, verbose=False)
    records = [
        json.loads(line)
        for line in (tmp_path / "run" / "retention.jsonl").read_text().splitlines()
    ]
    assert [r["params_sha256"] for r in records] == _hashes(result)[1:3]
    assert all(_log(config, s)[-1]["retention_penalty"] > 0 for s in (2, 3))
    assert all("pc_energy_end" in _log(config, s)[-1] for s in (2, 3))


@pytest.mark.parametrize(
    "learner,method",
    [
        ({}, "none"),
        ({}, "l2sp"),
        ({}, "ewc"),
        (EPC, "none"),
        (EPC, "l2sp"),
        (EPC, "ewc"),
    ],
)
def test_lambda_zero_is_bit_identical(setup, tmp_path, learner, method):
    checkpoint, data_dir = setup
    plain = run_domain_forgetting(
        tiny_config(checkpoint, data_dir, tmp_path / "plain", **learner), verbose=False
    )
    off = run_domain_forgetting(
        tiny_config(
            checkpoint,
            data_dir,
            tmp_path / "off",
            **learner,
            retention=method,
            retention_lambda=0.0,
        ),
        verbose=False,
    )
    assert _hashes(off) == _hashes(plain)
    assert not (tmp_path / "off" / "retention.jsonl").exists()


@pytest.mark.parametrize(
    "learner,method",
    [({}, "l2sp"), ({}, "ewc"), (EPC, "l2sp"), (EPC, "ewc")],
)
def test_negligible_penalty_runs_the_penalty_code_and_is_bit_identical(
    setup, tmp_path, learner, method
):
    checkpoint, data_dir = setup
    plain = run_domain_forgetting(
        tiny_config(checkpoint, data_dir, tmp_path / "plain", **learner), verbose=False
    )
    config = tiny_config(
        checkpoint,
        data_dir,
        tmp_path / "negligible",
        **learner,
        retention=method,
        retention_lambda=1e-30,
    )
    negligible = run_domain_forgetting(config, verbose=False)
    assert _hashes(negligible) == _hashes(plain)
    assert all("retention_penalty" in e for s in (2, 3) for e in _log(config, s))


@pytest.mark.parametrize("learner", [{}, EPC])
def test_uniform_all_scales_every_module_from_stage_two(setup, tmp_path, learner):
    checkpoint, data_dir = setup
    plain = tiny_config(checkpoint, data_dir, tmp_path / "plain", **learner)
    scaled = tiny_config(
        checkpoint,
        data_dir,
        tmp_path / "scaled",
        **learner,
        gate="uniform_all",
        gate_scale=0.5,
    )
    plain_hashes = _hashes(run_domain_forgetting(plain, verbose=False))
    scaled_hashes = _hashes(run_domain_forgetting(scaled, verbose=False))
    assert scaled_hashes[0] == plain_hashes[0]
    assert scaled_hashes[1] == plain_hashes[1]
    assert scaled_hashes[2] != plain_hashes[2]
    for name, value in _log(plain, 1)[0]["module_update_norms"].items():
        assert _log(scaled, 1)[0]["module_update_norms"][name] == pytest.approx(value)
    first = _log(plain, 2)[0]["module_update_norms"]
    assert len(first) > 1
    for name, value in first.items():
        assert _log(scaled, 2)[0]["module_update_norms"][name] == pytest.approx(
            0.5 * value
        )


def test_uniform_all_has_its_own_label_and_keeps_its_scale(setup, tmp_path):
    checkpoint, data_dir = setup
    config = tiny_config(
        checkpoint, data_dir, tmp_path / "run", gate="uniform_all", gate_scale=0.4
    )
    assert config.identity()["gate_scale"] == 0.4
    label = describe_gate({"gate": "uniform_all", "gate_scale": 0.4, "lr": 3e-5})[2]
    assert label == "uniform_all_s0.4@lr3e-05"


def test_off_settings_stay_out_of_the_identity(setup, tmp_path):
    checkpoint, data_dir = setup
    identity = tiny_config(checkpoint, data_dir, tmp_path / "run").identity()
    assert not {"retention", "retention_lambda", "keep_optimizer_state"} & set(identity)
    on = tiny_config(
        checkpoint, data_dir, tmp_path / "run", keep_optimizer_state=True, **EWC
    ).identity()
    assert on["retention"] == "ewc"
    assert on["retention_lambda"] == 10.0
    assert on["keep_optimizer_state"] is True


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"retention": "si"}, "retention must be one of"),
        ({"retention": "l2sp", "retention_lambda": -1.0}, "non-negative"),
        ({"retention": "l2sp", "retention_lambda": float("nan")}, "non-negative"),
        ({"retention_lambda": 1.0}, "retention_lambda needs"),
        ({**L2SP, "gate": "uniform", "gate_scale": 0.5}, "do not combine with gate"),
        ({"keep_optimizer_state": True, "gate": "uniform"}, "do not combine with gate"),
        ({**L2SP, "training": "joint"}, "sequential"),
        ({**L2SP, "pc_error_norm_penalty": 0.1}, "pc_error_norm_penalty"),
        ({**L2SP, "pc_mupc": "full"}, "pc_mupc"),
        ({**L2SP, "grad_noise_cosine": 0.5}, "grad_noise_cosine"),
        ({**EPC, **L2SP, "pc_output_energy_scale": 0.5}, "pc_output_energy_scale"),
    ],
)
def test_validation(overrides, message, tmp_path):
    config = tiny_config(tmp_path / "c", tmp_path / "d", tmp_path / "r", **overrides)
    with pytest.raises(ValueError, match=message):
        config.validate()


def test_cli_flags(tmp_path):
    args = cli.build_parser().parse_args(
        [
            "--checkpoint",
            "c",
            "--domains",
            "news",
            "wiki",
            "--output_dir",
            str(tmp_path),
            "--retention",
            "ewc",
            "--retention_lambda",
            "1e2",
            "--keep_optimizer_state",
        ]
    )
    config = cli.config_from_args(args)
    assert config.retention == "ewc"
    assert config.retention_lambda == 100.0
    assert config.keep_optimizer_state is True


def test_adam_step_from_carried_moments_known_answer():
    config = RunConfig(
        checkpoint="",
        data_dir="",
        domains=("news", "wiki"),
        output_dir="",
        lr=0.1,
        weight_decay=0.0,
        warmup_fraction=0.0,
        min_lr_ratio=1.0,
        grad_clip=0.0,
    )
    optimizer, _ = build_stage_optimizer(config, 10)
    params = toy_params({"a": [1.0]})
    count, mu, nu = 40, 0.2, 0.05
    carried = optax.ScaleByAdamState(
        count=jnp.asarray(count, jnp.int32),
        mu=toy_params({"a": [mu]}),
        nu=toy_params({"a": [nu]}),
    )
    state = with_adam_moments(optimizer.init(params), carried)
    updates, state = optimizer.update(toy_params({"a": [0.5]}), state, params)
    new_mu = 0.9 * mu + 0.1 * 0.5
    new_nu = 0.999 * nu + 0.001 * 0.25
    t = count + 1
    expected = (
        -0.1
        * (new_mu / (1 - 0.9**t))
        / (np.sqrt(new_nu / (1 - 0.999**t)) + config.adam_eps)
    )
    np.testing.assert_allclose(leaf(updates), [expected], rtol=1e-5)
    assert int(adam_moments(state).count) == t


def test_carry_keeps_adam_count_and_restarts_the_schedule_count(setup, tmp_path):
    config, base, data_dir = _tiny_base(setup, tmp_path, keep_optimizer_state=True)
    stream = _stream(config, base, data_dir, "news")
    gradient_fn = make_gradient_fn("backprop", base.structure)
    carry = OptimizerCarry()
    states = {}

    def stage_gate(stage):
        def record(update, params, opt_state):
            states[(stage, update)] = opt_state
            return opt_state

        return record

    params = base.params
    updates = 3
    expected_mu = None
    for stage in (1, 2):
        params, exposure, _ = train_stage(
            params,
            base.structure,
            stream,
            config=config,
            stage=stage,
            special_ids=base.special_ids,
            gradient_fn=gradient_fn,
            gate_fn=stage_gate(stage),
            carry=carry,
            verbose=False,
        )
        start = states[(stage, 0)]
        schedule_counts = [
            int(s.count)
            for s in jax.tree_util.tree_leaves(
                start, is_leaf=lambda x: isinstance(x, optax.ScaleByScheduleState)
            )
            if isinstance(s, optax.ScaleByScheduleState)
        ]
        assert schedule_counts == [0]
        assert int(adam_moments(start).count) == (stage - 1) * updates
        assert exposure["optimizer"]["adam_count_start"] == (stage - 1) * updates
        assert exposure["optimizer"]["adam_count_end"] == stage * updates
        if expected_mu is not None:
            assert params_hash(adam_moments(start).mu) == expected_mu
        expected_mu = params_hash(adam_moments(states[(stage, updates)]).mu)
        end_schedule = [
            int(s.count)
            for s in jax.tree_util.tree_leaves(
                states[(stage, updates)],
                is_leaf=lambda x: isinstance(x, optax.ScaleByScheduleState),
            )
            if isinstance(s, optax.ScaleByScheduleState)
        ]
        assert end_schedule == [updates]


def test_keep_optimizer_state_changes_only_later_stages(keep_run, plain_run):
    keep_config, kept = keep_run
    _, plain = plain_run
    assert _hashes(kept)[:2] == _hashes(plain)[:2]
    assert _hashes(kept)[2] != _hashes(plain)[2]
    output = Path(keep_config.output_dir)
    assert {p.name for p in output.glob("optimizer_state_stage_*.npz")} == {
        f"optimizer_state_stage_{s}.npz" for s in (1, 2, 3)
    }
    recorded = json.loads((output / "run_config.json").read_text())
    assert recorded["config"]["keep_optimizer_state"] is True
    assert recorded["optimizer_state"]["restarted"] == ["schedule count"]


def test_keep_false_writes_no_optimizer_state(plain_run):
    config, _ = plain_run
    assert not list(Path(config.output_dir).glob("optimizer_state_stage_*.npz"))


def test_moment_files_round_trip(setup, tmp_path):
    config, base, _ = _tiny_base(setup, tmp_path, keep_optimizer_state=True)
    moments = adam_moments(optax.adam(1e-3).init(base.params))
    moments = moments._replace(
        count=jnp.asarray(7, jnp.int32),
        mu=jax.tree_util.tree_map(lambda v: v + 1.0, moments.mu),
    )
    path = tmp_path / "moments.npz"
    save_adam_moments(path, moments)
    loaded = load_adam_moments(path, base.structure)
    assert int(loaded.count) == 7
    assert loaded.count.dtype == moments.count.dtype
    assert params_hash(loaded.mu) == params_hash(moments.mu)
    assert params_hash(loaded.nu) == params_hash(moments.nu)


class _Crash(RuntimeError):
    pass


def _crash_at_stage(monkeypatch, crash_stage):
    original = runner.train_stage

    def crashing(*args, stage, **kwargs):
        if stage == crash_stage:
            raise _Crash
        return original(*args, stage=stage, **kwargs)

    monkeypatch.setattr(runner, "train_stage", crashing)


@pytest.mark.parametrize("crash_stage", [2, 3])
def test_resumed_ewc_run_matches_an_uninterrupted_one(
    setup, tmp_path, monkeypatch, ewc_run, crash_stage
):
    full_config, full = ewc_run
    checkpoint, data_dir = setup
    config = tiny_config(checkpoint, data_dir, tmp_path / "run", **EWC)
    _crash_at_stage(monkeypatch, crash_stage)
    with pytest.raises(_Crash):
        run_domain_forgetting(config, verbose=False)
    monkeypatch.undo()
    resumed = run_domain_forgetting(config, resume=True, verbose=False)
    assert _hashes(resumed) == _hashes(full)
    assert resumed["matrices"] == full["matrices"]
    assert (tmp_path / "run" / "retention.jsonl").read_text().count("\n") == 2


def test_resumed_epc_ewc_run_matches_an_uninterrupted_one(setup, tmp_path, monkeypatch):
    checkpoint, data_dir = setup
    full = run_domain_forgetting(
        tiny_config(checkpoint, data_dir, tmp_path / "full", **EPC, **EWC),
        verbose=False,
    )
    config = tiny_config(checkpoint, data_dir, tmp_path / "run", **EPC, **EWC)
    _crash_at_stage(monkeypatch, 3)
    with pytest.raises(_Crash):
        run_domain_forgetting(config, verbose=False)
    monkeypatch.undo()
    resumed = run_domain_forgetting(config, resume=True, verbose=False)
    assert _hashes(resumed) == _hashes(full)


def test_resume_of_a_finished_ewc_run_computes_no_fisher(
    setup, tmp_path, monkeypatch, ewc_run
):
    full_config, full = ewc_run
    copy = tmp_path / "run"
    shutil.copytree(full_config.output_dir, copy)
    config = replace(full_config, output_dir=str(copy))
    before = (copy / "retention.jsonl").read_text()
    import fabricpc.experiments.domain_forgetting.retention as retention

    def forbidden(*args, **kwargs):
        raise AssertionError("no stage trains, so no Fisher is needed")

    monkeypatch.setattr(retention, "stage_fisher", forbidden)
    resumed = run_domain_forgetting(config, resume=True, verbose=False)
    assert _hashes(resumed) == _hashes(full)
    assert (copy / "retention.jsonl").read_text() == before


def test_resume_rejects_a_fisher_that_differs_from_its_record(setup, tmp_path, ewc_run):
    full_config, _ = ewc_run
    copy = tmp_path / "run"
    shutil.copytree(full_config.output_dir, copy)
    records = [
        json.loads(line) for line in (copy / "retention.jsonl").read_text().splitlines()
    ]
    records[0]["fisher_sha256"] = "0" * 64
    (copy / "retention.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n"
    )
    (copy / "stage_3.npz").unlink()
    config = replace(full_config, output_dir=str(copy))
    with pytest.raises(ValueError, match="recomputed Fisher"):
        run_domain_forgetting(config, resume=True, verbose=False)


def test_resumed_keep_optimizer_run_matches_an_uninterrupted_one(
    setup, tmp_path, monkeypatch, keep_run
):
    _, full = keep_run
    checkpoint, data_dir = setup
    config = tiny_config(
        checkpoint, data_dir, tmp_path / "run", keep_optimizer_state=True
    )
    _crash_at_stage(monkeypatch, 3)
    with pytest.raises(_Crash):
        run_domain_forgetting(config, verbose=False)
    monkeypatch.undo()
    resumed = run_domain_forgetting(config, resume=True, verbose=False)
    assert _hashes(resumed) == _hashes(full)


def test_missing_moments_retrain_the_stage_and_match(
    setup, tmp_path, monkeypatch, keep_run
):
    full_config, full = keep_run
    copy = tmp_path / "run"
    shutil.copytree(full_config.output_dir, copy)
    (copy / "optimizer_state_stage_2.npz").unlink()
    trained = []
    original = runner.train_stage

    def counting(*args, stage, **kwargs):
        trained.append(stage)
        return original(*args, stage=stage, **kwargs)

    monkeypatch.setattr(runner, "train_stage", counting)
    config = replace(full_config, output_dir=str(copy))
    resumed = run_domain_forgetting(config, resume=True, verbose=False)
    assert trained == [2, 3]
    assert _hashes(resumed) == _hashes(full)


def test_overwrite_removes_retention_artifacts(setup, tmp_path, keep_run):
    full_config, _ = keep_run
    copy = tmp_path / "run"
    shutil.copytree(full_config.output_dir, copy)
    (copy / "retention.jsonl").write_text("")
    removed = runner.remove_run_artifacts(copy)
    assert "retention.jsonl" in removed
    assert not list(copy.glob("optimizer_state_stage_*.npz"))


def test_labels_stay_out_of_the_reference_curve():
    base = {"lr": 3e-5, "mode": "backprop"}
    assert describe_gate({**base, "retention": "l2sp", "retention_lambda": 0.01})[
        2
    ] == ("lr3e-05_l2sp0.01")
    assert describe_gate({**base, "retention": "ewc", "retention_lambda": 1e6})[2] == (
        "lr3e-05_ewc1e+06"
    )
    assert describe_gate({**base, "keep_optimizer_state": True})[2] == "lr3e-05_keepopt"
    assert describe_gate({**base, "retention": "none"})[2] == "lr3e-05"
    labels = {
        describe_gate({**base, "retention": m, "retention_lambda": v})[2]
        for m, grid in RETENTION_GRIDS.items()
        for v in grid
    }
    assert len(labels) == 8


def _retention_run(root, name, *, x, y, seed, lr, domains, **config):
    path = _write_run(root, name, x=x, y=y, seed=seed, lr=lr, domains=domains)
    record = json.loads((path / "run_config.json").read_text())
    record["config"].update(config)
    (path / "run_config.json").write_text(json.dumps(record))
    return path


@pytest.mark.parametrize(
    "config",
    [
        {"retention": "l2sp", "retention_lambda": 0.1},
        {"retention": "ewc", "retention_lambda": 1.0},
        {"retention": "ewc", "retention_lambda": 0.0},
        {"keep_optimizer_state": True},
    ],
)
def test_retention_runs_never_enter_the_curve(tmp_path, config):
    path = _retention_run(
        tmp_path, "run", x=0.3, y=3.0, seed=0, lr=3e-5, domains=DOMAINS, **config
    )
    point = load_run_point(path)
    assert point["intervention"]
    assert not _is_reference(point)
    plain = _write_run(tmp_path, "ref", x=0.25, y=3.2, seed=0, lr=3e-5, domains=DOMAINS)
    result = compute_frontier(collect_runs([path, plain]), seq_lr=3e-5)
    curve = result["orders"][next(iter(result["orders"]))]["curves"]["0"]
    assert curve["x"] == [0.25]


def _pilot(tmp_path, values, *, reference=1.0):
    """``values`` maps (method, λ) to (gain, final CE) in both orders."""
    dirs = []
    for tag, domains in (("fwd", DOMAINS), ("rev", tuple(reversed(DOMAINS)))):
        dirs.append(
            _write_run(
                tmp_path,
                f"ref_{tag}",
                x=reference,
                y=3.0,
                seed=0,
                lr=3e-5,
                domains=domains,
            )
        )
        for (method, strength), (x, y) in values.items():
            dirs.append(
                _retention_run(
                    tmp_path,
                    f"{method}_{strength}_{tag}",
                    x=x,
                    y=y,
                    seed=0,
                    lr=3e-5,
                    domains=domains,
                    retention=method,
                    retention_lambda=strength,
                )
            )
    return collect_runs(dirs)


def _grid(method, rows):
    return {(method, v): row for v, row in zip(RETENTION_GRIDS[method], rows)}


def test_selection_lowest_final_ce_among_strengths_at_90_percent(tmp_path):
    values = _grid("l2sp", [(1.0, 3.1), (0.95, 3.0), (0.9, 2.9), (0.89, 2.5)])
    selection = select_retention(_pilot(tmp_path, values), 3e-5)
    entry = selection["methods"]["l2sp"]
    assert selection["threshold"] == pytest.approx(0.9)
    assert [row["qualifies"] for row in entry["strengths"]] == [True, True, True, False]
    assert entry["selected"] == RETENTION_GRIDS["l2sp"][2]
    assert entry["status"] == "selected"
    assert not entry["endpoint"]
    assert entry["tied"] == []
    assert "ewc" not in selection["methods"]
    assert "selected" in format_selection(selection)


def test_selection_flags_endpoints(tmp_path):
    values = _grid("ewc", [(1.0, 2.0), (1.0, 3.0), (1.0, 3.1), (0.5, 3.2)])
    entry = select_retention(_pilot(tmp_path, values), 3e-5)["methods"]["ewc"]
    assert entry["selected"] == 1.0
    assert entry["endpoint"]
    assert "ENDPOINT" in format_selection(
        {
            "reference_gain": 1.0,
            "threshold": 0.9,
            "seed": 0,
            "methods": {"ewc": entry},
        }
    )
    (tmp_path / "top").mkdir()
    values = _grid("ewc", [(1.0, 3.2), (1.0, 3.1), (1.0, 3.0), (1.0, 2.0)])
    entry = select_retention(_pilot(tmp_path / "top", values), 3e-5)["methods"]["ewc"]
    assert entry["selected"] == 1e6
    assert entry["endpoint"]


def test_selection_tie_takes_the_weaker_strength(tmp_path):
    values = _grid("l2sp", [(1.0, 3.1), (1.0, 3.0), (1.0, 3.0), (0.5, 2.0)])
    entry = select_retention(_pilot(tmp_path, values), 3e-5)["methods"]["l2sp"]
    assert entry["selected"] == RETENTION_GRIDS["l2sp"][1]
    assert entry["tied"] == list(RETENTION_GRIDS["l2sp"][1:3])


def test_selection_with_no_qualifying_strength(tmp_path):
    values = _grid("l2sp", [(0.1, 2.0), (0.2, 2.1), (0.3, 2.2), (0.89, 2.3)])
    selection = select_retention(_pilot(tmp_path, values), 3e-5)
    entry = selection["methods"]["l2sp"]
    assert entry["status"] == "no_qualifying_strength"
    assert entry["selected"] is None
    assert not entry["endpoint"]
    assert "NO QUALIFYING" in format_selection(selection)


def test_selection_averages_gain_over_orders_not_per_order(tmp_path):
    dirs = []
    for tag, domains, gain in (
        ("fwd", DOMAINS, 0.7),
        ("rev", tuple(reversed(DOMAINS)), 1.2),
    ):
        dirs.append(
            _write_run(
                tmp_path, f"ref_{tag}", x=1.0, y=3.0, seed=0, lr=3e-5, domains=domains
            )
        )
        for strength in RETENTION_GRIDS["l2sp"]:
            dirs.append(
                _retention_run(
                    tmp_path,
                    f"l2sp_{strength}_{tag}",
                    x=gain,
                    y=3.0 - strength / 10,
                    seed=0,
                    lr=3e-5,
                    domains=domains,
                    retention="l2sp",
                    retention_lambda=strength,
                )
            )
    entry = select_retention(collect_runs(dirs), 3e-5)["methods"]["l2sp"]
    assert all(row["qualifies"] for row in entry["strengths"])


def test_selection_ignores_other_seeds_and_lrs_and_rejects_gaps(tmp_path):
    values = _grid("l2sp", [(1.0, 3.1), (1.0, 3.0), (1.0, 2.9), (1.0, 2.95)])
    points = _pilot(tmp_path, values)
    other = _write_run(tmp_path, "s1", x=9.0, y=1.0, seed=1, lr=3e-5, domains=DOMAINS)
    lr = _write_run(tmp_path, "lr", x=9.0, y=1.0, seed=0, lr=1e-4, domains=DOMAINS)
    extended = points + collect_runs([other, lr])
    assert select_retention(extended, 3e-5)["reference_gain"] == 1.0
    partial = [p for p in points if not p["run_dir"].endswith("l2sp_1.0_fwd")]
    with pytest.raises(ValueError, match="one pilot run per order"):
        select_retention(partial, 3e-5)
    with pytest.raises(ValueError, match="no ungated run"):
        select_retention([p for p in points if p["retention"] != "none"], 3e-5)


def test_selection_diverged_strength_does_not_qualify(tmp_path):
    values = _grid("l2sp", [(1.0, 3.1), (1.0, 3.0), (1.0, 2.9), (1.0, 2.8)])
    points = _pilot(tmp_path, values)
    for point in points:
        if point["retention_lambda"] == 1.0:
            point.update(diverged=True, x=None, y=None)
    entry = select_retention(points, 3e-5)["methods"]["l2sp"]
    assert entry["selected"] == 0.1
    assert entry["strengths"][-1]["diverged"]


def _baseline_runs(tmp_path, *, baseline_y=3.40, matched_y=3.60, off_curve=False):
    dirs = []
    orders = {"fwd": DOMAINS, "rev": tuple(reversed(DOMAINS))}
    for tag, domains in orders.items():
        for seed in range(3):
            jitter = seed * 1e-3
            for lr, (x, y) in {
                1e-5: (0.25, 3.3),
                3e-5: (0.27, 3.7),
                1e-4: (0.30, 3.45),
            }.items():
                dirs.append(
                    _write_run(
                        tmp_path,
                        f"ref_{lr}_{tag}_{seed}",
                        x=x,
                        y=y + jitter,
                        seed=seed,
                        lr=lr,
                        domains=domains,
                    )
                )
            dirs.append(
                _retention_run(
                    tmp_path,
                    f"base_{tag}_{seed}",
                    x=0.5 if off_curve else 0.27,
                    y=baseline_y + jitter,
                    seed=seed,
                    lr=3e-5,
                    domains=domains,
                    retention="ewc",
                    retention_lambda=100.0,
                )
            )
            dirs.append(
                _write_run(
                    tmp_path,
                    f"match_{tag}_{seed}",
                    x=0.27,
                    y=matched_y + jitter,
                    seed=seed,
                    lr=3e-5,
                    domains=domains,
                    gate="uniform",
                    scale=0.48,
                )
            )
    return dirs


LABELS = ["lr3e-05_ewc100", "uniform_s0.48@lr3e-05"]


def test_baseline_verdict_passes_without_a_surprise_comparison(tmp_path):
    result = compute_frontier(
        collect_runs(_baseline_runs(tmp_path)),
        seq_lr=3e-5,
        baseline_verdicts=[LABELS],
        confirmation_seeds=[1, 2],
    )
    baseline = result["baseline_verdicts"][0]
    for scope, seeds in (("all_seeds", [0, 1, 2]), ("confirmation_seeds", [1, 2])):
        verdict = baseline[scope]
        assert verdict["missing_conditions"] == []
        assert verdict["surprise"] is None
        assert not verdict["descriptive_only"]
        for name in ("pre_registered", "envelope_post_hoc"):
            entry = verdict[name]
            assert set(entry["passes"]) == {"frontier", "vs_matched"}
            assert set(entry["status"]) == {"frontier", "vs_matched"}
            for rows in entry["orders"].values():
                assert rows["seeds"] == seeds
                assert "vs_surprise" not in rows
                assert len(rows["frontier"]["values"]) == len(seeds)
        assert verdict["pre_registered"]["pass"] is True
        assert verdict["pass_both"] is (
            verdict["pre_registered"]["pass"] and verdict["envelope_post_hoc"]["pass"]
        )
    assert (
        baseline["confirmation_seeds"]["pre_registered"]["status"]["frontier"] == "pass"
    )
    seq_sd = result["orders"]["news>reviews>wiki"]["seq"]["y"]["sd"]
    row = baseline["confirmation_seeds"]["pre_registered"]["orders"][
        "news>reviews>wiki"
    ]["frontier"]
    assert row["seq_y_sd"] == seq_sd
    table = format_table(result)
    assert "baseline lr3e-05_ewc100" in table
    assert "vs_surprise" not in table
    assert "confirmation_seeds" in table


def test_baseline_verdict_with_no_matched_runs_is_undefined(tmp_path):
    result = compute_frontier(
        collect_runs(_baseline_runs(tmp_path)),
        seq_lr=3e-5,
        baseline_verdicts=[[LABELS[0], "no_matched_control"]],
    )
    verdict = result["baseline_verdicts"][0]["all_seeds"]
    for name in ("pre_registered", "envelope_post_hoc"):
        assert verdict[name]["status"]["vs_matched"] == "undefined"
        assert not verdict[name]["pass"]
    assert verdict["pass_both"] is False
    assert "pass under both: False" in format_table(result)


def test_baseline_verdict_reports_fail_undefined_and_paired_values(tmp_path):
    (tmp_path / "fail").mkdir()
    failed = compute_frontier(
        collect_runs(_baseline_runs(tmp_path / "fail", matched_y=3.30)),
        seq_lr=3e-5,
        baseline_verdicts=[LABELS],
    )["baseline_verdicts"][0]
    assert failed["confirmation_seeds"] is None
    entry = failed["all_seeds"]["pre_registered"]
    assert entry["status"] == {"frontier": "pass", "vs_matched": "fail"}
    assert entry["pass"] is False
    first = failed["paired"]["news>reviews>wiki"][0]
    assert first["paired_y"] == pytest.approx(3.40 - 3.30)
    assert first["paired_bwt"] == pytest.approx(0.0)
    assert first["path"] is None

    (tmp_path / "off").mkdir()
    off = compute_frontier(
        collect_runs(_baseline_runs(tmp_path / "off", off_curve=True)),
        seq_lr=3e-5,
        baseline_verdicts=[LABELS],
    )["baseline_verdicts"][0]
    entry = off["all_seeds"]["pre_registered"]
    assert entry["status"] == {"frontier": "undefined", "vs_matched": "undefined"}
    assert entry["pass"] is False
    assert off["paired"]["news>reviews>wiki"][0]["paired_y"] == pytest.approx(-0.2)


def test_baseline_verdict_reads_the_stage_two_on_path(tmp_path):
    dirs = _baseline_runs(tmp_path)
    for path, step in ((tmp_path / "base_fwd_0", 2.0), (tmp_path / "match_fwd_0", 1.0)):
        for stage in (1, 2, 3):
            entry = {
                "update_norm": 100.0 if stage == 1 else step,
                "module_update_norms": {},
            }
            (path / f"training_log_stage_{stage}.jsonl").write_text(
                json.dumps(entry) + "\n"
            )
    baseline = compute_frontier(
        collect_runs(dirs), seq_lr=3e-5, baseline_verdicts=[LABELS]
    )["baseline_verdicts"][0]
    row = baseline["paired"]["news>reviews>wiki"][0]
    assert row["path"] == 4.0
    assert row["path_matched"] == 2.0
    assert row["path_ratio"] == 2.0


def test_descriptive_baseline_is_marked_and_compares_with_the_restarted_runs(tmp_path):
    dirs = []
    for tag, domains in (("fwd", DOMAINS), ("rev", tuple(reversed(DOMAINS)))):
        for seed in range(3):
            dirs.append(
                _write_run(
                    tmp_path,
                    f"low_{tag}_{seed}",
                    x=0.2,
                    y=3.5,
                    seed=seed,
                    lr=1e-5,
                    domains=domains,
                )
            )
            dirs.append(
                _write_run(
                    tmp_path,
                    f"ref_{tag}_{seed}",
                    x=0.3,
                    y=3.2,
                    seed=seed,
                    lr=3e-5,
                    domains=domains,
                )
            )
            dirs.append(
                _retention_run(
                    tmp_path,
                    f"keep_{tag}_{seed}",
                    x=0.3,
                    y=3.1,
                    seed=seed,
                    lr=3e-5,
                    domains=domains,
                    keep_optimizer_state=True,
                )
            )
    result = compute_frontier(
        collect_runs(dirs),
        seq_lr=3e-5,
        descriptive_baselines=[["lr3e-05_keepopt", "lr3e-05"]],
    )
    verdict = result["baseline_verdicts"][0]
    assert verdict["descriptive_only"]
    assert verdict["all_seeds"]["descriptive_only"]
    assert verdict["all_seeds"]["missing_conditions"] == []
    assert verdict["paired"]["news>reviews>wiki"][0]["paired_y"] == pytest.approx(-0.1)
    assert "descriptive only" in format_table(result)


def test_influence_verdict_with_surprise_is_unchanged_in_shape(tmp_path):
    dirs = _baseline_runs(tmp_path)
    verdict = compute_frontier(
        collect_runs(dirs),
        seq_lr=3e-5,
        influence_verdicts=[[*LABELS, "absent"]],
    )["influence_verdicts"][0]
    assert verdict["missing_conditions"] == ["absent"]
    assert set(verdict["pre_registered"]["passes"]) == {
        "frontier",
        "vs_matched",
        "vs_surprise",
    }


def test_analysis_cli_runs_selection_and_a_baseline_verdict(tmp_path, capsys):
    values = {
        **_grid("l2sp", [(1.0, 3.1), (1.0, 3.0), (1.0, 2.9), (1.0, 2.95)]),
        **_grid("ewc", [(0.1, 3.1), (0.1, 3.0), (0.1, 2.9), (0.1, 2.95)]),
    }
    (tmp_path / "pilot").mkdir()
    points = _pilot(tmp_path / "pilot", values)
    output = tmp_path / "selection"
    analysis.main(
        [
            "retention_select",
            "--run_dirs",
            *(p["run_dir"] for p in points),
            "--seq_lr",
            "3e-5",
            "--output_dir",
            str(output),
        ]
    )
    selection = json.loads((output / "retention_selection.json").read_text())
    assert selection["methods"]["l2sp"]["selected"] == 0.1
    assert selection["methods"]["ewc"]["selected"] is None
    assert "NO QUALIFYING" in capsys.readouterr().out
    (tmp_path / "base").mkdir()
    analysis.main(
        [
            "frontier",
            "--run_dirs",
            *map(str, _baseline_runs(tmp_path / "base")),
            "--seq_lr",
            "3e-5",
            "--baseline_verdict",
            *LABELS,
            "--confirmation_seeds",
            "1",
            "2",
            "--output_dir",
            str(tmp_path / "frontier"),
        ]
    )
    frontier = json.loads((tmp_path / "frontier" / "frontier.json").read_text())
    assert frontier["baseline_verdicts"][0]["gate"] == LABELS[0]


def test_run_name_prefix_is_outside_the_existing_globs():
    globs = (
        "online_protect*_seed?",
        "online_surprise*_seed?",
        "online_matched_*_seed?",
        "g3_bp_t*",
        "g4_bp_seed?",
        "g8_*_seed?",
        "g9_*_seed?",
        "epcsuite_*_seed?",
        "epc_fwd_lr3e-5_seed?",
        "epc_rev_lr3e-5_seed?",
        "commutator_*_seed*",
        "infl_*",
        "infl_bp_*_seed?",
    )
    names = (
        "ret_bp_l2sp_l1e-3_fwd_seed0",
        "ret_epc_ewc_l1e6_rev_seed2",
        "ret_bp_ewc_matched_fwd_seed0",
        "ret_epc_keepopt_rev_seed1",
        "ret_bp_uniform_s0.125_fwd_seed0",
    )
    for name in names:
        assert not any(fnmatch.fnmatch(name, glob) for glob in globs)


def test_script_grids_are_the_selection_grids():
    text = (
        Path(__file__).resolve().parent.parent / "scripts/run_retention_baselines.sh"
    ).read_text()
    for method, variable in (("l2sp", "L2SP_GRID"), ("ewc", "EWC_GRID")):
        match = re.search(rf'^{variable}="([^"]+)"$', text, re.MULTILINE)
        assert match is not None
        assert (
            tuple(float(v) for v in match.group(1).split()) == RETENTION_GRIDS[method]
        )


def test_map_batches_of_the_fisher_are_fixed_by_their_seed(setup, tmp_path):
    config, base, data_dir = _tiny_base(setup, tmp_path, **EWC)
    stream = _stream(config, base, data_dir, "news")
    first, windows = spaced_batches(
        stream,
        base.structure,
        base.special_ids,
        num_windows=16,
        mask_prob=config.mask_prob,
        seed=3,
    )
    second, same = spaced_batches(
        stream,
        base.structure,
        base.special_ids,
        num_windows=16,
        mask_prob=config.mask_prob,
        seed=3,
    )
    np.testing.assert_array_equal(windows, same)
    assert len(windows) == 16
    for a, b in zip(first, second, strict=True):
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])


STUB_PYTHON = """#!/bin/sh
if [ "$1" = "-" ]; then
  code=$(cat)
  case "$code" in
    *retention_selection*|*"entry = json.loads"*) echo 1e-2 ;;
    *read_paths*) echo "ERROR out of range" >&2; exit 4 ;;
    *describe_gate*)
      [ -f "$2/run_config.json" ] || exit 1
      echo label ;;
    *) echo checked ;;
  esac
else
  echo "$@" >> "$STUB_CALLS"
fi
"""


def _run_script(tmp_path, step):
    stub = tmp_path / "stub_python"
    stub.write_text(STUB_PYTHON)
    stub.chmod(0o755)
    calls = tmp_path / "calls"
    env = {
        **os.environ,
        "PYTHON": str(stub),
        "ROOT": str(tmp_path),
        "STUB_CALLS": str(calls),
        "LEARNERS": "bp",
        "METHODS": "l2sp ewc",
    }
    script = Path(__file__).parents[1] / "scripts" / "run_retention_baselines.sh"
    done = subprocess.run(
        ["bash", str(script), step], env=env, capture_output=True, text=True
    )
    return done, calls.read_text() if calls.exists() else ""


def test_matched_step_finishes_when_a_path_is_out_of_range(tmp_path):
    done, calls = _run_script(tmp_path, "matched")
    assert done.returncode == 3, done.stderr
    assert done.stderr.count("ERROR out of range") == 2
    assert calls == ""


def test_verdict_step_runs_without_matched_controls(tmp_path):
    runs = [
        "ret_bp_l2sp_l1e-2_fwd_seed0",
        "ret_bp_ewc_l1e-2_fwd_seed0",
        "ret_bp_keepopt_fwd_seed0",
        "g9_fwd_lr3e-5_seed0",
    ]
    for run in runs:
        (tmp_path / run).mkdir()
        (tmp_path / run / "run_config.json").write_text("{}")
    done, calls = _run_script(tmp_path, "verdict")
    assert done.returncode == 0, done.stderr
    lines = calls.splitlines()
    assert len(lines) == 3
    assert all("--baseline_verdict label no_matched_control" in x for x in lines[:2])
    assert "--descriptive_baseline_verdict label label" in lines[2]
