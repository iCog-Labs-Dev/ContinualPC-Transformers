"""Tests for the error-norm penalty and the per-module update scale (tiny BERT)."""

import json
import os
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fabricpc.core.inference_epc import EPCInference
from fabricpc.experiments.domain_forgetting import runner
from fabricpc.experiments.domain_forgetting.data import load_domain_cache
from fabricpc.experiments.domain_forgetting.error_maps import bp_signal_energies
from fabricpc.experiments.domain_forgetting.frontier import (
    _is_reference,
    collect_runs,
    compute_frontier,
    describe_gate,
)
from fabricpc.experiments.domain_forgetting.localization import load_run
from fabricpc.experiments.domain_forgetting.runner import (
    BLOCK_MODULE_PATTERN,
    accumulate_gradients,
    build_stage_optimizer,
    calibrate_error_norm_penalty,
    check_comparable_runs,
    error_norm_penalty_gradient,
    fixed_mlm_batches,
    module_grad_norms,
    flatten_params,
    load_base_model,
    make_error_norm_penalty_fn,
    make_gradient_fn,
    make_regime_probe,
    run_domain_forgetting,
    train_stage,
)
from fabricpc.graph_initialization import initialize_params
from fabricpc.nodes.encoder_transformer import create_encoder_mlm_model
from fabricpc.training.train_mlm import _prepare_pcn_mlm_batch
from tests.test_domain_forgetting_data import build_tiny_caches, tiny_tokenizer
from tests.test_domain_forgetting_error_maps import SCALE, SEQ_LEN, VOCAB, _batch
from tests.test_domain_forgetting_frontier import write_references, write_run
from tests.test_domain_forgetting_runner import (
    DOMAINS,
    tiny_config,
    write_tiny_bert_checkpoint,
)

EPC = {
    "mode": "pc",
    "pc_inference": "epc",
    "pc_eta_infer": 0.01,
    "pc_infer_steps": 3,
    "pc_forward_updates": "none",
}
MODES = {"pc": EPC, "backprop": {}}
SCALED = ("embeddings", "L0_mha", "L1_mha")
SCALED_MODULES = ["token_embeddings", "position_embeddings", "L0_mha", "L1_mha"]


def _tiny_bert(dropout_rate: float):
    return create_encoder_mlm_model(
        depth=2,
        embed_dim=16,
        num_heads=4,
        mlp_dim=32,
        seq_len=SEQ_LEN,
        vocab_size=VOCAB,
        inference=EPCInference(eta_infer=0.01, infer_steps=1),
        use_attention_mask=True,
        dropout_rate=dropout_rate,
        pc_output_energy_scale=SCALE,
        transformer_init_std=0.3,
        architecture="bert",
        residual_identity=True,
    )


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    directory = tmp_path_factory.mktemp("intervention")
    tokenizer = tiny_tokenizer(directory)
    checkpoint = write_tiny_bert_checkpoint(directory, tokenizer, depth=2)
    build_tiny_caches(directory, tokenizer)
    return checkpoint, directory / "data"


@pytest.fixture(scope="module")
def runs(setup, tmp_path_factory):
    checkpoint, data_dir = setup
    root = tmp_path_factory.mktemp("intervention_runs")
    settings = {
        "baseline": {},
        "zero": {"pc_error_norm_penalty": 0.0},
        "unit": {"update_scale_modules": SCALED, "update_scale": 1.0},
        "empty": {"update_scale_modules": (), "update_scale": 1.0},
        "penalty": {"pc_error_norm_penalty": 1e-2},
        "scaled": {"update_scale_modules": SCALED, "update_scale": 0.5},
    }
    probed = {
        "pc_regime_interval": 2,
        "pc_regime_eta": 0.01,
        "pc_regime_steps": 3,
        "pc_regime_iters": 4,
        "pc_regime_final": True,
    }
    result = {}
    for mode, extra in MODES.items():
        for name, values in {
            **settings,
            **({"probed": probed} if mode == "backprop" else {}),
        }.items():
            config = tiny_config(
                checkpoint, data_dir, root / f"{mode}_{name}", **extra, **values
            )
            result[mode, name] = (config, run_domain_forgetting(config, verbose=False))
    return result


def _hashes(result):
    return result["summary"]["stage_params_sha256"]


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def test_penalty_is_the_squared_backpropagated_error_at_zero_errors():
    """P equals ‖∇_ε E‖² of the ePC energy, BP's error signal and one ePC step."""
    with jax.enable_x64(True):
        structure = _tiny_bert(0.0)
        params = initialize_params(structure, jax.random.PRNGKey(0))
        batch = _batch()
        key = jax.random.PRNGKey(1)
        value, _ = make_error_norm_penalty_fn(structure)(params, batch, key)
        spectrum, _ = make_regime_probe(structure, 4)(params, batch, key)
        assert float(value) > 0
        assert float(value) == pytest.approx(
            float(spectrum.gradient_norm) ** 2, rel=1e-10
        )
        count = float(np.sum(np.asarray(batch["loss_mask"])))
        signal = bp_signal_energies(params, structure, batch, SCALE)
        independent = 2 * count**2 * sum(float(jnp.sum(v)) for v in signal.values())
        assert float(value) == pytest.approx(independent, rel=1e-6)
        _shared, _metrics, _free, initial, clamps = _prepare_pcn_mlm_batch(
            params, batch, structure, key
        )
        eta = 1e-3
        final = EPCInference.run_inference(
            params, initial, clamps, structure, eta_infer=eta
        )
        relaxed = EPCInference._relaxed_errors(structure, clamps)
        assert structure.task_map["y"] not in relaxed
        stepped = sum(
            float(jnp.sum(jnp.square(final.nodes[name].error))) for name in relaxed
        )
        assert stepped / eta**2 == pytest.approx(float(value), rel=1e-8)


def test_penalty_gradient_matches_central_differences():
    with jax.enable_x64(True):
        structure = _tiny_bert(0.1)
        params = initialize_params(structure, jax.random.PRNGKey(0))
        params = jax.tree_util.tree_map(lambda x: jnp.asarray(x, jnp.float64), params)
        batch = _batch()
        key = jax.random.PRNGKey(2)
        penalty_fn = make_error_norm_penalty_fn(structure)
        _, grads = penalty_fn(params, batch, key)
        leaves, treedef = jax.tree_util.tree_flatten(params)
        rng = np.random.default_rng(0)
        step = 1e-5
        for _ in range(3):
            direction = jax.tree_util.tree_unflatten(
                treedef, [jnp.asarray(rng.standard_normal(x.shape)) for x in leaves]
            )
            shifted = [
                jax.tree_util.tree_map(
                    lambda p, d: p + sign * step * d, params, direction
                )
                for sign in (1.0, -1.0)
            ]
            plus, minus = (float(penalty_fn(p, batch, key)[0]) for p in shifted)
            numeric = (plus - minus) / (2 * step)
            analytic = sum(
                float(jnp.vdot(g, d))
                for g, d in zip(
                    jax.tree_util.tree_leaves(grads),
                    jax.tree_util.tree_leaves(direction),
                )
            )
            assert analytic == pytest.approx(numeric, rel=1e-6)


@pytest.mark.parametrize("mode", list(MODES))
def test_zero_penalty_and_unit_scale_reproduce_the_baseline(runs, mode):
    baseline = _hashes(runs[mode, "baseline"][1])
    for name in ("zero", "unit", "empty"):
        assert _hashes(runs[mode, name][1]) == baseline, name
    zero_config = runs[mode, "zero"][0]
    log = _read_jsonl(Path(zero_config.output_dir) / "training_log_stage_1.jsonl")
    assert "error_norm_penalty" not in log[0]
    unit_config = runs[mode, "unit"][0]
    run_config = json.loads(
        (Path(unit_config.output_dir) / "run_config.json").read_text()
    )
    assert run_config["update_scale"]["modules"] == SCALED_MODULES
    exposure = json.loads((Path(unit_config.output_dir) / "exposure.json").read_text())
    assert all("update_scale" not in e["optimizer"] for e in exposure.values())


def test_zero_penalty_never_builds_the_penalty(setup, tmp_path, monkeypatch):
    checkpoint, data_dir = setup

    def forbidden(*args, **kwargs):
        raise AssertionError("penalty computed at weight 0")

    monkeypatch.setattr(runner, "make_error_norm_penalty_fn", forbidden)
    monkeypatch.setattr(runner, "error_norm_penalty_gradient", forbidden)
    config = tiny_config(checkpoint, data_dir, tmp_path / "run", **EPC)
    run_domain_forgetting(config, verbose=False)


def _stage_inputs(setup, tmp_path, mode, **values):
    checkpoint, data_dir = setup
    config = tiny_config(
        checkpoint,
        data_dir,
        tmp_path / "run",
        tokens_per_domain=20,
        **MODES[mode],
        **values,
    )
    base = load_base_model(config)
    stream = load_domain_cache(
        data_dir,
        DOMAINS[0],
        tokenizer_json=base.tokenizer_json,
        seq_len=int(base.model_config["seq_len"]),
        tokens_per_domain=config.tokens_per_domain,
    ).train
    gradient_fn = make_gradient_fn(
        config.mode, base.structure, config.pc_forward_updates, config.pc_inference
    )
    return config, base, stream, gradient_fn


@pytest.mark.parametrize("mode", list(MODES))
@pytest.mark.parametrize("stage", [1, 2])
def test_update_scale_halves_named_module_steps_in_every_stage(
    setup, tmp_path, mode, stage
):
    config, base, stream, gradient_fn = _stage_inputs(setup, tmp_path, mode)
    scaled = replace(config, update_scale_modules=SCALED, update_scale=0.5)
    start = flatten_params(base.params)
    after = {}
    for name, run in (("plain", config), ("scaled", scaled)):
        params, exposure, log = train_stage(
            base.params,
            base.structure,
            stream,
            config=run,
            stage=stage,
            special_ids=base.special_ids,
            gradient_fn=gradient_fn,
            verbose=False,
        )
        assert exposure["updates"] == len(log) == 1
        after[name] = (flatten_params(params), log[0], exposure)
    plain, plain_log, _ = after["plain"]
    halved, halved_log, exposure = after["scaled"]
    assert exposure["optimizer"]["update_scale"] == {m: 0.5 for m in SCALED_MODULES}
    assert halved_log["grad_norm"] == plain_log["grad_norm"]
    for key, value in start.items():
        module = key.split("/", 1)[0]
        if module in SCALED_MODULES:
            np.testing.assert_allclose(
                halved[key] - value,
                0.5 * (plain[key] - value),
                rtol=1e-4,
                atol=1e-6 * float(np.max(np.abs(value)) + 1.0),
            )
        else:
            np.testing.assert_array_equal(halved[key], plain[key])
    for module, norm in plain_log["module_update_norms"].items():
        expected = 0.5 * norm if module in SCALED_MODULES else norm
        assert halved_log["module_update_norms"][module] == pytest.approx(
            expected, rel=1e-4
        )


@pytest.mark.parametrize("mode", list(MODES))
def test_update_scale_applies_from_stage_one_in_a_run(runs, mode):
    config, _ = runs[mode, "scaled"]
    baseline, _ = runs[mode, "baseline"]
    output = Path(config.output_dir)
    exposure = json.loads((output / "exposure.json").read_text())
    for domain in DOMAINS:
        assert exposure[domain]["optimizer"]["update_scale"] == {
            m: 0.5 for m in SCALED_MODULES
        }
    first = _read_jsonl(output / "training_log_stage_1.jsonl")[0]
    reference = _read_jsonl(Path(baseline.output_dir) / "training_log_stage_1.jsonl")[0]
    for module, norm in reference["module_update_norms"].items():
        expected = 0.5 * norm if module in SCALED_MODULES else norm
        assert first["module_update_norms"][module] == pytest.approx(expected, rel=1e-4)


def test_update_scale_multiplies_a_gate_multiplier(setup, tmp_path):
    config, base, _stream, _gradient_fn = _stage_inputs(setup, tmp_path, "backprop")
    grads = jax.tree_util.tree_map(jnp.ones_like, base.params)
    gate = {"L0_mha": 0.5}
    scale = {"L0_mha": 0.5, "L1_mha": 0.5}
    updates = {}
    for name, (multipliers, scales) in {
        "plain": (None, None),
        "both": (gate, scale),
    }.items():
        optimizer, info = build_stage_optimizer(config, 10, multipliers, scales)
        state = optimizer.init(base.params)
        updates[name], _ = optimizer.update(grads, state, base.params)
    assert info["update_scale"] == scale and info["module_multipliers"] == gate
    for module, factor in (("L0_mha", 0.25), ("L1_mha", 0.5), ("L0_mlp1", 1.0)):
        for got, want in zip(
            jax.tree_util.tree_leaves(updates["both"].nodes[module]),
            jax.tree_util.tree_leaves(updates["plain"].nodes[module]),
        ):
            np.testing.assert_allclose(got, factor * want, rtol=1e-6)


@pytest.mark.parametrize("mode", list(MODES))
def test_penalty_changes_params_and_logs_a_positive_value(runs, mode):
    config, result = runs[mode, "penalty"]
    baseline = _hashes(runs[mode, "baseline"][1])
    assert _hashes(result)[0] == baseline[0]
    assert all(a != b for a, b in zip(_hashes(result)[1:], baseline[1:]))
    output = Path(config.output_dir)
    for stage in range(1, 4):
        for entry in _read_jsonl(output / f"training_log_stage_{stage}.jsonl"):
            value = entry["error_norm_penalty"]
            assert np.isfinite(value) and value > 0
            assert entry["error_norm_penalty_weighted"] == pytest.approx(
                config.pc_error_norm_penalty * value
            )
    run_config = json.loads((output / "run_config.json").read_text())
    assert run_config["config"]["pc_error_norm_penalty"] == 1e-2
    assert run_config["error_norm_penalty"]["weight"] == 1e-2


def test_resume_accepts_old_configs_and_refuses_changed_interventions(runs, tmp_path):
    config, result = runs["pc", "baseline"]
    source = Path(config.output_dir)
    recorded = json.loads((source / "run_config.json").read_text())
    for key in ("pc_error_norm_penalty", "update_scale_modules", "update_scale"):
        assert key not in recorded["config"]
    assert "update_scale" not in recorded and "error_norm_penalty" not in recorded
    resumed = run_domain_forgetting(config, resume=True, verbose=False)
    assert _hashes(resumed) == _hashes(result)
    for name, values in {
        "penalty": {"pc_error_norm_penalty": 1e-3},
        "scale": {"update_scale_modules": SCALED, "update_scale": 0.5},
    }.items():
        output = tmp_path / name
        output.mkdir()
        (output / "run_config.json").write_bytes(
            (source / "run_config.json").read_bytes()
        )
        changed = replace(config, output_dir=str(output), **values)
        with pytest.raises(ValueError, match="different setup"):
            run_domain_forgetting(changed, resume=True, verbose=False)
    scaled, _ = runs["pc", "scaled"]
    other = replace(scaled, update_scale=0.25)
    with pytest.raises(ValueError, match="different setup"):
        run_domain_forgetting(other, resume=True, verbose=False)


def test_analysis_loads_an_update_scaled_run(runs):
    config, _ = runs["backprop", "scaled"]
    context = load_run(config.output_dir)
    assert context.config.update_scale_modules == SCALED


@pytest.mark.parametrize(
    "values, message",
    [
        ({"pc_error_norm_penalty": -1.0}, "finite and non-negative"),
        ({"pc_error_norm_penalty": float("nan")}, "finite and non-negative"),
        (
            {"pc_error_norm_penalty": 1e-3, "grad_noise_cosine": 0.9},
            "does not combine with grad_noise",
        ),
        ({"update_scale_modules": SCALED, "update_scale": 0.0}, "finite and positive"),
        (
            {"update_scale_modules": SCALED, "update_scale": float("inf")},
            "finite and positive",
        ),
        ({"update_scale": 0.5}, "needs update_scale_modules"),
        (
            {"update_scale_modules": ("L0_mha", "L0_mha"), "update_scale": 0.5},
            "duplicates",
        ),
    ],
)
def test_intervention_validation(setup, tmp_path, values, message):
    checkpoint, data_dir = setup
    config = tiny_config(checkpoint, data_dir, tmp_path / "run", **values)
    with pytest.raises(ValueError, match=message):
        config.validate()


def test_unknown_update_scale_module_is_refused(setup, tmp_path):
    checkpoint, data_dir = setup
    config = tiny_config(
        checkpoint,
        data_dir,
        tmp_path / "run",
        update_scale_modules=("L9_mha",),
        update_scale=0.5,
    )
    with pytest.raises(ValueError, match="unknown module or group"):
        run_domain_forgetting(config, verbose=False)


def test_regime_probe_leaves_backprop_params_unchanged(runs):
    config, result = runs["backprop", "probed"]
    assert _hashes(result) == _hashes(runs["backprop", "baseline"][1])
    records = _read_jsonl(Path(config.output_dir) / "regime.jsonl")
    assert records and all("lambda_max_final" in r for r in records)


@pytest.mark.parametrize("mode", list(MODES))
def test_penalty_logs_task_and_penalty_gradient_norms(runs, mode):
    config, _ = runs[mode, "penalty"]
    baseline, _ = runs[mode, "baseline"]
    log = _read_jsonl(Path(config.output_dir) / "training_log_stage_1.jsonl")
    reference = _read_jsonl(Path(baseline.output_dir) / "training_log_stage_1.jsonl")
    first = log[0]
    assert first["task_grad_norm"] == pytest.approx(reference[0]["grad_norm"], rel=1e-6)
    for entry in log:
        task, penalty = entry["task_grad_norm"], entry["error_norm_penalty_grad_norm"]
        assert penalty > 0
        assert abs(task - penalty) - 1e-6 <= entry["grad_norm"] <= task + penalty + 1e-6
        assert set(entry["module_task_grad_norms"]) == set(entry["module_update_norms"])
        total = np.sqrt(sum(v**2 for v in entry["module_penalty_grad_norms"].values()))
        assert total == pytest.approx(penalty, rel=1e-5)
    assert "error_norm_penalty_grad_norm" not in reference[0]


@pytest.mark.parametrize("mode", list(MODES))
def test_calibration_ratio_matches_a_direct_computation(setup, tmp_path, mode):
    checkpoint, data_dir = setup
    config = tiny_config(checkpoint, data_dir, tmp_path / "run", **MODES[mode])
    report = calibrate_error_norm_penalty(config, "news", 2)
    assert report["mode"] == mode and report["num_batches"] == 2
    base = load_base_model(config)
    stream = load_domain_cache(
        data_dir,
        "news",
        tokenizer_json=base.tokenizer_json,
        seq_len=int(base.model_config["seq_len"]),
        tokens_per_domain=config.tokens_per_domain,
    ).train
    batches, keys = fixed_mlm_batches(
        stream,
        base.structure,
        base.special_ids,
        num_batches=2,
        batch_size=config.batch_size,
        mask_prob=config.mask_prob,
        seed=config.seed,
    )
    gradient_fn = make_gradient_fn(
        config.mode, base.structure, config.pc_forward_updates, config.pc_inference
    )
    penalty_fn = make_error_norm_penalty_fn(base.structure)
    task, penalty = {}, {}
    for batch, key in zip(batches, keys):
        grads, _, count, _ = accumulate_gradients(
            gradient_fn, base.params, [batch], [key], summed=mode == "pc"
        )
        term, _ = error_norm_penalty_gradient(
            penalty_fn, base.params, [batch], [key], count, 1.0
        )
        for name in grads.nodes:
            leaves = jax.tree_util.tree_leaves(grads.nodes[name])
            if not leaves:
                continue
            norm = lambda tree: float(
                np.sqrt(sum(np.sum(np.square(np.asarray(x))) for x in tree))
            )
            task.setdefault(name, []).append(norm(leaves))
            penalty.setdefault(name, []).append(
                norm(jax.tree_util.tree_leaves(term.nodes[name]))
            )
    blocks = [n for n in task if BLOCK_MODULE_PATTERN.match(n)]
    assert report["block_modules"] == blocks
    assert blocks == [f"L{i}_{t}" for i in (0, 1) for t in ("mha", "mlp1", "mlp2")]
    ratios = {n: np.mean(penalty[n]) / np.mean(task[n]) for n in task}
    for name, ratio in ratios.items():
        assert report["modules"][name]["ratio"] == pytest.approx(ratio, rel=1e-5)
    r1 = float(np.median([ratios[n] for n in blocks]))
    assert report["r1"] == pytest.approx(r1, rel=1e-5)
    target = 0.05
    gamma = target / report["r1"]
    scaled = []
    for name in blocks:
        batch, key = batches[0], keys[0]
        grads, _, count, _ = accumulate_gradients(
            gradient_fn, base.params, [batch], [key], summed=mode == "pc"
        )
        term, _ = error_norm_penalty_gradient(
            penalty_fn, base.params, [batch], [key], count, gamma
        )
        one, _ = error_norm_penalty_gradient(
            penalty_fn, base.params, [batch], [key], count, 1.0
        )
        scaled.append(
            float(module_grad_norms(term)[name]) / float(module_grad_norms(one)[name])
        )
    np.testing.assert_allclose(scaled, gamma, rtol=1e-5)
    assert gamma * report["r1"] == pytest.approx(target)


def test_calibrate_penalty_cli_writes_json(setup, tmp_path):
    from examples import mlm_domain_forgetting as cli

    checkpoint, data_dir = setup
    out = tmp_path / "calibration.json"
    report = cli.main(
        [
            "calibrate_penalty",
            "--checkpoint",
            str(checkpoint),
            "--data_dir",
            str(data_dir),
            "--skip_shape_check",
            "--batch_size",
            "4",
            "--mask_prob",
            "0.3",
            "--tokens_per_domain",
            "60",
            "--batches",
            "2",
            "--out",
            str(out),
        ]
    )
    written = json.loads(out.read_text())
    assert written["r1"] == pytest.approx(report["r1"]) and written["r1"] > 0
    assert written["mode"] == "backprop" and written["domain"] == "news"


def test_frontier_labels_intervention_runs_and_keeps_them_off_the_curve(tmp_path):
    base = {"lr": 3e-5}
    assert describe_gate({**base, "pc_error_norm_penalty": 1e-3})[2] == (
        "lr3e-05_pen0.001"
    )
    scaled = {**base, "update_scale_modules": ["embeddings"], "update_scale": 0.5}
    assert describe_gate(scaled)[2] == "lr3e-05_scale0.5[embeddings]"
    write_references(tmp_path)
    for name, extra in {
        "pen": {"pc_error_norm_penalty": 1e-3},
        "scale": {"update_scale_modules": ["L0_mha"], "update_scale": 0.25},
    }.items():
        path = write_run(tmp_path, f"bp_{name}", x=0.45, y=2.0, seed=0, lr=3e-5)
        run_config = json.loads((path / "run_config.json").read_text())
        run_config["config"].update(extra)
        (path / "run_config.json").write_text(json.dumps(run_config))
    points = collect_runs(sorted(tmp_path.iterdir()))
    marked = {p["run_dir"].rsplit("/", 1)[-1]: p for p in points}
    for name in ("bp_pen", "bp_scale"):
        assert marked[name]["gate"] == "none" and not _is_reference(marked[name])
    assert _is_reference(marked["fwd_lr3e-05_s0"])
    order = compute_frontier(points)["orders"]["news>reviews>wiki"]
    assert order["curves"]["0"]["y"] == pytest.approx([3.3, 3.1, 3.2])
    runs_by_name = {r["run_dir"].rsplit("/", 1)[-1]: r for r in order["runs"]}
    assert runs_by_name["bp_pen"]["d"] == pytest.approx(2.0 - 3.125)
    assert order["mean_curve"]["n"] == [3, 3, 3]


def test_compare_reports_a_penalty_difference():
    config = {"lr": 3e-5, "domains": list(DOMAINS), "mode": "backprop"}
    left = {
        "config": config,
        "valid_sha256": {},
        "train_array_sha256": {},
        "source": {"params_sha256": "a"},
    }
    right = {**left, "config": {**config, "pc_error_norm_penalty": 1e-3}}
    assert check_comparable_runs(left, left) == []
    problems = check_comparable_runs(left, right)
    assert problems and "pc_error_norm_penalty" in problems[0]
    scaled = {**left, "config": {**config, "update_scale": 0.5}}
    assert "update_scale" in check_comparable_runs(left, scaled)[0]
