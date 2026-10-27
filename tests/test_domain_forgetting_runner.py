"""Tests for the sequential MLM domain-forgetting runner (tiny BERT, offline)."""

import json
import os
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from examples.bert_demo import load_checkpoint
from examples.convert_hf_bert import (
    bert_model_config,
    build_bert_structure,
    convert_bert_state_dict,
    graph_params_for,
    write_bert_checkpoint,
)
from fabricpc.experiments.domain_forgetting import runner
from fabricpc.experiments.domain_forgetting.artifacts import load_npz
from fabricpc.experiments.domain_forgetting.runner import (
    RunConfig,
    accumulate_gradients,
    build_stage_optimizer,
    compare_pc_bp_runs,
    compare_runs,
    flatten_params,
    make_gradient_fn,
    perturb_params,
    load_base_model,
    load_stage_snapshot,
    num_stage_updates,
    run_domain_forgetting,
)
from fabricpc.experiments.forgetting_benchmark.sequential_trainer import params_hash
from tests.test_bert_hf import TINY_HF_CONFIG, _synthetic_state_dict
from tests.test_domain_forgetting_data import build_tiny_caches, tiny_tokenizer

DOMAINS = ("news", "reviews", "wiki")
SEQ_LEN = 6


def write_tiny_bert_checkpoint(directory, tokenizer, *, dtype=np.float32, depth=1):
    """Write a tiny ``architecture="bert"`` checkpoint with ``tokenizer``."""
    config = {
        **TINY_HF_CONFIG,
        "vocab_size": tokenizer.get_vocab_size(),
        "num_hidden_layers": depth,
    }
    state = _synthetic_state_dict(np.random.default_rng(0), **config)
    state = {key: 0.1 * value for key, value in state.items()}
    model_config = bert_model_config(config, SEQ_LEN)
    params = graph_params_for(
        convert_bert_state_dict(state, config=config, seq_len=SEQ_LEN),
        build_bert_structure(model_config),
    )
    params = jax.tree_util.tree_map(lambda leaf: np.asarray(leaf, dtype=dtype), params)
    path = directory / "bert.json"
    write_bert_checkpoint(path, params, tokenizer, model_config)
    return path


def make_setup(directory, *, dtype=np.float32):
    """Return ``(checkpoint, data_dir)`` for three synthetic domains."""
    directory.mkdir(parents=True, exist_ok=True)
    tokenizer = tiny_tokenizer(directory)
    checkpoint = write_tiny_bert_checkpoint(directory, tokenizer, dtype=dtype)
    build_tiny_caches(directory, tokenizer)
    return checkpoint, directory / "data"


def tiny_config(checkpoint, data_dir, output_dir, **overrides) -> RunConfig:
    """A run of three updates per stage on the tiny checkpoint."""
    values = {
        "checkpoint": str(checkpoint),
        "data_dir": str(data_dir),
        "domains": DOMAINS,
        "output_dir": str(output_dir),
        "tokens_per_domain": 60,
        "lr": 1e-2,
        "batch_size": 4,
        "eval_batch_size": 4,
        "mask_prob": 0.3,
        "dropout_rate": 0.1,
        "expected_shape": None,
        "log_interval": 0,
        **overrides,
    }
    return RunConfig(**values)


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    return make_setup(tmp_path_factory.mktemp("domain_forgetting"))


@pytest.fixture(scope="module")
def bp_run(setup, tmp_path_factory):
    checkpoint, data_dir = setup
    output = tmp_path_factory.mktemp("bp_run") / "run"
    config = tiny_config(checkpoint, data_dir, output)
    return config, run_domain_forgetting(config, verbose=False)


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _matrices(output):
    return {
        name: json.loads((output / f"{name}_matrix.json").read_text())["matrix"]
        for name in ("masked_ce", "masked_perplexity", "masked_accuracy")
    }


def test_run_writes_twelve_evaluations_and_all_artifacts(bp_run):
    config, result = bp_run
    output = Path(config.output_dir)
    records = _read_jsonl(output / "evaluations.jsonl")
    assert len(records) == 12
    assert {(r["stage"], r["domain"]) for r in records} == {
        (stage, name) for stage in range(4) for name in DOMAINS
    }
    for name in (
        "run_config.json",
        "checkpoint_manifest.json",
        "dataset_manifest.json",
        "tokenizer.json",
        "masked_ce_matrix.json",
        "masked_perplexity_matrix.json",
        "masked_accuracy_matrix.json",
        "forgetting_matrices.json",
        "summary.json",
        "exposure.json",
        "runtime.json",
        *(f"stage_{s}.npz" for s in range(4)),
        *(f"training_log_stage_{s}.jsonl" for s in range(1, 4)),
    ):
        assert (output / name).exists(), name
    matrices = _matrices(output)
    assert matrices == result["matrices"]
    for record in records:
        row = matrices["masked_ce"][record["stage"]]
        assert row[DOMAINS.index(record["domain"])] == record["masked_ce"]
    ce = np.asarray(matrices["masked_ce"])
    assert ce.shape == (4, 3)
    np.testing.assert_allclose(np.exp(ce), matrices["masked_perplexity"])
    summary = json.loads((output / "summary.json").read_text())
    assert summary["exposure_within_one_update"]
    assert summary["learning_gain_ce"]["news"] == pytest.approx(ce[0, 0] - ce[1, 0])


def test_stages_chain_from_previous_params(bp_run, setup):
    config, result = bp_run
    output = Path(config.output_dir)
    manifest = json.loads((output / "checkpoint_manifest.json").read_text())
    stages = manifest["stages"]
    assert [entry["domain"] for entry in stages] == [None, *DOMAINS]
    base = load_base_model(config)
    assert stages[0]["params_sha256"] == manifest["source"]["params_sha256"]
    assert stages[0]["params_sha256"] == params_hash(base.params)
    assert stages[0]["parent_params_sha256"] is None
    for previous, entry in zip(stages, stages[1:]):
        assert entry["initial_params_sha256"] == previous["params_sha256"]
        assert entry["parent_params_sha256"] == previous["params_sha256"]
        assert entry["params_sha256"] != previous["params_sha256"]
    for entry in stages:
        params = load_stage_snapshot(output / entry["file"], base.structure)
        assert params_hash(params) == entry["params_sha256"]
    source = load_checkpoint(config.checkpoint)
    assert params_hash(source["params"]) == stages[0]["params_sha256"]


def test_tokenizer_and_head_are_unchanged(bp_run):
    config, _ = bp_run
    output = Path(config.output_dir)
    source = load_checkpoint(config.checkpoint)
    assert (output / "tokenizer.json").read_text() == source["tokenizer_json"]
    vocab_size = len(source["vocab"])
    for stage in range(4):
        flat = load_npz(output / f"stage_{stage}.npz")
        assert flat["mlm_head/weights/W_out"].shape == (16, vocab_size)
        assert flat["mlm_head/biases/b_out"].shape == (vocab_size,)
        assert flat["token_embeddings/weights/embeddings"].shape[0] == vocab_size


def test_exposure_matches_budget_within_one_update(bp_run):
    config, result = bp_run
    updates = num_stage_updates(60, 4, SEQ_LEN)
    assert updates == 3
    for name in DOMAINS:
        exposure = result["exposure"][name]
        assert exposure["updates"] == updates
        assert exposure["within_one_update"]
        assert abs(exposure["stream_tokens"] - 60) <= 4 * (SEQ_LEN - 1)
        assert exposure["optimizer"]["total_steps"] == updates
    log = _read_jsonl(Path(config.output_dir) / "training_log_stage_1.jsonl")
    assert [entry["update"] for entry in log] == [1, 2, 3]


class _Crash(RuntimeError):
    pass


def _crash_at_stage(monkeypatch, crash_stage):
    original = runner.train_stage

    def train_stage(*args, stage, **kwargs):
        if stage == crash_stage:
            raise _Crash(f"simulated crash in stage {stage}")
        return original(*args, stage=stage, **kwargs)

    monkeypatch.setattr(runner, "train_stage", train_stage)


def test_resume_after_crash_equals_uninterrupted_run(
    bp_run, setup, tmp_path, monkeypatch
):
    config, result = bp_run
    checkpoint, data_dir = setup
    crashed = tiny_config(checkpoint, data_dir, tmp_path / "run")
    _crash_at_stage(monkeypatch, 2)
    with pytest.raises(_Crash):
        run_domain_forgetting(crashed, verbose=False)
    manifest = json.loads((tmp_path / "run" / "checkpoint_manifest.json").read_text())
    assert [entry["stage"] for entry in manifest["stages"]] == [0, 1]
    monkeypatch.undo()

    trained = []
    original = runner.train_stage

    def counting(*args, stage, **kwargs):
        trained.append(stage)
        return original(*args, stage=stage, **kwargs)

    monkeypatch.setattr(runner, "train_stage", counting)
    resumed = run_domain_forgetting(crashed, resume=True, verbose=False)
    assert trained == [2, 3]
    assert resumed["matrices"] == result["matrices"]
    assert _matrices(tmp_path / "run") == _matrices(Path(config.output_dir))
    assert resumed["summary"]["stage_params_sha256"] == (
        result["summary"]["stage_params_sha256"]
    )
    assert len(_read_jsonl(tmp_path / "run" / "evaluations.jsonl")) == 12

    trained.clear()
    again = run_domain_forgetting(crashed, resume=True, verbose=False)
    assert trained == []
    assert again["matrices"] == result["matrices"]


def test_resume_retrains_from_a_corrupted_snapshot(bp_run, setup, tmp_path):
    config, result = bp_run
    checkpoint, data_dir = setup
    source = Path(config.output_dir)
    output = tmp_path / "run"
    output.mkdir()
    for path in source.iterdir():
        (output / path.name).write_bytes(path.read_bytes())
    flat = load_npz(output / "stage_2.npz")
    flat["mlm_head/biases/b_out"] = flat["mlm_head/biases/b_out"] + 1.0
    np.savez(output / "stage_2.npz", **flat)
    copied = replace(config, output_dir=str(output))
    resumed = run_domain_forgetting(copied, resume=True, verbose=False)
    assert resumed["matrices"] == result["matrices"]
    assert params_hash(
        load_stage_snapshot(output / "stage_2.npz", load_base_model(config).structure)
    ) == (result["stages"][2]["params_sha256"])


def test_resume_rejects_a_different_setup(bp_run, tmp_path):
    config, _ = bp_run
    output = tmp_path / "run"
    output.mkdir()
    source = Path(config.output_dir)
    (output / "run_config.json").write_bytes((source / "run_config.json").read_bytes())
    changed = replace(config, output_dir=str(output), lr=2e-2)
    with pytest.raises(ValueError, match="different setup"):
        run_domain_forgetting(changed, resume=True, verbose=False)


def test_output_protection(bp_run, tmp_path):
    config, _ = bp_run
    with pytest.raises(FileExistsError, match="not empty"):
        run_domain_forgetting(config, verbose=False)
    with pytest.raises(ValueError, match="mutually exclusive"):
        run_domain_forgetting(config, resume=True, overwrite=True, verbose=False)
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    for kwargs in ({}, {"resume": True}, {"overwrite": True}):
        with pytest.raises(ValueError, match="symlink"):
            run_domain_forgetting(
                replace(config, output_dir=str(link / "run")), verbose=False, **kwargs
            )
    inner = tmp_path / "inner"
    inner.mkdir()
    (inner / "evil").symlink_to(real)
    with pytest.raises(ValueError, match="symlink"):
        run_domain_forgetting(
            replace(config, output_dir=str(inner)), overwrite=True, verbose=False
        )


def test_spike_only_pc_reproduces_backprop_in_float64(tmp_path):
    with jax.enable_x64(True):
        checkpoint, data_dir = make_setup(tmp_path / "setup", dtype=np.float64)
        bp_config = tiny_config(checkpoint, data_dir, tmp_path / "bp")
        pc_config = tiny_config(checkpoint, data_dir, tmp_path / "pc", mode="pc")
        bp = run_domain_forgetting(bp_config, verbose=False)
        pc = run_domain_forgetting(pc_config, verbose=False)
        agreement = compare_pc_bp_runs(tmp_path / "pc", tmp_path / "bp")
        base = load_base_model(pc_config)

    assert base.infer_steps == runner.max_free_depth(base.structure, "longest")
    stage_0 = load_npz(tmp_path / "pc" / "stage_0.npz")
    assert all(value.dtype == np.float64 for value in stage_0.values())
    run_config = json.loads((tmp_path / "pc" / "run_config.json").read_text())
    assert run_config["pc"]["infer_steps"] == base.infer_steps
    assert run_config["clip_norm"] == pc_config.grad_clip
    log = _read_jsonl(tmp_path / "pc" / "training_log_stage_1.jsonl")
    assert all(np.isfinite(entry["pc_total_energy"]) for entry in log)
    print(
        "pc_bp_agreement",
        json.dumps({k: v for k, v in agreement.items() if k != "definition"}),
    )
    assert agreement["stage_param_rel_distance"][0] == 0.0
    assert agreement["max_stage_param_rel_distance"] < 1e-6
    assert agreement["max_rel_ce_diff"] < 1e-6
    np.testing.assert_allclose(
        pc["matrices"]["masked_ce"], bp["matrices"]["masked_ce"], rtol=1e-6
    )
    assert pc["summary"]["stage_params_sha256"][0] == (
        bp["summary"]["stage_params_sha256"][0]
    )
    for stage in range(1, 4):
        bp_log = _read_jsonl(tmp_path / "bp" / f"training_log_stage_{stage}.jsonl")
        pc_log = _read_jsonl(tmp_path / "pc" / f"training_log_stage_{stage}.jsonl")
        assert [e["masked_tokens"] for e in bp_log] == [
            e["masked_tokens"] for e in pc_log
        ]
        np.testing.assert_allclose(
            [e["masked_loss"] for e in pc_log],
            [e["masked_loss"] for e in bp_log],
            rtol=1e-6,
        )


def test_stage_optimizer_warmup_cosine_floor_and_pc_clip(setup, tmp_path):
    checkpoint, data_dir = setup
    config = tiny_config(checkpoint, data_dir, tmp_path / "run", lr=1.0)
    assert num_stage_updates(5_000_000, 64, 128) == 616
    assert num_stage_updates(8128, 64, 128) == 1
    assert num_stage_updates(8129, 64, 128) == 2
    optimizer, info = build_stage_optimizer(config, 100)
    assert info["warmup_steps"] == 10
    assert info["min_lr"] == pytest.approx(0.1)
    assert info["clip_norm"] == 2.0
    params = {"w": jnp.zeros(3)}
    grads = {"w": jnp.full(3, 0.5)}
    state = optimizer.init(params)
    rates = []
    for _ in range(101):
        updates, state = optimizer.update(grads, state, params)
        rates.append(-float(updates["w"][0]))
    assert rates[0] == pytest.approx(0.0, abs=1e-6)
    assert rates[5] == pytest.approx(0.5, rel=1e-5)
    assert rates[10] == pytest.approx(1.0, rel=1e-5)
    assert rates[55] == pytest.approx(0.1 + 0.45, rel=1e-5)
    assert rates[100] == pytest.approx(0.1, rel=1e-5)
    assert all(a >= b for a, b in zip(rates[10:], rates[11:]))
    pc = replace(config, mode="pc", pc_output_energy_scale=4.0)
    assert pc.clip_norm == 8.0
    assert build_stage_optimizer(pc, 100)[1]["clip_norm"] == 8.0
    assert replace(config, pc_output_energy_scale=4.0).clip_norm == 2.0
    assert pc.optimizer_eps == pytest.approx(4.0 * config.adam_eps)
    assert build_stage_optimizer(pc, 100)[1]["adam_eps"] == pc.optimizer_eps
    assert replace(config, pc_output_energy_scale=4.0).optimizer_eps == (
        config.adam_eps
    )


def _rewrite_checkpoint(source, target, edit):
    metadata = json.loads(source.read_text())
    edit(metadata)
    target.write_text(json.dumps(metadata))
    (target.parent / metadata["tensor_file"]).write_bytes(
        (source.parent / metadata["tensor_file"]).read_bytes()
    )
    return target


def test_checkpoint_validation_rejects_mismatches(setup, tmp_path):
    checkpoint, data_dir = setup
    config = tiny_config(checkpoint, data_dir, tmp_path / "run")
    base = load_base_model(config)
    assert base.infer_steps is None
    assert base.source["checkpoint_sha256"] == runner.file_sha256(checkpoint)
    assert base.special_ids["[CLS]"] == base.vocab["[CLS]"]
    with pytest.raises(ValueError, match="shape"):
        load_base_model(replace(config, expected_shape=runner.BERT_MINI_SHAPE))
    with pytest.raises(FileNotFoundError):
        load_base_model(replace(config, checkpoint=str(tmp_path / "missing.json")))

    def encoder(metadata):
        metadata["config"]["architecture"] = "encoder"

    def vocab(metadata):
        metadata["vocab"]["extra"] = len(metadata["vocab"])

    def seq_len(metadata):
        metadata["config"]["seq_len"] = 8

    for edit, match in (
        (encoder, "bert"),
        (vocab, "vocab"),
        (seq_len, "parameters do not match"),
    ):
        broken = _rewrite_checkpoint(checkpoint, tmp_path / "broken.json", edit)
        with pytest.raises(ValueError, match=match):
            load_base_model(replace(config, checkpoint=str(broken)))


def test_pc_rejects_too_few_inference_steps(setup, tmp_path):
    checkpoint, data_dir = setup
    config = tiny_config(checkpoint, data_dir, tmp_path / "run", mode="pc")
    needed = load_base_model(config).infer_steps
    assert needed >= 4
    with pytest.raises(ValueError, match="longest free depth"):
        load_base_model(replace(config, pc_infer_steps=needed - 1))


def test_config_validation(setup, tmp_path):
    checkpoint, data_dir = setup
    config = tiny_config(checkpoint, data_dir, tmp_path / "run")
    for overrides, match in (
        ({"mode": "gates"}, "mode"),
        ({"domains": ("news", "news")}, "duplicate"),
        ({"batch_size": 0}, "batch_size"),
        ({"micro_batch_size": 3}, "micro_batch_size"),
        ({"warmup_fraction": 1.5}, "warmup_fraction"),
        ({"mode": "pc", "pc_inference": "sgd"}, "spiking"),
        ({"mode": "pc", "pc_output_energy_scale": 0.0}, "scale"),
    ):
        with pytest.raises(ValueError, match=match):
            replace(config, **overrides).validate()


def test_micro_batches_accumulate_to_the_same_update_count(setup, tmp_path):
    checkpoint, data_dir = setup
    config = tiny_config(
        checkpoint,
        data_dir,
        tmp_path / "run",
        micro_batch_size=2,
        domains=("wiki", "news"),
    )
    result = run_domain_forgetting(config, verbose=False)
    assert np.asarray(result["matrices"]["masked_ce"]).shape == (3, 2)
    for name in ("wiki", "news"):
        exposure = result["exposure"][name]
        assert exposure["updates"] == 3
        assert exposure["gradient_accumulation_steps"] == 2
        assert exposure["micro_batches"] == 6
        assert exposure["within_one_update"]
    assert len(_read_jsonl(tmp_path / "run" / "evaluations.jsonl")) == 6


def test_cli_run_matches_library_run_and_compare_writes_report(bp_run, tmp_path):
    from examples import mlm_domain_forgetting as cli

    config, result = bp_run
    output = tmp_path / "cli"
    argv = [
        "--checkpoint",
        config.checkpoint,
        "--data_dir",
        config.data_dir,
        "--domains",
        *DOMAINS,
        "--output_dir",
        str(output),
        "--mode",
        "backprop",
        "--tokens_per_domain",
        "60",
        "--lr",
        "1e-2",
        "--batch_size",
        "4",
        "--eval_batch_size",
        "4",
        "--mask_prob",
        "0.3",
        "--log_interval",
        "0",
        "--skip_shape_check",
    ]
    assert cli.config_from_args(cli.build_parser().parse_args(argv)) == replace(
        config, output_dir=str(output)
    )
    cli_result = cli.main(argv)
    assert cli_result["matrices"] == result["matrices"]
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([*argv, "--resume", "--overwrite"])
    report = cli.main(
        ["compare", "--pc_dir", str(output), "--bp_dir", config.output_dir]
    )
    assert report["max_abs_ce_diff"] == 0.0
    assert report["max_stage_param_rel_distance"] == 0.0
    saved = json.loads((output / "pc_bp_agreement.json").read_text())
    assert saved["max_abs_ce_diff"] == 0.0


def test_data_builder_cli_refuses_an_existing_cache(setup, tmp_path):
    from examples import build_domain_forgetting_data as builder

    checkpoint, data_dir = setup
    tokenizer_json, seq_len = builder.read_checkpoint_tokenizer(checkpoint)
    assert seq_len == SEQ_LEN
    assert tokenizer_json == load_checkpoint(str(checkpoint))["tokenizer_json"]
    args = builder.build_parser().parse_args(["--checkpoint", str(checkpoint)])
    assert args.domains == list(DOMAINS)
    assert args.tokens_per_domain == 5_000_000
    with pytest.raises(FileExistsError, match="--overwrite"):
        builder.main(["--checkpoint", str(checkpoint), "--out_dir", str(data_dir)])
    with pytest.raises(ValueError, match="unknown"):
        builder.main(["--checkpoint", str(checkpoint), "--domains", "poems"])


def _uneven_micro_batches(vocab_size, seq_len, rng):
    """Two micro-batches of two windows with 1 and 5 masked tokens."""
    batches = []
    for masked in ((1, 0), (3, 2)):
        mask = np.zeros((2, seq_len), dtype=bool)
        for row, count in enumerate(masked):
            mask[row, 1 : 1 + count] = True
        batches.append(
            {
                "x": jnp.asarray(rng.integers(5, vocab_size, (2, seq_len))),
                "y": jnp.asarray(rng.integers(5, vocab_size, (2, seq_len))),
                "loss_mask": jnp.asarray(mask),
                "attention_mask": jnp.ones((2, seq_len), dtype=bool),
            }
        )
    return batches


@pytest.mark.parametrize("mode", ["backprop", "pc"])
def test_micro_batch_accumulation_equals_the_full_batch_gradient(mode, tmp_path):
    with jax.enable_x64(True):
        checkpoint, data_dir = make_setup(tmp_path / "setup", dtype=np.float64)
        config = tiny_config(
            checkpoint, data_dir, tmp_path / "run", mode=mode, dropout_rate=0.0
        )
        base = load_base_model(config)
        gradient_fn = make_gradient_fn(mode, base.structure, config.pc_forward_updates)
        vocab_size = len(base.vocab)
        micro = _uneven_micro_batches(vocab_size, SEQ_LEN, np.random.default_rng(0))
        full = {k: jnp.concatenate([b[k] for b in micro]) for k in micro[0]}
        key = jax.random.PRNGKey(0)
        summed = mode == "pc"
        accumulated, loss_sum, count, extras = accumulate_gradients(
            gradient_fn, base.params, micro, [key, key], summed=summed
        )
        expected, full_loss, full_count, _ = gradient_fn(base.params, full, key)

        def mean(batch):
            grads, _loss, batch_count, _ = gradient_fn(base.params, batch, key)
            divisor = batch_count if summed else 1.0
            return jax.tree_util.tree_map(lambda g: g / divisor, grads)

        flat_acc = flatten_params(accumulated)
        flat_full = flatten_params(mean(full))
        mean_of_means = flatten_params(
            jax.tree_util.tree_map(
                lambda a, b: (a + b) / 2, mean(micro[0]), mean(micro[1])
            )
        )
    assert count == float(full_count) == 6.0
    assert loss_sum == pytest.approx(float(full_loss), rel=1e-10)
    assert len(extras) == 2
    scale = max(float(np.max(np.abs(v))) for v in flat_full.values())
    worst = max(float(np.max(np.abs(flat_acc[k] - flat_full[k]))) for k in flat_full)
    assert worst < 1e-10 * scale
    head = "mlm_head/weights/W_out"
    assert not np.allclose(mean_of_means[head], flat_full[head], rtol=1e-3)


def test_spike_only_pc_reproduces_backprop_at_energy_scale_four(tmp_path):
    with jax.enable_x64(True):
        checkpoint, data_dir = make_setup(tmp_path / "setup", dtype=np.float64)
        common = {"micro_batch_size": 2, "lr": 1e-3, "adam_eps": 1e-3}
        bp_config = tiny_config(checkpoint, data_dir, tmp_path / "bp", **common)
        pc_config = tiny_config(
            checkpoint,
            data_dir,
            tmp_path / "pc",
            mode="pc",
            pc_output_energy_scale=4.0,
            **common,
        )
        bp = run_domain_forgetting(bp_config, verbose=False)
        pc = run_domain_forgetting(pc_config, verbose=False)
        agreement = compare_runs(tmp_path / "pc", tmp_path / "bp")
    run_config = json.loads((tmp_path / "pc" / "run_config.json").read_text())
    assert run_config["clip_norm"] == 4.0 * pc_config.grad_clip
    assert run_config["adam_eps"] == pytest.approx(4.0 * pc_config.adam_eps)
    print(
        "pc_bp_agreement_scale4",
        json.dumps({k: v for k, v in agreement.items() if k != "definition"}),
    )
    assert agreement["modes"] == ["pc", "backprop"]
    assert agreement["max_stage_param_rel_distance"] < 1e-6
    assert agreement["max_stage_param_rel_distance_no_key_bias"] < 1e-6
    assert agreement["max_rel_ce_diff"] < 1e-6
    np.testing.assert_allclose(
        pc["matrices"]["masked_ce"], bp["matrices"]["masked_ce"], rtol=1e-6
    )


def test_init_perturbation_is_seeded_recorded_and_comparable(bp_run, setup, tmp_path):
    config, result = bp_run
    checkpoint, data_dir = setup
    base = load_base_model(config)
    assert perturb_params(base.params, 0.0, 0) is base.params
    first = flatten_params(perturb_params(base.params, 1e-3, 7))
    again = flatten_params(perturb_params(base.params, 1e-3, 7))
    other = flatten_params(perturb_params(base.params, 1e-3, 8))
    source = flatten_params(base.params)
    key = "mlm_head/weights/W_out"
    np.testing.assert_array_equal(first[key], again[key])
    assert not np.array_equal(first[key], other[key])
    ratio = first[key] / source[key] - 1.0
    assert 0.5e-3 < float(np.std(ratio)) < 2e-3
    assert first[key].dtype == source[key].dtype

    perturbed = replace(
        config,
        output_dir=str(tmp_path / "pert"),
        init_perturbation=1e-3,
        init_perturbation_seed=7,
    )
    run_domain_forgetting(perturbed, verbose=False)
    run_config = json.loads((tmp_path / "pert" / "run_config.json").read_text())
    assert run_config["init_perturbation"]["relative"] == 1e-3
    assert run_config["init_perturbation"]["seed"] == 7
    manifest = json.loads((tmp_path / "pert" / "checkpoint_manifest.json").read_text())
    assert manifest["stages"][0]["params_sha256"] == (
        run_config["init_perturbation"]["params_sha256"]
    )
    assert manifest["stages"][0]["params_sha256"] != (
        manifest["source"]["params_sha256"]
    )
    floor = compare_runs(tmp_path / "pert", config.output_dir)
    assert floor["modes"] == ["backprop", "backprop"]
    assert floor["init_perturbation"] == [1e-3, 0.0]
    assert floor["stage_param_rel_distance"][0] > 0
    assert floor["stage_param_rel_distance_no_key_bias"][0] > 0
    resumed = run_domain_forgetting(perturbed, resume=True, verbose=False)
    assert resumed["summary"]["stage_params_sha256"][0] == (
        manifest["stages"][0]["params_sha256"]
    )
    with pytest.raises(ValueError, match="init_perturbation"):
        RunConfig(**{**vars(config), "init_perturbation": -1.0}).validate()


def _copy_run(source, target):
    target.mkdir()
    for path in Path(source).iterdir():
        (target / path.name).write_bytes(path.read_bytes())
    return target


def test_compare_runs_rejects_incomparable_runs(bp_run, tmp_path):
    config, _ = bp_run
    for edit, match in (
        (lambda rc: rc["config"].update(lr=2e-2), "lr"),
        (lambda rc: rc["config"].update(seed=1), "seed"),
        (lambda rc: rc["valid_sha256"].update(news="0" * 64), "valid_sha256"),
        (
            lambda rc: rc["train_array_sha256"].update(wiki="0" * 64),
            "train_array_sha256",
        ),
        (
            lambda rc: rc["config"].update(domains=["reviews", "news", "wiki"]),
            "domain orders",
        ),
    ):
        copy = _copy_run(config.output_dir, tmp_path / f"copy_{match}")
        run_config = json.loads((copy / "run_config.json").read_text())
        edit(run_config)
        (copy / "run_config.json").write_text(json.dumps(run_config))
        with pytest.raises(ValueError, match=match):
            compare_runs(copy, config.output_dir)
    allowed = _copy_run(config.output_dir, tmp_path / "allowed")
    run_config = json.loads((allowed / "run_config.json").read_text())
    run_config["config"].update(
        mode="pc", pc_output_energy_scale=2.0, log_interval=5, init_perturbation=1e-7
    )
    (allowed / "run_config.json").write_text(json.dumps(run_config))
    assert compare_runs(allowed, config.output_dir)["max_abs_ce_diff"] == 0.0


def test_overwrite_removes_stale_run_and_analysis_artifacts(bp_run, tmp_path):
    config, result = bp_run
    output = _copy_run(config.output_dir, tmp_path / "run")
    stale = [
        "stage_9.npz",
        "training_log_stage_9.jsonl",
        "drift.json",
        "rollback.json",
        "error_maps.json",
        "predictors.json",
        "correlations.json",
        "pc_bp_agreement.json",
    ]
    for name in stale:
        (output / name).write_text("stale")
    (output / "notes.txt").write_text("keep")
    rerun = run_domain_forgetting(
        replace(config, output_dir=str(output)), overwrite=True, verbose=False
    )
    assert rerun["matrices"] == result["matrices"]
    for name in stale:
        assert not (output / name).exists(), name
    assert (output / "notes.txt").read_text() == "keep"
    assert (output / "stage_3.npz").exists()
