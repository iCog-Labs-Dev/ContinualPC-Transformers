"""Tests for the unified BERT demo entrypoint."""

import os
import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest

from examples.bert_demo import (
    PC_UNSUPPORTED_MESSAGE,
    build_mlm_loader,
    build_validation_mlm_loader,
    compute_gradient_accumulation_steps,
    derive_max_steps,
    deterministic_fraction_split,
    load_checkpoint,
    load_init_params,
    normalize_hf_dataset_args,
    parse_args,
    parse_validation_mask_seeds,
    run_finetune,
    run_pretrain,
    summarize_init_diagnostics,
)


def test_parse_pretrain_backprop_args():
    args = parse_args(["--stage", "pretrain", "--mode", "backprop"])

    assert args.stage == "pretrain"
    assert args.mode == "backprop"
    assert args.data_source == "tiny"


def test_parse_finetune_backprop_args():
    args = parse_args(
        ["--stage", "finetune", "--mode", "backprop", "--data_source", "synthetic"]
    )

    assert args.stage == "finetune"
    assert args.mode == "backprop"
    assert args.data_source == "synthetic"


def test_pc_pretrain_tiny_smoke(tmp_path):
    args = parse_args(
        [
            "--stage",
            "pretrain",
            "--mode",
            "pc",
            "--data_source",
            "tiny",
            "--seq_len",
            "6",
            "--embed_dim",
            "16",
            "--num_heads",
            "4",
            "--mlp_dim",
            "32",
            "--depth",
            "1",
            "--batch_size",
            "1",
            "--vocab_size",
            "16",
            "--max_steps",
            "1",
            "--checkpoint_interval",
            "1",
            "--validation_split",
            "validation",
            "--max_validation_sequences",
            "4",
            "--validation_interval",
            "0",
            "--log_interval",
            "0",
            "--dropout_rate",
            "0.0",
            "--checkpoint",
            str(tmp_path / "pc_transfer.json"),
        ]
    )

    run_pretrain(args)

    assert (tmp_path / "pc_transfer.json").exists()


def test_pc_finetune_fails_clearly():
    args = parse_args(["--stage", "finetune", "--mode", "pc"])

    with pytest.raises(NotImplementedError, match=PC_UNSUPPORTED_MESSAGE):
        run_finetune(args)


def test_normalize_hf_dataset_args_rejects_empty_dataset_name():
    with pytest.raises(ValueError, match="--dataset_name"):
        normalize_hf_dataset_args("", "")


def test_normalize_hf_dataset_args_treats_empty_revision_as_none():
    dataset_name, revision = normalize_hf_dataset_args("babylm/example", "")

    assert dataset_name == "babylm/example"
    assert revision is None


def test_deterministic_fraction_split_preserves_order_and_validates_sum():
    rows = list(range(10))

    first = deterministic_fraction_split(
        rows, train_fraction=0.6, dev_fraction=0.2, test_fraction=0.2, seed=3
    )
    second = deterministic_fraction_split(
        rows, train_fraction=0.6, dev_fraction=0.2, test_fraction=0.2, seed=3
    )
    different = deterministic_fraction_split(
        rows, train_fraction=0.6, dev_fraction=0.2, test_fraction=0.2, seed=4
    )

    assert first == second
    assert first != different
    assert all(part == sorted(part) for part in first)
    with pytest.raises(ValueError, match="sum to 1.0"):
        deterministic_fraction_split(
            rows, train_fraction=0.5, dev_fraction=0.2, test_fraction=0.2, seed=3
        )


def test_train_and_eval_window_shuffle_defaults_are_applied():
    args = parse_args(
        ["--stage", "pretrain", "--data_source", "tiny", "--seq_len", "6"]
    )

    train_loader, _, _ = build_mlm_loader(args, split="train", seed=0)
    dev_loader, _, _ = build_mlm_loader(args, split="dev", seed=0)

    assert train_loader.shuffle is True
    assert dev_loader.shuffle is False


def test_validation_sequence_cap_does_not_change_train_loader():
    args = parse_args(
        [
            "--stage",
            "pretrain",
            "--data_source",
            "tiny",
            "--validation_split",
            "dev",
            "--seq_len",
            "6",
            "--batch_size",
            "2",
            "--max_validation_sequences",
            "2",
        ]
    )
    train_loader, _, _ = build_mlm_loader(args, split="train", seed=0)
    validation_loader = build_validation_mlm_loader(args, seed=0)

    assert args.max_sequences is None
    assert train_loader.num_windows > validation_loader.num_windows
    assert validation_loader.num_windows == 2
    assert len(validation_loader) == 1


@pytest.mark.parametrize("cap", ["0", "-1", "1"])
def test_validation_sequence_cap_rejected_before_pretraining(
    tmp_path, cap, monkeypatch
):
    args = parse_args(
        [
            "--stage",
            "pretrain",
            "--mode",
            "pc",
            "--data_source",
            "tiny",
            "--validation_split",
            "dev",
            "--seq_len",
            "6",
            "--batch_size",
            "2",
            "--max_validation_sequences",
            cap,
            "--checkpoint",
            str(tmp_path / "never-written.json"),
        ]
    )

    def fail_if_loading(*_args, **_kwargs):
        pytest.fail("training loader must not be built for an invalid validation cap")

    monkeypatch.setattr("examples.bert_demo.build_mlm_loader", fail_if_loading)
    with pytest.raises(ValueError, match="max_validation_sequences"):
        run_pretrain(args)
    assert not (tmp_path / "never-written.json").exists()


def test_validation_sequence_cap_is_rejected_for_finetuning():
    args = parse_args(["--stage", "finetune", "--max_validation_sequences", "8"])
    with pytest.raises(ValueError, match="only valid for pretraining"):
        run_finetune(args)


def test_validation_mask_seed_parser_and_step_derivation(rng_key):
    args = parse_args(
        [
            "--stage",
            "pretrain",
            "--data_source",
            "tiny",
            "--seq_len",
            "6",
            "--batch_size",
            "2",
            "--tokens_per_update",
            "24",
        ]
    )
    loader, _, _ = build_mlm_loader(args, split="train", seed=0)

    assert parse_validation_mask_seeds("1, 2") == [1, 2]
    assert compute_gradient_accumulation_steps(args) == 2
    assert derive_max_steps(args, loader) == 3

    from fabricpc.core.inference import InferenceSGDNormClip
    from fabricpc.graph_initialization import initialize_params
    from fabricpc.nodes.encoder_transformer import create_encoder_mlm_model

    structure = create_encoder_mlm_model(
        depth=1,
        embed_dim=16,
        num_heads=4,
        mlp_dim=32,
        seq_len=6,
        vocab_size=loader.vocab_size,
        inference=InferenceSGDNormClip(eta_infer=0.1, infer_steps=3, max_norm=5.0),
    )
    diagnostics = summarize_init_diagnostics(initialize_params(structure, rng_key))
    assert all(np.isfinite(value) for value in diagnostics.values())
    assert "token_embedding_std" in diagnostics
    assert "position_embedding_std" in diagnostics


def test_tokenized_validation_requires_distinct_token_path(tmp_path):
    tokenizers = pytest.importorskip("tokenizers")
    tokenizer = tokenizers.Tokenizer(
        tokenizers.models.BPE(
            vocab={"[UNK]": 0, "warm": 1}, merges=[], unk_token="[UNK]"
        )
    )
    tokenizer.add_special_tokens(["[PAD]", "[MASK]", "[CLS]", "[SEP]"])
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    token_path = tmp_path / "tokens.npy"
    np.save(token_path, np.array([1, 1, 1, 1], dtype=np.int32))
    args = parse_args(
        [
            "--stage",
            "pretrain",
            "--data_source",
            "tokenized",
            "--token_path",
            str(token_path),
            "--tokenizer_path",
            str(tokenizer_path),
            "--validation_split",
            "validation",
            "--seq_len",
            "4",
        ]
    )

    with pytest.raises(ValueError, match="validation_token_path"):
        build_mlm_loader(args, split="validation", seed=0)


def test_pretrain_backprop_tiny_smoke(tmp_path):
    transfer_checkpoint = tmp_path / "transfer.json"
    training_checkpoint = tmp_path / "training.json"
    args = parse_args(
        [
            "--stage",
            "pretrain",
            "--mode",
            "backprop",
            "--data_source",
            "tiny",
            "--seq_len",
            "6",
            "--embed_dim",
            "16",
            "--num_heads",
            "4",
            "--mlp_dim",
            "32",
            "--depth",
            "1",
            "--batch_size",
            "2",
            "--vocab_size",
            "16",
            "--max_steps",
            "1",
            "--checkpoint_interval",
            "1",
            "--validation_interval",
            "0",
            "--log_interval",
            "0",
            "--checkpoint",
            str(transfer_checkpoint),
            "--training_checkpoint",
            str(training_checkpoint),
        ]
    )

    run_pretrain(args)

    assert transfer_checkpoint.exists()
    assert transfer_checkpoint.with_suffix(".json.npz").exists()
    assert training_checkpoint.exists()


def test_finetune_backprop_synthetic_smoke(tmp_path):
    args = parse_args(
        [
            "--stage",
            "finetune",
            "--mode",
            "backprop",
            "--data_source",
            "synthetic",
            "--seq_len",
            "6",
            "--embed_dim",
            "16",
            "--num_heads",
            "4",
            "--mlp_dim",
            "32",
            "--depth",
            "1",
            "--batch_size",
            "2",
            "--num_epochs",
            "1",
            "--checkpoint",
            str(tmp_path / "missing.json"),
        ]
    )

    run_finetune(args)


def test_pretrain_tiny_bpe_validation_split_reuses_tokenizer_cache(tmp_path):
    pytest.importorskip("tokenizers")
    args = parse_args(
        [
            "--stage",
            "pretrain",
            "--mode",
            "backprop",
            "--data_source",
            "tiny",
            "--cache_dir",
            str(tmp_path / "cache"),
            "--seq_len",
            "6",
            "--embed_dim",
            "16",
            "--num_heads",
            "4",
            "--mlp_dim",
            "32",
            "--depth",
            "1",
            "--batch_size",
            "2",
            "--vocab_size",
            "128",
            "--max_steps",
            "1",
            "--checkpoint_interval",
            "1",
            "--validation_interval",
            "1",
            "--validation_split",
            "valid",
            "--overwrite_cache",
            "--log_interval",
            "0",
            "--checkpoint",
            str(tmp_path / "transfer.json"),
        ]
    )

    run_pretrain(args)


def test_tokenized_loader_checkpoint_vocab_includes_added_tokens(tmp_path):
    tokenizers = pytest.importorskip("tokenizers")
    tokenizer = tokenizers.Tokenizer(
        tokenizers.models.BPE(
            vocab={"[UNK]": 0, "warm": 1}, merges=[], unk_token="[UNK]"
        )
    )
    tokenizer.add_special_tokens(["[PAD]", "[MASK]", "[CLS]", "[SEP]"])
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    token_path = tmp_path / "tokens.npy"
    np.save(token_path, np.array([tokenizer.token_to_id("warm")], dtype=np.int32))
    args = parse_args(
        [
            "--stage",
            "pretrain",
            "--data_source",
            "tokenized",
            "--token_path",
            str(token_path),
            "--tokenizer_path",
            str(tokenizer_path),
            "--seq_len",
            "4",
            "--batch_size",
            "1",
        ]
    )

    loader, vocab, tokenizer_json = build_mlm_loader(args, split="train", seed=0)

    assert len(vocab) == loader.vocab_size
    assert "[CLS]" in vocab
    assert tokenizer_json is not None


def test_bpe_pretrain_checkpoint_can_transfer_to_finetune(tmp_path):
    pytest.importorskip("tokenizers")
    checkpoint = tmp_path / "transfer.json"
    pretrain_args = parse_args(
        [
            "--stage",
            "pretrain",
            "--mode",
            "backprop",
            "--data_source",
            "tiny",
            "--cache_dir",
            str(tmp_path / "cache"),
            "--seq_len",
            "6",
            "--embed_dim",
            "16",
            "--num_heads",
            "4",
            "--mlp_dim",
            "32",
            "--depth",
            "1",
            "--batch_size",
            "2",
            "--vocab_size",
            "128",
            "--max_steps",
            "1",
            "--validation_interval",
            "0",
            "--log_interval",
            "0",
            "--checkpoint",
            str(checkpoint),
        ]
    )
    run_pretrain(pretrain_args)

    finetune_args = parse_args(
        [
            "--stage",
            "finetune",
            "--mode",
            "backprop",
            "--data_source",
            "synthetic",
            "--batch_size",
            "2",
            "--num_epochs",
            "1",
            "--checkpoint",
            str(checkpoint),
        ]
    )

    run_finetune(finetune_args)


@pytest.mark.parametrize(
    "extra, message",
    [
        (["--mode", "backprop"], "--mode pc"),
        (["--mode", "pc", "--tokens_per_update", "64"], "--tokens_per_update"),
        (["--mode", "pc", "--micro_batch_size", "1"], "--micro_batch_size"),
    ],
)
def test_pc_incremental_flag_validation(extra, message):
    args = parse_args(["--stage", "pretrain", "--pc_incremental", *extra])

    assert args.pc_incremental
    with pytest.raises(ValueError, match=message):
        run_pretrain(args)


def test_pc_incremental_pretrain_tiny_smoke(capsys):
    args = parse_args(
        [
            "--stage",
            "pretrain",
            "--mode",
            "pc",
            "--pc_incremental",
            "--pc_infer_steps",
            "2",
            "--pc_infer_guard",
            "decay",
            "--seq_len",
            "6",
            "--embed_dim",
            "16",
            "--num_heads",
            "4",
            "--mlp_dim",
            "32",
            "--depth",
            "1",
            "--vocab_size",
            "16",
            "--max_steps",
            "2",
            "--log_interval",
            "1",
            "--validation_interval",
            "0",
            "--checkpoint",
            "",
        ]
    )

    run_pretrain(args)

    out = capsys.readouterr().out
    assert "step=2 masked_loss=" in out
    assert "energy_rises=" in out


def test_adam_eps_flag_damps_updates_from_tiny_gradients():
    import jax.numpy as jnp

    from examples.bert_demo import build_optimizer, parse_args

    assert parse_args(["--stage", "pretrain"]).adam_eps == 1e-8
    with pytest.raises(ValueError, match="--adam_eps"):
        parse_args(["--stage", "pretrain", "--adam_eps", "0"])

    params = {"w": jnp.zeros(3)}
    grads = {"w": jnp.full(3, 1e-8)}
    steps = {}
    for eps in (1e-8, 1e-6):
        optimizer = build_optimizer(
            learning_rate=1e-3,
            weight_decay=0.0,
            grad_clip=0.0,
            warmup_steps=0,
            total_steps=10,
            adam_eps=eps,
        )
        updates, _ = optimizer.update(grads, optimizer.init(params), params)
        steps[eps] = float(jnp.max(jnp.abs(updates["w"])))

    assert steps[1e-8] == pytest.approx(0.5e-3, rel=1e-2)
    assert steps[1e-6] < 0.02e-3


TINY_PC_PRETRAIN = [
    "--stage",
    "pretrain",
    "--mode",
    "pc",
    "--seq_len",
    "6",
    "--embed_dim",
    "16",
    "--num_heads",
    "4",
    "--mlp_dim",
    "32",
    "--depth",
    "1",
    "--vocab_size",
    "16",
    "--max_steps",
    "1",
    "--log_interval",
    "1",
    "--validation_interval",
    "0",
    "--checkpoint",
    "",
    "--mask_prob",
    "0.5",
]


def test_spiking_flags_parse_and_build_inference():
    from examples.bert_demo import build_pretrain_inference
    from fabricpc.core.inference import InferenceSGDNormClip, InferenceSpikingPrecision

    default = parse_args(["--stage", "pretrain", "--mode", "pc"])
    assert default.pc_inference == "sgd_clip"
    assert default.pc_spike_eta == 1.0
    assert default.pc_spike_offset == 0
    assert default.pc_spike_depth == "longest"
    assert default.pc_forward_updates == "none"
    assert type(build_pretrain_inference(default)) is InferenceSGDNormClip

    args = parse_args(
        [
            "--stage",
            "pretrain",
            "--mode",
            "pc",
            "--pc_inference",
            "spiking",
            "--pc_spike_eta",
            "0.5",
            "--pc_spike_offset",
            "2",
            "--pc_spike_depth",
            "shortest",
            "--pc_forward_updates",
            "final",
        ]
    )
    inference = build_pretrain_inference(args)
    assert isinstance(inference, InferenceSpikingPrecision)
    assert inference.config["spike_eta"] == 0.5
    assert inference.config["spike_offset"] == 2
    assert inference.config["depth_mode"] == "shortest"
    with pytest.raises(ValueError, match="--pc_spike_eta"):
        parse_args(["--stage", "pretrain", "--pc_spike_eta", "0"])
    with pytest.raises(ValueError, match="--pc_spike_offset"):
        parse_args(["--stage", "pretrain", "--pc_spike_offset", "-1"])


@pytest.mark.parametrize(
    "extra, message",
    [
        (["--pc_inference", "spiking", "--pc_infer_guard", "decay"], "guard"),
        (["--pc_spike_eta", "0.5"], "requires --pc_inference spiking"),
        (["--pc_spike_depth", "shortest"], "requires --pc_inference spiking"),
        (["--pc_forward_updates", "final", "--pc_incremental"], "--pc_incremental"),
        (
            [
                "--pc_inference",
                "spiking",
                "--pc_incremental",
                "--pc_spike_depth",
                "shortest",
            ],
            "--pc_spike_depth longest",
        ),
        (
            [
                "--pc_inference",
                "spiking",
                "--pc_incremental",
                "--pc_eta_infer",
                "0",
            ],
            "--pc_eta_infer",
        ),
    ],
)
def test_spiking_flag_validation(extra, message, monkeypatch):
    def fail_if_loading(*_args, **_kwargs):
        raise AssertionError("validation must run before loading data")

    monkeypatch.setattr("examples.bert_demo.build_mlm_loader", fail_if_loading)
    args = parse_args(["--stage", "pretrain", "--mode", "pc", *extra])

    with pytest.raises(ValueError, match=message):
        run_pretrain(args)


@pytest.mark.parametrize(
    "extra, message",
    [
        (["--mode", "backprop", "--pc_inference", "spiking"], "--mode pc"),
        (["--mode", "backprop", "--pc_forward_updates", "initial"], "--mode pc"),
    ],
)
def test_spiking_and_forward_updates_require_pc_mode(extra, message):
    with pytest.raises(ValueError, match=message):
        run_pretrain(parse_args(["--stage", "pretrain", *extra]))


@pytest.mark.parametrize(
    "extra",
    [["--pc_inference", "spiking"], ["--pc_forward_updates", "final"]],
)
def test_spiking_and_forward_updates_rejected_for_finetuning(extra):
    args = parse_args(["--stage", "finetune", "--mode", "backprop", *extra])

    with pytest.raises(ValueError, match="only valid for pretraining"):
        run_finetune(args)


def _captured_train_config(monkeypatch, argv):
    captured = {}

    def fake_train_mlm(params, structure, *_args, **_kwargs):
        captured["config"] = _args[2]
        captured["structure"] = structure
        return params, {"step": 0}

    monkeypatch.setattr("examples.bert_demo.train_mlm", fake_train_mlm)
    monkeypatch.setattr(
        "examples.bert_demo.report_pc_gradient_reach", lambda *a, **k: {}
    )
    run_pretrain(parse_args([*TINY_PC_PRETRAIN, *argv]))
    return captured


def test_forward_updates_recorded_only_when_set(monkeypatch):
    default = _captured_train_config(monkeypatch, [])
    assert "pc_forward_updates" not in default["config"]
    assert type(default["structure"].config["inference"]).__name__ == (
        "InferenceSGDNormClip"
    )

    spiking = _captured_train_config(
        monkeypatch,
        ["--pc_inference", "spiking", "--pc_forward_updates", "initial"],
    )
    assert spiking["config"]["pc_forward_updates"] == "initial"
    inference = spiking["structure"].config["inference"]
    assert type(inference).__name__ == "InferenceSpikingPrecision"
    assert inference.config["spike_eta"] == 1.0


def test_spiking_pretrain_prints_depths_and_energy_rises(capsys):
    args = parse_args(
        [
            *TINY_PC_PRETRAIN,
            "--pc_inference",
            "spiking",
            "--pc_infer_steps",
            "3",
            "--pc_eta_infer",
            "0.05",
            "--pc_forward_updates",
            "final",
        ]
    )

    run_pretrain(args)

    captured = capsys.readouterr()
    assert "pc_spike_depths mode=longest max_depth=6" in captured.out
    assert "deepest=token_embeddings steps_needed=6" in captured.out
    assert (
        "pc_spike_schedule final_layernorm=1@0 L0_mlp2=2@1 L0_mlp1=3@2 "
        "L0_mha=4@3" in captured.out
    )
    assert "token_embeddings=6@5" in captured.out
    assert "mlm_head=" not in captured.out.split("pc_spike_schedule")[1].split("\n")[0]
    assert "--pc_infer_steps 3 < 6" in captured.err
    assert "step=1 masked_loss=" in captured.out
    assert "energy_rises=" in captured.out


def test_spiking_incremental_pretrain_tiny_smoke(capsys):
    args = parse_args(
        [
            *TINY_PC_PRETRAIN,
            "--pc_inference",
            "spiking",
            "--pc_incremental",
            "--pc_infer_steps",
            "6",
            "--pc_eta_infer",
            "0.05",
        ]
    )

    run_pretrain(args)

    captured = capsys.readouterr()
    assert "steps_needed=6" in captured.out
    assert "warning" not in captured.err
    assert "energy_rises=" in captured.out


def test_spiking_probe_uses_forward_updates(capsys):
    args = parse_args(
        [
            *TINY_PC_PRETRAIN,
            "--pc_inference",
            "spiking",
            "--pc_infer_steps",
            "6",
            "--pc_eta_infer",
            "0.0",
            "--pc_forward_updates",
            "initial",
            "--pc_bp_probe",
            "1",
        ]
    )

    run_pretrain(args)

    out = capsys.readouterr().out
    head = next(line for line in out.splitlines() if "node=mlm_head" in line)
    mlp1 = next(line for line in out.splitlines() if "node=L0_mlp1" in line)
    assert "cosine_mean=1.0000" in head
    assert "cosine_mean=1.0000" in mlp1


def _tiny_pretrain_argv(tmp_path, *extra):
    return [
        "--stage",
        "pretrain",
        "--data_source",
        "tiny",
        "--cache_dir",
        str(tmp_path / "cache"),
        "--seq_len",
        "6",
        "--embed_dim",
        "16",
        "--num_heads",
        "4",
        "--mlp_dim",
        "32",
        "--depth",
        "1",
        "--batch_size",
        "2",
        "--vocab_size",
        "128",
        "--max_steps",
        "1",
        "--validation_interval",
        "0",
        "--log_interval",
        "0",
        *extra,
    ]


def _save_tiny_checkpoint(tmp_path, name="source.json", *extra):
    checkpoint = tmp_path / name
    run_pretrain(
        parse_args(
            _tiny_pretrain_argv(
                tmp_path, "--mode", "backprop", "--checkpoint", str(checkpoint), *extra
            )
        )
    )
    return checkpoint


def test_load_init_params_round_trips_checkpoint_params(tmp_path):
    pytest.importorskip("tokenizers")
    checkpoint = _save_tiny_checkpoint(tmp_path)
    saved = load_checkpoint(str(checkpoint))

    loaded = load_init_params(str(checkpoint), saved["params"], saved["vocab"])

    for node_name, node in saved["params"].nodes.items():
        for name, value in node.weights.items():
            np.testing.assert_array_equal(loaded.nodes[node_name].weights[name], value)


def test_load_init_params_rejects_vocab_mismatch(tmp_path):
    pytest.importorskip("tokenizers")
    checkpoint = _save_tiny_checkpoint(tmp_path)
    saved = load_checkpoint(str(checkpoint))
    vocab = dict(saved["vocab"])
    vocab["[EXTRA]"] = len(vocab)

    with pytest.raises(ValueError, match="vocab does not match"):
        load_init_params(str(checkpoint), saved["params"], vocab)


def test_init_checkpoint_rejects_architecture_mismatch(tmp_path):
    pytest.importorskip("tokenizers")
    checkpoint = _save_tiny_checkpoint(tmp_path)
    argv = _tiny_pretrain_argv(
        tmp_path, "--mode", "backprop", "--init_checkpoint", str(checkpoint)
    )
    argv[argv.index("--mlp_dim") + 1] = "64"

    with pytest.raises(ValueError, match="do not match the model"):
        run_pretrain(parse_args(argv))


def test_init_checkpoint_rejects_resume_from(tmp_path):
    args = parse_args(
        _tiny_pretrain_argv(
            tmp_path,
            "--mode",
            "backprop",
            "--init_checkpoint",
            "a.json",
            "--resume_from",
            "b.json",
        )
    )

    with pytest.raises(ValueError, match="cannot be combined"):
        run_pretrain(args)


@pytest.mark.parametrize("mode", ["backprop", "pc"])
def test_pretrain_starts_from_init_checkpoint(tmp_path, capsys, mode):
    pytest.importorskip("tokenizers")
    checkpoint = _save_tiny_checkpoint(tmp_path, "source.json", "--seed", "7")
    expected = summarize_init_diagnostics(load_checkpoint(str(checkpoint))["params"])
    capsys.readouterr()
    run_pretrain(parse_args(_tiny_pretrain_argv(tmp_path, "--mode", mode)))
    assert f"param_std={expected['param_std']:.6f}" not in capsys.readouterr().out

    run_pretrain(
        parse_args(
            _tiny_pretrain_argv(
                tmp_path, "--mode", mode, "--init_checkpoint", str(checkpoint)
            )
        )
    )

    out = capsys.readouterr().out
    assert f"param_mean={expected['param_mean']:.6f}" in out
    assert f"param_std={expected['param_std']:.6f}" in out


def test_load_init_params_keeps_fresh_parameterless_nodes(tmp_path, monkeypatch):
    pytest.importorskip("tokenizers")
    import examples.bert_demo as bert_demo

    checkpoint = _save_tiny_checkpoint(tmp_path)
    saved = load_checkpoint(str(checkpoint))
    params = saved["params"]
    empty = [
        n for n, node in params.nodes.items() if not node.weights and not node.biases
    ]
    assert empty
    pruned = dict(
        saved,
        params=type(params)(
            nodes={n: node for n, node in params.nodes.items() if n not in empty}
        ),
    )
    monkeypatch.setattr(bert_demo, "load_checkpoint", lambda path: pruned)

    loaded = load_init_params(str(checkpoint), params, saved["vocab"])

    assert list(loaded.nodes) == list(params.nodes)
