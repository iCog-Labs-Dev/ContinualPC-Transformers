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
