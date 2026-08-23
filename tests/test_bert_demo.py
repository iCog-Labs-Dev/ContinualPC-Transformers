"""Tests for the unified BERT demo entrypoint."""

import os
import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest

from examples.bert_demo import (
    PC_UNSUPPORTED_MESSAGE,
    build_mlm_loader,
    normalize_hf_dataset_args,
    parse_args,
    run_finetune,
    run_pretrain,
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


def test_pc_pretrain_fails_clearly():
    args = parse_args(["--stage", "pretrain", "--mode", "pc"])

    with pytest.raises(NotImplementedError, match=PC_UNSUPPORTED_MESSAGE):
        run_pretrain(args)


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
