"""Tests for the unified BERT demo entrypoint."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest

from examples.bert_demo import (
    PC_UNSUPPORTED_MESSAGE,
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
