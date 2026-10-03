"""Smoke tests for examples/bert_demo.py (MLM pretraining demo)."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

DEMO_PATH = Path(__file__).resolve().parents[1] / "examples" / "bert_demo.py"
SMALL = ["--depth", "1", "--embed_dim", "16", "--num_heads", "2", "--mlp_dim", "32"]


@pytest.fixture(scope="module")
def demo():
    spec = importlib.util.spec_from_file_location("bert_demo_under_test", DEMO_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tiny_corpus_pretraining_reports_validation_metrics(demo):
    metrics = demo.main(SMALL + ["--num_epochs", "2"])

    assert metrics["num_masked_tokens"] > 0
    assert 0.0 <= metrics["masked_accuracy"] <= 1.0
    assert metrics["masked_loss"] > 0.0


def test_bpe_pretraining_from_text_files(demo, tmp_path):
    pytest.importorskip("tokenizers")
    train_file = tmp_path / "train.txt"
    val_file = tmp_path / "val.txt"
    train_file.write_text("\n".join(demo.TINY_TRAIN_TEXTS))
    val_file.write_text("\n".join(demo.TINY_VAL_TEXTS))

    metrics = demo.main(
        SMALL
        + ["--num_epochs", "1", "--tokenizer", "bpe"]
        + ["--vocab_size", "80"]
        + ["--train_file", str(train_file), "--val_file", str(val_file)]
    )

    assert metrics["num_masked_tokens"] > 0


def test_train_and_val_files_must_be_given_together(demo, tmp_path):
    with pytest.raises(SystemExit):
        demo.parse_args(["--train_file", str(tmp_path / "train.txt")])


def test_word_tokenizer_maps_unseen_validation_words_to_unk(demo):
    train_ids, val_ids, vocab = demo.word_tokenize(["the cat sat"], ["the dog sat"])

    assert "dog" not in vocab
    assert vocab["[UNK]"] in val_ids.tolist()
    assert train_ids[-1] == vocab["[SEP]"]


def test_too_little_text_fails_clearly(demo):
    args = demo.parse_args(SMALL)
    train_ids, _, vocab = demo.word_tokenize(["a short line"], ["a short line"])

    with pytest.raises(ValueError, match="no full batch"):
        demo.make_loader(train_ids, vocab, args, seed=0, shuffle=False)


@pytest.mark.parametrize("warmup_fraction", [-0.1, 1.0])
def test_optimizer_rejects_invalid_warmup_fraction(demo, warmup_fraction):
    args = SimpleNamespace(
        lr=1e-3, weight_decay=0.0, warmup_fraction=warmup_fraction, grad_clip=1.0
    )

    with pytest.raises(ValueError, match="warmup_fraction"):
        demo.build_optimizer(args, total_steps=10)


def test_optimizer_warms_up_then_decays(demo):
    import jax.numpy as jnp
    import optax

    args = SimpleNamespace(lr=1.0, weight_decay=0.0, warmup_fraction=0.5, grad_clip=0)
    optimizer = demo.build_optimizer(args, total_steps=4)
    params = {"w": jnp.zeros(())}
    state = optimizer.init(params)
    steps = []
    for _ in range(4):
        updates, state = optimizer.update({"w": jnp.ones(())}, state, params)
        steps.append(float(-updates["w"]))

    assert steps[0] == pytest.approx(0.0, abs=1e-6)
    assert steps[1] < steps[2]
    assert steps[3] < steps[2]
