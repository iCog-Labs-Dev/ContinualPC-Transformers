"""Tests for the character-level sequential text benchmark."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

from fabricpc.experiments.forgetting_benchmark.benchmark import (
    ASCII_VOCAB_SIZE,
    CharacterTokenizer,
    SequentialBenchmark,
    TASK_REGISTRY,
    TextTaskDataset,
    preprocess_ascii_text,
)


def test_character_tokenizer_roundtrips_known_characters():
    tokenizer = CharacterTokenizer()
    tokenizer.train(["To be, or not to be!\n"])

    assert tokenizer.decode(tokenizer.encode("To be!\n")) == "To be!\n"
    assert tokenizer.get_vocab_size() == ASCII_VOCAB_SIZE


def test_character_tokenizer_uses_unknown_token_for_test_only_character():
    tokenizer = CharacterTokenizer()
    tokenizer.train(["abc"])

    assert tokenizer.encode("a漢")[-1] == tokenizer.token_to_idx["<unk>"]
    assert tokenizer.decode(tokenizer.encode("a漢")) == "a�"


def test_preprocess_ascii_text_normalizes_unicode_and_removes_artifacts():
    cleaned, report = preprocess_ascii_text(
        "\u201cCaf\u00e9\u201d\u2014\ufb01anc\u00e9\u2026\r\nMojibake: \u00e2\u20ac\u2122 \u6771\u4eac\x00"
    )

    assert cleaned == "\"Cafe\"-fiance...\nMojibake: '"
    assert all(ord(character) < 128 for character in cleaned)
    assert report["normalization"] == "ascii_v1"
    assert report["non_ascii_input_characters"] == 12
    assert report["output_characters"] == len(cleaned)
    assert preprocess_ascii_text("\u0153uvre")[0] == "oeuvre"


def test_sequential_benchmark_writes_versioned_cleaned_cache(tmp_path, monkeypatch):
    fixture = {"train": "Caf\u00e9 \u6771\u4eac", "test": "\u201cTest\u201d"}

    def load_fixture(_data_dir, _max_chars):
        return fixture["train"], fixture["test"]

    monkeypatch.setitem(TASK_REGISTRY, "fixture", load_fixture)
    benchmark = SequentialBenchmark(
        ["fixture"], data_dir=str(tmp_path), max_chars_per_task=20
    )

    assert benchmark.vocab_size == ASCII_VOCAB_SIZE
    assert benchmark.preprocessing_manifest["profile"] == "ascii_v1"
    artifact_dir = tmp_path / benchmark.preprocessing_manifest["tasks"]["fixture"][
        "artifact_dir"
    ]
    assert (artifact_dir / "first_20_chars_train.txt").read_text(encoding="utf-8") == "Cafe"
    assert (artifact_dir / "first_20_chars_test.txt").read_text(encoding="utf-8") == '"Test"'
    first_report = benchmark.preprocessing_manifest["tasks"]["fixture"]["artifacts"]

    cached = SequentialBenchmark(["fixture"], data_dir=str(tmp_path), max_chars_per_task=20)
    assert cached.preprocessing_manifest["tasks"]["fixture"]["cache_hit"] is True
    assert cached.preprocessing_manifest["tasks"]["fixture"]["artifacts"] == first_report

    (artifact_dir / "first_20_chars_train.txt").write_text("corrupted")
    repaired = SequentialBenchmark(
        ["fixture"], data_dir=str(tmp_path), max_chars_per_task=20
    )
    assert repaired.preprocessing_manifest["tasks"]["fixture"]["cache_hit"] is False
    assert (artifact_dir / "first_20_chars_train.txt").read_text() == "Cafe"

    fixture["train"] = "na\u00efve"
    refreshed = SequentialBenchmark(
        ["fixture"], data_dir=str(tmp_path), max_chars_per_task=20
    )
    assert refreshed.preprocessing_manifest["tasks"]["fixture"]["cache_hit"] is False
    assert refreshed._train_texts["fixture"] == "naive"


def test_text_loader_yields_shifted_integer_batches():
    tokenizer = CharacterTokenizer()
    text = "the cat sat on the mat " * 100
    tokenizer.train([text])
    loader = TextTaskDataset(
        text,
        tokenizer,
        seq_len=8,
        batch_size=2,
        shuffle=False,
        seed=None,
        max_samples=10,
    )

    x, y = next(iter(loader))
    assert x.dtype == np.int32 and x.shape == (2, 8)
    assert np.array_equal(y[0, :-1], x[0, 1:])
    assert len(loader) == 5


def test_registry_contains_only_the_baseline_tasks():
    assert set(TASK_REGISTRY) == {"tiny_stories", "wikitext2", "tiny_shakespeare"}


def test_unknown_task_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown task"):
        SequentialBenchmark(["not-a-task"], data_dir=str(tmp_path))


def test_sequential_benchmark_uses_one_character_vocabulary(tmp_path):
    benchmark = SequentialBenchmark.__new__(SequentialBenchmark)
    benchmark.tasks = ["task_a", "task_b"]
    benchmark.seq_len = 8
    benchmark.batch_size = 2
    benchmark.seed = 42
    benchmark.max_samples_per_task = 10
    benchmark._train_texts = {"task_a": "abc " * 100, "task_b": "xyz " * 100}
    benchmark._test_texts = {"task_a": "abc " * 50, "task_b": "xyz " * 50}
    benchmark.tokenizer = CharacterTokenizer()
    benchmark.tokenizer.train(list(benchmark._train_texts.values()))

    assert (
        benchmark.get_train_loader(0).vocab_size
        == benchmark.get_train_loader(1).vocab_size
    )
    assert benchmark.get_task_name(1) == "task_b"
