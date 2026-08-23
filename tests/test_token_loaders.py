"""Tests for the shared _TokenSequenceLoader base: sliding-window batching and
the integer-target contract. BPE test is tokenizers-gated."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

from fabricpc.utils.data.dataloader import (
    _TokenSequenceLoader,
    BpeDataLoader,
    MlmTextBatcher,
    apply_mlm_mask,
    prepare_bpe_tokenized_texts,
)


class _ArrayTokenLoader(_TokenSequenceLoader):
    """Minimal concrete loader over an in-memory token array."""

    def __init__(
        self,
        data,
        vocab_size,
        seq_len,
        batch_size,
        shuffle=False,
        seed=0,
        max_samples=None,
    ):
        self.data = np.asarray(data, dtype=np.int32)
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self._init_sequence_indexing(max_samples)


def test_sequence_indexing_counts():
    loader = _ArrayTokenLoader(np.arange(100), vocab_size=100, seq_len=8, batch_size=4)
    assert loader.num_sequences == 100 - 8
    assert len(loader) == (100 - 8) // 4


def test_max_samples_caps_sequences():
    loader = _ArrayTokenLoader(
        np.arange(1000), vocab_size=50, seq_len=8, batch_size=4, max_samples=20
    )
    assert loader.num_sequences == 20
    assert len(loader) == 20 // 4


def test_yields_integer_targets_not_onehot():
    # Targets are integer ids (batch, seq_len); one-hot happens in
    # the training step, not the loader.
    loader = _ArrayTokenLoader(np.arange(50), vocab_size=50, seq_len=5, batch_size=2)
    x, y = next(iter(loader))
    assert x.dtype == np.int32 and x.shape == (2, 5)
    assert y.dtype == np.int32 and y.shape == (2, 5)


def test_next_token_alignment():
    data = np.arange(50, dtype=np.int32)
    loader = _ArrayTokenLoader(
        data, vocab_size=50, seq_len=5, batch_size=2, shuffle=False
    )
    x, y = next(iter(loader))
    assert np.array_equal(x[0], data[0:5])
    assert np.array_equal(y[0], data[1:6])
    assert np.array_equal(x[1], data[1:6])
    assert np.array_equal(y[1], data[2:7])


def test_drops_incomplete_last_batch():
    loader = _ArrayTokenLoader(
        np.arange(20), vocab_size=20, seq_len=4, batch_size=3, shuffle=False
    )
    batches = list(loader)
    assert len(batches) == (20 - 4) // 3
    assert all(x.shape[0] == 3 for x, _ in batches)


def test_bpe_dataloader_from_cache(tmp_path):
    """BpeDataLoader loads from a pre-built cache and yields
    integer targets. Skipped if the optional `tokenizers` dep is absent."""
    pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace

    bpe_dir = tmp_path / "bpe"
    bpe_dir.mkdir()

    corpus = tmp_path / "corpus.txt"
    corpus.write_text("to be or not to be that is the question " * 50, encoding="utf-8")
    tok = Tokenizer(BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    tok.train([str(corpus)], BpeTrainer(vocab_size=60, special_tokens=["[UNK]"]))
    tok.save(str(bpe_dir / "tokenizer.json"))

    ids = np.tile(
        np.array(
            tok.encode("to be or not to be that is the question").ids, dtype=np.int32
        ),
        20,
    )
    for split in ("train", "validation", "test"):
        np.save(bpe_dir / f"{split}.npy", ids)

    loader = BpeDataLoader(
        "train", seq_len=4, batch_size=2, shuffle=False, bpe_data_dir=str(bpe_dir)
    )
    assert loader.vocab_size == tok.get_vocab_size()
    assert loader.num_sequences == ids.size - 4  # len(data) - seq_len
    x, y = next(iter(loader))
    assert x.dtype == np.int32 and x.shape == (2, 4)
    assert y.dtype == np.int32 and y.shape == (2, 4)
    assert isinstance(loader.decode(x[0]), str)

def test_prepare_fits_tokenizer_on_train_split_only(tmp_path, monkeypatch):
    """_prepare fits vocab/merges on train only; a symbol appearing solely in
    validation/test must never enter the vocabulary, and all splits still cache."""
    pytest.importorskip("tokenizers")

    split_text = {
        "train": "alpha beta alpha beta alpha beta ",
        "validation": "alpha zeta zeta zeta ",
        "test": "beta zeta zeta zeta ",
    }
    monkeypatch.setattr(
        BpeDataLoader,
        "_load_split_text",
        staticmethod(lambda dataset, split: split_text[split]),
    )

    bpe_dir = tmp_path / "bpe"
    BpeDataLoader._prepare(bpe_dir, dataset="dummy", vocab_size=50, verbose=False)

    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(bpe_dir / "tokenizer.json"))
    # 'z' occurs only in held-out splits -> absent from a train-only fit
    assert not any("z" in token for token in tok.get_vocab())
    for split in ("train", "validation", "test"):
        assert (bpe_dir / f"{split}.npy").exists()


def test_apply_mlm_mask_has_no_forced_mask_fallback():
    tokens = np.array([[1, 5, 6, 2]], dtype=np.int32)

    _, _, loss_mask = apply_mlm_mask(
        tokens,
        mask_token_id=3,
        vocab_size=10,
        special_token_ids={0, 1, 2, 3},
        seed=0,
        mask_prob=0.0,
    )

    assert not np.any(loss_mask)


def test_mlm_text_batcher_allows_zero_mask_probability():
    batcher = MlmTextBatcher(
        token_ids=np.array([1, 5, 6, 7, 2, 1, 8, 9, 10, 2], dtype=np.int32),
        seq_len=5,
        batch_size=2,
        pad_token_id=0,
        mask_token_id=3,
        vocab_size=16,
        special_token_ids={0, 1, 2, 3},
        seed=0,
        mask_prob=0.0,
        shuffle=False,
    )

    batch = next(iter(batcher))

    assert not np.any(batch["loss_mask"])


def test_apply_mlm_mask_random_replacements_exclude_special_tokens():
    tokens = np.tile(np.array([[4, 5, 6, 7]], dtype=np.int32), (20, 1))

    masked, targets, loss_mask = apply_mlm_mask(
        tokens,
        mask_token_id=3,
        vocab_size=8,
        special_token_ids={0, 1, 2, 3},
        seed=1,
        mask_prob=1.0,
        random_replacement_excludes_special=True,
    )

    random_replacements = loss_mask & (masked != targets) & (masked != 3)
    assert np.any(random_replacements)
    assert not np.any(np.isin(masked[random_replacements], [0, 1, 2, 3]))


def test_mlm_text_batcher_prefixes_cls_and_uses_stream_stride():
    batcher = MlmTextBatcher(
        token_ids=np.arange(10, 20, dtype=np.int32),
        seq_len=5,
        batch_size=2,
        pad_token_id=0,
        mask_token_id=3,
        vocab_size=32,
        special_token_ids={0, 1, 2, 3},
        cls_token_id=1,
        seed=0,
        mask_prob=1.0,
        shuffle=False,
    )

    batch = next(iter(batcher))

    assert np.array_equal(batch["y"][:, 0], np.array([1, 1], dtype=np.int32))
    assert np.array_equal(batch["x"][:, 0], np.array([1, 1], dtype=np.int32))
    assert not np.any(batch["loss_mask"][:, 0])
    assert np.all(batch["attention_mask"][:, 0])
    assert np.array_equal(batch["y"][0, 1:], np.array([10, 11, 12, 13]))
    assert np.array_equal(batch["y"][1, 1:], np.array([14, 15, 16, 17]))


def test_prepare_bpe_tokenized_texts_rejects_mismatched_cache_manifest(tmp_path):
    pytest.importorskip("tokenizers")

    prepare_bpe_tokenized_texts(
        texts=["warm happy movie"],
        cache_dir=tmp_path,
        vocab_size=32,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
        token_stem="tokens_train",
        cache_metadata={"dataset_name": "demo", "split": "train"},
    )

    with pytest.raises(ValueError, match="overwrite_cache=True"):
        prepare_bpe_tokenized_texts(
            texts=["warm happy movie"],
            cache_dir=tmp_path,
            vocab_size=64,
            special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
            token_stem="tokens_train",
            cache_metadata={"dataset_name": "demo", "split": "train"},
        )


def test_prepare_bpe_overwrites_split_tokens_without_retraining_tokenizer(tmp_path):
    pytest.importorskip("tokenizers")
    special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    prepare_bpe_tokenized_texts(
        texts=["warm happy movie"],
        cache_dir=tmp_path,
        vocab_size=32,
        special_tokens=special_tokens,
        token_stem="tokens_train",
        cache_metadata={"dataset_name": "demo"},
        overwrite=True,
    )
    tokenizer_before = (tmp_path / "tokenizer.json").read_text()

    prepare_bpe_tokenized_texts(
        texts=["zzzz yyyy xxxx"],
        cache_dir=tmp_path,
        vocab_size=32,
        special_tokens=special_tokens,
        token_stem="tokens_valid",
        cache_metadata={"dataset_name": "demo"},
        overwrite=True,
        overwrite_tokenizer=False,
    )

    assert (tmp_path / "tokenizer.json").read_text() == tokenizer_before
    assert (tmp_path / "tokens_valid.npy").exists()
