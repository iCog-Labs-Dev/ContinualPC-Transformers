"""Masked-language-model data loaders for encoder pretraining."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

try:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.trainers import BpeTrainer

    _TOKENIZERS_AVAILABLE = True
except ImportError:
    _TOKENIZERS_AVAILABLE = False

_TOKENIZERS_INSTALL_HINT = (
    "MLM BPE loading requires the 'tokenizers' package. "
    "Install text dependencies with: pip install -e '.[text]'"
)


def _safe_cache_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def apply_mlm_mask(
    token_ids: np.ndarray,
    *,
    mask_token_id: int,
    vocab_size: int,
    special_token_ids: set[int],
    seed: int,
    mask_prob: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply BERT-style 80/10/10 MLM masking to a token batch."""
    if token_ids.ndim != 2:
        raise ValueError(
            f"token_ids must have shape (batch, seq_len), got {token_ids.shape}"
        )

    rng = np.random.default_rng(seed)
    targets = token_ids.copy()
    masked = token_ids.copy()
    valid_mask = ~np.isin(token_ids, list(special_token_ids))
    loss_mask = (rng.random(token_ids.shape) < mask_prob) & valid_mask

    for row_idx in range(token_ids.shape[0]):
        if np.any(loss_mask[row_idx]):
            continue
        candidates = np.flatnonzero(valid_mask[row_idx])
        if candidates.size > 0:
            loss_mask[row_idx, int(rng.choice(candidates))] = True

    policy = rng.random(token_ids.shape)
    random_tokens = rng.integers(0, vocab_size, size=token_ids.shape, dtype=np.int32)
    masked = np.where(loss_mask & (policy < 0.8), mask_token_id, masked)
    masked = np.where(
        loss_mask & (policy >= 0.8) & (policy < 0.9), random_tokens, masked
    )

    return masked.astype(np.int32), targets.astype(np.int32), loss_mask.astype(bool)


class MlmTextBatcher:
    """Batch a pre-tokenized stream into fixed-length MLM examples."""

    def __init__(
        self,
        *,
        token_ids: np.ndarray,
        seq_len: int,
        batch_size: int,
        pad_token_id: int,
        mask_token_id: int,
        vocab_size: int,
        special_token_ids: set[int],
        seed: int = 0,
        mask_prob: float = 0.15,
        stride: int | None = None,
        shuffle: bool = True,
        max_sequences: int | None = None,
    ):
        self.token_ids = np.asarray(token_ids, dtype=np.int32)
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.vocab_size = vocab_size
        self.special_token_ids = set(special_token_ids)
        self.seed = seed
        self.mask_prob = mask_prob
        self.stride = stride or seq_len
        self.shuffle = shuffle
        self._epoch = 0

        starts = list(range(0, len(self.token_ids), self.stride))
        if max_sequences is not None:
            starts = starts[:max_sequences]
        self.starts = np.array(starts, dtype=np.int32)
        self._num_batches = len(self.starts) // batch_size

    def __len__(self):
        return self._num_batches

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self._epoch}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self._epoch = int(state.get("epoch", 0))

    def __iter__(self):
        starts = self.starts.copy()
        rng = np.random.default_rng(self.seed + self._epoch)
        if self.shuffle:
            rng.shuffle(starts)
        self._epoch += 1

        for batch_start in range(0, len(starts), self.batch_size):
            batch_starts = starts[batch_start : batch_start + self.batch_size]
            if len(batch_starts) < self.batch_size:
                continue

            original = np.full(
                (self.batch_size, self.seq_len), self.pad_token_id, dtype=np.int32
            )
            attention_mask = np.zeros((self.batch_size, self.seq_len), dtype=bool)
            for row_idx, start in enumerate(batch_starts):
                segment = self.token_ids[start : start + self.seq_len]
                original[row_idx, : len(segment)] = segment
                attention_mask[row_idx, : len(segment)] = segment != self.pad_token_id

            masked, targets, loss_mask = apply_mlm_mask(
                original,
                mask_token_id=self.mask_token_id,
                vocab_size=self.vocab_size,
                special_token_ids=self.special_token_ids,
                seed=int(rng.integers(0, np.iinfo(np.int32).max)),
                mask_prob=self.mask_prob,
            )
            yield {
                "x": masked,
                "y": targets,
                "loss_mask": loss_mask,
                "attention_mask": attention_mask,
            }


def _load_hf_texts(
    dataset_name: str,
    split: str,
    text_field: str,
    streaming: bool,
    revision: str | None,
    max_texts: int | None,
    max_chars: int | None,
    max_total_chars: int | None,
) -> list[str]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "BabyLM/HuggingFace loading requires the 'datasets' package. "
            "Install with: pip install -e '.[text]' or pip install datasets"
        ) from exc

    dataset = load_dataset(
        dataset_name, split=split, streaming=streaming, revision=revision
    )
    texts: list[str] = []
    total_chars = 0
    for row in dataset:
        if not row.get(text_field):
            continue
        text = str(row[text_field])
        if max_chars is not None:
            text = text[:max_chars]
        if max_total_chars is not None:
            remaining = max_total_chars - total_chars
            if remaining <= 0:
                break
            text = text[:remaining]
        total_chars += len(text)
        texts.append(text)
        if max_texts is not None and len(texts) >= max_texts:
            break
    return texts


def prepare_bpe_tokenized_texts(
    *,
    texts: Sequence[str],
    cache_dir: str | Path,
    vocab_size: int,
    special_tokens: Sequence[str],
    token_stem: str = "tokens",
    overwrite: bool = False,
) -> tuple[np.ndarray, Tokenizer]:
    """Train/load a BPE tokenizer and encode texts into one token stream."""
    if not _TOKENIZERS_AVAILABLE:
        raise ImportError(_TOKENIZERS_INSTALL_HINT)

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    tokenizer_path = cache_path / "tokenizer.json"
    tokens_path = cache_path / f"{token_stem}.npy"

    if overwrite:
        for stale_tokens_path in cache_path.glob("tokens_*.npy"):
            stale_tokens_path.unlink()

    if tokenizer_path.exists() and tokens_path.exists() and not overwrite:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        return np.load(tokens_path, allow_pickle=False), tokenizer

    if tokenizer_path.exists() and not overwrite:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    else:
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=list(special_tokens))
        corpus_path = cache_path / "_corpus.txt"
        with corpus_path.open("w", encoding="utf-8") as fh:
            for text in texts:
                fh.write(text)
                fh.write("\n")
        tokenizer.train(files=[str(corpus_path)], trainer=trainer)
        corpus_path.unlink()
        tokenizer.save(str(tokenizer_path))

    ids: list[int] = []
    sep_id = tokenizer.token_to_id("[SEP]")
    for text in texts:
        ids.extend(tokenizer.encode(text).ids)
        if sep_id is not None:
            ids.append(sep_id)
    token_ids = np.asarray(ids, dtype=np.int32)
    np.save(tokens_path, token_ids)
    return token_ids, tokenizer


class BabyLmMlmLoader(MlmTextBatcher):
    """HuggingFace/BabyLM text loader that emits BPE MLM batches."""

    SPECIAL_TOKENS = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]

    def __init__(
        self,
        *,
        dataset_name: str,
        split: str,
        text_field: str,
        cache_dir: str,
        seq_len: int,
        batch_size: int,
        vocab_size: int = 16000,
        seed: int = 0,
        mask_prob: float = 0.15,
        stride: int | None = None,
        shuffle: bool = True,
        max_sequences: int | None = None,
        streaming: bool = False,
        revision: str | None = None,
        overwrite_cache: bool = False,
        max_texts: int | None = None,
        max_chars: int | None = None,
        max_total_chars: int | None = None,
    ):
        texts = _load_hf_texts(
            dataset_name,
            split,
            text_field,
            streaming,
            revision,
            max_texts,
            max_chars,
            max_total_chars,
        )
        token_ids, tokenizer = prepare_bpe_tokenized_texts(
            texts=texts,
            cache_dir=cache_dir,
            vocab_size=vocab_size,
            special_tokens=self.SPECIAL_TOKENS,
            token_stem=f"tokens_{_safe_cache_name(split)}",
            overwrite=overwrite_cache,
        )
        self.tokenizer = tokenizer
        self.vocab = tokenizer.get_vocab()
        pad_token_id = tokenizer.token_to_id("[PAD]")
        mask_token_id = tokenizer.token_to_id("[MASK]")
        if pad_token_id is None or mask_token_id is None:
            raise ValueError("Tokenizer must contain [PAD] and [MASK] special tokens")

        super().__init__(
            token_ids=token_ids,
            seq_len=seq_len,
            batch_size=batch_size,
            pad_token_id=pad_token_id,
            mask_token_id=mask_token_id,
            vocab_size=tokenizer.get_vocab_size(),
            special_token_ids={
                token_id
                for token in self.SPECIAL_TOKENS
                if (token_id := tokenizer.token_to_id(token)) is not None
            },
            seed=seed,
            mask_prob=mask_prob,
            stride=stride,
            shuffle=shuffle,
            max_sequences=max_sequences,
        )


def load_tokenized_mlm_loader(
    *,
    token_path: str,
    tokenizer_path: str,
    seq_len: int,
    batch_size: int,
    seed: int = 0,
    mask_prob: float = 0.15,
    stride: int | None = None,
    shuffle: bool = True,
    max_sequences: int | None = None,
) -> MlmTextBatcher:
    """Build an MLM batcher from pre-tokenized ids and a saved BPE tokenizer."""
    if not _TOKENIZERS_AVAILABLE:
        raise ImportError(_TOKENIZERS_INSTALL_HINT)

    tokenizer = Tokenizer.from_file(tokenizer_path)
    pad_token_id = tokenizer.token_to_id("[PAD]")
    mask_token_id = tokenizer.token_to_id("[MASK]")
    if pad_token_id is None or mask_token_id is None:
        raise ValueError("Tokenizer must contain [PAD] and [MASK] special tokens")
    special_ids = {
        token_id
        for token in BabyLmMlmLoader.SPECIAL_TOKENS
        if (token_id := tokenizer.token_to_id(token)) is not None
    }
    return MlmTextBatcher(
        token_ids=np.load(token_path, allow_pickle=False),
        seq_len=seq_len,
        batch_size=batch_size,
        pad_token_id=pad_token_id,
        mask_token_id=mask_token_id,
        vocab_size=tokenizer.get_vocab_size(),
        special_token_ids=special_ids,
        seed=seed,
        mask_prob=mask_prob,
        stride=stride,
        shuffle=shuffle,
        max_sequences=max_sequences,
    )
