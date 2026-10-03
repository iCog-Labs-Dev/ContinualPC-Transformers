"""Masked-language-model (MLM) data: BERT-style masking and batching.

``MlmTextBatcher`` turns one pre-tokenized token stream into fixed-length
windows and yields dict batches that ``create_encoder_mlm_model`` graphs train
on directly (``x``, ``y``, ``loss_mask``, ``attention_mask``).
``tokenize_texts_bpe`` builds such a stream from raw texts.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

from fabricpc.utils.data.dataloader import _train_bpe_tokenizer

MLM_SPECIAL_TOKENS = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]

# A batch that samples no masked positions is re-drawn (it would give zero loss
# but still advance momentum-based optimizers). After this many draws (only
# plausible for a tiny mask_prob) one eligible position is masked directly.
_MAX_MASK_REDRAWS = 100


def apply_mlm_mask(
    token_ids: np.ndarray,
    *,
    mask_token_id: int,
    vocab_size: int,
    special_token_ids: set[int],
    seed: int,
    mask_prob: float = 0.15,
    random_replacement_excludes_special: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply BERT-style 80/10/10 masking to a ``(batch, seq_len)`` token batch.

    Each non-special token is selected independently with ``mask_prob``.
    Selected tokens become ``[MASK]`` (80%), a random token (10%), or stay
    unchanged (10%).

    Returns:
        ``(masked_ids, targets, loss_mask)`` where ``targets`` are the original
        ids and ``loss_mask`` is a bool array marking the selected positions.
    """
    if token_ids.ndim != 2:
        raise ValueError(
            f"token_ids must have shape (batch, seq_len), got {token_ids.shape}"
        )
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if not 0.0 <= mask_prob <= 1.0:
        raise ValueError("mask_prob must be in [0, 1]")

    rng = np.random.default_rng(seed)
    eligible = ~np.isin(token_ids, list(special_token_ids))
    loss_mask = (rng.random(token_ids.shape) < mask_prob) & eligible

    policy = rng.random(token_ids.shape)
    if random_replacement_excludes_special:
        replacement_ids = np.array(
            [idx for idx in range(vocab_size) if idx not in special_token_ids],
            dtype=np.int32,
        )
        random_tokens = (
            rng.choice(replacement_ids, size=token_ids.shape)
            if replacement_ids.size
            else token_ids.copy()
        )
    else:
        random_tokens = rng.integers(0, vocab_size, size=token_ids.shape)

    masked = np.where(loss_mask & (policy < 0.8), mask_token_id, token_ids)
    masked = np.where(
        loss_mask & (policy >= 0.8) & (policy < 0.9), random_tokens, masked
    )
    return masked.astype(np.int32), token_ids.astype(np.int32), loss_mask


class MlmTextBatcher:
    """Batch a pre-tokenized stream into fixed-length MLM examples.

    Windows of ``seq_len`` tokens (``seq_len - 1`` plus a leading ``[CLS]`` when
    ``cls_token_id`` is set) start every ``stride`` tokens; short trailing
    windows are padded, and windows holding only special tokens (e.g. a lone
    trailing ``[SEP]``) are skipped. Each epoch reshuffles the windows (if
    ``shuffle``) and draws fresh masks; incomplete final batches are dropped.
    ``pad_token_id``, ``mask_token_id`` and ``cls_token_id`` must be listed in
    ``special_token_ids`` so they are never selected for masking.

    Yields dicts of numpy arrays: ``x`` (masked ids), ``y`` (original ids),
    and float32 ``loss_mask`` / ``attention_mask`` of shape (batch, seq_len).
    Every yielded batch has at least one masked position.
    """

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
        cls_token_id: int | None = None,
        seed: int = 0,
        mask_prob: float = 0.15,
        stride: int | None = None,
        shuffle: bool = True,
        max_sequences: int | None = None,
        random_replacement_excludes_special: bool = True,
    ):
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if not 0.0 < mask_prob <= 1.0:
            raise ValueError("mask_prob must be in (0, 1]")
        if cls_token_id is not None and seq_len < 2:
            raise ValueError("seq_len must be at least 2 when cls_token_id is set")

        required = {"pad_token_id": pad_token_id, "mask_token_id": mask_token_id}
        if cls_token_id is not None:
            required["cls_token_id"] = cls_token_id
        for name, token_id in required.items():
            if token_id not in special_token_ids:
                raise ValueError(f"{name}={token_id} must be in special_token_ids")

        self.stream_window = seq_len - 1 if cls_token_id is not None else seq_len
        self.stride = self.stream_window if stride is None else stride
        if self.stride <= 0:
            raise ValueError("stride must be positive")

        self.token_ids = np.asarray(token_ids, dtype=np.int32)
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.vocab_size = vocab_size
        self.special_token_ids = set(special_token_ids)
        self.cls_token_id = cls_token_id
        self.seed = seed
        self.mask_prob = mask_prob
        self.shuffle = shuffle
        self.random_replacement_excludes_special = random_replacement_excludes_special
        self._epoch = 0

        starts = self._maskable_window_starts()
        self.starts = starts if max_sequences is None else starts[:max_sequences]
        self._num_batches = len(self.starts) // batch_size

    def _maskable_window_starts(self) -> np.ndarray:
        """Window starts whose window contains at least one maskable token."""
        starts = np.arange(0, len(self.token_ids), self.stride, dtype=np.int64)
        eligible = ~np.isin(self.token_ids, list(self.special_token_ids))
        eligible_before = np.concatenate([[0], np.cumsum(eligible)])
        ends = np.minimum(starts + self.stream_window, len(self.token_ids))
        has_maskable = eligible_before[ends] - eligible_before[starts] > 0
        return starts[has_maskable].astype(np.int32)

    def __len__(self) -> int:
        return self._num_batches

    def state_dict(self) -> dict[str, int]:
        """Iteration state; restoring it replays the same windows and masks."""
        return {"epoch": self._epoch}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self._epoch = int(state.get("epoch", 0))

    def __iter__(self):
        starts = self.starts.copy()
        # Seed from (seed, epoch) so loaders with nearby seeds never share an
        # epoch's stream (seed + epoch would make seed=1/epoch 0 == seed=0/epoch 1).
        rng = np.random.default_rng([self.seed, self._epoch])
        if self.shuffle:
            rng.shuffle(starts)
        self._epoch += 1

        for batch_start in range(
            0, self._num_batches * self.batch_size, self.batch_size
        ):
            original, attention_mask = self._windows(
                starts[batch_start : batch_start + self.batch_size]
            )
            masked, targets, loss_mask = self._mask(original, rng)
            yield {
                "x": masked,
                "y": targets,
                "loss_mask": loss_mask.astype(np.float32),
                "attention_mask": attention_mask.astype(np.float32),
            }

    def _windows(self, batch_starts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        shape = (self.batch_size, self.seq_len)
        original = np.full(shape, self.pad_token_id, dtype=np.int32)
        attention_mask = np.zeros(shape, dtype=bool)
        offset = 0 if self.cls_token_id is None else 1
        if self.cls_token_id is not None:
            original[:, 0] = self.cls_token_id
        for row, start in enumerate(batch_starts):
            segment = self.token_ids[start : start + self.stream_window]
            original[row, offset : offset + len(segment)] = segment
            # Attend by position, not token value: a pad id inside real text
            # stays visible; only the padded tail is hidden.
            attention_mask[row, : offset + len(segment)] = True
        return original, attention_mask

    def _mask(self, original: np.ndarray, rng: np.random.Generator):
        for _ in range(_MAX_MASK_REDRAWS):
            masked, targets, loss_mask = apply_mlm_mask(
                original,
                mask_token_id=self.mask_token_id,
                vocab_size=self.vocab_size,
                special_token_ids=self.special_token_ids,
                seed=int(rng.integers(0, np.iinfo(np.int32).max)),
                mask_prob=self.mask_prob,
                random_replacement_excludes_special=(
                    self.random_replacement_excludes_special
                ),
            )
            if loss_mask.any():
                return masked, targets, loss_mask
        # Windows are pre-filtered to contain a maskable token, so one exists.
        eligible = np.argwhere(~np.isin(original, list(self.special_token_ids)))
        row, col = eligible[rng.integers(len(eligible))]
        loss_mask = np.zeros(original.shape, dtype=bool)
        loss_mask[row, col] = True
        masked = original.copy()
        masked[row, col] = self.mask_token_id
        return masked, original.copy(), loss_mask


def tokenize_texts_bpe(
    texts: Sequence[str],
    *,
    vocab_size: int,
    special_tokens: Iterable[str] = MLM_SPECIAL_TOKENS,
    tokenizer: Any = None,
) -> tuple[np.ndarray, Any]:
    """Encode texts into one BPE token stream, ``[SEP]``-separated.

    Trains a tokenizer on ``texts`` unless one is passed in (e.g. to encode a
    validation split with the training tokenizer). Requires ``tokenizers``.

    Returns:
        ``(token_ids, tokenizer)``
    """
    if tokenizer is None:
        tokenizer = _train_bpe_tokenizer(
            vocab_size=vocab_size, special_tokens=special_tokens, texts=texts
        )
    sep_id = tokenizer.token_to_id("[SEP]")
    ids: list[int] = []
    for text in texts:
        ids.extend(tokenizer.encode(text).ids)
        if sep_id is not None:
            ids.append(sep_id)
    return np.asarray(ids, dtype=np.int32), tokenizer
