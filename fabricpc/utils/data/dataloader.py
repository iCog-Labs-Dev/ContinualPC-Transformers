import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from fabricpc.utils.data.data_utils import one_hot, split_np_seed

try:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace

    _TOKENIZERS_AVAILABLE = True
except ImportError:  # optional dependency (BPE only)
    _TOKENIZERS_AVAILABLE = False

_TOKENIZERS_INSTALL_HINT = (
    "BpeDataLoader requires the 'tokenizers' package. "
    "Install the text-data extras with: pip install -e '.[text]'"
)

MLM_SPECIAL_TOKENS = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]


def _safe_cache_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def _manifest_payload(
    *,
    vocab_size: int,
    special_tokens: Sequence[str],
    cache_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "vocab_size": int(vocab_size),
        "special_tokens": list(special_tokens),
        "cache_metadata": cache_metadata or {},
    }


def _validate_manifest(
    manifest_path: Path,
    expected: dict[str, Any],
    *,
    artifact_name: str,
) -> None:
    if not manifest_path.exists():
        raise ValueError(
            f"Cached {artifact_name} is missing its manifest; rerun with "
            "overwrite_cache=True"
        )
    actual = json.loads(manifest_path.read_text())
    if actual != expected:
        raise ValueError(
            f"Cached {artifact_name} manifest does not match requested settings; "
            "rerun with overwrite_cache=True"
        )


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
    """Apply BERT-style 80/10/10 masking to a token batch.

    Eligible tokens are sampled independently; there is no forced fallback when
    a short row or batch happens to select no masked positions.
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
    targets = token_ids.copy()
    masked = token_ids.copy()
    valid_mask = ~np.isin(token_ids, list(special_token_ids))
    loss_mask = (rng.random(token_ids.shape) < mask_prob) & valid_mask

    policy = rng.random(token_ids.shape)
    if random_replacement_excludes_special:
        replacement_ids = np.array(
            [idx for idx in range(vocab_size) if idx not in special_token_ids],
            dtype=np.int32,
        )
        if replacement_ids.size == 0:
            random_tokens = token_ids.copy()
        else:
            random_tokens = rng.choice(replacement_ids, size=token_ids.shape)
    else:
        random_tokens = rng.integers(
            0, vocab_size, size=token_ids.shape, dtype=np.int32
        )

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
        cls_token_id: int | None = None,
        seed: int = 0,
        mask_prob: float = 0.15,
        stride: int | None = None,
        shuffle: bool = True,
        max_sequences: int | None = None,
        random_replacement_excludes_special: bool = True,
        token_unit: str = "unknown",
    ):
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if not 0.0 <= mask_prob <= 1.0:
            raise ValueError("mask_prob must be in [0, 1]")
        if token_unit not in {"word", "subword", "unknown"}:
            raise ValueError("token_unit must be 'word', 'subword', or 'unknown'")
        if cls_token_id is not None and seq_len < 2:
            raise ValueError("seq_len must be at least 2 when cls_token_id is set")

        stream_window = seq_len - 1 if cls_token_id is not None else seq_len
        effective_stride = stream_window if stride is None else stride
        if effective_stride <= 0:
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
        self.stride = effective_stride
        self.stream_window = stream_window
        self.shuffle = shuffle
        self.random_replacement_excludes_special = random_replacement_excludes_special
        self.source_token_count = int(np.sum(self.token_ids != self.pad_token_id))
        self.token_unit = token_unit
        self._epoch = 0

        starts = list(range(0, len(self.token_ids), self.stride))
        if max_sequences is not None:
            starts = starts[:max_sequences]
        self.starts = np.array(starts, dtype=np.int32)
        self._num_batches = len(self.starts) // batch_size
        self.num_windows = int(len(self.starts))
        self.num_full_batches = int(self._num_batches)
        self.max_tokens_per_batch = int(self.batch_size * self.seq_len)
        self.max_tokens_per_epoch = int(
            self.num_full_batches * self.max_tokens_per_batch
        )

    def __len__(self):
        return self._num_batches

    @property
    def metadata(self) -> dict[str, int | str]:
        return {
            "source_token_count": self.source_token_count,
            "token_unit": self.token_unit,
            "num_windows": self.num_windows,
            "num_full_batches": self.num_full_batches,
            "max_tokens_per_batch": self.max_tokens_per_batch,
            "max_tokens_per_epoch": self.max_tokens_per_epoch,
            "pad_token_id": int(self.pad_token_id),
        }

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
                segment = self.token_ids[start : start + self.stream_window]
                if self.cls_token_id is None:
                    original[row_idx, : len(segment)] = segment
                    attention_mask[row_idx, : len(segment)] = (
                        segment != self.pad_token_id
                    )
                else:
                    original[row_idx, 0] = self.cls_token_id
                    attention_mask[row_idx, 0] = True
                    original[row_idx, 1 : 1 + len(segment)] = segment
                    attention_mask[row_idx, 1 : 1 + len(segment)] = (
                        segment != self.pad_token_id
                    )

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
            yield {
                "x": masked,
                "y": targets,
                "loss_mask": loss_mask,
                "attention_mask": attention_mask,
            }


def prepare_bpe_tokenized_texts(
    *,
    texts: Sequence[str],
    cache_dir: str | Path,
    vocab_size: int,
    special_tokens: Sequence[str],
    token_stem: str = "tokens",
    overwrite: bool = False,
    overwrite_tokenizer: bool = True,
    cache_metadata: dict[str, Any] | None = None,
) -> tuple[np.ndarray, Any]:
    """Train/load a BPE tokenizer and encode texts into one token stream."""
    if not _TOKENIZERS_AVAILABLE:
        raise ImportError(_TOKENIZERS_INSTALL_HINT)

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    tokenizer_path = cache_path / "tokenizer.json"
    tokenizer_manifest_path = cache_path / "tokenizer.manifest.json"
    tokens_path = cache_path / f"{token_stem}.npy"
    tokens_manifest_path = cache_path / f"{token_stem}.manifest.json"
    manifest = _manifest_payload(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        cache_metadata=cache_metadata,
    )

    if overwrite:
        paths = [tokens_path, tokens_manifest_path]
        if overwrite_tokenizer:
            paths.extend([tokenizer_path, tokenizer_manifest_path])
        for path in paths:
            if path.exists():
                path.unlink()

    can_use_existing_tokenizer = tokenizer_path.exists() and (
        not overwrite or not overwrite_tokenizer
    )

    if can_use_existing_tokenizer and tokens_path.exists() and not overwrite:
        _validate_manifest(
            tokenizer_manifest_path, manifest, artifact_name="BPE tokenizer"
        )
        _validate_manifest(tokens_manifest_path, manifest, artifact_name="token stream")
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        return np.load(tokens_path, allow_pickle=False), tokenizer

    if can_use_existing_tokenizer:
        _validate_manifest(
            tokenizer_manifest_path, manifest, artifact_name="BPE tokenizer"
        )
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
        tokenizer_manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True)
        )

    ids: list[int] = []
    sep_id = tokenizer.token_to_id("[SEP]")
    for text in texts:
        ids.extend(tokenizer.encode(text).ids)
        if sep_id is not None:
            ids.append(sep_id)
    token_ids = np.asarray(ids, dtype=np.int32)
    np.save(tokens_path, token_ids)
    tokens_manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return token_ids, tokenizer


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
    cls_token_id = tokenizer.token_to_id("[CLS]")
    if pad_token_id is None or mask_token_id is None:
        raise ValueError("Tokenizer must contain [PAD] and [MASK] special tokens")
    special_ids = {
        token_id
        for token in MLM_SPECIAL_TOKENS
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
        cls_token_id=cls_token_id,
        seed=seed,
        mask_prob=mask_prob,
        stride=stride,
        shuffle=shuffle,
        max_sequences=max_sequences,
        token_unit="subword",
    )


class MnistLoader:
    """JAX-compatible data loader using TensorFlow Datasets.

    Provides the same interface as PyTorch DataLoader but uses tfds
    data parallelism based on C++ that bypasses GIL and does not inherit GPU state.
    Avoids os.fork warnings with JAX.

    Args:
        split: Dataset split to load. Use 'train' for training data or
               'test' for test data. Also supports slicing syntax like
               'train[:80%]' or 'train[80%:]' for custom splits.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle the data each epoch.
        seed: Random seed for reproducibility. When set, ensures deterministic
              shuffling across runs and machines. If None, shuffling is random.
        normalize_mean: Mean for normalization (default: MNIST mean).
        normalize_std: Std for normalization (default: MNIST std).
    """

    def __init__(
        self,
        split: str,
        batch_size: int,
        shuffle: bool = True,
        seed: int = None,
        tensor_format: str = "NHWC",  # image tensor 'flat' or 'NHWC' batch-height-width-channels
        normalize_mean: float = 0.1307,
        normalize_std: float = 0.3081,
    ):
        import tensorflow_datasets as tfds
        import tensorflow as tf

        # Disable GPU for TensorFlow (we only use it for data loading)
        tf.config.set_visible_devices([], "GPU")

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.tensor_format = tensor_format
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std

        # Split seed into two independent seeds for file and buffer shuffling
        file_seed, buffer_seed = split_np_seed(seed, n=2)

        # Configure read options for reproducibility
        read_config = tfds.ReadConfig(
            shuffle_seed=file_seed,
            interleave_cycle_length=1,  # Sequential reading for determinism
        )

        # Load dataset with pinned version for cross-machine reproducibility
        ds, info = tfds.load(
            "mnist:3.0.1",
            split=split,
            with_info=True,
            as_supervised=True,
            read_config=read_config,
            shuffle_files=shuffle and seed is not None,
        )
        self.num_examples = info.splits[split].num_examples
        self._num_batches = (self.num_examples + batch_size - 1) // batch_size

        # Build pipeline
        if shuffle:
            ds = ds.shuffle(
                buffer_size=self.num_examples, seed=buffer_seed
            )  # mnist fits in memory (~60MB) so the buffer is the full dataset
        ds = ds.batch(batch_size, drop_remainder=False)
        ds = ds.prefetch(tf.data.AUTOTUNE)

        self.ds = ds

    def __iter__(self):
        for images, labels in self.ds:
            # Convert to numpy, normalize, and flatten
            images = images.numpy().astype(np.float32) / 255.0
            images = (images - self.normalize_mean) / self.normalize_std

            # images shape is (Batch, 28, 28, 1)
            if self.tensor_format == "flat":
                images = images.reshape(images.shape[0], -1)  # Flatten to (Batch, 784)

            # One-hot encode labels
            labels = one_hot(labels.numpy(), num_classes=10)

            yield images, labels

    def __len__(self):
        return self._num_batches


class Cifar100Loader:
    """JAX-compatible CIFAR-100 data loader using TensorFlow Datasets.

    Loads the CIFAR-100 dataset (32x32 RGB images, 100 fine-grained classes)
    and yields batches of (images, one_hot_labels).

    Args:
        split: Dataset split to load ('train' or 'test').
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle the data each epoch.
        seed: Random seed for reproducibility.
        tensor_format: 'NHWC' for (batch, 32, 32, 3) or 'flat' for (batch, 3072).
        normalize_mean: Per-channel mean for normalization (default: CIFAR-100 mean).
        normalize_std: Per-channel std for normalization (default: CIFAR-100 std).
    """

    def __init__(
        self,
        split: str,
        batch_size: int,
        shuffle: bool = True,
        seed: int = None,
        tensor_format: str = "NHWC",
        normalize_mean: tuple = (0.5071, 0.4867, 0.4408),
        normalize_std: tuple = (0.2675, 0.2565, 0.2761),
    ):
        import tensorflow_datasets as tfds
        import tensorflow as tf

        tf.config.set_visible_devices([], "GPU")

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.tensor_format = tensor_format
        self.normalize_mean = np.array(normalize_mean, dtype=np.float32)
        self.normalize_std = np.array(normalize_std, dtype=np.float32)

        file_seed, buffer_seed = split_np_seed(seed, n=2)

        read_config = tfds.ReadConfig(
            shuffle_seed=file_seed,
            interleave_cycle_length=1,
        )

        ds, info = tfds.load(
            "cifar100",
            split=split,
            with_info=True,
            as_supervised=True,
            read_config=read_config,
            shuffle_files=shuffle and seed is not None,
        )
        self.num_examples = info.splits[split].num_examples
        self._num_batches = (self.num_examples + batch_size - 1) // batch_size

        if shuffle:
            ds = ds.shuffle(buffer_size=self.num_examples, seed=buffer_seed)
        ds = ds.batch(batch_size, drop_remainder=False)
        ds = ds.prefetch(tf.data.AUTOTUNE)

        self.ds = ds

    def __iter__(self):
        for images, labels in self.ds:
            images = images.numpy().astype(np.float32) / 255.0
            # Per-channel normalization: (H, W, C) broadcast
            images = (images - self.normalize_mean) / self.normalize_std

            if self.tensor_format == "flat":
                images = images.reshape(images.shape[0], -1)

            labels = one_hot(labels.numpy(), num_classes=100)

            yield images, labels

    def __len__(self):
        return self._num_batches


class Cifar10Loader:
    """JAX-compatible CIFAR-10 data loader using TensorFlow Datasets.

    Loads the CIFAR-10 dataset (32x32 RGB images, 10 classes)
    and yields batches of (images, one_hot_labels).

    Args:
        split: Dataset split to load ('train' or 'test').
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle the data each epoch.
        seed: Random seed for reproducibility.
        tensor_format: 'NHWC' for (batch, 32, 32, 3) or 'flat' for (batch, 3072).
        normalize_mean: Per-channel mean for normalization (default: CIFAR-10 mean).
        normalize_std: Per-channel std for normalization (default: CIFAR-10 std).
    """

    def __init__(
        self,
        split: str,
        batch_size: int,
        shuffle: bool = True,
        seed: int = None,
        tensor_format: str = "NHWC",
        normalize_mean: tuple = (0.4914, 0.4822, 0.4465),
        normalize_std: tuple = (0.2470, 0.2435, 0.2616),
    ):
        import tensorflow_datasets as tfds
        import tensorflow as tf

        tf.config.set_visible_devices([], "GPU")

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.tensor_format = tensor_format
        self.normalize_mean = np.array(normalize_mean, dtype=np.float32)
        self.normalize_std = np.array(normalize_std, dtype=np.float32)

        file_seed, buffer_seed = split_np_seed(seed, n=2)

        read_config = tfds.ReadConfig(
            shuffle_seed=file_seed,
            interleave_cycle_length=1,
        )

        ds, info = tfds.load(
            "cifar10",
            split=split,
            with_info=True,
            as_supervised=True,
            read_config=read_config,
            shuffle_files=shuffle and seed is not None,
        )
        self.num_examples = info.splits[split].num_examples
        self._num_batches = (self.num_examples + batch_size - 1) // batch_size

        if shuffle:
            ds = ds.shuffle(buffer_size=self.num_examples, seed=buffer_seed)
        ds = ds.batch(batch_size, drop_remainder=False)
        ds = ds.prefetch(tf.data.AUTOTUNE)

        self.ds = ds

    def __iter__(self):
        for images, labels in self.ds:
            images = images.numpy().astype(np.float32) / 255.0
            # Per-channel normalization: (H, W, C) broadcast
            images = (images - self.normalize_mean) / self.normalize_std

            if self.tensor_format == "flat":
                images = images.reshape(images.shape[0], -1)

            labels = one_hot(labels.numpy(), num_classes=10)

            yield images, labels

    def __len__(self):
        return self._num_batches


class _TokenSequenceLoader:
    """Shared sliding-window next-token batching for token-level loaders.

    Subclasses load their corpus into `self.data` (1D int32 token ids) and set
    `self.vocab_size`, `self.seq_len`, `self.batch_size`, `self.shuffle`, and
    `self.seed`, then call `self._init_sequence_indexing(max_samples)`. This base
    provides the identical batching for both char and BPE:
        x = data[i : i+seq_len],  y = data[i+1 : i+seq_len+1]

    Targets are yielded as integer token ids (batch, seq_len) int32. One-hot
    encoding to (batch, seq_len, vocab_size) for the CrossEntropy output node
    happens inside the training/eval step (compute_loss and the PC y-clamp),
    not here. This keeps the host->device transfer int32 and avoids a large
    one-hot per batch for big vocabularies (~96 MB at 16x128x11711 for BPE; the
    int ids are ~8 KB).
    """

    def _init_sequence_indexing(self, max_samples):
        # Each sequence needs seq_len input tokens + 1 shifted target token.
        self.num_sequences = len(self.data) - self.seq_len
        if max_samples is not None:
            self.num_sequences = min(self.num_sequences, max_samples)
        self._num_batches = self.num_sequences // self.batch_size
        self._epoch = 0

    def __iter__(self):
        indices = np.arange(self.num_sequences)
        if self.shuffle:
            epoch_seed = self.seed + self._epoch if self.seed is not None else None
            rng = np.random.default_rng(epoch_seed)
            rng.shuffle(indices)
        self._epoch += 1

        for start in range(0, len(indices), self.batch_size):
            batch_idx = indices[start : start + self.batch_size]
            if len(batch_idx) < self.batch_size:
                continue  # drop incomplete last batch

            x = np.stack([self.data[i : i + self.seq_len] for i in batch_idx])
            y = np.stack([self.data[i + 1 : i + self.seq_len + 1] for i in batch_idx])
            # Yield integer target ids; one-hot happens in the training/eval step
            # to keep the host->device transfer int32 (see class docstring).
            yield x, y

    def __len__(self):
        return self._num_batches


class CharDataLoader(_TokenSequenceLoader):
    """JAX-compatible character-level dataloader using TFDS.

    Loads the tiny_shakespeare dataset from TensorFlow Datasets and
    yields batches of (x_indices, y_indices) for next-character prediction.
    Targets are integer ids; one-hot encoding happens in the training/eval
    step (see _TokenSequenceLoader).

    The vocabulary is always built from the train split to ensure consistent
    char-to-index mappings across all splits.

    Args:
        split: Dataset split ('train', 'validation', or 'test').
        seq_len: Number of characters per input sequence.
        batch_size: Number of sequences per batch.
        shuffle: Whether to shuffle sequence start positions each epoch.
        seed: Random seed for reproducible shuffling.
        max_samples: If set, cap the number of sequences to this value.
            Useful for fast hyperparameter tuning on a subset of data.
    """

    # Class-level cache for vocabulary (built once from train split)
    _vocab = None

    def __init__(
        self,
        split: str,
        seq_len: int,
        batch_size: int,
        shuffle: bool = True,
        seed: int = None,
        max_samples: int = None,
    ):
        import tensorflow_datasets as tfds
        import tensorflow as tf

        tf.config.set_visible_devices([], "GPU")

        self.seq_len = seq_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

        # Build vocabulary from the train split (cached across instances)
        if CharDataLoader._vocab is None:
            train_ds = tfds.load("tiny_shakespeare", split="train")
            train_text = next(iter(train_ds))["text"].numpy().decode("utf-8")
            chars = sorted(set(train_text))
            CharDataLoader._vocab = {
                "chars": chars,
                "vocab_size": len(chars),
                "char_to_idx": {ch: i for i, ch in enumerate(chars)},
                "idx_to_char": {i: ch for i, ch in enumerate(chars)},
            }

        self.chars = CharDataLoader._vocab["chars"]
        self.vocab_size = CharDataLoader._vocab["vocab_size"]
        self.char_to_idx = CharDataLoader._vocab["char_to_idx"]
        self.idx_to_char = CharDataLoader._vocab["idx_to_char"]

        # Load the requested split and encode to indices
        ds = tfds.load("tiny_shakespeare", split=split)
        text = next(iter(ds))["text"].numpy().decode("utf-8")
        self.data = np.array([self.char_to_idx[ch] for ch in text], dtype=np.int32)

        self._init_sequence_indexing(max_samples)

    def decode(self, indices) -> str:
        """Convert an array of character indices back to a string."""
        return "".join(self.idx_to_char[int(i)] for i in indices)


class BpeDataLoader(_TokenSequenceLoader):
    """JAX-compatible BPE tokenized data loader using TFDS.

    Loads pre-encoded BPE token sequences from cached .npy files and yields
    batches of (x_indices, y_indices) for next-token prediction. Targets are
    integer token ids; one-hot encoding happens in the training/eval step.

    On first use the BPE tokenizer (vocab_size=8000 by default) is fit on the
    training split only; that frozen tokenizer then encodes every split and
    the results are cached to bpe_data_dir. Later runs load from cache.
    WikiText-103 is pulled via Hugging Face datasets; other datasets are
    loaded via TFDS.

    Args:
        split: Dataset split ('train', 'validation', or 'test').
        seq_len: Number of tokens per input sequence.
        batch_size: Number of sequences per batch.
        shuffle: Whether to shuffle sequence start positions each epoch.
        seed: Random seed for reproducible shuffling.
        max_samples: If set, cap the number of sequences to this value.
            Useful for fast hyperparameter tuning on a subset of data.
        dataset: Corpus identifier (default: "wikitext103").
        bpe_data_dir: Cache dir for tokenizer and encoded splits.
            Defaults to "data/bpe_{dataset}_v{vocab_size}".
        vocab_size: BPE vocabulary size (default: 8000).
        verbose: Print progress during dataset download/tokenization.
    """

    splits = ["train", "validation", "test"]

    def __init__(
        self,
        split: str,
        seq_len: int,
        batch_size: int,
        shuffle: bool = True,
        seed: int = None,
        max_samples: int = None,
        dataset: str = "wikitext103",
        bpe_data_dir: str = None,
        vocab_size: int = 8000,
        verbose: bool = True,
    ):
        from pathlib import Path

        if not _TOKENIZERS_AVAILABLE:
            raise ImportError(_TOKENIZERS_INSTALL_HINT)

        self.seq_len = seq_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

        if bpe_data_dir is None:
            bpe_data_dir = f"data/bpe_{dataset}_v{vocab_size}"

        bpe_dir = Path(bpe_data_dir)
        tokenizer_path = bpe_dir / "tokenizer.json"
        token_path = bpe_dir / f"{split}.npy"

        # Prepare tokenizer and encoded splits if not already cached
        if not tokenizer_path.exists() or not token_path.exists():
            self._prepare(bpe_dir, dataset=dataset, vocab_size=vocab_size, verbose=verbose)

        tok = Tokenizer.from_file(str(tokenizer_path))
        self.vocab_size = tok.get_vocab_size()
        vocab = tok.get_vocab()
        self.token_to_idx = vocab
        self.idx_to_token = {v: k for k, v in vocab.items()}
        self.data = np.load(token_path)
        self._tok = tok

        self._init_sequence_indexing(max_samples)

    @staticmethod
    def _load_split_text(dataset: str, split: str) -> str:
        """Return the raw text of a single split."""
        if dataset == "wikitext103":
            from datasets import load_dataset

            hf_split = {"train": "train", "validation": "validation", "test": "test"}[split]
            ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=hf_split)
            return "\n".join(t for t in ds["text"] if t.strip())

        import tensorflow_datasets as tfds
        import tensorflow as tf

        tf.config.set_visible_devices([], "GPU")
        ds = tfds.load(dataset, split=split)
        return "\n".join(ex["text"].numpy().decode("utf-8") for ex in ds)

    @staticmethod
    def _prepare(bpe_dir, dataset: str = "wikitext103", vocab_size: int = 8000, verbose: bool = True):
        """Fit the BPE tokenizer on the train split only, then encode every
        split with that frozen tokenizer and cache each one to .npy."""
        from pathlib import Path

        bpe_dir = Path(bpe_dir)
        bpe_dir.mkdir(parents=True, exist_ok=True)

        tokenizer_path = bpe_dir / "tokenizer.json"

        if not tokenizer_path.exists():
            if verbose:
                print(f"BpeDataLoader: training BPE tokenizer on {dataset} train split...")
            tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
            tokenizer.pre_tokenizer = Whitespace()
            trainer = BpeTrainer(
                vocab_size=vocab_size,
                special_tokens=["[UNK]", "[BOS]", "[EOS]", "[PAD]"],
            )
            tmp = bpe_dir / "_tmp_train.txt"
            tmp.write_text(
                BpeDataLoader._load_split_text(dataset, "train"), encoding="utf-8"
            )
            tokenizer.train(files=[str(tmp)], trainer=trainer)
            tmp.unlink()

            tokenizer.save(str(tokenizer_path))
            if verbose:
                print(f"BpeDataLoader: tokenizer saved to {tokenizer_path}")
        else:
            tokenizer = Tokenizer.from_file(str(tokenizer_path))

        for split in BpeDataLoader.splits:
            token_path = bpe_dir / f"{split}.npy"
            if not token_path.exists():
                if verbose:
                    print(f"BpeDataLoader: encoding {split} split...")
                text = BpeDataLoader._load_split_text(dataset, split)
                ids = tokenizer.encode(text).ids
                np.save(token_path, np.array(ids, dtype=np.int32))
                if verbose:
                    print(f"BpeDataLoader: {split} saved to {token_path}")

    def decode(self, indices) -> str:
        return self._tok.decode([int(i) for i in indices], skip_special_tokens=True)


class FashionMnistLoader:
    """JAX-compatible Fashion-MNIST data loader using TensorFlow Datasets.

    Drop-in replacement for MnistLoader with the Fashion-MNIST dataset
    (28x28 grayscale, 10 clothing categories).

    Args:
        split: Dataset split ('train' or 'test').
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle the data each epoch.
        seed: Random seed for reproducibility.
        tensor_format: 'NHWC' for (batch, 28, 28, 1) or 'flat' for (batch, 784).
        normalize_mean: Mean for normalization (default: Fashion-MNIST mean).
        normalize_std: Std for normalization (default: Fashion-MNIST std).
    """

    def __init__(
        self,
        split: str,
        batch_size: int,
        shuffle: bool = True,
        seed: int = None,
        tensor_format: str = "NHWC",
        normalize_mean: float = 0.2860,
        normalize_std: float = 0.3530,
    ):
        import tensorflow_datasets as tfds
        import tensorflow as tf

        tf.config.set_visible_devices([], "GPU")

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.tensor_format = tensor_format
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std

        file_seed, buffer_seed = split_np_seed(seed, n=2)

        read_config = tfds.ReadConfig(
            shuffle_seed=file_seed,
            interleave_cycle_length=1,
        )

        ds, info = tfds.load(
            "fashion_mnist",
            split=split,
            with_info=True,
            as_supervised=True,
            read_config=read_config,
            shuffle_files=shuffle and seed is not None,
        )
        self.num_examples = info.splits[split].num_examples
        self._num_batches = (self.num_examples + batch_size - 1) // batch_size

        if shuffle:
            ds = ds.shuffle(buffer_size=self.num_examples, seed=buffer_seed)
        ds = ds.batch(batch_size, drop_remainder=False)
        ds = ds.prefetch(tf.data.AUTOTUNE)

        self.ds = ds

    def __iter__(self):
        for images, labels in self.ds:
            images = images.numpy().astype(np.float32) / 255.0
            images = (images - self.normalize_mean) / self.normalize_std

            if self.tensor_format == "flat":
                images = images.reshape(images.shape[0], -1)

            labels = one_hot(labels.numpy(), num_classes=10)

            yield images, labels

    def __len__(self):
        return self._num_batches


class FewShotLoader:
    """Class-balanced K-shot data loader using TensorFlow Datasets.

    Loads a full dataset, subsamples exactly K examples per class
    (deterministically via seed), and yields shuffled minibatches.
    Both arms in an A/B experiment receive identical training data
    when given the same seed.

    Args:
        dataset_name: TFDS dataset name (e.g., 'fashion_mnist', 'mnist:3.0.1').
        split: Dataset split ('train' or 'test').
        k_per_class: Number of examples to keep per class.
        batch_size: Number of samples per batch.
        num_classes: Number of classes in the dataset (default: 10).
        shuffle: Whether to shuffle the subsample each epoch.
        seed: Random seed for both subsampling and shuffling.
        tensor_format: 'NHWC' or 'flat'.
        normalize_mean: Mean for normalization.
        normalize_std: Std for normalization.
    """

    def __init__(
        self,
        dataset_name: str,
        split: str,
        k_per_class: int,
        batch_size: int,
        num_classes: int = 10,
        shuffle: bool = True,
        seed: int = None,
        tensor_format: str = "flat",
        normalize_mean: float = 0.2860,
        normalize_std: float = 0.3530,
    ):
        import tensorflow_datasets as tfds
        import tensorflow as tf

        tf.config.set_visible_devices([], "GPU")

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.tensor_format = tensor_format
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.num_classes = num_classes
        self._epoch = 0

        # Load entire dataset into memory
        ds = tfds.load(dataset_name, split=split, as_supervised=True)
        all_images = []
        all_labels = []
        for img, label in ds:
            all_images.append(img.numpy())
            all_labels.append(int(label.numpy()))

        all_images = np.array(all_images, dtype=np.float32) / 255.0
        all_images = (all_images - self.normalize_mean) / self.normalize_std
        all_labels = np.array(all_labels, dtype=np.int32)

        # Class-balanced subsampling
        rng = np.random.default_rng(seed)
        selected_indices = []
        for c in range(num_classes):
            class_indices = np.where(all_labels == c)[0]
            if len(class_indices) < k_per_class:
                chosen = class_indices  # use all if fewer than K
            else:
                chosen = rng.choice(class_indices, size=k_per_class, replace=False)
            selected_indices.append(chosen)
        selected_indices = np.concatenate(selected_indices)

        self.images = all_images[selected_indices]
        self.labels = all_labels[selected_indices]
        self.num_samples = len(selected_indices)
        self._num_batches = self.num_samples // batch_size

    def __iter__(self):
        indices = np.arange(self.num_samples)
        if self.shuffle:
            epoch_seed = (
                self.seed + 10000 + self._epoch if self.seed is not None else None
            )
            rng = np.random.default_rng(epoch_seed)
            rng.shuffle(indices)
        self._epoch += 1

        for start in range(0, self.num_samples - self.batch_size + 1, self.batch_size):
            batch_idx = indices[start : start + self.batch_size]
            images = self.images[batch_idx]

            if self.tensor_format == "flat":
                images = images.reshape(images.shape[0], -1)

            labels = one_hot(self.labels[batch_idx], num_classes=self.num_classes)
            yield images, labels

    def __len__(self):
        return self._num_batches


class NoisyTestLoader:
    """Wrapper that adds Gaussian noise to a base loader's images at test time.

    Useful for evaluating noise robustness of trained models. Noise is
    applied after normalization, so noise_std is in normalized units.

    Args:
        base_loader: Any iterable loader yielding (images, labels) batches.
        noise_std: Standard deviation of Gaussian noise (0.0 = no noise).
        seed: Random seed for reproducible noise.
    """

    def __init__(self, base_loader, noise_std: float = 0.0, seed: int = None):
        self.base_loader = base_loader
        self.noise_std = noise_std
        self.seed = seed

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        for images, labels in self.base_loader:
            if self.noise_std > 0:
                noise = rng.normal(0, self.noise_std, images.shape).astype(np.float32)
                images = images + noise
            yield images, labels

    def __len__(self):
        return len(self.base_loader)
