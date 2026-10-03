"""Character-level text tasks for the sequential forgetting baseline."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import unicodedata
from pathlib import Path
from typing import Callable, Optional
from urllib.request import urlopen

import numpy as np


PREPROCESSING_PROFILE = "ascii_v1"
PREPROCESSING_IMPLEMENTATION_VERSION = "1"
ASCII_CHARACTERS = "\n" + "".join(chr(codepoint) for codepoint in range(32, 127))
ASCII_VOCAB_SIZE = 2 + len(ASCII_CHARACTERS)

# Repair the common UTF-8-as-Windows-1252 sequences before general Unicode
# normalization. These are observed in the current TinyStories cache.
MOJIBAKE_REPLACEMENTS = {
    "â€™": "'",
    "â€˜": "'",
    "â€œ": '"',
    "â€\u009d": '"',
    "â€“": "-",
    "â€”": "-",
    "â€¦": "...",
    "Â ": " ",
}
PUNCTUATION_REPLACEMENTS = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
        "–": "-",
        "—": "-",
        "―": "-",
        "…": "...",
        "œ": "oe",
        "Œ": "OE",
        "æ": "ae",
        "Æ": "AE",
        "ß": "ss",
        "\u00a0": " ",
        "\u200b": "",
        "\ufeff": "",
    }
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_text_atomically(path: Path, text: str) -> None:
    """Write a complete UTF-8 artifact before making it visible to other jobs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as temporary_file:
            temporary_file.write(text)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
            temporary_path = Path(temporary_file.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _artifact_matches_manifest(path: Path, expected_hash: object) -> bool:
    """Verify a cached processed artifact before it is reused for training."""
    if not path.exists() or not isinstance(expected_hash, str):
        return False
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    return _sha256(text) == expected_hash and all(
        character in ASCII_CHARACTERS for character in text
    )


def preprocess_ascii_text(text: str) -> tuple[str, dict[str, int | str]]:
    """Convert raw text into the benchmark's deterministic ASCII text contract.

    The contract preserves newlines and printable ASCII, canonicalizes common
    punctuation and mojibake, transliterates decomposable Latin characters, and
    replaces unsupported Unicode runs with whitespace. It deliberately avoids
    silently adding corpus-specific Unicode characters to the vocabulary.
    """
    input_characters = len(text)
    non_ascii_input_characters = sum(ord(character) > 127 for character in text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    for source, replacement in MOJIBAKE_REPLACEMENTS.items():
        text = text.replace(source, replacement)
    text = unicodedata.normalize("NFKC", text).translate(PUNCTUATION_REPLACEMENTS)

    output: list[str] = []
    unsupported_characters = 0
    control_characters = 0
    for character in text:
        if character == "\n":
            output.append(character)
        elif character in "\t\v\f":
            output.append(" ")
            control_characters += 1
        elif 32 <= ord(character) <= 126:
            output.append(character)
        elif unicodedata.category(character).startswith("C"):
            control_characters += 1
        else:
            transliterated = unicodedata.normalize("NFKD", character).encode(
                "ascii", "ignore"
            ).decode("ascii")
            transliterated = "".join(
                candidate
                for candidate in transliterated
                if 32 <= ord(candidate) <= 126
            )
            if transliterated:
                output.append(transliterated)
            else:
                output.append(" ")
                unsupported_characters += 1

    # Preserve line boundaries while avoiding whitespace-only evidence from
    # unsupported character runs and platform-specific line endings.
    cleaned = "\n".join(
        re.sub(r"[ ]+", " ", line).strip(" ") for line in "".join(output).split("\n")
    )
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned, {
        "normalization": PREPROCESSING_PROFILE,
        "input_characters": input_characters,
        "non_ascii_input_characters": non_ascii_input_characters,
        "unsupported_characters_replaced": unsupported_characters,
        "control_characters_removed": control_characters,
        "output_characters": len(cleaned),
        "output_sha256": _sha256(cleaned),
    }


def _limit(text: str, max_chars: Optional[int]) -> str:
    return text if max_chars is None else text[:max_chars]


def download_tiny_stories(
    data_dir: Path, max_chars: Optional[int] = None
) -> tuple[str, str]:
    """Load TinyStories through TFDS, using its train and validation splits."""
    legacy_suffix = "full" if max_chars is None else f"first_{max_chars}_chars"
    legacy_train = data_dir / f"tiny_stories_{legacy_suffix}_train.txt"
    legacy_test = data_dir / f"tiny_stories_{legacy_suffix}_test.txt"
    if legacy_train.exists() and legacy_test.exists():
        # A migration path for the manually prepared cache created before this
        # preprocessing contract existed. The content is still cleaned below.
        return legacy_train.read_text(encoding="utf-8"), legacy_test.read_text(
            encoding="utf-8"
        )
    try:
        import tensorflow as tf
        import tensorflow_datasets as tfds

        tf.config.set_visible_devices([], "GPU")
        splits = (
            tfds.load("tiny_stories", split="train", data_dir=str(data_dir / "tfds")),
            tfds.load(
                "tiny_stories", split="validation", data_dir=str(data_dir / "tfds")
            ),
        )
    except Exception as error:  # pragma: no cover - requires a local TFDS install.
        raise RuntimeError(
            "TinyStories requires the optional TFDS dependencies: pip install -e '.[tfds]'"
        ) from error

    def join(dataset) -> str:
        parts: list[str] = []
        size = 0
        for example in dataset:
            text = example["text"].numpy().decode("utf-8")
            parts.append(text)
            size += len(text) + 1
            if max_chars is not None and size >= max_chars:
                break
        return _limit("\n".join(parts), max_chars)

    return join(splits[0]), join(splits[1])


WIKITEXT_URL = (
    "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1"
)
TINY_SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def _download_text(url: str) -> str:
    try:
        return urlopen(url).read().decode("utf-8")
    except Exception as error:  # pragma: no cover - depends on network access.
        raise RuntimeError(f"Could not download {url}: {error}") from error


def download_wikitext2(
    data_dir: Path, max_chars: Optional[int] = None
) -> tuple[str, str]:
    """Download the WikiText-2 raw train and test files."""
    cache_dir = data_dir / "wikitext2_raw"
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_path, test_path = cache_dir / "train.txt", cache_dir / "test.txt"
    if not train_path.exists():
        train_path.write_text(
            _download_text(f"{WIKITEXT_URL}/wiki.train.raw"), encoding="utf-8"
        )
    if not test_path.exists():
        test_path.write_text(
            _download_text(f"{WIKITEXT_URL}/wiki.test.raw"), encoding="utf-8"
        )
    return _limit(train_path.read_text(encoding="utf-8"), max_chars), _limit(
        test_path.read_text(encoding="utf-8"), max_chars
    )


def download_tiny_shakespeare(
    data_dir: Path, max_chars: Optional[int] = None
) -> tuple[str, str]:
    """Download Tiny Shakespeare and split it deterministically into 90/10."""
    cache_dir = data_dir / "tiny_shakespeare_raw"
    cache_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = cache_dir / "input.txt"
    if not corpus_path.exists():
        corpus_path.write_text(_download_text(TINY_SHAKESPEARE_URL), encoding="utf-8")
    text = corpus_path.read_text(encoding="utf-8")
    split = int(len(text) * 0.9)
    return _limit(text[:split], max_chars), _limit(text[split:], max_chars)


TASK_REGISTRY: dict[str, Callable[[Path, Optional[int]], tuple[str, str]]] = {
    "tiny_stories": download_tiny_stories,
    "wikitext2": download_wikitext2,
    "tiny_shakespeare": download_tiny_shakespeare,
}


class CharacterTokenizer:
    """A shared fixed ASCII vocabulary for the cleaned benchmark corpus."""

    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"

    def __init__(self) -> None:
        self._token_to_idx: dict[str, int] = {}
        self._idx_to_token: dict[int, str] = {}

    def train(self, _texts: list[str]) -> None:
        """Initialize the predeclared vocabulary without inspecting task text."""
        self._token_to_idx = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
            **{
                character: index + 2
                for index, character in enumerate(ASCII_CHARACTERS)
            },
        }
        self._idx_to_token = {
            index: token for token, index in self._token_to_idx.items()
        }

    @property
    def token_to_idx(self) -> dict[str, int]:
        return self._token_to_idx

    def encode(self, text: str) -> list[int]:
        unknown = self._token_to_idx[self.UNK_TOKEN]
        return [self._token_to_idx.get(character, unknown) for character in text]

    def decode(self, ids: list[int]) -> str:
        chars = []
        for token_id in ids:
            token = self._idx_to_token.get(int(token_id), self.UNK_TOKEN)
            if token == self.PAD_TOKEN:
                continue
            chars.append("�" if token == self.UNK_TOKEN else token)
        return "".join(chars)

    def get_vocab_size(self) -> int:
        return len(self._token_to_idx)

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(self._token_to_idx, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


class TextTaskDataset:
    """Sliding-window next-character batches for one task."""

    def __init__(
        self,
        text: str,
        tokenizer: CharacterTokenizer,
        seq_len: int,
        batch_size: int,
        *,
        shuffle: bool,
        seed: Optional[int],
        max_samples: Optional[int],
    ) -> None:
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.vocab_size = tokenizer.get_vocab_size()
        tokens = tokenizer.encode(text)
        if max_samples is not None:
            tokens = tokens[: max_samples + seq_len]
        self.data = np.asarray(tokens, dtype=np.int32)
        self.num_sequences = max(0, len(self.data) - seq_len)
        self.num_sequences = (
            min(self.num_sequences, max_samples) if max_samples else self.num_sequences
        )
        self._num_batches = self.num_sequences // batch_size
        self._epoch = 0

    def __len__(self) -> int:
        return self._num_batches

    def __iter__(self):
        indices = np.arange(self.num_sequences)
        if self.shuffle:
            rng = np.random.default_rng(
                None if self.seed is None else self.seed + self._epoch
            )
            rng.shuffle(indices)
        self._epoch += 1
        for start in range(0, self._num_batches * self.batch_size, self.batch_size):
            batch_indices = indices[start : start + self.batch_size]
            x = np.stack(
                [self.data[index : index + self.seq_len] for index in batch_indices]
            )
            y = np.stack(
                [
                    self.data[index + 1 : index + self.seq_len + 1]
                    for index in batch_indices
                ]
            )
            yield x, y


class SequentialBenchmark:
    """Ordered A → B → C text tasks using one shared character vocabulary."""

    def __init__(
        self,
        tasks: list[str],
        *,
        seq_len: int = 128,
        batch_size: int = 16,
        data_dir: str = "data/forgetting_benchmark",
        seed: int = 42,
        max_samples_per_task: Optional[int] = None,
        max_chars_per_task: Optional[int] = None,
        preprocessing_profile: str = PREPROCESSING_PROFILE,
    ) -> None:
        unknown = [task for task in tasks if task not in TASK_REGISTRY]
        if unknown:
            raise ValueError(
                f"Unknown task(s): {unknown}. Available: {sorted(TASK_REGISTRY)}"
            )
        if preprocessing_profile != PREPROCESSING_PROFILE:
            raise ValueError(
                f"Unsupported preprocessing profile: {preprocessing_profile}. "
                f"Expected {PREPROCESSING_PROFILE}."
            )
        self.tasks = tasks
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.seed = seed
        self.max_samples_per_task = max_samples_per_task
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.preprocessing_profile = preprocessing_profile
        self.preprocessing_manifest: dict[str, object] = {
            "profile": preprocessing_profile,
            "implementation_version": PREPROCESSING_IMPLEMENTATION_VERSION,
            "unicode_data_version": unicodedata.unidata_version,
            "vocabulary": "printable_ascii_plus_newline",
            "vocab_size": ASCII_VOCAB_SIZE,
            "tasks": {},
        }
        self._train_texts: dict[str, str] = {}
        self._test_texts: dict[str, str] = {}
        self._load_tasks(max_chars_per_task)
        self.tokenizer = CharacterTokenizer()
        self.tokenizer.train([self._train_texts[task] for task in tasks])

    def _load_tasks(self, max_chars: Optional[int]) -> None:
        processed_dir = self.data_dir / "processed" / self.preprocessing_profile
        processed_dir.mkdir(parents=True, exist_ok=True)
        for task in self.tasks:
            suffix = "full" if max_chars is None else f"first_{max_chars}_chars"
            train_text, test_text = TASK_REGISTRY[task](self.data_dir, max_chars)
            source = {
                "train_sha256": _sha256(train_text),
                "test_sha256": _sha256(test_text),
            }
            cache_key = _sha256(
                json.dumps(
                    {
                        "source": source,
                        "profile": self.preprocessing_profile,
                        "implementation_version": PREPROCESSING_IMPLEMENTATION_VERSION,
                        "unicode_data_version": unicodedata.unidata_version,
                    },
                    sort_keys=True,
                )
            )
            artifact_dir = processed_dir / task / cache_key
            train_path = artifact_dir / f"{suffix}_train.txt"
            test_path = artifact_dir / f"{suffix}_test.txt"
            artifact_manifest_path = artifact_dir / "preprocessing_manifest.json"
            cache_manifest: Optional[dict] = None
            if artifact_manifest_path.exists():
                try:
                    parsed_manifest = json.loads(
                        artifact_manifest_path.read_text(encoding="utf-8")
                    )
                    if isinstance(parsed_manifest, dict):
                        cache_manifest = parsed_manifest
                except json.JSONDecodeError:
                    cache_manifest = None
            artifacts = (
                cache_manifest.get("artifacts")
                if cache_manifest is not None
                and isinstance(cache_manifest.get("artifacts"), dict)
                else {}
            )
            train_artifact = artifacts.get("train", {})
            test_artifact = artifacts.get("test", {})
            cache_is_valid = (
                cache_manifest is not None
                and cache_manifest.get("source") == source
                and cache_manifest.get("profile") == self.preprocessing_profile
                and cache_manifest.get("implementation_version")
                == PREPROCESSING_IMPLEMENTATION_VERSION
                and cache_manifest.get("unicode_data_version")
                == unicodedata.unidata_version
                and _artifact_matches_manifest(
                    train_path,
                    train_artifact.get("output_sha256")
                    if isinstance(train_artifact, dict)
                    else None,
                )
                and _artifact_matches_manifest(
                    test_path,
                    test_artifact.get("output_sha256")
                    if isinstance(test_artifact, dict)
                    else None,
                )
            )
            if not cache_is_valid:
                cleaned_train, train_report = preprocess_ascii_text(train_text)
                cleaned_test, test_report = preprocess_ascii_text(test_text)
                artifact_manifest = {
                    "task": task,
                    "profile": self.preprocessing_profile,
                    "implementation_version": PREPROCESSING_IMPLEMENTATION_VERSION,
                    "unicode_data_version": unicodedata.unidata_version,
                    "source": source,
                    "artifacts": {
                        "train": {"path": train_path.name, **train_report},
                        "test": {"path": test_path.name, **test_report},
                    },
                }
                _write_text_atomically(train_path, cleaned_train)
                _write_text_atomically(test_path, cleaned_test)
                _write_text_atomically(
                    artifact_manifest_path,
                    json.dumps(artifact_manifest, indent=2, sort_keys=True),
                )
                cache_manifest = artifact_manifest
                cache_hit = False
            else:
                cache_hit = True
            assert cache_manifest is not None
            self._train_texts[task] = train_path.read_text(encoding="utf-8")
            self._test_texts[task] = test_path.read_text(encoding="utf-8")
            self.preprocessing_manifest["tasks"][task] = {
                "cache_hit": cache_hit,
                "artifact_dir": str(artifact_dir.relative_to(self.data_dir)),
                **cache_manifest,
            }

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def __len__(self) -> int:
        return len(self.tasks)

    def get_task_name(self, index: int) -> str:
        return self.tasks[index]

    def get_train_loader(self, index: int) -> TextTaskDataset:
        return TextTaskDataset(
            self._train_texts[self.tasks[index]],
            self.tokenizer,
            self.seq_len,
            self.batch_size,
            shuffle=True,
            seed=self.seed + index,
            max_samples=self.max_samples_per_task,
        )

    def get_test_loader(self, index: int) -> TextTaskDataset:
        return TextTaskDataset(
            self._test_texts[self.tasks[index]],
            self.tokenizer,
            self.seq_len,
            self.batch_size,
            shuffle=False,
            seed=None,
            max_samples=self.max_samples_per_task,
        )
