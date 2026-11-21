"""Offline synthetic character tasks for smoke runs and tests."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from fabricpc.experiments.forgetting_benchmark.benchmark import TASK_REGISTRY

_VOCABULARIES = {
    "synthetic_a": (
        "the cat sat on a mat",
        "a dog ran to the red barn",
        "the sun is hot",
    ),
    "synthetic_b": (
        "we sing and dance at noon",
        "my song is long and loud",
        "dance all night",
    ),
    "synthetic_c": ("1 + 2 = 3", "4 + 5 = 9", "7 + 1 = 8", "2 + 2 = 4"),
}
_DEFAULT_CHARS = 20_000


def _synthetic_text(name: str, num_chars: int, seed: int) -> str:
    rng = np.random.default_rng(seed)
    phrases = _VOCABULARIES[name]
    pieces: list[str] = []
    total = 0
    while total < num_chars:
        phrase = phrases[int(rng.integers(len(phrases)))] + "\n"
        pieces.append(phrase)
        total += len(phrase)
    return "".join(pieces)[:num_chars]


def _loader(name: str):
    def load(_data_dir: Path, max_chars: Optional[int]) -> tuple[str, str]:
        size = max_chars or _DEFAULT_CHARS
        offset = sum(map(ord, name))
        train = _synthetic_text(name, size, offset)
        test = _synthetic_text(name, max(size // 4, 256), offset + 1)
        return train, test

    return load


def register_synthetic_tasks() -> None:
    """Add the synthetic tasks to the benchmark task registry."""
    for name in _VOCABULARIES:
        TASK_REGISTRY.setdefault(name, _loader(name))


register_synthetic_tasks()
