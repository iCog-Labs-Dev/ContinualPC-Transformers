"""Checksums and atomic file writes for the domain-forgetting benchmark."""

from __future__ import annotations

import hashlib
import io
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from fabricpc.experiments.forgetting_benchmark.sequential_trainer import dumps_strict


def sha256_text(text: str) -> str:
    """Return the SHA-256 hex digest of ``text`` encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 hex digest of a file's bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def arrays_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    """Return a SHA-256 digest over array names, dtypes, shapes and bytes.

    Unlike a file digest of an ``.npz`` archive, this does not depend on zip
    timestamps, so two builds of the same arrays hash equally.
    """
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(np.asarray(arrays[name]))
        digest.update(f"{name}/{value.dtype.str}/{value.shape}".encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _atomic_write(path: Path, write: Callable[[Any], None], mode: str) -> Path:
    """Write through a temporary file in the same directory, then rename."""
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"Refusing to write through symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, mode) as handle:
            write(handle)
        os.replace(tmp_name, path)
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise
    return path


def atomic_write_text(path: str | Path, text: str) -> Path:
    """Atomically write UTF-8 text."""
    return _atomic_write(
        Path(path), lambda handle: handle.write(text.encode("utf-8")), "wb"
    )


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    """Atomically write strict JSON (non-finite floats become null)."""
    return atomic_write_text(
        path, dumps_strict(payload, indent=2, sort_keys=True) + "\n"
    )


def atomic_save_npy(path: str | Path, array: np.ndarray) -> Path:
    """Atomically save one array in ``.npy`` format."""
    return _atomic_write(
        Path(path), lambda handle: np.save(handle, array, allow_pickle=False), "wb"
    )


def atomic_save_npz(path: str | Path, arrays: Mapping[str, np.ndarray]) -> Path:
    """Atomically save named arrays as an uncompressed ``.npz`` archive."""

    def write(handle: Any) -> None:
        buffer = io.BytesIO()
        np.savez(buffer, **{name: np.asarray(v) for name, v in arrays.items()})
        handle.write(buffer.getvalue())

    return _atomic_write(Path(path), write, "wb")


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Load every array of an ``.npz`` archive into memory."""
    with np.load(Path(path), allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}
