"""Build the frozen Tiny Shakespeare interpretation dataset.

The artifact contains the exact character windows shared by every BP, PC-F4,
and PC-FP analysis, together with leakage-safe split indices, structural
labels, n-gram controls, source positions, and reproducibility hashes.

Label definitions
-----------------
long_range_recurrence
    The final input character has appeared previously in the visible context,
    and its most recent previous occurrence is more than 16 positions back.

strict_induction
    The final input bigram appeared previously in the visible context; the
    continuation after its most recent earlier occurrence equals the true next
    character; and a trigram predictor fitted once on the official training
    split does not rank that target first. Ties for first are treated as
    locally predictable and are therefore excluded.

The examples come from three non-overlapping spans of the official validation
split. Adjacent spans are separated in start-position space by one complete
context length, so no character used by one split is reused by another.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from fabricpc.utils.data import CharDataLoader


FORMAT_VERSION = 1
DEFAULT_SEQ_LEN = 64
DEFAULT_RECURRENCE_THRESHOLD = 16
DEFAULT_SPLIT_SIZES = (12_000, 4_000, 8_000)
SPLIT_NAMES = ("train", "validation", "test")
PROBE_BUDGETS = (100, 500, 2_000, 10_000)
PROBE_REPEATS = 5
PROBE_SEED = 0
SHUFFLED_LABEL_SEED = PROBE_SEED + 101


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the frozen char-level interpretation dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("artifacts/interpretation/char_v1"),
    )
    parser.add_argument("--seq_len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument(
        "--recurrence_threshold",
        type=int,
        default=DEFAULT_RECURRENCE_THRESHOLD,
    )
    parser.add_argument(
        "--minimum_test_positives",
        type=int,
        default=200,
        help="Fail rather than report a probe factor with inadequate test power",
    )
    return parser.parse_args()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(str(array.shape).encode("utf-8"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fit_trigram_counts(token_ids: np.ndarray, vocab_size: int) -> np.ndarray:
    """Count next characters for every two-character context."""
    token_ids = np.asarray(token_ids, dtype=np.int32)
    counts = np.zeros((vocab_size, vocab_size, vocab_size), dtype=np.int64)
    np.add.at(counts, (token_ids[:-2], token_ids[1:-1], token_ids[2:]), 1)
    return counts


def recurrence_features(
    window: np.ndarray,
    *,
    threshold: int,
) -> tuple[np.int8, np.int16]:
    current = int(window[-1])
    prior = np.flatnonzero(window[:-1] == current)
    if prior.size == 0:
        return np.int8(0), np.int16(-1)
    distance = len(window) - 1 - int(prior[-1])
    return np.int8(distance > threshold), np.int16(distance)


def induction_features(
    window: np.ndarray,
    target: int,
    trigram_counts: np.ndarray,
) -> tuple[np.int8, np.int16, np.int16, np.int8]:
    """Return strict label, source offset, continuation, and trigram-top flag."""
    first, second = int(window[-2]), int(window[-1])
    candidates = np.flatnonzero(
        (window[:-2] == first) & (window[1:-1] == second)
    )

    if candidates.size:
        source = int(candidates[-1])
        continuation = int(window[source + 2])
        induction_consistent = continuation == int(target)
    else:
        source = -1
        continuation = -1
        induction_consistent = False

    local_counts = trigram_counts[first, second]
    maximum = int(local_counts.max())
    target_is_top = maximum > 0 and int(local_counts[int(target)]) == maximum
    strict = induction_consistent and not target_is_top
    return (
        np.int8(strict),
        np.int16(source),
        np.int16(continuation),
        np.int8(target_is_top),
    )


def make_split_starts(
    split_sizes: tuple[int, int, int],
    seq_len: int,
) -> dict[str, np.ndarray]:
    """Create fixed source positions with no token overlap between splits."""
    result: dict[str, np.ndarray] = {}
    offset = 0
    for name, size in zip(SPLIT_NAMES, split_sizes, strict=True):
        result[name] = np.arange(offset, offset + size, dtype=np.int64)
        offset += size + seq_len
    return result


def build_arrays(
    corpus_ids: np.ndarray,
    trigram_counts: np.ndarray,
    *,
    seq_len: int,
    recurrence_threshold: int,
    split_sizes: tuple[int, int, int] = DEFAULT_SPLIT_SIZES,
) -> dict[str, np.ndarray]:
    split_starts = make_split_starts(split_sizes, seq_len)
    source_positions = np.concatenate(
        [split_starts[name] for name in SPLIT_NAMES]
    )
    required = int(source_positions[-1]) + seq_len + 1
    if len(corpus_ids) < required:
        raise ValueError(
            f"validation corpus has {len(corpus_ids)} tokens but {required} are required"
        )

    n_samples = len(source_positions)
    inputs = np.empty((n_samples, seq_len), dtype=np.int32)
    targets = np.empty(n_samples, dtype=np.int32)
    recurrence = np.empty(n_samples, dtype=np.int8)
    recurrence_distance = np.empty(n_samples, dtype=np.int16)
    strict_induction = np.empty(n_samples, dtype=np.int8)
    induction_source_offset = np.empty(n_samples, dtype=np.int16)
    induction_continuation = np.empty(n_samples, dtype=np.int16)
    trigram_target_is_top = np.empty(n_samples, dtype=np.int8)
    trigram_target_count = np.empty(n_samples, dtype=np.int64)
    trigram_context_count = np.empty(n_samples, dtype=np.int64)
    trigram_surprisal = np.empty(n_samples, dtype=np.float64)

    vocab_size = trigram_counts.shape[-1]
    for row, start in enumerate(source_positions):
        window = np.asarray(corpus_ids[start : start + seq_len], dtype=np.int32)
        target = int(corpus_ids[start + seq_len])
        inputs[row] = window
        targets[row] = target

        recurrence[row], recurrence_distance[row] = recurrence_features(
            window,
            threshold=recurrence_threshold,
        )
        (
            strict_induction[row],
            induction_source_offset[row],
            induction_continuation[row],
            trigram_target_is_top[row],
        ) = induction_features(window, target, trigram_counts)

        context_counts = trigram_counts[int(window[-2]), int(window[-1])]
        target_count = int(context_counts[target])
        context_count = int(context_counts.sum())
        trigram_target_count[row] = target_count
        trigram_context_count[row] = context_count
        probability = (target_count + 1.0) / (context_count + vocab_size)
        trigram_surprisal[row] = -np.log(probability)

    n_train, n_validation, n_test = split_sizes
    split_train = np.arange(0, n_train, dtype=np.int64)
    split_validation = np.arange(
        n_train,
        n_train + n_validation,
        dtype=np.int64,
    )
    split_test = np.arange(
        n_train + n_validation,
        n_train + n_validation + n_test,
        dtype=np.int64,
    )

    arrays = {
        "inputs": inputs,
        "targets": targets,
        "source_positions": source_positions,
        "split_train": split_train,
        "split_validation": split_validation,
        "split_test": split_test,
        "long_range_recurrence": recurrence,
        "recurrence_distance": recurrence_distance,
        "strict_induction": strict_induction,
        "induction_source_offset": induction_source_offset,
        "induction_continuation": induction_continuation,
        "trigram_target_is_top": trigram_target_is_top,
        "trigram_target_count": trigram_target_count,
        "trigram_context_count": trigram_context_count,
        "trigram_surprisal": trigram_surprisal,
        "current_token": inputs[:, -1].copy(),
        "previous_token": inputs[:, -2].copy(),
    }

    # Freeze the exact nested training rows used by pcinterp's sample-efficiency
    # protocol. Each repeat uses one permutation, and each budget is its prefix.
    max_budget = max(PROBE_BUDGETS)
    if max_budget > len(split_train):
        raise ValueError("largest probe budget exceeds the training split")
    arrays["probe_budgets"] = np.asarray(PROBE_BUDGETS, dtype=np.int64)
    arrays["probe_train_rows"] = np.stack(
        [
            split_train[
                np.random.default_rng(PROBE_SEED + repeat).permutation(
                    len(split_train)
                )[:max_budget]
            ]
            for repeat in range(PROBE_REPEATS)
        ]
    )

    # Freeze pcinterp's selectivity control: only training labels are permuted;
    # validation and test labels remain untouched for like-for-like scoring.
    for label_name in ("long_range_recurrence", "strict_induction"):
        shuffled = arrays[label_name].copy()
        shuffled[split_train] = np.random.default_rng(
            SHUFFLED_LABEL_SEED
        ).permutation(shuffled[split_train])
        arrays[f"{label_name}_shuffled_control"] = shuffled

    return arrays


def validate_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    seq_len: int,
    minimum_test_positives: int,
) -> dict[str, dict[str, float | int]]:
    inputs = arrays["inputs"]
    if inputs.ndim != 2 or inputs.shape[1] != seq_len:
        raise AssertionError(f"unexpected input shape: {inputs.shape}")
    n_samples = len(inputs)
    sample_arrays = {
        name: value
        for name, value in arrays.items()
        if not name.startswith(("split_", "probe_"))
    }
    if not all(
        value.ndim > 0 and len(value) == n_samples
        for value in sample_arrays.values()
    ):
        raise AssertionError("not every saved array has one row per sample")

    split_rows = [arrays[f"split_{name}"] for name in SPLIT_NAMES]
    joined = np.concatenate(split_rows)
    if len(np.unique(joined)) != n_samples or set(joined.tolist()) != set(range(n_samples)):
        raise AssertionError("split rows must be disjoint and exhaust the dataset")

    positions = arrays["source_positions"]
    for left_name, right_name in zip(SPLIT_NAMES[:-1], SPLIT_NAMES[1:]):
        left = arrays[f"split_{left_name}"]
        right = arrays[f"split_{right_name}"]
        if int(positions[left].max()) + seq_len >= int(positions[right].min()):
            raise AssertionError(f"{left_name} and {right_name} contexts overlap")

    summary: dict[str, dict[str, float | int]] = {}
    test = arrays["split_test"]
    for label_name in ("long_range_recurrence", "strict_induction"):
        values = arrays[label_name]
        if not np.isin(values, (0, 1)).all():
            raise AssertionError(f"{label_name} is not binary")
        test_positives = int(values[test].sum())
        if test_positives < minimum_test_positives:
            raise ValueError(
                f"{label_name} has only {test_positives} test positives; "
                f"minimum is {minimum_test_positives}"
            )
        summary[label_name] = {
            "all_positives": int(values.sum()),
            "all_positive_rate": float(values.mean()),
            "test_positives": test_positives,
            "test_positive_rate": float(values[test].mean()),
        }

        shuffled = arrays[f"{label_name}_shuffled_control"]
        train = arrays["split_train"]
        validation = arrays["split_validation"]
        if int(shuffled[train].sum()) != int(values[train].sum()):
            raise AssertionError(f"{label_name} shuffled control changed prevalence")
        if not np.array_equal(shuffled[validation], values[validation]):
            raise AssertionError(f"{label_name} shuffled validation labels changed")
        if not np.array_equal(shuffled[test], values[test]):
            raise AssertionError(f"{label_name} shuffled test labels changed")

    probe_rows = arrays["probe_train_rows"]
    if probe_rows.shape != (PROBE_REPEATS, max(PROBE_BUDGETS)):
        raise AssertionError(f"unexpected probe row shape: {probe_rows.shape}")
    train_set = set(arrays["split_train"].tolist())
    for repeat_rows in probe_rows:
        if len(np.unique(repeat_rows)) != len(repeat_rows):
            raise AssertionError("probe training rows repeat within a schedule")
        if not set(repeat_rows.tolist()).issubset(train_set):
            raise AssertionError("probe training row outside the training split")
    return summary


def main() -> None:
    args = parse_args()
    if args.seq_len < 3:
        raise ValueError("seq_len must be at least 3")
    if not 0 < args.recurrence_threshold < args.seq_len:
        raise ValueError("recurrence_threshold must be between 1 and seq_len - 1")

    output_dir = args.output_dir.resolve()
    dataset_path = output_dir / "dataset.npz"
    manifest_path = output_dir / "manifest.json"
    if dataset_path.exists() or manifest_path.exists():
        raise FileExistsError(
            f"refusing to overwrite frozen artifact in {output_dir}; use a new version directory"
        )

    train_loader = CharDataLoader(
        "train",
        seq_len=args.seq_len,
        batch_size=1,
        shuffle=False,
    )
    validation_loader = CharDataLoader(
        "validation",
        seq_len=args.seq_len,
        batch_size=1,
        shuffle=False,
    )
    train_ids = np.asarray(train_loader.data, dtype=np.int32)
    validation_ids = np.asarray(validation_loader.data, dtype=np.int32)
    trigram_counts = fit_trigram_counts(train_ids, train_loader.vocab_size)

    arrays = build_arrays(
        validation_ids,
        trigram_counts,
        seq_len=args.seq_len,
        recurrence_threshold=args.recurrence_threshold,
    )
    label_summary = validate_arrays(
        arrays,
        seq_len=args.seq_len,
        minimum_test_positives=args.minimum_test_positives,
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(dataset_path, **arrays)

    manifest = {
        "format_version": FORMAT_VERSION,
        "name": "tiny_shakespeare_char_interpretation_v1",
        "source": {
            "dataset": "tensorflow_datasets/tiny_shakespeare",
            "vocabulary_source_split": "train",
            "example_source_split": "validation",
            "train_token_ids_sha256": array_sha256(train_ids),
            "validation_token_ids_sha256": array_sha256(validation_ids),
            "vocabulary": list(train_loader.chars),
        },
        "construction": {
            "seq_len": args.seq_len,
            "split_sizes": dict(zip(SPLIT_NAMES, DEFAULT_SPLIT_SIZES, strict=True)),
            "guard_in_start_positions": args.seq_len,
            "recurrence_threshold": args.recurrence_threshold,
            "strict_induction_uses": "most_recent_prior_bigram",
            "trigram_fit_split": "train",
            "trigram_ties_for_first_are_excluded": True,
            "trigram_surprisal_smoothing": "Laplace alpha=1",
        },
        "labels": label_summary,
        "controls": {
            "saved": [
                "trigram_target_is_top",
                "trigram_target_count",
                "trigram_context_count",
                "trigram_surprisal",
                "long_range_recurrence_shuffled_control",
                "strict_induction_shuffled_control",
            ],
            "study_time": [
                "untrained_seed0_adapter",
                "fixed_ngram_baseline",
            ],
            "shuffled_label_seed": SHUFFLED_LABEL_SEED,
            "shuffled_scope": "training rows only",
        },
        "probe_protocol": {
            "budgets": list(PROBE_BUDGETS),
            "repeats": PROBE_REPEATS,
            "seed": PROBE_SEED,
            "nested_training_rows_array": "probe_train_rows",
        },
        "array_sha256": {
            name: array_sha256(value) for name, value in sorted(arrays.items())
        },
        "dataset_file_sha256": file_sha256(dataset_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # Reopen the serialized artifact before declaring it frozen.
    with np.load(dataset_path, allow_pickle=False) as restored:
        for name, expected in arrays.items():
            if name not in restored or not np.array_equal(restored[name], expected):
                raise AssertionError(f"serialized array failed round-trip: {name}")

    print(f"Dataset:  {dataset_path}")
    print(f"Manifest: {manifest_path}")
    print(f"Dataset SHA-256:  {file_sha256(dataset_path)}")
    print(f"Manifest SHA-256: {file_sha256(manifest_path)}")
    for name, values in label_summary.items():
        print(
            f"{name}: test positives={values['test_positives']}, "
            f"test rate={100.0 * values['test_positive_rate']:.3f}%"
        )


if __name__ == "__main__":
    main()
