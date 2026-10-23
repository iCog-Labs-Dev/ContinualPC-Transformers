"""Fixed, pre-masked MLM validation sets and their evaluation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from fabricpc.core.inference import gather_inputs
from fabricpc.core.scaling import scale_inputs
from fabricpc.core.types import GraphParams, GraphStructure
from fabricpc.experiments.domain_forgetting.artifacts import (
    arrays_sha256,
    atomic_save_npz,
    load_npz,
)
from fabricpc.training.train_backprop import compute_forward_pass, evaluate_mlm
from fabricpc.utils.data.dataloader import apply_mlm_mask

VALID_SHUFFLE_SEED = 1234
VALID_MASK_SEED = 4321
DEFAULT_NUM_VALID_WINDOWS = 2048
VALID_ARRAY_NAMES = (
    "input_ids",
    "masked_input_ids",
    "labels",
    "loss_mask",
    "attention_mask",
)


def pack_windows(
    token_docs: Iterable[np.ndarray],
    *,
    seq_len: int,
    cls_token_id: int,
    sep_token_id: int,
    max_windows: int | None = None,
) -> np.ndarray:
    """Pack documents into full ``[CLS]`` + ``seq_len - 1`` token windows.

    Documents are joined with ``[SEP]`` between them and the stream is cut
    into consecutive ``seq_len - 1`` token segments, each prefixed with
    ``[CLS]``: the layout ``MlmTextBatcher`` produces with ``cls_token_id``
    and its default stride. A trailing partial segment is dropped, and
    reading stops once ``max_windows`` windows are full.
    """
    if seq_len < 2:
        raise ValueError("seq_len must be at least 2")
    if max_windows is not None and max_windows < 1:
        raise ValueError("max_windows must be positive")
    body = seq_len - 1
    pieces: list[np.ndarray] = []
    total = 0
    limit = None if max_windows is None else max_windows * body
    for index, doc in enumerate(token_docs):
        if index > 0:
            pieces.append(np.array([sep_token_id], dtype=np.int64))
            total += 1
        pieces.append(np.asarray(doc, dtype=np.int64))
        total += len(doc)
        if limit is not None and total >= limit:
            break
    stream = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.int64)
    num_windows = len(stream) // body
    if max_windows is not None:
        num_windows = min(num_windows, max_windows)
    segments = stream[: num_windows * body].reshape(num_windows, body)
    cls = np.full((num_windows, 1), cls_token_id, dtype=np.int64)
    return np.concatenate([cls, segments], axis=1).astype(np.int32)


def materialize_validation_set(
    windows: np.ndarray,
    *,
    mask_token_id: int,
    vocab_size: int,
    special_token_ids: set[int],
    seed: int = VALID_MASK_SEED,
    mask_prob: float = 0.15,
) -> dict[str, np.ndarray]:
    """Mask full windows once with BERT 80/10/10 through ``apply_mlm_mask``.

    Returns ``input_ids``, ``masked_input_ids``, ``labels``, ``loss_mask`` and
    ``attention_mask`` (all ones, since every window is full).
    """
    windows = np.asarray(windows, dtype=np.int32)
    masked, labels, loss_mask = apply_mlm_mask(
        windows,
        mask_token_id=mask_token_id,
        vocab_size=vocab_size,
        special_token_ids=set(special_token_ids),
        seed=seed,
        mask_prob=mask_prob,
    )
    return {
        "input_ids": windows,
        "masked_input_ids": masked,
        "labels": labels,
        "loss_mask": loss_mask,
        "attention_mask": np.ones(windows.shape, dtype=bool),
    }


def save_validation_arrays(path: str | Path, arrays: Mapping[str, np.ndarray]) -> str:
    """Atomically save a validation set and return its content SHA-256."""
    missing = [name for name in VALID_ARRAY_NAMES if name not in arrays]
    if missing:
        raise ValueError(f"validation arrays are missing {missing}")
    selected = {name: np.asarray(arrays[name]) for name in VALID_ARRAY_NAMES}
    atomic_save_npz(path, selected)
    return arrays_sha256(selected)


def load_validation_arrays(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    max_windows: int | None = None,
) -> dict[str, np.ndarray]:
    """Load a validation set, check its SHA-256 and keep the first windows."""
    arrays = load_npz(path)
    missing = [name for name in VALID_ARRAY_NAMES if name not in arrays]
    if missing:
        raise ValueError(f"{path}: validation arrays are missing {missing}")
    if expected_sha256 is not None and arrays_sha256(arrays) != expected_sha256:
        raise ValueError(f"{path}: validation arrays do not match their SHA-256")
    if max_windows is not None:
        if max_windows < 1:
            raise ValueError("max_windows must be positive")
        if max_windows > len(arrays["input_ids"]):
            raise ValueError(
                f"{path}: requested {max_windows} windows, "
                f"only {len(arrays['input_ids'])} stored"
            )
        arrays = {name: value[:max_windows] for name, value in arrays.items()}
    return arrays


class FixedMlmBatches:
    """Yield stored validation arrays in order as ``evaluate_mlm`` batches.

    Each batch is ``{"x": masked_input_ids, "y": labels, "loss_mask",
    "attention_mask"}``; the last batch may be smaller than ``batch_size``.
    """

    def __init__(self, arrays: Mapping[str, np.ndarray], batch_size: int) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.arrays = {name: np.asarray(arrays[name]) for name in VALID_ARRAY_NAMES}
        self.batch_size = int(batch_size)
        self.num_windows = int(len(self.arrays["input_ids"]))

    def __len__(self) -> int:
        return math.ceil(self.num_windows / self.batch_size)

    def __iter__(self) -> Iterator[dict[str, np.ndarray]]:
        for start in range(0, self.num_windows, self.batch_size):
            stop = start + self.batch_size
            yield {
                "x": self.arrays["masked_input_ids"][start:stop],
                "y": self.arrays["labels"][start:stop],
                "loss_mask": self.arrays["loss_mask"][start:stop],
                "attention_mask": self.arrays["attention_mask"][start:stop],
            }


def evaluate_fixed_set(
    params: GraphParams,
    structure: GraphStructure,
    arrays: Mapping[str, np.ndarray],
    *,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Evaluate masked CE, perplexity and accuracy on a fixed validation set.

    Reuses ``evaluate_mlm`` (dropout off), which sums loss and correct
    predictions over masked tokens, so CE is the mean over all masked tokens
    of the set rather than a mean of batch means.
    """
    metrics = evaluate_mlm(
        params, structure, FixedMlmBatches(arrays, batch_size), jax.random.PRNGKey(0)
    )
    num_masked = int(metrics["num_masked_tokens"])
    if num_masked == 0:
        raise ValueError("validation set has no masked tokens")
    ce = float(metrics["masked_loss"])
    return {
        "masked_ce": ce,
        "masked_perplexity": float(math.exp(ce)),
        "masked_accuracy": float(metrics["masked_accuracy"]),
        "num_masked_tokens": num_masked,
    }


def head_input(
    structure: GraphStructure, state: Any, output: str | None = None
) -> jax.Array:
    """Return the hidden states the MLM head projects (its ``in`` slot)."""
    output = output or structure.task_map["y"]
    info = structure.nodes[output].node_info
    inputs = scale_inputs(gather_inputs(info, structure, state), info.scaling_config)
    matches = [value for key, value in inputs.items() if key.endswith(":in")]
    if len(matches) != 1:
        raise ValueError(f"{output}: expected exactly one 'in' input")
    return matches[0]


class MultiDomainEvaluator:
    """Evaluate one model on every domain's fixed set in a single pass.

    The domains' windows are concatenated and padded to whole batches with
    windows whose loss mask is empty. Per batch the masked positions are
    gathered (padded to the largest masked count of any batch, so one jitted
    step serves every batch) before the vocabulary projection, so the head
    runs on masked tokens only. CE is the mean over each domain's masked
    tokens of ``-log(p + 1e-10)`` with ``p`` the head's softmax probability,
    the per-token loss of ``evaluate_fixed_set``; dropout is off.
    """

    def __init__(
        self,
        structure: GraphStructure,
        valid_sets: Mapping[str, Mapping[str, np.ndarray]],
        batch_size: int = 64,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.domains = list(valid_sets)
        self.batch_size = int(batch_size)
        arrays = {
            name: np.concatenate(
                [np.asarray(valid_sets[d][name]) for d in self.domains]
            )
            for name in ("masked_input_ids", "labels", "loss_mask", "attention_mask")
        }
        self.domain_index = np.concatenate(
            [
                np.full(len(valid_sets[d]["input_ids"]), i, dtype=np.int64)
                for i, d in enumerate(self.domains)
            ]
        )
        pad = -len(self.domain_index) % self.batch_size
        if pad:
            for name, value in arrays.items():
                filler = np.repeat(value[:1], pad, axis=0)
                if name == "loss_mask":
                    filler = np.zeros_like(filler)
                arrays[name] = np.concatenate([value, filler])
        self.arrays = arrays
        self.num_windows = len(self.domain_index)
        masks = arrays["loss_mask"].astype(bool)
        starts = range(0, len(masks), self.batch_size)
        self.max_masked = max(
            1, max(int(masks[i : i + self.batch_size].sum()) for i in starts)
        )
        self.positions = []
        for start in starts:
            flat = np.flatnonzero(masks[start : start + self.batch_size].reshape(-1))
            index = np.zeros(self.max_masked, dtype=np.int32)
            valid = np.zeros(self.max_masked, dtype=bool)
            index[: len(flat)] = flat
            valid[: len(flat)] = True
            self.positions.append((index, valid))
        output = structure.task_map["y"]
        batch_size_static = self.batch_size

        def step(params, batch, index, valid):
            state = compute_forward_pass(
                params, structure, batch, jax.random.PRNGKey(0), training=False
            )
            hidden = head_input(structure, state, output)
            seq_len = hidden.shape[1]
            gathered = hidden.reshape(-1, hidden.shape[-1])[index]
            head = params.nodes[output]
            logits = gathered @ head.weights["W_out"] + head.biases["b_out"]
            probs = jnp.exp(jax.nn.log_softmax(logits, axis=-1))
            labels = batch["y"].reshape(-1)[index]
            target = jnp.take_along_axis(probs, labels[:, None], axis=-1)[:, 0]
            weight = valid.astype(probs.dtype)
            token_ce = -jnp.log(target + 1e-10) * weight
            correct = (jnp.argmax(probs, axis=-1) == labels).astype(probs.dtype)
            window = index // seq_len
            sums = [
                jax.ops.segment_sum(v, window, num_segments=batch_size_static)
                for v in (token_ce, correct * weight, weight)
            ]
            return tuple(sums)

        self._step = jax.jit(step)

    def __call__(self, params: GraphParams) -> dict[str, dict[str, Any]]:
        sums = []
        for batch_index, start in enumerate(
            range(0, len(self.arrays["labels"]), self.batch_size)
        ):
            stop = start + self.batch_size
            batch = {
                "x": jnp.asarray(self.arrays["masked_input_ids"][start:stop]),
                "y": jnp.asarray(self.arrays["labels"][start:stop]),
                "loss_mask": jnp.asarray(self.arrays["loss_mask"][start:stop]),
                "attention_mask": jnp.asarray(
                    self.arrays["attention_mask"][start:stop]
                ),
            }
            index, valid = self.positions[batch_index]
            values = self._step(params, batch, jnp.asarray(index), jnp.asarray(valid))
            sums.append(np.stack([np.asarray(v) for v in values]))
        totals = np.concatenate(sums, axis=1)[:, : self.num_windows].astype(np.float64)
        results = {}
        for index, name in enumerate(self.domains):
            ce_sum, correct, count = totals[:, self.domain_index == index].sum(axis=1)
            if count == 0:
                raise ValueError(f"{name}: validation set has no masked tokens")
            ce = float(ce_sum / count)
            results[name] = {
                "masked_ce": ce,
                "masked_perplexity": float(math.exp(ce)),
                "masked_accuracy": float(correct / count),
                "num_masked_tokens": int(count),
            }
        return results
