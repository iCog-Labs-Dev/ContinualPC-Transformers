"""BP-only BERT sequential classification forgetting benchmark."""

# ruff: noqa: E402

from jax_setup import set_jax_flags_before_importing_jax

set_jax_flags_before_importing_jax()

import argparse
import inspect
import json
import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from examples.bert_demo import (
    build_optimizer,
    copy_encoder_params,
    create_encoder_sentiment_classifier,
    load_checkpoint,
    save_checkpoint,
)
from fabricpc.core.inference import InferenceSGDNormClip
from fabricpc.core.types import GraphParams
from fabricpc.experiments.forgetting_benchmark.classification_metrics import (
    compute_forgetting_metrics,
)
from fabricpc.graph_initialization import initialize_params
from fabricpc.training.train_backprop import compute_forward_pass, train_step_backprop


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    dataset_id: str
    config: str | None
    text_field: str
    label_field: str
    train_split: str = "train"
    eval_split: str = "test"


DATASET_REGISTRY = {
    "sst2": DatasetSpec(
        "sst2", "glue", "sst2", "sentence", "label", "train", "validation"
    ),
    "imdb": DatasetSpec("imdb", "imdb", None, "text", "label"),
    "yelp_polarity": DatasetSpec(
        "yelp_polarity", "yelp_polarity", None, "text", "label"
    ),
}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def _special_id(vocab: dict[str, int], token: str) -> int:
    if token not in vocab:
        raise ValueError(f"vocab must define {token}")
    return int(vocab[token])


def encode_with_vocab(
    text: str, vocab: dict[str, int], seq_len: int
) -> dict[str, list[int]]:
    """Encode text as [CLS] + tokens[:seq_len-2] + [SEP] + padding."""
    if seq_len < 2:
        raise ValueError("seq_len must be at least 2 to reserve [CLS] and [SEP]")
    pad_id = _special_id(vocab, "[PAD]")
    cls_id = _special_id(vocab, "[CLS]")
    sep_id = _special_id(vocab, "[SEP]")
    unk_id = int(vocab.get("[UNK]", pad_id))
    token_ids = [
        int(vocab.get(token, unk_id)) for token in _tokenize(text)[: seq_len - 2]
    ]
    input_ids = [cls_id, *token_ids, sep_id]
    attention_mask = [1] * len(input_ids)
    padding = [pad_id] * (seq_len - len(input_ids))
    return {
        "input_ids": input_ids + padding,
        "attention_mask": attention_mask + [0] * len(padding),
    }


def _encode_with_tokenizer(
    tokenizer: Any, text: str, seq_len: int
) -> dict[str, list[int]]:
    if seq_len < 2:
        raise ValueError("seq_len must be at least 2 to reserve [CLS] and [SEP]")
    cls_id = tokenizer.token_to_id("[CLS]")
    sep_id = tokenizer.token_to_id("[SEP]")
    pad_id = tokenizer.token_to_id("[PAD]")
    missing = [
        name
        for name, value in (("[CLS]", cls_id), ("[SEP]", sep_id), ("[PAD]", pad_id))
        if value is None
    ]
    if missing:
        raise ValueError(f"tokenizer must define {', '.join(missing)}")
    ids = [cls_id, *tokenizer.encode(text).ids[: seq_len - 2], sep_id]
    padding = [pad_id] * (seq_len - len(ids))
    return {
        "input_ids": ids + padding,
        "attention_mask": [1] * len(ids) + [0] * len(padding),
    }


@dataclass(frozen=True)
class ClassificationBatcher:
    examples: Sequence[tuple[str, int]]
    vocab: dict[str, int]
    seq_len: int
    batch_size: int
    seed: int = 0
    shuffle: bool = False
    tokenizer: Any | None = None

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if any(label not in (0, 1) for _, label in self.examples):
            raise ValueError("classification label must be 0 or 1")

    def __len__(self) -> int:
        return (len(self.examples) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        indices = np.arange(len(self.examples))
        if self.shuffle:
            indices = np.random.default_rng(self.seed).permutation(indices)
        for start in range(0, len(indices), self.batch_size):
            rows = [
                self.examples[int(index)]
                for index in indices[start : start + self.batch_size]
            ]
            encoded = [self._encode(text) for text, _ in rows]
            yield {
                "x": jnp.asarray(
                    [row["input_ids"] for row in encoded], dtype=jnp.int32
                ),
                "attention_mask": jnp.asarray(
                    [row["attention_mask"] for row in encoded], dtype=jnp.float32
                ),
                "y": jnp.asarray([label for _, label in rows], dtype=jnp.int32),
            }

    def _encode(self, text: str) -> dict[str, list[int]]:
        if self.tokenizer is not None:
            return _encode_with_tokenizer(self.tokenizer, text, self.seq_len)
        return encode_with_vocab(text, self.vocab, self.seq_len)


def validate_task_names(
    task_names: Sequence[str], registry: dict[str, DatasetSpec] = DATASET_REGISTRY
) -> list[str]:
    normalized = [task.strip() for task in task_names if task.strip()]
    if not normalized:
        raise ValueError("at least one task is required")
    duplicates = sorted({task for task in normalized if normalized.count(task) > 1})
    if duplicates:
        raise ValueError(f"Duplicate task names are not supported: {duplicates}")
    unsupported = [task for task in normalized if task not in registry]
    if unsupported:
        raise ValueError(
            f"Unsupported task names: {unsupported}. Supported: {sorted(registry)}"
        )
    return normalized


def ensure_output_dir(output_dir: str | Path, overwrite: bool) -> Path:
    path = Path(output_dir)
    _reject_symlink_components(path)
    if path.exists() and not overwrite and any(path.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {path}; pass --overwrite"
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


def _reject_symlink_components(path: Path) -> None:
    current = Path(path.anchor) if path.is_absolute() else Path(".")
    for part in path.parts:
        if part in {"", path.anchor}:
            continue
        current = current / part
        if current.exists() and current.is_symlink():
            raise ValueError(f"Output path must not contain symlinks: {current}")


def _reject_symlink_tree(path: Path) -> None:
    if not path.exists():
        return
    for root, dirs, files in os.walk(path):
        root_path = Path(root)
        for name in [*dirs, *files]:
            candidate = root_path / name
            if candidate.is_symlink():
                raise ValueError(
                    f"Refusing to write with symlink in output tree: {candidate}"
                )


def _assert_not_symlink(path: Path) -> None:
    if path.is_symlink():
        raise ValueError(f"Refusing to write through symlink: {path}")


def validate_sample_limit(value: int | None, flag_name: str) -> None:
    if value is not None and value <= 0:
        raise ValueError(f"{flag_name} must be positive when provided")


def make_evaluation_records(
    task_names: Sequence[str],
    accuracy_matrix: Sequence[Sequence[float]],
    losses: Sequence[Sequence[float | None]] | None = None,
) -> list[dict[str, Any]]:
    records = []
    for stage_index, row in enumerate(accuracy_matrix):
        for task_index, accuracy in enumerate(row):
            records.append(
                {
                    "stage": int(stage_index),
                    "after_task": (
                        None if stage_index == 0 else task_names[stage_index - 1]
                    ),
                    "task": task_names[task_index],
                    "eval_task": task_names[task_index],
                    "task_index": int(task_index),
                    "accuracy": float(accuracy),
                    "loss": None if losses is None else losses[stage_index][task_index],
                }
            )
    return records


def write_benchmark_outputs(
    output_dir: str | Path,
    task_names: Sequence[str],
    accuracy_matrix: Sequence[Sequence[float]],
    evaluation_records: Sequence[dict[str, Any]],
    summary: dict[str, Any],
    config: dict[str, Any],
    stages: Sequence[dict[str, Any]] | None = None,
    loss_matrix: Sequence[Sequence[float | None]] | None = None,
) -> None:
    path = Path(output_dir)
    _reject_symlink_components(path)
    _reject_symlink_tree(path)
    path.mkdir(parents=True, exist_ok=True)
    for output_file in (
        path / "summary.json",
        path / "accuracy_matrix.json",
        path / "evaluations.jsonl",
        path / "experiment_record.json",
    ):
        _assert_not_symlink(output_file)
    (path / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    (path / "accuracy_matrix.json").write_text(
        json.dumps(
            {
                "tasks": list(task_names),
                "accuracy_matrix": accuracy_matrix,
                "loss_matrix": loss_matrix,
                "matrix": accuracy_matrix,
            },
            indent=2,
            sort_keys=True,
        )
    )
    (path / "evaluations.jsonl").write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in evaluation_records)
        + "\n"
    )
    (path / "experiment_record.json").write_text(
        json.dumps(
            {
                "config": config,
                "tasks": list(task_names),
                "stages": list(stages or []),
                "summary": summary,
            },
            indent=2,
            sort_keys=True,
        )
    )


def build_classifier_graph(**kwargs: Any):
    kwargs = dict(kwargs)
    inference = kwargs.pop(
        "inference", InferenceSGDNormClip(eta_infer=0.05, infer_steps=5, max_norm=5.0)
    )
    return create_encoder_sentiment_classifier(
        **kwargs,
        inference=inference,
        use_attention_mask=True,
    )


def load_hf_classification_loader(
    task_name: str,
    split: str,
    batch_size: int,
    seq_len: int,
    seed: int,
    max_samples: int | None,
    vocab: dict[str, int],
    tokenizer: Any | None = None,
) -> ClassificationBatcher:
    validate_sample_limit(max_samples, "max_samples")
    spec = DATASET_REGISTRY[task_name]
    datasets = __import__("datasets")
    dataset_split = spec.train_split if split == "train" else spec.eval_split
    kwargs = {"split": dataset_split, "trust_remote_code": False}
    if spec.config is not None:
        dataset = datasets.load_dataset(spec.dataset_id, spec.config, **kwargs)
    else:
        dataset = datasets.load_dataset(spec.dataset_id, **kwargs)
    examples: list[tuple[str, int]] = []
    for row in dataset:
        text = row.get(spec.text_field)
        label = row.get(spec.label_field)
        if isinstance(text, str) and label in (0, 1):
            examples.append((text, int(label)))
        if max_samples is not None and len(examples) >= max_samples:
            break
    if not examples:
        raise ValueError(f"No examples loaded for task {task_name!r} split {split!r}")
    return ClassificationBatcher(
        examples,
        vocab,
        seq_len=seq_len,
        batch_size=batch_size,
        seed=seed,
        shuffle=split == "train",
        tokenizer=tokenizer,
    )


def evaluate_classifier_sample_weighted(
    params, structure, loader, rng_key
) -> dict[str, float]:
    correct = 0
    total = 0
    loss_sum = 0.0
    for batch in loader:
        state = compute_forward_pass(params, structure, batch, rng_key)
        probs = state.nodes[structure.task_map["y"]].z_mu
        labels = batch["y"]
        batch_size = int(labels.shape[0])
        preds = jnp.argmax(probs, axis=-1)
        correct += int(jnp.sum(preds == labels))
        total += batch_size
        batch_losses = -jnp.log(probs[jnp.arange(batch_size), labels] + 1e-10)
        loss_sum += float(jnp.sum(batch_losses))
    if total == 0:
        raise ValueError("cannot evaluate an empty loader")
    return {
        "accuracy": correct / total,
        "loss": loss_sum / total,
        "num_examples": total,
    }


def train_one_task_backprop(
    params,
    task_name: str,
    train_loader,
    stage_index: int,
    optimizer,
    structure,
    epochs_per_task: int,
    rng_key,
):
    opt_state = optimizer.init(params)
    current_params = params
    key = rng_key
    for _ in range(epochs_per_task):
        for batch in train_loader:
            key, step_key = jax.random.split(key)
            current_params, opt_state, _ = train_step_backprop(
                current_params, opt_state, batch, structure, optimizer, step_key
            )
    return current_params


def _checkpoint_tokenizer(checkpoint: dict[str, Any]) -> Any | None:
    tokenizer_json = checkpoint.get("tokenizer_json")
    if tokenizer_json is None:
        return None
    tokenizers = __import__("tokenizers")
    return tokenizers.Tokenizer.from_str(tokenizer_json)


def _default_optimizer_factory(**kwargs: Any):
    return build_optimizer(
        learning_rate=kwargs["lr"],
        weight_decay=kwargs["weight_decay"],
        grad_clip=kwargs["grad_clip"],
        warmup_steps=kwargs["warmup_steps"],
        total_steps=kwargs["total_steps"],
    )


def _call_loader_factory(loader_factory: Callable[..., Any], **kwargs: Any) -> Any:
    signature = inspect.signature(loader_factory)
    accepted = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
        or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
    }
    return loader_factory(**accepted)


def run_sequential_benchmark(
    *,
    checkpoint: str | Path,
    tasks: Sequence[str],
    output_dir: str | Path,
    epochs_per_task: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    warmup_steps: int,
    grad_clip: float,
    max_train_samples: int | None,
    max_eval_samples: int | None,
    seq_len: int,
    seed: int,
    overwrite: bool,
    save_stage_checkpoints: bool = False,
    checkpoint_loader: Callable[[str], dict[str, Any] | None] = load_checkpoint,
    loader_factory: Callable[..., Any] = load_hf_classification_loader,
    model_factory: Callable[..., Any] = build_classifier_graph,
    train_fn: Callable[..., Any] | None = None,
    evaluate_fn: Callable[..., dict[str, float]] | None = None,
    optimizer_factory: Callable[..., Any] = _default_optimizer_factory,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    task_names = validate_task_names(tasks)
    if epochs_per_task <= 0:
        raise ValueError("epochs_per_task must be positive")
    if seq_len < 2:
        raise ValueError("seq_len must be at least 2")
    validate_sample_limit(max_train_samples, "max_train_samples")
    validate_sample_limit(max_eval_samples, "max_eval_samples")

    loaded_checkpoint = checkpoint_loader(str(checkpoint_path))
    if loaded_checkpoint is None:
        raise FileNotFoundError(
            f"Checkpoint not found or unreadable: {checkpoint_path}"
        )
    output_path = ensure_output_dir(output_dir, overwrite=overwrite)

    vocab = loaded_checkpoint["vocab"]
    tokenizer = _checkpoint_tokenizer(loaded_checkpoint)
    checkpoint_config = loaded_checkpoint.get("config", {})
    effective_seq_len = int(checkpoint_config.get("seq_len", seq_len))
    structure = model_factory(
        depth=int(checkpoint_config.get("depth", 1)),
        embed_dim=int(checkpoint_config.get("embed_dim", 16)),
        num_heads=int(checkpoint_config.get("num_heads", 4)),
        mlp_dim=int(checkpoint_config.get("mlp_dim", 32)),
        seq_len=effective_seq_len,
        vocab_size=len(vocab),
        num_classes=2,
    )
    rng_key = jax.random.PRNGKey(seed)
    checkpoint_params = loaded_checkpoint["params"]
    if isinstance(checkpoint_params, GraphParams):
        params = copy_encoder_params(
            checkpoint_params, initialize_params(structure, rng_key)
        )
    elif train_fn is not None and evaluate_fn is not None:
        params = checkpoint_params
    else:
        raise TypeError("checkpoint params must be GraphParams for the default runner")

    train_loaders = {
        task: _call_loader_factory(
            loader_factory,
            task_name=task,
            split="train",
            batch_size=batch_size,
            seq_len=effective_seq_len,
            seed=seed + index,
            max_samples=max_train_samples,
            vocab=vocab,
            tokenizer=tokenizer,
        )
        for index, task in enumerate(task_names)
    }
    eval_loaders = {
        task: _call_loader_factory(
            loader_factory,
            task_name=task,
            split="eval",
            batch_size=batch_size,
            seq_len=effective_seq_len,
            seed=seed,
            max_samples=max_eval_samples,
            vocab=vocab,
            tokenizer=tokenizer,
        )
        for task in task_names
    }

    actual_train_fn = train_fn
    actual_evaluate_fn = evaluate_fn
    if actual_train_fn is None:

        def default_train_fn(p, task, loader, stage, optimizer):
            return train_one_task_backprop(
                p, task, loader, stage, optimizer, structure, epochs_per_task, rng_key
            )

        actual_train_fn = default_train_fn
    if actual_evaluate_fn is None:

        def default_evaluate_fn(p, task, loader, stage):
            return evaluate_classifier_sample_weighted(p, structure, loader, rng_key)

        actual_evaluate_fn = default_evaluate_fn

    accuracy_matrix: list[list[float]] = []
    loss_matrix: list[list[float | None]] = []
    stages: list[dict[str, Any]] = []

    for stage_index in range(len(task_names) + 1):
        stage_results = [
            actual_evaluate_fn(params, task, eval_loaders[task], stage_index)
            for task in task_names
        ]
        accuracy_matrix.append([float(result["accuracy"]) for result in stage_results])
        loss_matrix.append([result.get("loss") for result in stage_results])
        stages.append(
            {
                "stage": stage_index,
                "trained_task": (
                    None if stage_index == 0 else task_names[stage_index - 1]
                ),
                "evaluated_tasks": list(task_names),
            }
        )
        if stage_index == len(task_names):
            break
        train_task = task_names[stage_index]
        total_steps = max(1, epochs_per_task * len(train_loaders[train_task]))
        optimizer = optimizer_factory(
            lr=lr,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )
        stages[-1]["next_optimizer"] = {
            "reset_for_task": train_task,
            "lr": lr,
            "weight_decay": weight_decay,
            "grad_clip": grad_clip,
            "warmup_steps": warmup_steps,
            "total_steps": total_steps,
        }
        params = actual_train_fn(
            params, train_task, train_loaders[train_task], stage_index + 1, optimizer
        )
        if save_stage_checkpoints:
            stage_path = output_path / f"stage_{stage_index + 1}_{train_task}.json"
            _assert_not_symlink(stage_path)
            _assert_not_symlink(stage_path.with_suffix(stage_path.suffix + ".npz"))
            save_checkpoint(
                str(stage_path),
                params,
                vocab,
                {**checkpoint_config, "stage": stage_index + 1, "task": train_task},
            )

    metrics = compute_forgetting_metrics(accuracy_matrix, task_names)
    summary = {
        "tasks": list(task_names),
        **metrics,
        "FWT_baseline": "initial_pretrained_model",
        "final_accuracy": {
            task: metrics["per_task"][task]["final_accuracy"] for task in task_names
        },
        "optimizer_reset_each_task": True,
        "checkpoint": str(checkpoint_path),
        "seed": seed,
        "max_train_samples": max_train_samples,
        "max_eval_samples": max_eval_samples,
    }
    records = make_evaluation_records(task_names, accuracy_matrix, loss_matrix)
    config = {
        "checkpoint": str(checkpoint_path),
        "epochs_per_task": epochs_per_task,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "warmup_steps": warmup_steps,
        "grad_clip": grad_clip,
        "max_train_samples": max_train_samples,
        "max_eval_samples": max_eval_samples,
        "seq_len": effective_seq_len,
        "seed": seed,
    }
    write_benchmark_outputs(
        output_path,
        task_names,
        accuracy_matrix,
        records,
        summary,
        config,
        stages,
        loss_matrix=loss_matrix,
    )
    return {
        "accuracy_matrix": accuracy_matrix,
        "summary": summary,
        "evaluation_records": records,
        "stages": stages,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tasks", nargs="+", default=["sst2", "imdb", "yelp_polarity"])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs_per_task", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save_stage_checkpoints", action="store_true")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    tasks = [task for item in args.tasks for task in item.split(",") if task]
    run_sequential_benchmark(
        checkpoint=args.checkpoint,
        tasks=tasks,
        output_dir=args.output_dir,
        epochs_per_task=args.epochs_per_task,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        seq_len=args.seq_len,
        seed=args.seed,
        overwrite=args.overwrite,
        save_stage_checkpoints=args.save_stage_checkpoints,
    )


if __name__ == "__main__":
    main()
