"""Unified encoder-only BERT-style demo.

The demo mirrors ``transformer_v2_demo.py``: one entrypoint selects the
experiment stage and training mode with CLI flags, while reusable pieces live in
``fabricpc`` modules.
"""

# ruff: noqa: E402

from jax_setup import set_jax_flags_before_importing_jax

set_jax_flags_before_importing_jax()

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax

from fabricpc.core.inference import InferenceSGDNormClip
from fabricpc.core.types import GraphParams, NodeParams
from fabricpc.graph_initialization import initialize_params
from fabricpc.models import (
    copy_encoder_params,
    create_encoder_mlm_model,
    create_encoder_sentiment_classifier,
)
from fabricpc.training.optimizers import build_warmup_cosine_adamw
from fabricpc.training.train_backprop import (
    compute_forward_pass,
    evaluate_mlm,
    train_backprop_mlm,
    train_step_backprop,
)
from fabricpc.utils.data import (
    MLM_SPECIAL_TOKENS,
    MlmTextBatcher,
    load_tokenized_mlm_loader,
    prepare_bpe_tokenized_texts,
)


PC_UNSUPPORTED_MESSAGE = (
    "PC encoder pretraining/fine-tuning is not implemented yet; use --mode backprop."
)

TINY_SENTIMENT_DATASET = [
    ("a delightful and warm movie", 1),
    ("excellent acting and lovely story", 1),
    ("happy charming wonderful film", 1),
    ("the plot was bright and enjoyable", 1),
    ("a moving performance with heart", 1),
    ("funny sweet and satisfying", 1),
    ("a terrible and boring movie", 0),
    ("awful acting and dull story", 0),
    ("sad lifeless disappointing film", 0),
    ("the plot was flat and annoying", 0),
    ("a bad performance without heart", 0),
    ("tedious cold and unsatisfying", 0),
]
TINY_MLM_CORPUS = [text for text, _ in TINY_SENTIMENT_DATASET]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def build_tiny_vocab(texts: Iterable[str]) -> dict[str, int]:
    tokens = sorted({token for text in texts for token in tokenize(text)})
    vocab = {token: i for i, token in enumerate(MLM_SPECIAL_TOKENS)}
    vocab.update({token: i + len(vocab) for i, token in enumerate(tokens)})
    return vocab


def encode_tiny_text(text: str, vocab: dict[str, int], seq_len: int) -> list[int]:
    ids = [vocab["[CLS]"]]
    ids.extend(vocab.get(token, vocab["[UNK]"]) for token in tokenize(text))
    ids.append(vocab["[SEP]"])
    ids = ids[:seq_len]
    return ids + [vocab["[PAD]"]] * (seq_len - len(ids))


def build_tiny_token_stream(texts: Sequence[str], vocab: dict[str, int], seq_len: int):
    encoded = [encode_tiny_text(text, vocab, seq_len) for text in texts]
    return np.asarray([token for row in encoded for token in row], dtype=np.int32)


def _resolve_partition_split(args: argparse.Namespace, requested_split: str) -> str:
    if args.data_source == "hf" and args.validation_split == args.split:
        return "train" if requested_split == args.split else requested_split
    return requested_split


def _effective_batch_size(args: argparse.Namespace) -> int:
    return args.micro_batch_size or args.batch_size


def _split_source_texts(args: argparse.Namespace, texts: Sequence[str], split: str):
    if split not in {"train", "dev", "test"}:
        return list(texts)
    split_seed = args.seed if args.split_seed is None else args.split_seed
    train, dev, test = deterministic_fraction_split(
        list(texts),
        train_fraction=args.train_fraction,
        dev_fraction=args.dev_fraction,
        test_fraction=args.test_fraction,
        seed=split_seed,
    )
    return {"train": train, "dev": dev, "test": test}[split]


def _shuffle_for_split(args: argparse.Namespace, split: str) -> bool:
    return args.shuffle_train_windows if split == "train" else args.shuffle_eval_windows


@dataclass(frozen=True)
class SentimentBatcher:
    examples: Sequence[tuple[str, int]]
    seq_len: int
    batch_size: int
    encode_fn: Callable[[str, int], Sequence[int]]

    def __len__(self):
        return (len(self.examples) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        for start in range(0, len(self.examples), self.batch_size):
            rows = self.examples[start : start + self.batch_size]
            x = jnp.array(
                [self.encode_fn(text, self.seq_len) for text, _ in rows],
                dtype=jnp.int32,
            )
            y = jnp.array([label for _, label in rows], dtype=jnp.int32)
            yield {"x": x, "y": y}


def normalize_hf_dataset_args(
    dataset_name: str,
    revision: str | None,
) -> tuple[str, str | None]:
    """Validate HuggingFace dataset arguments before calling datasets."""
    normalized_dataset_name = dataset_name.strip()
    normalized_revision = revision.strip() if revision is not None else None
    if not normalized_dataset_name:
        raise ValueError(
            "--dataset_name must be a non-empty HuggingFace dataset id. "
            "Set DATASET_ID or pass --dataset_name explicitly."
        )
    return normalized_dataset_name, normalized_revision or None


def validate_split_fractions(
    train_fraction: float,
    dev_fraction: float,
    test_fraction: float,
    *,
    tolerance: float = 1e-6,
) -> None:
    fractions = (train_fraction, dev_fraction, test_fraction)
    if any(fraction < 0.0 for fraction in fractions):
        raise ValueError("split fractions must be non-negative")
    if abs(sum(fractions) - 1.0) > tolerance:
        raise ValueError("split fractions must sum to 1.0")


def deterministic_fraction_split(
    rows: Sequence[Any],
    *,
    train_fraction: float,
    dev_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[list[Any], list[Any], list[Any]]:
    validate_split_fractions(train_fraction, dev_fraction, test_fraction)
    n_rows = len(rows)
    fractions = (train_fraction, dev_fraction, test_fraction)
    requested = sum(1 for fraction in fractions if fraction > 0.0)
    if n_rows < requested:
        raise ValueError("dataset has too few rows for requested non-zero splits")

    counts = np.floor(np.array(fractions) * n_rows).astype(int)
    remainders = np.array(fractions) * n_rows - counts
    for idx in np.argsort(-remainders)[: n_rows - int(counts.sum())]:
        counts[idx] += 1
    for idx, fraction in enumerate(fractions):
        if fraction > 0.0 and counts[idx] == 0:
            donor = int(np.argmax(counts))
            counts[donor] -= 1
            counts[idx] = 1

    permuted = np.random.default_rng(seed).permutation(n_rows)
    train_end = counts[0]
    dev_end = train_end + counts[1]
    partitions = (permuted[:train_end], permuted[train_end:dev_end], permuted[dev_end:])
    return tuple(
        [rows[int(index)] for index in sorted(indices)] for indices in partitions
    )


def parse_validation_mask_seeds(value: str) -> list[int]:
    seeds = [part.strip() for part in value.split(",") if part.strip()]
    if not seeds:
        raise ValueError("--validation_mask_seeds must include at least one integer")
    return [int(seed) for seed in seeds]


def average_mlm_metrics(metrics: Sequence[dict[str, float]]) -> dict[str, float]:
    if not metrics:
        return {"masked_loss": 0.0, "masked_accuracy": 0.0, "num_masked_tokens": 0}
    return {
        "masked_loss": float(np.mean([m["masked_loss"] for m in metrics])),
        "masked_accuracy": float(np.mean([m["masked_accuracy"] for m in metrics])),
        "num_masked_tokens": int(sum(m["num_masked_tokens"] for m in metrics)),
    }


def evaluate_mlm_with_mask_seeds(
    params: GraphParams,
    structure,
    loader_factory: Callable[[int], MlmTextBatcher],
    rng_key,
    *,
    seeds: Sequence[int],
    strategy: str,
) -> dict[str, float]:
    selected_seeds = [seeds[0]] if strategy == "fixed" else list(seeds)
    return average_mlm_metrics(
        [evaluate_mlm(params, structure, loader_factory(seed), rng_key) for seed in selected_seeds]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BERT-style encoder demo")
    parser.add_argument("--stage", choices=["pretrain", "finetune"], required=True)
    parser.add_argument(
        "--mode",
        choices=["backprop", "pc"],
        default="backprop",
        help="Training mode. PC is reserved for future encoder objective work.",
    )
    parser.add_argument(
        "--data_source",
        choices=["tiny", "hf", "tokenized", "synthetic"],
        default="tiny",
        help="Data source for the selected stage.",
    )
    parser.add_argument("--dataset_name", default="")
    parser.add_argument("--split", default="train")
    parser.add_argument("--validation_split", default=None)
    parser.add_argument("--text_field", default="text")
    parser.add_argument("--label_field", default="label")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--token_path", default=None)
    parser.add_argument("--validation_token_path", default=None)
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--cache_dir", default="data/bert_demo_bpe")
    parser.add_argument("--checkpoint", default="data/bert_demo_mlm_params.json")
    parser.add_argument("--training_checkpoint", default=None)
    parser.add_argument("--resume_from", default=None)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--vocab_size", type=int, default=16000)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--mlp_dim", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--train_epochs_equivalent", type=float, default=1.0)
    parser.add_argument("--max_sequences", type=int, default=None)
    parser.add_argument("--max_texts", type=int, default=None)
    parser.add_argument("--max_chars", type=int, default=100_000)
    parser.add_argument("--max_total_chars", type=int, default=100_000_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--mask_prob", type=float, default=0.15)
    parser.add_argument("--checkpoint_interval", type=int, default=1000)
    parser.add_argument("--validation_interval", type=int, default=1000)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--rolling_loss_window", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train_fraction", type=float, default=0.98)
    parser.add_argument("--dev_fraction", type=float, default=0.01)
    parser.add_argument("--test_fraction", type=float, default=0.01)
    parser.add_argument("--split_seed", type=int, default=None)
    parser.add_argument(
        "--shuffle_train_windows",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--shuffle_eval_windows",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--validation_mask_seeds", default="0")
    parser.add_argument(
        "--validation_mask_strategy", choices=["fixed", "average"], default="fixed"
    )
    parser.add_argument("--dropout_rate", type=float, default=0.0)
    parser.add_argument("--embedding_init_std", type=float, default=1.0)
    parser.add_argument("--position_init_std", type=float, default=0.02)
    parser.add_argument("--transformer_init_std", type=float, default=0.02)
    parser.add_argument("--tokens_per_update", type=int, default=None)
    parser.add_argument("--micro_batch_size", type=int, default=None)
    parser.add_argument("--metrics_jsonl", default=None)
    parser.add_argument("--compare_scratch", action="store_true")
    parser.add_argument("--overwrite_cache", action="store_true")
    parser.add_argument("--use_pmap", action="store_true")
    parser.add_argument("--pmap_single_device", action="store_true")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _reject_unsupported_pc(args: argparse.Namespace) -> None:
    if args.mode == "pc":
        raise NotImplementedError(PC_UNSUPPORTED_MESSAGE)


def build_optimizer(
    *,
    learning_rate: float,
    weight_decay: float,
    grad_clip: float,
    warmup_steps: int,
    total_steps: int,
) -> optax.GradientTransformation:
    optimizer = build_warmup_cosine_adamw(
        learning_rate=learning_rate,
        total_steps=max(1, total_steps),
        warmup_steps=min(warmup_steps, max(1, total_steps)),
        weight_decay=weight_decay,
    )
    if grad_clip <= 0:
        return optimizer
    return optax.chain(optax.clip_by_global_norm(grad_clip), optimizer)


def _checkpoint_tensor_key(
    node_name: str,
    collection_name: str,
    param_name: str,
) -> str:
    return f"{node_name}/{collection_name}/{param_name}"


def save_checkpoint(
    path: str,
    params: GraphParams,
    vocab: dict[str, int],
    config: dict[str, Any],
    tokenizer_json: str | None = None,
) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tensor_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".npz")
    tensors = {}
    param_schema = {}
    for node_name, node_params in params.nodes.items():
        node_schema = {"weights": {}, "biases": {}}
        for collection_name in ("weights", "biases"):
            for param_name, value in getattr(node_params, collection_name).items():
                tensor_key = _checkpoint_tensor_key(
                    node_name,
                    collection_name,
                    param_name,
                )
                tensors[tensor_key] = np.asarray(value)
                node_schema[collection_name][param_name] = tensor_key
        param_schema[node_name] = node_schema

    np.savez(tensor_path, **tensors)
    checkpoint_path.write_text(
        json.dumps(
            {
                "format": "fabricpc-bert-demo-checkpoint",
                "version": 1,
                "tensor_file": tensor_path.name,
                "params": param_schema,
                "tokenizer_json": tokenizer_json,
                "vocab": vocab,
                "config": config,
            },
            indent=2,
            sort_keys=True,
        )
    )


def load_checkpoint(path: str) -> dict[str, Any] | None:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        return None

    metadata = json.loads(checkpoint_path.read_text())
    if metadata.get("format") != "fabricpc-bert-demo-checkpoint":
        raise ValueError("Unsupported checkpoint format")
    if metadata.get("version") != 1:
        raise ValueError("Unsupported checkpoint version")
    if not isinstance(metadata.get("vocab"), dict):
        raise ValueError("Checkpoint vocab must be a dictionary")
    if not isinstance(metadata.get("config"), dict):
        raise ValueError("Checkpoint config must be a dictionary")
    if not isinstance(metadata.get("params"), dict):
        raise ValueError("Checkpoint params schema must be a dictionary")
    tokenizer_json = metadata.get("tokenizer_json")
    if tokenizer_json is not None and not isinstance(tokenizer_json, str):
        raise ValueError("Checkpoint tokenizer_json must be a string or null")

    tensor_file = metadata.get("tensor_file")
    if not isinstance(tensor_file, str):
        raise ValueError("Checkpoint tensor_file must be a string")
    tensor_path = checkpoint_path.with_name(tensor_file)
    if not tensor_path.exists():
        raise ValueError(f"Checkpoint tensor file not found: {tensor_path}")

    with np.load(tensor_path, allow_pickle=False) as tensors:
        nodes = {}
        for node_name, node_schema in metadata["params"].items():
            if not isinstance(node_schema, dict):
                raise ValueError(f"Invalid parameter schema for node {node_name!r}")
            collections = {}
            for collection_name in ("weights", "biases"):
                collection_schema = node_schema.get(collection_name)
                if not isinstance(collection_schema, dict):
                    raise ValueError(
                        f"Invalid {collection_name} schema for node {node_name!r}"
                    )
                values = {}
                for param_name, tensor_key in collection_schema.items():
                    if not isinstance(tensor_key, str) or tensor_key not in tensors:
                        raise ValueError(f"Missing tensor {tensor_key!r}")
                    values[param_name] = jnp.array(tensors[tensor_key])
                collections[collection_name] = values
            nodes[node_name] = NodeParams(
                weights=collections["weights"], biases=collections["biases"]
            )

    return {
        "params": GraphParams(nodes=nodes),
        "tokenizer_json": tokenizer_json,
        "vocab": metadata["vocab"],
        "config": metadata["config"],
    }


def build_tiny_mlm_loader(
    args: argparse.Namespace,
    *,
    split: str = "train",
    seed: int,
) -> tuple[MlmTextBatcher, dict[str, int], str | None]:
    vocab = build_tiny_vocab(TINY_MLM_CORPUS)
    texts = _split_source_texts(args, TINY_MLM_CORPUS, split)
    token_ids = build_tiny_token_stream(texts, vocab, args.seq_len)
    loader = MlmTextBatcher(
        token_ids=token_ids,
        seq_len=args.seq_len,
        batch_size=_effective_batch_size(args),
        pad_token_id=vocab["[PAD]"],
        mask_token_id=vocab["[MASK]"],
        vocab_size=len(vocab),
        special_token_ids={vocab[token] for token in MLM_SPECIAL_TOKENS},
        seed=seed,
        mask_prob=args.mask_prob,
        max_sequences=args.max_sequences,
        shuffle=_shuffle_for_split(args, split),
        token_unit="word",
    )
    return loader, vocab, None


def _load_hf_texts(args: argparse.Namespace, split: str) -> list[str]:
    datasets = __import__("datasets")
    dataset_name, revision = normalize_hf_dataset_args(args.dataset_name, args.revision)
    dataset = datasets.load_dataset(dataset_name, split=split, revision=revision)
    texts = []
    total_chars = 0
    for row in dataset:
        value = row.get(args.text_field)
        if not isinstance(value, str) or not value.strip():
            continue
        text = value[: args.max_chars]
        texts.append(text)
        total_chars += len(text)
        if args.max_texts is not None and len(texts) >= args.max_texts:
            break
        if total_chars >= args.max_total_chars:
            break
    if not texts:
        raise ValueError(f"No text found in {args.dataset_name!r} split {split!r}")
    return texts


def build_text_mlm_loader(
    args: argparse.Namespace,
    *,
    split: str,
    seed: int,
    overwrite_tokenizer: bool = True,
) -> tuple[MlmTextBatcher, dict[str, int], str | None]:
    if args.data_source == "tiny":
        texts = TINY_MLM_CORPUS
    else:
        source_split = args.split if split in {"train", "dev", "test"} else split
        texts = _load_hf_texts(args, source_split)
    texts = _split_source_texts(args, texts, _resolve_partition_split(args, split))
    token_ids, tokenizer = prepare_bpe_tokenized_texts(
        texts=texts,
        cache_dir=args.cache_dir,
        vocab_size=args.vocab_size,
        special_tokens=MLM_SPECIAL_TOKENS,
        token_stem=f"tokens_{split}",
        cache_metadata={
            "dataset_name": "tiny" if args.data_source == "tiny" else args.dataset_name,
            "text_field": args.text_field,
        },
        overwrite=args.overwrite_cache,
        overwrite_tokenizer=overwrite_tokenizer,
    )
    token_to_id = tokenizer.token_to_id
    loader = MlmTextBatcher(
        token_ids=token_ids,
        seq_len=args.seq_len,
        batch_size=_effective_batch_size(args),
        pad_token_id=token_to_id("[PAD]"),
        mask_token_id=token_to_id("[MASK]"),
        vocab_size=tokenizer.get_vocab_size(),
        special_token_ids={token_to_id(token) for token in MLM_SPECIAL_TOKENS},
        cls_token_id=token_to_id("[CLS]"),
        seed=seed,
        mask_prob=args.mask_prob,
        max_sequences=args.max_sequences,
        shuffle=_shuffle_for_split(args, split),
        token_unit="subword",
    )
    return loader, tokenizer.get_vocab(), tokenizer.to_str()


def build_mlm_loader(
    args: argparse.Namespace,
    *,
    split: str,
    seed: int,
) -> tuple[MlmTextBatcher, dict[str, int], str | None]:
    if args.data_source == "synthetic":
        raise ValueError("--data_source synthetic is only valid for fine-tuning")
    if args.data_source == "tokenized":
        if not args.token_path or not args.tokenizer_path:
            raise ValueError("--token_path and --tokenizer_path are required")
        token_path = args.token_path
        if split != args.split:
            if not args.validation_token_path:
                raise ValueError(
                    "tokenized validation requires --validation_token_path; "
                    "validation_split on --token_path would leak train tokens"
                )
            token_path = args.validation_token_path
        loader = load_tokenized_mlm_loader(
            token_path=token_path,
            tokenizer_path=args.tokenizer_path,
            seq_len=args.seq_len,
            batch_size=_effective_batch_size(args),
            seed=seed,
            mask_prob=args.mask_prob,
            max_sequences=args.max_sequences,
            shuffle=_shuffle_for_split(args, split),
        )
        tokenizers = __import__("tokenizers")
        tokenizer_json = Path(args.tokenizer_path).read_text()
        tokenizer = tokenizers.Tokenizer.from_str(tokenizer_json)
        return (
            loader,
            {str(token): int(idx) for token, idx in tokenizer.get_vocab().items()},
            tokenizer_json,
        )
    if args.data_source == "tiny" and args.vocab_size <= len(
        build_tiny_vocab(TINY_MLM_CORPUS)
    ):
        return build_tiny_mlm_loader(args, split=split, seed=seed)
    return build_text_mlm_loader(args, split=split, seed=seed)


def evaluate_classifier(params, structure, loader, rng_key):
    correct = 0
    total = 0
    losses = []
    for batch in loader:
        state = compute_forward_pass(params, structure, batch, rng_key)
        probs = state.nodes[structure.task_map["y"]].z_mu
        labels = batch["y"]
        preds = jnp.argmax(probs, axis=-1)
        correct += int(jnp.sum(preds == labels))
        total += int(labels.shape[0])
        losses.append(
            float(
                -jnp.mean(jnp.log(probs[jnp.arange(labels.shape[0]), labels] + 1e-10))
            )
        )
    return {"accuracy": correct / total, "loss": sum(losses) / len(losses)}


def train_classifier(params, structure, train_loader, test_loader, args, rng_key):
    optimizer = build_optimizer(
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        warmup_steps=args.warmup_steps,
        total_steps=args.num_epochs * len(train_loader),
    )
    opt_state = optimizer.init(params)
    for _ in range(args.num_epochs):
        for batch in train_loader:
            rng_key, step_key = jax.random.split(rng_key)
            params, opt_state, _ = train_step_backprop(
                params,
                opt_state,
                batch,
                structure,
                optimizer,
                step_key,
            )
    return params, evaluate_classifier(params, structure, test_loader, rng_key)


def build_checkpoint_text_encoder(
    checkpoint: dict[str, Any] | None,
    vocab: dict[str, int],
) -> Callable[[str, int], Sequence[int]]:
    tokenizer_json = None if checkpoint is None else checkpoint.get("tokenizer_json")
    if tokenizer_json is None:
        return lambda text, seq_len: encode_tiny_text(text, vocab, seq_len)

    tokenizers = __import__("tokenizers")
    tokenizer = tokenizers.Tokenizer.from_str(tokenizer_json)
    cls_id = tokenizer.token_to_id("[CLS]")
    sep_id = tokenizer.token_to_id("[SEP]")
    pad_id = tokenizer.token_to_id("[PAD]")
    if cls_id is None or sep_id is None or pad_id is None:
        raise ValueError("Checkpoint tokenizer must define [CLS], [SEP], and [PAD]")

    def encode_with_tokenizer(text: str, seq_len: int) -> Sequence[int]:
        ids = [cls_id]
        ids.extend(tokenizer.encode(text).ids)
        ids.append(sep_id)
        ids = ids[:seq_len]
        return ids + [pad_id] * (seq_len - len(ids))

    return encode_with_tokenizer


def _split_tiny_sentiment_data():
    positives = [row for row in TINY_SENTIMENT_DATASET if row[1] == 1]
    negatives = [row for row in TINY_SENTIMENT_DATASET if row[1] == 0]
    return positives[:5] + negatives[:5], positives[5:] + negatives[5:]


def _model_config(args: argparse.Namespace, vocab_size: int) -> dict[str, int | float]:
    return {
        "depth": args.depth,
        "embed_dim": args.embed_dim,
        "num_heads": args.num_heads,
        "mlp_dim": args.mlp_dim,
        "seq_len": args.seq_len,
        "vocab_size": vocab_size,
        "dropout_rate": args.dropout_rate,
        "embedding_init_std": args.embedding_init_std,
        "position_init_std": args.position_init_std,
        "transformer_init_std": args.transformer_init_std,
    }


def compute_gradient_accumulation_steps(args: argparse.Namespace) -> int:
    if args.tokens_per_update is None:
        return 1
    micro_batch_size = _effective_batch_size(args)
    micro_tokens = micro_batch_size * args.seq_len
    return max(1, int(np.ceil(args.tokens_per_update / micro_tokens)))


def derive_max_steps(args: argparse.Namespace, loader: MlmTextBatcher) -> int:
    if args.max_steps is not None:
        return int(args.max_steps)
    metadata = loader.metadata
    tokens_per_update = args.tokens_per_update or metadata["max_tokens_per_batch"]
    if tokens_per_update <= 0:
        return max(1, len(loader))
    target_tokens = metadata["source_token_count"] * args.train_epochs_equivalent
    return max(1, int(np.ceil(target_tokens / tokens_per_update)))


def summarize_init_diagnostics(params: GraphParams) -> dict[str, float]:
    leaves = [np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(params)]
    flat = np.concatenate([leaf.reshape(-1) for leaf in leaves if leaf.size])
    diagnostics = {
        "param_mean": float(np.mean(flat)),
        "param_std": float(np.std(flat)),
        "param_abs_max": float(np.max(np.abs(flat))),
    }
    token_embeddings = params.nodes.get("token_embeddings")
    if token_embeddings is not None and "embeddings" in token_embeddings.weights:
        diagnostics["token_embedding_std"] = float(
            np.std(np.asarray(token_embeddings.weights["embeddings"]))
        )
    position_embeddings = params.nodes.get("position_embeddings")
    if (
        position_embeddings is not None
        and "position_embeddings" in position_embeddings.weights
    ):
        diagnostics["position_embedding_std"] = float(
            np.std(np.asarray(position_embeddings.weights["position_embeddings"]))
        )
    return diagnostics


def build_validation_mlm_loader(
    args: argparse.Namespace,
    *,
    seed: int,
) -> MlmTextBatcher:
    if args.validation_split is None:
        raise ValueError("--validation_split is required")
    validation_split = args.validation_split
    if args.data_source == "hf" and validation_split == args.split:
        validation_split = "dev"
    if args.data_source == "tiny" and args.vocab_size <= len(build_tiny_vocab(TINY_MLM_CORPUS)):
        loader, _, _ = build_mlm_loader(args, split=validation_split, seed=seed)
        return loader
    if args.data_source in {"tiny", "hf"}:
        loader, _, _ = build_text_mlm_loader(
            args,
            split=validation_split,
            seed=seed,
            overwrite_tokenizer=False,
        )
        return loader
    loader, _, _ = build_mlm_loader(args, split=validation_split, seed=seed)
    return loader


def run_pretrain(args: argparse.Namespace) -> None:
    _reject_unsupported_pc(args)
    validation_mask_seeds = parse_validation_mask_seeds(args.validation_mask_seeds)
    rng_key = jax.random.PRNGKey(args.seed)
    train_loader, vocab, tokenizer_json = build_mlm_loader(
        args, split=args.split, seed=args.seed
    )
    val_loader = None
    if args.validation_split:
        val_loader = build_validation_mlm_loader(args, seed=validation_mask_seeds[0])

    structure = create_encoder_mlm_model(
        depth=args.depth,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        mlp_dim=args.mlp_dim,
        seq_len=args.seq_len,
        vocab_size=train_loader.vocab_size,
        inference=InferenceSGDNormClip(eta_infer=0.05, infer_steps=5, max_norm=5.0),
        use_attention_mask=True,
        dropout_rate=args.dropout_rate,
        embedding_init_std=args.embedding_init_std,
        position_init_std=args.position_init_std,
        transformer_init_std=args.transformer_init_std,
    )
    params = initialize_params(structure, rng_key)
    init_diagnostics = {
        **summarize_init_diagnostics(params),
        "configured_embedding_init_std": args.embedding_init_std,
        "configured_position_init_std": args.position_init_std,
        "configured_transformer_init_std": args.transformer_init_std,
    }
    max_steps = derive_max_steps(args, train_loader)
    gradient_accumulation_steps = compute_gradient_accumulation_steps(args)
    optimizer = build_optimizer(
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        warmup_steps=args.warmup_steps,
        total_steps=max_steps,
    )
    params, state = train_backprop_mlm(
        params,
        structure,
        train_loader,
        optimizer,
        {
            "max_steps": max_steps,
            "checkpoint_path": args.training_checkpoint,
            "checkpoint_interval": args.checkpoint_interval,
            "validation_interval": args.validation_interval,
            "resume_from": args.resume_from,
            "use_pmap": args.use_pmap,
            "pmap_single_device": args.pmap_single_device,
            "log_interval": args.log_interval,
            "rolling_loss_window": args.rolling_loss_window,
            "tokens_per_update": args.tokens_per_update,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "metrics_jsonl": args.metrics_jsonl,
            "init_diagnostics": init_diagnostics,
            "model": _model_config(args, train_loader.vocab_size),
            "validation_fn": None,
        },
        rng_key,
        val_loader=val_loader,
    )
    if args.checkpoint:
        save_checkpoint(
            args.checkpoint,
            params,
            vocab,
            {
                **_model_config(args, train_loader.vocab_size),
                "init_diagnostics": init_diagnostics,
                "max_steps": max_steps,
                "tokens_per_update": args.tokens_per_update,
                "gradient_accumulation_steps": gradient_accumulation_steps,
            },
            tokenizer_json=tokenizer_json,
        )
        print(f"transfer_checkpoint={args.checkpoint}")
    if args.validation_split:
        validation_metrics = evaluate_mlm_with_mask_seeds(
            params,
            structure,
            lambda seed: build_validation_mlm_loader(args, seed=seed),
            rng_key,
            seeds=validation_mask_seeds,
            strategy=args.validation_mask_strategy,
        )
        print(
            f"final_validation_masked_loss={validation_metrics['masked_loss']:.4f} "
            f"final_validation_masked_accuracy={validation_metrics['masked_accuracy']:.4f} "
            f"final_validation_masked_tokens={validation_metrics['num_masked_tokens']:.0f}"
        )
    print(
        "init_diagnostics "
        f"param_mean={init_diagnostics['param_mean']:.6f} "
        f"param_std={init_diagnostics['param_std']:.6f} "
        f"param_abs_max={init_diagnostics['param_abs_max']:.6f}"
    )
    print(f"done stage=pretrain step={state['step']}")


def run_finetune(args: argparse.Namespace) -> None:
    _reject_unsupported_pc(args)
    if args.data_source not in {"tiny", "synthetic"}:
        raise ValueError("Fine-tuning currently supports --data_source tiny|synthetic")
    rng_key = jax.random.PRNGKey(args.seed)
    checkpoint = load_checkpoint(args.checkpoint) if args.checkpoint else None

    if checkpoint is not None:
        vocab = checkpoint["vocab"]
        checkpoint_config = checkpoint.get("config", {})
        args.seq_len = checkpoint_config.get("seq_len", args.seq_len)
        args.embed_dim = checkpoint_config.get("embed_dim", args.embed_dim)
        args.num_heads = checkpoint_config.get("num_heads", args.num_heads)
        args.mlp_dim = checkpoint_config.get("mlp_dim", args.mlp_dim)
        args.depth = checkpoint_config.get("depth", args.depth)
    else:
        vocab = build_tiny_vocab(text for text, _ in TINY_SENTIMENT_DATASET)
    encode_fn = build_checkpoint_text_encoder(checkpoint, vocab)

    train_data, test_data = _split_tiny_sentiment_data()
    train_loader = SentimentBatcher(
        train_data, args.seq_len, args.batch_size, encode_fn
    )
    test_loader = SentimentBatcher(test_data, args.seq_len, args.batch_size, encode_fn)

    structure = create_encoder_sentiment_classifier(
        depth=args.depth,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        mlp_dim=args.mlp_dim,
        seq_len=args.seq_len,
        vocab_size=len(vocab),
        num_classes=2,
        inference=InferenceSGDNormClip(eta_infer=0.05, infer_steps=5, max_norm=5.0),
        dropout_rate=args.dropout_rate,
        embedding_init_std=args.embedding_init_std,
        position_init_std=args.position_init_std,
        transformer_init_std=args.transformer_init_std,
    )

    rng_key, scratch_key, transfer_key = jax.random.split(rng_key, 3)
    if checkpoint is None or args.compare_scratch:
        scratch_params = initialize_params(structure, scratch_key)
        _, scratch_metrics = train_classifier(
            scratch_params,
            structure,
            train_loader,
            test_loader,
            args,
            scratch_key,
        )
        print(
            f"scratch test_loss={scratch_metrics['loss']:.4f} "
            f"test_acc={scratch_metrics['accuracy']:.2%}"
        )

    if checkpoint is None:
        print(f"transferred skipped: checkpoint not found at {args.checkpoint}")
        return

    transfer_params = initialize_params(structure, transfer_key)
    transfer_params = copy_encoder_params(checkpoint["params"], transfer_params)
    _, transfer_metrics = train_classifier(
        transfer_params,
        structure,
        train_loader,
        test_loader,
        args,
        transfer_key,
    )
    print(
        f"transferred test_loss={transfer_metrics['loss']:.4f} "
        f"test_acc={transfer_metrics['accuracy']:.2%}"
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.stage == "pretrain":
        run_pretrain(args)
    elif args.stage == "finetune":
        run_finetune(args)
    else:  # pragma: no cover - argparse enforces choices.
        raise ValueError(f"Unsupported stage: {args.stage}")


if __name__ == "__main__":
    main()
