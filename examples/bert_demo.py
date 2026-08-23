"""Unified encoder-only BERT-style demo.

The demo mirrors ``transformer_v2_demo.py``: one entrypoint selects the
experiment stage and training mode with CLI flags, while reusable pieces live in
``fabricpc`` modules.
"""

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
    parser.add_argument("--max_steps", type=int, default=1000)
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
    seed: int,
) -> tuple[MlmTextBatcher, dict[str, int], str | None]:
    vocab = build_tiny_vocab(TINY_MLM_CORPUS)
    token_ids = build_tiny_token_stream(TINY_MLM_CORPUS, vocab, args.seq_len)
    loader = MlmTextBatcher(
        token_ids=token_ids,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        pad_token_id=vocab["[PAD]"],
        mask_token_id=vocab["[MASK]"],
        vocab_size=len(vocab),
        special_token_ids={vocab[token] for token in MLM_SPECIAL_TOKENS},
        seed=seed,
        mask_prob=args.mask_prob,
        max_sequences=args.max_sequences,
        shuffle=False,
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
    texts = (
        TINY_MLM_CORPUS if args.data_source == "tiny" else _load_hf_texts(args, split)
    )
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
        batch_size=args.batch_size,
        pad_token_id=token_to_id("[PAD]"),
        mask_token_id=token_to_id("[MASK]"),
        vocab_size=tokenizer.get_vocab_size(),
        special_token_ids={token_to_id(token) for token in MLM_SPECIAL_TOKENS},
        cls_token_id=token_to_id("[CLS]"),
        seed=seed,
        mask_prob=args.mask_prob,
        max_sequences=args.max_sequences,
        shuffle=False,
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
        loader = load_tokenized_mlm_loader(
            token_path=args.token_path,
            tokenizer_path=args.tokenizer_path,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            seed=seed,
            mask_prob=args.mask_prob,
            max_sequences=args.max_sequences,
            shuffle=False,
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
        return build_tiny_mlm_loader(args, seed=seed)
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


def _model_config(args: argparse.Namespace, vocab_size: int) -> dict[str, int]:
    return {
        "depth": args.depth,
        "embed_dim": args.embed_dim,
        "num_heads": args.num_heads,
        "mlp_dim": args.mlp_dim,
        "seq_len": args.seq_len,
        "vocab_size": vocab_size,
    }


def run_pretrain(args: argparse.Namespace) -> None:
    _reject_unsupported_pc(args)
    rng_key = jax.random.PRNGKey(args.seed)
    train_loader, vocab, tokenizer_json = build_mlm_loader(
        args, split=args.split, seed=args.seed
    )
    val_loader = None
    if args.validation_split:
        if args.data_source == "tiny" and args.vocab_size <= len(
            build_tiny_vocab(TINY_MLM_CORPUS)
        ):
            val_loader, _, _ = build_mlm_loader(
                args,
                split=args.validation_split,
                seed=args.seed + 1,
            )
        elif args.data_source in {"tiny", "hf"}:
            val_loader, _, _ = build_text_mlm_loader(
                args,
                split=args.validation_split,
                seed=args.seed + 1,
                overwrite_tokenizer=False,
            )
        else:
            val_loader, _, _ = build_mlm_loader(
                args,
                split=args.validation_split,
                seed=args.seed + 1,
            )

    structure = create_encoder_mlm_model(
        depth=args.depth,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        mlp_dim=args.mlp_dim,
        seq_len=args.seq_len,
        vocab_size=train_loader.vocab_size,
        inference=InferenceSGDNormClip(eta_infer=0.05, infer_steps=5, max_norm=5.0),
        use_attention_mask=True,
    )
    params = initialize_params(structure, rng_key)
    optimizer = build_optimizer(
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        warmup_steps=args.warmup_steps,
        total_steps=args.max_steps,
    )
    params, state = train_backprop_mlm(
        params,
        structure,
        train_loader,
        optimizer,
        {
            "max_steps": args.max_steps,
            "checkpoint_path": args.training_checkpoint,
            "checkpoint_interval": args.checkpoint_interval,
            "validation_interval": args.validation_interval,
            "resume_from": args.resume_from,
            "use_pmap": args.use_pmap,
            "pmap_single_device": args.pmap_single_device,
            "log_interval": args.log_interval,
            "rolling_loss_window": args.rolling_loss_window,
            "model": _model_config(args, train_loader.vocab_size),
        },
        rng_key,
        val_loader=val_loader,
    )
    if args.checkpoint:
        save_checkpoint(
            args.checkpoint,
            params,
            vocab,
            _model_config(args, train_loader.vocab_size),
            tokenizer_json=tokenizer_json,
        )
        print(f"transfer_checkpoint={args.checkpoint}")
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
