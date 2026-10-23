#!/usr/bin/env python3
"""Standalone validation for FabricPC MLM checkpoints.

Loads a checkpoint and evaluates MLM loss/accuracy on the cached validation split.
No dependency on bert_demo.py — loads BPE cache directly.

Usage:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python examples/validate_checkpoint.py \\
        --checkpoint data/checkpoints/babylm_124m_50k_dropout01_training.json
"""

# ruff: noqa: E402

import argparse
import json
import sys
from pathlib import Path

from jax_setup import set_jax_flags_before_importing_jax

set_jax_flags_before_importing_jax()

import numpy as np
import jax
import jax.numpy as jnp

from examples.bert_demo import build_text_mlm_loader
from fabricpc.core.inference import InferenceSGDNormClip
from fabricpc.core.types import GraphParams, NodeParams
from fabricpc.nodes.encoder_transformer import create_encoder_mlm_model
from fabricpc.training.train import replicate_params, shard_batch
from fabricpc.training.train_backprop import compute_forward_pass, evaluate_mlm
from fabricpc.utils.data.dataloader import MlmTextBatcher

MLM_SPECIAL_TOKENS = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]")


def required_token_id(tokenizer, token: str) -> int:
    token_id = tokenizer.token_to_id(token)
    if token_id is None:
        raise ValueError(f"Tokenizer missing required token {token!r}")
    return int(token_id)


def normalize_vocab(vocab: dict | None) -> dict[str, int] | None:
    if not isinstance(vocab, dict):
        return None
    return {str(token): int(token_id) for token, token_id in vocab.items()}


def validate_tokenizer_compatibility(
    tokenizer_path: Path,
    expected_vocab_size: int,
    expected_vocab: dict[str, int] | None,
) -> None:
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer_vocab = {
        str(token): int(token_id) for token, token_id in tokenizer.get_vocab().items()
    }
    tokenizer_vocab_size = tokenizer.get_vocab_size()
    if tokenizer_vocab_size != expected_vocab_size:
        raise ValueError(
            f"Tokenizer vocab size {tokenizer_vocab_size} does not match "
            f"checkpoint vocab size {expected_vocab_size}"
        )
    if expected_vocab is not None and tokenizer_vocab != expected_vocab:
        raise ValueError(
            "Cached tokenizer vocabulary does not match checkpoint vocabulary. "
            "Use the original training cache for validation."
        )


def load_checkpoint(path: str) -> dict:
    ckpt_path = Path(path)
    meta = json.loads(ckpt_path.read_text())
    fmt = meta.get("format", "")
    tensor_file = meta.get("tensor_file")
    tensors = np.load(ckpt_path.with_name(tensor_file), allow_pickle=False)

    nodes = {}
    if fmt == "fabricpc-bert-demo-checkpoint":
        for node_name, schema in meta["params"].items():
            weights = {
                k: jnp.array(tensors[v]) for k, v in schema.get("weights", {}).items()
            }
            biases = {
                k: jnp.array(tensors[v]) for k, v in schema.get("biases", {}).items()
            }
            nodes[node_name] = NodeParams(weights=weights, biases=biases)
    elif fmt == "fabricpc-mlm-training-checkpoint":
        node_map: dict[str, dict[str, dict[str, jnp.ndarray]]] = {}
        for key in tensors.files:
            if not key.startswith("params/"):
                continue
            parts = key.split("/")
            if len(parts) >= 4:
                node_name = parts[1]
                collection = parts[2]
                param_name = "/".join(parts[3:])
                if node_name not in node_map:
                    node_map[node_name] = {"weights": {}, "biases": {}}
                node_map[node_name][collection][param_name] = jnp.array(tensors[key])
        for node_name, collections in node_map.items():
            nodes[node_name] = NodeParams(
                weights=collections.get("weights", {}),
                biases=collections.get("biases", {}),
            )
    else:
        raise ValueError(f"Unknown checkpoint format: {fmt!r}")

    config = meta.get("config", {})
    if "model" in config and isinstance(config["model"], dict):
        config = config["model"]

    return {
        "params": GraphParams(nodes=nodes),
        "config": config,
        "step": meta.get("step", 0),
        "format": fmt,
        "vocab": normalize_vocab(meta.get("vocab")),
    }


def load_val_loader(
    cache_dir: str,
    split: str,
    config: dict,
    mask_prob: float,
    seed: int,
    batch_size: int,
    expected_vocab_size: int,
    expected_vocab: dict[str, int] | None,
    max_sequences: int | None,
) -> MlmTextBatcher:
    """Load cached validation tokens and build MlmTextBatcher directly."""
    cache_path = Path(cache_dir)
    tokenizer_path = cache_path / "tokenizer.json"
    tokens_path = cache_path / f"tokens_{split}.npy"

    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    if not tokens_path.exists():
        raise FileNotFoundError(f"Validation tokens not found: {tokens_path}")

    token_ids = np.load(tokens_path, allow_pickle=False)

    from tokenizers import Tokenizer

    validate_tokenizer_compatibility(
        tokenizer_path,
        expected_vocab_size=expected_vocab_size,
        expected_vocab=expected_vocab,
    )
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer_vocab_size = tokenizer.get_vocab_size()
    special_token_ids = {
        token: required_token_id(tokenizer, token) for token in MLM_SPECIAL_TOKENS
    }

    seq_len = config.get("seq_len", 128)

    loader = MlmTextBatcher(
        token_ids=token_ids,
        seq_len=seq_len,
        batch_size=batch_size,
        pad_token_id=special_token_ids["[PAD]"],
        mask_token_id=special_token_ids["[MASK]"],
        vocab_size=tokenizer_vocab_size,
        special_token_ids=set(special_token_ids.values()),
        cls_token_id=special_token_ids["[CLS]"],
        seed=seed,
        mask_prob=mask_prob,
        max_sequences=max_sequences,
        shuffle=False,
        token_unit="subword",
    )
    return loader


def build_val_loader(
    args: argparse.Namespace,
    config: dict,
    vocab_size: int,
    expected_vocab: dict[str, int] | None,
) -> MlmTextBatcher:
    """Prefer cached validation tokens; otherwise build them from a HF split."""
    split = args.validation_split or "valid"
    tokens_path = Path(args.cache_dir) / f"tokens_{split}.npy"
    tokenizer_path = Path(args.cache_dir) / "tokenizer.json"
    if tokens_path.exists() and tokenizer_path.exists():
        print(f"Loading validation tokens from cache ({tokens_path.name})...")
        return load_val_loader(
            args.cache_dir,
            split,
            config,
            mask_prob=args.mask_prob,
            seed=args.seed,
            batch_size=args.batch_size,
            expected_vocab_size=vocab_size,
            expected_vocab=expected_vocab,
            max_sequences=args.max_sequences,
        )

    if not args.dataset_name:
        raise FileNotFoundError(
            f"Validation tokens not found: {tokens_path}. Pass --dataset_name to build them."
        )
    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Tokenizer not found: {tokenizer_path}. Reuse the original training cache; "
            "do not train a new tokenizer for checkpoint validation."
        )
    validate_tokenizer_compatibility(
        tokenizer_path,
        expected_vocab_size=vocab_size,
        expected_vocab=expected_vocab,
    )

    print(f"Building validation loader from HF split {split!r}...")
    args.data_source = "hf"
    args.split = args.source_split
    args.seq_len = config.get("seq_len", 128)
    args.vocab_size = int(vocab_size)
    val_loader, _, _ = build_text_mlm_loader(
        args,
        split=split,
        seed=args.seed,
        overwrite_tokenizer=False,
    )
    return val_loader


def describe_loader(loader: MlmTextBatcher) -> str:
    window_count = getattr(loader, "num_windows", None)
    full_batches = getattr(loader, "num_full_batches", None)
    if window_count is not None and full_batches is not None:
        return f"{window_count} windows ({full_batches} full batches)"
    try:
        return f"{len(loader)} batches"
    except TypeError:
        return "validation loader"


def create_pmap_eval_mlm_step(structure):
    def step_fn(params, batch, rng_key):
        state = compute_forward_pass(params, structure, batch, rng_key)
        predictions = state.nodes[structure.task_map["y"]].z_mu
        pred_ids = jnp.argmax(predictions, axis=-1)
        mask = batch["loss_mask"].astype(bool)
        targets = batch["y"]
        per_token_loss = -jnp.sum(
            jax.nn.one_hot(targets, predictions.shape[-1])
            * jnp.log(predictions + 1e-10),
            axis=-1,
        )
        loss_numerator = jnp.sum(per_token_loss * batch["loss_mask"])
        correct = jnp.sum((pred_ids == targets) & mask)
        masked_count = jnp.sum(batch["loss_mask"])
        return (
            jax.lax.psum(loss_numerator, axis_name="devices"),
            jax.lax.psum(correct, axis_name="devices"),
            jax.lax.psum(masked_count, axis_name="devices"),
        )

    return jax.pmap(step_fn, axis_name="devices")


def evaluate_mlm_pmap(
    params, structure, loader, rng_key, n_devices: int
) -> dict[str, float]:
    replicated_params = replicate_params(params, n_devices)
    eval_step = create_pmap_eval_mlm_step(structure)
    total_loss_numerator = 0.0
    total_correct = 0
    total_masked = 0

    for batch_data in loader:
        batch = {key: jnp.array(value) for key, value in batch_data.items()}
        sharded_batch = shard_batch(batch, n_devices)
        rng_key, step_key = jax.random.split(rng_key)
        step_keys = jax.random.split(step_key, n_devices)
        loss_numerator, correct, masked_count = eval_step(
            replicated_params, sharded_batch, step_keys
        )
        batch_masked = int(jax.device_get(masked_count[0]))
        if batch_masked == 0:
            continue
        total_loss_numerator += float(jax.device_get(loss_numerator[0]))
        total_correct += int(jax.device_get(correct[0]))
        total_masked += batch_masked

    if total_masked == 0:
        return {"masked_loss": 0.0, "masked_accuracy": 0.0, "num_masked_tokens": 0}
    return {
        "masked_loss": total_loss_numerator / total_masked,
        "masked_accuracy": total_correct / total_masked,
        "num_masked_tokens": total_masked,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate FabricPC MLM checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache_dir", default="data/babylm_bpe_100m_6k_2gpu")
    parser.add_argument("--mask_prob", type=float, default=0.15)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation_split", default="valid")
    parser.add_argument("--dataset_name", default="")
    parser.add_argument("--source_split", default="train")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--text_field", default="text")
    parser.add_argument("--max_sequences", type=int, default=None)
    parser.add_argument("--max_texts", type=int, default=None)
    parser.add_argument("--max_chars", type=int, default=100_000)
    parser.add_argument("--max_total_chars", type=int, default=100_000_000)
    parser.add_argument("--train_fraction", type=float, default=0.98)
    parser.add_argument("--dev_fraction", type=float, default=0.01)
    parser.add_argument("--test_fraction", type=float, default=0.01)
    parser.add_argument("--split_seed", type=int, default=None)
    parser.add_argument("--micro_batch_size", type=int, default=None)
    parser.add_argument("--overwrite_cache", action="store_true")
    parser.add_argument("--use_pmap", action="store_true")
    parser.add_argument(
        "--shuffle_train_windows", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--shuffle_eval_windows", action=argparse.BooleanOptionalAction, default=False
    )
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = load_checkpoint(args.checkpoint)
    config = checkpoint["config"]
    if args.seq_len:
        config["seq_len"] = args.seq_len

    # Infer vocab_size from embedding weights
    vocab_size = config.get("vocab_size")
    if not vocab_size:
        for node_name in ("embedding", "token_embeddings"):
            embed_node = checkpoint["params"].nodes.get(node_name)
            if embed_node:
                for key in ("embedding", "embeddings"):
                    ew = embed_node.weights.get(key)
                    if ew is not None:
                        vocab_size = int(ew.shape[0])
                        break
            if vocab_size:
                break
    if not vocab_size:
        print("ERROR: Cannot determine vocab_size from checkpoint")
        sys.exit(1)

    print(f"Step: {checkpoint['step']}")
    print(
        f"Vocab: {vocab_size}, embed: {config.get('embed_dim')}, depth: {config.get('depth')}"
    )

    structure = create_encoder_mlm_model(
        depth=config.get("depth", 4),
        embed_dim=config.get("embed_dim", 256),
        num_heads=config.get("num_heads", 4),
        mlp_dim=config.get("mlp_dim", 1024),
        seq_len=config.get("seq_len", 128),
        vocab_size=vocab_size,
        inference=InferenceSGDNormClip(eta_infer=0.05, infer_steps=5, max_norm=5.0),
        use_attention_mask=True,
        dropout_rate=0.0,
        architecture=config.get("architecture", "encoder"),
        ln_eps=config.get("ln_eps"),
    )

    val_loader = build_val_loader(args, config, int(vocab_size), checkpoint["vocab"])

    print(f"Running evaluation on {describe_loader(val_loader)}...")
    rng_key = jax.random.PRNGKey(args.seed)
    n_devices = jax.local_device_count()
    if args.use_pmap:
        if n_devices < 2:
            raise ValueError("--use_pmap requires at least 2 visible JAX devices")
        if args.batch_size % n_devices != 0:
            raise ValueError(
                f"--batch_size {args.batch_size} must be divisible by {n_devices} devices"
            )
        print(f"Using pmap evaluation across {n_devices} devices")
        metrics = evaluate_mlm_pmap(
            checkpoint["params"], structure, val_loader, rng_key, n_devices
        )
    else:
        metrics = evaluate_mlm(checkpoint["params"], structure, val_loader, rng_key)

    print(f"\n{'='*50}")
    print("Validation Results:")
    print(f"  Masked Loss:     {metrics['masked_loss']:.4f}")
    print(f"  Masked Accuracy: {metrics['masked_accuracy']:.2%}")
    print(f"  Masked Tokens:   {metrics['num_masked_tokens']:,}")
    print(f"{'='*50}")

    out = {
        "checkpoint": args.checkpoint,
        "step": checkpoint["step"],
        **metrics,
    }
    out_path = Path(args.checkpoint).with_suffix(".val.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
