#!/usr/bin/env python3
"""BLiMP evaluation for FabricPC MLM checkpoints.

Computes pseudo-log-likelihood on minimal pairs and reports accuracy
per phenomenon and overall. Works directly with JAX/Flax checkpoints.

Usage:
    PYTHONPATH=. python examples/evaluate_blimp.py \\
        --checkpoint data/checkpoints/babylm_100m_50k.json
"""

# ruff: noqa: E402

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from jax_setup import set_jax_flags_before_importing_jax

set_jax_flags_before_importing_jax()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from datasets import (
    concatenate_datasets,
    get_dataset_config_names,
    load_dataset,
)  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

from fabricpc.core.inference import InferenceSGDNormClip  # noqa: E402
from fabricpc.core.types import GraphParams, NodeParams  # noqa: E402
from fabricpc.nodes.encoder_transformer import create_encoder_mlm_model  # noqa: E402
from fabricpc.training.train_backprop import compute_forward_pass  # noqa: E402


def load_checkpoint(path: str) -> dict:
    ckpt_path = Path(path)
    meta = json.loads(ckpt_path.read_text())
    tensor_file = meta.get("tensor_file")
    tensors = np.load(ckpt_path.with_name(tensor_file), allow_pickle=False)

    nodes = {}
    for node_name, schema in meta["params"].items():
        weights = {
            k: jnp.array(tensors[v]) for k, v in schema.get("weights", {}).items()
        }
        biases = {k: jnp.array(tensors[v]) for k, v in schema.get("biases", {}).items()}
        nodes[node_name] = NodeParams(weights=weights, biases=biases)

    return {
        "params": GraphParams(nodes=nodes),
        "config": meta.get("config", {}),
        "vocab": meta.get("vocab", {}),
        "tokenizer_json": meta.get("tokenizer_json"),
    }


def load_tokenizer(checkpoint: dict) -> Tokenizer:
    tok_json = checkpoint.get("tokenizer_json")
    if tok_json:
        return Tokenizer.from_str(tok_json)
    raise ValueError("No tokenizer in checkpoint")


def build_structure(config: dict, vocab_size: int):
    return create_encoder_mlm_model(
        depth=config.get("depth", 4),
        embed_dim=config.get("embed_dim", 256),
        num_heads=config.get("num_heads", 4),
        mlp_dim=config.get("mlp_dim", 1024),
        seq_len=config.get("seq_len", 128),
        vocab_size=vocab_size,
        inference=InferenceSGDNormClip(eta_infer=0.05, infer_steps=5, max_norm=5.0),
        use_attention_mask=True,
        dropout_rate=0.0,
    )


def tokenize_to_ids(tokenizer: Tokenizer, text: str, max_len: int) -> list[int]:
    enc = tokenizer.encode(text)
    ids = enc.ids[:max_len]
    return ids


def get_position_capacity(params: GraphParams) -> int | None:
    position_node = params.nodes.get("position_embeddings")
    if position_node is None:
        return None
    position_embeddings = position_node.weights.get("position_embeddings")
    if position_embeddings is None:
        return None
    return int(position_embeddings.shape[0])


def get_required_token_id(tokenizer: Tokenizer, token: str) -> int:
    token_id = tokenizer.token_to_id(token)
    if token_id is None:
        raise ValueError(f"Tokenizer missing required token {token!r}")
    return int(token_id)


def sentence_log_prob(
    params,
    structure,
    tokenizer,
    text: str,
    rng_key,
    *,
    mask_id: int,
    pad_id: int,
    cls_id: int,
    sep_id: int,
) -> float | None:
    """Compute pseudo-log-likelihood: sum of log p(token_i | rest) for all positions."""
    seq_len = structure.config.get("seq_len", 128)
    content_ids = tokenize_to_ids(tokenizer, text, seq_len)
    if len(content_ids) < 1:
        return float("-inf")
    if len(content_ids) + 2 > seq_len:
        return None

    total_log_prob = 0.0
    ids = [cls_id, *content_ids, sep_id]

    for mask_pos in range(1, 1 + len(content_ids)):
        input_ids = list(ids)
        target_id = input_ids[mask_pos]
        input_ids[mask_pos] = mask_id

        padded = input_ids + [pad_id] * (seq_len - len(input_ids))
        attn_mask = [1.0] * len(input_ids) + [0.0] * (seq_len - len(input_ids))
        loss_mask = [0.0] * seq_len
        loss_mask[mask_pos] = 1.0

        batch = {
            "x": jnp.array([padded], dtype=jnp.int32),
            "y": jnp.array([target_id], dtype=jnp.int32),
            "attention_mask": jnp.array([attn_mask], dtype=jnp.float32),
            "loss_mask": jnp.array([loss_mask], dtype=jnp.float32),
        }

        rng_key, step_key = jax.random.split(rng_key)
        state = compute_forward_pass(params, structure, batch, step_key)
        probs = state.nodes[structure.task_map["y"]].z_mu[0]  # (seq_len, vocab)
        total_log_prob += float(jnp.log(probs[mask_pos, target_id] + 1e-9))

    return total_log_prob


def load_blimp(
    phenomena: str | None,
    limit: int | None,
    limit_per_phenomenon: int | None,
):
    selected = (
        None
        if phenomena is None
        else {p.strip() for p in phenomena.split(",") if p.strip()}
    )
    configs = get_dataset_config_names("nyu-mll/blimp")
    unknown = sorted(selected - set(configs)) if selected is not None else []
    if unknown:
        raise ValueError(
            f"Unknown BLiMP phenomena/configs: {', '.join(unknown)}. "
            f"Examples: {', '.join(configs[:10])}"
        )

    config_names = (
        configs if selected is None else [name for name in configs if name in selected]
    )
    datasets = []
    remaining = limit
    for config_name in config_names:
        split_dataset = load_dataset("nyu-mll/blimp", config_name, split="train")
        if limit_per_phenomenon is not None:
            take = min(limit_per_phenomenon, len(split_dataset))
            split_dataset = split_dataset.select(range(take))
        elif remaining is not None:
            take = min(remaining, len(split_dataset))
            if take <= 0:
                break
            split_dataset = split_dataset.select(range(take))
            remaining -= take
        datasets.append(
            split_dataset.map(lambda row, name=config_name: {"phenomenon": name})
        )
    if not datasets:
        raise ValueError("No BLiMP phenomena selected")
    return concatenate_datasets(datasets)


def main():
    parser = argparse.ArgumentParser(description="BLiMP evaluation for FabricPC MLM")
    parser.add_argument(
        "--checkpoint", required=True, help="Path to transfer checkpoint JSON"
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=None,
        help="Override seq_len (default: from checkpoint)",
    )
    parser.add_argument(
        "--phenomena",
        default=None,
        help="Comma-separated phenomenon subset to evaluate",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit total number of BLiMP pairs for fast smoke tests",
    )
    parser.add_argument(
        "--limit_per_phenomenon",
        type=int,
        default=None,
        help="Limit pairs per selected BLiMP phenomenon/config",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write results JSON (default: checkpoint path with .blimp.json)",
    )
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = load_checkpoint(args.checkpoint)
    config = checkpoint["config"]
    if args.seq_len:
        config["seq_len"] = args.seq_len

    vocab = checkpoint["vocab"]
    vocab_size = max(vocab.values()) + 1
    print(
        f"Vocab size: {vocab_size}, embed_dim: {config.get('embed_dim')}, depth: {config.get('depth')}"
    )

    tokenizer = load_tokenizer(checkpoint)
    mask_id = get_required_token_id(tokenizer, "[MASK]")
    pad_id = get_required_token_id(tokenizer, "[PAD]")
    cls_id = get_required_token_id(tokenizer, "[CLS]")
    sep_id = get_required_token_id(tokenizer, "[SEP]")
    position_capacity = get_position_capacity(checkpoint["params"])
    if (
        args.seq_len is not None
        and position_capacity is not None
        and args.seq_len > position_capacity
    ):
        raise ValueError(
            f"--seq_len {args.seq_len} exceeds checkpoint position capacity {position_capacity}"
        )
    structure = build_structure(config, vocab_size)
    params = checkpoint["params"]

    print("Loading BLiMP dataset...")
    blimp = load_blimp(args.phenomena, args.limit, args.limit_per_phenomenon)
    print(f"Loaded {len(blimp)} pairs across {len(set(blimp['phenomenon']))} phenomena")

    rng_key = jax.random.PRNGKey(0)
    total_correct = 0
    total_pairs = 0
    skipped_pairs = 0
    by_phenomenon = defaultdict(lambda: {"correct": 0, "total": 0})

    for i, example in enumerate(blimp):
        good_score = sentence_log_prob(
            params,
            structure,
            tokenizer,
            example["sentence_good"],
            rng_key,
            mask_id=mask_id,
            pad_id=pad_id,
            cls_id=cls_id,
            sep_id=sep_id,
        )
        bad_score = sentence_log_prob(
            params,
            structure,
            tokenizer,
            example["sentence_bad"],
            rng_key,
            mask_id=mask_id,
            pad_id=pad_id,
            cls_id=cls_id,
            sep_id=sep_id,
        )
        if good_score is None or bad_score is None:
            skipped_pairs += 1
            continue

        correct = good_score > bad_score
        phenomenon = example["phenomenon"]
        by_phenomenon[phenomenon]["total"] += 1
        if correct:
            by_phenomenon[phenomenon]["correct"] += 1
            total_correct += 1
        total_pairs += 1

        if (i + 1) % 500 == 0:
            print(
                f"  [{i+1}/{len(blimp)}] running accuracy: {total_correct/total_pairs:.1%}"
            )

    if total_pairs == 0:
        raise ValueError("No BLiMP examples evaluated")

    print(f"\n{'='*60}")
    print(
        f"Overall BLiMP Accuracy: {total_correct}/{total_pairs} = {total_correct/total_pairs:.1%}"
    )
    if skipped_pairs:
        print(f"Skipped overlength pairs: {skipped_pairs}")
    print(f"{'='*60}\n")

    print(f"{'Phenomenon':<45} {'Acc':>6} {'N':>5}")
    print("-" * 60)
    for phen in sorted(by_phenomenon.keys()):
        d = by_phenomenon[phen]
        acc = d["correct"] / d["total"] if d["total"] > 0 else 0
        print(f"{phen:<45} {acc:>5.1%} {d['total']:>5}")

    results = {
        "overall_accuracy": total_correct / total_pairs,
        "total_pairs": total_pairs,
        "skipped_pairs": skipped_pairs,
        "correct": total_correct,
        "by_phenomenon": {
            phen: {
                "accuracy": d["correct"] / d["total"],
                "correct": d["correct"],
                "total": d["total"],
            }
            for phen, d in by_phenomenon.items()
        },
    }
    out_path = (
        Path(args.output)
        if args.output
        else Path(args.checkpoint).with_suffix(".blimp.json")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
