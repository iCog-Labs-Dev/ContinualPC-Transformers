"""Fine-tune the BERT-style encoder with optional MLM checkpoint transfer.

Usage:
    PYTHONPATH=. python examples/bert_downstream_finetune_demo.py --num_epochs 1
"""

from jax_setup import set_jax_flags_before_importing_jax

set_jax_flags_before_importing_jax()

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from fabricpc.core.types import GraphParams, NodeParams
from fabricpc.core.inference import InferenceSGDNormClip
from fabricpc.graph_initialization import initialize_params
from fabricpc.models import copy_encoder_params, create_encoder_sentiment_classifier
from fabricpc.training.train_backprop import train_step_backprop


def load_sentiment_demo_helpers():
    demo_path = Path(__file__).resolve().parent / "bert_sentiment_demo.py"
    spec = importlib.util.spec_from_file_location(
        "bert_sentiment_demo_helpers", demo_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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
        "vocab": metadata["vocab"],
        "config": metadata["config"],
    }


def train_classifier(
    params, structure, train_loader, test_loader, args, rng_key, evaluate_fn
):
    optimizer = optax.adamw(args.lr, weight_decay=1e-4)
    opt_state = optimizer.init(params)
    for _ in range(args.num_epochs):
        for batch in train_loader:
            rng_key, step_key = jax.random.split(rng_key)
            params, opt_state, _ = train_step_backprop(
                params, opt_state, batch, structure, optimizer, step_key
            )
    return params, evaluate_fn(params, structure, test_loader, rng_key)


def parse_args():
    parser = argparse.ArgumentParser(description="Tiny downstream encoder fine-tuning")
    parser.add_argument("--seq_len", type=int, default=10)
    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--mlp_dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--checkpoint",
        default="data/bert_mlm_demo_params.json",
        help="Optional MLM checkpoint produced by bert_mlm_pretrain_demo.py.",
    )
    return parser.parse_args()


def main(args=None):
    args = parse_args() if args is None else args
    sentiment_demo = load_sentiment_demo_helpers()
    rng_key = jax.random.PRNGKey(args.seed)
    checkpoint = load_checkpoint(args.checkpoint)

    if checkpoint is not None:
        vocab = checkpoint["vocab"]
        checkpoint_config = checkpoint.get("config", {})
        args.seq_len = checkpoint_config.get("seq_len", args.seq_len)
        args.embed_dim = checkpoint_config.get("embed_dim", args.embed_dim)
        args.num_heads = checkpoint_config.get("num_heads", args.num_heads)
        args.mlp_dim = checkpoint_config.get("mlp_dim", args.mlp_dim)
        args.depth = checkpoint_config.get("depth", args.depth)
    else:
        vocab = sentiment_demo.build_vocab(sentiment_demo.DATASET)

    positives = [row for row in sentiment_demo.DATASET if row[1] == 1]
    negatives = [row for row in sentiment_demo.DATASET if row[1] == 0]
    train_data = positives[:5] + negatives[:5]
    test_data = positives[5:] + negatives[5:]
    train_loader = sentiment_demo.SentimentBatcher(
        train_data, vocab, args.seq_len, args.batch_size
    )
    test_loader = sentiment_demo.SentimentBatcher(
        test_data, vocab, args.seq_len, args.batch_size
    )

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

    rng_key, random_key, transfer_key = jax.random.split(rng_key, 3)
    random_params = initialize_params(structure, random_key)
    random_params, random_metrics = train_classifier(
        random_params,
        structure,
        train_loader,
        test_loader,
        args,
        random_key,
        sentiment_demo.evaluate,
    )
    print(
        f"random_init test_loss={random_metrics['loss']:.4f} "
        f"test_acc={random_metrics['accuracy']:.2%}"
    )

    if checkpoint is None:
        print(f"transferred skipped: checkpoint not found at {args.checkpoint}")
        return

    transfer_params = initialize_params(structure, transfer_key)
    transfer_params = copy_encoder_params(checkpoint["params"], transfer_params)
    transfer_params, transfer_metrics = train_classifier(
        transfer_params,
        structure,
        train_loader,
        test_loader,
        args,
        transfer_key,
        sentiment_demo.evaluate,
    )
    print(
        f"transferred test_loss={transfer_metrics['loss']:.4f} "
        f"test_acc={transfer_metrics['accuracy']:.2%}"
    )


if __name__ == "__main__":
    main()
