"""Standalone encoder-only transformer sentiment demo.

The demo reads separate training and evaluation CSV files with ``text,label``
columns. It uses a simple tokenizer and does not download data. It is
intentionally separate from the existing autoregressive transformer demos.

Scale note: this demo uses FabricPC graph-mode backprop through
``compute_forward_pass``, which calls ``initialize_graph_state`` for each
forward pass. That validates encoder graph semantics, but it is not meant to be
a production BERT-scale training loop. A future fast JAX execution path could
bypass graph-state traversal for larger backprop baselines.

Usage:
    PYTHONPATH=. python examples/bert_sentiment_demo.py \
        --train_csv train.csv --eval_csv eval.csv --num_epochs 5
"""

from jax_setup import set_jax_flags_before_importing_jax

set_jax_flags_before_importing_jax()

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import jax
import jax.numpy as jnp
import optax

from fabricpc.core.inference import InferenceSGDNormClip
from fabricpc.graph_initialization import initialize_params
from fabricpc.models import create_encoder_sentiment_classifier
from fabricpc.training.train_backprop import compute_forward_pass, train_step_backprop

PAD_TOKEN = "[PAD]"
CLS_TOKEN = "[CLS]"
UNK_TOKEN = "[UNK]"


@dataclass(frozen=True)
class SentimentBatcher:
    examples: Sequence[Tuple[str, int]]
    vocab: dict
    seq_len: int
    batch_size: int

    def __len__(self):
        return (len(self.examples) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        for start in range(0, len(self.examples), self.batch_size):
            rows = self.examples[start : start + self.batch_size]
            x = jnp.array(
                [encode(text, self.vocab, self.seq_len) for text, _ in rows],
                dtype=jnp.int32,
            )
            y = jnp.array([label for _, label in rows], dtype=jnp.int32)
            yield {"x": x, "y": y}


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z']+", text.lower())


def load_examples(path: Path) -> list[tuple[str, int]]:
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or not {"text", "label"}.issubset(reader.fieldnames):
            raise ValueError(f"{path} must have text and label columns")

        examples = []
        for line_number, row in enumerate(reader, start=2):
            text, label = row.get("text"), row.get("label")
            if not text or not text.strip() or label not in ("0", "1"):
                raise ValueError(f"{path}:{line_number} needs text and a 0/1 label")
            examples.append((text, int(label)))

    if not examples:
        raise ValueError(f"{path} has no examples")
    return examples


def build_vocab(examples: Iterable[Tuple[str, int]]) -> dict:
    tokens = sorted({token for text, _ in examples for token in tokenize(text)})
    vocab = {PAD_TOKEN: 0, CLS_TOKEN: 1, UNK_TOKEN: 2}
    vocab.update({token: i + len(vocab) for i, token in enumerate(tokens)})
    return vocab


def encode(text: str, vocab: dict, seq_len: int) -> List[int]:
    ids = [vocab[CLS_TOKEN]] + [
        vocab.get(token, vocab[UNK_TOKEN]) for token in tokenize(text)
    ]
    ids = ids[:seq_len]
    return ids + [vocab[PAD_TOKEN]] * (seq_len - len(ids))


def evaluate(params, structure, loader, rng_key):
    correct = 0
    total = 0
    loss_sum = 0.0
    for batch in loader:
        state = compute_forward_pass(params, structure, batch, rng_key)
        probs = state.nodes[structure.task_map["y"]].z_mu
        preds = jnp.argmax(probs, axis=-1)
        correct += int(jnp.sum(preds == batch["y"]))
        total += int(batch["y"].shape[0])
        loss_sum += float(
            -jnp.sum(
                jnp.log(probs[jnp.arange(batch["y"].shape[0]), batch["y"]] + 1e-10)
            )
        )
    return {"accuracy": correct / total, "loss": loss_sum / total}


def build_optimizer(
    *,
    learning_rate: float,
    weight_decay: float,
    grad_clip: float,
    warmup_steps: int,
    total_steps: int,
) -> optax.GradientTransformation:
    """Build the demo optimizer with optional clipping and LR warmup."""
    total_steps = max(1, total_steps)
    if warmup_steps > 0:
        clipped_warmup_steps = min(warmup_steps, max(1, total_steps - 1))
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=learning_rate,
            warmup_steps=clipped_warmup_steps,
            decay_steps=total_steps,
        )
        adamw_learning_rate: float | optax.Schedule = schedule
    else:
        adamw_learning_rate = learning_rate

    transforms = []
    if grad_clip > 0:
        transforms.append(optax.clip_by_global_norm(grad_clip))
    transforms.append(
        optax.adamw(
            learning_rate=adamw_learning_rate,
            weight_decay=weight_decay,
        )
    )
    return optax.chain(*transforms)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Encoder-only transformer sentiment demo"
    )
    parser.add_argument(
        "--mode",
        choices=["backprop"],
        default="backprop",
        help="Training mode. PC support is intentionally deferred for this demo.",
    )
    parser.add_argument("--train_csv", type=Path, required=True)
    parser.add_argument("--eval_csv", type=Path, required=True)
    parser.add_argument("--seq_len", type=int, default=8)
    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--mlp_dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=1.0,
        help="Global gradient clipping norm; set <= 0 to disable clipping.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="Optional optimizer warmup steps before cosine decay.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(args=None):
    args = parse_args() if args is None else args
    if args.train_csv.resolve() == args.eval_csv.resolve():
        raise ValueError("--train_csv and --eval_csv must be different files")

    rng_key = jax.random.PRNGKey(args.seed)
    train_data = load_examples(args.train_csv)
    test_data = load_examples(args.eval_csv)
    vocab = build_vocab(train_data)
    train_loader = SentimentBatcher(train_data, vocab, args.seq_len, args.batch_size)
    test_loader = SentimentBatcher(test_data, vocab, args.seq_len, args.batch_size)

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
    params = initialize_params(structure, rng_key)
    optimizer = build_optimizer(
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        warmup_steps=args.warmup_steps,
        total_steps=args.num_epochs * len(train_loader),
    )
    opt_state = optimizer.init(params)

    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            rng_key, step_key = jax.random.split(rng_key)
            params, opt_state, loss = train_step_backprop(
                params, opt_state, batch, structure, optimizer, step_key
            )
            epoch_loss += float(loss)
        metrics = evaluate(params, structure, test_loader, rng_key)
        print(
            f"epoch={epoch + 1} train_loss={epoch_loss / len(train_loader):.4f} "
            f"test_loss={metrics['loss']:.4f} test_acc={metrics['accuracy']:.2%}"
        )


if __name__ == "__main__":
    main()
