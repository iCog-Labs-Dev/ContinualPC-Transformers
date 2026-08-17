"""Tiny masked-language-model pretraining demo for the BERT-style encoder.

This intentionally uses a built-in corpus and demo-local batcher so the first
MLM path stays CPU-runnable and independent of external downloads.

Usage:
    PYTHONPATH=. python examples/bert_mlm_pretrain_demo.py --num_epochs 1
"""

from jax_setup import set_jax_flags_before_importing_jax

set_jax_flags_before_importing_jax()

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax

from fabricpc.core.types import GraphParams
from fabricpc.core.inference import InferenceSGDNormClip
from fabricpc.graph_initialization import initialize_params
from fabricpc.models import create_encoder_mlm_model
from fabricpc.training.train_backprop import evaluate_mlm, train_step_backprop_mlm

PAD_TOKEN = "[PAD]"
CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"
MASK_TOKEN = "[MASK]"
UNK_TOKEN = "[UNK]"
SPECIAL_TOKENS = [PAD_TOKEN, CLS_TOKEN, SEP_TOKEN, MASK_TOKEN, UNK_TOKEN]

CORPUS = [
    "a delightful and warm movie with excellent acting",
    "happy charming wonderful film with a lovely story",
    "the plot was bright enjoyable funny sweet and satisfying",
    "a moving performance with heart and careful pacing",
    "a terrible and boring movie with awful acting",
    "sad lifeless disappointing film with a dull story",
    "the plot was flat annoying tedious cold and unsatisfying",
    "a bad performance without heart or charm",
]


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z']+", text.lower())


def build_vocab(texts: Iterable[str]) -> dict:
    tokens = sorted({token for text in texts for token in tokenize(text)})
    vocab = {token: i for i, token in enumerate(SPECIAL_TOKENS)}
    vocab.update({token: i + len(vocab) for i, token in enumerate(tokens)})
    return vocab


def encode(text: str, vocab: dict, seq_len: int) -> List[int]:
    ids = [vocab[CLS_TOKEN]]
    ids.extend(vocab.get(token, vocab[UNK_TOKEN]) for token in tokenize(text))
    ids.append(vocab[SEP_TOKEN])
    ids = ids[:seq_len]
    return ids + [vocab[PAD_TOKEN]] * (seq_len - len(ids))


@dataclass(frozen=True)
class MlmBatcher:
    texts: Sequence[str]
    vocab: dict
    seq_len: int
    batch_size: int
    seed: int = 0
    mask_prob: float = 0.15

    def __len__(self):
        return (len(self.texts) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        mask_id = self.vocab[MASK_TOKEN]
        special_ids = {self.vocab[token] for token in SPECIAL_TOKENS}
        vocab_ids = list(range(len(self.vocab)))
        rng = jax.random.PRNGKey(self.seed)

        for start in range(0, len(self.texts), self.batch_size):
            rows = self.texts[start : start + self.batch_size]
            original = jnp.array(
                [encode(text, self.vocab, self.seq_len) for text in rows],
                dtype=jnp.int32,
            )
            inputs = original
            loss_mask = jnp.zeros(original.shape, dtype=bool)

            for row_idx in range(original.shape[0]):
                candidate_positions = [
                    col
                    for col, token_id in enumerate(list(original[row_idx]))
                    if int(token_id) not in special_ids
                ]
                if not candidate_positions:
                    continue

                rng, choose_key = jax.random.split(rng)
                selected = []
                for pos in candidate_positions:
                    rng, pos_key = jax.random.split(rng)
                    if bool(jax.random.bernoulli(pos_key, self.mask_prob)):
                        selected.append(pos)
                if not selected:
                    selected = [
                        candidate_positions[
                            int(
                                jax.random.randint(
                                    choose_key, (), 0, len(candidate_positions)
                                )
                            )
                        ]
                    ]

                for pos in selected:
                    rng, policy_key, random_key = jax.random.split(rng, 3)
                    policy = float(jax.random.uniform(policy_key))
                    if policy < 0.8:
                        replacement = mask_id
                    elif policy < 0.9:
                        replacement = int(
                            jax.random.choice(random_key, jnp.array(vocab_ids))
                        )
                    else:
                        replacement = int(original[row_idx, pos])
                    inputs = inputs.at[row_idx, pos].set(replacement)
                    loss_mask = loss_mask.at[row_idx, pos].set(True)

            yield {"x": inputs, "y": original, "loss_mask": loss_mask}


def save_checkpoint(
    path: str,
    params: GraphParams,
    vocab: dict[str, int],
    config: dict[str, Any],
) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tensor_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".npz")

    tensors = {}
    param_schema = {}
    for node_name, node_params in params.nodes.items():
        node_schema = {"weights": {}, "biases": {}}
        for collection_name in ("weights", "biases"):
            collection = getattr(node_params, collection_name)
            for param_name, value in collection.items():
                tensor_key = f"{node_name}/{collection_name}/{param_name}"
                tensors[tensor_key] = np.asarray(value)
                node_schema[collection_name][param_name] = tensor_key
        param_schema[node_name] = node_schema

    np.savez(tensor_path, **tensors)
    metadata = {
        "format": "fabricpc-bert-demo-checkpoint",
        "version": 1,
        "tensor_file": tensor_path.name,
        "params": param_schema,
        "vocab": vocab,
        "config": config,
    }
    checkpoint_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(description="Tiny BERT-style MLM pretraining")
    parser.add_argument("--seq_len", type=int, default=10)
    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--mlp_dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--checkpoint",
        default="data/bert_mlm_demo_params.json",
        help="Path to write the demo checkpoint.",
    )
    return parser.parse_args()


def main(args=None):
    args = parse_args() if args is None else args
    rng_key = jax.random.PRNGKey(args.seed)
    vocab = build_vocab(CORPUS)
    structure = create_encoder_mlm_model(
        depth=args.depth,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        mlp_dim=args.mlp_dim,
        seq_len=args.seq_len,
        vocab_size=len(vocab),
        inference=InferenceSGDNormClip(eta_infer=0.05, infer_steps=5, max_norm=5.0),
    )
    params = initialize_params(structure, rng_key)
    optimizer = optax.adamw(args.lr, weight_decay=1e-4)
    opt_state = optimizer.init(params)

    for epoch in range(args.num_epochs):
        train_loader = MlmBatcher(
            CORPUS, vocab, args.seq_len, args.batch_size, args.seed + epoch
        )
        epoch_loss = 0.0
        batches = 0
        for batch in train_loader:
            rng_key, step_key = jax.random.split(rng_key)
            params, opt_state, loss = train_step_backprop_mlm(
                params, opt_state, batch, structure, optimizer, step_key
            )
            epoch_loss += float(loss)
            batches += 1

        eval_loader = MlmBatcher(
            CORPUS, vocab, args.seq_len, args.batch_size, args.seed
        )
        metrics = evaluate_mlm(params, structure, eval_loader, rng_key)
        print(
            f"epoch={epoch + 1} train_loss={epoch_loss / batches:.4f} "
            f"masked_loss={metrics['masked_loss']:.4f} "
            f"masked_acc={metrics['masked_accuracy']:.2%}"
        )

    save_checkpoint(
        args.checkpoint,
        params,
        vocab,
        {
            "depth": args.depth,
            "embed_dim": args.embed_dim,
            "num_heads": args.num_heads,
            "mlp_dim": args.mlp_dim,
            "seq_len": args.seq_len,
        },
    )
    print(f"checkpoint={args.checkpoint}")


if __name__ == "__main__":
    main()
