"""
BERT-style encoder: masked-language-model pretraining with backprop.

Builds a bidirectional encoder (``create_encoder_mlm_model``), batches text
with ``MlmTextBatcher`` (BERT 80/10/10 masking, padding masks) and trains it
with the standard ``train_backprop_autoregressive`` loop. Validation reports
masked-token loss and accuracy each epoch.

Without arguments it runs on a small built-in corpus with a word-level
vocabulary (no extra dependencies). Pass text files (one document per line)
to pretrain on your own data; ``--tokenizer bpe`` needs the ``text`` extra
(``pip install -e '.[text]'``).

Usage:
    PYTHONPATH=. python examples/bert_demo.py
    PYTHONPATH=. python examples/bert_demo.py --train_file train.txt \\
        --val_file val.txt --tokenizer bpe --vocab_size 4000 --seq_len 64
"""

from jax_setup import set_jax_flags_before_importing_jax

set_jax_flags_before_importing_jax()

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import jax
import numpy as np
import optax

from fabricpc.core.inference import InferenceSGDNormClip
from fabricpc.graph_initialization import initialize_params
from fabricpc.nodes.encoder_transformer import create_encoder_mlm_model
from fabricpc.training import evaluate_mlm, train_backprop_autoregressive
from fabricpc.utils.data import MLM_SPECIAL_TOKENS, MlmTextBatcher, tokenize_texts_bpe

TINY_TRAIN_TEXTS = [
    "the cat sat on the warm mat by the door",
    "a small dog ran across the green park",
    "the bird sang a song in the tall tree",
    "the dog and the cat slept on the mat",
    "a child read a book under the tall tree",
    "the sun rose over the quiet green park",
    "the cat watched the bird in the tree",
    "a dog barked at the door of the house",
] * 4
TINY_VAL_TEXTS = [
    "the bird sat on the mat by the tree",
    "a cat ran across the park to the door",
    "the dog read a song under the warm sun",
    "a child watched the dog in the green park",
] * 4


def read_texts(path: Path) -> List[str]:
    texts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    texts = [text for text in texts if text]
    if not texts:
        raise ValueError(f"{path} contains no non-empty lines")
    return texts


def _words(text: str) -> List[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def word_tokenize(
    train_texts: Sequence[str], val_texts: Sequence[str]
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """Word-level vocabulary from the training texts; unknown words -> [UNK].

    Words are lowercased runs of ASCII letters, digits and apostrophes; any
    other character is dropped, so use ``--tokenizer bpe`` for non-ASCII text.
    """
    vocab = {token: idx for idx, token in enumerate(MLM_SPECIAL_TOKENS)}
    for word in sorted({w for text in train_texts for w in _words(text)}):
        vocab.setdefault(word, len(vocab))

    def encode(texts: Sequence[str]) -> np.ndarray:
        ids: List[int] = []
        for text in texts:
            ids.extend(vocab.get(word, vocab["[UNK]"]) for word in _words(text))
            ids.append(vocab["[SEP]"])
        return np.asarray(ids, dtype=np.int32)

    return encode(train_texts), encode(val_texts), vocab


def bpe_tokenize(
    train_texts: Sequence[str], val_texts: Sequence[str], vocab_size: int
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    train_ids, tokenizer = tokenize_texts_bpe(train_texts, vocab_size=vocab_size)
    val_ids, _ = tokenize_texts_bpe(
        val_texts, vocab_size=vocab_size, tokenizer=tokenizer
    )
    return train_ids, val_ids, tokenizer.get_vocab()


def make_loader(
    token_ids: np.ndarray, vocab: Dict[str, int], args, *, seed: int, shuffle: bool
) -> MlmTextBatcher:
    loader = MlmTextBatcher(
        token_ids=token_ids,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        pad_token_id=vocab["[PAD]"],
        mask_token_id=vocab["[MASK]"],
        vocab_size=len(vocab),
        special_token_ids={vocab[token] for token in MLM_SPECIAL_TOKENS},
        cls_token_id=vocab["[CLS]"],
        seed=seed,
        mask_prob=args.mask_prob,
        shuffle=shuffle,
    )
    if len(loader) == 0:
        raise ValueError(
            f"{len(token_ids)} tokens give no full batch of {args.batch_size} "
            f"windows at seq_len={args.seq_len}; use more text or smaller sizes"
        )
    return loader


def count_training_steps(num_epochs: float, batches_per_epoch: int) -> int:
    """Optimizer steps train_backprop_autoregressive runs for ``num_epochs``.

    The loop truncates a fractional final epoch to
    ``round(fraction * batches_per_epoch)`` batches and fails if that is 0.
    """
    if num_epochs <= 0:
        raise ValueError("--num_epochs must be positive")
    whole_epochs = math.floor(num_epochs)
    fraction = num_epochs - whole_epochs
    final_batches = round(fraction * batches_per_epoch)
    if fraction > 0 and final_batches == 0:
        raise ValueError(
            f"--num_epochs {num_epochs} leaves no batch in its final partial "
            f"epoch ({batches_per_epoch} batches per epoch); use a whole number "
            "or a larger fraction"
        )
    return whole_epochs * batches_per_epoch + final_batches


def build_optimizer(args, total_steps: int) -> optax.GradientTransformation:
    """AdamW with linear warmup, cosine decay and global-norm clipping."""
    if not 0.0 <= args.warmup_fraction < 1.0:
        raise ValueError("warmup_fraction must be in [0, 1)")
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=args.lr,
        warmup_steps=int(args.warmup_fraction * total_steps),
        decay_steps=total_steps,  # optax counts warmup inside decay_steps
    )
    adamw = optax.adamw(schedule, weight_decay=args.weight_decay)
    if args.grad_clip <= 0:
        return adamw
    return optax.chain(optax.clip_by_global_norm(args.grad_clip), adamw)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BERT-style MLM pretraining")
    parser.add_argument("--train_file", type=Path, default=None)
    parser.add_argument("--val_file", type=Path, default=None)
    parser.add_argument(
        "--tokenizer",
        choices=["word", "bpe"],
        default="word",
        help="word: ASCII words, no extra deps; bpe: needs the text extra",
    )
    parser.add_argument("--vocab_size", type=int, default=4000, help="BPE only")
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--mlp_dim", type=int, default=64)
    parser.add_argument("--dropout_rate", type=float, default=0.0)
    parser.add_argument("--mask_prob", type=float, default=0.15)
    parser.add_argument("--num_epochs", type=float, default=10)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument(
        "--warmup_fraction", type=float, default=0.1, help="of total steps"
    )
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if (args.train_file is None) != (args.val_file is None):
        parser.error("--train_file and --val_file must be given together")
    return args


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, float]:
    """Pretrain and return the final validation metrics."""
    args = parse_args(argv)
    if args.train_file is None:
        train_texts, val_texts = TINY_TRAIN_TEXTS, TINY_VAL_TEXTS
    else:
        train_texts, val_texts = read_texts(args.train_file), read_texts(args.val_file)

    if args.tokenizer == "bpe":
        train_ids, val_ids, vocab = bpe_tokenize(
            train_texts, val_texts, args.vocab_size
        )
    else:
        train_ids, val_ids, vocab = word_tokenize(train_texts, val_texts)
    train_loader = make_loader(train_ids, vocab, args, seed=args.seed, shuffle=True)
    val_loader = make_loader(val_ids, vocab, args, seed=args.seed + 1, shuffle=False)
    print(
        f"vocab={len(vocab)} train_tokens={len(train_ids)} "
        f"val_tokens={len(val_ids)} batches_per_epoch={len(train_loader)}"
    )

    structure = create_encoder_mlm_model(
        depth=args.depth,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        mlp_dim=args.mlp_dim,
        seq_len=args.seq_len,
        vocab_size=len(vocab),
        dropout_rate=args.dropout_rate,
        # Unused by backprop training; required by the graph builder.
        inference=InferenceSGDNormClip(eta_infer=0.05, infer_steps=5, max_norm=5.0),
    )
    init_key, train_key = jax.random.split(jax.random.PRNGKey(args.seed))
    params = initialize_params(structure, init_key)
    total_steps = count_training_steps(args.num_epochs, len(train_loader))

    def validate(epoch_idx, epoch_params, structure, config, key, **_):
        metrics = evaluate_mlm(epoch_params, structure, val_loader, key)
        print(
            f"  val masked_loss={metrics['masked_loss']:.4f} "
            f"masked_acc={metrics['masked_accuracy']:.2%} "
            f"tokens={metrics['num_masked_tokens']}"
        )
        return metrics

    _, _, epoch_metrics = train_backprop_autoregressive(
        params,
        structure,
        train_loader,
        build_optimizer(args, total_steps),
        {"num_epochs": args.num_epochs},
        train_key,
        epoch_callback=validate,
    )
    return epoch_metrics[-1]


if __name__ == "__main__":
    main()
