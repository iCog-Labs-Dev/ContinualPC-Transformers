"""Pretrain a small BERT-style encoder on a HuggingFace/BabyLM text corpus.

Example:
    PYTHONPATH=. python examples/bert_babylm_pretrain.py \
      --dataset_name <babylm-hf-dataset> --split train --text_field text \
      --seq_len 128 --batch_size 64 --max_steps 1000
"""

from jax_setup import set_jax_flags_before_importing_jax

set_jax_flags_before_importing_jax()

import argparse

import jax
import optax

from fabricpc.core.inference import InferenceSGDNormClip
from fabricpc.graph_initialization import initialize_params
from fabricpc.models import create_encoder_mlm_model
from fabricpc.training.train_backprop import train_backprop_mlm
from fabricpc.utils.data import BabyLmMlmLoader


def build_optimizer(
    *,
    learning_rate: float,
    weight_decay: float,
    grad_clip: float,
    warmup_steps: int,
    max_steps: int,
) -> optax.GradientTransformation:
    if warmup_steps > 0:
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=learning_rate,
            warmup_steps=min(warmup_steps, max(1, max_steps - 1)),
            decay_steps=max(1, max_steps),
        )
        adamw_lr: float | optax.Schedule = schedule
    else:
        adamw_lr = learning_rate

    transforms = []
    if grad_clip > 0:
        transforms.append(optax.clip_by_global_norm(grad_clip))
    transforms.append(optax.adamw(adamw_lr, weight_decay=weight_decay))
    return optax.chain(*transforms)


def parse_args():
    parser = argparse.ArgumentParser(description="BabyLM/HF BPE MLM pretraining")
    parser.add_argument("--dataset_name", required=True, help="HuggingFace dataset id")
    parser.add_argument("--split", default="train")
    parser.add_argument("--validation_split", default=None)
    parser.add_argument("--text_field", default="text")
    parser.add_argument("--revision", default=None, help="Optional pinned HF revision")
    parser.add_argument("--cache_dir", default="data/babylm_bpe")
    parser.add_argument("--checkpoint", default="data/checkpoints/babylm_mlm.json")
    parser.add_argument("--resume_from", default=None)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--vocab_size", type=int, default=16000)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--mlp_dim", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--max_sequences", type=int, default=None)
    parser.add_argument("--max_texts", type=int, default=None)
    parser.add_argument("--max_chars", type=int, default=100_000)
    parser.add_argument("--max_total_chars", type=int, default=100_000_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--mask_prob", type=float, default=0.15)
    parser.add_argument("--checkpoint_interval", type=int, default=1000)
    parser.add_argument("--validation_interval", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_pmap", action="store_true")
    parser.add_argument("--pmap_single_device", action="store_true")
    parser.add_argument("--overwrite_cache", action="store_true")
    return parser.parse_args()


def main(args=None):
    args = parse_args() if args is None else args
    if args.checkpoint_interval <= 0:
        raise ValueError("--checkpoint_interval must be positive")
    if args.seq_len <= 0 or args.batch_size <= 0 or args.vocab_size <= 0:
        raise ValueError("--seq_len, --batch_size, and --vocab_size must be positive")
    if not 0.0 < args.mask_prob <= 1.0:
        raise ValueError("--mask_prob must be in (0, 1]")
    rng_key = jax.random.PRNGKey(args.seed)

    train_loader = BabyLmMlmLoader(
        dataset_name=args.dataset_name,
        split=args.split,
        text_field=args.text_field,
        cache_dir=args.cache_dir,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        vocab_size=args.vocab_size,
        seed=args.seed,
        mask_prob=args.mask_prob,
        max_sequences=args.max_sequences,
        revision=args.revision,
        max_texts=args.max_texts,
        max_chars=args.max_chars,
        max_total_chars=args.max_total_chars,
        overwrite_cache=args.overwrite_cache,
    )
    val_loader = None
    if args.validation_split:
        val_loader = BabyLmMlmLoader(
            dataset_name=args.dataset_name,
            split=args.validation_split,
            text_field=args.text_field,
            cache_dir=args.cache_dir,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            vocab_size=args.vocab_size,
            seed=args.seed + 1,
            mask_prob=args.mask_prob,
            max_sequences=args.max_sequences,
            revision=args.revision,
            max_texts=args.max_texts,
            max_chars=args.max_chars,
            max_total_chars=args.max_total_chars,
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
        max_steps=args.max_steps,
    )
    params, state = train_backprop_mlm(
        params,
        structure,
        train_loader,
        optimizer,
        {
            "max_steps": args.max_steps,
            "checkpoint_path": args.checkpoint,
            "checkpoint_interval": args.checkpoint_interval,
            "validation_interval": args.validation_interval,
            "resume_from": args.resume_from,
            "use_pmap": args.use_pmap,
            "pmap_single_device": args.pmap_single_device,
            "model": {
                "depth": args.depth,
                "embed_dim": args.embed_dim,
                "num_heads": args.num_heads,
                "mlp_dim": args.mlp_dim,
                "seq_len": args.seq_len,
                "vocab_size": train_loader.vocab_size,
            },
        },
        rng_key,
        val_loader=val_loader,
    )
    print(f"done step={state['step']} checkpoint={args.checkpoint}")


if __name__ == "__main__":
    main()
