"""Unified encoder-only BERT-style demo.

The demo mirrors ``transformer_v2_demo.py``: one entrypoint selects the
experiment stage and training mode with CLI flags, while reusable pieces live in
``fabricpc`` modules.
"""

from jax_setup import set_jax_flags_before_importing_jax

set_jax_flags_before_importing_jax()

import argparse
from typing import Sequence


PC_UNSUPPORTED_MESSAGE = (
    "PC encoder pretraining/fine-tuning is not implemented yet; use --mode backprop."
)


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
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--vocab_size", type=int, default=16000)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--mlp_dim", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--mask_prob", type=float, default=0.15)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--rolling_loss_window", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compare_scratch", action="store_true")
    parser.add_argument("--overwrite_cache", action="store_true")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _reject_unsupported_pc(args: argparse.Namespace) -> None:
    if args.mode == "pc":
        raise NotImplementedError(PC_UNSUPPORTED_MESSAGE)


def run_pretrain(args: argparse.Namespace) -> None:
    _reject_unsupported_pc(args)
    raise NotImplementedError("BERT pretraining wiring is implemented in a later step.")


def run_finetune(args: argparse.Namespace) -> None:
    _reject_unsupported_pc(args)
    raise NotImplementedError("BERT fine-tuning wiring is implemented in a later step.")


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
