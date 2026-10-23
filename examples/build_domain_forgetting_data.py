#!/usr/bin/env python3
"""Build the pinned domain caches for the sequential MLM forgetting benchmark.

Downloads the pinned Hugging Face splits, normalizes, deduplicates and
tokenizes them with the checkpoint's tokenizer, and writes
``<out_dir>/<domain>/{train.npy, valid_windows.npz, manifest.json}`` plus
``<out_dir>/dataset_manifest.json``. Needs network access and ``datasets``.

Usage:
    PYTHONPATH=. python examples/build_domain_forgetting_data.py \\
        --checkpoint checkpoints/bert_mini_128.json \\
        --out_dir data/domain_forgetting --domains news reviews wiki \\
        --tokens_per_domain 5000000
"""

# ruff: noqa: E402

import argparse
import json
from pathlib import Path
from typing import Sequence

from jax_setup import set_jax_flags_before_importing_jax

set_jax_flags_before_importing_jax()

from examples.bert_forgetting_benchmark import _reject_symlink_components
from fabricpc.experiments.domain_forgetting.data import (
    DOMAIN_REGISTRY,
    build_domain_caches,
    load_hf_domain_rows,
    validate_domain_names,
)
from fabricpc.experiments.domain_forgetting.validation import (
    DEFAULT_NUM_VALID_WINDOWS,
)


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out_dir", default="data/domain_forgetting")
    parser.add_argument(
        "--domains", nargs="+", default=list(DOMAIN_REGISTRY), metavar="DOMAIN"
    )
    parser.add_argument("--tokens_per_domain", type=int, default=5_000_000)
    parser.add_argument("--data_seed", type=int, default=0)
    parser.add_argument(
        "--num_valid_windows", type=int, default=DEFAULT_NUM_VALID_WINDOWS
    )
    parser.add_argument("--mask_prob", type=float, default=0.15)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def read_checkpoint_tokenizer(path: str | Path) -> tuple[str, int]:
    """Return the checkpoint's ``tokenizer_json`` and ``seq_len``."""
    metadata = json.loads(Path(path).read_text())
    tokenizer_json = metadata.get("tokenizer_json")
    if not tokenizer_json:
        raise ValueError(f"{path}: checkpoint has no tokenizer_json")
    return tokenizer_json, int(metadata["config"]["seq_len"])


def main(argv: Sequence[str] | None = None) -> None:
    """Download, prepare and cache every requested domain."""
    args = build_parser().parse_args(argv)
    domains = validate_domain_names(args.domains)
    out_dir = Path(args.out_dir)
    _reject_symlink_components(out_dir)
    if (out_dir / "dataset_manifest.json").exists() and not args.overwrite:
        raise FileExistsError(
            f"{out_dir} already holds a domain cache; pass --overwrite"
        )
    tokenizer_json, seq_len = read_checkpoint_tokenizer(args.checkpoint)
    rows = {}
    for name in domains:
        print(f"loading {name}: {DOMAIN_REGISTRY[name].dataset_id}")
        rows[name] = load_hf_domain_rows(DOMAIN_REGISTRY[name])
    manifest = build_domain_caches(
        rows,
        tokenizer_json=tokenizer_json,
        out_dir=out_dir,
        tokens_per_domain=args.tokens_per_domain,
        seq_len=seq_len,
        data_seed=args.data_seed,
        num_valid_windows=args.num_valid_windows,
        mask_prob=args.mask_prob,
    )
    for name in domains:
        entry = manifest["per_domain"][name]
        stats = entry["train"]["token_stats"]
        print(
            f"{name}: train_tokens={entry['train']['tokens']} "
            f"valid_windows={entry['valid']['num_windows']} "
            f"tokens_per_word={stats['tokens_per_word']:.3f} "
            f"unk_rate={stats['unk_rate']:.5f}"
        )
    print(f"wrote {out_dir / 'dataset_manifest.json'}")


if __name__ == "__main__":
    main()
