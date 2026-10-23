#!/usr/bin/env python3
"""Sequential MLM domain-forgetting runner (backprop or spike-only PC).

S0 loads the checkpoint; stage s trains on the s-th domain with a fresh
optimizer and schedule, and every stage evaluates all domains on their fixed
validation sets. ``compare`` checks a run (spike-only PC, or BP from a
perturbed initialisation for the float32 noise floor) against a BP run of
the same setup.

Usage:
    PYTHONPATH=. python examples/mlm_domain_forgetting.py \\
        --checkpoint checkpoints/bert_mini_128.json \\
        --data_dir data/domain_forgetting --domains news reviews wiki \\
        --mode backprop --tokens_per_domain 5000000 --lr 1e-4 \\
        --seed 0 --output_dir results/domain_forgetting/bp_seed0

    PYTHONPATH=. python examples/mlm_domain_forgetting.py ... --mode pc \\
        --pc_inference spiking --pc_eta_infer 0 --pc_spike_eta 1 \\
        --pc_spike_depth longest --pc_forward_updates initial \\
        --output_dir results/domain_forgetting/pc_seed0

    PYTHONPATH=. python examples/mlm_domain_forgetting.py ... --mode backprop \\
        --init_perturbation 1e-7 --output_dir results/domain_forgetting/bp_pert

    PYTHONPATH=. python examples/mlm_domain_forgetting.py compare \\
        --pc_dir results/domain_forgetting/pc_seed0 \\
        --bp_dir results/domain_forgetting/bp_seed0
"""

# ruff: noqa: E402

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from jax_setup import set_jax_flags_before_importing_jax

set_jax_flags_before_importing_jax()

from fabricpc.experiments.domain_forgetting.artifacts import atomic_write_json
from fabricpc.experiments.domain_forgetting.data import DOMAIN_REGISTRY
from fabricpc.experiments.domain_forgetting.runner import (
    BERT_MINI_SHAPE,
    MODES,
    RunConfig,
    compare_runs,
    run_domain_forgetting,
)


def build_parser() -> argparse.ArgumentParser:
    """Return the parser of the run command."""
    defaults = RunConfig(checkpoint="", data_dir="", domains=(), output_dir="")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", default="data/domain_forgetting")
    parser.add_argument(
        "--domains", nargs="+", default=list(DOMAIN_REGISTRY), metavar="DOMAIN"
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mode", choices=MODES, default=defaults.mode)
    parser.add_argument(
        "--tokens_per_domain", type=int, default=defaults.tokens_per_domain
    )
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--batch_size", type=int, default=defaults.batch_size)
    parser.add_argument("--micro_batch_size", type=int, default=None)
    parser.add_argument(
        "--warmup_fraction", type=float, default=defaults.warmup_fraction
    )
    parser.add_argument("--min_lr_ratio", type=float, default=defaults.min_lr_ratio)
    parser.add_argument("--weight_decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--grad_clip", type=float, default=defaults.grad_clip)
    parser.add_argument("--adam_eps", type=float, default=defaults.adam_eps)
    parser.add_argument("--dropout_rate", type=float, default=defaults.dropout_rate)
    parser.add_argument("--mask_prob", type=float, default=defaults.mask_prob)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--eval_batch_size", type=int, default=defaults.eval_batch_size)
    parser.add_argument(
        "--max_valid_windows",
        type=int,
        default=None,
        help="Evaluate only the first N fixed windows per domain (smoke runs)",
    )
    parser.add_argument(
        "--skip_shape_check",
        action="store_true",
        help=f"Do not require the BERT-Mini shape {BERT_MINI_SHAPE}",
    )
    parser.add_argument(
        "--pc_inference", choices=("spiking",), default=defaults.pc_inference
    )
    parser.add_argument("--pc_eta_infer", type=float, default=defaults.pc_eta_infer)
    parser.add_argument("--pc_spike_eta", type=float, default=defaults.pc_spike_eta)
    parser.add_argument(
        "--pc_spike_depth",
        choices=("longest", "shortest"),
        default=defaults.pc_spike_depth,
    )
    parser.add_argument(
        "--pc_forward_updates",
        choices=("none", "initial", "final"),
        default=defaults.pc_forward_updates,
    )
    parser.add_argument(
        "--pc_infer_steps",
        type=int,
        default=None,
        help="Default: the longest free depth of the graph",
    )
    parser.add_argument("--pc_max_norm", type=float, default=defaults.pc_max_norm)
    parser.add_argument(
        "--pc_output_energy_scale",
        type=float,
        default=defaults.pc_output_energy_scale,
    )
    parser.add_argument(
        "--init_perturbation",
        type=float,
        default=defaults.init_perturbation,
        metavar="REL",
        help="Multiply every source parameter by (1 + REL * N(0, 1)); 0 = off",
    )
    parser.add_argument(
        "--init_perturbation_seed", type=int, default=defaults.init_perturbation_seed
    )
    parser.add_argument("--log_interval", type=int, default=defaults.log_interval)
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--resume", action="store_true")
    resume.add_argument("--overwrite", action="store_true")
    return parser


def build_compare_parser() -> argparse.ArgumentParser:
    """Return the parser of the ``compare`` command."""
    parser = argparse.ArgumentParser(
        prog="mlm_domain_forgetting.py compare",
        description=(
            "Compare a run (PC, or perturbed BP) with a BP reference run of "
            "the same setup."
        ),
    )
    parser.add_argument("--pc_dir", "--run_dir", dest="pc_dir", required=True)
    parser.add_argument("--bp_dir", "--reference_dir", dest="bp_dir", required=True)
    parser.add_argument(
        "--out", default=None, help="Default: <pc_dir>/pc_bp_agreement.json"
    )
    return parser


def config_from_args(args: argparse.Namespace) -> RunConfig:
    """Build the run configuration from parsed arguments."""
    return RunConfig(
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        domains=tuple(args.domains),
        output_dir=args.output_dir,
        mode=args.mode,
        tokens_per_domain=args.tokens_per_domain,
        lr=args.lr,
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        warmup_fraction=args.warmup_fraction,
        min_lr_ratio=args.min_lr_ratio,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        adam_eps=args.adam_eps,
        dropout_rate=args.dropout_rate,
        mask_prob=args.mask_prob,
        seed=args.seed,
        eval_batch_size=args.eval_batch_size,
        max_valid_windows=args.max_valid_windows,
        expected_shape=None if args.skip_shape_check else dict(BERT_MINI_SHAPE),
        pc_inference=args.pc_inference,
        pc_eta_infer=args.pc_eta_infer,
        pc_spike_eta=args.pc_spike_eta,
        pc_spike_depth=args.pc_spike_depth,
        pc_forward_updates=args.pc_forward_updates,
        pc_infer_steps=args.pc_infer_steps,
        pc_max_norm=args.pc_max_norm,
        pc_output_energy_scale=args.pc_output_energy_scale,
        init_perturbation=args.init_perturbation,
        init_perturbation_seed=args.init_perturbation_seed,
        log_interval=args.log_interval,
    )


def run_compare(argv: Sequence[str]) -> dict[str, Any]:
    """Compare two run directories and write the agreement report."""
    args = build_compare_parser().parse_args(argv)
    report = {
        "pc_dir": args.pc_dir,
        "bp_dir": args.bp_dir,
        **compare_runs(args.pc_dir, args.bp_dir),
    }
    out = Path(args.out) if args.out else Path(args.pc_dir) / "pc_bp_agreement.json"
    atomic_write_json(out, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """Run the benchmark, or ``compare`` two finished runs."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["compare"]:
        return run_compare(argv[1:])
    args = build_parser().parse_args(argv)
    result = run_domain_forgetting(
        config_from_args(args), resume=args.resume, overwrite=args.overwrite
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
