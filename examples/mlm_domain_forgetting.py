#!/usr/bin/env python3
"""Sequential MLM domain-forgetting runner (backprop, spike-only, standard PC, ePC).

S0 loads the checkpoint; stage s trains on the s-th domain with a fresh
optimizer and schedule, and every stage evaluates all domains on their fixed
validation sets. ``compare`` checks a run (spike-only PC, BP from a
perturbed initialisation for the float32 noise floor, or BP with gradient
noise at a fixed per-module cosine) against a BP run of the same setup. ``probe`` prints per-module PC vs backprop gradient cosines
and norm ratios on fixed batches of one domain, for one or a grid of PC
inference settings; for ePC it also measures the ε-spectrum and prints the
regime verdict of each setting. ``calibrate_penalty`` takes the run flags
and writes, on fixed training batches of one domain at the checkpoint, the
per-module norms of the task gradient and of the error-norm penalty's
gradient at weight 1, and ``r1``, their median ratio over the block modules:
``--pc_error_norm_penalty r / r1`` gives the block modules a ratio ``r``.

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

    PYTHONPATH=. python examples/mlm_domain_forgetting.py ... --mode pc \\
        --pc_inference standard --pc_eta_infer $ETA --pc_infer_steps 50 \\
        --pc_alignment_batches 4 --output_dir results/domain_forgetting/spc_seed0

    PYTHONPATH=. python examples/mlm_domain_forgetting.py ... --mode pc \\
        --pc_inference epc --pc_eta_infer $ETA --pc_infer_steps 5 \\
        --pc_regime_interval 50 --output_dir results/domain_forgetting/epc_seed0

    PYTHONPATH=. python examples/mlm_domain_forgetting.py ... --training joint \\
        --output_dir results/domain_forgetting/joint_seed0

    PYTHONPATH=. python examples/mlm_domain_forgetting.py ... --mode backprop \\
        --gate map --gate_beta 1 --gate_map_windows 256 \\
        --output_dir results/domain_forgetting/map_gate_b1_seed0

    Gates (stage 2 on, per-module multipliers after AdamW, in gates.jsonl):
    map / random / inverse / type (module-type means) from the onset error
    map, oracle from --gate_source runs' rollback.json (ungated runs of the
    other domain order), freeze for --gate_modules (module or group names:
    embeddings, blocks_0-1, blocks_2-3, blocks, head, mlp), uniform for a
    constant --gate_scale on every block module (the step-size control),
    online_protect / online_surprise for block multipliers refreshed every
    --gate_interval updates from the live energy smoothed with --gate_ema
    (schedules in gate_schedule_stage_<s>.jsonl), with --gate_variant live,
    type, shuffle (--gate_seed), inverse or replay (one --gate_source run).

    Interventions in every stage (any mode): --pc_error_norm_penalty GAMMA
    adds GAMMA times the backpropagated error norm at eps = 0 to the
    objective; --update_scale_modules NAME... --update_scale KAPPA multiplies
    those modules' final AdamW update by KAPPA (composes with --gate).

    $ETA has no default worth reusing. The MLM head energy is
    s * sum(masked CE) (a fixed per-token precision), so latent gradients, and
    the stable inference rate, scale with the number of masked tokens per
    micro-batch (about 1,200 at 64 x 128 x 0.15). Rates tuned under the
    earlier s * mean(masked CE) head are about that many times too large.
    Calibrate standard PC and ePC with ``probe`` at the run's micro-batch size
    (alignment for both; regime verdict and spectrum for ePC), over grids that
    span several decades:

    PYTHONPATH=. python examples/mlm_domain_forgetting.py probe \\
        --checkpoint checkpoints/bert_mini_128.json --domain news \\
        --pc_inference standard --batch_size 64 --batches 4 \\
        --grid "1e-6:50,1e-5:50,1e-4:50,1e-3:50"

    PYTHONPATH=. python examples/mlm_domain_forgetting.py probe \\
        --checkpoint checkpoints/bert_mini_128.json --domain news \\
        --pc_inference epc --batch_size 64 --batches 4 --lanczos_iters 20 \\
        --grid "1e-6:5,1e-5:5,1e-4:5,1e-3:5,1e-4:50" --final_spectrum

    Backprop or any PC run can log the ε-spectrum and regime verdict of an
    ePC setting during training (regime.jsonl): --pc_regime_interval K
    --pc_regime_eta ETA --pc_regime_steps T (ePC runs default to their own).
    ePC runs can hold the relative energy drop near a target by adapting the
    rate: --pc_eta_control energy --pc_energy_target 0.17.

    μPC (backprop and standard PC, run and probe): --pc_mupc jacobian
    multiplies the top-down error on each scaled edge by its Jacobian gain on
    the original weights; --pc_mupc full also loads those weights as W / a,
    the same function with μPC's optimiser behaviour (with --mode backprop it
    is the optimiser control).

    PYTHONPATH=. python examples/mlm_domain_forgetting.py ... --mode backprop \\
        --init_perturbation 1e-7 --output_dir results/domain_forgetting/bp_pert

    PYTHONPATH=. python examples/mlm_domain_forgetting.py ... --mode backprop \\
        --grad_noise_cosine 0.89 --output_dir results/domain_forgetting/bp_noise

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
from fabricpc.experiments.domain_forgetting.probe import parse_grid, probe_alignment
from fabricpc.experiments.domain_forgetting.runner import (
    BERT_MINI_SHAPE,
    DEFAULT_FORWARD_UPDATES,
    ETA_CONTROL,
    GRAD_NOISE_MODES,
    MODES,
    MUPC_MODES,
    PC_INFERENCE,
    GATES,
    GATE_VARIANTS,
    TRAINING,
    RunConfig,
    calibrate_error_norm_penalty,
    compare_runs,
    run_domain_forgetting,
)


def build_parser(output_required: bool = True) -> argparse.ArgumentParser:
    """Return the parser of the run command; ``output_required`` False makes
    ``--output_dir`` optional (``calibrate_penalty``)."""
    defaults = RunConfig(checkpoint="", data_dir="", domains=(), output_dir="")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", default="data/domain_forgetting")
    parser.add_argument(
        "--domains", nargs="+", default=list(DOMAIN_REGISTRY), metavar="DOMAIN"
    )
    parser.add_argument("--output_dir", required=output_required, default="")
    parser.add_argument("--mode", choices=MODES, default=defaults.mode)
    parser.add_argument(
        "--training",
        choices=TRAINING,
        default=defaults.training,
        help="sequential stages, or one joint stage on all domains mixed",
    )
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
        "--pc_inference", choices=PC_INFERENCE, default=defaults.pc_inference
    )
    parser.add_argument(
        "--pc_eta_infer",
        type=float,
        default=defaults.pc_eta_infer,
        help="Standard PC / ePC rate; latent gradients scale with the masked "
        "tokens per micro-batch (summed head energy), so calibrate it with probe",
    )
    parser.add_argument("--pc_spike_eta", type=float, default=defaults.pc_spike_eta)
    parser.add_argument(
        "--pc_spike_depth",
        choices=("longest", "shortest"),
        default=defaults.pc_spike_depth,
    )
    parser.add_argument(
        "--pc_forward_updates",
        choices=("none", "initial", "final"),
        default=None,
        help="Default: 'initial' for spiking, 'none' for standard and epc",
    )
    parser.add_argument(
        "--pc_infer_steps",
        type=int,
        default=None,
        help="Spiking default: the longest free depth; required for standard/epc",
    )
    parser.add_argument("--pc_max_norm", type=float, default=defaults.pc_max_norm)
    parser.add_argument(
        "--pc_output_energy_scale",
        type=float,
        default=defaults.pc_output_energy_scale,
    )
    parser.add_argument(
        "--pc_mupc",
        choices=MUPC_MODES,
        default=defaults.pc_mupc,
        help="μPC for backprop or standard PC: full = μPC with the W / a load "
        "(same function, μPC's optimiser behaviour); jacobian = μPC's top-down "
        "Jacobian gain only, original weights (no effect on backprop gradients)",
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
    parser.add_argument(
        "--grad_noise_cosine",
        type=float,
        default=defaults.grad_noise_cosine,
        metavar="C",
        help="Replace each module's gradient by one of equal norm at cosine C "
        "to it with a random orthogonal part, C in (0, 1) (backprop only)",
    )
    parser.add_argument(
        "--grad_noise_stage_cosines",
        type=float,
        nargs="+",
        default=list(defaults.grad_noise_stage_cosines),
        metavar="C",
        help="One cosine per stage, replacing --grad_noise_cosine stage by stage",
    )
    parser.add_argument("--grad_noise_seed", type=int, default=defaults.grad_noise_seed)
    parser.add_argument(
        "--grad_noise_mode",
        choices=GRAD_NOISE_MODES,
        default=defaults.grad_noise_mode,
        help="fresh = new noise direction every update; persistent = one per "
        "stage and module, re-projected against each update's gradient",
    )
    parser.add_argument(
        "--pc_alignment_batches",
        type=int,
        default=defaults.pc_alignment_batches,
        metavar="N",
        help="PC vs BP gradient alignment on N fixed batches at each stage's "
        "start and end, written to alignment.jsonl; 0 = off",
    )
    parser.add_argument(
        "--pc_regime_interval",
        type=int,
        default=defaults.pc_regime_interval,
        metavar="K",
        help="Log the eps-spectrum and ePC regime verdict at each stage's start, "
        "every K updates and its end, to regime.jsonl; 0 = off",
    )
    parser.add_argument("--pc_regime_eta", type=float, default=None)
    parser.add_argument("--pc_regime_steps", type=int, default=None)
    parser.add_argument(
        "--pc_regime_iters",
        type=int,
        default=defaults.pc_regime_iters,
        help="Lanczos steps per probe; each is one Hessian-vector product, so "
        "a probe costs about as much as tens of training micro-batch gradients",
    )
    parser.add_argument(
        "--pc_regime_final",
        action="store_true",
        help="Also measure the spectrum after pc_regime_steps ePC steps",
    )
    parser.add_argument(
        "--pc_eta_control",
        choices=ETA_CONTROL,
        default=defaults.pc_eta_control,
        help="energy = adapt the ePC rate after every update so the relative "
        "energy drop tracks --pc_energy_target; it restarts at --pc_eta_infer "
        "every stage (ePC only)",
    )
    parser.add_argument(
        "--pc_energy_target", type=float, default=defaults.pc_energy_target
    )
    parser.add_argument("--pc_eta_gain", type=float, default=defaults.pc_eta_gain)
    parser.add_argument(
        "--pc_eta_min",
        type=float,
        default=None,
        help="Controller floor (default pc_eta_infer / 100)",
    )
    parser.add_argument(
        "--pc_eta_max",
        type=float,
        default=None,
        help="Controller ceiling (default pc_eta_infer)",
    )
    parser.add_argument(
        "--pc_eta_backoff",
        type=float,
        default=defaults.pc_eta_backoff,
        help="Largest rate factor after an update whose inference raised the energy",
    )
    parser.add_argument(
        "--gate",
        choices=GATES,
        default=defaults.gate,
        help="Per-module update gate from stage 2 on; none = ungated",
    )
    parser.add_argument("--gate_beta", type=float, default=defaults.gate_beta)
    parser.add_argument(
        "--gate_modules",
        nargs="*",
        default=list(defaults.gate_modules),
        metavar="NAME",
        help="Modules or groups to freeze (--gate freeze)",
    )
    parser.add_argument(
        "--gate_source",
        nargs="*",
        default=list(defaults.gate_source),
        metavar="RUN_DIR",
        help="Run directories whose rollback.json gives the oracle scores, or "
        "the one online gate run to replay",
    )
    parser.add_argument(
        "--gate_map_windows", type=int, default=defaults.gate_map_windows
    )
    parser.add_argument("--gate_seed", type=int, default=defaults.gate_seed)
    parser.add_argument(
        "--gate_scale",
        type=float,
        default=defaults.gate_scale,
        help="Block-module multiplier of --gate uniform",
    )
    parser.add_argument(
        "--gate_interval",
        type=int,
        default=defaults.gate_interval,
        help="Updates between online gate refreshes",
    )
    parser.add_argument(
        "--gate_ema",
        type=float,
        default=defaults.gate_ema,
        help="EMA weight of a refresh's energy (online gates)",
    )
    parser.add_argument(
        "--gate_variant",
        choices=GATE_VARIANTS,
        default=defaults.gate_variant,
        help="Online gate variant",
    )
    parser.add_argument(
        "--pc_error_norm_penalty",
        type=float,
        default=defaults.pc_error_norm_penalty,
        metavar="GAMMA",
        help="Add GAMMA times the backpropagated error norm sum_l |dE/d eps_l|^2 "
        "at eps = 0, per masked token, to every update's objective (any mode); "
        "0 = off",
    )
    parser.add_argument(
        "--update_scale_modules",
        nargs="*",
        default=list(defaults.update_scale_modules),
        metavar="NAME",
        help="Modules or groups (as --gate_modules) whose final update "
        "--update_scale multiplies in every stage",
    )
    parser.add_argument(
        "--update_scale",
        type=float,
        default=defaults.update_scale,
        metavar="KAPPA",
        help="Multiplier of the --update_scale_modules updates, after AdamW and "
        "any gate",
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
            "Compare a run (PC, perturbed BP or gradient-noised BP) with a BP "
            "reference run of the same setup."
        ),
    )
    parser.add_argument("--pc_dir", "--run_dir", dest="pc_dir", required=True)
    parser.add_argument("--bp_dir", "--reference_dir", dest="bp_dir", required=True)
    parser.add_argument(
        "--out", default=None, help="Default: <pc_dir>/pc_bp_agreement.json"
    )
    return parser


def build_probe_parser() -> argparse.ArgumentParser:
    """Return the parser of the ``probe`` command."""
    defaults = RunConfig(checkpoint="", data_dir="", domains=(), output_dir="")
    parser = argparse.ArgumentParser(
        prog="mlm_domain_forgetting.py probe",
        description=(
            "Per-module PC vs backprop gradient cosine and norm ratio on fixed "
            "training batches of one domain."
        ),
    )
    parser.add_argument("--checkpoint", default=None, help="Default: the run's")
    parser.add_argument("--run_dir", default=None)
    parser.add_argument(
        "--stage", type=int, default=None, help="Probe stage_<s>.npz of --run_dir"
    )
    parser.add_argument("--data_dir", default=None, help="Default: the run's")
    parser.add_argument("--domain", required=True, choices=list(DOMAIN_REGISTRY))
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--pc_inference", choices=PC_INFERENCE, default="spiking")
    parser.add_argument("--pc_eta_infer", type=float, default=None)
    parser.add_argument("--pc_infer_steps", type=int, default=None)
    parser.add_argument("--pc_max_norm", type=float, default=None)
    parser.add_argument("--pc_output_energy_scale", type=float, default=None)
    parser.add_argument(
        "--pc_forward_updates", choices=("none", "initial", "final"), default=None
    )
    parser.add_argument(
        "--pc_mupc",
        choices=MUPC_MODES,
        default=None,
        help="μPC mode (standard PC only); default: the run's, else off. A "
        "stage of a run with another μPC parameterisation is converted",
    )
    parser.add_argument(
        "--grid",
        default=None,
        metavar="ETA:STEPS,...",
        help="Probe each eta:steps setting instead of --pc_eta_infer/--pc_infer_steps",
    )
    parser.add_argument("--dropout_rate", type=float, default=0.0)
    parser.add_argument("--mask_prob", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--lanczos_iters",
        type=int,
        default=20,
        help="Lanczos steps of the eps-spectrum (--pc_inference epc)",
    )
    parser.add_argument(
        "--final_spectrum",
        action="store_true",
        help="Also measure the eps-spectrum at each setting's relaxed errors "
        "(the regime verdict is a statement about eps = 0)",
    )
    parser.add_argument(
        "--skip_shape_check",
        action="store_true",
        help=f"Do not require the BERT-Mini shape {BERT_MINI_SHAPE}",
    )
    parser.add_argument("--out", default=None, help="Also write the JSON here")
    return parser


def run_probe(argv: Sequence[str]) -> dict[str, Any]:
    """Probe PC vs backprop gradient alignment and print one line per module."""
    args = build_probe_parser().parse_args(argv)
    if args.grid is not None:
        if args.pc_eta_infer is not None or args.pc_infer_steps is not None:
            raise ValueError("--grid replaces --pc_eta_infer and --pc_infer_steps")
        settings = parse_grid(args.grid)
    else:
        eta = 0.0 if args.pc_eta_infer is None else args.pc_eta_infer
        settings = [(eta, args.pc_infer_steps)]
    report = probe_alignment(
        domain=args.domain,
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        run_dir=args.run_dir,
        stage=args.stage,
        num_batches=args.batches,
        batch_size=args.batch_size,
        pc_inference=args.pc_inference,
        settings=settings,
        max_norm=args.pc_max_norm,
        output_energy_scale=args.pc_output_energy_scale,
        forward_updates=args.pc_forward_updates,
        dropout_rate=args.dropout_rate,
        mask_prob=args.mask_prob,
        seed=args.seed,
        expected_shape=None if args.skip_shape_check else dict(BERT_MINI_SHAPE),
        lanczos_iters=args.lanczos_iters,
        final_spectrum=args.final_spectrum,
        pc_mupc=args.pc_mupc,
    )
    if args.out:
        atomic_write_json(Path(args.out), report)
    return report


def config_from_args(args: argparse.Namespace) -> RunConfig:
    """Build the run configuration from parsed arguments."""
    return RunConfig(
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        domains=tuple(args.domains),
        output_dir=args.output_dir,
        mode=args.mode,
        training=args.training,
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
        pc_forward_updates=(
            args.pc_forward_updates or DEFAULT_FORWARD_UPDATES[args.pc_inference]
        ),
        pc_infer_steps=args.pc_infer_steps,
        pc_max_norm=args.pc_max_norm,
        pc_output_energy_scale=args.pc_output_energy_scale,
        pc_alignment_batches=args.pc_alignment_batches,
        pc_regime_interval=args.pc_regime_interval,
        pc_regime_eta=args.pc_regime_eta,
        pc_regime_steps=args.pc_regime_steps,
        pc_regime_iters=args.pc_regime_iters,
        pc_regime_final=args.pc_regime_final,
        pc_eta_control=args.pc_eta_control,
        pc_energy_target=args.pc_energy_target,
        pc_eta_gain=args.pc_eta_gain,
        pc_eta_min=args.pc_eta_min,
        pc_eta_max=args.pc_eta_max,
        pc_eta_backoff=args.pc_eta_backoff,
        init_perturbation=args.init_perturbation,
        init_perturbation_seed=args.init_perturbation_seed,
        grad_noise_cosine=args.grad_noise_cosine,
        grad_noise_seed=args.grad_noise_seed,
        grad_noise_mode=args.grad_noise_mode,
        grad_noise_stage_cosines=tuple(args.grad_noise_stage_cosines),
        pc_mupc=args.pc_mupc,
        gate=args.gate,
        gate_beta=args.gate_beta,
        gate_modules=tuple(args.gate_modules),
        gate_source=tuple(str(Path(p).resolve()) for p in args.gate_source),
        gate_map_windows=args.gate_map_windows,
        gate_seed=args.gate_seed,
        gate_scale=args.gate_scale,
        gate_interval=args.gate_interval,
        gate_ema=args.gate_ema,
        gate_variant=args.gate_variant,
        pc_error_norm_penalty=args.pc_error_norm_penalty,
        update_scale_modules=tuple(args.update_scale_modules),
        update_scale=args.update_scale,
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


def build_calibrate_parser() -> argparse.ArgumentParser:
    """Return the parser of ``calibrate_penalty``: the run flags plus its own."""
    parser = build_parser(output_required=False)
    parser.prog = "mlm_domain_forgetting.py calibrate_penalty"
    parser.add_argument("--domain", choices=list(DOMAIN_REGISTRY), default="news")
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--out", required=True, help="Write the JSON here")
    return parser


def run_calibrate_penalty(argv: Sequence[str]) -> dict[str, Any]:
    """Write the error-norm penalty calibration of one run setup."""
    args = build_calibrate_parser().parse_args(argv)
    report = calibrate_error_norm_penalty(
        config_from_args(args), args.domain, args.batches
    )
    atomic_write_json(args.out, report)
    print(json.dumps({"r1": report["r1"], "out": args.out}, sort_keys=True))
    return report


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """Run the benchmark, ``compare`` two finished runs, or ``probe`` alignment."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["compare"]:
        return run_compare(argv[1:])
    if argv[:1] == ["probe"]:
        return run_probe(argv[1:])
    if argv[:1] == ["calibrate_penalty"]:
        return run_calibrate_penalty(argv[1:])
    args = build_parser().parse_args(argv)
    result = run_domain_forgetting(
        config_from_args(args), resume=args.resume, overwrite=args.overwrite
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
