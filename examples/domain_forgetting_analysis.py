#!/usr/bin/env python3
"""Localisation and correlation analyses of finished domain-forgetting runs.

Every command reads a run directory written by ``mlm_domain_forgetting.py``
(stage snapshots and fixed validation sets) and writes its JSON there; none
retrains. Every JSON records the stage ``params_sha256`` values and
validation SHA-256s it used. ``correlations`` reads ``rollback.json``,
``predictors.json`` and ``error_maps.json`` of one or more runs (seeds),
refuses any computed from other stage snapshots, and writes
``correlations.json`` and ``correlations.png``. ``frontier`` reads only
``run_config.json`` and ``summary.json`` of the ``--run_dirs`` (default:
every run under ``--results_dir``) and writes ``frontier.json``,
``frontier.pdf`` and ``frontier.png``; ``all`` leaves it out.
``architecture`` reads the predictor tables of the ``--run_dirs``, groups
them by domain order and writes ``architecture_baseline.json`` (the
leave-one-order-out test of onset energy beyond type + depth).

Usage:
    PYTHONPATH=. python examples/domain_forgetting_analysis.py drift \\
        --run_dir results/domain_forgetting/bp_seed0
    PYTHONPATH=. python examples/domain_forgetting_analysis.py rollback \\
        --run_dir results/domain_forgetting/bp_seed0 --random_draws 3
    PYTHONPATH=. python examples/domain_forgetting_analysis.py error_maps \\
        --run_dir results/domain_forgetting/bp_seed0 \\
        --pc_infer_steps 120 --pc_eta_infer 0.1
    PYTHONPATH=. python examples/domain_forgetting_analysis.py predictors \\
        --run_dir results/domain_forgetting/bp_seed0
    PYTHONPATH=. python examples/domain_forgetting_analysis.py correlations \\
        --run_dirs results/domain_forgetting/bp_seed{0,1,2} \\
        --output_dir results/domain_forgetting/correlations
    PYTHONPATH=. python examples/domain_forgetting_analysis.py frontier \\
        --results_dir results/domain_forgetting \\
        --output_dir results/domain_forgetting/frontier
    PYTHONPATH=. python examples/domain_forgetting_analysis.py architecture \\
        --run_dirs results/domain_forgetting/g8_rev_bp_seed{0,1,2} ... \\
        --development_orders news,reviews,wiki wiki,reviews,news \\
        --output_dir results/domain_forgetting/architecture
    PYTHONPATH=. python examples/domain_forgetting_analysis.py all \\
        --run_dir results/domain_forgetting/bp_seed0
"""

# ruff: noqa: E402

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from jax_setup import set_jax_flags_before_importing_jax

set_jax_flags_before_importing_jax()

from fabricpc.experiments.domain_forgetting import architecture_baseline
from fabricpc.experiments.domain_forgetting.artifacts import atomic_write_json
from fabricpc.experiments.domain_forgetting.error_maps import (
    SETTINGS,
    ErrorMapConfig,
    compute_error_maps,
)
from fabricpc.experiments.domain_forgetting.correlations import (
    BOOTSTRAP_RESAMPLES,
    PERMUTATIONS,
    compute_correlations,
    compute_predictors,
    plot_correlations,
)
from fabricpc.experiments.domain_forgetting.frontier import (
    SEQ_LR,
    collect_runs,
    compute_frontier,
    format_table,
    plot_frontier,
)
from fabricpc.experiments.domain_forgetting.localization import (
    RANDOM_DRAWS,
    compute_drift,
    compute_rollback,
    load_run,
    run_provenance,
)

DEVELOPMENT = architecture_baseline.DEVELOPMENT_ORDERS
RUN_COMMANDS = ("drift", "rollback", "error_maps", "predictors", "correlations")
COMMANDS = (*RUN_COMMANDS, "frontier", "architecture", "all")


def build_parser() -> argparse.ArgumentParser:
    """Return the analysis parser."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--run_dir", help="run directory (all but correlations)")
    parser.add_argument(
        "--run_dirs",
        nargs="+",
        help="run directories (seeds) for correlations, or runs for frontier "
        "and architecture",
    )
    parser.add_argument(
        "--results_dir",
        default="results/domain_forgetting",
        help="frontier reads every run under it when --run_dirs is not given",
    )
    parser.add_argument("--seq_lr", type=float, default=SEQ_LR)
    parser.add_argument(
        "--output_dir",
        help="where correlations (default: the first run directory) or "
        "frontier (default: <results_dir>/frontier) writes",
    )
    parser.add_argument("--checkpoint", help="override the run's checkpoint path")
    parser.add_argument("--data_dir", help="override the run's data directory")
    parser.add_argument("--max_valid_windows", type=int)
    parser.add_argument("--eval_batch_size", type=int)
    parser.add_argument("--random_draws", type=int, default=RANDOM_DRAWS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_groups", action="store_true")
    parser.add_argument("--map_windows", type=int, default=256)
    parser.add_argument("--map_batch_size", type=int, default=8)
    parser.add_argument("--settings", nargs="+", choices=SETTINGS, default=SETTINGS)
    parser.add_argument("--pc_infer_steps", type=int, default=120)
    parser.add_argument("--pc_eta_infer", type=float, default=0.1)
    parser.add_argument("--pc_max_norm", type=float, default=5.0)
    parser.add_argument("--output_energy_scale", type=float, default=1.0)
    parser.add_argument("--predictor_windows", type=int, default=256)
    parser.add_argument("--predictor_batch_size", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument(
        "--permutations",
        type=int,
        help=f"default {PERMUTATIONS} (correlations) or "
        f"{architecture_baseline.PERMUTATIONS} (architecture)",
    )
    parser.add_argument(
        "--development_orders",
        nargs="+",
        default=[architecture_baseline.order_key(o) for o in DEVELOPMENT],
        help="architecture: development orders as comma-joined domains",
    )
    parser.add_argument(
        "--held_out_orders",
        nargs="+",
        help="architecture: held-out orders (default: all non-development orders)",
    )
    parser.add_argument("--no_figure", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def _context(args: argparse.Namespace):
    if not args.run_dir:
        raise SystemExit(f"{args.command} needs --run_dir")
    return load_run(
        args.run_dir,
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        max_valid_windows=args.max_valid_windows,
    )


def run_correlations(args: argparse.Namespace) -> dict[str, Any]:
    """Correlate predictors with rollback recovery over one or more runs."""
    run_dirs = args.run_dirs or ([args.run_dir] if args.run_dir else [])
    if not run_dirs:
        raise SystemExit("correlations needs --run_dirs or --run_dir")
    output = Path(args.output_dir or run_dirs[0])
    output.mkdir(parents=True, exist_ok=True)
    result = compute_correlations(
        run_dirs,
        resamples=args.bootstrap,
        permutations=PERMUTATIONS if args.permutations is None else args.permutations,
        seed=args.seed,
    )
    if not args.no_figure:
        figure = plot_correlations(result, output / "correlations.png")
        result["figure"] = str(figure) if figure else None
    atomic_write_json(output / "correlations.json", result)
    return result


def run_frontier(args: argparse.Namespace) -> dict[str, Any]:
    """Place every run against the uniform-lr frontier of its order and seed."""
    if args.run_dirs:
        run_dirs = [Path(p) for p in args.run_dirs]
    else:
        root = Path(args.results_dir)
        run_dirs = sorted(p for p in root.iterdir() if (p / "run_config.json").exists())
    output = Path(args.output_dir or Path(args.results_dir) / "frontier")
    output.mkdir(parents=True, exist_ok=True)
    result = compute_frontier(collect_runs(run_dirs), seq_lr=args.seq_lr)
    if not args.no_figure:
        figures = plot_frontier(result, output / "frontier.pdf")
        result["figures"] = [str(f) for f in figures] if figures else None
    atomic_write_json(output / "frontier.json", result)
    if not args.quiet:
        print(format_table(result))
    return result


def run_architecture(args: argparse.Namespace) -> dict[str, Any]:
    """Test whether onset energy predicts recovery beyond type + depth."""
    if not args.run_dirs:
        raise SystemExit("architecture needs --run_dirs")
    output = Path(args.output_dir or Path(args.results_dir) / "architecture")
    output.mkdir(parents=True, exist_ok=True)
    parse = architecture_baseline.parse_order
    result = architecture_baseline.run_architecture_baseline(
        args.run_dirs,
        development=[parse(o) for o in args.development_orders],
        held_out=(
            [parse(o) for o in args.held_out_orders] if args.held_out_orders else None
        ),
        permutations=(
            architecture_baseline.PERMUTATIONS
            if args.permutations is None
            else args.permutations
        ),
        seed=args.seed,
    )
    atomic_write_json(output / "architecture_baseline.json", result)
    if not args.quiet:
        print(architecture_baseline.format_table(result))
    return result


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """Run one analysis command, or ``all`` of them in order."""
    args = build_parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    verbose = not args.quiet
    commands = RUN_COMMANDS if args.command == "all" else (args.command,)
    results: dict[str, Any] = {}
    context = None
    for command in commands:
        started = time.time()
        if command == "correlations":
            results[command] = run_correlations(args)
        elif command == "frontier":
            results[command] = run_frontier(args)
        elif command == "architecture":
            results[command] = run_architecture(args)
        else:
            context = context or _context(args)
            output = Path(args.run_dir)
            if command == "drift":
                value = compute_drift(
                    context.snapshots,
                    context.structure,
                    context.domains,
                    provenance=run_provenance(context),
                )
            elif command == "rollback":
                value = compute_rollback(
                    context,
                    random_draws=args.random_draws,
                    seed=args.seed,
                    include_groups=not args.no_groups,
                    eval_batch_size=args.eval_batch_size,
                    verbose=verbose,
                )
            elif command == "error_maps":
                value = compute_error_maps(
                    context,
                    ErrorMapConfig(
                        num_windows=args.map_windows,
                        batch_size=args.map_batch_size,
                        pc_infer_steps=args.pc_infer_steps,
                        pc_eta_infer=args.pc_eta_infer,
                        pc_max_norm=args.pc_max_norm,
                        output_energy_scale=args.output_energy_scale,
                        settings=tuple(args.settings),
                    ),
                    verbose=verbose,
                )
            else:
                value = compute_predictors(
                    context,
                    num_windows=args.predictor_windows,
                    batch_size=args.predictor_batch_size,
                    verbose=verbose,
                )
            value["seconds"] = time.time() - started
            atomic_write_json(output / f"{command}.json", value)
            results[command] = value
        if verbose:
            print(f"{command}: {time.time() - started:.1f}s")
    return results[args.command] if args.command != "all" else results


if __name__ == "__main__":
    main()
