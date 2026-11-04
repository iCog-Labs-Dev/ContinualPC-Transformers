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
``frontier.pdf`` and ``frontier.png``; ``commutator`` reads the stage
snapshots and training logs of the ``--run_dirs`` (both domain orders per
seed and condition) and writes ``commutator.json``; ``all`` leaves both out.
``influence`` reads the stage snapshots, ``rollback.json`` and train streams
of the ``--run_dirs``, ablates and perturbs each module at each stage onset,
writes ``influence.json`` per run and a table of Spearman correlations to
``--output_dir``; it fails on any missing snapshot.
``architecture`` reads the predictor tables of the ``--run_dirs``, groups
them by domain order and writes ``architecture_baseline.json`` (the
leave-one-order-out test of onset energy beyond type + depth).

``--map_source learner`` selects the learner-native maps: ``error_maps``
adds the ``learner`` setting (the run's own learner errors, e.g. ePC's
relaxed errors), ``architecture`` tests its onset energy instead of the
spike-only (BP) map, and ``frontier`` runs its primary test on the
``gate_map_source="learner"`` map gates. ``frontier --reference_mode pc
--pc_inference epc`` builds the reference curves from ePC runs only.

Usage:
    PYTHONPATH=. python examples/domain_forgetting_analysis.py drift \\
        --run_dir results/domain_forgetting/bp_seed0
    PYTHONPATH=. python examples/domain_forgetting_analysis.py rollback \\
        --run_dir results/domain_forgetting/bp_seed0 --random_draws 3
    PYTHONPATH=. python examples/domain_forgetting_analysis.py error_maps \\
        --run_dir results/domain_forgetting/bp_seed0 \\
        --pc_infer_steps 120 --pc_eta_infer $ETA

    The standard_pc maps need --pc_eta_infer: the MLM head energy sums the
    masked-token CE, so the rate scales with the masked tokens per map batch
    and must be calibrated for --map_batch_size (mlm_domain_forgetting.py
    probe); --settings spike_only needs none.
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
    PYTHONPATH=. python examples/domain_forgetting_analysis.py commutator \\
        --run_dirs results/domain_forgetting/commutator_*_seed* \\
        --output_dir results/domain_forgetting/commutator_bert
    PYTHONPATH=. python examples/domain_forgetting_analysis.py influence \\
        --run_dirs results/domain_forgetting/g9_*_lr3e-5_seed* \\
        --output_dir results/domain_forgetting/influence_bp
    PYTHONPATH=. python examples/domain_forgetting_analysis.py all \\
        --run_dir results/domain_forgetting/bp_seed0
"""

# ruff: noqa: E402

import argparse
import json
import sys
import time
from pathlib import Path
from dataclasses import replace
from typing import Any, Sequence

from jax_setup import set_jax_flags_before_importing_jax

set_jax_flags_before_importing_jax()

from fabricpc.experiments.domain_forgetting import architecture_baseline
from fabricpc.experiments.domain_forgetting.artifacts import atomic_write_json
from fabricpc.experiments.domain_forgetting.commutator import (
    commutator_test,
    compute_commutator,
)
from fabricpc.experiments.domain_forgetting.commutator import (
    format_table as format_commutator_table,
)
from fabricpc.experiments.domain_forgetting.error_maps import (
    LEARNER,
    MAP_SOURCES,
    SETTINGS,
    SPIKE_ONLY,
    ErrorMapConfig,
    compute_error_maps,
    merge_error_maps,
    missing_settings,
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
from fabricpc.experiments.domain_forgetting.influence import (
    compute_influence_runs,
    format_influence_table,
)
from fabricpc.experiments.domain_forgetting.runner import PC_INFERENCE
from fabricpc.experiments.domain_forgetting.localization import (
    RANDOM_DRAWS,
    compute_drift,
    compute_rollback,
    load_run,
    run_provenance,
)

DEVELOPMENT = architecture_baseline.DEVELOPMENT_ORDERS
RUN_COMMANDS = ("drift", "rollback", "error_maps", "predictors", "correlations")
COMMANDS = (
    *RUN_COMMANDS,
    "frontier",
    "architecture",
    "commutator",
    "influence",
    "all",
)
REFERENCE_MODES = ("backprop", "pc")
MAP_SETTING = {"bp": SPIKE_ONLY, "learner": LEARNER}


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
        "--reference_mode",
        choices=REFERENCE_MODES,
        default="backprop",
        help="frontier: the learner whose runs are analysed and form the "
        "reference curves",
    )
    parser.add_argument(
        "--pc_inference",
        choices=PC_INFERENCE,
        default=None,
        help="frontier with --reference_mode pc (required there): keep only "
        "this PC inference",
    )
    parser.add_argument(
        "--expected_seeds",
        type=int,
        nargs="+",
        help="frontier: seeds the primary test needs (default: the seeds with "
        "a reference run in each order)",
    )
    parser.add_argument(
        "--map_source",
        choices=MAP_SOURCES,
        default="bp",
        help="error_maps: learner adds the run's own learner map; "
        "architecture: test that map's onset energy; frontier: primary test "
        "on map gates of this gate_map_source",
    )
    parser.add_argument(
        "--output_dir",
        help="where correlations (default: the first run directory) or "
        "frontier (default: <results_dir>/frontier) writes",
    )
    parser.add_argument(
        "--influence_verdict",
        nargs=3,
        action="append",
        default=[],
        metavar=("GATE", "MATCHED", "SURPRISE"),
        help="frontier: condition labels to judge on the frontier, against the "
        "matched control and against surprise only; repeatable",
    )
    parser.add_argument(
        "--descriptive_verdict",
        nargs=3,
        action="append",
        default=[],
        metavar=("GATE", "MATCHED", "SURPRISE"),
        help="frontier: as --influence_verdict, marked as not deciding the test",
    )
    parser.add_argument(
        "--commutator_test",
        nargs=2,
        action="append",
        default=[],
        metavar=("GATE", "CONTROL"),
        help="commutator: condition labels of a D_gate < D_control test on each "
        "pair holding both; repeatable",
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
    parser.add_argument(
        "--pc_eta_infer",
        type=float,
        default=None,
        help="Standard-PC map inference rate (required with the standard_pc "
        "setting; calibrate for --map_batch_size)",
    )
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
    parser.add_argument(
        "--no_divergence",
        action="store_true",
        help="commutator: skip the function-space distance",
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
        run_dirs = sorted(
            p
            for p in root.iterdir()
            if (p / "run_config.json").exists() and not p.name.startswith("infl_")
        )
    output = Path(args.output_dir or Path(args.results_dir) / "frontier")
    output.mkdir(parents=True, exist_ok=True)
    if args.pc_inference is not None and args.reference_mode != "pc":
        raise SystemExit("--pc_inference needs --reference_mode pc")
    if args.reference_mode == "pc" and args.pc_inference is None:
        raise SystemExit("--reference_mode pc needs --pc_inference")
    points = collect_runs(
        run_dirs, mode=args.reference_mode, pc_inference=args.pc_inference
    )
    result = compute_frontier(
        points,
        seq_lr=args.seq_lr,
        map_source=args.map_source,
        expected_seeds=args.expected_seeds,
        influence_verdicts=args.influence_verdict,
        descriptive_verdicts=args.descriptive_verdict,
    )
    if args.reference_mode != "backprop" or args.map_source != "bp":
        result["reference"] = {
            "mode": args.reference_mode,
            "pc_inference": args.pc_inference,
            "map_source": args.map_source,
        }
        if args.reference_mode == "pc" and args.map_source == "bp":
            result["reference"][
                "scope"
            ] = "primary test only (no contrasts, no matched control)"
    if not args.no_figure:
        figures = plot_frontier(result, output / "frontier.pdf")
        result["figures"] = [str(f) for f in figures] if figures else None
    atomic_write_json(output / "frontier.json", result)
    if not args.quiet:
        scope = result.get("reference", {}).get("scope")
        if scope:
            print(f"bp-source map gate: {scope}")
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
        setting=MAP_SETTING[args.map_source],
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


def _existing(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.is_file() else None


def run_error_maps(
    args: argparse.Namespace,
    context: Any,
    existing: dict[str, Any] | None,
    verbose: bool,
) -> dict[str, Any]:
    """Compute the error maps, merging into the run's existing ``error_maps.json``.

    Settings already in ``existing`` are kept, not recomputed, and
    ``existing`` itself is returned when it holds them all; a setting whose
    record would change, or other stages or windows, is refused.
    """
    settings = (
        tuple(args.settings) if args.map_source == "bp" else (*args.settings, LEARNER)
    )
    config = ErrorMapConfig(
        num_windows=args.map_windows,
        batch_size=args.map_batch_size,
        pc_infer_steps=args.pc_infer_steps,
        pc_eta_infer=args.pc_eta_infer,
        pc_max_norm=args.pc_max_norm,
        output_energy_scale=args.output_energy_scale,
        settings=settings,
    )
    if existing is None:
        return compute_error_maps(context, config, verbose=verbose)
    try:
        missing = missing_settings(existing, context, config)
    except ValueError as error:
        raise SystemExit(f"{args.run_dir}/error_maps.json: {error}") from error
    if not missing:
        if verbose:
            print(f"error_maps.json already holds {list(settings)}; kept")
        return existing
    new = compute_error_maps(
        context, replace(config, settings=missing), verbose=verbose
    )
    return merge_error_maps(existing, new)


def run_predictors(
    args: argparse.Namespace,
    context: Any,
    existing: dict[str, Any] | None,
    verbose: bool,
) -> dict[str, Any]:
    """Compute ``predictors.json`` unless an identical setup already wrote it.

    ``existing`` is returned when it has the same stages, windows and batch
    size; one of any other setup is refused, never overwritten.
    """
    if existing is not None:
        same = (
            existing.get("provenance") == run_provenance(context)
            and existing.get("num_windows") == args.predictor_windows
            and existing.get("batch_size") == args.predictor_batch_size
        )
        if not same:
            raise SystemExit(
                f"{args.run_dir}/predictors.json exists with another setup; "
                "refusing to overwrite it"
            )
        if verbose:
            print("predictors.json already computed with this setup; kept")
        return existing
    return compute_predictors(
        context,
        num_windows=args.predictor_windows,
        batch_size=args.predictor_batch_size,
        verbose=verbose,
    )


def run_commutator(args: argparse.Namespace) -> dict[str, Any]:
    """Order distances and noise floors of paired-order runs."""
    if not args.run_dirs:
        raise SystemExit("commutator needs --run_dirs")
    output = Path(args.output_dir or Path(args.results_dir) / "commutator")
    output.mkdir(parents=True, exist_ok=True)
    result = compute_commutator(
        args.run_dirs,
        divergence=not args.no_divergence,
        batch_size=args.eval_batch_size or 64,
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
    )
    tests = []
    for gate, control in args.commutator_test:
        pairs = [
            row["pair"]
            for row in result["results"]
            if row["condition"] == gate
            and any(
                other["condition"] == control and other["pair"] == row["pair"]
                for other in result["results"]
            )
        ]
        if not pairs:
            raise SystemExit(f"no pair holds conditions {gate} and {control}")
        tests.extend(commutator_test(result, gate, control, pair) for pair in pairs)
    if tests:
        result["tests"] = tests
    atomic_write_json(output / "commutator.json", result)
    if not args.quiet:
        print(format_commutator_table(result))
        for test in tests:
            print(
                f"test {test['gate']} < {test['control']} {'/'.join(test['pair'])}: "
                f"passed={test['passed']} mean_gap={test['mean_gap']} "
                f"control_sd={test['control_sd']} verdict={test.get('verdict')}"
            )
    return result


def run_influence(args: argparse.Namespace) -> dict[str, Any]:
    """Ablation and perturbation influence at each stage onset of finished runs."""
    if not args.run_dirs:
        raise SystemExit("influence needs --run_dirs")
    output = Path(args.output_dir or Path(args.results_dir) / "influence")
    output.mkdir(parents=True, exist_ok=True)
    result = compute_influence_runs(
        args.run_dirs,
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        verbose=not args.quiet,
    )
    atomic_write_json(output / "influence.json", result)
    if not args.quiet:
        print(format_influence_table(result))
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
        elif command == "commutator":
            results[command] = run_commutator(args)
        elif command == "influence":
            results[command] = run_influence(args)
        else:
            context = context or _context(args)
            output = Path(args.run_dir)
            existing = (
                _existing(output / f"{command}.json")
                if command in ("error_maps", "predictors")
                else None
            )
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
                value = run_error_maps(args, context, existing, verbose)
            else:
                value = run_predictors(args, context, existing, verbose)
            if value is not existing:
                value["seconds"] = time.time() - started
                atomic_write_json(output / f"{command}.json", value)
            results[command] = value
        if verbose:
            print(f"{command}: {time.time() - started:.1f}s")
    return results[args.command] if args.command != "all" else results


if __name__ == "__main__":
    main()
