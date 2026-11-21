"""CLI for the plasticity-gate comparison, lr matching and commutator runs.

Sub-commands:
  compare     seeds x conditions sequential retention benchmark
  lr-match    choose the lr_matched scalar on the smoke budget
  commutator  two-task A->B vs B->A order-dependence per condition

Use --preset smoke for an offline synthetic-data run. Real tasks download data;
full runs are launched by the user, not by tests.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import jax
import optax

import fabricpc.experiments.forgetting_benchmark.synthetic_tasks  # noqa: F401
from fabricpc.experiments.forgetting_benchmark.benchmark import SequentialBenchmark
from fabricpc.experiments.forgetting_benchmark.commutator import (
    commutator_report,
    make_batch_fn,
    noise_floor,
    num_sequences_of,
    precompute_streams,
    run_order,
    shared_warmup,
)
from fabricpc.experiments.forgetting_benchmark.gated_trainer import (
    GateEngine,
    GatedConfig,
    GatedSequentialTrainer,
    assert_records_comparable,
    data_manifest,
    take_batches,
)
from fabricpc.experiments.forgetting_benchmark.plasticity_analysis import (
    CONTRAST_NOTE,
    match_lr_scale,
    run_metrics,
    summarize_contrasts,
)
from fabricpc.experiments.forgetting_benchmark.run_benchmark import (
    BenchmarkConfig,
    _model_factory,
)
from fabricpc.experiments.forgetting_benchmark.sequential_trainer import (
    dumps_strict,
    params_hash,
)
from fabricpc.training.plasticity_gates import GateSchedule
from fabricpc.training.train_autoregressive import _eval_step_autoregressive

CONDITIONS: dict[str, tuple[str, dict[str, Any]]] = {
    "none": ("none", {}),
    "surprise": ("surprise", {}),
    "influence": ("influence", {}),
    "influence_current": ("influence", {"probe_source": "current"}),
    "replay_gates": ("replay_gates", {}),
    "replay_gates_cross": ("replay_gates", {}),
    "lr_matched": ("none", {}),
    "lr_matched_influence": ("none", {}),
    "lr_matched_learning": ("none", {}),
}
# condition -> (target condition, criterion): the matched run is plain SGD with
# the same ungated warmup, then one constant scale chosen per seed so that the
# criterion equals the target run's.
MATCHES = {
    "lr_matched": ("surprise", "path_length"),
    "lr_matched_influence": ("influence", "path_length"),
    "lr_matched_learning": ("surprise", "immediate_train_loss"),
}
MATCH_TOLERANCE = 0.05
DEFAULT_CONDITIONS = (
    "surprise",
    "none",
    "influence",
    "replay_gates",
    "replay_gates_cross",
    "lr_matched",
    "lr_matched_influence",
    "lr_matched_learning",
    "influence_current",
)
SMOKE_PRESET = dict(
    tasks=["synthetic_a", "synthetic_b"],
    embed_dim=32,
    seq_len=16,
    batch_size=8,
    max_samples=800,
    max_chars_per_task=4000,
    epochs_per_task=3,
    lr=0.01,
    warmup_steps=100,
    diag_every=50,
)
LR_MATCH_FILES = {
    "path": "lr_match_path.json",
    "path_influence": "lr_match_path_influence.json",
    "learning": "lr_match_learning.json",
}
CRITERION_OF = {
    "path": "lr_matched",
    "path_influence": "lr_matched_influence",
    "learning": "lr_matched_learning",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("smoke",), default=None)
    parser.add_argument("command", choices=("compare", "lr-match", "commutator"))
    parser.add_argument(
        "--tasks", nargs="+", default=["tiny_stories", "wikitext2", "tiny_shakespeare"]
    )
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--mlp-dim", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--max-chars-per-task", type=int, default=100_000)
    parser.add_argument("--epochs-per-task", type=float, default=2)
    parser.add_argument("--infer-steps", type=int, default=5)
    parser.add_argument("--eta-infer", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--diag-every", type=int, default=50)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--calib-batches", type=int, default=8)
    parser.add_argument("--probe-batches", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--data-dir", default="data/forgetting_benchmark")
    parser.add_argument("--output-dir", default="results/plasticity_gates")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=list(DEFAULT_CONDITIONS),
        choices=sorted(CONDITIONS),
    )
    parser.add_argument("--criterion", choices=sorted(LR_MATCH_FILES), default="path")
    parser.add_argument(
        "--match-seed",
        type=int,
        default=None,
        help="seed for lr-match and commutator matching; must not be in --seeds",
    )
    parser.add_argument("--steps-per-task", type=int, default=200)
    return parser


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = build_parser()
    preset_probe = argparse.ArgumentParser(add_help=False)
    preset_probe.add_argument("--preset", default=None)
    known, _ = preset_probe.parse_known_args(argv)
    if known.preset == "smoke":
        parser.set_defaults(**SMOKE_PRESET)
    return parser.parse_args(argv)


def build_benchmark(args: argparse.Namespace, seed: int) -> SequentialBenchmark:
    return SequentialBenchmark(
        list(args.tasks),
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        seed=seed,
        max_samples_per_task=args.max_samples,
        max_chars_per_task=args.max_chars_per_task,
    )


def build_factory(args: argparse.Namespace, vocab_size: int):
    return _model_factory(
        BenchmarkConfig(
            tasks=tuple(args.tasks),
            depth=args.depth,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            mlp_dim=args.mlp_dim,
            seq_len=args.seq_len,
            infer_steps=args.infer_steps,
            eta_infer=args.eta_infer,
        ),
        vocab_size,
    )


def gated_config(args: argparse.Namespace, **overrides: Any) -> GatedConfig:
    values = dict(
        lr=args.lr,
        num_epochs=args.epochs_per_task,
        diag_every=args.diag_every,
        calib_batches=args.calib_batches,
        warmup_steps=args.warmup_steps,
        alpha=args.alpha,
        probe_batches=args.probe_batches,
    )
    values.update(overrides)
    return GatedConfig(**values)


def run_settings(args: argparse.Namespace) -> dict[str, Any]:
    """Data and model settings that every compared run must share."""
    return {
        key: getattr(args, key)
        for key in (
            "tasks",
            "seq_len",
            "batch_size",
            "max_samples",
            "max_chars_per_task",
            "infer_steps",
            "eta_infer",
            "depth",
            "embed_dim",
            "num_heads",
            "mlp_dim",
        )
    } | {"tasks": list(args.tasks)}


def _theta0_hash(factory, seed: int) -> str:
    model_key = jax.random.split(jax.random.PRNGKey(seed), 3)[0]
    return params_hash(factory(model_key)[0])


def _bracket_center(args: argparse.Namespace, criterion: str) -> Optional[float]:
    path = Path(args.output_dir) / LR_MATCH_FILES[criterion]
    return float(json.loads(path.read_text())["scale"]) if path.exists() else None


def _target_value(outcome: dict[str, Any], criterion: str) -> float:
    record = outcome["record"]
    if criterion == "path_length":
        return float(record["path_length"])
    return float(
        sum(record["immediate_train_loss"]) / len(record["immediate_train_loss"])
    )


def _match_scale(
    args: argparse.Namespace,
    seed: int,
    condition: str,
    bench: SequentialBenchmark,
    factory,
    root: Path,
    target_outcome: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Per-seed constant post-warmup scale that matches the target run."""
    target_condition, criterion = MATCHES[condition]
    match_path = root / f"seed_{seed}" / f"{condition}_match.json"
    if match_path.exists():
        return json.loads(match_path.read_text())
    target = _target_value(target_outcome, criterion)

    def measure(scale: float) -> float:
        config = gated_config(args, post_warmup_scale=scale)
        run_dir = root / f"seed_{seed}" / f"{condition}_search" / f"{scale:.6f}"
        result = GatedSequentialTrainer(
            factory, bench, config, verbose=False, run_settings=settings
        ).run(str(run_dir), seed, "none")
        return _target_value({"record": result.run_record}, criterion)

    criterion_name = {v: k for k, v in CRITERION_OF.items()}[condition]
    center = _bracket_center(args, criterion_name)
    bracket = (0.7 * center, 1.4 * center) if center else (0.1, 2.0)
    found = match_lr_scale(measure, target, tolerance=0.01, bracket=bracket)
    found.update({"target_condition": target_condition, "criterion": criterion})
    match_path.write_text(dumps_strict(found, indent=2))
    return found


def run_condition(
    args: argparse.Namespace,
    seed: int,
    condition: str,
    bench: SequentialBenchmark,
    factory,
    root: Path,
    schedule_seed: Optional[int] = None,
    targets: Optional[dict[str, dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Run or resume one (seed, condition); refuse to resume a mismatched run.

    replay_gates replays the same seed's surprise gates; replay_gates_cross
    replays those recorded under `schedule_seed`. Matched conditions need the
    target condition's outcome in `targets`.
    """
    mode, overrides = CONDITIONS[condition]
    settings = run_settings(args)
    match = None
    if condition in MATCHES:
        target_condition = MATCHES[condition][0]
        match = _match_scale(
            args,
            seed,
            condition,
            bench,
            factory,
            root,
            (targets or {})[target_condition],
            settings,
        )
        overrides = dict(overrides, post_warmup_scale=match["scale"])
    config = gated_config(args, **overrides)
    trainer = GatedSequentialTrainer(
        factory, bench, config, verbose=False, run_settings=settings
    )
    tasks = [bench.get_task_name(i) for i in range(len(bench))]
    expected = trainer.fingerprint(seed, tasks)
    expected_theta0 = _theta0_hash(factory, seed)
    run_dir = root / f"seed_{seed}" / condition
    record_path = run_dir / "run_record.json"
    if record_path.exists():
        record = json.loads(record_path.read_text())
        if (
            record.get("contract_fingerprint") != expected
            or record.get("theta0_hash") != expected_theta0
            or record.get("config") != json.loads(json.dumps(asdict(config)))
            or record.get("mode") != mode
        ):
            raise ValueError(
                f"{run_dir} was produced under a different config fingerprint"
            )
        matrix = json.loads((run_dir / "forgetting_results.json").read_text())
        outcome = {
            "record": record,
            "metrics": run_metrics(matrix["performance_matrix"], tasks, record),
        }
    else:
        schedule = None
        if mode == "replay_gates":
            source = schedule_seed if condition == "replay_gates_cross" else seed
            if source is None:
                raise ValueError("replay_gates_cross needs a schedule_seed")
            schedule_path = root / f"seed_{source}" / "surprise" / "gate_schedule.json"
            if not schedule_path.exists():
                raise FileNotFoundError(f"missing gate schedule {schedule_path}")
            schedule = GateSchedule.load(schedule_path)
        result = trainer.run(str(run_dir), seed, mode, replay_schedule=schedule)
        if result.theta0_hash != expected_theta0:
            raise ValueError("theta_0 differs from the seed's initial parameters")
        metrics = run_metrics(result.performance_matrix, tasks, result.run_record)
        (run_dir / "metrics.json").write_text(dumps_strict(metrics, indent=2))
        outcome = {"record": result.run_record, "metrics": metrics}
    if match is not None:
        _, criterion = MATCHES[condition]
        ratio = _target_value(outcome, criterion) / _target_value(
            targets[MATCHES[condition][0]], criterion
        )
        outcome["match"] = {
            "scale": match["scale"],
            "criterion": criterion,
            "ratio": ratio,
            "within_tolerance": abs(ratio - 1.0) <= MATCH_TOLERANCE,
        }
        if not outcome["match"]["within_tolerance"]:
            print(
                f"WARNING seed {seed} {condition}: {criterion} ratio "
                f"{ratio:.3f} outside +-{MATCH_TOLERANCE:.0%}"
            )
    return outcome


def freeze_config(args: argparse.Namespace, root: Path) -> None:
    """Write the frozen hyper-parameters; refuse to continue under changed ones."""
    frozen = {
        key: value
        for key, value in vars(args).items()
        if key not in {"command", "conditions", "criterion", "steps_per_task"}
    }
    path = root / "frozen_config.json"
    root.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != json.loads(json.dumps(frozen)):
            raise ValueError(f"{path} exists with different hyper-parameters")
        return
    path.write_text(json.dumps(frozen, indent=2, sort_keys=True))


def _require_match_seed(args: argparse.Namespace) -> int:
    if args.match_seed is None:
        raise ValueError("--match-seed is required")
    assert args.match_seed not in args.seeds, "--match-seed must not be in --seeds"
    return args.match_seed


def command_compare(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.output_dir)
    freeze_config(args, root)
    requested = list(args.conditions)
    needed = {MATCHES[c][0] for c in requested if c in MATCHES}
    leading = ("surprise", "influence")
    conditions = [c for c in leading if c in requested or c in needed] + [
        c for c in requested if c not in leading
    ]
    per_run: dict[str, dict[int, dict[str, Any]]] = {c: {} for c in conditions}
    first_record: dict[int, dict[str, Any]] = {}
    matches: dict[str, dict[int, dict[str, Any]]] = {}
    setups = {}
    seeds = list(args.seeds)
    for seed in seeds:
        bench = build_benchmark(args, seed)
        setups[seed] = (bench, build_factory(args, bench.vocab_size))
    outcomes: dict[int, dict[str, dict[str, Any]]] = {seed: {} for seed in seeds}
    for condition in conditions:
        for index, seed in enumerate(seeds):
            bench, factory = setups[seed]
            outcome = run_condition(
                args,
                seed,
                condition,
                bench,
                factory,
                root,
                schedule_seed=seeds[(index + 1) % len(seeds)],
                targets=outcomes[seed],
            )
            outcomes[seed][condition] = outcome
            per_run[condition][seed] = outcome["metrics"]
            first = first_record.setdefault(seed, outcome["record"])
            assert_records_comparable([first, outcome["record"]])
            if "match" in outcome:
                matches.setdefault(condition, {})[seed] = outcome["match"]
            print(
                f"seed {seed} {condition}: {outcome['metrics']['average_forgetting']:.4f}"
            )
    summary = {
        "conditions": conditions,
        "seeds": seeds,
        "per_run": {
            c: {str(s): m for s, m in runs.items()} for c, runs in per_run.items()
        },
        "matches": {c: {str(s): m for s, m in v.items()} for c, v in matches.items()},
        "contrast_note": CONTRAST_NOTE,
        "primary_contrast": "surprise - none on average_forgetting",
        "contrasts": summarize_contrasts(per_run),
        "commands": [sys.argv],
    }
    (root / "comparison.json").write_text(dumps_strict(summary, indent=2))
    return summary


def command_lr_match(args: argparse.Namespace) -> dict[str, Any]:
    """Pick a bracket centre for the per-seed matching on a dedicated seed."""
    root = Path(args.output_dir)
    seed = _require_match_seed(args)
    bench = build_benchmark(args, seed)
    factory = build_factory(args, bench.vocab_size)
    settings = run_settings(args)
    condition = CRITERION_OF[args.criterion]
    target_condition, criterion = MATCHES[condition]
    targets = {
        target_condition: run_condition(
            args, seed, target_condition, bench, factory, root
        )
    }
    outcome = _match_scale(
        args,
        seed,
        condition,
        bench,
        factory,
        root,
        targets[target_condition],
        settings,
    )
    outcome.update({"criterion_name": args.criterion, "base_lr": args.lr, "seed": seed})
    (root / LR_MATCH_FILES[args.criterion]).write_text(dumps_strict(outcome, indent=2))
    print(f"lr scale {outcome['scale']:.4f} (rel err {outcome['relative_error']:.3%})")
    return outcome


def command_commutator(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    seed = args.seeds[0]
    if "lr_matched_learning" in args.conditions:
        raise ValueError("lr_matched_learning is not supported by the commutator")
    requested = list(args.conditions)
    needed = {MATCHES[c][0] for c in requested if c in MATCHES}
    leading = ("surprise", "influence")
    conditions = [c for c in leading if c in requested or c in needed] + [
        c for c in requested if c not in leading
    ]
    match_seed = _require_match_seed(args) if needed else None
    bench = build_benchmark(args, seed)
    factory = build_factory(args, bench.vocab_size)
    all_tasks = [bench.get_task_name(i) for i in range(len(bench))]
    tasks = all_tasks[:2]
    order_ab, order_ba = tuple(tasks), tuple(reversed(tasks))
    model_key = jax.random.split(jax.random.PRNGKey(seed), 3)[0]
    theta_init, structure = factory(model_key)
    batch_fn = make_batch_fn(bench, tasks)
    sizes = num_sequences_of(bench, tasks)

    def streams_for(stream_seed: int):
        return precompute_streams(
            stream_seed,
            tasks,
            args.steps_per_task,
            batch_size=args.batch_size,
            num_sequences=sizes,
            probe_batches=args.probe_batches,
        )

    streams, alt_streams = streams_for(seed), streams_for(seed + 1000)
    probe = {
        name: take_batches(bench.get_test_loader(all_tasks.index(name)), 1)[0]
        for name in tasks
    }
    eval_jit = jax.jit(
        lambda p, b: _eval_step_autoregressive(
            p, structure, b, jax.random.PRNGKey(0), True
        )
    )
    base_config = gated_config(args, warmup_steps=0)
    engine = GateEngine(structure, optax.sgd(base_config.lr), base_config)
    theta_0 = shared_warmup(
        theta_init, tasks, streams, args.warmup_steps, batch_fn, engine, base_config
    )
    calib = [
        batch_fn(name, idx)
        for name in tasks
        for idx in list(streams.probe_indices[name])
        + list(streams.indices[name][: args.calib_batches])
    ]
    tau = engine.calibrate(theta_0, calib, jax.random.PRNGKey(seed + 7))
    identity = data_manifest(bench)
    settings = run_settings(args) | {
        "warmup_steps": args.warmup_steps,
        "steps_per_task": args.steps_per_task,
    }

    def common(config: GatedConfig, **extra: Any) -> dict[str, Any]:
        return dict(
            structure=structure,
            config=config,
            batch_fn=batch_fn,
            tau=tau,
            engine=engine,
            data_identity=identity,
            settings=settings,
            **extra,
        )

    def total_path(mode: str, config: GatedConfig, stream: Any) -> float:
        return sum(
            run_order(theta_0, order, stream, mode, **common(config))[1]
            for order in (order_ab, order_ba)
        )

    schedules: dict[str, GateSchedule] = {}
    alt_schedules: dict[str, GateSchedule] = {}
    if "replay_gates_cross" in conditions:
        for order in (order_ab, order_ba):
            run_order(
                theta_0,
                order,
                alt_streams,
                "surprise",
                recorder_out=alt_schedules,
                **common(base_config),
            )
    match_streams = streams_for(match_seed) if needed else None
    match_targets: dict[str, float] = {}
    realized: dict[str, float] = {}
    results: dict[str, Any] = {
        "tasks": tasks,
        "theta0_hash": params_hash(theta_0),
        "tau": tau,
        "steps_per_task": args.steps_per_task,
        "warmup_steps": args.warmup_steps,
        "match_seed": match_seed,
        "commands": [sys.argv],
        "conditions": {},
    }
    for condition in conditions:
        mode, overrides = CONDITIONS[condition]
        match = None
        if condition in MATCHES:
            target_condition = MATCHES[condition][0]
            if target_condition not in match_targets:
                target_mode, target_over = CONDITIONS[target_condition]
                match_targets[target_condition] = total_path(
                    target_mode,
                    gated_config(args, warmup_steps=0, **target_over),
                    match_streams,
                )
            match = match_lr_scale(
                lambda scale: total_path(
                    "none",
                    gated_config(args, warmup_steps=0, post_warmup_scale=scale),
                    match_streams,
                ),
                match_targets[target_condition],
                tolerance=0.01,
            )
            overrides = dict(overrides, post_warmup_scale=match["scale"])
        config = gated_config(args, warmup_steps=0, **overrides)
        meta_ab: dict[str, Any] = {}
        meta_ba: dict[str, Any] = {}
        recorded: dict[str, GateSchedule] = {}
        if mode == "replay_gates":
            source = alt_schedules if condition == "replay_gates_cross" else schedules
            run_ab = common(config, replay_schedule=source["+".join(order_ab)])
            run_ba = common(config, replay_schedule=source["+".join(order_ba)])
        else:
            run_ab = run_ba = common(config)
        theta_ab, path_ab = run_order(
            theta_0,
            order_ab,
            streams,
            mode,
            meta=meta_ab,
            recorder_out=recorded,
            **run_ab,
        )
        theta_ba, path_ba = run_order(
            theta_0,
            order_ba,
            streams,
            mode,
            meta=meta_ba,
            recorder_out=recorded,
            **run_ba,
        )
        if condition == "surprise":
            schedules.update(recorded)
        realized[condition] = path_ab + path_ba
        report = commutator_report(
            theta_ab,
            theta_ba,
            path_ab,
            path_ba,
            probe,
            lambda t, b: eval_jit(t, b),
            theta_0=theta_0,
            meta_ab=meta_ab,
            meta_ba=meta_ba,
        )
        floor = noise_floor(theta_0, order_ab, streams, alt_streams, mode, **run_ab)
        report["noise_floor"] = floor
        report["distance_over_noise"] = report["normalized_distance"] / max(
            floor["normalized_distance"], 1e-12
        )
        report["meta"] = {"ab": meta_ab, "ba": meta_ba}
        if match is not None:
            ratio = realized[condition] / realized[MATCHES[condition][0]]
            report["match"] = {
                "scale": match["scale"],
                "search": match["search"],
                "path_ratio_on_match_seed": match["value"] / match["target"],
                "path_ratio": ratio,
                "within_tolerance": abs(ratio - 1.0) <= MATCH_TOLERANCE,
            }
            if not report["match"]["within_tolerance"]:
                print(f"WARNING {condition}: path ratio {ratio:.3f} outside tolerance")
        results["conditions"][condition] = report
        print(
            f"{condition}: normalized distance {report['normalized_distance']:.4f}, "
            f"noise {floor['normalized_distance']:.4f}"
        )
    (root / "commutator_results.json").write_text(dumps_strict(results, indent=2))
    return results


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    {
        "compare": command_compare,
        "lr-match": command_lr_match,
        "commutator": command_commutator,
    }[args.command](args)


if __name__ == "__main__":
    main()
