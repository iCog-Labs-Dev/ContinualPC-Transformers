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
import numpy as np
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
)
from fabricpc.experiments.forgetting_benchmark.gated_trainer import (
    GateEngine,
    GatedConfig,
    GatedSequentialTrainer,
    assert_records_comparable,
    contract_fingerprint,
    take_batches,
)
from fabricpc.experiments.forgetting_benchmark.plasticity_analysis import (
    match_lr_scale,
    run_metrics,
    summarize_contrasts,
)
from fabricpc.experiments.forgetting_benchmark.run_benchmark import (
    BenchmarkConfig,
    _model_factory,
)
from fabricpc.experiments.forgetting_benchmark.sequential_trainer import params_hash
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
    "lr_matched_learning": ("none", {}),
}
DEFAULT_CONDITIONS = (
    "surprise",
    "none",
    "influence",
    "replay_gates",
    "replay_gates_cross",
    "lr_matched",
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
    "lr_matched": "lr_match_path.json",
    "lr_matched_learning": "lr_match_learning.json",
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
    parser.add_argument("--criterion", choices=("path", "learning"), default="path")
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


def condition_lr_scale(args: argparse.Namespace, condition: str) -> float:
    if condition not in LR_MATCH_FILES:
        return 1.0
    path = Path(args.output_dir) / LR_MATCH_FILES[condition]
    if not path.exists():
        raise FileNotFoundError(
            f"{condition} needs {path}; run the lr-match command on the smoke budget first"
        )
    return float(json.loads(path.read_text())["scale"])


def run_condition(
    args: argparse.Namespace,
    seed: int,
    condition: str,
    bench: SequentialBenchmark,
    factory,
    root: Path,
    schedule_seed: Optional[int] = None,
) -> dict[str, Any]:
    """Run or resume one (seed, condition); refuse to resume a mismatched run.

    replay_gates replays the same seed's surprise gates; replay_gates_cross
    replays those recorded under `schedule_seed`.
    """
    mode, overrides = CONDITIONS[condition]
    scale = condition_lr_scale(args, condition)
    config = gated_config(args, lr=args.lr * scale, **overrides)
    trainer = GatedSequentialTrainer(factory, bench, config, verbose=False)
    tasks = [bench.get_task_name(i) for i in range(len(bench))]
    expected = contract_fingerprint(config, seed, tasks, trainer._data_manifest())
    run_dir = root / f"seed_{seed}" / condition
    record_path = run_dir / "run_record.json"
    if record_path.exists():
        record = json.loads(record_path.read_text())
        if (
            record.get("contract_fingerprint") != expected
            or record.get("config") != json.loads(json.dumps(asdict(config)))
            or record.get("mode") != mode
        ):
            raise ValueError(
                f"{run_dir} was produced under a different config fingerprint"
            )
        matrix = json.loads((run_dir / "forgetting_results.json").read_text())
        return {
            "record": record,
            "metrics": run_metrics(matrix["performance_matrix"], tasks, record),
        }
    schedule = None
    if mode == "replay_gates":
        source = schedule_seed if condition == "replay_gates_cross" else seed
        if source is None:
            raise ValueError("replay_gates_cross needs a schedule_seed")
        schedule_path = root / f"seed_{source}" / "surprise" / "gate_schedule.json"
        if not schedule_path.exists():
            raise FileNotFoundError(f"missing surprise gate schedule {schedule_path}")
        schedule = GateSchedule.load(schedule_path)
    result = trainer.run(str(run_dir), seed, mode, replay_schedule=schedule)
    metrics = run_metrics(result.performance_matrix, tasks, result.run_record)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return {"record": result.run_record, "metrics": metrics}


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


def command_compare(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.output_dir)
    freeze_config(args, root)
    conditions = list(args.conditions)
    conditions.sort(key=lambda c: c != "surprise")
    per_run: dict[str, dict[int, dict[str, Any]]] = {c: {} for c in conditions}
    records: dict[int, list[dict[str, Any]]] = {seed: [] for seed in args.seeds}
    setups = {}
    for seed in args.seeds:
        bench = build_benchmark(args, seed)
        setups[seed] = (bench, build_factory(args, bench.vocab_size))
    seeds = list(args.seeds)
    for condition in conditions:
        for index, seed in enumerate(seeds):
            bench, factory = setups[seed]
            other = seeds[(index + 1) % len(seeds)]
            outcome = run_condition(
                args, seed, condition, bench, factory, root, schedule_seed=other
            )
            per_run[condition][seed] = outcome["metrics"]
            records[seed].append(outcome["record"])
            print(
                f"seed {seed} {condition}: {outcome['metrics']['average_forgetting']:.4f}"
            )
    for seed_records in records.values():
        assert_records_comparable(seed_records)
    summary = {
        "conditions": conditions,
        "seeds": list(args.seeds),
        "per_run": {
            c: {str(s): m for s, m in runs.items()} for c, runs in per_run.items()
        },
        "contrasts": summarize_contrasts(per_run),
        "commands": [sys.argv],
    }
    (root / "comparison.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def command_lr_match(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.output_dir)
    seed = args.seeds[0]
    bench = build_benchmark(args, seed)
    factory = build_factory(args, bench.vocab_size)
    target_run = run_condition(args, seed, "surprise", bench, factory, root)
    key = "path_length" if args.criterion == "path" else "new_task_log_ppl"
    target = float(target_run["metrics"][key])
    tasks = [bench.get_task_name(i) for i in range(len(bench))]

    def measure(scale: float) -> float:
        config = gated_config(args, lr=args.lr * scale)
        run_dir = root / "lr_match_search" / f"{args.criterion}_{scale:.6f}"
        result = GatedSequentialTrainer(factory, bench, config, verbose=False).run(
            str(run_dir), seed, "none"
        )
        return run_metrics(result.performance_matrix, tasks, result.run_record)[key]

    outcome = match_lr_scale(measure, target)
    outcome.update({"criterion": args.criterion, "base_lr": args.lr, "seed": seed})
    name = LR_MATCH_FILES[
        "lr_matched" if args.criterion == "path" else "lr_matched_learning"
    ]
    (root / name).write_text(json.dumps(outcome, indent=2))
    print(f"lr scale {outcome['scale']:.4f} (rel err {outcome['relative_error']:.3%})")
    return outcome


def command_commutator(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    seed = args.seeds[0]
    bench = build_benchmark(args, seed)
    factory = build_factory(args, bench.vocab_size)
    all_tasks = [bench.get_task_name(i) for i in range(len(bench))]
    tasks = all_tasks[:2]
    order_ab, order_ba = tuple(tasks), tuple(reversed(tasks))
    model_key = jax.random.split(jax.random.PRNGKey(seed), 3)[0]
    theta_0, structure = factory(model_key)
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
    base_engine = GateEngine(structure, optax.sgd(base_config.lr), base_config)
    calib = [
        batch
        for name in tasks
        for batch in [batch_fn(name, idx) for idx in streams.probe_indices[name]]
    ]
    calib += [
        batch_fn(name, idx)
        for name in tasks
        for idx in streams.indices[name][: args.calib_batches]
    ]
    tau = base_engine.calibrate(theta_0, calib, jax.random.PRNGKey(seed + 7))

    conditions = sorted(args.conditions, key=lambda c: c != "surprise")
    schedules: dict[str, GateSchedule] = {}
    alt_schedules: dict[str, GateSchedule] = {}
    if "replay_gates_cross" in conditions:
        for order in (order_ab, order_ba):
            run_order(
                theta_0,
                order,
                alt_streams,
                "surprise",
                structure=structure,
                config=base_config,
                batch_fn=batch_fn,
                tau=tau,
                engine=base_engine,
                recorder_out=alt_schedules,
            )
    results: dict[str, Any] = {
        "tasks": tasks,
        "theta0_hash": params_hash(theta_0),
        "tau": tau,
        "steps_per_task": args.steps_per_task,
        "commands": [sys.argv],
        "conditions": {},
    }
    for condition in conditions:
        mode, overrides = CONDITIONS[condition]
        scale = condition_lr_scale(args, condition)
        config = gated_config(args, warmup_steps=0, lr=args.lr * scale, **overrides)
        engine = GateEngine(structure, optax.sgd(config.lr), config)
        kwargs: dict[str, Any] = dict(
            structure=structure,
            config=config,
            batch_fn=batch_fn,
            tau=tau,
            engine=engine,
        )
        meta_ab: dict[str, Any] = {}
        meta_ba: dict[str, Any] = {}
        recorded: dict[str, GateSchedule] = {}
        run_kwargs = dict(kwargs)
        if mode == "replay_gates":
            source = alt_schedules if condition == "replay_gates_cross" else schedules
            run_ab = dict(run_kwargs, replay_schedule=source["+".join(order_ab)])
            run_ba = dict(run_kwargs, replay_schedule=source["+".join(order_ba)])
        else:
            run_ab = run_ba = run_kwargs
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
        results["conditions"][condition] = report
        print(
            f"{condition}: normalized distance {report['normalized_distance']:.4f}, "
            f"noise {floor['normalized_distance']:.4f}"
        )
    (root / "commutator_results.json").write_text(
        json.dumps(results, indent=2, default=str)
    )
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
