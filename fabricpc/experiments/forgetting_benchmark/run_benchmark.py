"""Command-line runner for the character-level forgetting baseline."""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import jax
import numpy as np
import optax

try:
    import resource
except ImportError:  # pragma: no cover - Windows does not provide resource.
    resource = None

from fabricpc.core.inference import InferenceSGDNormClip
from fabricpc.experiments.forgetting_benchmark.analysis import (
    export_per_stage_performance_csv,
    export_results_json,
    generate_latex_table,
    generate_summary_report,
)
from fabricpc.experiments.forgetting_benchmark.benchmark import SequentialBenchmark
from fabricpc.experiments.forgetting_benchmark.forgetting_metrics import (
    ForgettingResult,
)
from fabricpc.experiments.forgetting_benchmark.sequential_trainer import (
    SequentialTrainer,
)
from fabricpc.graph_initialization import initialize_params
from fabricpc.nodes.transformer_v2 import create_deep_transformer


@dataclass(frozen=True)
class BenchmarkConfig:
    tasks: tuple[str, ...] = ("tiny_stories", "wikitext2", "tiny_shakespeare")
    depth: int = 1
    embed_dim: int = 32
    num_heads: int = 2
    mlp_dim: int = 64
    seq_len: int = 32
    batch_size: int = 8
    max_samples: Optional[int] = 500
    max_chars_per_task: Optional[int] = 100_000
    epochs_per_task: int = 2
    lr: float = 1e-3
    infer_steps: int = 5
    eta_infer: float = 0.05
    output_dir: str = "results/forgetting_baseline"
    seed: int = 42
    data_dir: str = "data/forgetting_benchmark"
    preprocessing_profile: str = "ascii_v1"
    optimizer: str = "adam"
    force_single_device: bool = False

    @classmethod
    def from_value(cls, value: "BenchmarkConfig | dict[str, Any]") -> "BenchmarkConfig":
        if isinstance(value, cls):
            return value
        unknown = set(value) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown benchmark config keys: {sorted(unknown)}")
        values = dict(value)
        if "tasks" in values:
            values["tasks"] = tuple(values["tasks"])
        return cls(**values)


def _model_factory(config: BenchmarkConfig, vocab_size: int) -> Callable:
    def create_model(key):
        structure = create_deep_transformer(
            depth=config.depth,
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            mlp_dim=config.mlp_dim,
            seq_len=config.seq_len,
            vocab_size=vocab_size,
            inference=InferenceSGDNormClip(
                eta_infer=config.eta_infer,
                infer_steps=config.infer_steps,
                max_norm=5.0,
            ),
        )
        return initialize_params(structure, key), structure

    return create_model


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _environment() -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            capture_output=True,
            check=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "jax_devices": [str(device) for device in jax.devices()],
        "git_revision": revision,
        "pid": os.getpid(),
    }


def _resource_report(start_usage, started: float) -> dict[str, float]:
    report = {"wall_time_seconds": time.perf_counter() - started}
    if resource is not None and start_usage is not None:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        max_rss = usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        report.update(
            {
                "cpu_user_seconds": usage.ru_utime - start_usage.ru_utime,
                "cpu_system_seconds": usage.ru_stime - start_usage.ru_stime,
                "max_rss_bytes": float(max_rss),
            }
        )
    return report


def run_full_benchmark(
    config: BenchmarkConfig | dict[str, Any],
    *,
    benchmark=None,
    model_factory: Optional[Callable] = None,
    generate_plots: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run A → B → C and save the metrics required for the baseline."""
    del generate_plots  # The baseline writes data artifacts, not diagnostic plots.
    config = BenchmarkConfig.from_value(config)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    start_usage = resource.getrusage(resource.RUSAGE_SELF) if resource else None
    record_path = output_dir / "experiment_record.json"
    record = {
        "objective": "Objective 2 / KR1 baseline forgetting benchmark",
        "status": "running",
        "created_at": _timestamp(),
        "config": asdict(config),
        "environment": _environment(),
        "optimizer_reset_each_task": True,
    }
    export_results_json(record, record_path)

    if benchmark is None:
        benchmark = SequentialBenchmark(
            list(config.tasks),
            seq_len=config.seq_len,
            batch_size=config.batch_size,
            data_dir=config.data_dir,
            seed=config.seed,
            max_samples_per_task=config.max_samples,
            max_chars_per_task=config.max_chars_per_task,
            preprocessing_profile=config.preprocessing_profile,
        )
    if model_factory is None:
        model_factory = _model_factory(config, benchmark.vocab_size)

    if config.optimizer not in ("adam", "sgd"):
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")
    optimizer = (optax.sgd if config.optimizer == "sgd" else optax.adam)(config.lr)
    train_config = {
        "num_epochs": config.epochs_per_task,
        "use_causal_mask": True,
        "force_single_device": config.force_single_device,
    }
    sequential = SequentialTrainer(
        model_factory=model_factory,
        benchmark=benchmark,
        optimizer=optimizer,
        config=train_config,
        verbose=verbose,
    ).run(output_dir=str(output_dir), seed=config.seed)

    task_names = [benchmark.get_task_name(index) for index in range(len(benchmark))]
    forgetting = ForgettingResult.from_performance_matrix(
        sequential.performance_matrix, task_names, metric="perplexity"
    )
    conclusion, next_diagnostic = forgetting.conclusion()
    np.save(output_dir / "forgetting_matrix.npy", forgetting.matrix)
    export_per_stage_performance_csv(
        sequential.performance_matrix, output_dir / "per_stage_performance.csv"
    )
    if hasattr(benchmark, "tokenizer"):
        benchmark.tokenizer.save(output_dir / "character_tokenizer.json")
    if hasattr(benchmark, "preprocessing_manifest"):
        export_results_json(
            benchmark.preprocessing_manifest, output_dir / "preprocessing_manifest.json"
        )

    summary = {
        "config": asdict(config),
        "num_tasks": len(benchmark),
        "tasks": task_names,
        "vocab_size": benchmark.vocab_size,
        "preprocessing": getattr(benchmark, "preprocessing_manifest", None),
        "total_time": sequential.total_time,
        "bwt": forgetting.bwt,
        "average_forgetting": forgetting.average_forgetting,
        "final_retention": forgetting.final_retention,
        "avg_final_ppl": forgetting.avg_final,
        "forgetting_scores": forgetting.forgetting_scores,
        "performance_matrix": sequential.performance_matrix,
        "conclusion": conclusion,
        "next_diagnostic": next_diagnostic,
    }
    export_results_json(summary, output_dir / "summary.json")
    (output_dir / "summary.txt").write_text(
        generate_summary_report(summary), encoding="utf-8"
    )
    (output_dir / "summary_table.tex").write_text(
        generate_latex_table(forgetting.matrix, task_names), encoding="utf-8"
    )
    record.update(
        {
            "status": "completed",
            "completed_at": _timestamp(),
            "resource_report": _resource_report(start_usage, started),
        }
    )
    export_results_json(record, record_path)
    export_results_json(record["resource_report"], output_dir / "resource_report.json")

    if verbose:
        print(forgetting.summary())
        print(f"Results saved to: {output_dir}")
    return {"sequential": sequential, "forgetting": forgetting, "summary": summary}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the baseline forgetting benchmark"
    )
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
    parser.add_argument("--epochs-per-task", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--infer-steps", type=int, default=5)
    parser.add_argument("--eta-infer", type=float, default=0.05)
    parser.add_argument("--output-dir", default="results/forgetting_baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", default="data/forgetting_benchmark")
    parser.add_argument("--preprocessing-profile", default="ascii_v1")
    parser.add_argument("--optimizer", choices=("adam", "sgd"), default="adam")
    parser.add_argument("--force-single-device", action="store_true")
    return parser.parse_args()


def main() -> None:
    run_full_benchmark(vars(parse_args()))


if __name__ == "__main__":
    main()
