"""Sequential training and checkpointing for the forgetting baseline."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np

from fabricpc.core.types import GraphParams, GraphStructure, NodeParams
from fabricpc.training.train_autoregressive import (
    evaluate_autoregressive,
    train_autoregressive,
)


def save_checkpoint(
    params: GraphParams,
    task_idx: int | str,
    output_dir: Path,
    metadata: Optional[dict[str, Any]] = None,
) -> Path:
    """Save model parameters after a training stage."""
    stem = task_idx if isinstance(task_idx, str) else f"task_{task_idx}"
    checkpoint_dir = output_dir / f"{stem}_checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for node_name, node_params in params.nodes.items():
        for name, value in node_params.weights.items():
            arrays[f"{node_name}/weights/{name}"] = np.asarray(value)
        for name, value in node_params.biases.items():
            arrays[f"{node_name}/biases/{name}"] = np.asarray(value)
    np.savez_compressed(checkpoint_dir / "params.npz", **arrays)
    if metadata is not None:
        (checkpoint_dir / "metadata.json").write_text(
            dumps_strict(metadata, indent=2), encoding="utf-8"
        )
    return checkpoint_dir


def load_checkpoint(checkpoint_dir: Path, structure: GraphStructure) -> GraphParams:
    """Reconstruct model parameters saved by :func:`save_checkpoint`."""
    archive = np.load(checkpoint_dir / "params.npz")
    nodes = {}
    for node_name in structure.nodes:
        weights = {
            key.split("/")[2]: jnp.asarray(archive[key])
            for key in archive.files
            if key.startswith(f"{node_name}/weights/")
        }
        biases = {
            key.split("/")[2]: jnp.asarray(archive[key])
            for key in archive.files
            if key.startswith(f"{node_name}/biases/")
        }
        nodes[node_name] = NodeParams(weights=weights, biases=biases)
    return GraphParams(nodes=nodes)


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def dumps_strict(obj: Any, **kwargs: Any) -> str:
    """JSON text with non-finite floats mapped to null (never NaN/Infinity)."""
    return json.dumps(_sanitize(obj), allow_nan=False, default=str, **kwargs)


def params_hash(params: GraphParams) -> str:
    """Return a SHA-256 digest over every parameter leaf, name and value."""
    digest = hashlib.sha256()
    for node_name in sorted(params.nodes):
        node = params.nodes[node_name]
        for group, entries in (("weights", node.weights), ("biases", node.biases)):
            for name in sorted(entries):
                value = np.ascontiguousarray(np.asarray(entries[name]))
                digest.update(f"{node_name}/{group}/{name}/{value.dtype}".encode())
                digest.update(str(value.shape).encode())
                digest.update(value.tobytes())
    return digest.hexdigest()


def save_results_matrix(results: dict[str, Any], output_dir: Path) -> None:
    """Write the triangular per-stage evaluation matrix as JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "forgetting_results.json").write_text(
        dumps_strict(results, indent=2), encoding="utf-8"
    )


@dataclass
class TaskResult:
    task_idx: int
    task_name: str
    train_loss: float
    train_perplexity: float
    eval_results: dict[str, dict[str, float]]


@dataclass
class SequentialResult:
    task_results: list[TaskResult]
    performance_matrix: dict[str, dict[str, dict[str, float]]]
    checkpoint_paths: list[Path]
    total_time: float
    final_params: GraphParams
    structure: GraphStructure
    theta0_hash: str = ""
    num_updates: int = 0
    num_tokens: int = 0
    run_record: Optional[dict[str, Any]] = None

    @property
    def num_tasks(self) -> int:
        return len(self.task_results)


class _CountingLoader:
    """Loader proxy that remembers the token count of the last yielded batch."""

    def __init__(self, loader) -> None:
        self.loader = loader
        self.last_tokens = 0

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self):
        for batch in self.loader:
            self.last_tokens = int(np.asarray(batch[0]).size)
            yield batch


class SequentialTrainer:
    """Train tasks in order, retaining model parameters between stages."""

    def __init__(
        self,
        model_factory: Callable,
        benchmark,
        optimizer,
        config: dict[str, Any],
        eval_config: Optional[dict[str, Any]] = None,
        verbose: bool = True,
    ) -> None:
        self.model_factory = model_factory
        self.benchmark = benchmark
        self.optimizer = optimizer
        self.config = config
        self.eval_config = eval_config or config
        self.verbose = verbose

    def _evaluate(
        self, params: GraphParams, structure: GraphStructure, task_idx: int, key
    ):
        return evaluate_autoregressive(
            params,
            structure,
            self.benchmark.get_test_loader(task_idx),
            self.eval_config,
            key,
        )

    def run(
        self, output_dir: str = "results/forgetting_benchmark", seed: int = 42
    ) -> SequentialResult:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        model_key, train_key, eval_key = jax.random.split(jax.random.PRNGKey(seed), 3)
        params, structure = self.model_factory(model_key)
        theta0_hash = params_hash(params)
        save_checkpoint(
            params, "theta_0", output_path, metadata={"theta0_hash": theta0_hash}
        )
        num_updates = 0
        num_tokens = 0
        task_results: list[TaskResult] = []
        performance_matrix: dict[str, dict[str, dict[str, float]]] = {}
        checkpoint_paths: list[Path] = []
        started = time.time()

        for task_idx in range(len(self.benchmark)):
            task_name = self.benchmark.get_task_name(task_idx)
            train_key, stage_key = jax.random.split(train_key)
            loader = _CountingLoader(self.benchmark.get_train_loader(task_idx))

            def count_update(_epoch, _batch, energy, loader=loader):
                nonlocal num_updates, num_tokens
                num_updates += 1
                num_tokens += loader.last_tokens
                return energy

            params, _, _ = train_autoregressive(
                params,
                structure,
                loader,
                self.optimizer,
                self.config,
                stage_key,
                verbose=self.verbose,
                iter_callback=count_update,
            )

            stage_metrics: dict[str, dict[str, float]] = {}
            for eval_idx in range(task_idx + 1):
                eval_key, task_key = jax.random.split(eval_key)
                metrics = self._evaluate(params, structure, eval_idx, task_key)
                stage_metrics[self.benchmark.get_task_name(eval_idx)] = {
                    name: float(metrics[name])
                    for name in ("loss", "perplexity", "accuracy")
                }
            performance_matrix[task_name] = stage_metrics
            current = stage_metrics[task_name]
            checkpoint_paths.append(
                save_checkpoint(
                    params,
                    task_idx,
                    output_path,
                    metadata={"task": task_name, **current},
                )
            )
            task_results.append(
                TaskResult(
                    task_idx=task_idx,
                    task_name=task_name,
                    train_loss=current["loss"],
                    train_perplexity=current["perplexity"],
                    eval_results=stage_metrics,
                )
            )

        result = SequentialResult(
            task_results=task_results,
            performance_matrix=performance_matrix,
            checkpoint_paths=checkpoint_paths,
            total_time=time.time() - started,
            final_params=params,
            structure=structure,
            theta0_hash=theta0_hash,
            num_updates=num_updates,
            num_tokens=num_tokens,
        )
        save_results_matrix(
            {
                "tasks": [
                    self.benchmark.get_task_name(index)
                    for index in range(len(self.benchmark))
                ],
                "performance_matrix": performance_matrix,
                "total_time": result.total_time,
                "theta0_hash": theta0_hash,
                "num_updates": num_updates,
                "num_tokens": num_tokens,
            },
            output_path,
        )
        return result
