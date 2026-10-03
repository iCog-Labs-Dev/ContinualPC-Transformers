"""Sequential PC training with node-local plasticity gates."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Literal, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax

from fabricpc.core.types import GraphParams, GraphStructure
from fabricpc.experiments.forgetting_benchmark.influence import (
    first_order_influence,
    influence_multipliers,
    make_probe_loss_fn,
    measure_update_influence,
    rank_agreement,
)
from fabricpc.experiments.forgetting_benchmark.sequential_trainer import (
    SequentialResult,
    TaskResult,
    params_hash,
    save_checkpoint,
    save_results_matrix,
)
from fabricpc.training.plasticity_gates import (
    GateFnState,
    GateSchedule,
    calibrate_tau,
    local_surprise,
    make_gate_state,
)
from fabricpc.training.train import _convert_batch
from fabricpc.training.train_autoregressive import (
    evaluate_autoregressive,
    gated_update_autoregressive,
    get_graph_param_gradient_autoregressive,
    train_step_autoregressive_gated,
    validate_gated_config,
)

Mode = Literal["none", "surprise", "influence", "replay_gates"]
MODES = ("none", "surprise", "influence", "replay_gates")
_GATE_MODE = {
    "none": "none",
    "surprise": "surprise",
    "influence": "influence",
    "replay_gates": "replay",
}
_CALIBRATION_STREAM = 1001
_DIAGNOSTIC_STREAM = 1002


@dataclass(frozen=True)
class GatedConfig:
    """Training and diagnostic settings; `lr` and `alpha` may differ per condition."""

    lr: float = 0.05
    num_epochs: float = 1
    use_causal_mask: bool = True
    diag_every: int = 50
    calib_batches: int = 8
    warmup_steps: int = 0
    alpha: float = 0.5
    probe_batches: int = 2
    probe_source: str = "past"

    def optimizer_cfg(self) -> dict[str, Any]:
        return {
            "name": "sgd",
            "momentum": 0.0,
            "weight_decay": 0.0,
            "gradient_accumulation_steps": 1,
            "pmap": False,
        }

    def contract(self) -> dict[str, Any]:
        """Settings that every condition of one comparison must share."""
        return {
            "num_epochs": self.num_epochs,
            "use_causal_mask": self.use_causal_mask,
            "diag_every": self.diag_every,
            "calib_batches": self.calib_batches,
            "warmup_steps": self.warmup_steps,
            "probe_batches": self.probe_batches,
        }


def _digest(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def contract_fingerprint(
    config: GatedConfig, seed: int, tasks: list[str], data: Any
) -> str:
    """Hash of everything conditions must share to be compared."""
    return _digest(
        {"config": config.contract(), "seed": seed, "tasks": tasks, "data": data}
    )


COMPARABLE_KEYS = ("contract_fingerprint", "theta0_hash", "num_tokens", "num_updates")


def assert_records_comparable(
    records: Iterable[dict[str, Any]], keys: Iterable[str] = COMPARABLE_KEYS
) -> None:
    """Raise if run records differ in fingerprint, theta_0, tokens or updates."""
    records = list(records)
    if not records:
        return
    for key in keys:
        values = {record.get(key) for record in records}
        if len(values) != 1:
            raise ValueError(f"runs are not comparable: mismatched {key}: {values}")


def _git_revision() -> Optional[str]:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            capture_output=True,
            check=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def stage_steps(
    loader: Any, key: jax.Array, num_epochs: float
) -> Iterator[tuple[dict[str, jnp.ndarray], jax.Array]]:
    """Yield (batch, key) pairs with the key schedule of train_autoregressive."""
    total_epochs = math.ceil(num_epochs)
    frac = num_epochs - math.floor(num_epochs)
    for epoch_idx in range(total_epochs):
        num_batches = len(loader)
        last = epoch_idx == total_epochs - 1
        max_batches = round(frac * num_batches) if last and frac > 0 else num_batches
        epoch_key, key = jax.random.split(key)
        batch_keys = jax.random.split(epoch_key, max_batches)
        for batch_idx, batch_data in enumerate(loader):
            if batch_idx >= max_batches:
                break
            yield _convert_batch(batch_data), batch_keys[batch_idx]


def take_batches(loader: Any, count: int, skip: int = 0) -> list[dict[str, Any]]:
    """The first `count` batches after `skip` from a freshly created loader."""
    batches = []
    for index, batch_data in enumerate(loader):
        if index < skip:
            continue
        if len(batches) >= count:
            break
        batches.append(_convert_batch(batch_data))
    return batches


@dataclass
class RunContext:
    """Mutable state shared by the training stages of one run."""

    mode: str
    config: GatedConfig
    calib_batches: list[dict[str, Any]]
    calib_key: jax.Array
    replay_schedule: Optional[GateSchedule] = None
    recorder: GateSchedule = field(default_factory=GateSchedule)
    tau: Optional[dict[str, float]] = None
    h: dict[str, float] = field(default_factory=dict)
    probe_batches: list[dict[str, Any]] = field(default_factory=list)
    diag_batch: Optional[dict[str, Any]] = None
    diag_key: Optional[jax.Array] = None
    step: int = 0
    num_updates: int = 0
    num_tokens: int = 0
    path_length: float = 0.0
    train_seconds: float = 0.0
    diagnostic_seconds: float = 0.0
    task_idx: int = 0
    step_log: list[dict[str, Any]] = field(default_factory=list)
    diag_log: list[dict[str, Any]] = field(default_factory=list)


class GateEngine:
    """Jitted gated step, calibration and diagnostics for one structure."""

    def __init__(
        self,
        structure: GraphStructure,
        optimizer: optax.GradientTransformation,
        config: GatedConfig,
    ) -> None:
        validate_gated_config(config.optimizer_cfg())
        self.structure = structure
        self.optimizer = optimizer
        self.config = config
        use_mask = config.use_causal_mask
        self._step = jax.jit(
            lambda p, o, b, k, g: train_step_autoregressive_gated(
                p, o, b, structure, optimizer, k, g, use_mask
            )
        )
        self._scores = jax.jit(
            lambda p, b, k: local_surprise(
                get_graph_param_gradient_autoregressive(p, b, structure, k, use_mask)[
                    3
                ],
                structure,
            )
        )
        self._candidate = jax.jit(
            lambda p, o, b, k, g: gated_update_autoregressive(
                p, o, b, structure, optimizer, k, g, use_mask
            )
        )
        self.probe_loss = make_probe_loss_fn(structure, use_mask, jax.random.PRNGKey(0))
        self.trainable = [
            name
            for name, node in structure.nodes.items()
            if node.node_info.in_degree > 0
        ]

    def calibrate(
        self, params: GraphParams, batches: list[dict[str, Any]], key: jax.Array
    ) -> dict[str, float]:
        """Per-node median surprise over training-only calibration batches."""
        keys = jax.random.split(key, len(batches))
        per_batch = [self._scores(params, b, k) for b, k in zip(batches, keys)]
        stacked = {
            name: np.asarray([float(s[name]) for s in per_batch])
            for name in self.trainable
        }
        return calibrate_tau(stacked)

    def gate_state(self, ctx: RunContext) -> GateFnState:
        tau = ctx.tau
        if tau is None:
            return make_gate_state("none", {n: 1.0 for n in self.trainable})
        mode = _GATE_MODE[ctx.mode]
        override = None
        if ctx.mode == "replay_gates":
            if ctx.replay_schedule is None or ctx.step not in ctx.replay_schedule:
                raise ValueError(f"no recorded gates for step {ctx.step}")
            override = ctx.replay_schedule.gates_at(ctx.step)
        return make_gate_state(
            mode, tau, h=ctx.h, alpha=ctx.config.alpha, override=override
        )

    def diagnose(self, params: GraphParams, opt_state: Any, ctx: RunContext) -> None:
        """Read-only influence measurement; never alters params or opt_state.

        The candidate is the surprise-gated update on the fixed diagnostic
        batch. Multipliers are cached for the influence condition only, but the
        measurement runs in every condition so overhead is comparable.
        """
        entry: dict[str, Any] = {"step": ctx.step, "task": ctx.task_idx}
        started = time.perf_counter()
        if ctx.tau is None or ctx.diag_batch is None:
            entry["skipped"] = "no calibration yet"
        elif not ctx.probe_batches:
            entry["skipped"] = "empty probe"
            ctx.h = {}
        else:
            state = make_gate_state("surprise", ctx.tau)
            updates, _, _, _, _, stats = self._candidate(
                params, opt_state, ctx.diag_batch, ctx.diag_key, state
            )
            probe = ctx.probe_batches
            base = float(np.mean([float(self.probe_loss(params, b)) for b in probe]))
            influence = measure_update_influence(
                params, updates, probe, self.probe_loss
            )
            first = first_order_influence(params, updates, probe, self.probe_loss)
            multipliers = influence_multipliers(influence)
            score = {n: float(v) for n, v in stats["score"].items()}
            rho_exact, flagged = rank_agreement(score, influence)
            rho_first, _ = rank_agreement(first, influence)
            ctx.h = multipliers if ctx.mode == "influence" else {}
            entry.update(
                {
                    "probe_loss": base,
                    "probe_size": len(probe),
                    "influence": influence,
                    "first_order": first,
                    "multiplier": multipliers,
                    "surprise": score,
                    "update_norm": {
                        n: float(v) for n, v in stats["update_norm"].items()
                    },
                    "rank_surprise_vs_influence": rho_exact,
                    "rank_first_order_vs_exact": rho_first,
                    "high_surprise_nonpositive_influence": flagged,
                }
            )
        entry["seconds"] = time.perf_counter() - started
        ctx.diag_log.append(entry)

    def train_stage(
        self,
        params: GraphParams,
        steps: Iterable[tuple[dict[str, Any], jax.Array]],
        ctx: RunContext,
    ) -> GraphParams:
        """Train one task with a fresh optimizer state."""
        cfg = self.config
        opt_state = self.optimizer.init(params)
        for index, (batch, key) in enumerate(steps):
            started = time.perf_counter()
            if ctx.tau is None and ctx.step >= cfg.warmup_steps:
                ctx.tau = self.calibrate(params, ctx.calib_batches, ctx.calib_key)
            if cfg.diag_every > 0 and index % cfg.diag_every == 0:
                self.diagnose(params, opt_state, ctx)
            diag_done = time.perf_counter()
            ctx.diagnostic_seconds += diag_done - started

            params, opt_state, energy, ce_loss, _, stats = self._step(
                params, opt_state, batch, key, self.gate_state(ctx)
            )
            stats = jax.device_get(stats)
            score = {n: float(v) for n, v in stats["score"].items()}
            gate = {n: float(v) for n, v in stats["gate"].items()}
            norm = {n: float(v) for n, v in stats["update_norm"].items()}
            if not all(np.isfinite(list(score.values()) + [float(energy)])):
                raise FloatingPointError(f"non-finite surprise at step {ctx.step}")
            ctx.recorder.record(ctx.step, gate)
            ctx.path_length += math.sqrt(sum(v * v for v in norm.values()))
            ctx.step_log.append(
                {
                    "step": ctx.step,
                    "task": ctx.task_idx,
                    "energy": float(energy),
                    "ce": float(ce_loss),
                    "score": score,
                    "gate": gate,
                    "update_norm": norm,
                }
            )
            ctx.num_updates += 1
            ctx.num_tokens += int(np.asarray(batch["x"]).size)
            ctx.step += 1
            ctx.train_seconds += time.perf_counter() - diag_done
        return params


class GatedSequentialTrainer:
    """Train tasks in order under one gating condition."""

    def __init__(
        self,
        model_factory: Callable,
        benchmark: Any,
        config: GatedConfig,
        verbose: bool = True,
    ) -> None:
        self.model_factory = model_factory
        self.benchmark = benchmark
        self.config = config
        self.verbose = verbose
        self.gate_schedule: Optional[GateSchedule] = None
        self.step_log: list[dict[str, Any]] = []
        self.diag_log: list[dict[str, Any]] = []

    def _data_manifest(self) -> Any:
        manifest = getattr(self.benchmark, "preprocessing_manifest", None)
        if not manifest:
            return None
        return {
            task: {"source": info.get("source"), "profile": info.get("profile")}
            for task, info in manifest.get("tasks", {}).items()
        }

    def run(
        self,
        output_dir: str,
        seed: int,
        mode: Mode,
        replay_schedule: Optional[GateSchedule] = None,
    ) -> SequentialResult:
        if mode not in MODES:
            raise ValueError(f"unknown mode '{mode}'; expected one of {MODES}")
        cfg = self.config
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        base_key = jax.random.PRNGKey(seed)
        model_key, train_key, eval_key = jax.random.split(base_key, 3)
        params, structure = self.model_factory(model_key)
        theta0_hash = params_hash(params)
        save_checkpoint(
            params, "theta_0", output_path, metadata={"theta0_hash": theta0_hash}
        )
        optimizer = optax.sgd(cfg.lr)
        engine = GateEngine(structure, optimizer, cfg)
        tasks = [self.benchmark.get_task_name(i) for i in range(len(self.benchmark))]

        calib_batches = take_batches(
            self.benchmark.get_train_loader(0), cfg.calib_batches
        )
        ctx = RunContext(
            mode=mode,
            config=cfg,
            calib_batches=calib_batches,
            calib_key=jax.random.fold_in(base_key, _CALIBRATION_STREAM),
            replay_schedule=replay_schedule,
        )
        eval_config = {
            "use_causal_mask": cfg.use_causal_mask,
            "force_single_device": True,
        }

        task_results: list[TaskResult] = []
        performance_matrix: dict[str, dict[str, dict[str, float]]] = {}
        checkpoint_paths: list[Path] = []
        started = time.time()

        for task_idx, task_name in enumerate(tasks):
            ctx.task_idx = task_idx
            train_key, stage_key = jax.random.split(train_key)
            loader = self.benchmark.get_train_loader(task_idx)
            ctx.diag_key = jax.random.fold_in(base_key, _DIAGNOSTIC_STREAM + task_idx)
            self._prepare_probe(ctx, task_idx)
            params = engine.train_stage(
                params, stage_steps(loader, stage_key, cfg.num_epochs), ctx
            )

            stage_metrics: dict[str, dict[str, float]] = {}
            for eval_idx in range(task_idx + 1):
                eval_key, task_key = jax.random.split(eval_key)
                metrics = evaluate_autoregressive(
                    params,
                    structure,
                    self.benchmark.get_test_loader(eval_idx),
                    eval_config,
                    task_key,
                )
                stage_metrics[tasks[eval_idx]] = {
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

        total_time = time.time() - started
        self.gate_schedule = ctx.recorder
        self.step_log = ctx.step_log
        self.diag_log = ctx.diag_log
        record = self._run_record(
            output_path, seed, mode, tasks, theta0_hash, ctx, total_time
        )
        result = SequentialResult(
            task_results=task_results,
            performance_matrix=performance_matrix,
            checkpoint_paths=checkpoint_paths,
            total_time=total_time,
            final_params=params,
            structure=structure,
            theta0_hash=theta0_hash,
            num_updates=ctx.num_updates,
            num_tokens=ctx.num_tokens,
            run_record=record,
        )
        self._write_artifacts(output_path, tasks, result, ctx, record)
        return result

    def _prepare_probe(self, ctx: RunContext, task_idx: int) -> None:
        """Select the diagnostic batch and probe set for the coming task.

        Probes come from training splits only; the past-task probe is empty at
        task 0, which leaves every multiplier at one.
        """
        cfg = self.config
        ctx.h = {}
        ctx.diag_batch = take_batches(self.benchmark.get_train_loader(task_idx), 1)[0]
        if cfg.probe_source == "current":
            ctx.probe_batches = take_batches(
                self.benchmark.get_train_loader(task_idx), cfg.probe_batches, skip=1
            )
        elif cfg.probe_source == "past":
            ctx.probe_batches = [
                batch
                for past in range(task_idx)
                for batch in take_batches(
                    self.benchmark.get_train_loader(past), cfg.probe_batches
                )
            ]
        else:
            raise ValueError(f"unknown probe_source '{cfg.probe_source}'")

    def _run_record(
        self,
        output_path: Path,
        seed: int,
        mode: str,
        tasks: list[str],
        theta0_hash: str,
        ctx: RunContext,
        total_time: float,
    ) -> dict[str, Any]:
        data = self._data_manifest()
        return {
            "git_revision": _git_revision(),
            "mode": mode,
            "seed": seed,
            "config": asdict(self.config),
            "tasks": tasks,
            "contract_fingerprint": contract_fingerprint(
                self.config, seed, tasks, data
            ),
            "data_hashes": data,
            "theta0_hash": theta0_hash,
            "device": [str(device) for device in jax.devices()],
            "elapsed_seconds": total_time,
            "train_seconds": ctx.train_seconds,
            "diagnostic_seconds": ctx.diagnostic_seconds,
            "num_updates": ctx.num_updates,
            "num_tokens": ctx.num_tokens,
            "path_length": ctx.path_length,
            "commands": [sys.argv],
            "status": "completed",
        }

    def _write_artifacts(
        self,
        output_path: Path,
        tasks: list[str],
        result: SequentialResult,
        ctx: RunContext,
        record: dict[str, Any],
    ) -> None:
        save_results_matrix(
            {
                "tasks": tasks,
                "performance_matrix": result.performance_matrix,
                "total_time": result.total_time,
                "theta0_hash": result.theta0_hash,
                "num_updates": result.num_updates,
                "num_tokens": result.num_tokens,
            },
            output_path,
        )
        (output_path / "run_record.json").write_text(
            json.dumps(record, indent=2, default=str), encoding="utf-8"
        )
        with open(output_path / "step_log.jsonl", "w", encoding="utf-8") as handle:
            for entry in ctx.step_log:
                handle.write(json.dumps(entry) + "\n")
        with open(output_path / "diagnostics.jsonl", "w", encoding="utf-8") as handle:
            for entry in ctx.diag_log:
                handle.write(json.dumps(entry) + "\n")
        ctx.recorder.save(output_path / "gate_schedule.json")
