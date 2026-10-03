"""Tests for the plain-PC and gated sequential character benchmarks."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np
import optax

from fabricpc.core.inference import InferenceSGDNormClip
from fabricpc.experiments.forgetting_benchmark.sequential_trainer import (
    SequentialTrainer,
    load_checkpoint,
    params_hash,
)
from fabricpc.graph_initialization import initialize_params
from fabricpc.nodes.transformer_v2 import create_deep_transformer
from fabricpc.training.train_autoregressive import evaluate_autoregressive

VOCAB = 16
SEQ = 8
BATCH = 8
STEPS = 12
EVAL_CONFIG = {"use_causal_mask": True, "force_single_device": True}


def _model(key=jax.random.PRNGKey(0)):
    structure = create_deep_transformer(
        depth=1,
        embed_dim=16,
        num_heads=2,
        mlp_dim=32,
        seq_len=SEQ,
        vocab_size=VOCAB,
        inference=InferenceSGDNormClip(eta_infer=0.05, infer_steps=2, max_norm=5.0),
    )
    return initialize_params(structure, key), structure


class ShiftLoader:
    """Next-token task y = (x + shift) mod VOCAB, with a fixed batch stream."""

    def __init__(self, shift: int, steps: int = STEPS, seed: int = 0) -> None:
        self.shift = shift
        self.steps = steps
        self.seed = seed
        self.num_sequences = steps * BATCH

    def __len__(self) -> int:
        return self.steps

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        for _ in range(self.steps):
            x = rng.integers(0, VOCAB, size=(BATCH, SEQ), dtype=np.int32)
            yield x, (x + self.shift) % VOCAB


class ShiftBenchmark:
    tasks = ["A", "B"]
    shifts = {"A": 1, "B": 5}

    def __len__(self) -> int:
        return len(self.tasks)

    def get_task_name(self, index: int) -> str:
        return self.tasks[index]

    def get_train_loader(self, index: int) -> ShiftLoader:
        return ShiftLoader(self.shifts[self.tasks[index]])

    def get_test_loader(self, index: int) -> ShiftLoader:
        return ShiftLoader(self.shifts[self.tasks[index]], steps=2, seed=99)


def _eval_loss(params, structure, task_idx: int) -> float:
    metrics = evaluate_autoregressive(
        params,
        structure,
        ShiftBenchmark().get_train_loader(task_idx),
        EVAL_CONFIG,
        jax.random.PRNGKey(7),
    )
    return float(metrics["loss"])


def test_two_task_synthetic_plain_pc_learns_both_tasks_from_one_checkpoint(tmp_path):
    params, structure = _model()
    trainer = SequentialTrainer(
        model_factory=lambda _key: (params, structure),
        benchmark=ShiftBenchmark(),
        optimizer=optax.sgd(0.05),
        config={"num_epochs": 2, "use_causal_mask": True, "force_single_device": True},
        eval_config=EVAL_CONFIG,
        verbose=False,
    )
    result = trainer.run(str(tmp_path), seed=3)

    theta_0 = load_checkpoint(tmp_path / "theta_0_checkpoint", structure)
    after_a = load_checkpoint(tmp_path / "task_0_checkpoint", structure)
    assert params_hash(theta_0) == params_hash(params) == result.theta0_hash

    assert _eval_loss(after_a, structure, 0) < _eval_loss(theta_0, structure, 0)
    assert _eval_loss(result.final_params, structure, 1) < _eval_loss(
        after_a, structure, 1
    )
    assert result.num_updates == 2 * STEPS * 2
    assert result.num_tokens == result.num_updates * BATCH * SEQ


def test_runner_accepts_sgd_and_single_device_config(tmp_path):
    from fabricpc.experiments.forgetting_benchmark.run_benchmark import (
        run_full_benchmark,
    )

    benchmark = ShiftBenchmark()
    benchmark.vocab_size = VOCAB
    result = run_full_benchmark(
        {
            "output_dir": str(tmp_path),
            "epochs_per_task": 1,
            "lr": 0.05,
            "optimizer": "sgd",
            "force_single_device": True,
            "seq_len": SEQ,
        },
        benchmark=benchmark,
        model_factory=lambda key: _model(key),
        verbose=False,
    )

    assert result["summary"]["config"]["optimizer"] == "sgd"
    assert result["sequential"].num_updates == 2 * STEPS
