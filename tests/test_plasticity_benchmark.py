"""Tests for the plain-PC and gated sequential character benchmarks."""

import json
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


# --- gated sequential trainer --------------------------------------------------

import jax.numpy as jnp
import pytest

from fabricpc.experiments.forgetting_benchmark.gated_trainer import (
    GatedConfig,
    GatedSequentialTrainer,
    assert_records_comparable,
)


class OneTaskBenchmark(ShiftBenchmark):
    tasks = ["A"]

    def get_train_loader(self, index):
        return ShiftLoader(1, steps=5)


def _leaf_equal(a, b) -> bool:
    return all(
        np.array_equal(x, y)
        for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b))
    )


def test_none_mode_matches_ported_plain_trainer_parameters(tmp_path):
    params, structure = _model()
    factory = lambda _key: (params, structure)
    plain = SequentialTrainer(
        model_factory=factory,
        benchmark=OneTaskBenchmark(),
        optimizer=optax.sgd(0.05),
        config={"num_epochs": 1, "use_causal_mask": True, "force_single_device": True},
        eval_config=EVAL_CONFIG,
        verbose=False,
    ).run(str(tmp_path / "plain"), seed=4)
    gated = GatedSequentialTrainer(
        model_factory=factory,
        benchmark=OneTaskBenchmark(),
        config=GatedConfig(lr=0.05, num_epochs=1, diag_every=2, calib_batches=2),
        verbose=False,
    ).run(str(tmp_path / "gated"), seed=4, mode="none")

    assert gated.num_updates == plain.num_updates == 5
    assert gated.num_tokens == plain.num_tokens
    assert gated.theta0_hash == plain.theta0_hash
    assert _leaf_equal(gated.final_params, plain.final_params)


def _record(**overrides):
    record = {
        "contract_fingerprint": "f",
        "theta0_hash": "t",
        "num_tokens": 100,
        "num_updates": 5,
    }
    record.update(overrides)
    return record


def test_run_record_rejects_mismatched_fingerprint_or_token_count():
    assert_records_comparable([_record(), _record()])
    for key, value in (
        ("contract_fingerprint", "other"),
        ("num_tokens", 99),
        ("theta0_hash", "other"),
        ("num_updates", 6),
    ):
        with pytest.raises(ValueError):
            assert_records_comparable([_record(), _record(**{key: value})])


def test_surprise_two_task_smoke_learns_both_and_logs_variable_gates(tmp_path):
    params, structure = _model()
    trainer = GatedSequentialTrainer(
        model_factory=lambda _key: (params, structure),
        benchmark=ShiftBenchmark(),
        config=GatedConfig(lr=0.2, num_epochs=2, diag_every=4, calib_batches=3),
        verbose=False,
    )
    result = trainer.run(str(tmp_path), seed=3, mode="surprise")

    theta_0 = load_checkpoint(tmp_path / "theta_0_checkpoint", structure)
    bench = ShiftBenchmark()
    for idx, name in enumerate(bench.tasks):
        before = float(
            evaluate_autoregressive(
                theta_0,
                structure,
                bench.get_test_loader(idx),
                EVAL_CONFIG,
                jax.random.PRNGKey(0),
            )["loss"]
        )
        assert result.performance_matrix[name][name]["loss"] < before

    gates = jnp.asarray([[v for v in e["gate"].values()] for e in trainer.step_log])
    assert float(gates.std(axis=1).max()) > 0.0
    assert float(gates.std(axis=0).max()) > 0.0
    assert all(v >= 0 for e in trainer.step_log for v in e["update_norm"].values())
    assert result.run_record["num_updates"] == 2 * STEPS * 2
    assert (tmp_path / "run_record.json").exists()


def _influence_run(tmp_path, mode, probe_source="past", seed=3):
    params, structure = _model()
    trainer = GatedSequentialTrainer(
        model_factory=lambda _key: (params, structure),
        benchmark=ShiftBenchmark(),
        config=GatedConfig(
            lr=0.2,
            num_epochs=2,
            diag_every=4,
            calib_batches=3,
            probe_batches=2,
            probe_source=probe_source,
        ),
        verbose=False,
    )
    result = trainer.run(str(tmp_path), seed=seed, mode=mode)
    return trainer, result


def test_influence_mode_is_neutral_at_task_zero_and_logs_diagnostics(tmp_path):
    trainer, result = _influence_run(tmp_path, "influence")

    first_task = [d for d in trainer.diag_log if d["task"] == 0]
    second_task = [d for d in trainer.diag_log if d["task"] == 1]
    assert first_task and all(d.get("skipped") == "empty probe" for d in first_task)
    measured = [d for d in second_task if "influence" in d]
    assert measured
    assert {"probe_loss", "surprise", "multiplier", "first_order"} <= set(measured[0])
    surprise, _ = _influence_run(tmp_path / "surprise", "surprise")
    for got, want in zip(trainer.step_log, surprise.step_log):
        if got["task"] == 0:
            assert got["gate"] == want["gate"]
    assert np.isfinite(result.performance_matrix["B"]["B"]["loss"])


def test_diagnostics_do_not_change_none_mode_training(tmp_path):
    params, structure = _model()

    def run(diag_every, sub):
        return GatedSequentialTrainer(
            model_factory=lambda _key: (params, structure),
            benchmark=ShiftBenchmark(),
            config=GatedConfig(lr=0.1, num_epochs=1, diag_every=diag_every),
            verbose=False,
        ).run(str(tmp_path / sub), seed=2, mode="none")

    assert _leaf_equal(run(3, "a").final_params, run(0, "b").final_params)


def test_strict_json_maps_nonfinite_to_null():
    from fabricpc.experiments.forgetting_benchmark.sequential_trainer import (
        dumps_strict,
    )

    text = dumps_strict(
        {"a": float("nan"), "b": [float("inf"), 1.5], "c": np.float32(2)}
    )

    assert json.loads(text) == {"a": None, "b": [None, 1.5], "c": 2.0}


def test_constant_post_warmup_scale_follows_the_surprise_schedule_shape(tmp_path):
    params, structure = _model()
    trainer = GatedSequentialTrainer(
        model_factory=lambda _key: (params, structure),
        benchmark=ShiftBenchmark(),
        config=GatedConfig(
            lr=0.1,
            num_epochs=1,
            diag_every=0,
            calib_batches=2,
            warmup_steps=5,
            post_warmup_scale=0.5,
        ),
        verbose=False,
        run_settings={"seq_len": SEQ},
    )
    result = trainer.run(str(tmp_path), seed=1, mode="none")

    gates = [set(e["gate"].values()) for e in trainer.step_log]
    assert all(g == {1.0} for g in gates[:5])
    assert all(g == {0.5} for g in gates[5:])
    assert len(result.run_record["immediate_train_loss"]) == 2


def test_fingerprint_depends_on_run_settings(tmp_path):
    params, structure = _model()

    def fingerprint(settings):
        return GatedSequentialTrainer(
            lambda _key: (params, structure),
            ShiftBenchmark(),
            GatedConfig(),
            run_settings=settings,
        ).fingerprint(0, ["A", "B"])

    assert fingerprint({"seq_len": 8}) == fingerprint({"seq_len": 8})
    assert fingerprint({"seq_len": 8}) != fingerprint({"seq_len": 16})
