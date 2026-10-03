"""Tests for sequential training and checkpoints."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np
import optax

from fabricpc.core.inference import InferenceSGDNormClip
from fabricpc.experiments.forgetting_benchmark.sequential_trainer import (
    SequentialTrainer,
    load_checkpoint,
    save_checkpoint,
    save_results_matrix,
)
from fabricpc.graph_initialization import initialize_params
from fabricpc.nodes.transformer_v2 import create_deep_transformer


def _model():
    structure = create_deep_transformer(
        depth=1,
        embed_dim=16,
        num_heads=2,
        mlp_dim=32,
        seq_len=8,
        vocab_size=20,
        inference=InferenceSGDNormClip(eta_infer=0.05, infer_steps=1, max_norm=5.0),
    )
    return initialize_params(structure, jax.random.PRNGKey(0)), structure


def _loader():
    class Loader:
        num_sequences = 4

        def __len__(self):
            return 1

        def __iter__(self):
            rng = np.random.default_rng(42)
            x = rng.integers(0, 20, size=(4, 8), dtype=np.int32)
            yield x, np.roll(x, -1, axis=1)

    return Loader()


def test_checkpoint_roundtrip(tmp_path):
    params, structure = _model()
    checkpoint = save_checkpoint(params, 0, tmp_path, metadata={"task": "A"})
    loaded = load_checkpoint(checkpoint, structure)

    assert (checkpoint / "metadata.json").exists()
    for node_name, node in params.nodes.items():
        for name, value in node.weights.items():
            np.testing.assert_array_equal(loaded.nodes[node_name].weights[name], value)


def test_sequential_training_records_all_observed_tasks(tmp_path):
    class Benchmark:
        tasks = ["A", "B"]

        def __len__(self):
            return len(self.tasks)

        def get_task_name(self, index):
            return self.tasks[index]

        def get_train_loader(self, _index):
            return _loader()

        def get_test_loader(self, _index):
            return _loader()

    params, structure = _model()
    trainer = SequentialTrainer(
        model_factory=lambda _key: (params, structure),
        benchmark=Benchmark(),
        optimizer=optax.adam(1e-3),
        config={"num_epochs": 1, "use_causal_mask": True},
        verbose=False,
    )
    result = trainer.run(str(tmp_path), seed=42)

    assert result.num_tasks == 2
    assert list(result.performance_matrix["A"]) == ["A"]
    assert list(result.performance_matrix["B"]) == ["A", "B"]
    assert len(result.checkpoint_paths) == 2
    assert (tmp_path / "forgetting_results.json").exists()


def test_save_results_matrix(tmp_path):
    save_results_matrix(
        {"tasks": ["A"], "performance_matrix": {}, "total_time": 1.0}, tmp_path
    )
    assert (tmp_path / "forgetting_results.json").exists()
