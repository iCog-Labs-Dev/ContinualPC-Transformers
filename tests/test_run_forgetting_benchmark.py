"""Small end-to-end check for the forgetting benchmark runner."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from pathlib import Path

import numpy as np

from fabricpc.core.inference import InferenceSGDNormClip
from fabricpc.experiments.forgetting_benchmark.run_benchmark import (
    BenchmarkConfig,
    run_full_benchmark,
)
from fabricpc.graph_initialization import initialize_params
from fabricpc.nodes.transformer_v2 import create_deep_transformer


class Loader:
    num_sequences = 4

    def __init__(self, offset: int):
        self.offset = offset

    def __len__(self):
        return 1

    def __iter__(self):
        rng = np.random.default_rng(42 + self.offset)
        x = rng.integers(0, 20, size=(4, 8), dtype=np.int32)
        yield x, np.roll(x, -1, axis=1)


class TinyBenchmark:
    tasks = ["tiny_stories", "wikitext2", "tiny_shakespeare"]
    vocab_size = 20

    def __len__(self):
        return len(self.tasks)

    def get_task_name(self, index):
        return self.tasks[index]

    def get_train_loader(self, index):
        return Loader(index)

    def get_test_loader(self, index):
        return Loader(index)


def _model(key):
    structure = create_deep_transformer(
        depth=1,
        embed_dim=16,
        num_heads=2,
        mlp_dim=32,
        seq_len=8,
        vocab_size=20,
        inference=InferenceSGDNormClip(eta_infer=0.05, infer_steps=1, max_norm=5.0),
    )
    return initialize_params(structure, key), structure


def test_baseline_defaults_are_the_three_character_tasks():
    assert BenchmarkConfig().tasks == ("tiny_stories", "wikitext2", "tiny_shakespeare")


def test_runner_writes_baseline_artifacts_without_external_data(tmp_path):
    result = run_full_benchmark(
        {"output_dir": str(tmp_path), "epochs_per_task": 1, "seed": 7},
        benchmark=TinyBenchmark(),
        model_factory=_model,
        verbose=False,
    )

    assert result["forgetting"].matrix.shape == (3, 3)
    assert list(result["sequential"].performance_matrix) == TinyBenchmark.tasks
    assert list(result["sequential"].performance_matrix["wikitext2"]) == [
        "tiny_stories",
        "wikitext2",
    ]
    assert list(result["sequential"].performance_matrix["tiny_shakespeare"]) == (
        TinyBenchmark.tasks
    )
    assert len(list(tmp_path.glob("task_*_checkpoint"))) == 3
    assert len((tmp_path / "per_stage_performance.csv").read_text().splitlines()) == 7
    assert result["summary"]["conclusion"]
    assert result["summary"]["next_diagnostic"]
    expected = {
        "experiment_record.json",
        "forgetting_matrix.npy",
        "forgetting_results.json",
        "per_stage_performance.csv",
        "resource_report.json",
        "summary.json",
        "summary.txt",
        "summary_table.tex",
    }
    assert expected <= {path.name for path in Path(tmp_path).iterdir()}
    assert "Conclusion:" in (tmp_path / "summary.txt").read_text()
