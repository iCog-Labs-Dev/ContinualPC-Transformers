import json
import pytest

from diagnostics.aggregate_seeds import load_metrics


def _write_metrics(path, **overrides):
    base = {
        "mode": "pc", "tokenizer": "char", "seed": 1,
        "n_params": 413000, "vocab_size": 65, "global_batch_size": 32,
        "config": {"embed_dim": 128, "num_epochs": 5},
        "test_ppl": 7.1, "test_loss": 1.9, "test_accuracy": 0.5,
        "final_train_ppl": 5.0, "final_train_energy": 100.0, "train_seconds": 10.0,
    }
    base.update(overrides)
    path.write_text(json.dumps(base))


def test_rejects_mismatched_n_params(tmp_path):
    """Seeds with different n_params (different architectures) are rejected."""
    _write_metrics(tmp_path / "metrics_pc_char_seed1.json", seed=1, n_params=413000)
    _write_metrics(tmp_path / "metrics_pc_char_seed2.json", seed=2, n_params=108000)
    with pytest.raises(ValueError, match="n_params"):
        load_metrics(str(tmp_path), tokenizer="char")


def test_rejects_mismatched_config(tmp_path):
    """Seeds with different configs (e.g. num_epochs) are rejected."""
    _write_metrics(tmp_path / "metrics_pc_char_seed1.json", seed=1,
                   config={"embed_dim": 128, "num_epochs": 5})
    _write_metrics(tmp_path / "metrics_pc_char_seed2.json", seed=2,
                   config={"embed_dim": 128, "num_epochs": 10})  # different epochs
    with pytest.raises(ValueError, match="config"):
        load_metrics(str(tmp_path), tokenizer="char")


def test_accepts_compatible_runs(tmp_path):
    """Seeds from the same experiment group and aggregate normally."""
    _write_metrics(tmp_path / "metrics_pc_char_seed1.json", seed=1)
    _write_metrics(tmp_path / "metrics_pc_char_seed2.json", seed=2)
    arms = load_metrics(str(tmp_path), tokenizer="char")
    assert len(arms[("pc", "char")]) == 2