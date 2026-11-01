"""Behavioral regressions for the standalone MNIST mechanism experiment."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import sys

from fabricpc.experiments.anchored_mnist import (
    init_params,
    loss,
    spike_gradient,
    local_gradients,
    energy,
    infer,
    gradient,
    split_indices,
    METHODS,
    latent_gradient,
    train_step,
)


@pytest.fixture
def tiny():
    with jax.enable_x64(True):
        p = init_params(3, (4, 5, 3, 2))
        x = jnp.asarray([[0.2, 0.3, 0.8, 0.1], [0.5, 0.9, 0.1, 0.4]], dtype=jnp.float64)
        y = jnp.asarray([0, 1])
        yield p, x, y


def test_spike_schedule_matches_autodiff_at_init_and_perturbed_weights(tiny):
    p, x, y = tiny
    for shift in (0.0, 0.03):
        params = jax.tree_util.tree_map(lambda v: v + shift, p)
        bp = jax.grad(loss)(params, x, y)
        spike = spike_gradient(params, x, y)
        for a, b in zip(
            jax.tree_util.tree_leaves(bp), jax.tree_util.tree_leaves(spike)
        ):
            np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12)


def test_local_weight_rule_is_partial_energy_gradient_and_nudge_scale(tiny):
    p, x, y = tiny
    z, _ = infer(p, x, y, 0.1, 20, 0.1)
    actual = local_gradients(p, x, y, z, 0.1)
    expected = jax.grad(energy)(p, x, y, z, 0.1)
    for a, b in zip(
        jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected)
    ):
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12)


def test_latent_rule_matches_energy_derivative_with_batch_factor(tiny):
    p, x, y = tiny
    z, _ = infer(p, x, y, 0.1, 3, 0.1)
    actual = latent_gradient(p, x, y, z, 0.1)
    expected = jax.grad(energy, argnums=3)(p, x, y, z, 0.1)
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, b * len(x), rtol=1e-10, atol=1e-12)


def test_unconverged_gradient_cannot_update_parameters(tiny):
    p, x, y = tiny
    after, stats, passed = train_step(p, x, y, 0.1, method="pc", solver=(0.1, 1))
    assert not bool(passed)
    assert float(stats[0]) > 1e-3
    for a, b in zip(jax.tree_util.tree_leaves(p), jax.tree_util.tree_leaves(after)):
        np.testing.assert_array_equal(a, b)


def test_relaxation_reduces_energy_and_reports_unclipped_residual(tiny):
    p, x, y = tiny
    _, trace = infer(p, x, y, 0.1, 160, 1.0)
    assert trace[-1, 0] < trace[0, 0]
    assert trace[-1, 1] / trace[0, 1] < 1e-3
    assert np.all(np.diff(np.asarray(trace[:, 0])) <= 1e-12)


def test_split_indices_are_disjoint_reproducible_and_leave_test_external():
    a = split_indices(100, 20, 30, 42)
    b = split_indices(100, 20, 30, 42)
    assert len(a[0]) == 20 and len(a[1]) == 30
    assert not set(a[0]) & set(a[1])
    assert all(np.array_equal(i, j) for i, j in zip(a, b))
    with pytest.raises(ValueError):
        split_indices(10, 8, 8, 0)


def test_beta_endpoints_and_norm_controls_use_same_weights(tiny):
    p, x, y = tiny
    solver = (0.1, 80)
    bp, _ = gradient(p, x, y, "bp", solver)
    spike, _ = gradient(p, x, y, "spike", solver)
    pc, _ = gradient(p, x, y, "pc", solver)
    blend, _ = gradient(p, x, y, "anchor_0.5", solver)
    matched, _ = gradient(p, x, y, "matched_0.5", solver)
    flat = lambda tree: np.concatenate(
        [np.ravel(v) for v in jax.tree_util.tree_leaves(tree)]
    )
    np.testing.assert_allclose(flat(bp), flat(spike), atol=1e-12)
    # A common clip follows blending, so compare the raw rule with a large clip.
    raw, _ = gradient(p, x, y, "anchor_0.5", solver, clip=1e6)
    np.testing.assert_allclose(flat(raw), (flat(spike) + flat(pc)) / 2, atol=1e-10)
    np.testing.assert_allclose(
        np.linalg.norm(flat(matched)), np.linalg.norm(flat(blend)), rtol=1e-10
    )
    np.testing.assert_allclose(
        flat(matched) / np.linalg.norm(flat(matched)),
        flat(bp) / np.linalg.norm(flat(bp)),
        atol=1e-10,
    )
    assert len(METHODS) == 8


def test_rate_sweep_has_all_paired_cells_without_tuning():
    experiment = sys.modules[init_params.__module__]

    schedule = experiment.rate_sweep_schedule()
    assert len(schedule) == len(set(schedule)) == 54
    for seed in (1, 2, 3):
        for lr in (0.3, 1.0, 3.0):
            assert {m for m, rate, s in schedule if (rate, s) == (lr, seed)} == {
                "bp",
                "pc",
                "anchor_0.1",
                "anchor_0.5",
                "matched_0.1",
                "matched_0.5",
            }


def test_sweep_failures_distinguish_inference_and_nonfinite_gradient():
    experiment = sys.modules[init_params.__module__]

    assert (
        experiment.gradient_failure(np.array([0.01, 0.02, 0.0]), False)
        == "inference_failed"
    )
    assert (
        experiment.gradient_failure(np.array([np.nan, 0.0, 0.0]), False)
        == "nonfinite_inference"
    )
    assert experiment.gradient_failure(np.zeros(3), False) == "nonfinite_gradient"
    assert experiment.gradient_failure(np.zeros(3), True) is None


def test_gpu_requirement_refuses_cpu_before_download_or_run(tmp_path):
    import os
    from pathlib import Path
    import subprocess
    import sys

    source = Path(sys.modules[init_params.__module__].__file__)
    result = subprocess.run(
        [
            sys.executable,
            str(source),
            "--data-dir",
            str(tmp_path / "data"),
            "--output-dir",
            str(tmp_path / "run"),
            "--require-gpu",
            "--download",
        ],
        env=dict(os.environ, JAX_PLATFORMS="cpu"),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "GPU required" in result.stderr
    assert not (tmp_path / "run").exists()
    assert not (tmp_path / "data").exists()


def test_download_checksum_failure_never_publishes_partial_file(tmp_path):
    experiment = sys.modules[init_params.__module__]

    source = tmp_path / "bad.gz"
    source.write_bytes(b"not an MNIST archive")
    destination = tmp_path / "downloaded.gz"
    with pytest.raises(ValueError, match="checksum"):
        experiment.download_verified(source.as_uri(), destination, "0" * 32)
    assert not destination.exists()
    assert not destination.with_suffix(".gz.part").exists()


def test_verified_download_preserves_other_partial_files_and_reuses_cache(tmp_path):
    import hashlib

    experiment = sys.modules[init_params.__module__]
    source = tmp_path / "source.gz"
    source.write_bytes(b"checksum-verified fixture")
    digest = hashlib.md5(source.read_bytes()).hexdigest()
    destination = tmp_path / "downloaded.gz"
    other_partial = destination.with_suffix(".gz.part")
    other_partial.write_bytes(b"another writer")
    experiment.download_verified(source.as_uri(), destination, digest)
    assert destination.read_bytes() == source.read_bytes()
    assert other_partial.read_bytes() == b"another writer"
    experiment.download_verified("file:///nonexistent", destination, digest)
