"""Tests for the order-dependence (commutator) runner."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fabricpc.core.inference import InferenceSGDNormClip
from fabricpc.core.types import GraphParams, NodeParams
from fabricpc.experiments.forgetting_benchmark.commutator import (
    commutator_report,
    noise_floor,
    precompute_streams,
    run_order,
)
from fabricpc.experiments.forgetting_benchmark.gated_trainer import GatedConfig
from fabricpc.graph_initialization import initialize_params
from fabricpc.nodes.transformer_v2 import create_deep_transformer
from fabricpc.training.train_autoregressive import _eval_step_autoregressive

VOCAB, SEQ, BATCH, STEPS = 16, 8, 4, 6
TASKS = ("A", "B")
SHIFT = {"A": 1, "B": 5}


def _tree(values) -> GraphParams:
    return GraphParams(
        nodes={
            "n": NodeParams(weights={"w": jnp.asarray(values, jnp.float32)}, biases={})
        }
    )


def _stub_eval(theta, batch):
    probs = jax.nn.softmax(jnp.stack([theta.nodes["n"].weights["w"]] * 2)[None])
    return jnp.sum(theta.nodes["n"].weights["w"] ** 2), probs


def _model():
    structure = create_deep_transformer(
        depth=1,
        embed_dim=16,
        num_heads=2,
        mlp_dim=32,
        seq_len=SEQ,
        vocab_size=VOCAB,
        inference=InferenceSGDNormClip(eta_infer=0.05, infer_steps=2, max_norm=5.0),
    )
    return initialize_params(structure, jax.random.PRNGKey(0)), structure


def _streams(seed=0, names=TASKS):
    return precompute_streams(
        seed, names, STEPS, batch_size=BATCH, num_sequences=64, probe_batches=1
    )


def _batch_fn(calls=None):
    data = {
        name: np.random.default_rng(i).integers(0, VOCAB, size=80).astype(np.int32)
        for i, name in enumerate(TASKS)
    }

    def batch_fn(task, idx):
        if calls is not None:
            calls.append((task, tuple(np.asarray(idx).ravel())))
        offsets = np.asarray(idx)[:, None] + np.arange(SEQ)[None, :]
        x = jnp.asarray(data[task][offsets])
        return {"x": x, "y": (x + SHIFT[task]) % VOCAB}

    return batch_fn


def test_identical_trees_give_zero_distance():
    theta = _tree([1.0, 2.0, 3.0])
    report = commutator_report(theta, theta, 2.0, 3.0, {"A": None}, _stub_eval)

    assert report["distance"] == 0.0
    assert report["normalized_distance"] == 0.0
    assert report["tasks"]["A"]["tv_disagreement"] == 0.0


def test_known_parameter_trees_give_expected_normalized_distance():
    report = commutator_report(
        _tree([0.0, 0.0]), _tree([3.0, 4.0]), 2.0, 3.0, {"A": None}, _stub_eval
    )

    assert report["distance"] == pytest.approx(5.0)
    assert report["normalized_distance"] == pytest.approx(1.0)
    assert report["path_ab"] == 2.0 and report["path_ba"] == 3.0


def test_distinct_paths_give_positive_distance():
    theta_0, structure = _model()
    config = GatedConfig(lr=0.1, diag_every=0)
    streams, batch_fn = _streams(), _batch_fn()
    kwargs = dict(structure=structure, config=config, batch_fn=batch_fn)

    ab, path_ab = run_order(theta_0, ("A", "B"), streams, "none", **kwargs)
    ba, path_ba = run_order(theta_0, ("B", "A"), streams, "none", **kwargs)

    def eval_fn(theta, batch):
        return _eval_step_autoregressive(
            theta, structure, batch, jax.random.PRNGKey(0), True
        )

    probe = {name: batch_fn(name, streams.probe_indices[name][0]) for name in TASKS}
    report = commutator_report(
        ab, ba, path_ab, path_ba, probe, eval_fn, theta_0=theta_0
    )

    assert path_ab > 0 and path_ba > 0
    assert report["normalized_distance"] > 0.0
    assert set(report["tasks"]) == set(TASKS)
    floor = noise_floor(
        theta_0,
        ("A", "B"),
        streams,
        _streams(seed=1),
        "none",
        **kwargs,
    )
    assert floor["normalized_distance"] > 0.0


def test_reversed_order_uses_identical_batches_and_rng_per_task():
    forward, reverse = _streams(names=("A", "B")), _streams(names=("B", "A"))
    for name in TASKS:
        np.testing.assert_array_equal(forward.indices[name], reverse.indices[name])
        np.testing.assert_array_equal(forward.keys[name], reverse.keys[name])

    theta_0, structure = _model()
    config = GatedConfig(lr=0.1, diag_every=0)
    calls_ab, calls_ba = [], []
    run_order(
        theta_0,
        ("A", "B"),
        forward,
        "none",
        structure=structure,
        config=config,
        batch_fn=_batch_fn(calls_ab),
    )
    run_order(
        theta_0,
        ("B", "A"),
        forward,
        "none",
        structure=structure,
        config=config,
        batch_fn=_batch_fn(calls_ba),
    )

    def per_task(calls, task):
        rows = {tuple(r) for r in forward.indices[task]}
        return [c for c in calls if c[0] == task and c[1] in rows]

    for name in TASKS:
        trained = per_task(calls_ab, name)
        assert len(trained) >= STEPS
        assert trained == per_task(calls_ba, name)


def test_commutator_rejects_mismatched_fingerprint_or_token_count():
    theta = _tree([1.0])
    base = {
        "contract_fingerprint": "f",
        "theta0_hash": "t",
        "num_tokens": 10,
        "num_updates": 2,
    }
    commutator_report(
        theta, theta, 1.0, 1.0, {}, _stub_eval, meta_ab=base, meta_ba=dict(base)
    )
    for key, value in (("contract_fingerprint", "x"), ("num_tokens", 11)):
        with pytest.raises(ValueError):
            commutator_report(
                theta,
                theta,
                1.0,
                1.0,
                {},
                _stub_eval,
                meta_ab=base,
                meta_ba={**base, key: value},
            )


def test_replayed_surprise_gates_reproduce_the_surprise_run_exactly():
    theta_0, structure = _model()
    config = GatedConfig(lr=0.1, diag_every=3, probe_batches=1)
    streams = _streams()
    tau = {
        name: 1e-3
        for name, node in structure.nodes.items()
        if node.node_info.in_degree > 0
    }
    kwargs = dict(structure=structure, config=config, batch_fn=_batch_fn(), tau=tau)
    schedules = {}
    surprise, path = run_order(
        theta_0, ("A", "B"), streams, "surprise", recorder_out=schedules, **kwargs
    )
    replay, replay_path = run_order(
        theta_0,
        ("A", "B"),
        streams,
        "replay_gates",
        replay_schedule=schedules["A+B"],
        **kwargs,
    )
    influence, _ = run_order(theta_0, ("A", "B"), streams, "influence", **kwargs)

    for got, want in zip(
        jax.tree_util.tree_leaves(replay), jax.tree_util.tree_leaves(surprise)
    ):
        np.testing.assert_array_equal(got, want)
    assert replay_path == path
    assert all(np.all(np.isfinite(x)) for x in jax.tree_util.tree_leaves(influence))
