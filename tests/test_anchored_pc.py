"""Regression checks for anchored-PC diagnostics."""

import jax.numpy as jnp
import jax
import numpy as np
import pytest

from fabricpc.experiments.anchored_pc import (
    blend_gradients,
    direction_metrics,
    match_norm,
    normalized_pc_gradient,
    convergence_passes,
    tiny_anchor_check,
    make_pc_trace,
    DiagnosticConfig,
    run_diagnostic,
)


def test_blend_endpoints_and_interior_preserve_inputs():
    bp = {"a": jnp.array([2.0, 0.0]), "empty": {}}
    pc = {"a": jnp.array([0.0, 4.0]), "empty": {}}
    np.testing.assert_array_equal(blend_gradients(bp, pc, 0)["a"], bp["a"])
    np.testing.assert_array_equal(blend_gradients(bp, pc, 1)["a"], pc["a"])
    np.testing.assert_allclose(blend_gradients(bp, pc, 0.25)["a"], [1.5, 1.0])
    np.testing.assert_array_equal(bp["a"], [2.0, 0.0])


@pytest.mark.parametrize("beta", [-1, 1.1, float("nan"), float("inf")])
def test_blend_rejects_invalid_beta(beta):
    with pytest.raises(ValueError):
        blend_gradients({"a": jnp.ones(2)}, {"a": jnp.ones(2)}, beta)


def test_blend_rejects_shape_broadcasting_and_nonfinite_even_at_endpoint():
    with pytest.raises(ValueError):
        blend_gradients({"a": jnp.ones(2)}, {"a": jnp.ones(1)}, 0.5)
    with pytest.raises(ValueError):
        blend_gradients({"a": jnp.ones(2)}, {"a": jnp.array([jnp.nan, 1])}, 0)


def test_norm_matching_removes_only_magnitude():
    bp = {"a": jnp.array([3.0, 4.0])}
    target = {"a": jnp.array([0.0, 10.0])}
    matched = match_norm(bp, target)
    np.testing.assert_allclose(matched["a"], [6.0, 8.0])
    assert match_norm({"a": jnp.zeros(2)}, target) is None
    metrics = direction_metrics(bp, target)
    assert metrics["cosine"] == pytest.approx(0.8)
    assert metrics["perpendicular_norm"] == pytest.approx(6.0)


def test_nudge_scale_compensation():
    grads = {"a": jnp.array([0.1, 0.2])}
    np.testing.assert_allclose(normalized_pc_gradient(grads, 0.1)["a"], [1, 2])
    with pytest.raises(ValueError):
        normalized_pc_gradient(grads, 0)


def test_energy_flattening_or_small_forcing_does_not_establish_convergence():
    assert convergence_passes(1e-5, 1.0, 0.001)
    assert not convergence_passes(1e-5, 1e-5, 0.001)
    assert not convergence_passes(0.0, 1.0, 0.2)
    assert not convergence_passes(float("nan"), 1.0, 0.0)


def test_tiny_anchor_is_exact_with_dropout_and_parameter_perturbation():
    results = tiny_anchor_check()
    assert len(results) == 4
    assert all(r["passed"] for r in results), results


def test_trace_recomputes_final_gradient_without_mutating_weights(tmp_path):
    from dataclasses import replace
    from tests.test_domain_forgetting_runner import make_setup, tiny_config
    from fabricpc.experiments.domain_forgetting.runner import (
        load_base_model,
        fixed_mlm_batches,
        state_total_energy,
    )
    from fabricpc.experiments.domain_forgetting.data import load_domain_cache
    from fabricpc.training.train_mlm import get_graph_param_gradient_pcn_mlm
    from fabricpc.experiments.forgetting_benchmark.sequential_trainer import params_hash

    checkpoint, data_dir = make_setup(tmp_path)
    config = replace(
        tiny_config(checkpoint, data_dir, tmp_path / "run"),
        mode="pc",
        dropout_rate=0,
        pc_inference="standard",
        pc_eta_infer=0.1,
        pc_infer_steps=4,
        pc_forward_updates="none",
    )
    base = load_base_model(config)
    cache = load_domain_cache(
        data_dir, "news", tokenizer_json=base.tokenizer_json, seq_len=6
    )
    batches, keys = fixed_mlm_batches(
        cache.train,
        base.structure,
        base.special_ids,
        num_batches=2,
        batch_size=2,
        mask_prob=0.3,
        seed=0,
    )
    before = params_hash(base.params)
    grads, trace = make_pc_trace(base.structure, 4)(base.params, batches[0], keys[0])
    reference, _, _, _, final = get_graph_param_gradient_pcn_mlm(
        base.params, batches[0], base.structure, keys[0], "none"
    )
    for a, b in zip(
        jax.tree_util.tree_leaves(grads), jax.tree_util.tree_leaves(reference)
    ):
        np.testing.assert_allclose(a, b, atol=2e-6, rtol=1e-4)
    assert trace.shape == (5, 4)
    np.testing.assert_allclose(
        trace[-1, 0], state_total_energy(base.params, final, base.structure), rtol=1e-5
    )
    assert params_hash(base.params) == before


def test_expired_budget_saves_no_go_report_without_loading_checkpoint(tmp_path):
    result = run_diagnostic(
        DiagnosticConfig(
            checkpoint="absent",
            data_dir="absent",
            output_dir=str(tmp_path / "output"),
            budget_seconds=1e-12,
        )
    )
    assert result["status"] == "budget_exhausted"
    assert result["decision"] == "no_go"
    assert (tmp_path / "output" / "summary.json").exists()


def test_tiny_end_to_end_records_an_explicit_stop_decision(tmp_path):
    from tests.test_domain_forgetting_runner import make_setup

    checkpoint, data_dir = make_setup(tmp_path / "input")
    result = run_diagnostic(
        DiagnosticConfig(
            checkpoint=str(checkpoint),
            data_dir=str(data_dir),
            output_dir=str(tmp_path / "output"),
            num_batches=2,
            batch_size=2,
            rates=(0.1,),
            step_pairs=(2,),
            budget_seconds=180,
        )
    )
    assert result["decision"] == "no_go"
    assert result["status"] == "convergence_failed"
    assert all(r["passed"] for r in result["anchor_checks"])
    assert result["tiny_anchor"]
    assert (tmp_path / "output" / "summary.json").exists()
    assert (tmp_path / "output" / "events.jsonl").exists()
