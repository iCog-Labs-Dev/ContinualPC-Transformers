"""Tests for local prediction-error maps (tiny BERT, offline)."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fabricpc.experiments.domain_forgetting.error_maps import (
    SPIKE_ONLY,
    STANDARD_PC,
    ErrorMapConfig,
    _clamps,
    bp_signal_energies,
    build_map_structure,
    compute_error_maps,
    hidden_nodes,
    map_specs,
    node_energies,
    spike_only_inference,
    window_batches,
)
from fabricpc.experiments.domain_forgetting.localization import load_run
from fabricpc.experiments.domain_forgetting.runner import max_free_depth
from fabricpc.graph_initialization import initialize_params
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.nodes.encoder_transformer import create_encoder_mlm_model
from tests.domain_forgetting_fixtures import build_tiny_run

VOCAB = 20
SEQ_LEN = 6
SCALE = 2.0


def _tiny_bert(inference):
    return create_encoder_mlm_model(
        depth=2,
        embed_dim=16,
        num_heads=4,
        mlp_dim=32,
        seq_len=SEQ_LEN,
        vocab_size=VOCAB,
        inference=inference,
        use_attention_mask=True,
        dropout_rate=0.1,
        pc_output_energy_scale=SCALE,
        transformer_init_std=0.3,
        architecture="bert",
        residual_identity=True,
    )


def _batch(seed=0, batch_size=3):
    rng = np.random.default_rng(seed)
    mask = rng.random((batch_size, SEQ_LEN)) < 0.5
    mask[:, 1] = True
    attention = np.ones((batch_size, SEQ_LEN))
    attention[0, -1] = 0.0
    return {
        "x": jnp.asarray(rng.integers(0, VOCAB, (batch_size, SEQ_LEN))),
        "y": jnp.asarray(rng.integers(0, VOCAB, (batch_size, SEQ_LEN))),
        "loss_mask": jnp.asarray(mask),
        "attention_mask": jnp.asarray(attention),
    }


def _bp_logit_signal(params, structure, batch):
    inputs, _ = _clamps(structure, batch)
    state = initialize_graph_state(
        structure, 3, jax.random.PRNGKey(0), clamps=inputs, params=params
    )
    hidden = state.nodes["mlm_transform"].z_latent
    head = params.nodes["mlm_head"]
    logits = hidden @ head.weights["W_out"] + head.biases["b_out"]
    mask = batch["loss_mask"].astype(logits.dtype)

    def loss(values):
        log_probs = jax.nn.log_softmax(values, axis=-1)
        ce = -jnp.take_along_axis(log_probs, batch["y"][..., None], axis=-1)[..., 0]
        return jnp.sum(ce * mask) / jnp.sum(mask)

    grad = jax.grad(loss)(logits)
    return 0.5 * SCALE**2 * jnp.sum(grad**2, axis=(1, 2))


def test_spike_only_map_equals_backprop_signal_in_float64():
    """Raw energies use the summed head energy: BP's mean-CE signal times N²."""
    with jax.enable_x64(True):
        probe = _tiny_bert(spike_only_inference(1))
        structure = _tiny_bert(spike_only_inference(max_free_depth(probe)))
        params = initialize_params(structure, jax.random.PRNGKey(0))
        batch = _batch()
        pc = jax.jit(lambda p, b: node_energies(p, structure, b))(params, batch)
        again = jax.jit(lambda p, b: node_energies(p, structure, b))(params, batch)
        bp = bp_signal_energies(params, structure, batch, SCALE)
        logit_signal = _bp_logit_signal(params, structure, batch)
        count = float(np.sum(np.asarray(batch["loss_mask"])))
        bp = {n: v * count**2 for n, v in bp.items()}
        logit_signal = logit_signal * count**2
    assert max_free_depth(probe) == 9
    assert set(bp) == set(hidden_nodes(structure))
    assert "L0_residual" in bp and "mlm_head" not in bp
    worst = 0.0
    final_ratio = {}
    for node, expected in bp.items():
        expected = np.asarray(expected)
        got = np.asarray(pc["forward_error_energy"][node])
        assert expected.dtype == np.float64
        assert (expected > 0).all()
        worst = max(worst, float(np.max(np.abs(got - expected) / expected)))
        final_ratio[node] = float(
            np.asarray(pc["final_energy"][node]).sum() / expected.sum()
        )
    print("spike_only_vs_bp_max_rel_err", worst, "final_over_bp", final_ratio)
    assert worst < 1e-6
    np.testing.assert_allclose(
        pc["head_energy"]["forward_error_energy"], logit_signal, rtol=1e-10
    )
    assert final_ratio["token_embeddings"] == pytest.approx(1.0, rel=1e-6)
    for field in ("final_energy", "forward_error_energy"):
        for node, value in pc[field].items():
            assert (np.asarray(value) >= 0).all()
            np.testing.assert_array_equal(value, again[field][node])
    assert (np.asarray(pc["output_ce_energy"]) >= 0).all()


def test_map_specs_cover_onset_and_retrospective_maps():
    specs = map_specs(["a", "b", "c"])
    assert [(s["kind"], s["domain"], s["stage"]) for s in specs] == [
        ("onset", "b", 1),
        ("onset", "c", 2),
        ("retrospective", "a", 2),
        ("retrospective", "a", 3),
        ("retrospective", "b", 3),
    ]


@pytest.fixture(scope="module")
def context(tmp_path_factory):
    config = build_tiny_run(tmp_path_factory.mktemp("error_maps"))
    return load_run(config.output_dir)


def test_run_error_maps_are_nonnegative_deterministic_and_batch_invariant(context):
    config = ErrorMapConfig(
        num_windows=8, batch_size=4, pc_infer_steps=5, pc_eta_infer=0.01
    )
    result = compute_error_maps(context, config)
    assert set(result["settings"]) == {SPIKE_ONLY, STANDARD_PC}
    assert result["settings"][SPIKE_ONLY]["infer_steps"] == 9
    assert result["settings"][SPIKE_ONLY]["eta_infer"] == 0.0
    assert result["settings"][STANDARD_PC]["infer_steps"] == 5
    assert len(result["maps"]) == 10
    for entry in result["maps"]:
        loss_mask = context.valid_sets[entry["domain"]]["loss_mask"][:8]
        assert entry["num_windows"] == int((loss_mask.sum(axis=1) > 0).sum()) > 0
        assert list(entry["modules"]) == result["modules"]
        assert set(entry["residuals"]) == {"L0_residual", "L1_residual"}
        for values in [*entry["modules"].values(), *entry["residuals"].values()]:
            assert all(v >= 0 and np.isfinite(v) for v in values.values())
        assert entry["output_ce_energy"] > 0
    first = result["maps"][0]["modules"]["L0_mlp1"]
    assert first["energy"] == first["forward_error_energy"]
    standard = next(m for m in result["maps"] if m["setting"] == STANDARD_PC)
    assert standard["modules"]["L0_mlp1"]["energy"] == (
        standard["modules"]["L0_mlp1"]["final_energy"]
    )
    assert compute_error_maps(context, config) == result
    for entry in result["maps"]:
        for module, values in entry["modules"].items():
            width = context.structure.nodes[module].node_info.shape[-1]
            assert values["output_dim"] == width
            assert values["energy_sum_ce_per_dim"] == pytest.approx(
                values["energy_sum_ce"] / width
            )
        if entry["setting"] == STANDARD_PC:
            trace = entry["energy_trace"]
            assert trace["steps"] == list(range(6))
            assert len(trace["total_energy"]) == 6
            assert trace["first"] == trace["total_energy"][0] > 0
            assert trace["last"] == trace["total_energy"][-1]
            assert trace["last"] <= trace["first"]
            assert all(np.isfinite(trace["total_energy"]))
        else:
            assert "energy_trace" not in entry
    assert result["provenance"]["stage_params_sha256"][0] == (
        context.run_config["source"]["params_sha256"]
    )
    assert result["provenance"]["valid_sha256"] == context.run_config["valid_sha256"]

    single = compute_error_maps(
        context, ErrorMapConfig(num_windows=8, batch_size=1, settings=(SPIKE_ONLY,))
    )
    for wide, narrow in zip(result["maps"][:5], single["maps"]):
        assert (wide["domain"], wide["stage"]) == (narrow["domain"], narrow["stage"])
        for module in result["modules"]:
            assert wide["modules"][module]["energy_sum_ce"] == pytest.approx(
                narrow["modules"][module]["energy_sum_ce"], rel=1e-3
            )


def test_standard_pc_trace_matches_the_untraced_inference(context):
    config = ErrorMapConfig(
        num_windows=4,
        batch_size=4,
        pc_infer_steps=4,
        pc_eta_infer=0.01,
        settings=(STANDARD_PC,),
    )
    structure = build_map_structure(context, STANDARD_PC, config)
    batch = window_batches(context.valid_sets["news"], 4, 4)[0]
    params = context.params(1)
    traced = jax.jit(lambda p, b: node_energies(p, structure, b, trace=True))(
        params, batch
    )
    plain = jax.jit(lambda p, b: node_energies(p, structure, b))(params, batch)
    for node, value in plain["final_energy"].items():
        np.testing.assert_allclose(traced["final_energy"][node], value, rtol=1e-6)
    trace = np.asarray(traced["energy_trace"])
    assert trace.shape == (5,)
    expected_last = sum(
        float(np.sum(v)) for v in traced["final_energy"].values()
    ) + float(np.sum(traced["output_ce_energy"]))
    assert trace[-1] == pytest.approx(expected_last, rel=1e-5)
    head_only = float(np.sum(plain["output_ce_energy"]))
    assert trace[0] >= head_only * (1 - 1e-6)
