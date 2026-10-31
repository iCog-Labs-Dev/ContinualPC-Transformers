"""Tests for μPC on the post-LN BERT graph: the W / a load and the jacobian mode."""

import json
import math
import os
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fabricpc.core.activations import GeluActivation
from fabricpc.core.inference import InferenceSGDNormClip
from fabricpc.core.mupc import MuPCScalingFactors, jacobian_only_scalings
from fabricpc.experiments.domain_forgetting.artifacts import load_npz
from fabricpc.experiments.domain_forgetting.localization import (
    compute_drift,
    load_run,
    module_keys,
    rollback_flat,
)
from fabricpc.experiments.domain_forgetting.probe import probe_alignment
from fabricpc.experiments.domain_forgetting.runner import (
    check_comparable_runs,
    gradient_alignment,
    load_base_model,
    make_alignment_fn,
    make_gradient_fn,
    module_weight_norms,
    params_from_flat,
    params_hash,
    run_domain_forgetting,
    state_total_energy,
    train_stage,
)
from fabricpc.experiments.domain_forgetting.validation import MultiDomainEvaluator
from fabricpc.graph_initialization import initialize_params
from fabricpc.nodes.encoder_transformer import (
    MUPC_LINEAR_INPUTS,
    create_encoder_mlm_model,
    mupc_function_preserving_params,
    mupc_plain_grads,
    mupc_plain_params,
    mupc_weight_scales,
)
from fabricpc.nodes.transformer_v2 import Mlp2ResidualNode
from fabricpc.training.train_backprop import _loss_and_grad_backprop_mlm_no_update
from fabricpc.training.train_mlm import (
    _prepare_pcn_mlm_batch,
    get_graph_param_gradient_pcn_mlm,
)
from tests.test_domain_forgetting_runner import make_setup, tiny_config
from tests.test_epc_eta_control import _stage_stream
from tests.test_inference_epc import bert_batch

GELU_GAIN = 1.168
STANDARD = {
    "mode": "pc",
    "pc_inference": "standard",
    "pc_eta_infer": 0.01,
    "pc_infer_steps": 5,
    "pc_forward_updates": "none",
}
HEAD_PINS = {
    "backprop": (
        "09a418db97950176bfc38efbe0b558ece7251b5b8eb958a55dbe21cd869cbe67",
        [3.417818069458008, 3.4387256622314455],
        [0.738935649394989, 0.6775385737419128],
    ),
    "standard": (
        "60c9f91a9774a2c771503a177f75ad744f297b55f0a3d469fc808225fb317082",
        [3.417818069458008, 3.4419506072998045],
        [0.6275948286056519, 0.5946071743965149],
    ),
}


def build(
    mupc=None,
    *,
    depth=2,
    eta=0.5,
    steps=12,
    dropout_rate=0.1,
    max_norm=5.0,
    residual_identity=True,
    **shape,
):
    values = {
        "embed_dim": 8,
        "num_heads": 2,
        "mlp_dim": 16,
        "seq_len": 6,
        "vocab_size": 20,
        **shape,
    }
    return create_encoder_mlm_model(
        depth=depth,
        inference=InferenceSGDNormClip(
            eta_infer=eta, infer_steps=steps, max_norm=max_norm
        ),
        use_attention_mask=True,
        dropout_rate=dropout_rate,
        architecture="bert",
        residual_identity=residual_identity,
        mupc=mupc,
        **values,
    )


def random_params(structure, seed=0):
    """Initialised params with every leaf perturbed, so biases and LN are generic."""
    params = initialize_params(structure, jax.random.PRNGKey(seed))
    leaves, treedef = jax.tree_util.tree_flatten(params)
    keys = jax.random.split(jax.random.PRNGKey(seed + 1), len(leaves))
    leaves = [
        leaf + 0.1 * jax.random.normal(key, leaf.shape, leaf.dtype)
        for leaf, key in zip(leaves, keys)
    ]
    return jax.tree_util.tree_unflatten(treedef, leaves)


def _leaves(tree):
    return [np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(tree)]


def _assert_trees_equal(left, right):
    for a, b in zip(_leaves(left), _leaves(right), strict=True):
        np.testing.assert_array_equal(a, b)


def _states_close(left, right, field, rtol, atol=0.0):
    for name in left.nodes:
        np.testing.assert_allclose(
            np.asarray(getattr(left.nodes[name], field)),
            np.asarray(getattr(right.nodes[name], field)),
            rtol=rtol,
            atol=atol,
            err_msg=f"{name}.{field}",
        )


def _feedforward(params, structure, key=jax.random.PRNGKey(7)):
    _, metrics, free, initial, _ = _prepare_pcn_mlm_batch(
        params, bert_batch(), structure, key
    )
    return metrics, free, initial


class TestScalings:
    def test_bert_mini_factors(self):
        """a = gain / sqrt(fan_in · L) with L = 2 · depth = 8; the GELU edges
        carry the top-down gain 1.168 and the residual identity edges none."""
        structure = build(
            "full", depth=4, embed_dim=256, num_heads=4, mlp_dim=1024, vocab_size=30
        )
        jacobian = build(
            "jacobian", depth=4, embed_dim=256, num_heads=4, mlp_dim=1024, vocab_size=30
        )
        expected = {
            "L{i}_mha": (1 / math.sqrt(256 * 8), 1.0),
            "L{i}_mlp1": (math.sqrt(2) / math.sqrt(256 * 8), GELU_GAIN),
            "L{i}_mlp2": (1 / math.sqrt(1024 * 8), 1.0),
        }
        scaled = {}
        for i in range(4):
            for pattern, (a, gain) in expected.items():
                scaled[pattern.format(i=i)] = (a, gain)
        scaled["mlm_transform"] = (1 / math.sqrt(256 * 8), 1.0)
        assert scaled["L0_mlp1"][0] == pytest.approx(1 / 32)
        for name in structure.node_order:
            full = structure.nodes[name].node_info.scaling_config
            jac = jacobian.nodes[name].node_info.scaling_config
            if name not in scaled:
                assert full is None and jac is None, name
                continue
            a, gain = scaled[name]
            (edge,) = full.forward_scale
            assert structure.edges[edge].slot == "in"
            assert full.forward_scale[edge] == pytest.approx(a, rel=1e-12)
            assert full.topdown_grad_scale[edge] == pytest.approx(a * gain, rel=1e-12)
            if gain == 1.0:
                assert jac is None, name
                continue
            assert jac.forward_scale == {}
            assert jac.topdown_grad_scale == {edge: pytest.approx(gain, rel=1e-12)}
            assert jac.weight_grad_scale == full.weight_grad_scale
            assert jac.self_grad_scale == full.self_grad_scale
        assert structure.config["mupc"] == "full"
        assert jacobian.config["mupc"] == "jacobian"

    def test_jacobian_only_scalings(self):
        factors = MuPCScalingFactors(
            forward_scale={"e": 0.25, "f": 0.5},
            self_grad_scale=1.0,
            topdown_grad_scale={"e": 0.25 * 1.168, "f": 0.5},
            weight_grad_scale={"e": 1.0, "f": 1.0},
        )
        jac = jacobian_only_scalings(factors)
        assert jac.forward_scale == {}
        assert jac.topdown_grad_scale == {"e": pytest.approx(1.168)}
        assert jac.weight_grad_scale == factors.weight_grad_scale

    def test_off_is_the_previous_graph(self):
        for mupc in (None, "off"):
            structure = build(mupc)
            assert "mupc" not in structure.config
            assert all(
                node.node_info.scaling_config is None
                for node in structure.nodes.values()
            )

    def test_mode_validation(self):
        with pytest.raises(ValueError, match="mupc must be one of"):
            build("on")
        encoder = {
            "depth": 1,
            "embed_dim": 8,
            "num_heads": 2,
            "mlp_dim": 16,
            "seq_len": 6,
            "vocab_size": 20,
            "inference": InferenceSGDNormClip(),
        }
        default = create_encoder_mlm_model(**encoder)
        explicit = create_encoder_mlm_model(**encoder, mupc="off")
        for name, node in default.nodes.items():
            assert (
                explicit.nodes[name].node_info.scaling_config
                == node.node_info.scaling_config
            )
        assert "mupc" not in explicit.config
        with pytest.raises(ValueError, match="only configurable"):
            create_encoder_mlm_model(**encoder, mupc="jacobian")

    def test_residual_identity_does_not_change_the_factors(self):
        """Stripping the residual identity edges equals μPC of the graph
        without them."""
        for mupc in ("full", "jacobian"):
            with_identity = build(mupc, depth=3)
            without = build(mupc, depth=3, residual_identity=False)
            for name, node in without.nodes.items():
                left = with_identity.nodes[name].node_info.scaling_config
                right = node.node_info.scaling_config
                if right is None:
                    assert left is None, name
                    continue
                rename = {
                    key: key.replace("_residual->", "_mha->")
                    for key in left.forward_scale
                }
                assert {
                    rename.get(k, k): v for k, v in left.forward_scale.items()
                } == right.forward_scale
                assert left.topdown_grad_scale == right.topdown_grad_scale
            for i in range(3):
                assert (
                    with_identity.nodes[f"L{i}_residual"].node_info.scaling_config
                    is None
                )


class TestFunctionPreservingLoad:
    def test_feedforward_and_energy_match_float64(self):
        with jax.enable_x64(True):
            off, full = build("off"), build("full")
            params = random_params(off)
            loaded = mupc_function_preserving_params(full, params)
            m_off, free_off, init_off = _feedforward(params, off)
            m_full, free_full, init_full = _feedforward(loaded, full)
            _states_close(free_off, free_full, "z_mu", 1e-12, 1e-13)
            _states_close(init_off, init_full, "z_mu", 1e-12, 1e-13)
            assert float(m_full["masked_loss"]) == pytest.approx(
                float(m_off["masked_loss"]), rel=1e-12
            )
            energy_off = float(state_total_energy(params, init_off, off))
            energy_full = float(state_total_energy(loaded, init_full, full))
            assert energy_off > 1e-3
            assert energy_full == pytest.approx(energy_off, rel=1e-12)

    def test_only_weights_behind_scaled_edges_change(self):
        with jax.enable_x64(True):
            full = build("full")
            params = random_params(full)
            loaded = mupc_function_preserving_params(full, params)
            scales = mupc_weight_scales(full)
            assert scales["L0_mha"].keys() == {"W_q", "W_k", "W_v"}
            assert scales["L1_mlp1"].keys() == {"W_ff1"}
            assert scales["L1_mlp2"].keys() == {"W_ff2"}
            assert scales["mlm_transform"].keys() == {"W_t"}
            for name, node in params.nodes.items():
                for weight, value in node.weights.items():
                    expected = value / scales.get(name, {}).get(weight, 1.0)
                    np.testing.assert_array_equal(
                        loaded.nodes[name].weights[weight], expected
                    )
                _assert_trees_equal(loaded.nodes[name].biases, node.biases)

    def test_runner_load_matches_off_float32(self, setup, tmp_path):
        checkpoint, data_dir = setup
        off = load_base_model(tiny_config(checkpoint, data_dir, tmp_path / "off"))
        full = load_base_model(
            tiny_config(checkpoint, data_dir, tmp_path / "full", pc_mupc="full")
        )
        assert full.source["params_sha256"] != off.source["params_sha256"]
        _, free_off, init_off = _feedforward(off.params, off.structure)
        _, free_full, init_full = _feedforward(full.params, full.structure)
        _states_close(free_off, free_full, "z_mu", 2e-6, 1e-6)
        energy_off = float(state_total_energy(off.params, init_off, off.structure))
        energy_full = float(state_total_energy(full.params, init_full, full.structure))
        assert energy_full == pytest.approx(energy_off, rel=1e-5)


class TestPlainConversion:
    def test_round_trip_and_function(self):
        with jax.enable_x64(True):
            off, full = build("off"), build("full")
            params = random_params(off)
            stored = mupc_function_preserving_params(full, params)
            plain = mupc_plain_params(full, stored)
            for a, b in zip(_leaves(plain), _leaves(params), strict=True):
                np.testing.assert_allclose(a, b, rtol=1e-15, atol=0.0)
            _, free_off, _ = _feedforward(params, off)
            _, free_plain, _ = _feedforward(plain, off)
            _states_close(free_off, free_plain, "z_mu", 1e-12, 1e-13)
            with pytest.raises(ValueError, match="mupc='full'"):
                mupc_plain_params(off, params)

    def test_plain_grads_divide_by_a(self):
        full = build("full")
        params = random_params(full)
        scales = mupc_weight_scales(full)
        grads = mupc_plain_grads(full, params)
        for name, node in params.nodes.items():
            for weight, value in node.weights.items():
                a = scales.get(name, {}).get(weight, 1.0)
                np.testing.assert_allclose(
                    np.asarray(grads.nodes[name].weights[weight]),
                    np.asarray(value) / a,
                    rtol=1e-6,
                )

    def test_weight_norms_are_of_plain_weights(self):
        off, full = build("off"), build("full")
        params = random_params(off)
        stored = mupc_function_preserving_params(full, params)
        for name, norm in module_weight_norms(params, off).items():
            assert module_weight_norms(stored, full)[name] == pytest.approx(
                norm, rel=1e-5
            )


class TestRefusals:
    def _with_node(self, structure, name, **changes):
        node = structure.nodes[name]
        info = replace(node.node_info, **changes)
        return structure._replace(
            nodes={**structure.nodes, name: node._with_graph_info(info)}
        )

    def test_needs_full_graph(self):
        params = random_params(build("off"))
        for mupc in ("off", "jacobian"):
            with pytest.raises(ValueError, match="mupc='full'"):
                mupc_function_preserving_params(build(mupc), params)

    def test_scaled_input_into_layernorm(self):
        full = build("full")
        info = full.nodes["L0_mlp1"].node_info
        pre_ln = self._with_node(
            full, "L0_mlp1", node_config={**info.node_config, "pre_ln": True}
        )
        with pytest.raises(ValueError, match="pre_ln=False"):
            mupc_weight_scales(pre_ln)
        mha = full.nodes["L0_mha"].node_info
        pre_norm = self._with_node(
            full, "L0_mha", node_config={**mha.node_config, "norm_position": "pre"}
        )
        with pytest.raises(ValueError, match="norm_position='post'"):
            mupc_weight_scales(pre_norm)

    def test_unknown_node_class(self):
        class FakeMlp2(Mlp2ResidualNode):
            pass

        full = build("full")
        fake = self._with_node(full, "L0_mlp2", node_class=FakeMlp2)
        assert (FakeMlp2, "in") not in MUPC_LINEAR_INPUTS
        with pytest.raises(ValueError, match="no function-preserving"):
            mupc_weight_scales(fake)

    def test_conflicting_factors_on_one_weight(self):
        full = build("full")
        info = full.nodes["L0_mlp1"].node_info
        (edge,) = info.scaling_config.forward_scale
        factors = replace(
            info.scaling_config,
            forward_scale={edge: 0.5, "extra->L0_mlp1:in": 0.25},
        )
        edges = {**full.edges, "extra->L0_mlp1:in": full.edges[edge]}
        broken = self._with_node(full, "L0_mlp1", scaling_config=factors)
        with pytest.raises(ValueError, match="different a"):
            mupc_weight_scales(broken._replace(edges=edges))


def _pc_step(params, structure, key=jax.random.PRNGKey(11)):
    grads, _metrics, _initial, _free, final = get_graph_param_gradient_pcn_mlm(
        params, bert_batch(), structure, key, "none"
    )
    return grads, final


class TestInferenceEquivalence:
    @pytest.mark.parametrize("steps", [1, 4, 12])
    def test_jacobian_and_full_give_identical_trajectories(self, steps):
        """Standard PC under full on W / a equals jacobian on W (float64)."""
        with jax.enable_x64(True):
            jac, full = build("jacobian", steps=steps), build("full", steps=steps)
            params = random_params(jac)
            loaded = mupc_function_preserving_params(full, params)
            grads_jac, final_jac = _pc_step(params, jac)
            grads_full, final_full = _pc_step(loaded, full)
            _states_close(final_jac, final_full, "z_latent", 1e-8, 1e-10)
            _states_close(final_jac, final_full, "z_mu", 1e-8, 1e-10)
            scales = mupc_weight_scales(full)
            for name, node in grads_jac.nodes.items():
                for weight, value in node.weights.items():
                    a = scales.get(name, {}).get(weight, 1.0)
                    np.testing.assert_allclose(
                        np.asarray(grads_full.nodes[name].weights[weight]),
                        a * np.asarray(value),
                        rtol=1e-8,
                        atol=1e-12,
                        err_msg=f"{name}.{weight}",
                    )
                for bias, value in node.biases.items():
                    np.testing.assert_allclose(
                        np.asarray(grads_full.nodes[name].biases[bias]),
                        np.asarray(value),
                        rtol=1e-8,
                        atol=1e-12,
                        err_msg=f"{name}.{bias}",
                    )

    def test_equivalence_with_active_clipping(self):
        """The norm clip sees the same latent gradients under both modes."""
        with jax.enable_x64(True):
            params = random_params(build("jacobian"))
            _, unclipped = _pc_step(params, build("jacobian", max_norm=1e9))
            jac, full = build("jacobian", max_norm=1e-2), build("full", max_norm=1e-2)
            _, final_jac = _pc_step(params, jac)
            _, final_full = _pc_step(
                mupc_function_preserving_params(full, params), full
            )
            moved = max(
                np.abs(
                    np.asarray(final_jac.nodes[n].z_latent)
                    - np.asarray(unclipped.nodes[n].z_latent)
                ).max()
                for n in final_jac.nodes
            )
            assert moved > 1e-3
            _states_close(final_jac, final_full, "z_latent", 1e-8, 1e-10)

    def test_alignment_is_the_same_under_full_and_jacobian(self):
        with jax.enable_x64(True):
            jac, full = build("jacobian"), build("full")
            params = random_params(jac)
            stored = mupc_function_preserving_params(full, params)
            batches, keys = [bert_batch()], [jax.random.PRNGKey(11)]
            left = gradient_alignment(
                make_alignment_fn(jac), params, batches, keys, jac
            )
            right = gradient_alignment(
                make_alignment_fn(full), stored, batches, keys, full
            )
            assert left["modules"].keys() == right["modules"].keys()
            for name, row in left["modules"].items():
                for key, value in row.items():
                    assert right["modules"][name][key] == pytest.approx(
                        value, rel=1e-8, abs=1e-12
                    ), (name, key)

    def test_trajectories_move(self):
        with jax.enable_x64(True):
            jac = build("jacobian", steps=12)
            params = random_params(jac)
            _, _, initial = _feedforward(params, jac, jax.random.PRNGKey(11))
            _, final = _pc_step(params, jac)
            moved = np.abs(
                np.asarray(final.nodes["L0_mha"].z_latent)
                - np.asarray(initial.nodes["L0_mha"].z_latent)
            ).max()
            assert moved > 1e-6


class TestJacobianAgainstOff:
    def test_forward_and_backprop_are_bit_identical(self):
        off, jac = build("off"), build("jacobian")
        params = random_params(off)
        _, free_off, init_off = _feedforward(params, off)
        _, free_jac, init_jac = _feedforward(params, jac)
        _states_close(free_off, free_jac, "z_mu", 0.0)
        _states_close(init_off, init_jac, "z_mu", 0.0)
        key = jax.random.PRNGKey(5)
        loss_off, grads_off, _ = _loss_and_grad_backprop_mlm_no_update(
            params, bert_batch(), off, key
        )
        loss_jac, grads_jac, _ = _loss_and_grad_backprop_mlm_no_update(
            params, bert_batch(), jac, key
        )
        assert float(loss_off) == float(loss_jac)
        _assert_trees_equal(grads_off, grads_jac)

    def test_standard_pc_differs_only_through_the_gelu_gain(self, monkeypatch):
        """With the GELU Jacobian gain set to 1 the jacobian mode is exactly off;
        with the real gain the PC trajectories differ."""
        off = build("off")
        params = random_params(off)
        grads_off, final_off = _pc_step(params, off)
        grads_jac, final_jac = _pc_step(params, build("jacobian"))
        diff = np.abs(
            np.asarray(final_off.nodes["L0_mha"].z_latent)
            - np.asarray(final_jac.nodes["L0_mha"].z_latent)
        ).max()
        assert diff > 0.0
        monkeypatch.setattr(
            GeluActivation, "jacobian_gain", staticmethod(lambda config=None: 1.0)
        )
        unit = build("jacobian")
        assert all(n.node_info.scaling_config is None for n in unit.nodes.values())
        grads_unit, final_unit = _pc_step(params, unit)
        _states_close(final_off, final_unit, "z_latent", 0.0)
        _assert_trees_equal(grads_off, grads_unit)


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    return make_setup(tmp_path_factory.mktemp("bert_mupc"))


def _two_updates(setup, tmp_path, **settings):
    checkpoint, data_dir = setup
    config = tiny_config(
        checkpoint, data_dir, tmp_path / "run", tokens_per_domain=40, **settings
    )
    config.validate()
    base = load_base_model(config)
    fn = make_gradient_fn(
        config.mode, base.structure, config.pc_forward_updates, config.pc_inference
    )
    params, _, log = train_stage(
        base.params,
        base.structure,
        _stage_stream(config, base, data_dir),
        config=config,
        stage=1,
        special_ids=base.special_ids,
        gradient_fn=fn,
        verbose=False,
    )
    return params, log


@pytest.fixture(scope="module")
def mupc_runs(setup, tmp_path_factory):
    checkpoint, data_dir = setup
    root = tmp_path_factory.mktemp("mupc_runs")
    off = tiny_config(checkpoint, data_dir, root / "off", domains=("news",))
    full = replace(off, output_dir=str(root / "full"), pc_mupc="full")
    return (
        (off, run_domain_forgetting(off, verbose=False)),
        (full, run_domain_forgetting(full, verbose=False)),
    )


class TestFullRunAnalyses:
    def test_load_run_converts_to_plain_weights(self, mupc_runs):
        _, ((full, result)) = mupc_runs
        context = load_run(full.output_dir)
        assert context.config.pc_mupc == "jacobian"
        assert context.structure.config["mupc"] == "jacobian"
        assert context.mupc_conversion["run"] == "full"
        record = json.loads((Path(full.output_dir) / "run_config.json").read_text())
        scales = record["mupc"]["weight_scales"]
        for stage, flat in enumerate(context.snapshots):
            raw = load_npz(Path(full.output_dir) / f"stage_{stage}.npz")
            assert set(raw) == set(flat)
            for key, value in raw.items():
                node, collection, leaf = key.split("/", 2)
                a = (
                    scales.get(node, {}).get(leaf, 1.0)
                    if collection == "weights"
                    else 1.0
                )
                np.testing.assert_allclose(flat[key], value * a, rtol=1e-6)
        drift = compute_drift(context.snapshots, context.structure, context.domains)
        plain = [
            {
                key: np.asarray(value, np.float64)
                * (
                    scales.get(key.split("/")[0], {}).get(key.split("/", 2)[2], 1.0)
                    if key.split("/")[1] == "weights"
                    else 1.0
                )
                for key, value in load_npz(
                    Path(full.output_dir) / f"stage_{stage}.npz"
                ).items()
            }
            for stage in range(len(context.snapshots))
        ]
        expected = compute_drift(plain, context.structure, context.domains)
        for module, stats in expected["transitions"][0]["modules"].items():
            for key, value in stats.items():
                assert drift["transitions"][0]["modules"][module][key] == pytest.approx(
                    value, rel=1e-5
                ), (module, key)

    def test_rollback_round_trip_restores_the_function(self, mupc_runs):
        _, (full, result) = mupc_runs
        context = load_run(full.output_dir)
        evaluate = MultiDomainEvaluator(context.structure, context.valid_sets, 4)
        for stage in range(len(context.snapshots)):
            ce = evaluate(context.params(stage))["news"]["masked_ce"]
            assert ce == pytest.approx(
                result["matrices"]["masked_ce"][stage][0], rel=1e-5
            )
        s0, s1 = context.snapshots
        rolled = rollback_flat(s1, s0, sorted(s0))
        ce = evaluate(context.params(rolled))["news"]["masked_ce"]
        assert ce == pytest.approx(result["matrices"]["masked_ce"][0][0], rel=1e-5)
        stored_structure = load_base_model(full).structure
        stored = [load_npz(Path(full.output_dir) / f"stage_{s}.npz") for s in range(2)]
        stored_evaluate = MultiDomainEvaluator(stored_structure, context.valid_sets, 4)
        keys = module_keys(s0, "L0_mlp1")
        plain_ce = evaluate(context.params(rollback_flat(s1, s0, keys)))
        stored_ce = stored_evaluate(
            params_from_flat(
                rollback_flat(stored[1], stored[0], keys), stored_structure
            )
        )
        assert plain_ce["news"]["masked_ce"] == pytest.approx(
            stored_ce["news"]["masked_ce"], rel=1e-5
        )

    def test_compare_refuses_with_a_mupc_message(self, mupc_runs):
        (off, _), (full, _) = mupc_runs
        left = json.loads((Path(full.output_dir) / "run_config.json").read_text())
        right = json.loads((Path(off.output_dir) / "run_config.json").read_text())
        problems = check_comparable_runs(left, right)
        assert any("μPC parameterisations differ" in p for p in problems)
        assert not any("source checkpoints" in p for p in problems)
        assert check_comparable_runs(left, left) == []

    def test_probe_converts_snapshots(self, mupc_runs):
        """A full run probed as jacobian (and an off run probed as full) is
        converted, so alignment matches probing in the run's own mode."""
        (off, _), (full, _) = mupc_runs
        common = {
            "domain": "news",
            "stage": 1,
            "num_batches": 1,
            "batch_size": 2,
            "pc_inference": "standard",
            "settings": [(0.3, 12)],
            "verbose": False,
        }
        for run, own, other in ((full, "full", "jacobian"), (off, "jacobian", "full")):
            mine = probe_alignment(**common, run_dir=run.output_dir, pc_mupc=own)
            converted = probe_alignment(**common, run_dir=run.output_dir, pc_mupc=other)
            run_mode = "full" if run is full else "off"
            assert converted["params_converted"] == f"{run_mode} -> {other}"
            assert "params_converted" not in mine
            for name, row in mine["settings"][0]["modules"].items():
                got = converted["settings"][0]["modules"][name]
                assert got["cosine"] == pytest.approx(row["cosine"], rel=1e-4, abs=1e-5)
                assert got["pc_norm"] == pytest.approx(row["pc_norm"], rel=1e-4)
                assert got["bp_norm"] == pytest.approx(row["bp_norm"], rel=1e-4)
        recorded = probe_alignment(**common, run_dir=full.output_dir)
        assert recorded["pc_mupc"] == "full"
        assert "params_converted" not in recorded


class TestRunner:
    @pytest.mark.parametrize("name", ["backprop", "standard"])
    def test_default_path_reproduces_head(self, setup, tmp_path, name):
        settings = STANDARD if name == "standard" else {}
        params, log = _two_updates(setup, tmp_path, **settings)
        again, explicit_log = _two_updates(setup, tmp_path, pc_mupc="off", **settings)
        _assert_trees_equal(params, again)
        assert explicit_log == log
        sha, losses, norms = HEAD_PINS[name]
        assert params_hash(params) == sha
        np.testing.assert_allclose([e["masked_loss"] for e in log], losses, rtol=1e-6)
        np.testing.assert_allclose([e["grad_norm"] for e in log], norms, rtol=1e-6)

    def test_jacobian_backprop_training_is_bit_identical(self, setup, tmp_path):
        params, log = _two_updates(setup, tmp_path)
        jac_params, jac_log = _two_updates(setup, tmp_path, pc_mupc="jacobian")
        _assert_trees_equal(params, jac_params)
        assert jac_log == log

    @pytest.mark.parametrize("mupc", ["jacobian", "full"])
    def test_standard_pc_trains(self, setup, tmp_path, mupc):
        _, log = _two_updates(setup, tmp_path, pc_mupc=mupc, **STANDARD)
        for entry in log:
            assert np.isfinite(entry["masked_loss"])
            assert entry["pc_energy_end"] < entry["pc_energy_start"]

    def test_validation(self, setup, tmp_path):
        checkpoint, data_dir = setup
        base = tiny_config(checkpoint, data_dir, tmp_path)
        for mupc in ("jacobian", "full"):
            replace(base, pc_mupc=mupc).validate()
            replace(base, pc_mupc=mupc, **STANDARD).validate()
            with pytest.raises(ValueError, match="not spiking"):
                replace(base, mode="pc", pc_mupc=mupc).validate()
            epc = {**STANDARD, "pc_inference": "epc"}
            with pytest.raises(ValueError, match="not epc"):
                replace(base, pc_mupc=mupc, **epc).validate()
            with pytest.raises(ValueError, match="pc_regime_interval"):
                replace(
                    base,
                    pc_mupc=mupc,
                    pc_regime_interval=5,
                    pc_regime_eta=0.1,
                    pc_regime_steps=2,
                ).validate()
        with pytest.raises(ValueError, match="pc_mupc must be one of"):
            replace(base, pc_mupc="on").validate()

    def test_identity_records_mupc_only_when_on(self, setup, tmp_path):
        checkpoint, data_dir = setup
        base = tiny_config(checkpoint, data_dir, tmp_path)
        assert "pc_mupc" not in base.identity()
        on = replace(base, pc_mupc="jacobian").identity()
        assert on["pc_mupc"] == "jacobian"
        assert {k: v for k, v in on.items() if k != "pc_mupc"} == base.identity()

    def test_full_run_records_and_refuses_mismatched_resume(self, mupc_runs):
        (off, off_result), (full, full_result) = mupc_runs
        np.testing.assert_allclose(
            full_result["matrices"]["masked_ce"][0],
            off_result["matrices"]["masked_ce"][0],
            rtol=1e-5,
        )
        run_config = json.loads((Path(full.output_dir) / "run_config.json").read_text())
        assert run_config["config"]["pc_mupc"] == "full"
        record = run_config["mupc"]
        assert record["mode"] == "full"
        assert set(record["weight_scales"]["L0_mha"]) == {"W_q", "W_k", "W_v"}
        off_config = json.loads((Path(off.output_dir) / "run_config.json").read_text())
        assert "mupc" not in off_config and "pc_mupc" not in off_config["config"]
        for other in ("jacobian", "off"):
            with pytest.raises(ValueError, match="cannot resume"):
                run_domain_forgetting(
                    replace(full, pc_mupc=other), resume=True, verbose=False
                )
        resumed = run_domain_forgetting(full, resume=True, verbose=False)
        assert resumed["matrices"] == full_result["matrices"]

    def test_probe(self, setup):
        checkpoint, data_dir = setup
        common = {
            "domain": "news",
            "checkpoint": str(checkpoint),
            "data_dir": str(data_dir),
            "num_batches": 1,
            "batch_size": 2,
            "pc_inference": "standard",
            "settings": [(0.01, 3)],
            "verbose": False,
        }
        plain = probe_alignment(**common)
        assert "pc_mupc" not in plain
        report = probe_alignment(**common, pc_mupc="jacobian")
        assert report["pc_mupc"] == "jacobian"
        assert report["settings"][0]["modules"]
        with pytest.raises(ValueError, match="not spiking"):
            probe_alignment(
                **{**common, "pc_inference": "spiking", "settings": [(0.0, None)]},
                pc_mupc="full",
            )


class TestCli:
    def test_run_flag(self):
        from examples import mlm_domain_forgetting as cli

        base = ["--checkpoint", "c.json", "--output_dir", "out"]
        args = cli.build_parser().parse_args(base)
        assert cli.config_from_args(args).pc_mupc == "off"
        args = cli.build_parser().parse_args([*base, "--pc_mupc", "full"])
        assert cli.config_from_args(args).pc_mupc == "full"
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args([*base, "--pc_mupc", "on"])

    def test_probe_flag(self):
        from examples import mlm_domain_forgetting as cli

        parser = cli.build_probe_parser()
        assert parser.parse_args(["--domain", "news"]).pc_mupc is None
        args = parser.parse_args(["--domain", "news", "--pc_mupc", "jacobian"])
        assert args.pc_mupc == "jacobian"
