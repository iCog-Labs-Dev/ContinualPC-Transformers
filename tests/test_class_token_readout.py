"""Unit tests for ClassTokenReadoutNode 

    y = LN(z_L^0)

Tests cover:
  - Construction-time validation (shape, embed_dim, cls_index, ln_eps)
  - Slot / muPC fan-in
  - Parameter initialization (LN gamma/beta, input shape validation)
  - Forward pass: CLS selection, LayerNorm, energy
  - Autodiff: CLS-only input gradients, LN parameter gradients
  - Graph builder and muPC integration
  - ViT-Base / Large / Huge geometries

Run with:
    pytest tests/test_class_token_readout.py -v
"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import pytest

from fabricpc.core.activations import IdentityActivation
from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.inference import InferenceSGD
from fabricpc.core.initializers import (
    NormalInitializer,
    OnesInitializer,
)
from fabricpc.core.mupc import MuPCConfig
from fabricpc.core.topology import Edge
from fabricpc.core.types import NodeInfo, NodeState
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.nodes import IdentityNode, Linear
from fabricpc.nodes.class_token_readout import ClassTokenReadoutNode
from fabricpc.utils.helpers import layernorm


EMBED_DIM = 768
SEQ_LEN = 197  # 1 CLS + 196 patches (ViT-Base/16 @ 224)


def _make_node_info(
    node_shape=(EMBED_DIM,),
    config=None,
    activation=None,
    energy_obj=None,
    latent_init=None,
    weight_init=None,
    in_edges=("encoder->readout:in",),
):
    if config is None:
        config = {
            "embed_dim": node_shape[0],
            "cls_index": 0,
            "ln_eps": 1e-6,
        }
    return NodeInfo(
        name="test_readout",
        shape=node_shape,
        node_type=ClassTokenReadoutNode.__name__,
        node_class=ClassTokenReadoutNode,
        node_config=config,
        activation=activation or IdentityActivation(),
        energy=energy_obj or GaussianEnergy(),
        latent_init=latent_init or NormalInitializer(),
        weight_init=weight_init or OnesInitializer(),
        slots={},
        in_degree=len(in_edges),
        out_degree=0,
        in_edges=in_edges,
        out_edges=(),
    )


def _make_state(key, batch_size, node_shape=(EMBED_DIM,)):
    z_latent = jax.random.normal(key, (batch_size, *node_shape))
    zeros = jnp.zeros_like(z_latent)
    return NodeState(
        z_latent=z_latent,
        z_mu=zeros,
        error=zeros,
        energy=jnp.zeros(batch_size),
        pre_activation=zeros,
        latent_grad=zeros,
    )


def _make_params(
    key=jax.random.PRNGKey(0),
    node_shape=(EMBED_DIM,),
    seq_len=SEQ_LEN,
    cls_index=0,
    ln_eps=1e-6,
    weight_init=None,
):
    config = {
        "embed_dim": node_shape[0],
        "cls_index": cls_index,
        "ln_eps": ln_eps,
    }
    return ClassTokenReadoutNode.initialize_params(
        key=key,
        node_shape=node_shape,
        input_shapes={"encoder->readout:in": (seq_len, node_shape[0])},
        weight_init=weight_init or OnesInitializer(),
        config=config,
    )


def _forward(
    params,
    x,
    state,
    *,
    cls_index=0,
    ln_eps=1e-6,
    node_shape=None,
):
    if node_shape is None:
        node_shape = (x.shape[-1],)
    node_info = _make_node_info(
        node_shape=node_shape,
        config={
            "embed_dim": node_shape[0],
            "cls_index": cls_index,
            "ln_eps": ln_eps,
        },
    )
    inputs = {node_info.in_edges[0]: x}
    return ClassTokenReadoutNode.forward(params, inputs, state, node_info)



class TestClassTokenReadoutConstruction:
    def test_valid_construction(self):
        node = ClassTokenReadoutNode(
            shape=(EMBED_DIM,),
            name="readout",
            embed_dim=EMBED_DIM,
        )
        assert node.shape == (EMBED_DIM,)
        assert node.name == "readout"
        assert node._extra_config["cls_index"] == 0
        assert node._extra_config["ln_eps"] == 1e-6
        assert ClassTokenReadoutNode.DEFAULT_ACTIVATION is IdentityActivation
        assert ClassTokenReadoutNode.DEFAULT_ENERGY is GaussianEnergy

    def test_shape_must_be_1d(self):
        with pytest.raises(ValueError, match="exactly 1 dimension"):
            ClassTokenReadoutNode(
                shape=(SEQ_LEN, EMBED_DIM),
                name="bad",
                embed_dim=EMBED_DIM,
            )

    def test_embed_dim_must_match_shape(self):
        with pytest.raises(ValueError, match="must equal embed_dim"):
            ClassTokenReadoutNode(
                shape=(512,),
                name="bad",
                embed_dim=EMBED_DIM,
            )

    def test_cls_index_must_be_nonneg_int(self):
        with pytest.raises(
            ValueError, match="cls_index must be a non-negative integer"
        ):
            ClassTokenReadoutNode(
                shape=(EMBED_DIM,),
                name="bad",
                embed_dim=EMBED_DIM,
                cls_index=-1,
            )
        with pytest.raises(
            ValueError, match="cls_index must be a non-negative integer"
        ):
            ClassTokenReadoutNode(
                shape=(EMBED_DIM,),
                name="bad",
                embed_dim=EMBED_DIM,
                cls_index=True,
            )

    def test_ln_eps_must_be_positive(self):
        for bad in (0.0, -1e-6, True, None):
            with pytest.raises(ValueError, match="ln_eps must be a positive float"):
                ClassTokenReadoutNode(
                    shape=(EMBED_DIM,),
                    name="bad",
                    embed_dim=EMBED_DIM,
                    ln_eps=bad,
                )



class TestClassTokenReadoutSlots:
    def test_single_input_slot(self):
        slots = ClassTokenReadoutNode.get_slots()
        assert set(slots) == {"in"}
        assert slots["in"].is_multi_input is False
        assert slots["in"].is_variance_scalable is False
        assert slots["in"].is_skip_connection is False

    def test_fan_in(self):
        fan_in = ClassTokenReadoutNode.get_weight_fan_in(
            source_shape=(SEQ_LEN, EMBED_DIM),
            config={"embed_dim": EMBED_DIM},
        )
        assert fan_in == 1



class TestClassTokenReadoutParams:
    def test_ln_gamma_shape(self):
        params = _make_params()
        assert params.weights["ln_gamma"].shape == (EMBED_DIM,)
        assert jnp.allclose(params.weights["ln_gamma"], 1.0)

    def test_ln_beta_shape(self):
        params = _make_params()
        assert params.biases["ln_beta"].shape == (EMBED_DIM,)
        assert jnp.allclose(params.biases["ln_beta"], 0.0)

    def test_input_must_be_2d_sequence(self):
        with pytest.raises(ValueError, match="2-D sequence input"):
            ClassTokenReadoutNode.initialize_params(
                key=jax.random.PRNGKey(0),
                node_shape=(EMBED_DIM,),
                input_shapes={"src->dst:in": (EMBED_DIM,)},
                weight_init=OnesInitializer(),
                config={"cls_index": 0, "ln_eps": 1e-6, "embed_dim": EMBED_DIM},
            )

    def test_embed_dim_must_match_input(self):
        with pytest.raises(ValueError, match="embed_dim mismatch"):
            ClassTokenReadoutNode.initialize_params(
                key=jax.random.PRNGKey(0),
                node_shape=(EMBED_DIM,),
                input_shapes={"src->dst:in": (SEQ_LEN, 512)},
                weight_init=OnesInitializer(),
                config={"cls_index": 0, "ln_eps": 1e-6, "embed_dim": EMBED_DIM},
            )

    def test_exactly_one_input_edge(self):
        with pytest.raises(ValueError, match="exactly one input edge"):
            ClassTokenReadoutNode.initialize_params(
                key=jax.random.PRNGKey(0),
                node_shape=(8,),
                input_shapes={},
                weight_init=OnesInitializer(),
                config={"cls_index": 0, "ln_eps": 1e-6, "embed_dim": 8},
            )

    def test_cls_index_must_fit_sequence(self):
        with pytest.raises(ValueError, match="out of range"):
            _make_params(seq_len=8, cls_index=8, node_shape=(4,))



class TestClassTokenReadoutForward:
    def test_output_shape(self):
        B = 2
        key = jax.random.PRNGKey(0)
        k_x, k_s, k_p = jax.random.split(key, 3)
        x = jax.random.normal(k_x, (B, SEQ_LEN, EMBED_DIM))
        params = _make_params(k_p)
        state = _make_state(k_s, B)
        _, new_state = _forward(params, x, state)

        assert new_state.z_mu.shape == (B, EMBED_DIM)
        assert new_state.error.shape == (B, EMBED_DIM)
        assert new_state.energy.shape == (B,)
        assert new_state.pre_activation.shape == (B, EMBED_DIM)

    def test_cls_token_selection(self):
        """Output matches LN of position 0, not of any other position."""
        B, S, D = 2, 8, 16
        key = jax.random.PRNGKey(1)
        k_x, k_s, k_p = jax.random.split(key, 3)
        x = jax.random.normal(k_x, (B, S, D))
        params = _make_params(k_p, node_shape=(D,), seq_len=S)
        state = _make_state(k_s, B, (D,))
        _, new_state = _forward(params, x, state, node_shape=(D,))

        expected = layernorm(
            x[:, 0, :],
            params.weights["ln_gamma"],
            params.biases["ln_beta"],
            eps=1e-6,
        )
        other = layernorm(
            x[:, 1, :],
            params.weights["ln_gamma"],
            params.biases["ln_beta"],
            eps=1e-6,
        )
        assert jnp.allclose(new_state.z_mu, expected, atol=1e-5)
        assert not jnp.allclose(new_state.z_mu, other, atol=1e-4)
        assert jnp.allclose(new_state.pre_activation, expected, atol=1e-6)

    def test_layernorm_applied(self):
        B, S, D = 3, 10, 32
        key = jax.random.PRNGKey(2)
        k_x, k_s, k_p = jax.random.split(key, 3)
        x = jax.random.normal(k_x, (B, S, D))
        params = _make_params(k_p, node_shape=(D,), seq_len=S)
        state = _make_state(k_s, B, (D,))
        _, new_state = _forward(params, x, state, node_shape=(D,))

        expected = layernorm(
            x[:, 0, :],
            params.weights["ln_gamma"],
            params.biases["ln_beta"],
            eps=1e-6,
        )
        assert jnp.allclose(new_state.z_mu, expected, atol=1e-5)

    def test_pre_activation_is_value_given_to_activation(self):
        """pre_activation must contain the LayerNorm output passed to activation."""
        batch_size, seq_len, embed_dim = 1, 2, 4
        x = jnp.array(
            [
                [
                    [1.0, 2.0, 4.0, 8.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            ],
            dtype=jnp.float32,
        )
        params = _make_params(node_shape=(embed_dim,), seq_len=seq_len)
        state = _make_state(
            jax.random.PRNGKey(0), batch_size, (embed_dim,)
        )

        _, new_state = _forward(
            params,
            x,
            state,
            node_shape=(embed_dim,),
        )

        expected_pre_activation = layernorm(
            x[:, 0, :],
            params.weights["ln_gamma"],
            params.biases["ln_beta"],
            eps=1e-6,
        )

        assert jnp.allclose(
            new_state.pre_activation,
            expected_pre_activation,
            atol=1e-6,
        )
        assert jnp.allclose(
            new_state.pre_activation,
            new_state.z_mu,
            atol=1e-6,
        )

    def test_energy_is_nonneg(self):
        B, S, D = 2, 8, 16
        key = jax.random.PRNGKey(3)
        k_x, k_s, k_p = jax.random.split(key, 3)
        x = jax.random.normal(k_x, (B, S, D))
        params = _make_params(k_p, node_shape=(D,), seq_len=S)
        state = _make_state(k_s, B, (D,))
        total, new_state = _forward(params, x, state, node_shape=(D,))

        assert jnp.all(new_state.energy >= 0)
        expected = 0.5 * jnp.sum((state.z_latent - new_state.z_mu) ** 2, axis=-1)
        assert jnp.allclose(new_state.energy, expected, atol=1e-5)
        assert jnp.allclose(total, jnp.sum(expected), atol=1e-5)

    def test_custom_cls_index(self):
        B, S, D = 2, 10, 16
        cls_index = 5
        key = jax.random.PRNGKey(4)
        k_x, k_s, k_p = jax.random.split(key, 3)
        x = jax.random.normal(k_x, (B, S, D))
        params = _make_params(k_p, node_shape=(D,), seq_len=S, cls_index=cls_index)
        state = _make_state(k_s, B, (D,))
        _, new_state = _forward(params, x, state, cls_index=cls_index, node_shape=(D,))

        expected = layernorm(
            x[:, cls_index, :],
            params.weights["ln_gamma"],
            params.biases["ln_beta"],
            eps=1e-6,
        )
        assert jnp.allclose(new_state.z_mu, expected, atol=1e-5)
        assert jnp.allclose(new_state.pre_activation, expected, atol=1e-6)

    def test_ln_eps_affects_output(self):
        """Near-zero variance makes the eps term dominate the LN denominator."""
        B, S, D = 1, 4, 8
        x = jnp.ones((B, S, D)) * 1e-8
        x = x.at[:, 0, 0].add(1e-8)
        params = _make_params(node_shape=(D,), seq_len=S)
        state = _make_state(jax.random.PRNGKey(5), B, (D,))

        _, state_small = _forward(params, x, state, ln_eps=1e-12, node_shape=(D,))
        _, state_large = _forward(params, x, state, ln_eps=1.0, node_shape=(D,))
        assert not jnp.allclose(state_small.z_mu, state_large.z_mu, atol=1e-8)



class TestClassTokenReadoutAutodiff:
    def test_jit_forward(self):
        B, S, D = 2, 8, 16
        key = jax.random.PRNGKey(0)
        k_x, k_s, k_p = jax.random.split(key, 3)
        x = jax.random.normal(k_x, (B, S, D))
        params = _make_params(k_p, node_shape=(D,), seq_len=S)
        state = _make_state(k_s, B, (D,))
        node_info = _make_node_info(node_shape=(D,))
        inputs = {node_info.in_edges[0]: x}

        def forward_jit(p, inp, st):
            _, new_state = ClassTokenReadoutNode.forward(p, inp, st, node_info)
            return new_state

        new_state = jax.jit(forward_jit)(params, inputs, state)
        assert new_state.z_mu.shape == (B, D)

    def test_grad_wrt_params(self):
        B, S, D = 2, 8, 16
        key = jax.random.PRNGKey(1)
        k_x, k_s, k_p = jax.random.split(key, 3)
        x = jax.random.normal(k_x, (B, S, D))
        params = _make_params(k_p, node_shape=(D,), seq_len=S)
        state = _make_state(k_s, B, (D,))
        node_info = _make_node_info(node_shape=(D,))
        inputs = {node_info.in_edges[0]: x}

        def loss_fn(p):
            energy, _ = ClassTokenReadoutNode.forward(p, inputs, state, node_info)
            return energy

        grads = jax.grad(loss_fn)(params)
        assert grads.weights["ln_gamma"].shape == (D,)
        assert grads.biases["ln_beta"].shape == (D,)
        assert jnp.any(grads.weights["ln_gamma"] != 0)
        assert jnp.any(grads.biases["ln_beta"] != 0)

    def test_grad_wrt_input(self):
        """Local dE/dx is supported only on the CLS position."""
        B, S, D = 2, 8, 16
        cls_index = 0
        key = jax.random.PRNGKey(2)
        k_x, k_s, k_p = jax.random.split(key, 3)
        x = jax.random.normal(k_x, (B, S, D))
        params = _make_params(k_p, node_shape=(D,), seq_len=S)
        state = _make_state(k_s, B, (D,))
        node_info = _make_node_info(node_shape=(D,))

        def loss_fn(seq):
            energy, _ = ClassTokenReadoutNode.forward(
                params, {node_info.in_edges[0]: seq}, state, node_info
            )
            return energy

        grad_x = jax.grad(loss_fn)(x)
        assert grad_x.shape == x.shape
        assert jnp.any(grad_x[:, cls_index, :] != 0)
        mask = jnp.ones((S,), dtype=bool).at[cls_index].set(False)
        assert jnp.allclose(grad_x[:, mask, :], 0.0)

    def test_fabricpc_local_gradient_apis(self):
        B, S, D = 2, 8, 16
        key = jax.random.PRNGKey(3)
        k_x, k_s, k_p = jax.random.split(key, 3)
        x = jax.random.normal(k_x, (B, S, D))
        params = _make_params(k_p, node_shape=(D,), seq_len=S)
        state = _make_state(k_s, B, (D,))
        node_info = replace(
            _make_node_info(node_shape=(D,)),
            out_degree=1,
            out_edges=("readout->head:in",),
        )
        edge_key = node_info.in_edges[0]
        inputs = {edge_key: x}

        _, input_grads, self_grad = ClassTokenReadoutNode.forward_and_latent_grads(
            params, inputs, state, node_info, is_clamped=False
        )
        _, param_grads = ClassTokenReadoutNode.forward_and_weight_grads(
            params, inputs, state, node_info
        )

        assert input_grads[edge_key].shape == x.shape
        assert jnp.any(input_grads[edge_key][:, 0, :] != 0)
        assert jnp.allclose(input_grads[edge_key][:, 1:, :], 0.0)
        assert jnp.any(self_grad != 0)
        assert jnp.any(param_grads.weights["ln_gamma"] != 0)
        assert jnp.any(param_grads.biases["ln_beta"] != 0)
        assert jnp.all(jnp.isfinite(param_grads.weights["ln_gamma"]))


def _build_readout_graph(seq_len=8, embed_dim=16, num_classes=4, with_mupc=False):
    encoder = IdentityNode(shape=(seq_len, embed_dim), name="encoder_out")
    readout = ClassTokenReadoutNode(
        shape=(embed_dim,),
        name="readout",
        embed_dim=embed_dim,
    )
    head = Linear(shape=(num_classes,), name="head")
    return graph(
        nodes=[encoder, readout, head],
        edges=[
            Edge(source=encoder, target=readout.slot("in")),
            Edge(source=readout, target=head.slot("in")),
        ],
        task_map=TaskMap(x=encoder, y=head),
        inference=InferenceSGD(eta_infer=0.05, infer_steps=3),
        scaling=MuPCConfig() if with_mupc else None,
    )


class TestClassTokenReadoutGraphIntegration:
    def test_real_graph_init_and_feedforward(self, rng_key):
        seq_len, embed_dim, num_classes = 8, 16, 4
        structure = _build_readout_graph(seq_len, embed_dim, num_classes)
        params = initialize_params(structure, rng_key)
        batch = 2
        x = jax.random.normal(rng_key, (batch, seq_len, embed_dim))
        y = jnp.zeros((batch, num_classes))
        state = initialize_graph_state(
            structure,
            batch_size=batch,
            rng_key=rng_key,
            clamps={"encoder_out": x, "head": y},
            params=params,
        )

        readout_params = params.nodes["readout"]
        assert readout_params.weights["ln_gamma"].shape == (embed_dim,)
        assert readout_params.biases["ln_beta"].shape == (embed_dim,)

        z_mu = state.nodes["readout"].z_mu
        assert z_mu.shape == (batch, embed_dim)
        expected = layernorm(
            x[:, 0, :],
            readout_params.weights["ln_gamma"],
            readout_params.biases["ln_beta"],
            eps=1e-6,
        )
        assert jnp.allclose(z_mu, expected, atol=1e-5)
        assert jnp.all(jnp.isfinite(z_mu))

    def test_mupc_scaling(self):
        embed_dim = 16
        structure = _build_readout_graph(embed_dim=embed_dim, with_mupc=True)
        readout_info = structure.nodes["readout"].node_info
        edge_key = readout_info.in_edges[0]
        scaling = readout_info.scaling_config

        assert scaling is not None
        assert edge_key not in scaling.forward_scale
        assert edge_key not in scaling.topdown_grad_scale
        assert edge_key not in scaling.weight_grad_scale

    def test_mupc_preserves_class_token_readout_equation(self, rng_key):
        """muPC must preserve the advertised y = LN(z_L^0) readout."""
        seq_len = 2
        embed_dim = 4
        num_classes = 2
        structure = _build_readout_graph(
            seq_len=seq_len,
            embed_dim=embed_dim,
            num_classes=num_classes,
            with_mupc=True,
        )
        params = initialize_params(structure, rng_key)

        cls_token = jnp.array(
            [0.0, 1e-3, -1e-3, 2e-3],
            dtype=jnp.float32,
        )
        x = jnp.stack(
            [cls_token, jnp.zeros_like(cls_token)],
            axis=0,
        )[None, ...]
        y = jnp.zeros((1, num_classes))

        state = initialize_graph_state(
            structure,
            batch_size=1,
            rng_key=rng_key,
            clamps={"encoder_out": x, "head": y},
            params=params,
        )

        readout_params = params.nodes["readout"]
        expected = layernorm(
            x[:, 0, :],
            readout_params.weights["ln_gamma"],
            readout_params.biases["ln_beta"],
            eps=1e-6,
        )
        actual = state.nodes["readout"].z_mu

        assert jnp.allclose(actual, expected, rtol=1e-6, atol=1e-7)


class TestClassTokenReadoutVariants:
    @pytest.mark.parametrize(
        "embed_dim,seq_len",
        [
            (768, 197),  # ViT-Base/16 @ 224
            (1024, 197),  # ViT-Large/16 @ 224
            (1280, 257),  # ViT-Huge/14 @ 224
        ],
    )
    def test_variant_shapes(self, embed_dim, seq_len):
        B = 1
        key = jax.random.PRNGKey(0)
        k_x, k_s, k_p = jax.random.split(key, 3)
        x = jax.random.normal(k_x, (B, seq_len, embed_dim))
        params = _make_params(k_p, node_shape=(embed_dim,), seq_len=seq_len)
        state = _make_state(k_s, B, (embed_dim,))
        _, new_state = _forward(params, x, state, node_shape=(embed_dim,))
        assert new_state.z_mu.shape == (B, embed_dim)
        expected = layernorm(
            x[:, 0, :],
            params.weights["ln_gamma"],
            params.biases["ln_beta"],
            eps=1e-6,
        )
        assert jnp.allclose(new_state.z_mu, expected, atol=1e-5)
