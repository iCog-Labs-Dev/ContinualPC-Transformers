"""Tests for InferencePCALM (augmented Lagrangian predictive coding).

Covers the reduction to standard PC at ``alpha = 0``, dual bookkeeping
(clamped / source nodes, per-run reset), convergence of the duals to the
backprop adjoint on a deep linear chain, step-for-step equivalence with a
plain-JAX port of the reference ``run_pcalm`` / ``method_grad``
(Seely & Gould 2026, arXiv 2605.31022), explicit-gradient nodes, the muPC fixed point,
the dashboarding history paths, the encoder MLM graph, and argument
validation.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from fabricpc.core.activations import IdentityActivation, TanhActivation
from fabricpc.core.inference import (
    InferencePCALM,
    InferenceSGD,
    InferenceSGDNormClip,
    run_inference,
)
from fabricpc.core.initializers import NormalInitializer
from fabricpc.core.mupc import MuPCConfig
from fabricpc.core.learning import compute_local_weight_gradients
from fabricpc.core.topology import Edge
from fabricpc.core.state_ops import drop_duals
from fabricpc.core.types import GraphParams, NodeParams
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.nodes import Linear
from fabricpc.nodes.encoder_transformer import create_encoder_mlm_model
from fabricpc.nodes.identity import IdentityNode
from fabricpc.nodes.linear_explicit_grad import LinearExplicitGrad
from fabricpc.nodes.transformer_v2 import EmbeddingNode
from fabricpc.training.train import get_graph_param_gradient
from fabricpc.training.train_autoregressive import (
    get_graph_param_gradient_autoregressive,
    train_step_autoregressive,
)
from fabricpc.training.train_backprop import compute_loss_autoregressive
from fabricpc.utils.dashboarding.inference_tracking import (
    run_inference_with_full_history,
    run_inference_with_history,
)
from fabricpc.utils.data import MlmTextBatcher

NODE_FIELDS = ("z_latent", "z_mu", "error", "energy", "pre_activation", "latent_grad")


class _ReluInputLinear(Linear):
    """Linear node applying ReLU to its input: z_mu = relu(x) @ W (reference block)."""

    @staticmethod
    def forward(params, inputs, state, node_info):
        relu_inputs = {k: jax.nn.relu(x) for k, x in inputs.items()}
        return Linear.forward(params, relu_inputs, state, node_info)


def _chain(
    inference,
    depth=3,
    width=4,
    activation=IdentityActivation(),
    node_cls=Linear,
):
    """x -> h0 -> ... -> h{depth-1} -> y, bias-free, Gaussian energy."""
    x = IdentityNode(shape=(width,), name="x")
    nodes, edges, prev = [x], [], x
    std = 1.0 / math.sqrt(width)
    for i in range(depth):
        h = node_cls(
            shape=(width,),
            name=f"h{i}",
            activation=activation,
            use_bias=False,
            weight_init=NormalInitializer(std=std),
        )
        nodes.append(h)
        edges.append(Edge(source=prev, target=h.slot("in")))
        prev = h
    y = node_cls(
        shape=(width,),
        name="y",
        use_bias=False,
        weight_init=NormalInitializer(std=std),
    )
    nodes.append(y)
    edges.append(Edge(source=prev, target=y.slot("in")))
    return graph(
        nodes=nodes, edges=edges, task_map=TaskMap(x=x, y=y), inference=inference
    )


def _chain_names(depth):
    return [f"h{i}" for i in range(depth)] + ["y"]


def _edge(prev, name):
    return f"{prev}->{name}:in"


def _chain_edges(depth):
    names = _chain_names(depth)
    return {n: _edge(p, n) for p, n in zip(["x"] + names[:-1], names)}


def _batch(key, batch_size, width_in, width_out):
    kx, ky = jax.random.split(key)
    return {
        "x": jax.random.normal(kx, (batch_size, width_in)),
        "y": jax.random.normal(ky, (batch_size, width_out)),
    }


def _pc_grads(structure, params, batch, key):
    grads, _, final_state = jax.jit(
        lambda p, b: get_graph_param_gradient(p, b, structure, key)
    )(params, batch)
    return grads, final_state


def _cos_ratio(a, b):
    a, b = jnp.ravel(a), jnp.ravel(b)
    na, nb = jnp.linalg.norm(a), jnp.linalg.norm(b)
    return float(jnp.dot(a, b) / (na * nb)), float(na / nb)


def _tree_dot(a, b):
    return sum(
        float(jnp.vdot(x, y))
        for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b))
    )


# =============================================================================
# alpha = 0 reduces to standard PC
# =============================================================================


class TestAlphaZeroIsStandardPC:
    @pytest.mark.parametrize(
        "pcalm_kwargs, baseline",
        [
            ({}, InferenceSGD(eta_infer=0.1, infer_steps=7)),
            (
                {"max_norm": 0.3},
                InferenceSGDNormClip(eta_infer=0.1, infer_steps=7, max_norm=0.3),
            ),
        ],
    )
    @pytest.mark.parametrize("dual_timing", ["sequential", "simultaneous"])
    def test_bit_identical_state_and_grads(
        self, rng_key, pcalm_kwargs, baseline, dual_timing
    ):
        base = _chain(baseline, activation=TanhActivation())
        alm = _chain(
            InferencePCALM(
                eta_infer=0.1,
                infer_steps=7,
                alpha=0.0,
                dual_timing=dual_timing,
                **pcalm_kwargs,
            ),
            activation=TanhActivation(),
        )
        params = initialize_params(base, rng_key)
        batch = _batch(jax.random.PRNGKey(1), 5, 4, 4)

        g_base, s_base = _pc_grads(base, params, batch, rng_key)
        g_alm, s_alm = _pc_grads(alm, params, batch, rng_key)

        for name in base.nodes:
            for field in NODE_FIELDS:
                assert jnp.array_equal(
                    getattr(s_base.nodes[name], field),
                    getattr(s_alm.nodes[name], field),
                ), (name, field)
        for a, b in zip(
            jax.tree_util.tree_leaves(g_base), jax.tree_util.tree_leaves(g_alm)
        ):
            assert jnp.array_equal(a, b)


# =============================================================================
# Dual bookkeeping
# =============================================================================


class TestDualBookkeeping:
    def _setup(self, rng_key, alpha=1.0):
        structure = _chain(InferencePCALM(eta_infer=0.1, infer_steps=6, alpha=alpha))
        params = initialize_params(structure, rng_key)
        batch = _batch(jax.random.PRNGKey(1), 3, 4, 4)
        return structure, params, batch

    def test_fresh_state_has_zero_duals(self, rng_key):
        structure, params, batch = self._setup(rng_key)
        state = initialize_graph_state(
            structure, 3, rng_key, clamps=batch, params=params
        )
        for name, ns in state.nodes.items():
            assert ns.dual is not None
            assert ns.dual.shape == ns.latent_grad.shape
            assert jnp.issubdtype(ns.dual.dtype, jnp.floating)
            assert not jnp.any(ns.dual)

    def test_standard_pc_state_has_no_dual(self, rng_key):
        structure = _chain(InferenceSGD())
        params = initialize_params(structure, rng_key)
        state = initialize_graph_state(structure, 3, rng_key, params=params)
        assert all(ns.dual is None for ns in state.nodes.values())

    def test_dual_zero_on_clamped_and_source_nodes(self, rng_key):
        structure, params, batch = self._setup(rng_key)
        _, final_state = _pc_grads(structure, params, batch, rng_key)

        assert not jnp.any(final_state.nodes["x"].dual)
        assert not jnp.any(final_state.nodes["y"].dual)
        for name in ("h0", "h1", "h2"):
            assert jnp.any(final_state.nodes[name].dual)

    def test_unclamped_output_has_no_dual(self, rng_key):
        structure, params, batch = self._setup(rng_key)
        clamps = {"x": batch["x"]}
        state = initialize_graph_state(
            structure, 3, rng_key, clamps=clamps, params=params
        )
        final_state = run_inference(params, state, clamps, structure)
        assert not jnp.any(final_state.nodes["y"].dual)

    def test_duals_reset_at_start_of_inference(self, rng_key):
        structure, params, batch = self._setup(rng_key)
        state = initialize_graph_state(
            structure, 3, rng_key, clamps=batch, params=params
        )
        stale = state._replace(
            nodes={
                n: ns._replace(dual=jnp.full_like(ns.dual, 3.0))
                for n, ns in state.nodes.items()
            }
        )
        fresh_final = run_inference(params, state, batch, structure)
        stale_final = run_inference(params, stale, batch, structure)
        for name in structure.nodes:
            assert jnp.array_equal(
                fresh_final.nodes[name].z_latent, stale_final.nodes[name].z_latent
            )
            assert jnp.array_equal(
                fresh_final.nodes[name].dual, stale_final.nodes[name].dual
            )

    def test_single_step_is_primal_only(self, rng_key):
        structure = _chain(InferencePCALM(eta_infer=0.1, infer_steps=1, alpha=1.0))
        params = initialize_params(structure, rng_key)
        batch = _batch(jax.random.PRNGKey(1), 3, 4, 4)
        _, final_state = _pc_grads(structure, params, batch, rng_key)
        assert all(not jnp.any(ns.dual) for ns in final_state.nodes.values())

    def test_non_dual_inference_ignores_leftover_duals(self, rng_key):
        structure = _chain(InferenceSGD(eta_infer=0.1, infer_steps=6))
        params = initialize_params(structure, rng_key)
        batch = _batch(jax.random.PRNGKey(1), 3, 4, 4)
        state = initialize_graph_state(
            structure, 3, rng_key, clamps=batch, params=params
        )
        stale = state._replace(
            nodes={
                n: ns._replace(dual=jnp.full_like(ns.latent_grad, 3.0))
                for n, ns in state.nodes.items()
            }
        )
        clean_final = run_inference(params, state, batch, structure)
        stale_final = run_inference(params, stale, batch, structure)
        for name in structure.nodes:
            assert stale_final.nodes[name].dual is None
            for field in NODE_FIELDS:
                assert jnp.array_equal(
                    getattr(clean_final.nodes[name], field),
                    getattr(stale_final.nodes[name], field),
                ), (name, field)


class TestHistoryPathsMatchRunInference:
    @pytest.mark.parametrize("dual_timing", ["sequential", "simultaneous"])
    def test_history_final_state_matches(self, rng_key, dual_timing):
        structure = _chain(
            InferencePCALM(
                eta_infer=0.1, infer_steps=6, alpha=1.0, dual_timing=dual_timing
            ),
            activation=TanhActivation(),
        )
        params = initialize_params(structure, rng_key)
        batch = _batch(jax.random.PRNGKey(1), 3, 4, 4)
        state = initialize_graph_state(
            structure, 3, rng_key, clamps=batch, params=params
        )
        no_dual = drop_duals(state)

        expected = run_inference(params, state, batch, structure)
        expected_grads = compute_local_weight_gradients(params, expected, structure)
        scanned, metrics = run_inference_with_history(params, no_dual, batch, structure)
        looped, history = run_inference_with_full_history(
            params, no_dual, batch, structure
        )

        assert len(history) == 6
        assert metrics["h0"]["energy"].shape == (6,)
        for final in (scanned, looped):
            grads = compute_local_weight_gradients(params, final, structure)
            for name in structure.nodes:
                for field in ("z_latent", "dual"):
                    np.testing.assert_allclose(
                        getattr(final.nodes[name], field),
                        getattr(expected.nodes[name], field),
                        rtol=1e-6,
                        atol=1e-6,
                    )
            for a, b in zip(
                jax.tree_util.tree_leaves(grads),
                jax.tree_util.tree_leaves(expected_grads),
            ):
                np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)


# =============================================================================
# Deep linear chain: duals -> BP adjoint, weight grads -> BP grads
# =============================================================================


def _orthogonal_params(structure, key):
    """Replace every weight matrix with a random orthogonal one (well-conditioned chain)."""
    params = initialize_params(structure, key)
    keys = jax.random.split(jax.random.PRNGKey(7), len(params.nodes))
    nodes = {}
    for k, (name, node_params) in zip(keys, params.nodes.items()):
        weights = {}
        for edge_key, w in node_params.weights.items():
            q, _ = jnp.linalg.qr(jax.random.normal(k, w.shape))
            weights[edge_key] = q
        nodes[name] = NodeParams(weights=weights, biases=node_params.biases)
    return GraphParams(nodes=nodes)


class TestDeepLinearChainMatchesBackprop:
    DEPTH, WIDTH, BATCH = 6, 8, 8
    ETA, STEPS = 0.2, 200

    def _bp(self, params, batch):
        edges = _chain_edges(self.DEPTH)
        names = _chain_names(self.DEPTH)

        def loss_from(z, start):
            for name in names[start:]:
                z = z @ params.nodes[name].weights[edges[name]]
            return 0.5 * jnp.sum((z - batch["y"]) ** 2)

        acts, z = {}, batch["x"]
        for name in names[:-1]:
            z = z @ params.nodes[name].weights[edges[name]]
            acts[name] = z
        adjoints = {
            name: jax.grad(loss_from)(acts[name], i + 1)
            for i, name in enumerate(names[:-1])
        }
        bp_grads = jax.grad(lambda p: self._loss(p, batch))(params)
        return adjoints, bp_grads

    def _loss(self, params, batch):
        edges = _chain_edges(self.DEPTH)
        z = batch["x"]
        for name in _chain_names(self.DEPTH):
            z = z @ params.nodes[name].weights[edges[name]]
        return 0.5 * jnp.sum((z - batch["y"]) ** 2)

    def test_duals_and_weight_grads_converge_to_backprop(self, rng_key):
        alm = _chain(
            InferencePCALM(eta_infer=self.ETA, infer_steps=self.STEPS, alpha=1.0),
            depth=self.DEPTH,
            width=self.WIDTH,
        )
        pc = _chain(
            InferenceSGD(eta_infer=self.ETA, infer_steps=self.STEPS),
            depth=self.DEPTH,
            width=self.WIDTH,
        )
        params = _orthogonal_params(alm, rng_key)
        batch = _batch(jax.random.PRNGKey(1), self.BATCH, self.WIDTH, self.WIDTH)
        adjoints, bp_grads = self._bp(params, batch)

        g_alm, s_alm = _pc_grads(alm, params, batch, rng_key)
        g_pc, _ = _pc_grads(pc, params, batch, rng_key)

        for name, adjoint in adjoints.items():
            np.testing.assert_allclose(
                s_alm.nodes[name].dual, -adjoint, rtol=1e-3, atol=1e-4
            )

        edges = _chain_edges(self.DEPTH)
        alm_stats, pc_stats = {}, {}
        for name, edge_key in edges.items():
            bp = bp_grads.nodes[name].weights[edge_key]
            alm_stats[name] = _cos_ratio(g_alm.nodes[name].weights[edge_key], bp)
            pc_stats[name] = _cos_ratio(g_pc.nodes[name].weights[edge_key], bp)
            cos, ratio = alm_stats[name]
            assert cos > 0.999, (name, cos)
            assert abs(ratio - 1.0) < 1e-2, (name, ratio)

        print("deep linear chain (cos, norm ratio) PC-ALM:", alm_stats)
        print("deep linear chain (cos, norm ratio) PC:", pc_stats)

        # Standard PC at the same T: bottom-layer gradient is a small fraction
        # of backprop's, and some layer is poorly aligned.
        assert pc_stats["h0"][1] < 0.5
        assert min(c for c, _ in pc_stats.values()) < 0.9
        assert min(c for c, _ in alm_stats.values()) > min(
            c for c, _ in pc_stats.values()
        )

    def test_explicit_grad_nodes_match_autodiff(self, rng_key):
        kwargs = dict(eta_infer=0.1, infer_steps=12, alpha=1.0)
        auto = _chain(InferencePCALM(**kwargs), activation=TanhActivation())
        explicit = _chain(
            InferencePCALM(**kwargs),
            activation=TanhActivation(),
            node_cls=LinearExplicitGrad,
        )
        params = initialize_params(auto, rng_key)
        batch = _batch(jax.random.PRNGKey(1), 5, 4, 4)

        g_auto, s_auto = _pc_grads(auto, params, batch, rng_key)
        g_expl, s_expl = _pc_grads(explicit, params, batch, rng_key)

        for name in auto.nodes:
            np.testing.assert_allclose(
                s_auto.nodes[name].z_latent,
                s_expl.nodes[name].z_latent,
                rtol=1e-5,
                atol=1e-6,
            )
            np.testing.assert_allclose(
                s_auto.nodes[name].dual, s_expl.nodes[name].dual, rtol=1e-5, atol=1e-6
            )
        for a, b in zip(
            jax.tree_util.tree_leaves(g_auto), jax.tree_util.tree_leaves(g_expl)
        ):
            np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)


def test_mupc_fixed_point_is_backprop_scaled_by_jacobian_gain(rng_key):
    """Under muPC, node k hops below the top free node gets gain**k times BP."""
    width, names, prevs = 8, ["h0", "h1", "h2", "y"], ["x", "h0", "h1", "h2"]
    x = IdentityNode(shape=(width,), name="x")
    hidden = [
        Linear(shape=(width,), name=n, activation=TanhActivation(), use_bias=False)
        for n in names[:-1]
    ]
    y = Linear(shape=(width,), name="y", use_bias=False)
    chain = [x, *hidden, y]
    structure = graph(
        nodes=chain,
        edges=[Edge(source=a, target=b.slot("in")) for a, b in zip(chain, chain[1:])],
        task_map=TaskMap(x=x, y=y),
        inference=InferencePCALM(eta_infer=0.2, infer_steps=800, alpha=1.0),
        scaling=MuPCConfig(),
    )
    params = initialize_params(structure, jax.random.PRNGKey(0))
    batch = _batch(jax.random.PRNGKey(1), 4, width, width)

    def bp_loss(p):
        z = batch["x"]
        for name, prev in zip(names, prevs):
            edge_key = _edge(prev, name)
            sc = structure.nodes[name].node_info.scaling_config
            a = sc.forward_scale[edge_key] if sc is not None else 1.0
            pre = (a * z) @ p.nodes[name].weights[edge_key]
            z = jnp.tanh(pre) if name != "y" else pre
        return 0.5 * jnp.sum((z - batch["y"]) ** 2)

    bp_grads = jax.grad(bp_loss)(params)
    grads, _ = _pc_grads(structure, params, batch, rng_key)

    gain = TanhActivation.jacobian_gain(TanhActivation().config)
    hops = {"h0": 2, "h1": 1, "h2": 0, "y": 0}
    for name, prev in zip(names, prevs):
        edge_key = _edge(prev, name)
        cos, ratio = _cos_ratio(
            grads.nodes[name].weights[edge_key], bp_grads.nodes[name].weights[edge_key]
        )
        assert cos > 0.9999, (name, cos)
        assert abs(ratio / gain ** hops[name] - 1.0) < 1e-2, (name, ratio)


# =============================================================================
# Equivalence with the reference implementation
# =============================================================================


def _ref_block(W, z_prev, is_first):
    inp = z_prev if is_first else jax.nn.relu(z_prev)
    return inp @ W.T


def _ref_forward(Ws, x):
    acts, z = [], x
    for i, W in enumerate(Ws):
        z = _ref_block(W, z, i == 0)
        acts.append(z)
    return acts


def _ref_residuals(Ws, x, free):
    return [
        z - _ref_block(Ws[i], x if i == 0 else free[i - 1], i == 0)
        for i, z in enumerate(free)
    ]


def _ref_al_energy(Ws, x, y, free, duals):
    batch_size = x.shape[0]
    y_pred = _ref_block(Ws[-1], free[-1], False)
    total = 0.5 * jnp.mean(jnp.sum((y_pred - y) ** 2, axis=-1))
    for r, lam in zip(_ref_residuals(Ws, x, free), duals):
        total = total + 0.5 * jnp.sum((r + lam) ** 2) / batch_size
    return total


def _ref_run_pcalm(Ws, x, y, state_lr, alpha, budget, sequential=True):
    """Reference run_pcalm (rho=1, inner_steps=1, pre_dual_energy)."""
    free = _ref_forward(Ws, x)[:-1]
    duals = [jnp.zeros_like(z) for z in free]
    lr = state_lr * x.shape[0]
    grad_free = jax.grad(_ref_al_energy, argnums=3)

    def primal(free, duals):
        g = grad_free(Ws, x, y, free, duals)
        return [z - lr * gz for z, gz in zip(free, g)]

    for _ in range(budget - 1):
        before = _ref_residuals(Ws, x, free)
        free = primal(free, duals)
        r = _ref_residuals(Ws, x, free) if sequential else before
        duals = [lam + alpha * ri for lam, ri in zip(duals, r)]
    free = primal(free, duals)
    return free, duals


def _ref_method_grad(Ws, x, y, free, duals):
    free = [jax.lax.stop_gradient(z) for z in free]
    duals = [jax.lax.stop_gradient(d) for d in duals]
    return jax.grad(_ref_al_energy)(Ws, x, y, free, duals)


class TestReferenceEquivalence:
    IN, HID, OUT, BATCH = 5, 6, 3, 4
    STATE_LR, ALPHA, BUDGET = 0.1, 1.0, 10

    def _fabricpc(self, dual_timing):
        x = IdentityNode(shape=(self.IN,), name="x")
        h0 = Linear(shape=(self.HID,), name="h0", use_bias=False)
        h1 = _ReluInputLinear(shape=(self.HID,), name="h1", use_bias=False)
        h2 = _ReluInputLinear(shape=(self.HID,), name="h2", use_bias=False)
        y = _ReluInputLinear(shape=(self.OUT,), name="y", use_bias=False)
        chain = [x, h0, h1, h2, y]
        edges = [Edge(source=a, target=b.slot("in")) for a, b in zip(chain, chain[1:])]
        return graph(
            nodes=chain,
            edges=edges,
            task_map=TaskMap(x=x, y=y),
            inference=InferencePCALM(
                eta_infer=self.STATE_LR,
                infer_steps=self.BUDGET,
                alpha=self.ALPHA,
                dual_timing=dual_timing,
            ),
        )

    def _setup(self, dual_timing):
        structure = self._fabricpc(dual_timing)
        keys = jax.random.split(jax.random.PRNGKey(3), 6)
        dims = [self.IN, self.HID, self.HID, self.HID, self.OUT]
        Ws = [
            jax.random.normal(k, (o, i)) / math.sqrt(i)
            for k, i, o in zip(keys, dims[:-1], dims[1:])
        ]
        names = ["h0", "h1", "h2", "y"]
        prevs = ["x", "h0", "h1", "h2"]
        params = GraphParams(
            nodes={
                "x": NodeParams(weights={}, biases={}),
                **{
                    n: NodeParams(weights={_edge(p, n): W.T}, biases={})
                    for n, p, W in zip(names, prevs, Ws)
                },
            }
        )
        batch = _batch(keys[-1], self.BATCH, self.IN, self.OUT)
        return structure, params, Ws, batch, names, prevs

    @pytest.mark.parametrize("dual_timing", ["sequential", "simultaneous"])
    def test_matches_reference(self, rng_key, dual_timing):
        structure, params, Ws, batch, names, prevs = self._setup(dual_timing)
        free, duals = _ref_run_pcalm(
            Ws,
            batch["x"],
            batch["y"],
            self.STATE_LR,
            self.ALPHA,
            self.BUDGET,
            sequential=dual_timing == "sequential",
        )
        ref_grads = _ref_method_grad(Ws, batch["x"], batch["y"], free, duals)

        grads, final_state = _pc_grads(structure, params, batch, rng_key)

        for i, name in enumerate(names[:-1]):
            np.testing.assert_allclose(
                final_state.nodes[name].z_latent, free[i], rtol=1e-5, atol=1e-5
            )
            np.testing.assert_allclose(
                final_state.nodes[name].dual, duals[i], rtol=1e-5, atol=1e-5
            )
        for name, prev, ref_g in zip(names, prevs, ref_grads):
            fabric_g = grads.nodes[name].weights[_edge(prev, name)].T / self.BATCH
            np.testing.assert_allclose(fabric_g, ref_g, rtol=1e-4, atol=1e-5)

    def test_simultaneous_close_to_sequential(self, rng_key):
        """Report the gap; the two timings differ by O(eta) per step."""
        seq, params, _, batch, names, prevs = self._setup("sequential")
        sim = self._fabricpc("simultaneous")
        g_seq, s_seq = _pc_grads(seq, params, batch, rng_key)
        g_sim, s_sim = _pc_grads(sim, params, batch, rng_key)

        gaps = {}
        for name, prev in zip(names, prevs):
            a = g_seq.nodes[name].weights[_edge(prev, name)]
            b = g_sim.nodes[name].weights[_edge(prev, name)]
            gaps[name] = float(jnp.linalg.norm(a - b) / jnp.linalg.norm(a))
        print("simultaneous vs sequential relative weight-grad gap:", gaps)
        assert all(np.isfinite(v) for v in gaps.values())
        assert max(gaps.values()) < 0.5


# =============================================================================
# Explicit self-gradient in EmbeddingNode
# =============================================================================


def test_embedding_node_self_grad_includes_dual(rng_key):
    x = IdentityNode(shape=(3,), name="ids")
    emb = EmbeddingNode(shape=(3, 4), name="emb", vocab_size=7, embed_dim=4)
    out = Linear(shape=(3, 2), name="out")
    structure = graph(
        nodes=[x, emb, out],
        edges=[
            Edge(source=x, target=emb.slot("in")),
            Edge(source=emb, target=out.slot("in")),
        ],
        task_map=TaskMap(x=x, y=out),
        inference=InferencePCALM(),
    )
    params = initialize_params(structure, rng_key)
    ids = jnp.array([[1, 2, 3], [4, 5, 6]], dtype=jnp.int32)
    state = initialize_graph_state(
        structure, 2, rng_key, clamps={"ids": ids}, params=params
    )
    node_info = structure.nodes["emb"].node_info
    dual = jax.random.normal(jax.random.PRNGKey(5), (2, 3, 4))
    z = jax.random.normal(jax.random.PRNGKey(6), (2, 3, 4))
    node_state = state.nodes["emb"]._replace(z_latent=z, dual=dual)

    new_state, _, self_grad = EmbeddingNode.forward_and_latent_grads(
        params.nodes["emb"], {"ids->emb:in": ids}, node_state, node_info, False
    )
    np.testing.assert_allclose(self_grad, z - new_state.z_mu + dual, rtol=1e-6)
    expected_energy = 0.5 * jnp.sum((z - new_state.z_mu) ** 2, axis=(1, 2)) + jnp.sum(
        dual * (z - new_state.z_mu), axis=(1, 2)
    )
    np.testing.assert_allclose(new_state.energy, expected_energy, rtol=1e-5)


# =============================================================================
# Encoder MLM graph
# =============================================================================


class TestEncoderMlm:
    SEQ_LEN, VOCAB = 8, 24
    STEPS = 20

    def _batch(self):
        loader = MlmTextBatcher(
            token_ids=np.tile(np.arange(4, self.VOCAB, dtype=np.int32), 8)[:96],
            seq_len=self.SEQ_LEN,
            batch_size=4,
            pad_token_id=0,
            mask_token_id=3,
            vocab_size=self.VOCAB,
            special_token_ids={0, 1, 2, 3},
            cls_token_id=1,
            seed=0,
            mask_prob=0.3,
        )
        return {k: jnp.asarray(v) for k, v in next(iter(loader)).items()}

    def _model(self, alpha):
        return create_encoder_mlm_model(
            depth=2,
            embed_dim=16,
            num_heads=2,
            mlp_dim=32,
            seq_len=self.SEQ_LEN,
            vocab_size=self.VOCAB,
            inference=InferencePCALM(
                eta_infer=0.1, infer_steps=self.STEPS, alpha=alpha, max_norm=5.0
            ),
        )

    def test_training_step_runs_under_jit(self, rng_key):
        structure = self._model(alpha=1.0)
        params = initialize_params(structure, rng_key)
        optimizer = optax.adam(1e-3)
        batch = self._batch()
        step = jax.jit(
            lambda p, o, b, k: train_step_autoregressive(
                p, o, b, structure, optimizer, k
            )
        )
        new_params, _, energy, ce, final_state = step(
            params, optimizer.init(params), batch, rng_key
        )
        assert np.isfinite(float(energy)) and np.isfinite(float(ce))
        assert all(
            bool(jnp.all(jnp.isfinite(leaf)))
            for leaf in jax.tree_util.tree_leaves((new_params, final_state))
        )
        assert jnp.any(final_state.nodes["L0_mha"].dual)
        assert not jnp.any(final_state.nodes["mlm_head"].dual)

    def test_lowest_layers_move_toward_backprop(self, rng_key):
        batch = self._batch()
        masked = float(jnp.sum(batch["loss_mask"]))
        low = ("token_embeddings", "position_embeddings", "L0_mha")

        stats = {}
        for alpha in (0.0, 1.0):
            structure = self._model(alpha)
            params = initialize_params(structure, rng_key)
            pc, _, _, _ = jax.jit(
                lambda p, b: get_graph_param_gradient_autoregressive(
                    p, b, structure, rng_key
                )
            )(params, batch)
            bp = jax.grad(
                lambda p: compute_loss_autoregressive(p, structure, batch, rng_key)[0]
            )(params)
            for name in low:
                g = jax.tree_util.tree_map(lambda v: v / masked, pc.nodes[name])
                b = bp.nodes[name]
                gg, bb, gb = _tree_dot(g, g), _tree_dot(b, b), _tree_dot(g, b)
                stats[alpha, name] = {
                    "cos": gb / math.sqrt(gg * bb),
                    "ratio": math.sqrt(gg / bb),
                    "rel_err": math.sqrt(max(gg - 2 * gb + bb, 0.0) / bb),
                }
        print("encoder MLM PC-vs-BP gradient stats:", stats)

        for name in low:
            pc0, alm = stats[0.0, name], stats[1.0, name]
            assert alm["rel_err"] < pc0["rel_err"], (name, pc0, alm)
            assert alm["ratio"] > 10 * pc0["ratio"], (name, pc0, alm)
            assert alm["cos"] > 0.95, (name, alm)


# =============================================================================
# Argument validation
# =============================================================================


class TestValidation:
    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"alpha": -0.1}, "alpha"),
            ({"infer_steps": 0}, "infer_steps"),
            ({"infer_steps": 2.5}, "infer_steps"),
            ({"dual_timing": "async"}, "dual_timing"),
            ({"max_norm": 0.0}, "max_norm"),
        ],
    )
    def test_invalid_args_raise(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            InferencePCALM(**kwargs)

    def test_defaults(self):
        config = InferencePCALM().config
        assert config["eta_infer"] == 0.1
        assert config["infer_steps"] == 20
        assert config["alpha"] == 1.0
        assert config["dual_timing"] == "sequential"
        assert config["latent_decay"] == 0.0
        assert config["max_norm"] is None
        assert config["eps"] == 1e-8
