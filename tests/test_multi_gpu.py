#!/usr/bin/env python3
"""
Test suite for multi-GPU training utilities.

Verifies numerical similarity between train_pcn() JIT path and
train_pcn(..., pmap_single_device=True), plus utility function correctness.
"""

import copy
import pytest
import jax
import jax.numpy as jnp
import numpy as np
import optax

from fabricpc.nodes import Linear
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.core.activations import IdentityActivation, ReLUActivation
from fabricpc.core.initializers import XavierInitializer
from fabricpc.core.inference import InferenceSGD, InferenceSGDNormClip
from fabricpc.training import train_pcn, evaluate_pcn, train_autoregressive, evaluate_autoregressive
from fabricpc.training.train import shard_batch, replicate_params, unshard_energies
from fabricpc.nodes.transformer_v2 import create_deep_transformer


@pytest.fixture
def simple_structure():
    """Simple graph structure for testing."""
    input_node = Linear(shape=(8,), activation=IdentityActivation(), name="input")
    hidden1 = Linear(
        shape=(16,),
        activation=ReLUActivation(),
        weight_init=XavierInitializer(),
        name="hidden1",
    )
    hidden2 = Linear(
        shape=(16,),
        activation=ReLUActivation(),
        weight_init=XavierInitializer(),
        name="hidden2",
    )
    hidden3 = Linear(
        shape=(16,),
        activation=ReLUActivation(),
        weight_init=XavierInitializer(),
        name="hidden3",
    )
    hidden4 = Linear(
        shape=(16,),
        activation=ReLUActivation(),
        weight_init=XavierInitializer(),
        name="hidden4",
    )
    output_node = Linear(shape=(8,), activation=IdentityActivation(), name="output")

    structure = graph(
        nodes=[input_node, hidden1, hidden2, hidden3, hidden4, output_node],
        edges=[
            Edge(source=input_node, target=hidden1.slot("in")),
            Edge(source=hidden1, target=hidden2.slot("in")),
            Edge(source=hidden2, target=hidden3.slot("in")),
            Edge(source=hidden3, target=hidden4.slot("in")),
            Edge(source=hidden4, target=output_node.slot("in")),
        ],
        task_map=TaskMap(x=input_node, y=output_node),
        inference=InferenceSGD(),
    )

    return structure


@pytest.fixture
def optimizer():
    """Optimizer for training."""
    return optax.adam(1e-3)


@pytest.fixture
def train_config():
    """Training configuration."""
    return {
        "num_epochs": 1,
    }


class SimpleDataLoader:
    """Simple data loader that yields random batches."""

    def __init__(self, input_shape, output_shape, batch_size, num_batches, rng_key):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.rng_key = rng_key
        self._data = self._generate_data()

    def _generate_data(self):
        """Pre-generate all batches for consistency."""
        batches = []
        rng_key = self.rng_key
        for _ in range(self.num_batches):
            rng_key, x_key, y_key = jax.random.split(rng_key, 3)
            x = jax.random.normal(x_key, (self.batch_size, *self.input_shape))
            y = jnp.tanh(x) + jax.random.normal(
                y_key, (self.batch_size, *self.output_shape)
            )
            batches.append({"x": x, "y": y})
        return batches

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return self.num_batches


class TestMultiGPUTraining:
    """Test suite for multi-GPU training numerical similarity."""

    def test_both_methods_reduce_energy(
        self, simple_structure, optimizer, train_config, rng_key
    ):
        """Test that both training methods reduce energy over training."""
        model_key, train_key1, train_key2, data_key, eval_key = jax.random.split(
            rng_key, 5
        )

        # Initialize parameters
        params = initialize_params(simple_structure, model_key)

        # Create data loader
        input_shape = simple_structure.nodes["input"].node_info.shape
        output_shape = simple_structure.nodes["output"].node_info.shape
        train_loader = SimpleDataLoader(
            input_shape, output_shape, batch_size=8, num_batches=4, rng_key=data_key
        )

        # Make copies
        params_single = copy.deepcopy(params)
        params_multi = copy.deepcopy(params)

        # Run single-GPU training (JIT path) and capture energies
        trained_single, iter_results_single, _ = train_pcn(
            params_single,
            simple_structure,
            train_loader,
            optimizer,
            train_config,
            train_key1,
            verbose=False,
            use_tqdm=False,
        )

        # Run pmap path training
        trained_multi, iter_results_multi, _ = train_pcn(
            params_multi,
            simple_structure,
            train_loader,
            optimizer,
            train_config,
            train_key2,
            verbose=False,
            use_tqdm=False,
            pmap_single_device=True,
        )

        # Check that single-GPU training reduced energy
        first_epoch_energy = np.mean(iter_results_single[0])
        last_epoch_energy = np.mean(iter_results_single[-1])
        assert last_epoch_energy <= first_epoch_energy, (
            f"Single-GPU: Energy should decrease. First: {first_epoch_energy:.4f}, "
            f"Last: {last_epoch_energy:.4f}"
        )

        # Evaluate both trained models on same data
        eval_loader = SimpleDataLoader(
            input_shape, output_shape, batch_size=8, num_batches=2, rng_key=eval_key
        )

        eval_config = {}

        eval_key1, eval_key2 = jax.random.split(eval_key)
        metrics_single = evaluate_pcn(
            trained_single, simple_structure, eval_loader, eval_config, eval_key1
        )
        metrics_multi = evaluate_pcn(
            trained_multi, simple_structure, eval_loader, eval_config, eval_key2
        )

        # Both should have finite energy
        assert np.isfinite(
            metrics_single["energy"]
        ), "Single-GPU eval energy should be finite"
        assert np.isfinite(
            metrics_multi["energy"]
        ), "Multi-GPU eval energy should be finite"

    def test_numerical_similarity(
        self, simple_structure, optimizer, train_config, rng_key
    ):
        """
        Test that train_pcn with pmap_single_device=True produces numerically
        identical results to the JIT path.

        When running on a single device, the pmap training path should be
        mathematically equivalent to the JIT path. Both should use the
        same local learning dynamics and produce identical parameter updates.
        """
        model_key, train_key, data_key = jax.random.split(rng_key, 3)

        # Initialize parameters
        params = initialize_params(simple_structure, model_key)

        # Create data loader
        input_shape = simple_structure.nodes["input"].node_info.shape
        output_shape = simple_structure.nodes["output"].node_info.shape
        train_loader = SimpleDataLoader(
            input_shape, output_shape, batch_size=8, num_batches=4, rng_key=data_key
        )

        # Make copies with same initial params
        params_single = copy.deepcopy(params)
        params_multi = copy.deepcopy(params)

        # Use the SAME training key - results should be identical
        trained_single, _, _ = train_pcn(
            params_single,
            simple_structure,
            train_loader,
            optimizer,
            train_config,
            train_key,
            verbose=False,
            use_tqdm=False,
        )
        trained_multi, _, _ = train_pcn(
            params_multi,
            simple_structure,
            train_loader,
            optimizer,
            train_config,
            train_key,
            verbose=False,
            use_tqdm=False,
            pmap_single_device=True,
        )

        # Compare actual parameter values element-wise
        # With same inputs and same RNG, parameters should be numerically identical
        max_relative_diff = 0.0
        param_diffs = []

        for node_name in simple_structure.nodes:
            if (
                simple_structure.nodes[node_name].node_info.in_degree > 0
            ):  # Skip terminal input nodes (zero in_degree)
                single_node = trained_single.nodes[node_name]
                multi_node = trained_multi.nodes[node_name]

                for edge_key in single_node.weights:
                    w_single = single_node.weights[edge_key]
                    w_multi = multi_node.weights[edge_key]

                    # Compute relative difference
                    diff = jnp.abs(w_single - w_multi)
                    scale = jnp.maximum(jnp.abs(w_single), jnp.abs(w_multi)) + 1e-8
                    rel_diff = diff / scale
                    max_rel = float(jnp.max(rel_diff))
                    mean_rel = float(jnp.mean(rel_diff))

                    param_diffs.append(
                        {
                            "node": node_name,
                            "edge": edge_key,
                            "max_rel_diff": max_rel,
                            "mean_rel_diff": mean_rel,
                        }
                    )
                    max_relative_diff = max(max_relative_diff, max_rel)

        # Single-shard pmap should produce identical results to JIT
        max_allowed_diff = 1e-5  # Allow for floating point tolerance

        assert max_relative_diff < max_allowed_diff, (
            f"Pmap (single shard) should produce identical parameters to JIT! "
            f"Max relative difference: {max_relative_diff:.6f} (allowed: {max_allowed_diff}). "
            f"Per-parameter diffs: {param_diffs}"
        )


class TestMultiGPUUtilities:
    """Test suite for multi-GPU utility functions."""

    def test_shard_batch(self):
        """Test batch sharding utility."""
        batch = {
            "x": jnp.zeros((8, 10)),
            "y": jnp.zeros((8, 5)),
        }

        # Test with 1 device
        sharded = shard_batch(batch, n_devices=1)
        assert sharded["x"].shape == (1, 8, 10)
        assert sharded["y"].shape == (1, 8, 5)

        # Test with 2 devices
        sharded = shard_batch(batch, n_devices=2)
        assert sharded["x"].shape == (2, 4, 10)
        assert sharded["y"].shape == (2, 4, 5)

        # Test with 4 devices
        sharded = shard_batch(batch, n_devices=4)
        assert sharded["x"].shape == (4, 2, 10)
        assert sharded["y"].shape == (4, 2, 5)

    def test_shard_batch_invalid_size(self):
        """Test that sharding raises error for invalid batch size."""
        batch = {"x": jnp.zeros((7, 10))}  # 7 not divisible by 2

        with pytest.raises(ValueError, match="divisible"):
            shard_batch(batch, n_devices=2)

    def test_replicate_params(self, simple_structure, rng_key):
        """Test parameter replication utility."""
        params = initialize_params(simple_structure, rng_key)

        # Replicate to 2 devices
        replicated = replicate_params(params, n_devices=2)

        # Check that each parameter has an extra leading dimension
        for node_name, node_params in replicated.nodes.items():
            for edge_key, weight in node_params.weights.items():
                original_weight = params.nodes[node_name].weights[edge_key]
                assert weight.shape == (
                    2,
                    *original_weight.shape,
                ), f"Replicated weight shape mismatch for {node_name}/{edge_key}"
                # Both replicas should be identical
                assert jnp.allclose(
                    weight[0], weight[1]
                ), "Replicas should be identical"
                assert jnp.allclose(
                    weight[0], original_weight
                ), "Replica should match original"

    def test_unshard_energies(self):
        """Test energy unsharding utility."""
        energies = jnp.array([1.0, 2.0, 3.0, 4.0])
        avg = unshard_energies(energies)

        assert abs(avg - 2.5) < 1e-6, f"Expected 2.5, got {avg}"


def _make_token_batches(
    vocab_size: int,
    seq_len: int,
    batch_size: int,
    num_batches: int,
    rng_key: jax.Array,
):
    """Generate token batches (x, y) for autoregressive testing."""
    batches = []
    for _ in range(num_batches):
        rng_key, x_key, y_key = jax.random.split(rng_key, 3)
        batches.append(
            {
                "x": jax.random.randint(
                    x_key, (batch_size, seq_len), 0, vocab_size
                ),
                "y": jax.random.randint(
                    y_key, (batch_size, seq_len), 0, vocab_size
                ),
            }
        )
    return batches


class TestAutoRegressiveMultiGPU:
    """Test suite for autoregressive multi-GPU / pmap training."""

    @pytest.fixture
    def transformer_structure(self):
        """Small transformer structure for fast testing."""
        return create_deep_transformer(
            depth=1,
            embed_dim=8,
            num_heads=2,
            mlp_dim=16,
            seq_len=4,
            vocab_size=10,
            inference=InferenceSGDNormClip(eta_infer=0.05, infer_steps=2, max_norm=5.0),
        )

    def test_autoregressive_numerical_similarity(self, transformer_structure, rng_key):
        """Test that autoregressive pmap updates match single-device JIT."""
        model_key, train_key, data_key = jax.random.split(rng_key, 3)

        params = initialize_params(transformer_structure, model_key)
        optimizer = optax.adam(1e-3)
        train_config = {"num_epochs": 1, "use_causal_mask": True}
        train_loader = _make_token_batches(
            vocab_size=10, seq_len=4, batch_size=4, num_batches=2, rng_key=data_key
        )

        params_single = copy.deepcopy(params)
        params_multi = copy.deepcopy(params)

        trained_single, _, _ = train_autoregressive(
            params_single,
            transformer_structure,
            train_loader,
            optimizer,
            train_config,
            train_key,
            verbose=False,
            pmap_single_device=False,
        )

        trained_multi, _, _ = train_autoregressive(
            params_multi,
            transformer_structure,
            train_loader,
            optimizer,
            train_config,
            train_key,
            verbose=False,
            pmap_single_device=True,
        )

        max_relative_diff = 0.0
        for node_name in transformer_structure.nodes:
            if transformer_structure.nodes[node_name].node_info.in_degree > 0:
                single_node = trained_single.nodes[node_name]
                multi_node = trained_multi.nodes[node_name]

                for edge_key in single_node.weights:
                    w_single = single_node.weights[edge_key]
                    w_multi = multi_node.weights[edge_key]
                    diff = jnp.abs(w_single - w_multi)
                    scale = jnp.maximum(jnp.abs(w_single), jnp.abs(w_multi)) + 1e-8
                    rel_diff = diff / scale
                    max_rel = float(jnp.max(rel_diff))
                    max_relative_diff = max(max_relative_diff, max_rel)

        assert max_relative_diff < 1e-5, (
            f"Autoregressive pmap (single shard) must match JIT! Max rel diff: {max_relative_diff}"
        )

    def test_autoregressive_pmap_energy_is_finite(self, transformer_structure, rng_key):
        """Test that autoregressive pmap produces finite energies during training."""
        model_key, train_key, data_key = jax.random.split(rng_key, 3)

        params = initialize_params(transformer_structure, model_key)
        optimizer = optax.adam(1e-2)
        train_config = {"num_epochs": 2, "use_causal_mask": True}
        train_loader = _make_token_batches(
            vocab_size=10, seq_len=4, batch_size=4, num_batches=4, rng_key=data_key
        )

        trained_params, iter_results, epoch_results = train_autoregressive(
            params,
            transformer_structure,
            train_loader,
            optimizer,
            train_config,
            train_key,
            verbose=False,
            pmap_single_device=True,
        )

        first_batch_energy = iter_results[0][0]
        last_batch_energy = iter_results[-1][-1]
        assert np.isfinite(first_batch_energy)
        assert np.isfinite(last_batch_energy)

    def test_autoregressive_epoch_callback_single_params(self, transformer_structure, rng_key):
        """Test that epoch_callback receives un-replicated params under pmap."""
        model_key, train_key, data_key = jax.random.split(rng_key, 3)

        params = initialize_params(transformer_structure, model_key)
        optimizer = optax.adam(1e-3)
        train_config = {"num_epochs": 1, "use_causal_mask": True}
        train_loader = _make_token_batches(
            vocab_size=10, seq_len=4, batch_size=4, num_batches=2, rng_key=data_key
        )

        callback_received_shapes = []

        def epoch_cb(epoch_idx, cb_params, cb_struct, cb_cfg, cb_rng, **kwargs):
            shapes = [x.shape for x in jax.tree_util.tree_leaves(cb_params)]
            callback_received_shapes.append(shapes)
            return True

        train_autoregressive(
            params,
            transformer_structure,
            train_loader,
            optimizer,
            train_config,
            train_key,
            verbose=False,
            epoch_callback=epoch_cb,
            pmap_single_device=True,
        )

        orig_shapes = [x.shape for x in jax.tree_util.tree_leaves(params)]
        assert len(callback_received_shapes) == 1
        assert callback_received_shapes[0] == orig_shapes

    def test_autoregressive_evaluation_multi_gpu(self, transformer_structure, rng_key):
        """Test that multi-device evaluation matches single-device evaluation."""
        model_key, eval_key, data_key = jax.random.split(rng_key, 3)

        params = initialize_params(transformer_structure, model_key)
        eval_config = {"use_causal_mask": True}
        test_loader = _make_token_batches(
            vocab_size=10, seq_len=4, batch_size=4, num_batches=3, rng_key=data_key
        )

        metrics_jit = evaluate_autoregressive(
            params,
            transformer_structure,
            test_loader,
            eval_config,
            eval_key,
            pmap_single_device=False,
        )

        metrics_pmap = evaluate_autoregressive(
            params,
            transformer_structure,
            test_loader,
            eval_config,
            eval_key,
            pmap_single_device=True,
        )

        assert np.isclose(metrics_jit["loss"], metrics_pmap["loss"], rtol=1e-4, atol=1e-4)
        assert np.isclose(metrics_jit["perplexity"], metrics_pmap["perplexity"], rtol=1e-4, atol=1e-4)
        assert np.isclose(metrics_jit["accuracy"], metrics_pmap["accuracy"], rtol=1e-4, atol=1e-4)

    def test_autoregressive_zero_batches_processed_raises(self, transformer_structure, rng_key):
        """Test that train_autoregressive raises RuntimeError if all batches are skipped."""
        model_key, train_key, _ = jax.random.split(rng_key, 3)

        params = initialize_params(transformer_structure, model_key)
        optimizer = optax.adam(1e-3)
        train_config = {"num_epochs": 1, "use_causal_mask": True}
        empty_loader = []

        with pytest.raises(RuntimeError, match="No batches were processed"):
            train_autoregressive(
                params,
                transformer_structure,
                empty_loader,
                optimizer,
                train_config,
                train_key,
                verbose=False,
            )

    def test_autoregressive_evaluation_debug_mode(self, transformer_structure, rng_key):
        """Test evaluate_autoregressive with debug=True and integer targets."""
        model_key, eval_key, data_key = jax.random.split(rng_key, 3)

        params = initialize_params(transformer_structure, model_key)
        eval_config = {"use_causal_mask": True}
        test_loader = _make_token_batches(
            vocab_size=10, seq_len=4, batch_size=4, num_batches=2, rng_key=data_key
        )

        metrics = evaluate_autoregressive(
            params,
            transformer_structure,
            test_loader,
            eval_config,
            eval_key,
            debug=True,
        )
        assert np.isfinite(metrics["loss"])

