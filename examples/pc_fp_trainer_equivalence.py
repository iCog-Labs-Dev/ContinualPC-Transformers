"""Equivalence checks for the Transformer V2 PC-FP trainer path."""

from jax_setup import set_jax_flags_before_importing_jax

set_jax_flags_before_importing_jax()

import argparse
import pickle

import jax
import jax.numpy as jnp
import optax

from fabricpc.core.fixed_prediction import (
    InferenceSGDFixedPrediction,
    fixed_prediction_inference_and_gradients,
)
from fabricpc.core.inference import run_inference
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.nodes.transformer_v2 import create_deep_transformer
from fabricpc.training.train_autoregressive import (
    compute_loss,
    create_causal_mask,
    create_dropout_clamps,
    get_graph_param_gradient_autoregressive,
    train_step_autoregressive,
)
from fabricpc.utils.data import CharDataLoader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--skip_batches", type=int, default=4)
    parser.add_argument("--audit_seed", type=int, default=0)
    return parser.parse_args()


def load_checkpoint(path):
    with open(path, "rb") as handle:
        checkpoint = pickle.load(handle)
    params = jax.tree_util.tree_map(jnp.asarray, checkpoint["params"])
    return params, dict(checkpoint["config"])


def convert_batch(raw_batch):
    if isinstance(raw_batch, dict):
        return {name: jnp.asarray(value) for name, value in raw_batch.items()}
    return {
        "x": jnp.asarray(raw_batch[0]),
        "y": jnp.asarray(raw_batch[1]),
    }


def tree_max_abs(left, right):
    left_leaves, left_def = jax.tree_util.tree_flatten(left)
    right_leaves, right_def = jax.tree_util.tree_flatten(right)
    if left_def != right_def:
        raise AssertionError("Pytree structures differ")
    maxima = [
        jnp.max(jnp.abs(a - b))
        for a, b in zip(left_leaves, right_leaves)
        if a.size > 0
    ]
    return float(jnp.max(jnp.stack(maxima))) if maxima else 0.0


def assert_tree_close(name, left, right, atol=1e-6, rtol=1e-6):
    left_leaves, left_def = jax.tree_util.tree_flatten(left)
    right_leaves, right_def = jax.tree_util.tree_flatten(right)
    if left_def != right_def:
        raise AssertionError(f"{name}: pytree structures differ")
    for index, (a, b) in enumerate(zip(left_leaves, right_leaves)):
        if not bool(jnp.allclose(a, b, atol=atol, rtol=rtol)):
            difference = float(jnp.max(jnp.abs(a - b)))
            raise AssertionError(
                f"{name}: leaf {index} differs; max_abs={difference:.6g}"
            )
    print(f"PASS {name}: max_abs={tree_max_abs(left, right):.6g}")


def make_training_clamps(structure, batch, rng_key):
    batch_size, seq_len = batch["x"].shape
    clamps = {}
    for task_name, value in batch.items():
        if task_name in structure.task_map:
            clamps[structure.task_map[task_name]] = value

    output_node = structure.task_map["y"]
    output_size = structure.nodes[output_node].node_info.shape[-1]
    targets = jax.nn.one_hot(batch["y"], output_size)
    clamps[output_node] = targets

    if "causal_mask" in structure.task_map:
        mask = create_causal_mask(seq_len)[None, None, :, :]
        clamps[structure.task_map["causal_mask"]] = jnp.broadcast_to(
            mask,
            (batch_size, 1, seq_len, seq_len),
        )

    _unused_key, dropout_key, state_key = jax.random.split(rng_key, 3)
    clamps.update(
        create_dropout_clamps(
            structure,
            batch_size,
            dropout_key,
            training=True,
        )
    )
    return clamps, targets, state_key


def direct_energy(state, structure, batch_size):
    energy = jnp.array(0.0)
    for node_name, node in structure.nodes.items():
        if node.node_info.in_degree > 0:
            energy = energy + jnp.sum(state.nodes[node_name].energy)
    return energy / batch_size


def main():
    args = parse_args()
    params, config = load_checkpoint(args.checkpoint)

    if config.get("tokenizer", "char") != "char":
        raise ValueError("This check currently targets the character model")

    batch_size = config.get("global_batch_size", config.get("batch_size", 32))
    loader = CharDataLoader(
        "validation",
        seq_len=config["seq_len"],
        batch_size=batch_size,
        shuffle=False,
    )
    raw_batch = None
    for batch_index, candidate in enumerate(loader):
        if batch_index == args.skip_batches:
            raw_batch = candidate
            break
    if raw_batch is None:
        raise RuntimeError("Validation loader did not provide the requested batch")
    batch = convert_batch(raw_batch)

    structure = create_deep_transformer(
        depth=config["depth"],
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        mlp_dim=config["mlp_dim"],
        seq_len=config["seq_len"],
        vocab_size=loader.vocab_size,
        inference=InferenceSGDFixedPrediction(
            eta_infer=config["eta_infer"],
            infer_steps=config["infer_steps"],
            max_norm=config.get("max_norm", 5.0),
            latent_decay=0.0,
        ),
        weight_init={"type": "normal", "std": config["weight_init_std"]},
        dropout_rate=config["dropout_rate"],
    )

    rng_key = jax.random.PRNGKey(args.audit_seed)
    clamps, targets, state_key = make_training_clamps(
        structure,
        batch,
        rng_key,
    )
    initial_state = initialize_graph_state(
        structure,
        batch["x"].shape[0],
        state_key,
        clamps=clamps,
        params=params,
    )

    direct_grads, direct_state, _ = fixed_prediction_inference_and_gradients(
        params,
        initial_state,
        clamps,
        structure,
    )
    expected_energy = direct_energy(
        direct_state,
        structure,
        batch["x"].shape[0],
    )
    expected_ce = compute_loss(
        direct_state,
        targets,
        structure.task_map["y"],
        loss_type="cross_entropy",
    )

    trainer_grads, trainer_energy, trainer_ce, trainer_state = (
        get_graph_param_gradient_autoregressive(
            params,
            batch,
            structure,
            rng_key,
            use_causal_mask=True,
        )
    )
    assert_tree_close("final state", trainer_state, direct_state)
    assert_tree_close("local gradients", trainer_grads, direct_grads)
    if not bool(jnp.allclose(trainer_energy, expected_energy, atol=1e-6, rtol=1e-6)):
        raise AssertionError(
            f"energy differs: trainer={trainer_energy}, direct={expected_energy}"
        )
    print(f"PASS energy: {float(trainer_energy):.8g}")
    if not bool(jnp.allclose(trainer_ce, expected_ce, atol=1e-6, rtol=1e-6)):
        raise AssertionError(
            f"CE differs: trainer={trainer_ce}, direct={expected_ce}"
        )
    print(f"PASS cross-entropy: {float(trainer_ce):.8g}")

    repeat_grads, repeat_energy, repeat_ce, repeat_state = (
        get_graph_param_gradient_autoregressive(
            params,
            batch,
            structure,
            rng_key,
            use_causal_mask=True,
        )
    )
    assert_tree_close("deterministic state", repeat_state, trainer_state)
    assert_tree_close("deterministic gradients", repeat_grads, trainer_grads)
    if not bool(jnp.array_equal(repeat_energy, trainer_energy)):
        raise AssertionError("deterministic energy check failed")
    if not bool(jnp.array_equal(repeat_ce, trainer_ce)):
        raise AssertionError("deterministic CE check failed")
    print("PASS deterministic scalar metrics")

    optimizer = optax.adam(float(config["lr"]))
    initial_opt_state = optimizer.init(params)
    stepped_params, stepped_opt_state, _, _, _ = train_step_autoregressive(
        params,
        initial_opt_state,
        batch,
        structure,
        optimizer,
        rng_key,
        use_causal_mask=True,
    )
    manual_updates, manual_opt_state = optimizer.update(
        direct_grads,
        initial_opt_state,
        params,
    )
    manual_params = optax.apply_updates(params, manual_updates)
    assert_tree_close("one Adam parameter update", stepped_params, manual_params)
    assert_tree_close("one Adam optimizer-state update", stepped_opt_state, manual_opt_state)

    # Fixed-prediction evaluation/generation must be an exact zero-step no-op.
    eval_key = jax.random.fold_in(rng_key, 999)
    _unused, dropout_key, eval_state_key = jax.random.split(eval_key, 3)
    eval_clamps = {structure.task_map["x"]: batch["x"]}
    eval_clamps.update(
        create_dropout_clamps(
            structure,
            batch["x"].shape[0],
            dropout_key,
            training=False,
        )
    )
    eval_state = initialize_graph_state(
        structure,
        batch["x"].shape[0],
        eval_state_key,
        clamps=eval_clamps,
        params=params,
    )
    eval_result = run_inference(
        params,
        eval_state,
        eval_clamps,
        structure,
    )
    assert_tree_close("eval/generation zero-step", eval_result, eval_state, atol=0.0, rtol=0.0)

    print("\nAll PC-FP trainer equivalence checks passed.")


if __name__ == "__main__":
    main()
