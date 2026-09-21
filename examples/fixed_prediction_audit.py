"""Same-checkpoint fixed-prediction PC audit for Transformer V2.

The ordinary PC loop recomputes every prediction and Jacobian as latents move.
This diagnostic freezes each node's inputs at the feed-forward state during
settling.  Errors remain local and are iteratively transported through the
cached node-local Jacobians.  Local parameter gradients are then evaluated
with the same feed-forward inputs and the settled local errors.
"""

from jax_setup import set_jax_flags_before_importing_jax

set_jax_flags_before_importing_jax()

import argparse
import csv
import os

import jax
import jax.numpy as jnp
import numpy as np

import same_checkpoint_grads as base

from fabricpc.core.inference import (
    InferenceBase,
    InferenceSGDNormClip,
    gather_inputs,
)
from fabricpc.core.scaling import (
    scale_input_grads,
    scale_inputs,
    scale_self_grad,
    scale_weight_grads,
)
from fabricpc.core.state_ops import update_node_in_state
from fabricpc.core.types import GraphParams, NodeParams
from fabricpc.graph_initialization.state_initializer import initialize_graph_state


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--steps", nargs="+", type=int, default=[0, 4, 30, 100])
    parser.add_argument("--eta_infer", type=float, default=None)
    parser.add_argument("--max_norm", type=float, default=None)
    parser.add_argument("--skip_batches", type=int, default=4)
    parser.add_argument("--num_batches", type=int, default=8)
    parser.add_argument("--audit_seed", type=int, default=0)
    parser.add_argument(
        "--output_dir",
        default="runs/fixed_prediction_audit",
    )
    return parser.parse_args()


def fixed_prediction_gradient_pass(
    params,
    state,
    feedforward_state,
    clamps,
    structure,
):
    """Refresh local errors and total latent gradients at cached inputs."""
    state = InferenceBase.zero_grads(params, state, clamps, structure)

    for node_name in structure.nodes:
        node = structure.nodes[node_name]
        info = node.node_info
        scaling = info.scaling_config

        # This is the defining fixed-prediction operation: every local
        # prediction and local Jacobian sees the original feed-forward inputs,
        # while its postsynaptic latent/error is the current inferred value.
        inputs = gather_inputs(info, structure, feedforward_state)
        inputs = scale_inputs(inputs, scaling)

        node_state, input_grads, self_grad = (
            info.node_class.forward_and_latent_grads(
                params.nodes[node_name],
                inputs,
                state.nodes[node_name],
                info,
                is_clamped=(node_name in clamps),
            )
        )

        input_grads = scale_input_grads(input_grads, scaling)
        self_grad = scale_self_grad(self_grad, scaling)
        node_state = node_state._replace(
            latent_grad=node_state.latent_grad + self_grad
        )
        state = state._replace(nodes={**state.nodes, node_name: node_state})

        for edge_key, gradient in input_grads.items():
            source_name = structure.edges[edge_key].source
            accumulated = state.nodes[source_name].latent_grad + gradient
            state = update_node_in_state(
                state,
                source_name,
                latent_grad=accumulated,
            )

    return state


def run_fixed_prediction_inference(
    params,
    initial_state,
    clamps,
    structure,
    inference_steps,
):
    """Settle latents against feed-forward predictions/Jacobians."""
    feedforward_state = initial_state
    config = structure.config["inference"].config

    def body_fn(_step, state):
        state = fixed_prediction_gradient_pass(
            params,
            state,
            feedforward_state,
            clamps,
            structure,
        )

        for node_name in structure.nodes:
            if node_name in clamps:
                continue
            new_latent = InferenceSGDNormClip.compute_new_latent(
                node_name,
                state.nodes[node_name],
                config,
            )
            state = update_node_in_state(
                state,
                node_name,
                z_latent=new_latent,
            )
        return state

    state = jax.lax.fori_loop(
        0,
        inference_steps,
        body_fn,
        initial_state,
    )
    state = fixed_prediction_gradient_pass(
        params,
        state,
        feedforward_state,
        clamps,
        structure,
    )
    return state, feedforward_state


def fixed_prediction_weight_gradients(
    params,
    settled_state,
    feedforward_state,
    structure,
):
    """Read local gradients using cached inputs and settled local errors."""
    gradients = {}
    for node_name, node in structure.nodes.items():
        info = node.node_info
        if info.in_degree == 0:
            gradients[node_name] = NodeParams(weights={}, biases={})
            continue

        scaling = info.scaling_config
        inputs = gather_inputs(info, structure, feedforward_state)
        inputs = scale_inputs(inputs, scaling)
        _, parameter_gradient = info.node_class.forward_and_weight_grads(
            params.nodes[node_name],
            inputs,
            settled_state.nodes[node_name],
            info,
        )
        gradients[node_name] = scale_weight_grads(
            parameter_gradient,
            scaling,
        )
    return GraphParams(nodes=gradients)


def pc_fixed_prediction_gradient(
    params,
    structure,
    batch,
    common_clamps,
    state_key,
    inference_steps,
):
    output_node = structure.task_map["y"]
    output_size = structure.nodes[output_node].node_info.shape[-1]

    clamps = dict(common_clamps)
    clamps[structure.task_map["x"]] = batch["x"]
    clamps[output_node] = jax.nn.one_hot(batch["y"], output_size)

    initial_state = initialize_graph_state(
        structure,
        batch["x"].shape[0],
        state_key,
        clamps=clamps,
        params=params,
    )
    settled_state, feedforward_state = run_fixed_prediction_inference(
        params,
        initial_state,
        clamps,
        structure,
        inference_steps,
    )
    gradients = fixed_prediction_weight_gradients(
        params,
        settled_state,
        feedforward_state,
        structure,
    )
    return gradients, settled_state


def finite_summary(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), 0
    return (
        float(np.mean(values)),
        float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        int(values.size),
    )


def paired_metrics(pc_raw, bp, token_count, eps=1e-12):
    cosines = []
    ratios = []
    for pc_vector_raw, bp_vector in zip(pc_raw, bp):
        pc_vector = pc_vector_raw / token_count
        pc_norm = float(jnp.linalg.norm(pc_vector))
        bp_norm = float(jnp.linalg.norm(bp_vector))
        denominator = pc_norm * bp_norm
        cosines.append(
            float(jnp.vdot(pc_vector, bp_vector) / denominator)
            if denominator > eps
            else float("nan")
        )
        ratios.append(pc_norm / bp_norm if bp_norm > eps else float("nan"))

    cosine, cosine_std, cosine_n = finite_summary(cosines)
    ratio, ratio_std, ratio_n = finite_summary(ratios)
    return {
        "cosine_similarity": cosine,
        "cosine_similarity_std": cosine_std,
        "cosine_valid_batches": cosine_n,
        "norm_ratio_pc_over_bp": ratio,
        "norm_ratio_pc_over_bp_std": ratio_std,
        "norm_ratio_valid_batches": ratio_n,
    }


def main():
    args = parse_args()
    if not args.steps or min(args.steps) < 0:
        raise ValueError("steps must be non-negative")
    if 0 not in args.steps:
        raise ValueError("Include T=0 for the logits sanity assertion")
    if args.eta_infer is not None and args.eta_infer <= 0.0:
        raise ValueError("eta_infer must be positive")
    if args.max_norm is not None and args.max_norm <= 0.0:
        raise ValueError("max_norm must be positive")

    os.makedirs(args.output_dir, exist_ok=True)
    params, checkpoint_config = base.load_checkpoint(args.checkpoint)
    config = dict(checkpoint_config)
    if args.eta_infer is not None:
        config["eta_infer"] = args.eta_infer
    if args.max_norm is not None:
        config["max_norm"] = args.max_norm

    loader = base.make_loader(config)
    reference_structure = base.build_structure(
        config,
        loader.vocab_size,
        infer_steps=0,
        disable_mlp2_scaling=False,
        clip_mode="global",
    )

    batches = []
    common_inputs = []
    key = jax.random.PRNGKey(args.audit_seed)
    for loader_index, raw_batch in enumerate(loader):
        if loader_index < args.skip_batches:
            continue
        if len(batches) >= args.num_batches:
            break
        batch = base.convert_batch(raw_batch)
        common_clamps, state_key = base.make_common_inputs(
            reference_structure,
            batch,
            jax.random.fold_in(key, loader_index),
        )
        batches.append(batch)
        common_inputs.append((common_clamps, state_key))

    if len(batches) != args.num_batches:
        raise RuntimeError(
            f"Requested {args.num_batches} batches, got {len(batches)}"
        )
    token_counts = {
        int(batch["x"].shape[0] * batch["x"].shape[1])
        for batch in batches
    }
    if len(token_counts) != 1:
        raise RuntimeError("Selected batches have inconsistent token counts")
    token_count = float(token_counts.pop())

    print("Audit data split: validation")
    print(
        f"Selected validation batches: {args.skip_batches}.."
        f"{args.skip_batches + len(batches) - 1}"
    )
    print(
        f"Fixed-prediction inference: eta={config['eta_infer']:.6g}, "
        f"max_norm={config.get('max_norm', 5.0):.6g}"
    )

    print("Computing BP gradients...")
    bp_by_node = {}
    for batch, (common_clamps, state_key) in zip(batches, common_inputs):
        gradients = base.bp_gradient(
            params,
            reference_structure,
            batch,
            common_clamps,
            state_key,
        )
        for node_name in base.MODULE_ORDER:
            vector = base.flatten_node_gradient(gradients, node_name)
            if vector is not None:
                bp_by_node.setdefault(node_name, []).append(vector)

    rows = []
    combined_rows = []
    output_node = reference_structure.task_map["y"]

    for step in sorted(set(args.steps)):
        print(f"Computing fixed-prediction PC gradients: T={step}")
        structure = base.build_structure(
            config,
            loader.vocab_size,
            infer_steps=step,
            disable_mlp2_scaling=False,
            clip_mode="global",
        )
        pc_by_node = {}
        pc_by_batch = []

        for batch, (common_clamps, state_key) in zip(
            batches,
            common_inputs,
        ):
            gradients, _ = pc_fixed_prediction_gradient(
                params,
                structure,
                batch,
                common_clamps,
                state_key,
                step,
            )
            batch_vectors = {}
            for node_name in base.MODULE_ORDER:
                vector = base.flatten_node_gradient(gradients, node_name)
                if vector is not None:
                    pc_by_node.setdefault(node_name, []).append(vector)
                    batch_vectors[node_name] = vector
            pc_by_batch.append(batch_vectors)

        module_cosines = []
        for node_name in base.MODULE_ORDER:
            if node_name not in pc_by_node or node_name not in bp_by_node:
                continue
            metrics = paired_metrics(
                pc_by_node[node_name],
                bp_by_node[node_name],
                token_count,
            )
            rows.append(
                {
                    "inference_step": step,
                    "module": node_name,
                    **metrics,
                }
            )
            if np.isfinite(metrics["cosine_similarity"]):
                module_cosines.append(metrics["cosine_similarity"])

            if step == 0 and node_name == output_node:
                if not np.isclose(
                    metrics["cosine_similarity"],
                    1.0,
                    rtol=1e-4,
                    atol=1e-6,
                ):
                    raise AssertionError(f"T=0 logits cosine failed: {metrics}")
                if not np.isclose(
                    metrics["norm_ratio_pc_over_bp"],
                    1.0,
                    rtol=1e-4,
                    atol=1e-6,
                ):
                    raise AssertionError(f"T=0 logits ratio failed: {metrics}")

        global_pc = []
        global_bp = []
        for batch_index, vectors in enumerate(pc_by_batch):
            names = [
                name
                for name in base.MODULE_ORDER
                if name in vectors and name in bp_by_node
            ]
            global_pc.append(jnp.concatenate([vectors[name] for name in names]))
            global_bp.append(
                jnp.concatenate(
                    [bp_by_node[name][batch_index] for name in names]
                )
            )
        global_metrics = paired_metrics(global_pc, global_bp, token_count)
        combined_rows.append(
            {
                "inference_step": step,
                "global_cosine": global_metrics["cosine_similarity"],
                "global_cosine_std": global_metrics[
                    "cosine_similarity_std"
                ],
                "global_norm_ratio_pc_over_bp": global_metrics[
                    "norm_ratio_pc_over_bp"
                ],
                "global_norm_ratio_pc_over_bp_std": global_metrics[
                    "norm_ratio_pc_over_bp_std"
                ],
                "mean_module_cosine": (
                    float(np.mean(module_cosines))
                    if module_cosines
                    else float("nan")
                ),
            }
        )

    checkpoint_tag = os.path.splitext(os.path.basename(args.checkpoint))[0]
    module_path = os.path.join(
        args.output_dir,
        f"{checkpoint_tag}_fixed_prediction_modules.csv",
    )
    combined_path = os.path.join(
        args.output_dir,
        f"{checkpoint_tag}_fixed_prediction_combined.csv",
    )
    with open(module_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(combined_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(combined_rows[0]))
        writer.writeheader()
        writer.writerows(combined_rows)

    print("\nCombined summary")
    for row in combined_rows:
        print(
            f"T={row['inference_step']:>3} "
            f"global_cos={row['global_cosine']:+.4f} "
            f"PC/BP={row['global_norm_ratio_pc_over_bp']:.4g} "
            f"mean_module_cos={row['mean_module_cosine']:+.4f}"
        )

    print("\nEarly-module summary")
    for row in rows:
        if row["module"] in ("L0_mha", "embed"):
            print(
                f"T={row['inference_step']:>3} {row['module']:<7} "
                f"cos={row['cosine_similarity']:+.4f} "
                f"sd={row['cosine_similarity_std']:.4f} "
                f"PC/BP={row['norm_ratio_pc_over_bp']:.4g}"
            )

    print(f"\nModule CSV:   {module_path}")
    print(f"Combined CSV: {combined_path}")


if __name__ == "__main__":
    main()