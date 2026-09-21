"""Cold-start Adam-update audit for fixed-prediction PC gradients.

This is deliberately separate from fixed_prediction_audit.py.  Old
Transformer V2 checkpoints do not contain Adam moment state, so every BP and
PC gradient is transformed with the same freshly initialized Adam state.

The primary comparison uses PC gradients divided by batch_size * seq_len,
matching the BP mean-loss gradient convention.  Raw-PC update metrics are
also retained because the current PC trainer sends unreduced local gradients
to Adam.
"""

from jax_setup import set_jax_flags_before_importing_jax

set_jax_flags_before_importing_jax()

import argparse
import csv
import os

import jax
import jax.numpy as jnp
import numpy as np
import optax

import fixed_prediction_audit as fixed
import same_checkpoint_grads as base


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        default=[0, 4, 30, 100, 200, 300],
    )
    parser.add_argument("--eta_infer", type=float, default=None)
    parser.add_argument("--max_norm", type=float, default=None)
    parser.add_argument("--skip_batches", type=int, default=4)
    parser.add_argument("--num_batches", type=int, default=8)
    parser.add_argument("--audit_seed", type=int, default=0)
    parser.add_argument(
        "--output_dir",
        default="runs/fixed_prediction_adam_audit",
    )
    return parser.parse_args()


def scale_tree(tree, divisor):
    return jax.tree_util.tree_map(lambda value: value / divisor, tree)


def cold_start_update(optimizer, initial_opt_state, gradients, params):
    updates, _ = optimizer.update(
        gradients,
        initial_opt_state,
        params,
    )
    return updates


def vectors_by_module(update_batches):
    result = {}
    for batch_updates in update_batches:
        for node_name in base.MODULE_ORDER:
            vector = base.flatten_node_gradient(batch_updates, node_name)
            if vector is not None:
                result.setdefault(node_name, []).append(vector)
    return result


def global_vectors(update_batches):
    result = []
    for batch_updates in update_batches:
        vectors = []
        for node_name in base.MODULE_ORDER:
            vector = base.flatten_node_gradient(batch_updates, node_name)
            if vector is not None:
                vectors.append(vector)
        result.append(jnp.concatenate(vectors))
    return result


def update_metrics(pc_vectors, bp_vectors):
    # The vectors are already optimizer updates; no further token reduction.
    return fixed.paired_metrics(pc_vectors, bp_vectors, token_count=1.0)


def main():
    args = parse_args()
    if not args.steps or min(args.steps) < 0:
        raise ValueError("steps must be non-negative")
    if 0 not in args.steps:
        raise ValueError("Include T=0 for the logits sanity assertion")

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

    # The learning-rate multiplier is shared and therefore does not affect
    # cosine or norm ratios.  We nevertheless use the checkpoint's real rate.
    optimizer = optax.adam(float(config["lr"]))
    initial_opt_state = optimizer.init(params)

    print("Audit data split: validation")
    print(
        f"Selected validation batches: {args.skip_batches}.."
        f"{args.skip_batches + len(batches) - 1}"
    )
    print(
        "Cold-start Adam update audit: "
        f"lr={float(config['lr']):.6g}, token reduction={int(token_count)}"
    )

    print("Computing BP gradients and cold-start Adam updates...")
    bp_updates_by_batch = []
    for batch, (common_clamps, state_key) in zip(batches, common_inputs):
        bp_gradients = base.bp_gradient(
            params,
            reference_structure,
            batch,
            common_clamps,
            state_key,
        )
        bp_updates_by_batch.append(
            cold_start_update(
                optimizer,
                initial_opt_state,
                bp_gradients,
                params,
            )
        )

    bp_by_module = vectors_by_module(bp_updates_by_batch)
    bp_global = global_vectors(bp_updates_by_batch)
    module_rows = []
    combined_rows = []
    output_node = reference_structure.task_map["y"]

    for step in sorted(set(args.steps)):
        print(f"Computing fixed-prediction PC Adam updates: T={step}")
        structure = base.build_structure(
            config,
            loader.vocab_size,
            infer_steps=step,
            disable_mlp2_scaling=False,
            clip_mode="global",
        )

        normalized_updates_by_batch = []
        raw_updates_by_batch = []
        for batch, (common_clamps, state_key) in zip(
            batches,
            common_inputs,
        ):
            raw_gradients, _ = fixed.pc_fixed_prediction_gradient(
                params,
                structure,
                batch,
                common_clamps,
                state_key,
                step,
            )
            normalized_gradients = scale_tree(raw_gradients, token_count)

            normalized_updates_by_batch.append(
                cold_start_update(
                    optimizer,
                    initial_opt_state,
                    normalized_gradients,
                    params,
                )
            )
            raw_updates_by_batch.append(
                cold_start_update(
                    optimizer,
                    initial_opt_state,
                    raw_gradients,
                    params,
                )
            )

        normalized_by_module = vectors_by_module(
            normalized_updates_by_batch
        )
        raw_by_module = vectors_by_module(raw_updates_by_batch)

        normalized_module_cosines = []
        for node_name in base.MODULE_ORDER:
            if node_name not in bp_by_module:
                continue
            normalized = update_metrics(
                normalized_by_module[node_name],
                bp_by_module[node_name],
            )
            raw = update_metrics(
                raw_by_module[node_name],
                bp_by_module[node_name],
            )
            module_rows.append(
                {
                    "inference_step": step,
                    "module": node_name,
                    "normalized_update_cosine": normalized[
                        "cosine_similarity"
                    ],
                    "normalized_update_cosine_std": normalized[
                        "cosine_similarity_std"
                    ],
                    "normalized_update_norm_ratio": normalized[
                        "norm_ratio_pc_over_bp"
                    ],
                    "normalized_update_norm_ratio_std": normalized[
                        "norm_ratio_pc_over_bp_std"
                    ],
                    "raw_update_cosine": raw["cosine_similarity"],
                    "raw_update_cosine_std": raw[
                        "cosine_similarity_std"
                    ],
                    "raw_update_norm_ratio": raw[
                        "norm_ratio_pc_over_bp"
                    ],
                    "raw_update_norm_ratio_std": raw[
                        "norm_ratio_pc_over_bp_std"
                    ],
                }
            )
            if np.isfinite(normalized["cosine_similarity"]):
                normalized_module_cosines.append(
                    normalized["cosine_similarity"]
                )

            if step == 0 and node_name == output_node:
                if not np.isclose(
                    normalized["cosine_similarity"],
                    1.0,
                    rtol=1e-4,
                    atol=1e-6,
                ):
                    raise AssertionError(
                        f"T=0 normalized logits update cosine failed: "
                        f"{normalized}"
                    )
                if not np.isclose(
                    normalized["norm_ratio_pc_over_bp"],
                    1.0,
                    rtol=1e-4,
                    atol=1e-6,
                ):
                    raise AssertionError(
                        f"T=0 normalized logits update ratio failed: "
                        f"{normalized}"
                    )

        normalized_global = update_metrics(
            global_vectors(normalized_updates_by_batch),
            bp_global,
        )
        raw_global = update_metrics(
            global_vectors(raw_updates_by_batch),
            bp_global,
        )
        combined_rows.append(
            {
                "inference_step": step,
                "normalized_global_update_cosine": normalized_global[
                    "cosine_similarity"
                ],
                "normalized_global_update_cosine_std": normalized_global[
                    "cosine_similarity_std"
                ],
                "normalized_global_update_norm_ratio": normalized_global[
                    "norm_ratio_pc_over_bp"
                ],
                "normalized_global_update_norm_ratio_std": normalized_global[
                    "norm_ratio_pc_over_bp_std"
                ],
                "raw_global_update_cosine": raw_global[
                    "cosine_similarity"
                ],
                "raw_global_update_cosine_std": raw_global[
                    "cosine_similarity_std"
                ],
                "raw_global_update_norm_ratio": raw_global[
                    "norm_ratio_pc_over_bp"
                ],
                "raw_global_update_norm_ratio_std": raw_global[
                    "norm_ratio_pc_over_bp_std"
                ],
                "mean_normalized_module_update_cosine": (
                    float(np.mean(normalized_module_cosines))
                    if normalized_module_cosines
                    else float("nan")
                ),
            }
        )

    checkpoint_tag = os.path.splitext(os.path.basename(args.checkpoint))[0]
    module_path = os.path.join(
        args.output_dir,
        f"{checkpoint_tag}_cold_adam_modules.csv",
    )
    combined_path = os.path.join(
        args.output_dir,
        f"{checkpoint_tag}_cold_adam_combined.csv",
    )
    with open(module_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(module_rows[0]))
        writer.writeheader()
        writer.writerows(module_rows)
    with open(combined_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(combined_rows[0]))
        writer.writeheader()
        writer.writerows(combined_rows)

    print("\nNormalized-PC cold-start Adam summary")
    for row in combined_rows:
        print(
            f"T={row['inference_step']:>3} "
            f"cos={row['normalized_global_update_cosine']:+.4f} "
            f"PC/BP={row['normalized_global_update_norm_ratio']:.4g} "
            f"mean_module_cos="
            f"{row['mean_normalized_module_update_cosine']:+.4f}"
        )

    print("\nEarly-module normalized-update summary")
    for row in module_rows:
        if row["module"] in ("L0_mha", "embed"):
            print(
                f"T={row['inference_step']:>3} "
                f"{row['module']:<7} "
                f"cos={row['normalized_update_cosine']:+.4f} "
                f"sd={row['normalized_update_cosine_std']:.4f} "
                f"PC/BP={row['normalized_update_norm_ratio']:.4g}"
            )

    print(f"\nModule CSV:   {module_path}")
    print(f"Combined CSV: {combined_path}")


if __name__ == "__main__":
    main()