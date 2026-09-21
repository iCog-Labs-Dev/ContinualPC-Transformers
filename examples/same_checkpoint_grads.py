"""
Same-checkpoint PC-vs-BP gradient audit.

Example:
PYTHONPATH=. python examples/same_checkpoint_grads.py \
    --checkpoint runs/checkpoints/ckpt_pc_char_seed0.pkl \
    --num_batches 4 \
    --max_steps 30 \
    --output_dir runs/same_checkpoint_audit
"""

from jax_setup import set_jax_flags_before_importing_jax

set_jax_flags_before_importing_jax()

import argparse
import csv
import os
import pickle
from dataclasses import replace
import math

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from fabricpc.core.inference import (
    InferenceBase,
    InferenceSGDNormClip,
    gather_inputs,
    run_inference,
)
from fabricpc.core.scaling import (
    scale_input_grads,
    scale_inputs,
    scale_self_grad,
)
from fabricpc.core.state_ops import update_node_in_state
from fabricpc.core.learning import compute_local_weight_gradients
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.nodes.transformer_v2 import (
    Mlp2ResidualNode,
    create_deep_transformer,
)
from fabricpc.training.train_autoregressive import (
    create_causal_mask,
    create_dropout_clamps,
)
from fabricpc.utils.data import CharDataLoader, BpeDataLoader

MODULE_ORDER = [
    "logits",
    "L1_mlp2",
    "L1_mlp1",
    "L1_mha",
    "L0_mlp2",
    "L0_mlp1",
    "L0_mha",
    "embed",
]

MHA_COMPONENTS = {
    "Q": (("weights", "W_q"), ("biases", "b_q")),
    "K": (("weights", "W_k"), ("biases", "b_k")),
    "V": (("weights", "W_v"), ("biases", "b_v")),
    "O": (("weights", "W_o"), ("biases", "b_o")),
    "LayerNorm": (("weights", "ln_gamma"), ("biases", "ln_beta")),
}

def is_mha_module(params, node_name):
    """Identify actual attention modules, excluding dropout nodes."""
    weights = params.nodes[node_name].weights
    return {"W_q", "W_k", "W_v", "W_o"}.issubset(weights)

# Use validation data for diagnostics and tau selection.
# Reserve test data for the final frozen evaluation.
AUDIT_SPLIT = "validation"

class InferenceSGDRMSClip(InferenceBase):
    """Clip each sample by its whole-state gradient RMS."""

    def __init__(
        self,
        eta_infer,
        infer_steps,
        max_norm,
        max_rms,
        latent_decay=0.0,
        eps=1e-8,
    ):
        super().__init__(
            eta_infer=eta_infer,
            infer_steps=infer_steps,
            latent_decay=latent_decay,
            max_norm=max_norm,
            max_rms=max_rms,
            eps=eps,
            clip_mode="rms",
        )

    @staticmethod
    def compute_new_latent(node_name, node_state, config):
        grad = node_state.latent_grad
        axes = tuple(range(1, grad.ndim))

        grad_rms = jnp.sqrt(
            jnp.mean(
                grad.conj() * grad,
                axis=axes,
                keepdims=True,
            ).real
        )

        factor = jnp.minimum(
            1.0,
            config["max_rms"] / (grad_rms + config["eps"]),
        )
        clipped_grad = grad * factor

        return (
            node_state.z_latent
            * (1.0 - config["eta_infer"] * config["latent_decay"])
            - config["eta_infer"] * clipped_grad
        )


class InferenceSGDPerTokenClip(InferenceBase):
    """Clip each token's feature-vector gradient independently."""

    def __init__(
        self,
        eta_infer,
        infer_steps,
        max_norm,
        max_token_norm,
        latent_decay=0.0,
        eps=1e-8,
    ):
        super().__init__(
            eta_infer=eta_infer,
            infer_steps=infer_steps,
            latent_decay=latent_decay,
            max_norm=max_norm,
            max_token_norm=max_token_norm,
            eps=eps,
            clip_mode="per_token",
        )

    @staticmethod
    def compute_new_latent(node_name, node_state, config):
        grad = node_state.latent_grad

        token_norm = jnp.sqrt(
            jnp.sum(
                grad.conj() * grad,
                axis=-1,
                keepdims=True,
            ).real
        )

        factor = jnp.minimum(
            1.0,
            config["max_token_norm"]
            / (token_norm + config["eps"]),
        )
        clipped_grad = grad * factor

        return (
            node_state.z_latent
            * (1.0 - config["eta_infer"] * config["latent_decay"])
            - config["eta_infer"] * clipped_grad
        )

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num_batches", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=30)
    parser.add_argument("--audit_seed", type=int, default=123)
    mlp2_ablation = parser.add_mutually_exclusive_group()
    mlp2_ablation.add_argument(
        "--disable_mlp2_input_scaling",
        action="store_true",
        help=(
            "Diagnostic only: remove muPC forward and top-down scaling from "
            "each Mlp2ResidualNode input edge. W_ff2 remains unchanged."
        ),
    )

    mlp2_ablation.add_argument(
        "--wff2_multiplier",
        type=float,
        default=1.0,
        help=(
            "Diagnostic only: multiply every checkpoint W_ff2 tensor by this "
            "positive value while retaining the original muPC scaling."
        ),
    )
    parser.add_argument(
        "--clip_mode",
        choices=("global", "rms", "per_token"),
        default="global",
        help=(
            "Latent-gradient clipping rule. Global is the existing "
            "per-sample whole-state L2 rule."
        ),
    )

    parser.add_argument(
        "--output_dir",
        default="runs/same_checkpoint_audit",
    )
    return parser.parse_args()


def load_checkpoint(path):
    with open(path, "rb") as f:
        checkpoint = pickle.load(f)

    params = jax.tree_util.tree_map(jnp.asarray, checkpoint["params"])
    config = checkpoint["config"]
    return params, config

def rescale_wff2(params, multiplier):
    """Multiply only W_ff2 checkpoint tensors."""

    if multiplier <= 0.0:
        raise ValueError(
            f"wff2_multiplier must be positive, got {multiplier}"
        )

    updated_nodes = dict(params.nodes)
    changed = []

    for node_name, node_params in params.nodes.items():
        if "W_ff2" not in node_params.weights:
            continue

        old_weight = node_params.weights["W_ff2"]
        new_weight = old_weight * multiplier

        new_weights = {
            **node_params.weights,
            "W_ff2": new_weight,
        }

        updated_nodes[node_name] = node_params._replace(
            weights=new_weights
        )

        old_rms = float(
            jnp.sqrt(jnp.mean(old_weight * old_weight))
        )
        new_rms = float(
            jnp.sqrt(jnp.mean(new_weight * new_weight))
        )
        changed.append((node_name, old_rms, new_rms))

    if not changed:
        raise RuntimeError(
            "No W_ff2 tensors were found in the checkpoint"
        )

    return params._replace(nodes=updated_nodes), tuple(changed)

def disable_mlp2_input_scaling(structure):
    """Remove only MLP2 compute-input muPC scaling."""

    updated_nodes = dict(structure.nodes)
    changed = []

    for node_name, node in structure.nodes.items():
        info = node.node_info

        if info.node_class is not Mlp2ResidualNode:
            continue

        scaling = info.scaling_config
        if scaling is None:
            raise RuntimeError(
                f"{node_name} has no muPC scaling configuration"
            )

        input_edges = [
            edge_key
            for edge_key in info.in_edges
            if structure.edges[edge_key].slot == "in"
        ]

        if len(input_edges) != 1:
            raise RuntimeError(
                f"Expected one input edge for {node_name}, "
                f"got {input_edges}"
            )

        edge_key = input_edges[0]

        if edge_key not in scaling.forward_scale:
            raise RuntimeError(
                f"{node_name} input edge has no forward scale"
            )

        if edge_key not in scaling.topdown_grad_scale:
            raise RuntimeError(
                f"{node_name} input edge has no top-down scale"
            )

        changed.append(node_name)

        new_scaling = replace(
            scaling,
            forward_scale={
                key: value
                for key, value in scaling.forward_scale.items()
                if key != edge_key
            },
            topdown_grad_scale={
                key: value
                for key, value in scaling.topdown_grad_scale.items()
                if key != edge_key
            },
            weight_grad_scale={
                key: value
                for key, value in scaling.weight_grad_scale.items()
                if key != edge_key
            },
        )

        new_info = replace(
            info,
            scaling_config=new_scaling,
        )

        updated_nodes[node_name] = node._with_graph_info(new_info)

    if not changed:
        raise RuntimeError(
            "No Mlp2ResidualNode input scaling was found"
        )

    return structure._replace(nodes=updated_nodes), tuple(changed)

def build_inference(config, infer_steps, clip_mode):
    eta_infer = config["eta_infer"]
    max_norm = config.get("max_norm", 5.0)

    if clip_mode == "global":
        return InferenceSGDNormClip(
            eta_infer=eta_infer,
            infer_steps=infer_steps,
            max_norm=max_norm,
            latent_decay=0.0,
        )

    if clip_mode == "rms":
        reference_numel = (
            config["seq_len"] * config["embed_dim"]
        )
        max_rms = max_norm / math.sqrt(reference_numel)

        return InferenceSGDRMSClip(
            eta_infer=eta_infer,
            infer_steps=infer_steps,
            max_norm=max_norm,
            max_rms=max_rms,
            latent_decay=0.0,
        )

    if clip_mode == "per_token":
        max_token_norm = (
            max_norm / math.sqrt(config["seq_len"])
        )

        return InferenceSGDPerTokenClip(
            eta_infer=eta_infer,
            infer_steps=infer_steps,
            max_norm=max_norm,
            max_token_norm=max_token_norm,
            latent_decay=0.0,
        )

    raise ValueError(f"Unknown clip_mode: {clip_mode}")

def build_structure(
    config,
    vocab_size,
    infer_steps,
    disable_mlp2_scaling=False,
    clip_mode="global",
):
    structure = create_deep_transformer(
        depth=config["depth"],
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        mlp_dim=config["mlp_dim"],
        seq_len=config["seq_len"],
        vocab_size=vocab_size,
        inference=build_inference(
            config,
            infer_steps,
            clip_mode,
        ),
        weight_init={
            "type": "normal",
            "std": config["weight_init_std"],
        },
        dropout_rate=config["dropout_rate"],
    )

    if disable_mlp2_scaling:
        structure, changed = disable_mlp2_input_scaling(
            structure
        )

        if len(changed) != config["depth"]:
            raise RuntimeError(
                "MLP2 scaling ablation changed "
                f"{len(changed)} nodes, expected {config['depth']}"
            )

    return structure


def make_loader(config):
    tokenizer = config.get("tokenizer", "char")
    batch_size = config.get(
        "global_batch_size",
        config.get("batch_size", 32),
    )

    if tokenizer == "bpe":
        return BpeDataLoader(
            AUDIT_SPLIT,
            seq_len=config["seq_len"],
            batch_size=batch_size,
            shuffle=False,
        )

    return CharDataLoader(
        AUDIT_SPLIT,
        seq_len=config["seq_len"],
        batch_size=batch_size,
        shuffle=False,
    )


def convert_batch(batch):
    if isinstance(batch, dict):
        return {key: jnp.asarray(value) for key, value in batch.items()}

    if isinstance(batch, (tuple, list)):
        return {
            "x": jnp.asarray(batch[0]),
            "y": jnp.asarray(batch[1]),
        }

    raise TypeError(f"Unsupported batch type: {type(batch)}")


def make_common_inputs(structure, batch, rng_key):
    """
    Create dropout masks and state key once.

    These exact values are reused by BP and by every PC inference-step count.
    """
    batch_size, seq_len = batch["x"].shape
    dropout_key, state_key = jax.random.split(rng_key)

    common_clamps = create_dropout_clamps(
        structure,
        batch_size,
        dropout_key,
        training=True,
    )

    if "causal_mask" in structure.task_map:
        mask = create_causal_mask(seq_len)[None, None, :, :]
        mask = jnp.broadcast_to(
            mask,
            (batch_size, 1, seq_len, seq_len),
        )
        common_clamps[structure.task_map["causal_mask"]] = mask

    return common_clamps, state_key


def per_sample_l2(value):
    """L2 norm over all non-batch axes, preserving one value per sample."""
    value = jnp.asarray(value)
    if value.ndim <= 1:
        return jnp.abs(value)
    axes = tuple(range(1, value.ndim))
    return jnp.sqrt(jnp.sum(value.conj() * value, axis=axes).real)


def per_sample_cosine(x, y, eps=1e-12):
    """Cosine similarity over non-batch axes, one value per sample."""
    if x.ndim <= 1:
        dot = (x.conj() * y).real
    else:
        axes = tuple(range(1, x.ndim))
        dot = jnp.sum((x.conj() * y).real, axis=axes)
    denominator = per_sample_l2(x) * per_sample_l2(y)
    return jnp.where(denominator > eps, dot / denominator, jnp.nan)


def finite_array_mean(values):
    """Mean over finite JAX values; return NaN when all values are invalid."""
    values = jnp.asarray(values)
    finite = jnp.isfinite(values)
    count = jnp.sum(finite)
    total = jnp.sum(jnp.where(finite, values, 0.0))
    return float(jnp.where(count > 0, total / count, jnp.nan))

def clipping_fractions(grad, inference_config):
    """Return applied-unit, sample-any and global-L2 fractions."""

    global_norm = per_sample_l2(grad)
    max_norm = inference_config["max_norm"]
    global_exceed = jnp.mean(global_norm > max_norm)

    clip_mode = inference_config.get(
        "clip_mode",
        "global",
    )

    if clip_mode == "global":
        clipped = global_norm > max_norm
        fraction = jnp.mean(clipped)

        return (
            float(fraction),
            float(fraction),
            float(global_exceed),
        )

    if clip_mode == "rms":
        axes = tuple(range(1, grad.ndim))
        grad_rms = jnp.sqrt(
            jnp.mean(
                grad.conj() * grad,
                axis=axes,
            ).real
        )
        clipped = grad_rms > inference_config["max_rms"]
        fraction = jnp.mean(clipped)

        return (
            float(fraction),
            float(fraction),
            float(global_exceed),
        )

    if clip_mode == "per_token":
        token_norm = jnp.sqrt(
            jnp.sum(
                grad.conj() * grad,
                axis=-1,
            ).real
        )
        clipped_tokens = (
            token_norm
            > inference_config["max_token_norm"]
        )
        token_fraction = jnp.mean(clipped_tokens)

        if clipped_tokens.ndim == 1:
            sample_any = clipped_tokens
        else:
            sample_any = jnp.any(
                clipped_tokens,
                axis=tuple(
                    range(1, clipped_tokens.ndim)
                ),
            )

        return (
            float(token_fraction),
            float(jnp.mean(sample_any)),
            float(global_exceed),
        )

    raise ValueError(f"Unknown clip_mode: {clip_mode}")

def refresh_state_and_collect_diagnostics(params, state, clamps, structure):
    """Refresh PC signals at the current latents without another update."""
    state = InferenceBase.zero_grads(params, state, clamps, structure)

    pre_mupc_topdown = {
        name: jnp.zeros_like(node_state.latent_grad)
        for name, node_state in state.nodes.items()
    }
    post_mupc_topdown = {
        name: jnp.zeros_like(node_state.latent_grad)
        for name, node_state in state.nodes.items()
    }
    mha_path_diagnostics = {}

    for node_name in structure.nodes:
        node = structure.nodes[node_name]
        node_info = node.node_info
        node_state = state.nodes[node_name]
        node_params = params.nodes[node_name]

        inputs = gather_inputs(node_info, structure, state)
        scaling = node_info.scaling_config
        scaled_inputs = scale_inputs(inputs, scaling)

        node_state, input_grads_pre, self_grad = (
            node_info.node_class.forward_and_latent_grads(
                node_params,
                scaled_inputs,
                node_state,
                node_info,
                is_clamped=(node_name in clamps),
            )
        )
        input_grads_post = scale_input_grads(input_grads_pre, scaling)

        # MhaResidualNode receives the same source through two distinct edges:
        # `in` traverses LayerNorm + attention, while `skip` is the identity
        # residual. Keep them separate before the inference loop sums them.
        if is_mha_module(params, node_name):
            pre_by_slot = {
                structure.edges[edge_key].slot: grad
                for edge_key, grad in input_grads_pre.items()
            }
            post_by_slot = {
                structure.edges[edge_key].slot: grad
                for edge_key, grad in input_grads_post.items()
            }

            required_slots = {"in", "skip"}
            if not required_slots.issubset(post_by_slot):
                raise RuntimeError(
                    f"{node_name} is missing MHA path gradients for "
                    f"{sorted(required_slots - set(post_by_slot))}"
                )

            attention_pre = pre_by_slot["in"]
            attention_post = post_by_slot["in"]
            skip_post = post_by_slot["skip"]
            combined_post = attention_post + skip_post

            attention_pre_norm = per_sample_l2(attention_pre)
            attention_post_norm = per_sample_l2(attention_post)
            skip_norm = per_sample_l2(skip_post)
            combined_norm = per_sample_l2(combined_post)
            path_norm_sum = attention_post_norm + skip_norm
            attention_retained = jnp.where(
                attention_pre_norm > 1e-12,
                attention_post_norm / attention_pre_norm,
                jnp.nan,
            )
            attention_to_skip = jnp.where(
                skip_norm > 1e-12,
                attention_post_norm / skip_norm,
                jnp.nan,
            )
            cancellation = jnp.where(
                path_norm_sum > 1e-12,
                combined_norm / path_norm_sum,
                jnp.nan,
            )

            mha_path_diagnostics[node_name] = {
                "attention_path_norm_pre_mupc": float(
                    jnp.mean(attention_pre_norm)
                ),
                "attention_path_norm_post_mupc": float(
                    jnp.mean(attention_post_norm)
                ),
                "attention_path_mupc_retained": finite_array_mean(
                    attention_retained
                ),
                "skip_path_norm": float(jnp.mean(skip_norm)),
                "attention_to_skip_norm_ratio": finite_array_mean(
                    attention_to_skip
                ),
                "combined_path_norm": float(jnp.mean(combined_norm)),
                "attention_skip_cosine": finite_array_mean(
                    per_sample_cosine(attention_post, skip_post)
                ),
                # 0 = exact cancellation, 1 = no cancellation.
                "cancellation_ratio": finite_array_mean(cancellation),
            }

        self_grad = scale_self_grad(self_grad, scaling)
        node_state = node_state._replace(
            latent_grad=node_state.latent_grad + self_grad
        )
        state = state._replace(nodes={**state.nodes, node_name: node_state})

        for edge_key, grad_pre in input_grads_pre.items():
            source_name = structure.edges[edge_key].source
            grad_post = input_grads_post[edge_key]

            pre_mupc_topdown[source_name] = (
                pre_mupc_topdown[source_name] + grad_pre
            )
            post_mupc_topdown[source_name] = (
                post_mupc_topdown[source_name] + grad_post
            )

            accumulated = state.nodes[source_name].latent_grad + grad_post
            state = update_node_in_state(
                state,
                source_name,
                latent_grad=accumulated,
            )

    inference_obj = structure.config["inference"]
    inference_config = inference_obj.config
    inference_cls = type(inference_obj)
    eps = inference_config.get("eps", 1e-8)

    diagnostics = {}

    for node_name, node_state in state.nodes.items():
        error_norm = per_sample_l2(node_state.error)
        latent_grad_norm = per_sample_l2(node_state.latent_grad)
        pre_topdown_norm = per_sample_l2(pre_mupc_topdown[node_name])
        post_topdown_norm = per_sample_l2(post_mupc_topdown[node_name])

        node_info = structure.nodes[node_name].node_info
        adaptive_eligible = (
            node_name not in clamps
            and node_info.in_degree > 0
            and node_info.out_degree > 0
        )

        if adaptive_eligible:
            # Exact signal used by InferenceSGDAdaptive.
            proposed_latent = inference_cls.compute_new_latent(
                node_name,
                node_state,
                inference_config,
            )
            delta = proposed_latent - node_state.z_latent

            delta_norm = jnp.sqrt(
                jnp.sum(delta.conj() * delta).real
            )
            latent_norm = jnp.sqrt(
                jnp.sum(
                    node_state.z_latent.conj()
                    * node_state.z_latent
                ).real
            )
            relative_proposed_update = float(
                delta_norm / (latent_norm + eps)
            )
        else:
            # Inputs, dropout masks and clamped output are not adaptive.
            relative_proposed_update = float("nan")

        if node_name in clamps:
            clipping_fraction = float("nan")
            sample_any_clipping_fraction = float("nan")
            global_norm_exceed_fraction = float("nan")
        else:
            (
                clipping_fraction,
                sample_any_clipping_fraction,
                global_norm_exceed_fraction,
            ) = clipping_fractions(
                node_state.latent_grad,
                inference_config,
            )

        diagnostics[node_name] = {
            "clipping_fraction": clipping_fraction,
            "sample_any_clipping_fraction": sample_any_clipping_fraction,
            "global_norm_exceed_fraction": global_norm_exceed_fraction,
            "local_error_norm": float(jnp.mean(error_norm)),
            "latent_grad_norm": float(jnp.mean(latent_grad_norm)),
            "topdown_grad_norm_pre_mupc": float(
                jnp.mean(pre_topdown_norm)
            ),
            "topdown_grad_norm_post_mupc": float(
                jnp.mean(post_topdown_norm)
            ),
            "relative_proposed_update": relative_proposed_update,
        }

    return state, diagnostics, mha_path_diagnostics

def pc_gradient(
    params,
    structure,
    batch,
    common_clamps,
    state_key,
):
    """
    Compute the local PC parameter gradient at the requested inference step.
    """
    batch_size = batch["x"].shape[0]

    clamps = dict(common_clamps)
    clamps[structure.task_map["x"]] = batch["x"]

    targets = jax.nn.one_hot(
        batch["y"],
        structure.nodes[
            structure.task_map["y"]
        ].node_info.shape[-1],
    )
    clamps[structure.task_map["y"]] = targets

    initial_state = initialize_graph_state(
        structure,
        batch_size,
        state_key,
        clamps=clamps,
        params=params,
    )

    final_state = run_inference(
        params,
        initial_state,
        clamps,
        structure,
    )

    diagnostic_state, diagnostics, mha_path_diagnostics = (
        refresh_state_and_collect_diagnostics(
            params,
            final_state,
            clamps,
            structure,
        )
    )

    gradients = compute_local_weight_gradients(
        params,
        diagnostic_state,
        structure,
    )

    return gradients, diagnostics, mha_path_diagnostics


def bp_gradient(
    params,
    structure,
    batch,
    common_clamps,
    state_key,
):
    """Compute BP using exactly the same output energy as PC."""

    output_node = structure.task_map["y"]

    targets = jax.nn.one_hot(
        batch["y"],
        structure.nodes[output_node].node_info.shape[-1],
    )

    token_count = batch["x"].shape[0] * batch["x"].shape[1]

    def loss_fn(current_params):
        clamps = dict(common_clamps)
        clamps[structure.task_map["x"]] = batch["x"]

        state = initialize_graph_state(
            structure,
            batch["x"].shape[0],
            state_key,
            clamps=clamps,
            params=current_params,
        )

        output_state = state.nodes[output_node]
        energy_obj = structure.nodes[output_node].node_info.energy

        # Use the identical CrossEntropyEnergy implementation used by the
        # clamped PC output node. Dividing by batch_size * seq_len matches
        # the normalization subsequently applied to raw PC gradients.
        per_sample_energy = type(energy_obj).energy(
            targets,
            output_state.z_mu,
            energy_obj.config,
        )

        return jnp.sum(per_sample_energy) / token_count

    return jax.grad(loss_fn)(params)


def flatten_node_gradient(grads, node_name):
    leaves = jax.tree_util.tree_leaves(grads.nodes[node_name])

    nonempty = [
        jnp.ravel(leaf)
        for leaf in leaves
        if leaf is not None and leaf.size > 0
    ]

    if not nonempty:
        return None

    return jnp.concatenate(nonempty)


def flatten_mha_component_gradient(grads, node_name, component):
    """Flatten one MHA projection (weight+bias) or LayerNorm group."""
    node_grads = grads.nodes[node_name]
    leaves = []

    for collection_name, parameter_name in MHA_COMPONENTS[component]:
        collection = getattr(node_grads, collection_name)
        if parameter_name in collection:
            value = collection[parameter_name]
            if value is not None and value.size > 0:
                leaves.append(jnp.ravel(value))

    if not leaves:
        return None

    return jnp.concatenate(leaves)


def cosine_similarity(x, y, eps=1e-12):
    denominator = jnp.linalg.norm(x) * jnp.linalg.norm(y)
    return jnp.where(
        denominator > eps,
        jnp.vdot(x, y) / denominator,
        jnp.nan,
    )


def average_vectors(vectors):
    return jnp.mean(jnp.stack(vectors), axis=0)

def finite_mean_std(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return float("nan"), float("nan"), 0

    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    return mean, std, int(values.size)


def summarize_paired_gradients(pc_vectors_raw, bp_vectors, reduction_factor):
    """Summarize matched PC/BP component gradients batch-first."""
    if len(pc_vectors_raw) != len(bp_vectors):
        raise RuntimeError(
            "Paired PC/BP component gradient counts do not match: "
            f"PC={len(pc_vectors_raw)}, BP={len(bp_vectors)}"
        )

    cosines = []
    norm_ratios = []
    pc_norms = []
    bp_norms = []
    eps = 1e-12

    for pc_raw, bp_vector in zip(pc_vectors_raw, bp_vectors):
        pc_vector = pc_raw / reduction_factor
        pc_norm = float(jnp.linalg.norm(pc_vector))
        bp_norm = float(jnp.linalg.norm(bp_vector))

        cosines.append(float(cosine_similarity(pc_vector, bp_vector)))
        norm_ratios.append(
            pc_norm / bp_norm if bp_norm > eps else float("nan")
        )
        pc_norms.append(pc_norm)
        bp_norms.append(bp_norm)

    cosine_mean, cosine_std, cosine_n = finite_mean_std(cosines)
    ratio_mean, ratio_std, ratio_n = finite_mean_std(norm_ratios)
    pc_norm_mean, pc_norm_std, _ = finite_mean_std(pc_norms)
    bp_norm_mean, bp_norm_std, _ = finite_mean_std(bp_norms)

    pc_mean = average_vectors(pc_vectors_raw) / reduction_factor
    bp_mean = average_vectors(bp_vectors)
    cosine_of_mean = float(cosine_similarity(pc_mean, bp_mean))
    bp_mean_norm = float(jnp.linalg.norm(bp_mean))
    norm_ratio_of_mean = (
        float(jnp.linalg.norm(pc_mean)) / bp_mean_norm
        if bp_mean_norm > eps
        else float("nan")
    )

    return {
        "cosine_similarity": cosine_mean,
        "cosine_similarity_std": cosine_std,
        "cosine_valid_batches": cosine_n,
        "norm_ratio_pc_over_bp": ratio_mean,
        "norm_ratio_pc_over_bp_std": ratio_std,
        "norm_ratio_valid_batches": ratio_n,
        "pc_grad_norm_normalized": pc_norm_mean,
        "pc_grad_norm_normalized_std": pc_norm_std,
        "bp_grad_norm": bp_norm_mean,
        "bp_grad_norm_std": bp_norm_std,
        "cosine_of_mean": cosine_of_mean,
        "norm_ratio_of_mean": norm_ratio_of_mean,
    }

def audit(
    params,
    config,
    loader,
    max_steps,
    num_batches,
    audit_seed,
    disable_mlp2_scaling=False,
    clip_mode="global",
):
    reference_structure = build_structure(
        config,
        loader.vocab_size,
        infer_steps=0,
        disable_mlp2_scaling=disable_mlp2_scaling,
        clip_mode=clip_mode,
    )

    batches = []
    common_inputs = []

    key = jax.random.PRNGKey(audit_seed)

    for batch_index, raw_batch in enumerate(loader):
        if batch_index >= num_batches:
            break

        batch = convert_batch(raw_batch)
        batch_key = jax.random.fold_in(key, batch_index)

        common_clamps, state_key = make_common_inputs(
            reference_structure,
            batch,
            batch_key,
        )

        batches.append(batch)
        common_inputs.append((common_clamps, state_key))

    if not batches:
        raise RuntimeError("The data loader returned no minibatches.")

    reduction_factors = {
        int(batch["x"].shape[0] * batch["x"].shape[1])
        for batch in batches
    }
    if len(reduction_factors) != 1:
        raise ValueError(
            "All audit batches must have the same batch_size and seq_len."
        )

    pc_reduction_factor = float(reduction_factors.pop())

    # BP is independent of PC inference step, so compute it once.
    bp_by_node = {}
    bp_mha_by_component = {}

    print("Computing BP gradients...")

    for batch, (common_clamps, state_key) in zip(
        batches,
        common_inputs,
    ):
        grads = bp_gradient(
            params,
            reference_structure,
            batch,
            common_clamps,
            state_key,
        )

        for node_name in reference_structure.nodes:
            vector = flatten_node_gradient(grads, node_name)
            if vector is not None:
                bp_by_node.setdefault(node_name, []).append(vector)

            if is_mha_module(params, node_name):
                for component in MHA_COMPONENTS:
                    component_vector = flatten_mha_component_gradient(
                        grads,
                        node_name,
                        component,
                    )
                    if component_vector is not None:
                        bp_mha_by_component.setdefault(
                            (node_name, component),
                            [],
                        ).append(component_vector)

    rows = []
    mha_component_rows = []
    mha_path_rows = []

    # Retain only the preceding sweep's PC gradients.
    previous_pc_by_node = {}

    for inference_step in range(max_steps + 1):
        print(f"Computing PC gradients: T={inference_step}")

        structure = build_structure(
            config,
            loader.vocab_size,
            infer_steps=inference_step,
            disable_mlp2_scaling=disable_mlp2_scaling,
            clip_mode=clip_mode,
        )

        pc_by_node = {}
        pc_mha_by_component = {}
        diagnostics_by_node = {}
        mha_paths_by_node = {}

        for batch, (common_clamps, state_key) in zip(
            batches,
            common_inputs,
        ):
            grads, batch_diagnostics, batch_mha_paths = pc_gradient(
                params,
                structure,
                batch,
                common_clamps,
                state_key,
            )

            for node_name in structure.nodes:
                vector = flatten_node_gradient(grads, node_name)
                if vector is not None:
                    pc_by_node.setdefault(node_name, []).append(vector)

                if node_name in batch_diagnostics:
                    node_diagnostics = diagnostics_by_node.setdefault(
                        node_name,
                        {
                            key: []
                            for key in batch_diagnostics[node_name]
                        },
                    )
                    for key, value in batch_diagnostics[node_name].items():
                        node_diagnostics[key].append(value)

                if is_mha_module(params, node_name):
                    for component in MHA_COMPONENTS:
                        component_vector = flatten_mha_component_gradient(
                            grads,
                            node_name,
                            component,
                        )
                        if component_vector is not None:
                            pc_mha_by_component.setdefault(
                                (node_name, component),
                                [],
                            ).append(component_vector)

            for node_name, path_metrics in batch_mha_paths.items():
                collected = mha_paths_by_node.setdefault(
                    node_name,
                    {key: [] for key in path_metrics},
                )
                for key, value in path_metrics.items():
                    collected[key].append(value)

        for node_name, pc_vectors_raw in pc_by_node.items():
            if node_name not in bp_by_node:
                continue

            bp_vectors = bp_by_node[node_name]

            if len(pc_vectors_raw) != len(bp_vectors):
                raise RuntimeError(
                    f"Batch mismatch for {node_name}: "
                    f"PC={len(pc_vectors_raw)}, BP={len(bp_vectors)}"
                )

            eps = 1e-12
            batch_cosines = []
            batch_norm_ratios = []

            for batch_index, (pc_raw, bp_vector) in enumerate(
                zip(pc_vectors_raw, bp_vectors)
            ):
                pc_vector = pc_raw / pc_reduction_factor

                cosine_value = float(
                    cosine_similarity(pc_vector, bp_vector)
                )
                pc_batch_norm = float(jnp.linalg.norm(pc_vector))
                bp_batch_norm = float(jnp.linalg.norm(bp_vector))

                norm_ratio = (
                    pc_batch_norm / bp_batch_norm
                    if bp_batch_norm > eps
                    else float("nan")
                )

                batch_cosines.append(cosine_value)
                batch_norm_ratios.append(norm_ratio)

                # Check every logits batch at T=0.
                if inference_step == 0 and node_name == structure.task_map["y"]:
                    if not np.isclose(cosine_value, 1.0, atol=1e-5):
                        raise AssertionError(
                            f"T=0 logits cosine failed on batch "
                            f"{batch_index}: {cosine_value}"
                        )

                    if not np.isclose(
                        norm_ratio, 1.0, rtol=1e-4, atol=1e-6
                    ):
                        raise AssertionError(
                            f"T=0 logits ratio failed on batch "
                            f"{batch_index}: {norm_ratio}"
                        )

            cosine_mean, cosine_std, cosine_n = finite_mean_std(
                batch_cosines
            )
            ratio_mean, ratio_std, ratio_n = finite_mean_std(
                batch_norm_ratios
            )

            consecutive_cosines = []
            relative_gradient_changes = []
            gradient_norm_ratios = []

            previous_vectors = previous_pc_by_node.get(node_name)

            if previous_vectors is not None:
                if len(previous_vectors) != len(pc_vectors_raw):
                    raise RuntimeError(
                        f"Previous/current batch mismatch for {node_name}"
                    )

                for current_raw, previous_raw in zip(
                    pc_vectors_raw,
                    previous_vectors,
                ):
                    current = current_raw / pc_reduction_factor
                    previous = previous_raw / pc_reduction_factor

                    current_norm = float(jnp.linalg.norm(current))
                    previous_norm = float(jnp.linalg.norm(previous))

                    if previous_norm > eps:
                        consecutive_cosines.append(
                            float(cosine_similarity(current, previous))
                        )
                        relative_gradient_changes.append(
                            float(jnp.linalg.norm(current - previous))
                            / previous_norm
                        )
                        gradient_norm_ratios.append(
                            current_norm / previous_norm
                        )
                    else:
                        consecutive_cosines.append(float("nan"))
                        relative_gradient_changes.append(float("nan"))
                        gradient_norm_ratios.append(float("nan"))

            consecutive_mean, consecutive_std, consecutive_n = (
                finite_mean_std(consecutive_cosines)
            )
            change_mean, change_std, change_n = finite_mean_std(
                relative_gradient_changes
            )
            norm_growth_mean, norm_growth_std, norm_growth_n = (
                finite_mean_std(gradient_norm_ratios)
            )

            diagnostic_summary = {}
            for key, values in diagnostics_by_node[node_name].items():
                mean, std, valid_batches = finite_mean_std(values)
                diagnostic_summary[key] = mean
                diagnostic_summary[f"{key}_std"] = std
                diagnostic_summary[f"{key}_valid_batches"] = valid_batches

            # Secondary: old cosine-of-averaged-gradients calculation.
            pc_vector_raw_mean = average_vectors(pc_vectors_raw)
            pc_vector_mean = pc_vector_raw_mean / pc_reduction_factor
            bp_vector_mean = average_vectors(bp_vectors)

            cosine_of_mean = float(
                cosine_similarity(pc_vector_mean, bp_vector_mean)
            )

            pc_norm_raw = float(jnp.linalg.norm(pc_vector_raw_mean))
            pc_norm = float(jnp.linalg.norm(pc_vector_mean))
            bp_norm = float(jnp.linalg.norm(bp_vector_mean))

            norm_ratio_of_mean = (
                pc_norm / bp_norm
                if bp_norm > eps
                else float("nan")
            )

            param_vector = flatten_node_gradient(params, node_name)
            param_norm = (
                float(jnp.linalg.norm(param_vector))
                if param_vector is not None
                else float("nan")
            )

            # Gradient-to-parameter ratios—not optimizer updates.
            pc_raw_grad_to_param_ratio = (
                pc_norm_raw / param_norm if param_norm > eps else float("nan")
            )
            pc_normalized_grad_to_param_ratio = (
                pc_norm / param_norm if param_norm > eps else float("nan")
            )
            bp_grad_to_param_ratio = (
                bp_norm / param_norm if param_norm > eps else float("nan")
            )

            rows.append(
                {
                    "inference_step": inference_step,
                    "module": node_name,
                    "cosine_similarity": cosine_mean,
                    "cosine_similarity_std": cosine_std,
                    "cosine_valid_batches": cosine_n,
                    "norm_ratio_pc_over_bp": ratio_mean,
                    "norm_ratio_pc_over_bp_std": ratio_std,
                    "norm_ratio_valid_batches": ratio_n,
                    "cosine_of_mean": cosine_of_mean,
                    "norm_ratio_of_mean": norm_ratio_of_mean,
                    "pc_raw_grad_to_param_ratio": pc_raw_grad_to_param_ratio,
                    "pc_normalized_grad_to_param_ratio": pc_normalized_grad_to_param_ratio,
                    "bp_grad_to_param_ratio": bp_grad_to_param_ratio,
                    "pc_reduction_factor": pc_reduction_factor,
                    "pc_grad_norm_raw": pc_norm_raw,
                    "pc_grad_norm_normalized": pc_norm,
                    "bp_grad_norm": bp_norm,
                    "pc_consecutive_gradient_cosine": consecutive_mean,
                    "pc_consecutive_gradient_cosine_std": consecutive_std,
                    "pc_consecutive_gradient_cosine_valid_batches": consecutive_n,
                    "pc_relative_gradient_change": change_mean,
                    "pc_relative_gradient_change_std": change_std,
                    "pc_relative_gradient_change_valid_batches": change_n,
                    "pc_gradient_norm_ratio_to_previous": norm_growth_mean,
                    "pc_gradient_norm_ratio_to_previous_std": norm_growth_std,
                    "pc_gradient_norm_ratio_to_previous_valid_batches": norm_growth_n,
                    **diagnostic_summary,
                }
            )

        for key, pc_component_vectors in pc_mha_by_component.items():
            if key not in bp_mha_by_component:
                continue

            node_name, component = key
            summary = summarize_paired_gradients(
                pc_component_vectors,
                bp_mha_by_component[key],
                pc_reduction_factor,
            )
            mha_component_rows.append(
                {
                    "inference_step": inference_step,
                    "module": node_name,
                    "component": component,
                    **summary,
                }
            )

        for node_name, metrics in mha_paths_by_node.items():
            row = {
                "inference_step": inference_step,
                "module": node_name,
            }
            for metric_name, values in metrics.items():
                mean, std, valid_batches = finite_mean_std(values)
                row[metric_name] = mean
                row[f"{metric_name}_std"] = std
                row[f"{metric_name}_valid_batches"] = valid_batches
            mha_path_rows.append(row)
        
        # These become the T-1 gradients on the next iteration.
        previous_pc_by_node = {
            node_name: tuple(vectors)
            for node_name, vectors in pc_by_node.items()
        }


    return rows, mha_component_rows, mha_path_rows


def save_csv(rows, output_path):
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "inference_step",
                "module",
                "cosine_similarity",
                "norm_ratio_pc_over_bp",
                "cosine_similarity_std",
                "cosine_valid_batches",
                "norm_ratio_pc_over_bp_std",
                "norm_ratio_valid_batches",
                "cosine_of_mean",
                "norm_ratio_of_mean",
                "pc_raw_grad_to_param_ratio",
                "pc_normalized_grad_to_param_ratio",
                "bp_grad_to_param_ratio",
                "pc_reduction_factor",
                "pc_grad_norm_raw",
                "pc_grad_norm_normalized",
                "bp_grad_norm",
                "clipping_fraction",
                "clipping_fraction_std",
                "clipping_fraction_valid_batches",
                "sample_any_clipping_fraction",
                "sample_any_clipping_fraction_std",
                "sample_any_clipping_fraction_valid_batches",
                "global_norm_exceed_fraction",
                "global_norm_exceed_fraction_std",
                "global_norm_exceed_fraction_valid_batches",
                "local_error_norm",
                "local_error_norm_std",
                "local_error_norm_valid_batches",
                "latent_grad_norm",
                "latent_grad_norm_std",
                "latent_grad_norm_valid_batches",
                "topdown_grad_norm_pre_mupc",
                "topdown_grad_norm_pre_mupc_std",
                "topdown_grad_norm_pre_mupc_valid_batches",
                "topdown_grad_norm_post_mupc",
                "topdown_grad_norm_post_mupc_std",
                "topdown_grad_norm_post_mupc_valid_batches",
                "relative_proposed_update",
                "relative_proposed_update_std",
                "relative_proposed_update_valid_batches",
                "pc_consecutive_gradient_cosine",
                "pc_consecutive_gradient_cosine_std",
                "pc_consecutive_gradient_cosine_valid_batches",
                "pc_relative_gradient_change",
                "pc_relative_gradient_change_std",
                "pc_relative_gradient_change_valid_batches",
                "pc_gradient_norm_ratio_to_previous",
                "pc_gradient_norm_ratio_to_previous_std",
                "pc_gradient_norm_ratio_to_previous_valid_batches",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def save_mha_component_csv(rows, output_path):
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "inference_step",
                "module",
                "component",
                "cosine_similarity",
                "cosine_similarity_std",
                "cosine_valid_batches",
                "norm_ratio_pc_over_bp",
                "norm_ratio_pc_over_bp_std",
                "norm_ratio_valid_batches",
                "pc_grad_norm_normalized",
                "pc_grad_norm_normalized_std",
                "bp_grad_norm",
                "bp_grad_norm_std",
                "cosine_of_mean",
                "norm_ratio_of_mean",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def save_mha_path_csv(rows, output_path):
    metrics = [
        "attention_path_norm_pre_mupc",
        "attention_path_norm_post_mupc",
        "attention_path_mupc_retained",
        "skip_path_norm",
        "attention_to_skip_norm_ratio",
        "combined_path_norm",
        "attention_skip_cosine",
        "cancellation_ratio",
    ]
    fieldnames = ["inference_step", "module"]
    for metric in metrics:
        fieldnames.extend(
            [
                metric,
                f"{metric}_std",
                f"{metric}_valid_batches",
            ]
        )

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def module_sort_key(name):
    """
    Put the output side near the top of the heatmap and the input side
    near the bottom. The graph's insertion order is otherwise preserved.
    """
    lower = name.lower()

    if "vocab" in lower or "logit" in lower:
        return (0, name)

    if "embed" in lower:
        return (3, name)

    return (1, name)


def plot_results(rows, output_dir, tag):
    steps = sorted({row["inference_step"] for row in rows})
    present_modules = {row["module"] for row in rows}

    modules = [
        module for module in MODULE_ORDER
        if module in present_modules
    ]

    lookup = {
        (row["module"], row["inference_step"]):
        row["cosine_similarity"]
        for row in rows
    }

    matrix = np.full(
        (len(modules), len(steps)),
        np.nan,
        dtype=float,
    )

    for module_index, module in enumerate(modules):
        for step_index, step in enumerate(steps):
            matrix[module_index, step_index] = lookup.get(
                (module, step),
                np.nan,
            )

    # Heatmap
    fig_width = max(9, 0.32 * len(steps))
    fig_height = max(5, 0.48 * len(modules))

    fig, ax = plt.subplots(
        figsize=(fig_width, fig_height),
        constrained_layout=True,
    )

    image = ax.imshow(
        matrix,
        aspect="auto",
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        interpolation="nearest",
    )

    ax.set_xticks(np.arange(len(steps)))
    ax.set_xticklabels(steps)
    ax.set_yticks(np.arange(len(modules)))
    ax.set_yticklabels(modules)
    ax.set_xlabel("PC inference steps")
    ax.set_ylabel("Module")
    ax.set_title("Same-checkpoint PC–BP gradient alignment")

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Cosine similarity")

    heatmap_path = os.path.join(
        output_dir,
        f"{tag}_heatmap.png",
    )
    fig.savefig(heatmap_path, dpi=300)
    plt.close(fig)

    # Mean alignment across modules
    mean_alignment = np.nanmean(matrix, axis=0)

    valid = np.asarray(steps) > 0

    fig, ax = plt.subplots(
        figsize=(7, 4.5),
        constrained_layout=True,
    )

    ax.plot(
        np.asarray(steps)[valid],
        mean_alignment[valid],
        marker="o",
        linewidth=2,
    )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlabel("PC inference steps")
    ax.set_ylabel("Mean cosine similarity")
    ax.set_title("Mean PC–BP gradient alignment")
    ax.grid(alpha=0.25)

    curve_path = os.path.join(
        output_dir,
        f"{tag}_mean_alignment.png",
    )
    fig.savefig(curve_path, dpi=300)
    plt.close(fig)

    return heatmap_path, curve_path


def plot_norm_ratio(rows, output_dir, tag):
    """Heatmap of log2(||g_PC|| / ||g_BP||): 0 = equal, blue = PC starved, red = PC larger."""
    modules = [
        module for module in MODULE_ORDER
        if any(r["module"] == module for r in rows)
    ]
    steps = sorted({r["inference_step"] for r in rows})
    matrix = np.full((len(modules), len(steps)), np.nan)
    for mi, module in enumerate(modules):
        for si, step in enumerate(steps):
            for r in rows:
                if r["module"] == module and r["inference_step"] == step:
                    ratio = r.get("norm_ratio_pc_over_bp", np.nan)
                    if ratio is not None and np.isfinite(ratio) and ratio > 0:
                        matrix[mi, si] = np.log2(ratio)
                    break

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-4, vmax=4)
    ax.set_xticks(np.arange(len(steps)))
    ax.set_xticklabels(steps)
    ax.set_yticks(np.arange(len(modules)))
    ax.set_yticklabels(modules)
    ax.set_xlabel("PC inference steps")
    ax.set_ylabel("Module")
    ax.set_title("PC/BP gradient magnitude  (log2 ratio: <0 = PC starved)")
    fig.colorbar(im, ax=ax, label="log2(||g_PC|| / ||g_BP||)")
    path = os.path.join(output_dir, f"{tag}_norm_ratio.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    params, config = load_checkpoint(args.checkpoint)
    loader = make_loader(config)

    print(f"Audit data split: {AUDIT_SPLIT}")
    checkpoint_tag = os.path.splitext(
        os.path.basename(args.checkpoint)
    )[0]

    active_ablations = sum(
        [
            bool(args.disable_mlp2_input_scaling),
            args.wff2_multiplier != 1.0,
            args.clip_mode != "global",
        ]
    )

    if active_ablations > 1:
        raise ValueError(
            "Run only one ablation at a time: MLP2 scaling, "
            "W_ff2, or clipping mode."
        )

    max_norm = config.get("max_norm", 5.0)

    if args.clip_mode == "global":
        print(
            f"Clipping: global sample L2, "
            f"threshold={max_norm:.6g}"
        )

    elif args.clip_mode == "rms":
        max_rms = max_norm / math.sqrt(
            config["seq_len"] * config["embed_dim"]
        )
        print(
            f"Clipping ablation: whole-state RMS, "
            f"threshold={max_rms:.6g}"
        )
        checkpoint_tag += "_clip_rms"

    elif args.clip_mode == "per_token":
        max_token_norm = (
            max_norm / math.sqrt(config["seq_len"])
        )
        print(
            f"Clipping ablation: per-token L2, "
            f"threshold={max_token_norm:.6g}"
        )
        checkpoint_tag += "_clip_per_token"

    if args.disable_mlp2_input_scaling:
        print(
            "Ablation: MLP2 input muPC scaling disabled; "
            "checkpoint W_ff2 unchanged"
        )
        checkpoint_tag += "_no_mlp2_input_scaling"
    
    if args.wff2_multiplier != 1.0:
        params, changed = rescale_wff2(
            params,
            args.wff2_multiplier,
        )

        if len(changed) != config["depth"]:
            raise RuntimeError(
                "W_ff2 rescaling changed "
                f"{len(changed)} nodes, expected {config['depth']}"
            )

        print(
            f"Ablation: W_ff2 multiplied by "
            f"{args.wff2_multiplier:.8g}; "
            "muPC scaling unchanged"
        )

        for node_name, old_rms, new_rms in changed:
            print(
                f"  {node_name}: W_ff2 RMS "
                f"{old_rms:.6g} -> {new_rms:.6g}"
            )

        multiplier_tag = (
            f"{args.wff2_multiplier:.4g}".replace(".", "p")
        )
        checkpoint_tag += f"_wff2_x{multiplier_tag}"

    rows, mha_component_rows, mha_path_rows = audit(
        params=params,
        config=config,
        loader=loader,
        max_steps=args.max_steps,
        num_batches=args.num_batches,
        audit_seed=args.audit_seed,
        disable_mlp2_scaling=args.disable_mlp2_input_scaling,
        clip_mode=args.clip_mode,
    )

    csv_path = os.path.join(
        args.output_dir,
        f"{checkpoint_tag}_alignment.csv",
    )
    save_csv(rows, csv_path)

    mha_component_csv_path = os.path.join(
        args.output_dir,
        f"{checkpoint_tag}_mha_components.csv",
    )
    save_mha_component_csv(mha_component_rows, mha_component_csv_path)

    mha_path_csv_path = os.path.join(
        args.output_dir,
        f"{checkpoint_tag}_mha_paths.csv",
    )
    save_mha_path_csv(mha_path_rows, mha_path_csv_path)

    heatmap_path, curve_path = plot_results(
        rows,
        args.output_dir,
        checkpoint_tag,
    )
    norm_ratio_path = plot_norm_ratio(rows, args.output_dir, checkpoint_tag)

    print("\nAudit complete")
    print(f"CSV:        {csv_path}")
    print(f"MHA parts:  {mha_component_csv_path}")
    print(f"MHA paths:  {mha_path_csv_path}")
    print(f"Cosine:     {heatmap_path}")
    print(f"Curve:      {curve_path}")
    print(f"Magnitude:  {norm_ratio_path}")


if __name__ == "__main__":
    main()
