"""Mode-dispatched masked-language-model training helpers."""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Tuple, cast

import jax
import jax.numpy as jnp
import numpy as np
import optax

from fabricpc.core.depth import output_node_names
from fabricpc.core.inference import (
    InferenceSGDNormClipDecay,
    InferenceSpikingPrecision,
    run_inference,
)
from fabricpc.core.learning import (
    FORWARD_UPDATE_MODES,
    compute_local_weight_gradients,
    compute_local_weight_gradients_forward_update,
)
from fabricpc.core.types import GraphParams, GraphState, GraphStructure
from fabricpc.graph_initialization.state_initializer import (
    FeedforwardStateInit,
    initialize_graph_state,
)
from fabricpc.training.train import _convert_batch
from fabricpc.training.train_autoregressive import compute_loss, create_dropout_clamps
from fabricpc.training.train_backprop import (
    _apply_optimizer_update_when_masked,
    _loader_metadata,
    _loss_and_grad_backprop_mlm_no_update,
    _serializable_training_config,
    evaluate_mlm,
    load_mlm_training_checkpoint,
    save_mlm_training_checkpoint,
    train_backprop_mlm,
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return repr(value)


def _object_identity(value: Any) -> dict[str, Any]:
    return {
        "class": type(value).__name__,
        "config": _json_safe(dict(getattr(value, "config", {}))),
        "repr": repr(value),
    }


def _config_training_identity(config: dict[str, Any]) -> dict[str, Any]:
    mutable_resume_keys = {
        "max_steps",
        "checkpoint_path",
        "checkpoint_interval",
        "resume_from",
        "log_interval",
        "validation_interval",
        "init_diagnostics",
    }
    serializable = _serializable_training_config(config)
    identity = {}
    for key, value in serializable.items():
        if key in mutable_resume_keys:
            continue
        if key == "optimizer_identity" and isinstance(value, dict):
            value = {
                sub_key: sub_value
                for sub_key, sub_value in value.items()
                if sub_key != "total_steps"
            }
        identity[key] = value
    return _json_safe(identity)


def _pc_training_identity(
    structure: GraphStructure,
    train_loader: Any,
    pc_incremental: bool = False,
) -> dict[str, Any]:
    inference = structure.config["inference"]
    identity: dict[str, Any] = {"pc_incremental": True} if pc_incremental else {}
    return {
        **identity,
        "inference": {
            "class": type(inference).__name__,
            "config": _json_safe(dict(inference.config)),
        },
        "node_order": list(structure.nodes),
        "edges": sorted(structure.edges),
        "nodes": {
            name: {
                "class": node.node_info.node_class.__name__,
                "shape": list(node.node_info.shape),
                "config": _json_safe(dict(node.node_info.node_config)),
                "activation": _object_identity(node.node_info.activation),
                "energy": _object_identity(node.node_info.energy),
                "scaling": _json_safe(node.node_info.scaling_config),
            }
            for name, node in structure.nodes.items()
        },
        "task_map": dict(structure.task_map),
        "loader": _loader_training_identity(train_loader),
    }


def _loader_training_identity(train_loader: Any) -> dict[str, Any]:
    identity = {
        "class": type(train_loader).__name__,
        "metadata": _loader_metadata(train_loader),
    }
    for name in (
        "seq_len",
        "batch_size",
        "pad_token_id",
        "mask_token_id",
        "vocab_size",
        "cls_token_id",
        "seed",
        "mask_prob",
        "stride",
        "stream_window",
        "shuffle",
        "random_replacement_excludes_special",
        "token_unit",
        "num_windows",
        "num_full_batches",
    ):
        if hasattr(train_loader, name):
            identity[name] = getattr(train_loader, name)
    if hasattr(train_loader, "special_token_ids"):
        identity["special_token_ids"] = sorted(train_loader.special_token_ids)
    if hasattr(train_loader, "starts"):
        starts = getattr(train_loader, "starts")
        identity["starts_hash"] = hashlib.sha256(starts.tobytes()).hexdigest()
        identity["starts_shape"] = list(starts.shape)
    if hasattr(train_loader, "token_ids"):
        token_ids = getattr(train_loader, "token_ids")
        identity["token_ids_hash"] = hashlib.sha256(token_ids.tobytes()).hexdigest()
        identity["token_ids_shape"] = list(token_ids.shape)
    return _json_safe(identity)


def _validate_pcn_mlm_batch(
    batch: Dict[str, jnp.ndarray], structure: GraphStructure
) -> None:
    if "loss_mask" not in batch:
        raise ValueError("MLM PC training requires batch['loss_mask']")
    if "loss_mask" not in structure.task_map:
        raise ValueError("MLM PC graph requires a 'loss_mask' task_map entry")
    if batch["loss_mask"].shape != batch["y"].shape:
        raise ValueError(
            "MLM PC loss_mask shape must match target token shape exactly; "
            f"got {batch['loss_mask'].shape}, expected {batch['y'].shape}"
        )


def _raw_batch_has_masked_tokens(batch_data: Any) -> bool:
    if not isinstance(batch_data, dict) or "loss_mask" not in batch_data:
        return True
    return bool(np.asarray(batch_data["loss_mask"]).sum() > 0)


def _masked_accuracy(
    free_state: GraphState,
    batch: Dict[str, jnp.ndarray],
    output_node: str,
) -> jax.Array:
    predictions = free_state.nodes[output_node].z_mu
    pred_ids = jnp.argmax(predictions, axis=-1)
    mask = batch["loss_mask"].astype(jnp.float32)
    correct = (pred_ids == batch["y"]).astype(jnp.float32) * mask
    return jnp.sum(correct) / jnp.maximum(jnp.sum(mask), 1.0)


def _prepare_pcn_mlm_batch(
    params: GraphParams,
    batch: Dict[str, jnp.ndarray],
    structure: GraphStructure,
    rng_key: jax.Array,
) -> tuple[
    dict[str, jnp.ndarray],
    dict[str, jax.Array],
    GraphState,
    GraphState,
    dict[str, jnp.ndarray],
]:
    """Build the clamps, free-state metrics and target-clamped initial state.

    Returns ``(shared_clamps, free_metrics, free_state, initial_state,
    clamps)``. ``free_metrics`` holds ``masked_loss``, ``masked_accuracy`` and
    ``num_masked_tokens`` of the feedforward state without target clamps.
    Dropout masks come from the same key split as ``compute_forward_pass``,
    so PC and backprop see identical masks for the same ``rng_key``.
    """
    _validate_pcn_mlm_batch(batch, structure)
    _, free_state_key, pc_state_key = jax.random.split(rng_key, 3)
    dropout_key, _ = jax.random.split(rng_key)
    batch_size = batch["x"].shape[0]
    output_node = structure.task_map["y"]

    shared_clamps = {
        structure.task_map["x"]: batch["x"],
        structure.task_map["loss_mask"]: batch["loss_mask"].astype(jnp.float32),
    }
    if "attention_mask" in structure.task_map and "attention_mask" in batch:
        shared_clamps[structure.task_map["attention_mask"]] = batch[
            "attention_mask"
        ].astype(jnp.float32)
    shared_clamps.update(
        create_dropout_clamps(structure, batch_size, dropout_key, training=True)
    )

    free_state = initialize_graph_state(
        structure,
        batch_size,
        free_state_key,
        clamps=shared_clamps,
        params=params,
    )
    free_metrics = {
        "masked_loss": compute_loss(
            free_state,
            batch["y"],
            output_node,
            loss_type="cross_entropy",
            loss_mask=batch["loss_mask"],
        ),
        "masked_accuracy": _masked_accuracy(free_state, batch, output_node),
        "num_masked_tokens": jnp.sum(batch["loss_mask"].astype(jnp.float32)),
    }

    vocab_size = structure.nodes[output_node].node_info.shape[-1]
    targets = jax.nn.one_hot(batch["y"], vocab_size, dtype=jnp.float32)
    clamps = {**shared_clamps, output_node: targets}
    initial_state = initialize_graph_state(
        structure,
        batch_size,
        pc_state_key,
        clamps=clamps,
        params=params,
    )
    return shared_clamps, free_metrics, free_state, initial_state, clamps


def _validate_forward_updates(forward_updates: str) -> None:
    allowed = ("none", *FORWARD_UPDATE_MODES)
    if forward_updates not in allowed:
        raise ValueError(
            f"pc_forward_updates must be one of {allowed}, got {forward_updates!r}"
        )


def _validate_feedforward_start(
    structure: GraphStructure, forward_updates: str
) -> None:
    """Require feedforward state init for forward updates and spiking inference.

    Forward updates read the feedforward prediction from the initial state and
    the spike schedule assumes zero hidden errors at the start.
    """
    spiking = isinstance(structure.config["inference"], InferenceSpikingPrecision)
    if forward_updates == "none" and not spiking:
        return
    if not isinstance(
        structure.config.get("graph_state_initializer"), FeedforwardStateInit
    ):
        raise ValueError(
            "pc_forward_updates and InferenceSpikingPrecision require "
            "FeedforwardStateInit as the graph_state_initializer"
        )


def ipc_weight_spike_steps(
    structure: GraphStructure, clamps: Dict[str, jnp.ndarray]
) -> dict[str, int]:
    """Return the iPC iteration at which each node's weight gradient is scaled.

    This is the spike step + 1, the first iteration whose state holds the
    spiked latent. Output nodes and clamped nodes never spike and are left out.
    """
    inference_obj = structure.config["inference"]
    outputs = set(output_node_names(structure))
    return {
        name: step + 1
        for name, step in type(inference_obj)
        .spike_steps(structure, dict(inference_obj.config))
        .items()
        if name not in outputs
        and name not in clamps
        and structure.nodes[name].node_info.in_degree > 0
    }


def get_graph_param_gradient_pcn_mlm(
    params: GraphParams,
    batch: Dict[str, jnp.ndarray],
    structure: GraphStructure,
    rng_key: jax.Array,
    forward_updates: str = "none",
) -> Tuple[GraphParams, dict[str, jax.Array], GraphState, GraphState, GraphState]:
    """Compute one BERT MLM PC local-gradient step without applying updates.

    ``forward_updates`` selects the weight-gradient rule: ``"none"`` uses
    ``compute_local_weight_gradients`` on the final state, ``"initial"`` and
    ``"final"`` use ``compute_local_weight_gradients_forward_update`` in that
    mode.
    """
    _validate_forward_updates(forward_updates)
    _validate_feedforward_start(structure, forward_updates)
    _shared_clamps, free_metrics, free_state, initial_state, clamps = (
        _prepare_pcn_mlm_batch(params, batch, structure, rng_key)
    )
    batch_size = batch["x"].shape[0]
    output_node = structure.task_map["y"]
    inference_cls = type(structure.config["inference"])
    if hasattr(inference_cls, "run_inference_with_stats"):
        final_state, inference_stats = inference_cls.run_inference_with_stats(
            params, initial_state, clamps, structure
        )
        energy_rises = inference_stats["energy_rises"]
    else:
        final_state = run_inference(params, initial_state, clamps, structure)
        energy_rises = jnp.array(0, dtype=jnp.int32)
    if forward_updates == "none":
        grads = compute_local_weight_gradients(params, final_state, structure)
    else:
        grads = compute_local_weight_gradients_forward_update(
            params, initial_state, final_state, structure, forward_updates
        )
    output_energy = jnp.sum(final_state.nodes[output_node].energy) / batch_size
    total_energy = jnp.array(0.0)
    for node_name, node in structure.nodes.items():
        if node.node_info.in_degree > 0:
            total_energy = total_energy + jnp.sum(final_state.nodes[node_name].energy)
    total_energy = total_energy / batch_size
    metrics = {
        **free_metrics,
        "pc_output_energy": output_energy,
        "pc_total_energy": total_energy,
        "pc_energy_rises": energy_rises,
    }
    return grads, metrics, initial_state, free_state, final_state


def get_graph_param_gradient_pcn_mlm_accumulation(
    params: GraphParams,
    batch: Dict[str, jnp.ndarray],
    structure: GraphStructure,
    rng_key: jax.Array,
    forward_updates: str = "none",
) -> tuple[GraphParams, jax.Array, jax.Array, jax.Array]:
    """Compute lean PC MLM grads, loss totals and inference energy rises."""
    grads, metrics, _initial_state, _free_state, _final_state = (
        get_graph_param_gradient_pcn_mlm(
            params,
            batch,
            structure,
            rng_key,
            forward_updates,
        )
    )
    masked_count = metrics["num_masked_tokens"]
    loss_numerator = metrics["masked_loss"] * masked_count
    return grads, loss_numerator, masked_count, metrics["pc_energy_rises"]


def pc_node_gradient_norms(
    params: GraphParams,
    batch: Dict[str, jnp.ndarray],
    structure: GraphStructure,
    rng_key: jax.Array,
    forward_updates: str = "none",
) -> dict[str, float]:
    """Return the global L2 norm of each parameterised node's PC gradient.

    Starting from feedforward latents, inference moves the output error back
    about one node per step, so with too few inference steps the earliest
    nodes, embeddings included, get an exactly zero gradient.
    """
    grads, _metrics, _initial, _free, _final = get_graph_param_gradient_pcn_mlm(
        params, batch, structure, rng_key, forward_updates
    )
    norms = {}
    for name, node_grads in grads.nodes.items():
        leaves = jax.tree_util.tree_leaves(node_grads)
        if leaves:
            norms[name] = float(optax.tree.norm(leaves))
    return norms


def create_pc_bp_gradient_comparison(
    structure: GraphStructure, forward_updates: str = "none"
):
    """Create a JIT-compiled per-node PC vs backprop gradient comparison.

    The returned function maps ``(params, batch, rng_key)`` to
    ``{node: {"pc_norm", "bp_norm", "cosine"}}`` for every parameterised node,
    comparing PC local gradients with backprop gradients of the masked loss on
    the same batch. Norm ratios carry a global loss-scale constant; compare
    them against the ``mlm_head`` ratio rather than against one. PC gradients
    use the training rule selected by ``forward_updates``.
    """
    _validate_forward_updates(forward_updates)
    _validate_feedforward_start(structure, forward_updates)

    def compare(params, batch, rng_key):
        pc_grads, _metrics, _initial, _free, _final = get_graph_param_gradient_pcn_mlm(
            params, batch, structure, rng_key, forward_updates
        )
        _loss, bp_grads, _count = _loss_and_grad_backprop_mlm_no_update(
            params, batch, structure, rng_key
        )
        result = {}
        for name, pc_node in pc_grads.nodes.items():
            pc_leaves = jax.tree_util.tree_leaves(pc_node)
            if not pc_leaves:
                continue
            bp_leaves = jax.tree_util.tree_leaves(bp_grads.nodes[name])
            dot = sum(jnp.vdot(p, b) for p, b in zip(pc_leaves, bp_leaves))
            pc_norm = optax.tree.norm(pc_leaves)
            bp_norm = optax.tree.norm(bp_leaves)
            result[name] = {
                "pc_norm": pc_norm,
                "bp_norm": bp_norm,
                "cosine": dot / jnp.maximum(pc_norm * bp_norm, 1e-30),
            }
        return result

    return jax.jit(compare)


def train_step_pcn_mlm(
    params: GraphParams,
    opt_state: optax.OptState,
    batch: Dict[str, jnp.ndarray],
    structure: GraphStructure,
    optimizer: optax.GradientTransformation,
    rng_key: jax.Array,
    forward_updates: str = "none",
) -> Tuple[
    GraphParams,
    optax.OptState,
    dict[str, jax.Array],
    GraphState,
    GraphState,
    GraphState,
]:
    """Single-device predictive-coding MLM training step.

    ``forward_updates`` selects the weight-gradient rule as in
    ``get_graph_param_gradient_pcn_mlm``.
    """
    grads, metrics, initial_state, free_state, final_state = (
        get_graph_param_gradient_pcn_mlm(
            params,
            batch,
            structure,
            rng_key,
            forward_updates,
        )
    )
    params, opt_state = _apply_optimizer_update_when_masked(
        params=params,
        opt_state=opt_state,
        grads=grads,
        optimizer=optimizer,
        masked_count=metrics["num_masked_tokens"],
    )
    return params, opt_state, metrics, initial_state, free_state, final_state


def create_jit_gradient_step_pcn_mlm(
    structure: GraphStructure, forward_updates: str = "none"
):
    """Create a JIT-compiled PC MLM gradient step for accumulation."""
    _validate_forward_updates(forward_updates)
    return jax.jit(
        lambda p, b, k: get_graph_param_gradient_pcn_mlm_accumulation(
            p, b, structure, k, forward_updates
        )
    )


def create_jit_update_step_pcn_mlm(optimizer: optax.GradientTransformation):
    """Create a JIT-compiled optimizer update for accumulated PC MLM grads."""
    return jax.jit(
        lambda p, o, g, c: _apply_optimizer_update_when_masked(
            params=p,
            opt_state=o,
            grads=g,
            optimizer=optimizer,
            masked_count=c,
        )
    )


def _train_step_pcn_mlm_incremental_with_state(
    params: GraphParams,
    opt_state: optax.OptState,
    batch: Dict[str, jnp.ndarray],
    structure: GraphStructure,
    optimizer: optax.GradientTransformation,
    rng_key: jax.Array,
) -> tuple[GraphParams, optax.OptState, dict[str, jax.Array], GraphState]:
    """Run one incremental PC MLM batch and also return the final state."""
    _shared_clamps, free_metrics, _free_state, initial_state, clamps = (
        _prepare_pcn_mlm_batch(params, batch, structure, rng_key)
    )
    inference_obj = structure.config["inference"]
    inference_cls = type(inference_obj)
    config = inference_obj.config
    guarded = issubclass(inference_cls, InferenceSGDNormClipDecay)
    spiking = issubclass(inference_cls, InferenceSpikingPrecision)
    if spiking:
        _validate_feedforward_start(structure, "none")
        if not config["eta_infer"] > 0.0:
            raise ValueError("incremental PC with spiking needs eta_infer > 0")
        if config["depth_mode"] != "longest":
            raise ValueError(
                "incremental PC with spiking supports only depth_mode='longest'"
            )
        weight_spike_steps = ipc_weight_spike_steps(structure, clamps)
        spike_ratio = config["spike_eta"] / config["eta_infer"]
    masked_count = free_metrics["num_masked_tokens"]
    batch_size = batch["x"].shape[0]
    float_dtype = jnp.result_type(float)
    if guarded:
        eta0, prev_energy0 = InferenceSGDNormClipDecay.initial_guard_carry(
            batch_size, dict(config)
        )
    elif spiking:
        eta0 = jnp.zeros((batch_size,), dtype=float_dtype)
        prev_energy0 = jnp.full((batch_size,), jnp.inf, dtype=float_dtype)
    else:
        eta0 = jnp.zeros((batch_size,), dtype=float_dtype)
        prev_energy0 = jnp.zeros((batch_size,), dtype=float_dtype)

    def body_fn(t, carry):
        params, opt_state, state, eta, prev_energy, rises, grad_norm_sum = carry
        grads = compute_local_weight_gradients(params, state, structure)
        grad_norm_sum = grad_norm_sum + optax.tree.norm(grads).astype(float_dtype)
        if guarded:
            state, eta, prev_energy, step_rises = inference_cls.guarded_step(
                params, state, eta, prev_energy, clamps, structure
            )
            rises = rises + step_rises
        elif spiking:
            state, prev_energy, step_rises = inference_cls.spiking_step(
                params, state, t, prev_energy, clamps, structure
            )
            rises = rises + step_rises
            grads = _scale_spike_step_grads(grads, weight_spike_steps, spike_ratio, t)
        else:
            state = inference_cls.inference_step(
                params, state, clamps, structure, config
            )
        params, opt_state = _apply_optimizer_update_when_masked(
            params=params,
            opt_state=opt_state,
            grads=grads,
            optimizer=optimizer,
            masked_count=masked_count,
        )
        return params, opt_state, state, eta, prev_energy, rises, grad_norm_sum

    infer_steps = config["infer_steps"]
    carry = (
        params,
        opt_state,
        initial_state,
        eta0,
        prev_energy0,
        jnp.array(0, dtype=jnp.int32),
        jnp.array(0.0, dtype=float_dtype),
    )
    params, opt_state, final_state, _eta, _energy, rises, grad_norm_sum = (
        jax.lax.fori_loop(0, infer_steps, body_fn, carry)
    )
    metrics = {
        **free_metrics,
        "pc_energy_rises": rises,
        "grad_norm_mean": grad_norm_sum / max(int(infer_steps), 1),
    }
    return params, opt_state, metrics, final_state


def _scale_spike_step_grads(
    grads: GraphParams,
    spike_steps: dict[str, int],
    spike_ratio: float,
    t: jax.Array,
) -> GraphParams:
    """Scale a node's gradient by ``spike_ratio`` where ``t`` equals its entry.

    ``spike_steps`` holds the spike step + 1 per node (see
    ``ipc_weight_spike_steps``). The ratio ``spike_eta / eta_infer`` matches
    the latent step ratio only approximately, since spike steps are unclipped
    and other steps are norm-clipped.
    """
    nodes = {}
    for name, node_grads in grads.nodes.items():
        if name in spike_steps:
            factor = jnp.where(t == spike_steps[name], spike_ratio, 1.0)
            node_grads = jax.tree_util.tree_map(
                lambda grad: grad * factor.astype(grad.dtype), node_grads
            )
        nodes[name] = node_grads
    return GraphParams(nodes=nodes)


def train_step_pcn_mlm_incremental(
    params: GraphParams,
    opt_state: optax.OptState,
    batch: Dict[str, jnp.ndarray],
    structure: GraphStructure,
    optimizer: optax.GradientTransformation,
    rng_key: jax.Array,
) -> tuple[GraphParams, optax.OptState, dict[str, jax.Array]]:
    """Incremental predictive-coding (iPC) MLM training step.

    Follows Algorithm 1 of Salvatori et al. 2024, "A Stable, Fast, and Fully
    Automatic Learning Algorithm for Predictive Coding Networks" (ICLR,
    arXiv 2212.00720): starting from the target-clamped feedforward state, each
    of the ``infer_steps`` iterations takes latent and local weight gradients
    at the same state, updates the latents (with the energy-rise guard when the
    inference is ``InferenceSGDNormClipDecay``) and applies one optimizer
    update. Batches without masked tokens leave params and optimizer state
    unchanged.

    With ``InferenceSpikingPrecision`` the latent step ``t`` uses the spike
    schedule, and each node's weight gradient is multiplied by
    ``spike_eta / eta_infer`` at the iteration after its spike step, the
    first one whose state holds the spiked latent, before the optimizer update:
    the ``eta / Sigma`` weight step of Alg. 1, Qi et al. 2025, for latents
    updated before weights. At the spike step itself the node's error is
    still zero from a feedforward start. The scaling acts on the gradient, so
    it scales the update linearly only for plain SGD; Adam-type optimizers
    renormalise it.

    Metrics hold ``masked_loss``, ``masked_accuracy`` and
    ``num_masked_tokens`` of the feedforward state before any update,
    ``pc_energy_rises`` (int32, zero without the guard) and ``grad_norm_mean``,
    the mean over iterations of the unclipped local weight-gradient norm.
    """
    params, opt_state, metrics, _final_state = (
        _train_step_pcn_mlm_incremental_with_state(
            params, opt_state, batch, structure, optimizer, rng_key
        )
    )
    return params, opt_state, metrics


def create_jit_train_step_pcn_mlm_incremental(
    structure: GraphStructure, optimizer: optax.GradientTransformation
):
    """Create a JIT-compiled incremental PC MLM training step."""
    return jax.jit(
        lambda p, o, b, k: train_step_pcn_mlm_incremental(
            p, o, b, structure, optimizer, k
        )
    )


def train_pcn_mlm(
    params: GraphParams,
    structure: GraphStructure,
    train_loader: Any,
    optimizer: optax.GradientTransformation,
    config: dict,
    rng_key: jax.Array,
    val_loader: Any = None,
    verbose: bool = True,
) -> tuple[GraphParams, dict[str, Any]]:
    """Train a BERT-style MLM graph with single-device PC updates.

    With ``config["pc_incremental"]`` true each batch runs incremental PC
    (``train_step_pcn_mlm_incremental``), applying one optimizer update per
    inference iteration; gradient accumulation is then rejected.
    ``config["pc_forward_updates"]`` ("none", "initial" or "final") selects
    the weight-gradient rule of standard PC and is rejected with incremental
    PC.
    """
    if config.get("use_pmap"):
        raise ValueError("pmap PC MLM training is not implemented yet")
    if config.get("pmap_single_device"):
        raise ValueError("pmap PC MLM training is not implemented yet")
    if config.get("metrics_jsonl"):
        raise ValueError("PC MLM metrics_jsonl logging is not implemented yet")
    if config.get("validation_fn") is not None:
        raise ValueError("custom PC MLM validation_fn is not implemented yet")
    train_metadata = _loader_metadata(train_loader)
    max_tokens_per_micro_batch = int(train_metadata.get("max_tokens_per_batch", 0) or 0)
    tokens_per_update = config.get("tokens_per_update")
    requested_accumulation_steps = config.get("gradient_accumulation_steps")
    if requested_accumulation_steps is None and tokens_per_update is not None:
        requested_tokens_per_update = int(tokens_per_update)
        if requested_tokens_per_update < 1:
            raise ValueError("tokens_per_update must be positive")
        if max_tokens_per_micro_batch < 1:
            raise ValueError(
                "tokens_per_update requires loader metadata with max_tokens_per_batch "
                "or an explicit gradient_accumulation_steps value"
            )
        gradient_accumulation_steps = max(
            1,
            (requested_tokens_per_update + max_tokens_per_micro_batch - 1)
            // max_tokens_per_micro_batch,
        )
    else:
        gradient_accumulation_steps = int(config.get("gradient_accumulation_steps", 1))
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    pc_incremental = bool(config.get("pc_incremental", False))
    forward_updates = config.get("pc_forward_updates", "none")
    _validate_forward_updates(forward_updates)
    _validate_feedforward_start(structure, forward_updates)
    if pc_incremental and forward_updates != "none":
        raise ValueError(
            "incremental PC MLM training does not support pc_forward_updates; "
            f"got {forward_updates!r}"
        )
    if pc_incremental and gradient_accumulation_steps > 1:
        raise ValueError(
            "incremental PC MLM training does not support gradient accumulation; "
            f"got gradient_accumulation_steps={gradient_accumulation_steps}"
        )

    max_steps = int(config.get("max_steps", len(train_loader)))
    checkpoint_path = config.get("checkpoint_path")
    checkpoint_requested = checkpoint_path or config.get("resume_from")
    if checkpoint_requested and "optimizer_identity" not in config:
        raise ValueError(
            "PC MLM checkpoint/resume requires config['optimizer_identity']"
        )
    checkpoint_interval = int(config.get("checkpoint_interval", max_steps))
    checkpoint_config = _serializable_training_config(
        {
            **config,
            "mode": "pc",
            "objective": "pc_mlm_masked_ce_v1",
            "config_training_identity": _config_training_identity(config),
            "pc_training_identity": _pc_training_identity(
                structure, train_loader, pc_incremental
            ),
        }
    )
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    if checkpoint_interval < 1:
        raise ValueError("checkpoint_interval must be positive")
    log_interval = int(config.get("log_interval", 1))
    validation_interval = int(config.get("validation_interval", 0))
    opt_state = optimizer.init(params)
    if pc_incremental:
        incremental_step_fn = create_jit_train_step_pcn_mlm_incremental(
            structure, optimizer
        )
    else:
        grad_fn = create_jit_gradient_step_pcn_mlm(structure, forward_updates)
        update_fn = create_jit_update_step_pcn_mlm(optimizer)
    losses: list[float] = []
    validation: list[dict[str, float]] = []
    if config.get("resume_from"):
        params, opt_state, step, saved_config, saved_rng_key, loader_state = (
            load_mlm_training_checkpoint(
                config["resume_from"], opt_state_template=opt_state
            )
        )
        if saved_config.get("mode") != "pc":
            raise ValueError("Cannot resume PC MLM training from a non-PC checkpoint")
        if saved_config.get("objective") != checkpoint_config["objective"]:
            raise ValueError("Cannot resume PC MLM training with a different objective")
        if (
            saved_config.get("config_training_identity")
            != checkpoint_config["config_training_identity"]
        ):
            saved_identity = saved_config.get("config_training_identity") or {}
            current_identity = checkpoint_config["config_training_identity"] or {}
            mismatched = sorted(
                key
                for key in set(saved_identity) | set(current_identity)
                if saved_identity.get(key) != current_identity.get(key)
            )
            raise ValueError(
                "Cannot resume PC MLM training with a different training config "
                f"(mismatched keys: {mismatched})"
            )
        if (
            saved_config.get("pc_training_identity")
            != checkpoint_config["pc_training_identity"]
        ):
            raise ValueError(
                "Cannot resume PC MLM training with a different training identity"
            )
        if saved_rng_key is not None:
            rng_key = saved_rng_key
        if hasattr(train_loader, "load_state_dict"):
            train_loader.load_state_dict(loader_state)
        start_batch_in_epoch = int(loader_state.get("batch_in_epoch", 0))
    else:
        step = 0
        start_batch_in_epoch = 0
        loader_state = {"epoch": 0, "batch_in_epoch": 0}

    if tokens_per_update is None and max_tokens_per_micro_batch:
        tokens_per_update = max_tokens_per_micro_batch * gradient_accumulation_steps
    micro_steps = int(loader_state.get("micro_steps", 0))
    pending_grads = None
    pending_micro_batches = 0
    pending_loss_numerator = jnp.array(0.0)
    pending_masked_count = jnp.array(0.0)
    pending_energy_rises = jnp.array(0, dtype=jnp.int32)

    while step < max_steps:
        progressed = False
        for batch_idx, batch_data in enumerate(train_loader):
            if start_batch_in_epoch and batch_idx < start_batch_in_epoch:
                continue
            start_batch_in_epoch = 0
            if step >= max_steps:
                break
            if not _raw_batch_has_masked_tokens(batch_data):
                continue
            batch = _convert_batch(batch_data)
            _validate_pcn_mlm_batch(batch, structure)
            rng_key, step_key = jax.random.split(rng_key)
            if pc_incremental:
                params, opt_state, step_metrics = incremental_step_fn(
                    params, opt_state, batch, step_key
                )
                micro_steps += 1
                progressed = True
                pending_loss_numerator = (
                    step_metrics["masked_loss"] * step_metrics["num_masked_tokens"]
                )
                pending_masked_count = step_metrics["num_masked_tokens"]
                pending_energy_rises = step_metrics["pc_energy_rises"]
                incremental_grad_norm = step_metrics["grad_norm_mean"]
            else:
                grads, loss_numerator, masked_count, energy_rises = grad_fn(
                    params, batch, step_key
                )
                scaled_grads = jax.tree_util.tree_map(
                    lambda grad: grad / gradient_accumulation_steps,
                    grads,
                )
                pending_grads = (
                    scaled_grads
                    if pending_grads is None
                    else jax.tree_util.tree_map(
                        lambda left, right: left + right,
                        pending_grads,
                        scaled_grads,
                    )
                )
                pending_loss_numerator = pending_loss_numerator + loss_numerator
                pending_masked_count = pending_masked_count + masked_count
                pending_energy_rises = pending_energy_rises + energy_rises
                pending_micro_batches += 1
                micro_steps += 1
                progressed = True

                if pending_micro_batches < gradient_accumulation_steps:
                    continue

                params, opt_state = update_fn(
                    params,
                    opt_state,
                    pending_grads,
                    pending_masked_count,
                )
            step += 1
            try:
                num_train_batches = len(train_loader)
            except TypeError:
                num_train_batches = 0
            next_batch = (batch_idx + 1) % (num_train_batches or 1)
            loader_state = {
                "epoch": (
                    getattr(train_loader, "_epoch", 0)
                    if next_batch == 0
                    else max(0, getattr(train_loader, "_epoch", 1) - 1)
                ),
                "batch_in_epoch": next_batch,
                "micro_steps": micro_steps,
            }
            loss = float(
                pending_loss_numerator / jnp.maximum(pending_masked_count, 1.0)
            )
            losses.append(loss)
            should_log = verbose and log_interval > 0 and step % log_interval == 0
            if should_log:
                grad_norm = float(
                    incremental_grad_norm
                    if pc_incremental
                    else optax.tree.norm(pending_grads)
                )
                logged_rises = int(pending_energy_rises)
            pending_grads = None
            pending_micro_batches = 0
            pending_loss_numerator = jnp.array(0.0)
            pending_masked_count = jnp.array(0.0)
            pending_energy_rises = jnp.array(0, dtype=jnp.int32)
            if should_log:
                print(
                    f"step={step} masked_loss={loss:.4f} grad_norm={grad_norm:.4e} "
                    f"param_norm={float(optax.tree.norm(params)):.4e} "
                    f"energy_rises={logged_rises}"
                )
            if checkpoint_path and step % checkpoint_interval == 0:
                save_mlm_training_checkpoint(
                    checkpoint_path,
                    params=params,
                    opt_state=opt_state,
                    step=step,
                    config=checkpoint_config,
                    rng_key=rng_key,
                    loader_state=loader_state,
                )
            if (
                val_loader is not None
                and validation_interval > 0
                and step % validation_interval == 0
            ):
                val_metrics = evaluate_mlm(params, structure, val_loader, rng_key)
                validation.append(cast(dict[str, float], val_metrics))
                if verbose:
                    print(
                        f"step={step} "
                        f"validation_masked_loss={val_metrics['masked_loss']:.4f} "
                        "validation_masked_accuracy="
                        f"{val_metrics['masked_accuracy']:.4f} "
                        "validation_masked_tokens="
                        f"{val_metrics['num_masked_tokens']:.0f}"
                    )
        if not progressed:
            raise ValueError("train_loader did not yield any supervised MLM batches")

    if checkpoint_path:
        save_mlm_training_checkpoint(
            checkpoint_path,
            params=params,
            opt_state=opt_state,
            step=step,
            config=checkpoint_config,
            rng_key=rng_key,
            loader_state=loader_state,
        )

    return params, {
        "step": step,
        "losses": losses,
        "validation": validation,
        "mode": "pc",
        "micro_steps": micro_steps,
        "tokens_per_update": tokens_per_update,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "max_tokens_per_micro_batch": max_tokens_per_micro_batch or None,
        "pc_incremental": pc_incremental,
    }


def train_mlm(
    params: GraphParams,
    structure: GraphStructure,
    train_loader: Any,
    optimizer: optax.GradientTransformation,
    config: dict,
    rng_key: jax.Array,
    *,
    mode: str,
    val_loader: Any = None,
    verbose: bool = True,
) -> tuple[GraphParams, dict[str, Any]]:
    """Dispatch MLM training to backprop or predictive coding by mode label."""
    if mode == "backprop":
        trained_params, state = train_backprop_mlm(
            params,
            structure,
            train_loader,
            optimizer,
            config,
            rng_key,
            val_loader=val_loader,
            verbose=verbose,
        )
        return trained_params, {**state, "mode": "backprop"}
    if mode == "pc":
        return train_pcn_mlm(
            params,
            structure,
            train_loader,
            optimizer,
            config,
            rng_key,
            val_loader=val_loader,
            verbose=verbose,
        )
    raise ValueError(f"Unknown MLM training mode: {mode!r}")
