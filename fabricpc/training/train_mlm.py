"""Mode-dispatched masked-language-model training helpers."""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Tuple, cast

import jax
import jax.numpy as jnp
import optax

from fabricpc.core.inference import run_inference
from fabricpc.core.learning import compute_local_weight_gradients
from fabricpc.core.types import GraphParams, GraphState, GraphStructure
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.training.train import _convert_batch
from fabricpc.training.train_autoregressive import compute_loss, create_dropout_clamps
from fabricpc.training.train_backprop import (
    _apply_optimizer_update_when_masked,
    _loader_metadata,
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
    }
    serializable = _serializable_training_config(config)
    return _json_safe(
        {
            key: value
            for key, value in serializable.items()
            if key not in mutable_resume_keys
        }
    )


def _pc_training_identity(
    structure: GraphStructure,
    train_loader: Any,
) -> dict[str, Any]:
    inference = structure.config["inference"]
    return {
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


def get_graph_param_gradient_pcn_mlm(
    params: GraphParams,
    batch: Dict[str, jnp.ndarray],
    structure: GraphStructure,
    rng_key: jax.Array,
) -> Tuple[GraphParams, dict[str, jax.Array], GraphState, GraphState, GraphState]:
    """Compute one BERT MLM PC local-gradient step without applying updates."""
    _validate_pcn_mlm_batch(batch, structure)
    dropout_key, free_state_key, pc_state_key = jax.random.split(rng_key, 3)
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
    masked_loss = compute_loss(
        free_state,
        batch["y"],
        output_node,
        loss_type="cross_entropy",
        loss_mask=batch["loss_mask"],
    )

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
    final_state = run_inference(params, initial_state, clamps, structure)
    grads = compute_local_weight_gradients(params, final_state, structure)
    output_energy = jnp.sum(final_state.nodes[output_node].energy) / batch_size
    total_energy = jnp.array(0.0)
    for node_name, node in structure.nodes.items():
        if node.node_info.in_degree > 0:
            total_energy = total_energy + jnp.sum(final_state.nodes[node_name].energy)
    total_energy = total_energy / batch_size
    metrics = {
        "masked_loss": masked_loss,
        "masked_accuracy": _masked_accuracy(free_state, batch, output_node),
        "num_masked_tokens": jnp.sum(batch["loss_mask"].astype(jnp.float32)),
        "pc_output_energy": output_energy,
        "pc_total_energy": total_energy,
    }
    return grads, metrics, initial_state, free_state, final_state


def train_step_pcn_mlm(
    params: GraphParams,
    opt_state: optax.OptState,
    batch: Dict[str, jnp.ndarray],
    structure: GraphStructure,
    optimizer: optax.GradientTransformation,
    rng_key: jax.Array,
) -> Tuple[
    GraphParams,
    optax.OptState,
    dict[str, jax.Array],
    GraphState,
    GraphState,
    GraphState,
]:
    """Single-device predictive-coding MLM training step."""
    grads, metrics, initial_state, free_state, final_state = (
        get_graph_param_gradient_pcn_mlm(
            params,
            batch,
            structure,
            rng_key,
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
    """Train a BERT-style MLM graph with single-device PC updates."""
    if config.get("use_pmap"):
        raise ValueError("pmap PC MLM training is not implemented yet")
    if config.get("pmap_single_device"):
        raise ValueError("pmap PC MLM training is not implemented yet")
    if config.get("tokens_per_update") is not None:
        raise ValueError("tokens_per_update is not implemented for PC MLM training yet")
    if int(config.get("gradient_accumulation_steps", 1)) != 1:
        raise ValueError(
            "gradient accumulation is not implemented for PC MLM training yet"
        )
    if config.get("metrics_jsonl"):
        raise ValueError("PC MLM metrics_jsonl logging is not implemented yet")
    if config.get("validation_fn") is not None:
        raise ValueError("custom PC MLM validation_fn is not implemented yet")
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
            "pc_training_identity": _pc_training_identity(structure, train_loader),
        }
    )
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    if checkpoint_interval < 1:
        raise ValueError("checkpoint_interval must be positive")
    log_interval = int(config.get("log_interval", 1))
    validation_interval = int(config.get("validation_interval", 0))
    opt_state = optimizer.init(params)
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
            raise ValueError(
                "Cannot resume PC MLM training with a different training config"
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

    while step < max_steps:
        progressed = False
        for batch_idx, batch_data in enumerate(train_loader):
            if start_batch_in_epoch and batch_idx < start_batch_in_epoch:
                continue
            start_batch_in_epoch = 0
            if step >= max_steps:
                break
            batch = _convert_batch(batch_data)
            _validate_pcn_mlm_batch(batch, structure)
            if float(jnp.sum(batch["loss_mask"].astype(jnp.float32))) <= 0:
                continue
            rng_key, step_key = jax.random.split(rng_key)
            params, opt_state, metrics, _initial, _free, _final = train_step_pcn_mlm(
                params,
                opt_state,
                batch,
                structure,
                optimizer,
                step_key,
            )
            step += 1
            progressed = True
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
            }
            loss = float(metrics["masked_loss"])
            losses.append(loss)
            if verbose and log_interval > 0 and step % log_interval == 0:
                print(f"step={step} masked_loss={loss:.4f}")
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
