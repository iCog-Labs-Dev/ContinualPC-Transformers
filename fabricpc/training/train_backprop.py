"""
Backpropagation training for FabricPC networks.

This module provides standard end-to-end backpropagation training as an
alternative to iterative predictive coding inference. Key differences:

1. No iterative latent inference - single forward pass
2. Latent states set to projections: z_latent = z_mu
3. End-to-end autodiff via jax.grad
4. Only input nodes are clamped (output computes freely)

Use cases:
- Baseline comparison with predictive coding
- Faster training when PC dynamics not needed
- Standard deep learning workflows
"""

from pathlib import Path
from typing import Dict, Tuple, Any, List, Optional, Callable, cast
from collections import deque
import json
import math
import jax
import jax.numpy as jnp
import numpy as np
import optax

from fabricpc.core.types import GraphParams, GraphStructure, GraphState, NodeParams
from fabricpc.graph_initialization.state_initializer import (
    initialize_graph_state,
    FeedforwardStateInit,
)
from fabricpc.training.train_autoregressive import (
    _validate_pmap_batch,
    create_causal_mask,
    create_dropout_clamps,
    compute_loss,
)
from fabricpc.training.train import (
    _convert_batch,
    replicate_opt_state,
    replicate_params,
    shard_batch,
)


def validate_feedforward_init(structure: GraphStructure):
    """
    Validate that the graph state initializer is feedforward type.

    Raises:
        ValueError if incompatible.
    """
    init = structure.config["graph_state_initializer"]
    if not isinstance(init, FeedforwardStateInit):
        raise ValueError(
            f"GraphState initializer must be FeedforwardStateInit for backprop training, "
            f"got {type(init).__name__}"
        )


def compute_forward_pass(
    params: GraphParams,
    structure: GraphStructure,
    batch: Dict[str, jnp.ndarray],
    rng_key: jax.Array,
    training: bool = False,
    extra_clamps: Optional[Dict[str, jnp.ndarray]] = None,
) -> GraphState:
    """
    Compute forward pass for backpropagation training.

    This runs a single forward pass through the network (no iterative
    inference) and computes output at the output node.

    Args:
        params: Model parameters
        structure: Graph structure
        batch: Batch data with keys matching task_map (e.g., 'x', 'y')
        rng_key: JAX random key for state initialization
        extra_clamps: Node values that override the computed ones

    Returns:
        graph state after feedforward pass
    """

    validate_feedforward_init(structure)

    batch_size = batch["x"].shape[0]

    # Clamp inputs and auxiliary non-target nodes (not output!) - key difference from PC
    clamps = {}
    for task_name, task_value in batch.items():
        if task_name in structure.task_map and task_name != "y":
            clamps[structure.task_map[task_name]] = task_value
    dropout_key, state_key = jax.random.split(rng_key)
    clamps.update(
        create_dropout_clamps(structure, batch_size, dropout_key, training=training)
    )
    clamps.update(extra_clamps or {})

    # Single forward pass via initialize_state with feedforward init
    state = initialize_graph_state(
        structure,
        batch_size,
        state_key,
        clamps=clamps,
        params=params,
    )
    return state


def train_step_backprop(
    params: GraphParams,
    opt_state: optax.OptState,
    batch: Dict[str, jnp.ndarray],
    structure: GraphStructure,
    optimizer: optax.GradientTransformation,
    rng_key: jax.Array,
    loss_type: str = "cross_entropy",
) -> Tuple[GraphParams, optax.OptState, jax.Array]:
    """
    Single backprop training step.

    Args:
        params: Current model parameters
        opt_state: Optimizer state
        batch: Batch data
        structure: Graph structure
        optimizer: Optax optimizer
        rng_key: JAX random key
        loss_type: Loss function type

    Returns:
        Tuple of (updated_params, updated_opt_state, loss_value)
    """

    def loss_fn(p):
        state = compute_forward_pass(p, structure, batch, rng_key, training=True)
        return compute_loss(state, batch["y"], structure.task_map["y"], loss_type)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = cast(GraphParams, optax.apply_updates(params, updates))

    return params, opt_state, loss


def _apply_optimizer_update_when_masked(
    *,
    params: GraphParams,
    opt_state: optax.OptState,
    grads: GraphParams,
    optimizer: optax.GradientTransformation,
    masked_count: jax.Array,
) -> tuple[GraphParams, optax.OptState]:
    def update_fn(args):
        current_params, current_opt_state, current_grads = args
        updates, new_opt_state = optimizer.update(
            current_grads,
            current_opt_state,
            current_params,
        )
        new_params = cast(GraphParams, optax.apply_updates(current_params, updates))
        return new_params, new_opt_state

    def noop_fn(args):
        current_params, current_opt_state, _ = args
        return current_params, current_opt_state

    return jax.lax.cond(
        masked_count > 0,
        update_fn,
        noop_fn,
        (params, opt_state, grads),
    )


def train_step_backprop_mlm(
    params: GraphParams,
    opt_state: optax.OptState,
    batch: Dict[str, jnp.ndarray],
    structure: GraphStructure,
    optimizer: optax.GradientTransformation,
    rng_key: jax.Array,
) -> Tuple[GraphParams, optax.OptState, jax.Array]:
    """Single backprop step for masked-language-model pretraining."""
    if "loss_mask" not in batch:
        raise ValueError("MLM training requires batch['loss_mask']")

    def loss_fn(p):
        state = compute_forward_pass(p, structure, batch, rng_key, training=True)
        return compute_loss(
            state,
            batch["y"],
            structure.task_map["y"],
            loss_type="cross_entropy",
            loss_mask=batch["loss_mask"],
        )

    loss, grads = jax.value_and_grad(loss_fn)(params)
    params, opt_state = _apply_optimizer_update_when_masked(
        params=params,
        opt_state=opt_state,
        grads=grads,
        optimizer=optimizer,
        masked_count=jnp.sum(batch["loss_mask"].astype(jnp.float32)),
    )

    return params, opt_state, loss


def _train_step_backprop_mlm_no_validate(
    params: GraphParams,
    opt_state: optax.OptState,
    batch: Dict[str, jnp.ndarray],
    structure: GraphStructure,
    optimizer: optax.GradientTransformation,
    rng_key: jax.Array,
) -> Tuple[GraphParams, optax.OptState, jax.Array]:
    def loss_fn(p):
        state = compute_forward_pass(p, structure, batch, rng_key, training=True)
        return compute_loss(
            state,
            batch["y"],
            structure.task_map["y"],
            loss_type="cross_entropy",
            loss_mask=batch["loss_mask"],
        )

    loss, grads = jax.value_and_grad(loss_fn)(params)
    params, opt_state = _apply_optimizer_update_when_masked(
        params=params,
        opt_state=opt_state,
        grads=grads,
        optimizer=optimizer,
        masked_count=jnp.sum(batch["loss_mask"].astype(jnp.float32)),
    )
    return params, opt_state, loss


def _loss_and_grad_backprop_mlm_no_update(
    params: GraphParams,
    batch: Dict[str, jnp.ndarray],
    structure: GraphStructure,
    rng_key: jax.Array,
) -> tuple[jax.Array, GraphParams, jax.Array]:
    def loss_fn(p):
        state = compute_forward_pass(p, structure, batch, rng_key, training=True)
        return compute_loss(
            state,
            batch["y"],
            structure.task_map["y"],
            loss_type="cross_entropy",
            loss_mask=batch["loss_mask"],
        )

    loss, grads = jax.value_and_grad(loss_fn)(params)
    masked_count = jnp.sum(batch["loss_mask"].astype(jnp.float32))
    return loss, grads, masked_count


def _apply_optimizer_update(
    params: GraphParams,
    opt_state: optax.OptState,
    grads: GraphParams,
    optimizer: optax.GradientTransformation,
) -> tuple[GraphParams, optax.OptState]:
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = cast(GraphParams, optax.apply_updates(params, updates))
    return params, opt_state


def _validate_resume_config(
    saved_config: dict[str, Any], current_config: dict[str, Any]
) -> None:
    for key in ("gradient_accumulation_steps", "tokens_per_update"):
        if saved_config.get(key) != current_config.get(key):
            raise ValueError(
                f"Checkpoint {key}={saved_config.get(key)!r} does not match "
                f"current {key}={current_config.get(key)!r}; resume with matching accumulation config"
            )


def train_step_backprop_mlm_pmap(
    params: GraphParams,
    opt_state: optax.OptState,
    batch: Dict[str, jnp.ndarray],
    structure: GraphStructure,
    optimizer: optax.GradientTransformation,
    rng_key: jax.Array,
) -> Tuple[GraphParams, optax.OptState, jax.Array]:
    """Data-parallel MLM backprop step for use under ``jax.pmap``."""
    mask = batch["loss_mask"].astype(jnp.float32)

    def loss_numerator(p):
        state = compute_forward_pass(p, structure, batch, rng_key, training=True)
        predictions = state.nodes[structure.task_map["y"]].z_mu
        targets = jax.nn.one_hot(batch["y"], predictions.shape[-1])
        per_token_loss = -jnp.sum(targets * jnp.log(predictions + 1e-10), axis=-1)
        return jnp.sum(per_token_loss * mask)

    numerator, grads = jax.value_and_grad(loss_numerator)(params)
    local_count = jnp.sum(mask)
    global_count = jax.lax.psum(local_count, axis_name="devices")
    global_numerator = jax.lax.psum(numerator, axis_name="devices")
    grads = jax.lax.psum(grads, axis_name="devices")
    grads = jax.tree_util.tree_map(lambda g: g / jnp.maximum(global_count, 1.0), grads)
    loss = global_numerator / jnp.maximum(global_count, 1.0)
    params, opt_state = _apply_optimizer_update_when_masked(
        params=params,
        opt_state=opt_state,
        grads=grads,
        optimizer=optimizer,
        masked_count=global_count,
    )
    return params, opt_state, loss


def create_pmap_train_step_backprop_mlm(
    structure: GraphStructure,
    optimizer: optax.GradientTransformation,
):
    """Create a pmap'd MLM backprop step with static graph/optimizer closure."""

    def step_fn(params, opt_state, batch, rng_key):
        return train_step_backprop_mlm_pmap(
            params, opt_state, batch, structure, optimizer, rng_key
        )

    return jax.pmap(step_fn, axis_name="devices")


def _validate_mlm_batch(batch: Dict[str, jnp.ndarray]) -> None:
    if "loss_mask" not in batch:
        raise ValueError("MLM training requires batch['loss_mask']")


def _validate_mlm_batch_data(batch_data: Any) -> None:
    if not isinstance(batch_data, dict):
        return
    if "loss_mask" not in batch_data:
        raise ValueError("MLM training requires batch['loss_mask']")


def _flatten_graph_params(
    params: GraphParams,
    key_prefix: str = "params",
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    tensors: dict[str, np.ndarray] = {}
    schema: dict[str, Any] = {}
    for node_name, node_params in params.nodes.items():
        node_schema = {"weights": {}, "biases": {}}
        for collection_name in ("weights", "biases"):
            for param_name, value in getattr(node_params, collection_name).items():
                tensor_key = f"{key_prefix}/{node_name}/{collection_name}/{param_name}"
                tensors[tensor_key] = np.asarray(value)
                node_schema[collection_name][param_name] = tensor_key
        schema[node_name] = node_schema
    return tensors, schema


def _graph_params_from_tensors(metadata: dict[str, Any], tensors: Any) -> GraphParams:
    nodes = {}
    for node_name, node_schema in metadata["params"].items():
        collections = {}
        for collection_name in ("weights", "biases"):
            values = {}
            for param_name, tensor_key in node_schema[collection_name].items():
                if tensor_key not in tensors:
                    raise ValueError(f"Missing tensor {tensor_key!r}")
                values[param_name] = jnp.array(tensors[tensor_key])
            collections[collection_name] = values
        nodes[node_name] = NodeParams(
            weights=collections["weights"], biases=collections["biases"]
        )
    return GraphParams(nodes=nodes)


def save_mlm_training_checkpoint(
    path: str,
    *,
    params: GraphParams,
    opt_state: optax.OptState,
    step: int,
    config: dict[str, Any],
    rng_key: Optional[jax.Array] = None,
    loader_state: Optional[dict[str, Any]] = None,
) -> None:
    """Save params, optimizer leaves, step, and config as JSON + NPZ."""
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tensor_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".npz")
    tensors, param_schema = _flatten_graph_params(params)

    opt_leaves = jax.tree_util.tree_leaves(opt_state)
    opt_schema = []
    for idx, leaf in enumerate(opt_leaves):
        key = f"opt_state/{idx}"
        tensors[key] = np.asarray(leaf)
        opt_schema.append(key)

    np.savez(tensor_path, **tensors)
    checkpoint_path.write_text(
        json.dumps(
            {
                "format": "fabricpc-mlm-training-checkpoint",
                "version": 1,
                "tensor_file": tensor_path.name,
                "step": step,
                "config": config,
                "rng_key": None if rng_key is None else np.asarray(rng_key).tolist(),
                "loader_state": loader_state or {},
                "params": param_schema,
                "opt_state": opt_schema,
            },
            indent=2,
            sort_keys=True,
        )
    )


def load_mlm_training_checkpoint(
    path: str,
    *,
    opt_state_template: optax.OptState,
) -> tuple[
    GraphParams,
    optax.OptState,
    int,
    dict[str, Any],
    Optional[jax.Array],
    dict[str, Any],
]:
    """Load a checkpoint saved by :func:`save_mlm_training_checkpoint`."""
    checkpoint_path = Path(path)
    metadata = json.loads(checkpoint_path.read_text())
    if metadata.get("format") != "fabricpc-mlm-training-checkpoint":
        raise ValueError("Unsupported MLM checkpoint format")
    if metadata.get("version") != 1:
        raise ValueError(
            f"Unsupported MLM checkpoint version: {metadata.get('version')!r}"
        )
    tensor_file = metadata.get("tensor_file")
    if not isinstance(tensor_file, str):
        raise ValueError("Checkpoint tensor_file must be a string")

    with np.load(checkpoint_path.with_name(tensor_file), allow_pickle=False) as tensors:
        params = _graph_params_from_tensors(metadata, tensors)
        template_leaves, treedef = jax.tree_util.tree_flatten(opt_state_template)
        opt_keys = metadata.get("opt_state", [])
        if len(opt_keys) != len(template_leaves):
            raise ValueError(
                "Checkpoint optimizer state does not match optimizer template"
            )
        opt_leaves = [jnp.array(tensors[key]) for key in opt_keys]
        opt_state = jax.tree_util.tree_unflatten(treedef, opt_leaves)

    saved_rng = metadata.get("rng_key")
    rng_key = None if saved_rng is None else jnp.array(saved_rng, dtype=jnp.uint32)
    return (
        params,
        opt_state,
        int(metadata["step"]),
        metadata.get("config", {}),
        rng_key,
        metadata.get("loader_state", {}),
    )


def evaluate_mlm(
    params: GraphParams,
    structure: GraphStructure,
    loader: Any,
    rng_key: jax.Array,
) -> Dict[str, float]:
    """Evaluate masked-token loss and accuracy over an MLM loader."""
    total_loss = 0.0
    total_correct = 0
    total_masked = 0

    loader_state = None
    if hasattr(loader, "state_dict") and hasattr(loader, "load_state_dict"):
        loader_state = loader.state_dict()

    try:
        for batch_data in loader:
            if not isinstance(batch_data, dict):
                raise ValueError("MLM loaders must yield dict batches")
            if "loss_mask" not in batch_data:
                raise ValueError("MLM evaluation requires batch['loss_mask']")
            batch = {k: jnp.array(v) for k, v in batch_data.items()}
            rng_key, step_key = jax.random.split(rng_key)
            state = compute_forward_pass(params, structure, batch, step_key)
            predictions = state.nodes[structure.task_map["y"]].z_mu
            pred_ids = jnp.argmax(predictions, axis=-1)
            if batch["loss_mask"].shape != pred_ids.shape:
                raise ValueError(
                    "MLM loss_mask shape must match token prediction shape exactly; "
                    f"got {batch['loss_mask'].shape}, expected {pred_ids.shape}"
                )

            mask = batch["loss_mask"].astype(bool)
            masked_count = int(jnp.sum(mask))
            if masked_count == 0:
                continue
            loss = compute_loss(
                state,
                batch["y"],
                structure.task_map["y"],
                loss_type="cross_entropy",
                loss_mask=batch["loss_mask"],
            )
            correct = int(jnp.sum((pred_ids == batch["y"]) & mask))

            total_loss += float(loss) * masked_count
            total_correct += correct
            total_masked += masked_count
    finally:
        if loader_state is not None:
            loader.load_state_dict(loader_state)

    if total_masked == 0:
        return {"masked_loss": 0.0, "masked_accuracy": 0.0, "num_masked_tokens": 0}
    return {
        "masked_loss": total_loss / total_masked,
        "masked_accuracy": total_correct / total_masked,
        "num_masked_tokens": total_masked,
    }


def _loader_metadata(loader: Any) -> dict[str, Any]:
    metadata = getattr(loader, "metadata", {})
    if callable(metadata):
        metadata = metadata()
    return dict(metadata) if isinstance(metadata, dict) else {}


def _count_exposure_tokens(
    batch_data: Any,
    loader_metadata: dict[str, Any],
) -> tuple[int, bool]:
    if isinstance(batch_data, dict) and "attention_mask" in batch_data:
        return int(np.asarray(batch_data["attention_mask"]).sum()), False
    if (
        isinstance(batch_data, dict)
        and "x" in batch_data
        and "pad_token_id" in loader_metadata
    ):
        pad_token_id = int(loader_metadata["pad_token_id"])
        return int(np.sum(np.asarray(batch_data["x"]) != pad_token_id)), False
    if isinstance(batch_data, dict) and "x" in batch_data:
        return int(np.asarray(batch_data["x"]).size), True
    return 0, True


def _append_metrics_jsonl(path: str | Path | None, event: dict[str, Any]) -> None:
    if not path:
        return
    metrics_path = Path(path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def _serializable_training_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if not callable(value)}


def train_backprop_mlm(
    params: GraphParams,
    structure: GraphStructure,
    train_loader: Any,
    optimizer: optax.GradientTransformation,
    config: dict,
    rng_key: jax.Array,
    val_loader: Any = None,
    verbose: bool = True,
) -> tuple[GraphParams, dict[str, Any]]:
    """Train a masked-language-model encoder with optional checkpoint/resume.

    Config keys:
        max_steps: total optimizer steps to run to.
        checkpoint_path: optional JSON path for periodic/final checkpoint.
        checkpoint_interval: save every N steps when checkpoint_path is set.
        resume_from: optional checkpoint path to resume from.
        validation_interval: evaluate val_loader every N steps when provided.
        use_pmap: use pmap when multiple devices are visible.
        pmap_single_device: force pmap shape path on one device for testing.
    """
    validate_feedforward_init(structure)

    try:
        num_train_batches = len(train_loader)
    except TypeError as exc:
        if "max_steps" not in config:
            raise TypeError(
                "train_loader must support len() when max_steps is omitted"
            ) from exc
        num_train_batches = None
    if num_train_batches == 0:
        raise ValueError("train_loader must yield at least one full MLM batch")

    max_steps = int(config.get("max_steps", num_train_batches))
    checkpoint_path = config.get("checkpoint_path")
    checkpoint_interval = int(config.get("checkpoint_interval", max_steps))
    validation_interval = int(config.get("validation_interval", 0))
    log_interval = int(config.get("log_interval", 1))
    rolling_loss_window = int(config.get("rolling_loss_window", 100))
    gradient_accumulation_steps = int(config.get("gradient_accumulation_steps", 1))
    metrics_jsonl = config.get("metrics_jsonl")
    validation_fn = config.get("validation_fn")
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    if checkpoint_interval < 1:
        raise ValueError("checkpoint_interval must be positive")
    if validation_interval < 0:
        raise ValueError("validation_interval must be non-negative")
    if log_interval < 0:
        raise ValueError("log_interval must be non-negative")
    if rolling_loss_window < 1:
        raise ValueError("rolling_loss_window must be positive")
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    use_pmap = bool(config.get("use_pmap", False))
    pmap_single_device = bool(config.get("pmap_single_device", False))
    n_devices = jax.device_count()
    run_pmap = use_pmap and ((n_devices > 1) or pmap_single_device)
    if run_pmap and gradient_accumulation_steps > 1:
        raise ValueError(
            "gradient accumulation is not supported with pmap MLM training"
        )

    opt_state = optimizer.init(params)
    start_step = 0
    start_batch_in_epoch = 0
    if config.get("resume_from"):
        loaded = load_mlm_training_checkpoint(
            config["resume_from"], opt_state_template=opt_state
        )
        params, opt_state, start_step, saved_config, saved_rng_key, loader_state = (
            loaded
        )
        _validate_resume_config(saved_config, config)
        if saved_rng_key is not None:
            rng_key = saved_rng_key
        if hasattr(train_loader, "load_state_dict"):
            train_loader.load_state_dict(loader_state)
        start_batch_in_epoch = int(loader_state.get("batch_in_epoch", 0))

    if run_pmap:
        from fabricpc.training.train import (
            replicate_params,
            replicate_opt_state,
            shard_batch,
        )

        params = replicate_params(params, n_devices)
        opt_state = replicate_opt_state(opt_state, n_devices)
        step_fn = create_pmap_train_step_backprop_mlm(structure, optimizer)
    else:
        grad_fn = jax.jit(
            lambda p, b, k: _loss_and_grad_backprop_mlm_no_update(p, b, structure, k)
        )
        update_fn = jax.jit(lambda p, o, g: _apply_optimizer_update(p, o, g, optimizer))

    losses: list[float] = []
    rolling_losses: deque[float] = deque(maxlen=rolling_loss_window)
    rolling_loss_sum = 0.0
    validation: list[dict[str, float]] = []
    update_step = start_step
    train_metadata = _loader_metadata(train_loader)
    source_token_count = int(train_metadata.get("source_token_count", 0) or 0)
    max_tokens_per_micro_batch = int(train_metadata.get("max_tokens_per_batch", 0) or 0)
    tokens_per_update = config.get("tokens_per_update")
    if tokens_per_update is None and max_tokens_per_micro_batch:
        tokens_per_update = max_tokens_per_micro_batch * gradient_accumulation_steps
    last_loader_state = (
        loader_state if config.get("resume_from") else {"epoch": 0, "batch_in_epoch": 0}
    )
    micro_steps = int(last_loader_state.get("micro_steps", 0))
    real_tokens_seen = int(last_loader_state.get("real_tokens_seen", 0))
    masked_positions_seen = int(last_loader_state.get("masked_positions_seen", 0))
    max_token_slots_seen = int(last_loader_state.get("max_token_slots_seen", 0))
    real_tokens_seen_estimated = bool(
        last_loader_state.get("real_tokens_seen_estimated", False)
    )
    pending_real_tokens = 0
    pending_masked_positions = 0
    pending_max_token_slots = 0
    pending_real_tokens_estimated = False
    pending_grads = None
    pending_losses: list[float] = []
    pending_micro_batches = 0
    while update_step < max_steps:
        progressed = False
        for batch_idx, batch_data in enumerate(train_loader):
            if start_batch_in_epoch and batch_idx < start_batch_in_epoch:
                continue
            start_batch_in_epoch = 0
            if update_step >= max_steps:
                break
            _validate_mlm_batch_data(batch_data)
            batch_real_tokens, estimated = _count_exposure_tokens(
                batch_data, train_metadata
            )
            batch_masked_positions = (
                int(np.asarray(batch_data["loss_mask"]).sum())
                if isinstance(batch_data, dict)
                else 0
            )
            batch_max_token_slots = (
                int(np.asarray(batch_data["x"]).size)
                if isinstance(batch_data, dict)
                else 0
            )
            batch = _convert_batch(batch_data)
            _validate_mlm_batch(batch)
            rng_key, step_key = jax.random.split(rng_key)
            progressed = True

            if run_pmap:
                batch = shard_batch(batch, n_devices)
                device_keys = jax.random.split(step_key, n_devices)
                params, opt_state, loss = step_fn(params, opt_state, batch, device_keys)
                loss_value = float(jnp.mean(loss))
                masked_count = batch_masked_positions
            else:
                loss, grads, masked_count_array = grad_fn(params, batch, step_key)
                loss_value = float(loss)
                masked_count = float(masked_count_array)

            if masked_count <= 0:
                continue

            micro_steps += 1
            pending_real_tokens += batch_real_tokens
            pending_masked_positions += batch_masked_positions
            pending_max_token_slots += batch_max_token_slots
            pending_real_tokens_estimated = pending_real_tokens_estimated or estimated
            if not run_pmap:
                scaled_grads = jax.tree_util.tree_map(
                    lambda grad: grad / gradient_accumulation_steps, grads
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
            pending_losses.append(loss_value)
            pending_micro_batches += 1

            if pending_micro_batches < gradient_accumulation_steps:
                continue

            if not run_pmap:
                params, opt_state = update_fn(params, opt_state, pending_grads)

            real_tokens_seen += pending_real_tokens
            masked_positions_seen += pending_masked_positions
            max_token_slots_seen += pending_max_token_slots
            real_tokens_seen_estimated = (
                real_tokens_seen_estimated or pending_real_tokens_estimated
            )
            loss_value = float(np.mean(pending_losses))

            update_step += 1
            losses.append(loss_value)
            if len(rolling_losses) == rolling_losses.maxlen:
                rolling_loss_sum -= rolling_losses[0]
            rolling_losses.append(loss_value)
            rolling_loss_sum += loss_value
            next_batch = (batch_idx + 1) % (num_train_batches or 1)
            last_loader_state = {
                "epoch": (
                    getattr(train_loader, "_epoch", 0)
                    if next_batch == 0
                    else max(0, getattr(train_loader, "_epoch", 1) - 1)
                ),
                "batch_in_epoch": next_batch,
                "micro_steps": micro_steps,
                "real_tokens_seen": real_tokens_seen,
                "real_tokens_seen_estimated": real_tokens_seen_estimated,
                "masked_positions_seen": masked_positions_seen,
                "max_token_slots_seen": max_token_slots_seen,
            }
            pending_real_tokens = 0
            pending_masked_positions = 0
            pending_max_token_slots = 0
            pending_real_tokens_estimated = False
            pending_grads = None
            pending_losses = []
            pending_micro_batches = 0

            if verbose and log_interval > 0 and update_step % log_interval == 0:
                rolling_average = rolling_loss_sum / len(rolling_losses)
                print(
                    f"step={update_step} masked_loss={loss_value:.4f} "
                    f"masked_loss_avg{len(rolling_losses)}={rolling_average:.4f}"
                )

            checkpoint_due = checkpoint_path and update_step % checkpoint_interval == 0
            if checkpoint_due:
                checkpoint_params = (
                    jax.tree_util.tree_map(lambda x: x[0], params)
                    if run_pmap
                    else params
                )
                checkpoint_opt_state = (
                    jax.tree_util.tree_map(lambda x: x[0], opt_state)
                    if run_pmap
                    else opt_state
                )
                save_mlm_training_checkpoint(
                    checkpoint_path,
                    params=checkpoint_params,
                    opt_state=checkpoint_opt_state,
                    step=update_step,
                    config=_serializable_training_config(config),
                    rng_key=rng_key,
                    loader_state=last_loader_state,
                )

            if (
                (val_loader is not None or validation_fn is not None)
                and validation_interval > 0
                and update_step % validation_interval == 0
            ):
                eval_params = (
                    jax.tree_util.tree_map(lambda x: x[0], params)
                    if run_pmap
                    else params
                )
                validation_metrics = (
                    validation_fn(eval_params, rng_key)
                    if validation_fn is not None
                    else evaluate_mlm(eval_params, structure, val_loader, rng_key)
                )
                validation_record = {"step": update_step, **validation_metrics}
                validation.append(validation_record)
                _append_metrics_jsonl(
                    metrics_jsonl,
                    {"event": "validation", **validation_record},
                )
                if verbose:
                    print(
                        f"step={update_step} validation_masked_loss="
                        f"{validation_metrics['masked_loss']:.4f} "
                        f"validation_masked_accuracy="
                        f"{validation_metrics['masked_accuracy']:.4f} "
                        f"validation_masked_tokens="
                        f"{validation_metrics['num_masked_tokens']:.0f}"
                    )
        if not progressed:
            raise ValueError("train_loader produced no MLM batches during an epoch")

    final_params = (
        jax.tree_util.tree_map(lambda x: x[0], params) if run_pmap else params
    )
    if checkpoint_path:
        final_opt_state = (
            jax.tree_util.tree_map(lambda x: x[0], opt_state) if run_pmap else opt_state
        )
        save_mlm_training_checkpoint(
            checkpoint_path,
            params=final_params,
            opt_state=final_opt_state,
            step=update_step,
            config=_serializable_training_config(config),
            rng_key=rng_key,
            loader_state=last_loader_state,
        )

    if verbose:
        final_loss = losses[-1] if losses else 0.0
        print(
            f"done step={update_step} final_masked_loss={final_loss:.4f} losses={len(losses)}"
        )

    epochs_equivalent = (
        real_tokens_seen / source_token_count if source_token_count > 0 else None
    )
    exposure_metadata = {
        "real_tokens_seen": real_tokens_seen,
        "real_tokens_seen_estimated": real_tokens_seen_estimated,
        "masked_positions_seen": masked_positions_seen,
        "max_token_slots_seen": max_token_slots_seen,
        "epochs_equivalent": epochs_equivalent,
        "tokens_per_update": tokens_per_update,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "micro_steps": micro_steps,
        "update_steps": update_step,
        "max_tokens_per_micro_batch": max_tokens_per_micro_batch or None,
        "loader_metadata": train_metadata,
    }
    _append_metrics_jsonl(
        metrics_jsonl,
        {"event": "final", "step": update_step, **exposure_metadata},
    )

    return final_params, {
        "step": update_step,
        "losses": losses,
        "validation": validation,
        **exposure_metadata,
    }


def train_backprop(
    params: GraphParams,
    structure: GraphStructure,
    train_loader: Any,
    optimizer: optax.GradientTransformation,
    config: dict,
    rng_key: jax.Array,
    verbose: bool = True,
    epoch_callback: Optional[Callable] = None,
    iter_callback: Optional[Callable] = None,
) -> Tuple[GraphParams, List[List[float]], List[Any]]:
    """
    Main backprop training loop.

    This provides standard end-to-end backpropagation training for FabricPC
    models, using the same graph structure but skipping iterative inference.

    Args:
        params: Initial parameters
        structure: Graph structure
        train_loader: Data loader yielding batches
        optimizer: Optax optimizer (e.g., optax.adam(1e-3))
        config: Training configuration:
            - num_epochs: Number of training epochs
            - loss_type: "cross_entropy" or "mse" (default: "cross_entropy")
        rng_key: JAX random key
        verbose: Print progress
        epoch_callback: Optional (epoch, params, structure, config, rng) -> any
        iter_callback: Optional (epoch, batch_idx, loss) -> any

    Returns:
        Tuple of (trained_params, loss_history, epoch_results)

    Example:
        >>> params = initialize_params(structure, rng_key)
        >>> optimizer = optax.adam(1e-3)
        >>> train_config = {
        ...     "num_epochs": 10,
        ...     "loss_type": "cross_entropy",
        ... }
        >>> trained_params, losses, _ = train_backprop(
        ...     params, structure, train_loader, optimizer, train_config, rng_key
        ... )
    """
    validate_feedforward_init(structure)

    opt_state = optimizer.init(params)

    # Training hyperparameters
    num_epochs = config.get("num_epochs", 10)  # supports float (e.g. 1.5)
    loss_type = config.get("loss_type", "cross_entropy")

    # Support fractional epochs: e.g. 1.5 -> 2 loop iterations, last stops at 50%
    total_epochs = math.ceil(num_epochs)
    frac = num_epochs - math.floor(num_epochs)

    # JIT compile training step
    jit_train_step = jax.jit(
        lambda p, o, b, k: train_step_backprop(
            p, o, b, structure, optimizer, k, loss_type
        )
    )

    iter_results = []
    epoch_results = []

    for epoch_idx in range(total_epochs):
        try:
            num_batches = len(train_loader)
        except TypeError:
            raise TypeError("train_loader must support len()")

        # On the final epoch, truncate if fractional
        is_last_epoch = epoch_idx == total_epochs - 1
        if is_last_epoch and frac > 0:
            max_batches = round(frac * num_batches)
        else:
            max_batches = num_batches

        # Split keys for actual batch count
        epoch_rng_key, rng_key = jax.random.split(rng_key)
        batch_keys = jax.random.split(epoch_rng_key, max_batches)

        batch_losses = []
        epoch_loss = 0.0
        batches_processed = 0

        for batch_idx, batch_data in enumerate(train_loader):
            if batch_idx >= max_batches:
                break

            # Convert batch to JAX format
            if isinstance(batch_data, dict):
                batch = {k: jnp.array(v) for k, v in batch_data.items()}
            elif isinstance(batch_data, (list, tuple)):
                batch = {
                    "x": jnp.array(batch_data[0]),
                    "y": jnp.array(batch_data[1]),
                }
            else:
                raise ValueError(f"Unsupported batch format: {type(batch_data)}")

            # Training step
            params, opt_state, loss = jit_train_step(
                params, opt_state, batch, batch_keys[batch_idx]
            )

            epoch_loss += float(loss)
            batches_processed += 1

            if iter_callback is not None:
                batch_losses.append(iter_callback(epoch_idx, batch_idx, loss))
            else:
                batch_losses.append(float(loss))

        iter_results.append(batch_losses)
        avg_loss = epoch_loss / batches_processed if batches_processed > 0 else 0.0

        # Epoch callback
        if epoch_callback is not None:
            epoch_results.append(
                epoch_callback(epoch_idx, params, structure, config, rng_key)
            )
        else:
            epoch_results.append(None)

        if verbose:
            print(f"Epoch {epoch_idx + 1}/{total_epochs}, Loss: {avg_loss:.4f}")

    return params, iter_results, epoch_results


def compute_loss_autoregressive(
    params: GraphParams,
    structure: GraphStructure,
    batch: Dict[str, jnp.ndarray],
    rng_key: jax.Array,
    use_causal_mask: bool = True,
    training: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute loss for autoregressive (next-token) prediction with backprop.

    This runs a single forward pass with causal masking and computes
    cross-entropy loss over all sequence positions.

    Args:
        params: Model parameters
        structure: Graph structure
        batch: Batch with "x" (input) and "y" (target) sequences
        rng_key: JAX random key
        use_causal_mask: Whether to apply causal masking
        training: If True, sample dropout masks; otherwise clamp ones.

    Returns:
        Tuple of (loss, predictions) where:
            - loss: Scalar loss (mean over batch)
            - predictions: Output logits of shape (batch, seq_len, vocab_size)
    """
    batch_size = batch["x"].shape[0]
    seq_len = batch["x"].shape[1]

    # Clamp input only (not output)
    clamps = {structure.task_map["x"]: batch["x"]}

    # Add an external causal mask only if the graph defines one (v1). The v2
    # decomposed MhaResidualNode masks internally via is_causal, so its task_map
    # has no "causal_mask" node and this is skipped.
    if use_causal_mask and "causal_mask" in structure.task_map:
        causal_mask = create_causal_mask(seq_len)
        causal_mask = causal_mask[None, None, :, :]  # (1, 1, seq, seq)
        causal_mask = jnp.broadcast_to(causal_mask, (batch_size, 1, seq_len, seq_len))
        clamps[structure.task_map["causal_mask"]] = causal_mask

    rng_key, dropout_key, state_key = jax.random.split(rng_key, 3)
    clamps.update(
        create_dropout_clamps(structure, batch_size, dropout_key, training=training)
    )

    # Single forward pass
    state = initialize_graph_state(
        structure,
        batch_size,
        state_key,
        clamps=clamps,
        params=params,
    )

    # Cross-entropy over all positions
    output_node = structure.task_map["y"]
    predictions = state.nodes[output_node].z_mu
    targets = batch["y"]

    loss = compute_loss(state, targets, output_node, "cross_entropy")

    return loss, predictions


def train_step_backprop_autoregressive(
    params: GraphParams,
    opt_state: optax.OptState,
    batch: Dict[str, jnp.ndarray],
    structure: GraphStructure,
    optimizer: optax.GradientTransformation,
    rng_key: jax.Array,
    use_causal_mask: bool = True,
) -> Tuple[GraphParams, optax.OptState, float, jnp.ndarray]:
    """
    Single autoregressive backprop training step.

    Args:
        params: Current model parameters
        opt_state: Optimizer state
        batch: Batch data with 'x' and 'y' sequences
        structure: Graph structure
        optimizer: Optax optimizer
        rng_key: JAX random key
        use_causal_mask: Whether to apply causal masking

    Returns:
        Tuple of (updated_params, updated_opt_state, loss_value, predictions)
    """

    def loss_fn(p):
        return compute_loss_autoregressive(
            p, structure, batch, rng_key, use_causal_mask
        )

    # has_aux=True: loss_fn returns (loss, aux), grads computed w.r.t. loss only
    (loss, predictions), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = cast(GraphParams, optax.apply_updates(params, updates))

    return params, opt_state, loss, predictions


def train_step_backprop_autoregressive_pmap(
    params: GraphParams,
    opt_state: optax.OptState,
    batch: Dict[str, jnp.ndarray],
    structure: GraphStructure,
    optimizer: optax.GradientTransformation,
    rng_key: jax.Array,
    use_causal_mask: bool = True,
) -> Tuple[GraphParams, optax.OptState, jnp.ndarray, jnp.ndarray]:
    """Single backprop pmap training step with mean-gradient reduction."""
    (loss, predictions), grads = jax.value_and_grad(
        lambda p: compute_loss_autoregressive(
            p, structure, batch, rng_key, use_causal_mask
        ),
        has_aux=True,
    )(params)
    grads = jax.lax.pmean(grads, axis_name="devices")
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = cast(GraphParams, optax.apply_updates(params, updates))
    return params, opt_state, loss, predictions


def create_pmap_train_step_backprop_autoregressive(
    structure: GraphStructure,
    optimizer: optax.GradientTransformation,
    use_causal_mask: bool = True,
):
    """Create a pmap'd backprop training step with static args in closure."""

    def step_fn(params, opt_state, batch, rng_key):
        return train_step_backprop_autoregressive_pmap(
            params, opt_state, batch, structure, optimizer, rng_key, use_causal_mask
        )

    return jax.pmap(step_fn, axis_name="devices")


def train_backprop_autoregressive(
    params: GraphParams,
    structure: GraphStructure,
    train_loader: Any,
    optimizer: optax.GradientTransformation,
    config: dict,
    rng_key: jax.Array,
    verbose: bool = True,
    epoch_callback: Optional[Callable] = None,
    iter_callback: Optional[Callable] = None,
    pmap_single_device: bool = False,
    force_single_device: bool = False,
) -> Tuple[GraphParams, List[List[float]], List[Any]]:
    """
    Main autoregressive backprop training loop.

    This provides standard end-to-end backpropagation training for
    autoregressive models (e.g., transformers for language modeling).

    Args:
        params: Initial parameters
        structure: Graph structure
        train_loader: Data loader yielding batches with 'x' and 'y' keys
        optimizer: Optax optimizer (e.g., optax.adam(1e-3))
        config: Training configuration:
            - num_epochs: Number of training epochs
            - use_causal_mask: Whether to use causal masking (default True)
        rng_key: JAX random key
        verbose: Print progress
        epoch_callback: Optional callback per epoch
        iter_callback: Optional callback per batch

    Returns:
        Tuple of (trained_params, loss_history, epoch_results)
    """
    validate_feedforward_init(structure)

    force_single = force_single_device or config.get("force_single_device", False)
    n_devices = 1 if force_single else jax.local_device_count()
    use_pmap = not force_single and ((n_devices > 1) or pmap_single_device)

    opt_state = optimizer.init(params)
    if use_pmap:
        params = replicate_params(params, n_devices)
        opt_state = replicate_opt_state(opt_state, n_devices)
        step_fn = create_pmap_train_step_backprop_autoregressive(
            structure, optimizer, use_causal_mask=config.get("use_causal_mask", True)
        )
    else:
        step_fn = jax.jit(
            lambda p, o, b, k: train_step_backprop_autoregressive(
                p, o, b, structure, optimizer, k, config.get("use_causal_mask", True)
            )
        )

    # Training hyperparameters
    num_epochs = config.get("num_epochs", 10)  # supports float (e.g. 1.5)

    # Support fractional epochs: e.g. 1.5 -> 2 loop iterations, last stops at 50%
    total_epochs = math.ceil(num_epochs)
    frac = num_epochs - math.floor(num_epochs)

    iter_results = []
    epoch_results = []

    for epoch_idx in range(total_epochs):
        try:
            num_batches = len(train_loader)
        except TypeError:
            raise TypeError("train_loader must support len()")

        # On the final epoch, truncate if fractional
        is_last_epoch = epoch_idx == total_epochs - 1
        if is_last_epoch and frac > 0:
            max_batches = round(frac * num_batches)
        else:
            max_batches = num_batches

        # Split keys for actual batch count
        epoch_rng_key, rng_key = jax.random.split(rng_key)
        if use_pmap:
            device_keys = (
                jax.random.split(epoch_rng_key, n_devices)
                if n_devices > 1
                else epoch_rng_key[None, :]
            )
            batch_keys = jax.vmap(lambda key: jax.random.split(key, max_batches))(
                device_keys
            )
        else:
            batch_keys = jax.random.split(epoch_rng_key, max_batches)

        batch_losses = []
        epoch_loss = 0.0
        batches_processed = 0

        for batch_idx, batch_data in enumerate(train_loader):
            if batch_idx >= max_batches:
                break

            # Convert batch to JAX format
            if isinstance(batch_data, dict):
                batch = {k: jnp.array(v) for k, v in batch_data.items()}
            elif isinstance(batch_data, (list, tuple)):
                batch = {
                    "x": jnp.array(batch_data[0]),
                    "y": jnp.array(batch_data[1]),
                }
            else:
                raise ValueError(f"Unsupported batch format: {type(batch_data)}")

            if use_pmap:
                _validate_pmap_batch(batch, n_devices)
                batch = shard_batch(batch, n_devices)
                params, opt_state, loss, _ = step_fn(
                    params, opt_state, batch, batch_keys[:, batch_idx]
                )
                loss = float(jnp.mean(loss))
            else:
                params, opt_state, loss, _ = step_fn(
                    params, opt_state, batch, batch_keys[batch_idx]
                )
                loss = float(loss)

            epoch_loss += float(loss)
            batches_processed += 1

            if iter_callback is not None:
                batch_losses.append(iter_callback(epoch_idx, batch_idx, loss))
            else:
                batch_losses.append(float(loss))

        iter_results.append(batch_losses)
        if batches_processed == 0:
            raise RuntimeError(
                f"No batches were processed in epoch {epoch_idx + 1}. "
                "Check that the train_loader is not empty."
            )
        avg_loss = epoch_loss / batches_processed

        # Epoch callback
        if epoch_callback is not None:
            callback_params = (
                jax.tree_util.tree_map(lambda value: value[0], params)
                if use_pmap
                else params
            )
            epoch_results.append(
                epoch_callback(
                    epoch_idx,
                    callback_params,
                    structure,
                    config,
                    rng_key,
                    ce_loss=avg_loss,
                )
            )
        else:
            epoch_results.append(None)

        if verbose:
            perplexity = float(jnp.exp(avg_loss))
            print(
                f"Epoch {epoch_idx + 1}/{total_epochs}, Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}"
            )

    if use_pmap:
        params = jax.tree_util.tree_map(lambda value: value[0], params)

    return params, iter_results, epoch_results


def eval_step_backprop(
    params: GraphParams,
    batch: Dict[str, jnp.ndarray],
    structure: GraphStructure,
    rng_key: jax.Array,
    loss_type: str = "cross_entropy",
) -> Tuple[float, int, int]:
    """
    Single evaluation step for backprop-trained model. No latent inference phase. Use structure configured for feedforward graph_state_initializer.

    Args:
        params: Model parameters
        batch: Batch data
        structure: Graph structure
        rng_key: Random key for initialization
        loss_type: Loss function type

    Returns:
        Tuple of (loss, correct_predictions, batch_size)
    """

    state = compute_forward_pass(params, structure, batch, rng_key)

    # Get predictions
    output_node = structure.task_map["y"]
    predictions = state.nodes[output_node].z_mu
    targets = batch["y"]

    loss = compute_loss(state, targets, output_node, loss_type)

    # Compute accuracy (for classification)
    pred_labels = jnp.argmax(predictions, axis=-1)
    if targets.ndim > 1 and targets.shape[-1] > 1:
        true_labels = jnp.argmax(targets, axis=-1)
    else:
        true_labels = targets

    # Handle sequence vs non-sequence
    correct = jnp.sum(pred_labels == true_labels)

    return loss, correct, jnp.prod(jnp.array(pred_labels.shape))


def evaluate_backprop(
    params: GraphParams,
    structure: GraphStructure,
    test_loader: Any,
    config: dict,
    rng_key: jax.Array = None,
) -> Dict[str, float]:
    """
    Evaluate backprop-trained model on test data.

    Args:
        params: Trained parameters
        structure: Graph structure
        test_loader: Test data loader
        config: Evaluation config with loss_type
        rng_key: Random key (optional, uses fixed key if None)

    Returns:
        Dictionary with "loss", "accuracy", and optionally "perplexity"
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    validate_feedforward_init(structure)

    loss_type = config.get("loss_type", "cross_entropy")

    # JIT compile eval step
    jit_eval_step = jax.jit(
        lambda p, b, k: eval_step_backprop(p, b, structure, k, loss_type)
    )

    try:
        num_batches = len(test_loader)
    except TypeError:
        raise TypeError("test_loader must support len()")

    batch_keys = jax.random.split(rng_key, num_batches)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_idx, batch_data in enumerate(test_loader):
        # Convert batch
        if isinstance(batch_data, dict):
            batch = {k: jnp.array(v) for k, v in batch_data.items()}
        elif isinstance(batch_data, (list, tuple)):
            batch = {"x": jnp.array(batch_data[0]), "y": jnp.array(batch_data[1])}
        else:
            raise ValueError(f"Unsupported batch format: {type(batch_data)}")

        loss, correct, count = jit_eval_step(params, batch, batch_keys[batch_idx])

        total_loss += loss * count
        total_correct += correct
        total_samples += count

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    result = {
        "loss": avg_loss,
        "accuracy": accuracy,
    }

    # Add perplexity for cross-entropy loss
    if loss_type == "cross_entropy":
        result["perplexity"] = float(jnp.exp(avg_loss))

    return result


def create_pmap_eval_step_backprop_autoregressive(
    structure: GraphStructure,
    use_causal_mask: bool,
):
    """Create a pmap'd backprop evaluation step with static args in closure."""

    def step_fn(params, batch, rng_key):
        return compute_loss_autoregressive(
            params, structure, batch, rng_key, use_causal_mask, training=False
        )

    return jax.pmap(step_fn, axis_name="devices")


def evaluate_backprop_autoregressive(
    params: GraphParams,
    structure: GraphStructure,
    test_loader: Any,
    config: dict,
    rng_key: jax.Array,
    debug: bool = False,
    pmap_single_device: bool = False,
    force_single_device: bool = False,
) -> Dict[str, float]:
    """
    Evaluate backprop-trained autoregressive model on test data.

    Computes:
    - Average cross-entropy loss
    - Perplexity (exp of average loss)
    - Accuracy (next-token prediction)

    Args:
        params: Trained model parameters
        structure: Graph structure
        test_loader: Test data loader
        config: Evaluation config with use_causal_mask (default True)
        rng_key: JAX random key
        debug: If True, print diagnostic info for first batch

    Returns:
        Dictionary with loss, perplexity, and accuracy
    """

    validate_feedforward_init(structure)

    force_single = force_single_device or config.get("force_single_device", False)
    n_devices = 1 if force_single else jax.local_device_count()
    use_pmap = not force_single and ((n_devices > 1) or pmap_single_device)

    use_causal_mask = config.get("use_causal_mask", True)

    try:
        num_batches_total = len(test_loader)
    except TypeError:
        raise TypeError("test_loader must support len() for evaluation")

    if use_pmap:
        epoch_key, rng_key = jax.random.split(rng_key)
        device_keys = (
            jax.random.split(epoch_key, n_devices)
            if n_devices > 1
            else epoch_key[None, :]
        )
        batch_keys = jax.vmap(lambda key: jax.random.split(key, num_batches_total))(
            device_keys
        )
    else:
        batch_keys = jax.random.split(rng_key, num_batches_total)

    if use_pmap:
        pmap_eval_step = create_pmap_eval_step_backprop_autoregressive(
            structure, use_causal_mask
        )
        replicated_params = replicate_params(params, n_devices)
    else:
        jit_forward = jax.jit(
            lambda p, b, k: compute_loss_autoregressive(
                p, structure, b, k, use_causal_mask, training=False
            )
        )

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    num_batches = 0

    for batch_idx, batch_data in enumerate(test_loader):
        # Convert batch to JAX format
        if isinstance(batch_data, dict):
            batch = {k: jnp.array(v) for k, v in batch_data.items()}
        elif isinstance(batch_data, (list, tuple)):
            batch = {"x": jnp.array(batch_data[0]), "y": jnp.array(batch_data[1])}
        else:
            raise ValueError(f"Unsupported batch format: {type(batch_data)}")

        batch_size = batch["x"].shape[0]
        seq_len = batch["x"].shape[1]
        original_batch = batch

        # Single forward pass returns both loss and predictions
        if use_pmap:
            # Pad incomplete batches so no test data is dropped.
            # Padding is sliced off before metrics so it has no effect.
            pad_size = (-batch_size) % n_devices
            if pad_size:
                batch = {
                    key: jnp.concatenate(
                        [
                            value,
                            jnp.zeros((pad_size,) + value.shape[1:], dtype=value.dtype),
                        ]
                    )
                    for key, value in batch.items()
                }
            batch = shard_batch(batch, n_devices)
            loss_sharded, preds_sharded = pmap_eval_step(
                replicated_params, batch, batch_keys[:, batch_idx]
            )
            padded_batch_size = batch_size + pad_size
            loss = float(jnp.mean(loss_sharded))
            predictions = preds_sharded.reshape(
                padded_batch_size, *preds_sharded.shape[2:]
            )[:batch_size]
        else:
            loss, predictions = jit_forward(params, batch, batch_keys[batch_idx])

        batch = original_batch
        if use_pmap and pad_size:
            log_predictions = jnp.log(predictions + 1e-10)
            if batch["y"].ndim == 3:
                valid_targets = batch["y"]
                loss = -float(
                    jnp.mean(jnp.sum(valid_targets * log_predictions, axis=-1))
                )
            else:
                loss = -float(
                    jnp.mean(
                        jnp.take_along_axis(
                            log_predictions, batch["y"][..., None], axis=-1
                        )
                    )
                )

        # Debug diagnostics for first batch
        if debug and batch_idx == 0:
            tgt = batch["y"]
            if tgt.ndim == predictions.ndim - 1:
                tgt = jax.nn.one_hot(tgt, predictions.shape[-1])

            # Check individual loss components
            log_preds = jnp.log(predictions + 1e-10)
            per_token_loss = -jnp.sum(tgt * log_preds, axis=-1)  # (batch, seq_len)
            print(
                f"  [DEBUG] per-token CE loss: min={float(jnp.min(per_token_loss)):.4f}, max={float(jnp.max(per_token_loss)):.4f}, mean={float(jnp.mean(per_token_loss)):.4f}"
            )

            token_intrinsic_perplexity = jnp.exp(
                -jnp.sum(predictions * log_preds, axis=-1)
            )  # (batch, seq_len)
            print(
                f"  [DEBUG] per-token intrinsic perplexity: min={float(jnp.min(token_intrinsic_perplexity)):.4f}, max={float(jnp.max(token_intrinsic_perplexity)):.4f}, mean={float(jnp.mean(token_intrinsic_perplexity)):.4f}"
            )

            # Check if there are extreme values
            correct_probs = jnp.sum(
                tgt * predictions, axis=-1
            )  # prob assigned to correct class
            print(
                f"  [DEBUG] prob of correct token: min={float(jnp.min(correct_probs)):.6f}, max={float(jnp.max(correct_probs)):.6f}, mean={float(jnp.mean(correct_probs)):.6f}"
            )

            print(f"  [DEBUG] batch loss: {float(loss):.4f}")

        total_loss += float(loss)

        # Get target tokens
        targets = batch["y"]
        if targets.ndim == 3:
            target_tokens = jnp.argmax(targets, axis=-1)
        else:
            target_tokens = targets

        pred_tokens = jnp.argmax(predictions, axis=-1)
        correct = jnp.sum(pred_tokens == target_tokens)
        total_correct += int(correct)
        total_tokens += batch_size * seq_len
        num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
    perplexity = float(jnp.exp(avg_loss))

    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "accuracy": accuracy,
        "num_batches": num_batches,
    }
