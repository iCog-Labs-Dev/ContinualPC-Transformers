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

from typing import Dict, Tuple, Any, List, Optional, Callable, cast
import math
import jax
import jax.numpy as jnp
import optax

from fabricpc.core.types import GraphParams, GraphStructure, GraphState
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


def _input_clamps(
    structure: GraphStructure, batch: Dict[str, jnp.ndarray]
) -> Dict[str, jnp.ndarray]:
    """Clamp every batch entry that names a task node, except the target ``y``.

    Besides ``x`` this covers auxiliary source nodes such as ``attention_mask``
    and ``loss_mask``; leaving those unclamped would give them random latents.
    Supplied entries take precedence: default ``dropout_*`` and ``causal_mask``
    inputs are generated only for tasks missing from the batch.
    """
    return {
        structure.task_map[task_name]: value
        for task_name, value in batch.items()
        if task_name in structure.task_map and task_name != "y"
    }


def compute_forward_pass(
    params: GraphParams,
    structure: GraphStructure,
    batch: Dict[str, jnp.ndarray],
    rng_key: jax.Array,
    training: bool = False,
) -> GraphState:
    """
    Compute forward pass for backpropagation training.

    This runs a single forward pass through the network (no iterative
    inference) and computes output at the output node. Every batch entry that
    names a task (except ``y``) is clamped as given; dropout masks are sampled
    (or set to ones) only for ``dropout_*`` tasks absent from the batch.

    Args:
        params: Model parameters
        structure: Graph structure
        batch: Batch data with keys matching task_map (e.g., 'x', 'y')
        rng_key: JAX random key for state initialization
        training: If True, sample dropout masks; otherwise clamp ones.

    Returns:
        graph state after feedforward pass
    """

    validate_feedforward_init(structure)

    batch_size = batch["x"].shape[0]

    # Clamp inputs, not the output - key difference from PC
    clamps = _input_clamps(structure, batch)
    if any(task_name.startswith("dropout_") for task_name in structure.task_map):
        rng_key, dropout_key = jax.random.split(rng_key)
        clamps.update(
            create_dropout_clamps(
                structure, batch_size, dropout_key, training, exclude=batch.keys()
            )
        )

    # Single forward pass via initialize_state with feedforward init
    state = initialize_graph_state(
        structure,
        batch_size,
        rng_key,
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
) -> Tuple[GraphParams, optax.OptState, float]:
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
        return compute_loss(
            state,
            batch["y"],
            structure.task_map["y"],
            loss_type,
            loss_mask=batch.get("loss_mask"),
        )

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = cast(GraphParams, optax.apply_updates(params, updates))

    return params, opt_state, loss


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


def _reject_masked_pmap_batch(batch: Dict[str, jnp.ndarray]) -> None:
    """Masked losses are normalised per device under pmap, which mis-weights
    the global masked mean, so refuse them until a token-weighted reduction
    exists."""
    if "loss_mask" in batch:
        raise NotImplementedError(
            "Batches with 'loss_mask' (masked-LM training) are not supported on "
            "the multi-device pmap path yet; run on a single device."
        )


def compute_loss_autoregressive(
    params: GraphParams,
    structure: GraphStructure,
    batch: Dict[str, jnp.ndarray],
    rng_key: jax.Array,
    use_causal_mask: bool = True,
    training: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute sequence cross-entropy loss with backprop.

    This runs a single forward pass (with causal masking when the graph
    defines a ``causal_mask`` task) and computes cross-entropy over all
    sequence positions, or only over ``batch["loss_mask"]`` positions when
    present (masked-LM training). A ``causal_mask`` or ``dropout_*`` entry in
    the batch is clamped as given instead of being generated.

    Args:
        params: Model parameters
        structure: Graph structure
        batch: Batch with "x" (input) and "y" (target) sequences, plus any
            auxiliary task inputs (e.g. "attention_mask", "loss_mask")
        rng_key: JAX random key
        use_causal_mask: Whether to apply causal masking
        training: If True, sample dropout masks; otherwise clamp ones.

    Returns:
        Tuple of (loss, predictions) where:
            - loss: Scalar loss (mean over positions, or over masked positions)
            - predictions: Output logits of shape (batch, seq_len, vocab_size)
    """
    batch_size = batch["x"].shape[0]
    seq_len = batch["x"].shape[1]

    # Clamp inputs (x and any auxiliary task nodes), not the output
    clamps = _input_clamps(structure, batch)

    # Add an external causal mask only if the graph defines one (v1). The v2
    # decomposed MhaResidualNode masks internally via is_causal, so its task_map
    # has no "causal_mask" node and this is skipped.
    if (
        use_causal_mask
        and "causal_mask" in structure.task_map
        and "causal_mask" not in batch
    ):
        causal_mask = create_causal_mask(seq_len)
        causal_mask = causal_mask[None, None, :, :]  # (1, 1, seq, seq)
        causal_mask = jnp.broadcast_to(causal_mask, (batch_size, 1, seq_len, seq_len))
        clamps[structure.task_map["causal_mask"]] = causal_mask

    rng_key, dropout_key, state_key = jax.random.split(rng_key, 3)
    clamps.update(
        create_dropout_clamps(
            structure, batch_size, dropout_key, training=training, exclude=batch.keys()
        )
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

    loss = compute_loss(
        state,
        targets,
        output_node,
        "cross_entropy",
        loss_mask=batch.get("loss_mask"),
    )

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
    """Single backprop pmap training step with mean-gradient reduction.

    Raises:
        NotImplementedError: If the batch carries a ``loss_mask``.
    """
    _reject_masked_pmap_batch(batch)
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
        Tuple of (trained_params, loss_history, epoch_results). The reported
        epoch loss is the mean of per-batch losses, not a token-weighted mean.
    """
    validate_feedforward_init(structure)

    n_devices = jax.local_device_count()
    use_pmap = (n_devices > 1) or pmap_single_device

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
    use_causal_mask = config.get("use_causal_mask", True)

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
            batch_keys = jax.vmap(
                lambda key: jax.random.split(key, max_batches)
            )(device_keys)
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
                _reject_masked_pmap_batch(batch)
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


def _target_labels(targets: jnp.ndarray, predictions: jnp.ndarray) -> jnp.ndarray:
    """Class labels from targets: argmax of one-hot targets, which have the
    predictions' exact shape; any other targets are already integer labels."""
    if targets.shape == predictions.shape:
        return jnp.argmax(targets, axis=-1)
    return targets


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

    loss_mask = batch.get("loss_mask")
    loss = compute_loss(state, targets, output_node, loss_type, loss_mask=loss_mask)

    # Compute accuracy (for classification)
    pred_labels = jnp.argmax(predictions, axis=-1)
    true_labels = _target_labels(targets, predictions)

    # Handle sequence vs non-sequence
    hits = pred_labels == true_labels
    if loss_mask is not None:
        weights = loss_mask.astype(jnp.float32)
        return loss, jnp.sum(hits * weights), jnp.sum(weights)
    return loss, jnp.sum(hits), jnp.prod(jnp.array(pred_labels.shape))


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
    """Create a pmap'd backprop evaluation step with static args in closure.

    The step raises ``NotImplementedError`` for batches with a ``loss_mask``.
    """

    def step_fn(params, batch, rng_key):
        _reject_masked_pmap_batch(batch)
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

    n_devices = jax.local_device_count()
    use_pmap = (n_devices > 1) or pmap_single_device

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
        batch_keys = jax.vmap(
            lambda key: jax.random.split(key, num_batches_total)
        )(device_keys)
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
        if "loss_mask" in batch:
            raise ValueError(
                "evaluate_backprop_autoregressive averages over every position; "
                "use evaluate_mlm for batches with a 'loss_mask'"
            )

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
                        [value, jnp.zeros((pad_size,) + value.shape[1:], dtype=value.dtype)]
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


def _masked_lm_batch_totals(
    params: GraphParams,
    structure: GraphStructure,
    batch: Dict[str, jnp.ndarray],
    rng_key: jax.Array,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return (summed masked CE, correct masked predictions, masked count)."""
    state = compute_forward_pass(params, structure, batch, rng_key)
    output_node = structure.task_map["y"]
    mask = batch["loss_mask"].astype(jnp.float32)
    count = jnp.sum(mask)
    mean_loss = compute_loss(
        state, batch["y"], output_node, "cross_entropy", loss_mask=mask
    )
    predictions = jnp.argmax(state.nodes[output_node].z_mu, axis=-1)
    correct = jnp.sum((predictions == batch["y"]) * mask)
    return mean_loss * count, correct, count


_masked_lm_eval_cache: Dict[str, Any] = {"structure": None, "fn": None}


def _jitted_masked_lm_batch_totals(structure: GraphStructure) -> Callable:
    if _masked_lm_eval_cache["structure"] is not structure:
        _masked_lm_eval_cache["fn"] = jax.jit(
            lambda p, b, k: _masked_lm_batch_totals(p, structure, b, k)
        )
        _masked_lm_eval_cache["structure"] = structure
    return _masked_lm_eval_cache["fn"]


def evaluate_mlm(
    params: GraphParams,
    structure: GraphStructure,
    loader: Any,
    rng_key: jax.Array,
) -> Dict[str, float]:
    """Masked-LM loss and accuracy over a loader, weighted by masked tokens.

    Runs a feedforward pass (dropout off) on every batch and averages only
    over ``loss_mask`` positions across the whole loader. If the loader has
    ``state_dict``/``load_state_dict`` (e.g. ``MlmTextBatcher``), its state is
    restored afterwards so repeated validation sees the same masks.

    Returns:
        ``masked_loss``, ``masked_accuracy`` and ``num_masked_tokens``.

    Raises:
        ValueError: If a batch lacks ``loss_mask`` or no token is masked.
    """
    batch_totals = _jitted_masked_lm_batch_totals(structure)
    saved_state = loader.state_dict() if hasattr(loader, "state_dict") else None
    loss_sum, correct, count = 0.0, 0.0, 0.0
    try:
        for batch_data in loader:
            if not isinstance(batch_data, dict) or "loss_mask" not in batch_data:
                raise ValueError("evaluate_mlm requires dict batches with 'loss_mask'")
            batch = {k: jnp.asarray(v) for k, v in batch_data.items()}
            rng_key, step_key = jax.random.split(rng_key)
            batch_loss, batch_correct, batch_count = batch_totals(
                params, batch, step_key
            )
            loss_sum += float(batch_loss)
            correct += float(batch_correct)
            count += float(batch_count)
    finally:
        if saved_state is not None:
            loader.load_state_dict(saved_state)

    if count == 0:
        raise ValueError("evaluate_mlm found no masked tokens in the loader")
    return {
        "masked_loss": loss_sum / count,
        "masked_accuracy": correct / count,
        "num_masked_tokens": int(count),
    }
