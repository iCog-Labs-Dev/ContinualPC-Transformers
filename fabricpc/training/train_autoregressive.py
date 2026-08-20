"""
Autoregressive training loop for transformer predictive coding networks.

This module implements training for autoregressive (next-token prediction) tasks
using the predictive coding framework. Key features:

1. Teacher forcing: During training, use ground truth tokens as input
2. Causal masking: Ensure predictions only depend on past tokens
3. Sequence-level energy: Aggregate prediction errors across positions
4. Local Hebbian learning: Weight updates based on local gradients

The training loop supports both:
- Full sequence prediction (all positions predict next token)
- Last-token prediction (only predict final token, for efficiency)
"""

from typing import Dict, Tuple, Any, List, Optional, Callable, cast
import math
import warnings
import jax
import jax.numpy as jnp
import optax

from fabricpc.core.types import GraphParams, GraphState, GraphStructure
from fabricpc.core.inference import run_inference
from fabricpc.core.learning import compute_local_weight_gradients
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.training.train import (
    _convert_batch,
    replicate_params,
    replicate_opt_state,
    shard_batch,
    unshard_energies,
)


def create_causal_mask(seq_len: int) -> jnp.ndarray:
    """
    Create a causal attention mask for autoregressive modeling.

    Returns:
        Mask of shape (seq_len, seq_len) where mask[i,j] = 1 if j <= i else 0
        This ensures position i can only attend to positions 0...i
    """
    return jnp.tril(jnp.ones((seq_len, seq_len)))

def create_dropout_clamps(
    structure: GraphStructure,
    batch_size: int,
    rng_key: jax.Array,
    training: bool,
) -> Dict[str, jnp.ndarray]:
    """Create clamps for transformer-v2 dropout mask source nodes.

    Dropout masks are graph inputs whose task names start with ``dropout_``.
    During training, each mask is Bernoulli(keep_prob) / keep_prob
    (inverted dropout). During evaluation/generation, masks are all ones, so
    no activation rescaling is applied at test time. Keeping dropout as clamped
    source data makes the mask fixed over the entire PC inference solve.
    """
    dropout_tasks = tuple(
        sorted(k for k in structure.task_map if k.startswith("dropout_"))
    )
    if not dropout_tasks:
        return {}

    dropout_rate = float(structure.config.get("dropout_rate", 0.0))
    if not 0.0 <= dropout_rate < 1.0:
        raise ValueError(f"dropout_rate must be in [0, 1), got {dropout_rate}")

    keep_prob = 1.0 - dropout_rate
    keys = jax.random.split(rng_key, len(dropout_tasks))
    clamps = {}

    for task_name, key in zip(dropout_tasks, keys):
        node_name = structure.task_map[task_name]
        node_shape = structure.nodes[node_name].node_info.shape
        mask_shape = (batch_size, *node_shape)

        if training and dropout_rate > 0.0:
            mask = jax.random.bernoulli(key, keep_prob, mask_shape)
            mask = mask.astype(jnp.float32) / keep_prob
        else:
            mask = jnp.ones(mask_shape, dtype=jnp.float32)

        clamps[node_name] = mask

    return clamps

def compute_loss(
    final_state: GraphState,
    targets: jnp.ndarray,
    output_node: str,
    loss_type: str = "cross_entropy",
) -> jnp.ndarray:
    """
    Compute differentiable loss

    Args:
        final_state: Final graph state after forward pass
        targets: Target values (labels) for output node, expects one-hot for cross-entropy
        output_node: Name of the output node
        loss_type: Loss function type: "cross_entropy" or "mse"

    Returns:
        Scalar loss value (mean over batch)
    """

    # Get prediction from output node
    predictions = final_state.nodes[output_node].z_mu

    # Compute loss
    # Targets may arrive as integer token ids (batch, seq_len) — the loaders
    # yield ints to avoid a large one-hot per batch. One-hot here to match the
    # prediction shape; already one-hot targets (equal ndim) pass through.
    if loss_type == "cross_entropy":
        if targets.ndim == predictions.ndim - 1:
            targets = jax.nn.one_hot(targets, predictions.shape[-1])
        log_preds = jnp.log(predictions + 1e-10)
        loss = -jnp.mean(jnp.sum(targets * log_preds, axis=-1))

    elif loss_type == "mse":
        loss = jnp.mean((predictions - targets) ** 2)

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    return loss


def get_graph_param_gradient_autoregressive(
    params: GraphParams,
    batch: Dict[str, jnp.ndarray],
    structure: GraphStructure,
    rng_key: jax.Array,
    use_causal_mask: bool = True,
) -> Tuple[GraphParams, float, float, GraphState]:
    """
    Compute parameter gradients and metrics for an autoregressive batch.

    Shared by both single-device (JIT) and multi-device (pmap) training.

    Args:
        params: Current model parameters
        batch: Batch dictionary with task-specific keys
        structure: Graph structure
        rng_key: JAX random key
        use_causal_mask: Whether to apply causal masking

    Returns:
        Tuple of (grads, avg_energy, output_cross_entropy, final_state)
    """
    batch_size = batch["x"].shape[0]
    seq_len = batch["x"].shape[1]

    # Map tasks to nodes using task_map
    clamps = {}
    for task_name, task_value in batch.items():
        if task_name in structure.task_map:
            node_name = structure.task_map[task_name]
            clamps[node_name] = task_value

    # Targets arrive as integer token ids (batch, seq_len) to keep the
    # host->device transfer int32 (a one-hot is vocab_size x larger). The
    # CrossEntropy output node is clamped with a one-hot latent, so encode here.
    # ndim is static under jit, so the check raises at trace time.
    y = batch["y"]
    if y.ndim != 2:
        raise ValueError(
            f"batch['y'] must be integer token ids of shape (batch, seq_len); "
            f"got shape {y.shape}. Token loaders yield integer targets; "
            f"one-hot encoding happens in this step."
        )
    y = jax.nn.one_hot(y, structure.nodes[structure.task_map["y"]].node_info.shape[-1])
    clamps[structure.task_map["y"]] = y

    # Clamp an external causal mask only if the graph defines one (the v1
    # monolithic TransformerBlock). The v2 decomposed MhaResidualNode masks
    # internally via is_causal, so its task_map has no "causal_mask" node and
    # this branch is skipped.
    if use_causal_mask and "causal_mask" in structure.task_map:
        # Create causal mask: (seq_len, seq_len) where mask[i,j] = 1 if j <= i
        causal_mask = create_causal_mask(seq_len)
        # Broadcast to (batch, 1, seq_len, seq_len) for attention scores
        causal_mask = causal_mask[None, None, :, :]
        causal_mask = jnp.broadcast_to(causal_mask, (batch_size, 1, seq_len, seq_len))

        # Clamp the causal mask to the node specified in task_map
        mask_node = structure.task_map["causal_mask"]
        clamps[mask_node] = causal_mask

    rng_key, dropout_key, state_key = jax.random.split(rng_key, 3)
    clamps.update(
        create_dropout_clamps(structure, batch_size, dropout_key, training=True)
    )

    # Initialize state
    init_state = initialize_graph_state(
        structure,
        batch_size,
        state_key,
        clamps=clamps,
        params=params,
    )

    # Run inference
    final_state = run_inference(params, init_state, clamps, structure)

    # Compute total energy (sum over nodes with in_degree>0)
    energy = jnp.array(0.0)
    for node_name, node in structure.nodes.items():
        if node.node_info.in_degree > 0:
            energy = energy + jnp.sum(final_state.nodes[node_name].energy)

    avg_energy = energy / batch_size

    # Compute local gradients
    grads = compute_local_weight_gradients(params, final_state, structure)

    # Compute output cross-entropy loss for perplexity metric - not used for gradients
    output_cross_entropy = compute_loss(
        final_state, y, structure.task_map["y"], loss_type="cross_entropy"
    )

    return grads, avg_energy, output_cross_entropy, final_state


def train_step_autoregressive(
    params: GraphParams,
    opt_state: optax.OptState,
    batch: Dict[str, jnp.ndarray],
    structure: GraphStructure,
    optimizer: optax.GradientTransformation,
    rng_key: jax.Array,
    use_causal_mask: bool = True,
) -> Tuple[GraphParams, optax.OptState, float, float, GraphState]:
    """
    Single autoregressive training step with local updates (JIT path).

    Args:
        params: Current model parameters
        opt_state: Optimizer state
        batch: Batch of data with task-specific keys
        structure: Graph structure
        optimizer: Optax optimizer
        rng_key: JAX random key
        use_causal_mask: Whether to apply causal masking

    Returns:
        Tuple of (updated_params, updated_opt_state, avg_energy, ce_loss, final_state)
    """
    grads, avg_energy, output_cross_entropy, final_state = (
        get_graph_param_gradient_autoregressive(
            params, batch, structure, rng_key, use_causal_mask
        )
    )

    # Update parameters using optimizer
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = cast(GraphParams, optax.apply_updates(params, updates))

    return (
        params,
        opt_state,
        avg_energy.astype(float),
        output_cross_entropy.astype(float),
        final_state,
    )


def train_step_autoregressive_pmap(
    params: GraphParams,
    opt_state: optax.OptState,
    batch: Dict[str, jnp.ndarray],
    structure: GraphStructure,
    optimizer: optax.GradientTransformation,
    rng_key: jax.Array,
    use_causal_mask: bool = True,
) -> Tuple[GraphParams, optax.OptState, jnp.ndarray, jnp.ndarray, GraphState]:
    """
    Training step for pmap: compute grads, average across devices, update.

    Args:
        params: Replicated parameters (has device axis)
        opt_state: Replicated optimizer state (has device axis)
        batch: Sharded batch (has device axis)
        structure: Graph structure
        optimizer: Optax optimizer
        rng_key: JAX random key for this device
        use_causal_mask: Whether to apply causal masking

    Returns:
        Tuple of (updated_params, updated_opt_state, energy_per_device, ce_loss_per_device, final_state)
    """
    grads, avg_energy, output_cross_entropy, final_state = (
        get_graph_param_gradient_autoregressive(
            params, batch, structure, rng_key, use_causal_mask
        )
    )

    # Sum per-device batch gradients
    grads = jax.lax.psum(grads, axis_name="devices")

    # Update parameters (each device now has same gradients)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = cast(GraphParams, optax.apply_updates(params, updates))

    return (
        params,
        opt_state,
        avg_energy.astype(float),
        output_cross_entropy.astype(float),
        final_state,
    )


def create_pmap_train_step_autoregressive(
    structure: GraphStructure,
    optimizer: optax.GradientTransformation,
    use_causal_mask: bool = True,
):
    """
    Create a pmap'd autoregressive training step with static args in closure.

    Args:
        structure: Graph structure (static)
        optimizer: Optimizer (static)
        use_causal_mask: Whether to use causal masking (static)

    Returns:
        Pmap'd training step function
    """

    def step_fn(params, opt_state, batch, rng_key):
        return train_step_autoregressive_pmap(
            params,
            opt_state,
            batch,
            structure,
            optimizer,
            rng_key,
            use_causal_mask,
        )

    return jax.pmap(step_fn, axis_name="devices")


def train_autoregressive(
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
    Main training loop for autoregressive predictive coding.

    Automatically detects available devices: uses JIT compilation on a single
    device, pmap data parallelism across multiple devices.

    Args:
        params: Initial parameters
        structure: Graph structure
        train_loader: Data loader yielding batches with 'x' and 'y' keys
        optimizer: Optax optimizer (e.g., optax.adam(1e-3))
        config: Training configuration:
            - num_epochs: Number of training epochs
            - use_causal_mask: Whether to use causal masking (default True)
            - gradient_accumulation_steps: Steps to accumulate gradients (default 1)
        rng_key: JAX random key
        verbose: Whether to print progress
        epoch_callback: Optional callback called at each epoch end as
            (epoch, params, structure, config, rng, *, energy, ce_loss) -> any.
            Receives the epoch's mean training energy and CE loss; should
            accept **kwargs.
        iter_callback: Optional callback (epoch, batch_idx, loss) -> any
        pmap_single_device: Force pmap code path for testing

    Returns:
        Tuple of (trained_params, energy_history, epoch_results)
    """
    # Training hyperparameters
    num_epochs = config.get("num_epochs", 10)  # supports float (e.g. 1.5)
    use_causal_mask = config.get("use_causal_mask", True)
    grad_accum_steps = config.get("gradient_accumulation_steps", 1)

    # Support fractional epochs: e.g. 1.5 -> 2 loop iterations, last stops at 50%
    total_epochs = math.ceil(num_epochs)
    frac = num_epochs - math.floor(num_epochs)

    # Device detection
    n_devices = jax.device_count()
    use_pmap = (n_devices > 1) or pmap_single_device

    if verbose:
        print(f"Training on {n_devices} device(s): {jax.devices()}")

    opt_state = optimizer.init(params)

    # Set up step function based on device path
    if use_pmap:
        params = replicate_params(params, n_devices)
        opt_state = replicate_opt_state(opt_state, n_devices)
        step_fn = create_pmap_train_step_autoregressive(
            structure, optimizer, use_causal_mask
        )
    else:
        step_fn = jax.jit(
            lambda p, o, b, k: train_step_autoregressive(
                p, o, b, structure, optimizer, k, use_causal_mask
            )
        )

    _shard_warned = False
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

        # Split keys for this epoch
        epoch_rng_key, rng_key = jax.random.split(rng_key)
        if use_pmap:
            dev_keys = (
                jax.random.split(epoch_rng_key, n_devices)
                if n_devices > 1
                else epoch_rng_key[None, :]
            )
            batch_keys = jax.vmap(
                lambda k: jax.random.split(k, max_batches)
            )(dev_keys)
        else:
            batch_keys = jax.random.split(epoch_rng_key, max_batches)

        batch_energies = []
        epoch_energy = 0.0
        epoch_ce_loss = 0.0
        batches_processed = 0

        for batch_idx, batch_data in enumerate(train_loader):
            if batch_idx >= max_batches:
                break

            batch = _convert_batch(batch_data)

            if use_pmap:
                try:
                    batch = shard_batch(batch, n_devices)
                except ValueError as e:
                    if not _shard_warned:
                        warnings.warn(
                            f"Skipping batch (size not divisible by {n_devices}): {e}"
                        )
                        _shard_warned = True
                    continue

                params, opt_state, energy, ce_loss, _ = step_fn(
                    params, opt_state, batch, batch_keys[:, batch_idx]
                )
                energy, ce_loss = unshard_energies(energy), float(jnp.mean(ce_loss))
            else:
                params, opt_state, energy, ce_loss, _ = step_fn(
                    params, opt_state, batch, batch_keys[batch_idx]
                )
                energy, ce_loss = float(energy), float(ce_loss)

            epoch_energy += energy
            epoch_ce_loss += ce_loss
            batches_processed += 1

            if iter_callback is not None:
                batch_energies.append(iter_callback(epoch_idx, batch_idx, energy))
            else:
                batch_energies.append(energy)

        iter_results.append(batch_energies)
        if batches_processed == 0:
            raise RuntimeError(
                f"No batches were processed in epoch {epoch_idx + 1}. "
                f"Ensure batch sizes are divisible by {n_devices} device(s)."
            )
        avg_energy = epoch_energy / batches_processed
        avg_ce_loss = epoch_ce_loss / batches_processed

        # Epoch callback. Pass the epoch's training metrics as keyword args so
        # callers (e.g. the tuner) can report per-epoch progress to a pruner
        # without recomputing them. Callbacks should accept **kwargs.
        if epoch_callback is not None:
            cb_params = (
                jax.tree_util.tree_map(lambda x: x[0], params)
                if use_pmap
                else params
            )
            epoch_results.append(
                epoch_callback(
                    epoch_idx,
                    cb_params,
                    structure,
                    config,
                    rng_key,
                    energy=avg_energy,
                    ce_loss=avg_ce_loss,
                )
            )
        else:
            epoch_results.append(None)

        if verbose:
            perplexity = float(jnp.exp(avg_ce_loss))
            print(
                f"Train Epoch {epoch_idx + 1}/{total_epochs}, Energy: {avg_energy:.4f}, Loss: {avg_ce_loss:.4f}, Perplexity: {perplexity:.2f}"
            )

    # Extract single-device params on return
    if use_pmap:
        params = jax.tree_util.tree_map(lambda x: x[0], params)

    return params, iter_results, epoch_results


def _generation_step(
    carry: Tuple[jnp.ndarray, jnp.ndarray, jax.Array],
    step_idx: int,
    params: GraphParams,
    structure: GraphStructure,
    input_node: str,
    output_node: str,
    seq_len: int,
    vocab_size: int,
    batch_size: int,
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray, jax.Array], jnp.ndarray]:
    """
    Single generation step for use with jax.lax.scan.

    Uses a sliding window approach with fixed-size buffers to maintain
    constant shapes required by JAX scan.

    Args:
        carry: Tuple of (context_window, output_buffer, rng_key)
            - context_window: Fixed-size (batch, seq_len) window of recent tokens
            - output_buffer: Fixed-size (batch, max_new_tokens) buffer for generated tokens
            - rng_key: JAX random key
        step_idx: Current step index, used to write to output buffer
        params, structure, etc.: Static parameters passed via closure

    Returns:
        Tuple of (new_carry, next_token)
    """
    context_window, output_buffer, rng_key = carry
    rng_key, sample_key, init_key = jax.random.split(rng_key, 3)

    # Format context window for the input node.
    # If input node shape is 1D (seq_len,), pass int token indices (EmbeddingNode).
    # If input node shape is 2D (seq_len, vocab_size), convert to one-hot (Linear).
    input_shape = structure.nodes[input_node].node_info.shape
    if len(input_shape) == 1:
        input_data = context_window
    else:
        input_data = jax.nn.one_hot(context_window, vocab_size)

    # Create clamps (only input, not output)
    clamps = {input_node: input_data}
    clamps.update(
        create_dropout_clamps(structure, batch_size, init_key, training=False)
    )

    # Initialize and run inference
    state = initialize_graph_state(
        structure,
        batch_size,
        init_key,
        clamps=clamps,
        params=params,
    )
    final_state = run_inference(params, state, clamps, structure)

    # Get output for the last position
    # z_mu contains the predicted output after activation (softmax for output node)
    # z_latent is the raw latent state before activation
    output_probs = final_state.nodes[output_node].z_mu
    output_last = output_probs[:, -1, :]  # (batch, vocab_size)

    # Convert to log-probabilities for sampling
    # z_mu after softmax should be probabilities, convert to log-probs
    # Adding epsilon avoids log(0)
    logits = jnp.log(output_last + 1e-10)

    # Apply temperature (divide log-probs, equivalent to taking prob^(1/T))
    logits = logits / temperature

    # Apply top-k filtering (always run but with large k if not specified)
    effective_top_k = top_k if top_k is not None else vocab_size
    top_k_logits, top_k_indices = jax.lax.top_k(logits, effective_top_k)
    # Set non-top-k logits to -inf
    neg_inf_mask = jnp.full_like(logits, float("-inf"))
    logits = neg_inf_mask.at[jnp.arange(batch_size)[:, None], top_k_indices].set(
        top_k_logits
    )

    # Apply top-p (nucleus) filtering
    if top_p is not None:
        sorted_indices = jnp.argsort(-logits, axis=-1)
        sorted_logits = jnp.take_along_axis(logits, sorted_indices, axis=-1)
        sorted_probs = jax.nn.softmax(sorted_logits, axis=-1)
        cumsum_probs = jnp.cumsum(sorted_probs, axis=-1)
        # Find cutoff
        cutoff_mask = cumsum_probs > top_p
        # Shift mask to keep at least one token
        cutoff_mask = jnp.concatenate(
            [jnp.zeros((batch_size, 1), dtype=bool), cutoff_mask[:, :-1]], axis=-1
        )
        sorted_logits = jnp.where(cutoff_mask, float("-inf"), sorted_logits)
        # Unsort
        unsort_indices = jnp.argsort(sorted_indices, axis=-1)
        logits = jnp.take_along_axis(sorted_logits, unsort_indices, axis=-1)

    # Sample from distribution
    next_token = jax.random.categorical(sample_key, logits, axis=-1)  # (batch,)

    # Update context window: shift left and append new token
    new_context = jnp.concatenate([context_window[:, 1:], next_token[:, None]], axis=1)

    # Write to output buffer at current step index
    new_output_buffer = output_buffer.at[:, step_idx].set(next_token)

    return (new_context, new_output_buffer, rng_key), next_token


def generate_autoregressive(
    params: GraphParams,
    structure: GraphStructure,
    prompt: jnp.ndarray,
    max_new_tokens: int,
    rng_key: jax.Array,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> jnp.ndarray:
    """
    Generate tokens autoregressively using the trained model.

    This function is JIT-compiled for efficient generation. The inner loop
    uses jax.lax.scan with fixed-size buffers for optimal performance.

    Args:
        params: Trained model parameters
        structure: Graph structure
        prompt: Initial token indices, shape (seq_len,) or (batch, seq_len)
        max_new_tokens: Number of new tokens to generate
        rng_key: JAX random key
        temperature: Sampling temperature (higher = more random, 1=neutral, <1=less random)
        top_k: If set, only sample from top-k tokens
        top_p: If set, use nucleus sampling with this probability threshold

    Returns:
        Generated token indices, shape (prompt_len + max_new_tokens,) or (batch, prompt_len + max_new_tokens)
    """
    # Handle batched vs unbatched input
    if prompt.ndim == 1:
        prompt = prompt[None, :]  # Add batch dimension
        unbatch = True
    else:
        unbatch = False

    batch_size, prompt_len = prompt.shape
    input_node = structure.task_map.get("x")
    output_node = structure.task_map.get("y")

    if input_node is None or output_node is None:
        raise ValueError("Structure must have 'x' and 'y' in task_map")

    vocab_size = structure.nodes[output_node].node_info.shape[-1]
    seq_len = structure.nodes[input_node].node_info.shape[0]

    # Prepare initial context window (pad or truncate prompt to seq_len)
    if prompt_len >= seq_len:
        # Use last seq_len tokens as context
        context_window = prompt[:, -seq_len:]
    else:
        # Pad at the beginning with zeros
        pad_len = seq_len - prompt_len
        context_window = jnp.pad(prompt, ((0, 0), (pad_len, 0)), constant_values=0)

    # Create JIT-compiled generation step with static arguments closed over
    @jax.jit
    def jit_generate_loop(context: jnp.ndarray, rng: jax.Array) -> jnp.ndarray:
        """JIT-compiled generation loop using lax.scan."""

        def scan_fn(carry, step_idx):
            return _generation_step(
                carry,
                step_idx,
                params=params,
                structure=structure,
                input_node=input_node,
                output_node=output_node,
                seq_len=seq_len,
                vocab_size=vocab_size,
                batch_size=batch_size,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

        # Initialize output buffer for generated tokens
        output_buffer = jnp.zeros((batch_size, max_new_tokens), dtype=jnp.int32)

        # Run the generation loop
        init_carry = (context, output_buffer, rng)
        (_, final_output_buffer, _), _ = jax.lax.scan(
            scan_fn, init_carry, jnp.arange(max_new_tokens)
        )

        return final_output_buffer

    # Run the JIT-compiled generation
    generated_tokens = jit_generate_loop(context_window, rng_key)

    # Concatenate prompt with generated tokens
    result = jnp.concatenate([prompt, generated_tokens], axis=1)

    if unbatch:
        result = result[0]  # Remove batch dimension

    return result


def _eval_step_autoregressive(
    params: GraphParams,
    structure: GraphStructure,
    batch: Dict[str, jnp.ndarray],
    rng_key: jax.Array,
    use_causal_mask: bool,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Single evaluation step for autoregressive model (JIT-compilable).

    Args:
        params: Model parameters
        structure: Graph structure
        batch: Batch with 'x' and 'y'
        rng_key: Random key
        use_causal_mask: Whether to use causal masking

    Returns:
        Tuple of (output_cross_entropy, predictions)
    """
    batch_size = batch["x"].shape[0]
    seq_len = batch["x"].shape[1]

    # Create clamps (input only for evaluation)
    clamps = {structure.task_map["x"]: batch["x"]}

    # Add an external causal mask only if the graph defines one (v1). v2 masks
    # internally via the node's is_causal flag, so this is skipped there.
    if use_causal_mask and "causal_mask" in structure.task_map:
        causal_mask = create_causal_mask(seq_len)
        causal_mask = causal_mask[None, None, :, :]
        causal_mask = jnp.broadcast_to(causal_mask, (batch_size, 1, seq_len, seq_len))
        clamps[structure.task_map["causal_mask"]] = causal_mask

    rng_key, dropout_key, state_key = jax.random.split(rng_key, 3)
    clamps.update(
        create_dropout_clamps(structure, batch_size, dropout_key, training=False)
    )
    # Initialize and run inference
    state = initialize_graph_state(
        structure,
        batch_size,
        state_key,
        clamps=clamps,
        params=params,
    )
    final_state = run_inference(params, state, clamps, structure)

    # Compute loss and get predictions
    output_node = structure.task_map["y"]
    output_cross_entropy = compute_loss(
        final_state, batch["y"], output_node, loss_type="cross_entropy"
    )  # For perplexity metric
    predictions = final_state.nodes[output_node].z_mu

    return output_cross_entropy, predictions


def create_pmap_eval_step_autoregressive(
    structure: GraphStructure,
    use_causal_mask: bool,
):
    """Create a pmap'd evaluation step with static args in closure."""

    def step_fn(params, batch, rng_key):
        return _eval_step_autoregressive(
            params, structure, batch, rng_key, use_causal_mask
        )

    return jax.pmap(step_fn, axis_name="devices")


def evaluate_autoregressive(
    params: GraphParams,
    structure: GraphStructure,
    test_loader: Any,
    config: dict,
    rng_key: jax.Array,
    debug: bool = False,
    pmap_single_device: bool = False,
) -> Dict[str, float]:
    """
    Evaluate autoregressive model on test data.

    Automatically detects available devices and uses pmap for multi-device
    evaluation, or JIT for single-device.

    Computes:
    - Average loss (cross-entropy)
    - Perplexity (if cross-entropy loss)
    - Accuracy (next-token prediction)

    Args:
        params: Trained parameters
        structure: Graph structure
        test_loader: Test data loader
        config: Evaluation config (use_causal_mask)
        rng_key: Random key
        debug: If True, print detailed diagnostics for first batch
        pmap_single_device: If True, forces pmap path on single device

    Returns:
        Dictionary of metrics
    """
    use_causal_mask = config["use_causal_mask"]

    output_node = structure.task_map.get("y")
    if output_node is None:
        raise ValueError("Structure must have 'y' in task_map")

    try:
        num_batches_total = len(test_loader)
    except TypeError:
        raise TypeError("test_loader must support len()")

    n_devices = jax.device_count()
    use_pmap = (n_devices > 1) or pmap_single_device

    if use_pmap:
        epoch_key, rng_key = jax.random.split(rng_key)
        if n_devices > 1:
            device_keys = jax.random.split(epoch_key, n_devices)
        else:
            device_keys = jnp.expand_dims(epoch_key, axis=0)

        replicated_params = replicate_params(params, n_devices)
        batch_keys_per_device = jax.vmap(
            lambda k: jax.random.split(k, num_batches_total)
        )(device_keys)

        pmap_eval_step = create_pmap_eval_step_autoregressive(
            structure, use_causal_mask
        )

        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        num_batches = 0

        for batch_idx, batch_data in enumerate(test_loader):
            batch_key_for_step = batch_keys_per_device[:, batch_idx]
            batch = _convert_batch(batch_data)
            batch_size = next(iter(batch.values())).shape[0]

            if batch_size % n_devices != 0:
                pad_size = n_devices - (batch_size % n_devices)
                batch = {
                    k: jnp.concatenate(
                        [v, jnp.zeros((pad_size,) + v.shape[1:], dtype=v.dtype)]
                    )
                    for k, v in batch.items()
                }
            else:
                pad_size = 0

            batch_sharded = shard_batch(batch, n_devices)
            loss_sharded, preds_sharded = pmap_eval_step(
                replicated_params, batch_sharded, batch_key_for_step
            )

            padded_batch_size = batch_size + pad_size
            predictions = preds_sharded.reshape(
                padded_batch_size, *preds_sharded.shape[2:]
            )
            predictions = predictions[:batch_size]
            targets = batch["y"][:batch_size]

            if pad_size > 0:
                log_preds = jnp.log(predictions + 1e-10)
                if targets.ndim == 3:
                    batch_loss = -float(jnp.mean(jnp.sum(targets * log_preds, axis=-1)))
                else:
                    batch_loss = -float(
                        jnp.mean(
                            jnp.take_along_axis(
                                log_preds, targets[..., None], axis=-1
                            )
                        )
                    )
            else:
                batch_loss = float(jnp.mean(loss_sharded))

            total_loss += batch_loss

            pred_tokens = jnp.argmax(predictions, axis=-1)
            if targets.ndim == 3:
                target_tokens = jnp.argmax(targets, axis=-1)
            else:
                target_tokens = targets

            correct = jnp.sum(pred_tokens == target_tokens)
            total_correct += int(correct)
            total_tokens += int(batch_size * predictions.shape[1])
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        perplexity = float(jnp.exp(avg_loss))
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0

        return {
            "loss": avg_loss,
            "perplexity": perplexity,
            "accuracy": accuracy,
            "num_batches": num_batches,
        }

    else:
        batch_keys = jax.random.split(rng_key, num_batches_total)

        # JIT compile the evaluation step
        jit_eval_step = jax.jit(
            lambda p, b, k: _eval_step_autoregressive(
                p, structure, b, k, use_causal_mask
            )
        )

        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        num_batches = 0

        for batch_idx, batch_data in enumerate(test_loader):
            batch = _convert_batch(batch_data)
            batch_size = batch["x"].shape[0]

            # Run JIT-compiled evaluation step
            loss, predictions = jit_eval_step(params, batch, batch_keys[batch_idx])
            total_loss += float(loss)

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

            # Compute accuracy
            targets = batch["y"]
            pred_tokens = jnp.argmax(predictions, axis=-1)
            if targets.ndim == 3:
                target_tokens = jnp.argmax(targets, axis=-1)
            else:
                target_tokens = targets

            correct = jnp.sum(pred_tokens == target_tokens)
            total_correct += int(correct)
            total_tokens += int(batch_size * predictions.shape[1])
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        perplexity = float(jnp.exp(avg_loss))
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0

        return {
            "loss": avg_loss,
            "perplexity": perplexity,
            "accuracy": accuracy,
            "num_batches": num_batches,
        }
