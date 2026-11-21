# Inference Algorithms API

All inference algorithms extend `InferenceBase` from `fabricpc.core.inference`.

## Overview

Inference is the inner optimization loop of predictive coding. Given fixed weights and clamped data, it iteratively updates latent states to minimize total network energy.

Each inference step has three phases:
1. **Zero gradients** — Reset accumulated latent gradients
2. **Forward pass** — Compute predictions, errors, and accumulate gradient contributions
3. **Latent update** — Apply the algorithm-specific update rule to z_latent

## InferenceSGD

Standard gradient descent inference: `z -= eta * grad`

```python
from fabricpc.core.inference import InferenceSGD

inference = InferenceSGD(eta_infer=0.05, infer_steps=20, latent_decay=0.0)
structure = graph(..., inference=inference)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `eta_infer` | `float` | `0.1` | Inference rate |
| `infer_steps` | `int` | `20` | Number of inference iterations |
| `latent_decay` | `float` | `0.0` | Weight decay on latent states |

**Update rule:**
```
z_new = z * (1 - eta * latent_decay) - eta * latent_grad
```

## InferenceSGDNormClip

SGD inference with per-node gradient norm clipping.

```python
from fabricpc.core.inference import InferenceSGDNormClip

inference = InferenceSGDNormClip(
    eta_infer=0.1, infer_steps=20,
    max_norm=1.0, latent_decay=0.0, eps=1e-8,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `eta_infer` | `float` | `0.1` | Inference rate |
| `infer_steps` | `int` | `20` | Number of inference iterations |
| `latent_decay` | `float` | `0.0` | Weight decay on latent states |
| `max_norm` | `float` | `1.0` | Maximum L2 norm per node per sample |
| `eps` | `float` | `1e-8` | Numerical stability constant |

**Update rule:**
```
grad_norm = ||latent_grad||_2  (per sample)
clip_factor = min(1.0, max_norm / (grad_norm + eps))
clipped_grad = latent_grad * clip_factor
z_new = z * (1 - eta * latent_decay) - eta * clipped_grad
```

## InferencePCALM

Augmented Lagrangian predictive coding (PC-ALM; Seely & Gould 2026, arXiv 2605.31022). Each free node (not clamped, with at least one parent and one child) carries a dual `λ` on the constraint `z = μ`, and its energy becomes `E(z, μ) + Σ λ·(z − μ)` (for Gaussian energy with precision 1, `½‖z − μ + λ‖²` up to a constant). Each step takes a primal SGD step on the augmented energy, then a dual ascent step `λ += α·(z − μ)`. Duals start at zero for every inference run, and the last step is primal-only, so the weight update uses the duals of the final primal step.

```python
from fabricpc.core.inference import InferencePCALM

inference = InferencePCALM(
    eta_infer=0.1, infer_steps=20, alpha=1.0,
    dual_timing="sequential", max_norm=None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `eta_infer` | `float` | `0.1` | Inference rate |
| `infer_steps` | `int` | `20` | Number of inference iterations (≥ 1) |
| `alpha` | `float` | `1.0` | Dual ascent rate (≥ 0) |
| `dual_timing` | `str` | `"sequential"` | `"sequential"`: residual after the primal step (reference); `"simultaneous"`: residual from the step's own forward pass (Arrow–Hurwicz) |
| `latent_decay` | `float` | `0.0` | Weight decay on latent states |
| `max_norm` | `float` or `None` | `None` | Per-node per-sample L2 clip on the primal gradient, as in `InferenceSGDNormClip` |
| `eps` | `float` | `1e-8` | Numerical stability constant for clipping |

**Update rule** (free nodes; clamped and source nodes keep `λ = 0`):
```
latent_grad = dE_aug/dz        (gradient of the augmented energy)
z_new = z * (1 - eta * latent_decay) - eta * latent_grad   (clipped if max_norm is set)
λ_new = λ + alpha * (z_new - μ(z_new))   # sequential
λ_new = λ + alpha * (z - μ(z))           # simultaneous
```

Notes:
- `alpha = 0` is standard PC: identical to `InferenceSGD` (`max_norm=None`) or `InferenceSGDNormClip` (`max_norm` set).
- `infer_steps ≈ 2 × graph depth` is the paper's budget heuristic (T = 2L), not a guarantee: with `alpha = 1` the primal-dual dynamics can oscillate, so `infer_steps`, `alpha` and `eta_infer` may need tuning.
- At a fixed point the duals equal the negative backprop adjoints `−dL/dz` and the weight gradients equal backprop's. This requires hidden-node energies with `∂E/∂z = 0` at `z = μ` (Gaussian, Huber), not cross-entropy, Bernoulli or KL on free hidden nodes. The Gaussian form `½‖z − μ + λ‖²` holds for precision 1; with precision `p` it is `(p/2)‖z − μ + λ/p‖²`.
- Under muPC the dual path goes through `topdown_grad_scale`, which includes `jacobian_gain`: each node's adjoint and weight gradient is backprop's scaled by the product of the `jacobian_gain` factors of the nodes above it (e.g. `1.261^k` for tanh, `k` hops below the top free node). Directions per node are unchanged; the match is exact with identity/ReLU activations or without muPC.
- `"sequential"` costs one extra forward pass per step; `"simultaneous"` adds none.
- Reported node energies include the dual term.
- Custom nodes that override `energy_functional` must apply `add_dual_energy` (`fabricpc.nodes.base`); nodes with hand-written gradients must add `+λ` to `dE/dz` and `−λ` to `dE/dμ`.

## Tuning Guidance

| Parameter | Typical Range | Notes |
|-----------|:------------:|-------|
| `eta_infer` | 0.01–0.2 | Lower for stability, higher for faster convergence |
| `infer_steps` | 10–50 | More steps = better convergence, slower training |
| `latent_decay` | 0.0 | Rarely needed; try 0.001 if latents drift |
| `max_norm` | 0.5–2.0 | For InferenceSGDNormClip; prevents gradient explosions |

For deep networks (>10 layers), consider:
- Increasing `infer_steps` to `max(20, 4 * num_layers)`
- Using `InferenceSGDNormClip` for stability

## Creating Custom Inference Algorithms

Subclass `InferenceBase` and implement `compute_new_latent()`:

```python
from fabricpc.core.inference import InferenceBase
import jax.numpy as jnp

class InferenceMomentum(InferenceBase):
    def __init__(self, eta_infer=0.1, infer_steps=20, momentum=0.9):
        super().__init__(eta_infer=eta_infer, infer_steps=infer_steps, momentum=momentum)

    @staticmethod
    def compute_new_latent(node_name, node_state, config):
        eta = config["eta_infer"]
        momentum = config["momentum"]
        # Your custom update rule here
        # Example: add momentum tracking via node_state auxiliary fields
        return node_state.z_latent - eta * node_state.latent_grad
```

For more radical changes, override `inference_step()`, `forward_value_and_grad()`, or `run_inference()`. `run_inference()` calls `prepare_state()` once (the default drops any `dual`), then `inference_step()` `infer_steps - 1` times, then `final_inference_step()` (default: `inference_step()`).

## Convenience Function

```python
from fabricpc.core.inference import run_inference

# Run inference using the algorithm stored in structure.config["inference"]
final_state = run_inference(params, initial_state, clamps, structure)
```

This is a convenience wrapper that extracts the inference object from the graph structure and delegates to its `run_inference()` method.
