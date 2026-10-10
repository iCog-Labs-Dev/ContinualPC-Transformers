"""
Optimizer utilities using Optax.
"""

import optax

from fabricpc.training.natural_gradients import (
    scale_by_natural_gradient_diag,
    scale_by_natural_gradient_layerwise,
)

__all__ = [
    "build_warmup_cosine_adamw",
    "scale_by_natural_gradient_diag",
    "scale_by_natural_gradient_layerwise",
]


def build_warmup_cosine_adamw(
    *,
    learning_rate: float,
    total_steps: int,
    warmup_steps: int = 0,
    weight_decay: float = 0.0,
    min_learning_rate: float = 0.0,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    schedule_step_divisor: int = 1,
) -> optax.GradientTransformation:
    """Build AdamW with optional linear warmup and cosine decay.

    The schedule is evaluated at ``count // schedule_step_divisor``, so with a
    divisor of ``k`` it advances once every ``k`` optimizer updates and
    ``total_steps`` and ``warmup_steps`` count groups of ``k`` updates.
    """
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if min_learning_rate < 0:
        raise ValueError("min_learning_rate must be non-negative")
    if min_learning_rate > learning_rate:
        raise ValueError("min_learning_rate must be <= learning_rate")
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if warmup_steps > total_steps:
        raise ValueError("warmup_steps must be <= total_steps")
    if schedule_step_divisor < 1:
        raise ValueError("schedule_step_divisor must be >= 1")

    decay_steps = max(1, total_steps - warmup_steps)
    alpha = min_learning_rate / learning_rate
    cosine = optax.cosine_decay_schedule(
        init_value=learning_rate,
        decay_steps=decay_steps,
        alpha=alpha,
    )
    if warmup_steps == 0:
        schedule = cosine
    else:
        warmup = optax.linear_schedule(
            init_value=0.0,
            end_value=learning_rate,
            transition_steps=warmup_steps,
        )
        schedule = optax.join_schedules(
            schedules=[warmup, cosine],
            boundaries=[warmup_steps],
        )

    def learning_rate_schedule(count):
        return schedule(count // schedule_step_divisor)

    return optax.adamw(
        learning_rate=(
            schedule if schedule_step_divisor == 1 else learning_rate_schedule
        ),
        b1=b1,
        b2=b2,
        eps=eps,
        weight_decay=weight_decay,
    )
