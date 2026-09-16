from __future__ import annotations

import math

import jax
import optax


def test_warmup_cosine_uses_full_schedule_length() -> None:
    lr = 1.0
    end_ratio = 0.1
    warmup_steps = 10
    total_steps = 100

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=lr * 0.05,
        peak_value=lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=lr * end_ratio,
    )

    step_90 = float(jax.device_get(schedule(90)))
    step_99 = float(jax.device_get(schedule(99)))
    step_100 = float(jax.device_get(schedule(100)))

    assert step_90 > lr * end_ratio
    assert step_99 > lr * end_ratio
    assert math.isclose(
        step_100,
        lr * end_ratio,
        rel_tol=0.0,
        abs_tol=1e-6,
    )


def test_warmup_cosine_reaches_peak_after_warmup() -> None:
    lr = 1.0
    warmup_steps = 10
    total_steps = 100

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=lr * 0.05,
        peak_value=lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=lr * 0.1,
    )

    step_warmup = float(jax.device_get(schedule(warmup_steps)))

    assert math.isclose(
        step_warmup,
        lr,
        rel_tol=0.0,
        abs_tol=1e-6,
    )


def main() -> None:
    test_warmup_cosine_uses_full_schedule_length()
    test_warmup_cosine_reaches_peak_after_warmup()
    print("PASS: warmup cosine scheduler regression tests")


if __name__ == "__main__":
    main()
