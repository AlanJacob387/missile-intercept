"""Fixed-timestep integrators (semi-implicit Euler, RK4) over batched tensors."""

from __future__ import annotations

import torch
from torch import Tensor


def step_semi_implicit(
    pos: Tensor, vel: Tensor, accel: Tensor, dt: float
) -> tuple[Tensor, Tensor]:
    """One semi-implicit Euler step. Returns new (pos, vel), inputs untouched.

    The oracle in reference/naive_sim.py updates velocity first, then advances
    position with the NEW velocity. That order is reproduced exactly here. Explicit
    Euler (position advanced with the OLD velocity) differs by O(dt) per step and
    accumulates well past the parity tolerance over a full trajectory.
    """
    new_vel = vel + accel * dt
    new_pos = pos + new_vel * dt
    return new_pos, new_vel


def step_rk4(
    pos: Tensor, vel: Tensor, accel_fn, dt: float
) -> tuple[Tensor, Tensor]:
    """Classical RK4. Takes an acceleration callable, not a fixed acceleration."""
    raise NotImplementedError("Phase 2")


def integrate(
    pos: Tensor, vel: Tensor, accel: Tensor, dt: float, integrator: str
) -> tuple[Tensor, Tensor]:
    """Dispatch on the config's integrator name."""
    if integrator == "semi_implicit":
        return step_semi_implicit(pos, vel, accel, dt)
    if integrator == "rk4":
        raise NotImplementedError("Phase 2")
    raise ValueError(f"unknown integrator: {integrator!r}")
