"""Batched closest-approach miss-distance test against the kill threshold."""

from __future__ import annotations

import torch
from torch import Tensor

EPS = 1e-12


def closest_approach(
    delta_pos: Tensor, delta_vel: Tensor, dt: float
) -> tuple[Tensor, Tensor]:
    """Minimum separation over one timestep, returning (distance, time_of_closest).

    Minimizes |delta_pos + delta_vel * tau| over tau in [0, dt]. The unconstrained
    minimum is at tau* = -(delta_pos . delta_vel) / |delta_vel|^2, clamped into the
    step.

    Testing only the endpoints would miss the entire engagement: at 6 km/s closing
    and dt = 0.05 s the bodies advance 300 m per step, so a pass well inside a 10 m
    kill radius can start and end hundreds of metres apart. That is the tunneling
    case, and it is the normal case here, not an edge case.
    """
    speed_sq = (delta_vel * delta_vel).sum(dim=-1).clamp(min=EPS)
    tau = (-(delta_pos * delta_vel).sum(dim=-1) / speed_sq).clamp(0.0, dt)
    separation = delta_pos + delta_vel * tau.unsqueeze(-1)
    return separation.norm(dim=-1), tau


def check_hits(
    threat_pos_before: Tensor,
    threat_pos_after: Tensor,
    interceptor_pos_before: Tensor,
    interceptor_pos_after: Tensor,
    threat_alive: Tensor,
    interceptor_alive: Tensor,
    dt: float,
    kill_radius_m: Tensor,
) -> tuple[Tensor, Tensor]:
    """Resolve every threat-interceptor pair over the step.

    kill_radius_m is per interceptor, [I] or broadcastable to [N, T, I]: lethal
    radius is a property of the warhead, so a hit-to-kill round and a
    blast-fragmentation round in the same battery do not share a threshold.

    Returns (hit [N, T, I], passed [N, T, I]). `passed` marks pairs whose closest
    approach fell strictly inside the step without killing -- the interceptor flew
    by and is spent, which is what ends a miss.

    Relative velocity is taken from the actual displacement over the step rather
    than from the instantaneous velocities, so the test matches the straight
    segments the integrator actually produced.
    """
    dp = interceptor_pos_before.unsqueeze(1) - threat_pos_before.unsqueeze(2)
    threat_step = (threat_pos_after - threat_pos_before).unsqueeze(2)
    interceptor_step = (interceptor_pos_after - interceptor_pos_before).unsqueeze(1)
    dv = (interceptor_step - threat_step) / dt

    distance, tau = closest_approach(dp, dv, dt)

    engaged = threat_alive.unsqueeze(2) & interceptor_alive.unsqueeze(1)
    hit = engaged & (distance <= kill_radius_m)
    passed = engaged & ~hit & (tau > 0.0) & (tau < dt)
    return hit, passed
