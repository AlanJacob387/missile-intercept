"""Interceptor kinematics under lateral-g limit and speed cap."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from mdsim.core.config import InterceptorSpec
from mdsim.core.dynamics import G

EPS = 1e-9


@dataclass(frozen=True)
class InterceptorParams:
    """Flight limits pulled from one arsenal entry in interceptors.json."""

    speed_mps: float
    max_accel_mps2: float
    envelope_min_m: float
    envelope_max_m: float
    reaction_time_s: float

    @classmethod
    def from_spec(cls, spec: InterceptorSpec) -> InterceptorParams:
        return cls(
            speed_mps=spec.intercept_speed_mps,
            max_accel_mps2=spec.max_g * G,
            envelope_min_m=spec.envelope_range_km[0] * 1000.0,
            envelope_max_m=spec.envelope_range_km[1] * 1000.0,
            reaction_time_s=spec.reaction_time_s,
        )


def clamp_accel(accel_cmd: Tensor, max_accel_mps2: float) -> Tensor:
    """Limit command magnitude, [N, I, 3].

    The limit is on the vector norm, not per axis: a per-axis clamp would allow
    sqrt(3) times the rated g along a diagonal and would silently rotate the
    commanded direction.
    """
    norm = accel_cmd.norm(dim=-1, keepdim=True).clamp(min=EPS)
    scale = (max_accel_mps2 / norm).clamp(max=1.0)
    return accel_cmd * scale


def clamp_speed(vel: Tensor, max_speed_mps: float) -> Tensor:
    """Cap speed while preserving heading, [N, I, 3]."""
    norm = vel.norm(dim=-1, keepdim=True).clamp(min=EPS)
    scale = (max_speed_mps / norm).clamp(max=1.0)
    return vel * scale
