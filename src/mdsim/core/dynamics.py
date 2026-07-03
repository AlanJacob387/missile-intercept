"""Acceleration terms: gravity, atmospheric drag, thrust."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from mdsim.core.config import Config

# Must equal reference.naive_sim.G. Parity with the oracle fails on the constant
# alone if these drift, so a gate test asserts they are equal.
G = 9.80665


@dataclass(frozen=True)
class PhysicsParams:
    """Everything the physics step needs, resolved from config once per run.

    Drag is off by default: Phase 0 is validated against a drag-free oracle, and
    `cd = 0` makes `drag` return zeros without touching the atmosphere model.
    """

    dt: float
    g: float = G
    cd: float = 0.0
    area_over_mass: float = 0.0
    rho0: float = 1.225
    scale_height_m: float = 8500.0
    threat_model: str = "ballistic"
    integrator: str = "semi_implicit"

    @classmethod
    def from_config(cls, config: Config) -> PhysicsParams:
        return cls(dt=config.sim.dt, integrator=config.sim.integrator)


def gravity(pos: Tensor, vel: Tensor, params: PhysicsParams) -> Tensor:
    """Constant gravity, -z. Returns [N, B, 3] for pos/vel of the same shape."""
    accel = torch.zeros_like(pos)
    accel[..., 2] = -params.g
    return accel


def drag(pos: Tensor, vel: Tensor, params: PhysicsParams) -> Tensor:
    """Quadratic drag against an exponential atmosphere. Returns [N, B, 3].

    Short-circuits to zeros when disabled: the speed norm below is not differentiable
    at v = 0, and there is no reason to evaluate an atmosphere the run does not use.
    """
    if params.cd == 0.0 or params.area_over_mass == 0.0:
        return torch.zeros_like(vel)

    altitude = pos[..., 2].clamp(min=0.0)
    rho = params.rho0 * torch.exp(-altitude / params.scale_height_m)
    speed = vel.norm(dim=-1, keepdim=True)
    coeff = 0.5 * params.cd * params.area_over_mass * rho.unsqueeze(-1)
    return -coeff * speed * vel


def total_accel(pos: Tensor, vel: Tensor, params: PhysicsParams) -> Tensor:
    """Sum of active acceleration terms, [N, B, 3]. Batched over envs and bodies."""
    return gravity(pos, vel, params) + drag(pos, vel, params)
