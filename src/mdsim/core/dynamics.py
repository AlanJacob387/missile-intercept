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

    beta is the ballistic coefficient in kg/m^2, m / (Cd A). It is the only drag
    parameter: mass, area and drag coefficient never appear separately because only
    their ratio affects the trajectory. beta = 0 disables drag entirely, which is
    what the drag-free ballistic parity checks use.
    """

    dt: float
    g: float = G
    beta: float = 0.0
    rho0: float = 1.225
    scale_height_m: float = 8500.0
    threat_model: str = "ballistic"
    integrator: str = "semi_implicit"

    @classmethod
    def from_config(cls, config: Config) -> PhysicsParams:
        spec = config.threats[config.scenario.threat]
        return cls(
            dt=config.sim.dt,
            beta=spec.ballistic_coefficient_beta,
            integrator=config.sim.integrator,
        )


def gravity(pos: Tensor, vel: Tensor, params: PhysicsParams) -> Tensor:
    """Constant gravity, -z. Returns [N, B, 3] for pos/vel of the same shape."""
    accel = torch.zeros_like(pos)
    accel[..., 2] = -params.g
    return accel


def drag(pos: Tensor, vel: Tensor, params: PhysicsParams) -> Tensor:
    """Quadratic drag against an exponential atmosphere. Returns [N, B, 3].

        a_drag = -(rho |v| / (2 beta)) v

    Short-circuits to zeros when beta is 0: the speed norm below is not
    differentiable at v = 0, and there is no reason to evaluate an atmosphere the run
    does not use.

    The atmosphere is a single exponential, rho0 exp(-h/H). That is a decent fit
    through the troposphere and stratosphere and increasingly poor above ~30 km,
    where it overstates density. It biases reentry deceleration high for anything
    entering steeply from exo-atmospheric altitudes.
    """
    if params.beta == 0.0:
        return torch.zeros_like(vel)

    altitude = pos[..., 2].clamp(min=0.0)
    rho = params.rho0 * torch.exp(-altitude / params.scale_height_m)
    speed = vel.norm(dim=-1, keepdim=True)
    return -(rho.unsqueeze(-1) * speed / (2.0 * params.beta)) * vel


def total_accel(pos: Tensor, vel: Tensor, params: PhysicsParams) -> Tensor:
    """Sum of active acceleration terms, [N, B, 3]. Batched over envs and bodies."""
    return gravity(pos, vel, params) + drag(pos, vel, params)
