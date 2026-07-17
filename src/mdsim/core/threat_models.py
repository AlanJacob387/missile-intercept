"""Swappable threat dynamics models (ballistic, maneuvering, glide)."""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from mdsim.core.dynamics import PhysicsParams, total_accel

_UP = (0.0, 0.0, 1.0)


def _lateral_axis(vel: Tensor) -> Tensor:
    """Unit vector perpendicular to vel and to +z, [N, B, 3].

    Degenerate when vel is nearly parallel to +z (near-vertical flight): the cross
    product norm goes to zero and the direction is undefined. The norm is clamped
    away from zero so the divide stays finite; the direction is not otherwise fixed
    up for that case, since a body flying straight up has no well-defined "lateral".
    """
    up = torch.zeros_like(vel)
    up[..., 2] = 1.0
    cross = torch.cross(vel, up, dim=-1)
    norm = cross.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return cross / norm


def ballistic(pos: Tensor, vel: Tensor, params: PhysicsParams, t: Tensor) -> Tensor:
    """Unpowered flight: gravity plus drag, no control. Returns [N, B, 3].

    t is unused: ballistic flight has no time-dependent control term.
    """
    return total_accel(pos, vel, params)


def weave(pos: Tensor, vel: Tensor, params: PhysicsParams, t: Tensor) -> Tensor:
    """Ballistic flight plus a sinusoidal lateral divert. Returns [N, B, 3].

        lateral_accel = maneuver_accel_mps2 * sin(2*pi*t / maneuver_period_s)

    applied along the axis perpendicular to velocity and to +z, so the body jinks
    side to side across its own heading rather than diving or climbing.
    """
    t_b = t.reshape(-1, 1, 1)
    phase = (2.0 * torch.pi / params.maneuver_period_s) * t_b
    lateral_accel = params.maneuver_accel_mps2 * torch.sin(phase)
    return total_accel(pos, vel, params) + _lateral_axis(vel) * lateral_accel


def pop_up(pos: Tensor, vel: Tensor, params: PhysicsParams, t: Tensor) -> Tensor:
    """Ballistic flight plus a sinusoidal vertical divert. Returns [N, B, 3].

    Same envelope as weave, applied along +z instead of the lateral axis, so the
    body porpoises above and below its unperturbed descent.
    """
    t_b = t.reshape(-1, 1, 1)
    phase = (2.0 * torch.pi / params.maneuver_period_s) * t_b
    vertical_accel = params.maneuver_accel_mps2 * torch.sin(phase)
    vertical = torch.zeros_like(pos)
    vertical[..., 2] = 1.0
    return total_accel(pos, vel, params) + vertical * vertical_accel


def lateral_step(pos: Tensor, vel: Tensor, params: PhysicsParams, t: Tensor) -> Tensor:
    """Ballistic flight plus a bang-bang lateral divert. Returns [N, B, 3].

    Same lateral axis as weave, but the command is a square wave rather than a
    sinusoid: full maneuver_accel_mps2 one way, then the other, switching every
    half period. sign(0) == 0 at the switch instants, which is a measure-zero set
    in continuous time and immaterial to the integrated trajectory.
    """
    t_b = t.reshape(-1, 1, 1)
    phase = (2.0 * torch.pi / params.maneuver_period_s) * t_b
    lateral_accel = params.maneuver_accel_mps2 * torch.sign(torch.sin(phase))
    return total_accel(pos, vel, params) + _lateral_axis(vel) * lateral_accel


# Adding a new pattern is a new entry here, not an edit at the call site.
THREAT_MODELS: dict[str, Callable[[Tensor, Tensor, PhysicsParams, Tensor], Tensor]] = {
    "ballistic": ballistic,
    "weave": weave,
    "pop_up": pop_up,
    "lateral_step": lateral_step,
}


def get_threat_model(
    name: str,
) -> Callable[[Tensor, Tensor, PhysicsParams, Tensor], Tensor]:
    try:
        return THREAT_MODELS[name]
    except KeyError:
        known = ", ".join(sorted(THREAT_MODELS))
        raise ValueError(f"unknown threat model {name!r}; known: {known}") from None
