"""3D proportional navigation guidance law.

Every function here reads TrackState and interceptor own-ship state. Truth tensors
are not an argument to anything in this module and must never become one -- the
defender flies on estimates or the simulation is telling itself the answer.
"""

from __future__ import annotations

import torch
from torch import Tensor

from mdsim.sensing.tracks import TrackState

EPS = 1e-6


def _gather_assigned(values: Tensor, assignment: Tensor) -> Tensor:
    """Pick each interceptor's assigned track. values [N, T, 3] -> [N, I, 3]."""
    index = assignment.unsqueeze(-1).expand(-1, -1, values.shape[-1])
    return torch.gather(values, 1, index)


def pn_accel(
    tracks: TrackState,
    interceptor_pos: Tensor,
    interceptor_vel: Tensor,
    n_gain: float,
    assignment: Tensor | None = None,
) -> Tensor:
    """True 3D proportional navigation command, [N, I, 3].

        a = N * V_c * (omega x u_los),  omega = (r x v_rel) / |r|^2,  V_c = -r_dot

    omega is the line-of-sight rotation vector, so the command is perpendicular to
    the line of sight and drives the LOS rate to zero -- a collision course. The
    command is zero for interceptors whose assigned track is not held.

    assignment [N, I] gives each interceptor its track index; it defaults to track 0,
    which is all Phase 0 needs. Phase 1 replaces the default with a WTA solution
    without changing this signature.
    """
    n_envs, n_interceptors, _ = interceptor_pos.shape
    if assignment is None:
        assignment = torch.zeros(
            (n_envs, n_interceptors), dtype=torch.int64, device=interceptor_pos.device
        )

    target_pos = _gather_assigned(tracks.position, assignment)
    target_vel = _gather_assigned(tracks.velocity, assignment)

    r = target_pos - interceptor_pos
    v_rel = target_vel - interceptor_vel

    r_sq = (r * r).sum(dim=-1, keepdim=True).clamp(min=EPS**2)
    r_norm = torch.sqrt(r_sq)
    u_los = r / r_norm

    omega = torch.cross(r, v_rel, dim=-1) / r_sq
    closing_speed = -(r * v_rel).sum(dim=-1, keepdim=True) / r_norm

    accel = n_gain * closing_speed * torch.cross(omega, u_los, dim=-1)

    held = torch.gather(tracks.detected, 1, assignment).unsqueeze(-1)
    return torch.where(held, accel, torch.zeros_like(accel))
