"""Launch timing and lead-angle solution against a tracked threat.

Reads TrackState only. Nothing here sees truth.
"""

from __future__ import annotations

import torch
from torch import Tensor

from mdsim.sensing.tracks import TrackState

EPS = 1e-9


def predicted_intercept(
    track_pos: Tensor,
    track_vel: Tensor,
    launch_pos: Tensor,
    interceptor_speed: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Constant-velocity intercept solution, returning (point, time_to_go, feasible).

    Solves |p_rel + v_t * t| = s * t for the earliest positive t, which is the
    quadratic (|v_t|^2 - s^2) t^2 + 2 (p_rel . v_t) t + |p_rel|^2 = 0.

    Simplification, per Phase 0 scope: the threat is extrapolated at constant
    velocity and the interceptor is assumed to fly a straight line at constant speed.
    Neither is true -- the threat is under gravity and the interceptor accelerates
    from rest -- so this is a launch-decision heuristic and an initial heading, not a
    guidance solution. Proportional navigation absorbs the error from there.
    """
    p_rel = track_pos - launch_pos

    a = (track_vel * track_vel).sum(-1) - interceptor_speed**2
    b = 2.0 * (p_rel * track_vel).sum(-1)
    c = (p_rel * p_rel).sum(-1)

    discriminant = b * b - 4.0 * a * c
    real = discriminant >= 0.0
    sqrt_disc = torch.sqrt(discriminant.clamp(min=0.0))

    denom = torch.where(a.abs() < EPS, torch.full_like(a, EPS), 2.0 * a)
    t1 = (-b - sqrt_disc) / denom
    t2 = (-b + sqrt_disc) / denom

    big = torch.full_like(t1, float("inf"))
    t1_pos = torch.where(t1 > 0.0, t1, big)
    t2_pos = torch.where(t2 > 0.0, t2, big)
    t_go = torch.minimum(t1_pos, t2_pos)

    feasible = real & torch.isfinite(t_go)
    t_go = torch.where(feasible, t_go, torch.zeros_like(t_go))

    point = track_pos + track_vel * t_go.unsqueeze(-1)
    return point, t_go, feasible


def should_launch(
    tracks: TrackState,
    launch_pos: Tensor,
    already_committed: Tensor,
    interceptor_speed: float,
    envelope_min_m: float,
    envelope_max_m: float,
    assignment: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Decide launch and give the initial velocity, ([N, I] bool, [N, I, 3]).

    Launches when a track is held, the predicted intercept point is inside the
    interceptor's range envelope, and this round has not already been committed.
    The interceptor leaves the rail at full speed pointed at the predicted intercept
    point; boost is not modelled in Phase 0.
    """
    n_envs, n_interceptors, _ = launch_pos.shape
    if assignment is None:
        assignment = torch.zeros(
            (n_envs, n_interceptors), dtype=torch.int64, device=launch_pos.device
        )

    index = assignment.unsqueeze(-1).expand(-1, -1, 3)
    track_pos = torch.gather(tracks.position, 1, index)
    track_vel = torch.gather(tracks.velocity, 1, index)
    held = torch.gather(tracks.detected, 1, assignment)

    point, _, feasible = predicted_intercept(
        track_pos, track_vel, launch_pos, interceptor_speed
    )

    reach = (point - launch_pos).norm(dim=-1)
    in_envelope = (reach >= envelope_min_m) & (reach <= envelope_max_m)

    launch = held & feasible & in_envelope & ~already_committed

    heading = point - launch_pos
    heading = heading / heading.norm(dim=-1, keepdim=True).clamp(min=EPS)
    velocity = heading * interceptor_speed
    return launch, velocity
