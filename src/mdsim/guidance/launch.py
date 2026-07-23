"""Launch timing and lead-angle solution against a tracked threat.

Reads TrackState only. Nothing here sees truth.

should_launch can also perturb the initial heading with a small, seeded aim
error (aim_dispersion_rad) so that multiple rounds bound to the same threat at
the same battery do not fly byte-identical geometry. See its docstring for the
model and the reproducibility guarantee.
"""

from __future__ import annotations

import torch
from torch import Tensor

from mdsim.core.rng import normal
from mdsim.sensing.tracks import TrackState

EPS = 1e-9

# Distinct from radar.py's STREAM_RANGE/STREAM_AZ/STREAM_EL (101-103) and from
# test_batch_equivalence.py's local _POS_STREAM/_VEL_STREAM (11-12), so this
# draw cannot alias onto any other noise source sharing a (seed, step).
STREAM_AIM_DISPERSION = 301


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
    seed: Tensor | None = None,
    step_index: Tensor | None = None,
    aim_dispersion_rad: float = 0.0,
) -> tuple[Tensor, Tensor]:
    """Decide launch and give the initial velocity, ([N, I] bool, [N, I, 3]).

    Launches when a track is held, the predicted intercept point is inside the
    interceptor's range envelope, and this round has not already been committed.
    The interceptor leaves the rail at full speed pointed at the predicted intercept
    point; boost is not modelled in Phase 0.

    aim_dispersion_rad, if > 0, perturbs the unit heading with isotropic 3D
    Gaussian noise of per-axis standard deviation aim_dispersion_rad before
    renormalizing. This is a small-angle aim-error model, not a literal
    rotation: for small aim_dispersion_rad the perturbed heading's angular
    deviation from the true heading is approximately aim_dispersion_rad
    radians (exactly, for a unit vector plus small orthogonal noise). It gives
    every interceptor slot its own draw, so rounds bound to the same threat
    and launched on the same step -- byte-identical geometry otherwise -- fly
    divergent initial headings instead. seed and step_index are required
    whenever aim_dispersion_rad > 0; they identify the draw the same way
    mdsim.sensing.radar.measure's noise does, not any property of the threat.

    The default aim_dispersion_rad=0.0 skips the perturbation entirely and
    reproduces the previous behaviour byte for byte -- callers that do not
    pass seed, step_index or aim_dispersion_rad are unaffected.

    Noise is drawn on every call, for every slot, whether or not that slot
    launches this step. Nothing needs to be stored across steps: the noise a
    round ends up flying is simply whatever was drawn on the one step its
    `launch` entry is True, and that draw is a deterministic function of
    (seed, step_index, slot index) alone, so it is reproducible from those
    three values without persisting any dispersion state.
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

    if aim_dispersion_rad > 0.0:
        if seed is None or step_index is None:
            raise ValueError("aim_dispersion_rad > 0 requires seed and step_index")
        noise = normal(
            seed, step_index, STREAM_AIM_DISPERSION, (n_interceptors, 3), launch_pos.dtype
        )
        heading = heading + noise * aim_dispersion_rad
        heading = heading / heading.norm(dim=-1, keepdim=True).clamp(min=EPS)

    velocity = heading * interceptor_speed
    return launch, velocity
