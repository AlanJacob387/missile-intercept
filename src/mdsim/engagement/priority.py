"""Threat scoring for weapon-target assignment, from track estimates only.

The defender does not know which city a threat is aimed at. It extrapolates the
track's own estimated state to the ground and takes the nearest surviving city as the
asset at risk. That inference is wrong sometimes, and being wrong sometimes is the
point -- a defender handed the true target list would never mis-prioritise.
"""

from __future__ import annotations

import torch
from torch import Tensor

from mdsim.sensing.tracks import TrackState

EPS = 1e-6

# Time-to-impact at which the flyout heuristic saturates, seconds. Above this there is
# comfortably enough time to launch, fly out and still manoeuvre; below it the
# achievable intercept geometry degrades roughly in proportion to the time left.
FLYOUT_REFERENCE_S = 60.0

# Ceiling on urgency so a track predicted to land essentially now cannot dominate the
# ordering by dividing by a near-zero interval.
MAX_URGENCY = 1.0


def time_to_impact(track_pos: Tensor, track_vel: Tensor, g: float) -> Tensor:
    """Seconds until the estimated state reaches z = 0, [N, T].

    Solves z + vz t - g t^2 / 2 = 0 for the positive root. Ballistic and drag-free:
    the defender is extrapolating a track, not running the threat's own dynamics, and
    it does not know the threat's ballistic coefficient.
    """
    height = track_pos[..., 2]
    vertical = track_vel[..., 2]

    discriminant = (vertical * vertical + 2.0 * g * height).clamp(min=0.0)
    root = (vertical + torch.sqrt(discriminant)) / g
    return root.clamp(min=0.0)


def predicted_impact_point(
    track_pos: Tensor, track_vel: Tensor, t_impact: Tensor
) -> Tensor:
    """Where the extrapolated track meets the ground, [N, T, 3]."""
    horizontal = track_pos[..., :2] + track_vel[..., :2] * t_impact.unsqueeze(-1)
    ground = torch.zeros_like(t_impact).unsqueeze(-1)
    return torch.cat((horizontal, ground), dim=-1)


def threatened_city(
    impact_point: Tensor, city_pos: Tensor, city_alive: Tensor
) -> tuple[Tensor, Tensor]:
    """Nearest surviving city to each predicted impact, ([N, T] index, [N, T] value mask).

    Returns the city index and whether any live city was found at all. A destroyed
    city is no longer worth defending, so it is excluded rather than merely
    deprioritised.
    """
    separation = (impact_point.unsqueeze(2) - city_pos.unsqueeze(1)).norm(dim=-1)

    unreachable = torch.full_like(separation, float("inf"))
    masked = torch.where(city_alive.unsqueeze(1), separation, unreachable)

    index = masked.argmin(dim=-1)
    any_live = city_alive.any(dim=-1, keepdim=True).expand(index.shape)
    return index, any_live


def flyout_confidence(t_impact: Tensor) -> Tensor:
    """Crude stand-in for kill probability, [N, T], in [0, 1].

    A geometry heuristic, not a calibrated probability and not fitted to measured Pk:
    it rises linearly with the time available to fly out and saturates at
    FLYOUT_REFERENCE_S. It exists so priority prefers threats the battery can
    plausibly still reach over ones it cannot. Phase 2 replaces it with the same
    degrading-Pk curve that replaces the hard envelope cutoff.
    """
    return (t_impact / FLYOUT_REFERENCE_S).clamp(0.0, 1.0)


def threat_priority(
    tracks: TrackState,
    city_pos: Tensor,
    city_value: Tensor,
    city_alive: Tensor,
    engageable: Tensor,
    g: float,
) -> tuple[Tensor, Tensor]:
    """Score every track for assignment, returning (priority [N, T], t_impact [N, T]).

    priority = asset value * flyout confidence * urgency.

    Urgency is the reciprocal of time-to-impact, not time-to-impact itself. The
    product as usually written is ambiguous, and only the reciprocal gives the
    ordering that matters: among two equally valuable threats the defender must shoot
    the one landing sooner, because the other can still be engaged afterwards.

    Priority is exactly zero for tracks that are not held or not engageable, so an
    undetected or out-of-envelope threat can never outrank a real candidate.
    """
    t_impact = time_to_impact(tracks.position, tracks.velocity, g)
    impact_point = predicted_impact_point(tracks.position, tracks.velocity, t_impact)

    city_index, any_live = threatened_city(impact_point, city_pos, city_alive)
    value = torch.gather(city_value, 1, city_index)

    urgency = (1.0 / (t_impact + EPS)).clamp(max=MAX_URGENCY)
    score = value * flyout_confidence(t_impact) * urgency

    live = tracks.detected & engageable & any_live
    return torch.where(live, score, torch.zeros_like(score)), t_impact
