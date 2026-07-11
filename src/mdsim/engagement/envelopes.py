"""Hard range and altitude cutoffs deciding which tracks a battery may engage.

Reads TrackState-derived positions and the battery's own location. No truth.
"""

from __future__ import annotations

from torch import Tensor

from mdsim.core.interceptor import InterceptorParams


def _as_broadcastable(battery_pos: Tensor) -> Tensor:
    return battery_pos.unsqueeze(1) if battery_pos.dim() == 2 else battery_pos


def envelope_weight(
    track_pos: Tensor, battery_pos: Tensor, params: InterceptorParams
) -> Tensor:
    """Engageability as a float weight in {0, 1}, [N, T].

    The weight is the seam a degrading-Pk curve slots into: Phase 2 replaces the
    step with a falloff toward the envelope edges and everything downstream keeps
    multiplying by the same quantity. Kept separate from `in_envelope` so that
    change touches one function rather than every caller.
    """
    offset = track_pos - _as_broadcastable(battery_pos)
    slant = offset.norm(dim=-1)
    altitude = track_pos[..., 2]

    inside = (
        (slant >= params.envelope_min_m)
        & (slant <= params.envelope_max_m)
        & (altitude >= params.envelope_alt_min_m)
        & (altitude <= params.envelope_alt_max_m)
    )
    return inside.to(track_pos.dtype)


def in_envelope(
    track_pos: Tensor, battery_pos: Tensor, params: InterceptorParams
) -> Tensor:
    """Whether each track sits inside the battery's engagement envelope, [N, T] bool.

    Slant range is measured from the battery; altitude is the track's z. Both bounds
    are hard cutoffs -- outside the band the battery cannot engage at all, which is
    the simpler of the two modelling choices and realistic enough while the layers
    are far apart.
    """
    return envelope_weight(track_pos, battery_pos, params) > 0.0
