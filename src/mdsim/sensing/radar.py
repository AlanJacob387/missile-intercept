"""Batched radar measurement model with range, azimuth, and elevation noise."""

from __future__ import annotations

import torch
from torch import Tensor

from mdsim.core.rng import normal

# Distinct RNG streams so the three noise sources cannot alias onto the same bits.
STREAM_RANGE = 101
STREAM_AZ = 102
STREAM_EL = 103


def to_spherical(rel_pos: Tensor) -> Tensor:
    """Cartesian offset from the radar to (range, azimuth, elevation), [..., 3].

    Azimuth is measured in the horizontal plane from +x toward +y; elevation is
    measured up from that plane. Range is clamped away from zero so the elevation
    quotient stays finite for a target at the radar.
    """
    x, y, z = rel_pos[..., 0], rel_pos[..., 1], rel_pos[..., 2]
    rng = rel_pos.norm(dim=-1).clamp(min=1e-6)
    az = torch.atan2(y, x)
    el = torch.asin((z / rng).clamp(-1.0, 1.0))
    return torch.stack((rng, az, el), dim=-1)


def to_cartesian(spherical: Tensor) -> Tensor:
    """Inverse of to_spherical, [..., 3]."""
    rng, az, el = spherical[..., 0], spherical[..., 1], spherical[..., 2]
    cos_el = torch.cos(el)
    return torch.stack(
        (rng * cos_el * torch.cos(az), rng * cos_el * torch.sin(az), rng * torch.sin(el)),
        dim=-1,
    )


def measure(
    truth_pos: Tensor,
    radar_pos: Tensor,
    seed: Tensor,
    step: Tensor,
    sigma_range_m: float,
    sigma_az_rad: float,
    sigma_el_rad: float,
    detect_range_m: float,
    period_steps: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Take a radar look at every target in every environment.

    Returns (measurement_cartesian [N, T, 3], measurement_spherical [N, T, 3],
    detected [N, T]). The Cartesian measurement is relative to the same origin as
    truth_pos; the spherical one is relative to the radar and is what the covariance
    transform needs.

    Noise is added in the sensor's own coordinates -- range and two angles -- which
    is why the resulting Cartesian error is range-dependent and not isotropic.

    Detection requires both that the target is inside detect_range_m and that this
    step falls on the radar's update cadence. Off-cadence steps return detected=False
    and the tracker coasts on its prediction.
    """
    rel = truth_pos - radar_pos
    true_spherical = to_spherical(rel)

    shape = (truth_pos.shape[1],)
    dtype = truth_pos.dtype
    noise = torch.stack(
        (
            normal(seed, step, STREAM_RANGE, shape, dtype) * sigma_range_m,
            normal(seed, step, STREAM_AZ, shape, dtype) * sigma_az_rad,
            normal(seed, step, STREAM_EL, shape, dtype) * sigma_el_rad,
        ),
        dim=-1,
    )
    measured_spherical = true_spherical + noise

    # step is a per-env tensor so the cadence test costs no host synchronization.
    on_cadence = ((step % period_steps) == 0).unsqueeze(-1)
    in_range = true_spherical[..., 0] <= detect_range_m
    detected = in_range & on_cadence

    measured_cartesian = to_cartesian(measured_spherical) + radar_pos
    return measured_cartesian, measured_spherical, detected
