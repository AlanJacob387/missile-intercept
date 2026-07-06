"""Batched Kalman filter estimating 3D threat position and velocity."""

from __future__ import annotations

import torch
from torch import Tensor


def transition(dt: float, dtype: torch.dtype, device: torch.device | str) -> Tensor:
    """Constant-velocity transition, [6, 6]."""
    F = torch.eye(6, dtype=dtype, device=device)
    F[0, 3] = dt
    F[1, 4] = dt
    F[2, 5] = dt
    return F


def process_noise(
    dt: float, q: float, dtype: torch.dtype, device: torch.device | str
) -> Tensor:
    """Continuous white-acceleration process noise discretized over dt, [6, 6].

    q is a spectral density in (m/s^2)^2/Hz. It stands in for everything the
    constant-velocity model omits -- for a ballistic threat that is gravity, so q
    must be large enough to cover g rather than tuned to sensor noise.
    """
    eye = torch.eye(3, dtype=dtype, device=device)
    Q = torch.zeros((6, 6), dtype=dtype, device=device)
    Q[:3, :3] = eye * (dt**4 / 4.0)
    Q[:3, 3:] = eye * (dt**3 / 2.0)
    Q[3:, :3] = eye * (dt**3 / 2.0)
    Q[3:, 3:] = eye * (dt**2)
    return Q * q


def measurement_matrix(dtype: torch.dtype, device: torch.device | str) -> Tensor:
    """Position-only observation, [3, 6]."""
    H = torch.zeros((3, 6), dtype=dtype, device=device)
    H[0, 0] = 1.0
    H[1, 1] = 1.0
    H[2, 2] = 1.0
    return H


def cartesian_measurement_covariance(
    measured_spherical: Tensor,
    sigma_range_m: float,
    sigma_az_rad: float,
    sigma_el_rad: float,
) -> Tensor:
    """Sensor covariance rotated into Cartesian, [..., 3, 3].

    The radar errs in range and two angles, so its Cartesian error ellipsoid is
    stretched across the line of sight by the range: the cross-range standard
    deviations are r*sigma_el and r*cos(el)*sigma_az. A fixed diagonal Cartesian R
    is wrong at every range but one, and badly wrong at long range where the angular
    terms dominate -- it would make the filter trust cross-range position far more
    than it should and drive the velocity estimate with angle noise.

    R_cart = J diag(sigma_r^2, sigma_az^2, sigma_el^2) J^T, with J the Jacobian of
    the spherical-to-Cartesian map. Linearization assumption: J is evaluated at the
    measured spherical point rather than the (unknown) true one, and the angular
    errors are small enough that the map is locally affine over the error ellipsoid.
    At 0.2 degrees and these ranges the resulting bias is far below the noise itself.
    """
    rng = measured_spherical[..., 0]
    az = measured_spherical[..., 1]
    el = measured_spherical[..., 2]

    cos_az, sin_az = torch.cos(az), torch.sin(az)
    cos_el, sin_el = torch.cos(el), torch.sin(el)

    zero = torch.zeros_like(rng)
    # Columns are d(x,y,z)/dr, d(x,y,z)/daz, d(x,y,z)/del.
    J = torch.stack(
        (
            torch.stack((cos_el * cos_az, -rng * cos_el * sin_az, -rng * sin_el * cos_az), dim=-1),
            torch.stack((cos_el * sin_az, rng * cos_el * cos_az, -rng * sin_el * sin_az), dim=-1),
            torch.stack((sin_el, zero, rng * cos_el), dim=-1),
        ),
        dim=-2,
    )

    variances = torch.stack(
        (
            torch.full_like(rng, sigma_range_m**2),
            torch.full_like(rng, sigma_az_rad**2),
            torch.full_like(rng, sigma_el_rad**2),
        ),
        dim=-1,
    )
    R_spherical = torch.diag_embed(variances)
    return J @ R_spherical @ J.transpose(-1, -2)


def predict(
    x: Tensor,
    P: Tensor,
    dt: float,
    q: float,
    known_accel: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Constant-velocity predict. x [N, T, 6], P [N, T, 6, 6].

    known_accel [3] is a deterministic acceleration the tracker is entitled to model
    -- gravity. It enters as a control input, not as state, so the filter stays
    linear and the covariance recursion is untouched.

    Leaving it None gives the pure constant-velocity model. That model has to absorb
    gravity through q instead, and q large enough to do so stops the filter smoothing
    angular noise, which at long range is the dominant error. Withholding gravity
    also models a defender that does not know Newton, which is not the intended
    uncertainty. Target *maneuver* remains unmodelled and is what q is for; the IMM
    upgrade in Phase 2 is aimed at that, not at this.
    """
    F = transition(dt, x.dtype, x.device)
    Q = process_noise(dt, q, x.dtype, x.device)

    x_pred = x @ F.transpose(-1, -2)
    if known_accel is not None:
        # Same semi-implicit convention the truth integrator uses: velocity first,
        # then position advanced with the updated velocity.
        delta_v = known_accel * dt
        x_pred = x_pred + torch.cat((delta_v * dt, delta_v), dim=-1)

    P_pred = F @ P @ F.transpose(-1, -2) + Q
    return x_pred, P_pred


def update(x: Tensor, P: Tensor, z: Tensor, R: Tensor) -> tuple[Tensor, Tensor]:
    """Standard KF update against a position measurement.

    The gain comes from a linear solve against the innovation covariance, never an
    explicit inverse. Covariance uses the Joseph form, which stays symmetric positive
    semidefinite under float32 rounding where (I - KH)P does not.
    """
    H = measurement_matrix(x.dtype, x.device)
    Ht = H.transpose(-1, -2)

    innovation = z - x @ Ht
    PHt = P @ Ht
    S = H @ PHt + R

    # K = PHt S^-1, obtained as (S^-T PHt^T)^T. S is symmetric, so S^-T == S^-1.
    K = torch.linalg.solve(S, PHt.transpose(-1, -2)).transpose(-1, -2)

    x_new = x + (K @ innovation.unsqueeze(-1)).squeeze(-1)

    eye = torch.eye(6, dtype=x.dtype, device=x.device)
    A = eye - K @ H
    P_new = A @ P @ A.transpose(-1, -2) + K @ R @ K.transpose(-1, -2)
    return x_new, P_new


def initialize(z: Tensor, R: Tensor, velocity_variance: float) -> tuple[Tensor, Tensor]:
    """Start a track from one measurement: position = z, velocity = 0, wide on velocity.

    A single position fix says nothing about velocity, so the velocity block is given
    a deliberately large variance and the filter is left to earn it from later looks.
    """
    zeros = torch.zeros_like(z)
    x = torch.cat((z, zeros), dim=-1)

    P = torch.zeros((*z.shape[:-1], 6, 6), dtype=z.dtype, device=z.device)
    P[..., :3, :3] = R
    eye = torch.eye(3, dtype=z.dtype, device=z.device)
    P[..., 3:, 3:] = eye * velocity_variance
    return x, P


def nees(x_est: Tensor, P: Tensor, x_true: Tensor) -> Tensor:
    """Normalized estimation error squared, [...]. Consistency diagnostic.

    A filter whose covariance is honest produces NEES with mean equal to the state
    dimension. Systematically low means it is over-reporting uncertainty; high means
    it is over-confident, which is what a wrong measurement covariance looks like.
    """
    error = (x_true - x_est).unsqueeze(-1)
    solved = torch.linalg.solve(P, error)
    return (error.transpose(-1, -2) @ solved).squeeze(-1).squeeze(-1)
