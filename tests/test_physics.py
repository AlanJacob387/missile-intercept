"""Analytic checks on the reference oracle: drag-free projectile range and flight time."""

from __future__ import annotations

import numpy as np
import pytest

from reference.naive_sim import G, propagate

DT = 0.05
SPEED_MPS = 1213.0

# Semi-implicit Euler lands one step early on a drag-free arc (the position error is
# g*dt*t/2), so relative error is ~dt/T. At these speeds T is 100 s+ and 1% is loose.
TOLERANCE = 0.01


def _ground_crossing(positions: np.ndarray, dt: float) -> tuple[float, float]:
    """Return (flight_time_s, horizontal_range_m) at the first descent through z=0."""
    z = positions[:, 2]
    below = np.flatnonzero(z < 0.0)
    if below.size == 0:
        raise AssertionError("trajectory never returned to the ground")

    k = int(below[0])
    z_hi, z_lo = z[k - 1], z[k]
    frac = z_hi / (z_hi - z_lo)  # linear interpolation between bracketing samples

    flight_time = (k - 1 + frac) * dt
    xy_hi, xy_lo = positions[k - 1, :2], positions[k, :2]
    impact_xy = xy_hi + frac * (xy_lo - xy_hi)
    return flight_time, float(np.linalg.norm(impact_xy - positions[0, :2]))


@pytest.mark.parametrize("theta_deg", [30.0, 45.0, 60.0])
def test_projectile_range_and_flight_time(theta_deg: float) -> None:
    theta = np.radians(theta_deg)
    vel0 = [SPEED_MPS * np.cos(theta), 0.0, SPEED_MPS * np.sin(theta)]

    expected_time = 2.0 * SPEED_MPS * np.sin(theta) / G
    expected_range = SPEED_MPS**2 * np.sin(2.0 * theta) / G

    n_steps = int(np.ceil((expected_time + 1.0) / DT))
    positions, _ = propagate([0.0, 0.0, 0.0], vel0, DT, n_steps)
    flight_time, horizontal_range = _ground_crossing(positions, DT)

    assert abs(flight_time - expected_time) / expected_time < TOLERANCE
    assert abs(horizontal_range - expected_range) / expected_range < TOLERANCE
