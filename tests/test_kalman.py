"""Filter consistency by NEES. This is the test that catches a wrong R transform.

A Kalman filter is consistent when its reported covariance matches the error it
actually makes. NEES measures exactly that, and it is the only cheap check that
notices a measurement covariance which is the wrong SHAPE rather than the wrong
size -- which is precisely the failure mode of a fixed diagonal Cartesian R.

The scenario is deliberately long-ranged. At 150 km a 0.2 degree angular error is
524 m of cross-range uncertainty against 100 m down-range, so the true error
ellipsoid is a 5:1 pancake lying across the line of sight. A filter told that its
errors are an isotropic 100 m sphere is over-confident across the LOS by a factor
of ~27 in variance, and NEES sees that immediately.
"""

from __future__ import annotations

import math

import pytest
import torch
from scipy.stats import chi2

from mdsim.core.rng import env_seeds, normal
from mdsim.sensing import kalman, radar

DTYPE = torch.float64
N_RUNS = 500
N_STEPS = 200
STATE_DIM = 6
DT = 0.05

SIGMA_RANGE_M = 100.0
SIGMA_AZ_RAD = 0.2 * math.pi / 180.0
SIGMA_EL_RAD = 0.2 * math.pi / 180.0
DETECT_RANGE_M = 1.0e9  # everything is in range; cadence and detection are not under test

# The truth process is exactly constant velocity, so the filter's process noise must
# be ~0 or it would inflate P beyond the error it actually makes and read as
# inconsistent for a reason that has nothing to do with R.
Q_NEGLIGIBLE = 1.0e-9

# The filter opens each track with velocity 0 and this variance. For NEES to mean
# anything, the truth must actually be drawn from that prior, so each env gets a
# true velocity sampled from N(0, VELOCITY_VARIANCE) per axis.
VELOCITY_VARIANCE = 1.0e6
VELOCITY_SIGMA = math.sqrt(VELOCITY_VARIANCE)

START_POS_M = (150000.0, 0.0, 30000.0)
_TRUE_VELOCITY_STREAM = 777

# Two-sided 95% interval on the mean of M chi-square(6) variables.
_LOWER = chi2.ppf(0.025, STATE_DIM * N_RUNS) / N_RUNS
_UPPER = chi2.ppf(0.975, STATE_DIM * N_RUNS) / N_RUNS


def _true_states() -> tuple[torch.Tensor, torch.Tensor]:
    """Common truth for both variants: fixed start, per-env velocity from the prior."""
    seeds = env_seeds(0, N_RUNS, "cpu")
    pos = torch.tensor(START_POS_M, dtype=DTYPE).expand(N_RUNS, 1, 3).contiguous()
    vel = normal(seeds, 0, _TRUE_VELOCITY_STREAM, (1, 3), DTYPE) * VELOCITY_SIGMA
    return pos, vel


def _mean_nees(transformed_r: bool) -> float:
    """Average NEES at the final step over N_RUNS independent environments."""
    seeds = env_seeds(0, N_RUNS, "cpu")
    radar_pos = torch.zeros(3, dtype=DTYPE)
    pos, vel = _true_states()

    wrong_r = (
        torch.eye(3, dtype=DTYPE).expand(N_RUNS, 1, 3, 3).contiguous()
        * SIGMA_RANGE_M**2
    )

    x = P = None
    for k in range(N_STEPS):
        if k > 0:
            pos = pos + vel * DT

        step = torch.full((N_RUNS,), k, dtype=torch.int64)
        measured_cart, measured_sph, _ = radar.measure(
            pos,
            radar_pos,
            seeds,
            step,
            SIGMA_RANGE_M,
            SIGMA_AZ_RAD,
            SIGMA_EL_RAD,
            DETECT_RANGE_M,
            1,
        )

        if transformed_r:
            R = kalman.cartesian_measurement_covariance(
                measured_sph, SIGMA_RANGE_M, SIGMA_AZ_RAD, SIGMA_EL_RAD
            )
        else:
            R = wrong_r

        if x is None:
            x, P = kalman.initialize(measured_cart, R, VELOCITY_VARIANCE)
        else:
            x, P = kalman.predict(x, P, DT, Q_NEGLIGIBLE)
            x, P = kalman.update(x, P, measured_cart, R)

    x_true = torch.cat((pos, vel), dim=-1)
    return float(kalman.nees(x, P, x_true).mean())


def test_chi_square_bounds_are_tight() -> None:
    """Sanity on the interval itself: it must bracket the state dimension closely."""
    assert _LOWER < STATE_DIM < _UPPER
    assert _UPPER - _LOWER < 1.0


def test_nees_consistent_with_transformed_r() -> None:
    """With the Jacobian-transformed R the filter must report honest covariance."""
    mean_nees = _mean_nees(transformed_r=True)
    print(f"NEES (transformed R) = {mean_nees:.4f}, bounds [{_LOWER:.4f}, {_UPPER:.4f}]")
    assert _LOWER <= mean_nees <= _UPPER, (
        f"filter inconsistent: NEES {mean_nees:.4f} outside [{_LOWER:.4f}, {_UPPER:.4f}]"
    )


def test_nees_rejects_untransformed_r() -> None:
    """The teeth: a fixed diagonal Cartesian R must be caught by the same check.

    Without this, a NEES test that passes proves nothing -- it could be insensitive
    rather than correct.
    """
    mean_nees = _mean_nees(transformed_r=False)
    print(f"NEES (fixed diagonal R) = {mean_nees:.4f}, bounds [{_LOWER:.4f}, {_UPPER:.4f}]")
    assert not (_LOWER <= mean_nees <= _UPPER), (
        f"NEES {mean_nees:.4f} accepted an untransformed R, so the test has no teeth"
    )
    # Direction matters: a wrong R here makes the filter over-confident, not under.
    assert mean_nees > _UPPER


def test_measurement_covariance_is_range_dependent() -> None:
    """Cross-range variance must grow with range; that is the whole point of J."""
    near = torch.tensor([[[10000.0, 0.0, 0.0]]], dtype=DTYPE)
    far = torch.tensor([[[200000.0, 0.0, 0.0]]], dtype=DTYPE)

    R_near = kalman.cartesian_measurement_covariance(
        near, SIGMA_RANGE_M, SIGMA_AZ_RAD, SIGMA_EL_RAD
    )
    R_far = kalman.cartesian_measurement_covariance(
        far, SIGMA_RANGE_M, SIGMA_AZ_RAD, SIGMA_EL_RAD
    )

    # Along the LOS (+x here) the variance is the range sigma at any distance.
    assert R_near[0, 0, 0, 0] == pytest.approx(SIGMA_RANGE_M**2, rel=1e-9)
    assert R_far[0, 0, 0, 0] == pytest.approx(SIGMA_RANGE_M**2, rel=1e-9)

    # Across the LOS it scales with range squared.
    ratio = float(R_far[0, 0, 1, 1] / R_near[0, 0, 1, 1])
    assert ratio == pytest.approx(20.0**2, rel=1e-9)
    assert float(R_far[0, 0, 1, 1]) > float(R_far[0, 0, 0, 0])
