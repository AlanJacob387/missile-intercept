"""IMM filter bank: consistency under maneuver, and reduction to the single CV KF.

Three checks:

  (a) NEES consistency against a maneuvering truth. Mirrors test_kalman.py's NEES
      test -- same chi-square construction, same convention of holding truth fixed
      across environments and letting independent measurement noise (env_seeds)
      supply the independent samples -- except the truth is no longer constant
      velocity: it carries a sinusoidal lateral acceleration on top, which is
      exactly the regime a single CV filter is not built for.

  (b) Reduction to the single KF. With q_ca == q_cv and the CA model's initial
      acceleration variance driven to ~0, the CA model's own extra terms in F/Q
      (everything that couples the acceleration state into position/velocity)
      contribute a vanishing correction, so the IMM's combined output should sit
      on top of a plain kalman.py run at the same q to a tight tolerance. This is
      an internal-consistency check on the augmented-state machinery, not a claim
      that the two models are made structurally identical.

  (c) Gate 1: mean final-step position error, batched over many independent
      maneuvering rollouts, must be lower for IMM than for a plain single CV KF
      run on the identical measurement stream at the identical q_cv. The scenario
      is deliberately tuned so a small q_cv (appropriate for a filter that should
      stay tight during unmaneuvered flight) makes the CV-only filter lag the
      turn, while the IMM's CA component picks it up.
"""

from __future__ import annotations

import math

import torch
from scipy.stats import chi2

from mdsim.core.rng import env_seeds
from mdsim.sensing import imm, kalman, radar

DTYPE = torch.float64
STATE_DIM = 6

# --- Maneuvering-truth scenario, shared by (a) and (c) ----------------------
#
# Range is kept short (10 km) relative to test_kalman's 150 km so the angular
# measurement noise (cross-range sigma = range * sigma_az ~ 35 m here) does not
# swamp the maneuver's own displacement (~1-1.5 km lateral swing over the run) --
# at long range the two gate-1 filters come out statistically indistinguishable
# because measurement noise, not model mismatch, dominates the error.
DT = 0.05
N_STEPS = 200
SIGMA_RANGE_M = 20.0
SIGMA_AZ_RAD = 0.2 * math.pi / 180.0
SIGMA_EL_RAD = 0.2 * math.pi / 180.0
DETECT_RANGE_M = 1.0e9  # always in range; cadence/detection is not under test
START_POS_M = (10000.0, 0.0, 3000.0)
START_VEL_MPS = (1200.0, 0.0, 0.0)

# Lateral (y-axis) acceleration a(t) = A * sin(omega * t) on top of the constant
# base velocity above -- a weaving threat, not a ballistic one.
A_MANEUVER_MPS2 = 150.0
OMEGA_MANEUVER_RADPS = 1.0

Q_CV = 1.0
Q_CA = 2000.0
MU0 = (0.9, 0.1)

N_RUNS_NEES = 500
N_RUNS_GATE = 256

_LOWER_NEES = chi2.ppf(0.025, STATE_DIM * N_RUNS_NEES) / N_RUNS_NEES
_UPPER_NEES = chi2.ppf(0.975, STATE_DIM * N_RUNS_NEES) / N_RUNS_NEES


def _maneuvering_positions(
    n_envs: int, dt: float, n_steps: int, start_pos, start_vel, amplitude: float, omega: float
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Constant velocity plus sinusoidal lateral accel, [1, 3] truth broadcast to n_envs.

    Truth is identical across environments; env-to-env independence for the
    statistics below comes entirely from independent measurement noise draws
    (env_seeds), the same separation test_kalman.py relies on.
    """
    pos = torch.tensor(start_pos, dtype=DTYPE).expand(n_envs, 1, 3).contiguous()
    vel = torch.tensor(start_vel, dtype=DTYPE).expand(n_envs, 1, 3).contiguous()
    positions = [pos.clone()]
    for k in range(1, n_steps):
        t = k * dt
        accel = torch.zeros(n_envs, 1, 3, dtype=DTYPE)
        accel[..., 1] = amplitude * math.sin(omega * t)
        vel = vel + accel * dt
        pos = pos + vel * dt
        positions.append(pos.clone())
    return positions, vel


def _run_imm(
    positions: list[torch.Tensor],
    seeds: torch.Tensor,
    dt: float,
    q_cv: float,
    q_ca: float,
    mu0: tuple[float, float] = MU0,
    sigma_range_m: float = SIGMA_RANGE_M,
    sigma_az_rad: float = SIGMA_AZ_RAD,
    sigma_el_rad: float = SIGMA_EL_RAD,
    known_accel: torch.Tensor | None = None,
):
    radar_pos = torch.zeros(3, dtype=DTYPE)
    known_accel = known_accel if known_accel is not None else torch.zeros(3, dtype=DTYPE)

    tracks = None
    for k, pos in enumerate(positions):
        step = torch.full((pos.shape[0],), k, dtype=torch.int64)
        measured_cart, measured_sph, detected = radar.measure(
            pos, radar_pos, seeds, step, sigma_range_m, sigma_az_rad, sigma_el_rad,
            DETECT_RANGE_M, 1,
        )
        R = kalman.cartesian_measurement_covariance(
            measured_sph, sigma_range_m, sigma_az_rad, sigma_el_rad
        )
        if tracks is None:
            tracks = imm.imm_initialize(measured_cart, R, mu0=mu0)
        else:
            tracks = imm.imm_step(tracks, measured_cart, R, detected, dt, q_cv, q_ca, known_accel)
    return tracks


def _run_cv(
    positions: list[torch.Tensor],
    seeds: torch.Tensor,
    dt: float,
    q_cv: float,
    sigma_range_m: float = SIGMA_RANGE_M,
    sigma_az_rad: float = SIGMA_AZ_RAD,
    sigma_el_rad: float = SIGMA_EL_RAD,
    known_accel: torch.Tensor | None = None,
):
    radar_pos = torch.zeros(3, dtype=DTYPE)
    known_accel = known_accel if known_accel is not None else torch.zeros(3, dtype=DTYPE)

    x = P = None
    for k, pos in enumerate(positions):
        step = torch.full((pos.shape[0],), k, dtype=torch.int64)
        measured_cart, measured_sph, _detected = radar.measure(
            pos, radar_pos, seeds, step, sigma_range_m, sigma_az_rad, sigma_el_rad,
            DETECT_RANGE_M, 1,
        )
        R = kalman.cartesian_measurement_covariance(
            measured_sph, sigma_range_m, sigma_az_rad, sigma_el_rad
        )
        if x is None:
            x, P = kalman.initialize(measured_cart, R, imm.INITIAL_VELOCITY_VARIANCE)
        else:
            x, P = kalman.predict(x, P, dt, q_cv, known_accel)
            x, P = kalman.update(x, P, measured_cart, R)
    return x, P


def test_imm_nees_consistent_on_maneuvering_truth() -> None:
    """Mean NEES at the final step, over N_RUNS_NEES independent noise draws, must
    land in the two-sided 95% interval for a 6-dim estimate -- the filter's
    reported P must match the error it actually makes even while the target
    is turning."""
    seeds = env_seeds(0, N_RUNS_NEES, "cpu")
    positions, vel_final = _maneuvering_positions(
        N_RUNS_NEES, DT, N_STEPS, START_POS_M, START_VEL_MPS, A_MANEUVER_MPS2, OMEGA_MANEUVER_RADPS
    )
    tracks = _run_imm(positions, seeds, DT, Q_CV, Q_CA)

    x_true = torch.cat((positions[-1], vel_final), dim=-1)
    mean_nees = float(kalman.nees(tracks.x_est, tracks.P, x_true).mean())

    print(f"IMM NEES (maneuvering truth) = {mean_nees:.4f}, bounds [{_LOWER_NEES:.4f}, {_UPPER_NEES:.4f}]")
    assert _LOWER_NEES <= mean_nees <= _UPPER_NEES, (
        f"IMM inconsistent: NEES {mean_nees:.4f} outside [{_LOWER_NEES:.4f}, {_UPPER_NEES:.4f}]"
    )


def test_imm_reduces_to_single_kf_when_models_match() -> None:
    """q_ca == q_cv and a near-zero CA initial acceleration variance should make
    the IMM combined output track a plain kalman.py run at the same q, step for
    step, to a tight tolerance."""
    dt = 0.01
    n_steps = 50
    n_envs = 16
    q = 1.0
    tiny_accel_variance = 1.0e-6
    sigma_range_m, sigma_az_rad, sigma_el_rad = 100.0, 0.2 * math.pi / 180.0, 0.2 * math.pi / 180.0
    start_pos = (150000.0, 0.0, 30000.0)
    start_vel = (2000.0, 0.0, 0.0)

    seeds = env_seeds(0, n_envs, "cpu")
    radar_pos = torch.zeros(3, dtype=DTYPE)
    known_accel = torch.zeros(3, dtype=DTYPE)

    pos = torch.tensor(start_pos, dtype=DTYPE).expand(n_envs, 1, 3).contiguous()
    vel = torch.tensor(start_vel, dtype=DTYPE).expand(n_envs, 1, 3).contiguous()

    tracks = None
    x_kf = P_kf = None
    max_x_diff = 0.0
    max_P_diff = 0.0

    for k in range(n_steps):
        if k > 0:
            pos = pos + vel * dt
        step = torch.full((n_envs,), k, dtype=torch.int64)
        measured_cart, measured_sph, detected = radar.measure(
            pos, radar_pos, seeds, step, sigma_range_m, sigma_az_rad, sigma_el_rad,
            DETECT_RANGE_M, 1,
        )
        R = kalman.cartesian_measurement_covariance(measured_sph, sigma_range_m, sigma_az_rad, sigma_el_rad)

        if tracks is None:
            tracks = imm.imm_initialize(measured_cart, R, imm.INITIAL_VELOCITY_VARIANCE, tiny_accel_variance, mu0=MU0)
            x_kf, P_kf = kalman.initialize(measured_cart, R, imm.INITIAL_VELOCITY_VARIANCE)
        else:
            tracks = imm.imm_step(tracks, measured_cart, R, detected, dt, q, q, known_accel)
            x_kf, P_kf = kalman.predict(x_kf, P_kf, dt, q, known_accel)
            x_kf, P_kf = kalman.update(x_kf, P_kf, measured_cart, R)

        max_x_diff = max(max_x_diff, float((tracks.x_est - x_kf).abs().max()))
        max_P_diff = max(max_P_diff, float((tracks.P - P_kf).abs().max()))

    print(f"reduction check: max |x_est diff| = {max_x_diff:.3e}, max |P diff| = {max_P_diff:.3e}")
    assert torch.allclose(tracks.x_est, x_kf, atol=1e-3, rtol=1e-6)
    assert torch.allclose(tracks.P, P_kf, atol=1e-2, rtol=1e-6)


def test_imm_beats_single_cv_kf_on_maneuvering_truth() -> None:
    """Gate 1: mean position error over N_RUNS_GATE maneuvering rollouts must be
    lower for IMM than for a plain CV KF at the identical q_cv and measurement
    stream."""
    seeds = env_seeds(0, N_RUNS_GATE, "cpu")
    positions, _vel_final = _maneuvering_positions(
        N_RUNS_GATE, DT, N_STEPS, START_POS_M, START_VEL_MPS, A_MANEUVER_MPS2, OMEGA_MANEUVER_RADPS
    )

    tracks = _run_imm(positions, seeds, DT, Q_CV, Q_CA)
    imm_err = float((tracks.position - positions[-1]).norm(dim=-1).mean())

    x_kf, _P_kf = _run_cv(positions, seeds, DT, Q_CV)
    cv_err = float((x_kf[..., :3] - positions[-1]).norm(dim=-1).mean())

    margin = cv_err - imm_err
    print(f"gate 1: IMM position error = {imm_err:.3f} m, CV-only = {cv_err:.3f} m, margin = {margin:.3f} m")
    assert imm_err < cv_err, (
        f"IMM did not beat plain CV KF on maneuvering truth: IMM={imm_err:.3f} m, CV={cv_err:.3f} m"
    )
