"""Scripted maneuvering threat models: jink geometry, and CV-KF degradation under them.

Part 1 checks the jink patterns land where their formulas say they should, propagated
directly with core.integrator against no other module. Part 2 checks the premise the
IMM tracker (built separately) is measured against: a plain constant-velocity Kalman
filter, tuned for a ballistic threat, tracks a maneuvering one worse.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import pytest
import torch

from mdsim.core.config import load_config
from mdsim.core.dynamics import G, PhysicsParams
from mdsim.core.integrator import integrate
from mdsim.core.threat_models import ballistic, get_threat_model, lateral_step, weave
from mdsim.core.rng import env_seeds
from mdsim.sensing import kalman, radar
from mdsim.world.scenario import launch_state

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
CONFIG = load_config(CONFIG_DIR)

DTYPE = torch.float64

# --- Part 1: jink geometry ------------------------------------------------------

DT = 0.05
PERIOD_S = 20.0
MANEUVER_ACCEL_MPS2 = 10.0 * G  # iskander_m's maneuver_g anchor, configs/arsenal/threats.json
N_STEPS_PERIOD = round(PERIOD_S / DT)

POS0 = (0.0, 0.0, 10_000.0)
VEL0 = (200.0, 0.0, 0.0)  # purely +x, so cross(vel, +z) is purely -y: a clean lateral axis


def _params(maneuver_accel: float = MANEUVER_ACCEL_MPS2) -> PhysicsParams:
    return PhysicsParams(
        dt=DT, beta=0.0, maneuver_accel_mps2=maneuver_accel, maneuver_period_s=PERIOD_S
    )


def _propagate(model_name: str, pos0, vel0, params: PhysicsParams, n_steps: int) -> torch.Tensor:
    """Single body, single env. Returns positions [n_steps + 1, 3]."""
    accel_fn = get_threat_model(model_name)
    pos = torch.tensor([[pos0]], dtype=DTYPE)
    vel = torch.tensor([[vel0]], dtype=DTYPE)
    t = torch.zeros(1, dtype=DTYPE)

    trace = [pos[0, 0].clone()]
    for _ in range(n_steps):
        accel = accel_fn(pos, vel, params, t)
        pos, vel = integrate(pos, vel, accel, params.dt, params.integrator)
        t = t + params.dt
        trace.append(pos[0, 0].clone())
    return torch.stack(trace)


@pytest.mark.parametrize("model_name", ["weave", "lateral_step"])
def test_lateral_maneuver_produces_sign_changing_displacement(model_name: str) -> None:
    """Over one period the lateral offset is nonzero and curves off a straight chord.

    The raw y(t) trace does not itself cross zero within one period -- both patterns
    start from rest and drift monotonically with the maneuver's net impulse. What
    "sign-changing" checks here is the wiggle relative to that drift: subtract the
    straight line from y(0) to y(T) and the residual must go both above and below it,
    which is the S-curve jink rather than a one-sided divert.
    """
    positions = _propagate(model_name, POS0, VEL0, _params(), N_STEPS_PERIOD)
    y = positions[:, 1]

    assert float(y.abs().max()) > 500.0, "lateral divert produced negligible displacement"

    chord = torch.linspace(float(y[0]), float(y[-1]), y.shape[0], dtype=DTYPE)
    residual = y - chord
    assert float(residual.min()) < -1.0, "no jink below the straight-line chord"
    assert float(residual.max()) > 1.0, "no jink above the straight-line chord"


def test_pop_up_produces_vertical_oscillation_beyond_ballistic_descent() -> None:
    params = _params()
    popup = _propagate("pop_up", POS0, VEL0, params, N_STEPS_PERIOD)
    baseline = _propagate("ballistic", POS0, VEL0, params, N_STEPS_PERIOD)

    delta = popup[:, 2] - baseline[:, 2]
    assert float(delta.abs().max()) > 500.0, "pop-up divert produced negligible altitude change"

    # Curvature (discrete second difference) must change sign: a body that porpoises
    # has an inflection, a body under a one-signed extra force does not.
    curvature = delta[2:] - 2.0 * delta[1:-1] + delta[:-2]
    assert float(curvature.min()) < -1e-3, "no downward inflection in the altitude offset"
    assert float(curvature.max()) > 1e-3, "no upward inflection in the altitude offset"


def test_lateral_step_stays_near_peak_accel_while_weave_passes_through_zero() -> None:
    """The distinguishing signature: bang-bang holds |a| near the peak, sine does not.

    Sampled at a fixed pos/vel across many phases of one period (not propagated, so
    the lateral axis is identical for every sample and only the phase varies).
    """
    params = _params()
    pos0 = torch.tensor(POS0, dtype=DTYPE)
    vel0 = torch.tensor(VEL0, dtype=DTYPE)

    # Even sample count: offsetting by half a sub-interval then keeps no sample
    # exactly on a sign-flip instant (t = 0, T/2, T), where sign(sin(.)) == 0 and
    # the two patterns' extra accel briefly coincide at zero.
    n_samples = 100
    fractions = (torch.arange(n_samples, dtype=DTYPE) + 0.5) / n_samples
    t = fractions * PERIOD_S

    pos = pos0.expand(n_samples, 1, 3).contiguous()
    vel = vel0.expand(n_samples, 1, 3).contiguous()

    weave_extra = weave(pos, vel, params, t) - ballistic(pos, vel, params, t)
    step_extra = lateral_step(pos, vel, params, t) - ballistic(pos, vel, params, t)

    weave_mag = weave_extra.norm(dim=-1).squeeze(-1)
    step_mag = step_extra.norm(dim=-1).squeeze(-1)

    assert float(weave_mag.min()) < 0.1 * MANEUVER_ACCEL_MPS2, "weave never nears zero accel"
    assert float(weave_mag.max()) > 0.9 * MANEUVER_ACCEL_MPS2, "weave never nears peak accel"
    assert float(step_mag.min()) > 0.9 * MANEUVER_ACCEL_MPS2, (
        "lateral_step accel dipped instead of holding near the peak"
    )


# --- Part 2: CV-KF degrades under maneuver --------------------------------------

N_RUNS = 100
N_STEPS = 1200  # 60 s at dt=0.05: three maneuver periods, well short of impact
VELOCITY_VARIANCE = 1.0e6
DETECT_RANGE_M = 1.0e9


def _terminal_tracking_error(config, threat_model_name: str, n_runs: int, n_steps: int) -> float:
    """Mean final-step position error, [3]-norm, of a plain CV Kalman filter.

    Truth is propagated here directly with core.integrator and the named threat
    model -- no engine.py involved. The filter only ever knows gravity (known_accel),
    exactly as kalman.predict's docstring describes; drag and any maneuver are both
    unmodelled, which is the whole point of this comparison.
    """
    physics = PhysicsParams.from_config(config)
    params = dataclasses.replace(physics, threat_model=threat_model_name)
    accel_fn = get_threat_model(threat_model_name)

    seeds = env_seeds(config.sim.seed, n_runs, "cpu")
    pos0, vel0 = launch_state(config)
    pos = torch.tensor(pos0, dtype=DTYPE).expand(n_runs, 1, 3).contiguous()
    vel = torch.tensor(vel0, dtype=DTYPE).expand(n_runs, 1, 3).contiguous()
    t = torch.zeros(n_runs, dtype=DTYPE)

    radar_pos = torch.tensor(config.scenario.battery_pos_m, dtype=DTYPE)
    sigma_range_m = config.radar.sigma_range_m
    sigma_az_rad = math.radians(config.radar.sigma_az_deg)
    sigma_el_rad = math.radians(config.radar.sigma_el_deg)
    known_accel = torch.tensor([0.0, 0.0, -params.g], dtype=DTYPE)

    x = P = None
    for k in range(n_steps):
        if k > 0:
            accel = accel_fn(pos, vel, params, t)
            pos, vel = integrate(pos, vel, accel, params.dt, params.integrator)
            t = t + params.dt

        step = torch.full((n_runs,), k, dtype=torch.int64)
        measured_cart, measured_sph, _ = radar.measure(
            pos,
            radar_pos,
            seeds,
            step,
            sigma_range_m,
            sigma_az_rad,
            sigma_el_rad,
            DETECT_RANGE_M,
            1,
        )
        R = kalman.cartesian_measurement_covariance(
            measured_sph, sigma_range_m, sigma_az_rad, sigma_el_rad
        )

        if x is None:
            x, P = kalman.initialize(measured_cart, R, VELOCITY_VARIANCE)
        else:
            x, P = kalman.predict(x, P, params.dt, config.sim.kf_process_noise_q, known_accel)
            x, P = kalman.update(x, P, measured_cart, R)

    error = (x[..., :3] - pos).norm(dim=-1)
    return float(error.mean())


def test_weave_degrades_cv_kalman_tracking_versus_ballistic() -> None:
    config = dataclasses.replace(
        CONFIG, scenario=dataclasses.replace(CONFIG.scenario, threat="iskander_m")
    )

    error_ballistic = _terminal_tracking_error(config, "ballistic", N_RUNS, N_STEPS)
    error_weave = _terminal_tracking_error(config, "weave", N_RUNS, N_STEPS)

    print(
        f"terminal position error (iskander_m, {N_STEPS * DT:.0f} s): "
        f"ballistic {error_ballistic:.1f} m, weave {error_weave:.1f} m"
    )
    assert error_weave > error_ballistic, (
        f"weave tracking error {error_weave:.1f} m not larger than "
        f"ballistic {error_ballistic:.1f} m at matched settings"
    )
