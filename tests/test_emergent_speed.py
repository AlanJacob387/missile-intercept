"""Terminal speed is emergent. Check it lands in each class's public Mach band.

This is the calibration instrument for `ballistic_coefficient_beta` and
`burnout_alt_km`, both of which are assumed. Nothing configures terminal speed any
more, so if a band fails the assumption behind it is wrong -- the band does not get
widened to accommodate it.

Speed is measured as the body DESCENDS through 10 km. "Mach" is ambiguous with
altitude, so the local speed of sound at 10 km is used throughout and stated here
rather than the sea-level value: at 10 km, a = 299.5 m/s against 340.3 m/s at sea
level, a 14% difference that would otherwise silently shift every Mach number.

Burnout speed comes from the SPHERICAL closed form, not from the scenario root-find.
The root-find solves "what speed makes this land on the battery in a flat-Earth
simulation", which is scenario geometry; at intercontinental range it returns
11.3 km/s, above escape velocity. The class band is a claim about real physics, so it
is anchored to real physics.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mdsim.core.config import load_config
from mdsim.core.dynamics import PhysicsParams
from mdsim.core.integrator import step_semi_implicit
from mdsim.core.threat_models import ballistic
from mdsim.world.scenario import minimum_energy_speed_spherical

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
CONFIG = load_config(CONFIG_DIR)

SPEED_OF_SOUND_10KM_MPS = 299.5
MEASURE_ALTITUDE_M = 10_000.0
ELEVATION_DEG = 45.0

# Public speed anchors for each class.
CLASS_MACH_BANDS = {
    "scud_b": (3.0, 8.0),
    "minuteman_iii": (15.0, 25.0),
}


def _fly_to_ground(speed_mps: float, burnout_alt_m: float, beta: float, dt: float):
    """Propagate one threat from burnout to impact using the engine's own physics.

    Returns (apogee_m, speed_at_measure_altitude_descending, impact_speed).
    """
    params = PhysicsParams(dt=dt, beta=beta)
    theta = torch.tensor(ELEVATION_DEG * torch.pi / 180.0, dtype=torch.float64)

    pos = torch.tensor([[[0.0, 0.0, burnout_alt_m]]], dtype=torch.float64)
    vel = torch.tensor(
        [[[speed_mps * float(torch.cos(theta)), 0.0, speed_mps * float(torch.sin(theta))]]],
        dtype=torch.float64,
    )

    apogee = burnout_alt_m
    measured = None
    ascending = True

    for _ in range(500_000):
        accel = ballistic(pos, vel, params)
        new_pos, new_vel = step_semi_implicit(pos, vel, accel, dt)

        z_old = float(pos[0, 0, 2])
        z_new = float(new_pos[0, 0, 2])
        apogee = max(apogee, z_new)
        if ascending and float(new_vel[0, 0, 2]) < 0.0:
            ascending = False
        if not ascending and measured is None and z_old >= MEASURE_ALTITUDE_M > z_new:
            measured = float(new_vel[0, 0].norm())

        pos, vel = new_pos, new_vel
        if z_new < 0.0 and not ascending:
            break

    return apogee, measured, float(vel[0, 0].norm())


def _measure(name: str):
    spec = CONFIG.threats[name]
    speed = minimum_energy_speed_spherical(spec.max_range_km)
    apogee, measured, impact = _fly_to_ground(
        speed,
        spec.burnout_alt_km * 1000.0,
        spec.ballistic_coefficient_beta,
        CONFIG.sim.dt,
    )
    assert measured is not None, f"{name} never descended through 10 km"
    return {
        "burnout_speed": speed,
        "apogee_km": apogee / 1000.0,
        "speed_10km": measured,
        "mach_10km": measured / SPEED_OF_SOUND_10KM_MPS,
        "impact_speed": impact,
    }


def test_spherical_sizing_is_physical_at_intercontinental_range() -> None:
    """13,000 km must size near 7 km/s, not the flat-Earth 11.3 km/s.

    The flat-Earth formula returns a burnout speed above Earth escape velocity at
    this range, which is the defect the spherical form exists to remove.
    """
    speed = minimum_energy_speed_spherical(13_000.0)
    assert 6_500.0 < speed < 8_500.0, f"got {speed:,.0f} m/s"
    assert speed < 11_186.0, "sized above escape velocity"


def test_spherical_matches_flat_earth_at_short_range() -> None:
    """The two forms must agree where curvature does not matter, or one is wrong."""
    from mdsim.world.scenario import flat_earth_speed

    spherical = minimum_energy_speed_spherical(300.0)
    flat = flat_earth_speed(300.0, 45.0)
    assert abs(spherical - flat) / flat < 0.02


def test_icbm_terminal_speed_in_class_band() -> None:
    """Minuteman III: ICBM reentry, Mach 15-25 at 10 km."""
    result = _measure("minuteman_iii")
    low, high = CLASS_MACH_BANDS["minuteman_iii"]
    print(
        f"minuteman_iii: burnout {result['burnout_speed']:,.0f} m/s, "
        f"apogee {result['apogee_km']:,.0f} km, "
        f"{result['speed_10km']:,.0f} m/s at 10 km = Mach {result['mach_10km']:.1f}"
    )
    assert low <= result["mach_10km"] <= high, (
        f"Mach {result['mach_10km']:.1f} outside [{low}, {high}]"
    )


def test_srbm_terminal_speed_in_class_band() -> None:
    """Scud-B: SRBM, Mach 3-8 at 10 km.

    Measured at beta = 4,000 kg/m^2: burnout 1,695 m/s at 25 km, apogee 86 km,
    993 m/s at 10 km, Mach 3.3. The band is cleared from below and with little
    margin, so this is the assertion that moves first if the atmosphere model or the
    burnout altitude changes. Beta 1,500 gives Mach 1.5 and 3,000 gives Mach 2.8.
    """
    result = _measure("scud_b")
    low, high = CLASS_MACH_BANDS["scud_b"]
    print(
        f"scud_b: burnout {result['burnout_speed']:,.0f} m/s, "
        f"apogee {result['apogee_km']:,.0f} km, "
        f"{result['speed_10km']:,.0f} m/s at 10 km = Mach {result['mach_10km']:.1f}"
    )
    assert low <= result["mach_10km"] <= high, (
        f"Mach {result['mach_10km']:.1f} outside [{low}, {high}]"
    )


def test_burnout_start_avoids_sea_level_drag_spike() -> None:
    """Guards the defect this initial condition exists to fix.

    Launching at sea level with full ballistic speed puts the body under ~60 g at
    t=0, which collapses a 150 km shot to about 7 km. Starting at burnout altitude
    must keep the initial drag deceleration at a small multiple of g.
    """
    from mdsim.core.dynamics import G, drag

    spec = CONFIG.threats["scud_b"]
    speed = minimum_energy_speed_spherical(spec.max_range_km)
    params = PhysicsParams(dt=CONFIG.sim.dt, beta=spec.ballistic_coefficient_beta)

    theta = torch.tensor(ELEVATION_DEG * torch.pi / 180.0, dtype=torch.float64)
    vel = torch.tensor(
        [[[speed * float(torch.cos(theta)), 0.0, speed * float(torch.sin(theta))]]],
        dtype=torch.float64,
    )

    at_burnout = torch.tensor(
        [[[0.0, 0.0, spec.burnout_alt_km * 1000.0]]], dtype=torch.float64
    )
    at_sea_level = torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float64)

    burnout_g = float(drag(at_burnout, vel, params).norm()) / G
    sea_level_g = float(drag(at_sea_level, vel, params).norm()) / G

    print(f"drag at burnout {burnout_g:.1f} g, at sea level {sea_level_g:.1f} g")
    assert burnout_g < 10.0, f"{burnout_g:.1f} g at burnout is still a spike"
    assert sea_level_g > 40.0, "sea-level comparison lost its meaning; recheck the model"
