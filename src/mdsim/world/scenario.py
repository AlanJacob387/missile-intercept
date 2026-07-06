"""Turn a scenario config into a burnout state. Speed is sized, never configured.

Two different sizing questions live here and they must not be confused:

`minimum_energy_speed_spherical` answers the physical one -- how fast does a body
actually have to be going to fly this far over a round Earth. It is the anchor the
class-band checks use, and at 13,000 km it returns ~7.6 km/s, which is what a real
ICBM burns out at.

`size_burnout_speed` answers the scenario-construction one -- how fast does it have to
be going for THIS simulation, whose gravity is flat and constant, to put it on the
battery. Those agree while the range is small against Earth's radius and diverge badly
beyond that: flat-Earth needs 11.3 km/s to reach 13,000 km, above escape velocity.
Scenario geometry uses the root-find so engagements actually connect; physics claims
use the closed form.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from mdsim.core.config import Config, ConfigError
from mdsim.core.dynamics import G

EARTH_RADIUS_M = 6_371_000.0
RHO0 = 1.225
SCALE_HEIGHT_M = 8500.0

# Sizing propagates at the engine's own dt. A coarser step is much faster but shifts
# the achieved range by ~2% at dt=0.1 against dt=0.05, which is larger than the 1%
# the root-find targets -- the sized speed would miss by more than the tolerance it
# was solved to.
SIZING_DT_S = 0.05
RANGE_TOLERANCE = 0.01
MAX_BISECTION_STEPS = 60


def minimum_energy_speed_spherical(range_km: float, g: float = G) -> float:
    """Minimum-energy ballistic speed over a spherical non-rotating Earth, m/s.

        v^2 = g R_e * 2 sin(phi/2) / (1 + sin(phi/2)),  phi = range / R_e

    As phi -> 0 this reduces to the flat-Earth sqrt(R g), so it agrees with the naive
    formula at short range and stays finite at intercontinental range where the naive
    one runs past escape velocity.

    Still idealized: non-rotating Earth, no drag, no boost, and the true minimum-energy
    launch angle (45 - phi/4 degrees) rather than any configured one.
    """
    if range_km <= 0.0:
        raise ValueError(f"range_km must be positive, got {range_km}")

    phi = (range_km * 1000.0) / EARTH_RADIUS_M
    half = math.sin(phi / 2.0)
    return math.sqrt(g * EARTH_RADIUS_M * (2.0 * half) / (1.0 + half))


def flat_earth_speed(range_km: float, elevation_deg: float, g: float = G) -> float:
    """Drag-free flat-Earth speed, v = sqrt(R g / sin 2 theta). Kept for comparison."""
    theta = math.radians(elevation_deg)
    return math.sqrt(range_km * 1000.0 * g / math.sin(2.0 * theta))


def _ground_range_m(
    speed_mps: float,
    elevation_deg: float,
    burnout_alt_m: float,
    beta: float,
    g: float = G,
    dt: float = SIZING_DT_S,
) -> float:
    """Downrange distance a dragged body covers from burnout to ground, metres.

    Semi-implicit Euler and the same exponential-atmosphere drag law the engine uses,
    so the sized speed lands where the engine will actually fly it.
    """
    theta = math.radians(elevation_deg)
    pos = np.array([0.0, 0.0, burnout_alt_m], dtype=np.float64)
    vel = np.array(
        [speed_mps * math.cos(theta), 0.0, speed_mps * math.sin(theta)], dtype=np.float64
    )

    gravity = np.array([0.0, 0.0, -g], dtype=np.float64)
    max_steps = int(2.0e6)

    for _ in range(max_steps):
        accel = gravity.copy()
        if beta > 0.0:
            rho = RHO0 * math.exp(-max(pos[2], 0.0) / SCALE_HEIGHT_M)
            accel -= (rho * float(np.linalg.norm(vel)) / (2.0 * beta)) * vel
        vel = vel + accel * dt
        new_pos = pos + vel * dt
        if new_pos[2] <= 0.0 < pos[2]:
            # Interpolate the ground crossing rather than overshooting by a step.
            frac = pos[2] / (pos[2] - new_pos[2])
            return float(pos[0] + frac * (new_pos[0] - pos[0]))
        pos = new_pos
    raise ConfigError("trajectory sizing did not reach the ground; check beta and speed")


@lru_cache(maxsize=None)
def size_burnout_speed(
    range_km: float,
    elevation_deg: float,
    burnout_alt_km: float,
    beta: float,
    entry_name: str = "<threat>",
    dt: float = SIZING_DT_S,
) -> float:
    """Burnout speed whose DRAGGED trajectory lands within 1% of range_km.

    Bisection, seeded from the spherical closed form. Ground range rises monotonically
    with burnout speed at fixed angle and beta, so a bracket is all bisection needs.

    Cached: this runs once per distinct scenario at build time, never in the step loop.
    """
    target_m = range_km * 1000.0
    burnout_alt_m = burnout_alt_km * 1000.0

    guess = minimum_energy_speed_spherical(range_km)
    low, high = guess * 0.25, guess * 1.5

    while _ground_range_m(low, elevation_deg, burnout_alt_m, beta, dt=dt) > target_m:
        low *= 0.5
        if low < 1.0:
            raise ConfigError(
                f"{entry_name}: cannot size a burnout speed for "
                f"{range_km} km -- even a near-zero speed overflies it from "
                f"{burnout_alt_km} km burnout altitude"
            )

    expansions = 0
    while _ground_range_m(high, elevation_deg, burnout_alt_m, beta, dt=dt) < target_m:
        high *= 1.5
        expansions += 1
        if expansions > 20:
            reached = (
                _ground_range_m(high, elevation_deg, burnout_alt_m, beta, dt=dt) / 1000.0
            )
            raise ConfigError(
                f"{entry_name}: {range_km} km is unreachable at beta={beta} kg/m^2 "
                f"from {burnout_alt_km} km burnout altitude and "
                f"{elevation_deg} deg -- drag caps the range near {reached:,.0f} km. "
                f"Raise beta, raise burnout_alt_km, or lower engagement_range_km."
            )

    for _ in range(MAX_BISECTION_STEPS):
        mid = 0.5 * (low + high)
        reached = _ground_range_m(mid, elevation_deg, burnout_alt_m, beta, dt=dt)
        if abs(reached - target_m) / target_m <= RANGE_TOLERANCE:
            return mid
        if reached < target_m:
            low = mid
        else:
            high = mid

    raise ConfigError(
        f"{entry_name}: burnout-speed bisection did not converge to "
        f"{RANGE_TOLERANCE:.0%} of {range_km} km"
    )


def launch_state(
    config: Config,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Threat burnout position and velocity, ((x, y, z), (vx, vy, vz)).

    The simulation starts at BURNOUT, not at the pad. Boost is not modelled: the
    segment covered here is the post-boost ballistic flight, which is the part
    midcourse and terminal interceptors actually engage. Starting at sea level with
    full ballistic speed instead would put the body under ~60 g of drag at t=0, which
    no missile experiences and which collapses the trajectory to a few km.

    A powered boost phase belongs in THREAT_MODELS as another entry later; it is not a
    rework of this.

    The body is placed one engagement range uprange of the battery at its burnout
    altitude, and its speed is sized so the dragged flight terminates on the battery.
    """
    scenario = config.scenario
    spec = config.threats[scenario.threat]

    speed = size_burnout_speed(
        scenario.engagement_range_km,
        scenario.launch_elevation_deg,
        spec.burnout_alt_km,
        spec.ballistic_coefficient_beta,
        spec.name,
        config.sim.dt,
    )
    theta = math.radians(scenario.launch_elevation_deg)

    battery_x, battery_y, battery_z = scenario.battery_pos_m
    position = (
        battery_x - scenario.engagement_range_km * 1000.0,
        battery_y,
        battery_z + spec.burnout_alt_km * 1000.0,
    )
    velocity = (speed * math.cos(theta), 0.0, speed * math.sin(theta))
    return position, velocity


def threat_beta(config: Config) -> float:
    """Ballistic coefficient of the scenario's threat, kg/m^2.

    One value because Phase 0 flies one threat type per scenario. Phase 1's mixed
    raids need this as a per-threat tensor.
    """
    return config.threats[config.scenario.threat].ballistic_coefficient_beta
