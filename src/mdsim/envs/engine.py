"""Batched engine step orchestrating physics, sensing, guidance, and intercept.

`step` is the torch.compile target; it stays free of host-side control flow over
environments so the whole step compiles to device work. Compilation is applied in a
later task -- eager first, once parity with the oracle holds.

Order within a tick: threat physics, radar look, tracker predict and update, launch
decision, guidance, interceptor motion, intercept resolution, impact scoring. The
tracker sees the post-move truth, and guidance sees only what the tracker produced.

Every interceptor still defaults to track slot 0 -- there is no weapon-target
assignment yet, so a raid with several threats is only ever contested by whichever
interceptor slots exist, all aimed at the same track.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import Tensor

from mdsim.core import intercept as intercept_mod
from mdsim.core import interceptor as interceptor_mod
from mdsim.core.config import Config
from mdsim.core.dynamics import PhysicsParams
from mdsim.core.integrator import integrate
from mdsim.core.state import EnvState
from mdsim.core.threat_models import get_threat_model
from mdsim.guidance.launch import should_launch
from mdsim.guidance.pro_nav import pn_accel
from mdsim.sensing import kalman, radar
from mdsim.sensing.tracks import TrackState
from mdsim.world import damage

# A single position fix carries no velocity information, so the track opens with a
# deliberately wide velocity block, in (m/s)^2.
INITIAL_VELOCITY_VARIANCE = 1.0e6


@dataclass(frozen=True)
class EngineParams:
    """Everything one tick needs, resolved from config once per run."""

    physics: PhysicsParams
    radar_pos: tuple[float, float, float]
    sigma_range_m: float
    sigma_az_rad: float
    sigma_el_rad: float
    detect_range_m: float
    radar_period_steps: int
    kf_q: float
    pn_gain: float
    interceptor: interceptor_mod.InterceptorParams
    city_impact_radius_m: float
    engage: bool = True

    @classmethod
    def from_config(
        cls, config: Config, engage: bool = True, noise_scale: float = 1.0
    ) -> EngineParams:
        """noise_scale multiplies every radar sigma, for the Pk-vs-noise sweep."""
        dt = config.sim.dt
        radar_cfg = config.radar
        spec = config.interceptors[config.scenario.interceptor]

        period = max(1, round(1.0 / (radar_cfg.update_hz * dt)))
        deg = torch.pi / 180.0

        return cls(
            physics=PhysicsParams.from_config(config),
            radar_pos=config.scenario.battery_pos_m,
            sigma_range_m=radar_cfg.sigma_range_m * noise_scale,
            sigma_az_rad=radar_cfg.sigma_az_deg * deg * noise_scale,
            sigma_el_rad=radar_cfg.sigma_el_deg * deg * noise_scale,
            detect_range_m=radar_cfg.detect_range_km * 1000.0,
            radar_period_steps=period,
            kf_q=config.sim.kf_process_noise_q,
            pn_gain=config.sim.pn_gain,
            interceptor=interceptor_mod.InterceptorParams.from_spec(spec),
            city_impact_radius_m=config.sim.city_impact_radius_m,
            engage=engage,
        )


def _advance_threats(state: EnvState, params: EngineParams) -> tuple[Tensor, Tensor]:
    physics = params.physics
    accel_fn = get_threat_model(physics.threat_model)
    accel = accel_fn(state.threat_pos, state.threat_vel, physics)
    pos, vel = integrate(
        state.threat_pos, state.threat_vel, accel, physics.dt, physics.integrator
    )
    alive = state.threat_alive.unsqueeze(-1)
    return (
        torch.where(alive, pos, state.threat_pos),
        torch.where(alive, vel, state.threat_vel),
    )


def _run_tracker(
    state: EnvState,
    params: EngineParams,
    threat_pos: Tensor,
    threat_alive: Tensor,
    radar_pos: Tensor,
) -> TrackState:
    tracks = state.tracks
    dt = params.physics.dt

    # The pre-increment step index, passed as a tensor: reading it back to Python
    # would force a device synchronization on every tick.
    measured_cart, measured_sph, detected = radar.measure(
        threat_pos,
        radar_pos,
        state.seed,
        state.step_index,
        params.sigma_range_m,
        params.sigma_az_rad,
        params.sigma_el_rad,
        params.detect_range_m,
        params.radar_period_steps,
    )
    detected = detected & threat_alive

    gravity = torch.tensor(
        (0.0, 0.0, -params.physics.g), dtype=threat_pos.dtype, device=threat_pos.device
    )
    x_pred, P_pred = kalman.predict(
        tracks.x_est, tracks.P, dt, params.kf_q, known_accel=gravity
    )

    R = kalman.cartesian_measurement_covariance(
        measured_sph, params.sigma_range_m, params.sigma_az_rad, params.sigma_el_rad
    )

    opening = detected & ~tracks.detected
    x_init, P_init = kalman.initialize(measured_cart, R, INITIAL_VELOCITY_VARIANCE)

    updating = detected & tracks.detected
    x_upd, P_upd = kalman.update(x_pred, P_pred, measured_cart, R)

    open_mask = opening.unsqueeze(-1)
    update_mask = updating.unsqueeze(-1)
    x_new = torch.where(open_mask, x_init, torch.where(update_mask, x_upd, x_pred))

    open_mask_cov = opening.unsqueeze(-1).unsqueeze(-1)
    update_mask_cov = updating.unsqueeze(-1).unsqueeze(-1)
    P_new = torch.where(
        open_mask_cov, P_init, torch.where(update_mask_cov, P_upd, P_pred)
    )

    held = tracks.detected | detected
    return TrackState(
        x_est=x_new,
        P=P_new,
        detected=held,
        age=tracks.age + torch.where(held, dt, torch.zeros_like(tracks.age)),
        updated=detected,
    )


def step(state: EnvState, params: EngineParams) -> EnvState:
    """Advance every environment by one fixed dt.

    Pure: returns a new EnvState and never writes into the input's tensors. Both the
    parity tests and torch.compile depend on that -- an in-place update would alias
    a recorded trajectory into a single mutating buffer.
    """
    dt = params.physics.dt
    threat_pos, threat_vel = _advance_threats(state, params)
    step_index = state.step_index + 1

    if not params.engage:
        return replace(
            state,
            threat_pos=threat_pos,
            threat_vel=threat_vel,
            t=state.t + dt,
            step_index=step_index,
        )

    radar_pos = torch.tensor(
        params.radar_pos, dtype=threat_pos.dtype, device=threat_pos.device
    )
    tracks = _run_tracker(state, params, threat_pos, state.threat_alive, radar_pos)

    launch, launch_vel = should_launch(
        tracks,
        state.interceptor_pos,
        state.interceptor_committed,
        params.interceptor.speed_mps,
        params.interceptor.envelope_min_m,
        params.interceptor.envelope_max_m,
    )
    launch_mask = launch.unsqueeze(-1)
    interceptor_vel = torch.where(launch_mask, launch_vel, state.interceptor_vel)
    interceptor_alive = state.interceptor_alive | launch
    interceptor_committed = state.interceptor_committed | launch

    command = pn_accel(tracks, state.interceptor_pos, interceptor_vel, params.pn_gain)
    command = interceptor_mod.clamp_accel(command, params.interceptor.max_accel_mps2)

    moved_pos, moved_vel = integrate(
        state.interceptor_pos,
        interceptor_vel,
        command,
        dt,
        params.physics.integrator,
    )
    moved_vel = interceptor_mod.clamp_speed(moved_vel, params.interceptor.speed_mps)

    in_flight = interceptor_alive.unsqueeze(-1)
    interceptor_pos = torch.where(in_flight, moved_pos, state.interceptor_pos)
    interceptor_vel = torch.where(in_flight, moved_vel, interceptor_vel)

    hit, passed = intercept_mod.check_hits(
        state.threat_pos,
        threat_pos,
        state.interceptor_pos,
        interceptor_pos,
        state.threat_alive,
        interceptor_alive,
        dt,
        # Per interceptor, [I]: lethal radius belongs to the warhead. One type per
        # battery, so every slot carries the same value today.
        torch.full(
            (state.n_interceptors,),
            params.interceptor.kill_radius_m,
            dtype=threat_pos.dtype,
            device=threat_pos.device,
        ),
    )

    threat_hit = hit.any(dim=2)
    interceptor_hit = hit.any(dim=1)
    # Unscoped: every interceptor targets slot 0, so there is only one thing a round
    # could have flown past. Scoping this to an assignment is an engagement-layer
    # concern.
    interceptor_passed = passed.any(dim=1)

    threat_killed = state.threat_killed | threat_hit
    survived = state.threat_alive & ~threat_hit

    newly_leaked, destroyed = damage.resolve_leaks(
        state.threat_pos,
        threat_pos,
        state.threat_target_city,
        state.threat_active,
        survived,
        state.threat_leaked,
        state.city_pos,
        state.city_alive,
        params.city_impact_radius_m,
        dt,
    )

    # A threat below ground is down, whether or not it landed on anything. Without
    # this a threat that misses every city keeps flying underground forever, holding
    # the run open and never resolving.
    grounded = threat_pos[..., 2] < 0.0

    # A miss ends the round for that interceptor: closest approach fell inside the
    # step and it did not kill, so it has flown past and is spent.
    interceptor_alive = interceptor_alive & ~interceptor_hit & ~interceptor_passed

    # replace() rather than a field-by-field rebuild: a new state field would
    # otherwise have to be threaded through every return site or be silently dropped.
    return replace(
        state,
        threat_pos=threat_pos,
        threat_vel=threat_vel,
        interceptor_pos=interceptor_pos,
        interceptor_vel=interceptor_vel,
        threat_alive=survived & ~newly_leaked & ~grounded,
        threat_killed=threat_killed,
        threat_leaked=state.threat_leaked | newly_leaked,
        interceptor_alive=interceptor_alive,
        interceptor_committed=interceptor_committed,
        city_alive=state.city_alive & ~destroyed,
        t=state.t + dt,
        step_index=step_index,
        tracks=tracks,
    )
