"""Batched engine step orchestrating physics, sensing, guidance, and intercept.

`step` is the torch.compile target; it stays free of host-side control flow over
environments so the whole step compiles to device work. Compilation is applied in a
later task -- eager first, once parity with the oracle holds.

Order within a tick: threat physics, radar look, tracker predict and update, launch
decision, guidance, interceptor motion, intercept resolution. The tracker sees the
post-move truth, and guidance sees only what the tracker produced.
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
from mdsim.engagement import doctrine, envelopes, inventory, priority
from mdsim.engagement.assignment import UNASSIGNED
from mdsim.guidance.launch import should_launch
from mdsim.guidance.pro_nav import pn_accel
from mdsim.sensing import imm, kalman, radar
from mdsim.sensing.tracks import TrackState
from mdsim.world import damage

# A single position fix carries no velocity information, so the track opens with a
# deliberately wide velocity block, in (m/s)^2.
INITIAL_VELOCITY_VARIANCE = 1.0e6

TRACKERS = ("kf", "imm")

# IMM's CA process noise, relative to the run's single-KF q (kf_q, reused as the IMM
# CV model's q so the two trackers see the same "how much do I trust straight-line
# flight" prior). This ratio is the one sensing/imm.py's own tests validated on a
# synthetic maneuvering target (q_cv=1.0, q_ca=2000.0) -- carried through as a ratio
# rather than a fixed absolute so it scales with whatever kf_q a scenario uses,
# instead of being a second, independently-tuned constant.
IMM_Q_CA_RATIO = 2000.0


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
    decision_interval_steps: int
    city_impact_radius_m: float
    engage: bool = True
    # Phase 2 additions, all defaulted to the Phase 0/1 behaviour: a caller that never
    # sets these gets byte-identical output to before these fields existed.
    tracker: str = "kf"
    imm_q_ca: float = 0.0
    salvo_size: int = 1

    def __post_init__(self) -> None:
        if self.tracker not in TRACKERS:
            raise ValueError(f"tracker must be one of {TRACKERS}, got {self.tracker!r}")
        if self.salvo_size < 1:
            raise ValueError(f"salvo_size must be >= 1, got {self.salvo_size}")
        if self.tracker == "imm" and self.imm_q_ca <= 0.0:
            raise ValueError("imm_q_ca must be positive when tracker='imm'")

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
            decision_interval_steps=config.sim.decision_interval_steps,
            city_impact_radius_m=config.sim.city_impact_radius_m,
            engage=engage,
            imm_q_ca=config.sim.kf_process_noise_q * IMM_Q_CA_RATIO,
        )


def _advance_threats(state: EnvState, params: EngineParams) -> tuple[Tensor, Tensor]:
    physics = params.physics
    accel_fn = get_threat_model(physics.threat_model)
    accel = accel_fn(state.threat_pos, state.threat_vel, physics, state.t)
    pos, vel = integrate(
        state.threat_pos, state.threat_vel, accel, physics.dt, physics.integrator
    )
    alive = state.threat_alive.unsqueeze(-1)
    return (
        torch.where(alive, pos, state.threat_pos),
        torch.where(alive, vel, state.threat_vel),
    )


def _imm_placeholder(tracks: TrackState, dtype: torch.dtype, device: torch.device) -> TrackState:
    """Materialize inert (mu, x_models, P_models) the first time IMM runs on a state.

    make_empty() (tracks.py) leaves these None regardless of which tracker a run
    selects, so a state built before the tracker choice is known stays valid for
    either. The first IMM step on a fresh state fills them with values every track
    slot can safely mix from even though nothing has been detected yet: mu at the
    prior, x_models at zero, P_models at identity (the same inert-covariance
    convention tracks.make_empty already uses for the single-KF P). Every one of
    these entries is overwritten by imm_initialize on that slot's first real
    detection; the placeholder only has to survive being mixed with itself.
    """
    n_envs, n_tracks = tracks.detected.shape
    mu0 = torch.tensor(imm.DEFAULT_MU0, dtype=dtype, device=device)
    mu = mu0.expand(n_envs, n_tracks, imm.N_MODELS).clone()
    x_models = torch.zeros(
        (n_envs, n_tracks, imm.N_MODELS, imm.STATE_DIM), dtype=dtype, device=device
    )
    eye = torch.eye(imm.STATE_DIM, dtype=dtype, device=device)
    P_models = eye.expand(n_envs, n_tracks, imm.N_MODELS, imm.STATE_DIM, imm.STATE_DIM).clone()
    return replace(tracks, mu=mu, x_models=x_models, P_models=P_models)


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
    R = kalman.cartesian_measurement_covariance(
        measured_sph, params.sigma_range_m, params.sigma_az_rad, params.sigma_el_rad
    )
    opening = detected & ~tracks.detected

    if params.tracker == "imm":
        imm_prior = tracks if tracks.mu is not None else _imm_placeholder(
            tracks, threat_pos.dtype, threat_pos.device
        )
        continued = imm.imm_step(
            imm_prior, measured_cart, R, detected, dt, params.kf_q, params.imm_q_ca, gravity
        )
        opened = imm.imm_initialize(measured_cart, R)

        open_mask = opening.unsqueeze(-1)
        open_mask_cov = open_mask.unsqueeze(-1)
        open_mask_models = opening[..., None, None]
        open_mask_models_cov = opening[..., None, None, None]

        x_new = torch.where(open_mask, opened.x_est, continued.x_est)
        P_new = torch.where(open_mask_cov, opened.P, continued.P)
        mu_new = torch.where(open_mask, opened.mu, continued.mu)
        x_models_new = torch.where(open_mask_models, opened.x_models, continued.x_models)
        P_models_new = torch.where(open_mask_models_cov, opened.P_models, continued.P_models)
    else:
        x_pred, P_pred = kalman.predict(
            tracks.x_est, tracks.P, dt, params.kf_q, known_accel=gravity
        )
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
        mu_new = None
        x_models_new = None
        P_models_new = None

    # A threat that is no longer alive -- killed, leaked, or grounded -- reflects
    # nothing back to the radar, so the track drops the same step truth marks it
    # gone. Without this, `detected` is a one-way OR: a destroyed threat stays
    # "held" forever, and priority/assignment keep proposing rounds against
    # wreckage since tracks.detected is the only truth signal they read.
    held = (tracks.detected | detected) & threat_alive
    return TrackState(
        x_est=x_new,
        P=P_new,
        detected=held,
        age=tracks.age + torch.where(held, dt, torch.zeros_like(tracks.age)),
        updated=detected,
        mu=mu_new,
        x_models=x_models_new,
        P_models=P_models_new,
    )


def _assign(
    state: EnvState, params: EngineParams, tracks: TrackState, radar_pos: Tensor
) -> Tensor:
    """Bind free interceptor slots to threats, [N, I] int64 with -1 for unbound.

    Runs on a fixed cadence rather than every step, and existing bindings survive
    untouched. An interceptor already flying at a threat has built geometry that
    re-aiming throws away, so reassignment thrash costs more than the better pairing
    it might find. Bindings are released only when the round or the threat resolves.

    Reads TrackState and own-side tensors only. The threatened city is inferred from
    the track's extrapolated impact point, never from the threat's true target.
    """
    engageable = envelopes.in_envelope(
        tracks.position, radar_pos, params.interceptor
    )
    scores, _ = priority.threat_priority(
        tracks,
        state.city_pos,
        state.city_value,
        state.city_alive,
        engageable,
        params.physics.g,
    )

    already_engaged = inventory.engaged_threats(
        state.interceptor_target, state.n_threats, params.salvo_size
    )
    assignable = engageable & tracks.detected & ~already_engaged
    free = inventory.available(state.interceptor_enabled, state.interceptor_committed)

    proposed = doctrine.salvo(scores, assignable, free, params.salvo_size)
    committed = inventory.commit(state.interceptor_target, proposed)

    decide = ((state.step_index % params.decision_interval_steps) == 0).unsqueeze(-1)
    return torch.where(decide, committed, state.interceptor_target)


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

    interceptor_target = _assign(state, params, tracks, radar_pos)

    # A slot may only fly at the threat it is bound to. Everything unbound, spent or
    # out of stock is blocked here rather than inside the launch solution, which has
    # no business knowing about magazines.
    bound = interceptor_target >= 0
    blocked = state.interceptor_committed | ~state.interceptor_enabled | ~bound
    assignment = interceptor_target.clamp(min=0)

    launch, launch_vel = should_launch(
        tracks,
        state.interceptor_pos,
        blocked,
        params.interceptor.speed_mps,
        params.interceptor.envelope_min_m,
        params.interceptor.envelope_max_m,
        assignment,
    )
    launch_mask = launch.unsqueeze(-1)
    interceptor_vel = torch.where(launch_mask, launch_vel, state.interceptor_vel)
    interceptor_alive = state.interceptor_alive | launch
    interceptor_committed = state.interceptor_committed | launch

    command = pn_accel(
        tracks, state.interceptor_pos, interceptor_vel, params.pn_gain, assignment
    )
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
        # Per interceptor, [I]: lethal radius belongs to the warhead. Phase 0 flies
        # one type per battery, so every slot carries the same value today.
        torch.full(
            (state.n_interceptors,),
            params.interceptor.kill_radius_m,
            dtype=threat_pos.dtype,
            device=threat_pos.device,
        ),
    )

    threat_hit = hit.any(dim=2)
    interceptor_hit = hit.any(dim=1)

    # Retirement is scoped to the pair an interceptor is actually flying at. Reducing
    # `passed` over every threat would retire a round the moment it swept past any
    # threat in the raid, including ones it was never chasing -- harmless with a
    # single threat, and with several it spends the magazine on geometry alone.
    # Kills stay unscoped: a warhead that gets close enough does not check assignment.
    slot = torch.arange(state.n_threats, device=hit.device).view(1, -1, 1)
    assigned_pair = interceptor_target.unsqueeze(1) == slot
    interceptor_passed = (passed & assigned_pair).any(dim=1)

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

    threat_resolved = threat_killed | state.threat_leaked | newly_leaked | grounded
    bound_index = interceptor_target.clamp(min=0)
    bound_resolved = torch.gather(threat_resolved, 1, bound_index)

    # A round whose bound threat resolves without that round scoring the kill has
    # nothing left to fly at. Commitment is a one-way expenditure -- inventory.py
    # never refunds a fired round -- so the only outcome consistent with that model
    # is to retire it in place. Left alive, it would keep flying under the next
    # step's PN command, which gathers track state at `interceptor_target.clamp(min=0)`
    # and would silently steer it onto threat slot 0.
    own_hit = torch.gather(hit, 1, bound_index.unsqueeze(1)).squeeze(1)
    orphaned = (interceptor_target >= 0) & bound_resolved & ~own_hit

    # A miss ends the round for that interceptor: closest approach fell inside the
    # step and it did not kill, so it has flown past and is spent.
    interceptor_alive = (
        interceptor_alive & ~interceptor_hit & ~interceptor_passed & ~orphaned
    )

    # Release a binding once its round is spent or its threat has resolved. Holding a
    # stale binding would leave the slot counted as engaging a threat that no longer
    # exists, and the next decision would skip a live threat to honour it.
    spent = state.interceptor_committed & ~interceptor_alive
    released = (interceptor_target >= 0) & (bound_resolved | spent)

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
        interceptor_target=inventory.release(interceptor_target, released),
        city_alive=state.city_alive & ~destroyed,
        t=state.t + dt,
        step_index=step_index,
        tracks=tracks,
    )
