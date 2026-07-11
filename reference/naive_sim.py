"""Un-batched NumPy oracle: deliberately slow and obvious, used to check the engine.

Two independent things live here. `propagate` is the ballistic reference used by the
physics gate, drag-free by default and optionally dragged. `run_engagement` is the
full engagement pipeline -- radar, Kalman tracker, threat prioritisation, greedy
weapon-target assignment, launch decision, proportional navigation, interceptor
motion, intercept resolution and impact scoring -- written as a single-environment
loop over plain NumPy arrays with an arbitrary number of threat and interceptor slots.

The assignment solver here is a naive argmax loop, deliberately unlike the engine's
sort-and-gather. Two implementations that agree only because they share a trick prove
nothing; these agree because greedy has one answer. The tie-break is the exception --
both must break ties toward the lower threat index or the comparison is meaningless.

Threats begin at burnout, not at the pad; the caller supplies that state. The
tracker's known-acceleration input stays gravity alone even when the truth includes
drag: the filter is entitled to model gravity, and drag on an unknown ballistic
coefficient is what process noise is for.

Nothing in this file imports from src/mdsim. The one thing that is duplicated rather
than shared is the counter-based integer hash (`_mix32`, `_bits`, `_normal`), which
must produce the same noise or the two implementations would be driven by different
measurements and could not be compared at all. That hash is a pure deterministic bit
function -- no physics, no estimation, no guidance -- so copying it leaves the
oracle's independence intact on everything the gate actually tests.

Tests hold the port to the original: `_bits` is asserted bit-for-bit identical, and
the normals derived from it agree to float64 rounding. The derived values are not
bit-identical because NumPy and PyTorch differ by up to one ulp in log, sqrt and cos;
that is a transcendental-library difference, not a divergence in the scheme.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

G = 9.80665

# Exponential atmosphere, mirroring the engine's defaults. Independently written --
# what must not diverge is the physical law, and a test compares the trajectories.
RHO0 = 1.225
SCALE_HEIGHT_M = 8500.0

# Mirrors src/mdsim/core/rng.py exactly. See module docstring for why this is copied.
_MASK32 = 0xFFFFFFFF
_MIX = 0x45D9F3B
_STEP_ODD = 0x9E3779B1
_STREAM_ODD = 0x85EBCA77
_UNIT = 2.0**-24

STREAM_RANGE = 101
STREAM_AZ = 102
STREAM_EL = 103


def _drag_accel(
    pos: np.ndarray,
    vel: np.ndarray,
    beta: float,
    rho0: float = RHO0,
    scale_height_m: float = SCALE_HEIGHT_M,
) -> np.ndarray:
    """Quadratic drag against an exponential atmosphere, a = -(rho |v| / 2 beta) v.

    beta = 0 means no atmosphere at all, which is what the drag-free ballistic gate
    uses. Altitude is floored at zero so a body below ground does not see a density
    that grows without bound.
    """
    if beta <= 0.0:
        return np.zeros(3, dtype=np.float64)
    rho = rho0 * np.exp(-max(float(pos[2]), 0.0) / scale_height_m)
    return -(rho * float(np.linalg.norm(vel)) / (2.0 * beta)) * vel


def propagate(
    pos0: np.ndarray | list[float],
    vel0: np.ndarray | list[float],
    dt: float,
    n_steps: int,
    g: float = G,
    beta: float = 0.0,
    rho0: float = RHO0,
    scale_height_m: float = SCALE_HEIGHT_M,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate one body for n_steps with semi-implicit Euler.

    Returns (positions, velocities), each [n_steps + 1, 3], including the initial state.

    beta defaults to 0, so the default behaviour is the drag-free ballistic arc the
    physics gate compares against. Pass a ballistic coefficient to add drag.

    Clarity beats speed here: this exists to be read and trusted, so the loop stays
    explicit and nothing is vectorized.
    """
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    if n_steps < 0:
        raise ValueError(f"n_steps must be non-negative, got {n_steps}")

    pos = np.asarray(pos0, dtype=np.float64).copy()
    vel = np.asarray(vel0, dtype=np.float64).copy()
    if pos.shape != (3,) or vel.shape != (3,):
        raise ValueError("pos0 and vel0 must each be length-3 vectors")

    positions = np.empty((n_steps + 1, 3), dtype=np.float64)
    velocities = np.empty((n_steps + 1, 3), dtype=np.float64)
    positions[0] = pos
    velocities[0] = vel

    gravity = np.array([0.0, 0.0, -g], dtype=np.float64)

    for k in range(n_steps):
        # Acceleration is evaluated at the OLD state, then semi-implicit Euler:
        # velocity updates first, then position uses the new velocity.
        accel = gravity + _drag_accel(pos, vel, beta, rho0, scale_height_m)
        vel = vel + accel * dt
        pos = pos + vel * dt
        positions[k + 1] = pos
        velocities[k + 1] = vel

    return positions, velocities


def _mix32(x: np.ndarray) -> np.ndarray:
    """32-bit avalanche hash over an int64 array holding 32-bit values."""
    x = x & _MASK32
    x = (x ^ (x >> 16)) & _MASK32
    x = (x * _MIX) & _MASK32
    x = (x ^ (x >> 16)) & _MASK32
    x = (x * _MIX) & _MASK32
    x = (x ^ (x >> 16)) & _MASK32
    return x


def _bits(seeds: np.ndarray, step: int, stream: int, n_lanes: int) -> np.ndarray:
    """Uniform 32-bit words, [N, n_lanes]."""
    root = _mix32(seeds ^ ((step * _STEP_ODD) & _MASK32))
    root = _mix32(root ^ ((stream * _STREAM_ODD) & _MASK32))
    lanes = _mix32(np.arange(n_lanes, dtype=np.int64))
    return _mix32(root[:, None] ^ lanes[None, :])


def _normal(seeds: np.ndarray, step: int, stream: int, n_lanes: int) -> np.ndarray:
    """Standard normal, [N, n_lanes], Box-Muller keeping only the cosine branch."""
    bits = _bits(seeds, step, stream, 2 * n_lanes)
    u1 = ((bits[:, :n_lanes] >> 8).astype(np.float64) + 1.0) * _UNIT
    u2 = (bits[:, n_lanes:] >> 8).astype(np.float64) * _UNIT
    radius = np.sqrt(-2.0 * np.log(u1))
    angle = (2.0 * np.pi) * u2
    return radius * np.cos(angle)


# Threat-scoring constants, mirroring the engagement layer's documented values.
FLYOUT_REFERENCE_S = 60.0
MAX_URGENCY = 1.0
PRIORITY_EPS = 1e-6


@dataclass(frozen=True)
class EngagementParams:
    """Scalar mirror of EngineParams, in the units the engine actually uses."""

    dt: float
    g: float
    beta: float
    radar_pos: tuple[float, float, float]
    sigma_range_m: float
    sigma_az_rad: float
    sigma_el_rad: float
    detect_range_m: float
    radar_period_steps: int
    kf_q: float
    pn_gain: float
    interceptor_speed_mps: float
    interceptor_max_accel_mps2: float
    envelope_min_m: float
    envelope_max_m: float
    envelope_alt_min_m: float
    envelope_alt_max_m: float
    kill_radius_m: float
    initial_velocity_variance: float
    decision_interval_steps: int
    city_impact_radius_m: float
    rho0: float = RHO0
    scale_height_m: float = SCALE_HEIGHT_M


def _to_spherical(rel: np.ndarray) -> np.ndarray:
    rng = max(float(np.linalg.norm(rel)), 1e-6)
    az = np.arctan2(rel[1], rel[0])
    el = np.arcsin(np.clip(rel[2] / rng, -1.0, 1.0))
    return np.array([rng, az, el], dtype=np.float64)


def _to_cartesian(sph: np.ndarray) -> np.ndarray:
    rng, az, el = sph[0], sph[1], sph[2]
    cos_el = np.cos(el)
    return np.array(
        [rng * cos_el * np.cos(az), rng * cos_el * np.sin(az), rng * np.sin(el)],
        dtype=np.float64,
    )


def _measurement_covariance(sph: np.ndarray, p: EngagementParams) -> np.ndarray:
    """R_cart = J diag(sigma_r^2, sigma_az^2, sigma_el^2) J^T at the measured point."""
    rng, az, el = sph[0], sph[1], sph[2]
    cos_az, sin_az = np.cos(az), np.sin(az)
    cos_el, sin_el = np.cos(el), np.sin(el)

    J = np.array(
        [
            [cos_el * cos_az, -rng * cos_el * sin_az, -rng * sin_el * cos_az],
            [cos_el * sin_az, rng * cos_el * cos_az, -rng * sin_el * sin_az],
            [sin_el, 0.0, rng * cos_el],
        ],
        dtype=np.float64,
    )
    R_sph = np.diag(
        np.array(
            [p.sigma_range_m**2, p.sigma_az_rad**2, p.sigma_el_rad**2],
            dtype=np.float64,
        )
    )
    return J @ R_sph @ J.T


def _transition(dt: float) -> np.ndarray:
    F = np.eye(6, dtype=np.float64)
    F[0, 3] = dt
    F[1, 4] = dt
    F[2, 5] = dt
    return F


def _process_noise(dt: float, q: float) -> np.ndarray:
    eye = np.eye(3, dtype=np.float64)
    Q = np.zeros((6, 6), dtype=np.float64)
    Q[:3, :3] = eye * (dt**4 / 4.0)
    Q[:3, 3:] = eye * (dt**3 / 2.0)
    Q[3:, :3] = eye * (dt**3 / 2.0)
    Q[3:, 3:] = eye * (dt**2)
    return Q * q


def _measurement_matrix() -> np.ndarray:
    H = np.zeros((3, 6), dtype=np.float64)
    H[0, 0] = 1.0
    H[1, 1] = 1.0
    H[2, 2] = 1.0
    return H


def _kf_predict(x, P, dt, q, known_accel=None):
    """CV predict with an optional deterministic acceleration input (gravity).

    The input enters semi-implicitly, matching the truth integrator: velocity takes
    delta_v, position takes delta_v * dt. The covariance recursion is untouched.
    """
    F = _transition(dt)
    Q = _process_noise(dt, q)
    x_pred = F @ x
    if known_accel is not None:
        delta_v = known_accel * dt
        x_pred = x_pred + np.concatenate([delta_v * dt, delta_v])
    return x_pred, F @ P @ F.T + Q


def _kf_update(x, P, z, R):
    H = _measurement_matrix()
    innovation = z - H @ x
    PHt = P @ H.T
    S = H @ PHt + R
    K = np.linalg.solve(S, PHt.T).T
    x_new = x + K @ innovation
    A = np.eye(6, dtype=np.float64) - K @ H
    P_new = A @ P @ A.T + K @ R @ K.T
    return x_new, P_new


def _kf_initialize(z, R, velocity_variance):
    x = np.concatenate([z, np.zeros(3, dtype=np.float64)])
    P = np.zeros((6, 6), dtype=np.float64)
    P[:3, :3] = R
    P[3:, 3:] = np.eye(3, dtype=np.float64) * velocity_variance
    return x, P


def _predicted_intercept(track_pos, track_vel, launch_pos, speed):
    """Earliest positive root of |p_rel + v t| = s t. Returns (point, feasible)."""
    eps = 1e-9
    p_rel = track_pos - launch_pos

    a = float(track_vel @ track_vel) - speed**2
    b = 2.0 * float(p_rel @ track_vel)
    c = float(p_rel @ p_rel)

    discriminant = b * b - 4.0 * a * c
    real = discriminant >= 0.0
    sqrt_disc = np.sqrt(max(discriminant, 0.0))

    denom = eps if abs(a) < eps else 2.0 * a
    t1 = (-b - sqrt_disc) / denom
    t2 = (-b + sqrt_disc) / denom

    t1_pos = t1 if t1 > 0.0 else np.inf
    t2_pos = t2 if t2 > 0.0 else np.inf
    t_go = min(t1_pos, t2_pos)

    feasible = bool(real and np.isfinite(t_go))
    if not feasible:
        t_go = 0.0

    return track_pos + track_vel * t_go, feasible


def _pn_accel(track_pos, track_vel, held, interceptor_pos, interceptor_vel, n_gain):
    if not held:
        return np.zeros(3, dtype=np.float64)

    eps = 1e-6
    r = track_pos - interceptor_pos
    v_rel = track_vel - interceptor_vel

    r_sq = max(float(r @ r), eps**2)
    r_norm = np.sqrt(r_sq)
    u_los = r / r_norm

    omega = np.cross(r, v_rel) / r_sq
    closing_speed = -float(r @ v_rel) / r_norm
    return n_gain * closing_speed * np.cross(omega, u_los)


def _clamp_norm(vector, limit, eps):
    norm = max(float(np.linalg.norm(vector)), eps)
    scale = min(limit / norm, 1.0)
    return vector * scale


def _closest_approach(dp, dv, dt):
    speed_sq = max(float(dv @ dv), 1e-12)
    tau = float(np.clip(-(dp @ dv) / speed_sq, 0.0, dt))
    separation = dp + dv * tau
    return float(np.linalg.norm(separation)), tau


def _in_envelope(track_pos, battery_pos, p: EngagementParams) -> bool:
    """Hard range and altitude cutoffs. Both bounds must hold."""
    slant = float(np.linalg.norm(track_pos - battery_pos))
    altitude = float(track_pos[2])
    return (
        p.envelope_min_m <= slant <= p.envelope_max_m
        and p.envelope_alt_min_m <= altitude <= p.envelope_alt_max_m
    )


def _time_to_impact(track_pos, track_vel, g: float) -> float:
    """Positive root of z + vz t - g t^2 / 2 = 0, drag-free.

    The defender extrapolates a track; it does not know the threat's ballistic
    coefficient, so this stays drag-free even though the truth is not.
    """
    height = float(track_pos[2])
    vertical = float(track_vel[2])
    discriminant = max(vertical * vertical + 2.0 * g * height, 0.0)
    return max((vertical + np.sqrt(discriminant)) / g, 0.0)


def _threat_priority(
    track_x, track_detected, engageable, city_pos, city_value, city_alive, g
):
    """Score every track slot, [T]. Zero for tracks that are not live candidates.

    The threatened city is inferred from the track's own extrapolated impact point.
    Nothing here reads the threat's true target.
    """
    n_threats = track_x.shape[0]
    priority = np.zeros(n_threats, dtype=np.float64)
    any_live = bool(np.any(city_alive))

    for t in range(n_threats):
        pos = track_x[t, :3]
        vel = track_x[t, 3:]
        t_impact = _time_to_impact(pos, vel, g)

        impact_point = np.array(
            [pos[0] + vel[0] * t_impact, pos[1] + vel[1] * t_impact, 0.0],
            dtype=np.float64,
        )

        best_city = 0
        best_distance = np.inf
        for c in range(city_pos.shape[0]):
            if not city_alive[c]:
                continue
            distance = float(np.linalg.norm(impact_point - city_pos[c]))
            if distance < best_distance:
                best_distance = distance
                best_city = c

        confidence = float(np.clip(t_impact / FLYOUT_REFERENCE_S, 0.0, 1.0))
        urgency = min(1.0 / (t_impact + PRIORITY_EPS), MAX_URGENCY)
        score = float(city_value[best_city]) * confidence * urgency

        if track_detected[t] and engageable[t] and any_live:
            priority[t] = score

    return priority


def _greedy_naive(priority, assignable, available):
    """Bind available slots to the best remaining threats, [I] int64 with -1 unbound.

    An argmax loop rather than a sort: strictly-greater comparison keeps the first
    slot seen at equal priority, which is the lower threat index and matches the
    engine's stable descending sort.
    """
    n_threats = priority.shape[0]
    target = np.full(available.shape[0], -1, dtype=np.int64)
    remaining = np.array(assignable, dtype=bool, copy=True)

    for i in range(available.shape[0]):
        if not available[i]:
            continue
        best = -1
        best_score = 0.0
        for t in range(n_threats):
            if not remaining[t]:
                continue
            if best < 0 or priority[t] > best_score:
                best = t
                best_score = float(priority[t])
        if best < 0:
            break
        target[i] = best
        remaining[best] = False

    return target


def _engaged_threats(interceptor_target, n_threats):
    """Threats that already have a shooter bound to them, [T] bool."""
    engaged = np.zeros(n_threats, dtype=bool)
    for bound in interceptor_target:
        if bound >= 0:
            engaged[bound] = True
    return engaged


def _reached_city(pos_before, pos_after, city, radius, dt) -> bool:
    """Whether this step's path passed within `radius` of a stationary city."""
    delta_pos = pos_before - city
    delta_vel = (pos_after - pos_before) / dt
    distance, _ = _closest_approach(delta_pos, delta_vel, dt)
    return distance <= radius


def run_engagement(
    params: EngagementParams,
    threat_pos0: np.ndarray | list,
    threat_vel0: np.ndarray | list,
    threat_target_city: np.ndarray | list,
    threat_active: np.ndarray | list,
    interceptor_enabled: np.ndarray | list,
    city_pos: np.ndarray | list,
    city_value: np.ndarray | list,
    seed: int,
    n_steps: int,
    battery_pos: np.ndarray | list[float] | None = None,
) -> dict[str, np.ndarray]:
    """Run the full engagement loop for one environment with T threats and I rounds.

    Returns per-step histories, each with a leading axis of length n_steps holding the
    post-step value: threat_pos [S,T,3], threat_vel [S,T,3], interceptor_pos [S,I,3],
    interceptor_vel [S,I,3], track_x [S,T,6], track_P [S,T,6,6], track_detected [S,T],
    threat_alive [S,T], threat_killed [S,T], threat_leaked [S,T],
    interceptor_alive [S,I], interceptor_committed [S,I], interceptor_target [S,I],
    city_alive [S,C].

    The step ordering mirrors the engine exactly: threat physics, radar look with the
    pre-increment step index, tracker predict then initiate-or-update, assignment on
    the decision cadence, launch check, guidance, interceptor motion, intercept
    resolution, then impact scoring.
    """
    p = params
    dt = p.dt

    threat_pos = np.asarray(threat_pos0, dtype=np.float64).copy()
    threat_vel = np.asarray(threat_vel0, dtype=np.float64).copy()
    target_city = np.asarray(threat_target_city, dtype=np.int64).copy()
    active = np.asarray(threat_active, dtype=bool).copy()
    enabled = np.asarray(interceptor_enabled, dtype=bool).copy()
    cities = np.asarray(city_pos, dtype=np.float64)
    values = np.asarray(city_value, dtype=np.float64)

    n_threats = threat_pos.shape[0]
    n_interceptors = enabled.shape[0]

    radar_pos = np.asarray(p.radar_pos, dtype=np.float64)
    launch_origin = (
        radar_pos.copy()
        if battery_pos is None
        else np.asarray(battery_pos, dtype=np.float64)
    )

    interceptor_pos = np.tile(launch_origin, (n_interceptors, 1))
    interceptor_vel = np.zeros((n_interceptors, 3), dtype=np.float64)

    # An inactive slot is never alive, so it never flies, is never tracked and is
    # never scored.
    threat_alive = active.copy()
    threat_killed = np.zeros(n_threats, dtype=bool)
    threat_leaked = np.zeros(n_threats, dtype=bool)

    interceptor_alive = np.zeros(n_interceptors, dtype=bool)
    interceptor_committed = np.zeros(n_interceptors, dtype=bool)
    interceptor_target = np.full(n_interceptors, -1, dtype=np.int64)

    city_alive = np.ones(cities.shape[0], dtype=bool)

    track_x = np.zeros((n_threats, 6), dtype=np.float64)
    track_P = np.stack([np.eye(6, dtype=np.float64) for _ in range(n_threats)])
    track_detected = np.zeros(n_threats, dtype=bool)

    seeds = np.array([seed], dtype=np.int64)
    gravity = np.array([0.0, 0.0, -p.g], dtype=np.float64)

    history: dict[str, list] = {
        key: []
        for key in (
            "threat_pos",
            "threat_vel",
            "interceptor_pos",
            "interceptor_vel",
            "track_x",
            "track_P",
            "track_detected",
            "threat_alive",
            "threat_killed",
            "threat_leaked",
            "interceptor_alive",
            "interceptor_committed",
            "interceptor_target",
            "city_alive",
        )
    }

    for k in range(n_steps):
        threat_pos_before = threat_pos.copy()
        interceptor_pos_before = interceptor_pos.copy()
        threat_alive_before = threat_alive.copy()
        interceptor_committed_before = interceptor_committed.copy()

        # 1. Threat physics. Gravity plus drag evaluated at the old state, then
        # semi-implicit Euler. Frozen once the threat is dead.
        for t in range(n_threats):
            accel = gravity + _drag_accel(
                threat_pos[t], threat_vel[t], p.beta, p.rho0, p.scale_height_m
            )
            new_vel = threat_vel[t] + accel * dt
            new_pos = threat_pos[t] + new_vel * dt
            if threat_alive[t]:
                threat_pos[t] = new_pos
                threat_vel[t] = new_vel

        # 2. Radar look, using the pre-increment step index as the engine does. One
        # noise lane per threat slot, so the lane layout matches the batched draw.
        noise_range = _normal(seeds, k, STREAM_RANGE, n_threats)[0]
        noise_az = _normal(seeds, k, STREAM_AZ, n_threats)[0]
        noise_el = _normal(seeds, k, STREAM_EL, n_threats)[0]
        on_cadence = (k % p.radar_period_steps) == 0

        # 3. Tracker: predict every step, initiate or update on a measurement. The
        # tracker models gravity as a known control input, as the engine does.
        for t in range(n_threats):
            true_sph = _to_spherical(threat_pos[t] - radar_pos)
            measured_sph = true_sph + np.array(
                [
                    noise_range[t] * p.sigma_range_m,
                    noise_az[t] * p.sigma_az_rad,
                    noise_el[t] * p.sigma_el_rad,
                ],
                dtype=np.float64,
            )
            measured_cart = _to_cartesian(measured_sph) + radar_pos

            in_range = true_sph[0] <= p.detect_range_m
            detected = bool(on_cadence and in_range and threat_alive_before[t])

            x_pred, P_pred = _kf_predict(
                track_x[t], track_P[t], dt, p.kf_q, known_accel=gravity
            )
            R = _measurement_covariance(measured_sph, p)

            if detected and not track_detected[t]:
                track_x[t], track_P[t] = _kf_initialize(
                    measured_cart, R, p.initial_velocity_variance
                )
            elif detected and track_detected[t]:
                track_x[t], track_P[t] = _kf_update(x_pred, P_pred, measured_cart, R)
            else:
                track_x[t], track_P[t] = x_pred, P_pred

            # A threat that is no longer alive reflects nothing back to the radar, so
            # the track drops the same step truth marks it gone -- otherwise this is
            # a one-way OR and a destroyed threat stays "held" forever.
            track_detected[t] = bool((track_detected[t] or detected) and threat_alive_before[t])

        # 4. Assignment, on the decision cadence and from the tracks only. Existing
        # bindings are untouched; a committed round is never re-aimed.
        engageable = np.array(
            [_in_envelope(track_x[t, :3], radar_pos, p) for t in range(n_threats)],
            dtype=bool,
        )
        priority = _threat_priority(
            track_x, track_detected, engageable, cities, values, city_alive, p.g
        )
        already_engaged = _engaged_threats(interceptor_target, n_threats)
        assignable = engageable & track_detected & ~already_engaged
        available = enabled & ~interceptor_committed

        if (k % p.decision_interval_steps) == 0:
            proposed = _greedy_naive(priority, assignable, available)
            for i in range(n_interceptors):
                if proposed[i] >= 0:
                    interceptor_target[i] = proposed[i]

        # 5. Launch. A slot may only fly at the threat it is bound to, and only if it
        # holds an unfired round.
        for i in range(n_interceptors):
            bound = int(interceptor_target[i])
            blocked = (
                interceptor_committed[i] or (not enabled[i]) or bound < 0
            )
            slot = max(bound, 0)

            point, feasible = _predicted_intercept(
                track_x[slot, :3],
                track_x[slot, 3:],
                interceptor_pos[i],
                p.interceptor_speed_mps,
            )
            reach = float(np.linalg.norm(point - interceptor_pos[i]))
            in_range_band = p.envelope_min_m <= reach <= p.envelope_max_m
            launch = bool(
                track_detected[slot] and feasible and in_range_band and not blocked
            )

            if launch:
                heading = point - interceptor_pos[i]
                heading = heading / max(float(np.linalg.norm(heading)), 1e-9)
                interceptor_vel[i] = heading * p.interceptor_speed_mps
                interceptor_alive[i] = True
                interceptor_committed[i] = True

        # 6. Guidance and interceptor motion, each round flying at its own binding.
        for i in range(n_interceptors):
            slot = max(int(interceptor_target[i]), 0)
            command = _pn_accel(
                track_x[slot, :3],
                track_x[slot, 3:],
                bool(track_detected[slot]),
                interceptor_pos[i],
                interceptor_vel[i],
                p.pn_gain,
            )
            command = _clamp_norm(command, p.interceptor_max_accel_mps2, 1e-9)

            moved_vel = interceptor_vel[i] + command * dt
            moved_pos = interceptor_pos[i] + moved_vel * dt
            moved_vel = _clamp_norm(moved_vel, p.interceptor_speed_mps, 1e-9)

            if interceptor_alive[i]:
                interceptor_pos[i] = moved_pos
                interceptor_vel[i] = moved_vel

        # 7. Intercept resolution over every threat-interceptor pair. A hit counts
        # against any threat; retirement on a fly-past counts only against the threat
        # the round was bound to.
        threat_hit = np.zeros(n_threats, dtype=bool)
        interceptor_hit = np.zeros(n_interceptors, dtype=bool)
        interceptor_passed = np.zeros(n_interceptors, dtype=bool)
        own_hit = np.zeros(n_interceptors, dtype=bool)

        for t in range(n_threats):
            for i in range(n_interceptors):
                if not (threat_alive_before[t] and interceptor_alive[i]):
                    continue
                dp = interceptor_pos_before[i] - threat_pos_before[t]
                dv = (
                    (interceptor_pos[i] - interceptor_pos_before[i])
                    - (threat_pos[t] - threat_pos_before[t])
                ) / dt
                distance, tau = _closest_approach(dp, dv, dt)

                if distance <= p.kill_radius_m:
                    threat_hit[t] = True
                    interceptor_hit[i] = True
                    if int(interceptor_target[i]) == t:
                        own_hit[i] = True
                elif 0.0 < tau < dt and int(interceptor_target[i]) == t:
                    interceptor_passed[i] = True

        threat_killed = threat_killed | threat_hit
        survived = threat_alive_before & ~threat_hit

        # 8. Impact scoring. Binary: the first threat to reach a city destroys it.
        newly_leaked = np.zeros(n_threats, dtype=bool)
        city_hit = np.zeros(cities.shape[0], dtype=bool)
        for t in range(n_threats):
            if not (active[t] and survived[t] and not threat_leaked[t]):
                continue
            if _reached_city(
                threat_pos_before[t],
                threat_pos[t],
                cities[int(target_city[t])],
                p.city_impact_radius_m,
                dt,
            ):
                newly_leaked[t] = True
                city_hit[int(target_city[t])] = True

        destroyed = city_hit & city_alive

        # A threat below ground is down, whether or not it landed on anything.
        grounded = threat_pos[:, 2] < 0.0

        threat_resolved = threat_killed | threat_leaked | newly_leaked | grounded

        # A round whose bound threat resolves without that round scoring the kill has
        # nothing left to fly at. Commitment is a one-way expenditure -- never
        # refunded -- so the only outcome consistent with that model is to retire it
        # in place. Left alive, it would keep flying under the next step's guidance
        # section, which clamps an unbound target to slot 0 and would silently steer
        # it there.
        orphaned = np.zeros(n_interceptors, dtype=bool)
        for i in range(n_interceptors):
            bound = int(interceptor_target[i])
            if bound >= 0 and threat_resolved[bound] and not own_hit[i]:
                orphaned[i] = True

        interceptor_alive = (
            interceptor_alive & ~interceptor_hit & ~interceptor_passed & ~orphaned
        )

        # Release a binding once its round is spent or its threat has resolved.
        spent = interceptor_committed_before & ~interceptor_alive
        for i in range(n_interceptors):
            bound = int(interceptor_target[i])
            if bound < 0:
                continue
            if threat_resolved[bound] or spent[i]:
                interceptor_target[i] = -1

        threat_alive = survived & ~newly_leaked & ~grounded
        threat_leaked = threat_leaked | newly_leaked
        city_alive = city_alive & ~destroyed

        history["threat_pos"].append(threat_pos.copy())
        history["threat_vel"].append(threat_vel.copy())
        history["interceptor_pos"].append(interceptor_pos.copy())
        history["interceptor_vel"].append(interceptor_vel.copy())
        history["track_x"].append(track_x.copy())
        history["track_P"].append(track_P.copy())
        history["track_detected"].append(track_detected.copy())
        history["threat_alive"].append(threat_alive.copy())
        history["threat_killed"].append(threat_killed.copy())
        history["threat_leaked"].append(threat_leaked.copy())
        history["interceptor_alive"].append(interceptor_alive.copy())
        history["interceptor_committed"].append(interceptor_committed.copy())
        history["interceptor_target"].append(interceptor_target.copy())
        history["city_alive"].append(city_alive.copy())

    return {key: np.asarray(values) for key, values in history.items()}
