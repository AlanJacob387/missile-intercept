"""The Phase 0 correctness gate: batched engine must agree with the reference oracle."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from mdsim.core import dynamics
from mdsim.core.config import load_config, resolve_device
from mdsim.core.rng import _bits as torch_bits, normal
from mdsim.core.state import EnvState, _resolve_battery_stocks, make_initial
from mdsim.envs.engine import INITIAL_VELOCITY_VARIANCE, EngineParams, step
from mdsim.envs.rollout import rollout
from mdsim.world.scenario import multi_launch_state
from reference.naive_sim import (
    G,
    STREAM_AZ as naive_STREAM_AZ,
    STREAM_EL as naive_STREAM_EL,
    STREAM_RANGE as naive_STREAM_RANGE,
    EngagementParams,
    _bits as naive_bits,
    _normal as naive_normal,
    propagate,
    run_engagement,
)

# Not bitwise. GPU reduction order varies with batch shape, so N=1 and N=1024 may
# differ in the last bits even when the computation is identical.
RTOL = 1e-5

# Comparing a float32 engine against the float64 oracle cannot reach RTOL, and not
# because the algorithm is wrong: in float64 the same engine tracks the oracle to
# ~2e-8. float32 plateaus at ~1.5e-5 relative (about 2.2 m at a 1.5e5 m scale, ~141
# ulps) and stops growing. That is the representation floor, so the algorithm check
# runs in float64 and the float32 cost is measured against this documented bound.
FLOAT32_FLOOR = 5e-5

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"

# 100 s at dt=0.05. The threat now starts at 25 km burnout altitude, so drag-free it
# stays airborne for ~235 s and this horizon sits comfortably mid-flight.
N_STEPS = 2000

# The full-loop check needs a longer horizon than the physics one. Nothing enters the
# launch envelope until the threat descends through the interceptor's 24 km ceiling
# around step 3340, and impacts land near step 4150, so a short run would compare radar
# and tracker while never touching assignment, launch, guidance or the intercept test.
# The assertions below fail loudly rather than quietly skipping that half of the loop.
FULL_LOOP_STEPS = 4400

# Both parity seeds are named explicitly rather than relying on the default. Whether
# a given seed kills depends on the tracker tuning and the ballistic coefficient, so a
# seed that misses today can start killing after a retune -- which would silently turn
# this into a second kill case and test nothing new. Re-pick with the recipe
# documented above KILLING_SEED.
MISS_SEED = 1

# The raid parity case. Fewer rounds than threats, so the run exercises the magazine
# running dry as well as the assignment itself: with a round for every threat the
# greedy solver never has to choose.
RAID_SEED = 0
RAID_THREATS = 3
RAID_INTERCEPTORS = 3
RAID_INVENTORY = 2

# Env-to-env spread for the batch-invariance check. Identical rows would pass
# trivially, so each env gets its own launch state from the counter-based RNG.
POS_SPREAD_M = 500.0
VEL_SPREAD_MPS = 20.0
_POS_STREAM = 11
_VEL_STREAM = 12


def _config(n_envs: int):
    config = load_config(CONFIG_DIR)
    return replace(config, sim=replace(config.sim, n_envs=n_envs))


def _devices() -> list[str]:
    """CPU always, plus the resolved accelerator when it is something else."""
    selected = resolve_device(load_config(CONFIG_DIR).sim.device, verbose=False)
    return ["cpu"] if selected == "cpu" else ["cpu", selected]


def _relative_deviation(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Max absolute deviation over the whole trajectory, scaled by its magnitude.

    Element-wise relative error is meaningless where a coordinate passes through
    zero, which this trajectory does. Scaling by the trajectory's own magnitude
    keeps the check strict without dividing by near-zero values.
    """
    scale = expected.abs().max()
    return float((actual - expected).abs().max() / scale)


def _burnout_states(config, n_threats: int):
    """Burnout positions, velocities and target cities for a raid.

    Deliberately taken from mdsim.world.scenario rather than recomputed here. This is
    config resolution -- a root-find over the scenario's engagement range -- not
    physics, and both sides must start from bit-identical initial conditions or the
    parity comparison measures nothing. Everything the gate actually tests (drag,
    integration, radar, tracking, prioritisation, assignment, guidance, intercept,
    impact scoring) stays independently implemented in reference/naive_sim.py.
    """
    return multi_launch_state(config, n_threats)


def _burnout_state(config) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Single-threat burnout state, for the physics-only checks."""
    positions, velocities, _ = _burnout_states(config, 1)
    return positions[0], velocities[0]


def _physics_params(config) -> EngineParams:
    """Engine params for the physics-only gate: no sensing, and no drag.

    Drag is on in the run configuration, but these checks compare against the
    drag-free ballistic oracle so they isolate the integrator, its update order and g.
    Leaving drag on here would fold the atmosphere model into a test that exists to
    catch integration bugs.
    """
    params = EngineParams.from_config(config, engage=False)
    return replace(params, physics=replace(params.physics, beta=0.0))


def _oracle_positions(config) -> torch.Tensor:
    """Drag-free oracle trajectory, aligned with rollout's post-step frames."""
    pos0, vel0 = _burnout_state(config)
    positions, _ = propagate(pos0, vel0, config.sim.dt, N_STEPS)
    return torch.from_numpy(np.ascontiguousarray(positions[1:]))


def _randomize(state: EnvState) -> EnvState:
    """Give each env its own launch state, derived from that env's seed."""
    shape = (state.n_threats, 3)
    pos_jitter = normal(state.seed, 0, _POS_STREAM, shape) * POS_SPREAD_M
    vel_jitter = normal(state.seed, 0, _VEL_STREAM, shape) * VEL_SPREAD_MPS
    return replace(
        state,
        threat_pos=state.threat_pos + pos_jitter,
        threat_vel=state.threat_vel + vel_jitter,
    )


def _slice_env(state: EnvState, index: int) -> EnvState:
    """Extract env `index` as a standalone N=1 state."""
    return EnvState.from_dict(
        {name: tensor[index : index + 1].clone() for name, tensor in state.to_dict().items()}
    )


def test_gravity_constant_matches_oracle() -> None:
    """Parity fails on the constant alone if these ever drift apart."""
    assert dynamics.G == G


def test_rng_port_matches_torch() -> None:
    """The oracle duplicates the RNG hash rather than importing it; hold it to that.

    If the port drifted, the two implementations would be driven by different
    measurements and the parity comparison would be meaningless rather than failing
    loudly.

    The integer hash is checked bit for bit. The normals derived from it are checked
    to float64 rounding instead: NumPy and PyTorch differ by up to one ulp in log,
    sqrt and cos, which is a transcendental-library difference and not a divergence
    in the scheme.
    """
    for seed in (0, 1, 8, 37, 1023, 999983):
        for step_index in (0, 1, 7, 2099, 2396):
            for stream in (naive_STREAM_RANGE, naive_STREAM_AZ, naive_STREAM_EL):
                mine_bits = naive_bits(
                    np.array([seed], dtype=np.int64), step_index, stream, 6
                )
                their_bits = torch_bits(
                    torch.tensor([seed], dtype=torch.int64), step_index, stream, 6
                ).numpy()
                np.testing.assert_array_equal(mine_bits, their_bits)

                mine = naive_normal(
                    np.array([seed], dtype=np.int64), step_index, stream, 3
                )
                theirs = normal(
                    torch.tensor([seed], dtype=torch.int64),
                    step_index,
                    stream,
                    (3,),
                    torch.float64,
                ).numpy()
                np.testing.assert_allclose(mine, theirs, rtol=1e-15, atol=0.0)


def test_engine_vs_oracle() -> None:
    """Engine at N=1 must track the float64 oracle at every step, in float64.

    Physics only (engage=False), so this isolates the integrator, its update order
    and g from everything the sensing and guidance layers add.
    """
    config = _config(1)
    params = _physics_params(config)
    expected = _oracle_positions(config)

    state = make_initial(config, "cpu", dtype=torch.float64)
    _, trajectory = rollout(state, params, N_STEPS, record=True)
    deviation = _relative_deviation(trajectory[:, 0, 0, :].cpu(), expected)

    print(f"engine (float64) vs oracle, max relative deviation: {deviation:.3e}")
    assert deviation < RTOL, f"engine deviates {deviation:.3e} from oracle"


def test_float32_stays_within_representation_floor() -> None:
    """Quantify what the float32 run path costs, on CPU and on device.

    Kept separate from the algorithm check so a real regression cannot hide inside
    a tolerance widened to accommodate float32.
    """
    config = _config(1)
    params = _physics_params(config)
    expected = _oracle_positions(config)

    deviations: dict[str, float] = {}
    for device in _devices():
        state = make_initial(config, device, dtype=torch.float32)
        _, trajectory = rollout(state, params, N_STEPS, record=True)
        actual = trajectory[:, 0, 0, :].double().cpu()
        deviations[device] = _relative_deviation(actual, expected)

    report = ", ".join(f"{d}={v:.3e}" for d, v in deviations.items())
    print(f"engine (float32) vs oracle, max relative deviation: {report}")

    for device, deviation in deviations.items():
        assert deviation < FLOAT32_FLOOR, f"{device} deviates {deviation:.3e} ({report})"


def test_engine_matches_across_devices() -> None:
    """CPU and device must agree with each other, not just with the oracle."""
    devices = _devices()
    if len(devices) == 1:
        pytest.skip("no accelerator available to compare against CPU")

    config = _config(1)
    params = _physics_params(config)

    trajectories = []
    for device in devices:
        state = make_initial(config, device, dtype=torch.float32)
        _, trajectory = rollout(state, params, N_STEPS, record=True)
        trajectories.append(trajectory.double().cpu())

    deviation = _relative_deviation(trajectories[1], trajectories[0])
    print(f"{devices[1]} vs cpu, max relative deviation: {deviation:.3e}")
    assert deviation < RTOL


@pytest.mark.parametrize("env_index", [0, 37, 512, 1023])
def test_batch_invariance(env_index: int) -> None:
    """Env i must produce the same trajectory at N=1024 as it does alone at N=1."""
    device = _devices()[-1]
    params = _physics_params(_config(1024))

    batched = _randomize(make_initial(_config(1024), device))
    _, batched_trajectory = rollout(batched, params, N_STEPS, record=True)

    solo = _slice_env(batched, env_index)
    _, solo_trajectory = rollout(solo, params, N_STEPS, record=True)

    expected = batched_trajectory[:, env_index].double().cpu()
    actual = solo_trajectory[:, 0].double().cpu()

    deviation = _relative_deviation(actual, expected)
    print(f"env {env_index} N=1 vs N=1024, max relative deviation: {deviation:.3e}")
    assert deviation < RTOL


def test_batch_invariance_uses_distinct_envs() -> None:
    """Guards the test above: if the envs were identical it would prove nothing."""
    batched = _randomize(make_initial(_config(1024), "cpu"))
    assert not torch.allclose(batched.threat_pos[0], batched.threat_pos[1])
    assert not torch.allclose(batched.threat_vel[0], batched.threat_vel[1])


def test_reference_self_consistency() -> None:
    """The oracle must be deterministic, or it cannot arbitrate anything."""
    config = load_config(CONFIG_DIR)
    pos0, vel0 = _burnout_state(config)
    beta = config.threats[config.scenario.threat].ballistic_coefficient_beta

    # Run it with drag on: the dragged path is the one the full-loop gate leans on,
    # and it exercises the atmosphere term as well as the integrator.
    pos_a, vel_a = propagate(pos0, vel0, config.sim.dt, N_STEPS, beta=beta)
    pos_b, vel_b = propagate(pos0, vel0, config.sim.dt, N_STEPS, beta=beta)

    np.testing.assert_array_equal(pos_a, pos_b)
    np.testing.assert_array_equal(vel_a, vel_b)


# ---------------------------------------------------------------------------
# Full Phase 0 loop: radar -> Kalman -> launch -> PN -> interceptor -> intercept
# ---------------------------------------------------------------------------


def _oracle_params(params: EngineParams) -> EngagementParams:
    """Scalar mirror of EngineParams for the un-batched oracle."""
    return EngagementParams(
        dt=params.physics.dt,
        g=params.physics.g,
        beta=params.physics.beta,
        radar_pos=params.radar_pos,
        sigma_range_m=params.sigma_range_m,
        sigma_az_rad=params.sigma_az_rad,
        sigma_el_rad=params.sigma_el_rad,
        detect_range_m=params.detect_range_m,
        radar_period_steps=params.radar_period_steps,
        kf_q=params.kf_q,
        pn_gain=params.pn_gain,
        interceptor_speed_mps=params.interceptor.speed_mps,
        interceptor_max_accel_mps2=params.interceptor.max_accel_mps2,
        envelope_min_m=params.interceptor.envelope_min_m,
        envelope_max_m=params.interceptor.envelope_max_m,
        envelope_alt_min_m=params.interceptor.envelope_alt_min_m,
        envelope_alt_max_m=params.interceptor.envelope_alt_max_m,
        kill_radius_m=params.interceptor.kill_radius_m,
        initial_velocity_variance=INITIAL_VELOCITY_VARIANCE,
        decision_interval_steps=params.decision_interval_steps,
        city_impact_radius_m=params.city_impact_radius_m,
        rho0=params.physics.rho0,
        scale_height_m=params.physics.scale_height_m,
        threat_model=params.physics.threat_model,
        maneuver_accel_mps2=params.physics.maneuver_accel_mps2,
        maneuver_period_s=params.physics.maneuver_period_s,
        tracker=params.tracker,
        imm_q_ca=params.imm_q_ca,
        assignment_method=params.assignment_method,
        salvo_size=params.salvo_size,
        battery_positions=params.battery_positions,
        battery_of_slot=params.battery_of_slot,
        aim_dispersion_rad=params.aim_dispersion_rad,
    )


# Fields recorded per step. Floats are compared with a relative deviation; the integer
# and boolean ones are compared exactly, because assignment and magazine state are
# discrete decisions and "close" is not a meaningful thing for them to be.
_FLOAT_FIELDS = ("threat_pos", "interceptor_pos", "track_x", "track_P")
_EXACT_FIELDS = (
    "threat_alive",
    "threat_killed",
    "threat_leaked",
    "track_detected",
    "interceptor_alive",
    "interceptor_committed",
    "interceptor_target",
    "city_alive",
)


def _engine_history(
    config,
    params: EngineParams,
    n_steps: int,
    inventory: int | list[int] | None = None,
    step_fn=step,
) -> dict[str, torch.Tensor]:
    """Run the engine at N=1 in float64, recording every field the oracle returns."""
    state = make_initial(config, "cpu", dtype=torch.float64, inventory=inventory)

    frames: dict[str, list[torch.Tensor]] = {
        key: [] for key in _FLOAT_FIELDS + _EXACT_FIELDS
    }

    for _ in range(n_steps):
        state = step_fn(state, params)
        frames["threat_pos"].append(state.threat_pos[0].clone())
        frames["interceptor_pos"].append(state.interceptor_pos[0].clone())
        frames["track_x"].append(state.tracks.x_est[0].clone())
        frames["track_P"].append(state.tracks.P[0].clone())
        frames["threat_alive"].append(state.threat_alive[0].clone())
        frames["threat_killed"].append(state.threat_killed[0].clone())
        frames["threat_leaked"].append(state.threat_leaked[0].clone())
        frames["track_detected"].append(state.tracks.detected[0].clone())
        frames["interceptor_alive"].append(state.interceptor_alive[0].clone())
        frames["interceptor_committed"].append(state.interceptor_committed[0].clone())
        frames["interceptor_target"].append(state.interceptor_target[0].clone())
        frames["city_alive"].append(state.city_alive[0].clone())

    return {key: torch.stack(values) for key, values in frames.items()}


def _oracle_enabled(config, inventory: int | list[int]) -> list[bool]:
    """Flat per-slot enabled mask, mirroring make_initial's inventory resolution.

    Single-battery: a flat index<inventory count over the whole slot range.
    Multi-battery: inventory resolution (the None/int/Sequence[int] contract) is
    config resolution shared with make_initial, not part of what this file's
    parity gate is testing -- reusing state._resolve_battery_stocks here is the
    same kind of deliberate sharing _burnout_states already does for launch
    geometry via mdsim.world.scenario.multi_launch_state.
    """
    batteries = config.scenario.batteries
    if not batteries:
        n_interceptors = config.scenario.n_interceptors
        return [index < inventory for index in range(n_interceptors)]

    stocks = _resolve_battery_stocks(batteries, inventory)
    enabled: list[bool] = []
    for spec, stock in zip(batteries, stocks):
        enabled.extend(index < stock for index in range(spec.n_interceptors))
    return enabled


def _oracle_history(
    config,
    params: EngineParams,
    n_steps: int,
    inventory: int | list[int],
    n_active: int | None = None,
):
    """Run the oracle from the same resolved initial conditions the engine uses.

    Mirrors make_initial's padding: geometry is built for the ACTIVE raid size and
    inactive slots are parked on the last active launch point. Building it for the
    slot count instead would spread a small raid differently on the two sides and the
    comparison would fail for a reason that has nothing to do with the engagement.
    """
    n_threats = config.scenario.n_threats

    active_count = n_threats if n_active is None else n_active
    active = max(active_count, 1)

    positions, velocities, targets = _burnout_states(config, active)
    pad = n_threats - active
    if pad > 0:
        positions = list(positions) + [positions[-1]] * pad
        velocities = list(velocities) + [velocities[-1]] * pad
        targets = list(targets) + [targets[-1]] * pad

    enabled = _oracle_enabled(config, inventory)

    return run_engagement(
        _oracle_params(params),
        positions,
        velocities,
        targets,
        [index < active_count for index in range(n_threats)],
        enabled,
        [city.position_m for city in config.cities],
        [city.value for city in config.cities],
        config.sim.seed,
        n_steps,
    )


def _compare(engine, oracle, upto: int | None = None) -> dict[str, float]:
    """Assert exact agreement on the discrete fields, return float deviations."""
    limit = slice(None) if upto is None else slice(0, upto)

    for name in _EXACT_FIELDS:
        np.testing.assert_array_equal(
            engine[name][limit].numpy(),
            np.asarray(oracle[name], dtype=engine[name].numpy().dtype)[limit],
            err_msg=f"{name} diverged",
        )

    return {
        name: _relative_deviation(
            engine[name][limit],
            torch.from_numpy(np.ascontiguousarray(oracle[name]))[limit],
        )
        for name in _FLOAT_FIELDS
    }


def _first_true(flags) -> int | None:
    """Index of the first True in a 1-D boolean sequence, or None."""
    indices = np.flatnonzero(np.asarray(flags, dtype=bool))
    return int(indices[0]) if indices.size else None


def _raid_config(seed: int, n_threats: int, n_interceptors: int):
    """A one-environment raid scenario at a given seed."""
    config = load_config(CONFIG_DIR)
    return replace(
        config,
        sim=replace(config.sim, n_envs=1, seed=seed),
        scenario=replace(
            config.scenario,
            n_threats=n_threats,
            n_interceptors=n_interceptors,
        ),
    )


@pytest.mark.slow
def test_full_loop_engine_vs_oracle() -> None:
    """A multi-threat raid must match the oracle step for step, covariance included.

    Position parity alone would not catch a wrong measurement covariance: R only
    reaches the trajectory through the Kalman gain, so P is compared directly. The
    discrete fields -- magazine state and the assignment itself -- are compared
    exactly, so a prioritisation or tie-break disagreement fails here rather than
    showing up later as an unexplained trajectory drift.
    """
    config = _raid_config(RAID_SEED, RAID_THREATS, RAID_INTERCEPTORS)
    params = EngineParams.from_config(config, engage=True)

    engine = _engine_history(config, params, FULL_LOOP_STEPS, inventory=RAID_INVENTORY)
    oracle = _oracle_history(config, params, FULL_LOOP_STEPS, RAID_INVENTORY)

    deviations = _compare(engine, oracle)
    report = ", ".join(f"{k}={v:.3e}" for k, v in deviations.items())
    print(f"multi-threat loop engine vs oracle: {report}")

    for name, deviation in deviations.items():
        assert deviation < RTOL, f"{name} deviates {deviation:.3e} ({report})"

    # Without these the comparison could pass while only ever exercising radar and the
    # tracker: nothing reaches the launch envelope until late in the flight.
    committed = int(engine["interceptor_committed"][-1].sum())
    bound_ever = int((engine["interceptor_target"] >= 0).any(dim=0).sum())
    leaked = int(engine["threat_leaked"][-1].sum())
    killed = int(engine["threat_killed"][-1].sum())
    print(
        f"raid outcome: committed={committed} bound_ever={bound_ever} "
        f"killed={killed} leaked={leaked}"
    )
    assert committed > 0, "no round ever launched"
    assert bound_ever > 1, "assignment never bound more than one interceptor"
    assert killed + leaked > 0, "the raid never resolved"


@pytest.mark.slow
def test_single_threat_regression() -> None:
    """One threat and one round must still behave as a single engagement, end to end.

    The launch STEP moved when the altitude envelope arrived, so this asserts
    engine-oracle agreement rather than a remembered step number -- a constant would
    only record whatever the engine happens to do today.
    """
    config = _raid_config(MISS_SEED, 1, 1)
    params = EngineParams.from_config(config, engage=True)

    engine = _engine_history(config, params, FULL_LOOP_STEPS, inventory=1)
    oracle = _oracle_history(config, params, FULL_LOOP_STEPS, 1)

    deviations = _compare(engine, oracle)
    report = ", ".join(f"{k}={v:.3e}" for k, v in deviations.items())
    print(f"single-threat regression engine vs oracle: {report}")

    for name, deviation in deviations.items():
        assert deviation < RTOL, f"{name} deviates {deviation:.3e} ({report})"

    assert bool(engine["interceptor_committed"][-1, 0]), "interceptor never launched"

    # This seed misses, so the run also covers the `passed` transition: the round's
    # closest approach falls inside a step, it fails to kill, and it is retired.
    spent = _first_true(
        ~engine["interceptor_alive"][:, 0].numpy()
        & engine["interceptor_committed"][:, 0].numpy()
    )
    print(f"miss case: round retired at step {spent}")
    assert spent is not None, "the miss case never retired the round"
    assert not bool(engine["threat_killed"][-1, 0]), "MISS_SEED is no longer a miss"


# A killing seed is named here because MISS_SEED does not kill. Without one the
# intercept resolution -- closest_approach, the hit branch, and the alive/killed
# transitions -- would never be compared against the oracle at all.
#
# THIS CONSTANT IS ENVELOPE- AND TRAJECTORY-DEPENDENT and must be re-picked whenever
# the scenario's interceptor entry, the threat's flight, or the physics change. It has
# broken three times: when phase0_single moved from `generic_mid_tier` ([5, 60] km) to
# `pac3_mse` ([3, 70] km), when threats moved to a burnout initial condition with
# atmospheric drag, and when the engagement layer added the altitude envelope, which
# pushed launch several hundred steps later.
#
# To re-pick: run a batched N=256 float64 engagement at one threat and one round, roll
# until the rounds are spent, read `threat_killed[:, 0]`, and take any env index (env i
# carries seed base_seed + i). The killing rate has swung between 1-in-256 and 1-in-3
# across retunings, so scan a wide batch rather than assuming an early index will do.
KILLING_SEED = 4


@pytest.mark.slow
def test_full_loop_kill_parity() -> None:
    """A seed that kills, so the hit path is compared and not just the miss path."""
    config = _raid_config(KILLING_SEED, 1, 1)
    params = EngineParams.from_config(config, engage=True)

    engine = _engine_history(config, params, FULL_LOOP_STEPS, inventory=1)
    oracle = _oracle_history(config, params, FULL_LOOP_STEPS, 1)

    engine_kill = _first_true(engine["threat_killed"][:, 0].numpy())
    oracle_kill = _first_true(np.asarray(oracle["threat_killed"])[:, 0])
    print(f"kill step: engine={engine_kill} oracle={oracle_kill}")

    assert engine_kill is not None, f"seed {KILLING_SEED} no longer kills"
    # Exact, not within tolerance: the kill is a discrete event and a one-step
    # disagreement would mean the two closest-approach tests disagree.
    assert engine_kill == oracle_kill

    deviations = _compare(engine, oracle, upto=engine_kill + 1)
    report = ", ".join(f"{k}={v:.3e}" for k, v in deviations.items())
    print(f"kill case engine vs oracle (to kill step): {report}")
    for name, deviation in deviations.items():
        assert deviation < RTOL, f"{name} deviates {deviation:.3e} ({report})"

    assert bool(engine["threat_killed"][-1, 0])
    assert not bool(engine["threat_alive"][-1, 0])
    assert not bool(engine["interceptor_alive"][-1, 0])
    assert bool(engine["interceptor_committed"][-1, 0])


def _phase2_config(seed: int, n_threats: int, n_interceptors: int, threat: str | None = None):
    """A raid config with an optional threat override, for Phase 2 parity cases."""
    config = _raid_config(seed, n_threats, n_interceptors)
    if threat is not None:
        config = replace(config, scenario=replace(config.scenario, threat=threat))
    return config


@pytest.mark.slow
def test_full_loop_weave_parity() -> None:
    """Maneuvering threat physics (weave) against the independent oracle.

    iskander_m carries a real maneuver_g anchor; the KF tracker is Phase 1's, kept
    fixed here so this isolates the threat-model physics from the tracker choice.
    """
    config = _phase2_config(0, 1, 1, threat="iskander_m")
    params = EngineParams.from_config(config, engage=True)
    params = replace(params, physics=replace(params.physics, threat_model="weave"))

    # Long enough for the interceptor to actually launch (~step 3300 for iskander_m
    # under weave) -- short of that, interceptor_pos never leaves the battery origin
    # on either side and the relative-deviation scale is degenerately zero.
    engine = _engine_history(config, params, FULL_LOOP_STEPS, inventory=1)
    oracle = _oracle_history(config, params, FULL_LOOP_STEPS, 1)

    deviations = _compare(engine, oracle)
    report = ", ".join(f"{k}={v:.3e}" for k, v in deviations.items())
    print(f"weave engine vs oracle: {report}")
    for name, deviation in deviations.items():
        assert deviation < RTOL, f"{name} deviates {deviation:.3e} ({report})"


@pytest.mark.slow
def test_full_loop_imm_parity() -> None:
    """IMM tracker (states, per-model covariances folded into track_P) against the
    independent oracle. Ballistic flight, so this isolates the tracker from the
    threat-model choice."""
    config = _raid_config(0, 2, 2)
    params = EngineParams.from_config(config, engage=True)
    params = replace(params, tracker="imm")

    engine = _engine_history(config, params, FULL_LOOP_STEPS, inventory=2)
    oracle = _oracle_history(config, params, FULL_LOOP_STEPS, 2)

    deviations = _compare(engine, oracle)
    report = ", ".join(f"{k}={v:.3e}" for k, v in deviations.items())
    print(f"imm engine vs oracle: {report}")
    for name, deviation in deviations.items():
        assert deviation < RTOL, f"{name} deviates {deviation:.3e} ({report})"


@pytest.mark.slow
def test_full_loop_hungarian_parity() -> None:
    """Hungarian assignment decisions (interceptor_target, exact) against the oracle."""
    config = _raid_config(2, 3, 3)
    params = EngineParams.from_config(config, engage=True)
    params = replace(params, assignment_method="hungarian")

    engine = _engine_history(config, params, FULL_LOOP_STEPS, inventory=3)
    oracle = _oracle_history(config, params, FULL_LOOP_STEPS, 3)

    deviations = _compare(engine, oracle)
    report = ", ".join(f"{k}={v:.3e}" for k, v in deviations.items())
    print(f"hungarian engine vs oracle: {report}")
    for name, deviation in deviations.items():
        assert deviation < RTOL, f"{name} deviates {deviation:.3e} ({report})"


@pytest.mark.slow
def test_full_loop_salvo_parity() -> None:
    """Salvo (size 2) inventory/assignment bookkeeping against the oracle."""
    config = _raid_config(4, 1, 3)
    params = EngineParams.from_config(config, engage=True)
    params = replace(params, salvo_size=2)

    engine = _engine_history(config, params, FULL_LOOP_STEPS, inventory=3)
    oracle = _oracle_history(config, params, FULL_LOOP_STEPS, 3)

    deviations = _compare(engine, oracle)
    report = ", ".join(f"{k}={v:.3e}" for k, v in deviations.items())
    print(f"salvo engine vs oracle: {report}")
    for name, deviation in deviations.items():
        assert deviation < RTOL, f"{name} deviates {deviation:.3e} ({report})"

    committed = int(engine["interceptor_committed"][-1].sum())
    assert committed > 1, "salvo never committed more than one round"


@pytest.mark.slow
def test_full_loop_combined_phase2_parity() -> None:
    """All four Phase 2 features at once: weave + IMM + Hungarian + salvo.

    The parity gate has to hold with every axis on simultaneously, not just
    pairwise -- a bug in how two features compose would not necessarily show up
    with only one enabled at a time.
    """
    config = _phase2_config(4, 2, 4, threat="iskander_m")
    params = EngineParams.from_config(config, engage=True)
    params = replace(
        params,
        physics=replace(params.physics, threat_model="weave"),
        tracker="imm",
        assignment_method="hungarian",
        salvo_size=2,
    )

    engine = _engine_history(config, params, FULL_LOOP_STEPS, inventory=4)
    oracle = _oracle_history(config, params, FULL_LOOP_STEPS, 4)

    deviations = _compare(engine, oracle)
    report = ", ".join(f"{k}={v:.3e}" for k, v in deviations.items())
    print(f"combined phase 2 engine vs oracle: {report}")
    for name, deviation in deviations.items():
        assert deviation < RTOL, f"{name} deviates {deviation:.3e} ({report})"


def _multi_battery_config(seed: int):
    config = load_config(CONFIG_DIR, scenario="phase2_multi_battery")
    return replace(config, sim=replace(config.sim, n_envs=1, seed=seed))


@pytest.mark.slow
def test_full_loop_multi_battery_greedy_parity() -> None:
    """Multiple batteries, greedy assignment, against the independent oracle.

    Per-battery inventory (a list, not a scalar) exercises make_initial's
    multi-battery inventory resolution and _oracle_enabled's mirror of it together.
    """
    config = _multi_battery_config(0)
    params = EngineParams.from_config(config, engage=True)

    engine = _engine_history(config, params, 5000, inventory=[2, 2, 2])
    oracle = _oracle_history(config, params, 5000, [2, 2, 2])

    deviations = _compare(engine, oracle)
    report = ", ".join(f"{k}={v:.3e}" for k, v in deviations.items())
    print(f"multi-battery greedy engine vs oracle: {report}")
    for name, deviation in deviations.items():
        assert deviation < RTOL, f"{name} deviates {deviation:.3e} ({report})"


@pytest.mark.slow
def test_full_loop_multi_battery_hungarian_parity() -> None:
    """Multiple batteries, Hungarian assignment, against the independent oracle.

    This is the parity case that actually exercises the tie-break in
    assignment.hungarian and reference.naive_sim's _hungarian_naive: same-battery
    slots see identical per-(slot, threat) value, so the optimum is genuinely
    non-unique here, and only a matching deterministic tie-break makes the discrete
    interceptor_target field comparable at all -- see assignment.py's _tie_break
    docstring for why an additive tie-break fails this specific case.
    """
    config = _multi_battery_config(0)
    params = EngineParams.from_config(config, engage=True)
    params = replace(params, assignment_method="hungarian")

    engine = _engine_history(config, params, 5000, inventory=[2, 2, 2])
    oracle = _oracle_history(config, params, 5000, [2, 2, 2])

    deviations = _compare(engine, oracle)
    report = ", ".join(f"{k}={v:.3e}" for k, v in deviations.items())
    print(f"multi-battery hungarian engine vs oracle: {report}")
    for name, deviation in deviations.items():
        assert deviation < RTOL, f"{name} deviates {deviation:.3e} ({report})"


@pytest.mark.slow
def test_full_loop_dispersion_parity() -> None:
    """Per-round aim dispersion (single battery), against the independent oracle."""
    config = _raid_config(4, 1, 3)
    params = EngineParams.from_config(config, engage=True)
    params = replace(params, salvo_size=2, aim_dispersion_rad=0.02)

    engine = _engine_history(config, params, FULL_LOOP_STEPS, inventory=3)
    oracle = _oracle_history(config, params, FULL_LOOP_STEPS, 3)

    deviations = _compare(engine, oracle)
    report = ", ".join(f"{k}={v:.3e}" for k, v in deviations.items())
    print(f"dispersion engine vs oracle: {report}")
    for name, deviation in deviations.items():
        assert deviation < RTOL, f"{name} deviates {deviation:.3e} ({report})"


@pytest.mark.slow
def test_full_loop_combined_multi_battery_dispersion_parity() -> None:
    """Multi-battery + dispersion + salvo + Hungarian all at once.

    The parity gate has to hold with every axis on simultaneously here too, the
    same reasoning as test_full_loop_combined_phase2_parity.
    """
    config = _multi_battery_config(2)
    params = EngineParams.from_config(config, engage=True)
    params = replace(params, assignment_method="hungarian", salvo_size=2, aim_dispersion_rad=0.02)

    engine = _engine_history(config, params, 5000, inventory=[3, 3, 3])
    oracle = _oracle_history(config, params, 5000, [3, 3, 3])

    deviations = _compare(engine, oracle)
    report = ", ".join(f"{k}={v:.3e}" for k, v in deviations.items())
    print(f"combined multi-battery+dispersion engine vs oracle: {report}")
    for name, deviation in deviations.items():
        assert deviation < RTOL, f"{name} deviates {deviation:.3e} ({report})"


@pytest.mark.slow
def test_single_battery_zero_dispersion_regression() -> None:
    """The opt-in guard: 1 battery and aim_dispersion_rad=0 must reproduce Phase 2
    exactly. Regenerates a Phase 2 case explicitly through the new fields' defaults
    (rather than simply omitting them) so a future default change cannot silently
    break this guarantee without this test noticing."""
    config = _raid_config(RAID_SEED, RAID_THREATS, RAID_INTERCEPTORS)
    params = EngineParams.from_config(config, engage=True)
    params = replace(params, assignment_method="greedy", salvo_size=1, aim_dispersion_rad=0.0)
    assert params.battery_positions == (config.scenario.battery_pos_m,)
    assert params.battery_of_slot == (0,) * config.scenario.n_interceptors

    engine = _engine_history(config, params, FULL_LOOP_STEPS, inventory=RAID_INVENTORY)
    oracle = _oracle_history(config, params, FULL_LOOP_STEPS, RAID_INVENTORY)

    deviations = _compare(engine, oracle)
    report = ", ".join(f"{k}={v:.3e}" for k, v in deviations.items())
    print(f"single-battery zero-dispersion regression: {report}")
    for name, deviation in deviations.items():
        assert deviation < RTOL, f"{name} deviates {deviation:.3e} ({report})"


@pytest.mark.slow
def test_full_loop_batch_invariance() -> None:
    """Batch invariance must survive the sensing and guidance layers, not just physics.

    Radar noise is drawn per environment, so this is where a batch-dependent RNG or a
    shape-dependent reduction would show up.
    """
    device = _devices()[-1]
    params = EngineParams.from_config(_config(1024), engage=True)

    batched = _randomize(make_initial(_config(1024), device))
    solo_states = {index: _slice_env(batched, index) for index in (0, 37, 512, 1023)}

    batched_track = []
    for _ in range(400):
        batched = step(batched, params)
        batched_track.append(batched.tracks.x_est.clone())
    batched_history = torch.stack(batched_track)

    worst = 0.0
    for index, solo in solo_states.items():
        frames = []
        for _ in range(400):
            solo = step(solo, params)
            frames.append(solo.tracks.x_est.clone())
        solo_history = torch.stack(frames)

        deviation = _relative_deviation(
            solo_history[:, 0].double().cpu(), batched_history[:, index].double().cpu()
        )
        print(f"full loop env {index} N=1 vs N=1024 track deviation: {deviation:.3e}")
        worst = max(worst, deviation)

    assert worst < RTOL
