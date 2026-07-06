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
from mdsim.core.state import EnvState, make_initial
from mdsim.envs.engine import INITIAL_VELOCITY_VARIANCE, EngineParams, step
from mdsim.envs.rollout import rollout
from mdsim.world.scenario import launch_state
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

# The full-loop check needs a longer horizon than the physics one. The threat enters
# the launch envelope partway through the flight (step ~1872) and the engagement does
# not resolve until step ~2625, so a short run would compare radar and tracker while
# never touching launch, guidance, interceptor motion or the intercept test. The
# assertions below fail loudly rather than quietly skipping that half of the loop.
FULL_LOOP_STEPS = 2700

# Both parity seeds are named explicitly rather than relying on the default. Whether
# a given seed kills depends on the tracker tuning and the ballistic coefficient, so a
# seed that misses today can start killing after a retune -- which would silently turn
# this into a second kill case and test nothing new. Re-pick with the recipe
# documented above KILLING_SEED.
MISS_SEED = 1

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


def _burnout_state(config) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """The threat's burnout position and velocity.

    Deliberately taken from mdsim.world.scenario rather than recomputed here. This is
    config resolution -- a root-find over the scenario's engagement range -- not
    physics, and both sides must start from bit-identical initial conditions or the
    parity comparison measures nothing. Everything the gate actually tests (drag,
    integration, radar, tracking, guidance, intercept) stays independently implemented
    in reference/naive_sim.py.
    """
    return launch_state(config)


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
        kill_radius_m=params.interceptor.kill_radius_m,
        initial_velocity_variance=INITIAL_VELOCITY_VARIANCE,
        rho0=params.physics.rho0,
        scale_height_m=params.physics.scale_height_m,
    )


def _engine_history(config, params: EngineParams, n_steps: int) -> dict[str, torch.Tensor]:
    """Run the engine at N=1 in float64, recording every field the oracle returns."""
    state = make_initial(config, "cpu", dtype=torch.float64)

    frames: dict[str, list[torch.Tensor]] = {
        key: [] for key in ("threat_pos", "interceptor_pos", "track_x", "track_P")
    }
    flags: dict[str, list[bool]] = {
        key: []
        for key in (
            "threat_killed",
            "threat_alive",
            "interceptor_alive",
            "interceptor_committed",
        )
    }

    for _ in range(n_steps):
        state = step(state, params)
        frames["threat_pos"].append(state.threat_pos[0, 0].clone())
        frames["interceptor_pos"].append(state.interceptor_pos[0, 0].clone())
        frames["track_x"].append(state.tracks.x_est[0, 0].clone())
        frames["track_P"].append(state.tracks.P[0, 0].clone())
        flags["threat_killed"].append(bool(state.threat_killed[0, 0]))
        flags["threat_alive"].append(bool(state.threat_alive[0, 0]))
        flags["interceptor_alive"].append(bool(state.interceptor_alive[0, 0]))
        flags["interceptor_committed"].append(bool(state.interceptor_committed[0, 0]))

    history = {key: torch.stack(values) for key, values in frames.items()}
    history.update({key: torch.tensor(values) for key, values in flags.items()})
    return history


def _first_true(flags) -> int | None:
    """Index of the first True in a 1-D boolean sequence, or None."""
    indices = np.flatnonzero(np.asarray(flags, dtype=bool))
    return int(indices[0]) if indices.size else None


def test_full_loop_engine_vs_oracle() -> None:
    """The whole Phase 0 chain must match the oracle step for step, covariance included.

    Position parity alone would not catch a wrong measurement covariance: R only
    reaches the trajectory through the Kalman gain, so P is compared directly.
    """
    config = _config(1)
    config = replace(config, sim=replace(config.sim, seed=MISS_SEED))
    params = EngineParams.from_config(config, engage=True)

    pos0, vel0 = _burnout_state(config)
    engine = _engine_history(config, params, FULL_LOOP_STEPS)
    oracle = run_engagement(
        _oracle_params(params),
        pos0,
        vel0,
        config.sim.seed,
        FULL_LOOP_STEPS,
        battery_pos=config.scenario.battery_pos_m,
    )

    deviations = {
        name: _relative_deviation(
            engine[name], torch.from_numpy(np.ascontiguousarray(oracle[name]))
        )
        for name in ("threat_pos", "interceptor_pos", "track_x", "track_P")
    }

    report = ", ".join(f"{k}={v:.3e}" for k, v in deviations.items())
    print(f"full loop engine vs oracle: {report}")

    engine_kills = int(engine["threat_killed"].sum())
    oracle_kills = int(oracle["threat_killed"].sum())
    print(f"kill steps: engine={engine_kills} oracle={oracle_kills}")

    for name, deviation in deviations.items():
        assert deviation < RTOL, f"{name} deviates {deviation:.3e} ({report})"
    assert engine_kills == oracle_kills

    # Without this the comparison could pass while only ever exercising radar and the
    # tracker: the threat does not reach the launch envelope until ~step 1872.
    assert bool(engine["interceptor_committed"][-1]), "interceptor never launched"
    assert bool(oracle["interceptor_alive"].any()), "oracle interceptor never flew"

    # The default seed misses, so this run also covers the `passed` transition: the
    # interceptor's closest approach falls inside a step, it fails to kill, and it is
    # retired. Both sides must retire it on the same step.
    for name in ("threat_alive", "interceptor_alive"):
        np.testing.assert_array_equal(
            engine[name].numpy(), np.asarray(oracle[name], dtype=bool)
        )
    engine_spent = _first_true(~engine["interceptor_alive"].numpy() & engine["interceptor_committed"].numpy())
    print(f"miss case: interceptor retired at step {engine_spent}")
    assert engine_spent is not None, "the miss case never retired the interceptor"


# A killing seed is named here because MISS_SEED does not kill. Without one the
# intercept resolution -- closest_approach, the hit branch, and the alive/killed
# transitions -- would never be compared against the oracle at all.
#
# BOTH CONSTANTS ARE ENVELOPE- AND TRAJECTORY-DEPENDENT and must be re-picked whenever
# the scenario's interceptor entry, the threat's flight, or the physics change. They
# have broken twice already: first when phase0_single moved from `generic_mid_tier`
# ([5, 60] km) to `pac3_mse` ([3, 70] km), and again when threats moved to a burnout
# initial condition with atmospheric drag, which reshaped the whole trajectory.
#
# To re-pick: run a batched N=256 float64 engagement, roll until the interceptors are
# spent, read `threat_killed[:, 0]`, and take any env index (env i carries seed
# base_seed + i). Killing seeds run about 1 in 10 at nominal noise, but that rate has
# swung between 1-in-8 and 1-in-256 across retunings, so scan a wide batch rather than
# assuming an early index will do.
KILLING_SEED = 15
KILL_STEP = 2614


def test_full_loop_kill_parity() -> None:
    """A seed that kills, so the hit path is compared and not just the miss path."""
    config = load_config(CONFIG_DIR)
    config = replace(config, sim=replace(config.sim, n_envs=1, seed=KILLING_SEED))
    params = EngineParams.from_config(config, engage=True)

    pos0, vel0 = _burnout_state(config)
    engine = _engine_history(config, params, FULL_LOOP_STEPS)
    oracle = run_engagement(
        _oracle_params(params),
        pos0,
        vel0,
        KILLING_SEED,
        FULL_LOOP_STEPS,
        battery_pos=config.scenario.battery_pos_m,
    )

    engine_kill = _first_true(engine["threat_killed"].numpy())
    oracle_kill = _first_true(oracle["threat_killed"])
    print(f"kill step: engine={engine_kill} oracle={oracle_kill}")

    assert engine_kill is not None, f"seed {KILLING_SEED} no longer kills"
    # Exact, not within tolerance: the kill is a discrete event and a one-step
    # disagreement would mean the two closest-approach tests disagree.
    assert engine_kill == oracle_kill
    assert engine_kill == KILL_STEP

    # Trajectories up to and including the kill step.
    upto = engine_kill + 1
    deviations = {
        name: _relative_deviation(
            engine[name][:upto],
            torch.from_numpy(np.ascontiguousarray(oracle[name][:upto])),
        )
        for name in ("threat_pos", "interceptor_pos", "track_x", "track_P")
    }
    report = ", ".join(f"{k}={v:.3e}" for k, v in deviations.items())
    print(f"kill case engine vs oracle (to kill step): {report}")
    for name, deviation in deviations.items():
        assert deviation < RTOL, f"{name} deviates {deviation:.3e} ({report})"

    for name in ("threat_killed", "threat_alive", "interceptor_alive"):
        np.testing.assert_array_equal(
            engine[name].numpy(), np.asarray(oracle[name], dtype=bool)
        )

    assert bool(engine["threat_killed"][-1])
    assert not bool(engine["threat_alive"][-1])
    assert not bool(engine["interceptor_alive"][-1])
    assert bool(engine["interceptor_committed"][-1])


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
