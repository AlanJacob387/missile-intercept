"""The correctness gate: the batched engine must agree with the reference oracle."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from mdsim.core import dynamics
from mdsim.core.config import load_config, resolve_device
from mdsim.core.dynamics import PhysicsParams
from mdsim.core.rng import normal
from mdsim.core.state import EnvState, make_initial
from mdsim.envs.rollout import rollout
from reference.naive_sim import G, propagate

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
N_STEPS = 2000  # 100 s at dt=0.05; the 45-degree arc is airborne for ~124 s

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


def _oracle_positions(config) -> torch.Tensor:
    """Oracle trajectory in float64, aligned with rollout's post-step frames."""
    positions, _ = propagate(
        config.scenario.threat_launch_pos_m,
        config.scenario.threat_launch_vel_mps,
        config.sim.dt,
        N_STEPS,
    )
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


def test_engine_vs_oracle() -> None:
    """Engine at N=1 must track the float64 oracle at every step, in float64.

    This is the algorithm check: matching dtypes, so any deviation above the float64
    accumulation floor means the integrator, its update order, or g is wrong.
    """
    config = _config(1)
    params = PhysicsParams.from_config(config)
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
    params = PhysicsParams.from_config(config)
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
    params = PhysicsParams.from_config(config)

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
    params = PhysicsParams.from_config(_config(1024))

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
    args = (
        config.scenario.threat_launch_pos_m,
        config.scenario.threat_launch_vel_mps,
        config.sim.dt,
        N_STEPS,
    )
    pos_a, vel_a = propagate(*args)
    pos_b, vel_b = propagate(*args)

    np.testing.assert_array_equal(pos_a, pos_b)
    np.testing.assert_array_equal(vel_a, vel_b)
