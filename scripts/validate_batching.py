"""Correctness gate: report which checks pass, fail, or await the engine."""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from mdsim.core.config import load_config, resolve_device  # noqa: E402
from mdsim.core.dynamics import PhysicsParams  # noqa: E402
from mdsim.core.rng import normal  # noqa: E402
from mdsim.core.state import EnvState, make_initial  # noqa: E402
from mdsim.envs.rollout import rollout  # noqa: E402
from reference.naive_sim import G, propagate  # noqa: E402

N_STEPS = 2000
RTOL = 1e-5
# float32 cannot reach RTOL against a float64 oracle; see tests/test_batch_equivalence.
FLOAT32_FLOOR = 5e-5

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"

_DEVICE = "cpu"


@dataclass
class Result:
    name: str
    status: str
    deviation: float | None = None
    detail: str = ""


def _config(n_envs: int):
    config = load_config(ROOT / "configs")
    return replace(config, sim=replace(config.sim, n_envs=n_envs))


def _relative_deviation(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual - expected).abs().max() / expected.abs().max())


def _oracle_positions(config) -> torch.Tensor:
    positions, _ = propagate(
        config.scenario.threat_launch_pos_m,
        config.scenario.threat_launch_vel_mps,
        config.sim.dt,
        N_STEPS,
    )
    return torch.from_numpy(np.ascontiguousarray(positions[1:]))


def _engine_positions(config, device: str, dtype: torch.dtype) -> torch.Tensor:
    state = make_initial(config, device, dtype=dtype)
    _, trajectory = rollout(state, PhysicsParams.from_config(config), N_STEPS, record=True)
    return trajectory[:, 0, 0, :].double().cpu()


def check_oracle_determinism() -> Result:
    """The oracle arbitrates every other check, so it must be reproducible first."""
    config = _config(1)
    args = (
        config.scenario.threat_launch_pos_m,
        config.scenario.threat_launch_vel_mps,
        config.sim.dt,
        N_STEPS,
    )
    pos_a, _ = propagate(*args)
    pos_b, _ = propagate(*args)
    deviation = float(np.max(np.abs(pos_a - pos_b)))
    return Result("oracle determinism", PASS if deviation == 0.0 else FAIL, deviation)


def check_oracle_vs_analytic() -> Result:
    """Drag-free range against the closed-form solution."""
    speed, theta = 1213.0, np.radians(45.0)
    vel0 = [speed * np.cos(theta), 0.0, speed * np.sin(theta)]
    expected_time = 2.0 * speed * np.sin(theta) / G
    expected_range = speed**2 * np.sin(2.0 * theta) / G

    n_steps = int(np.ceil((expected_time + 1.0) / 0.05))
    positions, _ = propagate([0.0, 0.0, 0.0], vel0, 0.05, n_steps)

    z = positions[:, 2]
    below = np.flatnonzero(z < 0.0)
    if below.size == 0:
        return Result("oracle vs analytic range", FAIL, None, "no ground crossing")

    k = int(below[0])
    frac = z[k - 1] / (z[k - 1] - z[k])
    impact = positions[k - 1, :2] + frac * (positions[k, :2] - positions[k - 1, :2])
    relative = abs(float(np.linalg.norm(impact)) - expected_range) / expected_range
    return Result("oracle vs analytic range", PASS if relative < 0.01 else FAIL,
                  relative, "relative error")


def check_engine_vs_oracle() -> Result:
    """Algorithm parity, float64 both sides."""
    config = _config(1)
    deviation = _relative_deviation(
        _engine_positions(config, "cpu", torch.float64), _oracle_positions(config)
    )
    return Result("engine N=1 vs oracle (f64)", PASS if deviation < RTOL else FAIL,
                  deviation, f"rtol {RTOL:g}")


def check_float32_floor() -> Result:
    """What the float32 run path costs against the same oracle, on the run device."""
    config = _config(1)
    deviation = _relative_deviation(
        _engine_positions(config, _DEVICE, torch.float32), _oracle_positions(config)
    )
    return Result(f"engine f32 on {_DEVICE} vs oracle",
                  PASS if deviation < FLOAT32_FLOOR else FAIL,
                  deviation, f"float32 floor {FLOAT32_FLOOR:g}")


def check_batch_invariance() -> Result:
    """Env i alone must match env i inside a batch of 1024, with varied launch states."""
    config = _config(1024)
    params = PhysicsParams.from_config(config)

    state = make_initial(config, _DEVICE)
    shape = (state.n_threats, 3)
    state = replace(
        state,
        threat_pos=state.threat_pos + normal(state.seed, 0, 11, shape) * 500.0,
        threat_vel=state.threat_vel + normal(state.seed, 0, 12, shape) * 20.0,
    )

    _, batched = rollout(state, params, N_STEPS, record=True)

    worst = 0.0
    for index in (0, 37, 512, 1023):
        solo_state = EnvState.from_dict(
            {name: t[index : index + 1].clone() for name, t in state.to_dict().items()}
        )
        _, solo = rollout(solo_state, params, N_STEPS, record=True)
        expected = batched[:, index].double().cpu()
        worst = max(worst, _relative_deviation(solo[:, 0].double().cpu(), expected))

    return Result("N=1 vs N=1024 agreement", PASS if worst < RTOL else FAIL,
                  worst, f"worst of 4 envs, rtol {RTOL:g}")


CHECKS = (
    check_oracle_determinism,
    check_oracle_vs_analytic,
    check_engine_vs_oracle,
    check_float32_floor,
    check_batch_invariance,
)


def main() -> int:
    global _DEVICE

    config = load_config(ROOT / "configs")
    _DEVICE = resolve_device(config.sim.device)

    results = [check() for check in CHECKS]

    width = max(len(r.name) for r in results)
    print(f"\nPhysics gate  (device={_DEVICE}, T={N_STEPS})")
    print("-" * (width + 44))
    for r in results:
        deviation = "-" if r.deviation is None else f"{r.deviation:.3e}"
        detail = f"  {r.detail}" if r.detail else ""
        print(f"{r.status:<5} {r.name:<{width}}  max dev {deviation:>10}{detail}")
    print("-" * (width + 44))

    failed = sum(r.status == FAIL for r in results)
    skipped = sum(r.status == SKIP for r in results)
    print(f"{len(results) - failed - skipped} passed, {failed} failed, {skipped} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
