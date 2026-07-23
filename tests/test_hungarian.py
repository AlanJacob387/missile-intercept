"""Hungarian assignment: matches an independent oracle, and beats greedy where
slot-threat coupling actually matters.

scipy.optimize.linear_sum_assignment is the oracle here specifically because
`hungarian` is not allowed to call it -- the two solvers have to agree despite being
built independently, or the hand-rolled Kuhn-Munkres implementation is wrong.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from scipy.optimize import linear_sum_assignment

from mdsim.core.config import BatterySpec, load_config
from mdsim.core.state import make_initial
from mdsim.engagement.assignment import UNASSIGNED, greedy, hungarian
from mdsim.envs.engine import EngineParams, step
from mdsim.eval.metrics import kills, value_destroyed

DTYPE = torch.float64
CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _check_invariants(
    result: torch.Tensor, assignable: torch.Tensor, available: torch.Tensor
) -> None:
    bound = result >= 0

    assert not bool((bound & ~available).any()), "bound an unavailable slot"

    picked_assignable = torch.gather(assignable, 1, result.clamp(min=0))
    assert not bool((bound & ~picked_assignable).any()), "bound a non-assignable threat"

    for env in range(result.shape[0]):
        chosen = result[env][bound[env]].tolist()
        assert len(chosen) == len(set(chosen)), f"threat bound twice in env {env}"


def _realized_total(value_env: torch.Tensor, result_env: torch.Tensor) -> float:
    """Sum of value at the (slot, threat) pairs a per-env result actually picked."""
    total = 0.0
    for slot, threat in enumerate(result_env.tolist()):
        if threat != UNASSIGNED:
            total += float(value_env[slot, threat])
    return total


def _scipy_optimum(
    value_env: torch.Tensor, assignable_env: torch.Tensor, available_env: torch.Tensor
) -> float:
    """Best achievable total value for one env, available slots and legal targets
    only -- restricted the same way `hungarian` restricts its own rows, so the two
    are optimising over the same feasible set."""
    if not bool(available_env.any()):
        return 0.0
    rows = value_env[available_env].numpy().copy()
    cost = -rows
    cost[:, ~assignable_env.numpy()] = 1e12
    row_ind, col_ind = linear_sum_assignment(cost)
    total = 0.0
    for r, c in zip(row_ind, col_ind):
        if assignable_env[c]:
            total += float(rows[r, c])
    return total


def test_matches_scipy_optimum_on_random_batches() -> None:
    torch.manual_seed(0)
    shapes = [
        (1, 2, 2),
        (1, 3, 5),  # I < T
        (1, 5, 3),  # I > T
        (2, 4, 4),  # I == T
        (3, 1, 1),
        (1, 6, 6),
    ]
    for n_envs, n_interceptors, n_threats in shapes:
        value = torch.rand((n_envs, n_interceptors, n_threats), dtype=DTYPE) + 0.01
        assignable = torch.ones((n_envs, n_threats), dtype=torch.bool)
        available = torch.ones((n_envs, n_interceptors), dtype=torch.bool)

        result = hungarian(value, assignable, available)
        assert result.shape == (n_envs, n_interceptors)
        _check_invariants(result, assignable, available)

        for env in range(n_envs):
            got = _realized_total(value[env], result[env])
            want = _scipy_optimum(value[env], assignable[env], available[env])
            assert got == pytest.approx(want), (n_envs, n_interceptors, n_threats, env)


def test_matches_scipy_optimum_with_masking() -> None:
    torch.manual_seed(1)
    n_envs, n_interceptors, n_threats = 4, 5, 6
    value = torch.rand((n_envs, n_interceptors, n_threats), dtype=DTYPE) + 0.01
    assignable = torch.rand((n_envs, n_threats)) > 0.4
    available = torch.rand((n_envs, n_interceptors)) > 0.4

    result = hungarian(value, assignable, available)
    _check_invariants(result, assignable, available)

    # A False `available` slot returns -1 always.
    assert bool((result[~available] == UNASSIGNED).all())

    for env in range(n_envs):
        got = _realized_total(value[env], result[env])
        want = _scipy_optimum(value[env], assignable[env], available[env])
        assert got == pytest.approx(want), env


def test_contested_case_hungarian_beats_greedy() -> None:
    """The case greedy is structurally blind to: slot0 pairs well with threat1,
    slot1 pairs well with threat0, and the cross pairs are bad.

    greedy only ever sees a per-threat priority, so it cannot express this coupling
    even in principle. Collapsing `value` the only way greedy's signature allows --
    the max over slots -- is what it would be handed if asked to decide here.
    """
    value = torch.tensor([[[1.0, 10.0], [10.0, 1.0]]], dtype=DTYPE)
    assignable = torch.ones((1, 2), dtype=torch.bool)
    available = torch.ones((1, 2), dtype=torch.bool)

    priority = value.amax(dim=1)
    greedy_result = greedy(priority, assignable, available)
    hungarian_result = hungarian(value, assignable, available)

    greedy_total = _realized_total(value[0], greedy_result[0])
    hungarian_total = _realized_total(value[0], hungarian_result[0])

    print(f"greedy total: {greedy_total}, hungarian total: {hungarian_total}")
    assert hungarian_total > greedy_total


@pytest.mark.slow
def test_multi_battery_engine_contested_case_hungarian_beats_greedy() -> None:
    """The same coupling, in the real engine rather than a hand-built value matrix.

    Two batteries: battery_wrong sits 500 km out, past pac3_mse's 70 km max
    envelope for every threat this raid can produce, so nothing launched from it
    can ever reach an intercept point -- should_launch's own range check from its
    actual position is what stops it, not any assignment logic. It is declared
    FIRST, so its slots get the LOW indices. battery_right sits at the origin, the
    position phase0_single already proves can score kills, declared second (high
    indices).

    greedy ranks available slots by raw index and pairs the j-th slot with the
    j-th ranked threat, blind to which battery a slot belongs to. With exactly as
    many threats as battery_wrong has slots, greedy's index-first allocation is
    entirely consumed by the unreachable battery -- those rounds commit and never
    launch -- and battery_right's slots are never even proposed, since greedy stops
    once n_assignable threats are covered. Hungarian's per-(slot, threat) value
    marks every battery_wrong pairing UNREACHABLE_VALUE (engine.py), so it routes
    the same threats to battery_right instead and actually kills some of them.
    """
    config = load_config(CONFIG_DIR, scenario="phase2_multi_battery")
    config = replace(config, sim=replace(config.sim, n_envs=256))
    batteries = (
        BatterySpec("battery_wrong", (500_000.0, 500_000.0, 0.0), 2),
        BatterySpec("battery_right", (0.0, 0.0, 0.0), 2),
    )
    config = replace(
        config,
        scenario=replace(config.scenario, batteries=batteries, n_interceptors=4, n_threats=2),
    )
    base = EngineParams.from_config(config, engage=True)

    damage: dict[str, float] = {}
    killed: dict[str, float] = {}
    for method in ("greedy", "hungarian"):
        params = replace(base, assignment_method=method)
        state = make_initial(config, "cpu", dtype=torch.float32, inventory=[2, 2])
        for _ in range(5000):
            state = step(state, params)
        damage[method] = value_destroyed(state)
        killed[method] = kills(state)

    print(
        f"greedy damage={damage['greedy']:.4f} kills={killed['greedy']:.0f}, "
        f"hungarian damage={damage['hungarian']:.4f} kills={killed['hungarian']:.0f}"
    )
    # value_destroyed is a loss metric -- lower is better defense.
    assert damage["hungarian"] < damage["greedy"]
    assert killed["hungarian"] > killed["greedy"]
