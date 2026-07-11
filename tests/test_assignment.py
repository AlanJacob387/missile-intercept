"""Greedy assignment invariants, and the tie-break the oracle has to reproduce.

Greedy is solved batch-wide with a sort and a gather rather than a loop, so the
failure modes are not the ones a loop would have: an off-by-one in the slot ranking
double-books a threat, and an unstable sort makes the whole assignment
backend-dependent. Both are asserted here rather than inferred from a passing run.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch

from mdsim.core.config import load_config
from mdsim.core.interceptor import InterceptorParams
from mdsim.core.state import make_initial
from mdsim.engagement.assignment import UNASSIGNED, greedy
from mdsim.engagement.envelopes import in_envelope
from mdsim.envs.engine import EngineParams, step

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"

DTYPE = torch.float64

# A clean band to test the cutoffs against, rather than whichever arsenal entry the
# scenario happens to name.
PARAMS = InterceptorParams(
    speed_mps=1700.0,
    max_accel_mps2=294.2,
    envelope_min_m=3_000.0,
    envelope_max_m=70_000.0,
    envelope_alt_min_m=500.0,
    envelope_alt_max_m=24_000.0,
    reaction_time_s=5.0,
    kill_radius_m=10.0,
)
BATTERY = torch.zeros(3, dtype=DTYPE)


def _check_invariants(
    result: torch.Tensor, assignable: torch.Tensor, available: torch.Tensor
) -> None:
    """The four properties greedy must hold for any input."""
    bound = result >= 0

    assert not bool((bound & ~available).any()), "bound an unavailable slot"

    picked_assignable = torch.gather(assignable, 1, result.clamp(min=0))
    assert not bool((bound & ~picked_assignable).any()), "bound a non-assignable threat"

    assert bool((bound.sum(dim=1) <= available.sum(dim=1)).all()), "exceeded stock"
    assert bool((bound.sum(dim=1) <= assignable.sum(dim=1)).all()), "exceeded targets"

    # Double-booking is per environment, so it needs a per-row uniqueness check. A
    # loop is fine in a test; the point is that the batched implementation cannot do
    # this, not that the check itself is fast.
    for env in range(result.shape[0]):
        chosen = result[env][bound[env]].tolist()
        assert len(chosen) == len(set(chosen)), f"threat bound twice in env {env}"


def test_hand_case_matches_known_greedy_answer() -> None:
    """Distinct priorities, one slot already spent: the pairing is fully determined."""
    priority = torch.tensor([[0.1, 0.9, 0.5, 0.0]], dtype=DTYPE)
    assignable = torch.tensor([[True, True, True, False]])
    available = torch.tensor([[True, False, True, True]])

    result = greedy(priority, assignable, available)

    # Ranked targets are [1, 2, 0]; available slots are 0, 2, 3 in that order.
    assert result.tolist() == [[1, UNASSIGNED, 2, 0]]
    _check_invariants(result, assignable, available)


def test_tie_breaks_toward_lower_threat_index() -> None:
    """Load-bearing: the oracle's independent argmax must break ties the same way.

    An unstable sort would leave this free to differ between backends, and the
    engine-oracle comparison would be comparing two defensible answers.
    """
    priority = torch.tensor([[0.5, 0.5, 0.1]], dtype=DTYPE)
    assignable = torch.tensor([[True, True, True]])
    available = torch.tensor([[True, False, False]])

    result = greedy(priority, assignable, available)
    assert result[0, 0].item() == 0


def test_tie_break_holds_across_the_whole_ranking() -> None:
    """Every tied group orders by index, not just the first."""
    priority = torch.tensor([[0.5, 0.5, 0.5, 0.5]], dtype=DTYPE)
    assignable = torch.ones((1, 4), dtype=torch.bool)
    available = torch.ones((1, 4), dtype=torch.bool)

    assert greedy(priority, assignable, available).tolist() == [[0, 1, 2, 3]]


def test_more_slots_than_threats_leaves_the_surplus_unbound() -> None:
    priority = torch.tensor([[0.9, 0.2, 0.0, 0.0]], dtype=DTYPE)
    assignable = torch.tensor([[True, True, False, False]])
    available = torch.ones((1, 4), dtype=torch.bool)

    result = greedy(priority, assignable, available)
    assert result.tolist() == [[0, 1, UNASSIGNED, UNASSIGNED]]
    _check_invariants(result, assignable, available)


def test_more_threats_than_slots_takes_the_highest_priority() -> None:
    priority = torch.tensor([[0.1, 0.9, 0.5, 0.7]], dtype=DTYPE)
    assignable = torch.ones((1, 4), dtype=torch.bool)
    available = torch.tensor([[True, True, False, False]])

    result = greedy(priority, assignable, available)
    assert result.tolist() == [[1, 3, UNASSIGNED, UNASSIGNED]]


def test_no_stock_binds_nothing() -> None:
    priority = torch.tensor([[0.9, 0.5]], dtype=DTYPE)
    assignable = torch.ones((1, 2), dtype=torch.bool)
    available = torch.zeros((1, 2), dtype=torch.bool)

    assert greedy(priority, assignable, available).tolist() == [[UNASSIGNED, UNASSIGNED]]


def test_no_assignable_threats_binds_nothing() -> None:
    priority = torch.tensor([[0.9, 0.5]], dtype=DTYPE)
    assignable = torch.zeros((1, 2), dtype=torch.bool)
    available = torch.ones((1, 2), dtype=torch.bool)

    assert greedy(priority, assignable, available).tolist() == [[UNASSIGNED, UNASSIGNED]]


def test_environments_are_solved_independently() -> None:
    """One env running dry must not affect its neighbour's pairing."""
    priority = torch.tensor([[0.1, 0.9], [0.9, 0.1]], dtype=DTYPE)
    assignable = torch.ones((2, 2), dtype=torch.bool)
    available = torch.tensor([[True, True], [True, False]])

    result = greedy(priority, assignable, available)
    assert result.tolist() == [[1, 0], [0, UNASSIGNED]]


def test_out_of_envelope_tracks_are_not_engageable() -> None:
    """Range low, range high, altitude high, altitude low -- each cutoff separately."""
    track_pos = torch.tensor(
        [
            [
                [10_000.0, 0.0, 10_000.0],  # inside both bands
                [1_000.0, 0.0, 1_000.0],  # slant 1.4 km, under the 3 km floor
                [80_000.0, 0.0, 10_000.0],  # slant 80 km, over the 70 km ceiling
                [10_000.0, 0.0, 30_000.0],  # 30 km up, over the 24 km ceiling
                [10_000.0, 0.0, 100.0],  # 100 m up, under the 500 m floor
            ]
        ],
        dtype=DTYPE,
    )

    assert in_envelope(track_pos, BATTERY, PARAMS).tolist() == [
        [True, False, False, False, False]
    ]


def test_greedy_never_binds_an_out_of_envelope_threat() -> None:
    """The envelope mask has to reach greedy through `assignable`, not by luck."""
    track_pos = torch.tensor(
        [[[80_000.0, 0.0, 10_000.0], [10_000.0, 0.0, 10_000.0]]], dtype=DTYPE
    )
    engageable = in_envelope(track_pos, BATTERY, PARAMS)

    # The unreachable threat carries the higher score, so a missing mask shows up.
    priority = torch.tensor([[0.9, 0.1]], dtype=DTYPE)
    available = torch.ones((1, 2), dtype=torch.bool)

    result = greedy(priority, engageable, available)
    assert result.tolist() == [[1, UNASSIGNED]]


def test_invariants_hold_across_randomised_batches() -> None:
    """Shapes and masks the hand cases do not reach, including empty rows."""
    torch.manual_seed(0)

    for trial in range(120):
        n_envs = int(torch.randint(1, 9, ()).item())
        n_threats = int(torch.randint(1, 7, ()).item())
        n_interceptors = int(torch.randint(1, 7, ()).item())

        priority = torch.rand((n_envs, n_threats), dtype=DTYPE)
        assignable = torch.rand((n_envs, n_threats)) > 0.4
        available = torch.rand((n_envs, n_interceptors)) > 0.4

        result = greedy(priority, assignable, available)
        assert result.shape == (n_envs, n_interceptors), f"trial {trial}"
        _check_invariants(result, assignable, available)


def test_repeated_priorities_across_a_batch_stay_deterministic() -> None:
    """Same input twice, same answer -- ties included."""
    torch.manual_seed(1)
    priority = torch.randint(0, 3, (6, 5)).to(DTYPE)
    assignable = torch.rand((6, 5)) > 0.3
    available = torch.rand((6, 5)) > 0.3

    first = greedy(priority, assignable, available)
    second = greedy(priority, assignable, available)
    assert torch.equal(first, second)


def test_no_binding_ever_points_at_a_dead_threat() -> None:
    """A live commitment always points at a threat still in the fight.

    Assignment never reads threat_killed or threat_leaked -- it has no such
    parameter, and cannot -- so this exercises the real engine loop and reads truth
    only to check the invariant, not to enforce it. What actually excludes a dead
    threat is that its track stops being detected the step it dies; if that stopped
    working, a later decision could rebind a fresh round onto wreckage.
    """
    config = load_config(CONFIG_DIR)
    config = replace(
        config,
        sim=replace(config.sim, n_envs=1),
        scenario=replace(config.scenario, n_threats=3, n_interceptors=6),
    )
    state = make_initial(config, "cpu", dtype=DTYPE, n_active_threats=3, inventory=6)
    params = EngineParams.from_config(config)

    checked_a_dead_threat = False
    for _ in range(4600):
        state = step(state, params)
        bound = state.interceptor_target
        live = bound >= 0
        if not bool(live.any()):
            if not bool(state.threat_alive.any()):
                break
            continue

        target = bound.clamp(min=0)
        bound_killed = torch.gather(state.threat_killed, 1, target) & live
        bound_leaked = torch.gather(state.threat_leaked, 1, target) & live
        assert not bool(bound_killed.any()), "a live binding points at a killed threat"
        assert not bool(bound_leaked.any()), "a live binding points at a leaked threat"
        checked_a_dead_threat = checked_a_dead_threat or bool(
            state.threat_killed.any() or state.threat_leaked.any()
        )

    assert checked_a_dead_threat, "no threat died during the run -- the check never fired"
