"""Magazine accounting and the commitment lifecycle.

The helpers are small enough to check directly, but the property that matters --
a binding holds until its engagement ends and never silently switches target -- is a
property of the loop, not of any one function. It is driven through engine.step here
for that reason.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch

from mdsim.core.config import load_config
from mdsim.core.state import make_initial
from mdsim.engagement.assignment import UNASSIGNED, greedy
from mdsim.engagement.inventory import available, commit, engaged_threats, release
from mdsim.envs.engine import EngineParams, step

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
DTYPE = torch.float64


def test_available_is_enabled_and_unspent() -> None:
    enabled = torch.tensor([[True, True, False, False]])
    committed = torch.tensor([[True, False, True, False]])

    assert available(enabled, committed).tolist() == [[False, True, False, False]]


def test_commit_binds_proposals_and_leaves_existing_bindings_alone() -> None:
    """A round already flying keeps its target: reassignment happens nowhere else."""
    existing = torch.tensor([[3, UNASSIGNED, 7]])
    proposal = torch.tensor([[UNASSIGNED, 2, 5]])

    # Slot 0 keeps 3, slot 1 takes the proposal, slot 2 would be overwritten -- but
    # assignment only ever proposes for available slots, so that case cannot arise
    # from the engine. The helper is deliberately unguarded; the guard is upstream.
    assert commit(existing, proposal).tolist() == [[3, 2, 5]]


def test_commit_is_a_noop_when_nothing_is_proposed() -> None:
    existing = torch.tensor([[3, UNASSIGNED, 7]])
    nothing = torch.full_like(existing, UNASSIGNED)

    assert torch.equal(commit(existing, nothing), existing)


def test_release_clears_exactly_the_named_slots() -> None:
    targets = torch.tensor([[3, 1, 7, UNASSIGNED]])
    released = torch.tensor([[True, False, True, True]])

    assert release(targets, released).tolist() == [[UNASSIGNED, 1, UNASSIGNED, UNASSIGNED]]


def test_engaged_threats_marks_bound_targets_only() -> None:
    targets = torch.tensor([[2, UNASSIGNED, 0]])

    assert engaged_threats(targets, 4).tolist() == [[True, False, True, False]]


def test_unbound_slots_do_not_mark_threat_zero() -> None:
    """The sentinel column exists so -1 does not read as a binding on threat 0."""
    targets = torch.full((1, 3), UNASSIGNED)

    assert engaged_threats(targets, 3).tolist() == [[False, False, False]]


def test_engaged_threats_handles_duplicate_bindings() -> None:
    """Two slots on one threat mark it once, not twice."""
    targets = torch.tensor([[1, 1, UNASSIGNED]])

    assert engaged_threats(targets, 3).tolist() == [[False, True, False]]


def test_bindings_never_exceed_stock() -> None:
    """Greedy plus commit cannot produce more bindings than there are enabled rounds."""
    torch.manual_seed(0)

    for _ in range(80):
        n_envs = int(torch.randint(1, 6, ()).item())
        n_threats = int(torch.randint(1, 7, ()).item())
        n_rounds = int(torch.randint(1, 7, ()).item())

        enabled = torch.rand((n_envs, n_rounds)) > 0.3
        committed = enabled & (torch.rand((n_envs, n_rounds)) > 0.6)
        targets = torch.where(
            committed,
            torch.randint(0, n_threats, (n_envs, n_rounds)),
            torch.full((n_envs, n_rounds), UNASSIGNED),
        )

        proposal = greedy(
            torch.rand((n_envs, n_threats), dtype=DTYPE),
            torch.rand((n_envs, n_threats)) > 0.3,
            available(enabled, committed),
        )
        bound = commit(targets, proposal) >= 0

        assert bool((bound.sum(dim=1) <= enabled.sum(dim=1)).all())


def _engagement_state(threat_pos, threat_vel, target_city=None, n_threats=1):
    """A one-env state with the threats hand-placed inside the engagement envelope."""
    config = load_config(CONFIG_DIR)
    config = replace(
        config,
        sim=replace(config.sim, n_envs=1),
        scenario=replace(
            config.scenario, n_threats=n_threats, n_interceptors=n_threats
        ),
    )
    state = make_initial(
        config, "cpu", dtype=DTYPE, n_active_threats=n_threats, inventory=n_threats
    )
    state = replace(
        state,
        threat_pos=torch.tensor([threat_pos], dtype=DTYPE),
        threat_vel=torch.tensor([threat_vel], dtype=DTYPE),
    )
    if target_city is not None:
        state = replace(
            state, threat_target_city=torch.tensor([target_city], dtype=torch.int64)
        )
    return state, EngineParams.from_config(config)


def _run(state, params, max_steps):
    """Advance until every threat resolves, recording binding transitions."""
    previous = state.interceptor_target.clone()
    record = {"bound": None, "released": None, "thrash": 0, "stopped": max_steps}

    for index in range(max_steps):
        state = step(state, params)
        current = state.interceptor_target

        if record["bound"] is None and bool((current >= 0).any()):
            record["bound"] = index
        switched = (previous >= 0) & (current >= 0) & (previous != current)
        record["thrash"] += int(switched.sum())
        if record["released"] is None and bool(((previous >= 0) & (current < 0)).any()):
            record["released"] = index

        previous = current.clone()
        if not bool(state.threat_alive.any()):
            record["stopped"] = index
            break

    return state, record


def test_commitment_holds_and_never_switches_target() -> None:
    """Two threats, two rounds, run to resolution: no binding may change target.

    A binding moving from one threat to another is the reassignment thrash the fixed
    decision interval exists to prevent, and it would not fail any other assertion --
    the counts stay legal throughout.
    """
    state, params = _engagement_state(
        threat_pos=[[20_000.0, 0.0, 15_000.0], [22_000.0, 9_000.0, 14_000.0]],
        threat_vel=[[-1_000.0, 0.0, -800.0], [-900.0, -400.0, -800.0]],
        n_threats=2,
    )
    final, record = _run(state, params, max_steps=700)

    assert record["bound"] is not None, "nothing was ever committed"
    assert record["released"] is not None, "no binding was ever released"
    assert record["thrash"] == 0, "a committed round changed target mid-flight"
    assert final.interceptor_target.tolist() == [[UNASSIGNED, UNASSIGNED]]


def test_binding_survives_a_full_decision_interval() -> None:
    """The binding made at the first decision is still in place at the next one."""
    state, params = _engagement_state(
        threat_pos=[[20_000.0, 0.0, 15_000.0]],
        threat_vel=[[-1_000.0, 0.0, -800.0]],
    )
    interval = params.decision_interval_steps

    state = step(state, params)
    first = state.interceptor_target.clone()
    assert bool((first >= 0).all()), "expected a binding at the first decision"

    for _ in range(interval + 1):
        state = step(state, params)

    assert torch.equal(state.interceptor_target, first)


def test_binding_releases_when_the_threat_leaks() -> None:
    """Aimed at the far city, too fast to reach: the round is released on the leak."""
    state, params = _engagement_state(
        threat_pos=[[25_000.0, 12_000.0, 9_000.0]],
        threat_vel=[[0.0, 0.0, -1_400.0]],
        target_city=[1],
    )
    final, record = _run(state, params, max_steps=400)

    assert bool(final.threat_leaked.any()), "the threat did not reach its city"
    assert record["released"] is not None
    assert final.interceptor_target.tolist() == [[UNASSIGNED]]


def test_stock_is_never_spent_below_zero_over_a_run() -> None:
    """Committed rounds can only ever be a subset of enabled ones."""
    state, params = _engagement_state(
        threat_pos=[[20_000.0, 0.0, 15_000.0], [22_000.0, 9_000.0, 14_000.0]],
        threat_vel=[[-1_000.0, 0.0, -800.0], [-900.0, -400.0, -800.0]],
        n_threats=2,
    )

    for _ in range(300):
        state = step(state, params)
        spent = state.interceptor_committed
        assert not bool((spent & ~state.interceptor_enabled).any()), "fired a dry slot"
        assert bool(
            (available(state.interceptor_enabled, spent).sum(dim=1) >= 0).all()
        )


def test_committed_count_stops_growing_once_the_only_threat_is_dead() -> None:
    """More rounds than threats: the extras must stay in the magazine.

    Before a dead threat's track dropped out of `detected`, it stayed "detected"
    forever, so once its binding released on death it looked assignable again and
    drew a second round -- and a third, until the magazine ran out. One threat,
    six rounds available: at most one round may ever be committed against it.
    """
    config = load_config(CONFIG_DIR)
    config = replace(
        config,
        sim=replace(config.sim, n_envs=1),
        scenario=replace(config.scenario, n_threats=1, n_interceptors=6),
    )
    state = make_initial(config, "cpu", dtype=DTYPE, n_active_threats=1, inventory=6)
    # Close, slow, and head-on: PN kills this inside the first decision interval.
    state = replace(
        state,
        threat_pos=torch.tensor([[[8_000.0, 0.0, 8_000.0]]], dtype=DTYPE),
        threat_vel=torch.tensor([[[-200.0, 0.0, -150.0]]], dtype=DTYPE),
    )
    params = EngineParams.from_config(config)

    for _ in range(400):
        state = step(state, params)
        if bool(state.threat_killed.any()):
            break
    assert bool(state.threat_killed.any()), "the threat was never killed"

    for _ in range(200):
        state = step(state, params)

    committed = int(state.interceptor_committed.sum())
    assert committed == 1, f"expected exactly one round spent, got {committed}"
    remaining = int(available(state.interceptor_enabled, state.interceptor_committed).sum())
    assert remaining == 5, f"expected 5 rounds left in the magazine, got {remaining}"
