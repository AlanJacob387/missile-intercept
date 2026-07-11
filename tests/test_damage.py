"""Leak resolution and binary city destruction.

Two things are easy to get wrong here and neither fails loudly. A second threat
arriving at a city already destroyed must still count as leakage while adding no
value, or the damage total double-counts. And an inactive slot sitting on top of a
city must score nothing, or every raid smaller than the slot count reports damage it
never took.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch

from mdsim.core.config import load_config
from mdsim.core.state import make_initial
from mdsim.envs.engine import EngineParams, step
from mdsim.eval import metrics
from mdsim.world.damage import damage_value, leakage_count, reached_city, resolve_leaks

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
DTYPE = torch.float64
DT = 0.05
RADIUS_M = 100.0

CITY_POS = torch.tensor([[[0.0, 0.0, 0.0], [1_000.0, 0.0, 0.0]]], dtype=DTYPE)
CITY_VALUE = torch.tensor([[1.0, 0.6]], dtype=DTYPE)


def _resolve(before, after, target_city, active, alive, leaked, city_alive=None):
    """resolve_leaks over one env, with the shared two-city layout."""
    if city_alive is None:
        city_alive = torch.ones((1, 2), dtype=torch.bool)
    return resolve_leaks(
        torch.tensor([before], dtype=DTYPE),
        torch.tensor([after], dtype=DTYPE),
        torch.tensor([target_city], dtype=torch.int64),
        torch.tensor([active]),
        torch.tensor([alive]),
        torch.tensor([leaked]),
        CITY_POS,
        city_alive,
        RADIUS_M,
        DT,
    )


def test_endpoints_alone_would_miss_the_impact() -> None:
    """Establishes the premise for the test below."""
    before = torch.tensor([[[-400.0, 50.0, 0.0]]], dtype=DTYPE)
    after = torch.tensor([[[400.0, 50.0, 0.0]]], dtype=DTYPE)
    city = CITY_POS[:, 0]

    assert float((before[0, 0] - city[0]).norm()) > RADIUS_M
    assert float((after[0, 0] - city[0]).norm()) > RADIUS_M


def test_reached_city_finds_the_sub_timestep_pass() -> None:
    """Closest approach lands mid-step at 50 m, inside a 100 m footprint."""
    before = torch.tensor([[[-400.0, 50.0, 0.0]]], dtype=DTYPE)
    after = torch.tensor([[[400.0, 50.0, 0.0]]], dtype=DTYPE)
    target = torch.tensor([[0]], dtype=torch.int64)

    assert reached_city(before, after, target, CITY_POS, RADIUS_M, DT).tolist() == [[True]]


def test_a_clean_miss_is_not_a_leak() -> None:
    before = torch.tensor([[[-400.0, 5_000.0, 0.0]]], dtype=DTYPE)
    after = torch.tensor([[[400.0, 5_000.0, 0.0]]], dtype=DTYPE)
    target = torch.tensor([[0]], dtype=torch.int64)

    assert reached_city(before, after, target, CITY_POS, RADIUS_M, DT).tolist() == [[False]]


def test_each_threat_is_scored_against_its_own_city() -> None:
    """Sitting on city 0 is not an impact for a threat aimed at city 1."""
    leaked, destroyed = _resolve(
        before=[[0.0, 0.0, 10.0]],
        after=[[0.0, 0.0, -10.0]],
        target_city=[1],
        active=[True],
        alive=[True],
        leaked=[False],
    )

    assert leaked.tolist() == [[False]]
    assert destroyed.tolist() == [[False, False]]


def test_unopposed_threat_destroys_its_city_and_scores_its_value() -> None:
    """Full engine path with an empty magazine: nothing engages, the city falls."""
    config = load_config(CONFIG_DIR)
    config = replace(
        config,
        sim=replace(config.sim, n_envs=1),
        scenario=replace(config.scenario, n_threats=1, n_interceptors=1),
    )
    state = make_initial(config, "cpu", dtype=DTYPE, n_active_threats=1, inventory=0)
    state = replace(
        state,
        threat_pos=torch.tensor([[[0.0, 0.0, 8_000.0]]], dtype=DTYPE),
        threat_vel=torch.tensor([[[0.0, 0.0, -800.0]]], dtype=DTYPE),
    )
    params = EngineParams.from_config(config)
    target_value = float(state.city_value[0, 0])

    for _ in range(400):
        state = step(state, params)
        if not bool(state.threat_alive.any()):
            break

    assert bool(state.threat_leaked[0, 0]), "the threat never reached its city"
    assert not bool(state.interceptor_committed.any()), "an empty magazine fired"
    assert not bool(state.city_alive[0, 0]), "the city survived a direct impact"
    assert metrics.value_destroyed(state) == target_value


def test_killed_threat_scores_nothing() -> None:
    """Same geometry with a round available: the city is not touched."""
    config = load_config(CONFIG_DIR)
    config = replace(
        config,
        sim=replace(config.sim, n_envs=1),
        scenario=replace(config.scenario, n_threats=1, n_interceptors=1),
    )
    state = make_initial(config, "cpu", dtype=DTYPE, n_active_threats=1, inventory=1)
    state = replace(
        state,
        threat_pos=torch.tensor([[[0.0, 0.0, 8_000.0]]], dtype=DTYPE),
        threat_vel=torch.tensor([[[0.0, 0.0, -800.0]]], dtype=DTYPE),
    )
    params = EngineParams.from_config(config)

    for _ in range(400):
        state = step(state, params)
        if not bool(state.threat_alive.any()):
            break

    assert bool(state.threat_killed[0, 0]), "the round did not intercept"
    assert not bool(state.threat_leaked[0, 0])
    assert bool(state.city_alive.all())
    assert metrics.value_destroyed(state) == 0.0


def test_two_threats_on_one_city_destroy_it_once() -> None:
    """Simultaneous arrivals: both leak, the value is scored once."""
    leaked, destroyed = _resolve(
        before=[[0.0, 0.0, 10.0], [0.0, 20.0, 10.0]],
        after=[[0.0, 0.0, -10.0], [0.0, 20.0, -10.0]],
        target_city=[0, 0],
        active=[True, True],
        alive=[True, True],
        leaked=[False, False],
    )

    assert leaked.tolist() == [[True, True]]
    assert destroyed.tolist() == [[True, False]]

    city_alive_after = torch.ones((1, 2), dtype=torch.bool) & ~destroyed
    assert damage_value(CITY_VALUE, torch.ones((1, 2), dtype=torch.bool), city_alive_after).tolist() == [1.0]


def test_second_arrival_at_a_dead_city_leaks_but_adds_no_value() -> None:
    """Sequential arrivals: leakage still counts, damage does not double-score."""
    already_dead = torch.tensor([[False, True]])

    leaked, destroyed = _resolve(
        before=[[0.0, 0.0, 10.0]],
        after=[[0.0, 0.0, -10.0]],
        target_city=[0],
        active=[True],
        alive=[True],
        leaked=[False],
        city_alive=already_dead,
    )

    assert leaked.tolist() == [[True]], "leakage must count regardless of city state"
    assert destroyed.tolist() == [[False, False]], "a dead city cannot be destroyed twice"
    assert damage_value(CITY_VALUE, already_dead, already_dead & ~destroyed).tolist() == [0.0]


def test_inactive_slot_on_top_of_a_city_never_leaks() -> None:
    """The slot is geometrically on target; the active mask is the only thing stopping it."""
    geometry = ([[0.0, 0.0, 10.0]], [[0.0, 0.0, -10.0]])
    target = torch.tensor([[0]], dtype=torch.int64)

    # The geometry does register as an impact -- the mask is doing the work.
    assert reached_city(
        torch.tensor([geometry[0]], dtype=DTYPE),
        torch.tensor([geometry[1]], dtype=DTYPE),
        target,
        CITY_POS,
        RADIUS_M,
        DT,
    ).tolist() == [[True]]

    leaked, destroyed = _resolve(
        before=geometry[0],
        after=geometry[1],
        target_city=[0],
        active=[False],
        alive=[True],
        leaked=[False],
    )

    assert leaked.tolist() == [[False]]
    assert destroyed.tolist() == [[False, False]]


def test_already_leaked_threat_does_not_leak_again() -> None:
    leaked, destroyed = _resolve(
        before=[[0.0, 0.0, 10.0]],
        after=[[0.0, 0.0, -10.0]],
        target_city=[0],
        active=[True],
        alive=[True],
        leaked=[True],
    )

    assert leaked.tolist() == [[False]]
    assert destroyed.tolist() == [[False, False]]


def test_dead_threat_does_not_leak() -> None:
    leaked, _destroyed = _resolve(
        before=[[0.0, 0.0, 10.0]],
        after=[[0.0, 0.0, -10.0]],
        target_city=[0],
        active=[True],
        alive=[False],
        leaked=[False],
    )

    assert leaked.tolist() == [[False]]


def test_leakage_count_excludes_inactive_slots() -> None:
    leaked = torch.tensor([[True, True, True]])
    active = torch.tensor([[True, False, True]])

    assert leakage_count(leaked, active).tolist() == [2]
