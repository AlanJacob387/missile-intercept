"""Salvo doctrine: up to k rounds per threat per decision, and the inventory change
that lets a threat stay assignable until it holds a full salvo.

`salvo` at k=1 has to be exactly `greedy` -- the engine calls `salvo` unconditionally
with `salvo_size` from config, defaulting to 1, so any drift here is a Phase 1
regression, not a Phase 2 feature gap.
"""

from __future__ import annotations

import torch

from mdsim.engagement import doctrine
from mdsim.engagement.assignment import UNASSIGNED, greedy
from mdsim.engagement.inventory import commit, engaged_threats

DTYPE = torch.float64


def test_salvo_binds_up_to_salvo_size_slots_to_one_threat() -> None:
    """One threat, three free slots, salvo_size=2: exactly two slots bind, one stays free."""
    priority = torch.tensor([[1.0]], dtype=DTYPE)
    assignable = torch.tensor([[True]])
    available = torch.tensor([[True, True, True]])

    result = doctrine.salvo(priority, assignable, available, salvo_size=2)

    bound = result >= 0
    assert int(bound.sum()) == 2, f"expected 2 bindings, got {result.tolist()}"
    assert bool((result[bound] == 0).all()), "a bound slot points somewhere other than threat 0"
    assert int((result == UNASSIGNED).sum()) == 1, "expected exactly one slot left unbound"


def test_salvo_result_composes_with_commit() -> None:
    """The two committed slots from a salvo land in `interceptor_target` as expected."""
    priority = torch.tensor([[1.0]], dtype=DTYPE)
    assignable = torch.tensor([[True]])
    available = torch.tensor([[True, True, True]])
    interceptor_target = torch.full((1, 3), UNASSIGNED)

    proposal = doctrine.salvo(priority, assignable, available, salvo_size=2)
    committed = commit(interceptor_target, proposal)

    assert int((committed == 0).sum()) == 2
    assert int((committed == UNASSIGNED).sum()) == 1


def test_salvo_size_one_matches_greedy_on_hand_cases() -> None:
    cases = [
        (
            torch.tensor([[0.1, 0.9, 0.5, 0.0]], dtype=DTYPE),
            torch.tensor([[True, True, True, False]]),
            torch.tensor([[True, False, True, True]]),
        ),
        (
            torch.tensor([[0.5, 0.5, 0.1]], dtype=DTYPE),
            torch.tensor([[True, True, True]]),
            torch.tensor([[True, False, False]]),
        ),
        (
            torch.tensor([[0.9, 0.5]], dtype=DTYPE),
            torch.zeros((1, 2), dtype=torch.bool),
            torch.ones((1, 2), dtype=torch.bool),
        ),
    ]
    for priority, assignable, available in cases:
        salvo_result = doctrine.salvo(priority, assignable, available, salvo_size=1)
        greedy_result = greedy(priority, assignable, available)
        assert torch.equal(salvo_result, greedy_result)


def test_salvo_size_one_matches_greedy_across_randomised_batches() -> None:
    """Property test: salvo(size=1) must be bit-for-bit greedy, any shape or mask."""
    torch.manual_seed(0)

    for trial in range(100):
        n_envs = int(torch.randint(1, 9, ()).item())
        n_threats = int(torch.randint(1, 7, ()).item())
        n_interceptors = int(torch.randint(1, 7, ()).item())

        priority = torch.rand((n_envs, n_threats), dtype=DTYPE)
        assignable = torch.rand((n_envs, n_threats)) > 0.4
        available = torch.rand((n_envs, n_interceptors)) > 0.4

        salvo_result = doctrine.salvo(priority, assignable, available, salvo_size=1)
        greedy_result = greedy(priority, assignable, available)
        assert torch.equal(salvo_result, greedy_result), f"trial {trial}"


def test_engaged_threats_default_salvo_size_matches_phase_one_behaviour() -> None:
    """salvo_size defaults to 1, so one bound slot is enough to mark a threat engaged."""
    targets = torch.tensor([[2, UNASSIGNED, 0]])

    assert engaged_threats(targets, 4).tolist() == [[True, False, True, False]]


def test_engaged_threats_waits_for_a_full_salvo_before_marking_engaged() -> None:
    """One slot bound to threat 0: engaged at salvo_size=1, not yet at salvo_size=2."""
    interceptor_target = torch.tensor([[0, UNASSIGNED, UNASSIGNED]])
    n_threats = 2

    assert engaged_threats(interceptor_target, n_threats, salvo_size=1).tolist() == [
        [True, False]
    ]
    assert engaged_threats(interceptor_target, n_threats, salvo_size=2).tolist() == [
        [False, False]
    ]

    # A second slot binds to the same threat: now salvo_size=2 reads it engaged too.
    interceptor_target = torch.tensor([[0, 0, UNASSIGNED]])
    assert engaged_threats(interceptor_target, n_threats, salvo_size=2).tolist() == [
        [True, False]
    ]


def test_shoot_look_shoot_is_documented_as_not_yet_implemented() -> None:
    """Guards against the hook text silently disappearing or getting marked done."""
    text = doctrine.salvo.__doc__.lower()
    assert "shoot-look-shoot" in text
    assert "not implement" in text

    module_text = doctrine.__doc__.lower()
    assert "shoot-look-shoot" in module_text
