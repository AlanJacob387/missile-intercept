"""Sub-timestep closest approach: the tunneling case that endpoint testing misses.

At 6 km/s closing and dt = 0.05 s the pair advances 300 m per step. A kill radius of
10 m is 3% of one step, so an intercept that passes dead centre begins and ends the
step ~150 m apart. Endpoint-only testing does not merely lose accuracy here; it
fails to detect the engagement at all.
"""

from __future__ import annotations

import torch

from mdsim.core.intercept import check_hits, closest_approach

DTYPE = torch.float64
DT = 0.05
KILL_RADIUS_M = 10.0

# Crossing at 3 km/s each, head-on components opposed: 6 km/s closing.
THREAT_VEL = (3000.0, 0.0, 0.0)
INTERCEPTOR_VEL = (-3000.0, 0.0, 0.0)

# Offsets chosen so closest approach lands mid-step at 5 m -- inside the kill radius --
# while both endpoints sit ~150 m apart.
MISS_OFFSET_M = 5.0
START_LEAD_M = 150.0


def _tunneling_geometry() -> dict[str, torch.Tensor]:
    threat_before = torch.tensor([[[0.0, 0.0, 0.0]]], dtype=DTYPE)
    interceptor_before = torch.tensor(
        [[[START_LEAD_M, MISS_OFFSET_M, 0.0]]], dtype=DTYPE
    )
    threat_step = torch.tensor([[list(THREAT_VEL)]], dtype=DTYPE) * DT
    interceptor_step = torch.tensor([[list(INTERCEPTOR_VEL)]], dtype=DTYPE) * DT
    return {
        "threat_before": threat_before,
        "threat_after": threat_before + threat_step,
        "interceptor_before": interceptor_before,
        "interceptor_after": interceptor_before + interceptor_step,
    }


def test_endpoints_alone_would_miss_the_intercept() -> None:
    """Establishes the premise: without this, the test below proves nothing."""
    geometry = _tunneling_geometry()

    start_gap = (geometry["interceptor_before"] - geometry["threat_before"]).norm()
    end_gap = (geometry["interceptor_after"] - geometry["threat_after"]).norm()

    assert float(start_gap) > KILL_RADIUS_M
    assert float(end_gap) > KILL_RADIUS_M
    # Both endpoints are an order of magnitude outside the lethal radius.
    assert float(start_gap) > 10.0 * KILL_RADIUS_M
    assert float(end_gap) > 10.0 * KILL_RADIUS_M


def test_closest_approach_finds_the_mid_step_pass() -> None:
    geometry = _tunneling_geometry()

    delta_pos = geometry["interceptor_before"] - geometry["threat_before"]
    threat_step = geometry["threat_after"] - geometry["threat_before"]
    interceptor_step = geometry["interceptor_after"] - geometry["interceptor_before"]
    delta_vel = (interceptor_step - threat_step) / DT

    # Confirm the closing speed is the 6 km/s the docstring claims.
    assert float(delta_vel.norm()) == 6000.0

    distance, tau = closest_approach(delta_pos, delta_vel, DT)

    assert float(distance) < KILL_RADIUS_M
    assert float(distance) == 5.0
    assert 0.0 < float(tau) < DT
    assert float(tau) == 0.025


def test_check_hits_registers_the_tunneling_intercept() -> None:
    geometry = _tunneling_geometry()
    alive = torch.ones((1, 1), dtype=torch.bool)

    hit, passed = check_hits(
        geometry["threat_before"],
        geometry["threat_after"],
        geometry["interceptor_before"],
        geometry["interceptor_after"],
        alive,
        alive,
        DT,
        KILL_RADIUS_M,
    )

    assert bool(hit[0, 0, 0])
    assert not bool(passed[0, 0, 0])


def test_check_hits_marks_a_near_miss_as_passed() -> None:
    """A pass that closes inside the step but stays outside the radius ends the round."""
    threat_before = torch.tensor([[[0.0, 0.0, 0.0]]], dtype=DTYPE)
    interceptor_before = torch.tensor([[[START_LEAD_M, 50.0, 0.0]]], dtype=DTYPE)
    threat_step = torch.tensor([[list(THREAT_VEL)]], dtype=DTYPE) * DT
    interceptor_step = torch.tensor([[list(INTERCEPTOR_VEL)]], dtype=DTYPE) * DT
    alive = torch.ones((1, 1), dtype=torch.bool)

    hit, passed = check_hits(
        threat_before,
        threat_before + threat_step,
        interceptor_before,
        interceptor_before + interceptor_step,
        alive,
        alive,
        DT,
        KILL_RADIUS_M,
    )

    assert not bool(hit[0, 0, 0])
    assert bool(passed[0, 0, 0])


def test_dead_bodies_cannot_be_hit() -> None:
    geometry = _tunneling_geometry()
    alive = torch.ones((1, 1), dtype=torch.bool)
    dead = torch.zeros((1, 1), dtype=torch.bool)

    hit, _ = check_hits(
        geometry["threat_before"],
        geometry["threat_after"],
        geometry["interceptor_before"],
        geometry["interceptor_after"],
        dead,
        alive,
        DT,
        KILL_RADIUS_M,
    )
    assert not bool(hit[0, 0, 0])


def test_receding_pair_clamps_to_the_step_start() -> None:
    """When separation only grows, the minimum is at tau = 0, not inside the step."""
    delta_pos = torch.tensor([[[100.0, 0.0, 0.0]]], dtype=DTYPE)
    delta_vel = torch.tensor([[[6000.0, 0.0, 0.0]]], dtype=DTYPE)

    distance, tau = closest_approach(delta_pos, delta_vel, DT)
    assert float(tau) == 0.0
    assert float(distance) == 100.0
