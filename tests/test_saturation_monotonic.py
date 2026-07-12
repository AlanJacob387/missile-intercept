"""Saturation has a shape, and this pins it.

Two claims, one per axis. More rounds must never let more threats through; a larger
raid against fixed stock must never let fewer through. Both are asserted non-strictly,
because plateaus are real: once every round is already committed, two extra rounds that
arrive after the last engagement window change nothing, and a strict assertion would
fail on correct behaviour.

The failure this exists to catch is a FLAT curve. Flat means adding interceptors
changes nothing, which means assignment, launch or inventory is broken and the
engagement is not resolving at all -- a state in which every other test here still
passes. So the extremes are compared strictly as well.

Scale is deliberately small. The statistical claim belongs to scripts/run_saturation.py
at 4096 envs; this guards the shape at a size the suite can afford.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from mdsim.core.config import load_config
from mdsim.core.state import make_initial
from mdsim.envs.engine import EngineParams, step
from mdsim.eval.metrics import leakage

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"

N_ENVS = 64
THREAT_SLOTS = 8
INTERCEPTOR_SLOTS = 8

# Threats fly from burnout to ground over roughly 4,200 steps at dt=0.05, and nothing
# is engageable until they descend through the interceptor's altitude ceiling near
# step 3,340. A horizon short of impact reports threats as neither killed nor leaked,
# which reads as zero leakage everywhere and passes every assertion vacuously.
MAX_STEPS = 4600
IDLE_CHECK_EVERY = 200

# Held fixed while the other axis varies.
FIXED_RAID = 4
FIXED_INVENTORY = 2

INVENTORY_AXIS = (0, 2, 8)
RAID_AXIS = (2, 4, 8)

# 64 envs x 4 threats is 256 outcomes, so the binomial standard error near p = 0.9 is
# about 0.02. A real drop has to clear sampling noise by a comfortable margin.
MIN_EXTREME_DROP = 0.05

# Nothing engages at zero stock, so every active threat must arrive. Anything less
# means threats are not reaching their cities and the run is measuring the wrong thing.
UNDEFENDED_LEAKAGE = 0.99


def _leakage(raid: int, inventory: int) -> float:
    """Run one batched raid to resolution and report the leaked fraction.

    CPU rather than the resolved device: at this batch size the accelerator is
    launch-bound and slower, and a test should not contend with whatever else is
    using it.
    """
    config = load_config(CONFIG_DIR)
    config = replace(
        config,
        sim=replace(config.sim, n_envs=N_ENVS),
        scenario=replace(
            config.scenario,
            n_threats=THREAT_SLOTS,
            n_interceptors=INTERCEPTOR_SLOTS,
        ),
    )
    params = EngineParams.from_config(config)

    state = make_initial(
        config,
        "cpu",
        dtype=torch.float32,
        n_active_threats=raid,
        inventory=inventory,
    )
    for index in range(MAX_STEPS):
        state = step(state, params)
        if (index + 1) % IDLE_CHECK_EVERY == 0 and not bool(state.threat_alive.any()):
            break
    return leakage(state)


@pytest.fixture(scope="module")
def grid() -> dict[tuple[int, int], float]:
    """Every cell both axes need, computed once.

    The two axes share the (FIXED_RAID, FIXED_INVENTORY) cell, so five runs cover
    six points.
    """
    cells = {(FIXED_RAID, inv) for inv in INVENTORY_AXIS}
    cells |= {(raid, FIXED_INVENTORY) for raid in RAID_AXIS}
    measured = {cell: _leakage(*cell) for cell in sorted(cells)}

    # Printed so a passing run still shows the curve. The assertions below quote the
    # numbers they fail on, but a shape that is drifting toward flat is worth seeing
    # before it crosses the threshold.
    for (raid, inventory), value in measured.items():
        print(f"raid {raid:>2} inventory {inventory:>2}  leakage {value:.4f}")
    return measured


def test_undefended_raid_leaks_almost_entirely(grid: dict[tuple[int, int], float]) -> None:
    """Guards every other assertion here against a truncated horizon."""
    undefended = grid[(FIXED_RAID, 0)]
    assert undefended >= UNDEFENDED_LEAKAGE, (
        f"leakage with no interceptors is {undefended:.4f}, not ~1.0 -- threats are "
        "not reaching their cities, so the run is too short or targeting is wrong"
    )


def test_leakage_does_not_rise_with_inventory(grid: dict[tuple[int, int], float]) -> None:
    values = [grid[(FIXED_RAID, inv)] for inv in INVENTORY_AXIS]
    for (low_inv, low), (high_inv, high) in zip(
        zip(INVENTORY_AXIS, values), zip(INVENTORY_AXIS[1:], values[1:])
    ):
        assert high <= low + 1e-9, (
            f"raid {FIXED_RAID}: leakage rose from {low:.4f} at inventory {low_inv} "
            f"to {high:.4f} at {high_inv}"
        )


def test_more_inventory_strictly_helps_at_the_extremes(
    grid: dict[tuple[int, int], float],
) -> None:
    """The anti-flatline check. A curve that never moves means nothing is engaging."""
    empty = grid[(FIXED_RAID, INVENTORY_AXIS[0])]
    full = grid[(FIXED_RAID, INVENTORY_AXIS[-1])]
    assert empty - full >= MIN_EXTREME_DROP, (
        f"leakage barely moved across the inventory axis: {empty:.4f} at "
        f"{INVENTORY_AXIS[0]} rounds vs {full:.4f} at {INVENTORY_AXIS[-1]}"
    )


def test_leakage_does_not_fall_with_raid_size(grid: dict[tuple[int, int], float]) -> None:
    values = [grid[(raid, FIXED_INVENTORY)] for raid in RAID_AXIS]
    for (small_raid, small), (large_raid, large) in zip(
        zip(RAID_AXIS, values), zip(RAID_AXIS[1:], values[1:])
    ):
        assert large >= small - 1e-9, (
            f"inventory {FIXED_INVENTORY}: leakage fell from {small:.4f} at raid "
            f"{small_raid} to {large:.4f} at {large_raid}"
        )


def test_a_bigger_raid_strictly_saturates_a_fixed_magazine(
    grid: dict[tuple[int, int], float],
) -> None:
    """Fixed stock spread over more threats must let a larger share through."""
    smallest = grid[(RAID_AXIS[0], FIXED_INVENTORY)]
    largest = grid[(RAID_AXIS[-1], FIXED_INVENTORY)]
    assert largest - smallest >= MIN_EXTREME_DROP, (
        f"raid size barely moved leakage at inventory {FIXED_INVENTORY}: "
        f"{smallest:.4f} at raid {RAID_AXIS[0]} vs {largest:.4f} at {RAID_AXIS[-1]}"
    )
