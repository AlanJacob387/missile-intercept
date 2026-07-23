"""Multi-battery placement, per-battery inventory, and envelope generality.

phase2_multi_battery.yaml declares three batteries of 4 slots each, in order:
battery_north, battery_southwest, battery_southeast -- slots [0,4), [4,8), [8,12).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from mdsim.core.config import BatterySpec, load_config
from mdsim.core.interceptor import InterceptorParams
from mdsim.core.state import make_initial
from mdsim.engagement.envelopes import in_envelope
from mdsim.engagement.inventory import remaining_by_battery

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
DTYPE = torch.float64


def _enabled_counts(state, batteries: tuple[BatterySpec, ...]) -> list[int]:
    counts = []
    slot = 0
    for battery in batteries:
        counts.append(int(state.interceptor_enabled[0, slot : slot + battery.n_interceptors].sum()))
        slot += battery.n_interceptors
    return counts


def _battery_of_slot(batteries: tuple[BatterySpec, ...]) -> torch.Tensor:
    return torch.repeat_interleave(
        torch.arange(len(batteries)), torch.tensor([b.n_interceptors for b in batteries])
    )


# -- (a) placement: each slot block launches from its own battery's position --


def test_battery_slots_launch_from_their_own_battery_position() -> None:
    config = load_config(CONFIG_DIR, scenario="phase2_multi_battery")
    state = make_initial(config, "cpu", dtype=DTYPE)

    slot = 0
    for battery in config.scenario.batteries:
        expected = torch.tensor(battery.position_m, dtype=DTYPE)
        block = state.interceptor_pos[:, slot : slot + battery.n_interceptors, :]
        assert torch.allclose(block, expected.expand_as(block)), battery.name
        slot += battery.n_interceptors
    assert slot == state.n_interceptors


# -- (b) per-battery inventory: make_initial's three modes, and remaining_by_battery --


def test_inventory_none_fills_every_battery() -> None:
    config = load_config(CONFIG_DIR, scenario="phase2_multi_battery")
    state = make_initial(config, "cpu", dtype=DTYPE, inventory=None)
    assert _enabled_counts(state, config.scenario.batteries) == [4, 4, 4]


def test_inventory_int_applies_same_count_to_every_battery() -> None:
    """A plain int is not a flat total split across batteries -- it repeats."""
    config = load_config(CONFIG_DIR, scenario="phase2_multi_battery")
    state = make_initial(config, "cpu", dtype=DTYPE, inventory=2)
    assert _enabled_counts(state, config.scenario.batteries) == [2, 2, 2]


def test_inventory_sequence_applies_per_battery_in_declared_order() -> None:
    config = load_config(CONFIG_DIR, scenario="phase2_multi_battery")
    state = make_initial(config, "cpu", dtype=DTYPE, inventory=[1, 3, 4])
    assert _enabled_counts(state, config.scenario.batteries) == [1, 3, 4]


def test_inventory_sequence_wrong_length_raises() -> None:
    config = load_config(CONFIG_DIR, scenario="phase2_multi_battery")
    with pytest.raises(ValueError):
        make_initial(config, "cpu", dtype=DTYPE, inventory=[1, 2])


def test_inventory_over_capacity_names_the_battery() -> None:
    config = load_config(CONFIG_DIR, scenario="phase2_multi_battery")
    with pytest.raises(ValueError, match="battery_north"):
        make_initial(config, "cpu", dtype=DTYPE, inventory=[5, 4, 4])


def test_remaining_by_battery_counts_unfired_rounds_per_battery() -> None:
    config = load_config(CONFIG_DIR, scenario="phase2_multi_battery")
    batteries = config.scenario.batteries
    battery_of_slot = _battery_of_slot(batteries)

    state = make_initial(config, "cpu", dtype=DTYPE, inventory=[3, 2, 4])
    committed = torch.zeros_like(state.interceptor_committed)
    # battery_north: 2 of its 3 loaded slots committed -> 1 left.
    committed[:, 0] = True
    committed[:, 1] = True
    # battery_southeast: 1 of its 4 loaded slots committed -> 3 left.
    committed[:, 8] = True

    remaining = remaining_by_battery(
        state.interceptor_enabled, committed, battery_of_slot, len(batteries)
    )
    # battery_north=1, battery_southwest=2 (untouched), battery_southeast=3.
    assert remaining[0].tolist() == [1, 2, 3]
    assert bool((remaining == remaining[0]).all()), "every env is built identically"


# -- (c) envelope check is already battery-position generic, no code change needed --


def test_envelope_check_is_battery_position_generic() -> None:
    config = load_config(CONFIG_DIR)
    params = InterceptorParams.from_spec(config.interceptors["pac3_mse"])

    # pac3_mse envelope: slant [3, 70] km, altitude [0.5, 24] km.
    track_pos = torch.tensor([[[10_000.0, 0.0, 10_000.0]]], dtype=DTYPE)
    near_battery = torch.tensor([[0.0, 0.0, 0.0]], dtype=DTYPE)
    far_battery = torch.tensor([[300_000.0, 0.0, 0.0]], dtype=DTYPE)

    assert bool(in_envelope(track_pos, near_battery, params)[0, 0])
    assert not bool(in_envelope(track_pos, far_battery, params)[0, 0])


# -- (d) single-battery regression: batteries == () must behave exactly as before --


def test_single_battery_pos_is_the_shared_battery_pos_m() -> None:
    config = load_config(CONFIG_DIR)
    assert config.scenario.batteries == ()

    state = make_initial(config, "cpu", dtype=DTYPE)
    expected = torch.tensor(config.scenario.battery_pos_m, dtype=DTYPE)
    assert torch.allclose(state.interceptor_pos, expected.expand_as(state.interceptor_pos))


def test_single_battery_inventory_is_still_a_flat_global_stock() -> None:
    config = load_config(CONFIG_DIR)
    config = replace(config, scenario=replace(config.scenario, n_interceptors=5))
    assert config.scenario.batteries == ()

    full = make_initial(config, "cpu", dtype=DTYPE, inventory=None)
    assert bool(full.interceptor_enabled.all())

    empty = make_initial(config, "cpu", dtype=DTYPE, inventory=0)
    assert not bool(empty.interceptor_enabled.any())

    # All but the last, by flat index order -- the pre-multi-battery behaviour.
    partial = make_initial(config, "cpu", dtype=DTYPE, inventory=4)
    assert partial.interceptor_enabled[0].tolist() == [True, True, True, True, False]
