"""Batched truth state for all environments as one device-resident tensor bundle."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields, replace

import torch
from torch import Tensor

from mdsim.core.config import BatterySpec, Config
from mdsim.core.rng import env_seeds
from mdsim.sensing.tracks import TrackState, make_empty
from mdsim.world.scenario import multi_launch_state

_TRACK_PREFIX = "tracks."


@dataclass(frozen=True)
class EnvState:
    """Ground truth for N environments. Only the engine may produce a new one.

    Shapes, with N environments, T threat slots, I interceptor slots, C cities:
        threat_pos, threat_vel            [N, T, 3]
        interceptor_pos, interceptor_vel  [N, I, 3]
        threat_alive                      [N, T]   still flying
        threat_killed                     [N, T]   killed by an interceptor
        threat_active                     [N, T]   slot is in use this scenario
        threat_target_city                [N, T]   city index this threat is aimed at
        threat_leaked                     [N, T]   reached its city
        interceptor_alive                 [N, I]   in flight now
        interceptor_committed             [N, I]   launched at some point
        interceptor_enabled               [N, I]   magazine holds a round for this slot
        interceptor_target                [N, I]   bound threat slot, -1 uncommitted
        city_pos                          [N, C, 3]
        city_value                        [N, C]
        city_alive                        [N, C]
        t                                 [N]
        step_index                        [N]
        seed                              [N]
        tracks                            TrackState

    Threat slots are fixed at T and masked by `threat_active`. Inactive slots are
    excluded from physics, radar, assignment and scoring, which keeps every raid
    size the same tensor shape -- a saturation sweep varies how many slots are
    active, not how big the arrays are.

    Inventory is `interceptor_enabled`: a disabled slot has no round and can never
    launch. Remaining stock is `enabled & ~committed`, so magazine depth is a mask
    rather than a counter that could drift out of step with the slots.

    Every field carries the leading N_envs dimension, so operations on the state are
    written batched by hand rather than mapped over a per-environment function.

    `tracks` is estimate-side state and lives here only so one object carries the
    whole simulation forward. Nothing downstream reaches through it to truth.
    """

    threat_pos: Tensor
    threat_vel: Tensor
    interceptor_pos: Tensor
    interceptor_vel: Tensor
    threat_alive: Tensor
    threat_killed: Tensor
    threat_active: Tensor
    threat_target_city: Tensor
    threat_leaked: Tensor
    interceptor_alive: Tensor
    interceptor_committed: Tensor
    interceptor_enabled: Tensor
    interceptor_target: Tensor
    city_pos: Tensor
    city_value: Tensor
    city_alive: Tensor
    t: Tensor
    step_index: Tensor
    seed: Tensor
    tracks: TrackState

    @property
    def n_envs(self) -> int:
        return self.threat_pos.shape[0]

    @property
    def n_threats(self) -> int:
        return self.threat_pos.shape[1]

    @property
    def n_interceptors(self) -> int:
        return self.interceptor_pos.shape[1]

    @property
    def device(self) -> torch.device:
        return self.threat_pos.device

    def to(self, device: torch.device | str) -> EnvState:
        """Move every field to `device` as a unit; no field may be left behind."""
        moved = {}
        for f in fields(self):
            value = getattr(self, f.name)
            moved[f.name] = value.to(device)
        return replace(self, **moved)

    def to_dict(self) -> dict[str, Tensor]:
        """Flat field mapping, for torch.save and for handoff to the renderer."""
        flat: dict[str, Tensor] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, TrackState):
                for name, tensor in value.to_dict().items():
                    flat[_TRACK_PREFIX + name] = tensor
            else:
                flat[f.name] = value
        return flat

    @classmethod
    def from_dict(cls, data: dict[str, Tensor]) -> EnvState:
        track_data = {
            name[len(_TRACK_PREFIX) :]: tensor
            for name, tensor in data.items()
            if name.startswith(_TRACK_PREFIX)
        }
        expected = {f.name for f in fields(cls)} - {"tracks"}
        missing = expected - data.keys()
        if missing:
            raise ValueError(f"state is missing field(s): {', '.join(sorted(missing))}")
        values = {name: data[name] for name in expected}
        return cls(**values, tracks=TrackState.from_dict(track_data))


def _resolve_battery_stocks(
    batteries: tuple[BatterySpec, ...], inventory: int | Sequence[int] | None
) -> list[int]:
    """Per-battery starting stock, in `batteries` order.

    None fills every battery. A plain int applies that SAME count to every battery
    (not a total split across them). A Sequence[int] gives one entry per battery.
    """
    if inventory is None:
        stocks = [b.n_interceptors for b in batteries]
    elif isinstance(inventory, int):
        stocks = [inventory for _ in batteries]
    else:
        stocks = list(inventory)
        if len(stocks) != len(batteries):
            raise ValueError(
                f"inventory sequence length ({len(stocks)}) must match battery count "
                f"({len(batteries)})"
            )

    for spec, stock in zip(batteries, stocks):
        if not 0 <= stock <= spec.n_interceptors:
            raise ValueError(
                f"battery '{spec.name}': inventory must be in [0, {spec.n_interceptors}], "
                f"got {stock}"
            )
    return stocks


def make_initial(
    config: Config,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    n_active_threats: int | None = None,
    inventory: int | Sequence[int] | None = None,
) -> EnvState:
    """Build the t=0 batched state for a scenario.

    Interceptors start not-alive: `alive` means in flight, and nothing has been
    launched yet. They sit at the battery so their position is defined before launch.

    dtype selects the float precision of the physical fields. float32 is the run
    default; float64 exists so parity against the float64 oracle measures the
    algorithm rather than float32 rounding. float64 runs on XPU but is emulated
    there, so the parity gate uses it on CPU.

    n_active_threats overrides the scenario's active threat count without changing
    tensor shapes, which is what the saturation sweep varies. It defaults to fully
    active.

    inventory means two different things depending on config.scenario.batteries, and
    conflating them is the easiest mistake to make here:

    - Single-battery (batteries == (), the Phase 0-2 default): unchanged from before
      multi-battery existed. inventory is a flat count over the whole n_interceptors
      range -- None means every slot starts loaded, an int means exactly that many
      of the flat n_interceptors slots (by index order) start loaded.
    - Multi-battery (batteries non-empty): inventory is resolved PER BATTERY.
        * None: every battery starts full, at its own n_interceptors.
        * int: that SAME count is applied to EVERY battery -- this is NOT the flat
          total from the single-battery case split across batteries. Passing the
          scenario's n_interceptors here would overflow every battery, not fill the
          fleet once.
        * Sequence[int]: one stock value per battery, in `batteries` declaration
          order; a length mismatch raises ValueError. Each resolved value must fall
          within its own battery's [0, n_interceptors], or ValueError names the
          battery that overflowed.
    """
    n = config.sim.n_envs
    n_threats = config.scenario.n_threats
    n_interceptors = config.scenario.n_interceptors
    n_cities = len(config.cities)
    batteries = config.scenario.batteries

    active_count = n_threats if n_active_threats is None else n_active_threats
    if not 0 <= active_count <= n_threats:
        raise ValueError(f"n_active_threats must be in [0, {n_threats}], got {active_count}")

    # Launch state is derived from the scenario's engagement range, not written in
    # config: terminal speed has to emerge from the flight, not be declared.
    #
    # Geometry is built for the ACTIVE raid, not for the slot count. Spreading all T
    # slots across the arc and then activating the first k would bunch a small raid
    # at one edge, so raid size and angular spread would vary together and the
    # saturation sweep could not attribute a change to either. Inactive slots are
    # parked on the last active launch point; they are masked everywhere and never
    # fly, so their position only has to be defined.
    active = max(active_count, 1)
    positions, velocities, targets = multi_launch_state(config, active)
    pad = n_threats - active
    if pad > 0:
        positions = positions + [positions[-1]] * pad
        velocities = velocities + [velocities[-1]] * pad
        targets = targets + [targets[-1]] * pad

    threat_pos = torch.tensor(positions, dtype=dtype, device=device).expand(n, n_threats, 3)
    threat_vel = torch.tensor(velocities, dtype=dtype, device=device).expand(n, n_threats, 3)
    target_city = torch.tensor(targets, dtype=torch.int64, device=device).expand(n, n_threats)

    if not batteries:
        # Single-battery, byte-identical to the pre-multi-battery code path: one
        # shared position, one flat stock count over the whole slot range.
        stock = n_interceptors if inventory is None else inventory
        if not 0 <= stock <= n_interceptors:
            raise ValueError(f"inventory must be in [0, {n_interceptors}], got {stock}")

        battery = torch.tensor(config.scenario.battery_pos_m, dtype=dtype, device=device)
        battery_pos = battery.expand(n, n_interceptors, 3)

        round_index = torch.arange(n_interceptors, device=device)
        interceptor_enabled = (round_index < stock).expand(n, n_interceptors)
    else:
        # Multi-battery: each slot block gets its own battery's position and its own
        # resolved stock, concatenated in `batteries` declaration order.
        stocks = _resolve_battery_stocks(batteries, inventory)

        pos_blocks = []
        enabled_blocks = []
        for spec, spec_stock in zip(batteries, stocks):
            spec_pos = torch.tensor(spec.position_m, dtype=dtype, device=device)
            pos_blocks.append(spec_pos.expand(spec.n_interceptors, 3))
            spec_round_index = torch.arange(spec.n_interceptors, device=device)
            enabled_blocks.append(spec_round_index < spec_stock)

        battery_pos = torch.cat(pos_blocks, dim=0).expand(n, n_interceptors, 3)
        interceptor_enabled = torch.cat(enabled_blocks, dim=0).expand(n, n_interceptors)

    slot_index = torch.arange(n_threats, device=device)
    threat_active = (slot_index < active_count).expand(n, n_threats)

    city_pos = torch.tensor(
        [city.position_m for city in config.cities], dtype=dtype, device=device
    ).expand(n, n_cities, 3)
    city_value = torch.tensor(
        [city.value for city in config.cities], dtype=dtype, device=device
    ).expand(n, n_cities)

    false_threats = torch.zeros((n, n_threats), dtype=torch.bool, device=device)
    false_interceptors = torch.zeros((n, n_interceptors), dtype=torch.bool, device=device)

    return EnvState(
        threat_pos=threat_pos.contiguous(),
        threat_vel=threat_vel.contiguous(),
        interceptor_pos=battery_pos.contiguous(),
        interceptor_vel=torch.zeros(
            (n, n_interceptors, 3), dtype=dtype, device=device
        ),
        # An inactive slot is not alive: it must never fly, be tracked, or be scored.
        threat_alive=threat_active.clone(),
        threat_killed=false_threats.clone(),
        threat_active=threat_active.contiguous(),
        threat_target_city=target_city.contiguous(),
        threat_leaked=false_threats.clone(),
        interceptor_alive=false_interceptors.clone(),
        interceptor_committed=false_interceptors.clone(),
        interceptor_enabled=interceptor_enabled.contiguous(),
        interceptor_target=torch.full(
            (n, n_interceptors), -1, dtype=torch.int64, device=device
        ),
        city_pos=city_pos.contiguous(),
        city_value=city_value.contiguous(),
        city_alive=torch.ones((n, n_cities), dtype=torch.bool, device=device),
        t=torch.zeros((n,), dtype=dtype, device=device),
        step_index=torch.zeros((n,), dtype=torch.int64, device=device),
        seed=env_seeds(config.sim.seed, n, device),
        tracks=make_empty(n, n_threats, device, dtype),
    )
