"""Impact scoring: which threats reached their city, and what that cost.

This module reads truth tensors, and that is correct rather than an information-
separation violation. The separation rule constrains what the DEFENDER may act on --
guidance and assignment see estimates only. Scoring is the outcome ledger, evaluated
after the fact by the simulation itself, and an outcome computed from estimates would
be scoring the defender's opinion instead of what happened.

Destruction is binary. One leaked threat destroys its city and scores the full value;
there is no accumulating damage and no partial credit. A second threat reaching an
already-destroyed city still counts as leakage but adds no value, which is why the
leakage count and the damage total are separate numbers rather than one rescaled.
"""

from __future__ import annotations

import torch
from torch import Tensor

from mdsim.core.intercept import closest_approach


def _target_city_pos(city_pos: Tensor, threat_target_city: Tensor) -> Tensor:
    """Position of the city each threat is aimed at, [N, T, 3]."""
    index = threat_target_city.unsqueeze(-1).expand(-1, -1, city_pos.shape[-1])
    return torch.gather(city_pos, 1, index)


def reached_city(
    threat_pos_before: Tensor,
    threat_pos_after: Tensor,
    threat_target_city: Tensor,
    city_pos: Tensor,
    impact_radius_m: float,
    dt: float,
) -> Tensor:
    """[N, T] bool: this step's path passed within impact_radius_m of the target city.

    Closest approach over the step rather than an endpoint test, matching the
    threat-interceptor geometry in core.intercept. A terminal threat covers tens of
    metres per step against a city radius of hundreds, so endpoint testing would
    rarely miss an impact outright -- but the two geometry tests in the engine
    disagreeing about what "reached" means is a worse failure than either being
    slightly coarse, and it would surface as an unexplained parity break.

    Pure geometry: no active or alive masking here. resolve_leaks applies those.
    """
    target = _target_city_pos(city_pos, threat_target_city)
    delta_pos = threat_pos_before - target
    # The city does not move, so relative velocity is the threat's own step
    # displacement, taken from the segment the integrator actually produced.
    delta_vel = (threat_pos_after - threat_pos_before) / dt

    distance, _ = closest_approach(delta_pos, delta_vel, dt)
    return distance <= impact_radius_m


def resolve_leaks(
    threat_pos_before: Tensor,
    threat_pos_after: Tensor,
    threat_target_city: Tensor,
    threat_active: Tensor,
    threat_alive: Tensor,
    threat_leaked: Tensor,
    city_pos: Tensor,
    city_alive: Tensor,
    impact_radius_m: float,
    dt: float,
) -> tuple[Tensor, Tensor]:
    """Resolve impacts for one step, returning (newly_leaked [N, T], destroyed [N, C]).

    A threat leaks when it is active, still flying, has not already leaked, and
    reached its city this step. `destroyed` marks cities that were standing at the
    start of the step and are not any more, so a caller can score the value once
    without tracking which threat got there first.

    Nothing is mutated. The caller clears threat_alive for the leaked slots and
    clears city_alive for the destroyed cities.
    """
    arrived = reached_city(
        threat_pos_before,
        threat_pos_after,
        threat_target_city,
        city_pos,
        impact_radius_m,
        dt,
    )
    newly_leaked = threat_active & threat_alive & ~threat_leaked & arrived

    # Fan the per-threat mask out to per-city: city c is hit if any threat aimed at
    # it leaked this step. One-hot comparison rather than a scatter, so the reduction
    # stays a plain boolean any() and needs no index-collision handling.
    n_cities = city_pos.shape[1]
    city_index = torch.arange(n_cities, device=city_pos.device).view(1, 1, n_cities)
    aimed_at = threat_target_city.unsqueeze(-1) == city_index
    city_hit = (newly_leaked.unsqueeze(-1) & aimed_at).any(dim=1)

    return newly_leaked, city_hit & city_alive


def damage_value(
    city_value: Tensor, city_alive_before: Tensor, city_alive_after: Tensor
) -> Tensor:
    """[N] value destroyed between the two city-alive masks."""
    destroyed = city_alive_before & ~city_alive_after
    return (city_value * destroyed.to(city_value.dtype)).sum(dim=-1)


def leakage_count(threat_leaked: Tensor, threat_active: Tensor) -> Tensor:
    """[N] leaked threats per environment, inactive slots excluded."""
    return (threat_leaked & threat_active).sum(dim=-1)
