"""Kill probability, leakage, expenditure counts, and value-weighted damage.

Every threat-side metric is masked by `threat_active`. Threat slots are fixed at T
and a raid uses only the first few, so an unmasked mean divides by the slot count
rather than the raid size and reports a smaller leakage the emptier the raid gets --
exactly backwards, and exactly the shape a saturation curve is supposed to show.
"""

from __future__ import annotations

import torch

from mdsim.core.state import EnvState

_ACC = torch.float64  # reduce in float64 so a 4096-env mean is not float32-noisy


def _active_count(state: EnvState) -> torch.Tensor:
    """Active threat slots per env, [N], clamped so empty raids cannot divide by zero."""
    return state.threat_active.to(_ACC).sum(dim=1).clamp(min=1.0)


def kill_probability(state: EnvState) -> float:
    """Fraction of environments in which every active threat was killed."""
    unkilled = state.threat_active & ~state.threat_killed
    return float((~unkilled.any(dim=1)).to(_ACC).mean())


def leakage(state: EnvState) -> float:
    """Fraction of active threats that reached their city.

    Leakage is what got through, not merely what went unkilled: a threat still in
    flight when the run ends is neither. The saturation curve is drawn from this.
    """
    leaked = (state.threat_leaked & state.threat_active).to(_ACC).sum(dim=1)
    return float((leaked / _active_count(state)).mean())


def leaked_count(state: EnvState) -> float:
    """Total active threats that reached a city, across the batch."""
    return float((state.threat_leaked & state.threat_active).to(_ACC).sum())


def value_destroyed(state: EnvState) -> float:
    """Mean per-env city value lost. Destruction is binary per city."""
    lost = (~state.city_alive).to(_ACC) * state.city_value.to(_ACC)
    return float(lost.sum(dim=1).mean())


def value_defended(state: EnvState) -> float:
    """Mean per-env city value still standing."""
    held = state.city_alive.to(_ACC) * state.city_value.to(_ACC)
    return float(held.sum(dim=1).mean())


def damage_prevented_fraction(state: EnvState) -> float:
    """Share of total city value still standing, in [0, 1].

    The defended fraction rather than an absolute, so runs with different city sets
    stay comparable.
    """
    total = float(state.city_value.to(_ACC).sum(dim=1).mean())
    if total == 0.0:
        return 1.0
    return value_defended(state) / total


def interceptors_committed(state: EnvState) -> float:
    """Total interceptors committed across the whole batch. A count, not a rate."""
    return float(state.interceptor_committed.to(_ACC).sum())


def inventory_remaining(state: EnvState) -> float:
    """Mean unfired rounds per env. Stock is a slot mask, not a counter."""
    unfired = state.interceptor_enabled & ~state.interceptor_committed
    return float(unfired.to(_ACC).sum(dim=1).mean())


def kills(state: EnvState) -> float:
    """Total active threats killed across the whole batch. A count, not a rate."""
    return float((state.threat_killed & state.threat_active).to(_ACC).sum())


def committed_per_kill(state: EnvState) -> float | None:
    """Batch-level ratio of interceptors committed to threats killed.

    Returns None when nothing was killed, because the ratio is undefined there --
    not infinite, and not zero. Callers must render the absence rather than let a
    sentinel float propagate into a plot or a table.

    This is a batch aggregate, NOT a salvo size. Read it beside the raw counts,
    never instead of them.
    """
    killed = kills(state)
    if killed == 0.0:
        return None
    return interceptors_committed(state) / killed


def tracks_held(state: EnvState) -> float:
    """Fraction of active threats the tracker was still holding at the end.

    Not a scoring metric. It separates a miss from a never-detected threat, which is
    the first thing to check when leakage is total.
    """
    held = (state.tracks.detected & state.threat_active).to(_ACC).sum(dim=1)
    return float((held / _active_count(state)).mean())


def summarize(state: EnvState) -> dict[str, float | None]:
    """Counts first, derived ratios after, so the ratio is read in context."""
    return {
        "pk": kill_probability(state),
        "leakage": leakage(state),
        "leaked_count": leaked_count(state),
        "value_destroyed": value_destroyed(state),
        "damage_prevented_fraction": damage_prevented_fraction(state),
        "interceptors_committed": interceptors_committed(state),
        "inventory_remaining": inventory_remaining(state),
        "kills": kills(state),
        "committed_per_kill_batch_level": committed_per_kill(state),
        "tracks_held": tracks_held(state),
    }
