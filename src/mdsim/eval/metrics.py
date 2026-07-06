"""Kill probability, leakage, expenditure counts, and value-weighted damage."""

from __future__ import annotations

import torch

from mdsim.core.state import EnvState

_ACC = torch.float64  # reduce in float64 so a 4096-env mean is not float32-noisy


def kill_probability(state: EnvState) -> float:
    """Fraction of environments in which every threat was killed."""
    return float(state.threat_killed.all(dim=1).to(_ACC).mean())


def leakage(state: EnvState) -> float:
    """Fraction of all threats that were not killed."""
    return float((~state.threat_killed).to(_ACC).mean())


def interceptors_committed(state: EnvState) -> float:
    """Total interceptors committed across the whole batch. A count, not a rate."""
    return float(state.interceptor_committed.to(_ACC).sum())


def kills(state: EnvState) -> float:
    """Total threats killed across the whole batch. A count, not a rate."""
    return float(state.threat_killed.to(_ACC).sum())


def committed_per_kill(state: EnvState) -> float | None:
    """Batch-level ratio of interceptors committed to threats killed.

    Returns None when nothing was killed, because the ratio is undefined there --
    not infinite, and not zero. Callers must render the absence rather than let a
    sentinel float propagate into a plot or a table.

    This is a batch aggregate, NOT a salvo size. With one interceptor committed per
    environment it is simply 1/Pk, so at Pk = 0.05 it reads 20 by construction and
    says nothing about how many rounds any single engagement fired. Read it beside
    the raw counts, never instead of them.
    """
    killed = kills(state)
    if killed == 0.0:
        return None
    return interceptors_committed(state) / killed


def tracks_held(state: EnvState) -> float:
    """Fraction of threats the tracker was still holding at the end of the run.

    Not a scoring metric. It separates a miss from a never-detected threat, which is
    the first thing to check when Pk collapses.
    """
    return float(state.tracks.detected.to(_ACC).mean())


def summarize(state: EnvState) -> dict[str, float | None]:
    """Counts first, derived ratios after, so the ratio is read in context."""
    return {
        "pk": kill_probability(state),
        "leakage": leakage(state),
        "interceptors_committed": interceptors_committed(state),
        "kills": kills(state),
        "committed_per_kill_batch_level": committed_per_kill(state),
        "tracks_held": tracks_held(state),
    }
