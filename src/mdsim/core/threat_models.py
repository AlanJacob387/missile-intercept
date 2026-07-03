"""Swappable threat dynamics models (ballistic, maneuvering, glide)."""

from __future__ import annotations

from typing import Callable

from torch import Tensor

from mdsim.core.dynamics import PhysicsParams, total_accel


def ballistic(pos: Tensor, vel: Tensor, params: PhysicsParams) -> Tensor:
    """Unpowered flight: gravity plus drag, no control. Returns [N, B, 3]."""
    return total_accel(pos, vel, params)


# Adding MANEUVERING in Phase 2 is a new entry here, not an edit at the call site.
THREAT_MODELS: dict[str, Callable[[Tensor, Tensor, PhysicsParams], Tensor]] = {
    "ballistic": ballistic,
}


def get_threat_model(
    name: str,
) -> Callable[[Tensor, Tensor, PhysicsParams], Tensor]:
    try:
        return THREAT_MODELS[name]
    except KeyError:
        known = ", ".join(sorted(THREAT_MODELS))
        raise ValueError(f"unknown threat model {name!r}; known: {known}") from None
