"""Un-batched NumPy oracle: deliberately slow and obvious, used to check the engine."""

from __future__ import annotations

import numpy as np

G = 9.80665


def propagate(
    pos0: np.ndarray | list[float],
    vel0: np.ndarray | list[float],
    dt: float,
    n_steps: int,
    g: float = G,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate one drag-free body for n_steps with semi-implicit Euler.

    Returns (positions, velocities), each [n_steps + 1, 3], including the initial state.

    Clarity beats speed here: this exists to be read and trusted, so the loop stays
    explicit and nothing is vectorized.
    """
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    if n_steps < 0:
        raise ValueError(f"n_steps must be non-negative, got {n_steps}")

    pos = np.asarray(pos0, dtype=np.float64).copy()
    vel = np.asarray(vel0, dtype=np.float64).copy()
    if pos.shape != (3,) or vel.shape != (3,):
        raise ValueError("pos0 and vel0 must each be length-3 vectors")

    positions = np.empty((n_steps + 1, 3), dtype=np.float64)
    velocities = np.empty((n_steps + 1, 3), dtype=np.float64)
    positions[0] = pos
    velocities[0] = vel

    accel = np.array([0.0, 0.0, -g], dtype=np.float64)

    for k in range(n_steps):
        # Semi-implicit: velocity updates first, then position uses the new velocity.
        vel = vel + accel * dt
        pos = pos + vel * dt
        positions[k + 1] = pos
        velocities[k + 1] = vel

    return positions, velocities
