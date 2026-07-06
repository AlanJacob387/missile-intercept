"""Multi-step rollouts: a Python loop over time, batched across environments.

The loop runs over T timesteps, never over environments -- each step is a single
batched call, so the host-side loop costs one dispatch per step, not per env.
"""

from __future__ import annotations

import torch
from torch import Tensor

from mdsim.core.state import EnvState
from mdsim.envs.engine import EngineParams, step


def rollout(
    state: EnvState,
    params: EngineParams,
    n_steps: int,
    record: bool = False,
) -> tuple[EnvState, Tensor | None]:
    """Advance `n_steps` ticks. Returns (final state, trajectory or None).

    With record=True the trajectory is threat positions stacked as [T, N, B, 3],
    where index k holds the state after step k+1 -- the initial state is not
    included, so it aligns with the oracle's positions[1:].

    Recording is off by default because it is the memory cost of the run: T=2000,
    N=1024 float32 positions is ~25 MB per body, and it grows linearly in both.
    """
    if n_steps < 0:
        raise ValueError(f"n_steps must be non-negative, got {n_steps}")

    frames: list[Tensor] = []
    for _ in range(n_steps):
        state = step(state, params)
        if record:
            frames.append(state.threat_pos)

    if not record:
        return state, None
    if not frames:
        empty = state.threat_pos.new_empty((0, *state.threat_pos.shape))
        return state, empty
    return state, torch.stack(frames)
