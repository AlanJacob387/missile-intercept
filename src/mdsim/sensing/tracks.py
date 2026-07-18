"""Track state container, the only threat view guidance and assignment may read.

Nothing downstream of the tracker imports EnvState. TrackState holds estimates
derived from noisy measurements and never aliases a truth tensor -- a test asserts
that by tensor storage identity, not by comparing values, because an estimate that
happens to equal truth on one step is not the same thing as an estimate that IS
truth.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import torch
from torch import Tensor


@dataclass(frozen=True)
class TrackState:
    """Tracker output for N environments and T tracks.

        x_est     [N, T, 6]        estimated (x, y, z, vx, vy, vz)
        P         [N, T, 6, 6]     estimate covariance
        detected  [N, T]           a track has been initiated and is being held
        age       [N, T]           seconds since track initiation
        updated   [N, T]           a measurement was applied on this step

    The three fields below are optional and only populated by an IMM-aware
    tracker (sensing.imm). A track not using IMM leaves them None, and
    to_dict/from_dict/to treat that as the track's ordinary shape rather than a
    missing field.

        mu        [N, T, 2]        IMM model probabilities
        x_models  [N, T, 2, 9]     per-model IMM states
        P_models  [N, T, 2, 9, 9]  per-model IMM covariances
    """

    x_est: Tensor
    P: Tensor
    detected: Tensor
    age: Tensor
    updated: Tensor
    mu: Tensor | None = None
    x_models: Tensor | None = None
    P_models: Tensor | None = None

    @property
    def position(self) -> Tensor:
        return self.x_est[..., :3]

    @property
    def velocity(self) -> Tensor:
        return self.x_est[..., 3:]

    @property
    def n_envs(self) -> int:
        return self.x_est.shape[0]

    @property
    def n_tracks(self) -> int:
        return self.x_est.shape[1]

    def to(self, device: torch.device | str) -> TrackState:
        updates = {}
        for f in fields(self):
            value = getattr(self, f.name)
            updates[f.name] = value.to(device) if value is not None else None
        return replace(self, **updates)

    def to_dict(self) -> dict[str, Tensor]:
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if getattr(self, f.name) is not None
        }

    @classmethod
    def from_dict(cls, data: dict[str, Tensor]) -> TrackState:
        required = {"x_est", "P", "detected", "age", "updated"}
        missing = required - data.keys()
        if missing:
            raise ValueError(f"tracks missing field(s): {', '.join(sorted(missing))}")
        optional = {f.name for f in fields(cls)} - required
        return cls(
            **{name: data[name] for name in required},
            **{name: data[name] for name in optional if name in data},
        )


def make_empty(
    n_envs: int,
    n_tracks: int,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> TrackState:
    """No tracks held. Covariance starts at identity; it is overwritten on initiation."""
    eye = torch.eye(6, dtype=dtype, device=device)
    return TrackState(
        x_est=torch.zeros((n_envs, n_tracks, 6), dtype=dtype, device=device),
        P=eye.expand(n_envs, n_tracks, 6, 6).clone(),
        detected=torch.zeros((n_envs, n_tracks), dtype=torch.bool, device=device),
        age=torch.zeros((n_envs, n_tracks), dtype=dtype, device=device),
        updated=torch.zeros((n_envs, n_tracks), dtype=torch.bool, device=device),
    )
