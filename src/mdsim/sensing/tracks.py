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

        x_est     [N, T, 6]     estimated (x, y, z, vx, vy, vz)
        P         [N, T, 6, 6]  estimate covariance
        detected  [N, T]        a track has been initiated and is being held
        age       [N, T]        seconds since track initiation
        updated   [N, T]        a measurement was applied on this step
    """

    x_est: Tensor
    P: Tensor
    detected: Tensor
    age: Tensor
    updated: Tensor

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
        return replace(
            self, **{f.name: getattr(self, f.name).to(device) for f in fields(self)}
        )

    def to_dict(self) -> dict[str, Tensor]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: dict[str, Tensor]) -> TrackState:
        expected = {f.name for f in fields(cls)}
        missing = expected - data.keys()
        if missing:
            raise ValueError(f"tracks missing field(s): {', '.join(sorted(missing))}")
        return cls(**{name: data[name] for name in expected})


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
