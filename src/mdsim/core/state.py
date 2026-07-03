"""Batched truth state for all environments as one device-resident tensor bundle."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import torch
from torch import Tensor

from mdsim.core.config import Config
from mdsim.core.rng import env_seeds


@dataclass(frozen=True)
class EnvState:
    """Ground truth for N environments. Only the engine may produce a new one.

    Shapes, with N environments, T threats, and I interceptors:
        threat_pos, threat_vel            [N, T, 3]
        interceptor_pos, interceptor_vel  [N, I, 3]
        threat_alive                      [N, T]
        interceptor_alive                 [N, I]
        t                                 [N]
        seed                              [N]

    Every field carries the leading N_envs dimension, so operations on the state are
    written batched by hand rather than mapped over a per-environment function.
    """

    threat_pos: Tensor
    threat_vel: Tensor
    interceptor_pos: Tensor
    interceptor_vel: Tensor
    threat_alive: Tensor
    interceptor_alive: Tensor
    t: Tensor
    seed: Tensor

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
        return replace(
            self, **{f.name: getattr(self, f.name).to(device) for f in fields(self)}
        )

    def to_dict(self) -> dict[str, Tensor]:
        """Flat field mapping, for torch.save and for handoff to the renderer."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: dict[str, Tensor]) -> EnvState:
        expected = {f.name for f in fields(cls)}
        missing = expected - data.keys()
        if missing:
            raise ValueError(f"state is missing field(s): {', '.join(sorted(missing))}")
        return cls(**{name: data[name] for name in expected})


def make_initial(
    config: Config,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> EnvState:
    """Build the t=0 batched state for a scenario.

    Interceptors start not-alive: `alive` means in flight, and nothing has been
    launched yet. They sit at the battery so their position is defined before launch.

    dtype selects the float precision of the physical fields. float32 is the run
    default; float64 exists so parity against the float64 oracle measures the
    algorithm rather than float32 rounding. float64 runs on XPU but is emulated
    there, so the parity gate uses it on CPU.
    """
    n = config.sim.n_envs
    n_threats = config.scenario.n_threats
    n_interceptors = config.scenario.n_interceptors

    def _vec(values: tuple[float, float, float]) -> Tensor:
        return torch.tensor(values, dtype=dtype, device=device)

    # expand() would alias one row of storage across all N; the engine writes into
    # these in place, so materialize real per-environment storage up front.
    threat_pos = _vec(config.scenario.threat_launch_pos_m).expand(n, n_threats, 3)
    threat_vel = _vec(config.scenario.threat_launch_vel_mps).expand(n, n_threats, 3)
    battery_pos = _vec(config.scenario.battery_pos_m).expand(n, n_interceptors, 3)

    return EnvState(
        threat_pos=threat_pos.contiguous(),
        threat_vel=threat_vel.contiguous(),
        interceptor_pos=battery_pos.contiguous(),
        interceptor_vel=torch.zeros(
            (n, n_interceptors, 3), dtype=dtype, device=device
        ),
        threat_alive=torch.ones((n, n_threats), dtype=torch.bool, device=device),
        interceptor_alive=torch.zeros(
            (n, n_interceptors), dtype=torch.bool, device=device
        ),
        t=torch.zeros((n,), dtype=dtype, device=device),
        seed=env_seeds(config.sim.seed, n, device),
    )
