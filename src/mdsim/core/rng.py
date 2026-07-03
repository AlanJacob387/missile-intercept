"""Counter-based per-environment RNG: environment i's stream depends on its seed alone.

Scheme: each environment carries an integer seed, `base_seed + env_index`. Noise is
drawn by hashing the tuple (env seed, step, stream, lane) through a 32-bit avalanche
mixer, not by advancing a stateful generator. The hash input never includes N_envs or
any batch-relative offset, so environment i draws identical values whether it runs
alone or inside a batch of 1024. That property is what the batch-equivalence gate
checks, and it holds across devices.

`torch.Generator` is deliberately not used for simulation noise: which element of a
generator's stream lands at a given index depends on the request shape and the
backend's thread layout, so env i's noise would shift with N_envs.

`stream` separates independent noise sources (radar range error, azimuth error, and
so on) that are drawn at the same step. Callers pass distinct stream ids so the
sources do not alias onto the same bits.
"""

from __future__ import annotations

import torch
from torch import Tensor

_MASK32 = 0xFFFFFFFF
# Avalanche constant from the lowbias32 mixer family. Kept under 2**27 so the
# int64 product of a 32-bit operand cannot exceed int64 range before masking.
_MIX = 0x45D9F3B
_STEP_ODD = 0x9E3779B1
_STREAM_ODD = 0x85EBCA77

_UNIT = 2.0**-24  # 24 bits is the exactly-representable range of float32


def _mix32(x: Tensor) -> Tensor:
    """32-bit avalanche hash over an int64 tensor holding 32-bit values."""
    x = x & _MASK32
    x = (x ^ (x >> 16)) & _MASK32
    x = (x * _MIX) & _MASK32
    x = (x ^ (x >> 16)) & _MASK32
    x = (x * _MIX) & _MASK32
    x = (x ^ (x >> 16)) & _MASK32
    return x


def env_seeds(base_seed: int, n_envs: int, device: torch.device | str) -> Tensor:
    """Per-environment seeds, [N] int64. Env i is `base_seed + i` for any N."""
    if n_envs < 1:
        raise ValueError(f"n_envs must be >= 1, got {n_envs}")
    return base_seed + torch.arange(n_envs, dtype=torch.int64, device=device)


def _bits(seeds: Tensor, step: int, stream: int, n_lanes: int) -> Tensor:
    """Uniform 32-bit words, [N, n_lanes]. Independent of N and of n_lanes."""
    root = _mix32(seeds ^ ((step * _STEP_ODD) & _MASK32))
    root = _mix32(root ^ ((stream * _STREAM_ODD) & _MASK32))
    lanes = _mix32(torch.arange(n_lanes, dtype=torch.int64, device=seeds.device))
    return _mix32(root.unsqueeze(1) ^ lanes.unsqueeze(0))


def uniform(
    seeds: Tensor,
    step: int,
    stream: int,
    shape: tuple[int, ...] = (),
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Uniform on [0, 1), shaped [N, *shape]."""
    n_lanes = 1
    for dim in shape:
        n_lanes *= dim
    bits = _bits(seeds, step, stream, n_lanes)
    values = (bits >> 8).to(dtype) * _UNIT
    return values.reshape(seeds.shape[0], *shape)


def normal(
    seeds: Tensor,
    step: int,
    stream: int,
    shape: tuple[int, ...] = (),
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Standard normal, shaped [N, *shape], via Box-Muller.

    Each output consumes two uniforms and keeps only the cosine branch. Discarding
    the sine branch halves throughput and costs nothing else; keeping it would make
    a lane's value depend on its parity, which is easy to get wrong later.
    """
    n_lanes = 1
    for dim in shape:
        n_lanes *= dim

    bits = _bits(seeds, step, stream, 2 * n_lanes)
    # Shift off zero so the log is finite: u1 lands in (0, 1].
    u1 = ((bits[:, :n_lanes] >> 8).to(dtype) + 1.0) * _UNIT
    u2 = (bits[:, n_lanes:] >> 8).to(dtype) * _UNIT

    radius = torch.sqrt(-2.0 * torch.log(u1))
    angle = (2.0 * torch.pi) * u2
    return (radius * torch.cos(angle)).reshape(seeds.shape[0], *shape)
