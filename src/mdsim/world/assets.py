"""Defended assets as batched tensors."""

from __future__ import annotations

import torch
from torch import Tensor

from mdsim.core.config import Config


def city_tensors(
    config: Config,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    n_envs: int | None = None,
) -> tuple[Tensor, Tensor]:
    """City positions [N, C, 3] and values [N, C], replicated across environments.

    Every environment holds its own copy so a later scenario can perturb asset
    layout or value per env without changing any shape downstream. The tensors are
    materialized rather than left as broadcast views: an expanded view aliases one
    row of storage across all N, and a caller writing into it would silently edit
    every environment at once.
    """
    count = config.sim.n_envs if n_envs is None else n_envs
    if count < 1:
        raise ValueError(f"n_envs must be >= 1, got {count}")
    if not config.cities:
        raise ValueError("scenario defines no cities")

    positions = torch.tensor(
        [city.position_m for city in config.cities], dtype=dtype, device=device
    )
    values = torch.tensor(
        [city.value for city in config.cities], dtype=dtype, device=device
    )

    n_cities = positions.shape[0]
    return (
        positions.expand(count, n_cities, 3).contiguous(),
        values.expand(count, n_cities).contiguous(),
    )
