"""Environment i must draw the same noise regardless of how many envs run beside it."""

from __future__ import annotations

import torch

from mdsim.core.rng import env_seeds, normal, uniform

STEP = 7
STREAM = 3
SHAPE = (2, 3)


def _seeds(n: int) -> torch.Tensor:
    return env_seeds(0, n, "cpu")


def test_env_seeds_are_index_offsets() -> None:
    assert _seeds(4).tolist() == [0, 1, 2, 3]
    assert env_seeds(100, 3, "cpu").tolist() == [100, 101, 102]


def test_normal_is_independent_of_batch_size() -> None:
    """The property the batch-equivalence gate rests on. Bitwise here: same device,
    same shape per env, so only the batch dimension differs."""
    solo = normal(_seeds(1), STEP, STREAM, SHAPE)
    mid = normal(_seeds(37), STEP, STREAM, SHAPE)
    batch = normal(_seeds(1024), STEP, STREAM, SHAPE)

    torch.testing.assert_close(solo[0], batch[0], rtol=0, atol=0)
    torch.testing.assert_close(mid[36], batch[36], rtol=0, atol=0)


def test_uniform_is_independent_of_batch_size() -> None:
    solo = uniform(_seeds(1), STEP, STREAM, SHAPE)
    batch = uniform(_seeds(1024), STEP, STREAM, SHAPE)
    torch.testing.assert_close(solo[0], batch[0], rtol=0, atol=0)


def test_streams_and_steps_decorrelate() -> None:
    seeds = _seeds(256)
    base = normal(seeds, STEP, STREAM, (8,))
    assert not torch.allclose(base, normal(seeds, STEP, STREAM + 1, (8,)))
    assert not torch.allclose(base, normal(seeds, STEP + 1, STREAM, (8,)))


def test_envs_decorrelate() -> None:
    draw = normal(_seeds(2), STEP, STREAM, (8,))
    assert not torch.allclose(draw[0], draw[1])


def test_uniform_stays_in_unit_interval() -> None:
    values = uniform(_seeds(512), STEP, STREAM, (16,))
    assert values.min() >= 0.0
    assert values.max() < 1.0


def test_normal_moments_are_sane() -> None:
    """Loose moment check: catches a broken mixer, not a subtle bias."""
    values = normal(_seeds(4096), STEP, STREAM, (16,))
    assert abs(values.mean().item()) < 0.02
    assert abs(values.std().item() - 1.0) < 0.02
