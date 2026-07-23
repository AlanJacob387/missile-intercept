"""Per-round aim dispersion in should_launch.

Without dispersion, every interceptor slot bound to the same threat on the
same step computes the identical predicted intercept point and flies the
identical heading -- multiple rounds at one threat are then not independent
shots, they are one shot copied. aim_dispersion_rad exists to break that
lockstep with a small, seeded angular aim error per slot.
"""

from __future__ import annotations

import torch

from mdsim.core.rng import env_seeds
from mdsim.guidance.launch import should_launch
from mdsim.sensing.tracks import TrackState

DTYPE = torch.float64
INTERCEPTOR_SPEED_MPS = 1700.0
ENVELOPE_MIN_M = 5000.0
ENVELOPE_MAX_M = 60000.0
AIM_DISPERSION_RAD = 0.02

# Crossing geometry re-used from test_pro_nav.py: known to land the predicted
# intercept point inside the envelope above, so `launch` comes back True.
TARGET_POS_M = (40000.0, 20000.0, 10000.0)
TARGET_VEL_MPS = (-800.0, -300.0, -50.0)


def _track(n_envs: int, pos: tuple[float, ...], vel: tuple[float, ...]) -> TrackState:
    """A held, single-threat track, identical across environments."""
    x_est = torch.tensor([list(pos) + list(vel)], dtype=DTYPE)
    x_est = x_est.unsqueeze(0).expand(n_envs, 1, 6).clone()
    return TrackState(
        x_est=x_est,
        P=torch.eye(6, dtype=DTYPE).reshape(1, 1, 6, 6).expand(n_envs, 1, 6, 6).clone(),
        detected=torch.ones((n_envs, 1), dtype=torch.bool),
        age=torch.zeros((n_envs, 1), dtype=DTYPE),
        updated=torch.ones((n_envs, 1), dtype=torch.bool),
    )


def _slice(state: TrackState, index: int) -> TrackState:
    return TrackState.from_dict(
        {name: tensor[index : index + 1].clone() for name, tensor in state.to_dict().items()}
    )


def test_two_rounds_at_one_battery_diverge_with_dispersion() -> None:
    """Two slots at an identical launch point must fly distinct headings.

    This is exactly the lockstep case the feature fixes: two rounds bound to
    the same threat, launched from the same point on the same step.
    """
    tracks = _track(1, TARGET_POS_M, TARGET_VEL_MPS)
    launch_pos = torch.zeros((1, 2, 3), dtype=DTYPE)
    already_committed = torch.zeros((1, 2), dtype=torch.bool)
    seed = torch.tensor([7], dtype=torch.int64)
    step_index = torch.tensor([100], dtype=torch.int64)

    launch, velocity = should_launch(
        tracks,
        launch_pos,
        already_committed,
        INTERCEPTOR_SPEED_MPS,
        ENVELOPE_MIN_M,
        ENVELOPE_MAX_M,
        seed=seed,
        step_index=step_index,
        aim_dispersion_rad=AIM_DISPERSION_RAD,
    )

    assert bool(launch[0, 0]) and bool(launch[0, 1]), "geometry must be launchable or nothing is tested"
    assert not torch.equal(velocity[0, 0], velocity[0, 1])

    # Contrast case: with no dispersion, the two identically-placed slots must
    # still be byte-identical -- that lockstep is the bug this feature fixes.
    _, velocity_flat = should_launch(
        tracks,
        launch_pos,
        already_committed,
        INTERCEPTOR_SPEED_MPS,
        ENVELOPE_MIN_M,
        ENVELOPE_MAX_M,
    )
    assert torch.equal(velocity_flat[0, 0], velocity_flat[0, 1])


def test_aim_dispersion_rad_requires_seed_and_step() -> None:
    tracks = _track(1, TARGET_POS_M, TARGET_VEL_MPS)
    launch_pos = torch.zeros((1, 2, 3), dtype=DTYPE)
    already_committed = torch.zeros((1, 2), dtype=torch.bool)

    try:
        should_launch(
            tracks,
            launch_pos,
            already_committed,
            INTERCEPTOR_SPEED_MPS,
            ENVELOPE_MIN_M,
            ENVELOPE_MAX_M,
            aim_dispersion_rad=AIM_DISPERSION_RAD,
        )
    except ValueError:
        return
    raise AssertionError("aim_dispersion_rad > 0 without seed/step_index must raise")


def test_dispersion_is_deterministic() -> None:
    """Same (seed, step_index) inputs, called twice, must draw bit-identical noise."""
    tracks = _track(1, TARGET_POS_M, TARGET_VEL_MPS)
    launch_pos = torch.zeros((1, 2, 3), dtype=DTYPE)
    already_committed = torch.zeros((1, 2), dtype=torch.bool)
    seed = torch.tensor([7], dtype=torch.int64)
    step_index = torch.tensor([100], dtype=torch.int64)

    kwargs = dict(
        tracks=tracks,
        launch_pos=launch_pos,
        already_committed=already_committed,
        interceptor_speed=INTERCEPTOR_SPEED_MPS,
        envelope_min_m=ENVELOPE_MIN_M,
        envelope_max_m=ENVELOPE_MAX_M,
        seed=seed,
        step_index=step_index,
        aim_dispersion_rad=AIM_DISPERSION_RAD,
    )
    launch_a, velocity_a = should_launch(**kwargs)
    launch_b, velocity_b = should_launch(**kwargs)

    assert torch.equal(launch_a, launch_b)
    assert torch.equal(velocity_a, velocity_b)


def test_dispersion_is_batch_invariant_across_envs() -> None:
    """Env i's draw must match whether it runs alone or inside a larger batch.

    Mirrors tests/test_batch_equivalence.py::test_batch_invariance: slice one
    env out of a batched call and compare against that env run solo.
    """
    n_envs = 4
    env_index = 2
    base_seed = 123

    tracks = _track(n_envs, TARGET_POS_M, TARGET_VEL_MPS)
    launch_pos = torch.zeros((n_envs, 2, 3), dtype=DTYPE)
    already_committed = torch.zeros((n_envs, 2), dtype=torch.bool)
    seed = env_seeds(base_seed, n_envs, "cpu")
    step_index = torch.full((n_envs,), 50, dtype=torch.int64)

    launch_batched, velocity_batched = should_launch(
        tracks,
        launch_pos,
        already_committed,
        INTERCEPTOR_SPEED_MPS,
        ENVELOPE_MIN_M,
        ENVELOPE_MAX_M,
        seed=seed,
        step_index=step_index,
        aim_dispersion_rad=AIM_DISPERSION_RAD,
    )

    launch_solo, velocity_solo = should_launch(
        _slice(tracks, env_index),
        launch_pos[env_index : env_index + 1],
        already_committed[env_index : env_index + 1],
        INTERCEPTOR_SPEED_MPS,
        ENVELOPE_MIN_M,
        ENVELOPE_MAX_M,
        seed=seed[env_index : env_index + 1],
        step_index=step_index[env_index : env_index + 1],
        aim_dispersion_rad=AIM_DISPERSION_RAD,
    )

    assert torch.equal(launch_batched[env_index], launch_solo[0])
    assert torch.equal(velocity_batched[env_index], velocity_solo[0])


def test_zero_dispersion_reproduces_prior_behaviour_exactly() -> None:
    """aim_dispersion_rad=0.0 is the regression guard: byte-identical to before.

    Existing callers pass none of seed, step_index or aim_dispersion_rad, so
    this compares that call form against the same call with the new
    parameters present but inert.
    """
    tracks = _track(1, TARGET_POS_M, TARGET_VEL_MPS)
    launch_pos = torch.zeros((1, 3, 3), dtype=DTYPE)
    already_committed = torch.zeros((1, 3), dtype=torch.bool)
    seed = torch.tensor([7], dtype=torch.int64)
    step_index = torch.tensor([100], dtype=torch.int64)

    launch_old, velocity_old = should_launch(
        tracks,
        launch_pos,
        already_committed,
        INTERCEPTOR_SPEED_MPS,
        ENVELOPE_MIN_M,
        ENVELOPE_MAX_M,
    )

    launch_new, velocity_new = should_launch(
        tracks,
        launch_pos,
        already_committed,
        INTERCEPTOR_SPEED_MPS,
        ENVELOPE_MIN_M,
        ENVELOPE_MAX_M,
        seed=seed,
        step_index=step_index,
        aim_dispersion_rad=0.0,
    )

    assert torch.equal(launch_old, launch_new)
    assert torch.equal(velocity_old, velocity_new)
