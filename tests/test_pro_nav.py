"""Proportional navigation closes on a crossing target when the track is perfect.

Noise-free tracks isolate the guidance law: any miss here is PN or the interceptor
flight limits, not the sensor or the filter. Building a TrackState from truth values
inside a test is fine -- the architectural constraint is that no function in
guidance/ may take a truth tensor as a parameter, not that tests may not construct
estimates that happen to be exact.
"""

from __future__ import annotations

import math

import torch

from mdsim.core.intercept import closest_approach
from mdsim.core.integrator import step_semi_implicit
from mdsim.core.interceptor import clamp_accel, clamp_speed
from mdsim.guidance.launch import should_launch
from mdsim.guidance.pro_nav import pn_accel
from mdsim.sensing.tracks import TrackState

DTYPE = torch.float64
DT = 0.05
KILL_RADIUS_M = 10.0

# The collision solution puts intercept at ~18.1 s, i.e. step ~362. The run is capped
# just past that on purpose: the interceptor is twice the target's speed, so given
# hundreds of extra steps even pure pursuit would win a stern chase and the test
# would pass without PN doing anything. Bounding the run makes it measure the
# intercept it is supposed to measure.
N_STEPS = 380
EXPECTED_INTERCEPT_STEP = 362
INTERCEPT_STEP_TOLERANCE = 25

PN_GAIN = 4.0
INTERCEPTOR_SPEED_MPS = 1700.0
MAX_ACCEL_MPS2 = 30.0 * 9.80665
ENVELOPE_MIN_M = 5000.0
ENVELOPE_MAX_M = 60000.0

# Crossing geometry: the target is not closing along the launch bearing, so the
# collision triangle has a real lead angle and PN has something to do.
TARGET_POS_M = (40000.0, 20000.0, 10000.0)
TARGET_VEL_MPS = (-800.0, -300.0, -50.0)


def _perfect_track(pos: tuple[float, ...], vel: tuple[float, ...]) -> TrackState:
    """A held track whose estimate carries no error."""
    x_est = torch.tensor([[list(pos) + list(vel)]], dtype=DTYPE)
    return TrackState(
        x_est=x_est,
        P=torch.eye(6, dtype=DTYPE).reshape(1, 1, 6, 6).clone(),
        detected=torch.ones((1, 1), dtype=torch.bool),
        age=torch.zeros((1, 1), dtype=DTYPE),
        updated=torch.ones((1, 1), dtype=torch.bool),
    )


def _fly(initial_velocity: torch.Tensor | None = None) -> tuple[float, int]:
    """Run the engagement, returning (minimum closest approach, step it occurred on)."""
    target_pos = torch.tensor([[list(TARGET_POS_M)]], dtype=DTYPE)
    target_vel = torch.tensor([[list(TARGET_VEL_MPS)]], dtype=DTYPE)

    interceptor_pos = torch.zeros((1, 1, 3), dtype=DTYPE)
    tracks = _perfect_track(TARGET_POS_M, TARGET_VEL_MPS)

    launch, launch_velocity = should_launch(
        tracks,
        interceptor_pos,
        torch.zeros((1, 1), dtype=torch.bool),
        INTERCEPTOR_SPEED_MPS,
        ENVELOPE_MIN_M,
        ENVELOPE_MAX_M,
    )
    assert bool(launch[0, 0]), "geometry must be inside the envelope or nothing is tested"

    interceptor_vel = launch_velocity if initial_velocity is None else initial_velocity

    best = float("inf")
    best_step = -1
    for index in range(N_STEPS):
        tracks = _perfect_track(
            tuple(target_pos[0, 0].tolist()), tuple(target_vel[0, 0].tolist())
        )

        command = pn_accel(tracks, interceptor_pos, interceptor_vel, PN_GAIN)
        command = clamp_accel(command, MAX_ACCEL_MPS2)

        new_interceptor_pos, new_interceptor_vel = step_semi_implicit(
            interceptor_pos, interceptor_vel, command, DT
        )
        new_interceptor_vel = clamp_speed(new_interceptor_vel, INTERCEPTOR_SPEED_MPS)

        new_target_pos = target_pos + target_vel * DT

        delta_pos = interceptor_pos - target_pos
        delta_vel = (
            (new_interceptor_pos - interceptor_pos) - (new_target_pos - target_pos)
        ) / DT
        distance, _ = closest_approach(delta_pos, delta_vel, DT)
        if float(distance) < best:
            best, best_step = float(distance), index

        interceptor_pos, interceptor_vel = new_interceptor_pos, new_interceptor_vel
        target_pos = new_target_pos

    return best, best_step


def test_launch_solution_is_a_real_collision_triangle() -> None:
    """The predicted intercept point must be reachable in the same time by both."""
    from mdsim.guidance.launch import predicted_intercept

    tracks = _perfect_track(TARGET_POS_M, TARGET_VEL_MPS)
    launch_pos = torch.zeros((1, 1, 3), dtype=DTYPE)

    point, time_to_go, feasible = predicted_intercept(
        tracks.position, tracks.velocity, launch_pos, INTERCEPTOR_SPEED_MPS
    )

    assert bool(feasible[0, 0])
    assert float(time_to_go[0, 0]) > 0.0
    # The interceptor must cover the distance to the point in exactly t_go.
    reach = float((point - launch_pos).norm())
    assert math.isclose(
        reach, INTERCEPTOR_SPEED_MPS * float(time_to_go[0, 0]), rel_tol=1e-9
    )


def test_pn_intercepts_crossing_target() -> None:
    miss, at_step = _fly()
    print(f"PN miss distance, nominal launch: {miss:.6f} m at step {at_step}")
    assert miss < KILL_RADIUS_M
    assert abs(at_step - EXPECTED_INTERCEPT_STEP) < INTERCEPT_STEP_TOLERANCE, (
        f"closest approach at step {at_step}, not near the predicted intercept "
        f"({EXPECTED_INTERCEPT_STEP}); this is a stern chase, not an intercept"
    )


def test_pn_corrects_a_launch_heading_error() -> None:
    """PN must earn its place: fly off the collision triangle and still hit.

    A perfect launch solution puts the interceptor on a collision course before PN
    does anything, so the nominal case alone cannot distinguish working guidance
    from no guidance at all.
    """
    tracks = _perfect_track(TARGET_POS_M, TARGET_VEL_MPS)
    _, launch_velocity = should_launch(
        tracks,
        torch.zeros((1, 1, 3), dtype=DTYPE),
        torch.zeros((1, 1), dtype=torch.bool),
        INTERCEPTOR_SPEED_MPS,
        ENVELOPE_MIN_M,
        ENVELOPE_MAX_M,
    )

    # Rotate the launch heading 10 degrees about z, keeping the speed.
    angle = math.radians(10.0)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    rotation = torch.tensor(
        [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]], dtype=DTYPE
    )
    skewed = launch_velocity @ rotation.T

    assert not torch.allclose(skewed, launch_velocity)

    miss, at_step = _fly(initial_velocity=skewed)
    print(f"PN miss distance, 10 deg launch error: {miss:.6f} m at step {at_step}")
    assert miss < KILL_RADIUS_M
    assert abs(at_step - EXPECTED_INTERCEPT_STEP) < INTERCEPT_STEP_TOLERANCE


def test_pn_command_is_perpendicular_to_line_of_sight() -> None:
    """Structural check on the law: PN never commands along the LOS."""
    tracks = _perfect_track(TARGET_POS_M, TARGET_VEL_MPS)
    interceptor_pos = torch.zeros((1, 1, 3), dtype=DTYPE)
    interceptor_vel = torch.tensor([[[1200.0, 400.0, 300.0]]], dtype=DTYPE)

    command = pn_accel(tracks, interceptor_pos, interceptor_vel, PN_GAIN)

    line_of_sight = tracks.position - interceptor_pos
    projection = (command * line_of_sight).sum(-1) / line_of_sight.norm(dim=-1)
    assert abs(float(projection)) < 1e-9
    assert float(command.norm()) > 0.0


def test_pn_is_silent_without_a_track() -> None:
    tracks = _perfect_track(TARGET_POS_M, TARGET_VEL_MPS)
    dropped = TrackState(
        x_est=tracks.x_est,
        P=tracks.P,
        detected=torch.zeros((1, 1), dtype=torch.bool),
        age=tracks.age,
        updated=tracks.updated,
    )

    command = pn_accel(
        dropped,
        torch.zeros((1, 1, 3), dtype=DTYPE),
        torch.tensor([[[1200.0, 400.0, 300.0]]], dtype=DTYPE),
        PN_GAIN,
    )
    assert float(command.abs().max()) == 0.0
