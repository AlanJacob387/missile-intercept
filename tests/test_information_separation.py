"""The architectural constraint: the defender flies on estimates, never on truth.

Per-side information separation is load-bearing for the whole design, and
ground-truth leakage is the failure that breaks it. A leak of this kind produces no
test failure anywhere else -- the simulation simply gets quietly better at
intercepting than any real system could be. So it is asserted structurally, on
signatures and on tensor storage, rather than on behaviour.
"""

from __future__ import annotations

import inspect
import typing
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from mdsim.core.config import load_config
from mdsim.core.state import EnvState, make_initial
from mdsim.envs.engine import EngineParams, step
from mdsim.guidance import launch as launch_module
from mdsim.guidance import pro_nav as pro_nav_module
from mdsim.sensing.tracks import TrackState
from mdsim.world import damage as damage_module

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"

GUIDANCE_MODULES = (pro_nav_module, launch_module)

DECIDING_MODULES = GUIDANCE_MODULES

# Names that would betray a truth argument even if the annotation were loose. The
# specific field names matter as much as the generic words: threat_target_city is the
# one most likely to creep in later, because reading it makes prioritisation trivially
# correct and silently deletes the uncertainty the defender is supposed to face.
FORBIDDEN_SUBSTRINGS = (
    "truth",
    "true_",
    "actual",
    "env_state",
    "envstate",
    "threat_pos",
    "threat_vel",
    "threat_alive",
    "threat_target_city",
    "threat_active",
)

TRUTH_FIELDS = (
    "threat_pos",
    "threat_vel",
    "interceptor_pos",
    "interceptor_vel",
    "threat_target_city",
    "threat_active",
)


def _public_functions(module):
    for name, obj in vars(module).items():
        if name.startswith("_"):
            continue
        if inspect.isfunction(obj) and obj.__module__ == module.__name__:
            yield name, obj


@pytest.mark.parametrize("module", DECIDING_MODULES, ids=lambda m: m.__name__)
def test_deciding_modules_expose_functions(module) -> None:
    """Guards the walks below from silently inspecting nothing."""
    assert list(_public_functions(module)), f"{module.__name__} exposes no functions"


@pytest.mark.parametrize("module", DECIDING_MODULES, ids=lambda m: m.__name__)
def test_no_parameter_is_named_for_truth(module) -> None:
    for name, function in _public_functions(module):
        for parameter in inspect.signature(function).parameters:
            lowered = parameter.lower()
            for forbidden in FORBIDDEN_SUBSTRINGS:
                assert forbidden not in lowered, (
                    f"{module.__name__}.{name} takes '{parameter}', "
                    "which names a truth quantity"
                )


@pytest.mark.parametrize("module", DECIDING_MODULES, ids=lambda m: m.__name__)
def test_no_parameter_is_annotated_with_env_state(module) -> None:
    for name, function in _public_functions(module):
        try:
            hints = typing.get_type_hints(function)
        except NameError:
            # EnvState is not importable from this module's namespace, which is
            # itself the property under test.
            continue
        for parameter, annotation in hints.items():
            assert annotation is not EnvState, (
                f"{module.__name__}.{name} annotates '{parameter}' as EnvState"
            )
            assert "EnvState" not in str(annotation), (
                f"{module.__name__}.{name} annotates '{parameter}' as {annotation}"
            )


@pytest.mark.parametrize("module", DECIDING_MODULES, ids=lambda m: m.__name__)
def test_deciding_modules_do_not_import_env_state(module) -> None:
    """A module that cannot name EnvState cannot accidentally accept one."""
    for attribute, value in vars(module).items():
        assert value is not EnvState, (
            f"{module.__name__} imports EnvState as '{attribute}'"
        )


def test_damage_is_allowed_to_read_truth() -> None:
    """Deliberate asymmetry, asserted so it is not "fixed" later.

    world.damage scores what happened. It is outcome accounting, not a defender
    decision, so it reads truth by design -- which city a threat was aimed at, and
    whether it was still alive. Holding it to the estimates-only rule would mean
    scoring the run from the defender's beliefs, and a defender that missed a leak
    would also fail to be charged for it.
    """
    parameters = inspect.signature(damage_module.resolve_leaks).parameters
    assert "threat_target_city" in parameters
    assert "threat_alive" in parameters


def test_guidance_accepts_track_state_only() -> None:
    """The estimate-side entry points must be typed to TrackState."""
    signature = inspect.signature(pro_nav_module.pn_accel)
    assert typing.get_type_hints(pro_nav_module.pn_accel)["tracks"] is TrackState
    assert "tracks" in signature.parameters

    launch_hints = typing.get_type_hints(launch_module.should_launch)
    assert launch_hints["tracks"] is TrackState


@pytest.fixture(scope="module")
def evolved_state() -> EnvState:
    """A state carried far enough for the tracker to have opened and updated tracks.

    Several threat slots, so the storage check covers the multi-threat tensors the
    world layer reads rather than a degenerate single column.
    """
    config = load_config(CONFIG_DIR)
    config = replace(
        config,
        sim=replace(config.sim, n_envs=8),
        scenario=replace(config.scenario, n_threats=3, n_interceptors=1),
    )
    params = EngineParams.from_config(config, engage=True)

    state = make_initial(config, "cpu", dtype=torch.float64)
    for _ in range(25):
        state = step(state, params)
    return state


def test_tracker_actually_ran(evolved_state: EnvState) -> None:
    """Storage checks below are vacuous if no track was ever opened."""
    assert bool(evolved_state.tracks.detected.any())


def test_track_tensors_share_no_storage_with_truth(evolved_state: EnvState) -> None:
    """Storage identity, not value equality.

    An estimate that happens to equal truth on one step is a coincidence; an estimate
    that IS truth is a leak. Only the data pointer distinguishes them.
    """
    truth_pointers = {}
    for field in TRUTH_FIELDS:
        tensor = getattr(evolved_state, field)
        truth_pointers[field] = tensor.untyped_storage().data_ptr()

    for track_field, tensor in evolved_state.tracks.to_dict().items():
        pointer = tensor.untyped_storage().data_ptr()
        for truth_field, truth_pointer in truth_pointers.items():
            assert pointer != truth_pointer, (
                f"tracks.{track_field} shares storage with truth {truth_field}"
            )


def test_track_estimates_differ_from_truth(evolved_state: EnvState) -> None:
    """Corroborates the storage check: a noisy sensor cannot reproduce truth exactly."""
    estimate = evolved_state.tracks.position
    truth = evolved_state.threat_pos
    assert not torch.allclose(estimate, truth, rtol=0.0, atol=1e-9)
