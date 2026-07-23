"""Salvo doctrine: up to k rounds per threat per decision, and the inventory change
that lets a threat stay assignable until it holds a full salvo.

`salvo` at k=1 has to be exactly `greedy` -- the engine calls `salvo` unconditionally
with `salvo_size` from config, defaulting to 1, so any drift here is a Phase 1
regression, not a Phase 2 feature gap.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from mdsim.core.config import load_config
from mdsim.core.state import make_initial
from mdsim.engagement import doctrine
from mdsim.engagement.assignment import UNASSIGNED, greedy
from mdsim.engagement.inventory import commit, engaged_threats
from mdsim.envs.engine import EngineParams, step

DTYPE = torch.float64

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_salvo_binds_up_to_salvo_size_slots_to_one_threat() -> None:
    """One threat, three free slots, salvo_size=2: exactly two slots bind, one stays free."""
    priority = torch.tensor([[1.0]], dtype=DTYPE)
    assignable = torch.tensor([[True]])
    available = torch.tensor([[True, True, True]])

    result = doctrine.salvo(priority, assignable, available, salvo_size=2)

    bound = result >= 0
    assert int(bound.sum()) == 2, f"expected 2 bindings, got {result.tolist()}"
    assert bool((result[bound] == 0).all()), "a bound slot points somewhere other than threat 0"
    assert int((result == UNASSIGNED).sum()) == 1, "expected exactly one slot left unbound"


def test_salvo_result_composes_with_commit() -> None:
    """The two committed slots from a salvo land in `interceptor_target` as expected."""
    priority = torch.tensor([[1.0]], dtype=DTYPE)
    assignable = torch.tensor([[True]])
    available = torch.tensor([[True, True, True]])
    interceptor_target = torch.full((1, 3), UNASSIGNED)

    proposal = doctrine.salvo(priority, assignable, available, salvo_size=2)
    committed = commit(interceptor_target, proposal)

    assert int((committed == 0).sum()) == 2
    assert int((committed == UNASSIGNED).sum()) == 1


def test_salvo_size_one_matches_greedy_on_hand_cases() -> None:
    cases = [
        (
            torch.tensor([[0.1, 0.9, 0.5, 0.0]], dtype=DTYPE),
            torch.tensor([[True, True, True, False]]),
            torch.tensor([[True, False, True, True]]),
        ),
        (
            torch.tensor([[0.5, 0.5, 0.1]], dtype=DTYPE),
            torch.tensor([[True, True, True]]),
            torch.tensor([[True, False, False]]),
        ),
        (
            torch.tensor([[0.9, 0.5]], dtype=DTYPE),
            torch.zeros((1, 2), dtype=torch.bool),
            torch.ones((1, 2), dtype=torch.bool),
        ),
    ]
    for priority, assignable, available in cases:
        salvo_result = doctrine.salvo(priority, assignable, available, salvo_size=1)
        greedy_result = greedy(priority, assignable, available)
        assert torch.equal(salvo_result, greedy_result)


def test_salvo_size_one_matches_greedy_across_randomised_batches() -> None:
    """Property test: salvo(size=1) must be bit-for-bit greedy, any shape or mask."""
    torch.manual_seed(0)

    for trial in range(100):
        n_envs = int(torch.randint(1, 9, ()).item())
        n_threats = int(torch.randint(1, 7, ()).item())
        n_interceptors = int(torch.randint(1, 7, ()).item())

        priority = torch.rand((n_envs, n_threats), dtype=DTYPE)
        assignable = torch.rand((n_envs, n_threats)) > 0.4
        available = torch.rand((n_envs, n_interceptors)) > 0.4

        salvo_result = doctrine.salvo(priority, assignable, available, salvo_size=1)
        greedy_result = greedy(priority, assignable, available)
        assert torch.equal(salvo_result, greedy_result), f"trial {trial}"


def test_engaged_threats_default_salvo_size_matches_phase_one_behaviour() -> None:
    """salvo_size defaults to 1, so one bound slot is enough to mark a threat engaged."""
    targets = torch.tensor([[2, UNASSIGNED, 0]])

    assert engaged_threats(targets, 4).tolist() == [[True, False, True, False]]


def test_engaged_threats_waits_for_a_full_salvo_before_marking_engaged() -> None:
    """One slot bound to threat 0: engaged at salvo_size=1, not yet at salvo_size=2."""
    interceptor_target = torch.tensor([[0, UNASSIGNED, UNASSIGNED]])
    n_threats = 2

    assert engaged_threats(interceptor_target, n_threats, salvo_size=1).tolist() == [
        [True, False]
    ]
    assert engaged_threats(interceptor_target, n_threats, salvo_size=2).tolist() == [
        [False, False]
    ]

    # A second slot binds to the same threat: now salvo_size=2 reads it engaged too.
    interceptor_target = torch.tensor([[0, 0, UNASSIGNED]])
    assert engaged_threats(interceptor_target, n_threats, salvo_size=2).tolist() == [
        [True, False]
    ]


def test_shoot_look_shoot_is_documented_as_not_yet_implemented() -> None:
    """Guards against the hook text silently disappearing or getting marked done."""
    text = doctrine.salvo.__doc__.lower()
    assert "shoot-look-shoot" in text
    assert "not implement" in text

    module_text = doctrine.__doc__.lower()
    assert "shoot-look-shoot" in module_text


# ---------------------------------------------------------------------------
# Salvo with per-round aim dispersion: the compounding effect dispersion exists
# to unlock. Without dispersion, every round bound to the same threat in the
# same decision flies bit-identical geometry (see scripts/run_phase2_gates.py's
# gate 3 mechanism note) and salvo's Pk is unchanged from single-shot's. With
# it, two rounds independently miss or hit, so salvo's Pk should measurably
# exceed a lone round's.
# ---------------------------------------------------------------------------

# Picked by direct measurement, not the module default of 0.0: small enough to
# read as plausible aim-point scatter per should_launch's docstring, and large
# enough that the Pk margin below clears sampling noise by a comfortable
# multiple. 0.005-0.01 rad showed the same direction but a margin too close to
# the batch's binomial noise floor to assert on confidently; 0.02 was
# borderline; 0.04 gave a clean, repeatable double-digit-percentage-point
# margin across widely separated seeds (0 and 1000).
AIM_DISPERSION_RAD = 0.04

# base_seed + arange(n_envs) is env_seeds' per-env formula (mdsim.core.rng), so
# nearby base seeds share almost their whole env set shifted by one and are NOT
# independent draws of the aggregate Pk -- confirmed directly (seed 0 and seed
# 1 at N=256 reproduced the same Pk to 4 decimal places). 1000 is far enough
# from 0 to be a genuinely separate draw, and the margin was re-checked at
# both to confirm it is a real effect rather than one batch's coincidence.
DISPERSION_SEED = 1000
N_ENVS = 256

# One threat, three interceptor slots so salvo_size=2 leaves a spare slot
# rather than exhausting every one.
N_THREATS = 1
N_INTERCEPTORS = 3

# phase0_single's threat (scud_b) flies burnout-to-impact over roughly 4200-
# 4400 steps at dt=0.05 in this scenario; both engagements resolve (kill, miss,
# or leak) well inside this horizon, and the idle check exits early once the
# one threat is no longer alive.
N_STEPS = 4600
IDLE_CHECK_EVERY = 200

# Measured margins were +0.078 (seed 0) and +0.090 (seed 1000) at N=256 -- this
# threshold sits comfortably below both so the assertion is not a hair-trigger
# on ordinary batch-to-batch noise, while staying well above the ~0.03 binomial
# standard error at this N and Pk.
MIN_PK_MARGIN = 0.03


def _dispersion_scenario(seed: int):
    config = load_config(CONFIG_DIR)
    return replace(
        config,
        sim=replace(config.sim, n_envs=N_ENVS, seed=seed),
        scenario=replace(
            config.scenario, n_threats=N_THREATS, n_interceptors=N_INTERCEPTORS
        ),
    )


def _run_to_resolution(config, params: EngineParams, inventory: int):
    state = make_initial(config, "cpu", dtype=torch.float32, inventory=inventory)
    for index in range(N_STEPS):
        state = step(state, params)
        if (index + 1) % IDLE_CHECK_EVERY == 0 and not bool(state.threat_alive.any()):
            break
    return state


@pytest.mark.slow
def test_dispersion_makes_salvo_rounds_diverge_and_compound_pk() -> None:
    """End to end: dispersion breaks salvo's lockstep and lifts its Pk over single-shot's.

    Both runs share the same aim_dispersion_rad, so this isolates the salvo-vs-
    single-shot comparison from the separate question of what dispersion alone
    does to a lone round's own Pk -- the fair comparison is dispersed single-
    shot vs dispersed salvo, matched rounds-per-threat aside.
    """
    config = _dispersion_scenario(DISPERSION_SEED)
    base_params = EngineParams.from_config(config, engage=True)

    single_params = replace(base_params, salvo_size=1, aim_dispersion_rad=AIM_DISPERSION_RAD)
    single_state = _run_to_resolution(config, single_params, inventory=1)
    single_pk = float(single_state.threat_killed[:, 0].float().mean())

    salvo_params = replace(base_params, salvo_size=2, aim_dispersion_rad=AIM_DISPERSION_RAD)
    salvo_state = _run_to_resolution(config, salvo_params, inventory=2)
    salvo_pk = float(salvo_state.threat_killed[:, 0].float().mean())

    margin = salvo_pk - single_pk
    print(
        f"single-shot Pk={single_pk:.4f}  salvo(2) Pk={salvo_pk:.4f}  "
        f"margin={margin:+.4f}  (N_envs={N_ENVS}, aim_dispersion_rad={AIM_DISPERSION_RAD})"
    )

    # Divergence: envs where both salvo slots bound to the one threat must not
    # share a frozen terminal position -- interceptor_pos stops updating the
    # step a round leaves interceptor_alive, so this is each round's own
    # retirement point, not a snapshot mid-flight.
    both_bound = (
        salvo_state.interceptor_committed[:, 0]
        & salvo_state.interceptor_committed[:, 1]
        & (salvo_state.interceptor_target[:, 0] == salvo_state.interceptor_target[:, 1])
    )
    n_pairs = int(both_bound.sum())
    assert n_pairs > 0, "no env bound both salvo slots to the same threat"

    pos_a = salvo_state.interceptor_pos[both_bound, 0]
    pos_b = salvo_state.interceptor_pos[both_bound, 1]
    identical = torch.all(pos_a == pos_b, dim=-1)
    n_diverged = int((~identical).sum())
    print(f"salvo pairs bound to the same threat: {n_pairs}, diverged: {n_diverged}")
    assert n_diverged > 0, "no salvo pair's rounds diverged despite aim_dispersion_rad > 0"

    assert margin > MIN_PK_MARGIN, (
        f"salvo(2) Pk ({salvo_pk:.4f}) did not clear single-shot Pk ({single_pk:.4f}) "
        f"by more than {MIN_PK_MARGIN} -- got margin {margin:+.4f}"
    )
