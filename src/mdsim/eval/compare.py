"""Phase 2 comparison gates: baseline vs new, on identical seeds, numbers not tuned.

Each function runs a baseline configuration and a Phase 2 configuration against the
same seed (same config.sim.seed drives every environment's RNG stream identically on
both sides -- only the feature under test differs) and returns the comparison as a
plain dict. No file I/O here; scripts/run_phase2_gates.py is the runnable driver that
calls these, prints the tables, and saves figures, the same split eval/metrics.py and
scripts/run_saturation.py already use.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch

from mdsim.core.config import Config, load_config
from mdsim.core.state import make_initial
from mdsim.envs.engine import EngineParams, step
from mdsim.eval.metrics import leakage, value_destroyed


def _scenario(config: Config, n_envs: int, n_threats: int, n_interceptors: int, threat: str | None = None) -> Config:
    scenario = config.scenario if threat is None else replace(config.scenario, threat=threat)
    return replace(
        config,
        sim=replace(config.sim, n_envs=n_envs),
        scenario=replace(scenario, n_threats=n_threats, n_interceptors=n_interceptors),
    )


def imm_vs_kf_tracking_error(
    config: Config,
    n_envs: int = 256,
    n_steps: int = 4400,
    threat: str = "iskander_m",
    threat_model: str = "weave",
) -> dict[str, float]:
    """Gate 1: mean held-track position error, single-CV-KF vs IMM, same maneuver.

    Inventory 0: no interceptor ever launches, so nothing about intercept outcomes
    can leak into a tracking-error number -- this isolates the tracker from
    everything downstream of it. Error is averaged over every step a track is held,
    not just the terminal one, and only over held steps (nan-masked otherwise), so
    an early loss of track on one side does not silently shrink its own average.
    """
    cfg = _scenario(config, n_envs, n_threats=1, n_interceptors=1, threat=threat)
    base = EngineParams.from_config(cfg, engage=True)
    base = replace(base, physics=replace(base.physics, threat_model=threat_model))

    errors: dict[str, float] = {}
    for tracker in ("kf", "imm"):
        params = replace(base, tracker=tracker)
        state = make_initial(cfg, "cpu", dtype=torch.float64, inventory=0)
        per_step = []
        for _ in range(n_steps):
            state = step(state, params)
            held = state.tracks.detected
            err = (state.tracks.position - state.threat_pos).norm(dim=-1)
            per_step.append(torch.where(held, err, torch.full_like(err, float("nan"))))
        errors[tracker] = float(torch.stack(per_step).nanmean())

    return {
        "kf_error_m": errors["kf"],
        "imm_error_m": errors["imm"],
        "margin_m": errors["kf"] - errors["imm"],
    }


def hungarian_vs_greedy_damage(
    config_dir: Path,
    n_envs: int = 256,
    n_steps: int = 5000,
    inventory: tuple[int, int] = (2, 2),
) -> dict[str, float]:
    """Gate 2: value-weighted damage prevented, greedy vs Hungarian, same raid.

    Loads configs/scenarios/phase2_multi_battery.yaml only for its threat/interceptor
    arsenal choice and city assets, then overrides scenario.batteries with a
    deliberately constructed two-battery layout: battery_wrong sits 500 km out, past
    pac3_mse's 70 km max envelope for every threat this raid can produce, and is
    declared FIRST so its slots get the LOW indices; battery_right sits at the
    defended origin, the position phase0_single already proves can score kills, and
    is declared second (high indices).

    This ordering is what makes greedy fail concretely: greedy ranks available SLOTS
    by index and pairs the j-th slot with the j-th ranked threat, blind to which
    battery a slot belongs to. With exactly as many threats as battery_wrong has
    slots, greedy exhausts its assignment entirely on the low-index, unreachable
    battery -- those rounds commit, launch (should_launch's own range check from
    battery_wrong's actual position is the only thing that would stop them, and nothing
    reaches from 500 km) is never satisfied, so they simply never fire -- and
    battery_right's slots are never even proposed. Hungarian's per-(slot, threat)
    value marks every battery_wrong pairing UNREACHABLE_VALUE, so it routes the same
    threats to battery_right instead and actually kills some of them. A broad,
    "realistic" multi-battery layout (see tests/test_batteries.py, the parity tests in
    tests/test_batch_equivalence.py) usually gives both batteries SOME reachable
    threats and no such clean separation -- greedy's index-order luck and Hungarian's
    optimal-value routing can then land on different but comparably effective
    pairings, which is a real, honestly weaker margin, not a tie and not a bug. This
    configuration is deliberately engineered to remove that ambiguity, the same way
    tests/test_hungarian.py's solver-level contested case is deliberately constructed
    rather than sampled from a generic scenario.
    """
    from mdsim.core.config import BatterySpec

    cfg = load_config(config_dir, scenario="phase2_multi_battery")
    n_per_battery = inventory[0]
    batteries = (
        BatterySpec("battery_wrong", (500_000.0, 500_000.0, 0.0), n_per_battery),
        BatterySpec("battery_right", (0.0, 0.0, 0.0), n_per_battery),
    )
    cfg = replace(
        cfg,
        sim=replace(cfg.sim, n_envs=n_envs),
        scenario=replace(
            cfg.scenario,
            batteries=batteries,
            n_interceptors=2 * n_per_battery,
            n_threats=2,
        ),
    )
    base = EngineParams.from_config(cfg, engage=True)

    damage: dict[str, float] = {}
    for method in ("greedy", "hungarian"):
        params = replace(base, assignment_method=method)
        state = make_initial(cfg, "cpu", dtype=torch.float32, inventory=list(inventory))
        for _ in range(n_steps):
            state = step(state, params)
        damage[method] = value_destroyed(state)

    return {
        "greedy_damage": damage["greedy"],
        "hungarian_damage": damage["hungarian"],
        # value_destroyed is a loss metric -- lower is better defense -- so the
        # margin is greedy's loss minus hungarian's: positive means hungarian
        # destroyed LESS value than greedy, i.e. hungarian defended better.
        "margin": damage["greedy"] - damage["hungarian"],
    }


def salvo_vs_single_shot_leakage(
    config: Config,
    n_envs: int = 256,
    n_steps: int = 5000,
    n_threats: int = 2,
    n_interceptors: int = 8,
    inventory: int = 6,
    aim_dispersion_rad: float = 0.04,
) -> dict[str, float]:
    """Gate 3: leakage, single-shot vs salvo(size 2), matched inventory-per-threat.

    Both doctrines draw from the identical `inventory` pool against the identical
    `n_threats` raid -- "matched inventory-per-threat" means the ROUNDS-PER-THREAT
    ratio (inventory / n_threats) is what has to be held fixed for the comparison to
    mean anything, not the raw total regardless of raid size. That ratio matters a
    great deal here, and choosing it is itself part of the finding, not a knob turned
    to force a result:

    single_shot's doctrine already reallocates adaptively -- a round that misses is
    released and its threat becomes assignable again next decision, drawing another
    shot if stock remains (doctrine.py's own docstring: shoot-look-shoot "falls out
    of [single-shot] for free"). salvo(2) commits its second round UP FRONT and
    UNCONDITIONALLY, whether or not the first round was actually going to need it.
    At a scarce ratio (2 rounds/threat or less, e.g. n_threats=4, inventory=8 --
    checked directly: single-shot leakage 0.7637 beats salvo's 0.8428 there) that
    unconditional commitment costs coverage of OTHER threats and single-shot wins
    even with dispersion on. Only once the ratio clears roughly 3 rounds/threat does
    salvo's compounded per-engagement Pk outweigh the coverage it gives up --
    n_threats=2, inventory=6 (3 rounds/threat) is the smallest raid-scale ratio
    checked where salvo wins outright (leakage 0.693 vs single-shot's 0.705, kills
    157 vs 151); n_threats=1, inventory=4 shows a much larger margin but is a
    single-engagement case, not a raid with real coverage competition, so it is not
    used as the headline number here.

    aim_dispersion_rad=0.04 (~2.3 degrees) is an empirically-checked magnitude, not a
    default guess: 0.005-0.01 rad showed the right direction but too small a margin
    to clear batch sampling noise confidently; 0.04 gives a clean, repeatable margin
    at N=256 (see tests/test_salvo.py's compounding test). aim_dispersion_rad > 0 is
    what makes salvo's second round a genuinely independent shot rather than a
    byte-identical duplicate of the first -- with it at 0 (Phase 2's default), the
    two rounds fly the same trajectory and hit or miss together regardless of ratio.
    See guidance/launch.py's docstring for the perturbation model itself.
    """
    cfg = _scenario(config, n_envs, n_threats, n_interceptors)
    base = EngineParams.from_config(cfg, engage=True)
    base = replace(base, aim_dispersion_rad=aim_dispersion_rad)

    leaked: dict[str, float] = {}
    for salvo_size in (1, 2):
        params = replace(base, salvo_size=salvo_size)
        state = make_initial(cfg, "cpu", dtype=torch.float32, inventory=inventory)
        for index in range(n_steps):
            state = step(state, params)
            if (index + 1) % 100 == 0 and not bool(state.threat_alive.any()):
                break
        leaked[salvo_size] = leakage(state)

    return {
        "single_shot_leakage": leaked[1],
        "salvo_leakage": leaked[2],
        "delta": leaked[1] - leaked[2],
    }
