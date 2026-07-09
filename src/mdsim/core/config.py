"""Config loading and validation for sim, radar, arsenal, assets, and scenarios."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

INTEGRATORS = ("semi_implicit", "rk4")
DEVICES = ("xpu", "cuda", "cpu")

# Tried in this order when the requested device is unavailable.
DEVICE_FALLBACK = ("xpu", "cuda", "cpu")

# Required arsenal fields. Missing any is a config error, not a defaulted value:
# a silently defaulted weapon parameter is indistinguishable from a modelled one.
THREAT_NUMERIC_FIELDS = (
    "max_range_km",
    "ballistic_coefficient_beta",
    "burnout_alt_km",
    "maneuver_g",
    "divert_budget",
    "rcs_proxy",
    "payload_value",
)
INTERCEPTOR_NUMERIC_FIELDS = (
    "intercept_speed_mps",
    "max_g",
    "envelope_range_km",
    "envelope_alt_km",
    "reaction_time_s",
    "inventory",
    "kill_radius_m",
)

# Carried for display and for attacker-mode UI later. No engine code reads these:
# terminal speed is emergent from the launch solution and drag, not configured.
THREAT_REFERENCE_FIELDS = ("reference_terminal_speed_mps",)

# Provenance fields, required on every arsenal entry. `assumed` names the numeric
# fields that no public source states, so a reader can tell a transcribed figure
# from a modeling choice without cross-checking the citations by hand.
PROVENANCE_FIELDS = ("sources", "notes", "assumed")

THREAT_FIELDS = (
    "class",
    *THREAT_NUMERIC_FIELDS,
    *THREAT_REFERENCE_FIELDS,
    *PROVENANCE_FIELDS,
)
INTERCEPTOR_FIELDS = (*INTERCEPTOR_NUMERIC_FIELDS, *PROVENANCE_FIELDS)

# Names permitted in an entry's `assumed` list.
THREAT_FLAGGABLE_FIELDS = (*THREAT_NUMERIC_FIELDS, *THREAT_REFERENCE_FIELDS)


class ConfigError(ValueError):
    """Raised when a config file is missing fields or holds an invalid value."""


@dataclass(frozen=True)
class SimConfig:
    dt: float
    n_envs: int
    integrator: str
    seed: int
    device: str
    pn_gain: float
    kf_process_noise_q: float
    city_impact_radius_m: float


@dataclass(frozen=True)
class RadarConfig:
    sigma_range_m: float
    sigma_az_deg: float
    sigma_el_deg: float
    detect_range_km: float
    update_hz: float


@dataclass(frozen=True)
class Provenance:
    """Where an arsenal entry's numbers came from.

    `assumed` lists numeric fields that no cited source states -- modeling choices,
    which a citation must never be read as backing.

    `calibrated` lists fields tuned until the model reproduced an independent
    observable, such as a published terminal-speed band. Neither sourced nor
    arbitrary, and worth separating from both: a calibrated value carries evidence,
    but the evidence is the model's own output rather than a measurement.
    """

    sources: tuple[str, ...]
    notes: str
    assumed: frozenset[str]
    calibrated: frozenset[str] = frozenset()

    def is_assumed(self, field: str) -> bool:
        return field in self.assumed

    def is_unsourced(self, field: str) -> bool:
        return field in self.assumed or field in self.calibrated


@dataclass(frozen=True)
class ThreatSpec:
    name: str
    missile_class: str
    max_range_km: float
    ballistic_coefficient_beta: float
    burnout_alt_km: float
    reference_terminal_speed_mps: float
    maneuver_g: float
    divert_budget: float
    rcs_proxy: float
    payload_value: float
    provenance: Provenance


@dataclass(frozen=True)
class InterceptorSpec:
    name: str
    intercept_speed_mps: float
    max_g: float
    envelope_range_km: tuple[float, float]
    envelope_alt_km: tuple[float, float]
    reaction_time_s: float
    inventory: int
    kill_radius_m: float
    provenance: Provenance


@dataclass(frozen=True)
class CityAsset:
    name: str
    position_m: tuple[float, float, float]
    value: float


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    n_threats: int
    n_interceptors: int
    threat: str
    interceptor: str
    engagement_range_km: float
    launch_elevation_deg: float
    battery_pos_m: tuple[float, float, float]


@dataclass(frozen=True)
class Config:
    sim: SimConfig
    radar: RadarConfig
    threats: Mapping[str, ThreatSpec]
    interceptors: Mapping[str, InterceptorSpec]
    cities: tuple[CityAsset, ...]
    scenario: ScenarioConfig


def _device_available(name: str) -> bool:
    import torch

    if name == "cpu":
        return True
    backend = getattr(torch, name, None)
    return backend is not None and backend.is_available()


def resolve_device(requested: str, verbose: bool = True) -> str:
    """Pick a usable backend, falling back xpu -> cuda -> cpu rather than failing.

    A missing accelerator degrades throughput but not correctness, so it warns and
    keeps running. Silently landing on CPU is the failure mode worth avoiding.
    """
    import torch

    if requested not in DEVICES:
        raise ConfigError(f"sim: device must be one of {DEVICES}, got {requested!r}")

    order = (requested, *(d for d in DEVICE_FALLBACK if d != requested))
    selected = next((d for d in order if _device_available(d)), "cpu")

    if verbose:
        print(f"torch {torch.__version__} | requested {requested} | using {selected}")
    if selected != requested:
        print(
            f"WARNING: {requested} is unavailable, running on {selected}. "
            "Batched throughput will be far below target.",
            file=sys.stderr,
        )
    return selected


def configure_determinism(warn_only: bool = True) -> None:
    """Request deterministic kernels.

    warn_only leaves the run alive when a backend has no deterministic implementation
    for an op. A silently non-deterministic kernel breaks reproducibility without
    failing anything, so the warning is the only signal that it happened.
    """
    import torch

    torch.use_deterministic_algorithms(True, warn_only=warn_only)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"missing config file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level")
    return data


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"missing config file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected an object at the top level")
    return data


def _require(mapping: Mapping[str, Any], fields: tuple[str, ...], where: str) -> None:
    missing = [f for f in fields if f not in mapping]
    if missing:
        raise ConfigError(f"{where}: missing required field(s): {', '.join(missing)}")


def _positive(value: Any, name: str, where: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ConfigError(f"{where}: {name} must be a number, got {value!r}")
    if value <= 0:
        raise ConfigError(f"{where}: {name} must be positive, got {value}")
    return float(value)


def _non_negative(value: Any, name: str, where: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ConfigError(f"{where}: {name} must be a number, got {value!r}")
    if value < 0:
        raise ConfigError(f"{where}: {name} must be non-negative, got {value}")
    return float(value)


def _vec3(value: Any, name: str, where: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ConfigError(f"{where}: {name} must be a length-3 vector, got {value!r}")
    return tuple(float(v) for v in value)  # type: ignore[return-value]


def _interval(value: Any, name: str, where: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ConfigError(f"{where}: {name} must be a [min, max] pair, got {value!r}")
    lo, hi = float(value[0]), float(value[1])
    if lo >= hi:
        raise ConfigError(f"{where}: {name} min must be below max, got [{lo}, {hi}]")
    return lo, hi


def parse_sim(data: Mapping[str, Any]) -> SimConfig:
    fields = (
        "dt",
        "n_envs",
        "integrator",
        "seed",
        "device",
        "pn_gain",
        "kf_process_noise_q",
        "city_impact_radius_m",
    )
    _require(data, fields, "sim")
    dt = _positive(data["dt"], "dt", "sim")

    n_envs = data["n_envs"]
    if not isinstance(n_envs, int) or isinstance(n_envs, bool) or n_envs < 1:
        raise ConfigError(f"sim: n_envs must be an integer >= 1, got {n_envs!r}")

    integrator = data["integrator"]
    if integrator not in INTEGRATORS:
        raise ConfigError(f"sim: integrator must be one of {INTEGRATORS}, got {integrator!r}")

    seed = data["seed"]
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ConfigError(f"sim: seed must be a non-negative integer, got {seed!r}")

    device = data["device"]
    if device not in DEVICES:
        raise ConfigError(f"sim: device must be one of {DEVICES}, got {device!r}")

    return SimConfig(
        dt=dt,
        n_envs=n_envs,
        integrator=integrator,
        seed=seed,
        device=device,
        pn_gain=_positive(data["pn_gain"], "pn_gain", "sim"),
        kf_process_noise_q=_positive(
            data["kf_process_noise_q"], "kf_process_noise_q", "sim"
        ),
        city_impact_radius_m=_positive(
            data["city_impact_radius_m"], "city_impact_radius_m", "sim"
        ),
    )


def parse_radar(data: Mapping[str, Any]) -> RadarConfig:
    fields = (
        "sigma_range_m",
        "sigma_az_deg",
        "sigma_el_deg",
        "detect_range_km",
        "update_hz",
    )
    _require(data, fields, "radar")
    return RadarConfig(**{f: _positive(data[f], f, "radar") for f in fields})


def parse_provenance(
    data: Mapping[str, Any], numeric_fields: tuple[str, ...], where: str
) -> Provenance:
    """Validate an entry's citations and assumption flags."""
    _require(data, PROVENANCE_FIELDS, where)

    raw_sources = data["sources"]
    if not isinstance(raw_sources, list):
        raise ConfigError(f"{where}: sources must be a list of URLs")
    for url in raw_sources:
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            raise ConfigError(f"{where}: source is not a URL: {url!r}")

    notes = data["notes"]
    if not isinstance(notes, str) or not notes.strip():
        raise ConfigError(f"{where}: notes must be a non-empty string")

    raw_assumed = data["assumed"]
    if not isinstance(raw_assumed, list):
        raise ConfigError(f"{where}: assumed must be a list of field names")
    unknown = [f for f in raw_assumed if f not in numeric_fields]
    if unknown:
        raise ConfigError(
            f"{where}: assumed names non-numeric or unknown field(s): "
            f"{', '.join(map(str, unknown))}"
        )

    raw_calibrated = data.get("calibrated", [])
    if not isinstance(raw_calibrated, list):
        raise ConfigError(f"{where}: calibrated must be a list of field names")
    unknown = [f for f in raw_calibrated if f not in numeric_fields]
    if unknown:
        raise ConfigError(
            f"{where}: calibrated names non-numeric or unknown field(s): "
            f"{', '.join(map(str, unknown))}"
        )

    assumed = frozenset(raw_assumed)
    calibrated = frozenset(raw_calibrated)
    overlap = assumed & calibrated
    if overlap:
        raise ConfigError(
            f"{where}: field(s) both assumed and calibrated: {', '.join(sorted(overlap))}"
        )

    # A transcribed figure has to be traceable. Entries whose numbers are all
    # modeling choices are allowed to carry no citation -- what is forbidden is a
    # sourced number with nothing to source it to.
    if set(numeric_fields) - assumed - calibrated and not raw_sources:
        raise ConfigError(f"{where}: non-assumed numeric fields require a source URL")

    return Provenance(
        sources=tuple(raw_sources),
        notes=notes,
        assumed=assumed,
        calibrated=calibrated,
    )


def parse_threat(name: str, data: Mapping[str, Any]) -> ThreatSpec:
    where = f"threat '{name}'"
    _require(data, THREAT_FIELDS, where)

    missile_class = data["class"]
    if not isinstance(missile_class, str) or not missile_class:
        raise ConfigError(f"{where}: class must be a non-empty string")

    return ThreatSpec(
        name=name,
        missile_class=missile_class,
        max_range_km=_positive(data["max_range_km"], "max_range_km", where),
        ballistic_coefficient_beta=_positive(
            data["ballistic_coefficient_beta"], "ballistic_coefficient_beta", where
        ),
        burnout_alt_km=_positive(data["burnout_alt_km"], "burnout_alt_km", where),
        reference_terminal_speed_mps=_positive(
            data["reference_terminal_speed_mps"], "reference_terminal_speed_mps", where
        ),
        maneuver_g=_non_negative(data["maneuver_g"], "maneuver_g", where),
        divert_budget=_non_negative(data["divert_budget"], "divert_budget", where),
        rcs_proxy=_positive(data["rcs_proxy"], "rcs_proxy", where),
        payload_value=_non_negative(data["payload_value"], "payload_value", where),
        provenance=parse_provenance(data, THREAT_FLAGGABLE_FIELDS, where),
    )


def parse_interceptor(name: str, data: Mapping[str, Any]) -> InterceptorSpec:
    where = f"interceptor '{name}'"
    _require(data, INTERCEPTOR_FIELDS, where)

    inventory = data["inventory"]
    if not isinstance(inventory, int) or isinstance(inventory, bool) or inventory < 1:
        raise ConfigError(f"{where}: inventory must be an integer >= 1, got {inventory!r}")

    return InterceptorSpec(
        name=name,
        intercept_speed_mps=_positive(data["intercept_speed_mps"], "intercept_speed_mps", where),
        max_g=_positive(data["max_g"], "max_g", where),
        envelope_range_km=_interval(data["envelope_range_km"], "envelope_range_km", where),
        envelope_alt_km=_interval(data["envelope_alt_km"], "envelope_alt_km", where),
        reaction_time_s=_non_negative(data["reaction_time_s"], "reaction_time_s", where),
        inventory=inventory,
        kill_radius_m=_positive(data["kill_radius_m"], "kill_radius_m", where),
        provenance=parse_provenance(data, INTERCEPTOR_NUMERIC_FIELDS, where),
    )


def parse_city(index: int, data: Mapping[str, Any]) -> CityAsset:
    where = f"city[{index}]"
    _require(data, ("name", "position_m", "value"), where)
    name = data["name"]
    if not isinstance(name, str) or not name:
        raise ConfigError(f"{where}: name must be a non-empty string")
    return CityAsset(
        name=name,
        position_m=_vec3(data["position_m"], "position_m", where),
        value=_positive(data["value"], "value", where),
    )


def parse_scenario(
    data: Mapping[str, Any],
    threats: Mapping[str, ThreatSpec],
    interceptors: Mapping[str, InterceptorSpec],
) -> ScenarioConfig:
    fields = (
        "name",
        "n_threats",
        "n_interceptors",
        "threat",
        "interceptor",
        "engagement_range_km",
        "launch_elevation_deg",
        "battery_pos_m",
    )
    _require(data, fields, "scenario")
    where = f"scenario '{data['name']}'"

    for count_field in ("n_threats", "n_interceptors"):
        count = data[count_field]
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ConfigError(f"{where}: {count_field} must be an integer >= 1, got {count!r}")

    if data["threat"] not in threats:
        raise ConfigError(f"{where}: unknown threat '{data['threat']}'")
    if data["interceptor"] not in interceptors:
        raise ConfigError(f"{where}: unknown interceptor '{data['interceptor']}'")

    engagement_range_km = _positive(
        data["engagement_range_km"], "engagement_range_km", where
    )
    max_range_km = threats[data["threat"]].max_range_km
    if engagement_range_km > max_range_km:
        raise ConfigError(
            f"{where}: engagement_range_km {engagement_range_km} exceeds "
            f"{data['threat']} max_range_km {max_range_km}"
        )

    elevation = data["launch_elevation_deg"]
    if not isinstance(elevation, (int, float)) or not 0.0 < elevation < 90.0:
        raise ConfigError(
            f"{where}: launch_elevation_deg must be strictly between 0 and 90, "
            f"got {elevation!r}"
        )

    return ScenarioConfig(
        name=data["name"],
        n_threats=data["n_threats"],
        n_interceptors=data["n_interceptors"],
        threat=data["threat"],
        interceptor=data["interceptor"],
        engagement_range_km=engagement_range_km,
        launch_elevation_deg=float(elevation),
        battery_pos_m=_vec3(data["battery_pos_m"], "battery_pos_m", where),
    )


def load_config(config_dir: str | Path, scenario: str = "phase0_single") -> Config:
    """Load and validate the full config tree. Raises ConfigError on any violation."""
    root = Path(config_dir)

    sim = parse_sim(_read_yaml(root / "sim.yaml"))
    radar = parse_radar(_read_yaml(root / "radar.yaml"))

    threat_data = _read_json(root / "arsenal" / "threats.json").get("threats")
    if not isinstance(threat_data, dict) or not threat_data:
        raise ConfigError("threats.json: expected a non-empty 'threats' object")
    threats = {name: parse_threat(name, spec) for name, spec in threat_data.items()}

    interceptor_data = _read_json(root / "arsenal" / "interceptors.json").get("interceptors")
    if not isinstance(interceptor_data, dict) or not interceptor_data:
        raise ConfigError("interceptors.json: expected a non-empty 'interceptors' object")
    interceptors = {
        name: parse_interceptor(name, spec) for name, spec in interceptor_data.items()
    }

    city_data = _read_json(root / "assets" / "cities.json").get("cities")
    if not isinstance(city_data, list) or not city_data:
        raise ConfigError("cities.json: expected a non-empty 'cities' array")
    cities = tuple(parse_city(i, spec) for i, spec in enumerate(city_data))

    scenario_cfg = parse_scenario(
        _read_yaml(root / "scenarios" / f"{scenario}.yaml"), threats, interceptors
    )

    return Config(
        sim=sim,
        radar=radar,
        threats=threats,
        interceptors=interceptors,
        cities=cities,
        scenario=scenario_cfg,
    )
