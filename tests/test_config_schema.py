"""Config loader accepts the shipped configs and rejects malformed ones."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from mdsim.core.config import ConfigError, load_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture
def configs(tmp_path: Path) -> Path:
    """A writable copy of the shipped configs, so tests can corrupt one file."""
    target = tmp_path / "configs"
    shutil.copytree(CONFIG_DIR, target)
    return target


def _patch_yaml(path: Path, **changes: object) -> None:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data.update(changes)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def test_shipped_configs_load() -> None:
    config = load_config(CONFIG_DIR)

    assert config.sim.dt == 0.05
    assert config.sim.n_envs == 1024
    assert config.sim.integrator == "semi_implicit"
    assert config.radar.detect_range_km == 400.0
    assert config.scenario.name == "phase0_single"
    assert config.threats[config.scenario.threat].missile_class == "SRBM"
    assert config.interceptors[config.scenario.interceptor].inventory == 8
    assert len(config.cities) >= 1


def test_negative_dt_raises(configs: Path) -> None:
    _patch_yaml(configs / "sim.yaml", dt=-0.05)
    with pytest.raises(ConfigError, match="dt must be positive"):
        load_config(configs)


def test_zero_dt_raises(configs: Path) -> None:
    _patch_yaml(configs / "sim.yaml", dt=0.0)
    with pytest.raises(ConfigError, match="dt must be positive"):
        load_config(configs)


def test_missing_threat_field_raises(configs: Path) -> None:
    path = configs / "arsenal" / "threats.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    del data["threats"]["generic_srbm"]["terminal_speed_mps"]
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ConfigError, match="terminal_speed_mps"):
        load_config(configs)


def test_missing_interceptor_field_raises(configs: Path) -> None:
    path = configs / "arsenal" / "interceptors.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    del data["interceptors"]["generic_mid_tier"]["reaction_time_s"]
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ConfigError, match="reaction_time_s"):
        load_config(configs)


def test_inverted_range_envelope_raises(configs: Path) -> None:
    path = configs / "arsenal" / "interceptors.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["interceptors"]["generic_mid_tier"]["envelope_range_km"] = [60, 5]
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ConfigError, match="envelope_range_km min must be below max"):
        load_config(configs)


def test_inverted_alt_envelope_raises(configs: Path) -> None:
    path = configs / "arsenal" / "interceptors.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["interceptors"]["generic_mid_tier"]["envelope_alt_km"] = [25, 25]
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ConfigError, match="envelope_alt_km min must be below max"):
        load_config(configs)


def test_unknown_scenario_threat_raises(configs: Path) -> None:
    _patch_yaml(configs / "scenarios" / "phase0_single.yaml", threat="does_not_exist")
    with pytest.raises(ConfigError, match="unknown threat"):
        load_config(configs)


def test_bad_integrator_raises(configs: Path) -> None:
    _patch_yaml(configs / "sim.yaml", integrator="leapfrog")
    with pytest.raises(ConfigError, match="integrator must be one of"):
        load_config(configs)
