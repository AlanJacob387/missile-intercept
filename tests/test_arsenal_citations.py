"""Every arsenal number is either cited or flagged as an assumption. No third option."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mdsim.core.config import (
    INTERCEPTOR_NUMERIC_FIELDS,
    THREAT_FLAGGABLE_FIELDS,
    THREAT_NUMERIC_FIELDS,
    ConfigError,
    load_config,
    parse_provenance,
)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
CONFIG = load_config(CONFIG_DIR)

THREATS = sorted(CONFIG.threats)
INTERCEPTORS = sorted(CONFIG.interceptors)

EXPECTED_THREATS = {
    "scud_b",
    "iskander_m",
    "df21",
    "df26",
    "minuteman_iii",
    "hwasong_15",
    "df17_hgv",
}
EXPECTED_INTERCEPTORS = {"pac3_mse", "thaad", "sm3_iia", "sm6"}


def _raw(filename: str, key: str) -> dict:
    path = CONFIG_DIR / "arsenal" / filename
    return json.loads(path.read_text(encoding="utf-8"))[key]


def test_expected_entries_present() -> None:
    assert set(THREATS) == EXPECTED_THREATS
    assert set(INTERCEPTORS) == EXPECTED_INTERCEPTORS


@pytest.mark.parametrize("name", THREATS)
def test_threat_entry_is_traceable(name: str) -> None:
    provenance = CONFIG.threats[name].provenance
    cited = set(THREAT_NUMERIC_FIELDS) - provenance.assumed
    if cited:
        assert provenance.sources, f"{name} has cited fields {cited} but no source URL"
    assert provenance.notes.strip(), f"{name} has no notes"


@pytest.mark.parametrize("name", INTERCEPTORS)
def test_interceptor_entry_is_traceable(name: str) -> None:
    provenance = CONFIG.interceptors[name].provenance
    cited = set(INTERCEPTOR_NUMERIC_FIELDS) - provenance.assumed
    if cited:
        assert provenance.sources, f"{name} has cited fields {cited} but no source URL"
    assert provenance.notes.strip(), f"{name} has no notes"


@pytest.mark.parametrize("name", THREATS + INTERCEPTORS)
def test_sources_are_urls(name: str) -> None:
    spec = CONFIG.threats.get(name) or CONFIG.interceptors[name]
    for url in spec.provenance.sources:
        assert url.startswith("https://"), f"{name}: not an https URL: {url}"


@pytest.mark.parametrize("name", THREATS)
def test_threat_reference_speed_is_flagged_assumed(name: str) -> None:
    """No cited threat page states a speed figure, so none may be presented as sourced.

    Checked against the pages: only the DF-17 entry's source prints a speed at all,
    and it prints a Mach 5-10 band rather than the single value used here.
    """
    assert "reference_terminal_speed_mps" in CONFIG.threats[name].provenance.assumed


@pytest.mark.parametrize("name", THREATS)
def test_drag_and_burnout_are_not_presented_as_sourced(name: str) -> None:
    """Neither trajectory parameter may read as transcribed from a citation.

    beta may be assumed or calibrated depending on the entry; burnout altitude is
    assumed throughout. What matters is that neither is silently sourced.
    """
    provenance = CONFIG.threats[name].provenance
    assert provenance.is_unsourced("ballistic_coefficient_beta")
    assert provenance.is_unsourced("burnout_alt_km")


@pytest.mark.parametrize("name", THREATS + INTERCEPTORS)
def test_assumed_and_calibrated_are_disjoint(name: str) -> None:
    """A field is one or the other. Both would mean the entry contradicts itself."""
    provenance = (CONFIG.threats.get(name) or CONFIG.interceptors[name]).provenance
    assert not (provenance.assumed & provenance.calibrated)


def test_calibrated_fields_are_explained() -> None:
    """A tuned number needs its target written down, or it is just a magic constant."""
    for name in THREATS:
        provenance = CONFIG.threats[name].provenance
        if provenance.calibrated:
            assert "calibrat" in provenance.notes.lower(), name


def test_field_both_assumed_and_calibrated_raises() -> None:
    data = dict(_raw("threats.json", "threats")["scud_b"])
    data["assumed"] = list(data["assumed"]) + ["ballistic_coefficient_beta"]
    with pytest.raises(ConfigError, match="both assumed and calibrated"):
        parse_provenance(data, THREAT_FLAGGABLE_FIELDS, "threat 'x'")


def test_unknown_calibrated_field_raises() -> None:
    data = dict(_raw("threats.json", "threats")["scud_b"])
    data["calibrated"] = ["wingspan"]
    with pytest.raises(ConfigError, match="unknown field"):
        parse_provenance(data, THREAT_FLAGGABLE_FIELDS, "threat 'x'")


@pytest.mark.parametrize("name", INTERCEPTORS)
def test_kill_radius_is_flagged_assumed(name: str) -> None:
    """No cited source states a lethal radius for any of these interceptors."""
    assert "kill_radius_m" in CONFIG.interceptors[name].provenance.assumed


def test_no_engine_consumed_speed_is_configured() -> None:
    """Terminal speed is emergent. Nothing the engine reads may declare one.

    `reference_terminal_speed_mps` survives as display metadata, which is why it is
    excluded from the engine's numeric field set rather than merely renamed.
    """
    assert "terminal_speed_mps" not in THREAT_NUMERIC_FIELDS
    assert "reference_terminal_speed_mps" not in THREAT_NUMERIC_FIELDS

    raw = _raw("threats.json", "threats")
    for name, entry in raw.items():
        assert "terminal_speed_mps" not in entry, f"{name} still declares a terminal speed"


@pytest.mark.parametrize("name", INTERCEPTORS)
def test_interceptor_max_g_is_flagged_assumed(name: str) -> None:
    """No public source states an interceptor's lateral-g limit."""
    assert "max_g" in CONFIG.interceptors[name].provenance.assumed


@pytest.mark.parametrize("name", THREATS)
def test_maneuvering_threats_have_divert_budget(name: str) -> None:
    """A threat that pulls g must be given something to spend doing it."""
    spec = CONFIG.threats[name]
    if spec.maneuver_g > 0:
        assert spec.divert_budget > 0
        assert "maneuver_g" in spec.provenance.assumed
    else:
        assert spec.divert_budget == 0


@pytest.mark.parametrize("name", THREATS + INTERCEPTORS)
def test_assumed_fields_are_explained(name: str) -> None:
    """An assumption nobody wrote down is indistinguishable from a made-up number."""
    spec = CONFIG.threats.get(name) or CONFIG.interceptors[name]
    if spec.provenance.assumed:
        assert "assum" in spec.provenance.notes.lower()


# The 33-40 km seam between the SM-6 ceiling and the THAAD floor. Both bounds are
# assumed layer boundaries, so this is a deliberate modeling choice, not an oversight.
KNOWN_ALTITUDE_GAPS_KM = ((33.0, 40.0),)


def test_known_coverage_seam() -> None:
    """The 33-40 km altitude seam is deliberate layered-defense modeling.

    Real layered BMD has handoff seams, and modeling one keeps the Phase 2 assignment
    problem honest: a threat flying through it cannot be engaged by anything, and the
    engine will not announce that -- it simply never launches.

    This asserts the seam's exact bounds rather than merely allowing it, so closing
    the seam by editing an envelope fails here. Coverage changes should be explicit,
    not a side effect of tuning an altitude band.
    """
    bands = sorted(
        (float(s.envelope_alt_km[0]), float(s.envelope_alt_km[1]))
        for s in CONFIG.interceptors.values()
    )

    gaps = []
    reach = bands[0][1]
    for low, high in bands[1:]:
        if low > reach:
            gaps.append((reach, low))
        reach = max(reach, high)

    assert tuple(gaps) == KNOWN_ALTITUDE_GAPS_KM, (
        f"altitude coverage changed: {gaps} vs documented {KNOWN_ALTITUDE_GAPS_KM}"
    )


def test_missing_sources_key_raises() -> None:
    data = dict(_raw("threats.json", "threats")["scud_b"])
    del data["sources"]
    with pytest.raises(ConfigError, match="sources"):
        parse_provenance(data, THREAT_FLAGGABLE_FIELDS, "threat 'x'")


def test_non_url_source_raises() -> None:
    data = dict(_raw("threats.json", "threats")["scud_b"])
    data["sources"] = ["Jane's, 2019, page 44"]
    with pytest.raises(ConfigError, match="not a URL"):
        parse_provenance(data, THREAT_FLAGGABLE_FIELDS, "threat 'x'")


def test_unknown_assumed_field_raises() -> None:
    data = dict(_raw("threats.json", "threats")["scud_b"])
    data["assumed"] = ["wingspan"]
    with pytest.raises(ConfigError, match="unknown field"):
        parse_provenance(data, THREAT_FLAGGABLE_FIELDS, "threat 'x'")


def test_cited_field_without_source_raises() -> None:
    """The rule with teeth: a number presented as sourced must have a source."""
    data = dict(_raw("threats.json", "threats")["scud_b"])
    data["sources"] = []
    with pytest.raises(ConfigError, match="require a source"):
        parse_provenance(data, THREAT_FLAGGABLE_FIELDS, "threat 'x'")


def test_all_assumed_entry_may_omit_sources() -> None:
    """The converse: an entry that claims nothing needs to cite nothing."""
    data = dict(_raw("threats.json", "threats")["scud_b"])
    data["sources"] = []
    data["assumed"] = list(THREAT_FLAGGABLE_FIELDS)
    data["calibrated"] = []
    provenance = parse_provenance(data, THREAT_FLAGGABLE_FIELDS, "threat 'x'")
    assert provenance.sources == ()


def test_empty_notes_raises() -> None:
    data = dict(_raw("threats.json", "threats")["scud_b"])
    data["notes"] = "   "
    with pytest.raises(ConfigError, match="notes"):
        parse_provenance(data, THREAT_FLAGGABLE_FIELDS, "threat 'x'")
