# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Rulebook loading, validation, and the change-the-YAML-change-the-behaviour
acceptance guarantee for the hazard resolution machine."""

from __future__ import annotations

import datetime as dt

import pytest
import yaml

from resolver.hazard_resolution.rulebook import (
    DEFAULT_RULEBOOK_PATH,
    RulebookError,
    load_rulebook,
    validate_rulebook,
)
from resolver.hazard_resolution.rules import (
    cyclone_track_qualifies,
    flood_alert_qualifies,
    freeze_deadline,
    within_sanity_ceiling,
)


def _shipped_data() -> dict:
    with DEFAULT_RULEBOOK_PATH.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def _write_rulebook(tmp_path, data: dict):
    path = tmp_path / "rulebook.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_shipped_rulebook_loads_and_validates():
    rb = load_rulebook()
    assert rb.get("cyclone.buffer_km") == 200
    assert rb.get("cyclone.min_wind_kt") == 34
    assert rb.get("flood.gdacs_trigger_level") == "orange"
    assert rb.get("drought.rule") == "ipc_phase3plus_delta"
    assert rb.get("freeze_days") == 60
    assert rb.get("ladder") == ["emdat", "reliefweb_extracted", "ifrc_go", "idmc_idu"]
    assert rb.get("sanity.ceiling_source") == "gdacs_exposed"
    assert rb.get("sanity.ceiling_multiplier") == 1.0
    assert rb.get("sanity.population_cap") is True
    assert rb.get("conflict_rule") == "ladder_with_flag"
    assert rb.get("severity_base_rate_window_start") == 2015
    assert rb.get("backcast") == {"cyclone": 2000, "flood": 2010, "drought": 2017}


def test_dotted_get_default_and_keyerror():
    rb = load_rulebook()
    assert rb.get("cyclone.no_such_key", "fallback") == "fallback"
    with pytest.raises(KeyError):
        rb.get("cyclone.no_such_key")


def test_missing_key_fails_validation(tmp_path):
    data = _shipped_data()
    del data["freeze_days"]
    with pytest.raises(RulebookError, match="freeze_days"):
        load_rulebook(_write_rulebook(tmp_path, data))


def test_unknown_ladder_rung_fails_validation(tmp_path):
    data = _shipped_data()
    # GDACS must never be a ladder rung: detection + ceiling only.
    data["ladder"] = ["emdat", "gdacs_exposed"]
    with pytest.raises(RulebookError, match="unknown rungs"):
        load_rulebook(_write_rulebook(tmp_path, data))


def test_duplicate_ladder_rung_fails_validation(tmp_path):
    data = _shipped_data()
    data["ladder"] = ["emdat", "emdat"]
    with pytest.raises(RulebookError, match="duplicate"):
        load_rulebook(_write_rulebook(tmp_path, data))


def test_bad_gdacs_level_fails_validation(tmp_path):
    data = _shipped_data()
    data["flood"]["gdacs_trigger_level"] = "purple"
    with pytest.raises(RulebookError, match="gdacs_trigger_level"):
        load_rulebook(_write_rulebook(tmp_path, data))


def test_credential_looking_key_is_rejected(tmp_path):
    # Hard rule: API keys come from env vars, never from config.
    data = _shipped_data()
    data["emdat_api_key"] = "abc123"
    with pytest.raises(RulebookError, match="credential"):
        load_rulebook(_write_rulebook(tmp_path, data))


def test_validation_reports_all_problems_at_once():
    data = _shipped_data()
    del data["freeze_days"]
    data["conflict_rule"] = "nope"
    data["backcast"]["cyclone"] = "yesterday"
    problems = validate_rulebook(data)
    joined = "\n".join(problems)
    assert "freeze_days" in joined
    assert "conflict_rule" in joined
    assert "backcast.cyclone" in joined


# ---------------------------------------------------------------------------
# Acceptance: changing a rulebook VALUE changes behaviour, with no code change.
# ---------------------------------------------------------------------------


def test_changing_cyclone_buffer_changes_detection(tmp_path):
    baseline = load_rulebook()
    # 150 km inside the default 200 km buffer, at gale strength.
    assert cyclone_track_qualifies(distance_km=150, max_wind_kt=40, rulebook=baseline) is True

    data = _shipped_data()
    data["cyclone"]["buffer_km"] = 100
    tightened = load_rulebook(_write_rulebook(tmp_path, data))
    assert cyclone_track_qualifies(distance_km=150, max_wind_kt=40, rulebook=tightened) is False


def test_changing_flood_trigger_level_changes_detection(tmp_path):
    baseline = load_rulebook()
    assert flood_alert_qualifies("orange", baseline) is True
    assert flood_alert_qualifies("green", baseline) is False
    assert flood_alert_qualifies("red", baseline) is True

    data = _shipped_data()
    data["flood"]["gdacs_trigger_level"] = "red"
    stricter = load_rulebook(_write_rulebook(tmp_path, data))
    assert flood_alert_qualifies("orange", stricter) is False
    assert flood_alert_qualifies("red", stricter) is True

    with pytest.raises(ValueError, match="unknown GDACS alert level"):
        flood_alert_qualifies("purple", baseline)


def test_changing_freeze_days_moves_the_freeze_deadline(tmp_path):
    baseline = load_rulebook()
    assert freeze_deadline(2026, 1, baseline) == dt.date(2026, 4, 1)  # Jan 31 + 60d

    data = _shipped_data()
    data["freeze_days"] = 30
    shorter = load_rulebook(_write_rulebook(tmp_path, data))
    assert freeze_deadline(2026, 1, shorter) == dt.date(2026, 3, 2)  # Jan 31 + 30d


def test_changing_ceiling_multiplier_changes_sanity_check(tmp_path):
    baseline = load_rulebook()
    # Default multiplier 1.0: a figure above GDACS exposure fails the ceiling.
    assert within_sanity_ceiling(120_000, exposed_population=100_000, rulebook=baseline) is False
    assert within_sanity_ceiling(80_000, exposed_population=100_000, rulebook=baseline) is True
    # No exposure estimate -> no ceiling to apply.
    assert within_sanity_ceiling(120_000, exposed_population=None, rulebook=baseline) is True

    data = _shipped_data()
    data["sanity"]["ceiling_multiplier"] = 1.5
    looser = load_rulebook(_write_rulebook(tmp_path, data))
    assert within_sanity_ceiling(120_000, exposed_population=100_000, rulebook=looser) is True
