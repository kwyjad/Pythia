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


# ---------------------------------------------------------------------------
# Phase 1 keys: IBTrACS connector + ReliefWeb silence sweep.
# ---------------------------------------------------------------------------


def test_shipped_rulebook_carries_phase1_detection_keys():
    rb = load_rulebook()
    assert rb.get("cyclone.wind_source_priority") == ["usa_wind", "wmo_wind"]
    assert "{scope}" in rb.get("cyclone.ibtracs.url_template")
    assert rb.get("cyclone.ibtracs.default_scope") == "last3years"
    assert rb.get("cyclone.ibtracs.coverage_grace_days") == 7
    assert rb.get("cyclone.reliefweb_sweep.disaster_types") == ["Tropical Cyclone"]
    assert "cyclone" in rb.get("cyclone.reliefweb_sweep.keywords")
    assert rb.get("cyclone.reliefweb_sweep.max_hits_for_silence") == 0
    assert rb.get("reliefweb.api_base_url").startswith("https://")


def test_unknown_wind_source_fails_validation(tmp_path):
    data = _shipped_data()
    data["cyclone"]["wind_source_priority"] = ["usa_wind", "jtwc_wind"]
    with pytest.raises(RulebookError, match="wind_source_priority"):
        load_rulebook(_write_rulebook(tmp_path, data))


def test_url_template_without_scope_placeholder_fails_validation(tmp_path):
    data = _shipped_data()
    data["cyclone"]["ibtracs"]["url_template"] = "https://example.test/ibtracs.csv"
    with pytest.raises(RulebookError, match="url_template"):
        load_rulebook(_write_rulebook(tmp_path, data))


def test_missing_sweep_keywords_fail_validation(tmp_path):
    data = _shipped_data()
    del data["cyclone"]["reliefweb_sweep"]["keywords"]
    with pytest.raises(RulebookError, match="keywords"):
        load_rulebook(_write_rulebook(tmp_path, data))


# ---------------------------------------------------------------------------
# Phase 2: flood detection, the ladder's sources, and conflict handling
# ---------------------------------------------------------------------------


def test_shipped_rulebook_carries_phase2_keys():
    """Everything Phase 2 reads must be present and correctly typed."""
    rb = load_rulebook()

    # Flood detection + its own silence sweep.
    assert rb.get("flood.gdacs.coverage_grace_days") >= 0
    assert rb.get("flood.reliefweb_sweep.keywords")
    assert rb.get("flood.reliefweb_sweep.disaster_types")

    # Ladder semantics.
    assert rb.get("ladder")[0] == "emdat"
    assert "idmc_idu" in rb.get("lower_bound_rungs")
    assert rb.get("conflict_detection.order_of_magnitude_factor") > 1
    assert rb.get("event_attribution.figure") == "start_month"
    assert rb.get("event_attribution.detection") == "overlap"

    # Source connectors.
    for prefix in ("emdat", "ifrc_go", "idmc_idu"):
        assert rb.get(f"{prefix}.lookback_months") >= 0
        assert rb.get(f"{prefix}.lookahead_months") >= 0
    assert rb.get("emdat.api_url").startswith("https://")
    assert rb.get("ifrc_go.api_base_url").startswith("https://")
    assert rb.get("idmc_idu.api_url").startswith("https://")


def test_every_sweeping_hazard_has_a_complete_sweep_block():
    """A half-configured hazard must fail loudly, not sweep with defaults."""
    from resolver.hazard_resolution.rulebook import SWEEP_HAZARD_KEYS

    rb = load_rulebook()
    for hazard_key in SWEEP_HAZARD_KEYS:
        for key in (
            "disaster_types", "keywords", "publication_pad_days",
            "max_hits_for_silence", "sample_size", "request_timeout_sec",
            "request_delay_sec",
        ):
            assert rb.get(f"{hazard_key}.reliefweb_sweep.{key}") is not None


def test_missing_flood_sweep_block_fails_validation(tmp_path):
    data = _shipped_data()
    del data["flood"]["reliefweb_sweep"]
    with pytest.raises(RulebookError, match="flood.reliefweb_sweep"):
        load_rulebook(_write_rulebook(tmp_path, data))


def test_lower_bound_rung_outside_the_ladder_fails_validation(tmp_path):
    """A lower-bound rung the ladder never walks is a silent no-op."""
    data = _shipped_data()
    data["ladder"] = ["emdat", "ifrc_go"]
    data["lower_bound_rungs"] = ["idmc_idu"]
    with pytest.raises(RulebookError, match="not in the ladder"):
        load_rulebook(_write_rulebook(tmp_path, data))


def test_unknown_lower_bound_rung_fails_validation(tmp_path):
    data = _shipped_data()
    data["lower_bound_rungs"] = ["gdacs"]
    with pytest.raises(RulebookError, match="lower_bound_rungs"):
        load_rulebook(_write_rulebook(tmp_path, data))


@pytest.mark.parametrize("factor", [1.0, 0.5, 0, "ten"])
def test_conflict_factor_must_exceed_one(tmp_path, factor):
    """A factor <= 1 would flag every pair of rungs."""
    data = _shipped_data()
    data["conflict_detection"]["order_of_magnitude_factor"] = factor
    with pytest.raises(RulebookError, match="order_of_magnitude_factor"):
        load_rulebook(_write_rulebook(tmp_path, data))


def test_unknown_hazard_code_in_a_source_map_fails_validation(tmp_path):
    data = _shipped_data()
    data["emdat"]["classif_keys"]["XX"] = ["nat-xxx"]
    with pytest.raises(RulebookError, match="classif_keys"):
        load_rulebook(_write_rulebook(tmp_path, data))


def test_non_https_source_endpoint_fails_validation(tmp_path):
    data = _shipped_data()
    data["idmc_idu"]["api_url"] = "http://insecure.example/idus"
    with pytest.raises(RulebookError, match="idmc_idu.api_url"):
        load_rulebook(_write_rulebook(tmp_path, data))


def test_a_credential_in_a_source_block_is_rejected(tmp_path):
    """API keys come from the environment; the rulebook must refuse one."""
    data = _shipped_data()
    data["emdat"]["api_key"] = "sk-not-allowed-here"
    with pytest.raises(RulebookError, match="looks like a credential"):
        load_rulebook(_write_rulebook(tmp_path, data))


def test_changing_the_ladder_changes_the_winner(tmp_path):
    """The acceptance guarantee, for Phase 2's central decision."""
    from resolver.hazard_resolution.rules import ladder_rank

    data = _shipped_data()
    data["ladder"] = ["ifrc_go", "emdat", "reliefweb_extracted", "idmc_idu"]
    rb = load_rulebook(_write_rulebook(tmp_path, data))

    assert ladder_rank("ifrc_go", rb) == 0
    assert ladder_rank("emdat", rb) == 1
    assert ladder_rank("gdacs", rb) is None  # never a rung


def test_changing_the_conflict_factor_changes_flagging(tmp_path):
    from resolver.hazard_resolution.rules import orders_of_magnitude_apart

    data = _shipped_data()
    data["conflict_detection"]["order_of_magnitude_factor"] = 2.0
    strict = load_rulebook(_write_rulebook(tmp_path, data))

    assert orders_of_magnitude_apart(10_000, 50_000, strict) is True
    assert orders_of_magnitude_apart(10_000, 50_000, load_rulebook()) is False
