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
    # Above 1.0 since Sept 2026: at exactly 1.0 there was no room between a
    # modelled estimate and a reported one, so every report above the model
    # was flagged. The mechanism is what matters, not the number.
    assert float(rb.get("sanity.ceiling_multiplier")) > 1.0
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
    multiplier = float(baseline.get("sanity.ceiling_multiplier"))
    exposure = 100_000

    # Derived from the shipped multiplier, not hardcoded: the value moved
    # 1.0 -> 3.0 and a test that pins the number has to be edited whenever
    # policy does, which teaches nothing about the mechanism.
    assert within_sanity_ceiling(
        exposure * multiplier * 1.5, exposed_population=exposure, rulebook=baseline
    ) is False
    assert within_sanity_ceiling(
        exposure * multiplier * 0.5, exposed_population=exposure, rulebook=baseline
    ) is True
    # No exposure estimate -> no ceiling to apply.
    assert within_sanity_ceiling(
        exposure * 90, exposed_population=None, rulebook=baseline
    ) is True

    data = _shipped_data()
    data["sanity"]["ceiling_multiplier"] = multiplier * 2
    looser = load_rulebook(_write_rulebook(tmp_path, data))
    assert within_sanity_ceiling(
        exposure * multiplier * 1.5, exposed_population=exposure, rulebook=looser
    ) is True


def test_a_zero_exposure_is_not_a_ceiling_of_zero():
    """It means GDACS declined to say, and unknown imposes no bound.

    GDACS discovery carries no population figure; the per-event RSS fetch
    that fills it in tolerates 404s and leaves the field at its 0.0 default.
    Read as a ceiling, that rejected every figure for the event — 147 of the
    199 extracted figures rejected in the August 2026 run were rejected
    against a ceiling of zero.
    """

    baseline = load_rulebook()
    assert within_sanity_ceiling(
        250_000, exposed_population=0.0, rulebook=baseline
    ) is True
    # A real exposure still binds. (1.0 would not: below the plausibility
    # floor an exposure is a parse failure, not a bound — see the floor test.)
    assert within_sanity_ceiling(
        250_000, exposed_population=10_000.0, rulebook=baseline
    ) is False


def test_the_population_fallback_share_is_a_fraction():
    """0 disables it; above 1 it would sit above the population cap and be inert."""

    share = load_rulebook().get("sanity.population_fallback_share")
    assert 0.0 <= float(share) <= 1.0


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


# ---------------------------------------------------------------------------
# Phase 3: document selection, figure precedence, extraction policy
# ---------------------------------------------------------------------------


def test_shipped_rulebook_carries_phase3_keys():
    """Everything Phase 3 reads must be present and correctly typed."""
    rb = load_rulebook()

    # Document selection.
    assert rb.get("reliefweb.documents.formats")
    assert rb.get("reliefweb.documents.max_docs_per_cell") >= 1
    assert rb.get("reliefweb.documents.candidate_pool_size") >= rb.get(
        "reliefweb.documents.max_docs_per_cell"
    )
    assert rb.get("reliefweb.documents.source_priority")[0].lower() == "ocha"
    assert rb.get("reliefweb.documents.body_char_limit") > 0

    # Figure precedence: the spec's order, top to bottom.
    tiers = [entry["tier"] for entry in rb.get("reliefweb.authority_precedence")]
    assert tiers == ["government", "un_agency", "ifrc_ngo", "media"]

    # Households conversion.
    assert rb.get("reliefweb.household_conversion.default_multiplier") > 0
    assert rb.get("reliefweb.household_conversion.by_iso3")["PHL"] > 0

    # Extraction policy — a ROLE, never a model id.
    assert rb.get("extraction.enabled") is True
    assert ":" not in rb.get("extraction.model_role")
    assert rb.get("extraction.max_calls_per_month") >= 0
    assert rb.get("extraction.skip_when_higher_rung_populated") is True


def test_a_model_ref_in_place_of_a_role_fails_validation(tmp_path):
    """Model ids live in the model registry; the rulebook names a role."""
    data = _shipped_data()
    data["extraction"]["model_role"] = "anthropic:claude-haiku-4-5-20251001"
    with pytest.raises(RulebookError, match="ROLE name"):
        load_rulebook(_write_rulebook(tmp_path, data))


def test_a_candidate_pool_smaller_than_the_cap_fails_validation(tmp_path):
    """Otherwise 'most authoritative first' silently becomes 'most recent'."""
    data = _shipped_data()
    data["reliefweb"]["documents"]["candidate_pool_size"] = 5
    data["reliefweb"]["documents"]["max_docs_per_cell"] = 30
    with pytest.raises(RulebookError, match="candidate_pool_size"):
        load_rulebook(_write_rulebook(tmp_path, data))


def test_a_duplicate_authority_tier_fails_validation(tmp_path):
    data = _shipped_data()
    data["reliefweb"]["authority_precedence"].append(
        {"tier": "government", "keywords": ["another"]}
    )
    with pytest.raises(RulebookError, match="duplicate tiers"):
        load_rulebook(_write_rulebook(tmp_path, data))


def test_a_non_positive_household_multiplier_fails_validation(tmp_path):
    data = _shipped_data()
    data["reliefweb"]["household_conversion"]["by_iso3"]["PHL"] = 0
    with pytest.raises(RulebookError, match="by_iso3.PHL"):
        load_rulebook(_write_rulebook(tmp_path, data))


def test_a_numeric_token_budget_is_config_not_a_credential():
    """The credential guard fires on secrets, not on settings named *_tokens.

    `extraction.max_output_tokens` is a token BUDGET. Rejecting it would
    push the next person to rename the setting to satisfy a check that was
    wrong, which costs the rulebook its readability.
    """
    data = _shipped_data()
    assert isinstance(data["extraction"]["max_output_tokens"], int)
    assert validate_rulebook(data) == []


def test_a_string_valued_credential_key_is_still_rejected(tmp_path):
    """The refinement must not open a hole: a real key is still refused."""
    data = _shipped_data()
    data["extraction"]["api_token"] = "sk-ant-not-a-real-key"
    with pytest.raises(RulebookError, match="looks like a credential"):
        load_rulebook(_write_rulebook(tmp_path, data))


def test_changing_the_authority_order_changes_the_preferred_figure(tmp_path):
    """Change the YAML, change the answer — no code edit."""
    from resolver.hazard_resolution.figures import authority_tier

    rb = load_rulebook()
    assert authority_tier("NDRRMC", rb)[0] < authority_tier("OCHA", rb)[0]

    data = _shipped_data()
    data["reliefweb"]["authority_precedence"].reverse()
    flipped = load_rulebook(_write_rulebook(tmp_path, data))
    assert authority_tier("NDRRMC", flipped)[0] > authority_tier("OCHA", flipped)[0]


def test_changing_the_household_multiplier_changes_the_conversion(tmp_path):
    from resolver.hazard_resolution.figures import household_multiplier

    data = _shipped_data()
    data["reliefweb"]["household_conversion"]["by_iso3"]["PHL"] = 9.0
    changed = load_rulebook(_write_rulebook(tmp_path, data))
    assert household_multiplier("PHL", changed) == (9.0, "by_iso3")
