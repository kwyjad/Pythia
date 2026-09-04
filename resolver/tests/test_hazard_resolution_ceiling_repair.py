# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The GDACS exposure ceiling, document attribution and the honest ledger.

Run 33841370196 (Group C). Across 310 ledger rows there were exactly three
distinct ceilings — 5, 20 and 955 — and 72 well-sourced figures were
rejected against them; two Spanish- and English-language reports about
March-to-May floods were filed under AFG / FL / 2026-07; and "187 families
affected" was written as value 935 under unit "households". Each test here
pins the repair by what lands in the table or the ledger.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from resolver.connectors.gdacs import parse_gdacs_population
from resolver.hazard_resolution import figures as figures_mod
from resolver.hazard_resolution.extract import ExtractedFigure
from resolver.hazard_resolution.rulebook import load_rulebook
from resolver.hazard_resolution.rules import usable_exposure, within_sanity_ceiling


@pytest.fixture()
def rulebook():
    return load_rulebook()


# ---------------------------------------------------------------------------
# C1: what GDACS actually publishes, and the plausibility floor
# ---------------------------------------------------------------------------

class TestPopulationUnit:
    def test_people_units_pass_the_value_through(self):
        for unit in ("people", "Pop74", "Population in 100km", ""):
            value, detail = parse_gdacs_population("1300", unit, "1300 people in the flooded area")
            assert value == 1300.0, unit
            assert detail["raw_unit"] == unit
            assert detail["text"].startswith("1300 people")

    def test_a_multiplicative_unit_is_scaled_never_taken_bare(self):
        assert parse_gdacs_population("1.67", "Million")[0] == pytest.approx(1_670_000.0)
        assert parse_gdacs_population("6.67", "thousand")[0] == pytest.approx(6_670.0)
        _v, detail = parse_gdacs_population("1.67", "Million")
        assert detail["outcome"] == "scaled" and detail["multiplier"] == 1_000_000.0

    def test_an_unrecognised_unit_is_unknown_and_logged(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            value, detail = parse_gdacs_population("5", "furlongs")
        assert value is None
        assert detail["outcome"] == "unrecognised_unit"
        assert any("furlongs" in r.message and "UNKNOWN" in r.message for r in caplog.records)

    def test_no_value_is_no_value(self):
        assert parse_gdacs_population(None, "people")[0] is None
        assert parse_gdacs_population("", "people")[1]["outcome"] == "no_value"

    def test_the_connector_keeps_unit_and_text_on_the_event(self):
        from resolver.connectors.gdacs import GdacsConnector

        rss = b"""<?xml version="1.0"?>
<rss xmlns:gdacs="http://www.gdacs.org" version="2.0"><channel><item>
  <gdacs:eventtype>FL</gdacs:eventtype><gdacs:eventid>1102000</gdacs:eventid>
  <gdacs:country>Afghanistan</gdacs:country><gdacs:iso3>AFG</gdacs:iso3>
  <gdacs:fromdate>Mon, 01 Jul 2026 00:00:00 GMT</gdacs:fromdate>
  <gdacs:todate>Thu, 04 Jul 2026 00:00:00 GMT</gdacs:todate>
  <gdacs:alertlevel>Orange</gdacs:alertlevel>
  <gdacs:population value="6.67" unit="Million">6.67 Million people in the flooded area</gdacs:population>
</item></channel></rss>"""
        events = GdacsConnector()._parse_rss(rss, {"afghanistan": "AFG"})
        assert len(events) == 1
        ev = events[0]
        assert ev["population"] == pytest.approx(6_670_000.0)
        assert ev["population_unit"] == "Million"
        assert ev["population_raw"] == "6.67"
        assert ev["population_parse"] == "scaled"
        assert "6.67 Million" in ev["population_text"]


class TestPlausibilityFloor:
    def test_the_three_september_ceilings_are_not_bounds(self, rulebook):
        """Exposures of 1.67, 6.67 and 318 people: parse failures, not ceilings."""

        for exposure in (1.67, 6.67, 318.0):
            assert usable_exposure(exposure, rulebook) is None
            # "306 houses were affected, impacting 1,917 people" against a
            # ceiling of 20 — the figure from the September ledger.
            assert within_sanity_ceiling(1_917, exposure, rulebook) is True

    def test_a_real_exposure_still_binds(self, rulebook):
        assert usable_exposure(50_000.0, rulebook) == 50_000.0
        assert within_sanity_ceiling(1_000_000, 50_000.0, rulebook) is False
        assert within_sanity_ceiling(120_000, 50_000.0, rulebook) is True

    def test_the_floor_is_read_from_the_rulebook(self, rulebook):
        assert float(rulebook.get("sanity.min_plausible_exposure")) == 1000.0
        assert usable_exposure(999.0, rulebook) is None
        assert usable_exposure(1000.0, rulebook) == 1000.0

    def test_reconcile_applies_the_floor_to_the_effective_ceiling(self, rulebook):
        from resolver.hazard_resolution.candidates import Candidate, VALUE_CEILING
        from resolver.hazard_resolution.reconcile import effective_ceiling

        tiny = Candidate(
            iso3="AFG", ym="2026-07", hazard="FL", value=6.67,
            value_type=VALUE_CEILING, source="gdacs", source_ref="FL-1",
        )
        ceiling, basis = effective_ceiling([tiny], rulebook, national_population=40_000_000)
        assert basis == "population_share"      # GDACS said nothing usable
        assert ceiling == pytest.approx(20_000_000.0)

    def test_the_ceiling_basis_counts_events_below_the_floor(self, rulebook, monkeypatch):
        from resolver.hazard_resolution import candidates as cand_mod

        events = [
            {"exposed_population": 6.67, "event_id": "1", "exposed_population_unit": "people"},
            {"exposed_population": 250_000.0, "event_id": "2", "exposed_population_unit": "people"},
        ]
        monkeypatch.setattr(cand_mod.gdacs_mod, "events_for_country_month", lambda *a, **k: events)
        basis = cand_mod.exposure_ceiling_basis(None, "AFG", "2026-07", "FL", rulebook)
        assert basis["value"] == 250_000.0
        assert basis["n_events_below_plausible_floor"] == 1
        assert basis["exposure_units_seen"] == ["people"]


# ---------------------------------------------------------------------------
# C2: a document supplies a figure only to the cell it is about
# ---------------------------------------------------------------------------

def _figure(**overrides) -> ExtractedFigure:
    base = dict(
        value=82_000.0, unit="people", quote="82,000 people were affected",
        stated_by="OCHA", area="", date="", cumulative_or_new="unstated",
        doc_id="rw-4222929", doc_url="https://reliefweb.int/x", doc_title="Floods",
        doc_date="2026-07-20T00:00:00+00:00", doc_source_rank=0, model="test",
        doc_date_original="", doc_primary_country="", doc_country_iso3s=(),
    )
    base.update(overrides)
    return ExtractedFigure(**base)


class TestAttribution:
    def test_a_document_about_another_country_supplies_nothing(self, rulebook):
        pairs = [(_figure(doc_primary_country="COL", doc_country_iso3s=("COL", "AFG")), 82_000.0)]
        kept, rejected = figures_mod.apply_attribution(pairs, "AFG", "2026-07", rulebook)
        assert kept == []
        assert rejected[0]["reason"] == "document_primary_country_mismatch"
        assert rejected[0]["doc_primary_country"] == "COL"

    def test_a_figure_about_a_date_outside_the_window_is_rejected(self, rulebook):
        march = _figure(date="2026-04-15", doc_primary_country="AFG")
        kept, rejected = figures_mod.apply_attribution([(march, 82_000.0)], "AFG", "2026-07", rulebook)
        assert kept == []
        assert rejected[0]["reason"] == "outside_reporting_window"
        assert rejected[0]["figure_about"] == "2026-04-15"
        assert rejected[0]["figure_about_field"] == "figure_date"

    def test_the_documents_original_date_stands_in_for_an_undated_figure(self, rulebook):
        stale = _figure(doc_primary_country="AFG", doc_date_original="2026-05-02T00:00:00+00:00")
        kept, rejected = figures_mod.apply_attribution([(stale, 82_000.0)], "AFG", "2026-07", rulebook)
        assert kept == [] and rejected[0]["figure_about_field"] == "doc_date_original"

    def test_a_figure_inside_the_window_and_about_the_country_is_kept(self, rulebook):
        good = _figure(date="2026-07-03", doc_primary_country="AFG")
        padded = _figure(date="", doc_primary_country="AFG", doc_date_original="2026-08-10T00:00:00+00:00")
        kept, rejected = figures_mod.apply_attribution(
            [(good, 82_000.0), (padded, 1_917.0)], "AFG", "2026-07", rulebook,
        )
        assert [v for _f, v in kept] == [82_000.0, 1_917.0]
        assert rejected == []

    def test_a_pre_september_cached_document_with_no_evidence_is_kept(self, rulebook):
        legacy = _figure(doc_date="", doc_primary_country="", doc_date_original="")
        kept, rejected = figures_mod.apply_attribution([(legacy, 1.0)], "AFG", "2026-07", rulebook)
        assert len(kept) == 1 and rejected == []

    def test_the_reporting_window_is_month_plus_the_publication_pad(self, rulebook):
        start, end = figures_mod.reporting_window("2026-07", rulebook)
        pad = int(rulebook.get("reliefweb.documents.publication_pad_days"))
        assert start == dt.date(2026, 7, 1)
        assert end == dt.date(2026, 7, 31) + dt.timedelta(days=pad)


# ---------------------------------------------------------------------------
# C3: the ledger says what the source said AND what the ladder uses
# ---------------------------------------------------------------------------

class TestHonestLedger:
    def test_households_become_whole_persons_with_the_factor_recorded(self, rulebook, monkeypatch, tmp_path):
        from resolver.diagnostics import run_log

        monkeypatch.setenv("PYTHIA_RUN_LOG_DIR", str(tmp_path))
        run_log.reset_for_tests()

        class _Extraction:
            figures = [
                _figure(value=15_600.0, unit="households",
                        quote="15,600 households (75,000 people)",
                        doc_primary_country="BGD", date="2026-07-05"),
            ]
            rejected: list = []

        candidates, rejected = figures_mod.build_candidates(
            _Extraction(), iso3="BGD", ym="2026-07", hazard="FL", rulebook=rulebook,
            exposure_ceiling=None,
        )
        assert rejected == []
        assert candidates[0].value == 63_960.0          # 15,600 x 4.1, whole persons
        assert candidates[0].value == int(candidates[0].value)

        rows = list(run_log.read_stream(tmp_path / f"{run_log.STREAM_FIGURES}.jsonl"))
        (row,) = rows
        assert row["stated_value"] == 15_600.0
        assert row["stated_unit"] == "households"
        assert row["value_persons"] == 63_960
        assert row["conversion_factor"] == 4.1
        assert row["figure_date"] == "2026-07-05"
        assert row["doc_primary_country"] == "BGD"

    def test_a_people_figure_has_factor_one_and_matches_itself(self, rulebook, monkeypatch, tmp_path):
        from resolver.diagnostics import run_log

        monkeypatch.setenv("PYTHIA_RUN_LOG_DIR", str(tmp_path))
        run_log.reset_for_tests()

        class _Extraction:
            figures = [_figure(value=1_917.0, doc_primary_country="AFG", date="2026-07-02")]
            rejected: list = []

        figures_mod.build_candidates(
            _Extraction(), iso3="AFG", ym="2026-07", hazard="FL", rulebook=rulebook,
            exposure_ceiling=None,
        )
        (row,) = list(run_log.read_stream(tmp_path / f"{run_log.STREAM_FIGURES}.jsonl"))
        assert row["stated_value"] == 1_917.0 and row["value_persons"] == 1_917
        assert row["conversion_factor"] == 1.0


# ---------------------------------------------------------------------------
# C1.4: the corrected ceiling reaches the rows it would have changed
# ---------------------------------------------------------------------------

def test_cells_with_ceiling_rejections_are_found_from_their_own_provenance(tmp_path):
    duckdb = pytest.importorskip("duckdb")
    from resolver.hazard_resolution import impact as impact_mod
    from resolver.hazard_resolution.schema import ensure_haz_schema

    con = duckdb.connect(str(tmp_path / "haz.duckdb"))
    ensure_haz_schema(con)
    provenance = json.dumps({"reliefweb_extraction": {"extraction": {"rejected": [
        {"reason": impact_mod.CEILING_REJECTION_REASON, "value": 1917.0}]}}})
    con.execute(
        "INSERT INTO haz_resolutions (iso3, year, month, hazard, status, value, provenance_json, rule_fired, flagged) "
        "VALUES ('AFG', 2026, 7, 'FL', 'NO_DATA', NULL, ?, 'x', TRUE)",
        [provenance],
    )
    con.execute(
        "INSERT INTO haz_resolutions (iso3, year, month, hazard, status, value, provenance_json, rule_fired, flagged) "
        "VALUES ('PHL', 2026, 7, 'FL', 'RESOLVED_VALUE', 100.0, '{}', 'y', FALSE)",
    )
    assert impact_mod.cells_with_ceiling_rejections(con, "FL") == {"2026-07": ["AFG"]}
    assert impact_mod.cells_with_ceiling_rejections(con, "TC") == {}


def test_reconsideration_replays_the_ladder_for_exactly_those_cells(tmp_path, monkeypatch, rulebook):
    duckdb = pytest.importorskip("duckdb")
    from resolver.hazard_resolution import impact as impact_mod
    from resolver.hazard_resolution.schema import ensure_haz_schema

    con = duckdb.connect(str(tmp_path / "haz.duckdb"))
    ensure_haz_schema(con)
    con.execute(
        "INSERT INTO haz_resolutions (iso3, year, month, hazard, status, value, provenance_json, rule_fired, flagged) "
        "VALUES ('AFG', 2026, 7, 'FL', 'NO_DATA', NULL, ?, 'x', TRUE)",
        [json.dumps({"rejected": [{"reason": impact_mod.CEILING_REJECTION_REASON}]})],
    )
    walked: list[tuple[str, list[str], bool]] = []

    def fake_resolve(con, *, ym, hazard, iso3s, fetch_documents, extract, **kwargs):
        walked.append((ym, iso3s, fetch_documents))
        run = impact_mod.LadderRun(hazard=hazard, ym=ym)
        run.cells = len(iso3s)
        run.resolved_value = 1
        return run

    monkeypatch.setattr(impact_mod, "resolve_triggered_cells", fake_resolve)
    monkeypatch.setattr(impact_mod, "fetch_ladder_sources", lambda *a, **k: {})
    summary = impact_mod.reconsider_rejected_cells(con, hazard="FL", rulebook=rulebook)
    assert walked == [("2026-07", ["AFG"], False)]    # no document fetch; cache only
    assert summary["cells_reconsidered"] == 1 and summary["cells_rewritten"] == 1
    assert summary["months"] == ["2026-07"]
