# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The November 2026 figure rules, held to the cells that motivated them.

Every fixture in TestReviewCells uses the verbatim quote, the figure date
and the document dates the canonical DB held for the cell in the November
2026 resolver review (run 42999258374). Each must be rejected or kept with
the stated reason.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import pytest

from resolver.hazard_resolution import figures as figures_mod
from resolver.hazard_resolution.extract import ExtractedFigure
from resolver.hazard_resolution.rulebook import load_rulebook
from resolver.hazard_resolution.sources import parse_period


@pytest.fixture(scope="module")
def rulebook():
    return load_rulebook()


def _fig(**overrides) -> ExtractedFigure:
    base = dict(
        value=1000.0, unit="people", quote="1,000 people were affected",
        stated_by="", area="", date="", cumulative_or_new="unstated",
        doc_id="d1", doc_url="https://reliefweb.int/node/1", doc_title="Floods",
        doc_date="2026-09-20T00:00:00+00:00", doc_source_rank=0, model="test",
        doc_date_original="2026-09-20T00:00:00+00:00", doc_primary_country="",
        doc_country_iso3s=(), doc_publisher="", doc_disaster_types=("Flood",),
    )
    base.update(overrides)
    return ExtractedFigure(**base)


class _Extraction:
    def __init__(self, figures):
        self.figures = list(figures)
        self.rejected: list = []


def _run(figs, *, iso3, ym, hazard, rulebook):
    return figures_mod.build_candidates(
        _Extraction(figs), iso3=iso3, ym=ym, hazard=hazard, rulebook=rulebook,
        exposure_ceiling=None,
    )


def _reasons(rejected, quote_part):
    return sorted({r["reason"] for r in rejected if quote_part in (r.get("quote") or "")})


# ---------------------------------------------------------------------------
# The context table
# ---------------------------------------------------------------------------

PAK_DOC = dict(
    doc_id="4232445", doc_title="Addressing a new climate reality: Why we're working in Pakistan",
    doc_date="2026-10-02T16:03:06+00:00", doc_date_original="2026-10-02T00:00:00+00:00",
    doc_primary_country="PAK", doc_disaster_types=("Flood", "Heat Wave"),
)
CUB_DOC = dict(
    doc_id="4233281", doc_title="Cuba | Hurricane Melissa - Operation Update #5 (MDRCU013)",
    doc_date="2026-10-09T02:14:27+00:00", doc_date_original="2026-10-06T00:00:00+00:00",
    doc_primary_country="CUB", doc_disaster_types=("Flood", "Tropical Cyclone"),
)
THA_DOC = dict(
    doc_id="4232755", doc_title="THA: Flood - 09-2026 - Heavy rainfall caused flooding #3 (2026-10-06)",
    doc_date="2026-10-06T03:36:15+00:00", doc_date_original="2026-10-06T00:00:00+00:00",
    doc_primary_country="THA", doc_disaster_types=("Flood",),
)
THA_QUOTE = (
    "From 16 September to 6 October 2026, flooding was reported across 50 provinces, "
    "affecting 1,474,981 households (4,188,452 people), with 31 deaths reported."
)
LAO_DOC = dict(
    doc_id="4230629", doc_title="Lao PDR Monsoon Floods: Flash Update (09 to 14 September 2026)",
    doc_date="2026-09-21T04:30:11+00:00", doc_date_original="2026-09-20T00:00:00+00:00",
    doc_primary_country="LAO",
    doc_disaster_types=("Flash Flood", "Flood", "Land Slide", "Tropical Cyclone"),
)
LAO_QUOTE = (
    "According to the latest Government data, 403,133 people from 81,484 families have "
    "been affected across 16 provinces, including 203,857 women."
)
VNM_DOC = dict(
    doc_id="4236244", doc_title="Viet Nam Monsoon Flood 2025 - DREF Final Report (MDRVN025)",
    doc_date="2026-10-29T14:10:27+00:00", doc_date_original="2026-10-29T00:00:00+00:00",
    doc_primary_country="VNM", doc_disaster_types=("Flood", "Land Slide", "Tropical Cyclone"),
)
VNM_QUOTE = (
    "While the initial storms in July and August aected an estimated 100,000 people, "
    "mainly in Nghệ An, Điện Biên and Sơn La provinces"
)


class TestReviewCells:
    def test_pak_2022_floods_33_million_is_an_earlier_period(self, rulebook):
        quote = ("The 2022 floods were a catastrophe that submerged one-third of the country, "
                 "affecting 33 million people, claiming over 1,700 lives, ...")
        cands, rej = _run([_fig(value=33e6, quote=quote, **PAK_DOC)],
                          iso3="PAK", ym="2026-10", hazard="FL", rulebook=rulebook)
        assert cands == []
        assert _reasons(rej, "2022 floods") == ["quote_names_earlier_period"]

    def test_pak_figure_dated_2025_is_outside_september_2026(self, rulebook):
        quote = ("Fast forward to 2025, where floods from the Sutlej and Ravi rivers flooded 8 "
                 "districts in Punjab, affecting 3.9 million people and displacing 1.9 million.")
        cands, rej = _run([_fig(value=3.9e6, quote=quote, date="2025", **PAK_DOC)],
                          iso3="PAK", ym="2026-09", hazard="FL", rulebook=rulebook)
        assert cands == []
        assert _reasons(rej, "Fast forward") == ["quote_names_earlier_period"]

    def test_pak_partial_date_alone_is_dated_by_its_year(self, rulebook):
        """With no year in the quote, the figure's own "2025" still dates it."""

        cands, rej = _run([_fig(value=3.9e6, quote="3.9 million people were affected",
                                date="2025", **PAK_DOC)],
                          iso3="PAK", ym="2026-09", hazard="FL", rulebook=rulebook)
        assert cands == []
        assert rej[0]["reason"] == "outside_reporting_window"
        assert rej[0]["figure_about"] == "2025-01-01"

    def test_yem_september_2025(self, rulebook):
        quote = ("As of September 2025, an estimated 73,218 households approximately 512,526 "
                 "people were affected across 20 governorates of Yemen.")
        cands, rej = _run([_fig(value=512_526.0, quote=quote, date="2025-09",
                                doc_id="4228594", doc_primary_country="YEM",
                                doc_title="Yemen Flood 2025 | DREF Final Report (MDRYE016)",
                                doc_date="2026-09-07T01:27:43+00:00",
                                doc_date_original="2026-09-04T00:00:00+00:00")],
                          iso3="YEM", ym="2026-09", hazard="FL", rulebook=rulebook)
        assert cands == []
        assert _reasons(rej, "As of September 2025") == ["quote_names_earlier_period"]

    def test_som_2023_deyr_floods(self, rulebook):
        quote = ("Somalia's 2023 Deyr floods affected 2.5 million people and displaced 1.2 "
                 "million, with around 118 deaths reported.")
        cands, rej = _run([_fig(value=2.5e6, quote=quote, doc_id="4233308",
                                doc_primary_country="SOM",
                                doc_title="Somalia - El Niño flood risk in displacement sites (September 2026)",
                                doc_date="2026-10-09T06:20:33+00:00",
                                doc_date_original="2026-10-09T00:00:00+00:00")],
                          iso3="SOM", ym="2026-09", hazard="FL", rulebook=rulebook)
        assert cands == []
        assert _reasons(rej, "2023 Deyr") == ["quote_names_earlier_period"]

    def test_cub_citation_year_2025a(self, rulebook):
        quote = "Close to 2.2 million people were affected, with no fatalities reported (OCHA, 2025a)."
        for hazard in ("FL", "TC"):
            cands, rej = _run([_fig(value=2.2e6, quote=quote, stated_by="OCHA, 2025a", **CUB_DOC)],
                              iso3="CUB", ym="2026-09", hazard=hazard, rulebook=rulebook)
            assert cands == []
            assert _reasons(rej, "2025a") == ["quote_names_earlier_period"]

    def test_cub_645000_from_a_document_about_an_earlier_event(self, rulebook):
        dated = _fig(value=2.2e6, date="2025-10-29",
                     quote="Close to 2.2 million people were affected, with no fatalities reported (OCHA, 2025a).",
                     **CUB_DOC)
        undated = _fig(value=645_000.0,
                       quote=("More than 215,000 homes were damaged or destroyed, affecting an "
                              "estimated 645,000 people, half of whom are considered vulnerable."),
                       **CUB_DOC)
        cands, rej = _run([dated, undated], iso3="CUB", ym="2026-10", hazard="TC", rulebook=rulebook)
        assert cands == []
        assert _reasons(rej, "645,000") == ["document_about_earlier_event"]
        # In the flood cell it is caught twice over; the attribution rule runs first.
        cands, rej = _run([dated, undated], iso3="CUB", ym="2026-10", hazard="FL", rulebook=rulebook)
        assert cands == []
        assert _reasons(rej, "645,000") == ["document_about_earlier_event"]

    def test_vnm_july_and_august_in_an_october_cell(self, rulebook):
        for hazard in ("FL", "TC"):
            cands, rej = _run([_fig(value=100_000.0, quote=VNM_QUOTE, **VNM_DOC)],
                              iso3="VNM", ym="2026-10", hazard=hazard, rulebook=rulebook)
            assert cands == []
            assert _reasons(rej, "initial storms") == ["quote_names_earlier_period"] or \
                _reasons(rej, "initial storms") == ["outside_reporting_window"]

    def test_vnm_quote_period_is_july_to_august_of_the_document_year(self):
        start, end, evidence = figures_mod.quote_period(
            VNM_QUOTE, reference_date=dt.date(2026, 10, 29)
        )
        assert (start, end) == (dt.date(2026, 7, 1), dt.date(2026, 8, 31))
        assert evidence == ["July", "August"]

    def test_tha_people_reading_wins_september(self, rulebook):
        households = _fig(value=1_474_981.0, unit="households", quote=THA_QUOTE,
                          date="2026-10-06", **THA_DOC)
        people = _fig(value=4_188_452.0, unit="people", quote=THA_QUOTE,
                      date="2026-10-06", **THA_DOC)
        cands, rej = _run([households, people], iso3="THA", ym="2026-09", hazard="FL",
                          rulebook=rulebook)
        assert [c.value for c in cands] == [4_188_452.0]
        assert "household_superseded_by_stated_people" in {r["reason"] for r in rej}

    def test_tha_16_sep_to_6_oct_is_septembers_only(self, rulebook):
        people = _fig(value=4_188_452.0, quote=THA_QUOTE, date="2026-10-06", **THA_DOC)
        cands, rej = _run([people], iso3="THA", ym="2026-10", hazard="FL", rulebook=rulebook)
        assert cands == []
        assert rej[0]["reason"] == "outside_reporting_window"
        assert rej[0]["figure_about"] == "2026-09-16"
        assert rej[0]["figure_about_field"] == "figure_date+quote"

    def test_lao_403133_is_a_cyclone_figure(self, rulebook):
        fig = _fig(value=403_133.0, quote=LAO_QUOTE, date="2026-09-14",
                   stated_by="Government", **LAO_DOC)
        cands, rej = _run([fig], iso3="LAO", ym="2026-09", hazard="FL", rulebook=rulebook)
        assert cands == []
        assert rej[0]["reason"] == "cyclone_figure_in_flood_cell"
        cands, rej = _run([fig], iso3="LAO", ym="2026-09", hazard="TC", rulebook=rulebook)
        assert [c.value for c in cands] == [403_133.0]


# ---------------------------------------------------------------------------
# Dating
# ---------------------------------------------------------------------------

class TestParsePeriod:
    def test_year(self):
        assert parse_period("2025") == (dt.date(2025, 1, 1), dt.date(2025, 12, 31))

    def test_year_month(self):
        assert parse_period("2025-09") == (dt.date(2025, 9, 1), dt.date(2025, 9, 30))

    def test_full_date(self):
        assert parse_period("2025-09-14") == (dt.date(2025, 9, 14), dt.date(2025, 9, 14))

    @pytest.mark.parametrize("value", ["", None, "garbage", "2025-13", "99", "September"])
    def test_empty_and_garbage(self, value):
        assert parse_period(value) is None


class TestQuotePeriod:
    REF = dt.date(2026, 10, 29)

    def test_as_of_day_month_takes_the_reference_year(self):
        start, end, _ = figures_mod.quote_period("As of 6 October, 5,000 people", reference_date=self.REF)
        assert start == end == dt.date(2026, 10, 6)

    def test_a_range_with_one_year(self):
        start, end, ev = figures_mod.quote_period(
            "From 16 September to 6 October 2026, flooding", reference_date=None
        )
        assert (start, end) == (dt.date(2026, 9, 16), dt.date(2026, 10, 6))
        assert ev == ["16 September", "6 October 2026"]

    def test_a_citation_year(self):
        start, end, _ = figures_mod.quote_period("affected (OCHA, 2025a).", reference_date=self.REF)
        assert (start, end) == (dt.date(2025, 1, 1), dt.date(2025, 12, 31))

    def test_months_without_a_year(self):
        start, end, _ = figures_mod.quote_period("in July and August", reference_date=self.REF)
        assert (start, end) == (dt.date(2026, 7, 1), dt.date(2026, 8, 31))

    def test_a_month_after_the_document_belongs_to_the_year_before(self):
        start, _end, _ = figures_mod.quote_period(
            "In December, 4,000 people", reference_date=dt.date(2026, 3, 1)
        )
        assert start == dt.date(2025, 12, 1)

    @pytest.mark.parametrize("quote,expected", [
        ("le 12 octobre 2026, 4 000 personnes", dt.date(2026, 10, 12)),
        ("du 3 au 5 mars 2026", dt.date(2026, 3, 3)),
        ("el 12 de septiembre de 2026", dt.date(2026, 9, 12)),
        ("en agosto de 2026", dt.date(2026, 8, 1)),
    ])
    def test_french_and_spanish(self, quote, expected):
        start, _end, _ = figures_mod.quote_period(quote, reference_date=self.REF)
        assert start == expected

    @pytest.mark.parametrize("quote", [
        "1,000 people were affected",
        "The affected people may need shelter",
        "2025 people were affected",
    ])
    def test_no_time_named(self, quote):
        assert figures_mod.quote_period(quote, reference_date=self.REF) is None

    def test_since_compares_and_does_not_date(self):
        assert figures_mod.names_only_earlier_years("the worst since 1998, affecting 50,000 people", 2026) == []


# ---------------------------------------------------------------------------
# Households
# ---------------------------------------------------------------------------

class TestHouseholds:
    def test_both_readings_keep_the_people_one(self, rulebook):
        q = "329,000 households / 700,000 people were affected"
        cands, rej = _run(
            [_fig(value=329_000.0, unit="households", quote=q, doc_primary_country="BGD"),
             _fig(value=700_000.0, unit="people", quote=q, doc_primary_country="BGD")],
            iso3="BGD", ym="2026-09", hazard="FL", rulebook=rulebook,
        )
        assert [c.value for c in cands] == [700_000.0]
        assert [r["reason"] for r in rej] == ["household_superseded_by_stated_people"]

    def test_households_alone_beside_a_stated_people_count_is_rejected(self, rulebook):
        q = "73,218 households approximately 512,526 people were affected"
        cands, rej = _run([_fig(value=73_218.0, unit="households", quote=q)],
                          iso3="BGD", ym="2026-09", hazard="FL", rulebook=rulebook)
        assert cands == []
        assert rej[0]["reason"] == "household_with_stated_people"
        assert rej[0]["stated_people"] == [512_526.0]

    def test_a_smaller_people_count_is_not_the_same_group(self, rulebook):
        q = "5,000 households were affected, including 300 people with disabilities"
        cands, _rej = _run([_fig(value=5_000.0, unit="households", quote=q)],
                           iso3="BGD", ym="2026-09", hazard="FL", rulebook=rulebook)
        assert [c.value for c in cands] == [5_000.0 * 4.1]


# ---------------------------------------------------------------------------
# One hazard
# ---------------------------------------------------------------------------

class TestOneHazard:
    def test_a_tc_tagged_document_is_rejected_in_flood_and_kept_in_cyclone(self, rulebook):
        fig = _fig(doc_disaster_types=("Flood", "Tropical Cyclone"), doc_primary_country="PHL")
        assert _run([fig], iso3="PHL", ym="2026-09", hazard="FL", rulebook=rulebook)[0] == []
        assert len(_run([fig], iso3="PHL", ym="2026-09", hazard="TC", rulebook=rulebook)[0]) == 1

    def test_a_flood_only_document_is_unaffected(self, rulebook):
        fig = _fig(doc_disaster_types=("Flood",), doc_primary_country="PHL")
        assert len(_run([fig], iso3="PHL", ym="2026-09", hazard="FL", rulebook=rulebook)[0]) == 1

    def test_a_named_storm_in_the_title_without_disaster_types(self, rulebook):
        fig = _fig(doc_disaster_types=(), doc_title="Typhoon Haikui: Flash Update",
                   doc_primary_country="PHL")
        cands, rej = _run([fig], iso3="PHL", ym="2026-09", hazard="FL", rulebook=rulebook)
        assert cands == [] and rej[0]["reason"] == "cyclone_figure_in_flood_cell"

    def test_initial_storms_is_not_a_cyclone(self, rulebook):
        fig = _fig(doc_disaster_types=("Flood",), quote="the initial storms affected 9,000 people",
                   doc_primary_country="PHL")
        assert len(_run([fig], iso3="PHL", ym="2026-09", hazard="FL", rulebook=rulebook)[0]) == 1

    def test_named_flooding_with_a_separate_cyclone_figure_stays_flood(self, rulebook):
        types = ("Flood", "Tropical Cyclone")
        flood = _fig(value=12_000.0, quote="Floods affected 12,000 people in the north",
                     doc_disaster_types=types, doc_primary_country="PHL")
        storm = _fig(value=90_000.0, quote="The typhoon affected 90,000 people",
                     doc_disaster_types=types, doc_primary_country="PHL")
        cands, rej = _run([flood, storm], iso3="PHL", ym="2026-09", hazard="FL", rulebook=rulebook)
        assert [c.value for c in cands] == [12_000.0]
        assert _reasons(rej, "typhoon") == ["cyclone_figure_in_flood_cell"]


class TestRegression:
    def test_a_dated_figure_in_window_with_an_undated_quote_is_accepted(self, rulebook):
        fig = _fig(value=4_000.0, date="2026-09-12", quote="4,000 people were affected",
                   doc_date="2026-09-15T00:00:00+00:00", doc_date_original="2026-09-14T00:00:00+00:00",
                   doc_primary_country="KEN")
        cands, rej = _run([fig], iso3="KEN", ym="2026-09", hazard="FL", rulebook=rulebook)
        assert [c.value for c in cands] == [4_000.0]
        assert rej == []

    def test_switches_off_restore_the_old_rules(self, rulebook):
        from resolver.hazard_resolution.quote_period_audit import rulebook_with_switches

        old = rulebook_with_switches(rulebook, False)
        quote = "Somalia's 2023 Deyr floods affected 2.5 million people"
        cands, _ = _run([_fig(value=2.5e6, quote=quote, doc_primary_country="SOM")],
                        iso3="SOM", ym="2026-09", hazard="FL", rulebook=old)
        assert [c.value for c in cands] == [2.5e6]


# ---------------------------------------------------------------------------
# Audit and retract, on a small DB
# ---------------------------------------------------------------------------

def _seed(con, rulebook, *, iso3, ym, hazard, figs, doc, stored_value=None, frozen=None):
    from resolver.hazard_resolution import extract as ex
    from resolver.hazard_resolution import reliefweb_docs as docs
    from resolver.hazard_resolution.schema import ensure_haz_schema
    from resolver.hazard_resolution.sources import RawRecord, store_raw_records

    ensure_haz_schema(con)
    store_raw_records(con, docs.SOURCE, [RawRecord(
        record_id=f"rw-{doc['doc_id']}", payload=doc, iso3=iso3, ym=ym, hazard=hazard,
    )])
    ex.write_extraction(
        con, document=doc, iso3=iso3, ym=ym, hazard=hazard,
        model=ex.resolve_extraction_model(rulebook),
        prompt_version=str(rulebook.get("extraction.prompt_version")),
        status="ok", figures=figs, rejected=[], usage={}, cost_usd=0.0, error=None,
    )
    if stored_value is not None:
        ref = figures_mod._source_ref(figs[0])
        prov = {
            "source_record_ids": [ref], "winning_rung": "reliefweb_extracted",
            "decision": {"candidates": [{"source_ref": ref, "detail": {
                "quote": figs[0].quote, "doc_id": figs[0].doc_id, "unit_as_stated": figs[0].unit}}]},
        }
        year, month = (int(p) for p in ym.split("-"))
        con.execute(
            "INSERT INTO haz_resolutions (iso3, year, month, hazard, status, value, "
            "provenance_json, rule_fired, provisional, frozen_at) VALUES "
            "(?, ?, ?, ?, 'RESOLVED_VALUE', ?, ?, 'ladder:reliefweb_extracted', ?, ?)",
            [iso3, year, month, hazard, stored_value, json.dumps(prov), frozen is None,
             frozen or dt.datetime(2026, 11, 29)],
        )


def _doc(fig: ExtractedFigure) -> dict:
    return {
        "doc_id": fig.doc_id, "title": fig.doc_title, "url": fig.doc_url, "body": fig.quote,
        "date_created": fig.doc_date, "date_original": fig.doc_date_original,
        "primary_country_iso3": fig.doc_primary_country, "country_iso3s": [fig.doc_primary_country],
        "sources": [], "disaster_types": list(fig.doc_disaster_types), "source_rank": 0,
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_audit_cli_is_read_only(tmp_path, rulebook):
    import duckdb

    from resolver.hazard_resolution import quote_period_audit as audit

    db = tmp_path / "a.duckdb"
    con = duckdb.connect(str(db))
    pak = _fig(value=33e6, quote="The 2022 floods affected 33 million people", **PAK_DOC)
    _seed(con, rulebook, iso3="PAK", ym="2026-10", hazard="FL", figs=[pak], doc=_doc(pak),
          stored_value=33e6)
    con.execute("CHECKPOINT")
    con.close()
    before = _sha(db)
    out_json = tmp_path / "audit.json"
    rc = audit.main(["--db", str(db), "--months", "2026-10", "--json-out", str(out_json),
                     "--md-out", str(tmp_path / "audit.md"), "--today", "2026-11-24"])
    assert rc == 0
    assert _sha(db) == before
    report = json.loads(out_json.read_text())
    (cell,) = [c for c in report["cells"] if c["iso3"] == "PAK"]
    assert cell["stored_value"] == 33e6 and cell["old_rules_value"] == 33e6
    assert cell["stored_winner_fails_because"] == "quote_names_earlier_period"
    assert cell["changed"] is True
    assert report["model_calls"] == 0


def test_the_retract_selects_a_frozen_question_cell_and_applies_once(tmp_path, rulebook):
    import duckdb

    from resolver.hazard_resolution import extraction_rule_retract as rr
    from pythia.db.schema import ensure_schema

    db = tmp_path / "r.duckdb"
    con = duckdb.connect(str(db))
    ensure_schema(con)
    som = _fig(value=1.2e6, doc_id="9", doc_primary_country="SOM",
               quote="this year's El Niño could rival the floods of 2023, which displaced 1.2 million people",
               doc_date="2026-08-20T00:00:00+00:00", doc_date_original="2026-08-20T00:00:00+00:00")
    _seed(con, rulebook, iso3="SOM", ym="2026-08", hazard="FL", figs=[som], doc=_doc(som),
          stored_value=1.2e6, frozen=dt.datetime(2026, 10, 30))
    con.execute(
        "INSERT INTO questions (question_id, iso3, hazard_code, metric, window_start_date, "
        "target_month, wording, status) VALUES "
        "('SOM_FL_PA_2026-08', 'SOM', 'FL', 'PA', DATE '2026-08-01', '2027-01', 'w', 'active')"
    )
    rows = rr.select_rows(con, rulebook, today=dt.date(2026, 11, 24))
    assert [(r["iso3"], r["ym"], r["reason"]) for r in rows] == [
        ("SOM", "2026-08", "quote_names_earlier_period")
    ]
    rr.stage_selection(con, rows)
    before = rr.integrity_report(con)
    result = rr.apply_retraction(con)
    assert result["applied"] is True
    after = rr.integrity_report(con)
    assert rr.compare_reports(before, after, result["deleted"]) == []
    assert con.execute("SELECT COUNT(*) FROM haz_resolutions").fetchone()[0] == 0
    assert rr.apply_retraction(con)["applied"] is False  # one-shot token
    con.close()


def test_a_cell_no_question_resolves_from_is_not_selected(tmp_path, rulebook):
    import duckdb

    from resolver.hazard_resolution import extraction_rule_retract as rr
    from pythia.db.schema import ensure_schema

    con = duckdb.connect(str(tmp_path / "n.duckdb"))
    ensure_schema(con)
    som = _fig(value=1.2e6, doc_id="9", doc_primary_country="SOM",
               quote="the floods of 2023 displaced 1.2 million people",
               doc_date="2026-08-20T00:00:00+00:00", doc_date_original="2026-08-20T00:00:00+00:00")
    _seed(con, rulebook, iso3="SOM", ym="2026-08", hazard="FL", figs=[som], doc=_doc(som),
          stored_value=1.2e6, frozen=dt.datetime(2026, 10, 30))
    assert rr.select_rows(con, rulebook, today=dt.date(2026, 11, 24)) == []
    con.close()


def test_the_drought_fingerprint_does_not_read_the_figure_switches(rulebook):
    from resolver.hazard_resolution.quote_period_audit import rulebook_with_switches

    off = rulebook_with_switches(rulebook, False)
    assert off.hazard_fingerprint("drought") == rulebook.hazard_fingerprint("drought")
    assert off.hazard_fingerprint("flood") != rulebook.hazard_fingerprint("flood")
    assert off.hazard_fingerprint("cyclone") != rulebook.hazard_fingerprint("cyclone")
