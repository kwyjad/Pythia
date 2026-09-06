# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Group G of the run-33946954189 repairs: extraction quality and denominators.

From ``figures_ledger.csv``, grouped by ``(stated_unit, unit,
conversion_factor)``:

    122 rows  households -> households, factor 5.0
     74 rows  households -> households, factor EMPTY
    229 rows  people     -> people,     factor EMPTY

A household count with no factor beside it is either being multiplied by an
unrecorded number or read as persons, and a factor of five is the difference
between 2,000 people and 10,000. Beside that: 71 of 1,324 quotes were
rejected as unverifiable, 558 of 950 accepted figures carried
``(unattributed)``, and 36 of the 252 countries had no population
denominator at all — so a cell with no GDACS exposure had no upper bound.

Network-free.
"""

from __future__ import annotations

import pytest

from resolver.hazard_resolution import extract as extract_mod
from resolver.hazard_resolution import figures as figures_mod
from resolver.tests.hazard_resolution_utils import make_rulebook


class TestEveryRowCarriesAConversionFactor:
    def test_a_household_unit_carries_the_country_factor(self):
        rulebook = make_rulebook()
        factor, origin = figures_mod.conversion_for_unit("households", "KEN", rulebook)

        assert factor > 1.0
        assert origin in ("by_iso3", "default")

    def test_a_person_unit_carries_an_explicit_one(self):
        factor, origin = figures_mod.conversion_for_unit("people", "KEN", make_rulebook())

        assert factor == 1.0
        assert origin == "identity", (
            "blank must mean unresolved, never trivially one"
        )

    def test_an_absent_unit_is_not_treated_as_households(self):
        factor, _origin = figures_mod.conversion_for_unit(None, "KEN", make_rulebook())

        assert factor == 1.0


class TestQuoteNormalisation:
    """G2 — a PDF artefact is not a fabrication."""

    def _found(self, quote: str, body: str) -> bool:
        return extract_mod._normalise_for_match(quote) in extract_mod._normalise_for_match(body)

    def test_a_word_broken_across_a_line_still_matches(self):
        assert self._found(
            "82,000 people were affected",
            "the floods: 82,000 people were affec-\nted across three provinces",
        )

    def test_a_soft_hyphen_still_matches(self):
        assert self._found("people affected", "people af­fected by the floods")

    def test_a_ligature_still_matches(self):
        assert self._found("families affected", "families aﬀected by the floods")

    def test_a_narrow_no_break_space_still_matches(self):
        assert self._found("12 000 people", "some 12 000 people were displaced")

    def test_a_quote_the_document_does_not_contain_still_fails(self):
        assert not self._found(
            "400,000 people were affected",
            "the floods affected 82,000 people across three provinces",
        )

    def test_a_reordered_quote_still_fails(self):
        assert not self._found(
            "affected people 82,000",
            "82,000 people affected",
        ), "this is a containment check, not a fuzzy match"

    def test_the_strict_comparison_is_kept_for_counting(self):
        # The pre-Sept-2026 comparison, so a run can say how many quotes the
        # wider normalisation recovered rather than leaving it to be guessed.
        assert extract_mod.strict_normalise("  A  B ") == "a b"
        assert extract_mod.strict_normalise("af­fected") != "affected"


class TestDocumentAttribution:
    """G3 — an unnamed figure in an OCHA report is OCHA's."""

    def _figure(self, **kwargs):
        base = dict(
            value=100.0, unit="people", quote="q", stated_by="", area="",
            date="", cumulative_or_new="unstated", doc_id="rw-1",
            doc_url="", doc_title="", doc_date="", doc_source_rank=0,
            model="haiku",
        )
        base.update(kwargs)
        return extract_mod.ExtractedFigure(**base)

    def test_the_text_attribution_wins(self):
        figure = self._figure(stated_by="UNHCR", doc_publisher="OCHA")

        assert figures_mod._attribute(figure) == ("UNHCR", "text")

    def test_the_document_attributes_a_figure_the_text_does_not(self):
        figure = self._figure(stated_by="", doc_publisher="OCHA")

        assert figures_mod._attribute(figure) == ("OCHA", "document")

    def test_neither_leaves_it_unattributed(self):
        figure = self._figure(stated_by="", doc_publisher="")

        assert figures_mod._attribute(figure) == (figures_mod.UNATTRIBUTED, "none")

    def test_the_inference_is_labelled_not_passed_off_as_a_transcription(self):
        text = figures_mod._attribute(self._figure(stated_by="UNHCR"))
        document = figures_mod._attribute(self._figure(doc_publisher="OCHA"))

        assert text[1] != document[1], (
            "a reader must be able to tell 'the report said UNHCR' from "
            "'the report is an OCHA report and said nothing'"
        )


class TestPopulationDenominators:
    """G4 — a country with no denominator has no upper bound at all."""

    def test_every_country_in_the_list_has_a_denominator_or_is_uninhabited(self):
        import csv
        from pathlib import Path

        pop = set()
        with open(
            Path("resolver/data/population.csv"), newline="", encoding="utf-8-sig"
        ) as handle:
            for row in csv.DictReader(handle):
                pop.add((row["iso3"] or "").strip().upper())
        countries = set()
        with open(
            Path("resolver/data/countries.csv"), newline="", encoding="utf-8-sig"
        ) as handle:
            for row in csv.DictReader(handle):
                code = (row.get("iso3") or "").strip().upper()
                if code:
                    countries.add(code)

        # Antarctica, the French Southern Territories, Bouvet, Heard &
        # McDonald, South Georgia and the US Minor Outlying Islands have no
        # resident population; ANT (Netherlands Antilles) was dissolved in
        # 2010. Inventing a figure for any of them would be worse than the
        # hole, and no hazard cell resolves for them.
        uninhabited = {"ANT", "ATA", "ATF", "BVT", "HMD", "SGS", "UMI"}
        missing = countries - pop - uninhabited

        assert missing == set(), (
            f"{len(missing)} inhabited countries have no population "
            f"denominator, so a cell of theirs with no GDACS exposure "
            f"reconciles with no upper bound: {sorted(missing)}"
        )

    def test_the_added_territories_parse(self):
        from resolver.hazard_resolution.population import load_population_records

        records = load_population_records()
        by_iso3 = dict(zip(records["iso3"], records["population"]))

        assert by_iso3["TWN"] > 20_000_000
        assert by_iso3["REU"] > 500_000
        assert by_iso3["BES"] > 10_000  # the name carries a comma
        assert by_iso3["PCN"] > 0


class TestDfoArchiveAddresses:
    """G5 — a source address is a candidate list."""

    def test_the_rulebook_offers_more_than_one_route(self):
        from resolver.hazard_resolution.dfo import _archive_urls
        from resolver.hazard_resolution.rulebook import load_rulebook

        urls = _archive_urls(load_rulebook())

        assert len(urls) >= 2
        assert len(set(urls)) == len(urls)

    def test_a_moved_route_falls_through_to_the_next(self, tmp_path):
        import duckdb

        from resolver.hazard_resolution import dfo as dfo_mod
        from resolver.hazard_resolution.schema import ensure_haz_schema

        con = duckdb.connect(str(tmp_path / "haz.duckdb"))
        ensure_haz_schema(con)
        tried: list[str] = []

        def _get(url, _timeout):
            tried.append(url)
            raise RuntimeError("404")

        outcome = dfo_mod.fetch_dfo(con, make_rulebook(), get=_get)

        assert not outcome.ok
        assert len(tried) >= 2, "every candidate route is tried"
        assert all(url in outcome.error for url in tried), (
            "when every route fails the error names each attempt: "
            "'unreachable' and 'moved' want different repairs"
        )
        con.close()


class TestConflictForecastPeriodOrdering:
    """The latent bug: a dedup that worked only because the file arrives sorted.

    ``drop_duplicates(subset=[iso3], keep="last")`` with no sort kept 182 of
    36,200 rows. The file spans periods from 201001, so a provider that ever
    reordered it would silently store 2010 values as the current forecast
    and nothing in the run would notice.
    """

    def _frame(self, rows):
        pd = pytest.importorskip("pandas")
        return pd.DataFrame(rows)

    def test_the_latest_period_wins_whatever_the_arrival_order(self):
        from resolver.connectors.conflictforecast import ConflictForecastOrgConnector

        df = self._frame([
            {"iso3": "KEN", "period": 202608, "ons_armedconf_03_all": 0.9},
            {"iso3": "KEN", "period": 201001, "ons_armedconf_03_all": 0.1},
        ])

        rows = ConflictForecastOrgConnector._transform_csv(
            df, "cf_armed_conflict_risk_3m", 3, __import__("datetime").date(2026, 9, 1),
            "ons_armedconf_03",
        )

        assert len(rows) == 1
        assert rows[0]["value"] == pytest.approx(0.9), (
            "the 2010 row arrived last and would have been stored as the "
            "current forecast"
        )

    def test_a_row_from_an_older_period_is_dropped_not_restamped(self):
        from resolver.connectors.conflictforecast import ConflictForecastOrgConnector

        df = self._frame([
            {"iso3": "KEN", "period": 202608, "ons_armedconf_03_all": 0.9},
            {"iso3": "SDN", "period": 201001, "ons_armedconf_03_all": 0.1},
        ])

        rows = ConflictForecastOrgConnector._transform_csv(
            df, "cf_armed_conflict_risk_3m", 3, __import__("datetime").date(2026, 9, 1),
            "ons_armedconf_03",
        )

        assert [r["iso3"] for r in rows] == ["KEN"], (
            "every surviving row is about to be stamped with one issue date, "
            "so a row from another period is not this vintage's"
        )

    def test_a_file_with_no_period_column_still_works(self):
        from resolver.connectors.conflictforecast import ConflictForecastOrgConnector

        df = self._frame([
            {"iso3": "KEN", "ons_armedconf_03_all": 0.9},
            {"iso3": "SDN", "ons_armedconf_03_all": 0.2},
        ])

        rows = ConflictForecastOrgConnector._transform_csv(
            df, "cf_armed_conflict_risk_3m", 3, __import__("datetime").date(2026, 9, 1),
            "ons_armedconf_03",
        )

        assert sorted(r["iso3"] for r in rows) == ["KEN", "SDN"]
