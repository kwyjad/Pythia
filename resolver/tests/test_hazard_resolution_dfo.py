# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Phase-5 tests for the Dartmouth Flood Observatory cross-check.

The load-bearing test in this file is
:func:`test_dfo_never_writes_to_any_resolution_table`. Everything else is
parsing and arithmetic; that one is the guarantee. An archive assembled
from press reporting is a different quantity from a resolved
people-affected figure, and a source admitted "just for calibration" is one
refactor away from being admitted as an answer — which is exactly the
mistake GDACS exposure once caused in ``facts_resolved``.

The other thing pinned here is that a feed which resolves nothing is
REFUSED rather than read as a world without floods, and that the comparison
window never overlaps the machine's own — a cross-check against the years
the machine already resolved would confirm nothing.

Network-free: the transport is injected.
"""

from __future__ import annotations

import csv
import datetime as dt
import io

import duckdb
import pytest

from resolver.hazard_resolution import base_rates as br
from resolver.hazard_resolution import dfo as dfo_mod
from resolver.hazard_resolution.rulebook import RulebookError, validate_rulebook
from resolver.hazard_resolution.schema import ensure_haz_schema
from resolver.tests.hazard_resolution_utils import make_rulebook, seed_trigger

TODAY = dt.date(2026, 8, 5)


@pytest.fixture()
def rulebook():
    # The shipped rulebook points at the xlsx export; these tests drive the
    # csv path because it is inspectable in a fixture. The xlsx reader has
    # its own test below, so both shapes stay covered.
    return make_rulebook({"dfo": {"format": "csv"}})


@pytest.fixture()
def con():
    con = duckdb.connect(":memory:")
    ensure_haz_schema(con)
    return con


def archive_csv(rows: list[dict[str, object]], headers: list[str] | None = None) -> bytes:
    """A DFO-shaped CSV export."""

    headers = headers or ["ID", "Country", "OtherCountry", "Began", "Ended", "Dead", "Displaced"]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=headers, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue().encode("utf-8")


def _flood(record_id: str, country: str, began: str, **extra: object) -> dict[str, object]:
    return {
        "ID": record_id,
        "Country": country,
        "Began": began,
        "Ended": began,
        "Dead": 3,
        "Displaced": 1000,
        **extra,
    }


def _getter(blob: bytes):
    def get(url: str, timeout: float) -> bytes:
        return blob

    return get


# ---------------------------------------------------------------------------
# The guarantee
# ---------------------------------------------------------------------------


def test_dfo_never_writes_to_any_resolution_table(con, rulebook):
    """DFO is calibration. It may cache itself and nothing else.

    If this ever fails, an archive of press-reported floods has become
    capable of triggering a hazard or supplying a people-affected figure.
    """

    blob = archive_csv([_flood("1", "Philippines", "2005-08-03")])
    dfo_mod.fetch_dfo(con, rulebook, get=_getter(blob))
    dfo_mod.cross_check(con, rulebook)

    assert con.execute("SELECT COUNT(*) FROM haz_raw_dfo").fetchone()[0] == 1
    for table in ("haz_triggers", "haz_impact_candidates", "haz_resolutions"):
        assert con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0, table


def test_the_rulebook_refuses_a_cross_check_window_that_overlaps_the_backcast():
    """Comparing the machine against the years it already resolved from
    would be a self-confirmation, so the validator rejects it."""

    data = make_rulebook().raw
    data["dfo"] = {**data["dfo"], "cross_check_end_year": 2015}  # backcast.flood is 2010
    problems = validate_rulebook(data)
    assert any("cross_check_end_year" in p for p in problems)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_parses_a_csv_archive_into_events():
    blob = archive_csv(
        [
            _flood("1", "Philippines", "2005-08-03"),
            _flood("2", "Bangladesh", "2004-07-11"),
        ]
    )
    events, report = dfo_mod.parse_archive(blob, "csv")

    assert {e.iso3 for e in events} == {"PHL", "BGD"}
    assert report["parsed"] == 2
    assert report["dropped_no_iso3"] == 0


def test_column_names_are_matched_tolerantly():
    """The archive has shipped several spellings; a rename should cost a
    line in the alias table, not a debugging session."""

    blob = archive_csv(
        [{"Register #": "1", "Main Country": "Kenya", "Start Date": "2006-04-02"}],
        headers=["Register #", "Main Country", "Start Date"],
    )
    events, report = dfo_mod.parse_archive(blob, "csv")

    assert [e.iso3 for e in events] == ["KEN"]
    assert report["parsed"] == 1


def test_parses_the_xlsx_export_the_rulebook_ships_with():
    """The shipped dfo.format is xlsx, so that reader needs its own cover —
    the csv path used by the rest of this file would otherwise be the only
    one ever exercised."""

    from openpyxl import Workbook

    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["ID", "Country", "Began", "Ended"])
    sheet.append(["1", "Philippines", "2005-08-03", "2005-08-09"])
    buffer = io.BytesIO()
    workbook.save(buffer)

    events, report = dfo_mod.parse_archive(buffer.getvalue(), "xlsx")

    assert [e.iso3 for e in events] == ["PHL"]
    assert report["parsed"] == 1


def test_a_cross_border_flood_counts_for_every_country_it_names():
    """Occurrence asks whether a flood happened here, not whose figure it is."""

    blob = archive_csv([_flood("1", "Nepal", "2007-08-01", OtherCountry="India; Bangladesh")])
    events, _ = dfo_mod.parse_archive(blob, "csv")

    assert {e.iso3 for e in events} == {"NPL", "IND", "BGD"}
    # The spillover entries get distinct record ids so the cache keeps them all.
    assert len({e.record_id for e in events}) == 3


def test_rows_without_a_usable_date_or_country_are_counted_not_silently_dropped():
    blob = archive_csv(
        [
            _flood("1", "Philippines", "2005-08-03"),
            _flood("2", "Atlantis", "2005-08-03"),
            _flood("3", "Kenya", ""),
        ]
    )
    events, report = dfo_mod.parse_archive(blob, "csv")

    assert len(events) == 1
    assert report["dropped_no_iso3"] == 1
    assert report["dropped_no_date"] == 1
    assert "Atlantis" in report["unresolved_country_samples"]


def test_a_missing_required_column_is_an_error_not_an_empty_archive():
    blob = archive_csv([{"ID": "1", "Dead": 3}], headers=["ID", "Dead"])
    events, report = dfo_mod.parse_archive(blob, "csv")

    assert events == []
    assert "missing required column" in report["error"]


# ---------------------------------------------------------------------------
# Fetch outcomes
# ---------------------------------------------------------------------------


def test_fetch_caches_events_and_is_idempotent(con, rulebook):
    blob = archive_csv([_flood("1", "Philippines", "2005-08-03")])

    first = dfo_mod.fetch_dfo(con, rulebook, get=_getter(blob))
    second = dfo_mod.fetch_dfo(con, rulebook, get=_getter(blob))

    assert first.ok and second.ok
    assert second.inserted == 0  # identical content is a no-op
    assert con.execute("SELECT COUNT(*) FROM haz_raw_dfo").fetchone()[0] == 1


def test_an_unreachable_archive_is_unavailable_not_empty(con, rulebook):
    def boom(url: str, timeout: float) -> bytes:
        raise ConnectionError("host unreachable")

    outcome = dfo_mod.fetch_dfo(con, rulebook, get=boom)

    assert outcome.ok is False
    assert "unreachable" in (outcome.error or "")
    assert con.execute("SELECT COUNT(*) FROM haz_raw_dfo").fetchone()[0] == 0


def test_an_archive_that_resolves_nothing_is_refused(con, rulebook):
    """A feed where 0 of N rows resolve has changed shape. Caching it would
    record 'no floods anywhere' as a fact about the world."""

    blob = archive_csv([_flood(str(i), "Atlantis", "2005-08-03") for i in range(5)])
    outcome = dfo_mod.fetch_dfo(con, rulebook, get=_getter(blob))

    assert outcome.ok is False
    assert "0 of 5" in (outcome.error or "")
    assert con.execute("SELECT COUNT(*) FROM haz_raw_dfo").fetchone()[0] == 0


def test_disabling_the_cross_check_in_the_rulebook_stops_the_fetch(con):
    disabled = make_rulebook({"dfo": {"format": "csv", "enabled": False}})

    def explode(url: str, timeout: float) -> bytes:  # pragma: no cover - must not run
        raise AssertionError("fetch must not be attempted when dfo.enabled is false")

    outcome = dfo_mod.fetch_dfo(con, disabled, get=explode)
    assert outcome.ok is False


# ---------------------------------------------------------------------------
# Occurrence arithmetic
# ---------------------------------------------------------------------------


def test_occurrence_denominator_is_the_archives_span_not_the_countrys():
    """A country seen only in its wet years would otherwise show p = 1.0 for
    every month it appears in — an artefact of counting only the years it
    was seen."""

    events = [
        dfo_mod.DfoEvent(record_id="1", iso3="PHL", began=dt.date(2000, 8, 1)),
        dfo_mod.DfoEvent(record_id="2", iso3="PHL", began=dt.date(2002, 8, 1)),
        dfo_mod.DfoEvent(record_id="3", iso3="BGD", began=dt.date(2004, 6, 1)),
    ]
    rates = dfo_mod.occurrence_rates(events)

    # Archive spans 2000..2004 = 5 years for everyone.
    assert rates[("PHL", 8)]["p_occurrence"] == pytest.approx(2 / 5)
    assert rates[("PHL", 8)]["n_years"] == 5
    assert rates[("BGD", 6)]["p_occurrence"] == pytest.approx(1 / 5)


def test_two_floods_in_one_month_of_one_year_count_once():
    events = [
        dfo_mod.DfoEvent(record_id="1", iso3="PHL", began=dt.date(2000, 8, 1)),
        dfo_mod.DfoEvent(record_id="2", iso3="PHL", began=dt.date(2000, 8, 20)),
    ]
    rates = dfo_mod.occurrence_rates(events)
    assert rates[("PHL", 8)]["n_years_with_event"] == 1


def test_load_events_honours_the_pre_backcast_cutoff(con, rulebook):
    blob = archive_csv(
        [
            _flood("1", "Philippines", "2005-08-03"),
            _flood("2", "Philippines", "2015-08-03"),
        ]
    )
    dfo_mod.fetch_dfo(con, rulebook, get=_getter(blob))

    before = dfo_mod.load_events(con, before_year=2010)
    assert [e.began.year for e in before] == [2005]


# ---------------------------------------------------------------------------
# Cross-check
# ---------------------------------------------------------------------------


def _seed_machine_flood_rate(con, rulebook, iso3: str, month: int, triggered_years: set[int]):
    for year in range(2011, 2021):
        seed_trigger(
            con, iso3=iso3, ym=f"{year}-{month:02d}", hazard="FL",
            triggered=year in triggered_years,
        )
    br.compute_occurrence(con, rulebook, hazards=["FL"], today=TODAY)


def test_cross_check_flags_a_country_the_machine_barely_sees(con, rulebook):
    # DFO: floods in 8 of the 10 pre-2010 Augusts. Machine: 1 of 10.
    blob = archive_csv(
        [_flood(str(year), "Philippines", f"{year}-08-03") for year in range(2000, 2008)]
        + [_flood("pad", "Philippines", "2009-08-03")]
    )
    dfo_mod.fetch_dfo(con, rulebook, get=_getter(blob))
    _seed_machine_flood_rate(con, rulebook, "PHL", 8, {2015})

    check = dfo_mod.cross_check(con, rulebook)

    assert check.available
    row = next(r for r in check.rows if r.iso3 == "PHL")
    assert row.dfo_p > row.machine_p
    assert row.diverges
    assert check.diverging


def test_cross_check_does_not_flag_agreement(con, rulebook):
    blob = archive_csv(
        [_flood(str(year), "Philippines", f"{year}-08-03") for year in (2001, 2003, 2005, 2007, 2009)]
    )
    dfo_mod.fetch_dfo(con, rulebook, get=_getter(blob))
    _seed_machine_flood_rate(con, rulebook, "PHL", 8, {2012, 2014, 2016, 2018, 2020})

    check = dfo_mod.cross_check(con, rulebook)

    assert check.available
    assert not check.diverging


def test_cross_check_reports_why_it_could_not_run(con, rulebook):
    """Unavailable is a finding, not a silence — and it must be
    distinguishable from 'the rates agree'."""

    empty = dfo_mod.cross_check(con, rulebook)
    assert empty.available is False
    assert "no cached DFO events" in (empty.reason or "")

    # An archive too short to compare is its own distinct reason.
    blob = archive_csv([_flood("1", "Philippines", "2005-08-03")])
    dfo_mod.fetch_dfo(con, rulebook, get=_getter(blob))
    too_short = dfo_mod.cross_check(con, rulebook)
    assert too_short.available is False
    assert "min_years_for_comparison" in (too_short.reason or "")

    # With a long enough archive but no computed base rates, the reason
    # names the missing table rather than blaming the archive.
    long_enough = archive_csv(
        [_flood(str(year), "Philippines", f"{year}-08-03") for year in range(2000, 2010)]
    )
    dfo_mod.fetch_dfo(con, rulebook, get=_getter(long_enough))
    no_base_rates = dfo_mod.cross_check(con, rulebook)
    assert no_base_rates.available is False
    assert "haz_base_rates_occurrence" in (no_base_rates.reason or "")


def test_cross_check_divergence_factor_comes_from_the_rulebook(con, rulebook):
    blob = archive_csv(
        [_flood(str(year), "Philippines", f"{year}-08-03") for year in range(2000, 2010)]
    )
    dfo_mod.fetch_dfo(con, rulebook, get=_getter(blob))
    # Machine sees 5 of 10 Augusts; DFO sees 10 of 10 — a 2x difference.
    _seed_machine_flood_rate(con, rulebook, "PHL", 8, {2011, 2013, 2015, 2017, 2019})

    assert not dfo_mod.cross_check(con, rulebook).diverging  # factor 3.0

    strict = make_rulebook({"dfo": {"format": "csv", "divergence_factor": 1.5}})
    assert dfo_mod.cross_check(con, strict).diverging
