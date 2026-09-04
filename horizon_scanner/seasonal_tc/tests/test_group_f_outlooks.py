# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Group F of the run-33841370196 faults: seasonal TC outlooks.

F1 the table had no key but the clock; F2 the August Atlantic update carried
the May issue date; F3 the NW Pacific May outlook changed category between
runs; F4 BoM stored "October 2025" in a date column and four basins carried
no date at all; F5 no NOAA 2026 release was reachable.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from horizon_scanner.seasonal_tc import bom_scraper, mfr_swio_scraper
from horizon_scanner.seasonal_tc.dates import (
    REASON_CLIMATOLOGY,
    REASON_MONTH_PRECISION,
    REASON_NO_DATE_IN_SOURCE,
    REASON_PDF_METADATA,
    REASON_UNPARSEABLE,
    REASON_URL_MONTH_DISAGREES,
    parse_issue_date,
)
from horizon_scanner.seasonal_tc.tsr_seasonal_extractor import (
    resolve_forecast_type,
    resolve_issue_date,
)

duckdb = pytest.importorskip("duckdb")


# --------------------------------------------------------------------------
# F4: a date column holds a date, or NULL and a reason
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("2026-05-28", ("2026-05-28", None)),
        ("5th August 2026", ("2026-08-05", None)),
        ("August 5, 2026", ("2026-08-05", None)),
        ("21 octobre 2025", ("2025-10-21", None)),
        ("October 2025", ("2025-10-01", REASON_MONTH_PRECISION)),
        ("", (None, REASON_NO_DATE_IN_SOURCE)),
        (None, (None, REASON_NO_DATE_IN_SOURCE)),
        ("latest", (None, REASON_UNPARSEABLE)),
        (dt.date(2026, 8, 5), ("2026-08-05", None)),
    ],
)
def test_every_shape_a_scraper_produced_becomes_a_date_or_a_reason(raw, expected):
    assert parse_issue_date(raw) == expected


def test_bom_reads_its_own_issued_line_and_invents_nothing():
    dated = bom_scraper.extract_australian_outlook(
        "Issued: 10 October 2025\nThe 2025-26 Australian tropical cyclone season is expected "
        "to be like the long-term average. Averages since 1980-81."
    )
    assert (dated.issue_date, dated.issue_date_reason) == ("2025-10-10", "")
    assert dated.season == "2025-26" and dated.season_reason == ""

    month_only = bom_scraper.extract_south_pacific_outlook(
        "South Pacific tropical cyclone outlook. Issued October 2025. Averages since 1980-81."
    )
    assert month_only.issue_date == "2025-10-01"
    assert month_only.issue_date_reason == REASON_MONTH_PRECISION
    # No season phrase and only a baseline pair: nothing, and the reason why.
    assert month_only.season == ""
    assert month_only.season_reason

    undated = bom_scraper.extract_australian_outlook("Averages since 1980-81.")
    # The old code wrote "October {season_year}" here — free text, invented.
    assert undated.issue_date == ""
    assert undated.issue_date_reason == REASON_NO_DATE_IN_SOURCE


def test_swi_reads_the_articles_own_date_line():
    f = mfr_swio_scraper.extract_swio_outlook(
        "Prévisions saisonnières SAISON 2025-2026, publié le 21 octobre 2025."
    )
    assert (f.issue_date, f.issue_date_reason) == ("2025-10-21", "")
    assert f.season == "2025-26"
    bare = mfr_swio_scraper.extract_swio_outlook("Une saison cyclonique à venir.")
    assert bare.issue_date == "" and bare.issue_date_reason == REASON_NO_DATE_IN_SOURCE
    assert bare.season_reason


def test_the_nio_climatology_block_says_why_it_has_no_date():
    from horizon_scanner.seasonal_tc.imd_nio_scraper import build_nio_context

    (block,) = build_nio_context(fetch_live=False)
    assert block.issue_date == ""
    assert block.issue_date_reason == REASON_CLIMATOLOGY


# --------------------------------------------------------------------------
# F2: the August Atlantic update is dated August
# --------------------------------------------------------------------------


AUGUST_QUOTING_MAY_FIRST = (
    "August Forecast Update for Atlantic Hurricane Activity in 2026\n"
    "Summary: our pre-season forecast (Issued: 28th May 2026) called for 11 named "
    "storms; this update lowers that to 10.\n"
    + "text\n" * 60
    + "Issued: 5th August 2026\n"
)


def test_the_line_matching_the_filename_month_wins_wherever_it_sits():
    """The May quote sat inside the 1,500-character header of the real PDF,
    so header-first still returned May. The filename says August."""

    assert resolve_issue_date(AUGUST_QUOTING_MAY_FIRST, issue_month="August") == (
        "2026-08-05", "",
    )


def test_the_pdf_metadata_settles_a_line_that_contradicts_the_filename():
    text = "August Forecast Update\nIssued: 28th May 2026\n"
    assert resolve_issue_date(text, issue_month="August", document_date="2026-08-05") == (
        "2026-08-05", REASON_PDF_METADATA,
    )


def test_an_unsettled_contradiction_is_kept_and_flagged_never_silent():
    text = "August Forecast Update\nIssued: 28th May 2026\n"
    assert resolve_issue_date(text, issue_month="August") == (
        "2026-05-28", REASON_URL_MONTH_DISAGREES,
    )


def test_a_document_with_no_issued_line_falls_back_to_its_metadata():
    assert resolve_issue_date("Pre-Season Forecast\n", issue_month="May", document_date="2026-05-28") == (
        "2026-05-28", REASON_PDF_METADATA,
    )
    assert resolve_issue_date("nothing here", issue_month="May") == ("", REASON_NO_DATE_IN_SOURCE)


# --------------------------------------------------------------------------
# F3: the category is what the document calls itself
# --------------------------------------------------------------------------


NWP_MAY = (
    "Extended Range Forecast for Northwest Pacific Typhoon Activity in 2026\n"
    "Issued: 11th May 2026\n"
    "Summary: TSR's pre-season outlook ...\n"
)


def test_the_title_decides_the_category_whatever_the_url_month_says():
    """The same 11 May NW Pacific outlook was 'extended_range' in one run and
    'pre_season' in the next: May means pre-season for the Atlantic and the
    extended-range product for the NW Pacific, and only the title knows."""

    assert resolve_forecast_type(NWP_MAY, "2026-05-11", issue_month="May") == (
        "extended_range", "title",
    )
    assert resolve_forecast_type(NWP_MAY, "2026-05-11", issue_month="") == (
        "extended_range", "title",
    )


def test_the_summary_paragraph_is_never_read_for_the_product_name():
    text = (
        "TSR ATLANTIC HURRICANE FORECAST AUGUST UPDATE\nIssued: 5th August 2026\n\n"
        "Our pre-season forecast, Issued: 21st May 2026, called for an above-normal season.\n"
    )
    assert resolve_forecast_type(text, "2026-08-05") == ("august_update", "title")


def test_the_url_month_is_only_the_fallback_for_a_title_that_names_nothing():
    assert resolve_forecast_type("Untitled\nIssued: 11th May 2026", "2026-05-11", "May") == (
        "pre_season", "url",
    )
    assert resolve_forecast_type("Untitled\n", "2026-07-09") == ("july_update", "issue_month")


# --------------------------------------------------------------------------
# F1: one outlook is one row
# --------------------------------------------------------------------------


@pytest.fixture()
def outlook_db(tmp_path, monkeypatch):
    from pythia.db import schema

    path = tmp_path / "p.duckdb"
    monkeypatch.setenv("PYTHIA_DB_URL", f"duckdb:///{path}")
    con = duckdb.connect(str(path))
    schema.ensure_schema(con)
    con.close()
    yield path


def _rows(path):
    # The store holds a pooled read-write handle on this path; a second
    # read-only handle in the same process is refused by DuckDB, so read
    # through the same pool.
    from pythia.db.schema import connect

    con = connect()
    return con.execute(
        "SELECT basin, source, forecast_season, category, issue_date, issue_date_reason, "
        "named_storms_forecast, fetched_at FROM seasonal_tc_outlooks ORDER BY 1,2,3,4,5"
    ).fetchall()


def test_storing_the_same_outlook_twice_yields_one_row_with_its_first_fetch_time(outlook_db):
    from horizon_scanner.seasonal_tc import store_seasonal_tc_outlooks

    outlook = {"basin": "ATL", "source": "TSR", "season_year": 2026, "issue_date": "2026-07-07",
               "forecast_type": "july_update", "named_storms": 11}
    assert store_seasonal_tc_outlooks([outlook]) == 1
    first = _rows(outlook_db)
    assert store_seasonal_tc_outlooks([dict(outlook, named_storms=12)]) == 1
    second = _rows(outlook_db)
    assert len(first) == len(second) == 1
    assert second[0][6] == "12"  # updated in place
    assert second[0][7] == first[0][7]  # the first fetch time survives


def test_a_reparse_with_a_corrected_date_supersedes_the_misread_row(outlook_db):
    """The August update stored under 2026-05-28 is replaced, not joined."""

    from horizon_scanner.seasonal_tc import store_seasonal_tc_outlooks

    misread = {"basin": "ATL", "source": "TSR", "season_year": 2026, "issue_date": "2026-05-28",
               "forecast_type": "august_update", "named_storms": 10}
    pre_season = {"basin": "ATL", "source": "TSR", "season_year": 2026, "issue_date": "2026-05-28",
                  "forecast_type": "pre_season", "named_storms": 11}
    store_seasonal_tc_outlooks([misread, pre_season])
    assert len(_rows(outlook_db)) == 2
    store_seasonal_tc_outlooks([dict(misread, issue_date="2026-08-05")])
    rows = _rows(outlook_db)
    assert [(r[3], str(r[4])) for r in rows] == [
        ("august_update", "2026-08-05"), ("pre_season", "2026-05-28"),
    ]


def test_an_undated_outlook_upserts_onto_itself(outlook_db):
    from horizon_scanner.seasonal_tc import store_seasonal_tc_outlooks

    block = {"basin": "NIO", "source": "IMD_RSMC_NewDelhi", "season": "2026", "issue_date": "",
             "issue_date_reason": REASON_CLIMATOLOGY, "forecast_type": "climatology_context"}
    store_seasonal_tc_outlooks([block])
    store_seasonal_tc_outlooks([block])
    rows = _rows(outlook_db)
    assert len(rows) == 1 and rows[0][4] is None and rows[0][5] == REASON_CLIMATOLOGY


def test_the_legacy_table_is_rebuilt_deduplicated_and_dated(tmp_path):
    from pythia.db import schema

    con = duckdb.connect(str(tmp_path / "legacy.duckdb"))
    con.execute(
        "CREATE TABLE seasonal_tc_outlooks (basin VARCHAR NOT NULL, source VARCHAR NOT NULL, "
        "forecast_season VARCHAR, named_storms_forecast VARCHAR, category VARCHAR, "
        "raw_json VARCHAR, fetched_at TIMESTAMP, PRIMARY KEY (basin, source, fetched_at))"
    )
    pre = json.dumps({"issue_date": "2026-05-28", "season_year": 2026, "forecast_type": "pre_season"})
    con.executemany(
        "INSERT INTO seasonal_tc_outlooks VALUES (?,?,?,?,?,?,?)",
        [
            ("ATL", "TSR", "", "11", "pre_season", pre, "2026-07-09 08:47:57"),
            ("ATL", "TSR", "", "11", "pre_season", pre, "2026-08-28 16:11:40"),
            ("ATL", "TSR", "", "11", "pre_season", pre, "2026-09-04 06:52:29"),
            ("AUS", "BoM", "1980-81", "", "seasonal_outlook",
             json.dumps({"issue_date": "October 2025", "season": "1980-81"}), "2026-07-12 13:02:24"),
            ("AUS", "BoM", "", "", "seasonal_outlook",
             json.dumps({"issue_date": "October 2025"}), "2026-09-04 06:52:29"),
            ("SP", "BoM", "", "", "seasonal_outlook", json.dumps({}), "2026-09-04 06:52:29"),
        ],
    )
    schema.ensure_schema(con)
    rows = con.execute(
        "SELECT basin, forecast_season, issue_date, issue_date_reason, fetched_at "
        "FROM seasonal_tc_outlooks ORDER BY basin"
    ).fetchall()
    con.close()
    by_basin = {r[0]: r for r in rows}
    # Three fetches of one outlook are one row, keeping the earliest fetch.
    assert len(rows) == 3
    assert by_basin["ATL"][1] == "2026" and by_basin["ATL"][4] == dt.datetime(2026, 7, 9, 8, 47, 57)
    # "October 2025" became a date with its precision stated.
    assert str(by_basin["AUS"][2]) == "2025-10-01" and by_basin["AUS"][3] == REASON_MONTH_PRECISION
    # The 1980-81 "season" was a climatology baseline: deleted, not carried.
    assert by_basin["AUS"][1] == ""
    # Nothing at all becomes NULL with the reason.
    assert by_basin["SP"][2] is None and by_basin["SP"][3] == REASON_NO_DATE_IN_SOURCE


# --------------------------------------------------------------------------
# F5: NOAA 2026
# --------------------------------------------------------------------------


def test_the_2026_noaa_releases_are_curated_and_the_runner_looks_like_a_browser():
    from horizon_scanner.seasonal_tc import noaa_cpc_scraper as noaa

    urls = noaa.candidate_urls(2026)
    assert urls["ATL_initial"].endswith("noaa-predicts-below-normal-2026-atlantic-hurricane-season")
    assert urls["ATL_update"].endswith("noaa-maintains-prediction-for-below-normal-atlantic-hurricane-season")
    assert "CP" in urls
    assert "Mozilla" in noaa.HEADERS["User-Agent"]
