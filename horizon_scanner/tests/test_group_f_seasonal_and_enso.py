# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Group F of the run-33946954189 repairs: seasonal TC and ENSO.

Three things the run showed:

* ``NWP/TSR/2026 @ 2026-05-11`` sat in ``seasonal_tc_outlooks`` twice, once
  ``extended_range`` (fetched July) and once ``pre_season`` (fetched
  September), both quoting 27 named storms — one document classified two
  ways by two versions of the classifier;
* NOAA ERDDAP answered HTTP 403 with "Your IP address is on this ERDDAP's
  request blacklist. Did you often submit more than one request at a time?
  Did you often submit identical requests?", leaving the index ladder on
  two live ranks, the bare minimum the corroboration check demands; and
* ``python -m horizon_scanner.enso.enso_module`` loaded its own module
  twice, under a runpy warning, immediately before backfilling 919 rows.

Network-free.
"""

from __future__ import annotations

import datetime as dt

import pytest

from horizon_scanner.enso import indices as idx

TODAY = dt.date(2026, 9, 6)


# --------------------------------------------------------------------------
# F1/F2 — one document, one row
# --------------------------------------------------------------------------


@pytest.fixture()
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHIA_DB_URL", f"duckdb:///{tmp_path / 'pythia.duckdb'}")
    from pythia.db.schema import connect, ensure_schema

    con = connect(read_only=False)
    ensure_schema(con)
    con.close()
    yield


def _outlook(**kwargs):
    base = {
        "basin": "NWP",
        "source": "TSR",
        "season": "2026",
        "forecast_type": "extended_range",
        "issue_date": "2026-05-11",
        "named_storms": 27,
    }
    base.update(kwargs)
    return base


def _rows():
    from pythia.db.schema import connect

    con = connect(read_only=True)
    try:
        return con.execute(
            "SELECT basin, source, forecast_season, category, issue_date "
            "FROM seasonal_tc_outlooks ORDER BY category"
        ).fetchall()
    finally:
        con.close()


class TestOneDocumentOneRow:
    def test_a_re_categorised_document_supersedes_its_earlier_row(self, db):
        """Same date, same figures: one document read twice."""

        from horizon_scanner.seasonal_tc import store_seasonal_tc_outlooks

        store_seasonal_tc_outlooks([_outlook(forecast_type="extended_range")])
        store_seasonal_tc_outlooks([_outlook(forecast_type="pre_season")])

        rows = _rows()
        assert len(rows) == 1, (
            "one basin, one source, one season, one issue date and one storm "
            "count is ONE document, and a document has one category"
        )
        assert rows[0][3] == "pre_season"

    def test_two_documents_misdated_onto_one_day_both_stand(self, db):
        """Same date, DIFFERENT figures: two documents, one misdated.

        This is the shape PR #892 was about — the August Atlantic update
        stored under the May date beside the real May forecast. Collapsing
        them would delete a correct row, so both stand and the store logs
        the contradiction.
        """

        from horizon_scanner.seasonal_tc import store_seasonal_tc_outlooks

        store_seasonal_tc_outlooks([
            _outlook(basin="ATL", forecast_type="august_update",
                     issue_date="2026-05-28", named_storms=10),
            _outlook(basin="ATL", forecast_type="pre_season",
                     issue_date="2026-05-28", named_storms=11),
        ])

        assert len(_rows()) == 2

    def test_a_row_with_no_figure_supersedes_nothing(self, db):
        from horizon_scanner.seasonal_tc import store_seasonal_tc_outlooks

        store_seasonal_tc_outlooks([_outlook(forecast_type="extended_range")])
        store_seasonal_tc_outlooks([
            _outlook(forecast_type="pre_season", named_storms=None),
        ])

        assert len(_rows()) == 2, (
            "a row carrying no figure cannot establish that it is the same "
            "document, so it never deletes one"
        )

    def test_two_documents_on_different_dates_both_stand(self, db):
        from horizon_scanner.seasonal_tc import store_seasonal_tc_outlooks

        store_seasonal_tc_outlooks([
            _outlook(forecast_type="pre_season", issue_date="2026-05-11"),
        ])
        store_seasonal_tc_outlooks([
            _outlook(forecast_type="august_update", issue_date="2026-08-07"),
        ])

        assert len(_rows()) == 2

    def test_a_supersede_never_reaches_another_season(self, db):
        from horizon_scanner.seasonal_tc import store_seasonal_tc_outlooks

        store_seasonal_tc_outlooks([
            _outlook(season="2025", forecast_type="pre_season",
                     issue_date="2025-05-12"),
        ])
        store_seasonal_tc_outlooks([
            _outlook(season="2026", forecast_type="extended_range"),
        ])
        store_seasonal_tc_outlooks([
            _outlook(season="2026", forecast_type="pre_season"),
        ])

        seasons = sorted(r[2] for r in _rows())
        assert seasons == ["2025", "2026"], (
            "the 56-row table became 16 rows in one migration; a store must "
            "never be able to take a season with it"
        )

    def test_an_undated_outlook_is_never_collapsed_by_date(self, db):
        from horizon_scanner.seasonal_tc import store_seasonal_tc_outlooks

        store_seasonal_tc_outlooks([
            _outlook(forecast_type="climatology_context", issue_date=None),
        ])
        store_seasonal_tc_outlooks([
            _outlook(forecast_type="pre_season", issue_date=None),
        ])

        assert len(_rows()) == 2, (
            "two undated products share an issue_date_key of 'undated:...' "
            "and are not the same document"
        )


# --------------------------------------------------------------------------
# F3 — the index ladder
# --------------------------------------------------------------------------


PSL_BODY = """  1950 2026
  2025 -0.61 -0.42 -0.20  0.05  0.24  0.35  0.44  0.51  0.62  0.80  1.02  1.25
  2026  1.41  1.55  1.62  1.70  1.74  1.78  1.80  1.82 -99.99 -99.99 -99.99 -99.99
  -99.99
  NOAA/PSL
"""


class TestPslRank:
    def test_the_monthly_series_parses(self):
        observations = idx.parse_psl_monthly(PSL_BODY)

        assert observations[0].date == dt.date(2025, 1, 1)
        assert observations[-1].date == dt.date(2026, 8, 1)
        assert observations[-1].anomaly == pytest.approx(1.82)

    def test_the_missing_sentinel_is_not_a_reading(self):
        values = [o.anomaly for o in idx.parse_psl_monthly(PSL_BODY)]

        assert all(abs(v) <= 4.0 for v in values)
        assert -99.99 not in values

    def test_metadata_lines_are_ignored(self):
        assert idx.parse_psl_monthly("  1950 2026\n  -99.99\n  NOAA/PSL\n") == []

    def test_it_is_a_fourth_independent_rank(self):
        ladder = idx.source_ladder(TODAY)
        names = [spec["name"] for spec in ladder]

        assert "noaa_psl_nina34_monthly" in names
        assert len(ladder) >= 4, (
            "the ladder was down to two live ranks after the blacklist, "
            "which is the corroboration floor and no margin at all"
        )
        hosts = {spec["url"].split("/")[2] for spec in ladder}
        assert len(hosts) >= 3, "a rank on the same host is not independent"

    def test_a_monthly_series_is_never_published_as_a_weekly_reading(self):
        resolution = idx.resolve_indices(
            get=lambda url, timeout: (
                PSL_BODY if "psl.noaa.gov" in url else _raise(url)
            ),
            today=TODAY,
        )

        assert resolution.oni is not None
        assert resolution.source_name == "noaa_psl_nina34_monthly"
        assert resolution.nino34_weekly is None, (
            "a monthly mean under a weekly label hides that the run was one "
            "source short"
        )


def _raise(url):
    raise RuntimeError(f"no fixture for {url}")


class TestResponseCache:
    def test_a_cached_body_is_served_without_a_request(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PYTHIA_ENSO_CACHE_DIR", str(tmp_path))
        url = "https://coastwatch.pfeg.noaa.gov/erddap/x.csv"
        idx.write_cached_body(url, "hello")

        body, age = idx.read_cached_body(url)

        assert body == "hello"
        assert age < 60

    def test_a_failed_fetch_serves_the_cache_rather_than_leaving_the_rank_unread(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("PYTHIA_ENSO_CACHE_DIR", str(tmp_path))
        monkeypatch.setenv("PYTHIA_ENSO_CACHE_TTL_SEC", "0")
        url = "https://coastwatch.pfeg.noaa.gov/erddap/tabledap/x.csv"
        idx.write_cached_body(url, "cached-body")

        class _Resp:
            status_code = 403
            text = "Your IP address is on this ERDDAP's request blacklist."

        class _Requests:
            @staticmethod
            def get(*_a, **_k):
                return _Resp()

        monkeypatch.setitem(__import__("sys").modules, "requests", _Requests)
        monkeypatch.setattr(idx, "_throttle", lambda _url: None)

        assert idx._default_get(url, 5.0) == "cached-body"

    def test_a_failed_fetch_with_no_cache_still_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PYTHIA_ENSO_CACHE_DIR", str(tmp_path))

        class _Resp:
            status_code = 403
            text = "blacklisted"

        class _Requests:
            @staticmethod
            def get(*_a, **_k):
                return _Resp()

        monkeypatch.setitem(__import__("sys").modules, "requests", _Requests)
        monkeypatch.setattr(idx, "_throttle", lambda _url: None)

        with pytest.raises(RuntimeError, match="403"):
            idx._default_get("https://example.invalid/never-cached", 5.0)

    def test_the_cache_can_be_turned_off(self, monkeypatch):
        monkeypatch.setenv("PYTHIA_ENSO_CACHE_DIR", "")

        assert idx._cache_path("https://example.invalid/x") is None


# --------------------------------------------------------------------------
# F4 — the package runs its module once
# --------------------------------------------------------------------------


class TestPackageEntryPoint:
    def test_the_package_has_a_main_module(self):
        import importlib

        module = importlib.import_module("horizon_scanner.enso.__main__")

        assert callable(module.main)

    def test_the_workflow_invokes_the_package_not_the_module(self):
        import pathlib

        workflow = pathlib.Path(".github/workflows/resolver_update.yml").read_text()

        assert "python -m horizon_scanner.enso --backfill-oni" in workflow
        assert "python -m horizon_scanner.enso.enso_module" not in workflow
