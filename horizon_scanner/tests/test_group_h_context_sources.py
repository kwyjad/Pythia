# Pythia / Copyright (c) 2025 Kevin Wyjad
"""Group H — the context sources that lied quietly.

H3 CrisisWatch: a regional heading warned as an unmatched country in every
run, and four editions of 2026 were permanent holes because the monthly
path only ever reads the newest snapshot. H5 HDX Signals: freshness was
measured from a file mtime that git resets on every checkout, so the
Horizon Scanner has been reading a committed snapshot. And the ACLED CAST
block, absent since January, was omitted in silence while the closing
paragraph went on describing it to the model.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

bs4 = pytest.importorskip("bs4")


def _load_refresh():
    spec = importlib.util.spec_from_file_location(
        "_h_refresh", REPO_ROOT / "scripts" / "refresh_crisiswatch.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["_h_refresh"] = module
    spec.loader.exec_module(module)
    return module


_PAGE = """
<html><body>
  <div class="c-crisiswatch-entry" data-entry-country="nile-waters" id="a">
    <h3><span class="o-icon"><svg><use xlink:href="#deteriorated"></use></svg></span>Nile Waters</h3>
    <time>March 2026</time>
    <div class="o-crisis-states__detail"><p><strong>Dam talks stalled.</strong></p></div>
  </div>
  <div class="c-crisiswatch-entry" data-entry-country="somalia" id="b">
    <h3>Somalia</h3>
    <time>March 2026</time>
    <div class="o-crisis-states__detail"><p><strong>Al-Shabaab offensive.</strong></p></div>
  </div>
  <div class="c-crisiswatch-entry" data-entry-country="atlantis" id="c">
    <h3>Atlantis</h3>
    <time>March 2026</time>
    <div class="o-crisis-states__detail"><p><strong>Submerged.</strong></p></div>
  </div>
</body></html>
"""


class TestRegionalHeadingsAreNotUnmatchedCountries:
    @pytest.fixture(scope="class")
    def refresh(self):
        return _load_refresh()

    def test_a_regional_heading_raises_no_unmatched_warning(self, refresh, caplog):
        soup = bs4.BeautifulSoup(_PAGE, "html.parser")
        with caplog.at_level("WARNING"):
            entries, _, _ = refresh._parse_country_entries(soup)
        assert "Nile Waters" not in caplog.text
        assert "Korean Peninsula" not in caplog.text

    def test_a_genuinely_unmatched_country_still_warns(self, refresh, caplog):
        soup = bs4.BeautifulSoup(_PAGE, "html.parser")
        with caplog.at_level("WARNING"):
            refresh._parse_country_entries(soup)
        # The warning must still reach a reader for a name nothing resolves.
        assert "Atlantis" in caplog.text

    def test_the_regional_heading_still_expands_into_its_member_states(self, refresh):
        soup = bs4.BeautifulSoup(_PAGE, "html.parser")
        entries, _, _ = refresh._parse_country_entries(soup)
        expanded = {e["iso3"] for e in entries if e.get("regional_source") == "Nile Waters"}
        assert expanded == {"ETH", "SDN", "EGY"}

    def test_every_expanded_row_carries_its_reason(self, refresh):
        soup = bs4.BeautifulSoup(_PAGE, "html.parser")
        entries, _, _ = refresh._parse_country_entries(soup)
        for entry in entries:
            if entry.get("regional_source"):
                assert entry["iso3_reason"] == "regional_expansion"


class TestEditionBackfill:
    @pytest.fixture(scope="class")
    def refresh(self):
        return _load_refresh()

    @pytest.fixture(autouse=True)
    def _no_cdx(self, refresh, monkeypatch):
        """Keep the targeted per-edition probe off the network.

        Since the walk became targeted, pass 1 asks CDX for the captures in
        the month after each wanted edition. These tests hand the walk a
        snapshot list directly, so the windowed query must answer nothing
        and let pass 2 — the general listing they DO patch — do the work.
        """
        monkeypatch.setattr(refresh, "_cdx_timestamps", lambda **k: [])

    def test_wanted_editions_parse(self, refresh):
        assert refresh._wanted_editions("2026-03, 2026-4") == [(2026, 3), (2026, 4)]
        assert refresh._wanted_editions("") == []

    def test_a_bad_edition_is_refused(self, refresh):
        with pytest.raises(ValueError):
            refresh._wanted_editions("March 2026")
        with pytest.raises(ValueError):
            refresh._wanted_editions("2026-13")

    def test_a_wanted_edition_is_recovered_and_written(self, refresh, tmp_path, monkeypatch):
        monkeypatch.setattr(
            refresh, "_list_wayback_snapshots",
            lambda **k: ["20260405000000", "20260310000000"],
        )
        pages = {
            "20260310000000": _PAGE,                              # March 2026
            "20260405000000": _PAGE.replace("March 2026", "April 2026"),
        }
        monkeypatch.setattr(
            refresh, "_fetch_snapshot_html", lambda ts, **k: pages[ts]
        )
        out = refresh.backfill_editions(
            "2026-03", backfill_dir=tmp_path,
        )
        assert out["recovered"] == ["2026-03"]
        assert out["still_missing"] == []
        written = json.loads((tmp_path / "crisiswatch_2026-03.json").read_text())
        assert written["month"].startswith("March")
        assert written["entries"], "a recovered edition must carry rows"

    def test_the_latest_edition_file_is_never_touched(self, refresh, tmp_path, monkeypatch):
        monkeypatch.setattr(
            refresh, "_list_wayback_snapshots", lambda **k: ["20260310000000"],
        )
        monkeypatch.setattr(refresh, "_fetch_snapshot_html", lambda ts, **k: _PAGE)
        refresh.backfill_editions("2026-03", backfill_dir=tmp_path)
        assert not (tmp_path / "crisiswatch_latest.json").exists()
        names = sorted(p.name for p in tmp_path.iterdir())
        assert names == ["crisiswatch_2026-03.json"]

    def test_an_edition_the_archive_cannot_supply_is_named_not_hidden(
        self, refresh, tmp_path, monkeypatch,
    ):
        monkeypatch.setattr(
            refresh, "_list_wayback_snapshots", lambda **k: ["20260310000000"],
        )
        monkeypatch.setattr(refresh, "_fetch_snapshot_html", lambda ts, **k: _PAGE)
        out = refresh.backfill_editions("2026-03,2026-07", backfill_dir=tmp_path)
        assert out["recovered"] == ["2026-03"]
        assert out["still_missing"] == ["2026-07"]

    def test_an_empty_parse_is_never_written(self, refresh, tmp_path, monkeypatch):
        monkeypatch.setattr(
            refresh, "_list_wayback_snapshots", lambda **k: ["20260310000000"],
        )
        empty = _PAGE.replace('class="c-crisiswatch-entry"', 'class="other"')
        monkeypatch.setattr(refresh, "_fetch_snapshot_html", lambda ts, **k: empty)
        out = refresh.backfill_editions("2026-03", backfill_dir=tmp_path)
        assert out["recovered"] == []
        assert list(tmp_path.iterdir()) == []

    def test_one_unreadable_snapshot_does_not_end_the_walk(
        self, refresh, tmp_path, monkeypatch,
    ):
        monkeypatch.setattr(
            refresh, "_list_wayback_snapshots",
            lambda **k: ["20260215000000", "20260310000000"],
        )

        def _fetch(ts, **k):
            if ts == "20260215000000":
                raise RuntimeError("archive.org refused")
            return _PAGE

        monkeypatch.setattr(refresh, "_fetch_snapshot_html", _fetch)
        out = refresh.backfill_editions("2026-03", backfill_dir=tmp_path)
        assert out["recovered"] == ["2026-03"]


class TestMissingEditions:
    def test_the_current_month_is_never_asked_for(self, monkeypatch, tmp_path):
        import duckdb

        from horizon_scanner import crisiswatch as cw

        db = tmp_path / "cw.duckdb"
        con = duckdb.connect(str(db))
        con.execute(
            "CREATE TABLE crisiswatch_entries "
            "(iso3 TEXT, year INTEGER, month INTEGER)"
        )
        con.close()

        opened = duckdb.connect(str(db))
        monkeypatch.setattr(
            "pythia.db.schema.connect", lambda *a, **k: opened, raising=False,
        )
        try:
            missing = cw.missing_editions(3)
        finally:
            opened.close()
        today = datetime.utcnow().date()
        assert f"{today.year:04d}-{today.month:02d}" not in missing
        assert len(missing) == 3

    def test_a_month_already_in_the_table_is_not_asked_for(self, monkeypatch, tmp_path):
        import duckdb

        from horizon_scanner import crisiswatch as cw

        today = datetime.utcnow().date()
        prev_year, prev_month = (
            (today.year, today.month - 1) if today.month > 1 else (today.year - 1, 12)
        )
        db = tmp_path / "cw2.duckdb"
        con = duckdb.connect(str(db))
        con.execute(
            "CREATE TABLE crisiswatch_entries "
            "(iso3 TEXT, year INTEGER, month INTEGER)"
        )
        con.execute(
            "INSERT INTO crisiswatch_entries VALUES ('SOM', ?, ?)",
            [prev_year, prev_month],
        )
        con.close()

        opened = duckdb.connect(str(db))
        monkeypatch.setattr(
            "pythia.db.schema.connect", lambda *a, **k: opened, raising=False,
        )
        try:
            missing = cw.missing_editions(3)
        finally:
            opened.close()
        assert f"{prev_year:04d}-{prev_month:02d}" not in missing

    def test_an_unreadable_table_asks_for_nothing(self, monkeypatch):
        from horizon_scanner import crisiswatch as cw

        def _boom(*a, **k):
            raise RuntimeError("no database here")

        monkeypatch.setattr("pythia.db.schema.connect", _boom, raising=False)
        # A backfill that cannot tell what is missing must ask for nothing,
        # never for everything.
        assert cw.missing_editions(12) == []


class TestHdxCacheFreshness:
    def test_freshness_is_measured_from_the_newest_signal(self):
        from horizon_scanner import hdx_signals as hdx

        rows = [
            {"date": "2026-08-01 00:00:00"},
            {"date": "2026-09-02"},
            {"date": ""},
        ]
        assert hdx._newest_signal_date(rows).date().isoformat() == "2026-09-02"

    def test_a_cache_with_no_parseable_date_reports_none(self):
        from horizon_scanner import hdx_signals as hdx

        assert hdx._newest_signal_date([{"date": "sometime"}]) is None
        assert hdx._newest_signal_date([]) is None

    def test_a_freshly_checked_out_stale_cache_is_refreshed(self, monkeypatch, tmp_path):
        """The bug: git stamps the checkout time on a committed cache file."""

        from horizon_scanner import hdx_signals as hdx

        cache = tmp_path / "hdx_signals.csv"
        old = (datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d %H:%M:%S")
        cache.write_text(f"iso3,indicator_id,date\nSDN,x,{old}\n", encoding="utf-8")
        monkeypatch.setattr(hdx, "CACHE_FILE", cache)
        monkeypatch.setattr(hdx, "_SIGNALS_CACHE", None, raising=False)

        called: list[bool] = []

        def _fetch():
            called.append(True)
            return cache

        monkeypatch.setattr(hdx, "fetch_and_cache", _fetch)
        hdx.ensure_cache_fresh()
        assert called, (
            "an mtime test called this fresh on every CI run; the content is "
            "over a year old and must trigger a download"
        )

    def test_a_genuinely_fresh_cache_is_not_refetched(self, monkeypatch, tmp_path):
        from horizon_scanner import hdx_signals as hdx

        cache = tmp_path / "hdx_signals.csv"
        recent = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d %H:%M:%S")
        cache.write_text(f"iso3,indicator_id,date\nSDN,x,{recent}\n", encoding="utf-8")
        monkeypatch.setattr(hdx, "CACHE_FILE", cache)
        monkeypatch.setattr(hdx, "_SIGNALS_CACHE", None, raising=False)

        called: list[bool] = []
        monkeypatch.setattr(
            hdx, "fetch_and_cache", lambda: (called.append(True), cache)[1],
        )
        assert hdx.ensure_cache_fresh() is True
        assert not called


class TestCastUnavailabilityIsStated:
    def test_the_block_says_cast_is_unavailable(self):
        from horizon_scanner import conflict_forecasts as cf

        text = cf.format_conflict_forecasts_for_prompt(
            {
                "views_fatalities": [{"lead_months": 1, "value": 12.0}],
                "views_issue_date": "2026-08-01",
                "cast_unavailable_reason": (
                    "ACLED CAST is UNAVAILABLE: the newest vintage this system "
                    "holds was issued 2025-12-10 and the last month it forecasts "
                    "is 2026-05-01, which is now in the past."
                ),
            }
        )
        assert "ACLED CAST" in text
        assert "UNAVAILABLE" in text
        assert "2025-12-10" in text

    def test_the_closing_paragraph_stops_describing_a_source_never_given(self):
        from horizon_scanner import conflict_forecasts as cf

        text = cf.format_conflict_forecasts_for_prompt(
            {
                "views_fatalities": [{"lead_months": 1, "value": 12.0}],
                "views_issue_date": "2026-08-01",
                "cast_unavailable_reason": "ACLED CAST is UNAVAILABLE: no vintage.",
            }
        )
        assert "ACLED CAST provides event count predictions" not in text
        assert "No ACLED CAST event-count forecast was available" in text

    def test_a_live_cast_forecast_still_reads_as_before(self):
        from horizon_scanner import conflict_forecasts as cf

        text = cf.format_conflict_forecasts_for_prompt(
            {
                "cast_total": [{"lead_months": 1, "value": 40.0}],
                "cast_issue_date": "2026-09-01",
            }
        )
        assert "ACLED CAST provides event count predictions" in text
        assert "NO DATA" not in text

    def test_the_reason_distinguishes_never_published_from_a_frozen_vintage(self, tmp_path):
        import duckdb

        from horizon_scanner import conflict_forecasts as cf

        con = duckdb.connect(str(tmp_path / "cf.duckdb"))
        try:
            con.execute(
                "CREATE TABLE conflict_forecasts (source TEXT, iso3 TEXT, "
                "forecast_issue_date DATE, target_month DATE)"
            )
            assert "never published" in cf.cast_unavailable_reason(con, "SDN")
            con.execute(
                "INSERT INTO conflict_forecasts VALUES "
                "('ACLED_CAST', 'SDN', DATE '2025-12-10', DATE '2026-05-01')"
            )
            reason = cf.cast_unavailable_reason(con, "SDN")
        finally:
            con.close()
        assert "2025-12-10" in reason
        assert "2026-05-01" in reason
        assert "not as a forecast of no events" in reason
