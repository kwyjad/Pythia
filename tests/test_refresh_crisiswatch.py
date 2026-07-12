# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Tests for the CrisisWatch refresh script's Wayback acquisition path.

Pure unit tests — no network, no DB.  The fixture
``tests/fixtures/crisiswatch_wayback_sample.html`` is a trimmed but
structurally faithful CrisisWatch page (April 2026 edition, 3 entries)
padded past the 20 KB snapshot-validation threshold.
"""

from __future__ import annotations

import gzip
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

import scripts.refresh_crisiswatch as rc
from horizon_scanner.crisiswatch import _resolve_iso3

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "crisiswatch_wayback_sample.html"


@pytest.fixture()
def fixture_html() -> str:
    return FIXTURE_PATH.read_text(encoding="utf-8")


class _StubResponse:
    def __init__(self, *, content: bytes = b"", json_data=None):
        self.content = content
        self._json_data = json_data
        self.status_code = 200
        self.headers: dict[str, str] = {}

    def json(self):
        if self._json_data is None:
            raise ValueError("no json")
        return self._json_data


# ---------------------------------------------------------------------------
# 1-2: parsing the fixture with the existing (unchanged) parser
# ---------------------------------------------------------------------------


def test_parse_fixture_overview(fixture_html):
    soup = BeautifulSoup(fixture_html, "html.parser")
    overview = rc._parse_overview_section(soup)

    assert overview["report_month"] == "April 2026"
    assert overview["outlook_month"] == "May 2026"
    assert overview["conflict_risk_alerts"] == ["KWT", "YEM"]
    assert overview["resolution_opportunities"] == ["IRN"]
    assert overview["deteriorated"] == ["COL", "MLI"]
    assert overview["improved"] == ["IRN"]


def test_parse_fixture_entries(fixture_html):
    soup = BeautifulSoup(fixture_html, "html.parser")
    entries, month, year = rc._parse_country_entries(soup)

    assert month == "April 2026"
    assert year == 2026
    assert len(entries) == 3

    by_iso3 = {e["iso3"]: e for e in entries}
    assert set(by_iso3) == {"SOM", "KWT", "COL"}

    assert by_iso3["SOM"]["arrow"] == "unchanged"
    assert by_iso3["SOM"]["alert_type"] == ""
    assert by_iso3["SOM"]["summary"].startswith(
        "Security forces sustained pressure"
    )

    assert by_iso3["KWT"]["arrow"] == "deteriorated"
    assert by_iso3["KWT"]["alert_type"] == "conflict_risk"

    assert by_iso3["COL"]["arrow"] == "improved"

    # The "past entries" database link paragraph must not leak into
    # summaries.
    for e in entries:
        assert "past entries" not in e["summary"]


# ---------------------------------------------------------------------------
# 3: snapshot validation
# ---------------------------------------------------------------------------


def test_validate_rejects_challenge_page(fixture_html):
    # Cloudflare challenge pages are ~6 KB with zero entry divs.
    challenge = "<html><title>Just a moment...</title></html>"
    reason = rc._validate_snapshot_html(challenge)
    assert reason is not None and "too small" in reason

    # Large enough but no entry divs (e.g. a redesigned or error page).
    big_empty = "x" * 30_000
    reason = rc._validate_snapshot_html(big_empty)
    assert reason is not None and "entry divs" in reason

    # The fixture passes at default thresholds except entry count (it has
    # only 3 entries), so exercise both threshold parameters.
    assert rc._validate_snapshot_html(fixture_html, min_entries=2) is None
    assert rc._validate_snapshot_html(fixture_html) is not None


# ---------------------------------------------------------------------------
# 4: gzip sniffing
# ---------------------------------------------------------------------------


def test_gzip_sniff(fixture_html, monkeypatch):
    payloads = {
        "single": gzip.compress(fixture_html.encode("utf-8")),
        "double": gzip.compress(gzip.compress(fixture_html.encode("utf-8"))),
        "plain": fixture_html.encode("utf-8"),
    }

    for label, payload in payloads.items():
        monkeypatch.setattr(
            rc, "_get_with_retries",
            lambda url, **kw: _StubResponse(content=payload),
        )
        monkeypatch.setattr(rc, "_MIN_ENTRY_COUNT", 2)
        html = rc._fetch_snapshot_html("20260601184337")
        assert html is not None, f"{label} payload should decode"
        assert "c-crisiswatch-entry" in html


# ---------------------------------------------------------------------------
# 5: CDX walk-back over an invalid newest snapshot
# ---------------------------------------------------------------------------


def test_cdx_walkback(fixture_html, monkeypatch):
    challenge_bytes = b"<html><title>Just a moment...</title></html>"
    cdx_rows = [
        ["timestamp", "statuscode", "mimetype"],  # header row must be skipped
        ["20260601184337", "200", "text/html"],
        ["20260626094355", "200", "text/html"],  # newer but challenged
        ["20260520000000", "301", "text/html"],  # non-200 must be dropped
    ]

    def fake_get(url, *, params=None, **kw):
        if url == rc._WAYBACK_CDX_URL:
            return _StubResponse(json_data=cdx_rows)
        if "20260626094355" in url:
            return _StubResponse(content=challenge_bytes)
        if "20260601184337" in url:
            return _StubResponse(content=fixture_html.encode("utf-8"))
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(rc, "_get_with_retries", fake_get)
    monkeypatch.setattr(rc, "_MIN_ENTRY_COUNT", 2)

    html, timestamp = rc._fetch_wayback_html(spn=False)
    # Newest (June 26) is a challenge page -> walked back to June 1.
    assert timestamp == "20260601184337"
    assert "c-crisiswatch-entry" in html


def test_wayback_exits_when_no_valid_snapshot(monkeypatch):
    def fake_get(url, *, params=None, **kw):
        if url == rc._WAYBACK_CDX_URL:
            return _StubResponse(
                json_data=[["timestamp", "statuscode", "mimetype"]],
            )
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(rc, "_get_with_retries", fake_get)
    with pytest.raises(SystemExit):
        rc._fetch_wayback_html(spn=False)


# ---------------------------------------------------------------------------
# 6: --only-if-newer gate in run()
# ---------------------------------------------------------------------------


def _seed(output_path: Path, month: str, year: int) -> None:
    output_path.write_text(
        json.dumps({"month": month, "year": year, "entries": []}),
        encoding="utf-8",
    )


def _run_wayback(output_path: Path, fixture_html: str, monkeypatch, **kw):
    monkeypatch.setattr(
        rc, "_fetch_wayback_html",
        lambda **_: (fixture_html, "20260601184337"),
    )
    return rc.run(
        output_path=output_path, source="wayback", **kw,
    )


def test_only_if_newer_skips_when_not_newer(
    tmp_path, fixture_html, monkeypatch,
):
    out = tmp_path / "cw.json"
    _seed(out, "May 2026", 2026)
    before = out.read_text(encoding="utf-8")

    result = _run_wayback(out, fixture_html, monkeypatch, only_if_newer=True)

    assert result == {}
    assert out.read_text(encoding="utf-8") == before


def test_only_if_newer_writes_when_newer(tmp_path, fixture_html, monkeypatch):
    out = tmp_path / "cw.json"
    _seed(out, "February 2026", 2026)

    result = _run_wayback(out, fixture_html, monkeypatch, only_if_newer=True)

    assert result["month"] == "April 2026"
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["month"] == "April 2026"
    assert written["source"] == "wayback:20260601184337"
    assert len(written["entries"]) == 3


def test_only_if_newer_force_overrides(tmp_path, fixture_html, monkeypatch):
    out = tmp_path / "cw.json"
    _seed(out, "May 2026", 2026)

    result = _run_wayback(
        out, fixture_html, monkeypatch, only_if_newer=True, force=True,
    )

    assert result["month"] == "April 2026"
    assert json.loads(out.read_text(encoding="utf-8"))["month"] == "April 2026"


def test_only_if_newer_writes_when_no_existing_file(
    tmp_path, fixture_html, monkeypatch,
):
    out = tmp_path / "cw.json"

    result = _run_wayback(out, fixture_html, monkeypatch, only_if_newer=True)

    assert result["month"] == "April 2026"
    assert out.exists()


# ---------------------------------------------------------------------------
# 7: edition key + staleness alarm
# ---------------------------------------------------------------------------


def test_edition_key():
    assert rc._edition_key("April 2026", 2026) == (2026, 4)
    assert rc._edition_key("Aug 2025", 0) == (2025, 8)
    assert rc._edition_key("outlook for August 2025", 0) == (2025, 8)
    assert rc._edition_key("garbage", 0) is None
    assert rc._edition_key("", 2026) is None


def test_check_staleness():
    april = (2026, 4)  # ends 2026-04-30

    # 46 days after month end: fine at a 60-day threshold.
    rc._check_staleness(
        april, 60, now=datetime(2026, 6, 15, tzinfo=timezone.utc),
    )

    # 62 days after month end: alarm.
    with pytest.raises(SystemExit):
        rc._check_staleness(
            april, 60, now=datetime(2026, 7, 1, tzinfo=timezone.utc),
        )

    # Disabled or unknown edition: never raises.
    rc._check_staleness(april, 0, now=datetime(2030, 1, 1, tzinfo=timezone.utc))
    rc._check_staleness(None, 60)


# ---------------------------------------------------------------------------
# 8: newly added Gulf-state ISO3 mappings
# ---------------------------------------------------------------------------


def test_gulf_states_resolve():
    assert _resolve_iso3("Kuwait") == "KWT"
    assert _resolve_iso3("Oman") == "OMN"
    assert _resolve_iso3("Qatar") == "QAT"
