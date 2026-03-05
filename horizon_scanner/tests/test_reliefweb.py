# Pythia / Copyright (c) 2025 Kevin Wyjad

"""Tests for the ReliefWeb data connector.

Validates :func:`fetch_reliefweb_reports` (live API),
:func:`format_reliefweb_for_prompt`, :func:`format_reliefweb_for_spd`,
and the HTML stripping helper.
"""

from __future__ import annotations

import json
from datetime import date, timedelta

import pytest

from horizon_scanner.reliefweb import (
    _strip_html,
    fetch_reliefweb_reports,
    format_reliefweb_for_prompt,
    format_reliefweb_for_spd,
)


# ---------------------------------------------------------------------------
# Sample data fixtures
# ---------------------------------------------------------------------------

def _sample_reports(n: int = 3) -> list[dict]:
    """Return *n* sample report dicts with realistic data."""
    base_date = date.today() - timedelta(days=5)
    reports = []
    for i in range(n):
        dt = (base_date - timedelta(days=i)).isoformat()
        reports.append({
            "report_id": 1000 + i,
            "iso3": "SDN",
            "title": f"Sudan: Humanitarian Update #{i + 1}",
            "published_date": f"{dt}T00:00:00+00:00",
            "sources": json.dumps(["OCHA", "UNHCR"]),
            "disaster_types": json.dumps(["Flood", "Epidemic"]),
            "themes": json.dumps(["Food and Nutrition", "Protection/Human Rights"]),
            "body_excerpt": f"Excerpt of report {i + 1} describing the humanitarian situation.",
            "url": f"https://reliefweb.int/report/{1000 + i}",
            "fetched_at": f"{date.today().isoformat()}T12:00:00+00:00",
        })
    return reports


def _stale_reports() -> list[dict]:
    """Return reports with dates older than the staleness threshold."""
    old_date = (date.today() - timedelta(days=90)).isoformat()
    return [{
        "report_id": 9999,
        "iso3": "SDN",
        "title": "Sudan: Old Situation Report",
        "published_date": f"{old_date}T00:00:00+00:00",
        "sources": json.dumps(["OCHA"]),
        "disaster_types": json.dumps(["Flood"]),
        "themes": json.dumps(["Shelter and Non-Food Items"]),
        "body_excerpt": "This is an old report.",
        "url": "https://reliefweb.int/report/9999",
        "fetched_at": f"{old_date}T12:00:00+00:00",
    }]


def _many_reports() -> list[dict]:
    """Return 13 sample reports (more than the 10-report display limit)."""
    return _sample_reports(n=13)


# ---------------------------------------------------------------------------
# _strip_html
# ---------------------------------------------------------------------------

class TestStripHtml:
    """Tests for the HTML stripping helper."""

    def test_removes_tags(self):
        assert _strip_html("<p>Hello <b>world</b></p>") == "Hello world"

    def test_plain_text_unchanged(self):
        assert _strip_html("no tags here") == "no tags here"

    def test_empty_string(self):
        assert _strip_html("") == ""

    def test_nested_tags(self):
        html = "<div><p>A <a href='#'>link</a> and <em>emphasis</em></p></div>"
        assert _strip_html(html) == "A link and emphasis"

    def test_br_tags(self):
        assert _strip_html("line1<br>line2<br/>line3") == "line1line2line3"


# ---------------------------------------------------------------------------
# fetch_reliefweb_reports (live API)
# ---------------------------------------------------------------------------

class TestFetchReliefwebReports:
    """Live API tests — require network access."""

    @pytest.mark.network
    def test_fetch_returns_list(self):
        reports = fetch_reliefweb_reports("SDN", days_back=30, max_reports=3)
        assert isinstance(reports, list)

    @pytest.mark.network
    def test_report_structure(self):
        reports = fetch_reliefweb_reports("SDN", days_back=30, max_reports=3)
        if not reports:
            pytest.skip("No reports returned — API may be down or no data")
        r = reports[0]
        expected_keys = {
            "report_id", "iso3", "title", "published_date", "sources",
            "disaster_types", "themes", "body_excerpt", "url", "fetched_at",
        }
        assert expected_keys == set(r.keys())

    @pytest.mark.network
    def test_body_excerpt_truncated(self):
        reports = fetch_reliefweb_reports("SDN", days_back=60, max_reports=5)
        for r in reports:
            assert len(r["body_excerpt"]) <= 600  # 500 + tolerance for word boundary

    @pytest.mark.network
    def test_iso3_uppercase(self):
        reports = fetch_reliefweb_reports("sdn", days_back=30, max_reports=2)
        for r in reports:
            assert r["iso3"] == "SDN"


# ---------------------------------------------------------------------------
# format_reliefweb_for_prompt
# ---------------------------------------------------------------------------

class TestFormatReliefwebForPrompt:
    """Tests for the RC/Triage prompt formatter."""

    def test_none_returns_empty_string(self):
        assert format_reliefweb_for_prompt(None) == ""

    def test_empty_list_returns_empty_string(self):
        assert format_reliefweb_for_prompt([]) == ""

    def test_contains_header(self):
        result = format_reliefweb_for_prompt(_sample_reports())
        assert "RECENT RELIEFWEB REPORTS" in result
        assert "SDN" in result

    def test_contains_report_count(self):
        result = format_reliefweb_for_prompt(_sample_reports())
        assert "3 reports found" in result

    def test_contains_themes(self):
        result = format_reliefweb_for_prompt(_sample_reports())
        assert "Food and Nutrition" in result

    def test_contains_numbered_entries(self):
        result = format_reliefweb_for_prompt(_sample_reports())
        assert "1. [" in result
        assert "2. [" in result
        assert "3. [" in result

    def test_contains_sources(self):
        result = format_reliefweb_for_prompt(_sample_reports())
        assert "OCHA" in result

    def test_contains_disaster_types(self):
        result = format_reliefweb_for_prompt(_sample_reports())
        assert "Flood" in result

    def test_contains_body_excerpt(self):
        result = format_reliefweb_for_prompt(_sample_reports())
        assert "Excerpt of report" in result

    def test_contains_interpretation_footer(self):
        result = format_reliefweb_for_prompt(_sample_reports())
        assert "situational context" in result

    def test_stale_data_contains_warning(self):
        result = format_reliefweb_for_prompt(_stale_reports())
        assert "WARNING" in result
        assert "STALE" in result

    def test_fresh_data_no_warning(self):
        result = format_reliefweb_for_prompt(_sample_reports())
        assert "WARNING" not in result

    def test_many_reports_shows_remainder(self):
        result = format_reliefweb_for_prompt(_many_reports())
        assert "+3 additional reports" in result

    def test_many_reports_limits_display(self):
        result = format_reliefweb_for_prompt(_many_reports())
        assert "10. [" in result
        assert "11. [" not in result


# ---------------------------------------------------------------------------
# format_reliefweb_for_spd
# ---------------------------------------------------------------------------

class TestFormatReliefwebForSpd:
    """Tests for the compact SPD prompt formatter."""

    def test_none_returns_empty_string(self):
        assert format_reliefweb_for_spd(None) == ""

    def test_empty_list_returns_empty_string(self):
        assert format_reliefweb_for_spd([]) == ""

    def test_contains_header(self):
        result = format_reliefweb_for_spd(_sample_reports())
        assert "RECENT REPORTS (SDN)" in result

    def test_compact_format(self):
        result = format_reliefweb_for_spd(_sample_reports())
        # Each entry is a single line starting with "- "
        lines = [l for l in result.split("\n") if l.startswith("- ")]
        assert len(lines) == 3

    def test_includes_disaster_types(self):
        result = format_reliefweb_for_spd(_sample_reports())
        assert "(Flood, Epidemic)" in result

    def test_stale_data_contains_warning(self):
        result = format_reliefweb_for_spd(_stale_reports())
        assert "WARNING" in result
        assert "STALE" in result
