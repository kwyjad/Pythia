# Pythia / Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.
"""CrisisWatch repairs from the review of the 1 October 2026 run.

* "Israel/Palestine" resolved to ISR alone, so PSE's two conflict questions
  ran with no CrisisWatch entry.
* ``_resolve_iso3`` fell back to a substring test that can put Niger inside
  Nigeria and Guinea inside Guinea-Bissau.
* The formatter printed nothing at all for a country absent from the newest
  edition or not covered by ICG: 33 of 120 conflict RC prompts.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from horizon_scanner import crisiswatch as cw

duckdb = pytest.importorskip("duckdb")

TODAY = datetime(2026, 10, 13, tzinfo=timezone.utc)


@pytest.fixture()
def db(tmp_path, monkeypatch):
    path = tmp_path / "pythia.duckdb"
    monkeypatch.setenv("PYTHIA_DB_URL", f"duckdb:///{path}")
    return path


# --------------------------------------------------------------- resolution


def test_israel_palestine_names_both_countries():
    members = dict((iso3, name) for name, iso3 in cw.MULTI_COUNTRY_HEADINGS["Israel/Palestine"])
    assert set(members) == {"ISR", "PSE"}
    # Not a single-country alias any more: resolving it to one code is
    # exactly the fault.
    assert cw._resolve_iso3("Israel/Palestine") is None
    assert cw.multi_country_heading("israel/palestine") == "Israel/Palestine"


@pytest.mark.parametrize(
    "heading",
    ["Niger Delta", "Guinea Gulf", "Mali Border", "Upper Sudan Region"],
)
def test_no_substring_guess(heading):
    # Each of these contains a mapped name and used to resolve by substring.
    assert cw._resolve_iso3(heading) is None
    assert heading in cw.unmapped_headings()


def test_explicit_aliases_still_resolve():
    assert cw._resolve_iso3("Niger") == "NER"
    assert cw._resolve_iso3("Nigeria") == "NGA"
    assert cw._resolve_iso3("Guinea-Bissau") == "GNB"
    assert cw._resolve_iso3("Russia (Internal)") == "RUS"
    assert cw._resolve_iso3("China-U.S.") == "CHN"
    assert cw._resolve_iso3("South China Sea") == "CHN"
    assert cw._resolve_iso3("Kosovo") == "RKS"
    assert cw._resolve_iso3("Northern Ireland (UK)") == "GBR"


def test_scraper_expands_israel_palestine(tmp_path):
    from scripts import refresh_crisiswatch as rc

    html = (
        '<div class="c-crisiswatch-entry" data-entry-country="israel-palestine">'
        '<h3><svg><use xlink:href="#deteriorated"></use></svg>Israel/Palestine</h3>'
        "<time>August 2026</time>"
        '<div class="o-crisis-states__detail"><p><strong>Strikes on Gaza.</strong></p></div>'
        "</div>"
    )
    from bs4 import BeautifulSoup

    entries, _month, _year = rc._parse_country_entries(BeautifulSoup(html, "html.parser"))
    assert {e["iso3"] for e in entries} == {"ISR", "PSE"}
    assert all(e["regional_source"] == "Israel/Palestine" for e in entries)


def test_a_legacy_json_file_gives_palestine_its_entry(tmp_path):
    # A file written before the fix: the scraper stamped ISR on the heading.
    path = tmp_path / "cw.json"
    path.write_text(json.dumps({
        "month": "August 2026", "year": 2026,
        "entries": [{"country": "Israel/Palestine", "iso3": "ISR", "arrow": "deteriorated",
                     "alert_type": "", "summary": "Strikes on Gaza.", "regional_source": ""}],
    }))
    data = cw._load_from_json(path)
    assert set(data) == {"ISR", "PSE"}
    assert data["PSE"]["country"] == "Israel/Palestine"
    assert data["PSE"]["arrow"] == "deteriorated"
    acc = cw.load_accounting()
    assert acc["parsed"] == sum(acc["reasons"].values())


def test_stored_heading_rows_are_repaired_once(db):
    from pythia.db.schema import connect, ensure_schema

    con = connect(read_only=False)
    ensure_schema(con)
    con.execute(
        "INSERT INTO crisiswatch_entries (iso3, year, month, arrow, alert_type, summary, "
        "country_name, content_hash) VALUES "
        "('ISR', 2026, 6, 'unchanged', '', 'June text', 'Israel/Palestine', 'h1'),"
        "('XKX', 2026, 6, 'unchanged', '', 'Kosovo text', 'Kosovo', 'h2')"
    )
    first = cw.repair_stored_headings(con)
    second = cw.repair_stored_headings(con)
    rows = dict(
        ((r[0], r[1]), r[2])
        for r in con.execute(
            "SELECT iso3, month, summary FROM crisiswatch_entries"
        ).fetchall()
    )
    con.close()
    assert first == {"heading_rows_added": 1, "iso3_renamed": 1}
    assert second == {"heading_rows_added": 0, "iso3_renamed": 0}
    assert rows[("PSE", 6)] == "June text"
    assert ("RKS", 6) in rows and ("XKX", 6) not in rows


# ---------------------------------------------------------------- the prompt

_EDITIONS = [(2026, m) for m in (1, 2, 3, 4, 6, 7, 8, 9)]


def _data():
    return {
        "PSE": {"country": "Israel/Palestine", "iso3": "PSE", "arrow": "deteriorated",
                "alert_type": "conflict_risk", "summary": "Strikes on Gaza.",
                "month": "September 2026", "year": 2026},
        "NIC": {"country": "Nicaragua", "iso3": "NIC", "arrow": "unchanged",
                "alert_type": "", "summary": "Repression continued.",
                "month": "June 2026", "year": 2026},
    }


def test_prompt_for_palestine_names_the_heading_and_the_edition():
    text = cw.format_crisiswatch_for_prompt("PSE", _data(), today=TODAY, editions=_EDITIONS)
    assert "newest edition held: September 2026" in text
    assert "Israel/Palestine (September 2026):" in text
    assert "Arrow: Deteriorated" in text
    assert "not listed" not in text


def test_prompt_for_a_country_absent_from_the_newest_edition():
    text = cw.format_crisiswatch_for_prompt("NIC", _data(), today=TODAY, editions=_EDITIONS)
    assert "Nicaragua is not listed in the September 2026 edition" in text
    assert "last entry is from June 2026" in text
    assert "Context: Repression continued." in text


def test_prompt_for_a_country_icg_does_not_cover(monkeypatch):
    monkeypatch.setattr(cw, "load_crisiswatch_for_country", lambda iso3: None)
    text = cw.format_crisiswatch_for_prompt("ZMB", _data(), today=TODAY, editions=_EDITIONS)
    assert text is not None
    assert "ICG does not cover Zambia" in text
    assert "8 edition(s) held (January 2026 to September 2026)" in text
    # One line beyond the header: nothing invented.
    assert len(text.splitlines()) == 2
