# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Several scraped CrisisWatch rows can land on one ISO3 — they are merged.

CrisisWatch publishes headings that are not a single country: regional entries
the scraper expands (Nile Waters -> ETH/SDN/EGY), bilateral headings that
resolve by name (China-U.S. -> CHN) and territories (Somaliland -> SOM).  The
loader used to build its ISO3-keyed dict by assignment, so the last row seen
won and everything the earlier rows knew was dropped.

Measured on the real August 2026 edition: Ethiopia's own entry said
``deteriorated``; the Nile Waters expansion that follows it said ``unchanged``
and overwrote it, so the loader reported 3 deteriorating countries where the
scraped file said 4 — and Ethiopia's deterioration reached no ACE prompt.
"""

from __future__ import annotations

import json

import pytest

from horizon_scanner import crisiswatch


def _rows(*rows: dict) -> list[dict]:
    base = {"arrow": "unchanged", "alert_type": "", "summary": "", "regional_source": ""}
    return [{**base, **r} for r in rows]


# ---------------------------------------------------------------------------
# The merge itself
# ---------------------------------------------------------------------------


def test_regional_row_never_erases_a_countrys_own_signal():
    """The August 2026 Ethiopia case, which is why this module exists."""
    merged = crisiswatch._merge_country_rows(
        "ETH",
        _rows(
            {"country": "Ethiopia", "arrow": "deteriorated", "summary": "Tigray clashes"},
            {"country": "Ethiopia", "arrow": "unchanged",
             "summary": "Dam dispute", "regional_source": "Nile Waters"},
        ),
    )
    assert merged["arrow"] == "deteriorated"
    assert merged["country"] == "Ethiopia"


def test_a_regional_signal_survives_a_quiet_own_entry():
    """The reverse direction: a deterioration is never dropped either."""
    merged = crisiswatch._merge_country_rows(
        "SDN",
        _rows(
            {"country": "Sudan", "arrow": "unchanged"},
            {"country": "Sudan", "arrow": "deteriorated", "regional_source": "Nile Waters"},
        ),
    )
    assert merged["arrow"] == "deteriorated"


@pytest.mark.parametrize(
    "arrows,expected",
    [
        (["unchanged", "improved"], "improved"),
        (["improved", "deteriorated"], "deteriorated"),
        (["", "unchanged"], "unchanged"),
        (["unchanged", "unchanged"], "unchanged"),
    ],
)
def test_arrow_takes_the_strongest_value(arrows, expected):
    merged = crisiswatch._merge_country_rows(
        "XXX", _rows(*({"country": "X", "arrow": a} for a in arrows))
    )
    assert merged["arrow"] == expected


def test_an_alert_on_any_row_survives():
    merged = crisiswatch._merge_country_rows(
        "EGY",
        _rows(
            {"country": "Egypt", "alert_type": ""},
            {"country": "Egypt", "alert_type": "conflict_risk",
             "regional_source": "Nile Waters"},
        ),
    )
    assert merged["alert_type"] == "conflict_risk"


def test_conflict_risk_outranks_resolution_opportunity():
    merged = crisiswatch._merge_country_rows(
        "XXX",
        _rows(
            {"country": "X", "alert_type": "resolution_opportunity"},
            {"country": "X", "alert_type": "conflict_risk"},
        ),
    )
    assert merged["alert_type"] == "conflict_risk"


def test_identity_comes_from_the_countrys_own_entry_not_the_regional_one():
    """Somaliland resolves to SOM and used to overwrite Somalia."""
    merged = crisiswatch._merge_country_rows(
        "SOM",
        _rows(
            {"country": "Somalia", "summary": "Own entry"},
            {"country": "Somaliland", "summary": "Territory entry"},
        ),
    )
    assert merged["country"] == "Somalia"
    assert merged["summary"].startswith("Own entry")


def test_own_entry_wins_identity_even_when_the_regional_row_comes_first():
    merged = crisiswatch._merge_country_rows(
        "ETH",
        _rows(
            {"country": "Ethiopia", "summary": "Dam dispute",
             "regional_source": "Nile Waters"},
            {"country": "Ethiopia", "summary": "Tigray clashes"},
        ),
    )
    assert merged["summary"].startswith("Tigray clashes")


def test_no_summary_is_silently_dropped():
    merged = crisiswatch._merge_country_rows(
        "CHN",
        _rows(
            {"country": "China-U.S.", "summary": "Trade tensions"},
            {"country": "China/Japan", "summary": "East China Sea"},
            {"country": "South China Sea", "summary": "Manila and Beijing"},
        ),
    )
    for text in ("Trade tensions", "East China Sea", "Manila and Beijing"):
        assert text in merged["summary"]


def test_merged_summary_is_bounded():
    merged = crisiswatch._merge_country_rows(
        "XXX",
        _rows(*({"country": f"C{i}", "summary": "x" * 480} for i in range(5))),
    )
    assert len(merged["summary"]) <= crisiswatch._MERGED_SUMMARY_MAX_CHARS + 16


def test_a_duplicate_summary_is_not_repeated():
    merged = crisiswatch._merge_country_rows(
        "XXX",
        _rows(
            {"country": "A", "summary": "Same text"},
            {"country": "B", "summary": "Same text"},
        ),
    )
    assert merged["summary"] == "Same text"


def test_a_self_label_is_dropped_on_a_legacy_unmarked_row():
    """An older JSON file carries no regional_source; the fallback label would
    otherwise be the country's own name, which says nothing."""
    rows = [
        {"country": "Ethiopia", "arrow": "deteriorated", "alert_type": "", "summary": "Own"},
        {"country": "Ethiopia", "arrow": "unchanged", "alert_type": "", "summary": "Regional"},
    ]
    merged = crisiswatch._merge_country_rows("ETH", rows)
    assert merged["summary"] == "Own | Regional"


def test_a_single_row_country_round_trips_unchanged():
    merged = crisiswatch._merge_country_rows(
        "KEN",
        _rows({"country": "Kenya", "arrow": "improved",
               "alert_type": "resolution_opportunity", "summary": "Talks"}),
    )
    assert merged == {
        "country": "Kenya",
        "iso3": "KEN",
        "arrow": "improved",
        "alert_type": "resolution_opportunity",
        "summary": "Talks",
    }


# ---------------------------------------------------------------------------
# End to end through the loader, which is what feeds prompts AND the DB
# ---------------------------------------------------------------------------


def _write(tmp_path, entries):
    path = tmp_path / "crisiswatch_latest.json"
    path.write_text(json.dumps({
        "month": "August 2026",
        "year": 2026,
        "fetched_at": "2026-09-03T08:36:03+00:00",
        "entries": entries,
    }), encoding="utf-8")
    return path


def test_loader_keeps_the_deterioration_count_the_scraper_reported(tmp_path, monkeypatch):
    entries = [
        {"country": "Afghanistan", "iso3": "AFG", "arrow": "deteriorated",
         "alert_type": "", "summary": "a", "regional_source": ""},
        {"country": "Ethiopia", "iso3": "ETH", "arrow": "deteriorated",
         "alert_type": "", "summary": "b", "regional_source": ""},
        {"country": "Ethiopia", "iso3": "ETH", "arrow": "unchanged",
         "alert_type": "", "summary": "c", "regional_source": "Nile Waters"},
        {"country": "Sudan", "iso3": "SDN", "arrow": "unchanged",
         "alert_type": "", "summary": "d", "regional_source": "Nile Waters"},
    ]
    monkeypatch.setattr(crisiswatch, "_FALLBACK_PATH", _write(tmp_path, entries))
    result = crisiswatch._load_from_json()

    scraped_deteriorated = {e["iso3"] for e in entries if e["arrow"] == "deteriorated"}
    loaded_deteriorated = {k for k, v in result.items() if v["arrow"] == "deteriorated"}
    assert loaded_deteriorated == scraped_deteriorated == {"AFG", "ETH"}


def test_loader_still_merges_a_legacy_file_with_no_provenance(tmp_path, monkeypatch):
    """Files written before the scraper stamped regional_source must not lose
    their signal either — document order is the fallback."""
    entries = [
        {"country": "Ethiopia", "iso3": "ETH", "arrow": "deteriorated",
         "alert_type": "", "summary": "own"},
        {"country": "Ethiopia", "iso3": "ETH", "arrow": "unchanged",
         "alert_type": "", "summary": "regional"},
    ]
    monkeypatch.setattr(crisiswatch, "_FALLBACK_PATH", _write(tmp_path, entries))
    result = crisiswatch._load_from_json()
    assert result["ETH"]["arrow"] == "deteriorated"


def test_loader_carries_edition_month_and_year_onto_every_entry(tmp_path, monkeypatch):
    entries = [
        {"country": "Kenya", "iso3": "KEN", "arrow": "unchanged",
         "alert_type": "", "summary": "s", "regional_source": ""},
    ]
    monkeypatch.setattr(crisiswatch, "_FALLBACK_PATH", _write(tmp_path, entries))
    result = crisiswatch._load_from_json()
    assert result["KEN"]["month"] == "August 2026"
    assert result["KEN"]["year"] == 2026
