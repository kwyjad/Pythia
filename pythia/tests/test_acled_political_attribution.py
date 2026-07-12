# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Country-attribution tests for the ACLED political events connector.

Regression coverage for the July 2026 contamination bug: the fetcher passed
``iso3`` as an unsupported ACLED filter param, received the same global
events for every country, and stored them stamped with the requested iso3 —
so Somalia's SPD prompts contained Iranian/Ecuadorian/Mozambican events
(11,592 rows = the identical 46 events under 252 countries).
"""

from __future__ import annotations

import pytest

pytest.importorskip("duckdb")

from pythia.acled_political import (
    _filter_events_to_country,
    purge_contaminated_events,
    store_acled_political_events,
)


MIXED_EVENTS = [
    {"event_id_cnty": "SOM1", "iso3": "SOM", "event_type": "Riots", "fatalities": 3},
    {"event_id_cnty": "IRN1", "iso3": "IRN", "event_type": "Protests", "fatalities": 0},
    {"event_id_cnty": "ECU1", "iso3": "ECU", "event_type": "Strategic developments", "fatalities": 0},
    {"event_id_cnty": "SOM2", "iso3": "som", "event_type": "Protests", "fatalities": 1},
    {"event_id_cnty": "NOISO", "event_type": "Riots", "fatalities": 2},
]


def test_filter_keeps_only_matching_iso3():
    kept = _filter_events_to_country(MIXED_EVENTS, "SOM")
    assert [e["event_id_cnty"] for e in kept] == ["SOM1", "SOM2"]


def test_filter_returns_empty_when_nothing_matches():
    kept = _filter_events_to_country(MIXED_EVENTS, "KEN")
    assert kept == []


# The live ACLED API does not reliably return the code under the literal key
# ``iso3`` — it may use the HXL tag ``#country+code``, the underscore variant
# ``country_iso3``, or only the ``country`` name. The July-2026 contamination
# fix read a single ``ev.get("iso3")`` key, so in production every event
# resolved to "" and was discarded (5,257 events / 182 countries, 0 stored).
# These fixtures pin the real response shapes so the guard can't silently
# over-reject again.
REAL_SHAPE_EVENTS = [
    {"event_id_cnty": "NGA1", "#country+code": "NGA", "event_type": "Riots"},
    {"event_id_cnty": "NGA2", "country_iso3": "NGA", "event_type": "Protests"},
    {"event_id_cnty": "NGA3", "country": "Nigeria", "event_type": "Protests"},
    {"event_id_cnty": "RUS1", "country": "Russia", "event_type": "Protests"},
    {"event_id_cnty": "IRN1", "iso3": "IRN", "event_type": "Protests"},
]


def test_filter_resolves_iso3_from_alternate_keys():
    """HXL tag / underscore variant / country name all attribute correctly."""
    kept = _filter_events_to_country(REAL_SHAPE_EVENTS, "NGA")
    assert [e["event_id_cnty"] for e in kept] == ["NGA1", "NGA2", "NGA3"]


def test_filter_resolves_iso3_from_country_name_only():
    kept = _filter_events_to_country(REAL_SHAPE_EVENTS, "RUS")
    assert [e["event_id_cnty"] for e in kept] == ["RUS1"]


def test_store_guard_drops_misattributed_events(tmp_path, monkeypatch):
    db_path = tmp_path / "acled_pol.duckdb"
    monkeypatch.setenv("PYTHIA_DB_URL", f"duckdb:///{db_path}")

    events = [
        {"event_id": "SOM1", "iso3": "SOM", "event_date": "2026-06-01",
         "event_type": "Riots", "sub_event_type": "", "actor1": "", "actor2": "",
         "admin1": "", "location": "", "notes_excerpt": "", "fatalities": 3},
        # Wrong-country event must be dropped by the store guard.
        {"event_id": "IRN1", "iso3": "IRN", "event_date": "2026-06-02",
         "event_type": "Protests", "sub_event_type": "", "actor1": "", "actor2": "",
         "admin1": "", "location": "", "notes_excerpt": "", "fatalities": 0},
        # Legacy dicts without iso3 are trusted (pre-attribution shape).
        {"event_id": "SOM2", "event_date": "2026-06-03",
         "event_type": "Protests", "sub_event_type": "", "actor1": "", "actor2": "",
         "admin1": "", "location": "", "notes_excerpt": "", "fatalities": 1},
    ]
    store_acled_political_events("SOM", events)

    from pythia.db.schema import connect

    con = connect(read_only=False)
    try:
        rows = con.execute(
            "SELECT event_id FROM acled_political_events ORDER BY event_id"
        ).fetchall()
    finally:
        con.close()
    assert [r[0] for r in rows] == ["SOM1", "SOM2"]


def test_purge_detects_and_wipes_contamination(tmp_path, monkeypatch):
    db_path = tmp_path / "acled_pol_purge.duckdb"
    monkeypatch.setenv("PYTHIA_DB_URL", f"duckdb:///{db_path}")

    from pythia.db.schema import connect, ensure_schema

    ensure_schema()
    con = connect(read_only=False)
    try:
        # Same event_id under two countries = the contamination signature.
        for iso3 in ("SOM", "IRN"):
            con.execute(
                """
                INSERT OR REPLACE INTO acled_political_events
                    (iso3, event_id, event_date, event_type, sub_event_type,
                     actor1, actor2, admin1, location, notes_excerpt,
                     fatalities, fetched_at)
                VALUES (?, 'GLOBAL1', '2026-06-01', 'Riots', '', '', '', '', '', '', 0, '')
                """,
                [iso3],
            )
    finally:
        con.close()

    assert purge_contaminated_events() == 2

    con = connect(read_only=False)
    try:
        n = con.execute("SELECT COUNT(*) FROM acled_political_events").fetchone()[0]
    finally:
        con.close()
    assert n == 0
    # Idempotent: second call is a no-op.
    assert purge_contaminated_events() == 0


def test_purge_noop_on_clean_table(tmp_path, monkeypatch):
    db_path = tmp_path / "acled_pol_clean.duckdb"
    monkeypatch.setenv("PYTHIA_DB_URL", f"duckdb:///{db_path}")

    from pythia.db.schema import connect, ensure_schema

    ensure_schema()
    con = connect(read_only=False)
    try:
        con.execute(
            """
            INSERT OR REPLACE INTO acled_political_events
                (iso3, event_id, event_date, event_type, sub_event_type,
                 actor1, actor2, admin1, location, notes_excerpt,
                 fatalities, fetched_at)
            VALUES ('SOM', 'SOM1', '2026-06-01', 'Riots', '', '', '', '', '', '', 0, '')
            """
        )
    finally:
        con.close()

    assert purge_contaminated_events() == 0
