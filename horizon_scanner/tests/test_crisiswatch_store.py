# Pythia / Copyright (c) 2025 Kevin Wyjad
"""The CrisisWatch DB writer is the one writer, and it is change-aware.

Group H of the run-33841370196 repairs. The refresh script logged
"Candidate edition 2026-08 is not newer than existing 2026-08 — skipping
write" and 78 rows landed in ``crisiswatch_entries`` with a fresh
``fetched_at`` regardless, because the DB writer was ``INSERT OR REPLACE``
for every entry on every run. And 85 entries were parsed against 78 rows
stored with no line for the other seven.
"""

from __future__ import annotations

import json

import pytest

from horizon_scanner import crisiswatch as cw

duckdb = pytest.importorskip("duckdb")


@pytest.fixture()
def db(tmp_path, monkeypatch):
    path = tmp_path / "pythia.duckdb"
    monkeypatch.setenv("PYTHIA_DB_URL", f"duckdb:///{path}")
    return path


def _entries(**overrides):
    base = {
        "ETH": {"country": "Ethiopia", "iso3": "ETH", "arrow": "deteriorated",
                "alert_type": "conflict_risk", "summary": "Drone strikes hit Mekelle.",
                "month": "August 2026", "year": 2026},
        "SOM": {"country": "Somalia", "iso3": "SOM", "arrow": "unchanged",
                "alert_type": "", "summary": "Al-Shabaab attacks continued.",
                "month": "August 2026", "year": 2026},
    }
    for iso3, changes in overrides.items():
        base[iso3] = {**base[iso3], **changes}
    return base


def _rows(db):
    from pythia.db.schema import connect

    con = connect(read_only=False)
    try:
        return {
            row[0]: row[1:]
            for row in con.execute(
                "SELECT iso3, arrow, summary, fetched_at, content_hash "
                "FROM crisiswatch_entries ORDER BY iso3"
            ).fetchall()
        }
    finally:
        con.close()


def test_an_unchanged_republication_leaves_rows_and_fetched_at_alone(db):
    first = cw.store_crisiswatch_entries(_entries())
    assert (first["inserted"], first["updated"], first["unchanged"]) == (2, 0, 0)
    before = _rows(db)

    second = cw.store_crisiswatch_entries(_entries())
    assert (second["inserted"], second["updated"], second["unchanged"]) == (0, 0, 2)
    assert second["rows_for_edition"] == 2
    after = _rows(db)
    assert after == before, "an identical edition must not move fetched_at"


def test_a_revised_entry_is_rewritten_and_only_that_row_moves(db):
    cw.store_crisiswatch_entries(_entries())
    before = _rows(db)

    revised = _entries(ETH={"summary": "Drone strikes hit Mekelle; ceasefire talks collapsed."})
    counts = cw.store_crisiswatch_entries(revised)
    assert (counts["inserted"], counts["updated"], counts["unchanged"]) == (0, 1, 1)
    after = _rows(db)
    assert after["SOM"] == before["SOM"]
    assert after["ETH"][1].endswith("collapsed.")
    assert after["ETH"][3] != before["ETH"][3]  # content hash moved
    assert after["ETH"][2] >= before["ETH"][2]  # fetched_at moved with it


def test_an_entry_with_no_edition_month_is_skipped_and_counted(db):
    entries = _entries(SOM={"month": "", "year": 0})
    counts = cw.store_crisiswatch_entries(entries)
    assert counts["skipped_no_month"] == 1
    assert counts["inserted"] == 1
    assert set(_rows(db)) == {"ETH"}


def test_the_content_hash_covers_what_a_row_carries():
    a = {"arrow": "deteriorated", "alert_type": "", "summary": "x", "country": "A"}
    b = {**a, "fetched_at": "2026-09-05T00:00:00", "month": "August 2026"}
    assert cw.entry_content_hash(a) == cw.entry_content_hash(b)
    assert cw.entry_content_hash({**a, "arrow": "improved"}) != cw.entry_content_hash(a)


def test_every_parsed_entry_carries_a_reason(tmp_path, monkeypatch):
    """85 parsed and 78 stored must add up, entry by entry."""

    payload = {
        "month": "August 2026", "year": 2026, "fetched_at": "2026-09-04T00:00:00+00:00",
        "entries": [
            {"country": "Ethiopia", "iso3": "ETH", "arrow": "deteriorated",
             "alert_type": "", "summary": "own entry", "regional_source": ""},
            {"country": "Ethiopia", "iso3": "ETH", "arrow": "unchanged",
             "alert_type": "", "summary": "Nile waters", "regional_source": "Nile Waters"},
            {"country": "Somalia", "iso3": "SOM", "arrow": "unchanged",
             "alert_type": "", "summary": "own", "regional_source": ""},
            {"country": "Somaliland", "iso3": "SOM", "arrow": "unchanged",
             "alert_type": "", "summary": "territory", "regional_source": ""},
            {"country": "Atlantis Federation", "iso3": "", "arrow": "unchanged",
             "alert_type": "", "summary": "nowhere", "regional_source": ""},
        ],
    }
    path = tmp_path / "crisiswatch_latest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(cw, "_FALLBACK_PATH", path)
    monkeypatch.setattr(cw, "_resolve_iso3", lambda name: None)

    loaded = cw._load_from_json()
    assert set(loaded) == {"ETH", "SOM"}
    accounting = cw.load_accounting()
    assert accounting["parsed"] == 5
    assert accounting["loaded"] == 2
    assert accounting["reasons"] == {
        "stored": 2, "merged_into_ETH": 1, "merged_into_SOM": 1, "unresolved_iso3": 1,
    }
    assert sum(accounting["reasons"].values()) == accounting["parsed"]
    by_country = {e["country"]: e["reason"] for e in accounting["not_stored"]}
    assert by_country == {
        "Ethiopia": "merged_into_ETH", "Somaliland": "merged_into_SOM",
        "Atlantis Federation": "unresolved_iso3",
    }


def test_bulk_store_records_the_accounting_for_the_bundle(db, tmp_path, monkeypatch):
    payload = {
        "month": "August 2026", "year": 2026, "fetched_at": "2026-09-04T00:00:00+00:00",
        "entries": [
            {"country": "Ethiopia", "iso3": "ETH", "arrow": "deteriorated",
             "alert_type": "", "summary": "own entry", "regional_source": ""},
            {"country": "Nowhere", "iso3": "", "arrow": "unchanged",
             "alert_type": "", "summary": "x", "regional_source": ""},
        ],
    }
    path = tmp_path / "crisiswatch_latest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(cw, "_FALLBACK_PATH", path)
    monkeypatch.setattr(cw, "_resolve_iso3", lambda name: None)
    log_dir = tmp_path / "run_log"
    monkeypatch.setenv("PYTHIA_RUN_LOG_DIR", str(log_dir))
    from resolver.diagnostics import run_log

    run_log.reset_for_tests()

    n = cw.bulk_store_crisiswatch()
    assert n == 1
    records = list(run_log.read_stream(log_dir / f"{cw.STORE_ACCOUNTING_STREAM}.jsonl"))
    assert len(records) == 1
    rec = records[0]
    assert rec["parsed"] == 2 and rec["loaded"] == 1
    assert rec["inserted"] == 1 and rec["rows_for_edition"] == 1
    assert rec["load_reasons"] == {"stored": 1, "unresolved_iso3": 1}
    assert rec["edition"] == "August 2026"
