# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Within a country-month, the LATEST revision must win, not the first report.

``IFRCAdapter`` snaps ``as_of_date`` to month-end, so every IFRC GO record for a
country-hazard-metric-month lands in a single ``resolve_sources`` dedup group.
The sort was ascending on every column with ``keep="first"``, so the winner was
the row with the earliest ``publication_date`` — the initial field report, filed
when least was known, beating every later revision of the same disaster.

Field reports are progressive: a flood reported at 5,000 affected on day two is
routinely revised to 80,000 a fortnight later. Keeping the first one is a
systematic downward bias on PA ground truth, and it is silent.

resolver/tools/precedence_config.yml already declares
``tiebreak: [as_of_desc, value_nonnull, source_alpha]``; this pins the other
precedence implementation to the same policy.
"""

from __future__ import annotations

import pandas as pd
import pytest

duckdb = pytest.importorskip("duckdb")

from resolver.transform.resolve_sources import resolve_sources


_COLUMNS = """
    ym TEXT, iso3 TEXT, hazard_code TEXT, hazard_label TEXT, hazard_class TEXT,
    metric TEXT, series_semantics TEXT, value DOUBLE, unit TEXT,
    as_of DATE, as_of_date VARCHAR, publication_date VARCHAR,
    publisher TEXT, source_id TEXT, source_type TEXT, source_url TEXT,
    doc_title TEXT, definition_text TEXT, precedence_tier TEXT, event_id TEXT,
    proxy_for TEXT, confidence TEXT, series TEXT,
    provenance_source TEXT, provenance_rank INTEGER
"""


def _row(*, value, publication_date, event_id, source_id="ifrc_go"):
    return {
        "ym": "2024-03",
        "iso3": "PHL",
        "hazard_code": "TC",
        "hazard_label": "Tropical Cyclone",
        "hazard_class": "natural",
        "metric": "affected",
        "series_semantics": "stock",
        "value": value,
        "unit": "persons",
        # Month-end, as IFRCAdapter snaps it — this is what puts every record
        # for the month into one dedup group.
        "as_of": "2024-03-31",
        "as_of_date": "2024-03-31",
        "publication_date": publication_date,
        "publisher": "IFRC",
        "source_id": source_id,
        "source_type": None,
        "source_url": None,
        "doc_title": None,
        "definition_text": None,
        "precedence_tier": None,
        "event_id": event_id,
        "proxy_for": None,
        "confidence": None,
        "series": None,
        "provenance_source": None,
        "provenance_rank": None,
    }


def _resolve(tmp_path, rows):
    conn = duckdb.connect(str(tmp_path / "resolver.duckdb"))
    try:
        conn.execute(f"CREATE TABLE facts_resolved ({_COLUMNS})")
        conn.register("seed", pd.DataFrame(rows))
        conn.execute("INSERT INTO facts_resolved SELECT * FROM seed")
        conn.unregister("seed")
        resolve_sources(conn, sources=["ifrc_go"])
        return conn.execute(
            "SELECT value, publication_date, event_id FROM facts_resolved"
        ).fetchall()
    finally:
        conn.close()


def test_latest_revision_wins_over_the_first_field_report(tmp_path):
    rows = [
        _row(value=5_000.0, publication_date="2024-03-04", event_id="fr-first"),
        _row(value=80_000.0, publication_date="2024-03-19", event_id="fr-revised"),
    ]

    result = _resolve(tmp_path, rows)

    assert len(result) == 1, "expected the country-month to collapse to one row"
    value, pub_date, event_id = result[0]
    assert event_id == "fr-revised", (
        "resolve_sources kept the earliest field report over a later revision. "
        "That is the silent downward bias on PA ground truth this test exists "
        "to prevent."
    )
    assert value == pytest.approx(80_000.0)
    assert pub_date == "2024-03-19"


def test_seed_order_does_not_decide_the_winner(tmp_path):
    """Same rows, reversed insertion order — the winner must not move."""
    rows = [
        _row(value=80_000.0, publication_date="2024-03-19", event_id="fr-revised"),
        _row(value=5_000.0, publication_date="2024-03-04", event_id="fr-first"),
    ]

    result = _resolve(tmp_path, rows)

    assert len(result) == 1
    assert result[0][2] == "fr-revised"


def test_a_null_value_never_beats_a_real_one(tmp_path):
    """A later row carrying no value must not displace an earlier real figure."""
    rows = [
        _row(value=42_000.0, publication_date="2024-03-10", event_id="fr-real"),
        _row(value=None, publication_date="2024-03-25", event_id="fr-empty"),
    ]

    result = _resolve(tmp_path, rows)

    assert len(result) == 1
    value, _pub, event_id = result[0]
    assert event_id == "fr-real", (
        "a null-valued row won on recency alone; value_nonnull must rank above "
        "as_of_desc, as precedence_config.yml declares"
    )
    assert value == pytest.approx(42_000.0)


def test_source_priority_still_outranks_recency(tmp_path):
    """Recency is a tiebreak WITHIN a source tier, never across tiers."""
    rows = [
        _row(value=10_000.0, publication_date="2024-03-02", event_id="ifrc",
             source_id="ifrc_go"),
        _row(value=99_000.0, publication_date="2024-03-28", event_id="idmc",
             source_id="IDMC"),
    ]

    conn = duckdb.connect(str(tmp_path / "resolver.duckdb"))
    try:
        conn.execute(f"CREATE TABLE facts_resolved ({_COLUMNS})")
        conn.register("seed", pd.DataFrame(rows))
        conn.execute("INSERT INTO facts_resolved SELECT * FROM seed")
        conn.unregister("seed")
        resolve_sources(conn, sources=["ifrc_go", "IDMC"])
        result = conn.execute("SELECT event_id FROM facts_resolved").fetchall()
    finally:
        conn.close()

    assert result == [("ifrc",)], (
        "the newer IDMC row displaced the higher-priority IFRC row; source "
        "priority must be applied before the recency tiebreak"
    )
