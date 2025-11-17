"""Freeze snapshot DuckDB routing tests."""

from pathlib import Path

import pandas as pd
import pytest

from resolver.db import duckdb_io
from resolver.tools import freeze_snapshot


@pytest.mark.duckdb
def test_preview_routing_preserves_parity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("duckdb")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RESOLVER_WRITE_DB", "0")

    data = pd.DataFrame(
        [
            {
                "iso3": "AAA",
                "ym": "2024-01",
                "event_id": "evt-001",
                "metric": "in_need",
                "value": 100,
                "series_semantics": "stock",
            },
            {
                "iso3": "BBB",
                "ym": "2024-01",
                "event_id": "evt-002",
                "metric": "affected",
                "value": 50,
                "series_semantics": "new",
            },
        ]
    )

    facts_path = tmp_path / "facts.csv"
    data.to_csv(facts_path, index=False)

    db_path = tmp_path / "preview.duckdb"
    db_url = f"duckdb:///{db_path.as_posix()}"

    freeze_snapshot.freeze_snapshot(
        facts=facts_path,
        month="2024-01",
        outdir=tmp_path / "snapshots",
        overwrite=True,
        write_db=True,
        db_url=db_url,
    )

    conn = duckdb_io.get_db(db_url)
    try:
        rows = conn.execute(
            """
            SELECT event_id, metric, series_semantics
            FROM facts_resolved
            WHERE ym = '2024-01'
            ORDER BY event_id
            """
        ).fetchall()
    finally:
        duckdb_io.close_db(conn)

    assert len(rows) == len(data)
    assert [row[0] for row in rows] == ["evt-001", "evt-002"]


@pytest.mark.duckdb
def test_emdat_flow_override_routes_all_to_deltas(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("duckdb")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RESOLVER_WRITE_DB", "0")

    rows = []
    for idx in range(4):
        rows.append(
            {
                "iso3": f"CCC",
                "ym": "2024-02",
                "event_id": f"emdat-{idx}",
                "metric": "affected",
                "value": 10 * (idx + 1),
                "publisher": "CRED / UCLouvain (EM-DAT)",
                "series_semantics": "",
            }
        )

    facts_path = tmp_path / "emdat_facts.csv"
    pd.DataFrame(rows).to_csv(facts_path, index=False)

    db_path = tmp_path / "emdat.duckdb"
    db_url = f"duckdb:///{db_path.as_posix()}"

    freeze_snapshot.freeze_snapshot(
        facts=facts_path,
        month="2024-02",
        outdir=tmp_path / "snapshots",
        overwrite=True,
        write_db=True,
        db_url=db_url,
    )

    conn = duckdb_io.get_db(db_url)
    try:
        resolved_count = conn.execute(
            "SELECT COUNT(*) FROM facts_resolved WHERE ym = '2024-02'"
        ).fetchone()[0]
        deltas_rows = conn.execute(
            "SELECT event_id, series_semantics FROM facts_deltas WHERE ym = '2024-02' ORDER BY event_id"
        ).fetchall()
    finally:
        duckdb_io.close_db(conn)

    assert resolved_count == 0
    assert len(deltas_rows) == 4
    assert all(sem == "new" for _, sem in deltas_rows)


@pytest.mark.duckdb
def test_preview_only_routes_rows_to_deltas(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("duckdb")

    monkeypatch.chdir(tmp_path)

    rows = [
        {
            "iso3": "ZZZ",
            "ym": "",
            "hazard_code": "TC",
            "metric": "affected",
            "value": 5,
            "series_semantics": "",
            "as_of_date": "2024-03-01",
        },
        {
            "iso3": "ZZZ",
            "ym": "",
            "hazard_code": "TC",
            "metric": "in_need",
            "value": 7,
            "series_semantics": "new",
            "as_of_date": "2024-03-01",
        },
    ]

    facts_path = tmp_path / "preview_only.csv"
    pd.DataFrame(rows).to_csv(facts_path, index=False)

    db_path = tmp_path / "preview_only.duckdb"
    db_url = f"duckdb:///{db_path.as_posix()}"

    freeze_snapshot._maybe_write_db(
        facts_path=facts_path,
        resolved_path=None,
        deltas_path=None,
        manifest_path=None,
        month="2024-03",
        db_url=db_url,
        write_db=True,
    )

    conn = duckdb_io.get_db(db_url)
    try:
        resolved_count = conn.execute(
            "SELECT COUNT(*) FROM facts_resolved WHERE ym = '2024-03'"
        ).fetchone()[0]
        deltas_count = conn.execute(
            "SELECT COUNT(*) FROM facts_deltas WHERE ym = '2024-03'"
        ).fetchone()[0]
    finally:
        duckdb_io.close_db(conn)

    assert resolved_count == 0
    assert deltas_count == len(rows)
