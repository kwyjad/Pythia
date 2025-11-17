from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from resolver.db import duckdb_io

pytest.importorskip("duckdb")


@pytest.mark.duckdb
def test_event_id_used_for_upsert_keys(tmp_path: Path) -> None:
    db_path = tmp_path / "event_keys.duckdb"
    db_url = f"duckdb:///{db_path.as_posix()}"

    resolved = pd.DataFrame(
        [
            {
                "event_id": "evt-1",
                "iso3": "AAA",
                "hazard_code": "TC",
                "metric": "in_need",
                "value": 10,
                "series_semantics": "stock",
                "as_of_date": "2024-01-05",
                "publication_date": "2024-01-07",
                "ym": "2024-01",
            },
            {
                "event_id": "evt-2",
                "iso3": "AAA",
                "hazard_code": "TC",
                "metric": "in_need",
                "value": 25,
                "series_semantics": "stock",
                "as_of_date": "2024-01-10",
                "publication_date": "2024-01-12",
                "ym": "2024-01",
            },
        ]
    )

    deltas = pd.DataFrame(
        [
            {
                "event_id": "flow-1",
                "iso3": "AAA",
                "hazard_code": "TC",
                "metric": "in_need",
                "value_new": 5,
                "series_semantics": "new",
                "as_of": "2024-01-15",
                "ym": "2024-01",
            },
            {
                "event_id": "flow-2",
                "iso3": "AAA",
                "hazard_code": "TC",
                "metric": "in_need",
                "value_new": 7,
                "series_semantics": "new",
                "as_of": "2024-01-20",
                "ym": "2024-01",
            },
        ]
    )

    conn = duckdb_io.get_db(db_url)
    try:
        duckdb_io.write_snapshot(
            conn,
            ym="2024-01",
            facts_resolved=resolved,
            facts_deltas=deltas,
            manifests=None,
            meta=None,
        )
        resolved_rows = conn.execute(
            "SELECT event_id, value FROM facts_resolved WHERE ym = '2024-01' ORDER BY event_id"
        ).fetchall()
        deltas_rows = conn.execute(
            "SELECT event_id, value_new FROM facts_deltas WHERE ym = '2024-01' ORDER BY event_id"
        ).fetchall()
    finally:
        duckdb_io.close_db(conn)

    assert [row[0] for row in resolved_rows] == ["evt-1", "evt-2"]
    assert [row[1] for row in resolved_rows] == [10, 25]
    assert [row[0] for row in deltas_rows] == ["flow-1", "flow-2"]
    assert [row[1] for row in deltas_rows] == [5, 7]
    assert len(resolved_rows) == 2
    assert len(deltas_rows) == 2
