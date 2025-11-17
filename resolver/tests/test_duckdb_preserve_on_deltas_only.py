from pathlib import Path

import pandas as pd
import pytest

from resolver.db import duckdb_io

pytest.importorskip("duckdb")


@pytest.mark.duckdb
def test_deltas_only_write_preserves_resolved(tmp_path: Path) -> None:
    db_path = tmp_path / "preserve.duckdb"
    db_url = f"duckdb:///{db_path.as_posix()}"

    resolved = pd.DataFrame(
        [
            {
                "event_id": "evt-1",
                "iso3": "AAA",
                "hazard_code": "TC",
                "metric": "in_need",
                "value": 100,
                "as_of_date": "2024-01-01",
                "publication_date": "2024-01-02",
                "series_semantics": "stock",
                "ym": "2024-01",
            },
            {
                "event_id": "evt-2",
                "iso3": "AAA",
                "hazard_code": "EQ",
                "metric": "affected",
                "value": 50,
                "as_of_date": "2024-01-05",
                "publication_date": "2024-01-06",
                "series_semantics": "stock",
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
                "value_new": 25,
                "series_semantics": "new",
                "as_of": "2024-01-15",
                "ym": "2024-01",
            }
        ]
    )

    conn = duckdb_io.get_db(db_url)
    try:
        duckdb_io.write_snapshot(
            conn,
            ym="2024-01",
            facts_resolved=resolved,
            facts_deltas=None,
            manifests=None,
            meta=None,
        )
        duckdb_io.write_snapshot(
            conn,
            ym="2024-01",
            facts_resolved=None,
            facts_deltas=deltas,
            manifests=None,
            meta=None,
        )

        resolved_count = conn.execute(
            "SELECT COUNT(*) FROM facts_resolved WHERE ym = '2024-01'"
        ).fetchone()[0]
        deltas_count = conn.execute(
            "SELECT COUNT(*) FROM facts_deltas WHERE ym = '2024-01'"
        ).fetchone()[0]
    finally:
        duckdb_io.close_db(conn)

    assert resolved_count == len(resolved)
    assert deltas_count == len(deltas)
