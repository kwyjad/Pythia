# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

import pandas as pd

from resolver.db import duckdb_io


def test_write_snapshot_initialises_schema(tmp_path):
    db_path = tmp_path / "auto-init.duckdb"
    url = f"duckdb:///{db_path}"
    conn = duckdb_io.get_db(url)

    resolved = pd.DataFrame(
        [
            {
                "ym": "2024-02",
                "iso3": "PHL",
                "hazard_code": "TC",
                "metric": "in_need",
                "series_semantics": "stock",
                "value": "123",
            }
        ]
    )

    duckdb_io.write_snapshot(
        conn,
        ym="2024-02",
        facts_resolved=resolved,
        facts_deltas=None,
        manifests=None,
        meta={},
    )

    count = conn.execute("SELECT COUNT(*) FROM facts_resolved").fetchone()[0]
    assert count == 1

    indexes = conn.execute(
        "SELECT index_name FROM duckdb_indexes() WHERE table_name='facts_resolved'"
    ).fetchall()
    names = {row[0] for row in indexes}
    assert "ux_facts_resolved_series" in names
