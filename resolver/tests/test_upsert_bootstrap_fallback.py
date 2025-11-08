import duckdb
import pandas as pd
import pytest

from resolver.db import duckdb_io


@pytest.mark.parametrize(
    "table,keys,frame",
    [
        (
            "facts_resolved",
            duckdb_io.FACTS_RESOLVED_KEY_COLUMNS,
            pd.DataFrame(
                [
                    {
                        "ym": "2024-01",
                        "iso3": "COL",
                        "hazard_code": "FLD",
                        "metric": "affected",
                        "series_semantics": "stock",
                        "value": "1.5",
                        "as_of_date": "2024-01-15",
                        "publication_date": "2024-01-20",
                        "publisher": "Unit Test",
                        "hazard_label": "Flood",
                        "hazard_class": "hydro",
                        "unit": "persons",
                    }
                ]
            ),
        ),
    ],
)
def test_fallback_writes_when_no_constraints(monkeypatch, tmp_path, table, keys, frame):
    conn = duckdb.connect(database=str(tmp_path / "bootstrap.duckdb"))

    # Ensure schema exists but simulate a case where key detection fails entirely.
    duckdb_io.init_schema(conn)
    monkeypatch.setattr(duckdb_io, "_has_declared_key", lambda *args, **kwargs: False)
    monkeypatch.setattr(duckdb_io, "_attempt_heal_missing_key", lambda *args, **kwargs: False)

    written = duckdb_io.upsert_dataframe(conn, table, frame, keys=keys)
    assert written.rows_written == len(frame)
    assert written.rows_delta == len(frame)

    rows = conn.execute(
        f"SELECT value FROM {table} WHERE ym='2024-01' AND iso3='COL'"
    ).fetchall()
    assert rows and pytest.approx(rows[0][0]) == 1.5

    # Update the same key to confirm deterministic delete+insert behaviour.
    frame_updated = frame.copy()
    frame_updated.loc[0, "value"] = "2.0"
    written_second = duckdb_io.upsert_dataframe(conn, table, frame_updated, keys=keys)
    assert written_second.rows_written == len(frame_updated)
    rows = conn.execute(
        f"SELECT value FROM {table} WHERE ym='2024-01' AND iso3='COL'"
    ).fetchall()
    assert rows and pytest.approx(rows[0][0]) == 2.0


def test_numeric_coercion_to_nulls(tmp_path):
    conn = duckdb.connect(database=str(tmp_path / "coercion.duckdb"))
    duckdb_io.init_schema(conn)

    frame = pd.DataFrame(
        [
            {
                "ym": "2024-02",
                "iso3": "COL",
                "hazard_code": "FLD",
                "metric": "affected",
                "series_semantics": "new",
                "value_new": "200",
                "value_stock": "None",
                "as_of": "2024-02-15",
            }
        ]
    )

    written = duckdb_io.upsert_dataframe(
        conn,
        "facts_deltas",
        frame,
        keys=duckdb_io.FACTS_DELTAS_KEY_COLUMNS,
    )
    assert written.rows_written == len(frame)

    value_new, value_stock = conn.execute(
        "SELECT value_new, value_stock FROM facts_deltas WHERE ym='2024-02'"
    ).fetchone()
    assert pytest.approx(value_new) == 200.0
    assert value_stock is None
