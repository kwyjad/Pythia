from __future__ import annotations

import json
import sys

import pandas as pd
import pytest

try:  # pragma: no cover - exercised in CI
    from resolver.db import duckdb_io
except RuntimeError:  # DuckDB optional dependency missing
    duckdb_io = None

from resolver.common import compute_series_semantics, dict_counts, df_schema


def _emit_parity_diagnostics(
    expected_resolved: pd.DataFrame,
    db_rows: pd.DataFrame,
    join_cols: list[str],
    artifacts_dir,
):
    print("\n=== DB PARITY DIAGNOSTICS ===")
    print("CSV schema:", df_schema(expected_resolved))
    print("DB  schema:", df_schema(db_rows))
    print(
        "CSV series_semantics:",
        dict_counts(expected_resolved.get("series_semantics", pd.Series(dtype=str))),
    )
    print(
        "DB  series_semantics:",
        dict_counts(db_rows.get("series_semantics", pd.Series(dtype=str))),
    )

    missing_in_db = (
        expected_resolved.merge(
            db_rows,
            on=join_cols,
            how="left",
            indicator=True,
        )
        .query("_merge == 'left_only'")
        .drop(columns=["_merge"])
    )
    missing_in_csv = (
        db_rows.merge(
            expected_resolved,
            on=join_cols,
            how="left",
            indicator=True,
        )
        .query("_merge == 'left_only'")
        .drop(columns=["_merge"])
    )
    print("Missing in DB:", len(missing_in_db))
    print("Missing in CSV:", len(missing_in_csv))
    expected_path = artifacts_dir / "parity_mismatch_expected.csv"
    db_path = artifacts_dir / "parity_mismatch_db.csv"
    if not missing_in_db.empty:
        missing_in_db.to_csv(expected_path, index=False)
        print("Saved expected-only rows to", expected_path)
    if not missing_in_csv.empty:
        missing_in_csv.to_csv(db_path, index=False)
        print("Saved DB-only rows to", db_path)

    loose_cols = [col for col in join_cols if col != "series_semantics"]
    diff_records = []
    if loose_cols:
        joined = expected_resolved.merge(
            db_rows,
            on=loose_cols,
            how="inner",
            suffixes=("_csv", "_db"),
        )
        for col in expected_resolved.columns:
            left = f"{col}_csv"
            right = f"{col}_db"
            if left in joined.columns and right in joined.columns:
                mask = joined[left] != joined[right]
                mask = mask.fillna(False)
                for _, row in joined.loc[mask].head(5).iterrows():
                    diff = {key: row[key] for key in loose_cols}
                    diff.update({"column": col, "csv": row[left], "db": row[right]})
                    diff_records.append(diff)
    if diff_records:
        diff_frame = pd.DataFrame(diff_records)
        diff_path = artifacts_dir / "parity_mismatch_cell_diffs.csv"
        diff_frame.to_csv(diff_path, index=False)
        print("Saved column-level diffs to", diff_path)
    else:
        print("Column-level diffs: 0")


def _expected_resolved(df: pd.DataFrame) -> pd.DataFrame:
    expected = df.copy()
    expected["ym"] = ""
    if "as_of_date" in expected.columns:
        derived = pd.to_datetime(expected["as_of_date"], errors="coerce").dt.strftime("%Y-%m")
        expected.loc[expected["ym"].astype(str).str.len() == 0, "ym"] = derived.fillna("")
    if "publication_date" in expected.columns:
        mask = expected["ym"].astype(str).str.len() == 0
        fallback = pd.to_datetime(expected.loc[mask, "publication_date"], errors="coerce").dt.strftime("%Y-%m")
        expected.loc[mask, "ym"] = fallback.fillna("")
    expected["value"] = pd.to_numeric(expected.get("value", 0), errors="coerce")
    expected["series_semantics"] = expected.apply(
        lambda row: compute_series_semantics(
            metric=row.get("metric"), existing=row.get("series_semantics")
        ),
        axis=1,
    )
    subset = expected[
        [
            "event_id",
            "iso3",
            "hazard_code",
            "metric",
            "value",
            "unit",
            "as_of_date",
            "publication_date",
            "series_semantics",
            "ym",
        ]
    ].copy()
    subset["value"] = pd.to_numeric(subset["value"], errors="coerce")
    subset["series_semantics"] = subset["series_semantics"].fillna("").astype(str)
    subset["ym"] = subset["ym"].fillna("").astype(str)
    return subset


@pytest.mark.usefixtures("monkeypatch")
@pytest.mark.skipif(
    duckdb_io is None,
    reason="duckdb not installed — run `pip install -e .[db]` or `make dev-setup`",
)
def test_exporter_dual_writes_to_duckdb(tmp_path, monkeypatch):
    staging = tmp_path / "staging.csv"
    data = pd.DataFrame(
        [
            {
                "event_id": "E1",
                "country_name": "Philippines",
                "iso3": "PHL",
                "hazard_code": "TC",
                "hazard_label": "Tropical Cyclone",
                "hazard_class": "Cyclone",
                "metric": "in_need",
                "value": "1000",
                "unit": "persons",
                "as_of_date": "2024-01-15",
                "publication_date": "2024-01-16",
                "publisher": "OCHA",
                "source_type": "situation_report",
                "source_url": "https://example.org/report",
                "doc_title": "SitRep",
                "definition_text": "People in need",
                "method": "survey",
                "confidence": "medium",
                "revision": "1",
                "ingested_at": "2024-01-17T00:00:00Z",
            },
            {
                "event_id": "E2",
                "country_name": "Philippines",
                "iso3": "PHL",
                "hazard_code": "EQ",
                "hazard_label": "Earthquake",
                "hazard_class": "Geophysical",
                "metric": "affected",
                "value": "500",
                "unit": "persons",
                "as_of_date": "2024-01-10",
                "publication_date": "2024-01-11",
                "publisher": "OCHA",
                "source_type": "situation_report",
                "source_url": "https://example.org/report2",
                "doc_title": "SitRep 2",
                "definition_text": "People affected",
                "method": "survey",
                "confidence": "high",
                "revision": "1",
                "ingested_at": "2024-01-12T00:00:00Z",
            },
        ]
    )
    data.to_csv(staging, index=False)

    config = {
        "mapping": {c: [c] for c in data.columns},
        "constants": {},
    }
    config_path = tmp_path / "config.yml"
    config_path.write_text(json.dumps(config))

    out_dir = tmp_path / "exports"
    db_path = tmp_path / "resolver.duckdb"
    monkeypatch.setenv("RESOLVER_DB_URL", f"duckdb:///{db_path}")

    module = __import__("resolver.tools.export_facts", fromlist=["main"])
    argv = [
        "export_facts",
        "--in",
        str(staging),
        "--config",
        str(config_path),
        "--out",
        str(out_dir),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    module.main()

    csv_path = out_dir / "facts.csv"
    assert csv_path.exists(), "CSV export missing"

    exported = pd.read_csv(csv_path)

    conn = duckdb_io.get_db(f"duckdb:///{db_path}")
    duckdb_io.init_schema(conn)
    db_rows = conn.execute(
        "SELECT event_id, iso3, hazard_code, metric, value, unit, as_of_date, publication_date, series_semantics, ym "
        "FROM facts_resolved ORDER BY event_id"
    ).fetch_df()

    expected_resolved = _expected_resolved(exported)
    assert len(db_rows) == len(expected_resolved)
    join_cols = [
        "event_id",
        "iso3",
        "hazard_code",
        "metric",
        "unit",
        "as_of_date",
        "publication_date",
        "value",
        "series_semantics",
        "ym",
    ]
    joined = db_rows.merge(
        expected_resolved,
        on=join_cols,
        how="inner",
    )
    if len(joined) != len(expected_resolved):
        _emit_parity_diagnostics(expected_resolved, db_rows, join_cols, tmp_path)
    assert len(joined) == len(expected_resolved), "DuckDB facts_resolved rows diverge from CSV output"

    conn.close()

    freeze = __import__("resolver.tools.freeze_snapshot", fromlist=["main"])
    monkeypatch.setattr(freeze, "run_validator", lambda _path: None)
    monkeypatch.setattr(freeze, "SNAPSHOTS", tmp_path / "snapshots")
    argv = [
        "freeze_snapshot",
        "--facts",
        str(csv_path),
        "--month",
        "2024-01",
        "--overwrite",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    freeze.main()

    snapshot_dir = tmp_path / "snapshots" / "2024-01"
    facts_parquet = snapshot_dir / "facts.parquet"
    assert facts_parquet.exists(), "Snapshot facts parquet missing"
    snap_df = pd.read_parquet(facts_parquet)
    snap_expected = _expected_resolved(snap_df)

    conn = duckdb_io.get_db(f"duckdb:///{db_path}")
    duckdb_io.init_schema(conn)
    db_snapshot = conn.execute(
        "SELECT event_id, iso3, hazard_code, metric, value, unit, as_of_date, publication_date, series_semantics, ym "
        "FROM facts_resolved WHERE ym = '2024-01' ORDER BY event_id"
    ).fetch_df()
    merged_snapshot = db_snapshot.merge(
        snap_expected,
        on=join_cols,
        how="inner",
    )
    if len(merged_snapshot) != len(snap_expected):
        _emit_parity_diagnostics(snap_expected, db_snapshot, join_cols, tmp_path)
    assert len(merged_snapshot) == len(snap_expected)

    snapshots_db = conn.execute("SELECT ym, git_sha, created_at FROM snapshots").fetch_df()
    assert not snapshots_db.empty
    assert (snapshots_db["ym"] == "2024-01").any()
    conn.close()


def test_emit_parity_diagnostics_outputs(tmp_path, capsys):
    expected = pd.DataFrame(
        [
            {
                "event_id": "E1",
                "iso3": "AAA",
                "hazard_code": "TC",
                "metric": "in_need",
                "unit": "people",
                "as_of_date": "2024-01-01",
                "publication_date": "2024-01-02",
                "value": 1000,
                "series_semantics": "stock",
                "ym": "2024-01",
            },
            {
                "event_id": "E2",
                "iso3": "AAA",
                "hazard_code": "EQ",
                "metric": "affected",
                "unit": "people",
                "as_of_date": "2024-01-05",
                "publication_date": "2024-01-06",
                "value": 500,
                "series_semantics": "",
                "ym": "2024-01",
            },
        ]
    )
    db_rows = expected.copy()
    db_rows.loc[1, "value"] = 600

    join_cols = [
        "event_id",
        "iso3",
        "hazard_code",
        "metric",
        "unit",
        "as_of_date",
        "publication_date",
        "value",
        "series_semantics",
        "ym",
    ]

    _emit_parity_diagnostics(expected, db_rows, join_cols, tmp_path)
    captured = capsys.readouterr().out
    assert "DB PARITY DIAGNOSTICS" in captured
    assert (tmp_path / "parity_mismatch_expected.csv").exists()
    assert (tmp_path / "parity_mismatch_db.csv").exists()
