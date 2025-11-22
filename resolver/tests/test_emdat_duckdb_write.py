import csv
from pathlib import Path

import pytest

from resolver.db import duckdb_io
from resolver.ingestion.emdat_normalize import (
    normalize_emdat_pa,
    write_emdat_pa_to_duckdb,
)
from resolver.ingestion.emdat_stub import fetch_raw
from resolver.ingestion.emdat_client import CANONICAL_HEADERS
from resolver.tools.export_facts import export_facts
from resolver.tools.freeze_snapshot import freeze_snapshot


@pytest.mark.duckdb
def test_emdat_duckdb_write_idempotent(tmp_path):
    pytest.importorskip("duckdb")

    db_path = tmp_path / "emdat.duckdb"
    db_url = f"duckdb:///{db_path.as_posix()}"
    conn = duckdb_io.get_db(db_url)
    try:
        raw = fetch_raw(2010, 2030)
        normalized = normalize_emdat_pa(
            raw, info={"timestamp": "2024-01-31T00:00:00Z"}
        )

        first = write_emdat_pa_to_duckdb(conn, normalized)
        assert first.rows_written == len(normalized)

        second = write_emdat_pa_to_duckdb(conn, normalized)
        assert second.rows_delta == 0

        row_count = conn.execute("SELECT COUNT(*) FROM emdat_pa").fetchone()[0]
        assert row_count == len(normalized)

        columns = [row[1] for row in conn.execute("PRAGMA table_info('emdat_pa')").fetchall()]
        for column in (
            "iso3",
            "ym",
            "shock_type",
            "pa",
            "as_of_date",
            "publication_date",
            "source_id",
            "disno_first",
        ):
            assert column in columns

        pa_values = [value for (value,) in conn.execute("SELECT pa FROM emdat_pa").fetchall()]
        assert all(isinstance(value, int) for value in pa_values)
        assert all(value >= 0 for value in pa_values)

        indexes = conn.execute(
            "SELECT index_name FROM duckdb_indexes() WHERE table_name = 'emdat_pa'"
        ).fetchall()
        assert any(name == "ux_emdat_pa" for (name,) in indexes)
    finally:
        duckdb_io.close_db(conn)


@pytest.mark.duckdb
@pytest.mark.legacy_freeze
@pytest.mark.xfail(
    reason="Legacy freeze_snapshot pipeline is retired and replaced by DB-backed snapshot builder."
)
def test_emdat_export_and_freeze_to_duckdb(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("duckdb")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RESOLVER_WRITE_DB", "0")
    staging_dir = Path("resolver/staging")
    staging_dir.mkdir(parents=True, exist_ok=True)

    raw = fetch_raw(2015, 2016)
    normalized = normalize_emdat_pa(
        raw,
        info={"timestamp": "2024-01-15T00:00:00Z"},
    )
    assert not normalized.empty

    sample = normalized.head(8).copy()
    csv_path = staging_dir / "emdat_pa.csv"
    sample.loc[:, list(CANONICAL_HEADERS)].to_csv(csv_path, index=False)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
    assert header == list(CANONICAL_HEADERS)

    repo_root = Path(__file__).resolve().parents[2]
    export_facts(
        inp=staging_dir,
        config_path=repo_root / "resolver" / "tools" / "export_config.yml",
        out_dir=Path("resolver/exports/backfill"),
        write_db=False,
    )

    preview_path = Path("diagnostics/ingestion/export_preview/facts.csv")
    assert preview_path.exists()
    with preview_path.open(newline="", encoding="utf-8") as handle:
        preview_rows = list(csv.DictReader(handle))
    assert preview_rows

    month = preview_rows[0]["ym"]
    db_path = Path("data/resolver_backfill.duckdb")
    db_path.parent.mkdir(parents=True, exist_ok=True)

    freeze_snapshot(
        facts=preview_path,
        month=month,
        outdir=Path("snapshots"),
        write_db=True,
        db_url=f"duckdb:///{db_path.as_posix()}",
    )

    preview_map = {
        (row["iso3"], row["ym"], row["hazard_code"], row["metric"]): int(float(row["value"]))
        for row in preview_rows
    }

    conn = duckdb_io.get_db(f"duckdb:///{db_path.as_posix()}")
    try:
        count = conn.execute("SELECT COUNT(*) FROM facts_deltas").fetchone()[0]
        assert count == len(preview_rows)

        db_rows = conn.execute(
            "SELECT iso3, ym, hazard_code, metric, series_semantics, value_new FROM facts_deltas"
        ).fetchall()
        assert len(db_rows) == len(preview_map)
        for iso3, ym, hazard_code, metric, semantics, value_new in db_rows:
            key = (iso3, ym, hazard_code, metric)
            assert key in preview_map
            assert int(value_new) == preview_map[key]
            assert semantics == "new"
    finally:
        duckdb_io.close_db(conn)
