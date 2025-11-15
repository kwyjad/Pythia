import json
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("duckdb")

from resolver.tools.freeze_snapshot import _maybe_write_db


def _write_snapshot_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    facts_csv = tmp_path / "facts.csv"
    resolved_csv = tmp_path / "resolved.csv"
    deltas_csv = tmp_path / "deltas.csv"
    manifest_path = tmp_path / "manifest.json"

    payload = {
        "iso3": ["AFG"],
        "metric": ["events"],
        "series_semantics": ["new"],
        "value": [3],
        "as_of_date": ["2025-11-30"],
        "publication_date": ["2025-12-01"],
        "ym": ["2025-11"],
    }
    pd.DataFrame(payload).to_csv(facts_csv, index=False)
    pd.DataFrame(payload).to_csv(resolved_csv, index=False)
    pd.DataFrame({**payload, "value_new": [3], "as_of": ["2025-11-30"]}).to_csv(
        deltas_csv, index=False
    )
    manifest_path.write_text(
        json.dumps({"created_at_utc": "2025-12-01T00:00:00Z"}), encoding="utf-8"
    )
    return facts_csv, resolved_csv, deltas_csv, manifest_path


def test_maybe_write_db_writes_snapshot(tmp_path: Path) -> None:
    facts_csv, resolved_csv, deltas_csv, manifest_path = _write_snapshot_inputs(tmp_path)
    db_path = tmp_path / "snapshot.duckdb"
    db_url = f"duckdb:///{db_path}"

    _maybe_write_db(
        facts_path=facts_csv,
        resolved_path=resolved_csv,
        deltas_path=deltas_csv,
        manifest_path=manifest_path,
        month="2025-11",
        db_url=db_url,
        write_db=True,
    )

    import duckdb  # delayed import for pytest.importorskip

    conn = duckdb.connect(str(db_path))
    try:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
            ).fetchall()
        }
    finally:
        conn.close()

    assert "facts_resolved" in tables
    assert "snapshots" in tables


def test_maybe_write_db_noop_when_write_disabled(tmp_path: Path) -> None:
    facts_csv, resolved_csv, deltas_csv, manifest_path = _write_snapshot_inputs(tmp_path)
    db_path = tmp_path / "snapshot.duckdb"
    db_url = f"duckdb:///{db_path}"

    _maybe_write_db(
        facts_path=facts_csv,
        resolved_path=resolved_csv,
        deltas_path=deltas_csv,
        manifest_path=manifest_path,
        month="2025-11",
        db_url=db_url,
        write_db=False,
    )

    assert not db_path.exists()
