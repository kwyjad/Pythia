import json
from pathlib import Path
from typing import Tuple

import pandas as pd
import pytest

pytest.importorskip("duckdb")

from resolver.db import duckdb_io
from resolver.tools import freeze_snapshot


def _sample_facts() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_id": "E1",
                "iso3": "PHL",
                "hazard_code": "TC",
                "hazard_label": "Cyclone",
                "metric": "in_need",
                "value": "1000",
                "unit": "persons",
                "as_of_date": "2024-01-15",
                "publication_date": "2024-01-16",
                "publisher": "OCHA",
                "series_semantics": "",
                "ym": "2024-01",
            },
            {
                "event_id": "E2",
                "iso3": "PHL",
                "hazard_code": "EQ",
                "hazard_label": "Earthquake",
                "metric": "affected",
                "value": "250",
                "unit": "persons",
                "as_of_date": "2024-01-10",
                "publication_date": "2024-01-11",
                "publisher": "OCHA",
                "series_semantics": "Stock estimate",
                "ym": "2024-01",
            },
        ]
    )


def _write_snapshot_inputs(tmp_path: Path) -> Tuple[Path, Path, Path]:
    facts = _sample_facts()

    facts_csv = tmp_path / "facts.csv"
    resolved_csv = tmp_path / "resolved.csv"
    manifest_path = tmp_path / "manifest.json"

    facts.to_csv(facts_csv, index=False)
    facts.to_csv(resolved_csv, index=False)

    manifest = {
        "created_at_utc": "2024-01-31T00:00:00Z",
        "source_commit_sha": "abc123",
        "resolved_rows": len(facts),
        "artifacts": {
            "facts_resolved_csv": str(resolved_csv),
            "facts_resolved_parquet": str(resolved_csv.with_suffix(".parquet")),
        },
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    return facts_csv, resolved_csv, manifest_path


def test_maybe_write_db_snapshot_idempotent(tmp_path: Path) -> None:
    facts_csv, resolved_csv, manifest_path = _write_snapshot_inputs(tmp_path)
    facts = _sample_facts()
    db_path = tmp_path / "resolver.duckdb"
    db_url = f"duckdb:///{db_path}"

    freeze_snapshot._maybe_write_db(
        facts_path=facts_csv,
        resolved_path=resolved_csv,
        deltas_path=None,
        manifest_path=manifest_path,
        month="2024-01",
        db_url=db_url,
        write_db=True,
    )
    freeze_snapshot._maybe_write_db(
        facts_path=facts_csv,
        resolved_path=resolved_csv,
        deltas_path=None,
        manifest_path=manifest_path,
        month="2024-01",
        db_url=db_url,
        write_db=True,
    )

    conn = duckdb_io.get_db(db_url)
    try:
        facts_rows = conn.execute(
            "SELECT COUNT(*) FROM facts_resolved WHERE ym = '2024-01'"
        ).fetchone()[0]
        deltas_rows = conn.execute(
            "SELECT COUNT(*) FROM facts_deltas WHERE ym = '2024-01'"
        ).fetchone()[0]
        snapshots_rows = conn.execute(
            "SELECT COUNT(*) FROM snapshots WHERE ym = '2024-01'"
        ).fetchone()[0]
    finally:
        duckdb_io.close_db(conn)

    assert facts_rows == len(facts)
    assert deltas_rows == len(facts)
    assert snapshots_rows == 1
