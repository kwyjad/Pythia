from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pandas as pd

from resolver.tests.utils import run as run_proc

CANONICAL_COLUMNS = [
    "event_id",
    "country_name",
    "iso3",
    "hazard_code",
    "hazard_label",
    "hazard_class",
    "metric",
    "unit",
    "as_of_date",
    "value",
    "series_semantics",
    "source",
]


def _write_canonical_fixture(base: Path) -> None:
    canonical_dir = base / "data" / "staging" / "2025Q3" / "canonical"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        [
            [
                "evt-1",
                "Exampleland",
                "EXL",
                "DR",
                "Drought",
                "natural",
                "in_need",
                "persons",
                "2025-07-15",
                100,
                "stock",
                "Relief Org",
            ],
            [
                "evt-2",
                "Exampleland",
                "EXL",
                "DR",
                "Drought",
                "natural",
                "in_need",
                "persons",
                "2025-08-15",
                150,
                "stock",
                "Relief Org",
            ],
            [
                "evt-3",
                "Exampleland",
                "EXL",
                "DR",
                "Drought",
                "natural",
                "in_need",
                "persons",
                "2025-09-15",
                120,
                "stock",
                "Relief Org",
            ],
            [
                "evt-4",
                "Exampleland",
                "EXL",
                "DR",
                "Drought",
                "natural",
                "in_need",
                "persons",
                "2025-09-20",
                25,
                "new",
                "Relief Org",
            ],
        ],
        columns=CANONICAL_COLUMNS,
    )
    frame.to_csv(canonical_dir / "canonical.csv", index=False)


def test_cli_loads_derives_and_exports(tmp_path: Path, monkeypatch) -> None:
    _write_canonical_fixture(tmp_path)

    db_path = tmp_path / "resolver.duckdb"
    snapshots_root = tmp_path / "data" / "snapshots"

    cmd = [
        sys.executable,
        "-m",
        "resolver.tools.load_and_derive",
        "--period",
        "2025Q3",
        "--staging-root",
        str(tmp_path / "data" / "staging"),
        "--snapshots-root",
        str(snapshots_root),
        "--db",
        str(db_path),
        "--allow-negatives",
        "1",
    ]

    result = run_proc(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise AssertionError(
            f"CLI failed: {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    conn = duckdb.connect(str(db_path))
    try:
        raw_count = conn.execute("SELECT COUNT(*) FROM facts_raw").fetchone()[0]
        resolved = conn.execute(
            "SELECT ym, value, provenance_source, provenance_rank FROM facts_resolved ORDER BY ym"
        ).df()
        deltas = conn.execute(
            "SELECT ym, value_new, value_stock FROM facts_deltas ORDER BY ym"
        ).df()
    finally:
        conn.close()

    assert raw_count == 4
    assert len(resolved) == 3
    assert len(deltas) == 3

    assert set(resolved["provenance_source"]) == {"Relief Org"}
    assert (resolved["provenance_rank"] >= 1).all()

    deltas["cumulative"] = deltas["value_new"].cumsum()
    merged = resolved.merge(deltas, on="ym", how="inner")
    assert len(merged) == 3
    assert merged.loc[merged["ym"] == "2025-09", "value_new"].iloc[0] == -30
    assert (merged["value"] == merged["cumulative"]).all()

    period_dir = snapshots_root / "2025Q3"
    resolved_parquet = period_dir / "facts_resolved.parquet"
    deltas_parquet = period_dir / "facts_deltas.parquet"
    assert resolved_parquet.exists()
    assert deltas_parquet.exists()

    # Ensure DuckDB can read the snapshot parquet outputs.
    duck_conn = duckdb.connect()
    try:
        duck_conn.execute(f"SELECT COUNT(*) FROM read_parquet('{resolved_parquet}')")
        duck_conn.execute(f"SELECT COUNT(*) FROM read_parquet('{deltas_parquet}')")
    finally:
        duck_conn.close()
