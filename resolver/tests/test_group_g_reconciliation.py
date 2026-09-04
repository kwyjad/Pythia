# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Group G of the run-33841370196 faults: the reconciliation checks.

G1: an idempotent upsert moves no table count, so a check that compares a
claim against ``rows_after - rows_before`` calls NMME a failure for merging
2,408 rows into a table that stayed at 4,816. G2: three sources write one
table, so one delta cannot say which of them wrote. Both are settled by the
rows carrying THIS run's write stamp within each connector's own source
filter.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd
import pytest

duckdb = pytest.importorskip("duckdb")

from resolver.db import duckdb_io  # noqa: E402
from scripts import build_resolver_debug_bundle as bundle  # noqa: E402

RUN_START = "2026-09-04T05:00:00Z"
BEFORE = "2026-09-03 12:00:00"   # written by an earlier run
DURING = "2026-09-04 06:00:00"   # written by this run


def _db(path: Path) -> None:
    con = duckdb.connect(str(path))
    con.execute(
        "CREATE TABLE seasonal_forecasts (iso3 TEXT, variable TEXT, lead_months INTEGER, "
        "anomaly_value DOUBLE, forecast_issue_date DATE, created_at TIMESTAMP, fetched_at TIMESTAMP)"
    )
    # An idempotent re-upsert: created_at from the earlier run, fetched_at
    # from this one, and the row count unchanged.
    con.execute(
        f"INSERT INTO seasonal_forecasts VALUES "
        f"('SOM','prate',1,-1.2,DATE '2026-08-08',TIMESTAMP '{BEFORE}',TIMESTAMP '{DURING}'),"
        f"('KEN','prate',1,0.4,DATE '2026-08-08',TIMESTAMP '{BEFORE}',TIMESTAMP '{DURING}')"
    )
    con.execute(
        "CREATE TABLE conflict_forecasts (source TEXT, iso3 TEXT, hazard_code TEXT, metric TEXT, "
        "lead_months INTEGER, forecast_issue_date DATE, value DOUBLE, created_at TIMESTAMP)"
    )
    # VIEWS wrote this run; ACLED CAST's rows are all from an earlier run.
    con.execute(
        f"INSERT INTO conflict_forecasts VALUES "
        f"('VIEWS','SOM','ACE','fatalities',1,DATE '2026-08-01',12.0,TIMESTAMP '{DURING}'),"
        f"('ACLED_CAST','SOM','ACE','cast_total_events',1,DATE '2025-12-01',10.0,TIMESTAMP '{BEFORE}')"
    )
    con.close()


def _diagnostics(tmp_path: Path, records: list[dict]) -> Path:
    diagnostics = tmp_path / "diagnostics"
    (diagnostics / "ingestion").mkdir(parents=True)
    (diagnostics / "ingestion" / "connectors_report.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8",
    )
    # A table delta of zero for seasonal_forecasts: the old check's false alarm.
    (diagnostics / "db_signature_before.json").write_text(
        json.dumps({"all_counts": {"seasonal_forecasts": 2, "conflict_forecasts": 2}}),
        encoding="utf-8",
    )
    return diagnostics


def _build(tmp_path: Path, db: Path, diagnostics: Path, environ: dict | None):
    out = tmp_path / "bundle.zip"
    manifest = bundle.build_bundle(
        out_path=out, db_path=db, diagnostics_dir=diagnostics, run_log_dir=None,
        max_bytes=bundle.DEFAULT_MAX_BYTES, staging=tmp_path / "staging",
        environ={} if environ is None else environ,
    )
    return out, {c["name"]: c for c in manifest["checks"]}


NAME = "connectors_claiming_rows_touched_their_source_rows"


def test_an_idempotent_upsert_is_not_a_failure(tmp_path):
    """G1: NMME merged 2,408 rows into a table that stayed at 4,816."""

    db = tmp_path / "r.duckdb"; _db(db)
    diagnostics = _diagnostics(tmp_path, [
        {"connector_id": "nmme_seasonal_forecasts", "status": "ok",
         "counts": {"fetched": 2, "written": 1, "rows_written": 2}},
    ])
    _out, checks = _build(tmp_path, db, diagnostics, {"PYTHIA_RUN_STARTED_AT": RUN_START})
    assert checks[NAME]["verdict"] == "PASS", checks[NAME]


def test_a_shared_table_is_reconciled_per_source(tmp_path):
    """G2: VIEWS wrote this run; CAST claimed rows and wrote none of its own."""

    db = tmp_path / "r.duckdb"; _db(db)
    diagnostics = _diagnostics(tmp_path, [
        {"connector_id": "views_forecasts", "status": "ok",
         "counts": {"written": 1}, "extras": {"conflict_rows": 1}},
        {"connector_id": "acledcast_forecasts", "status": "ok",
         "counts": {"written": 1}, "extras": {"conflict_rows": 545}},
    ])
    out, checks = _build(tmp_path, db, diagnostics, {"PYTHIA_RUN_STARTED_AT": RUN_START})
    check = checks[NAME]
    assert check["verdict"] == "FAIL"
    assert "acledcast_forecasts" in check["detail"]
    assert "views_forecasts" not in check["detail"]
    assert "source = 'ACLED_CAST'" in check["detail"]

    import zipfile

    with zipfile.ZipFile(out) as zf:
        reconciliation = zf.read("checks/reconciliation.md").decode("utf-8")
    assert "touched since run start" in reconciliation and "source filter" in reconciliation
    assert "| views_forecasts |" in reconciliation and "**NO**" in reconciliation


def test_without_a_run_start_nothing_is_attributed(tmp_path):
    db = tmp_path / "r.duckdb"; _db(db)
    diagnostics = _diagnostics(tmp_path, [
        {"connector_id": "views_forecasts", "status": "ok", "counts": {"written": 1}},
    ])
    _out, checks = _build(tmp_path, db, diagnostics, {})
    assert checks[NAME]["verdict"] == "SKIP"
    assert "PYTHIA_RUN_STARTED_AT" in checks[NAME]["detail"]


# --------------------------------------------------------------------------
# The stamp the check reads: facts tables carry updated_at on every MERGE
# --------------------------------------------------------------------------


def _facts_frame() -> pd.DataFrame:
    return pd.DataFrame([{
        "ym": "2026-08", "iso3": "SOM", "hazard_code": "ACE", "metric": "fatalities",
        "value": 415.0, "unit": "persons", "series_semantics": "new",
        "as_of": "2026-08-31", "as_of_date": "2026-08-31", "publication_date": "2026-09-01",
        "publisher": "ACLED", "source_id": "acled", "event_id": "",
        "provenance_source": "acled", "series": "",
    }])


def test_every_facts_write_moves_updated_at_and_never_created_at(tmp_path):
    con = duckdb.connect(str(tmp_path / "f.duckdb"))
    duckdb_io.init_schema(con)
    duckdb_io.write_facts_tables(con, facts_resolved=_facts_frame())
    first = con.execute("SELECT created_at, updated_at FROM facts_resolved").fetchone()
    assert first[1] is not None
    import time

    time.sleep(0.01)
    duckdb_io.write_facts_tables(con, facts_resolved=_facts_frame())
    rows = con.execute("SELECT created_at, updated_at FROM facts_resolved").fetchall()
    assert len(rows) == 1, "an idempotent re-write is one row"
    assert rows[0][0] == first[0], "created_at marks the first insert"
    assert rows[0][1] > first[1], "updated_at moves on the matched MERGE"
    con.close()


def test_a_pre_stamp_facts_table_gains_the_column_on_init(tmp_path):
    con = duckdb.connect(str(tmp_path / "old.duckdb"))
    con.execute(
        "CREATE TABLE facts_resolved (ym TEXT, iso3 TEXT, hazard_code TEXT, metric TEXT, "
        "value DOUBLE, series_semantics TEXT, created_at TIMESTAMP)"
    )
    con.execute("CREATE TABLE facts_deltas (ym TEXT, iso3 TEXT, hazard_code TEXT, metric TEXT, "
                "value_new DOUBLE, series_semantics TEXT, created_at TIMESTAMP)")
    duckdb_io.init_schema(con)
    for table in ("facts_resolved", "facts_deltas"):
        columns = {r[1] for r in con.execute(f"PRAGMA table_info('{table}')").fetchall()}
        assert "updated_at" in columns, table
    con.close()
