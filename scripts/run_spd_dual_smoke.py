"""Run a minimal SPD v2 dual-run smoke compare on real configs."""

import asyncio
import os
from pathlib import Path

import duckdb

from pythia.db import schema as db_schema
from forecaster import cli


def main() -> None:
    compare_dir = os.getenv("PYTHIA_SPD_COMPARE_DIR", "debug/spd_compare_smoke")
    os.environ["PYTHIA_SPD_COMPARE_DIR"] = compare_dir

    default_run_id = f"smoke_{os.getenv('GITHUB_RUN_ID', 'dual')}"
    run_id = os.getenv("SPD_SMOKE_RUN_ID", default_run_id)
    os.environ["SPD_SMOKE_RUN_ID"] = run_id

    iso3 = os.getenv("SPD_SMOKE_ISO3", "ETH")
    hz = os.getenv("SPD_SMOKE_HAZARD", "DR")
    metric = os.getenv("SPD_SMOKE_METRIC", "PA")
    target_month = os.getenv("SPD_SMOKE_TARGET_MONTH", "2025-12")
    default_qid = f"SMOKE_{iso3}_{hz}_{metric}_{target_month}"
    qid = os.getenv("SPD_SMOKE_QID", default_qid)
    os.environ["SPD_SMOKE_QID"] = qid

    db_path = Path("data") / "spd_smoke.duckdb"
    os.environ["PYTHIA_DB_URL"] = f"duckdb:///{db_path}"

    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    try:
        db_schema.ensure_schema(con)
        con.execute(
            """
            INSERT INTO questions (
              question_id, hs_run_id, scenario_ids_json, iso3, hazard_code, metric,
              target_month, window_start_date, window_end_date, wording, status, pythia_metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                qid,
                "",
                "[]",
                iso3,
                hz,
                metric,
                target_month,
                None,
                None,
                "Smoke SPD question",
                "active",
                None,
            ],
        )
        question_row = con.execute(
            "SELECT * FROM questions WHERE question_id = ?", [qid]
        ).fetchone()
    finally:
        con.close()

    asyncio.run(cli._run_spd_for_question(run_id, question_row))


if __name__ == "__main__":
    main()
