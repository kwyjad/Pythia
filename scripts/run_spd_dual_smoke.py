import os
import asyncio
from pathlib import Path
import duckdb

from pythia.db import schema as db_schema
from forecaster import cli


def main() -> None:
    # Always run compare in smoke
    os.environ["PYTHIA_SPD_V2_DUAL_RUN"] = os.getenv("PYTHIA_SPD_V2_DUAL_RUN", "1")
    os.environ["PYTHIA_SPD_COMPARE_DIR"] = os.getenv(
        "PYTHIA_SPD_COMPARE_DIR", "debug/spd_compare_smoke"
    )

    run_id = os.getenv("SPD_SMOKE_RUN_ID", f"smoke_{os.getenv('GITHUB_RUN_ID','local')}")
    iso3 = os.getenv("SPD_SMOKE_ISO3", "SOM").upper()
    hz = os.getenv("SPD_SMOKE_HAZARD", "ACE").upper()
    metric = os.getenv("SPD_SMOKE_METRIC", "PA").upper()
    target_month = os.getenv("SPD_SMOKE_TARGET_MONTH", "2025-12")

    qid = os.getenv("SPD_SMOKE_QID", f"SMOKE_{iso3}_{hz}_{metric}_{target_month}")

    print(f"[smoke] run_id={run_id}")
    print(f"[smoke] question_id={qid}")
    print(f"[smoke] compare_dir={os.environ['PYTHIA_SPD_COMPARE_DIR']}")
    print(f"[smoke] bayesmc_write={os.getenv('PYTHIA_SPD_V2_USE_BAYESMC','0')}")

    db_path = Path("data") / "spd_smoke.duckdb"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ["PYTHIA_DB_URL"] = f"duckdb:///{db_path}"

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
            "SELECT * FROM questions WHERE question_id = ?",
            [qid],
        ).fetchone()
    finally:
        con.close()

    asyncio.run(cli._run_spd_for_question(run_id, question_row))


if __name__ == "__main__":
    main()
