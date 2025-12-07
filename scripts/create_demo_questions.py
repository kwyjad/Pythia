from __future__ import annotations

import argparse
import os
from datetime import date

import duckdb

from pythia.db.schema import ensure_schema
from scripts.create_questions_from_triage import _build_question_wording

DEMO_QUESTIONS = [
    {
        "question_id": "ETH_ACE_FATALITIES",
        "iso3": "ETH",
        "hazard_code": "ACE",
        "metric": "FATALITIES",
        "target_month": "2026-01",
        "window_start_date": "2026-01-01",
        "window_end_date": "2026-06-30",
    },
    {
        "question_id": "ETH_ACE_PA",
        "iso3": "ETH",
        "hazard_code": "ACE",
        "metric": "PA",
        "target_month": "2026-01",
        "window_start_date": "2026-01-01",
        "window_end_date": "2026-06-30",
    },
    {
        "question_id": "SOM_ACE_FATALITIES",
        "iso3": "SOM",
        "hazard_code": "ACE",
        "metric": "FATALITIES",
        "target_month": "2026-01",
        "window_start_date": "2026-01-01",
        "window_end_date": "2026-06-30",
    },
    {
        "question_id": "SOM_ACE_PA",
        "iso3": "SOM",
        "hazard_code": "ACE",
        "metric": "PA",
        "target_month": "2026-01",
        "window_start_date": "2026-01-01",
        "window_end_date": "2026-06-30",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create demo questions in DuckDB (gated by PYTHIA_ENABLE_DEMO_QUESTIONS=1)"
        )
    )
    parser.add_argument("--db", default="duckdb:///data/resolver.duckdb", help="DuckDB URL")
    args = parser.parse_args()

    if os.getenv("PYTHIA_ENABLE_DEMO_QUESTIONS", "0") != "1":
        print(
            "[info] Demo questions disabled (set PYTHIA_ENABLE_DEMO_QUESTIONS=1 to enable)"
        )
        return

    db_url = args.db
    if db_url.startswith("duckdb:///"):
        db_path = db_url[len("duckdb:///") :]
    else:
        db_path = db_url

    con = duckdb.connect(db_path, read_only=False)
    ensure_schema(con)
    try:
        inserted = 0
        for q in DEMO_QUESTIONS:
            qid = q["question_id"]
            iso3 = q["iso3"]
            hz = q["hazard_code"]
            metric = q["metric"]
            target_month = q["target_month"]
            window_start = date.fromisoformat(q["window_start_date"])
            window_end = date.fromisoformat(q["window_end_date"])
            wording = _build_question_wording(
                iso3, hz, metric, window_start, window_end
            )

            # delete any existing question with this id so we can reinsert cleanly
            con.execute(
                "DELETE FROM questions WHERE question_id = ?",
                [qid],
            )

            con.execute(
                """
                INSERT INTO questions (
                    question_id, iso3, hazard_code, metric,
                    target_month, window_start_date, window_end_date,
                    wording, status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active')
                """,
                [
                    qid,
                    iso3,
                    hz,
                    metric,
                    target_month,
                    window_start,
                    window_end,
                    wording,
                ],
            )
            inserted += 1
        print(f"create_demo_questions: ensured {inserted} demo questions exist")
    finally:
        con.close()


if __name__ == "__main__":
    main()
