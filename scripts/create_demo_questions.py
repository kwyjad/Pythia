from __future__ import annotations

import argparse

import duckdb

from pythia.db.schema import ensure_schema

DEMO_QUESTIONS = [
    {
        "question_id": "ETH_ACO_FATALITIES",
        "iso3": "ETH",
        "hazard_code": "ACO",
        "metric": "FATALITIES",
        "target_month": "2026-01",
        "window_start_date": "2026-01-01",
        "window_end_date": "2026-06-30",
        "wording": "Monthly conflict fatalities in Ethiopia associated with armed conflict (ACLED).",
    },
    {
        "question_id": "ETH_ACO_PA",
        "iso3": "ETH",
        "hazard_code": "ACO",
        "metric": "PA",
        "target_month": "2026-01",
        "window_start_date": "2026-01-01",
        "window_end_date": "2026-06-30",
        "wording": "Monthly IDPs in Ethiopia due to armed conflict (IDMC as resolution).",
    },
    {
        "question_id": "SOM_ACO_FATALITIES",
        "iso3": "SOM",
        "hazard_code": "ACO",
        "metric": "FATALITIES",
        "target_month": "2026-01",
        "window_start_date": "2026-01-01",
        "window_end_date": "2026-06-30",
        "wording": "Monthly conflict fatalities in Somalia associated with armed conflict (ACLED).",
    },
    {
        "question_id": "SOM_ACO_PA",
        "iso3": "SOM",
        "hazard_code": "ACO",
        "metric": "PA",
        "target_month": "2026-01",
        "window_start_date": "2026-01-01",
        "window_end_date": "2026-06-30",
        "wording": "Monthly IDPs in Somalia due to armed conflict (IDMC as resolution).",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create demo questions in DuckDB")
    parser.add_argument("--db", default="duckdb:///data/resolver.duckdb", help="DuckDB URL")
    args = parser.parse_args()

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
            con.execute(
                """
                INSERT INTO questions (
                    question_id, iso3, hazard_code, metric,
                    target_month, window_start_date, window_end_date,
                    wording, status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active')
                ON CONFLICT(question_id) DO NOTHING
                """,
                [
                    q["question_id"],
                    q["iso3"],
                    q["hazard_code"],
                    q["metric"],
                    q["target_month"],
                    q["window_start_date"],
                    q["window_end_date"],
                    q["wording"],
                ],
            )
            inserted += 1
        print(f"create_demo_questions: ensured {inserted} demo questions exist")
    finally:
        con.close()


if __name__ == "__main__":
    main()
