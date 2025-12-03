from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import duckdb

from pythia.db.schema import ensure_schema


def import_questions(con, path: Path, run_id: str | None = None) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        questions = data
    else:
        questions = data.get("questions") or []

    logging.info("Importing %d candidate questions from %s", len(questions), path)

    inserted = 0
    for q in questions:
        qid = str(q.get("question_id") or q.get("id") or "")
        iso3 = (q.get("iso3") or "").upper()
        hazard_code = (q.get("hazard_code") or q.get("hazard") or "").upper()
        metric = (q.get("metric") or "").upper()

        if not qid or not iso3 or not hazard_code or not metric:
            continue

        target_month = str(q.get("target_month") or "")
        window_start = str(q.get("window_start_date") or "")
        window_end = str(q.get("window_end_date") or "")
        wording = str(q.get("wording") or q.get("question") or "")

        con.execute(
            """
            INSERT INTO questions (
                question_id, iso3, hazard_code, metric,
                target_month, window_start_date, window_end_date,
                wording, status, hs_run_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active', ?)
            """,
            [
                qid,
                iso3,
                hazard_code,
                metric,
                target_month,
                window_start,
                window_end,
                wording,
                run_id or q.get("hs_run_id") or "",
            ],
        )
        inserted += 1

    logging.info("Inserted %d questions into questions table", inserted)


def main() -> None:
    parser = argparse.ArgumentParser(description="Import questions from a JSON file into DuckDB")
    parser.add_argument("--db", default="duckdb:///data/resolver.duckdb", help="DuckDB URL")
    parser.add_argument("--path", default="data/test_questions.json", help="Path to questions JSON")
    parser.add_argument("--run-id", default=None, help="Optional HS run_id to associate with questions")
    args = parser.parse_args()

    db_url = args.db
    if db_url.startswith("duckdb:///"):
        db_path = db_url[len("duckdb:///") :]
    else:
        db_path = db_url

    con = duckdb.connect(db_path, read_only=False)
    ensure_schema(con)
    try:
        import_questions(con, Path(args.path), run_id=args.run_id)
    finally:
        con.close()


if __name__ == "__main__":
    main()
