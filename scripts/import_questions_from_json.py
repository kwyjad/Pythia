from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from pythia.db.schema import connect as pythia_connect, ensure_schema


def import_questions(path: Path, run_id: str | None = None) -> None:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    questions: List[Dict[str, Any]]
    if isinstance(data, list):
        questions = data
    else:
        questions = data.get("questions", []) if isinstance(data, dict) else []

    if not questions:
        return

    con = pythia_connect(read_only=False)
    ensure_schema(con)
    try:
        for q in questions:
            qid = str(q.get("question_id") or q.get("id") or "")
            iso3 = str(q.get("iso3") or "")
            hazard_code = str(q.get("hazard_code") or "")
            metric = str(q.get("metric") or "")
            target_month = str(q.get("target_month") or "")
            window_start = str(q.get("window_start_date") or "")
            window_end = str(q.get("window_end_date") or "")
            wording = str(q.get("wording") or "")

            if not qid or not iso3 or not hazard_code or not metric:
                continue

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
    finally:
        con.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Import questions from a JSON file into DuckDB")
    parser.add_argument("--db", default=None, help="DuckDB URL, e.g., duckdb:///data/resolver.duckdb")
    parser.add_argument("--path", default="data/test_questions.json", help="Path to questions JSON")
    parser.add_argument("--run-id", default=None, help="Optional HS run_id to associate with questions")
    args = parser.parse_args()

    if args.db:
        os.environ["PYTHIA_DB_URL"] = args.db

    import_questions(Path(args.path), run_id=args.run_id)


if __name__ == "__main__":
    main()
