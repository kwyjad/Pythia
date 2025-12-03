from __future__ import annotations

import argparse

import duckdb

from pythia.db.schema import ensure_schema


def _latest_hs_run_id(con) -> str | None:
    row = con.execute(
        """
        SELECT run_id
        FROM hs_triage
        GROUP BY run_id
        ORDER BY MAX(created_at) DESC
        LIMIT 1
        """
    ).fetchone()
    return row[0] if row else None


def _metrics_for_hazard(hz: str) -> list[tuple[str, str]]:
    hz = hz.upper()
    if hz in {"ACO", "ACE", "CU"}:
        return [("FATALITIES", "ACLED"), ("PA", "IDMC")]
    if hz in {"DR", "FL", "TC", "HW"}:
        return [("PA", "EM-DAT")]
    if hz == "DI":
        return [("PA", "NONE")]
    return []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="duckdb:///data/resolver.duckdb")
    args = parser.parse_args()

    db_url = args.db
    if db_url.startswith("duckdb:///"):
        db_path = db_url[len("duckdb:///") :]
    else:
        db_path = db_url

    con = duckdb.connect(db_path, read_only=False)
    ensure_schema(con)
    try:
        hs_run_id = _latest_hs_run_id(con)
        if not hs_run_id:
            print("No hs_triage rows found; nothing to create.")
            return

        rows = con.execute(
            """
            SELECT iso3, hazard_code, tier, triage_score
            FROM hs_triage
            WHERE run_id = ? AND need_full_spd = TRUE
            """,
            [hs_run_id],
        ).fetchall()
        inserted = 0
        for iso3, hz, tier, score in rows:
            for metric, _res_src in _metrics_for_hazard(hz):
                qid = f"{iso3}_{hz}_{metric}"
                target_month = ""
                con.execute(
                    """
                    INSERT INTO questions (
                        question_id, hs_run_id, iso3, hazard_code, metric,
                        target_month, window_start_date, window_end_date,
                        wording, status
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'active')
                    ON CONFLICT(question_id) DO NOTHING
                    """,
                    [
                        qid,
                        hs_run_id,
                        iso3,
                        hz,
                        metric,
                        target_month,
                        "",
                        "",
                        f"{iso3} / {hz} / {metric} auto-generated from HS triage",
                    ],
                )
                inserted += 1
        print(f"Created {inserted} questions from hs_triage run_id={hs_run_id}")
    finally:
        con.close()


if __name__ == "__main__":
    main()
