from __future__ import annotations

import sys
import duckdb

from pythia.db.schema import get_db_url


def _resolve_db_path() -> str:
    db_url = get_db_url()
    if db_url.startswith("duckdb:///"):
        return db_url[len("duckdb:///") :]
    return db_url


def _get_latest_fc_run_id(con) -> str | None:
    row = con.execute(
        """
        SELECT run_id
        FROM forecasts_ensemble
        WHERE run_id LIKE 'fc_%'
        ORDER BY coalesce(created_at, CURRENT_TIMESTAMP) DESC, run_id DESC
        LIMIT 1
        """
    ).fetchone()
    if row and row[0]:
        return row[0]

    row = con.execute(
        """
        SELECT run_id
        FROM forecasts_ensemble
        ORDER BY coalesce(created_at, CURRENT_TIMESTAMP) DESC, run_id DESC
        LIMIT 1
        """
    ).fetchone()
    return row[0] if row and row[0] else None


def main() -> None:
    db_path = _resolve_db_path()
    if not db_path:
        print("No database URL configured (PYTHIA_DB_URL).")
        sys.exit(1)

    con = duckdb.connect(db_path, read_only=True)

    run_id = _get_latest_fc_run_id(con)
    if not run_id:
        print("No forecasts_ensemble rows found; nothing to summarise.")
        con.close()
        sys.exit(0)

    print(f"Pythia v2 run summary (run_id={run_id})")

    hs_row = con.execute(
        """
        SELECT DISTINCT q.hs_run_id
        FROM forecasts_ensemble fe
        JOIN questions q ON q.question_id = fe.question_id
        WHERE fe.run_id = ? AND q.hs_run_id IS NOT NULL
        ORDER BY q.hs_run_id DESC
        LIMIT 1
        """,
        [run_id],
    ).fetchone()
    hs_run_id = hs_row[0] if hs_row and hs_row[0] else None
    if hs_run_id:
        triage_count = con.execute(
            "SELECT COUNT(*) FROM hs_triage WHERE run_id = ?", [hs_run_id]
        ).fetchone()[0]
        print(f"Linked hs_run_id: {hs_run_id} (hs_triage rows: {triage_count})")
    else:
        print("Linked hs_run_id: (none)")

    questions = con.execute(
        """
        SELECT DISTINCT q.iso3, q.hazard_code, q.metric, q.question_id
        FROM questions q
        WHERE q.status = 'active'
        ORDER BY q.iso3, q.hazard_code, q.metric, q.question_id
        """
    ).fetchall()

    with_spd_rows = con.execute(
        """
        SELECT DISTINCT q.question_id
        FROM questions q
        JOIN forecasts_ensemble fe ON fe.question_id = q.question_id
        WHERE fe.run_id = ?
          AND fe.status = 'ok'
          AND q.status = 'active'
        """,
        [run_id],
    ).fetchall()

    qids_with_spd = {row[0] for row in with_spd_rows}
    print(f"\nActive questions total: {len(questions)}")
    print(
        f"Active questions with SPD rows for run_id={run_id}: {len(qids_with_spd)}"
    )

    q_stats = con.execute(
        """
        SELECT iso3, hazard_code, metric, COUNT(*) AS n
        FROM forecasts_ensemble
        WHERE run_id = ?
          AND status = 'ok'
        GROUP BY iso3, hazard_code, metric
        ORDER BY iso3, hazard_code, metric
        """,
        [run_id],
    ).fetchall()

    print("\nQuestions with forecasts (status='ok'):")
    if not q_stats:
        print("  (none)")
    else:
        for iso3, hz, metric, n in q_stats:
            print(f"  {iso3}/{hz}/{metric}: {n} rows")

    nf_stats = con.execute(
        """
        SELECT q.iso3, q.hazard_code, q.metric, COUNT(*) AS n
        FROM questions AS q
        LEFT JOIN forecasts_ensemble AS f
          ON f.run_id = ?
         AND f.question_id = q.question_id
         AND f.status = 'ok'
        WHERE q.status = 'active'
        GROUP BY q.iso3, q.hazard_code, q.metric
        HAVING n = 0
        ORDER BY q.iso3, q.hazard_code, q.metric
        """,
        [run_id],
    ).fetchall()

    print("\nActive questions with no SPD rows:")
    if not nf_stats:
        print("  (none)")
    else:
        for iso3, hz, metric, _n in nf_stats:
            print(f"  {iso3}/{hz}/{metric}")

    con.close()


if __name__ == "__main__":
    main()
