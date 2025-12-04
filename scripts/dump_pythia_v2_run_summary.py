from __future__ import annotations

import sys
import duckdb

from pythia.db.schema import get_db_url


def _resolve_db_path() -> str:
    db_url = get_db_url()
    if db_url.startswith("duckdb:///"):
        return db_url[len("duckdb:///") :]
    return db_url


def main() -> None:
    db_path = _resolve_db_path()
    if not db_path:
        print("No database URL configured (PYTHIA_DB_URL).")
        sys.exit(1)

    con = duckdb.connect(db_path, read_only=True)

    row = con.execute(
        """
        SELECT run_id, MAX(created_at) AS t
        FROM forecasts_ensemble
        WHERE run_id LIKE 'fc_%'
        GROUP BY run_id
        ORDER BY t DESC
        LIMIT 1
        """
    ).fetchone()

    if not row:
        print("No fc_* runs found in forecasts_ensemble.")
        con.close()
        sys.exit(0)

    run_id = row[0]
    print(f"=== Pythia v2 run summary for run_id={run_id} ===")

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
