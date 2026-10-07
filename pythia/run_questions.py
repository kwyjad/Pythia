# Pythia / Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
"""Which questions a Horizon Scanner run asked for, apart from where a
question came from.

``questions.hs_run_id`` is a question's ORIGIN: the production scan that
created it (or the latest production scan that re-ran its epoch). Question
ids are epoch-keyed, so a second run in the same month asks for the same
questions. A test run must not change a production question (Oct 2026), which
left the run unable to find them: the 6 October 2026 rehearsal
(``hs_20261006T085029``) forecast 2 of its 21 questions, because the other 19
still named the 1 October production scan.

``run_questions`` records a run's question SET, one row per (run, question),
with the track, tier and triage score as THAT run saw them. Question creation
writes it for every run, test or production; the forecaster, the scenario
writer, Sibyl's selection and the bundle builders read it. The forecaster
stamps ``forecaster_run_id`` on the rows it forecast, which is how a forecaster
run is mapped back to the HS run it served.

Every reader also accepts ``questions.hs_run_id = <run>`` (the legacy rule), so
a database that predates the table, or a run the backfill could not
reconstruct, still behaves as before.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

LOG = logging.getLogger(__name__)

TABLE = "run_questions"

RUN_QUESTIONS_DDL = """
CREATE TABLE IF NOT EXISTS run_questions (
    hs_run_id TEXT NOT NULL,
    question_id TEXT NOT NULL,
    iso3 TEXT,
    hazard_code TEXT,
    metric TEXT,
    track INTEGER,
    tier TEXT,
    triage_score DOUBLE,
    metadata_json TEXT,
    is_test BOOLEAN DEFAULT FALSE,
    forecaster_run_id TEXT,
    source TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (hs_run_id, question_id)
)
"""

# What a hazard's triage row asks for, by question metric. Mirrors
# scripts/create_questions_from_triage.py (DR/PA becomes PHASE3PLUS_IN_NEED);
# used only to reconstruct the links of runs written before the table.
_METRICS_BY_HAZARD = {
    "ACE": ("FATALITIES", "PA"),
    "CU": ("PA",),
    "DI": ("PA",),
    "DR": ("PHASE3PLUS_IN_NEED", "EVENT_OCCURRENCE"),
    "FL": ("PA", "EVENT_OCCURRENCE"),
    "TC": ("PA", "EVENT_OCCURRENCE"),
}


def table_exists(con, table: str = TABLE) -> bool:
    try:
        row = con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?", [table]
        ).fetchone()
        return bool(row and row[0])
    except Exception:  # noqa: BLE001
        return False


def _columns(con, table: str) -> set[str]:
    try:
        # PRAGMA table_info yields (cid, name, ...): the NAME is column 1.
        return {str(r[1]).lower() for r in con.execute(f"PRAGMA table_info('{table}')").fetchall()}
    except Exception:  # noqa: BLE001
        return set()


def ensure_run_questions(con) -> None:
    con.execute(RUN_QUESTIONS_DDL)


def record_run_question(
    con,
    *,
    hs_run_id: str,
    question_id: str,
    iso3: Optional[str] = None,
    hazard_code: Optional[str] = None,
    metric: Optional[str] = None,
    track: Optional[int] = None,
    tier: Optional[str] = None,
    triage_score: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
    is_test: bool = False,
    source: str = "question_creation",
) -> None:
    """Link *question_id* to *hs_run_id* as that run saw it. Idempotent.

    A re-run of question creation for the same run refreshes the run's view
    (track, tier, score) and keeps any ``forecaster_run_id`` already stamped.
    """
    ensure_run_questions(con)
    con.execute(
        """
        INSERT INTO run_questions (
            hs_run_id, question_id, iso3, hazard_code, metric, track, tier,
            triage_score, metadata_json, is_test, source, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT (hs_run_id, question_id) DO UPDATE SET
            iso3 = excluded.iso3,
            hazard_code = excluded.hazard_code,
            metric = excluded.metric,
            track = excluded.track,
            tier = excluded.tier,
            triage_score = excluded.triage_score,
            metadata_json = excluded.metadata_json,
            is_test = excluded.is_test,
            source = excluded.source
        """,
        [
            hs_run_id,
            question_id,
            iso3,
            hazard_code,
            metric,
            track,
            tier,
            triage_score,
            json.dumps(metadata, ensure_ascii=False) if metadata is not None else None,
            bool(is_test),
            source,
        ],
    )


def in_run_clause(con, alias: str = "q") -> Tuple[str, int]:
    """SQL predicate selecting the questions of one HS run, and how many
    times the run id must be bound.

    ``(q.hs_run_id = ? OR q.question_id IN (links of ?))`` when the link table
    exists, else the legacy ``q.hs_run_id = ?``.
    """
    if table_exists(con):
        return (
            f"({alias}.hs_run_id = ? OR {alias}.question_id IN "
            f"(SELECT question_id FROM run_questions WHERE hs_run_id = ?))",
            2,
        )
    return f"{alias}.hs_run_id = ?", 1


def run_question_ids(con, hs_run_id: str) -> List[str]:
    """Every question the run asked for (links and legacy origin)."""
    clause, n = in_run_clause(con, "q")
    rows = con.execute(
        f"SELECT DISTINCT q.question_id FROM questions q WHERE {clause} ORDER BY 1",
        [hs_run_id] * n,
    ).fetchall()
    return [str(r[0]) for r in rows]


def stamp_forecaster_run(
    con, *, hs_run_id: str, forecaster_run_id: str, question_ids: Iterable[str]
) -> int:
    """Record that *forecaster_run_id* forecast these questions for *hs_run_id*.

    A question the run reached only through the legacy origin rule (a run
    from before the table) is linked here too, so the mapping holds for it.
    Returns the number of links stamped. Never raises.
    """
    qids = [str(q) for q in question_ids if q]
    if not (hs_run_id and forecaster_run_id and qids):
        return 0
    try:
        ensure_run_questions(con)
        is_test_row = con.execute(
            "SELECT COALESCE(is_test, FALSE) FROM hs_runs WHERE hs_run_id = ?", [hs_run_id]
        ).fetchone() if table_exists(con, "hs_runs") else None
        run_is_test = bool(is_test_row and is_test_row[0])
        for qid in qids:
            con.execute(
                """
                INSERT INTO run_questions (hs_run_id, question_id, iso3, hazard_code,
                                           metric, track, is_test, source, forecaster_run_id)
                SELECT ?, q.question_id, q.iso3, q.hazard_code, q.metric, q.track, ?,
                       'forecaster_legacy_origin', ?
                FROM questions q WHERE q.question_id = ?
                ON CONFLICT (hs_run_id, question_id) DO UPDATE SET
                    forecaster_run_id = excluded.forecaster_run_id
                """,
                [hs_run_id, run_is_test, forecaster_run_id, qid],
            )
        return len(qids)
    except Exception as exc:  # noqa: BLE001 - a link must never stop a forecast
        LOG.warning("run_questions: could not stamp forecaster run %s: %s", forecaster_run_id, exc)
        return 0


def hs_run_for_forecaster_run(con, forecaster_run_id: Optional[str]) -> Optional[str]:
    """The HS run a forecaster run served.

    The link's ``forecaster_run_id`` first. A run from before the table falls
    back to the most common origin among the questions it forecast.
    """
    if not forecaster_run_id:
        return None
    try:
        if table_exists(con) and "forecaster_run_id" in _columns(con, TABLE):
            row = con.execute(
                "SELECT hs_run_id, COUNT(*) AS n FROM run_questions "
                "WHERE forecaster_run_id = ? GROUP BY hs_run_id ORDER BY n DESC, hs_run_id DESC LIMIT 1",
                [forecaster_run_id],
            ).fetchone()
            if row and row[0]:
                return str(row[0])
        for table in ("forecasts_ensemble", "forecasts_raw"):
            if not table_exists(con, table):
                continue
            row = con.execute(
                f"SELECT q.hs_run_id, COUNT(DISTINCT q.question_id) AS n FROM questions q "
                f"JOIN {table} f ON f.question_id = q.question_id "
                f"WHERE f.run_id = ? AND q.hs_run_id IS NOT NULL "
                f"GROUP BY q.hs_run_id ORDER BY n DESC, q.hs_run_id DESC LIMIT 1",
                [forecaster_run_id],
            ).fetchone()
            if row and row[0]:
                return str(row[0])
    except Exception as exc:  # noqa: BLE001
        LOG.warning("run_questions: hs run for %s unresolved: %s", forecaster_run_id, exc)
    return None


def link_overlay(con, alias: str = "q", link_alias: str = "rq") -> Tuple[str, str, str]:
    """(join, hs_run_expr, track_expr) for a query over one forecaster run.

    The join binds the forecaster run id once. Without the table the overlay
    is empty and the expressions read the questions row.
    """
    if table_exists(con) and "forecaster_run_id" in _columns(con, TABLE):
        return (
            f"LEFT JOIN run_questions {link_alias} ON {link_alias}.question_id = {alias}.question_id "
            f"AND {link_alias}.forecaster_run_id = ?",
            f"COALESCE({link_alias}.hs_run_id, {alias}.hs_run_id)",
            f"COALESCE({link_alias}.track, {alias}.track)",
        )
    return "", f"{alias}.hs_run_id", f"{alias}.track"


def latest_hs_run_with_questions(con) -> Optional[str]:
    """The most recent HS run that asked for any active question."""
    has_links = table_exists(con)
    union = (
        "UNION SELECT rq.hs_run_id FROM run_questions rq "
        "JOIN questions q ON q.question_id = rq.question_id WHERE q.status = 'active'"
        if has_links
        else ""
    )
    has_runs = table_exists(con, "hs_runs")
    order_join = "LEFT JOIN hs_runs r ON r.hs_run_id = s.hs_run_id" if has_runs else ""
    order = "r.generated_at DESC NULLS LAST, s.hs_run_id DESC" if has_runs else "s.hs_run_id DESC"
    row = con.execute(
        f"""
        SELECT s.hs_run_id FROM (
            SELECT q.hs_run_id FROM questions q
            WHERE q.status = 'active' AND q.hs_run_id IS NOT NULL
            {union}
        ) s {order_join}
        WHERE s.hs_run_id IS NOT NULL
        ORDER BY {order}
        LIMIT 1
        """
    ).fetchone()
    return str(row[0]) if row and row[0] else None


def backfill_run_questions(con) -> Dict[str, Any]:
    """Reconstruct the links of runs written before the table. Idempotent.

    For every ``hs_runs`` row with no link yet: the questions whose origin is
    that run, plus the questions its triage asked for (a ``need_full_spd`` row
    for the country and hazard, in the epoch the run's month opens, i.e. the
    month after it). Track, tier and score are the run's own triage values.

    Returns counts and the runs that could not be reconstructed (no triage
    rows, or triage asking for nothing that exists in ``questions``).
    """
    report: Dict[str, Any] = {"runs_examined": 0, "runs_linked": 0, "links_written": 0,
                              "unreconstructable": []}
    if not (table_exists(con, "hs_runs") and table_exists(con, "questions")):
        return report
    ensure_run_questions(con)
    runs = con.execute(
        """
        SELECT h.hs_run_id, COALESCE(h.is_test, FALSE), h.generated_at
        FROM hs_runs h
        WHERE h.hs_run_id IS NOT NULL
          AND NOT EXISTS (SELECT 1 FROM run_questions rq WHERE rq.hs_run_id = h.hs_run_id)
        ORDER BY h.hs_run_id
        """
    ).fetchall()
    triage_cols = _columns(con, "hs_triage") if table_exists(con, "hs_triage") else set()
    has_triage = bool(triage_cols)
    track_expr = "t.track" if "track" in triage_cols else "NULL"
    for hs_run_id, is_test, generated_at in runs:
        report["runs_examined"] += 1
        before = con.execute(
            "SELECT COUNT(*) FROM run_questions WHERE hs_run_id = ?", [hs_run_id]
        ).fetchone()[0]
        # Origin rows: the questions this run created or adopted.
        con.execute(
            """
            INSERT INTO run_questions (hs_run_id, question_id, iso3, hazard_code, metric,
                                       track, metadata_json, is_test, source)
            SELECT q.hs_run_id, q.question_id, q.iso3, q.hazard_code, q.metric, q.track,
                   q.pythia_metadata_json, ?, 'backfill:origin'
            FROM questions q WHERE q.hs_run_id = ?
            ON CONFLICT (hs_run_id, question_id) DO NOTHING
            """,
            [bool(is_test), hs_run_id],
        )
        if has_triage:
            # The run's month: hs_YYYYMMDDT... when the id says it, else the
            # generation time. The window opens the month after.
            for hz, metrics in _METRICS_BY_HAZARD.items():
                for metric in metrics:
                    con.execute(
                        f"""
                        INSERT INTO run_questions (hs_run_id, question_id, iso3, hazard_code,
                                                   metric, track, tier, triage_score, is_test, source)
                        SELECT t.run_id, q.question_id, q.iso3, q.hazard_code, q.metric,
                               {track_expr}, t.tier, t.triage_score, ?, 'backfill:hs_triage'
                        FROM hs_triage t
                        JOIN questions q
                          ON upper(q.iso3) = upper(t.iso3)
                         AND upper(q.hazard_code) = upper(t.hazard_code)
                         AND upper(q.metric) = ?
                        WHERE t.run_id = ?
                          AND upper(t.hazard_code) = ?
                          AND COALESCE(t.need_full_spd, FALSE)
                          AND strftime(CAST(q.window_start_date AS DATE), '%Y-%m') = strftime(
                              CAST(COALESCE(
                                  TRY_STRPTIME(substr(t.run_id, 4, 8), '%Y%m%d'),
                                  CAST(? AS TIMESTAMP)
                              ) AS DATE) + INTERVAL 1 MONTH, '%Y-%m')
                        ON CONFLICT (hs_run_id, question_id) DO NOTHING
                        """,
                        [bool(is_test), metric, hs_run_id, hz, generated_at],
                    )
        after = con.execute(
            "SELECT COUNT(*) FROM run_questions WHERE hs_run_id = ?", [hs_run_id]
        ).fetchone()[0]
        written = int(after) - int(before)
        report["links_written"] += written
        if after:
            report["runs_linked"] += 1
        else:
            report["unreconstructable"].append(str(hs_run_id))
    return report


def _main(argv: Optional[List[str]] = None) -> int:
    import argparse  # noqa: PLC0415

    import duckdb  # noqa: PLC0415

    ap = argparse.ArgumentParser(description="Backfill run_questions from what the DB holds.")
    ap.add_argument("--db", required=True)
    args = ap.parse_args(argv)
    path = args.db[len("duckdb:///"):] if args.db.startswith("duckdb:///") else args.db
    con = duckdb.connect(path)
    try:
        report = backfill_run_questions(con)
    finally:
        con.close()
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
