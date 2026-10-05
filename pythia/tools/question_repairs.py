# Pythia / Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
"""Repairs to the questions table that more than one stage must run.

``repair_questions_pointing_at_test_scans`` ran only at question creation,
which happens once a month on the 13th; between cycles the release kept
the 28 production questions a same-epoch test scan had re-pointed (release
of 2026-10-03). It now also runs at the start of every
``compute_resolutions``, which follows every Resolver Update.
"""

from __future__ import annotations

import duckdb


def repair_questions_pointing_at_test_scans(con: duckdb.DuckDBPyConnection) -> dict:
    """Point production questions back at a production scan. Idempotent.

    A production question whose ``hs_run_id`` names a test scan was re-pointed
    by a same-epoch test run before that was stopped. It goes back to the
    latest production scan that triaged the same country and hazard in the
    month before its window opens (the scan that set its epoch). A question
    with no such scan is left alone and counted.
    """
    try:
        bad = con.execute(
            """
            SELECT q.question_id, q.iso3, q.hazard_code,
                   strftime(CAST(q.window_start_date AS DATE) - INTERVAL 1 MONTH, '%Y%m')
            FROM questions q JOIN hs_runs h ON h.hs_run_id = q.hs_run_id
            WHERE NOT COALESCE(q.is_test, FALSE) AND COALESCE(h.is_test, FALSE)
            """
        ).fetchall()
    except Exception as exc:  # noqa: BLE001 - a repair must never stop question creation
        print(f"repair_questions_pointing_at_test_scans: skipped ({exc})")
        return {"repaired": 0, "unrepairable": 0}
    repaired = unrepairable = 0
    for qid, iso3, hz, ym in bad:
        row = con.execute(
            """
            SELECT MAX(t.run_id) FROM hs_triage t JOIN hs_runs h ON h.hs_run_id = t.run_id
            WHERE NOT COALESCE(h.is_test, FALSE) AND t.iso3 = ? AND t.hazard_code = ?
              AND substr(t.run_id, 4, 6) = ?
            """,
            [iso3, hz, ym],
        ).fetchone()
        target = row[0] if row else None
        if not target:
            unrepairable += 1
            continue
        con.execute(
            "UPDATE questions SET hs_run_id = ?, "
            "pythia_metadata_json = CASE WHEN json_valid(pythia_metadata_json) "
            "THEN CAST(json_merge_patch(pythia_metadata_json, json_object('hs_run_id', ?)) AS VARCHAR) "
            "ELSE pythia_metadata_json END "
            "WHERE question_id = ?",
            [target, target, qid],
        )
        repaired += 1
    if bad:
        print(
            f"repair_questions_pointing_at_test_scans: repaired {repaired}, "
            f"left {unrepairable} with no production scan to point at"
        )
    return {"repaired": repaired, "unrepairable": unrepairable}
