# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Did a stored Sibyl forecast rest on evidence?

``sibyl_forecasts.evidence_ok`` says so. ``sibyl.run`` writes it for every
row it stores from October 2026 on. Rows written before then carry NULL, and
``backfill_evidence_ok`` fills them from the trial traces kept in
``trials_json``: a stored forecast had evidence when at least two of its
trials made at least one search whose tool call succeeded.

Why this exists: in the production run of 15 July 2026
(``sibyl_1784113515141``) the Brave circuit breaker tripped on the first
three calls and all 216 searches failed, yet Sibyl stored ten forecasts with
status ``ok``. All six of Sibyl's scored forecasts came from that run. The
rows and their scores are kept; every reader of Sibyl's record (the advice
loop, the head-to-head comparison, the Sibyl API, the interpreter's second
opinion) leaves out the ones flagged FALSE.

The backfill is idempotent (it touches NULL rows only) and never raises: it
runs at the start of ``sibyl.run.run_sibyl`` and of ``sibyl.advice.generate``.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Iterable

logger = logging.getLogger(__name__)

#: Trials with a successful search needed for a legacy row to count.
LEGACY_MIN_TRIALS_WITH_SEARCH = 2


def _has_column(con, table: str, column: str) -> bool:
    try:
        cols = {str(r[1]).lower() for r in con.execute(f"PRAGMA table_info('{table}')").fetchall()}
    except Exception:  # noqa: BLE001
        return False
    return column.lower() in cols


def trial_had_successful_search(trial: Any) -> bool:
    """One stored trial dict: did any search step have ``tool_ok`` true?"""
    if not isinstance(trial, dict):
        return False
    for step in trial.get("belief_trace") or []:
        if not isinstance(step, dict):
            continue
        if step.get("action") == "brave_search" and step.get("tool_ok") is True:
            return True
    return False


def legacy_evidence_ok(trials: Iterable[Any]) -> bool:
    """The backfill rule over a stored ``trials_json`` list."""
    n = sum(1 for t in (trials or []) if trial_had_successful_search(t))
    return n >= LEGACY_MIN_TRIALS_WITH_SEARCH


def backfill_evidence_ok(con) -> int:
    """Fill ``evidence_ok`` on rows that carry NULL. Returns rows updated.

    Adds the column when an older database lacks it. Never raises.
    """
    try:
        try:
            tables = {str(r[0]).lower() for r in con.execute(
                "SELECT table_name FROM information_schema.tables"
            ).fetchall()}
        except Exception:  # noqa: BLE001
            tables = set()
        if "sibyl_forecasts" not in tables:
            return 0
        if not _has_column(con, "sibyl_forecasts", "evidence_ok"):
            con.execute("ALTER TABLE sibyl_forecasts ADD COLUMN evidence_ok BOOLEAN")
        rows = con.execute(
            "SELECT sibyl_run_id, question_id, trials_json FROM sibyl_forecasts "
            "WHERE evidence_ok IS NULL"
        ).fetchall()
        updated = 0
        flagged = 0
        for srid, qid, raw in rows:
            try:
                trials = json.loads(raw) if isinstance(raw, str) else (raw or [])
            except (TypeError, ValueError):
                trials = []
            ok = legacy_evidence_ok(trials if isinstance(trials, list) else [])
            con.execute(
                "UPDATE sibyl_forecasts SET evidence_ok = ? "
                "WHERE sibyl_run_id IS NOT DISTINCT FROM ? AND question_id IS NOT DISTINCT FROM ? "
                "AND evidence_ok IS NULL",
                [ok, srid, qid],
            )
            updated += 1
            flagged += 0 if ok else 1
        if updated:
            logger.info(
                "sibyl.evidence: backfilled evidence_ok on %d row(s); %d flagged "
                "as resting on no evidence", updated, flagged,
            )
        return updated
    except Exception as exc:  # noqa: BLE001 - a backfill must never stop a run
        logger.warning("sibyl.evidence: evidence_ok backfill failed: %s", exc)
        return 0


def evidence_filter_sql(con, alias: str = "") -> str:
    """`` AND COALESCE(<alias>.evidence_ok, TRUE)`` when the column exists.

    NULL (not yet backfilled) counts as evidence, so a reader never hides a
    row merely because the backfill has not run on its database.
    """
    if not _has_column(con, "sibyl_forecasts", "evidence_ok"):
        return ""
    prefix = f"{alias}." if alias else ""
    return f" AND COALESCE({prefix}evidence_ok, TRUE)"
