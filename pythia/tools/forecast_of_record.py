# Pythia / Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
"""Which run's forecast of a question stands: the forecast of record.

Question ids are keyed by forecast epoch (``SOM_ACE_PA_2026-11``), so every
run whose window opens in the same month forecasts the same question ids.
The 13 October 2026 run opens epoch 2026-11, the epoch of the 1 October run
(``fc_1790831584``), and re-asks most of its questions.

The rule, one implementation for every reader that learns from or reports
on forecasts:

* The forecast of record for a question is the one from the LATEST
  PRODUCTION run that wrote a ``forecasts_ensemble`` row for it. Run ids are
  ``fc_<unix seconds>``, so the greatest id is the newest run.
* A question a later run did not ask again keeps its earlier forecast: the
  rule is per question, never "the latest run".
* A test run never supplies the forecast of record of a production reader,
  however late it ran (the epoch-2026-11 test runs of 2, 6 and 7 October
  2026 all postdate the 1 October production run).
* Every run's rows stay in the tables. Scores, deviations and the per-run
  analyses (RC split arms, advice arms, prompt versions) can still select
  any run by its id; only readers that count a question once use this rule.

Rows with no run id (the ``__ext_*`` reference forecasters) are kept by
:func:`record_run_clause`, since they are not tied to a run.
"""

from __future__ import annotations

from typing import Any, Dict

#: The table whose rows decide which runs forecast a question.
AUTHORITY_TABLE = "forecasts_ensemble"


def _has_column(con: Any, table: str, column: str) -> bool:
    try:
        rows = con.execute(f"PRAGMA table_info('{table}')").fetchall()
    except Exception:  # noqa: BLE001 - absent table reads as absent column
        return False
    return any(str(r[1]).lower() == column.lower() for r in rows)


def record_run_subquery(con: Any, question_expr: str, *, include_test: bool = False,
                        authority: str = AUTHORITY_TABLE) -> str:
    """SQL scalar subquery: the forecast-of-record run id of ``question_expr``."""
    not_test = (
        " AND NOT COALESCE(_rec.is_test, FALSE)"
        if not include_test and _has_column(con, authority, "is_test") else ""
    )
    return (
        f"(SELECT MAX(_rec.run_id) FROM {authority} _rec "
        f"WHERE _rec.question_id = {question_expr} "
        f"AND _rec.run_id IS NOT NULL AND _rec.run_id <> ''{not_test})"
    )


def record_run_clause(con: Any, alias: str, *, include_test: bool = False,
                      keep_runless: bool = True, authority: str | None = None) -> str:
    """`` AND ...`` keeping only rows of ``alias`` from their question's run of
    record (plus run-less rows when ``keep_runless``).

    ``alias`` must carry ``question_id`` and ``run_id``. Empty when the
    authority table has no ``run_id`` column, so an older database degrades to
    counting every run rather than failing every query. ``authority`` falls
    back to ``alias``'s own table when ``forecasts_ensemble`` is absent.
    """
    table = authority or AUTHORITY_TABLE
    if not _has_column(con, table, "run_id"):
        return ""
    sub = record_run_subquery(con, f"{alias}.question_id", include_test=include_test,
                              authority=table)
    if keep_runless:
        return f" AND ({alias}.run_id IS NULL OR {alias}.run_id = {sub})"
    return f" AND {alias}.run_id = {sub}"


def record_runs(con: Any, *, include_test: bool = False) -> Dict[str, str]:
    """``{question_id: run of record}`` for every question with a forecast."""
    if not _has_column(con, AUTHORITY_TABLE, "run_id"):
        return {}
    not_test = (
        " AND NOT COALESCE(is_test, FALSE)"
        if not include_test and _has_column(con, AUTHORITY_TABLE, "is_test") else ""
    )
    rows = con.execute(
        f"SELECT question_id, MAX(run_id) FROM {AUTHORITY_TABLE} "
        f"WHERE run_id IS NOT NULL AND run_id <> ''{not_test} GROUP BY 1"
    ).fetchall()
    return {str(q): str(r) for q, r in rows if q and r}


__all__ = [
    "AUTHORITY_TABLE",
    "record_run_clause",
    "record_run_subquery",
    "record_runs",
]
