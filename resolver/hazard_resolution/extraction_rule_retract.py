# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Retract frozen ReliefWeb answers the November 2026 figure rules reject.

The rules in :mod:`figures` (a figure is dated by what it says, a stated
people count beats a household conversion, one figure one hazard) reach
every cell still before its freeze deadline on the next Resolver Update. A
cell already frozen keeps its answer: a re-walk only logs a revision. The
August 2026 flood and cyclone cells froze on 2026-10-30, and two carry the
same ReliefWeb figure under both hazards (Laos 187,734 and 403,133).

This is the audited, predicate-scoped exception for them, on the pattern of
:mod:`sweep_era_rewalk`. The predicate:

* hazard FL or TC, status ``RESOLVED_VALUE``, ``rule_fired`` naming the
  ReliefWeb rung, frozen (past its deadline);
* the stored winning figure is REJECTED when the cell's cached extractions
  are replayed through the current rules (replayed with no network and no
  model call, :func:`quote_period_audit.replay_cell`); a winner that is
  merely outranked is not selected;
* some question resolves from the cell: a ``questions`` row for the
  country and hazard, metric PA, whose six-month window holds the month.

An audit run (``--audit-only``) writes the plan and an audit copy and
changes nothing. An apply run, in one transaction, deletes the rows, stamps
the trigger rows ``no_row_reason = extraction_rule_retract_pending``, frees
their (hazard, month) resume-ledger rows under a one-shot
``haz_feed_restale`` token so the nightly backcast re-walks them once, and
compares counts and checksums over every untouched row before and after,
exiting 1 on a mismatch. Kevin decides whether to apply it.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import logging
from pathlib import Path
from typing import Any

LOG = logging.getLogger(__name__)

HAZARDS = ("FL", "TC")
FEED = "extraction_rule_retract"
TOKEN = "figure-rules-2026-11"
PENDING_REASON = "extraction_rule_retract_pending"
KEYS_TABLE = "_extraction_rule_keys"
MONTHS_TABLE = "_extraction_rule_months"


def _tables(con) -> set[str]:
    return {str(r[0]) for r in con.execute(
        "SELECT table_name FROM information_schema.tables"
    ).fetchall()}


def _question_cells(con) -> set[tuple[str, str, str]]:
    """(iso3, hazard, ym) for every PA question month on FL or TC."""

    if "questions" not in _tables(con):
        return set()
    rows = con.execute(
        """
        SELECT upper(iso3), upper(hazard_code), CAST(window_start_date AS DATE)
        FROM questions
        WHERE upper(hazard_code) IN ('FL', 'TC') AND upper(metric) = 'PA'
          AND window_start_date IS NOT NULL
        """
    ).fetchall()
    cells: set[tuple[str, str, str]] = set()
    for iso3, hazard, start in rows:
        if start is None:
            continue
        index = start.year * 12 + start.month - 1
        for offset in range(6):
            i = index + offset
            cells.add((str(iso3), str(hazard), f"{i // 12:04d}-{i % 12 + 1:02d}"))
    return cells


def select_rows(con, rulebook, *, today: dt.date | None = None) -> list[dict[str, Any]]:
    """The frozen rows the predicate selects, with the reason each fails."""

    from resolver.hazard_resolution.quote_period_audit import (
        replay_cell,
        stored_resolution,
        why_stored_winner_failed,
    )

    today = today or dt.date.today()
    question_cells = _question_cells(con)
    candidates = con.execute(
        """
        SELECT iso3, hazard, year, month, value, frozen_at
        FROM haz_resolutions
        WHERE hazard IN ('FL', 'TC') AND status = 'RESOLVED_VALUE'
          AND rule_fired LIKE '%reliefweb_extracted%'
          AND frozen_at IS NOT NULL AND CAST(frozen_at AS DATE) < ?
        ORDER BY hazard, year, month, iso3
        """,
        [today],
    ).fetchall()
    selected: list[dict[str, Any]] = []
    for iso3, hazard, year, month, value, frozen_at in candidates:
        ym = f"{int(year):04d}-{int(month):02d}"
        if (str(iso3).upper(), str(hazard).upper(), ym) not in question_cells:
            continue
        stored = stored_resolution(con, iso3, ym, hazard) or {}
        try:
            replay = replay_cell(
                con, iso3=iso3, ym=ym, hazard=hazard, rulebook=rulebook, with_ladder=False
            )
        except Exception as exc:
            LOG.warning("[retract] replay %s %s %s failed: %s", iso3, hazard, ym, exc)
            continue
        reason = why_stored_winner_failed(stored, replay)
        if not reason or reason in ("outranked", "not_extracted"):
            continue
        selected.append({
            "iso3": str(iso3), "hazard": str(hazard), "year": int(year), "month": int(month),
            "ym": ym, "value": float(value) if value is not None else None,
            "frozen_at": frozen_at.isoformat() if hasattr(frozen_at, "isoformat") else frozen_at,
            "reason": reason, "winning_quote": stored.get("winning_quote"),
        })
    return selected


def stage_selection(con, rows: list[dict[str, Any]]) -> None:
    con.execute(f"DROP TABLE IF EXISTS {KEYS_TABLE}")
    con.execute(f"DROP TABLE IF EXISTS {MONTHS_TABLE}")
    con.execute(
        f"CREATE TEMP TABLE {KEYS_TABLE} (iso3 TEXT, hazard TEXT, year INTEGER, month INTEGER)"
    )
    con.execute(f"CREATE TEMP TABLE {MONTHS_TABLE} (hazard TEXT, ym TEXT)")
    for row in rows:
        con.execute(
            f"INSERT INTO {KEYS_TABLE} VALUES (?, ?, ?, ?)",
            [row["iso3"], row["hazard"], row["year"], row["month"]],
        )
    for hazard, ym in sorted({(r["hazard"], r["ym"]) for r in rows}):
        con.execute(f"INSERT INTO {MONTHS_TABLE} VALUES (?, ?)", [hazard, ym])


def plan_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_hazard: dict[str, int] = {}
    by_reason: dict[str, int] = {}
    for row in rows:
        by_hazard[row["hazard"]] = by_hazard.get(row["hazard"], 0) + 1
        by_reason[row["reason"]] = by_reason.get(row["reason"], 0) + 1
    return {
        "rows": len(rows),
        "by_hazard": by_hazard,
        "by_reason": by_reason,
        "months": sorted({f"{r['hazard']} {r['ym']}" for r in rows}),
        "cells": rows,
    }


def export_audit(con, rows: list[dict[str, Any]], audit_dir: Path) -> dict[str, int]:
    audit_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, int] = {}
    for table in ("haz_resolutions", "haz_triggers"):
        cur = con.execute(
            f"""
            SELECT x.* FROM {table} x JOIN {KEYS_TABLE} k
              ON k.iso3 = x.iso3 AND k.year = x.year AND k.month = x.month AND k.hazard = x.hazard
            ORDER BY x.hazard, x.year, x.month, x.iso3
            """
        )
        cols = [d[0] for d in cur.description]
        data = cur.fetchall()
        with (audit_dir / f"{table}.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(cols)
            writer.writerows(data)
        written[table] = len(data)
    with (audit_dir / "plan.json").open("w", encoding="utf-8") as fh:
        json.dump(plan_summary(rows), fh, indent=2, default=str)
    with (audit_dir / "README.txt").open("w", encoding="utf-8") as fh:
        fh.write(
            "Rows selected by compact_resolver_db.yml maintenance=extraction_rule_retract.\n"
            "Predicate: see resolver/hazard_resolution/extraction_rule_retract.py.\n"
            f"Rows: {json.dumps(written)}\n"
        )
    return written


_CELL_EXCLUDE = (
    f"SELECT 1 FROM {KEYS_TABLE} k WHERE k.iso3 = x.iso3 AND k.year = x.year "
    f"AND k.month = x.month AND k.hazard = x.hazard"
)
_MONTH_EXCLUDE = f"SELECT 1 FROM {MONTHS_TABLE} m WHERE m.hazard = x.hazard AND m.ym = x.ym"


def integrity_report(con) -> dict[str, Any]:
    from resolver.hazard_resolution.sweep_era_rewalk import _checksum

    tables = _tables(con)
    report: dict[str, Any] = {
        "counts": [
            {"hazard": h, "status": s, "rows": int(n)}
            for h, s, n in con.execute(
                "SELECT hazard, status, COUNT(*) FROM haz_resolutions GROUP BY 1, 2 ORDER BY 1, 2"
            ).fetchall()
        ],
        "checksums": {
            "haz_resolutions_untouched": _checksum(
                con, "haz_resolutions", "x.iso3, x.year, x.month, x.hazard", _CELL_EXCLUDE
            ),
            "haz_triggers_untouched": _checksum(
                con, "haz_triggers", "x.iso3, x.year, x.month, x.hazard", _CELL_EXCLUDE
            ),
        },
        "trigger_rows": int(con.execute("SELECT COUNT(*) FROM haz_triggers").fetchone()[0]),
    }
    if "haz_backcast_progress" in tables:
        report["checksums"]["haz_backcast_progress_untouched"] = _checksum(
            con, "haz_backcast_progress", "x.hazard, x.ym", _MONTH_EXCLUDE
        )
    if "haz_revisions" in tables:
        report["checksums"]["haz_revisions_whole"] = _checksum(
            con, "haz_revisions",
            "x.iso3, x.year, x.month, x.hazard, x.source, x.observed_at, x.new_value", None,
        )
        report["revisions_rows"] = int(
            con.execute("SELECT COUNT(*) FROM haz_revisions").fetchone()[0]
        )
    return report


def already_applied(con) -> bool:
    if "haz_feed_restale" not in _tables(con):
        return False
    return bool(con.execute(
        "SELECT COUNT(*) FROM haz_feed_restale WHERE feed = ? AND token = ?", [FEED, TOKEN]
    ).fetchone()[0])


def apply_retraction(con) -> dict[str, Any]:
    """Delete the staged rows, stamp their triggers, free their ledger months."""

    from resolver.hazard_resolution.schema import ensure_haz_schema

    ensure_haz_schema(con)
    if already_applied(con):
        return {"applied": False, "reason": f"token {TOKEN} already applied"}
    n = int(con.execute(f"SELECT COUNT(*) FROM {KEYS_TABLE}").fetchone()[0])
    if not n:
        return {"applied": False, "reason": "nothing selected"}
    con.execute("BEGIN TRANSACTION")
    try:
        deleted = con.execute(
            f"""
            SELECT r.hazard, r.status, COUNT(*) FROM haz_resolutions r JOIN {KEYS_TABLE} k
              ON k.iso3 = r.iso3 AND k.year = r.year AND k.month = r.month AND k.hazard = r.hazard
            GROUP BY 1, 2
            """
        ).fetchall()
        con.execute(
            f"""
            DELETE FROM haz_resolutions r WHERE EXISTS (
                SELECT 1 FROM {KEYS_TABLE} k WHERE k.iso3 = r.iso3 AND k.year = r.year
                  AND k.month = r.month AND k.hazard = r.hazard)
            """
        )
        patch = json.dumps({"no_row_reason": PENDING_REASON})
        con.execute(
            f"""
            UPDATE haz_triggers t SET trigger_detail_json = CAST(json_merge_patch(
                CASE WHEN json_valid(t.trigger_detail_json)
                          AND json_type(t.trigger_detail_json) = 'OBJECT'
                     THEN t.trigger_detail_json ELSE '{{}}' END, ?) AS VARCHAR)
            WHERE EXISTS (SELECT 1 FROM {KEYS_TABLE} k WHERE k.iso3 = t.iso3
                  AND k.year = t.year AND k.month = t.month AND k.hazard = t.hazard)
            """,
            [patch],
        )
        freed: dict[str, int] = {}
        for (hazard,) in con.execute(
            f"SELECT DISTINCT hazard FROM {MONTHS_TABLE} ORDER BY 1"
        ).fetchall():
            months = [str(r[0]) for r in con.execute(
                f"SELECT ym FROM {MONTHS_TABLE} WHERE hazard = ? ORDER BY ym", [hazard]
            ).fetchall()]
            held = 0
            if "haz_backcast_progress" in _tables(con):
                held = int(con.execute(
                    f"""
                    SELECT COUNT(*) FROM haz_backcast_progress p JOIN {MONTHS_TABLE} m
                      ON m.hazard = p.hazard AND m.ym = p.ym WHERE p.hazard = ?
                    """,
                    [hazard],
                ).fetchone()[0])
                con.execute(
                    f"""
                    DELETE FROM haz_backcast_progress p WHERE p.hazard = ? AND EXISTS (
                        SELECT 1 FROM {MONTHS_TABLE} m WHERE m.hazard = p.hazard AND m.ym = p.ym)
                    """,
                    [hazard],
                )
            con.execute(
                "INSERT INTO haz_feed_restale (feed, hazard, token, months, rows_deleted) "
                "VALUES (?, ?, ?, ?, ?)",
                [FEED, hazard, TOKEN, ",".join(months), held],
            )
            freed[str(hazard)] = held
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    return {
        "applied": True,
        "deleted": {f"{h}/{s}": int(c) for h, s, c in deleted},
        "ledger_rows_freed": freed,
    }


def compare_reports(before: dict[str, Any], after: dict[str, Any], deleted: dict[str, int]) -> list[str]:
    from resolver.hazard_resolution.sweep_era_rewalk import compare_reports as _compare

    keyed = {tuple(k.split("/", 1)): v for k, v in deleted.items()}
    problems = _compare(before, after, keyed)
    # The trigger rows are stamped, not removed: their count must not move,
    # and _compare already asserts that.
    return problems


def format_plan(plan: dict[str, Any]) -> list[str]:
    lines = [
        f"Frozen FL/TC ReliefWeb answers the current figure rules reject, on cells a "
        f"question resolves from: {plan['rows']} "
        f"(FL {plan['by_hazard'].get('FL', 0)}, TC {plan['by_hazard'].get('TC', 0)}).",
        "",
        "| reason | rows |",
        "|---|---:|",
    ]
    for reason, n in sorted(plan["by_reason"].items()):
        lines.append(f"| {reason} | {n} |")
    lines += ["", "| cell | value | frozen_at | reason | winning quote |", "|---|---:|---|---|---|"]
    for row in plan["cells"]:
        quote = str(row.get("winning_quote") or "").replace("|", "/")
        quote = quote if len(quote) <= 110 else quote[:109] + "…"
        value = row.get("value")
        lines.append(
            f"| {row['iso3']} {row['hazard']} {row['ym']} | "
            f"{value:,.0f} | {str(row.get('frozen_at') or '')[:10]} | {row['reason']} | {quote} |"
            if value is not None else
            f"| {row['iso3']} {row['hazard']} {row['ym']} | | | {row['reason']} | {quote} |"
        )
    return lines


def main(argv: list[str] | None = None) -> int:
    import argparse

    import duckdb

    from resolver.hazard_resolution import schema as schema_mod
    from resolver.hazard_resolution.rulebook import load_rulebook

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", required=True)
    parser.add_argument("--audit-dir", default=None)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--audit-only", action="store_true")
    mode.add_argument("--apply", action="store_true")
    parser.add_argument("--read-only", action="store_true",
                        help="open the DB read-only (audit only; no temp-table export)")
    parser.add_argument("--today", default=None)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--md-out", default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    if args.apply and args.read_only:
        parser.error("--apply cannot be read-only")
    rulebook = load_rulebook()
    today = dt.date.fromisoformat(args.today) if args.today else None
    con = duckdb.connect(args.db, read_only=bool(args.read_only))
    if args.read_only:
        schema_mod._ENSURED_CONNECTIONS.add(con)
    try:
        rows = select_rows(con, rulebook, today=today)
        plan = plan_summary(rows)
        out: dict[str, Any] = {"plan": plan, "token": TOKEN}
        if not args.read_only:
            stage_selection(con, rows)
            if args.audit_dir:
                out["audit_rows"] = export_audit(con, rows, Path(args.audit_dir))
        if args.apply:
            before = integrity_report(con)
            result = apply_retraction(con)
            out["result"] = result
            if result.get("applied"):
                after = integrity_report(con)
                problems = compare_reports(before, after, result.get("deleted") or {})
                out["integrity"] = {"before": before, "after": after, "problems": problems}
                if problems:
                    LOG.error("[retract] integrity check failed: %s", problems)
                    _write(args, out)
                    return 1
        _write(args, out)
        return 0
    finally:
        con.close()


def _write(args, out: dict[str, Any]) -> None:
    lines = format_plan(out["plan"])
    result = out.get("result")
    if result:
        lines += ["", f"Apply: {json.dumps(result, default=str)}"]
    text = "\n".join(lines) + "\n"
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(out, indent=2, default=str))
    if args.md_out:
        Path(args.md_out).write_text(text)
    else:
        print(text)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
