# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Retract, once, the flood and cyclone rows written under the old sweep rule.

Until 17 September 2026 the ReliefWeb silence sweep matched any report that
NAMED a country, and a hit promoted the cell to ``triggered=true,
trigger_source='reliefweb_sweep'``. The ladder then walked a cell no detector
had triggered, usually found nothing, and once the freeze deadline passed it
wrote ``NO_DATA`` with ``rule_fired = 'no_data:no_candidate_on_any_rung'``:
a row saying a hazard was detected and no source stated a figure. Under the
current, scoped sweep most of those cells are silent and would be a credible
zero; the rest are undecided and write no row at all. None of them would be
a NO_DATA.

Those rows are frozen, and the freeze guard (``resolutions._frozen_row``)
keeps a re-walk from touching them. The owner approved ONE exception
(2026-10-12), scoped by a predicate and audited: a freeze protects an
answer, and a row that read no source under a rule since withdrawn carries
no answer.

The predicate (:data:`SELECT_SQL`), every clause required:

* ``hazard`` is FL or TC (drought never had the sweep promotion);
* ``status = 'NO_DATA'`` and ``rule_fired = 'no_data:no_candidate_on_any_rung'``;
* ``created_at`` before :data:`SWEEP_FIX_CUTOFF`, the merge of PR #937
  (commit ``e97a136a``, 2026-09-17 08:51:01 UTC), which completed the fix:
  PR #936 (07:24:17 UTC the same morning) stopped the promotion and scoped
  the keyword query, and #937 scoped the taxonomy query, which was where
  every hit came from;
* the provenance shows no rung stated a figure: ``decision.rungs_populated``
  is empty AND the ReliefWeb extraction extracted no figure. A row whose
  extraction read figures and rejected them all against a ceiling DID read
  a source; it belongs to ``impact.reconsider_rejected_cells`` and is left
  alone (29 FL rows in the 2025-08..2026-07 acceptance window);
* the cell's ``haz_triggers`` row says ``triggered = false`` (the post-fix
  re-walk already withdrew the detection) OR still says
  ``trigger_source = 'reliefweb_sweep'`` (the retired promotion). A row on a
  detector trigger (GDACS, IBTrACS) or a ladder-confirmed sweep hit
  (``reliefweb_ladder``) is a detection the current rule still makes.

On apply: the selected ``haz_resolutions`` rows are deleted (``haz_revisions``
is an audit table and is kept whole); their trigger rows are stamped
``no_row_reason = 'sweep_era_rewalk_pending'`` and the ones still carrying
the retired promotion are reset to the current semantics (``triggered =
false``, ``trigger_source = 'none'`` as detect.py writes it); and the affected (hazard, month) rows of
``haz_backcast_progress`` are freed under a one-shot token recorded in
``haz_feed_restale`` (feed ``sweep_era_rewalk``), so the nightly backcast
re-walks those months once, under today's rule, and never again. A re-walked
cell has no row, so the backcast writes its first answer, which the freeze
guard allows (freezing protects existing answers, not late ones).

:func:`integrity_report` is taken before and after: counts by hazard and
status, and checksums over every row the retraction does NOT touch in
``haz_resolutions``, ``haz_triggers`` and ``haz_backcast_progress``, plus
the whole of ``haz_revisions``. Any untouched row that moves, or a count
that moves by other than the rows deleted, is a mismatch and the CLI exits 1
so the workflow stops before the canonical upload.

Usage::

    python -m resolver.hazard_resolution.sweep_era_rewalk --db PATH \\
        --audit-dir audit/sweep_era_rows [--audit-only | --apply] \\
        [--json-out plan.json] [--md-out plan.md] [--read-only]
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    import duckdb

LOG = logging.getLogger(__name__)

#: The merge of PR #937 (commit e97a136a), in UTC: the moment the scoped
#: sweep was complete on main. ``created_at`` is a naive UTC timestamp.
SWEEP_FIX_CUTOFF = "2026-09-17 08:51:01"
SWEEP_FIX_COMMIT = "e97a136a (PR #937, merged 2026-09-17 08:51:01 UTC)"

HAZARDS = ("FL", "TC")
RULE = "no_data:no_candidate_on_any_rung"
RETIRED_SOURCE = "reliefweb_sweep"
#: What detect.py writes for a cell no detector triggered (never NULL).
TRIGGER_SOURCE_NONE = "none"
PENDING_REASON = "sweep_era_rewalk_pending"

#: haz_feed_restale feed name and the one-shot token. The token names the
#: rule being retracted, not the day the run happened, so a second dispatch
#: finds it and frees nothing.
FEED = "sweep_era_rewalk"
TOKEN = "sweep-rule-before-2026-09-17"

KEYS_TABLE = "_sweep_era_keys"
MONTHS_TABLE = "_sweep_era_months"

#: The selector. Every clause is in the module docstring; keep them in step.
SELECT_SQL = f"""
SELECT r.iso3, r.year, r.month, r.hazard,
       printf('%04d-%02d', r.year, r.month) AS ym,
       t.triggered, t.trigger_source
FROM haz_resolutions r
JOIN haz_triggers t
  ON t.iso3 = r.iso3 AND t.year = r.year AND t.month = r.month AND t.hazard = r.hazard
WHERE r.hazard IN ('FL', 'TC')
  AND r.status = 'NO_DATA'
  AND r.rule_fired = '{RULE}'
  AND r.created_at < TIMESTAMP '{SWEEP_FIX_CUTOFF}'
  AND COALESCE(json_array_length(
        json_extract(r.provenance_json, '$.decision.rungs_populated')), 0) = 0
  AND COALESCE(TRY_CAST(json_extract_string(
        r.provenance_json, '$.reliefweb_extraction.extraction.figures_extracted'
      ) AS BIGINT), 0) = 0
  AND (t.triggered = FALSE OR t.trigger_source = '{RETIRED_SOURCE}')
"""

#: Measured silent shares below this many decided cells fall back to this.
DEFAULT_SILENT_SHARE = 0.75
MIN_CELLS_FOR_SHARE = 50


def _tables(con) -> set[str]:
    return {
        str(r[0])
        for r in con.execute(
            "SELECT table_name FROM information_schema.tables"
        ).fetchall()
    }


def stage_selection(con) -> int:
    """Materialise the selected keys and months in temp tables; return the row count."""

    con.execute(f"DROP TABLE IF EXISTS {KEYS_TABLE}")
    con.execute(f"DROP TABLE IF EXISTS {MONTHS_TABLE}")
    con.execute(f"CREATE TEMP TABLE {KEYS_TABLE} AS {SELECT_SQL}")
    con.execute(
        f"CREATE TEMP TABLE {MONTHS_TABLE} AS "
        f"SELECT DISTINCT hazard, ym FROM {KEYS_TABLE}"
    )
    return int(con.execute(f"SELECT COUNT(*) FROM {KEYS_TABLE}").fetchone()[0])


def selection_counts(con) -> dict[str, Any]:
    """Counts by hazard and year, by hazard and trigger state, and months."""

    by_year = [
        {"hazard": h, "year": int(y), "rows": int(n)}
        for h, y, n in con.execute(
            f"SELECT hazard, year, COUNT(*) FROM {KEYS_TABLE} GROUP BY 1, 2 ORDER BY 1, 2"
        ).fetchall()
    ]
    by_trigger = [
        {"hazard": h, "trigger": str(s), "rows": int(n)}
        for h, s, n in con.execute(
            f"""
            SELECT hazard,
                   CASE WHEN trigger_source = '{RETIRED_SOURCE}' THEN 'reliefweb_sweep'
                        ELSE 'triggered=false' END,
                   COUNT(*)
            FROM {KEYS_TABLE} GROUP BY 1, 2 ORDER BY 1, 2
            """
        ).fetchall()
    ]
    months = [
        {"hazard": h, "months": int(n)}
        for h, n in con.execute(
            f"SELECT hazard, COUNT(*) FROM {MONTHS_TABLE} GROUP BY 1 ORDER BY 1"
        ).fetchall()
    ]
    held = 0
    if "haz_backcast_progress" in _tables(con):
        held = int(
            con.execute(
                f"""
                SELECT COUNT(*) FROM haz_backcast_progress p
                JOIN {MONTHS_TABLE} m ON m.hazard = p.hazard AND m.ym = p.ym
                """
            ).fetchone()[0]
        )
    total = sum(r["rows"] for r in by_year)
    return {
        "rows": total,
        "by_hazard": {
            h: sum(r["rows"] for r in by_year if r["hazard"] == h) for h in HAZARDS
        },
        "by_year": by_year,
        "by_trigger": by_trigger,
        "months_by_hazard": months,
        "ledger_rows_to_free": held,
    }


# ---------------------------------------------------------------------------
# Budget estimate
# ---------------------------------------------------------------------------

def _silent_share(con, hazard: str) -> tuple[float, int, str]:
    """Share of non-detector cells the CURRENT sweep finds silent.

    Measured on cells decided after the fix: trigger rows written after the
    cutoff that no detector triggered (``triggered = false`` or a
    ladder-confirmed sweep hit), whose resolution, if any, was also written
    after the cutoff, in months that have ended. A zero is silence; a value
    or no row is a sweep hit the ladder had to read.
    """

    row = con.execute(
        f"""
        SELECT COUNT(*) FILTER (WHERE r.status = 'RESOLVED_ZERO'), COUNT(*)
        FROM haz_triggers t
        LEFT JOIN haz_resolutions r
          ON r.iso3 = t.iso3 AND r.year = t.year AND r.month = t.month AND r.hazard = t.hazard
        WHERE t.hazard = ?
          AND t.created_at >= TIMESTAMP '{SWEEP_FIX_CUTOFF}'
          AND (t.triggered = FALSE OR t.trigger_source = 'reliefweb_ladder')
          AND (r.iso3 IS NULL OR r.created_at >= TIMESTAMP '{SWEEP_FIX_CUTOFF}')
          AND make_date(t.year, t.month, 1) < date_trunc('month', CAST(t.created_at AS DATE))
        """,
        [hazard],
    ).fetchone()
    zeros, cells = int(row[0] or 0), int(row[1] or 0)
    if cells < MIN_CELLS_FOR_SHARE:
        return DEFAULT_SILENT_SHARE, cells, "default"
    return zeros / cells, cells, "measured"


def _ledger_rates(con, hazard: str) -> dict[str, Any]:
    """Cost per call and calls per cell, from the last 30 days of billed calls."""

    base = """
        FROM haz_doc_extractions
        WHERE hazard = ?
          AND COALESCE(prompt_tokens, 0) + COALESCE(completion_tokens, 0) > 0
    """
    newest = con.execute(f"SELECT MAX(created_at) {base}", [hazard]).fetchone()[0]
    if newest is None:
        return {"calls": 0, "cost_usd": 0.0, "cells": 0, "window": "none"}
    since = newest - dt.timedelta(days=30)
    row = con.execute(
        f"SELECT COUNT(*), COALESCE(SUM(cost_usd), 0), "
        f"COUNT(DISTINCT (iso3, year, month)) {base} AND created_at >= ?",
        [hazard, since],
    ).fetchone()
    return {
        "calls": int(row[0] or 0), "cost_usd": float(row[1] or 0.0),
        "cells": int(row[2] or 0),
        "window": f"{since:%Y-%m-%d}..{newest:%Y-%m-%d}",
    }


def budget_estimate(con, rulebook=None) -> dict[str, Any]:
    """What the re-walk will cost in extraction calls, priced at the ledger.

    A re-walked cell the scoped sweep finds silent writes a zero and costs
    nothing; a cell with a sweep hit goes to the ladder under the backcast
    share. Documents a cell already extracted at the current prompt version
    are cache hits and cost nothing, so the estimate is given gross and net
    of cells that already hold extractions.
    """

    if rulebook is None:
        from resolver.hazard_resolution.rulebook import load_rulebook

        rulebook = load_rulebook()
    share = int(rulebook.get("extraction.backcast_max_calls_per_month") or 0)
    prompt_version = str(rulebook.get("extraction.prompt_version") or "")
    tables = _tables(con)
    out: dict[str, Any] = {
        "backcast_share_per_month": share,
        "prompt_version": prompt_version,
        "by_hazard": {},
    }
    total_calls_net = 0.0
    total_calls_gross = 0.0
    total_cost_net = 0.0
    for hz in HAZARDS:
        n = int(
            con.execute(f"SELECT COUNT(*) FROM {KEYS_TABLE} WHERE hazard = ?", [hz]).fetchone()[0]
        )
        silent, measured_cells, basis = _silent_share(con, hz)
        rates = (
            _ledger_rates(con, hz)
            if "haz_doc_extractions" in tables
            else {"calls": 0, "cost_usd": 0.0, "cells": 0, "window": "none"}
        )
        calls_per_cell = rates["calls"] / rates["cells"] if rates["cells"] else None
        cost_per_call = rates["cost_usd"] / rates["calls"] if rates["calls"] else None
        cached = 0
        if "haz_doc_extractions" in tables and n:
            cached = int(
                con.execute(
                    f"""
                    SELECT COUNT(*) FROM {KEYS_TABLE} k
                    WHERE k.hazard = ? AND EXISTS (
                        SELECT 1 FROM haz_doc_extractions e
                        WHERE e.iso3 = k.iso3 AND e.year = k.year AND e.month = k.month
                          AND e.hazard = k.hazard AND e.status = 'ok'
                          AND e.prompt_version = ?)
                    """,
                    [hz, prompt_version],
                ).fetchone()[0]
            )
        hit_cells = n * (1.0 - silent)
        hit_cells_uncached = max(n - cached, 0) * (1.0 - silent)
        gross = hit_cells * calls_per_cell if calls_per_cell is not None else None
        net = hit_cells_uncached * calls_per_cell if calls_per_cell is not None else None
        cost = net * cost_per_call if net is not None and cost_per_call is not None else None
        total_calls_gross += gross or 0.0
        total_calls_net += net or 0.0
        total_cost_net += cost or 0.0
        out["by_hazard"][hz] = {
            "cells": n,
            "silent_share": round(silent, 3),
            "silent_share_basis": basis,
            "silent_share_cells": measured_cells,
            "expected_zero_cells": round(n * silent),
            "expected_ladder_cells": round(hit_cells),
            "cells_with_cached_extractions": cached,
            "calls_per_cell": round(calls_per_cell, 2) if calls_per_cell is not None else None,
            "cost_per_call_usd": round(cost_per_call, 5) if cost_per_call is not None else None,
            "ledger_window": rates["window"],
            "calls_gross": round(gross) if gross is not None else None,
            "calls_net_of_cache": round(net) if net is not None else None,
            "cost_usd_net_of_cache": round(cost, 2) if cost is not None else None,
        }
    out["calls_gross"] = round(total_calls_gross)
    out["calls_net_of_cache"] = round(total_calls_net)
    out["cost_usd_net_of_cache"] = round(total_cost_net, 2)
    capacity = 2 * share
    total = int(rulebook.get("extraction.max_calls_per_month") or 0)
    reserve = int(rulebook.get("extraction.live_reserve_calls") or 0)
    # The one-dispatch override is clamped to the monthly total less the live
    # reserve (extract.load_budget), so that is the most a month can absorb
    # without a rulebook change.
    max_override = max(total - reserve, 0)
    out["two_month_capacity"] = capacity
    out["max_override_per_month"] = max_override
    out["months_at_share"] = (
        round(total_calls_gross / share, 1) if share else None
    )
    out["months_at_max_override"] = (
        round(total_calls_gross / max_override, 1) if max_override else None
    )
    # The gross figure decides: the cache is a document-level bet on the
    # same documents coming back, and the share must absorb the worst case.
    if share and total_calls_gross > capacity:
        out["recommendation"] = (
            f"the gross estimate ({round(total_calls_gross):,} calls) exceeds what the "
            f"backcast share absorbs in two months ({capacity:,}); raise it for ONE "
            f"dispatch of haz_backcast.yml with backcast_extraction_share "
            f"(PYTHIA_HAZ_BACKCAST_EXTRACTION_SHARE), never in the rulebook. The override "
            f"is clamped to {max_override:,} a month (the total less the live reserve), so "
            f"the re-walk takes about {out['months_at_max_override']} months at the "
            f"maximum and {out['months_at_share']} at the standing share; cells the budget "
            f"cuts short are deferred, never resolved, and the nightly run resumes them"
        )
    else:
        out["recommendation"] = (
            f"the backcast share ({share:,} a month) absorbs the estimate within two "
            f"months; no override needed"
        )
    return out


# ---------------------------------------------------------------------------
# Audit copy
# ---------------------------------------------------------------------------

def export_audit(con, audit_dir: Path) -> dict[str, int]:
    """Write the selected resolution rows and their trigger rows as CSV."""

    audit_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, int] = {}
    for table in ("haz_resolutions", "haz_triggers"):
        cur = con.execute(
            f"""
            SELECT x.* FROM {table} x
            JOIN {KEYS_TABLE} k
              ON k.iso3 = x.iso3 AND k.year = x.year AND k.month = x.month AND k.hazard = x.hazard
            ORDER BY x.hazard, x.year, x.month, x.iso3
            """
        )
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        path = audit_dir / f"{table}.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(cols)
            writer.writerows(rows)
        written[table] = len(rows)
    ledger = con.execute(
        f"""
        SELECT p.* FROM haz_backcast_progress p
        JOIN {MONTHS_TABLE} m ON m.hazard = p.hazard AND m.ym = p.ym
        ORDER BY p.hazard, p.ym
        """
    ) if "haz_backcast_progress" in _tables(con) else None
    if ledger is not None:
        cols = [d[0] for d in ledger.description]
        rows = ledger.fetchall()
        with (audit_dir / "haz_backcast_progress.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(cols)
            writer.writerows(rows)
        written["haz_backcast_progress"] = len(rows)
    with (audit_dir / "README.txt").open("w", encoding="utf-8") as fh:
        fh.write(
            "Rows retracted by compact_resolver_db.yml maintenance=sweep_era_rewalk.\n"
            f"Predicate: see resolver/hazard_resolution/sweep_era_rewalk.py "
            f"(cutoff {SWEEP_FIX_CUTOFF} UTC, {SWEEP_FIX_COMMIT}).\n"
            f"Rows: {json.dumps(written)}\n"
        )
    return written


# ---------------------------------------------------------------------------
# Integrity
# ---------------------------------------------------------------------------

def _columns(con, table: str) -> list[str]:
    # PRAGMA table_info: (cid, NAME, ...)
    return [str(r[1]) for r in con.execute(f"PRAGMA table_info('{table}')").fetchall()]


def _checksum(con, table: str, order_by: str, exclude_join: str | None) -> str:
    cols = _columns(con, table)
    row_expr = "concat_ws('|', " + ", ".join(
        f"coalesce(CAST(x.\"{c}\" AS VARCHAR), '<null>')" for c in cols
    ) + ")"
    where = f"WHERE NOT EXISTS ({exclude_join})" if exclude_join else ""
    return str(
        con.execute(
            f"""
            SELECT md5(coalesce(string_agg(md5({row_expr}), ',' ORDER BY {order_by}), ''))
            FROM {table} x {where}
            """
        ).fetchone()[0]
    )


_CELL_EXCLUDE = (
    f"SELECT 1 FROM {KEYS_TABLE} k WHERE k.iso3 = x.iso3 AND k.year = x.year "
    f"AND k.month = x.month AND k.hazard = x.hazard"
)
_MONTH_EXCLUDE = f"SELECT 1 FROM {MONTHS_TABLE} m WHERE m.hazard = x.hazard AND m.ym = x.ym"


def integrity_report(con) -> dict[str, Any]:
    """Counts and checksums over everything the retraction must not touch."""

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
    report["trigger_rows"] = int(con.execute("SELECT COUNT(*) FROM haz_triggers").fetchone()[0])
    return report


def compare_reports(
    before: dict[str, Any], after: dict[str, Any], deleted: dict[tuple[str, str], int]
) -> list[str]:
    """Every difference that must not exist; an empty list means they match."""

    problems = []
    for name, value in before.get("checksums", {}).items():
        if after.get("checksums", {}).get(name) != value:
            problems.append(f"checksum moved: {name}")
    expected = {}
    for c in before.get("counts", []):
        key = (c["hazard"], c["status"])
        n = c["rows"] - deleted.get(key, 0)
        if n:
            expected[key] = n
    got = {(c["hazard"], c["status"]): c["rows"] for c in after.get("counts", [])}
    if got != expected:
        problems.append(f"counts moved by other than the deleted rows: {expected} -> {got}")
    if before.get("trigger_rows") != after.get("trigger_rows"):
        problems.append("haz_triggers row count moved")
    if before.get("revisions_rows") != after.get("revisions_rows"):
        problems.append("haz_revisions row count moved")
    return problems


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------

def already_applied(con) -> dict[str, int]:
    if "haz_feed_restale" not in _tables(con):
        return {}
    return {
        str(h): int(n or 0)
        for h, n in con.execute(
            "SELECT hazard, rows_deleted FROM haz_feed_restale WHERE feed = ? AND token = ?",
            [FEED, TOKEN],
        ).fetchall()
    }


def apply_retraction(con) -> dict[str, Any]:
    """Delete the staged rows, reset their triggers, free their ledger months.

    One transaction. The staged temp tables say exactly what to touch, so
    the integrity report's exclusions and the writes cannot disagree.
    """

    from resolver.hazard_resolution.schema import ensure_haz_schema

    ensure_haz_schema(con)
    done = already_applied(con)
    result: dict[str, Any] = {
        "applied": False, "deleted": {}, "triggers_reset": 0, "triggers_stamped": 0,
        "ledger_rows_freed": {}, "already_applied": done,
    }
    hazards = [
        str(r[0]) for r in con.execute(
            f"SELECT DISTINCT hazard FROM {KEYS_TABLE} ORDER BY 1"
        ).fetchall()
    ]
    todo = [h for h in hazards if h not in done]
    if not todo:
        result["reason"] = (
            "nothing selected" if not hazards
            else f"token {TOKEN} already applied for {', '.join(hazards)}"
        )
        return result
    hz_list = ", ".join(f"'{h}'" for h in todo)
    con.execute("BEGIN TRANSACTION")
    try:
        deleted = con.execute(
            f"""
            SELECT r.hazard, r.status, COUNT(*) FROM haz_resolutions r
            JOIN {KEYS_TABLE} k
              ON k.iso3 = r.iso3 AND k.year = r.year AND k.month = r.month AND k.hazard = r.hazard
            WHERE r.hazard IN ({hz_list})
            GROUP BY 1, 2
            """
        ).fetchall()
        con.execute(
            f"""
            DELETE FROM haz_resolutions r
            WHERE r.hazard IN ({hz_list}) AND EXISTS (
                SELECT 1 FROM {KEYS_TABLE} k
                WHERE k.iso3 = r.iso3 AND k.year = r.year
                  AND k.month = r.month AND k.hazard = r.hazard)
            """
        )
        reset = con.execute(
            f"""
            SELECT COUNT(*) FROM haz_triggers t JOIN {KEYS_TABLE} k
              ON k.iso3 = t.iso3 AND k.year = t.year AND k.month = t.month AND k.hazard = t.hazard
            WHERE t.hazard IN ({hz_list}) AND t.trigger_source = '{RETIRED_SOURCE}'
            """
        ).fetchone()[0]
        patch = json.dumps({"no_row_reason": PENDING_REASON})
        con.execute(
            f"""
            UPDATE haz_triggers t SET
                triggered = CASE WHEN t.trigger_source = '{RETIRED_SOURCE}'
                                 THEN FALSE ELSE t.triggered END,
                trigger_source = CASE WHEN t.trigger_source = '{RETIRED_SOURCE}'
                                      THEN '{TRIGGER_SOURCE_NONE}' ELSE t.trigger_source END,
                trigger_detail_json = CAST(json_merge_patch(
                    CASE WHEN json_valid(t.trigger_detail_json)
                              AND json_type(t.trigger_detail_json) = 'OBJECT'
                         THEN t.trigger_detail_json ELSE '{{}}' END,
                    ?) AS VARCHAR)
            WHERE t.hazard IN ({hz_list}) AND EXISTS (
                SELECT 1 FROM {KEYS_TABLE} k
                WHERE k.iso3 = t.iso3 AND k.year = t.year
                  AND k.month = t.month AND k.hazard = t.hazard)
            """,
            [patch],
        )
        stamped = con.execute(
            f"SELECT COUNT(*) FROM {KEYS_TABLE} WHERE hazard IN ({hz_list})"
        ).fetchone()[0]
        freed: dict[str, int] = {}
        for hz in todo:
            months = [
                str(r[0]) for r in con.execute(
                    f"SELECT ym FROM {MONTHS_TABLE} WHERE hazard = ? ORDER BY ym", [hz]
                ).fetchall()
            ]
            held = int(
                con.execute(
                    f"""
                    SELECT COUNT(*) FROM haz_backcast_progress p
                    JOIN {MONTHS_TABLE} m ON m.hazard = p.hazard AND m.ym = p.ym
                    WHERE p.hazard = ?
                    """,
                    [hz],
                ).fetchone()[0]
            )
            con.execute(
                f"""
                DELETE FROM haz_backcast_progress p
                WHERE p.hazard = ? AND EXISTS (
                    SELECT 1 FROM {MONTHS_TABLE} m WHERE m.hazard = p.hazard AND m.ym = p.ym)
                """,
                [hz],
            )
            con.execute(
                "INSERT INTO haz_feed_restale (feed, hazard, token, months, rows_deleted) "
                "VALUES (?, ?, ?, ?, ?)",
                [FEED, hz, TOKEN, ",".join(months), held],
            )
            freed[hz] = held
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    result.update({
        "applied": True,
        "deleted": {f"{h}/{s}": int(n) for h, s, n in deleted},
        "triggers_reset": int(reset),
        "triggers_stamped": int(stamped),
        "ledger_rows_freed": freed,
    })
    return result


# ---------------------------------------------------------------------------
# Report + CLI
# ---------------------------------------------------------------------------

def format_plan(plan: dict[str, Any], budget: dict[str, Any] | None) -> list[str]:
    lines = [
        f"Selected rows written before {SWEEP_FIX_CUTOFF} UTC ({SWEEP_FIX_COMMIT}): "
        f"{plan['rows']:,} (FL {plan['by_hazard'].get('FL', 0):,}, "
        f"TC {plan['by_hazard'].get('TC', 0):,}); "
        f"ledger rows to free: {plan['ledger_rows_to_free']:,}",
        "",
        "| hazard | year | rows |",
        "|---|---:|---:|",
    ]
    for r in plan["by_year"]:
        lines.append(f"| {r['hazard']} | {r['year']} | {r['rows']:,} |")
    lines += ["", "| hazard | trigger row | rows |", "|---|---|---:|"]
    for r in plan["by_trigger"]:
        lines.append(f"| {r['hazard']} | {r['trigger']} | {r['rows']:,} |")
    lines += ["", "| hazard | months to re-walk |", "|---|---:|"]
    for r in plan["months_by_hazard"]:
        lines.append(f"| {r['hazard']} | {r['months']:,} |")
    if budget:
        lines += [
            "",
            "Extraction estimate (a silent cell writes a zero and costs nothing; "
            "a sweep hit goes to the ladder under the backcast share):",
            "",
            "| hazard | cells | silent share | ladder cells | cached cells | calls/cell | "
            "$/call | calls gross | calls net of cache | $ net of cache |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for hz, b in budget["by_hazard"].items():
            lines.append(
                f"| {hz} | {b['cells']:,} | {b['silent_share']:.2f} ({b['silent_share_basis']}, "
                f"n={b['silent_share_cells']}) | {b['expected_ladder_cells']:,} | "
                f"{b['cells_with_cached_extractions']:,} | {b['calls_per_cell']} | "
                f"{b['cost_per_call_usd']} | {b['calls_gross']} | {b['calls_net_of_cache']} | "
                f"{b['cost_usd_net_of_cache']} |"
            )
        lines += [
            "",
            f"Total: {budget['calls_gross']:,} calls gross, {budget['calls_net_of_cache']:,} "
            f"net of cache (${budget['cost_usd_net_of_cache']:,.2f}); backcast share "
            f"{budget['backcast_share_per_month']:,} a month, "
            f"{budget['two_month_capacity']:,} over two months.",
            f"Recommendation: {budget['recommendation']}.",
        ]
    return lines


def main(argv: list[str] | None = None) -> int:
    import argparse

    import duckdb

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", required=True)
    parser.add_argument("--audit-dir", default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--audit-only", action="store_true",
                      help="plan + audit copy + estimate, write nothing (the default)")
    mode.add_argument("--apply", action="store_true")
    parser.add_argument("--read-only", action="store_true",
                        help="open the database read-only (a dry run on a copy)")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--md-out", default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.apply and args.read_only:
        parser.error("--apply cannot run on a read-only connection")

    path = args.db.replace("duckdb:///", "")
    con = duckdb.connect(path, read_only=args.read_only)
    problems: list[str] = []
    payload: dict[str, Any] = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "cutoff_utc": SWEEP_FIX_CUTOFF, "cutoff_commit": SWEEP_FIX_COMMIT,
        "apply": bool(args.apply),
    }
    try:
        stage_selection(con)
        plan = selection_counts(con)
        payload["plan"] = plan
        try:
            payload["budget"] = budget_estimate(con)
        except Exception as exc:  # noqa: BLE001 - an estimate never stops the plan
            payload["budget"] = None
            payload["budget_error"] = f"{type(exc).__name__}: {exc}"
        if args.audit_dir:
            payload["audit"] = export_audit(con, Path(args.audit_dir))
        if args.apply:
            from resolver.hazard_resolution.schema import ensure_haz_schema

            ensure_haz_schema(con)
            before = integrity_report(con)
            result = apply_retraction(con)
            con.execute("CHECKPOINT")
            after = integrity_report(con)
            deleted = {
                tuple(k.split("/", 1)): v for k, v in result.get("deleted", {}).items()
            }
            problems = compare_reports(before, after, deleted)
            payload.update({"before": before, "after": after, "result": result})
    finally:
        con.close()
    payload["problems"] = problems

    lines = format_plan(payload["plan"], payload.get("budget"))
    if payload.get("budget_error"):
        lines += ["", f"Estimate unavailable: {payload['budget_error']}"]
    if args.apply:
        r = payload["result"]
        lines += [
            "",
            f"Applied: {r['applied']} ({r.get('reason', 'done')}); deleted {r['deleted']}; "
            f"triggers reset {r['triggers_reset']}, stamped {r['triggers_stamped']}; "
            f"ledger rows freed {r['ledger_rows_freed']} under token {TOKEN}.",
        ]
        for name, value in payload["before"]["checksums"].items():
            lines.append(
                f"- {name}: `{value}` -> `{payload['after']['checksums'].get(name)}`"
            )
    if problems:
        lines += ["", "**MISMATCH:** " + "; ".join(problems)]
    text = "\n".join(lines)
    print(text)
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
    if args.md_out:
        Path(args.md_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.md_out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    budget = payload.get("budget") or {}
    if budget.get("recommendation", "").startswith("the gross estimate"):
        print(f"::warning title=Sweep-era re-walk budget::{budget['recommendation']}")
    if problems:
        print("::error::untouched rows moved under the sweep-era retraction: "
              + "; ".join(problems))
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
