# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Run the selection gate over every run in a database and print what it does.

    python -m interpreter.gatecheck --db "$PYTHIA_DB_URL" [--include-test]

Task 1.6 of the v3 brief: before a threshold change ships, run the gate over
the runs already in the DB and look at what it admits and drops. If 25% empties
the climate section, that is a finding, and a finding is worth more than a
silently blank page.

`gating.calibration_table` has existed since the thresholds landed and had no
caller, which in this repo means it did not exist. This is the caller.

Read-only, and `main()` returns 0 in every outcome: a diagnostic must never be
the thing that fails a pipeline.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

from interpreter import config, gating, names

LOGGER = logging.getLogger(__name__)


def _rows_for_run(con, run_id: str, *, include_test: bool) -> list[dict[str, Any]]:
    """One deviation row per question, best-available aggregate, as the pack
    builder would see them."""
    test_clause = "" if include_test else " AND COALESCE(is_test, FALSE) = FALSE"
    order = " ".join(
        f"WHEN model_name = '{m}' THEN {i}"
        for i, m in enumerate(names.AGGREGATE_PREFERENCE)
    )
    # The v4 movement columns are added by migration, so a DB from before it
    # will not have them. Their absence is the finding, not a crash: the gate
    # can admit nothing as worsening and the caller is told so by name.
    extra = [c for c in _MOVEMENT_COLUMNS if _column_exists(con, c)]
    select_extra = "".join(f", {c}" for c in extra)
    sql = f"""
        WITH ranked AS (
            SELECT question_id, iso3, hazard_code, metric, js_vs_baserate,
                   excess_nominal, exceedances_json, baserate_n_obs,
                   score_family{select_extra},
                   ROW_NUMBER() OVER (
                       PARTITION BY question_id
                       ORDER BY CASE {order} ELSE 99 END, model_name
                   ) AS rn
            FROM forecast_deviation
            WHERE run_id = ?{test_clause}
        )
        SELECT question_id, iso3, hazard_code, metric, js_vs_baserate,
               excess_nominal, exceedances_json, baserate_n_obs,
               score_family{select_extra}
        FROM ranked WHERE rn = 1
    """
    out: list[dict[str, Any]] = []
    for row in con.execute(sql, [run_id]).fetchall():
        exceedances: list[float] = []
        if row[6]:
            try:
                exceedances = [float(v) for v in json.loads(row[6]) if v is not None]
            except (TypeError, ValueError):
                exceedances = []
        record = {
            "question_id": row[0],
            "iso3": row[1],
            "hazard_code": row[2],
            "metric": row[3],
            "js_vs_baserate": row[4],
            "excess_nominal": row[5],
            "exceedances": exceedances,
            "baserate_n_obs": row[7],
            "score_family": row[8],
        }
        for i, name in enumerate(extra):
            record[name] = row[9 + i]
        out.append(record)
    return out


_MOVEMENT_COLUMNS = ("delta_p50", "delta_p90", "movement_threshold")


def _column_exists(con, column: str) -> bool:
    try:
        rows = con.execute("PRAGMA table_info('forecast_deviation')").fetchall()
    except Exception:  # noqa: BLE001
        return False
    return column.lower() in {str(r[1]).lower() for r in rows}


def calibrate(
    con, *, include_test: bool = False, mode: str | None = None
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """({run_id: gate counts}, {run_id: gated rows}) for every run.

    The rows come back as well as the counts, because "how many cleared" is
    not what a recalibration needs: it needs to see WHICH cleared and what
    the near misses were. A threshold that empties a section is a finding to
    look at, and a finding is worth more than a silently blank page.
    """
    test_clause = "" if include_test else " WHERE COALESCE(is_test, FALSE) = FALSE"
    runs = [
        str(r[0]) for r in con.execute(
            "SELECT DISTINCT run_id FROM forecast_deviation"
            f"{test_clause} ORDER BY run_id"
        ).fetchall()
        if r[0]
    ]
    out: dict[str, dict[str, Any]] = {}
    by_run: dict[str, list[dict[str, Any]]] = {}
    for run_id in runs:
        rows = _rows_for_run(con, run_id, include_test=include_test)
        if not rows:
            continue
        out[run_id] = gating.gate_rows(
            rows,
            unusual_percentile=config.unusual_percentile(),
            min_probability=config.min_probability(),
            thin_min_obs=config.baserate_min_obs(),
            mode=mode or config.gate_mode(),
        )
        by_run[run_id] = rows
    return out, by_run


def _movement_words(row: dict[str, Any]) -> str:
    movement = row.get("material_movement")
    threshold = row.get("movement_threshold")
    if movement is None or threshold in (None, 0):
        return "no movement figure"
    unit = "points" if str(row.get("score_family") or "") == "binary" else ""
    share = float(movement) / float(threshold)
    return f"{float(movement):+,.0f}{unit} ({share:.2f} of its threshold)"


def _run_detail(rows: list[dict[str, Any]]) -> list[str]:
    """Which questions cleared, and what the near misses were."""
    lines: list[str] = []
    cleared = [r for r in rows if r.get("gate") == gating.GATE_WORSENING]
    cleared.sort(key=gating.movement_rank_key)
    lines.append(f"  worsening ({len(cleared)}):")
    for row in cleared:
        lines.append(
            f"    {names.describe_pair(row.get('iso3'), row.get('hazard_code'), row.get('metric'))}"
            f" — {_movement_words(row)}"
        )
    if not cleared:
        lines.append("    none. That is a statement about the month, and the "
                     "report says so rather than reaching for filler.")
    misses = [
        r for r in rows
        if r.get("gate") != gating.GATE_WORSENING
        and r.get("material_movement") is not None
        and r.get("movement_threshold")
        and float(r["material_movement"]) >= 0.5 * float(r["movement_threshold"])
    ]
    misses.sort(key=gating.movement_rank_key)
    lines.append(f"  near misses ({len(misses)}):")
    for row in misses[:10]:
        why = []
        if not row.get("passed_unusual"):
            why.append("not unusual enough")
        if not row.get("passed_worsening"):
            why.append("movement under the threshold")
        lines.append(
            f"    {names.describe_pair(row.get('iso3'), row.get('hazard_code'), row.get('metric'))}"
            f" — {_movement_words(row)}; {', '.join(why) or 'other'}"
        )
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, help="DuckDB URL or path")
    parser.add_argument("--include-test", action="store_true")
    parser.add_argument(
        "--mode", choices=["delta", "level"], default=None,
        help="Override the materiality mode for this check only. Running "
             "both is how a change is argued about rather than asserted.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[gatecheck] %(message)s")
    try:
        from resolver.db import duckdb_io

        con = duckdb_io.get_db(args.db or duckdb_io.DEFAULT_DB_URL)
        try:
            runs, rows_by_run = calibrate(
                con, include_test=args.include_test, mode=args.mode
            )
        finally:
            duckdb_io.close_db(con)
    except Exception as exc:  # noqa: BLE001 - a diagnostic never fails a run
        LOGGER.error("[gatecheck] failed: %s", exc)
        return 0

    if not runs:
        print("[gatecheck] no runs with deviation rows in this database")
        return 0
    print(
        f"[gatecheck] mode {args.mode or config.gate_mode()}, "
        f"unusual percentile {config.unusual_percentile():.2f}, "
        f"minimum probability {config.min_probability():.2f}, "
        f"thin below {config.baserate_min_obs()} observations, "
        f"at most {config.max_entries()} entries"
    )
    print(gating.calibration_table(runs))
    for run_id, rows in rows_by_run.items():
        print(f"\n[gatecheck] {run_id}")
        for line in _run_detail(rows):
            print(line)
    empty = [r for r, c in runs.items() if not c.get("both") and not c.get("major")]
    if empty:
        print(
            "[gatecheck] these runs would produce NO entries at all under "
            f"these thresholds: {', '.join(empty)}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
