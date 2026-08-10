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
    sql = f"""
        WITH ranked AS (
            SELECT question_id, iso3, hazard_code, metric, js_vs_baserate,
                   excess_nominal, exceedances_json, baserate_n_obs,
                   ROW_NUMBER() OVER (
                       PARTITION BY question_id
                       ORDER BY CASE {order} ELSE 99 END, model_name
                   ) AS rn
            FROM forecast_deviation
            WHERE run_id = ?{test_clause}
        )
        SELECT question_id, iso3, hazard_code, metric, js_vs_baserate,
               excess_nominal, exceedances_json, baserate_n_obs
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
        out.append({
            "question_id": row[0],
            "iso3": row[1],
            "hazard_code": row[2],
            "metric": row[3],
            "js_vs_baserate": row[4],
            "excess_nominal": row[5],
            "exceedances": exceedances,
            "baserate_n_obs": row[7],
        })
    return out


def calibrate(con, *, include_test: bool = False) -> dict[str, dict[str, Any]]:
    """{run_id: gate counts} for every run with deviation rows."""
    test_clause = "" if include_test else " WHERE COALESCE(is_test, FALSE) = FALSE"
    runs = [
        str(r[0]) for r in con.execute(
            "SELECT DISTINCT run_id FROM forecast_deviation"
            f"{test_clause} ORDER BY run_id"
        ).fetchall()
        if r[0]
    ]
    out: dict[str, dict[str, Any]] = {}
    for run_id in runs:
        rows = _rows_for_run(con, run_id, include_test=include_test)
        if not rows:
            continue
        out[run_id] = gating.gate_rows(
            rows,
            unusual_percentile=config.unusual_percentile(),
            min_probability=config.min_probability(),
            thin_min_obs=config.baserate_min_obs(),
        )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, help="DuckDB URL or path")
    parser.add_argument("--include-test", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[gatecheck] %(message)s")
    try:
        from resolver.db import duckdb_io

        con = duckdb_io.get_db(args.db or duckdb_io.DEFAULT_DB_URL)
        try:
            runs = calibrate(con, include_test=args.include_test)
        finally:
            duckdb_io.close_db(con)
    except Exception as exc:  # noqa: BLE001 - a diagnostic never fails a run
        LOGGER.error("[gatecheck] failed: %s", exc)
        return 0

    if not runs:
        print("[gatecheck] no runs with deviation rows in this database")
        return 0
    print(
        f"[gatecheck] unusual percentile {config.unusual_percentile():.2f}, "
        f"minimum probability {config.min_probability():.2f}, "
        f"thin below {config.baserate_min_obs()} observations, "
        f"at most {config.max_entries()} entries"
    )
    print(gating.calibration_table(runs))
    empty = [r for r, c in runs.items() if not c.get("both") and not c.get("major")]
    if empty:
        print(
            "[gatecheck] these runs would produce NO entries at all under "
            f"these thresholds: {', '.join(empty)}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
