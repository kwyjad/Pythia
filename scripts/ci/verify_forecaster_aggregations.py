#!/usr/bin/env python3
# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Validate that Forecaster wrote both aggregation methods for the latest run."""

from __future__ import annotations

import argparse
import re
from typing import Iterable, Sequence, Tuple

from resolver.db import duckdb_io


_TS_RE = re.compile(r"(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify that ensemble_mean_v2 and ensemble_bayesmc_v2 exist for the latest forecasts_ensemble run.",
    )
    parser.add_argument(
        "--db",
        required=True,
        help="DuckDB URL or path for the resolver database (e.g., duckdb:///data/resolver.duckdb).",
    )
    return parser.parse_args()


def _run_id_key(run_id: str) -> tuple[int, str]:
    match = _TS_RE.search(run_id or "")
    if not match:
        return (-1, run_id or "")
    return (int(match.group(1)), run_id)


def get_latest_run_id(con: duckdb_io.duckdb.DuckDBPyConnection) -> str:
    columns = [row[1] for row in con.execute("PRAGMA table_info('forecasts_ensemble')").fetchall()]
    for column in ("created_at", "timestamp", "created_time", "created_time_iso", "ts"):
        if column in columns:
            row = con.execute(
                f"SELECT run_id FROM forecasts_ensemble WHERE run_id IS NOT NULL ORDER BY {column} DESC LIMIT 1"
            ).fetchone()
            if row and row[0]:
                return str(row[0])

    rows = con.execute(
        "SELECT DISTINCT run_id FROM forecasts_ensemble WHERE run_id IS NOT NULL"
    ).fetchall()
    run_ids = [str(row[0]) for row in rows if row and row[0]]
    if not run_ids:
        raise SystemExit("No forecasts_ensemble.run_id values found; Forecaster has not written ensembles yet.")
    return max(run_ids, key=_run_id_key)


def get_model_counts(
    con: duckdb_io.duckdb.DuckDBPyConnection, run_id: str
) -> Sequence[Tuple[str, int]]:
    rows: Iterable[Tuple[str, int]] = con.execute(
        """
        SELECT model_name, COUNT(*) AS n
        FROM forecasts_ensemble
        WHERE run_id = ?
        GROUP BY 1
        ORDER BY model_name
        """,
        [run_id],
    ).fetchall()
    return [(str(model_name), int(count)) for model_name, count in rows]


def find_missing_questions(
    con: duckdb_io.duckdb.DuckDBPyConnection, run_id: str
) -> Sequence[Tuple[str, int, int]]:
    rows: Iterable[Tuple[str, int, int]] = con.execute(
        """
        SELECT question_id,
               SUM(CASE WHEN model_name = 'ensemble_mean_v2' THEN 1 ELSE 0 END) AS n_mean,
               SUM(CASE WHEN model_name = 'ensemble_bayesmc_v2' THEN 1 ELSE 0 END) AS n_bmc
        FROM forecasts_ensemble
        WHERE run_id = ?
        GROUP BY 1
        HAVING n_mean = 0 OR n_bmc = 0
        ORDER BY question_id
        LIMIT 20
        """,
        [run_id],
    ).fetchall()
    return [(str(qid), int(n_mean), int(n_bmc)) for qid, n_mean, n_bmc in rows]


def main() -> int:
    args = parse_args()
    con = duckdb_io.get_db(args.db)
    try:
        run_id = get_latest_run_id(con)
        print(f"Latest forecasts_ensemble run_id: {run_id}")

        counts = get_model_counts(con, run_id)
        if not counts:
            raise SystemExit(f"No forecasts_ensemble rows found for run_id {run_id}.")

        present = {model_name for model_name, _ in counts}
        track1_models = {m for m in present if m != "track2_flash"}
        track2_count = sum(c for m, c in counts if m == "track2_flash")
        track1_count = sum(c for m, c in counts if m != "track2_flash")

        print(f"Track breakdown: track1_models={len(track1_models)} ({track1_count} rows), track2_flash={track2_count} rows")
        print("model_name counts for latest run:")
        for model_name, count in counts:
            print(f"- {model_name}: {count}")

        if not track1_models:
            print("All questions are track2 — ensemble aggregation not expected. OK.")
            return 0

        expected = ("ensemble_bayesmc_v2", "ensemble_mean_v2")
        missing_models = [name for name in expected if name not in present]
        if missing_models:
            missing_list = ", ".join(sorted(missing_models))
            raise SystemExit(f"Missing expected model_names in forecasts_ensemble: {missing_list}")

        missing_questions = find_missing_questions(con, run_id)
        if missing_questions:
            formatted = ", ".join(
                f"{question_id} (mean={n_mean}, bayesmc={n_bmc})"
                for question_id, n_mean, n_bmc in missing_questions
            )
            raise SystemExit(
                "Some questions missing one aggregation method (showing up to 20): "
                f"{formatted}"
            )

        print(f"OK: both aggregation methods present for all questions in run {run_id}.")
    finally:
        duckdb_io.close_db(con)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
