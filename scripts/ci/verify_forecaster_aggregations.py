#!/usr/bin/env python3
# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Validate that Forecaster wrote both aggregation methods for the latest run."""

from __future__ import annotations

import argparse
from typing import Iterable, Sequence, Tuple

from resolver.db import duckdb_io


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


def get_latest_run_id(con: duckdb_io.duckdb.DuckDBPyConnection) -> int:
    row = con.execute("SELECT max(run_id) FROM forecasts_ensemble").fetchone()
    run_id = row[0] if row else None
    if run_id is None:
        raise SystemExit("No forecasts_ensemble.run_id values found; Forecaster has not written ensembles yet.")
    return int(run_id)


def get_model_counts(
    con: duckdb_io.duckdb.DuckDBPyConnection, run_id: int
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
    con: duckdb_io.duckdb.DuckDBPyConnection, run_id: int
) -> Sequence[Tuple[int, int, int]]:
    rows: Iterable[Tuple[int, int, int]] = con.execute(
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
    return [(int(qid), int(n_mean), int(n_bmc)) for qid, n_mean, n_bmc in rows]


def main() -> int:
    args = parse_args()
    con = duckdb_io.get_db(args.db)
    try:
        run_id = get_latest_run_id(con)
        print(f"Latest forecasts_ensemble run_id: {run_id}")

        counts = get_model_counts(con, run_id)
        if not counts:
            raise SystemExit(f"No forecasts_ensemble rows found for run_id {run_id}.")

        print("model_name counts for latest run:")
        for model_name, count in counts:
            print(f"- {model_name}: {count}")

        expected = ("ensemble_bayesmc_v2", "ensemble_mean_v2")
        present = {model_name for model_name, _ in counts}
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
