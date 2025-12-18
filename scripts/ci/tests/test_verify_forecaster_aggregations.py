# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

from pathlib import Path
import sys

import pytest

from resolver.db import duckdb_io
from resolver.db._duckdb_available import DUCKDB_AVAILABLE
from scripts.ci import verify_forecaster_aggregations as vfa

pytestmark = pytest.mark.skipif(not DUCKDB_AVAILABLE, reason="duckdb not installed")


def _create_forecasts_table(conn: duckdb_io.duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        CREATE TABLE forecasts_ensemble (
            run_id TEXT,
            model_name TEXT,
            question_id INTEGER
        )
        """
    )


def test_get_latest_run_id_prefers_numeric_suffix(tmp_path: Path) -> None:
    db_path = tmp_path / "resolver.duckdb"
    conn = duckdb_io.get_db(str(db_path))
    try:
        _create_forecasts_table(conn)
        conn.execute(
            "INSERT INTO forecasts_ensemble VALUES ('fc_100', 'ensemble_mean_v2', 1), ('fc_200', 'ensemble_mean_v2', 2)"
        )
        assert vfa.get_latest_run_id(conn) == "fc_200"
    finally:
        duckdb_io.close_db(conn)


def test_script_fails_when_latest_run_missing_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "resolver.duckdb"
    conn = duckdb_io.get_db(str(db_path))
    try:
        _create_forecasts_table(conn)
        conn.execute(
            """
            INSERT INTO forecasts_ensemble VALUES
            ('fc_100', 'ensemble_mean_v2', 1),
            ('fc_100', 'ensemble_bayesmc_v2', 1),
            ('fc_200', 'ensemble_mean_v2', 2)
            """
        )
    finally:
        duckdb_io.close_db(conn)

    monkeypatch.setattr(sys, "argv", ["verify_forecaster_aggregations.py", "--db", str(db_path)])
    with pytest.raises(SystemExit) as excinfo:
        vfa.main()

    assert "Missing expected model_names" in str(excinfo.value)
