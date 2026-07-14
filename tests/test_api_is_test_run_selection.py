# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""is_test gating in latest-run selection (July 2026 pre-production audit).

Covers two gaps of the same class as the #796 run_summary fix:
- core._run_filter_cte: MAX(run_id) per question must not pick a test run
  when the test filter is active.
- /v1/diagnostics/summary: the hs_runs fallback for latest_hs_run must not
  surface a test run with Test OFF.
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator

import pytest
import yaml

duckdb = pytest.importorskip("duckdb")
fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from pythia import config as pythia_config
from pythia.api.app import app
from pythia.api.core import _run_filter_cte


def _cte_selected_run(con, include_test: bool) -> str:
    cte, _join = _run_filter_cte(con, None, include_test=include_test)
    assert cte, "expected a non-empty CTE"
    row = con.execute(
        f"WITH {cte} SELECT run_id FROM fc_run_filter WHERE question_id = 'q1'"
    ).fetchone()
    return row[0]


def test_run_filter_cte_excludes_test_runs() -> None:
    con = duckdb.connect(":memory:")
    con.execute(
        "CREATE TABLE forecasts_ensemble (question_id TEXT, run_id TEXT, "
        "month_index INTEGER, bucket_index INTEGER, probability DOUBLE, "
        "is_test BOOLEAN DEFAULT FALSE)"
    )
    # Production forecast, then a LATER test-run forecast for the same question.
    con.execute(
        "INSERT INTO forecasts_ensemble VALUES "
        "('q1', 'fc_1000_prod', 1, 1, 0.5, FALSE), "
        "('q1', 'fc_2000_test', 1, 1, 0.9, TRUE)"
    )
    assert _cte_selected_run(con, include_test=False) == "fc_1000_prod"
    assert _cte_selected_run(con, include_test=True) == "fc_2000_test"


def test_run_filter_cte_without_is_test_column_backcompat() -> None:
    con = duckdb.connect(":memory:")
    con.execute(
        "CREATE TABLE forecasts_ensemble (question_id TEXT, run_id TEXT, "
        "month_index INTEGER, bucket_index INTEGER, probability DOUBLE)"
    )
    con.execute("INSERT INTO forecasts_ensemble VALUES ('q1', 'fc_1', 1, 1, 0.5)")
    assert _cte_selected_run(con, include_test=False) == "fc_1"


def _write_config(tmp_path: Path, db_path: Path) -> Path:
    cfg = {"app": {"db_url": f"duckdb:///{db_path}"}}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


@pytest.fixture()
def api_env_hs_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[None, None, None]:
    """DB with an older production hs_run and a NEWER test hs_run."""
    db_path = tmp_path / "api-istest-hsruns.duckdb"
    con = duckdb.connect(str(db_path), read_only=False)
    con.execute(
        "CREATE TABLE hs_runs (hs_run_id TEXT, created_at TIMESTAMP, "
        "countries_json TEXT, is_test BOOLEAN DEFAULT FALSE)"
    )
    con.execute(
        "INSERT INTO hs_runs VALUES "
        "('hs_prod', TIMESTAMP '2026-07-01 00:00:00', '[]', FALSE), "
        "('hs_test', TIMESTAMP '2026-07-14 00:00:00', '[]', TRUE)"
    )
    con.close()

    config_path = _write_config(tmp_path, db_path)
    monkeypatch.setenv("PYTHIA_CONFIG_PATH", str(config_path))
    pythia_config.load.cache_clear()
    import pythia.api.app as _app_mod

    _app_mod._READ_CON = None
    # The manifest-cached latest-hs path must not shadow the fallback query.
    import pythia.api.routes.diagnostics as _diag_mod

    monkeypatch.setattr(_diag_mod, "get_cached_latest_hs", lambda: None)
    try:
        yield
    finally:
        _app_mod._READ_CON = None
        pythia_config.load.cache_clear()


def test_diagnostics_summary_latest_hs_respects_test_filter(
    api_env_hs_runs: None,
) -> None:
    client = TestClient(app)

    resp = client.get("/v1/diagnostics/summary")  # include_test defaults False
    assert resp.status_code == 200
    latest = resp.json().get("latest_hs_run") or {}
    assert latest.get("run_id") == "hs_prod"

    resp = client.get("/v1/diagnostics/summary", params={"include_test": "true"})
    assert resp.status_code == 200
    latest = resp.json().get("latest_hs_run") or {}
    assert latest.get("run_id") == "hs_test"
