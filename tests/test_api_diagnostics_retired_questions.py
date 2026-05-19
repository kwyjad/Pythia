# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Regression: `/v1/diagnostics/resolution_rates` and `/v1/diagnostics/summary`
must exclude retired questions so legacy DI/HW/CU rows don't clutter the
dashboard with permanent 0% tiles."""

from __future__ import annotations

from pathlib import Path
from typing import Generator

import pytest
import yaml

duckdb = pytest.importorskip("duckdb")
fastapi_testclient = pytest.importorskip("fastapi.testclient")

from fastapi.testclient import TestClient  # noqa: E402

from pythia import config as pythia_config  # noqa: E402
from pythia.api import app as _app_mod  # noqa: E402
from pythia.api.app import app  # noqa: E402


def _write_config(tmp_path: Path, db_path: Path) -> Path:
    cfg = {"app": {"db_url": f"duckdb:///{db_path}"}}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


@pytest.fixture()
def api_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[Path, None, None]:
    db_path = tmp_path / "api.duckdb"
    con = duckdb.connect(str(db_path), read_only=False)
    con.execute(
        """
        CREATE TABLE questions (
            question_id TEXT,
            hazard_code TEXT,
            metric TEXT,
            status TEXT,
            is_test BOOLEAN DEFAULT FALSE
        );
        """
    )
    con.execute(
        """
        CREATE TABLE resolutions (
            question_id TEXT,
            horizon_m INTEGER,
            value DOUBLE,
            is_test BOOLEAN DEFAULT FALSE
        );
        """
    )
    con.execute(
        """
        CREATE TABLE forecasts_ensemble (
            question_id TEXT,
            is_test BOOLEAN DEFAULT FALSE
        );
        """
    )
    con.execute(
        """
        CREATE TABLE scores (
            question_id TEXT,
            value DOUBLE
        );
        """
    )

    # Two ACE/FATALITIES questions: one active (resolved), one resolved-but-active.
    # One DI/PA question: retired (legacy, should NOT appear in the response).
    # One HW/PA question: also retired.
    # One FL/PA question: active, unresolved.
    con.executemany(
        "INSERT INTO questions(question_id, hazard_code, metric, status) VALUES (?,?,?,?)",
        [
            ("ace1", "ACE", "FATALITIES", "active"),
            ("ace2", "ACE", "FATALITIES", "active"),
            ("di1", "DI", "PA", "retired"),
            ("hw1", "HW", "PA", "retired"),
            ("fl1", "FL", "PA", "active"),
        ],
    )
    con.executemany(
        "INSERT INTO resolutions(question_id, horizon_m, value) VALUES (?,?,?)",
        [
            ("ace1", 1, 5.0),
            ("ace2", 2, 7.0),
            # A stray resolution for a retired question — should still be filtered out.
            ("di1", 1, 0.0),
        ],
    )
    con.executemany(
        "INSERT INTO forecasts_ensemble(question_id) VALUES (?)",
        [("ace1",), ("ace2",), ("di1",), ("fl1",)],
    )
    con.executemany(
        "INSERT INTO scores(question_id, value) VALUES (?,?)",
        [("ace1", 0.5), ("ace2", 0.4), ("di1", 0.9)],
    )
    con.close()

    config_path = _write_config(tmp_path, db_path)
    monkeypatch.setenv("PYTHIA_CONFIG_PATH", str(config_path))
    pythia_config.load.cache_clear()
    # Reset any cached DB connection from prior API tests (test pollution
    # would otherwise leave _READ_CON pointing at a deleted tmp_path DB).
    _app_mod._READ_CON = None
    _app_mod._READ_CON_MTIME = None

    try:
        yield db_path
    finally:
        pythia_config.load.cache_clear()
        _app_mod._READ_CON = None
        _app_mod._READ_CON_MTIME = None


def test_resolution_rates_excludes_retired_questions(api_env: Path) -> None:
    client = TestClient(app)
    resp = client.get("/v1/diagnostics/resolution_rates")
    assert resp.status_code == 200, resp.text
    rows = resp.json()["rows"]
    by_key = {(r["hazard_code"], r["metric"]): r for r in rows}

    # DI/PA and HW/PA are retired → must not appear.
    assert ("DI", "PA") not in by_key, f"retired DI/PA row should be hidden: {rows}"
    assert ("HW", "PA") not in by_key, f"retired HW/PA row should be hidden: {rows}"

    # ACE/FATALITIES and FL/PA still visible.
    assert by_key[("ACE", "FATALITIES")]["total_questions"] == 2
    assert by_key[("ACE", "FATALITIES")]["resolved_questions"] == 2
    assert by_key[("FL", "PA")]["total_questions"] == 1
    assert by_key[("FL", "PA")]["resolved_questions"] == 0


def test_summary_excludes_retired_from_with_x_counts(api_env: Path) -> None:
    client = TestClient(app)
    resp = client.get("/v1/diagnostics/summary")
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # questions_by_status still surfaces the retired bucket for auditing.
    by_status = {r["status"]: r["n"] for r in body["questions_by_status"]}
    assert by_status.get("retired", 0) == 2  # di1 + hw1
    assert by_status.get("active", 0) == 3  # ace1 + ace2 + fl1

    # But the "with X" counts must exclude retired questions:
    #   forecasts: ace1, ace2, fl1 → 3 (di1 has a forecast row but is retired)
    #   resolutions: ace1, ace2 → 2 (di1 has a resolution row but is retired)
    #   scores: ace1, ace2 → 2 (di1 has a score row but is retired)
    assert body["questions_with_forecasts"] == 3
    assert body["questions_with_resolutions"] == 2
    assert body["questions_with_scores"] == 2
