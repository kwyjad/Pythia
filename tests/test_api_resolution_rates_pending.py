# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""`/v1/diagnostics/resolution_rates` must return a `pending_too_new` count
per (hazard, metric) group so the dashboard can distinguish "all questions
from the latest epoch, haven't had a chance to resolve yet" from real 0%
calibration failures."""

from __future__ import annotations

from datetime import date
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
    """A DB with three groups:

      - ACE/FATALITIES: 2 questions from a past epoch (1 resolved, 1 unresolved)
      - FL/PA: 2 questions ENTIRELY from the current month (all pending_too_new)
      - TC/PA: 1 past + 1 future (partially pending — must not flip to "all pending")
    """
    db_path = tmp_path / "api.duckdb"
    con = duckdb.connect(str(db_path), read_only=False)
    con.execute(
        """
        CREATE TABLE questions (
            question_id TEXT,
            hazard_code TEXT,
            metric TEXT,
            status TEXT,
            window_start_date DATE,
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

    today = date.today()
    # "Past" = first day of the month two months ago — guaranteed to be on or
    # before the calendar cutoff (previous complete month).
    if today.month <= 2:
        past = date(today.year - 1, today.month + 10, 1)
    else:
        past = date(today.year, today.month - 2, 1)
    # "Future" = first day of next month — guaranteed to be after the cutoff.
    if today.month == 12:
        future = date(today.year + 1, 1, 1)
    else:
        future = date(today.year, today.month + 1, 1)

    con.executemany(
        """
        INSERT INTO questions(question_id, hazard_code, metric, status, window_start_date)
        VALUES (?,?,?,?,?)
        """,
        [
            ("ace_past_a", "ACE", "FATALITIES", "active", past),
            ("ace_past_b", "ACE", "FATALITIES", "active", past),
            ("fl_future_a", "FL", "PA", "active", future),
            ("fl_future_b", "FL", "PA", "active", future),
            ("tc_past", "TC", "PA", "active", past),
            ("tc_future", "TC", "PA", "active", future),
        ],
    )
    con.execute(
        "INSERT INTO resolutions(question_id, horizon_m, value) VALUES ('ace_past_a', 1, 3.0)"
    )
    con.close()

    config_path = _write_config(tmp_path, db_path)
    monkeypatch.setenv("PYTHIA_CONFIG_PATH", str(config_path))
    pythia_config.load.cache_clear()
    _app_mod._READ_CON = None
    _app_mod._READ_CON_MTIME = None

    try:
        yield db_path
    finally:
        pythia_config.load.cache_clear()
        _app_mod._READ_CON = None
        _app_mod._READ_CON_MTIME = None


def test_pending_too_new_counts(api_env: Path) -> None:
    client = TestClient(app)
    resp = client.get("/v1/diagnostics/resolution_rates")
    assert resp.status_code == 200, resp.text
    rows = resp.json()["rows"]
    by_key = {(r["hazard_code"], r["metric"]): r for r in rows}

    # ACE/FATALITIES — both questions are from a past epoch. None pending.
    ace = by_key[("ACE", "FATALITIES")]
    assert ace["total_questions"] == 2
    assert ace["pending_too_new"] == 0
    assert ace["resolved_questions"] == 1

    # FL/PA — both questions are from the future epoch. Group is fully pending,
    # which is the empty-state condition the dashboard renders specially.
    fl = by_key[("FL", "PA")]
    assert fl["total_questions"] == 2
    assert fl["pending_too_new"] == 2
    assert fl["resolved_questions"] == 0

    # TC/PA — mixed. Pending count is 1 (not equal to total), so the dashboard
    # will NOT flip this tile into empty-state — it's a genuine low-resolution
    # case worth surfacing.
    tc = by_key[("TC", "PA")]
    assert tc["total_questions"] == 2
    assert tc["pending_too_new"] == 1
    assert tc["resolved_questions"] == 0
