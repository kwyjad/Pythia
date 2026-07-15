# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Test toggle (include_test) must actually hide test data (July 2026).

Regression tests for the 2026-07-15 dashboard review: with the Test toggle
OFF the Countries and Forecasts pages showed Iran/Somalia rows from
test-mode runs, and the homepage "Last updated" reflected a test run's
timestamps. Root causes covered here:

- ``core._latest_questions_view`` (the latest_only CTE behind /v1/questions
  and /v1/forecasts/ensemble) had no is_test filter.
- ``resolver.query.countries_index`` filtered is_test only in ``_base_rows``;
  ``_latest_hs_run_id``, ``_add_hs_country_list``, ``_update_last_triaged``,
  ``_update_forecasts`` and ``_update_highest_rc`` all leaked test rows.
- ``compute_questions_forecast_summary`` could pick a test run's forecast
  rows for a production question (same-epoch question reuse).
- ``/v1/version`` staleness probes had no test filter, so a test-mode
  publish advanced "Last updated" while the Test-OFF dashboard showed
  nothing new.
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
import pythia.api.app as _app_mod
from pythia.api.app import app
from resolver.query.countries_index import compute_countries_index
from resolver.query.questions_index import compute_questions_forecast_summary


def _write_config(tmp_path: Path, db_path: Path) -> Path:
    cfg = {"app": {"db_url": f"duckdb:///{db_path}"}}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def _build_db(db_path: Path) -> None:
    """One production run (ETH) plus a NEWER test-mode run (IRN, SOM)."""
    con = duckdb.connect(str(db_path), read_only=False)
    con.execute(
        """
        CREATE TABLE questions (
            question_id TEXT, iso3 TEXT, hazard_code TEXT, metric TEXT,
            target_month TEXT, hs_run_id TEXT, status TEXT,
            is_test BOOLEAN DEFAULT FALSE
        )
        """
    )
    con.execute(
        """
        CREATE TABLE hs_runs (
            hs_run_id TEXT, created_at TIMESTAMP, is_test BOOLEAN DEFAULT FALSE
        )
        """
    )
    con.execute(
        """
        CREATE TABLE hs_triage (
            run_id TEXT, iso3 TEXT, hazard_code TEXT, tier TEXT,
            triage_score DOUBLE, need_full_spd BOOLEAN, created_at TIMESTAMP,
            regime_change_level INTEGER, regime_change_score DOUBLE,
            is_test BOOLEAN DEFAULT FALSE
        )
        """
    )
    # Modern columns (month_index/bucket_index/probability) plus the legacy
    # ones /v1/forecasts/ensemble selects (horizon_m/class_bin/p/...).
    con.execute(
        """
        CREATE TABLE forecasts_ensemble (
            question_id TEXT, run_id TEXT, created_at TIMESTAMP,
            model_name TEXT, status TEXT, hazard_code TEXT, metric TEXT,
            month_index INTEGER, bucket_index INTEGER, probability DOUBLE,
            horizon_m INTEGER, class_bin TEXT, p DOUBLE,
            aggregator TEXT, ensemble_version TEXT,
            is_test BOOLEAN DEFAULT FALSE
        )
        """
    )
    con.execute(
        """
        CREATE TABLE scores (
            question_id TEXT, created_at TIMESTAMP, score_type TEXT,
            value DOUBLE, model_name TEXT, horizon_m INTEGER,
            is_test BOOLEAN DEFAULT FALSE
        )
        """
    )
    con.execute(
        "CREATE TABLE bucket_centroids (hazard_code TEXT, metric TEXT, "
        "bucket_index INTEGER, centroid DOUBLE)"
    )

    # Production run (older) covering ETH.
    con.execute(
        "INSERT INTO hs_runs VALUES ('hs_prod', TIMESTAMP '2026-07-01 10:00:00', FALSE)"
    )
    con.execute(
        "INSERT INTO questions VALUES "
        "('ETH_ACE_PA_2026-08', 'ETH', 'ACE', 'PA', '2027-01', 'hs_prod', 'active', FALSE)"
    )
    con.execute(
        "INSERT INTO hs_triage VALUES "
        "('hs_prod', 'ETH', 'ACE', 'priority', 0.8, TRUE, TIMESTAMP '2026-07-01 10:00:00', 1, 0.4, FALSE)"
    )
    con.execute(
        "INSERT INTO forecasts_ensemble "
        "(question_id, run_id, created_at, model_name, status, hazard_code, metric, "
        " month_index, bucket_index, probability, horizon_m, class_bin, p, is_test) VALUES "
        "('ETH_ACE_PA_2026-08', 'fc_prod', TIMESTAMP '2026-07-01 11:00:00', "
        "'ensemble_mean_v2', 'ok', 'ACE', 'PA', 1, 1, 1.0, 1, 'B1', 1.0, FALSE)"
    )

    # Test-mode run (NEWER) covering IRN + SOM, plus a test forecast for the
    # production ETH question (same-epoch question reuse writes test forecast
    # rows against production question_ids).
    con.execute(
        "INSERT INTO hs_runs VALUES ('hs_test', TIMESTAMP '2026-07-14 10:00:00', TRUE)"
    )
    con.execute(
        "INSERT INTO questions VALUES "
        "('IRN_ACE_PA_2026-08', 'IRN', 'ACE', 'PA', '2027-01', 'hs_test', 'active', TRUE), "
        "('SOM_ACE_PA_2026-08', 'SOM', 'ACE', 'PA', '2027-01', 'hs_test', 'active', TRUE)"
    )
    con.execute(
        "INSERT INTO hs_triage VALUES "
        "('hs_test', 'IRN', 'ACE', 'priority', 0.9, TRUE, TIMESTAMP '2026-07-14 10:00:00', 2, 0.6, TRUE), "
        "('hs_test', 'SOM', 'ACE', 'priority', 0.9, TRUE, TIMESTAMP '2026-07-14 10:00:00', 2, 0.6, TRUE)"
    )
    con.execute(
        "INSERT INTO forecasts_ensemble "
        "(question_id, run_id, created_at, model_name, status, hazard_code, metric, "
        " month_index, bucket_index, probability, horizon_m, class_bin, p, is_test) VALUES "
        "('IRN_ACE_PA_2026-08', 'fc_test', TIMESTAMP '2026-07-14 11:00:00', "
        "'ensemble_mean_v2', 'ok', 'ACE', 'PA', 1, 1, 1.0, 1, 'B1', 1.0, TRUE), "
        "('ETH_ACE_PA_2026-08', 'fc_test', TIMESTAMP '2026-07-14 11:00:00', "
        "'ensemble_mean_v2', 'ok', 'ACE', 'PA', 1, 1, 1.0, 1, 'B1', 1.0, TRUE)"
    )
    con.execute(
        "INSERT INTO scores VALUES "
        "('IRN_ACE_PA_2026-08', TIMESTAMP '2026-07-14 12:00:00', 'brier', 0.1, "
        "'ensemble_mean_v2', 1, TRUE)"
    )
    con.execute(
        "INSERT INTO bucket_centroids VALUES ('*', 'PA', 1, 10.0)"
    )
    con.close()


@pytest.fixture()
def api_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[Path, None, None]:
    db_path = tmp_path / "api-test-toggle.duckdb"
    _build_db(db_path)
    config_path = _write_config(tmp_path, db_path)
    monkeypatch.setenv("PYTHIA_CONFIG_PATH", str(config_path))
    pythia_config.load.cache_clear()
    _app_mod._READ_CON = None
    _app_mod._VERSION_PROBE_CACHE = None
    monkeypatch.setattr(
        _app_mod,
        "maybe_sync_latest_db",
        lambda: {
            "db_sha256": "test",
            "latest_hs_run_id": "hs_test",
            "latest_hs_created_at": "2026-07-14T10:00:00",
        },
    )
    try:
        yield db_path
    finally:
        _app_mod._READ_CON = None
        _app_mod._VERSION_PROBE_CACHE = None
        pythia_config.load.cache_clear()


# ---------------------------------------------------------------------------
# /v1/questions (Forecasts page uses latest_only=true)
# ---------------------------------------------------------------------------

def test_questions_latest_only_hides_test_questions(api_env: Path) -> None:
    client = TestClient(app)
    resp = client.get("/v1/questions", params={"latest_only": "true"})
    assert resp.status_code == 200
    rows = resp.json()["rows"]
    assert [r["question_id"] for r in rows] == ["ETH_ACE_PA_2026-08"]


def test_questions_latest_only_shows_test_questions_when_opted_in(api_env: Path) -> None:
    client = TestClient(app)
    resp = client.get(
        "/v1/questions", params={"latest_only": "true", "include_test": "true"}
    )
    assert resp.status_code == 200
    ids = {r["question_id"] for r in resp.json()["rows"]}
    assert ids == {"ETH_ACE_PA_2026-08", "IRN_ACE_PA_2026-08", "SOM_ACE_PA_2026-08"}


def test_forecast_summary_ignores_test_forecast_rows(api_env: Path) -> None:
    """A production question carrying a NEWER test-run forecast row must keep
    its production forecast_date with the test filter active."""
    con = duckdb.connect(str(api_env), read_only=True)
    try:
        summary = compute_questions_forecast_summary(
            con, question_ids=["ETH_ACE_PA_2026-08"], include_test=False
        )
        assert summary["ETH_ACE_PA_2026-08"]["forecast_date"] == "2026-07-01"
        summary_with_test = compute_questions_forecast_summary(
            con, question_ids=["ETH_ACE_PA_2026-08"], include_test=True
        )
        assert summary_with_test["ETH_ACE_PA_2026-08"]["forecast_date"] == "2026-07-14"
    finally:
        con.close()


# ---------------------------------------------------------------------------
# /v1/countries (Countries page + homepage)
# ---------------------------------------------------------------------------

def test_countries_hides_test_run_countries(api_env: Path) -> None:
    client = TestClient(app)
    resp = client.get("/v1/countries")
    assert resp.status_code == 200
    rows = {r["iso3"]: r for r in resp.json()["rows"]}
    # IRN/SOM exist only in the test run: they must not appear at all
    # (previously leaked via the unfiltered hs_triage country-list query).
    assert set(rows) == {"ETH"}
    # And ETH's dates/counters must come from production rows only.
    assert rows["ETH"]["last_triaged"] == "2026-07-01"
    assert rows["ETH"]["last_forecasted"] == "2026-07-01"
    assert rows["ETH"]["n_forecasted"] == 1


def test_countries_shows_test_run_countries_when_opted_in(api_env: Path) -> None:
    client = TestClient(app)
    resp = client.get("/v1/countries", params={"include_test": "true"})
    assert resp.status_code == 200
    rows = {r["iso3"]: r for r in resp.json()["rows"]}
    assert {"ETH", "IRN", "SOM"}.issubset(set(rows))


def test_countries_index_latest_run_selection_skips_test_runs(api_env: Path) -> None:
    """The 'latest HS run' used for the country list must be the newest
    PRODUCTION run when the test filter is active — not the newer test run."""
    con = duckdb.connect(str(api_env), read_only=True)
    try:
        rows = compute_countries_index(con, include_test=False)
        isos = {r["iso3"] for r in rows}
        assert isos == {"ETH"}
        rows_with_test = compute_countries_index(con, include_test=True)
        isos_with_test = {r["iso3"] for r in rows_with_test}
        assert {"ETH", "IRN", "SOM"}.issubset(isos_with_test)
    finally:
        con.close()


# ---------------------------------------------------------------------------
# /v1/forecasts/ensemble (latest_only view)
# ---------------------------------------------------------------------------

def test_forecasts_ensemble_latest_only_hides_test_questions(api_env: Path) -> None:
    client = TestClient(app)
    resp = client.get("/v1/forecasts/ensemble", params={"latest_only": "true"})
    assert resp.status_code == 200
    isos = {r["iso3"] for r in resp.json()["rows"]}
    assert isos == {"ETH"}


# ---------------------------------------------------------------------------
# /v1/version ("Last updated" on the homepage)
# ---------------------------------------------------------------------------

def test_version_last_updated_excludes_test_runs_by_default(api_env: Path) -> None:
    client = TestClient(app)
    resp = client.get("/v1/version")
    assert resp.status_code == 200
    payload = resp.json()
    # The manifest advertises the test run (it has no is_test concept), but
    # the test-aware probe must override it with the latest PRODUCTION run.
    assert payload["latest_hs_run_id"] == "hs_prod"
    assert payload["latest_hs_created_at"] == "2026-07-01T10:00:00"
    # Latest production activity: the 11:00 production forecast write.
    assert payload["latest_data_at"] == "2026-07-01T11:00:00"
    assert payload["latest_forecast_run_id"] == "fc_prod"


def test_version_last_updated_includes_test_runs_when_opted_in(api_env: Path) -> None:
    client = TestClient(app)
    resp = client.get("/v1/version", params={"include_test": "true"})
    assert resp.status_code == 200
    payload = resp.json()
    # With Test ON the newest activity is the test run's 12:00 score write.
    assert payload["latest_data_at"] == "2026-07-14T12:00:00"
    assert payload["latest_forecast_run_id"] == "fc_test"
