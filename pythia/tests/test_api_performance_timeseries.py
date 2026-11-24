# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""/v1/performance/timeseries and the paired blocks of /v1/performance/scores.

Pins: the forecast-month axis (a run dated through run_provenance, a run-less
reference row dated by its question's forecast of record), mean against SQL
median, the month range, the paired benchmark (only cells both sides scored),
include_test, the blocked-hazard and indicative exclusions, the track filter,
an empty or older DB degrading to empty lists, and ``active_models``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
duckdb = pytest.importorskip("duckdb")

from fastapi.testclient import TestClient

from pythia import config as pythia_config
import pythia.api.app as _app_mod
from pythia.api.app import app
from pythia.api.routes import performance as perf


def _make_db(db_path: Path) -> None:
    con = duckdb.connect(str(db_path))
    con.execute(
        """
        CREATE TABLE hs_runs (hs_run_id TEXT, generated_at TIMESTAMP, is_test BOOLEAN DEFAULT FALSE);
        CREATE TABLE run_provenance (forecaster_run_id TEXT, hs_run_id TEXT);
        CREATE TABLE questions (
          question_id TEXT, iso3 TEXT, hazard_code TEXT, metric TEXT, hs_run_id TEXT,
          track INTEGER, is_test BOOLEAN DEFAULT FALSE
        );
        CREATE TABLE forecasts_ensemble (question_id TEXT, run_id TEXT, is_test BOOLEAN DEFAULT FALSE);
        CREATE TABLE resolutions (question_id TEXT, horizon_m INTEGER, scoring_class TEXT);
        CREATE TABLE scores (
          question_id TEXT, horizon_m INTEGER, score_type TEXT, model_name TEXT,
          value DOUBLE, run_id TEXT, is_test BOOLEAN DEFAULT FALSE
        );
        """
    )
    con.execute(
        """INSERT INTO hs_runs VALUES
        ('hs_aug', TIMESTAMP '2026-08-13 01:00:00', FALSE),
        ('hs_sep', TIMESTAMP '2026-09-13 01:00:00', FALSE),
        ('hs_oct', TIMESTAMP '2026-10-13 01:00:00', FALSE);"""
    )
    con.execute(
        """INSERT INTO run_provenance VALUES
        ('fc_8','hs_aug'), ('fc_9','hs_sep'), ('fc_10','hs_oct');"""
    )
    con.execute(
        """INSERT INTO questions VALUES
        ('QA1','SOM','ACE','FATALITIES','hs_aug',1,FALSE),
        ('QA2','SDN','ACE','FATALITIES','hs_aug',1,FALSE),
        ('QA3','ETH','ACE','FATALITIES','hs_aug',2,FALSE),
        ('QS1','KEN','FL','PA','hs_sep',1,FALSE),
        ('QB1','KEN','FL','EVENT_OCCURRENCE','hs_sep',1,FALSE),
        ('QO1','PHL','TC','PA','hs_oct',1,FALSE),
        ('QDI','SOM','DI','PA','hs_aug',1,FALSE),
        ('QIND','COD','ACE','PA','hs_aug',1,FALSE),
        ('QT','TST','ACE','FATALITIES','hs_aug',1,TRUE);"""
    )
    con.execute(
        """INSERT INTO forecasts_ensemble VALUES
        ('QA1','fc_8',FALSE),('QA2','fc_8',FALSE),('QA3','fc_8',FALSE),
        ('QS1','fc_9',FALSE),('QB1','fc_9',FALSE),('QO1','fc_10',FALSE),
        ('QDI','fc_8',FALSE),('QIND','fc_8',FALSE),('QT','fc_8',TRUE);"""
    )
    con.execute(
        """INSERT INTO resolutions VALUES
        ('QA1',1,NULL),('QA2',1,NULL),('QA3',1,NULL),('QS1',1,NULL),('QB1',1,NULL),
        ('QO1',1,NULL),('QDI',1,NULL),('QIND',1,'indicative'),('QT',1,NULL);"""
    )
    rows = [
        # August: three ensemble cells -> mean 0.4, median 0.2.
        ("QA1", 1, "brier", "ensemble_mean_v2", 0.1, "fc_8", False),
        ("QA2", 1, "brier", "ensemble_mean_v2", 0.2, "fc_8", False),
        ("QA3", 1, "brier", "ensemble_mean_v2", 0.9, "fc_8", False),
        ("QA1", 1, "crps", "ensemble_mean_v2", 0.05, "fc_8", False),
        # A member on QA1/QA2, for the skill block.
        ("QA1", 1, "brier", "gpt-6-sol", 0.15, "fc_8", False),
        ("QA2", 1, "brier", "gpt-6-sol", 0.25, "fc_8", False),
        # Climatology only on QA1/QA2 (run-less, dated by the record run).
        ("QA1", 1, "brier", "__ext_climatology", 0.3, None, False),
        ("QA2", 1, "brier", "__ext_climatology", 0.5, None, False),
        # Sibyl's own variant is not a benchmark.
        ("QA1", 1, "brier", "__ext_sibyl_ref", 0.01, None, False),
        # September SPD and binary.
        ("QS1", 1, "brier", "ensemble_mean_v2", 0.6, "fc_9", False),
        ("QB1", 1, "brier", "ensemble_mean_v2", 0.02, "fc_9", False),
        ("QB1", 1, "brier", "__ext_uniform", 0.25, None, False),
        # October.
        ("QO1", 1, "brier", "ensemble_mean_v2", 0.7, "fc_10", False),
        # Excluded: blocked hazard, indicative ACE/PA, a test question.
        ("QDI", 1, "brier", "ensemble_mean_v2", 1.9, "fc_8", False),
        ("QIND", 1, "brier", "ensemble_mean_v2", 1.8, "fc_8", False),
        ("QT", 1, "brier", "ensemble_mean_v2", 1.7, "fc_8", True),
    ]
    con.executemany("INSERT INTO scores VALUES (?,?,?,?,?,?,?)", rows)
    con.close()


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def _boot(make=True) -> TestClient:
        db_path = tmp_path / "perf.duckdb"
        if make:
            _make_db(db_path)
        else:
            duckdb.connect(str(db_path)).close()
        cfg = tmp_path / "config.yaml"
        cfg.write_text(f"app:\n  db_url: 'duckdb:///{db_path}'\n", encoding="utf-8")
        monkeypatch.setenv("PYTHIA_CONFIG_PATH", str(cfg))
        pythia_config.load.cache_clear()
        _app_mod._READ_CON = None
        return TestClient(app)

    try:
        yield _boot
    finally:
        _app_mod._READ_CON = None
        pythia_config.load.cache_clear()


def _row(rows, **match):
    hits = [r for r in rows if all(r.get(k) == v for k, v in match.items())]
    assert len(hits) == 1, (match, hits)
    return hits[0]


def test_mean_and_median_by_forecast_month(client) -> None:
    body = client().get("/v1/performance/timeseries").json()
    assert body["month_basis"] == "forecast_month"
    aug = _row(body["rows"], month="2026-08", model_name="ensemble_mean_v2",
               score_type="brier", score_family="spd")
    assert aug["avg_value"] == pytest.approx(0.4)
    assert aug["median_value"] == pytest.approx(0.2)
    assert aug["n_questions"] == 3 and aug["n_samples"] == 3
    months = {r["month"] for r in body["rows"]}
    assert months == {"2026-08", "2026-09", "2026-10"}
    # A run-less reference row takes its question's record-run month.
    clim = _row(body["rows"], month="2026-08", model_name="__ext_climatology")
    assert clim["avg_value"] == pytest.approx(0.4)
    # Families are never blended.
    sep_bin = _row(body["rows"], month="2026-09", model_name="ensemble_mean_v2",
                   score_family="binary")
    assert sep_bin["avg_value"] == pytest.approx(0.02)
    # Members and Sibyl's variants are not series on this chart.
    assert not {r["model_name"] for r in body["rows"]} & {"gpt-6-sol", "__ext_sibyl_ref"}


def test_exclusions_match_the_scores_endpoint(client) -> None:
    c = client()
    body = c.get("/v1/performance/timeseries").json()
    vals = [r["avg_value"] for r in body["rows"]]
    # 1.9 (blocked DI), 1.8 (indicative ACE/PA), 1.7 (test) never appear.
    assert max(vals) < 1.0
    with_test = c.get("/v1/performance/timeseries", params={"include_test": "true"}).json()
    aug = _row(with_test["rows"], month="2026-08", model_name="ensemble_mean_v2",
               score_type="brier", score_family="spd")
    assert aug["n_questions"] == 4  # QT joins
    t2 = c.get("/v1/performance/timeseries", params={"track": 2}).json()
    aug2 = _row(t2["rows"], month="2026-08", model_name="ensemble_mean_v2",
                score_type="brier", score_family="spd")
    assert aug2["n_questions"] == 1 and aug2["avg_value"] == pytest.approx(0.9)


def test_month_range(client, monkeypatch: pytest.MonkeyPatch) -> None:
    c = client()
    real = perf._month_cutoff
    monkeypatch.setattr(perf, "_month_cutoff", lambda m: real(m, "2026-10-20"))
    body = c.get("/v1/performance/timeseries", params={"months": 2}).json()
    assert {r["month"] for r in body["rows"]} == {"2026-09", "2026-10"}
    assert {r["month"] for r in body["paired_rows"]} <= {"2026-09", "2026-10"}
    assert real(12, "2026-01-05") == "2025-02"


def test_paired_benchmark_uses_only_shared_cells(client) -> None:
    c = client()
    body = c.get("/v1/performance/timeseries",
                 params={"paired_with": "__ext_climatology"}).json()
    p = _row(body["paired_rows"], month="2026-08", model_name="ensemble_mean_v2",
             reference="__ext_climatology", score_type="brier")
    # Only QA1 and QA2: 0.1 and 0.2 against 0.3 and 0.5; QA3 is left out.
    assert p["n_questions"] == 2
    assert p["model_avg"] == pytest.approx(0.15)
    assert p["reference_avg"] == pytest.approx(0.4)
    assert p["reference_median"] == pytest.approx(0.4)
    assert {r["reference"] for r in body["paired_rows"]} == {"__ext_climatology"}
    # Without the parameter every reference is paired; Sibyl's variants never.
    allp = c.get("/v1/performance/timeseries").json()["paired_rows"]
    assert {r["reference"] for r in allp} == {"__ext_climatology", "__ext_uniform"}


def test_empty_and_older_db(client) -> None:
    body = client(make=False).get("/v1/performance/timeseries").json()
    assert body == {"month_basis": "forecast_month", "rows": [], "paired_rows": []}
    s = client(make=False).get("/v1/performance/scores").json()
    assert s["summary_rows"] == [] and s["run_rows"] == []


def test_no_hs_runs_degrades_to_empty(client, tmp_path: Path) -> None:
    c = client()
    con = duckdb.connect(str(tmp_path / "perf.duckdb"))
    con.execute("DROP TABLE hs_runs")
    con.close()
    _app_mod._READ_CON = None
    resp = c.get("/v1/performance/timeseries")
    assert resp.status_code == 200
    assert resp.json()["rows"] == []


def test_scores_carries_paired_blocks_and_active_models(client) -> None:
    body = client().get("/v1/performance/scores").json()
    ref = _row(body["reference_rows"], model_name="ensemble_mean_v2",
               reference="__ext_climatology", score_type="brier")
    assert ref["n_questions"] == 2 and ref["reference_avg"] == pytest.approx(0.4)
    skill = _row(body["skill_rows"], model_name="gpt-6-sol", score_type="brier")
    assert skill["reference"] == "__ext_climatology"
    assert skill["model_avg"] == pytest.approx(0.2)
    assert skill["reference_avg"] == pytest.approx(0.4)
    assert not [r for r in body["skill_rows"] if r["model_name"].startswith("__ext_")]
    assert isinstance(body["active_models"], list)
    # A true median across hazards, not a median of per-hazard medians.
    ov = _row(body["overall_rows"], model_name="ensemble_mean_v2",
              score_type="brier", score_family="spd")
    # ACE 0.1, 0.2, 0.9; FL 0.6; TC 0.7 -> median 0.6.
    assert ov["median_value"] == pytest.approx(0.6)
    assert ov["n_questions"] == 5
    # The blocked hazard no longer reaches the summary either.
    assert "DI" not in {r["hazard_code"] for r in body["summary_rows"]}


def test_every_configured_model_maps_to_the_name_its_rows_carry() -> None:
    """A configured model whose name matches no stored row would be shown as
    retired. The forecaster names member rows with _provider_display_name and
    the Track 2 rows with the literal track2_flash; active_scored_models must
    agree with both."""
    providers = pytest.importorskip("forecaster.providers")
    from pythia import llm_profiles

    pythia_config.load.cache_clear()
    active = llm_profiles.active_scored_models()
    roles = {a["role"] for a in active}
    assert roles == {"ensemble", "track2"}
    by_id = {e["model_id"]: e for e in llm_profiles.get_ensemble_resolved()}
    for a in active:
        assert a["model_name"], a
        if a["role"] == "ensemble":
            e = by_id[a["model_id"]]
            assert a["model_name"] == providers._provider_display_name(
                e["provider"], e["model_id"], e
            )
    voting = set(llm_profiles.voting_ensemble_model_ids())
    assert {a["model_id"] for a in active if a["role"] == "ensemble"} == voting
    cli_src = (Path(__file__).resolve().parents[2] / "forecaster" / "cli.py").read_text()
    assert f'"{llm_profiles.TRACK2_MODEL_NAME}"' in cli_src
