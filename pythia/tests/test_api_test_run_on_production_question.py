# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""A production question forecast by a test run shows no change in any
production view (Oct 2026).

Question ids are epoch-keyed, so a TEST-mode run in the same month as a
production run forecasts the SAME production question row
(``questions.is_test = FALSE``) and writes its forecast rows with
``is_test = TRUE``. Any query that filters only on the question and then picks
the "latest" forecast would put the test forecast on a production page.

Each test builds two databases from ``pythia.db.schema.ensure_schema``: one
holding the production run alone, one holding the production run plus a NEWER
test run on the same question, and asserts that every production view
(``include_test`` false, the default) answers identically on both — and that
``include_test=true`` does show the test run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pytest
import yaml

duckdb = pytest.importorskip("duckdb")
fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from pythia import config as pythia_config
import pythia.api.app as _app_mod
from pythia.api.app import app
from pythia.db.schema import ensure_schema
from resolver.query.countries_index import compute_countries_index
from resolver.query.downloads import (
    build_ensemble_scores_export,
    build_forecast_spd_export,
    build_model_scores_export,
    build_rationale_export,
)

QID = "SOM_ACE_PA_2026-11"
PROD_RUN = "fc_100"
TEST_RUN = "fc_200"
PROD_VEC = [0.50, 0.20, 0.10, 0.10, 0.05, 0.05]
TEST_VEC = [0.05, 0.05, 0.10, 0.10, 0.20, 0.50]


def _cols(con, table: str) -> set[str]:
    return {r[1] for r in con.execute(f"PRAGMA table_info('{table}')").fetchall()}


def _insert(con, table: str, row: dict[str, Any]) -> None:
    cols = _cols(con, table)
    row = {k: v for k, v in row.items() if k in cols}
    names = ", ".join(row)
    marks = ", ".join(["?"] * len(row))
    con.execute(f"INSERT INTO {table} ({names}) VALUES ({marks})", list(row.values()))


def _write_run(con, *, run_id: str, hs_run_id: str, is_test: bool, ts: str,
               vec: list[float], text: str) -> None:
    _insert(con, "hs_runs", {
        "hs_run_id": hs_run_id, "generated_at": ts, "created_at": ts,
        "is_test": is_test,
    })
    _insert(con, "hs_triage", {
        "run_id": hs_run_id, "iso3": "SOM", "hazard_code": "ACE",
        "tier": "rc_promoted", "triage_score": 0.0, "need_full_spd": True,
        "regime_change_level": 2 if is_test else 1,
        "regime_change_score": 0.9 if is_test else 0.3,
        "track": 1, "created_at": ts, "is_test": is_test,
    })
    for model in ("ensemble_mean_v2", "gpt-6-sol"):
        for m in range(1, 7):
            for b, p in enumerate(vec, start=1):
                base = {
                    "run_id": run_id, "question_id": QID, "model_name": model,
                    "month_index": m, "bucket_index": b, "probability": p,
                    "horizon_m": m, "class_bin": str(b), "p": p,
                    "status": "ok", "human_explanation": text,
                    "is_test": is_test,
                }
                if model == "ensemble_mean_v2":
                    _insert(con, "forecasts_ensemble", {
                        **base, "iso3": "SOM", "hazard_code": "ACE",
                        "metric": "PA", "created_at": ts,
                        "aggregator": "mean", "ensemble_version": "v2",
                    })
                _insert(con, "forecasts_raw", {**base, "ok": True})
    _insert(con, "scenarios", {
        "run_id": run_id, "iso3": "SOM", "hazard_code": "ACE", "metric": "PA",
        "scenario_type": "base", "bucket_label": "1", "probability": vec[0],
        "text": text, "created_at": ts, "is_test": is_test,
    })
    _insert(con, "forecast_deviation", {
        "run_id": run_id, "question_id": QID, "model_name": "ensemble_mean_v2",
        "iso3": "SOM", "hazard_code": "ACE", "metric": "PA",
        "score_family": "spd", "js_vs_baserate": 0.6 if is_test else 0.1,
        "log_ev_ratio": 0.0, "eiv_nominal": 1000.0, "created_at": ts,
        "is_test": is_test,
    })
    _insert(con, "question_run_metrics", {
        "run_id": run_id, "question_id": QID, "iso3": "SOM",
        "hazard_code": "ACE", "metric": "PA", "wall_ms": 10, "cost_usd": 1.0,
        "is_test": is_test,
    })
    con.execute(
        "INSERT INTO scores (question_id, horizon_m, metric, score_type, model_name, "
        "value, created_at, run_id, is_test) VALUES (?, 1, 'PA', 'brier', "
        "'ensemble_mean_v2', ?, ?, ?, ?)",
        [QID, 0.9 if is_test else 0.2, ts, run_id, is_test],
    )


def _build_db(db_path: Path, *, with_test_run: bool) -> None:
    con = duckdb.connect(str(db_path))
    ensure_schema(con)
    # Columns the legacy /v1/forecasts/ensemble route selects.
    for col in ("aggregator", "ensemble_version"):
        if col not in _cols(con, "forecasts_ensemble"):
            con.execute(f"ALTER TABLE forecasts_ensemble ADD COLUMN {col} TEXT")
    con.execute(
        "CREATE TABLE IF NOT EXISTS scores (question_id TEXT, horizon_m INTEGER, "
        "metric TEXT, score_type TEXT, model_name TEXT, value DOUBLE, "
        "created_at TIMESTAMP, run_id TEXT, is_test BOOLEAN DEFAULT FALSE)"
    )
    for col, typ in (("run_id", "TEXT"), ("is_test", "BOOLEAN")):
        if col not in _cols(con, "scores"):
            con.execute(f"ALTER TABLE scores ADD COLUMN {col} {typ}")
    _insert(con, "questions", {
        "question_id": QID, "hs_run_id": "hs_prod", "iso3": "SOM",
        "hazard_code": "ACE", "metric": "PA", "target_month": "2027-04",
        "window_start_date": "2026-11-01", "window_end_date": "2027-04-30",
        "wording": "People displaced by conflict in Somalia",
        "status": "active", "track": 1, "is_test": False,
    })
    _write_run(con, run_id=PROD_RUN, hs_run_id="hs_prod", is_test=False,
               ts="2026-10-13 05:00:00", vec=PROD_VEC, text="production reasoning")
    if with_test_run:
        _write_run(con, run_id=TEST_RUN, hs_run_id="hs_test", is_test=True,
                   ts="2026-10-20 05:00:00", vec=TEST_VEC, text="test reasoning")
    con.close()


@pytest.fixture()
def dbs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    prod_only = tmp_path / "prod_only.duckdb"
    with_test = tmp_path / "with_test.duckdb"
    _build_db(prod_only, with_test_run=False)
    _build_db(with_test, with_test_run=True)
    monkeypatch.setattr(
        _app_mod, "maybe_sync_latest_db", lambda *a, **k: {"db_sha256": "test"},
    )
    monkeypatch.setenv("PYTHIA_EXPORT_TMP_DIR", str(tmp_path / "exports"))
    yield {"prod_only": prod_only, "with_test": with_test, "tmp": tmp_path}
    _reset(None, monkeypatch)


def _reset(db_path: Path | None, monkeypatch: pytest.MonkeyPatch) -> None:
    try:
        from pythia.db.schema import close_pooled_connections

        close_pooled_connections()
    except Exception:
        pass
    if _app_mod._READ_CON is not None:
        try:
            _app_mod._READ_CON.close()
        except Exception:
            pass
    _app_mod._READ_CON = None
    _app_mod._VERSION_PROBE_CACHE = None
    pythia_config.load.cache_clear()
    if db_path is not None:
        cfg = db_path.parent / f"{db_path.stem}.config.yaml"
        cfg.write_text(
            yaml.safe_dump({"app": {"db_url": f"duckdb:///{db_path}"}}), encoding="utf-8"
        )
        monkeypatch.setenv("PYTHIA_CONFIG_PATH", str(cfg))


def _get(db_path: Path, monkeypatch, path: str, **params) -> Any:
    _reset(db_path, monkeypatch)
    resp = TestClient(app).get(path, params=params)
    assert resp.status_code == 200, (path, resp.status_code, resp.text[:500])
    return resp.json()


def _with_con(db_path: Path, fn: Callable) -> Any:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        return fn(con)
    finally:
        con.close()


def _same_in_production(dbs, monkeypatch, path: str, **params) -> tuple[Any, Any]:
    before = _get(dbs["prod_only"], monkeypatch, path, **params)
    after = _get(dbs["with_test"], monkeypatch, path, **params)
    assert after == before, f"{path} changed in a production view after a test run"
    return before, after


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

def test_question_bundle_keeps_the_production_forecast(dbs, monkeypatch) -> None:
    params = {"question_id": QID, "include_llm_calls": "false"}
    before, _ = _same_in_production(dbs, monkeypatch, "/v1/question_bundle", **params)
    assert before["forecast"]["forecaster_run_id"] == PROD_RUN
    probs = sorted(
        (r["bucket_index"], r["probability"])
        for r in before["forecast"]["ensemble_spd"] if r["month_index"] == 1
    )
    assert [p for _, p in probs] == PROD_VEC
    assert {r["run_id"] for r in before["forecast"]["raw_spd"]} == {PROD_RUN}
    assert {r["text"] for r in before["forecast"]["scenario_writer"]} == {"production reasoning"}
    assert {s["value"] for s in before["context"]["scores"]} == {0.2}

    opted_in = _get(dbs["with_test"], monkeypatch, "/v1/question_bundle",
                    include_test="true", **params)
    assert opted_in["forecast"]["forecaster_run_id"] == TEST_RUN


def test_forecasts_ensemble_latest_only(dbs, monkeypatch) -> None:
    before, _ = _same_in_production(dbs, monkeypatch, "/v1/forecasts/ensemble")
    assert {r["p"] for r in before["rows"]} == set(PROD_VEC)
    opted_in = _get(dbs["with_test"], monkeypatch, "/v1/forecasts/ensemble",
                    include_test="true")
    assert len(opted_in["rows"]) == 2 * len(before["rows"])


def test_questions_page(dbs, monkeypatch) -> None:
    before, _ = _same_in_production(dbs, monkeypatch, "/v1/questions", latest_only="true")
    assert before["rows"][0]["forecast_date"] == "2026-10-13"
    opted_in = _get(dbs["with_test"], monkeypatch, "/v1/questions",
                    latest_only="true", include_test="true")
    assert opted_in["rows"][0]["forecast_date"] == "2026-10-20"


@pytest.mark.parametrize("params", [
    {"metric": "PA"},
    {"metric": "PA", "target_month": "2027-04"},
])
def test_risk_index(dbs, monkeypatch, params) -> None:
    before, _ = _same_in_production(dbs, monkeypatch, "/v1/risk_index", **params)
    opted_in = _get(dbs["with_test"], monkeypatch, "/v1/risk_index",
                    include_test="true", **params)
    assert opted_in != before


def test_countries(dbs, monkeypatch) -> None:
    before, _ = _same_in_production(dbs, monkeypatch, "/v1/countries")
    assert before["rows"][0]["last_forecasted"] == "2026-10-13"


def test_run_summary(dbs, monkeypatch) -> None:
    before, _ = _same_in_production(dbs, monkeypatch, "/v1/diagnostics/run_summary")
    assert before.get("run_id") == PROD_RUN


def test_kpi_scopes_and_summary(dbs, monkeypatch) -> None:
    _same_in_production(dbs, monkeypatch, "/v1/diagnostics/kpi_scopes")
    _same_in_production(dbs, monkeypatch, "/v1/diagnostics/summary")


def test_interpreter_attention_map(dbs, monkeypatch) -> None:
    before, _ = _same_in_production(dbs, monkeypatch, "/v1/interpreter/attention_map")
    assert before["run_id"] == PROD_RUN
    opted_in = _get(dbs["with_test"], monkeypatch, "/v1/interpreter/attention_map",
                    include_test="true")
    assert opted_in["run_id"] == TEST_RUN


def test_version_probes(dbs, monkeypatch) -> None:
    before = _get(dbs["prod_only"], monkeypatch, "/v1/version")
    after = _get(dbs["with_test"], monkeypatch, "/v1/version")
    for key in ("latest_forecast_month", "latest_forecast_run_id", "latest_forecast_at"):
        assert after.get(key) == before.get(key), key
    assert before.get("latest_forecast_run_id") == PROD_RUN


# ---------------------------------------------------------------------------
# Query modules (downloads, countries index)
# ---------------------------------------------------------------------------

def _frames_equal(a, b) -> bool:
    a = a.reset_index(drop=True)
    b = b.reset_index(drop=True)
    return a.fillna("<NA>").astype(str).equals(b.fillna("<NA>").astype(str))


@pytest.mark.parametrize("builder", [
    lambda con, t: build_forecast_spd_export(con, include_test=t),
    lambda con, t: build_rationale_export(con, "ACE", include_test=t),
    lambda con, t: build_ensemble_scores_export(con, "ensemble_mean", include_test=t),
    lambda con, t: build_model_scores_export(con, include_test=t),
], ids=["forecast_spd", "rationale", "ensemble_scores", "model_scores"])
def test_downloads_exports(dbs, builder) -> None:
    before = _with_con(dbs["prod_only"], lambda c: builder(c, False))
    after = _with_con(dbs["with_test"], lambda c: builder(c, False))
    assert not before.empty
    assert _frames_equal(before, after), (before.head().to_dict(), after.head().to_dict())
    opted_in = _with_con(dbs["with_test"], lambda c: builder(c, True))
    assert not _frames_equal(before, opted_in)


def test_forecast_spd_export_carries_production_probabilities(dbs) -> None:
    df = _with_con(dbs["with_test"], lambda c: build_forecast_spd_export(c, include_test=False))
    row = df[df["model"] == "ensemble_mean_v2"].iloc[0]
    assert [row[f"SPD_{i}"] for i in range(1, 7)] == PROD_VEC


def test_countries_index(dbs) -> None:
    before = _with_con(dbs["prod_only"], lambda c: compute_countries_index(c))
    after = _with_con(dbs["with_test"], lambda c: compute_countries_index(c))
    assert after == before
