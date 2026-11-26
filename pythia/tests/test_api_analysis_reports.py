# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""/v1/analysis_reports/*: grouping, reruns, placeholders, 404s, degraded index.

The index comes from the analysis-reports release; the HTTP seam
(``routes.analysis_reports._fetch``) is patched, so nothing here reaches the
network.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
duckdb = pytest.importorskip("duckdb")

from fastapi.testclient import TestClient

import pythia.api.app as _app_mod
from pythia import config as pythia_config
from pythia.api.app import app
from pythia.api.routes import analysis_reports as ar

SCORING = {"date": "2026-11-11", "kind": "scoring", "fc_run_id": None,
           "publish_run_id": "43009722469", "folder_name": "2026-11-11__scoring__gh43009722469"}
FORECAST = {"date": "2026-10-13", "kind": "forecast", "fc_run_id": "fc_1791866322",
            "publish_run_id": "38355281121",
            "folder_name": "2026-10-13__forecast__fc_1791866322__gh38355281121"}


def _rec(report_id, kind, period, chain=None, **extra):
    return {
        "report_id": report_id, "type": kind, "display_name": ar.DISPLAY_NAMES[kind],
        "period": period, "chain": chain, "rerun_of": None, "rerun_date": None,
        "title": report_id, "dry_run": False, "prepared": "2026-11-20", "summary": "s",
        "verdict": None, "sha256": report_id, "md_asset": f"{report_id}.md",
        "pdf_asset": f"{report_id}.pdf", **extra,
    }


INDEX = {
    "schema_version": 1,
    "generated_at": "2026-11-26T09:41:00+00:00",
    "reports": [
        _rec("fred_performance_2026-11", "skill", "2026-11", SCORING, dry_run=True),
        _rec("fred_performance_2026-11__rerun_2026-11-28", "skill", "2026-11", SCORING,
             rerun_of="fred_performance_2026-11", rerun_date="2026-11-28",
             prepared="2026-11-28"),
        _rec("fred_forecast_review_2026-10-13__fc_1791866322", "forecast", "2026-10", FORECAST,
             verdict="worked with faults", file_fc_run_id="fc_1791866322",
             file_date="2026-10-13"),
    ],
}


class _Resp:
    def __init__(self, payload=None, text=""):
        self._payload, self.text = payload, text

    def json(self):
        return self._payload


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db = tmp_path / "api.duckdb"
    con = duckdb.connect(str(db))
    con.execute("CREATE TABLE forecasts_ensemble (run_id TEXT, question_id TEXT, "
                "created_at TIMESTAMP, is_test BOOLEAN)")
    con.execute("INSERT INTO forecasts_ensemble VALUES "
                "('fc_1791866322', 'q1', '2026-10-13 05:10:00', FALSE),"
                "('fc_1794543371', 'q1', '2026-11-13 05:10:00', FALSE),"
                "('fc_1794600000', 'q1', '2026-11-14 05:10:00', TRUE),"
                "('fc_1788237725', 'q1', '2026-09-01 05:10:00', FALSE)")
    con.close()
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"app:\n  db_url: 'duckdb:///{db}'\n", encoding="utf-8")
    monkeypatch.setenv("PYTHIA_CONFIG_PATH", str(cfg))
    monkeypatch.setenv("PYTHIA_DATA_REPO", "owner/repo")
    pythia_config.load.cache_clear()
    _app_mod._READ_CON = None
    ar.reset_cache()
    calls = {"n": 0, "fail": False}

    def fake_fetch(url, timeout=10.0):
        calls["n"] += 1
        if calls["fail"]:
            raise RuntimeError("github down")
        if "index.json" in url:
            return _Resp(INDEX)
        return _Resp(text="# A report\n\n## 1. Summary\n\nText.\n")

    monkeypatch.setattr(ar, "_fetch", fake_fetch)
    try:
        yield TestClient(app), calls
    finally:
        _app_mod._READ_CON = None
        pythia_config.load.cache_clear()
        ar.reset_cache()


def test_groups_by_cycle_newest_first_with_reruns_nested(client) -> None:
    c, _ = client
    body = c.get("/v1/analysis_reports").json()
    assert body["available"] is True
    periods = [cy["period"] for cy in body["cycles"]]
    assert periods == ["2026-11", "2026-10"]
    nov = body["cycles"][0]
    rnd = nov["scoring_rounds"][0]
    assert rnd["publish_run_id"] == "43009722469" and rnd["date"] == "2026-11-11"
    skill, resolver = rnd["reports"]
    # The rerun is the default and the original sits under it as a version.
    assert skill["report_id"] == "fred_performance_2026-11__rerun_2026-11-28"
    assert [v["report_id"] for v in skill["versions"]] == [
        "fred_performance_2026-11__rerun_2026-11-28", "fred_performance_2026-11"]
    # The round has no resolver review: a placeholder stands in its place.
    assert resolver["type"] == "resolver" and resolver["available"] is False


def test_forecast_runs_and_pending_placeholders(client) -> None:
    c, _ = client
    body = c.get("/v1/analysis_reports").json()
    by_period = {cy["period"]: cy for cy in body["cycles"]}
    oct_runs = by_period["2026-10"]["forecast_runs"]
    assert [r["fc_run_id"] for r in oct_runs] == ["fc_1791866322"]
    assert oct_runs[0]["reports"][0]["verdict"] == "worked with faults"
    nov_runs = {r["fc_run_id"]: r for r in by_period["2026-11"]["forecast_runs"]}
    # The 13 November production run has no review yet; the test run never gets one,
    # and runs before reviews began (1 September) are left out.
    assert set(nov_runs) == {"fc_1794543371"}
    pending = nov_runs["fc_1794543371"]["reports"][0]
    assert pending["available"] is False and pending["pending"] is True
    assert "2026-09" not in by_period


def test_filters_type_year_and_dry_runs(client) -> None:
    c, _ = client
    body = c.get("/v1/analysis_reports", params={"type": "forecast"}).json()
    for cy in body["cycles"]:
        assert cy["scoring_rounds"] == []
    no_dry = c.get("/v1/analysis_reports", params={"include_dry_runs": "false"}).json()
    nov = {cy["period"]: cy for cy in no_dry["cycles"]}["2026-11"]
    # Only the rerun is a non-dry report, so it is still the round's skill report.
    assert nov["scoring_rounds"][0]["reports"][0]["report_id"].endswith("rerun_2026-11-28")
    assert c.get("/v1/analysis_reports", params={"year": "2025"}).json()["cycles"] == []


def test_unknown_id_is_404_and_downloads_redirect(client) -> None:
    c, _ = client
    assert c.get("/v1/analysis_reports/nope/content").status_code == 404
    assert c.get("/v1/analysis_reports/../../etc/download").status_code == 404
    r = c.get("/v1/analysis_reports/fred_performance_2026-11/download",
              params={"format": "md"}, follow_redirects=False)
    assert r.status_code == 302
    assert r.headers["location"] == (
        "https://github.com/owner/repo/releases/download/analysis-reports/fred_performance_2026-11.md")
    body = c.get("/v1/analysis_reports/fred_performance_2026-11/content").json()
    assert body["markdown"].startswith("# A report")


def test_index_failure_serves_last_good_copy_then_empty(client, monkeypatch) -> None:
    c, calls = client
    assert c.get("/v1/analysis_reports").json()["available"] is True
    calls["fail"] = True
    monkeypatch.setattr(ar, "INDEX_TTL_S", 0)
    body = c.get("/v1/analysis_reports").json()
    assert body["available"] is True and body["stale"] is True
    ar.reset_cache()
    empty = c.get("/v1/analysis_reports")
    assert empty.status_code == 200
    assert empty.json() == {"available": False, "generated_at": None, "cycles": [], "years": []}


def test_group_reports_without_a_db_has_no_placeholders() -> None:
    cycles = ar.group_reports(INDEX["reports"], forecast_runs=None)
    runs = [r for cy in cycles for r in cy["forecast_runs"]]
    assert [r["fc_run_id"] for r in runs] == ["fc_1791866322"]
