# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""/v1/interpreter/* — report serving + the attention map.

Pins: latest-selection (highest version of the newest run, test-filtered),
server-side placeholder resolution, the has_interpretation:false empty state
on a pre-interpreter DB (never a 500), version listing, 404 on unknown id,
and the attention map's model preference + test filtering + ln2 scaling.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
duckdb = pytest.importorskip("duckdb")

from fastapi.testclient import TestClient

from pythia import config as pythia_config
import pythia.api.app as _app_mod
from pythia.api.app import app

QID_ACE = "ETH_ACE_FATALITIES_2026-08"
QID_FL = "ETH_FL_PA_2026-08"
QID_KEN = "KEN_FL_PA_2026-08"
QID_TEST = "TST_ACE_FATALITIES_2026-08"


def _content(kind: str = "combined") -> dict:
    return {
        "schema_version": "1",
        "template_version": "v1",
        "kind": kind,
        "headline": "One risk stands out.",
        "attention": [{
            "rank": 1,
            "reason_code": "base_rate_deviation",
            "iso3": "ETH",
            "hazard_code": "ACE",
            "metric": "FATALITIES",
            "question_ids": [QID_ACE],
            "why_it_stands_out": "Moved {{fig:js_vs_baserate}} from its base rate.",
            "decision_point": {"action": "Decide whether to preposition stock.", "deadline_month": "2026-09", "basis": "peak_horizon"},
            "lead_time_months": 2,
        }],
        "performance": {"plain_summary": "No scored run yet."},
    }


def _insert_interp(con, *, iid, kind, run_id, version, status="ok",
                   is_test=False, created_at="2026-08-08 10:00:00"):
    figures = {"per_question": {QID_ACE: {"js_vs_baserate": 0.3466}}, "global": {}}
    con.execute(
        """
        INSERT INTO interpretations
            (interpretation_id, kind, run_id, hs_run_id, scored_run_id, version,
             template_version, model_name, thinking_level, prompt_hash, pack_hash,
             content_json, content_md, figures_json, status, validation_json,
             cost_usd, input_tokens, output_tokens, created_at, is_test)
        VALUES (?, ?, ?, 'hs_1', NULL, ?, 'v1', 'claude-opus-5', 'high', 'ph', 'pk',
                ?, '# md', ?, ?, '{}', 2.0, 100, 50, ?, ?)
        """,
        [iid, kind, run_id, version, json.dumps(_content(kind)),
         json.dumps(figures), status, created_at, is_test],
    )


def _make_db(db_path: Path, *, with_tables: bool = True) -> None:
    con = duckdb.connect(str(db_path))
    if not with_tables:
        con.close()
        return
    from interpreter.store import ensure_table

    ensure_table(con)
    _insert_interp(con, iid="int_combined_fc_1_v1", kind="combined",
                   run_id="fc_1", version=1, created_at="2026-08-08 10:00:00")
    _insert_interp(con, iid="int_combined_fc_1_v2", kind="combined",
                   run_id="fc_1", version=2, created_at="2026-08-08 11:00:00")
    _insert_interp(con, iid="int_scored_2026-08_v1", kind="scored",
                   run_id=None, version=1, created_at="2026-08-08 09:00:00")
    # A NEWER test-mode row that must be invisible with the toggle off.
    _insert_interp(con, iid="int_combined_fc_9_v1", kind="combined",
                   run_id="fc_9", version=1, is_test=True,
                   created_at="2026-08-09 10:00:00")

    con.execute(
        "CREATE TABLE questions (question_id TEXT, iso3 TEXT, hazard_code TEXT, "
        "metric TEXT, is_test BOOLEAN DEFAULT FALSE)"
    )
    con.execute(
        "INSERT INTO questions VALUES "
        f"('{QID_ACE}','ETH','ACE','FATALITIES',FALSE),"
        f"('{QID_FL}','ETH','FL','PA',FALSE),"
        f"('{QID_KEN}','KEN','FL','PA',FALSE),"
        f"('{QID_TEST}','TST','ACE','FATALITIES',TRUE)"
    )
    con.execute(
        "CREATE TABLE forecast_deviation (run_id TEXT, question_id TEXT, "
        "model_name TEXT, iso3 TEXT, hazard_code TEXT, metric TEXT, "
        "score_family TEXT, js_vs_baserate DOUBLE, log_ev_ratio DOUBLE, "
        "eiv_nominal DOUBLE, eiv_per_100k DOUBLE, baserate_source TEXT, "
        "baserate_json TEXT, is_test BOOLEAN DEFAULT FALSE)"
    )
    rows = [
        # ETH/ACE: the mean must be preferred over the bayesmc row, and the
        # figures are deliberately set so preferring the wrong one is visible
        # (mean 0.60 vs bayesmc 0.3466). Owner decision 2026-08-10: BayesMC
        # and the mean diverge most on thin anchors, and BayesMC's figure is
        # what led the August report astray.
        ("fc_1", QID_ACE, "ensemble_bayesmc_v2", "ETH", "ACE", "FATALITIES",
         "spd", 0.3466, 1.2, 200000.0, 180.0, False),
        ("fc_1", QID_ACE, "ensemble_mean_v2", "ETH", "ACE", "FATALITIES",
         "spd", 0.60, 1.2, 200000.0, 180.0, False),
        ("fc_1", QID_FL, "ensemble_mean_v2", "ETH", "FL", "PA",
         "spd", 0.10, 0.2, 50000.0, 40.0, False),
        ("fc_1", QID_KEN, "track2_flash", "KEN", "FL", "PA",
         "spd", 0.20, 0.3, 30000.0, 25.0, False),
        ("fc_1", QID_TEST, "ensemble_mean_v2", "TST", "ACE", "FATALITIES",
         "spd", 0.69, 2.0, 900000.0, 900.0, True),
    ]
    con.executemany(
        "INSERT INTO forecast_deviation VALUES (?,?,?,?,?,?,?,?,?,?,?,NULL,NULL,?)",
        rows,
    )
    con.close()


@pytest.fixture()
def api_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def _boot(with_tables: bool = True) -> TestClient:
        db_path = tmp_path / f"api_{with_tables}.duckdb"
        _make_db(db_path, with_tables=with_tables)
        config_path = tmp_path / f"config_{with_tables}.yaml"
        config_path.write_text(
            f"app:\n  db_url: 'duckdb:///{db_path}'\n", encoding="utf-8"
        )
        monkeypatch.setenv("PYTHIA_CONFIG_PATH", str(config_path))
        pythia_config.load.cache_clear()
        _app_mod._READ_CON = None
        return TestClient(app)

    try:
        yield _boot
    finally:
        _app_mod._READ_CON = None
        pythia_config.load.cache_clear()


def test_latest_highest_version_of_newest_production_run(api_env) -> None:
    client = api_env()
    body = client.get("/v1/interpreter/latest?kind=combined").json()
    assert body["has_interpretation"] is True
    interp = body["interpretation"]
    # The newer fc_9 row is test-mode — invisible with the toggle off.
    assert interp["interpretation_id"] == "int_combined_fc_1_v2"
    assert interp["version"] == 2
    # Server-side placeholder resolution: the raw content keeps the
    # placeholder, the resolved copy substitutes the pack value.
    raw = interp["content"]["attention"][0]["why_it_stands_out"]
    resolved = interp["content_resolved"]["attention"][0]["why_it_stands_out"]
    assert "{{fig:js_vs_baserate}}" in raw
    assert "{{fig:" not in resolved
    assert "a long way from its usual pattern" in resolved
    assert interp["unresolved_figures"] == []
    # Names are stamped server-side from interpreter/names.py, so the
    # dashboard never carries a second copy of the country table.
    entry = interp["content_resolved"]["attention"][0]
    assert entry["country_name"] and entry["country_name"] != entry["iso3"]
    assert entry["hazard_name"] and entry["metric_name"]


def test_latest_include_test_surfaces_test_row(api_env) -> None:
    client = api_env()
    body = client.get("/v1/interpreter/latest?kind=combined&include_test=true").json()
    assert body["interpretation"]["interpretation_id"] == "int_combined_fc_9_v1"


def test_latest_scored_kind_and_bad_kind(api_env) -> None:
    client = api_env()
    body = client.get("/v1/interpreter/latest?kind=scored").json()
    assert body["interpretation"]["kind"] == "scored"
    assert client.get("/v1/interpreter/latest?kind=nonsense").status_code == 400


def test_pre_interpreter_db_has_interpretation_false(api_env) -> None:
    client = api_env(with_tables=False)
    body = client.get("/v1/interpreter/latest").json()
    assert body == {"has_interpretation": False, "kind": "combined"}
    assert client.get("/v1/interpreter/versions").json() == {"rows": []}
    assert client.get("/v1/interpreter/attention_map").json() == {
        "has_data": False, "run_id": None, "rows": [],
    }


def test_versions_listing_and_filters(api_env) -> None:
    client = api_env()
    rows = client.get("/v1/interpreter/versions").json()["rows"]
    assert [r["interpretation_id"] for r in rows] == [
        "int_combined_fc_1_v2", "int_combined_fc_1_v1", "int_scored_2026-08_v1",
    ]
    assert all("content" not in r and "content_json" not in r for r in rows)
    kind_rows = client.get("/v1/interpreter/versions?kind=scored").json()["rows"]
    assert [r["kind"] for r in kind_rows] == ["scored"]
    run_rows = client.get("/v1/interpreter/versions?run_id=fc_1").json()["rows"]
    assert len(run_rows) == 2


def test_get_by_id_and_404(api_env) -> None:
    client = api_env()
    body = client.get("/v1/interpreter/int_combined_fc_1_v1").json()
    assert body["interpretation"]["version"] == 1
    assert client.get("/v1/interpreter/does_not_exist").status_code == 404


def test_attention_map_preference_scaling_and_test_filter(api_env) -> None:
    client = api_env()
    body = client.get("/v1/interpreter/attention_map").json()
    assert body["has_data"] is True
    assert body["run_id"] == "fc_1"
    by_iso3 = {r["iso3"]: r for r in body["rows"]}
    # The test-mode question is filtered out entirely.
    assert set(by_iso3) == {"ETH", "KEN"}
    eth = by_iso3["ETH"]
    # The mean (0.60) is preferred over the bayesmc row (0.3466) for QID_ACE,
    # and ACE beats FL (0.10) as ETH's top entry.
    assert eth["hazard_code"] == "ACE"
    assert eth["js_vs_baserate"] == pytest.approx(0.60)
    assert eth["attention"] == pytest.approx(0.60 / math.log(2.0))
    assert eth["n_questions"] == 2
    assert by_iso3["KEN"]["attention"] == pytest.approx(0.20 / math.log(2.0))


def test_attention_map_include_test(api_env) -> None:
    client = api_env()
    body = client.get("/v1/interpreter/attention_map?include_test=true").json()
    by_iso3 = {r["iso3"]: r for r in body["rows"]}
    assert "TST" in by_iso3
    assert by_iso3["TST"]["attention"] == pytest.approx(0.69 / math.log(2.0), abs=1e-6)
