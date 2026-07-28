# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""/v1/llm/costs and /v1/llm/costs/summary against the LIVE llm_calls schema.

The July 2026 audit found both endpoints referencing the legacy column set
(created_at/component/llm_profile/tokens_in/tokens_out) that the canonical
llm_calls table does not have — every request 500ed on a production DB and
no test existed to catch it. These tests run against a DB created by
ensure_schema (the real schema), not a hand-rolled fixture shape.
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator

import pytest

pytest.importorskip("fastapi")
duckdb = pytest.importorskip("duckdb")

from fastapi.testclient import TestClient

from pythia import config as pythia_config
import pythia.api.app as _app_mod
from pythia.api.app import app
from pythia.db.schema import ensure_schema


@pytest.fixture()
def api_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    db_path = tmp_path / "api.duckdb"
    con = duckdb.connect(str(db_path))
    ensure_schema(con)  # the REAL schema — the point of these tests
    con.execute(
        """
        INSERT INTO llm_calls (
            call_id, run_id, hs_run_id, phase, model_name, provider, model_id,
            elapsed_ms, prompt_tokens, completion_tokens, total_tokens,
            cost_usd, timestamp, is_test
        ) VALUES
        ('c1', 'fc_1', 'hs_1', 'spd_v2', 'claude-opus-5', 'anthropic',
         'claude-opus-5', 1000, 5000, 900, 5900, 0.05,
         TIMESTAMP '2026-07-20 10:00:00', FALSE),
        ('c2', 'fc_1', 'hs_1', 'hs_triage', 'gemini-3.5-flash', 'google',
         'gemini-3.5-flash', 500, 2000, 300, 2300, 0.01,
         TIMESTAMP '2026-07-21 10:00:00', FALSE),
        ('c3', 'fc_2', 'hs_2', 'spd_v2', 'claude-opus-5', 'anthropic',
         'claude-opus-5', 1000, 4000, 800, 4800, 0.04,
         TIMESTAMP '2026-07-22 10:00:00', TRUE)
        """
    )
    con.close()

    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"app:\n  db_url: 'duckdb:///{db_path}'\n", encoding="utf-8")
    monkeypatch.setenv("PYTHIA_CONFIG_PATH", str(config_path))
    pythia_config.load.cache_clear()
    _app_mod._READ_CON = None
    try:
        yield
    finally:
        _app_mod._READ_CON = None
        pythia_config.load.cache_clear()


def test_llm_costs_returns_rows_on_live_schema(api_env: None) -> None:
    client = TestClient(app)
    resp = client.get("/v1/llm/costs")
    assert resp.status_code == 200, resp.text
    rows = resp.json()["rows"]
    assert len(rows) == 2  # test row excluded by default
    # Newest first by the real timestamp column.
    assert rows[0]["call_id"] == "c2"
    # Multi-KB text columns are excluded from the payload.
    assert "prompt_text" not in rows[0]


def test_llm_costs_phase_and_since_filters(api_env: None) -> None:
    client = TestClient(app)
    resp = client.get("/v1/llm/costs", params={"phase": "spd_v2"})
    assert resp.status_code == 200
    assert [r["call_id"] for r in resp.json()["rows"]] == ["c1"]

    # Legacy alias still accepted.
    resp = client.get("/v1/llm/costs", params={"component": "hs_triage"})
    assert [r["call_id"] for r in resp.json()["rows"]] == ["c2"]

    resp = client.get("/v1/llm/costs", params={"since": "2026-07-21"})
    assert [r["call_id"] for r in resp.json()["rows"]] == ["c2"]


def test_llm_costs_summary_groups_on_live_schema(api_env: None) -> None:
    client = TestClient(app)
    resp = client.get("/v1/llm/costs/summary")
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["group_by"] == ["phase", "model_name"]
    rows = {(r["phase"], r["model_name"]): r for r in payload["rows"]}
    spd = rows[("spd_v2", "claude-opus-5")]
    assert spd["calls"] == 1  # test row excluded
    assert spd["tokens_in"] == 5000
    assert spd["tokens_out"] == 900

    with_test = client.get("/v1/llm/costs/summary", params={"include_test": "true"}).json()
    rows_t = {(r["phase"], r["model_name"]): r for r in with_test["rows"]}
    assert rows_t[("spd_v2", "claude-opus-5")]["calls"] == 2


def test_llm_costs_summary_rejects_unknown_and_absent_fields(api_env: None) -> None:
    client = TestClient(app)
    assert client.get(
        "/v1/llm/costs/summary", params={"group_by": "nope"}
    ).status_code == 400
    # llm_profile is a legacy column the canonical schema doesn't carry —
    # explicit 400, not a binder 500.
    assert client.get(
        "/v1/llm/costs/summary", params={"group_by": "llm_profile"}
    ).status_code == 400
