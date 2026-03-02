# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import json
from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb")

from horizon_scanner.llm_logging import log_hs_llm_call
from pythia.db.schema import connect, ensure_schema


class _DummySpec:
    provider = "test-provider"
    model_id = "test-model"
    temperature = 0.1


def test_log_hs_llm_call_records_hs_triage(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    db_path = tmp_path / "hs_logging.duckdb"
    monkeypatch.setenv("PYTHIA_DB_URL", f"duckdb:///{db_path}")

    con = connect(read_only=False)
    try:
        ensure_schema(con)
    finally:
        con.close()

    usage = {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12, "elapsed_ms": 42}

    log_hs_llm_call(
        hs_run_id="hs_test",
        iso3="ETH",
        hazard_code="ACE",
        model_spec=_DummySpec(),
        prompt_text="PROMPT",
        response_text="RESPONSE",
        usage=usage,
        error_text=None,
    )

    con = connect(read_only=False)
    try:
        rows = con.execute(
            "SELECT phase, iso3, hazard_code, usage_json FROM llm_calls"
        ).fetchall()
    finally:
        con.close()

    assert rows, "llm_calls should contain the logged HS triage call"
    phase, iso3, hazard_code, usage_json = rows[0]
    assert phase == "hs_triage"
    assert iso3 == "ETH"
    assert hazard_code == "ACE"

    usage_loaded = json.loads(usage_json)
    assert usage_loaded.get("elapsed_ms") == 42


def test_run_hs_logs_single_llm_call_per_country(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    db_path = tmp_path / "hs_llm_call.duckdb"
    monkeypatch.setenv("PYTHIA_DB_URL", f"duckdb:///{db_path}")

    con = connect(read_only=False)
    try:
        ensure_schema(con)
    finally:
        con.close()

    from horizon_scanner import horizon_scanner
    from forecaster.providers import ModelSpec

    triage_payload = {
        "hazards": {
            "FL": {
                "tier": "priority",
                "triage_score": 0.82,
                "drivers": [],
                "regime_shifts": [],
                "data_quality": {},
            },
            "DR": {
                "tier": "priority",
                "triage_score": 0.55,
                "drivers": [],
                "regime_shifts": [],
                "data_quality": {},
            },
        }
    }
    response_text = f"```json\n{json.dumps(triage_payload)}\n```"
    usage = {"elapsed_ms": 120, "total_tokens": 10}

    async def _fake_call(prompt_text: str, *, run_id: str | None = None, fallback_specs=None):
        return response_text, usage, None, ModelSpec(name="Test", provider="test", model_id="test-model")

    monkeypatch.setattr(horizon_scanner, "_call_hs_model", _fake_call)

    horizon_scanner._run_hs_for_country("hs_test", "ETH", "Ethiopia")

    con = connect(read_only=False)
    try:
        llm_rows = con.execute(
            "SELECT phase, iso3, hazard_code FROM llm_calls WHERE phase = 'hs_triage'"
        ).fetchall()
        triage_llm_rows = con.execute(
            "SELECT phase, iso3, hazard_code FROM llm_calls WHERE phase = 'hs_triage' AND hazard_code LIKE 'TRIAGE_%'"
        ).fetchall()
        triage_rows = con.execute(
            "SELECT hazard_code FROM hs_triage WHERE run_id = ? ORDER BY hazard_code", ["hs_test"]
        ).fetchall()
    finally:
        con.close()

    assert len(llm_rows) >= 2, "Expected at least two hs_triage llm_calls rows (RC + triage)"
    assert len(triage_llm_rows) >= 2, "Expected triage LLM calls to be logged"
    for row in triage_llm_rows:
        phase, iso3, hazard_code = row
        assert phase == "hs_triage"
        assert iso3 == "ETH"
        assert hazard_code.startswith("TRIAGE_")

    hazard_codes = {row[0] for row in triage_rows}
    assert "DR" in hazard_codes
    assert "FL" in hazard_codes
