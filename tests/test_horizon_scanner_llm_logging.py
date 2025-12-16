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
