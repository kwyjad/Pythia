# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import json
from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb")

from horizon_scanner.llm_logging import (
    GROUNDING_BREAKER_MODEL_ID,
    GROUNDING_FAILED_MODEL_ID,
    GROUNDING_UNAVAILABLE_MODEL_ID,
    log_hs_llm_call,
    resolve_grounding_model_id,
)
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


def test_log_hs_llm_call_records_call_type_purpose(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """call_type carries the call's purpose; the default stays 'chat'.

    phase must remain 'hs_triage' for every HS row (the costs page groups by
    phase) — the purpose lives in call_type, making HS calls queryable without
    string-parsing the synthetic hazard_code encoding.
    """
    db_path = tmp_path / "hs_call_type.duckdb"
    monkeypatch.setenv("PYTHIA_DB_URL", f"duckdb:///{db_path}")

    con = connect(read_only=False)
    try:
        ensure_schema(con)
    finally:
        con.close()

    usage = {"total_tokens": 3}
    log_hs_llm_call(
        hs_run_id="hs_ct",
        iso3="ETH",
        hazard_code="rc_ACE_pass_1",
        model_spec=_DummySpec(),
        prompt_text="P",
        response_text="R",
        usage=usage,
        error_text=None,
        call_type="rc_pass_1",
    )
    log_hs_llm_call(
        hs_run_id="hs_ct",
        iso3="ETH",
        hazard_code="grounding_ACE",
        model_spec=_DummySpec(),
        prompt_text="P",
        response_text="R",
        usage=usage,
        error_text=None,
        call_type="rc_grounding",
    )
    # A call site that doesn't pass call_type keeps the historical default.
    log_hs_llm_call(
        hs_run_id="hs_ct",
        iso3="ETH",
        hazard_code="ACE",
        model_spec=_DummySpec(),
        prompt_text="P",
        response_text="R",
        usage=usage,
        error_text=None,
    )

    con = connect(read_only=False)
    try:
        rows = con.execute(
            "SELECT hazard_code, call_type, phase FROM llm_calls WHERE hs_run_id = 'hs_ct' "
            "ORDER BY hazard_code"
        ).fetchall()
    finally:
        con.close()

    # The writer uppercases hazard_code (the synthetic-label convention).
    by_hazard = {r[0]: (r[1], r[2]) for r in rows}
    assert by_hazard["RC_ACE_PASS_1"] == ("rc_pass_1", "hs_triage")
    assert by_hazard["GROUNDING_ACE"] == ("rc_grounding", "hs_triage")
    assert by_hazard["ACE"] == ("chat", "hs_triage")


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


class TestResolveGroundingModelId:
    """A grounding attempt that never reached a backend must not be logged as a model.

    Before this, both grounding log sites defaulted ``model_id`` to the bare string
    "unknown", which surfaced on the Costs page as if "unknown" were a model when it
    actually meant "this country-hazard pair got no grounding evidence at all".

    The pack shapes below mirror the real producers — keep them in sync with
    ``pythia/web_research/backends/brave_search.py`` and the ``all_failed`` fallback
    packs in ``regime_change_llm.py`` / ``triage.py``.
    """

    def test_real_backend_model_id_passes_through(self) -> None:
        # brave_search.py success path — including a run that found 0 sources, which
        # still cost money and must stay attributed to the backend that ran.
        pack = {
            "sources": [],
            "grounded": False,
            "error": {"type": "no_results", "message": "Brave Search returned no results"},
            "debug": {
                "provider": "brave",
                "model_id": "brave-web-search",
                "selected_model_id": "brave-web-search",
                "grounding_backend": "brave_search",
            },
        }
        assert resolve_grounding_model_id(pack) == "brave-web-search"

    def test_selected_model_id_is_used_when_model_id_absent(self) -> None:
        pack = {"debug": {"selected_model_id": "gemini-2.5-flash", "grounding_backend": "gemini"}}
        assert resolve_grounding_model_id(pack) == "gemini-2.5-flash"

    def test_all_backends_failed(self) -> None:
        # The synthetic pack built when brave, openai AND gemini all returned nothing.
        pack = {
            "sources": [],
            "grounded": False,
            "markdown": "",
            "debug": {"grounding_backend": "all_failed"},
        }
        assert resolve_grounding_model_id(pack) == GROUNDING_FAILED_MODEL_ID

    def test_brave_circuit_breaker_tripped(self) -> None:
        pack = {
            "sources": [],
            "grounded": False,
            "error": {
                "type": "circuit_breaker_tripped",
                "message": "Brave Search circuit breaker is open — budget likely exhausted",
            },
            "debug": {"grounding_backend": "brave_circuit_breaker_tripped"},
        }
        assert resolve_grounding_model_id(pack) == GROUNDING_BREAKER_MODEL_ID

    def test_missing_api_key(self) -> None:
        # This pack carries no grounding_backend at all — only debug["error"].
        pack = {
            "sources": [],
            "error": {"type": "missing_api_key", "message": "BRAVE_SEARCH_API_KEY not set"},
            "debug": {"error": "missing_api_key"},
        }
        assert resolve_grounding_model_id(pack) == GROUNDING_UNAVAILABLE_MODEL_ID

    def test_no_backend_available(self) -> None:
        pack = {"error": {"type": "no_backend_available", "message": "no backend attempts"}, "debug": {}}
        assert resolve_grounding_model_id(pack) == GROUNDING_UNAVAILABLE_MODEL_ID

    def test_degenerate_packs_never_raise(self) -> None:
        assert resolve_grounding_model_id({}) == GROUNDING_FAILED_MODEL_ID
        assert resolve_grounding_model_id({"debug": None}) == GROUNDING_FAILED_MODEL_ID
        assert resolve_grounding_model_id({"debug": "not-a-dict"}) == GROUNDING_FAILED_MODEL_ID

    def test_sentinels_are_not_priced_as_models(self) -> None:
        # These are not models: a cost-table entry would silently invent spend.
        from forecaster.providers import resolve_price_per_1m

        for sentinel in (
            GROUNDING_FAILED_MODEL_ID,
            GROUNDING_BREAKER_MODEL_ID,
            GROUNDING_UNAVAILABLE_MODEL_ID,
        ):
            assert resolve_price_per_1m(sentinel) is None
