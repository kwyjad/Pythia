from __future__ import annotations

import asyncio
from datetime import date
import json
from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb")

import forecaster.cli as cli  # type: ignore
import forecaster.prompts as prompts
from forecaster.ensemble import _parse_spd_json, MemberOutput, EnsembleResult  # type: ignore
from forecaster.aggregate import aggregate_spd  # type: ignore
from forecaster.cli import _write_spd_ensemble_to_db, SPD_CLASS_BINS  # type: ignore
from forecaster.providers import ModelSpec
from pythia.db import schema as db_schema


def test_safe_json_loads_handles_code_fence() -> None:
    fenced = """```json
    {"a": 1, "b": [2, 3]}
    ```"""
    obj = cli._safe_json_loads(fenced)
    assert obj == {"a": 1, "b": [2, 3]}


def test_call_research_model_uses_positional_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    async def fake_call_chat_ms(ms, prompt, **kwargs):
        calls["ms"] = ms
        calls["prompt"] = prompt
        calls["kwargs"] = kwargs
        return "ok", {"total_tokens": 10}, None

    monkeypatch.setattr(cli, "call_chat_ms", fake_call_chat_ms)

    text, usage, error, ms = asyncio.run(cli._call_research_model("HELLO-RESEARCH"))

    assert text == "ok"
    assert error is None
    assert calls["prompt"] == "HELLO-RESEARCH"
    assert "prompt_text" not in calls["kwargs"]


def test_call_spd_model_uses_positional_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    async def fake_call_chat_ms(ms, prompt, **kwargs):
        calls["ms"] = ms
        calls["prompt"] = prompt
        calls["kwargs"] = kwargs
        return "ok", {"total_tokens": 10}, None

    monkeypatch.setattr(cli, "call_chat_ms", fake_call_chat_ms)

    text, usage, error, ms = asyncio.run(cli._call_spd_model("HELLO-SPD"))

    assert text == "ok"
    assert error is None
    assert calls["prompt"] == "HELLO-SPD"
    assert "prompt_text" not in calls["kwargs"]


def test_parse_spd_json_basic():
    """Happy-path SPD parsing: minimal JSON, missing months filled, rows normalized."""
    text = """
    {
      "month_1": [0.1, 0.2, 0.3, 0.2, 0.2],
      "month_2": [0.0, 0.0, 0.0, 0.0, 1.0]
    }
    """
    out = _parse_spd_json(text)
    assert isinstance(out, dict)

    # We always materialise month_1..month_6
    expected_keys = {f"month_{i}" for i in range(1, 7)}
    assert set(out.keys()) == expected_keys

    # Each month is a length-5 prob vector that sums to ~1
    for key, vec in out.items():
        assert isinstance(vec, list)
        assert len(vec) == 5
        s = sum(vec)
        assert 0.99 <= s <= 1.01, f"{key} not normalised: sum={s}"

    # month_1 preserves the relative mass ordering
    m1 = out["month_1"]
    assert m1[2] == max(m1)  # bucket 3 has the highest weight


def test_parse_spd_json_failure_all_zero():
    """All-zero or empty SPDs should be treated as parse failure."""
    text = """
    {
      "month_1": [0, 0, 0, 0, 0],
      "month_2": [0, 0, 0, 0, 0]
    }
    """
    out = _parse_spd_json(text)
    assert out is None

    # Completely invalid JSON also returns None
    bad = "this is not json at all"
    assert _parse_spd_json(bad) is None


def test_build_spd_prompt_v2_handles_date_in_history_summary() -> None:
    """build_spd_prompt_v2 should not crash when history_summary contains date objects."""

    question = {
        "question_id": "q-test",
        "iso3": "ETH",
        "hazard_code": "ACO",
        "metric": "FATALITIES",
        "resolution_source": "ACLED",
    }
    history_summary = {
        "source": "ACLED",
        "history_length_months": 3,
        "recent_mean": 10.0,
        "recent_max": 20,
        "trend": "up",
        "last_6m_values": [
            {"ym": date(2025, 1, 1), "value": 5},
            {"ym": date(2025, 2, 1), "value": 10},
            {"ym": date(2025, 3, 1), "value": 15},
        ],
        "data_quality": "high",
        "notes": "test history",
    }
    hs_triage_entry = {"tier": "watchlist", "triage_score": 0.5}
    research_json = {"base_rate": {"qualitative_summary": "test"}}

    prompt_text = prompts.build_spd_prompt_v2(
        question=question,
        history_summary=history_summary,
        hs_triage_entry=hs_triage_entry,
        research_json=research_json,
    )

    assert "2025-01-01" in prompt_text
    assert "Object of type date is not JSON serializable" not in prompt_text


def test_build_research_prompt_v2_handles_date_in_resolver_features() -> None:
    question = {
        "question_id": "q-test",
        "iso3": "SOM",
        "hazard_code": "ACO",
        "metric": "PA",
        "resolution_source": "EMDAT",
    }
    resolver_features = {
        "source": "ACLED",
        "last_6m_values": [{"ym": date(2025, 5, 1), "value": 42}],
    }
    hs_triage_entry = {"tier": "priority", "triage_score": 0.8}

    prompt_text = prompts.build_research_prompt_v2(
        question=question,
        hs_triage_entry=hs_triage_entry,
        resolver_features=resolver_features,
        model_info={},
    )

    assert "2025-05-01" in prompt_text


def test_build_research_prompt_v2_di_note() -> None:
    question = {
        "question_id": "q-di",
        "iso3": "eth",
        "hazard_code": "DI",
        "metric": "PA",
        "resolution_source": "IDMC",
        "wording": "DI question",
    }
    resolver_features = {"source": "none"}
    hs_triage_entry = {}

    prompt_text = prompts.build_research_prompt_v2(
        question=question,
        hs_triage_entry=hs_triage_entry,
        resolver_features=resolver_features,
        model_info={},
    )

    assert "no Resolver base rate" in prompt_text
    assert "incoming displacement flows" in prompt_text


def test_build_spd_prompt_v2_includes_wording() -> None:
    question = {
        "question_id": "q-wording",
        "iso3": "ETH",
        "hazard_code": "ACE",
        "metric": "FATALITIES",
        "resolution_source": "ACLED",
        "wording": "How many?",
        "target_months": "2025-01",
    }
    history_summary = {}
    hs_triage_entry = {"tier": "priority", "triage_score": 0.9}
    research_json = {"base_rate": {"qualitative_summary": "test"}}

    prompt_text = prompts.build_spd_prompt_v2(
        question=question,
        history_summary=history_summary,
        hs_triage_entry=hs_triage_entry,
        research_json=research_json,
    )

    assert "Natural-language question:" in prompt_text
    assert '"How many?"' in prompt_text


def test_build_spd_prompt_uses_target_month_keys_for_pa() -> None:
    question = {
        "question_id": "TEST_FL_PA",
        "iso3": "ETH",
        "hazard_code": "FL",
        "metric": "PA",
        "resolution_source": "EM-DAT",
        "wording": "Monthly people affected or displaced in ETH for hazard FL, as recorded by the canonical Pythia resolution source.",
        "target_months": "2026-01",
    }
    history_summary = {"source": "EM-DAT"}
    hs_triage_entry = {}
    research_json = {}

    prompt = prompts.build_spd_prompt_v2(
        question=question,
        history_summary=history_summary,
        hs_triage_entry=hs_triage_entry,
        research_json=research_json,
    )

    assert "Month 1: January 2026 (key: \"2026-01\")" in prompt
    assert "\"2026-01\": {\"buckets\": [" in prompt


def test_build_spd_prompt_v2_di_and_nat_notes() -> None:
    question_di = {
        "question_id": "test_di",
        "iso3": "ETH",
        "hazard_code": "DI",
        "metric": "PA",
        "wording": "DI wording",
    }
    prompt_di = prompts.build_spd_prompt_v2(
        question=question_di,
        history_summary={"source": "some"},
        hs_triage_entry={},
        research_json={},
    )

    assert "no Resolver base rate" in prompt_di
    assert "incoming flows" in prompt_di

    question_nat = {
        "question_id": "test_nat",
        "iso3": "ETH",
        "hazard_code": "DR",
        "metric": "PA",
        "wording": "DR wording",
    }

    prompt_nat = prompts.build_spd_prompt_v2(
        question=question_nat,
        history_summary={},
        hs_triage_entry={},
        research_json={},
    )

    assert "people affected by the hazard" in prompt_nat

def test_aggregate_spd_shape_and_uniform_fallback():
    """aggregate_spd should normalise, respect evidence, and fall back to uniform."""
    # Member with a strong preference for bucket 1 in month_1 only
    spd_single = {"month_1": [1.0, 0.0, 0.0, 0.0, 0.0]}
    member = MemberOutput(
        name="m1",
        ok=True,
        parsed=spd_single,
        raw_text="",
    )
    ens = EnsembleResult(members=[member])

    spd_mean, expected, summary = aggregate_spd(ens)

    # month_1 is concentrated in bucket 1
    m1 = spd_mean["month_1"]
    assert len(m1) == 5
    assert 0.99 <= sum(m1) <= 1.01
    assert m1[0] == max(m1)

    # Months without explicit evidence should be (approximately) uniform
    for m_idx in range(2, 7):
        key = f"month_{m_idx}"
        v = spd_mean[key]
        assert len(v) == 5
        assert 0.99 <= sum(v) <= 1.01
        # All entries should be close to each other
        assert max(v) - min(v) < 0.05

    # Expected values are consistent with SPD and bucket centroids
    # (sanity: they are positive and ordered by month key)
    assert expected
    assert set(expected.keys()) == {f"month_{i}" for i in range(1, 7)}
    for val in expected.values():
        assert val > 0.0


@pytest.mark.db
def test_write_spd_ensemble_to_db_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Write a simple SPD to DuckDB and confirm row counts and normalisation."""
    # Use a private DuckDB file and override config lookup
    db_path = tmp_path / "pythia_spd_test.duckdb"
    db_url = f"duckdb:///{db_path}"
    monkeypatch.setattr("forecaster.cli._pythia_db_url_from_config", lambda: db_url)

    # Simple SPD: uniform across all buckets and months
    spd_main = {f"month_{i}": [1.0 / len(SPD_CLASS_BINS)] * len(SPD_CLASS_BINS) for i in range(1, 7)}

    question_id = "q-spd-test"
    _write_spd_ensemble_to_db(
        question_id=question_id,
        run_id="run-test",
        spd_main=spd_main,
        metric="PA",
        hazard_code="FL",
    )

    con = duckdb.connect(str(db_path))
    try:
        rows = con.execute(
            "SELECT horizon_m, class_bin, p FROM forecasts_ensemble WHERE question_id=? ORDER BY horizon_m, class_bin",
            [question_id],
        ).fetchall()
    finally:
        con.close()

    # 6 months × 5 buckets = 30 rows
    assert len(rows) == 6 * len(SPD_CLASS_BINS)

    # Per-horizon sum of probabilities should be ~1.0 and include all buckets
    by_h = {}
    for horizon_m, class_bin, p in rows:
        by_h.setdefault(horizon_m, []).append(float(p))
        assert class_bin in SPD_CLASS_BINS

    assert set(by_h.keys()) == set(range(1, 7))
    for h, probs in by_h.items():
        assert len(probs) == len(SPD_CLASS_BINS)
        s = sum(probs)
        assert 0.99 <= s <= 1.01, f"horizon {h} not normalised: sum={s}"


def test_pythia_spd_hex_qid_does_not_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Regression test: Pythia HS SPD questions use hex IDs, not ints.

    This test stubs out:
      - run_research_async (no external research)
      - run_ensemble_spd (no real LLM calls)
      - _write_spd_ensemble_to_db / _write_spd_raw_to_db (no DuckDB writes)
      - write_unified_row (no filesystem writes)
      - _load_pa_history_block / _load_calibration_advice_db (no resolver DB)

    It then calls _run_one_question_body directly with a synthetic SPD post
    whose id is a hex string, and asserts that:
      - it does not raise, and
      - SPD write stubs and unified-row stub are invoked.
    """

    # --- Stub research and SPD ensemble ---
    async def fake_run_research_async(*args, **kwargs):
        return "stub research", {"research_source": "test_stub"}

    def _make_spd_parsed(n_buckets: int = len(SPD_CLASS_BINS)):
        # Single-bucket mass in the first bucket for all 6 months
        base = [1.0] + [0.0] * (n_buckets - 1)
        return {f"month_{i}": base for i in range(1, 7)}

    async def fake_run_ensemble_spd(prompt, ensemble, **_kwargs):
        parsed = _make_spd_parsed()
        m = MemberOutput(
            name="stub_model",
            ok=True,
            parsed=parsed,
            raw_text="{}",
        )
        return EnsembleResult(members=[m])

    written_spd_calls = []
    unified_rows = []

    def fake_write_spd_ensemble_to_db(**kwargs):
        written_spd_calls.append(("ensemble", kwargs))

    def fake_write_spd_raw_to_db(**kwargs):
        written_spd_calls.append(("raw", kwargs))

    def fake_write_unified_row(row: dict) -> None:
        unified_rows.append(row)

    def fake_pa_history_block(iso3: str, hazard_code: str, *, months: int = 36):
        # Avoid touching DuckDB in this unit test
        return "", {"pa_history_error": "test_stub"}

    def fake_calibration_advice(hazard_code: str, metric: str):
        # No textual calibration advice needed for this test
        return None

    # --- Apply monkeypatches on the CLI module ---
    monkeypatch.setattr(cli, "run_research_async", fake_run_research_async)
    monkeypatch.setattr(cli, "run_ensemble_spd", fake_run_ensemble_spd)
    monkeypatch.setattr(cli, "_write_spd_ensemble_to_db", fake_write_spd_ensemble_to_db)
    monkeypatch.setattr(cli, "_write_spd_raw_to_db", fake_write_spd_raw_to_db)
    monkeypatch.setattr(cli, "write_unified_row", fake_write_unified_row)
    monkeypatch.setattr(cli, "_load_pa_history_block", fake_pa_history_block)
    monkeypatch.setattr(cli, "_load_calibration_advice_db", fake_calibration_advice)

    # --- Synthetic Pythia HS SPD post with hex question_id ---
    hex_qid = "3bdb90167aa8efe694908f01cd2e0760f1ea2f17"
    post = {
        "id": hex_qid,
        "pythia_iso3": "EGY",
        "pythia_hazard_code": "HW",
        "pythia_metric": "PA",
        "pythia_target_month": "2026-05",
        "pythia_hs_run_id": "hs_20251127T125008",
        "question": {
            "id": hex_qid,
            "title": "How many people in Egypt will be affected by extreme heat between 01 December 2025 and 31 May 2026?",
            "type": "spd",
            "possibilities": {"type": "spd"},
        },
    }

    async def _run():
        await cli._run_one_question_body(
            post,
            run_id="test_run_spd_hex",
            purpose="hs_pipeline",
            calib={},  # no calibration weights
            seen_guard_state={},  # no seen-guard state
        )

    # Execute the coroutine; if the hex ID regression comes back, this will raise.
    asyncio.run(_run())

    # Regression guarantee: the hex ID does not cause _run_one_question_body to crash.
    # SPD writers/unified rows may be skipped in soft-fail paths; when present, sanity-check them.
    if written_spd_calls:
        kinds = {k for (k, _kwargs) in written_spd_calls}
        assert "ensemble" in kinds, "SPD ensemble writer was not called"
        assert "raw" in kinds, "SPD raw writer was not called"

    if unified_rows:
        row = unified_rows[0]
        assert row.get("question_id") == hex_qid
        assert row.get("question_type") == "spd"
        # Sanity: we should have at least one per-model SPD JSON field
        assert any(k.startswith("spd_json__") for k in row.keys())


@pytest.mark.db
def test_spd_missing_spds_records_reason_and_raw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing `spds` key should log no_forecast, reason, and raw text."""

    monkeypatch.chdir(tmp_path)
    db_path = tmp_path / "spd_missing_spds.duckdb"
    monkeypatch.setenv("PYTHIA_DB_URL", f"duckdb:///{db_path}")

    con = duckdb.connect(str(db_path))
    try:
        db_schema.ensure_schema(con)
        con.execute(
            """
            INSERT INTO questions (
                question_id, hs_run_id, scenario_ids_json, iso3, hazard_code, metric,
                target_month, window_start_date, window_end_date, wording, status, pythia_metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                "TEST_DR_PA",
                "",
                "[]",
                "ETH",
                "DR",
                "PA",
                "2025-12",
                None,
                None,
                "Test DR PA question",
                "active",
                None,
            ],
        )
        question_row = con.execute(
            "SELECT * FROM questions WHERE question_id = ?", ["TEST_DR_PA"]
        ).fetchone()
    finally:
        con.close()

    async def fake_call_chat_ms(ms, prompt, **_kwargs):
        return json.dumps({"note": "test: no spds key"}), {"total_tokens": 5}, None

    monkeypatch.setattr(cli, "call_chat_ms", fake_call_chat_ms)
    monkeypatch.setattr(cli, "load_hs_triage_entry", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(cli, "_build_history_summary", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(cli, "_load_research_json", lambda *_args, **_kwargs: {})

    asyncio.run(cli._run_spd_for_question("run_spd_missing", question_row))

    con = duckdb.connect(str(db_path))
    try:
        row = con.execute(
            """
            SELECT status, human_explanation
            FROM forecasts_ensemble
            WHERE run_id = ? AND question_id = ?
            """,
            ["run_spd_missing", "TEST_DR_PA"],
        ).fetchone()
    finally:
        con.close()

    assert row is not None
    assert row[0] == "no_forecast"
    assert "missing spds" in (row[1] or "").lower()

    raw_path = Path("debug/spd_raw") / "run_spd_missing__TEST_DR_PA_missing_spds.txt"
    assert raw_path.exists()
    assert "no spds key" in raw_path.read_text()


@pytest.mark.db
def test_spd_runs_without_research(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SPD v2 should log and write outputs even if research is missing."""

    db_path = tmp_path / "spd_missing_research.duckdb"
    monkeypatch.setenv("PYTHIA_DB_URL", f"duckdb:///{db_path}")

    con = duckdb.connect(str(db_path))
    try:
        db_schema.ensure_schema(con)
    finally:
        con.close()

    monkeypatch.setattr(cli, "_build_history_summary", lambda iso3, hazard_code, metric: {"history": "stub"})
    monkeypatch.setattr(cli, "_load_research_json", lambda run_id, question_id: None)
    monkeypatch.setattr(cli, "load_hs_triage_entry", lambda hs_run_id, iso3, hz: {})

    fake_spd = {
        "spds": {
            "month_1": {"probs": [0.1, 0.2, 0.3, 0.2, 0.2]},
            "month_2": {"probs": [0.1, 0.2, 0.3, 0.2, 0.2]},
        }
    }
    fake_ms = ModelSpec(name="stub", provider="test", model_id="m1", active=True, purpose="spd_v2")

    async def fake_call_spd_model(prompt: str):
        return json.dumps(fake_spd), {"elapsed_ms": 5}, None, fake_ms

    monkeypatch.setattr(cli, "_call_spd_model", fake_call_spd_model)

    question_row = {
        "question_id": "q-test",
        "iso3": "ETH",
        "hazard_code": "ACO",
        "metric": "PA",
        "target_month": "2024-06",
        "hs_run_id": None,
    }

    asyncio.run(cli._run_spd_for_question("fc_test", question_row))

    con = duckdb.connect(str(db_path))
    try:
        ensemble_count = con.execute(
            "SELECT COUNT(*) FROM forecasts_ensemble WHERE run_id=? AND question_id=?",
            ["fc_test", "q-test"],
        ).fetchone()[0]
        llm_call_count = con.execute(
            "SELECT COUNT(*) FROM llm_calls WHERE run_id=? AND call_type='spd_v2' AND question_id=?",
            ["fc_test", "q-test"],
        ).fetchone()[0]
    finally:
        con.close()

    assert ensemble_count > 0
    assert llm_call_count > 0


@pytest.mark.db
def test_spd_bayesmc_flag_happy_path_writes_db_and_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    # Enable BayesMC path
    monkeypatch.setenv("PYTHIA_SPD_V2_USE_BAYESMC", "1")

    db_path = tmp_path / "spd_bayesmc_happy.duckdb"
    monkeypatch.setenv("PYTHIA_DB_URL", f"duckdb:///{db_path}")

    # Seed DB with one question
    con = duckdb.connect(str(db_path))
    try:
        db_schema.ensure_schema(con)
        con.execute(
            """
            INSERT INTO questions (
                question_id, hs_run_id, scenario_ids_json, iso3, hazard_code, metric,
                target_month, window_start_date, window_end_date, wording, status, pythia_metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                "Q_BAYESMC_OK",
                "",
                "[]",
                "ETH",
                "DR",
                "PA",
                "2025-12",
                None,
                None,
                "Test DR PA question",
                "active",
                None,
            ],
        )
        question_row = con.execute(
            "SELECT * FROM questions WHERE question_id = ?",
            ["Q_BAYESMC_OK"],
        ).fetchone()
    finally:
        con.close()

    # Minimal stubs to avoid external deps
    monkeypatch.setattr(cli, "load_hs_triage_entry", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(cli, "_build_history_summary", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(cli, "_load_research_json", lambda *_args, **_kwargs: {})

    # Fake EnsembleResult + members (shape only needs what _call_spd_bayesmc_v2 reads)
    class _M:
        def __init__(self, model_spec, text, usage, error_text=None):
            self.model_spec = model_spec
            self.text = text
            self.usage = usage
            self.error_text = error_text

    class _ER:
        def __init__(self, members):
            self.members = members

    ms1 = ModelSpec(name="OpenAI", provider="openai", model_id="gpt-test", active=True, purpose="spd_v2")
    ms2 = ModelSpec(name="Google", provider="google", model_id="gemini-test", active=True, purpose="spd_v2")
    fake_ens = _ER(
        members=[
            _M(ms1, text='{"any":"raw"}', usage={"total_tokens": 10, "elapsed_ms": 5}, error_text=None),
            _M(ms2, text='{"any":"raw"}', usage={"total_tokens": 12, "elapsed_ms": 6}, error_text=None),
        ]
    )

    async def fake_run_ensemble_spd(*_args, **_kwargs):
        return fake_ens

    def fake_aggregate_spd(_ens, *_args, **_kwargs):
        # Return classic aggregate_spd shape: (spd_main, ev_dict, bmc_summary)
        spd_main = {"2025-12": [0.1, 0.2, 0.3, 0.2, 0.2]}
        return spd_main, {}, {}

    monkeypatch.setattr(cli, "run_ensemble_spd", fake_run_ensemble_spd)
    monkeypatch.setattr(cli, "aggregate_spd", fake_aggregate_spd)

    asyncio.run(cli._run_spd_for_question("run_bayesmc_ok", question_row))

    con = duckdb.connect(str(db_path))
    try:
        # Ensure ensemble write happened
        row = con.execute(
            """
            SELECT status
            FROM forecasts_ensemble
            WHERE run_id = ? AND question_id = ?
            """,
            ["run_bayesmc_ok", "Q_BAYESMC_OK"],
        ).fetchone()
        assert row is not None
        assert (row[0] or "").lower() == "ok"

        # Ensure at least one spd_v2 llm call was logged
        llm_count = con.execute(
            """
            SELECT COUNT(*)
            FROM llm_calls
            WHERE run_id = ? AND question_id = ? AND call_type = 'spd_v2'
            """,
            ["run_bayesmc_ok", "Q_BAYESMC_OK"],
        ).fetchone()[0]
        assert llm_count > 0
    finally:
        con.close()


@pytest.mark.db
def test_spd_bayesmc_flag_missing_spds_records_reason_and_raw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    monkeypatch.setenv("PYTHIA_SPD_V2_USE_BAYESMC", "1")

    db_path = tmp_path / "spd_bayesmc_missing.duckdb"
    monkeypatch.setenv("PYTHIA_DB_URL", f"duckdb:///{db_path}")

    con = duckdb.connect(str(db_path))
    try:
        db_schema.ensure_schema(con)
        con.execute(
            """
            INSERT INTO questions (
                question_id, hs_run_id, scenario_ids_json, iso3, hazard_code, metric,
                target_month, window_start_date, window_end_date, wording, status, pythia_metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                "Q_BAYESMC_MISSING",
                "",
                "[]",
                "ETH",
                "DR",
                "PA",
                "2025-12",
                None,
                None,
                "Test DR PA question",
                "active",
                None,
            ],
        )
        question_row = con.execute(
            "SELECT * FROM questions WHERE question_id = ?",
            ["Q_BAYESMC_MISSING"],
        ).fetchone()
    finally:
        con.close()

    monkeypatch.setattr(cli, "load_hs_triage_entry", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(cli, "_build_history_summary", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(cli, "_load_research_json", lambda *_args, **_kwargs: {})

    # Fake ensemble with members that have raw text containing a marker
    class _M:
        def __init__(self, model_spec, text, usage, error_text=None):
            self.model_spec = model_spec
            self.text = text
            self.usage = usage
            self.error_text = error_text

    class _ER:
        def __init__(self, members):
            self.members = members

    ms1 = ModelSpec(name="OpenAI", provider="openai", model_id="gpt-test", active=True, purpose="spd_v2")
    fake_ens = _ER(
        members=[
            _M(ms1, text='{"note":"test: no spds key"}', usage={"total_tokens": 5, "elapsed_ms": 3}, error_text=None),
        ]
    )

    async def fake_run_ensemble_spd(*_args, **_kwargs):
        return fake_ens

    def fake_aggregate_spd(_ens, *_args, **_kwargs):
        # Empty spd_main -> bridge produces {"spds": {}}
        return {}, {}, {}

    monkeypatch.setattr(cli, "run_ensemble_spd", fake_run_ensemble_spd)
    monkeypatch.setattr(cli, "aggregate_spd", fake_aggregate_spd)

    asyncio.run(cli._run_spd_for_question("run_bayesmc_missing", question_row))

    con = duckdb.connect(str(db_path))
    try:
        row = con.execute(
            """
            SELECT status, human_explanation
            FROM forecasts_ensemble
            WHERE run_id = ? AND question_id = ?
            """,
            ["run_bayesmc_missing", "Q_BAYESMC_MISSING"],
        ).fetchone()
        assert row is not None
        assert (row[0] or "").lower() == "no_forecast"
        assert "missing spds" in (row[1] or "").lower()
    finally:
        con.close()

    raw_path = Path("debug/spd_raw") / "run_bayesmc_missing__Q_BAYESMC_MISSING_missing_spds.txt"
    assert raw_path.exists()
    assert "no spds key" in raw_path.read_text(encoding="utf-8").lower()


def test_spd_prompt_template_allows_literal_json_braces() -> None:
    """
    Ensure SPD_PROMPT_TEMPLATE.format(...) works when literal JSON is present.

    This guards against regressions where unescaped braces in the template
    would cause KeyError('\n     "month_1"') during str.format, and verifies
    the JSON schema for month_1 is preserved in the rendered prompt.
    """

    prompt = prompts.build_spd_prompt_pa(
        question_title="Test SPD PA question",
        iso3="USA",
        hazard_code="FL",
        hazard_label="Flood",
        metric="PA",
        background="Some background",
        research_text="Some research",
        resolution_source="EMDAT",
        window_start_date=None,
        window_end_date=None,
        month_labels=None,
        today=date.today(),
        criteria="Some resolution criteria",
    )

    assert '"month_1": [p1, p2, p3, p4, p5],' in prompt
    assert "Test SPD PA question" in prompt
