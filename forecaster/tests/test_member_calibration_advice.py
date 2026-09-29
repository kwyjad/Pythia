# Pythia / Copyright (c) 2025 Kevin Wyjad
"""Per-model calibration advice reaches the member it was written for.

``generate_calibration_advice`` has written per-model advice every cycle,
keyed by the model name a member's scores are stored under. Nothing read it:
Track 1 builds one prompt for all members and passes no model name, and the
only caller that passed one was Track 2, whose name (``track2_flash``) is an
aggregate that never has advice. On 2026-09-28 it also wrote advice for the
two external benchmarks, and the prompt loader's last fallback served "any
most-recent row" with no hazard or model filter at all.

Tests are synchronous and wrap async entry points in asyncio.run() —
forecaster-ci installs plain pytest without pytest-asyncio.
"""

from __future__ import annotations

import asyncio

import pytest

duckdb = pytest.importorskip("duckdb")

import forecaster.cli as cli  # type: ignore
import forecaster.prompts as prompts
from forecaster.providers import ModelSpec

_ANSWER = '{"spds": {"month_1": {"probs": [0.1,0.1,0.2,0.2,0.2,0.2]}}}'
_A = ModelSpec(name="model-a", provider="google", model_id="model-a", active=True)
_B = ModelSpec(name="model-b", provider="google", model_id="model-b", active=True)


def _fake_advice(hz, metric, name):
    return "You over-forecast the top bucket." if name == "model-a" else ""


def _call(ms, monkeypatch):
    sent: list[str] = []

    async def fake_call_chat_ms(ms_, prompt, **_kwargs):
        sent.append(prompt)
        return _ANSWER, {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}, None

    monkeypatch.setattr(cli, "call_chat_ms", fake_call_chat_ms)
    monkeypatch.setattr(cli, "load_member_calibration_advice", _fake_advice)
    _text, usage, error, _ = asyncio.run(
        cli._call_spd_model_for_spec(
            ms, "BASE PROMPT", run_id="r1", question_id="Q1", iso3="NGA",
            hazard_code="ACE", metric="FATALITIES",
        )
    )
    assert error is None
    return sent, usage


def test_only_the_advised_member_carries_its_note(monkeypatch) -> None:
    sent_a, usage_a = _call(_A, monkeypatch)
    sent_b, usage_b = _call(_B, monkeypatch)

    assert sent_a[0].startswith("BASE PROMPT")
    assert "You over-forecast the top bucket." in sent_a[0]
    assert "CALIBRATION NOTE FOR THIS MODEL" in sent_a[0]
    # The note is recorded as the true sent prompt, so llm_calls shows it.
    assert usage_a["sent_prompt_text"] == sent_a[0]

    assert sent_b == ["BASE PROMPT"]
    assert "sent_prompt_text" not in usage_b


def test_the_batch_body_carries_the_same_note(monkeypatch) -> None:
    """A batch body and a sync body must carry the same text."""
    enqueued: list[str] = []

    def fake_try_batch_phase(ms, prompt, **_kwargs):
        enqueued.append(prompt)
        return "", {"batched": True}, None, ms

    monkeypatch.setattr(cli, "_try_batch_phase", fake_try_batch_phase)
    monkeypatch.setattr(cli, "load_member_calibration_advice", _fake_advice)
    _text, usage, _err, _ = asyncio.run(
        cli._call_spd_model_for_spec(
            _A, "BASE PROMPT", run_id="r1", question_id="Q1", iso3="NGA",
            hazard_code="ACE", metric="FATALITIES", batch_family="spd_v2",
        )
    )
    assert len(enqueued) == 1
    assert "You over-forecast the top bucket." in enqueued[0]
    assert usage["sent_prompt_text"] == enqueued[0]


def test_a_replayed_note_is_logged_even_when_the_stored_prompt_matches(monkeypatch) -> None:
    """Replay only stashes when the stored prompt DIFFERS from the one handed
    in; with the note appended they are equal, so the wrapper must stash."""

    def fake_try_batch_phase(ms, prompt, **_kwargs):
        return _ANSWER, {"elapsed_ms": 0}, None, ms

    monkeypatch.setattr(cli, "_try_batch_phase", fake_try_batch_phase)
    monkeypatch.setattr(cli, "load_member_calibration_advice", _fake_advice)
    _text, usage, _err, _ = asyncio.run(
        cli._call_spd_model_for_spec(
            _A, "BASE PROMPT", hazard_code="ACE", metric="FATALITIES", batch_family="spd_v2",
        )
    )
    assert "CALIBRATION NOTE FOR THIS MODEL" in usage["sent_prompt_text"]


def _advice_db(tmp_path, monkeypatch):
    db = tmp_path / "advice.duckdb"
    con = duckdb.connect(str(db))
    con.execute(
        "CREATE TABLE calibration_advice (as_of_month TEXT, hazard_code TEXT, metric TEXT, "
        "model_name TEXT, advice TEXT, findings_json TEXT, advice_version TEXT, created_at TIMESTAMP)"
    )
    rows = [
        ("2026-09", "ACE", "FATALITIES", "__shared__", "SHARED ACE ADVICE"),
        ("2026-09", "ACE", "FATALITIES", "model-a", "MODEL A ACE ADVICE"),
        ("2026-09", "ACE", "FATALITIES", "__ext_uniform", "BENCHMARK ADVICE"),
    ]
    for r in rows:
        con.execute(
            "INSERT INTO calibration_advice VALUES (?, ?, ?, ?, ?, NULL, 'v1', now())", list(r)
        )
    con.close()
    monkeypatch.setattr(prompts, "_pythia_db_url_from_config", lambda: f"duckdb:///{db}")
    monkeypatch.delenv("PYTHIA_ADVICE_VERSION", raising=False)
    prompts.reset_member_calibration_advice_cache()


def test_member_loader_returns_only_that_models_row(tmp_path, monkeypatch) -> None:
    _advice_db(tmp_path, monkeypatch)
    assert prompts.load_member_calibration_advice("ACE", "FATALITIES", "model-a") == "MODEL A ACE ADVICE"
    assert prompts.load_member_calibration_advice("ACE", "FATALITIES", "model-b") == ""
    assert prompts.load_member_calibration_advice("FL", "PA", "model-a") == ""
    prompts.reset_member_calibration_advice_cache()


def test_another_hazards_advice_never_fills_an_empty_one(tmp_path, monkeypatch) -> None:
    """The old last-resort fallback returned any row at all, so a flood
    prompt with no advice of its own could be handed model A's ACE advice or
    a benchmark's."""
    _advice_db(tmp_path, monkeypatch)
    text = prompts._load_calibration_advice_for_hazard("FL", "PA")
    assert "ACE" not in text
    assert "BENCHMARK" not in text
    assert text == ""


def test_generator_ranks_forecasters_only_and_counts_questions(tmp_path) -> None:
    from pythia.tools.generate_calibration_advice import _compute_per_model_brier

    con = duckdb.connect(str(tmp_path / "scores.duckdb"))
    con.execute(
        "CREATE TABLE questions (question_id TEXT, hazard_code TEXT, metric TEXT, is_test BOOLEAN)"
    )
    con.execute(
        "CREATE TABLE scores (question_id TEXT, horizon_m INTEGER, score_type TEXT, "
        "model_name TEXT, value DOUBLE)"
    )
    con.execute("INSERT INTO questions VALUES ('Q1','ACE','FATALITIES',FALSE),('Q2','ACE','FATALITIES',FALSE)")
    for qid in ("Q1", "Q2"):
        for h in (1, 2, 3):
            con.execute("INSERT INTO scores VALUES (?, ?, 'brier', 'model-a', 0.3)", [qid, h])
            con.execute("INSERT INTO scores VALUES (?, ?, 'brier', '__ext_uniform', 0.1)", [qid, h])
            con.execute("INSERT INTO scores VALUES (?, ?, 'brier', '__ext_views', 0.05)", [qid, h])
    out = _compute_per_model_brier(con, "ACE", "FATALITIES")
    con.close()

    names = [m["name"] for m in out["all_models"]]
    assert names == ["model-a"]
    # Six score rows over two questions: the threshold reads questions.
    assert out["all_models"][0]["n"] == 2
