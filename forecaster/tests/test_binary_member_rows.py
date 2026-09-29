# Pythia / Copyright (c) 2025 Kevin Wyjad
"""Binary members are stored, scored and weighted like SPD members.

Until Sept 2026 ``_run_binary_forecast_for_question`` parsed every member's
answer and then wrote only the pooled rows. ``compute_calibration_pythia``
therefore reported "No member-model Brier samples" for every
EVENT_OCCURRENCE group on 2026-09-28: a binary member could never be scored,
so binary questions could never be calibrated.
"""

from __future__ import annotations

import asyncio
import json
from datetime import date
from unittest.mock import patch

import pytest

duckdb = pytest.importorskip("duckdb")

import forecaster.cli as cli
from forecaster.providers import ModelSpec
from pythia.db import schema as db_schema

_QROW = {
    "question_id": "TST_FL_EVENT_OCCURRENCE_2026-04",
    "iso3": "TST",
    "hazard_code": "FL",
    "metric": "EVENT_OCCURRENCE",
    "wording": "test question",
    "window_start_date": "2026-04-01",
    "target_month": "2026-09",
    "hs_run_id": "hs_test",
}
_WINDOW = [f"2026-{m:02d}" for m in range(4, 10)]
_SPECS = [
    ModelSpec(name="model-a", provider="openai", model_id="model-a", active=True),
    ModelSpec(name="model-b", provider="google", model_id="model-b", active=True),
    ModelSpec(name="model-c", provider="anthropic", model_id="model-c", active=True),
]


def _answer(p: float) -> str:
    return json.dumps({"months": {m: {"posterior": p} for m in _WINDOW}})


def _run(answers: list[str], *, track: int = 1, weights: dict | None = None):
    writes: list[dict] = []

    async def fake_members(prompt, specs, **kwargs):
        calls = [
            {"text": t, "usage": {"cost_usd": 0.01}, "error": None, "model_spec": ms}
            for ms, t in zip(specs, answers)
        ]
        return [], {}, calls, {}

    async def fake_log(**kwargs):
        return None

    def fake_write(run_id, question_row, month_probs, *, resolution_source, usage,
                   model_name="ensemble", raw_only=False):
        writes.append({"model_name": model_name, "raw_only": raw_only, "months": dict(month_probs)})

    specs = _SPECS if track == 1 else [cli.TRACK2_MODEL_SPEC]
    cli._CALIB_WEIGHTS_CACHE.clear()
    with patch.object(cli, "_call_spd_members_v2_compat", fake_members), \
         patch.object(cli, "log_forecaster_llm_call", fake_log), \
         patch.object(cli, "_write_binary_outputs", fake_write), \
         patch.object(cli, "_record_no_forecast", lambda *a, **k: None), \
         patch.object(cli, "_load_structured_data", lambda *a, **k: {}), \
         patch.object(cli, "load_hs_triage_entry", lambda *a, **k: {}), \
         patch.object(cli, "build_binary_base_rate", lambda *a, **k: {}), \
         patch.object(cli, "build_binary_event_prompt", lambda **k: "PROMPT"), \
         patch.object(cli, "connect", side_effect=RuntimeError("no db in test")), \
         patch.object(cli, "_load_calibration_weights_db", lambda hz, mt: weights), \
         patch.object(cli, "_select_spd_specs_for_run", lambda: (_SPECS, [])):
        asyncio.run(cli._run_binary_forecast_for_question("run_test", _QROW, track=track))
    cli._CALIB_WEIGHTS_CACHE.clear()
    return writes


def test_track1_writes_each_member_to_raw_only_beside_the_pooled_rows() -> None:
    writes = _run([_answer(0.1), _answer(0.2), _answer(0.6)])
    members = {w["model_name"]: w for w in writes if w["raw_only"]}
    pooled = {w["model_name"] for w in writes if not w["raw_only"]}
    assert set(members) == {"model-a", "model-b", "model-c"}
    assert pooled == {"ensemble_mean_v2", "ensemble_bayesmc_v2"}
    assert members["model-c"]["months"] == {m: pytest.approx(0.6) for m in _WINDOW}


def test_track2_adds_no_member_row() -> None:
    writes = _run([_answer(0.3)], track=2)
    assert [w["raw_only"] for w in writes] == [False]
    assert writes[0]["model_name"] == "track2_flash"


def test_the_pooled_mean_is_plain_without_weights() -> None:
    writes = _run([_answer(0.1), _answer(0.2), _answer(0.6)])
    mean = next(w for w in writes if w["model_name"] == "ensemble_mean_v2")
    assert mean["months"]["2026-04"] == pytest.approx(0.3)


def test_the_pooled_mean_uses_calibration_weights() -> None:
    # model-a weighted three times model-b; model-c has no record and takes
    # the calibrated average (neutral).
    writes = _run(
        [_answer(0.1), _answer(0.2), _answer(0.6)],
        weights={"model-a": 0.75, "model-b": 0.25},
    )
    mean = next(w for w in writes if w["model_name"] == "ensemble_mean_v2")
    # Matched rescaled to mean 1: a=1.5, b=0.5; c=1.0. (1.5*.1+.5*.2+1*.6)/3
    assert mean["months"]["2026-04"] == pytest.approx((0.15 + 0.1 + 0.6) / 3.0)


def test_raw_only_writes_nothing_to_forecasts_ensemble(tmp_path) -> None:
    db_path = str(tmp_path / "binary_members.duckdb")
    con = duckdb.connect(db_path)
    db_schema.ensure_schema(con)
    con.close()

    row = dict(_QROW, window_start_date=date(2026, 4, 1))
    with patch("forecaster.cli.connect") as mock_connect:
        mock_connect.return_value = duckdb.connect(db_path)
        cli._write_binary_outputs(
            "run-x", row, {m: 0.25 for m in _WINDOW},
            resolution_source="GDACS", usage={}, model_name="model-a", raw_only=True,
        )

    con = duckdb.connect(db_path)
    try:
        raw = con.execute(
            "SELECT COUNT(*) FROM forecasts_raw WHERE model_name = 'model-a' AND bucket_index = 1"
        ).fetchone()[0]
        ens = con.execute("SELECT COUNT(*) FROM forecasts_ensemble").fetchone()[0]
    finally:
        con.close()
    assert raw == len(_WINDOW)
    assert ens == 0
