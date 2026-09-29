# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""What a binary drought question is shown, and what is kept of its members.

Five Track-1 drought event questions (ETH, SLV, SSD, HND, SOM) resolved "yes"
in August 2026 against ensemble probabilities of 1.7% to 15%. Their prompts
said "ENSO: Current state: Neutral" during a strong El Niño, and the GDACS
history block read "0 of 9 months" for a window of 2026-05 to 2026-07: nine
ROWS over three months of the country's own span, printed as a base rate.
The members' individual forecasts were pooled and discarded, so nobody could
say which of them had been right.
"""

from __future__ import annotations

import asyncio
import json
from datetime import date
from pathlib import Path
from unittest.mock import patch

import duckdb
import pytest

from forecaster import binary_prompts as bp
from forecaster import cli
from forecaster.gdacs_history import gdacs_calendar_series

_WINDOW = [f"2026-{m:02d}" for m in range(10, 13)] + [f"2027-{m:02d}" for m in range(1, 4)]


def _question(**extra):
    q = {
        "question_id": "SLV_DR_EVENT_OCCURRENCE_2026-10",
        "iso3": "SLV",
        "hazard_code": "DR",
        "metric": "EVENT_OCCURRENCE",
        "country_name": "El Salvador",
        "window_start_date": "2026-10-01",
        "wording": "test",
    }
    q.update(extra)
    return q


def _facts(con) -> None:
    con.execute(
        "CREATE TABLE facts_resolved (ym TEXT, iso3 TEXT, hazard_code TEXT, metric TEXT, "
        "value DOUBLE, alertlevel TEXT, series_semantics TEXT)"
    )


def _month_seq(start: str, end: str) -> list[str]:
    y, m = int(start[:4]), int(start[5:])
    out = []
    while f"{y:04d}-{m:02d}" <= end:
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return out


# ---------------------------------------------------------------------------
# (b) the GDACS history is counted in calendar months of the source's window
# ---------------------------------------------------------------------------


def _slv_db(global_start: str = "2016-01", *, slv_orange: bool = True):
    con = duckdb.connect(":memory:")
    _facts(con)
    # Another country keeps GDACS drought coverage running from global_start.
    for ym in _month_seq(global_start, "2026-08"):
        con.execute(
            "INSERT INTO facts_resolved VALUES (?, 'ETH', 'DR', 'event_occurrence', 1, 'Orange', 'new')",
            [ym],
        )
    # El Salvador: three Green months, each carried by three rows — the
    # shape that printed "2026-05 to 2026-07 ... 0 of 9 months".
    for ym in ("2026-05", "2026-06", "2026-07"):
        for sem in ("new", "stock", ""):
            con.execute(
                "INSERT INTO facts_resolved VALUES (?, 'SLV', 'DR', 'event_occurrence', 0, 'Green', ?)",
                [ym, sem],
            )
    # And one Orange month for SLV, twice over.
    for sem in (("new", "stock") if slv_orange else ()):
        con.execute(
            "INSERT INTO facts_resolved VALUES ('2023-08', 'SLV', 'DR', 'event_occurrence', 1, 'Orange', ?)",
            [sem],
        )
    return con


def test_denominator_is_calendar_months_of_the_source_window() -> None:
    con = _slv_db()
    series = gdacs_calendar_series(con, "SLV", "DR", today=date(2026, 9, 29))
    assert series["window_start"] == "2016-01"
    assert series["window_end"] == "2026-08"
    assert series["total_months"] == len(_month_seq("2016-01", "2026-08"))
    assert series["event_months"] == 1  # the duplicated Orange month counts once
    assert series["country_rows"] == 11
    assert series["history_available"] is True


def test_base_rate_section_no_longer_prints_rows_as_months() -> None:
    con = _slv_db()
    base = bp._query_base_rate(con, "SLV", "DR", today=date(2026, 9, 29))
    text = bp._section_base_rate("El Salvador", "drought", base)
    n = len(_month_seq("2016-01", "2026-08"))
    assert f"in 1 of the {n} calendar months" in text
    assert "2016-01 to 2026-08" in text
    assert "0 of 9" not in text


def test_a_thin_source_window_says_history_unavailable() -> None:
    con = _slv_db(global_start="2026-05", slv_orange=False)
    base = bp._query_base_rate(con, "SLV", "DR", today=date(2026, 9, 29))
    assert base["history_available"] is False
    text = bp._section_base_rate("El Salvador", "drought", base)
    assert "History unavailable" in text
    assert "Do NOT read this as a 0% base rate" in text
    assert "0.0%" not in text


def test_future_rows_and_the_current_month_are_outside_the_window() -> None:
    con = _slv_db()
    con.execute(
        "INSERT INTO facts_resolved VALUES ('2026-09', 'ETH', 'DR', 'event_occurrence', 1, 'Red', 'new')"
    )
    con.execute(
        "INSERT INTO facts_resolved VALUES ('2027-02', 'ETH', 'DR', 'event_occurrence', 1, 'Red', 'new')"
    )
    series = gdacs_calendar_series(con, "ETH", "DR", today=date(2026, 9, 29))
    assert series["window_end"] == "2026-08"


def test_event_history_block_uses_the_same_window(monkeypatch) -> None:
    from forecaster import history_loaders
    from forecaster.prompts import _format_gdacs_event_history_for_prompt

    con = _slv_db()

    class _NoClose:
        def __init__(self, inner):
            self._inner = inner

        def execute(self, *a, **k):
            return self._inner.execute(*a, **k)

        def close(self):
            pass

    monkeypatch.setattr(history_loaders, "connect", lambda read_only=True: _NoClose(con))
    hist = history_loaders._build_gdacs_event_history("SLV", "DR")
    assert hist["event_months"] == 1
    assert hist["data_range"].startswith("2016-01 to ")
    # August 2023 was an event month; every August in the window observed.
    aug = hist["seasonal"][8]
    assert aug["years_with_event"] == 1 and aug["years_observed"] >= 10
    block = _format_gdacs_event_history_for_prompt(hist, [10, 11, 12, 1, 2, 3])
    assert "calendar months" in block

    thin = _slv_db(global_start="2026-05", slv_orange=False)
    monkeypatch.setattr(history_loaders, "connect", lambda read_only=True: _NoClose(thin))
    hist = history_loaders._build_gdacs_event_history("SLV", "DR")
    block = _format_gdacs_event_history_for_prompt(hist, [10, 11, 12, 1, 2, 3])
    assert "History unavailable" in block
    assert "Overall event rate" not in block


# ---------------------------------------------------------------------------
# (a) the September ENSO fix reaches the binary prompt, dated
# ---------------------------------------------------------------------------


def test_binary_prompt_carries_el_nino_from_the_db_with_its_observation_date(
    tmp_path: Path, monkeypatch
) -> None:
    from horizon_scanner.enso import enso_module
    from pythia.db.schema import ensure_schema

    db = tmp_path / "enso.duckdb"
    monkeypatch.setenv("PYTHIA_DB_URL", f"duckdb:///{db}")
    con = duckdb.connect(str(db))
    ensure_schema(con)
    today = date.today()
    observed = date(today.year, today.month, 1)
    con.execute(
        """
        INSERT INTO enso_state (fetch_date, enso_phase, nino34_anomaly, oni, enso_strength,
                                oni_basis, observation_date, status, row_kind, age_days,
                                nino34_source, raw_context)
        VALUES (?, 'El Niño', 1.8, 1.8, 'strong', 'oni_table', ?, 'fresh', 'live', 0,
                'cpc_oni_ascii', '')
        """,
        [today, observed],
    )
    # A historical row must never answer for the present.
    con.execute(
        "INSERT INTO enso_state (fetch_date, enso_phase, oni, observation_date, status, row_kind) "
        "VALUES (DATE '1950-01-01', 'La Niña', -1.5, DATE '1950-01-01', 'historical', 'historical')"
    )
    con.close()

    enso_text = enso_module.get_enso_prompt_context()
    assert "El Ni" in enso_text and "Neutral" not in enso_text

    prompt = bp.build_binary_event_prompt(
        question=_question(),
        base_rate={},
        current_alerts=[],
        structured_data={"enso_context": enso_text},
        today=today.isoformat(),
    )
    assert f"ENSO STATE (index observed {observed.isoformat()}):" in prompt
    assert "El Ni" in prompt


def test_an_undated_enso_block_says_it_may_be_stale() -> None:
    header = bp._enso_header("## ENSO State and Forecast\nCurrent state: Neutral.")
    assert "observation date not stated" in header


# ---------------------------------------------------------------------------
# (c) a regime-change flag must be reconciled in the answer
# ---------------------------------------------------------------------------


def test_rc_flag_requires_a_reconciliation_field() -> None:
    flagged = bp.build_binary_event_prompt(
        question=_question(), base_rate={}, current_alerts=[], structured_data={},
        today="2026-09-29", rc_level=2,
    )
    quiet = bp.build_binary_event_prompt(
        question=_question(), base_rate={}, current_alerts=[], structured_data={},
        today="2026-09-29", rc_level=0,
    )
    assert "REGIME CHANGE FLAG" in flagged and '"rc_reconciliation"' in flagged
    assert "REGIME CHANGE FLAG" not in quiet


def test_rc_reconciliation_is_parsed_and_the_months_still_are() -> None:
    raw = "```json\n" + json.dumps({
        "rc_reconciliation": "The flag rests on a failed Primera; I raised the prior.",
        "months": {m: {"posterior": 0.3} for m in _WINDOW},
    }) + "\n```"
    assert bp.parse_rc_reconciliation(raw).startswith("The flag rests")
    assert bp.parse_binary_response(raw, expected_months=_WINDOW) == {m: 0.3 for m in _WINDOW}
    assert bp.parse_rc_reconciliation('{"months": {}}') is None


# ---------------------------------------------------------------------------
# (d) members are stored beside the pooled rows, and scored
# ---------------------------------------------------------------------------


class _Spec:
    def __init__(self, name):
        self.name = name


_QROW = {
    "question_id": "SLV_DR_EVENT_OCCURRENCE_2026-10",
    "iso3": "SLV",
    "hazard_code": "DR",
    "metric": "EVENT_OCCURRENCE",
    "wording": "test",
    "window_start_date": "2026-10-01",
    "target_month": "2027-03",
    "hs_run_id": "hs_test",
}


def _run(track: int, hs_entry: dict):
    writes: list[dict] = []
    texts = [
        json.dumps({"rc_reconciliation": f"note {i}",
                    "months": {m: {"posterior": 0.1 * (i + 1)} for m in _WINDOW}})
        for i in range(2)
    ]

    async def fake_members(prompt, specs, **kwargs):
        calls = [
            {"text": t, "usage": {"cost_usd": 0.01}, "error": None,
             "model_spec": _Spec(f"model-{i}")}
            for i, t in enumerate(texts)
        ]
        return [], {}, calls, {}

    async def fake_log(**kwargs):
        return None

    def fake_write(run_id, question_row, month_probs, **kwargs):
        writes.append({"months": dict(month_probs), **kwargs})

    prompts: list[dict] = []

    def fake_prompt(**kwargs):
        prompts.append(kwargs)
        return "PROMPT"

    with patch.object(cli, "_call_spd_members_v2_compat", fake_members), \
         patch.object(cli, "log_forecaster_llm_call", fake_log), \
         patch.object(cli, "_write_binary_outputs", fake_write), \
         patch.object(cli, "_record_no_forecast", lambda *a, **k: None), \
         patch.object(cli, "_load_structured_data", lambda *a, **k: {}), \
         patch.object(cli, "load_hs_triage_entry", lambda *a, **k: hs_entry), \
         patch.object(cli, "build_binary_base_rate", lambda *a, **k: {}), \
         patch.object(cli, "build_binary_event_prompt", fake_prompt), \
         patch.object(cli, "connect", side_effect=RuntimeError("no db in test")), \
         patch.object(cli, "_select_spd_specs_for_run", lambda: ([_Spec("model-0"), _Spec("model-1")], [])):
        asyncio.run(cli._run_binary_forecast_for_question("run_test", _QROW, track=track))
    return writes, prompts


def test_track1_members_are_written_to_raw_only_with_their_reconciliation() -> None:
    writes, prompts = _run(1, {"regime_change_level": 2})
    assert prompts[0]["rc_level"] == 2
    by_name = {w["model_name"]: w for w in writes}
    assert set(by_name) == {"ensemble_mean_v2", "ensemble_bayesmc_v2", "model-0", "model-1"}
    for member in ("model-0", "model-1"):
        assert by_name[member]["write_ensemble"] is False
        assert by_name[member]["extra_json"]["rc_level"] == 2
        assert by_name[member]["extra_json"]["rc_reconciliation"].startswith("note")
    assert by_name["model-1"]["months"]["2026-10"] == pytest.approx(0.2)
    assert by_name["ensemble_mean_v2"]["months"]["2026-10"] == pytest.approx(0.15)
    assert by_name["ensemble_mean_v2"].get("write_ensemble", True) is True


def test_track2_writes_no_separate_member_row() -> None:
    writes, _ = _run(2, {"regime_change": {"level": 0}})
    assert {w["model_name"] for w in writes} == {"track2_flash"}


def test_member_rows_land_in_raw_and_are_scored_as_binary(tmp_path: Path, monkeypatch) -> None:
    """The real writer puts a member in forecasts_raw only, and compute_scores
    scores it with the binary Brier — its own family, never the SPD one."""

    from pythia.db.schema import ensure_schema

    db = tmp_path / "w.duckdb"
    monkeypatch.setenv("PYTHIA_DB_URL", f"duckdb:///{db}")
    con = duckdb.connect(str(db))
    ensure_schema(con)
    con.close()

    qrow = dict(_QROW)
    cli._write_binary_outputs(
        "fc_1", qrow, {m: 0.3 for m in _WINDOW},
        resolution_source="GDACS", usage={}, model_name="model-0",
        write_ensemble=False, extra_json={"member": True, "rc_reconciliation": "x"},
    )
    cli._write_binary_outputs(
        "fc_1", qrow, {m: 0.2 for m in _WINDOW},
        resolution_source="GDACS", usage={}, model_name="ensemble_mean_v2",
    )
    con = duckdb.connect(str(db))
    raw = con.execute(
        "SELECT model_name, COUNT(DISTINCT month_index) FROM forecasts_raw "
        "WHERE question_id = ? GROUP BY 1 ORDER BY 1", [qrow["question_id"]]
    ).fetchall()
    ens = con.execute(
        "SELECT DISTINCT model_name FROM forecasts_ensemble WHERE question_id = ?",
        [qrow["question_id"]],
    ).fetchall()
    note = con.execute(
        "SELECT spd_json FROM forecasts_raw WHERE model_name = 'model-0' LIMIT 1"
    ).fetchone()[0]
    con.close()
    assert raw == [("ensemble_mean_v2", 6), ("model-0", 6)]
    assert ens == [("ensemble_mean_v2",)]
    assert json.loads(note)["rc_reconciliation"] == "x"

    # compute_scores scores the member with the binary Brier and nothing else.
    from pythia.tools.compute_scores import compute_scores

    con = duckdb.connect(str(db))
    con.execute("INSERT INTO hs_runs (hs_run_id) VALUES ('hs_test')")
    con.execute(
        "INSERT INTO questions (question_id, hs_run_id, iso3, hazard_code, metric, "
        "target_month, window_start_date, status) VALUES (?, 'hs_test', 'SLV', 'DR', "
        "'EVENT_OCCURRENCE', '2027-03', DATE '2026-10-01', 'active')",
        [qrow["question_id"]],
    )
    con.execute(
        "CREATE TABLE IF NOT EXISTS resolutions (question_id TEXT, horizon_m INTEGER, "
        "observed_month TEXT, value DOUBLE, source_desc TEXT, is_test BOOLEAN DEFAULT FALSE)"
    )
    con.execute(
        "INSERT INTO resolutions (question_id, horizon_m, observed_month, value) "
        "VALUES (?, 1, '2026-10', 1.0)",
        [qrow["question_id"]],
    )
    con.close()
    compute_scores(f"duckdb:///{db}")
    con = duckdb.connect(str(db))
    scores = con.execute(
        "SELECT model_name, score_type, value FROM scores WHERE question_id = ? ORDER BY 1, 2",
        [qrow["question_id"]],
    ).fetchall()
    con.close()
    assert [(m, t) for m, t, _v in scores] == [
        ("ensemble_mean_v2", "brier"), ("model-0", "brier"),
    ]
    by = {m: v for m, _t, v in scores}
    assert by["model-0"] == pytest.approx(0.49)
    assert by["ensemble_mean_v2"] == pytest.approx(0.64)
