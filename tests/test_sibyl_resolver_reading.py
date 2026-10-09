# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The resolving source's latest reading (Oct 2026, review Part 5).

The live ACLED read against the real response shape (``#country+code`` or a
country name only), a 200 HTML body and a 403 read as unavailable, one retry,
pacing, the Phase 3+ and PA blocks, nothing in backtest or when switched off,
the prompt unchanged when nothing is shown, and the record a run leaves.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("duckdb")

import sibyl.agent as sibyl_agent
import sibyl.config as sibyl_config
import sibyl.run as sibyl_run
from sibyl import resolver_reading as rr
from sibyl.belief_state import empty_plan, initial_belief
from sibyl.select_questions import SibylQuestion
from tests.sibyl_test_utils import HS_RUN_ID, Q1, stub_reference
from tests.test_sibyl_lanes import LOW, _lane_model, run_env  # noqa: F401

TODAY = date(2026, 10, 9)


class _Resp:
    def __init__(self, status=200, payload=None, text=None, ctype="application/json"):
        self.status_code = status
        self._payload = payload
        self.text = text if text is not None else json.dumps(payload)
        self.headers = {"Content-Type": ctype}
        self.url = "https://acleddata.com/api/acled/read?iso=231"
        self.history = []
        self.reason = ""

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


def _events():
    # The live shape: the ISO3 under the HXL tag, or only a country name.
    return [
        {"event_date": "2026-10-02", "event_type": "Battles", "fatalities": "12",
         "#country+code": "ETH", "notes": "a named place"},
        {"event_date": "2026-10-06", "event_type": "Violence against civilians",
         "fatalities": "3", "country": "Ethiopia"},
        {"event_date": "2026-10-05", "event_type": "Battles", "fatalities": "0",
         "country": "Ethiopia"},
        # Another country's event the gateway returned anyway.
        {"event_date": "2026-10-07", "event_type": "Battles", "fatalities": "40",
         "#country+code": "SOM"},
    ]


def _getter(responses, seen=None):
    queue = list(responses)

    def get(url, params, headers, timeout):
        if seen is not None:
            seen.append(dict(params))
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    return get


def _read(responses, **kw):
    sleeps = []
    out = rr.acled_month_to_date(
        "ETH", TODAY, http_get=_getter(responses, kw.get("seen")),
        token_fn=kw.get("token_fn", lambda: "tok"), sleep=sleeps.append,
        clock=lambda: 0.0)
    return out, sleeps


# --- the live ACLED read -----------------------------------------------------------

def test_the_read_aggregates_this_country_only():
    seen = []
    out, _ = _read([_Resp(payload={"data": _events(), "next_cursor": None})], seen=seen)
    assert out["status"] == rr.LIVE_OK
    assert (out["deaths"], out["events"], out["events_other_country"]) == (15, 3, 1)
    assert out["newest_event_date"] == "2026-10-06"
    assert out["by_event_type"]["Battles"] == {"events": 2, "deaths": 12}
    p = seen[0]
    assert p["iso"] == "231" and p["event_date"] == "2026-10-01|2026-10-09"
    assert "iso3" not in p  # the numeric code is ACLED's filter


def test_the_block_carries_aggregates_and_never_an_event():
    out, _ = _read([_Resp(payload={"data": _events()})])
    text = "\n".join(rr._acled_lines("Ethiopia", TODAY, out))
    assert "reported deaths, all event types: 15" in text
    assert "9 of 31 days" in text and "part month" in text
    assert "Battles 2 events, 12 deaths" in text
    assert "a named place" not in text


def test_a_200_html_body_is_unavailable_never_zero():
    html = "<!DOCTYPE html><html><title>Unauthorized</title></html>"
    out, _ = _read([_Resp(text=html, ctype="text/html")])
    assert out["status"] == rr.LIVE_UNAVAILABLE and "deaths" not in out
    text = "\n".join(rr._acled_lines("Ethiopia", TODAY, out))
    assert "could not be read" in text and "do not read its absence as a quiet month" in text


def test_a_403_is_unavailable_and_not_retried():
    seen = []
    out, _ = _read([_Resp(status=403, payload={"message": "denied"})], seen=seen)
    assert out["status"] == rr.LIVE_UNAVAILABLE and len(seen) == 1


def test_one_retry_on_a_5xx_and_requests_are_paced(monkeypatch):
    monkeypatch.setattr(rr, "_LAST_REQUEST_AT", [0.0])
    seen = []
    out, sleeps = _read([_Resp(status=503, payload={}), _Resp(payload={"data": _events()})],
                        seen=seen)
    assert out["status"] == rr.LIVE_OK and len(seen) == 2
    assert sleeps and all(s <= rr.ACLED_MIN_INTERVAL_SEC for s in sleeps)


def test_a_failed_token_is_unavailable_and_scrubbed():
    def boom():
        raise RuntimeError("refused key=sk-ant-api03-" + "a" * 40)

    out, _ = _read([], token_fn=boom)
    assert out["status"] == rr.LIVE_UNAVAILABLE
    assert "a" * 40 not in out["reason"]


# --- the database readings ---------------------------------------------------------

def _facts(con, rows):
    con.execute(
        "CREATE TABLE facts_resolved (ym TEXT, iso3 TEXT, hazard_code TEXT, metric TEXT, "
        "value DOUBLE, value_high DOUBLE, publisher TEXT, publication_date TEXT, "
        "alertlevel TEXT)")
    for r in rows:
        con.execute("INSERT INTO facts_resolved VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", r)


def _q(hazard, metric, iso3="SOM"):
    return SimpleNamespace(question_id=f"{iso3}_{hazard}_{metric}", iso3=iso3,
                           hazard_code=hazard, metric=metric)


def test_the_phase3_block_states_the_lower_bound_and_projections(monkeypatch):
    import duckdb

    monkeypatch.setattr(sibyl_config, "BACKTEST_MODE", False)
    con = duckdb.connect()
    _facts(con, [
        ("2026-06", "SOM", "DR", "phase3plus_in_need", 1_000_000, 2_490_000, "FEWS NET",
         "2026-07-15", None),
        ("2026-11", "SOM", "DR", "phase3plus_projection", 1_500_000, None, "FEWS NET", None, None),
    ])
    keys = ["2026-11", "2026-12", "2027-01", "2027-02", "2027-03", "2027-04"]
    reading = rr.build_resolver_reading(con, _q("DR", "PHASE3PLUS_IN_NEED"), date.today(),
                                        forecast_keys=keys, country_name="Somalia")
    t = reading.text
    assert t.startswith("\n\n=== RESOLVING SOURCE: LATEST READING (as of ")
    assert "2026-06 (FEWS NET, published 2026-07-15): 1,000,000 to 2,490,000" in t
    assert "LOWER bound" in t and "not a measurement" in t and "2026-11: 1,500,000" in t
    assert reading.record["rows"] == 1 and reading.live is None


def test_the_pa_block_states_months_with_no_record_and_gdacs_is_detection(monkeypatch):
    import duckdb

    monkeypatch.setattr(sibyl_config, "BACKTEST_MODE", False)
    con = duckdb.connect()
    today = date.today()
    prev = f"{today.year - (today.month == 1):04d}-{(today.month - 2) % 12 + 1:02d}"
    _facts(con, [(prev, "SOM", "FL", "affected", 42000, None, "IFRC", None, None),
                 (prev, "SOM", "FL", "event_occurrence", 1, None, "GDACS", None, "Orange")])
    reading = rr.build_resolver_reading(con, _q("FL", "PA"), today, country_name="Somalia")
    t = reading.text
    assert f"- {prev}: 42,000 affected (IFRC)" in t
    assert "no record held for this month" in t
    assert f"{prev} Orange" in t and "not the figure this question resolves on" in t


def test_nothing_in_backtest_or_when_switched_off(monkeypatch):
    import duckdb

    con = duckdb.connect()
    q = _q("DR", "PHASE3PLUS_IN_NEED")
    assert rr.build_resolver_reading(con, q, date(2020, 1, 1)).text == ""
    monkeypatch.setattr(sibyl_config, "BACKTEST_MODE", True)
    assert rr.build_resolver_reading(con, q, date.today()).text == ""
    monkeypatch.setattr(sibyl_config, "BACKTEST_MODE", False)
    monkeypatch.setattr(sibyl_config, "RESOLVER_READING", False)
    assert rr.build_resolver_reading(con, q, date.today()).text == ""


def test_conflict_shows_nothing_without_live_lookups(monkeypatch):
    monkeypatch.setattr(sibyl_config, "LIVE_LOOKUPS_ENABLED", False)
    reading = rr.build_resolver_reading(None, _q("ACE", "FATALITIES"), date.today())
    assert reading.text == "" and reading.live is None


def test_a_failing_read_never_raises(monkeypatch):
    class Broken:
        def execute(self, *a, **k):
            raise RuntimeError("boom")

    reading = rr.build_resolver_reading(Broken(), _q("FL", "PA"), date.today())
    assert reading.text == ""


# --- the prompt --------------------------------------------------------------------

def _old_agent():
    path = Path(__file__).resolve().parents[1] / "docs" / "prompts" / "2026-10-09-3" / "sibyl_agent.py"
    spec = importlib.util.spec_from_file_location("sibyl_agent_before_part5", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(spec.name, None)
    mod._CARD_DIR = sibyl_agent._CARD_DIR
    return mod


def _prompt_args():
    q = SibylQuestion(question_id=Q1, hs_run_id=HS_RUN_ID, iso3="ETH", hazard_code="ACE",
                      metric="FATALITIES", target_month="2027-01",
                      window_start_date=date(2026, 11, 1), wording="How many?",
                      volatility_score=0.8, triage_score=0.9)
    ref = stub_reference()
    start = initial_belief(ref.by_month, "FATALITIES")
    start.plan = empty_plan()
    kw = dict(step=2, as_of=TODAY, forecast_months=["2026-11"] * 6,
              transcript_text="=== STEP 1 ===\nx\n", country_name="Ethiopia",
              track_record="t", lessons="", perspective=sibyl_agent.TRIAL_LANES["A"])
    return q, ref, start, kw


def test_no_reading_leaves_the_prompt_byte_identical():
    old = _old_agent()
    q, ref, start, kw = _prompt_args()
    assert sibyl_agent.build_step_prompt(q, ref, start, **kw) == \
        old.build_step_prompt(q, ref, start, **kw)


def test_the_reading_sits_after_the_reference_and_before_the_track_record():
    q, ref, start, kw = _prompt_args()
    block = "\n\n=== RESOLVING SOURCE: LATEST READING (as of 2026-10-09) ===\nx"
    segs = sibyl_agent.build_step_prompt(q, ref, start, resolver_reading=block,
                                         return_segments=True, **kw)
    question_seg = segs[1][0]
    assert block in question_seg
    assert question_seg.index("=== REFERENCE") < question_seg.index("RESOLVING SOURCE") \
        < question_seg.index("=== YOUR TRACK RECORD ===")


# --- a run -------------------------------------------------------------------------

def test_a_run_shows_the_reading_to_every_lane_and_records_it(run_env, monkeypatch):  # noqa: F811
    import resolver.ingestion.acled_auth as acled_auth

    monkeypatch.setattr(sibyl_config, "LIVE_LOOKUPS_ENABLED", True)
    monkeypatch.setattr(rr, "ACLED_MIN_INTERVAL_SEC", 0.0)
    monkeypatch.setattr(acled_auth, "get_access_token", lambda: "tok")
    monkeypatch.setattr(rr, "_default_get", _getter([_Resp(payload={"data": _events()})]))
    prompts = []
    base = _lane_model({"*": LOW}, [])

    def model(prompt):
        prompts.append(prompt)
        return base(prompt)

    monkeypatch.setattr(sibyl_run, "extra_trials_rule",
                        lambda *a, **k: (None, {"max_pairwise_jsd": 0.0}))
    sibyl_run.run_sibyl(HS_RUN_ID, n_questions=1, model_call=model)
    assert prompts and all("RESOLVING SOURCE: LATEST READING" in p for p in prompts)
    assert all("reported deaths, all event types: 15" in p for p in prompts)
    from pythia.db.schema import connect

    con = connect(read_only=False)
    try:
        rec = json.loads(con.execute(
            "SELECT resolver_reading_json FROM sibyl_forecasts WHERE question_id = ?",
            [Q1]).fetchone()[0])
        assert rec["source"] == "acled_live" and rec["deaths"] == 15
        ev = con.execute("SELECT trial_index, step, tool, role FROM sibyl_evidence "
                         "WHERE tool = 'resolver_reading'").fetchall()
        assert ev == [(-1, 0, "resolver_reading", "resolver_reading")]
        ok, failed, nowcast = con.execute(
            "SELECT n_resolver_live_ok, n_resolver_live_failed, share_nowcast_done "
            "FROM sibyl_runs").fetchone()
        assert (ok, failed) == (1, 0) and nowcast is not None
    finally:
        con.close()
