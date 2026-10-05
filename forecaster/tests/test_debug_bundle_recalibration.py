# Pythia / Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
"""Family recalibration, as the debug bundle reports it (Oct 2026).

The 1 October 2026 run wrote no ``__recal`` copy at all, because the
factors were first fitted the next day, and nothing said so. The bundle now
states per hazard and metric whether recalibration was applied, shadowed or
absent, and why. The November case is pinned as well: factors fitted under
the old ACE/FATALITIES prompt meet the prior-anchor prompt and are shadowed,
in both RC split-test arms.
"""

from __future__ import annotations

import json

import pytest

duckdb = pytest.importorskip("duckdb")

from scripts import dump_pythia_debug_bundle as bundle


def _db(rows, fitted=()):
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE questions (question_id TEXT PRIMARY KEY, hazard_code TEXT, metric TEXT)")
    con.execute("CREATE TABLE forecasts_raw (run_id TEXT, question_id TEXT, model_name TEXT, recalibration_json TEXT)")
    con.execute("CREATE TABLE family_recalibration (hazard_code TEXT, metric TEXT)")
    for qid, hz, m, model, meta in rows:
        con.execute("INSERT OR IGNORE INTO questions VALUES (?, ?, ?)", [qid, hz, m])
        con.execute("INSERT INTO forecasts_raw VALUES ('r1', ?, ?, ?)",
                    [qid, model, json.dumps(meta) if meta else None])
    for hz, m in fitted:
        con.execute("INSERT INTO family_recalibration VALUES (?, ?)", [hz, m])
    return con


def test_states_and_reasons_per_group():
    con = _db([
        ("q1", "ACE", "FATALITIES", "gpt-6-sol", {"mode": "auto_shadow", "reason": "factors fitted under block=None rc=None"}),
        ("q1", "ACE", "FATALITIES", "gpt-6-sol__recal", None),
        ("q2", "FL", "PA", "gpt-6-sol", {"mode": "none", "reason": "no factors for this family and group"}),
        ("q3", "DR", "PHASE3PLUS_IN_NEED", "gpt-6-sol", {"mode": "apply"}),
        ("q3", "DR", "PHASE3PLUS_IN_NEED", "gpt-6-sol__raw", None),
    ], fitted=[("ACE", "FATALITIES"), ("DR", "PHASE3PLUS_IN_NEED")])
    got = {(g["hazard_code"], g["metric"]): g for g in bundle._load_recalibration_health(con, "r1")}
    assert got[("ACE", "FATALITIES")]["state"] == "shadowed"
    assert "block=None" in got[("ACE", "FATALITIES")]["why"]
    assert got[("FL", "PA")]["state"] == "absent"
    assert got[("FL", "PA")]["why"] == "no factors for this family and group"
    assert got[("DR", "PHASE3PLUS_IN_NEED")]["state"] == "applied"


def test_the_health_table_carries_the_line():
    data = bundle.BundleData()
    data.recalibration_health = [{"hazard_code": "ACE", "metric": "FATALITIES", "state": "shadowed",
                                  "why": "factors fitted under block=None rc=None"}]
    rows = {c["subsystem"]: c for c in bundle._evaluate_pipeline_health(data)}
    assert "ACE/FATALITIES: shadowed" in rows["Family Recalibration"]["detail"]


def test_no_run_id_reports_nothing():
    assert bundle._load_recalibration_health(_db([]), None) == []


@pytest.mark.parametrize("rc_guidance", [None, "shift_v1"])
def test_november_shadows_ace_fatalities_in_both_rc_arms(monkeypatch, rc_guidance):
    from pythia.tools import family_recalibration as fr

    monkeypatch.setenv("PYTHIA_FAMILY_RECALIBRATION_MODE", "apply")
    fr.reset_factor_cache()
    # The factors the 2026-10-02 calibration round fitted: old prompt only.
    fr._FACTOR_CACHE["as_of"] = "2026-10"
    fr._FACTOR_CACHE["factors"] = {
        ("gpt", "ACE", "FATALITIES", "spd", None, None): {b: 1.0 for b in range(1, 8)},
    }
    try:
        info = fr.lookup("gpt-6-sol", "ACE", "FATALITIES",
                         base_rate_block_version="prior_anchor_v1", rc_guidance=rc_guidance)
    finally:
        fr.reset_factor_cache()
    assert info["mode"] == "auto_shadow"
    assert info["factors"]
