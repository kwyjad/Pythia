# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""monthly_report_inputs.json: each section on a small fixture, the stub
paths, the Track 1 rule, the confound detector and a rewritten resolution."""

from __future__ import annotations

import json
from datetime import date

import duckdb
import pytest

from scripts.ai_bundle import error_attribution as ea
from scripts.ai_bundle import monthly_report_inputs as mri

ROUND = date(2026, 11, 11)


def _meta(qid, hz, metric, track, run, **flags):
    base = {"question_id": qid, "hazard_code": hz, "metric": metric, "track": track, "run_id": run,
            "lineup_id": "L1", "base_rate_block_version": "none", "rc_guidance": "legacy",
            "advice_arm": "none", "recalibration_mode": "off", "input_partial_month": False,
            "base_rate_block_group": "none", "rc_shift_arm": "none"}
    base.update(flags)
    return base


def _ctx(con=None, *, extra_scores=None, indicative=()):
    """Ten Track 1 SPD questions, ten Track 1 binary, two Track 2 SPD."""
    qmeta, scores = {}, {}
    for i in range(10):
        q = f"A{i:02d}_ACE_FATALITIES_2026-10"
        run = "fc_1789534892" if i < 5 else "fc_1785557077"
        qmeta[q] = _meta(q, "ACE", "FATALITIES", 1, run,
                         lineup_id="L1" if i < 5 else "L2",
                         input_partial_month=i >= 5)
        for h in (1, 2):
            scores[(q, "ensemble_bayesmc_v2", h, "crps")] = 0.20
            scores[(q, "ensemble_mean_v2", h, "crps")] = 0.10
            scores[(q, "gpt-6-sol", h, "crps")] = 0.05
            scores[(q, "__ext_climatology", h, "crps")] = 0.30 + 0.01 * i
            scores[(q, "sibyl", h, "crps")] = 0.15
        b = f"B{i:02d}_DR_EVENT_OCCURRENCE_2026-10"
        qmeta[b] = _meta(b, "DR", "EVENT_OCCURRENCE", 1, "fc_1789534892")
        scores[(b, "ensemble_bayesmc_v2", 1, "brier")] = 0.04
        scores[(b, "ensemble_mean_v2", 1, "brier")] = 0.06
        scores[(b, "__ext_climatology", 1, "brier")] = 0.05
    for i in range(2):
        q = f"T{i:02d}_ACE_FATALITIES_2026-10"
        qmeta[q] = _meta(q, "ACE", "FATALITIES", 2, "fc_1789534892")
        # Wildly different, so a Track 2 leak would move every figure.
        scores[(q, "track2_flash", 1, "crps")] = 9.0
        scores[(q, "ensemble_bayesmc_v2", 1, "crps")] = 9.0
        scores[(q, "ensemble_mean_v2", 1, "crps")] = -9.0
        scores[(q, "sibyl", 1, "crps")] = -9.0
        scores[(q, "__ext_climatology", 1, "crps")] = 0.3
    scores.update(extra_scores or {})
    ctx = ea.Context(con=con, include_test=False, qmeta=qmeta, outcomes={},
                     bundle_qids=sorted(qmeta), scores=scores)
    ctx.index_scores()
    ctx.indicative = set(indicative)
    return ctx


def test_forecast_vintage_counts_flags_and_names_single_value_ones():
    v = mri.build_forecast_vintage(_ctx())
    runs = {r["latest_run_id"]: r for r in v["runs"]}
    assert runs["fc_1789534892"]["run_date"] == "2026-09-16"
    assert runs["fc_1785557077"]["n_questions"] == 5
    assert runs["fc_1785557077"]["flags"]["input_partial_month"] == {"True": 5}
    single = {s["flag"] for s in v["single_value_flags"]}
    assert {"rc_guidance", "advice_arm", "recalibration_mode"} <= single
    assert "lineup_id" not in single


def test_paired_comparisons_are_track_1_only_and_never_blend_families():
    out = mri.build_paired_comparisons(_ctx())
    rows = out["comparisons"]
    for r in rows:
        assert r["tracks_seen"] in (["1"], [])
    spd = next(r for r in rows if r["comparison"] == "bayesmc_minus_mean" and r["score_family"] == "spd")
    assert spd["difference"] == pytest.approx(0.10) and spd["n_paired_questions"] == 10
    assert spd["verdict"] == "ensemble_mean_v2 better"
    binary = next(r for r in rows if r["comparison"] == "bayesmc_minus_mean"
                  and r["score_family"] == "binary")
    assert binary["score_type"] == "brier" and binary["difference"] == pytest.approx(-0.02)
    member = next(r for r in rows if r["comparison"] == "member_minus_primary")
    assert member["a"] == "gpt-6-sol" and member["difference"] == pytest.approx(-0.15)
    sib = [r for r in rows if r["comparison"] == "sibyl_minus_primary"]
    assert {r["score_type"] for r in sib} == {"crps", "brier"}
    # Track 2 would have pulled the bayesmc - mean difference to ~3.
    assert all(abs(r["difference"] or 0) < 1 for r in rows)


def test_indicative_months_are_left_out_of_the_pairs():
    ind = {(f"A{i:02d}_ACE_FATALITIES_2026-10", h) for i in range(10) for h in (1, 2)}
    out = mri.build_paired_comparisons(_ctx(indicative=ind))
    spd = next(r for r in out["comparisons"]
               if r["comparison"] == "bayesmc_minus_mean" and r["score_family"] == "spd")
    assert spd["n_paired_questions"] == 0 and spd["verdict"] == "too few"


def test_the_confound_detector_finds_two_flags_with_identical_arms():
    out = mri.build_confounded_experiments(_ctx())
    pairs = [p for p in out["pairs"] if p["hazard_code"] == "ACE" and p["track"] == 1]
    assert len(pairs) == 1
    p = pairs[0]
    assert {p["flag_a"], p["flag_b"]} == {"input_partial_month", "lineup_id"}
    assert p["arms"] == {"False": "L1", "True": "L2"} and p["derived"] is False


def test_crisiswatch_split_flags_classes_drawn_from_different_runs():
    ctx = _ctx()
    facts = {}
    for i in range(10):
        q = f"A{i:02d}_ACE_FATALITIES_2026-10"
        age = 0 if i < 5 else 4
        facts[q] = {"crisiswatch": {"available": True, "edition_age_months": age}}
    out = mri.build_crisiswatch_split(ctx, facts)
    t1 = out["by_track"]["1"]
    cls = {c["class"]: c for c in t1["classes"]}
    assert cls["current"]["n_questions"] == 5 and cls["stale"]["n_questions"] == 5
    assert cls["current"]["latest_run_ids"] == ["fc_1789534892"]
    assert t1["confounded_with_run"] is True
    assert ["current", "stale"] in t1["class_pairs_with_no_shared_run"]
    assert "2" in out["by_track"]  # Track 2 reported apart, never pooled
    assert mri.crisiswatch_class({"available": False}) == "absent"
    assert mri.crisiswatch_class({"available": True, "edition_age_months": 2}) == "other"


NIC_REPEATED = """\
GDACS EVENT HISTORY (NIC — drought):
Coverage: 2026-06 to 2026-09 (4 months)
Overall event rate: 0 events in 4 months (0%)

Recent event timeline:
  2026-08: -- | 2026-08: -- | 2026-09: --

INTERPRETATION: x
"""
UNAVAILABLE = """\
GDACS EVENT HISTORY (ERI — drought):
History unavailable for ERI / drought: the database holds GDACS DR alerts for only 4 month(s) (2026-06 to 2026-09); fewer than 24. Do NOT read this as a 0% base rate.

X
"""


def test_gdacs_checks_read_the_stored_prompt():
    ctx = _ctx()
    qs = sorted(q for q in ctx.qmeta if q.startswith("B"))
    facts = {
        qs[0]: mri.question_facts({"spd_prompt": NIC_REPEATED,
                                   "inject_status": {"gdacs_history": {"event_months": 4}}}),
        qs[1]: mri.question_facts({"spd_prompt": UNAVAILABLE, "inject_status": {}}),
        qs[2]: mri.question_facts({"spd_prompt": "no block here", "inject_status": {}}),
    }
    dr = mri.build_gdacs_prompt_checks(ctx, facts)["by_hazard"]["DR"]
    assert dr["prompts"] == 3 and dr["with_block"] == 2
    assert dr["repeated_months"] == 1 and dr["zero_where_db_has_events"] == 1
    assert dr["unavailable"] == 1 and dr["rate_under_min_months"] == 1
    assert dr["questions"] == 10


def _res_db(tmp_path):
    con = duckdb.connect(str(tmp_path / "r.duckdb"))
    con.execute("CREATE TABLE resolutions (question_id VARCHAR, horizon_m INTEGER, value DOUBLE, "
                "source_snapshot_ym VARCHAR)")
    con.execute("CREATE TABLE resolution_vintages (question_id VARCHAR, horizon_m INTEGER, "
                "vintage VARCHAR, value DOUBLE, recorded_at TIMESTAMP)")
    ins = "INSERT INTO resolutions VALUES (?, ?, ?, ?)"
    a0, a1 = "A00_ACE_FATALITIES_2026-10", "A01_ACE_FATALITIES_2026-10"
    con.execute(ins, [a0, 1, 12, "2026-11-11T10:27:49+00:00"])  # new, read first this round
    con.execute(ins, [a0, 2, 40, "2026-11-11T10:27:49+00:00"])  # new, rewritten
    con.execute(ins, [a1, 1, 3, "2026-10-11T09:00:00"])          # earlier round
    con.execute(ins, [a1, 2, 0, None])                           # zero default, first read now
    con.execute(ins, ["B00_DR_EVENT_OCCURRENCE_2026-10", 1, 0, None])  # undated
    vin = "INSERT INTO resolution_vintages VALUES (?, ?, 'first', ?, ?)"
    con.execute(vin, [a0, 1, 12, "2026-11-11 10:57:00"])
    con.execute(vin, [a0, 2, 35, "2026-10-12 10:00:00"])
    con.execute(vin, [a1, 1, 3, "2026-10-12 10:00:00"])
    con.execute(vin, [a1, 2, 0, "2026-11-11 10:57:00"])
    return con


def test_new_this_round_counts_a_rewritten_resolution_as_new(tmp_path):
    con = _res_db(tmp_path)
    ctx = _ctx(con)
    assert mri.latest_snapshot_date(con) == ROUND
    dates = mri.resolution_dates(con, ctx.bundle_qids)
    section, new_pairs = mri.build_new_this_round(ctx, ROUND, dates)
    ace = next(g for g in section["groups"] if g["hazard_code"] == "ACE" and g["track"] == 1)
    assert ace["horizon_months_new"] == 3 and ace["horizon_months_earlier"] == 1
    assert ace["questions_new"] == 2 and ace["new_rewritten"] == 1
    assert section["rewritten"] == [{"question_id": "A00_ACE_FATALITIES_2026-10", "horizon_m": 2,
                                     "rewritten": True, "first_recorded": "2026-10-12",
                                     "first_value": 35.0, "value": 40.0}]
    dr = next(g for g in section["groups"] if g["hazard_code"] == "DR")
    assert dr["horizon_months_undated"] == 1 and dr["horizon_months_new"] == 0
    assert ("A01_ACE_FATALITIES_2026-10", 1) not in new_pairs
    assert ("A01_ACE_FATALITIES_2026-10", 2) in new_pairs


def test_case_candidates_rank_new_pairs_against_climatology_and_write_records(tmp_path):
    ctx = _ctx()
    new = {(f"A{i:02d}_ACE_FATALITIES_2026-10", 1) for i in range(10)}
    new |= {(f"B{i:02d}_DR_EVENT_OCCURRENCE_2026-10", 1) for i in range(3)}
    facts = {q: {"resolutions": {1: {"value": 7, "source_desc": "acled", "resolution_reading": "first",
                                     "observed_month": "2026-10", "scoring_class": None}}}
             for q, _h in new}
    cand = mri.build_case_candidates(ctx, facts, new)
    spd = cand["by_family"]["spd"]
    assert len(spd["worst"]) == 5 and len(spd["best"]) == 5
    # Primary is bayesmc (0.20); climatology rises with i, so A00 is the worst excess.
    assert spd["worst"][0]["question_id"] == "A00_ACE_FATALITIES_2026-10"
    assert spd["worst"][0]["excess_over_climatology"] == pytest.approx(-0.10)
    assert spd["best"][0]["question_id"] == "A09_ACE_FATALITIES_2026-10"
    assert spd["worst"][0]["sibyl"] == 0.15 and spd["worst"][0]["source_desc"] == "acled"
    assert all(r["track"] == 1 for r in spd["worst"] + spd["best"])
    assert cand["by_family"]["binary"]["score_type"] == "brier"

    staging = tmp_path / "st"
    (staging / "questions").mkdir(parents=True)
    for q, _h in new:
        size = 5000 if q.startswith("A00") else 100
        (staging / "questions" / f"{q}.json").write_text(json.dumps({"x": "y" * size}))
    out = mri.write_case_studies(staging, cand, cap_bytes=1000)
    assert {d["question_id"] for d in out["dropped_for_size"]} == {"A00_ACE_FATALITIES_2026-10"}
    assert out["total_bytes"] <= 1000
    written = {p.stem for p in (staging / "case_studies_new").iterdir()}
    assert "A00_ACE_FATALITIES_2026-10" not in written and len(written) == len(out["written"])


def test_build_writes_every_section_and_stubs_without_a_context(tmp_path):
    con = _res_db(tmp_path)
    staging = tmp_path / "st"
    staging.mkdir()
    doc = mri.build(con, None, {}, staging, ctx_error="boom")
    assert doc["section_status"]["round"] == "ok"
    assert doc["round"]["round_date"] == "2026-11-11"
    for name in ("forecast_vintage", "new_this_round", "paired_comparisons", "crisiswatch_split",
                 "gdacs_prompt_checks", "confounded_experiments", "case_candidates",
                 "case_studies_new"):
        assert doc[name]["stub"] is True and "boom" in doc[name]["reason"]
    assert json.loads((staging / mri.FILE_NAME).read_text())["new_this_round"]["stub"] is True


def test_a_failing_section_is_stubbed_and_the_rest_still_written(tmp_path, monkeypatch):
    con = _res_db(tmp_path)
    staging = tmp_path / "st"
    (staging / "questions").mkdir(parents=True)
    monkeypatch.setattr(mri, "build_paired_comparisons", lambda ctx: 1 / 0)
    doc = mri.build(con, _ctx(con), {}, staging, round_date=date(2026, 11, 1))
    assert doc["round"]["round_date"] == "2026-11-01" and doc["round"]["round_date_source"] == "--round-date"
    assert doc["paired_comparisons"]["stub"] is True
    assert "ZeroDivisionError" in doc["paired_comparisons"]["reason"]
    assert doc["section_status"]["forecast_vintage"] == "ok"
    assert doc["section_status"]["new_this_round"] == "ok"


def test_no_round_date_stubs_new_this_round_and_the_cases(tmp_path):
    con = duckdb.connect(str(tmp_path / "e.duckdb"))
    staging = tmp_path / "st"
    staging.mkdir()
    doc = mri.build(con, _ctx(con), {}, staging)
    assert doc["new_this_round"]["stub"] is True
    assert doc["case_candidates"]["stub"] is True and doc["case_studies_new"]["stub"] is True
    assert doc["section_status"]["paired_comparisons"] == "ok"
