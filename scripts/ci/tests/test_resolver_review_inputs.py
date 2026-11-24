# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The resolver review's inputs and the ``resolver_review/`` folder in Drive.

A small staging tree shaped like the 11 November 2026 scoring chain: a
Resolver Update with its debug bundle (a zip inside the tree), the three
post-ingest stages' chain diagnostics, a scored bundle and a Publish run. No
network, no DB, no Drive: Drive is a directory.
"""

from __future__ import annotations

import csv
import dataclasses
import io
import json
import shutil
import zipfile
from pathlib import Path

import pytest

from scripts.ci import build_cycle_diagnostics as bcd
from scripts.ci import drive_archive as da
from scripts.ci import resolver_review_inputs as rri
from scripts.ci.tests.test_drive_archive import FakeRclone

RU, CR, CS, CC, PUB, CW = "100", "101", "102", "103", "104", "99"


def _csv(header, rows, preamble="") -> str:
    buf = io.StringIO()
    if preamble:
        for line in preamble.splitlines():
            buf.write(f"# {line}\n")
    w = csv.writer(buf)
    w.writerow(header)
    w.writerows(rows)
    return buf.getvalue()


CELL_HEADER = ["iso3", "hazard", "ym", "status", "value", "flags", "provisional", "reason_code",
               "answering_rung", "ceiling_basis", "run_type"]

#: (iso3, hazard, ym, status, value, flags, provisional, reason, rung, ceiling basis)
CELLS = [
    ("PAK", "FL", "2026-10", "RESOLVED_VALUE", 33000000, "", "True", "", "reliefweb_extracted", "population_share"),
    ("CUB", "FL", "2026-10", "RESOLVED_VALUE", 645000, "", "True", "", "reliefweb_extracted", "population_share"),
    ("CUB", "TC", "2026-10", "RESOLVED_VALUE", 645000, "", "True", "", "reliefweb_extracted", "population_share"),
    ("THA", "FL", "2026-09", "RESOLVED_VALUE", 7374905, "", "True", "", "reliefweb_extracted", "population_share"),
    ("THA", "FL", "2026-10", "RESOLVED_VALUE", 7374905, "", "True", "", "reliefweb_extracted", "population_share"),
    ("BGD", "FL", "2026-09", "RESOLVED_VALUE", 1300000, "", "True", "", "reliefweb_extracted", ""),
    ("HTI", "TC", "2026-09", "RESOLVED_VALUE", 5000, "ceiling_exceeded", "True", "", "ifrc_go", "gdacs_exposed"),
    ("NGA", "FL", "2026-09", "RESOLVED_VALUE", 20000, "", "True", "", "idmc_idu", "population_share"),
    # History: above 10% of population, but long frozen and in no scored question.
    ("AFG", "FL", "2010-08", "RESOLVED_VALUE", 9000000, "", "False", "", "reliefweb_extracted", ""),
    ("SOM", "DR", "2026-09", "RESOLVED_ZERO", 0, "", "True", "", "", ""),
    ("ETH", "DR", "2026-09", "", "", "", "", "indicator_no_coverage", "", ""),
    ("ATA", "FL", "2012-01", "", "", "", "", "unexplained_no_row", "", ""),
    ("PHL", "FL", "2026-11", "", "", "", "", "month_in_progress", "", ""),
]

FIG_HEADER = ["iso3", "hazard", "ym", "outcome", "doc_id", "value", "unit", "stated_value",
              "stated_unit", "value_persons", "conversion_factor", "preference_rank", "quote"]
FIGURES = [
    ("PAK", "FL", "2026-10", "accepted", "d1", 33000000, "people", 33000000, "people", 33000000, 1, 0,
     "The 2022 floods affected 33 million people across the country."),
    ("CUB", "FL", "2026-10", "accepted", "d2", 645000, "people", 645000, "people", 645000, 1, 0,
     "Some 645,000 people were affected by Hurricane Melissa."),
    ("CUB", "TC", "2026-10", "accepted", "d2", 645000, "people", 645000, "people", 645000, 1, 0,
     "Some 645,000 people were affected by Hurricane Melissa."),
    ("THA", "FL", "2026-09", "accepted", "d3", 7374905, "households", 1474981, "households", 7374905, 5, 0,
     "From 16 September to 6 October 2026, 1,474,981 households (4,188,452 people) were affected."),
    ("THA", "FL", "2026-10", "accepted", "d3", 7374905, "households", 1474981, "households", 7374905, 5, 0,
     "From 16 September to 6 October 2026, 1,474,981 households (4,188,452 people) were affected."),
    ("BGD", "FL", "2026-09", "accepted", "d4", 1300000, "people", 1300000, "people", 1300000, 1, 0,
     "Flooding in September 2026 affected 1.3 million people."),
    ("BGD", "FL", "2026-09", "rejected", "d5", 50, "people", 50, "people", 50, 1, 1, "50 people"),
]

#: (question_id, iso3, hazard, metric, horizon, observed, value, bucket, source_desc)
RESOLUTIONS = [
    ("PAK_FL_PA_2026-10", "PAK", "FL", "PA", 1, "2026-10-01", 33000000, 6, "haz_resolutions:reliefweb_extracted:affected"),
    ("CUB_FL_PA_2026-10", "CUB", "FL", "PA", 1, "2026-10-01", 645000, 6, "haz_resolutions:reliefweb_extracted:affected"),
    ("CUB_TC_PA_2026-10", "CUB", "TC", "PA", 1, "2026-10-01", 645000, 6, "haz_resolutions:reliefweb_extracted:affected"),
    ("THA_FL_PA_2026-09", "THA", "FL", "PA", 1, "2026-09-01", 7374905, 6, "haz_resolutions:reliefweb_extracted:affected"),
    ("THA_FL_PA_2026-09", "THA", "FL", "PA", 2, "2026-10-01", 7374905, 6, "haz_resolutions:reliefweb_extracted:affected"),
    ("BGD_FL_PA_2026-09", "BGD", "FL", "PA", 1, "2026-09-01", 1300000, 6, "haz_resolutions:reliefweb_extracted:affected"),
    ("HTI_TC_PA_2026-09", "HTI", "TC", "PA", 1, "2026-09-01", 5000, 2, "haz_resolutions:ifrc_go:affected"),
    ("NGA_FL_PA_2026-09", "NGA", "FL", "PA", 1, "2026-09-01", 20000, 3, "haz_resolutions:idmc_idu:displaced_lower_bound"),
]
MODELS = ("ensemble_mean_v2", "gpt-6-sol")
OLD, RATE = 837934.0, 0.1


def _bucket6_values():
    return [r[6] for r in RESOLUTIONS if r[2] == "FL" and r[7] == 6]


def _scored_zip(*, gap: bool = False) -> bytes:
    buf = io.BytesIO()
    by_q: dict[str, list] = {}
    for r in RESOLUTIONS:
        by_q.setdefault(r[0], []).append(r)
    flat = []
    with zipfile.ZipFile(buf, "w") as zf:
        for qid, rows in by_q.items():
            q = {"question_id": qid, "iso3": rows[0][1], "hazard_code": rows[0][2], "metric": rows[0][3]}
            horizons = [r[4] for r in rows]
            scores = []
            for r in rows:
                for m in MODELS:
                    if gap and qid == "BGD_FL_PA_2026-09" and m == "gpt-6-sol":
                        continue
                    scores.append({"horizon_m": r[4], "model_name": m, "score_type": "brier", "value": 0.4})
                    flat.append([qid, r[1], r[2], r[3], "spd", r[4], m, "brier", 0.4, r[6], r[5], "",
                                 r[8], "fc_1", "True", "scored", ""])
            rec = {
                "question": q,
                "outcome": {"resolutions": [{"horizon_m": r[4], "observed_month": r[5], "value": r[6],
                                             "realized_bucket": r[7], "source_desc": r[8]} for r in rows]},
                "scores": scores,
                "ensemble": {"ensemble_mean_v2": {"months": {str(h): {"1": 1.0} for h in horizons}}},
                "members": [{"model_name": "gpt-6-sol", "ok": True, "status": "ok"}],
            }
            zf.writestr(f"questions/{qid}.json", json.dumps(rec))
        zf.writestr("scores_flat.csv", _csv(
            ["question_id", "iso3", "hazard_code", "metric", "score_family", "horizon_m", "model_name",
             "score_type", "value", "resolved_value", "observed_month", "source_snapshot_ym",
             "source_desc", "run_id", "is_latest_run", "scoring_class", "resolution_reading"], flat))
        zf.writestr("digest.md", "# digest\n")
    return buf.getvalue()


def _debug_zip(*, big_ledger_rows: int = 0) -> bytes:
    files: dict[str, str] = {}
    files["README.md"] = "# Resolver debug bundle\n"
    files["checks/issues.json"] = json.dumps({"issues": [
        {"id": "connector_failed_conflictforecast_forecasts", "severity": "degraded", "title": "empty",
         "first_seen": "2026-11-08", "runs_seen": 2, "owner": "pythia"},
        {"id": "acled_cast_stale_vintage", "severity": "known", "title": "stale", "first_seen": "2026-01-15",
         "runs_seen": 14, "owner": "external"},
    ]})
    files["checks/issues.md"] = "# Issues\n"
    files["checks/contradictions.md"] = (
        "# Contradictions\n\n| verdict | check | observed | expected |\n| --- | --- | --- | --- |\n"
        "| PASS | no_partial_acled_month | 0 | 0 |\n"
        "| FAIL | no_stored_forecast_vintage_past_its_threshold | 2 of 3 sources | at most 45 days |\n"
        "| SKIP | emdat_is_read_when_a_key_is_configured | no key | |\n\n### PASS — x\n\ndetail\n")
    files["checks/reconciliation.md"] = (
        "# Reconciliation\n\nRun start: `2026-11-11T03:00:00Z`\n\n"
        "| connector | status | fetched | normalised | claimed written | rows written (MERGE) | target table "
        "| source filter | touched since run start | table delta | agrees |\n"
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | --- |\n"
        "| acled_client | ok | 424 | 424 | 424 | | facts_resolved | | 424 | 0 | yes |\n"
        "| crisiswatch | ok | 75 | 75 | 75 | | crisiswatch_entries | | 0 | 0 | unchanged on purpose |\n"
        "| idmc | superseded by idmc_helix | | | | | | | | | n/a — deliberately skipped |\n"
        "| conflictforecast_forecasts | error | 0 | 0 | | | conflict_forecasts | | 0 | 0 | — |\n"
        "| nmme | ok | 3276 | 3276 | 3276 | | seasonal_forecasts | | 0 | 0 | **NO** |\n")
    files["logs/log_index.md"] = "# Log index\n"
    files["logs/phase26_conflict_extract.log"] = (
        "Traceback (most recent call last):\n  File x\nValueError: Unknown level: ''\n")
    files["logs/phase25_haz_flood.log"] = "INFO ok\n"
    files["db/freshness.csv"] = _csv(
        ["table", "date_column", "max_value", "age_days", "staleness_threshold_days", "rows", "verdict"],
        [["conflict_forecasts", "forecast_issue_date", "2025-12-01", 345, 45, 10, "stale"],
         ["facts_resolved", "updated_at", "2026-11-11", 0, 30, 100, "fresh"]], preamble="Freshness")
    sig_before = {"all_counts": {"conflict_displacement_candidates": 117,
                                 "conflict_displacement_cell_ledger": 1128,
                                 "dtm_conflict_rounds": 346, "dtm_conflict_lower_bounds": 106}}
    sig_after = json.loads(json.dumps(sig_before))
    files["db/signature_before.json"] = json.dumps(sig_before)
    files["db/signature_after.json"] = json.dumps(sig_after)
    files["db/queries/crisiswatch_editions.csv"] = _csv(
        ["year", "month", "entries", "countries", "deteriorated", "fetched_at"],
        [[2026, 10, 75, 75, 7, "2026-11-01"], [2026, 9, 74, 74, 6, "2026-10-01"]], preamble="SQL")
    files["db/queries/conflict_forecast_vintages.csv"] = _csv(["source", "vintage"], [["ACLED_CAST", "2025-12-01"]])
    files["hazard/acceptance_report.md"] = (
        "# Acceptance\n\n## Rates\n\n| hazard | months in window | cells assessed | resolved | rate |\n"
        "|---|---:|---:|---:|---:|\n| DR | 12 of 12 | 3,024 | 2,549 | 84.3% |\n"
        "| FL | 12 of 12 | 3,024 | 2,112 | 69.8% |\n")
    cells = [list(c) + ["live"] for c in CELLS]
    for i in range(big_ledger_rows):
        cells.append(["ZZZ", "TC", f"{2000 + i % 20}-01", "RESOLVED_ZERO", 0, "", "False",
                      "", "", "", "backcast"])
    files["hazard/cell_ledger.csv"] = _csv(CELL_HEADER, cells, preamble="One row per cell")
    files["hazard/figures_ledger.csv"] = _csv(FIG_HEADER, FIGURES, preamble="Every figure")
    files["hazard/extraction_budget.csv"] = _csv(
        ["caller", "day", "calls", "billed_calls", "cost_usd"],
        [["live", "2026-11-11", 400, 400, 0.75], ["backcast", "2026-11-02", 10, 10, 0.02],
         ["backcast", "2026-10-20", 6826, 6826, 13.1], ["backcast", "2026-08-01", 5, 5, 0.01]],
        preamble="SQL")
    files["hazard/backcast_progress.csv"] = _csv(
        ["hazard", "ym", "status"], [["TC", "2026-01", "ok"], ["FL", "2026-01", "ok"], ["FL", "2026-02", "deferred"]])
    files["hazard/conflict_displacement/cell_ledger.csv"] = _csv(
        ["iso3", "ym", "reason_code", "assessed_at"],
        [["SDN", "2026-09", "admitted_figure", "2026-10-14T10:00:00"],
         ["COD", "2026-08", "no_documents", "2026-10-14T09:00:00"]])
    files["config/connectors/known_issues.yml"] = (
        "issues:\n"
        "  - id: acled_cast_stale_vintage\n    owner: external\n    review_by: 2026-10-15\n"
        "    first_seen: 2026-01-15\n"
        "  - id: emdat_auth_rejected\n    owner: external\n    review_by: 2027-03-10\n"
        "  - id: spei3_commit_credential\n    owner: pythia\n    review_by: never\n"
        "    no_review_reason: refusal-based guard\n")
    files["config/rulebook.yaml"] = "freeze_days: 60\n"
    files["run/step_timings.csv"] = _csv(
        ["number", "step", "status", "conclusion", "started_at", "completed_at", "duration_sec"],
        [[30, "Phase 2.5 flood", "completed", "success", "a", "b", 1200],
         [31, "Phase 2.6: conflict displacement extraction", "completed", "success", "a", "a", 0],
         [32, "Phase 2.6: DTM conflict lower bounds", "completed", "success", "a", "a", 0],
         [33, "Set up job", "completed", "success", "a", "a", 0]], preamble="source: api")
    files["run/workflow_run.json"] = json.dumps({"id": int(RU), "run_started_at": "2026-11-11T03:00:00Z",
                                                 "head_sha": "a1186a0"})
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, body in files.items():
            zf.writestr(name, body)
    return buf.getvalue()


def _api(rid: str, name: str, conclusion: str = "success") -> str:
    return json.dumps({"id": int(rid), "name": name, "conclusion": conclusion, "event": "schedule",
                       "run_started_at": "2026-11-11T03:00:00Z", "updated_at": "2026-11-11T04:00:00Z"})


def _w(path: Path, body) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body) if isinstance(body, bytes) else path.write_text(body, encoding="utf-8")


def build_staging(root: Path, *, with_resolver: bool = True, big_ledger_rows: int = 0,
                  scored_gap: bool = False) -> Path:
    st = root / "staging"
    _w(st / "README.md", "# cycle\n")
    _w(st / "manifest.json", json.dumps({"runs": [{"folder": "x"}]}))
    _w(st / f"08_refresh_crisiswatch__{CW}" / "api_run.json", _api(CW, "Refresh CrisisWatch Data"))
    if with_resolver:
        ru = st / f"09_resolver_update__{RU}"
        _w(ru / "api_run.json", _api(RU, "Resolver Update"))
        _w(ru / f"resolver-debug-bundle-{RU}" / f"resolver-debug-bundle-{RU}.zip",
           _debug_zip(big_ledger_rows=big_ledger_rows))
        bf = ru / "backfill-diagnostics"
        _w(bf / "dtm_conflict.json", json.dumps({"countries": 3}))
        _w(bf / "ingestion" / "idmc" / "summary.json", json.dumps({"rows_written": 506}))
    cr = st / f"10_compute_resolutions__{CR}"
    _w(cr / "api_run.json", _api(CR, "Pythia — Compute Resolutions"))
    d = cr / f"chain-diagnostics-compute_resolutions-{CR}"
    _w(d / "summary.md", "# CR\n")
    _w(d / "issues.md", "# issues\n")
    _w(d / "issues.json", json.dumps({"issues": [{"id": "stale_columns", "severity": "info"}]}))
    _w(d / "lineage.json", json.dumps({"canonical_db_run_downloaded": RU, "canonical_db_run_uploaded": CR,
                                       "upload_guard_verdict": "ok", "trigger_run": RU}))
    _w(d / "resolutions_by_source_desc.csv", _csv(["source_desc", "rows"], [["zero_default", 195]]))
    _w(d / "reports" / "compute_resolutions.json", json.dumps({
        "calendar_cutoff": "2026-10", "rows_written": 853, "vintages_written": 245,
        "purged_stale_beyond_cutoff": 0,
        "outcome_counts": {
            "FL/PA": {"sourced": 100, "zero_default": 0, "unresolved_for_lag": 36,
                      "unresolved_machine_no_row": 6, "unresolved_future": 470},
            "TC/EVENT_OCCURRENCE": {"sourced": 25, "zero_default": 45, "unresolved_no_coverage": 230},
        }}))
    cs = st / f"11_compute_scores__{CS}"
    _w(cs / "api_run.json", _api(CS, "Pythia — Compute SPD Scores"))
    d = cs / f"chain-diagnostics-compute_scores-{CS}"
    _w(d / "summary.md", "# CS\n")
    _w(d / "issues.md", "# issues\n")
    for name in ("compute_scores", "score_baselines", "score_views"):
        _w(d / "reports" / f"{name}.json", json.dumps({"name": name}))
    cc = st / f"12_compute_calibration__{CC}"
    _w(cc / "api_run.json", _api(CC, "Pythia — Compute Calibration Weights & Advice"))
    d = cc / f"chain-diagnostics-compute_calibration-{CC}"
    _w(d / "summary.md", "# CC\n")
    _w(d / "issues.md", "# issues\n")
    emp = sum(_bucket6_values()) / len(_bucket6_values())
    _w(d / "centroid_ema_updates.csv", _csv(
        ["hazard_code", "metric", "bucket_index", "old", "new", "empirical", "n_resolutions"],
        [["FL", "PA", 6, OLD, OLD + RATE * (emp - OLD), emp, len(_bucket6_values())],
         ["ACE", "FATALITIES", 7, 1500, 1629, 2793, 12]]))
    _w(d / "family_recalibration.csv", _csv(["family", "clip_share"], [["claude", 0.143]]))
    _w(d / "calibration_weights.csv", _csv(
        ["hazard_code", "metric", "questions_with_member_scores", "floor", "weight_rows_written_this_run",
         "status"], [["ACE", "FATALITIES", 66, 20, 5, "weighted"], ["FL", "PA", 14, 20, 0, "below floor"]]))
    _w(d / "sibyl_calibration_steps.json", json.dumps({"sibyl_llm_calls_this_run": 17}))
    _w(cc / "pythia-ai-analysis-bundle" / "scored_forecast_analysis__2026-11.zip", _scored_zip(gap=scored_gap))
    pub = st / f"13_publish_latest_data__{PUB}"
    _w(pub / "api_run.json", _api(PUB, "Publish Latest Data (Release)"))
    _w(pub / f"chain-diagnostics-publish_latest_data-{PUB}" / "summary.md", "# pub\n")
    _w(pub / "resolver-inspect" / "resolver_inspect.md",
       "# Inspect\n\n| Source | Coverage |\n|---|---|\n| CrisisWatch | 70 / 120 |\n")
    return st


@pytest.fixture()
def payload(tmp_path):
    st = build_staging(tmp_path)
    return rri.write_inputs(st, tmp_path / "out.json", tmp_path / "work")


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def test_the_file_has_every_section_and_writes_valid_json(tmp_path, payload):
    for key in ("chain", "issues", "ingestion", "contradictions", "pa_machine", "conflict_displacement",
                "resolutions", "plausibility", "scoring", "calibration", "known_issues"):
        assert key in payload, key
        assert not (isinstance(payload[key], dict) and payload[key].get("stub")), (key, payload[key])
    assert json.loads((tmp_path / "out.json").read_text())["run_date"] == "2026-11-11"
    assert payload["sources"]["debug_bundle"] == f"resolver-debug-bundle-{RU}.zip"


def test_chain_stages_lineage_and_steps_that_did_nothing(payload):
    stages = {s["slug"]: s for s in payload["chain"]["stages"]}
    assert stages["compute_resolutions"]["db_downloaded_from"] == RU
    assert stages["compute_resolutions"]["upload_guard"] == "ok"
    assert stages["resolver_update"]["duration_min"] == 60.0
    ru = payload["chain"]["resolver_update"]
    zero = {z["step"]: z for z in ru["zero_second_steps"]}
    step = "Phase 2.6: conflict displacement extraction"
    assert step in zero and zero[step]["logs_with_traceback"] == ["phase26_conflict_extract.log"]
    assert "Set up job" in ru["zero_second_steps_without_logs"]
    assert ru["logs_with_traceback"] == ["phase26_conflict_extract.log"]


def test_issues_count_by_severity_and_read_the_other_stages(payload):
    ru = payload["issues"]["resolver_update"]
    assert ru["counts"] == {"degraded": 1, "known": 1}
    assert ru["issues"][0]["first_seen"] == "2026-11-08" and ru["issues"][0]["runs_seen"] == 2
    assert payload["issues"]["other_stages"]["compute_resolutions"][0]["id"] == "stale_columns"


def test_ingestion_names_only_the_connectors_that_are_not_fine(payload):
    ing = payload["ingestion"]
    bad = {r["connector"] for r in ing["reconciliation"]["not_ok_or_disagreeing"]}
    assert bad == {"conflictforecast_forecasts", "nmme"}
    assert [r["table"] for r in ing["freshness_past_threshold"]] == ["conflict_forecasts"]
    cw = ing["crisiswatch"]
    assert cw["latest_edition"] == "2026-10" and cw["entries"] == 75
    assert "70 / 120" in cw["inspect_line"]
    assert cw["refresh_runs"] == {"ok": 1, "failed": 0, "runs": [int(CW)]}


def test_contradictions_keep_verdict_observed_expected(payload):
    c = payload["contradictions"]
    assert c["counts"] == {"PASS": 1, "FAIL": 1, "SKIP": 1}
    fail = [x for x in c["checks"] if x["verdict"] == "FAIL"][0]
    assert fail["observed"] == "2 of 3 sources" and fail["expected"] == "at most 45 days"


def test_pa_machine_counts_come_from_the_ledger(payload):
    pa = payload["pa_machine"]
    led = pa["cell_ledger"]
    assert led["rows"] == len(CELLS) and led["unexplained_no_row"] == 1
    reasons = {(r["reason_code"], r["hazard"]): r["cells"] for r in led["no_row_by_reason_hazard"]}
    assert reasons[("indicator_no_coverage", "DR")] == 1
    assert led["resolved_value_by_ceiling_basis"]["unrecorded"] == 2
    assert {(f["flag"], f["hazard"]) for f in led["flags_by_hazard"]} == {("ceiling_exceeded", "TC")}
    assert led["months_not_yet_ended"] == [{"ym": "2026-11", "hazard": "FL", "status": "no_row", "cells": 1}]
    budget = {(b["month"], b["caller"]): b for b in pa["extraction_budget"]}
    assert budget[("2026-11", "live")]["calls"] == 400 and ("2026-08", "backcast") not in budget
    assert budget[("2026-10", "backcast")]["cost_usd"] == 13.1
    assert {"hazard": "FL", "status": "deferred", "months": 1} in pa["backcast_progress"]
    freezes = {(f["ym"], f["hazard"]): f for f in pa["freeze_dates"]["provisional_months"]}
    assert freezes[("2026-09", "FL")]["freezes_on"] == "2026-11-29"
    assert freezes[("2026-09", "FL")]["frozen_by_run_date"] is False
    assert pa["acceptance"][0]["rows"][1]["rate"] == "69.8%"


def test_conflict_displacement_tables_summaries_and_ledger(payload):
    cd = payload["conflict_displacement"]
    assert cd["tables"]["dtm_conflict_rounds"] == {"before": 346, "after": 346, "changed": False}
    s = cd["phase26_summaries"]
    assert s["conflict_displacement_extract.json"]["stub"] is True
    assert s["dtm_conflict.json"] == {"countries": 3}
    assert cd["ledger"]["latest_ym"] == "2026-09"
    assert cd["ledger"]["assessed_at_range"] == ["2026-10-14T09:00:00", "2026-10-14T10:00:00"]
    assert cd["idmc_conflict"] == {"rows_written": 506}


def test_resolutions_keep_the_unresolved_split(payload):
    r = payload["resolutions"]
    assert r["calendar_cutoff"] == "2026-10" and r["vintages_written"] == 245
    assert r["unresolved_split"]["FL/PA"] == {"unresolved_for_lag": 36, "unresolved_machine_no_row": 6,
                                              "unresolved_future": 470}
    assert r["by_source_desc"] == [{"source_desc": "zero_default", "rows": "195"}]


def test_plausibility_reproduces_the_november_findings(payload):
    p = payload["plausibility"]
    same = p["identical_fl_tc_pa_value"]
    assert same["identical_country_months_in_scored_set"] == 1
    assert same["country_months_both_resolved_in_scored_set"] == 1
    assert same["examples"][0]["iso3"] == "CUB" and "Melissa" in same["examples"][0]["quote"]
    early = p["winning_quote_names_only_earlier_years"]
    assert early["count"] == 1 and early["examples"][0]["years_named"] == [2022]
    hh = p["household_conversion_where_quote_states_people"]
    assert hh["accepted_figures"] == 2 and hh["set_a_cell_value"] == 2 and hh["count"] == 2
    two = p["one_figure_wins_two_months"]
    assert two["count"] == 2 and two["examples"][0]["months"] == ["2026-09", "2026-10"]
    over = p["value_above_10pct_of_population"]
    # PAK 13%, THA 10.3% (two months); AFG 2010 is history: in the all-months count only.
    assert over["count"] == 3 and over["count_all_months"] == 4
    assert {e["iso3"] for e in over["examples"][:3]} == {"PAK", "THA"}
    assert p["round_extracted_figure"]["count"] >= 1
    lb = p["displacement_lower_bound_as_people_affected"]
    assert lb["count"] == 1 and lb["scored_pa_resolutions"] == len(RESOLUTIONS)
    shares = {s["group"]: s for s in p["zero_default_share_by_group"]}
    assert shares["TC/EVENT_OCCURRENCE"]["above_half"] and shares["TC/EVENT_OCCURRENCE"]["exempt"]
    union = p["union_of_flagged_cells"]
    # PAK FL 10, CUB FL+TC 10, THA FL 09+10: five scored cells, six resolutions, two models each.
    assert union["count"] == 5
    assert union["scored_resolutions"] == 5 and union["scored_forecaster_rows"] == 10
    assert "_flagged_cells" not in p


def test_scoring_reports_and_a_hand_off_gap(tmp_path):
    st = build_staging(tmp_path, scored_gap=True)
    out = rri.write_inputs(st, tmp_path / "o.json", tmp_path / "w")
    sc = out["scoring"]
    assert sc["reports"]["score_views"] == {"name": "score_views"}
    h = sc["hand_off"]
    assert h["resolved_question_horizons"] == len(RESOLUTIONS)
    assert h["pairs_with_forecast_but_no_score"] == 1
    assert h["gaps"] == [{"question_id": "BGD_FL_PA_2026-09", "horizon_m": 1, "model_name": "gpt-6-sol"}]


def test_hand_off_is_clean_when_every_pair_is_scored(payload):
    assert payload["scoring"]["hand_off"]["pairs_with_forecast_but_no_score"] == 0


def test_centroid_counterfactual_leaves_out_flagged_resolutions(payload):
    moves = {(m["hazard_code"], m["bucket_index"]): m for m in payload["calibration"]["centroid_moves"]}
    fl = moves[("FL", 6.0)]
    assert fl["bundle_resolutions"] == len(_bucket6_values())
    assert fl["bundle_mean"] == pytest.approx(fl["empirical"], rel=1e-6)
    assert fl["flagged_resolutions"] == 4 and fl["clean_mean"] == 1300000
    assert fl["ema_rate"] == pytest.approx(RATE)
    assert fl["counterfactual_new"] == pytest.approx(OLD + RATE * (1300000 - OLD), rel=1e-6)
    assert moves[("ACE", 7.0)]["bundle_resolutions"] == 0
    assert payload["calibration"]["groups_weighted"] == 1


def test_known_issues_overdue_against_the_run_date(payload):
    ki = {e["id"]: e for e in payload["known_issues"]["entries"]}
    assert ki["acled_cast_stale_vintage"]["overdue"] is True
    assert ki["emdat_auth_rejected"]["overdue"] is False
    assert ki["spei3_commit_credential"]["overdue"] is False
    assert payload["known_issues"]["source"] == "debug bundle"


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


def test_no_debug_bundle_degrades_section_by_section(tmp_path):
    st = build_staging(tmp_path)
    for z in st.rglob("resolver-debug-bundle-*.zip"):
        z.unlink()
    out = rri.write_inputs(st, tmp_path / "o.json", tmp_path / "w")
    for key in ("ingestion", "contradictions", "pa_machine", "conflict_displacement", "plausibility"):
        assert out[key]["stub"] is True and "debug bundle" in out[key]["reason"]
    assert out["chain"]["resolver_update"]["stub"] is True
    # Sections that do not need the bundle still answer.
    assert out["resolutions"]["calendar_cutoff"] == "2026-10"
    assert out["known_issues"]["source"] == "repository"


def test_missing_files_inside_the_bundle_are_stubs_not_errors(tmp_path):
    st = build_staging(tmp_path)
    zpath = next(st.rglob("resolver-debug-bundle-*.zip"))
    keep = io.BytesIO()
    with zipfile.ZipFile(zpath) as src, zipfile.ZipFile(keep, "w") as dst:
        for n in src.namelist():
            if n not in ("checks/contradictions.md", "db/freshness.csv", "hazard/extraction_budget.csv"):
                dst.writestr(n, src.read(n))
    zpath.write_bytes(keep.getvalue())
    for d in st.rglob("chain-diagnostics-compute_calibration-*"):
        shutil.rmtree(d)
    out = rri.write_inputs(st, tmp_path / "o.json", tmp_path / "w")
    assert out["contradictions"]["stub"] is True
    assert out["ingestion"]["freshness_past_threshold"]["stub"] is True
    assert out["pa_machine"]["extraction_budget"]["stub"] is True
    assert out["calibration"]["stub"] is True


def test_a_section_that_raises_becomes_a_stub(tmp_path, monkeypatch):
    st = build_staging(tmp_path)

    def boom(*_a, **_k):
        raise ValueError("bad shape")

    monkeypatch.setattr(rri, "section_contradictions", boom)
    out = rri.write_inputs(st, tmp_path / "o.json", tmp_path / "w")
    assert out["contradictions"] == {"stub": True, "reason": "ValueError: bad shape"}
    assert out["pa_machine"]["cell_ledger"]["rows"] == len(CELLS)


def test_cell_ledger_counts_on_a_ledger_larger_than_the_cap(tmp_path):
    st = build_staging(tmp_path, big_ledger_rows=30000)
    ledger_bytes = None
    with zipfile.ZipFile(next(st.rglob("resolver-debug-bundle-*.zip"))) as zf:
        ledger_bytes = zf.getinfo("hazard/cell_ledger.csv").file_size
    assert ledger_bytes > 1024 * 1024  # bigger than the per-file cap
    out = rri.write_inputs(st, tmp_path / "o.json", tmp_path / "w")
    led = out["pa_machine"]["cell_ledger"]
    assert led["rows"] == len(CELLS) + 30000
    zeros = {(r["status"], r["hazard"]): r["cells"] for r in led["by_status_hazard"]}
    assert zeros[("RESOLVED_ZERO", "TC")] == 30000
    review = da.extract_resolver_review(st, tmp_path / "rr", tmp_path / "w2", env={})
    assert not (tmp_path / "rr" / "hazard__cell_ledger.csv").exists()
    assert (tmp_path / "rr" / da.RESOLVER_REVIEW_INPUTS).stat().st_size < 1024 * 1024
    assert any(s["file"] == "hazard__cell_ledger.csv" for s in review["skipped"])


# ---------------------------------------------------------------------------
# resolver_review/ beside the chain zip
# ---------------------------------------------------------------------------


def test_the_folder_holds_the_inputs_and_the_lifted_files(tmp_path):
    st = build_staging(tmp_path)
    out = da.extract_resolver_review(st, tmp_path / "rr", tmp_path / "w", env={})
    folder = tmp_path / "rr"
    got = {p.name for p in folder.iterdir()}
    assert da.RESOLVER_REVIEW_INPUTS in got
    for name in ("README.md", "checks__issues.md", "checks__contradictions.md", "hazard__figures_ledger.csv",
                 "config__connectors__known_issues.yml", "run__workflow_run.json",
                 "compute_resolutions__summary.md", "compute_resolutions__compute_resolutions.json",
                 "compute_scores__score_views.json", "compute_scores__compute_scores.json",
                 "compute_calibration__centroid_ema_updates.csv",
                 "compute_calibration__sibyl_calibration_steps.json",
                 "publish_latest_data__summary.md", "resolver_inspect.md", "SKIPPED.md"):
        assert name in got, name
    assert "hazard__cell_ledger.csv" not in got
    assert out["files"][0]["file"] == da.RESOLVER_REVIEW_INPUTS
    assert out["total_bytes"] == sum(f["bytes"] for f in out["files"])
    skipped = (folder / "SKIPPED.md").read_text()
    assert "`publish_latest_data__issues.md`" in skipped and "not in this chain" in skipped
    assert "`hazard__cell_ledger.csv`" in skipped and "never copied" in skipped


def test_both_caps_skip_a_file_and_name_it(tmp_path):
    st = build_staging(tmp_path)
    out = da.extract_resolver_review(st, tmp_path / "rr", tmp_path / "w", file_cap=300, env={})
    sk = {s["file"]: s for s in out["skipped"]}
    assert "per-file cap" in sk["hazard__figures_ledger.csv"]["reason"]
    assert sk["hazard__figures_ledger.csv"]["bytes"] > 300
    assert not (tmp_path / "rr" / "hazard__figures_ledger.csv").exists()
    out2 = da.extract_resolver_review(st, tmp_path / "rr2", tmp_path / "w2", total_cap=1, env={})
    assert len(out2["files"]) == 1  # the inputs file always goes in, and nothing after it fits
    assert any("folder past" in s["reason"] for s in out2["skipped"])


def test_no_folder_when_the_chain_has_no_resolver_update(tmp_path):
    st = build_staging(tmp_path, with_resolver=False)
    assert da.extract_resolver_review(st, tmp_path / "rr", tmp_path / "w", env={}) is None
    assert not (tmp_path / "rr").exists()


def test_the_folder_is_scanned_for_credentials(tmp_path):
    st = build_staging(tmp_path)
    secret = "ghs_" + "Q1w2E3r4T5y6U7i8O9p0" * 2
    _w(st / f"10_compute_resolutions__{CR}" / f"chain-diagnostics-compute_resolutions-{CR}" / "summary.md",
       f"leaked {secret}")
    with pytest.raises(bcd.SecretLeak):
        da.extract_resolver_review(st, tmp_path / "rr", tmp_path / "w", env={"GH_TOKEN": secret})


def _chain_zip(staging: Path, out: Path) -> Path:
    with zipfile.ZipFile(out, "w") as zf:
        for p in sorted(staging.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(staging).as_posix())
    return out


def _spec() -> dict:
    names = da.archive_names(kind="scoring", start_date="2026-11-11", publish_run_id="43009722469")
    return {"publish_run_id": "43009722469", "names": dataclasses.asdict(names)}


def test_the_backfill_adds_only_resolver_review(tmp_path):
    st = build_staging(tmp_path)
    spec = _spec()
    names = da.ArchiveNames(**spec["names"])
    rc = FakeRclone(tmp_path / "drive")
    run_dir = tmp_path / "drive" / "Fred_archive" / names.run_dir
    run_dir.mkdir(parents=True)
    _chain_zip(st, run_dir / names.zip_name)
    _w(run_dir / "CYCLE_SUMMARY.md", "# cycle\n")
    _w(run_dir / "scored_bundle" / "digest.md", "# d\n")
    before = {p.relative_to(run_dir).as_posix(): (p.stat().st_mtime_ns, p.read_bytes())
              for p in run_dir.rglob("*") if p.is_file()}

    out = da.backfill_resolver_review(spec, rclone=rc, work=tmp_path / "bf", env={})
    assert out["status"] == "archived", out
    assert rc.copies == [("subtree", f"{names.run_dir}/resolver_review")]
    after = {p.relative_to(run_dir).as_posix(): (p.stat().st_mtime_ns, p.read_bytes())
             for p in run_dir.rglob("*") if p.is_file()}
    for rel, val in before.items():
        assert after[rel] == val  # nothing outside the new folder moved
    new = {k for k in after if k not in before}
    assert new and all(k.startswith("resolver_review/") for k in new)
    inputs = json.loads((run_dir / "resolver_review" / da.RESOLVER_REVIEW_INPUTS).read_text())
    assert inputs["plausibility"]["union_of_flagged_cells"]["count"] == 5
    assert all(p.startswith(f"Fred_archive/{names.run_dir}/resolver_review/") for p in out["drive_paths"])

    rc.copies.clear()
    again = da.backfill_resolver_review(spec, rclone=rc, work=tmp_path / "bf2", env={})
    assert again["status"] == "already_present" and rc.copies == []


def test_the_backfill_refuses_a_chain_not_in_drive_and_one_with_no_resolver(tmp_path):
    spec = _spec()
    names = da.ArchiveNames(**spec["names"])
    rc = FakeRclone(tmp_path / "drive")
    out = da.backfill_resolver_review(spec, rclone=rc, work=tmp_path / "bf", env={})
    assert out["status"] == "failed" and "not in Drive" in out["reason"]
    st = build_staging(tmp_path, with_resolver=False)
    run_dir = tmp_path / "drive" / "Fred_archive" / names.run_dir
    run_dir.mkdir(parents=True)
    _chain_zip(st, run_dir / names.zip_name)
    out = da.backfill_resolver_review(spec, rclone=rc, work=tmp_path / "bf2", env={})
    assert out["status"] == "no_resolver_update" and rc.copies == []
    assert not (run_dir / "resolver_review").exists()
