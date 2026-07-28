# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Tests for the scored-forecast analysis bundle builder.

Fixture strategy mirrors pythia/tests/test_summarize_all_phases.py: a temp
DuckDB with hand-built mini tables. The builder must be defensive about
missing tables/columns (sparse or pre-Sibyl DBs).
"""

from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb")

from scripts.ai_bundle.build_scored_forecast_bundle import build_bundle

QID_SPD = "ETH_ACE_FATALITIES_2026-01"
QID_BIN = "ETH_FL_EVENT_OCCURRENCE_2026-01"
QID_UNSCORED = "KEN_FL_PA_2026-01"
QID_TEST = "SOM_ACE_FATALITIES_2026-01"
HS_RUN = "hs_run_1"
FC_RUN = "fc_run_1"

SPD_PROMPT = "FULL SPD PROMPT with all injects " + "x" * 500
RAW_RESPONSE = '{"spds": {...}} RAW MODEL RESPONSE'
GROUNDING_MD = "## Evidence\n" + ("long untruncated grounding evidence line\n" * 200)

# Prior-only trace: composite = 0.4*0.7 (no base rate) + 0.4*1.0 + 0.2*1.0 = 0.88
TRACE = {
    "prior": {"spd": [0.5, 0.2, 0.1, 0.1, 0.05, 0.03, 0.02], "rationale": "base rate anchored"},
    "updates": [],
    "point_estimate": "~40 fatalities",
    "rc_assessment": "accepted",
}


@pytest.fixture
def mini_db(tmp_path: Path) -> str:
    db_path = tmp_path / "pythia.duckdb"
    con = duckdb.connect(str(db_path))

    con.execute(
        """
        CREATE TABLE questions (
            question_id TEXT, hs_run_id TEXT, iso3 TEXT, hazard_code TEXT,
            metric TEXT, target_month TEXT, window_start_date DATE,
            window_end_date DATE, wording TEXT, status TEXT, track INTEGER,
            pythia_metadata_json TEXT, is_test BOOLEAN DEFAULT FALSE
        )
        """
    )
    con.execute(
        "INSERT INTO questions VALUES "
        f"('{QID_SPD}', '{HS_RUN}', 'ETH', 'ACE', 'FATALITIES', '2026-06', "
        "DATE '2026-01-01', DATE '2026-06-30', 'How many conflict fatalities…', "
        "'active', 1, NULL, FALSE),"
        f"('{QID_BIN}', '{HS_RUN}', 'ETH', 'FL', 'EVENT_OCCURRENCE', '2026-06', "
        "DATE '2026-01-01', DATE '2026-06-30', 'Will a GDACS Orange/Red flood…', "
        "'active', 1, NULL, FALSE),"
        f"('{QID_UNSCORED}', '{HS_RUN}', 'KEN', 'FL', 'PA', '2026-06', "
        "DATE '2026-01-01', DATE '2026-06-30', 'unscored question', 'active', 1, NULL, FALSE),"
        f"('{QID_TEST}', '{HS_RUN}', 'SOM', 'ACE', 'FATALITIES', '2026-06', "
        "DATE '2026-01-01', DATE '2026-06-30', 'test-mode question', 'active', 1, NULL, TRUE)"
    )

    con.execute(
        """
        CREATE TABLE scores (
            question_id TEXT, horizon_m INTEGER, metric TEXT, score_type TEXT,
            model_name TEXT, value DOUBLE, run_id TEXT,
            created_at TIMESTAMP DEFAULT now(), is_test BOOLEAN DEFAULT FALSE
        )
        """
    )
    con.execute(
        "INSERT INTO scores (question_id, horizon_m, metric, score_type, model_name, value, run_id) VALUES "
        # SPD question: ensemble + one member, two horizons.
        f"('{QID_SPD}', 1, 'FATALITIES', 'brier', 'ensemble_mean_v2', 0.30, '{FC_RUN}'),"
        f"('{QID_SPD}', 2, 'FATALITIES', 'brier', 'ensemble_mean_v2', 0.50, '{FC_RUN}'),"
        f"('{QID_SPD}', 1, 'FATALITIES', 'crps', 'ensemble_mean_v2', 0.10, '{FC_RUN}'),"
        f"('{QID_SPD}', 1, 'FATALITIES', 'brier', 'model-a', 1.20, '{FC_RUN}'),"
        # Binary question: mean ensemble only.
        f"('{QID_BIN}', 1, 'EVENT_OCCURRENCE', 'brier', 'ensemble_mean_v2', 0.04, '{FC_RUN}'),"
        # Test-mode question also has scores — must be excluded by default.
        f"('{QID_TEST}', 1, 'FATALITIES', 'brier', 'ensemble_mean_v2', 0.90, '{FC_RUN}')"
    )

    con.execute(
        """
        CREATE TABLE resolutions (
            question_id TEXT, horizon_m INTEGER, observed_month TEXT,
            value DOUBLE, source_snapshot_ym TEXT, source_desc TEXT,
            created_at TIMESTAMP DEFAULT now(), is_test BOOLEAN DEFAULT FALSE
        )
        """
    )
    con.execute(
        "INSERT INTO resolutions (question_id, horizon_m, observed_month, value, source_snapshot_ym, source_desc) VALUES "
        f"('{QID_SPD}', 1, '2026-01', 40.0, '2026-02', 'facts_resolved:ACLED:2026-02'),"
        f"('{QID_SPD}', 2, '2026-02', 700.0, '2026-03', 'facts_resolved:ACLED:2026-03'),"
        f"('{QID_BIN}', 1, '2026-01', 0.0, NULL, 'zero_default')"
    )

    con.execute(
        """
        CREATE TABLE hs_triage (
            run_id TEXT, iso3 TEXT, hazard_code TEXT, tier TEXT,
            triage_score DOUBLE, need_full_spd BOOLEAN, drivers_json TEXT,
            regime_shifts_json TEXT, data_quality_json TEXT, scenario_stub TEXT,
            regime_change_likelihood DOUBLE, regime_change_magnitude DOUBLE,
            regime_change_score DOUBLE, regime_change_level INTEGER,
            regime_change_direction TEXT, regime_change_window TEXT,
            regime_change_json TEXT, track INTEGER, is_test BOOLEAN DEFAULT FALSE
        )
        """
    )
    rc_json = json.dumps(
        {
            "rationale_bullets": ["escalating offensives", "peace talks collapsed"],
            "trigger_signals": ["troop build-up near border"],
        }
    )
    con.execute(
        "INSERT INTO hs_triage VALUES "
        f"('{HS_RUN}', 'ETH', 'ACE', 'priority', 0.8, TRUE, '[\"driver1\"]', NULL, "
        f"NULL, 'stub', 0.4, 0.6, 0.24, 2, 'up', '1-3m', '{rc_json}', 1, FALSE)"
    )

    con.execute(
        """
        CREATE TABLE hs_hazard_tail_packs (
            hs_run_id TEXT, iso3 TEXT, hazard_code TEXT, rc_level INTEGER,
            rc_score DOUBLE, rc_direction TEXT, rc_window TEXT, query TEXT,
            report_markdown TEXT, sources_json TEXT, grounded BOOLEAN,
            grounding_debug_json TEXT, structural_context TEXT,
            recent_signals_json TEXT, is_test BOOLEAN DEFAULT FALSE
        )
        """
    )
    con.execute(
        "INSERT INTO hs_hazard_tail_packs (hs_run_id, iso3, hazard_code, query, report_markdown, sources_json, grounded) VALUES "
        f"('{HS_RUN}', 'ETH', 'ACE', 'rc_grounding: Ethiopia conflict', ?, "
        "'[\"http://example.com/a\"]', TRUE),"
        f"('{HS_RUN}', 'ETH', 'ACE', 'triage_grounding: Ethiopia situation', 'triage pack', "
        "'[]', TRUE)",
        [GROUNDING_MD],
    )

    con.execute(
        """
        CREATE TABLE hs_adversarial_checks (
            hs_run_id TEXT, iso3 TEXT, hazard_code TEXT, rc_level INTEGER,
            net_assessment TEXT, summary TEXT, payload_json TEXT,
            sources_json TEXT, grounded BOOLEAN, model_id TEXT,
            is_test BOOLEAN DEFAULT FALSE
        )
        """
    )
    con.execute(
        "INSERT INTO hs_adversarial_checks VALUES "
        f"('{HS_RUN}', 'ETH', 'ACE', 2, 'partially_supported', 'counter-evidence exists', "
        "'{\"counter_evidence\": []}', '[]', TRUE, 'gemini-3.5-flash', FALSE)"
    )

    con.execute(
        """
        CREATE TABLE llm_calls (
            call_id TEXT, run_id TEXT, hs_run_id TEXT, question_id TEXT,
            call_type TEXT, phase TEXT, model_name TEXT, provider TEXT,
            model_id TEXT, prompt_text TEXT, response_text TEXT,
            error_text TEXT, timestamp TIMESTAMP DEFAULT now(),
            is_test BOOLEAN DEFAULT FALSE
        )
        """
    )
    con.execute(
        "INSERT INTO llm_calls (call_id, run_id, question_id, call_type, phase, model_name, provider, model_id, prompt_text, response_text) VALUES "
        f"('c1', '{FC_RUN}', '{QID_SPD}', 'spd_v2', 'spd_v2', 'model-a', 'google', "
        "'model-a-id', ?, ?),"
        f"('c2', '{FC_RUN}', '{QID_BIN}', 'binary_v2', 'binary_v2', 'model-a', 'google', "
        "'model-a-id', 'BINARY PROMPT', 'binary raw')",
        [SPD_PROMPT, RAW_RESPONSE],
    )

    con.execute(
        """
        CREATE TABLE forecasts_raw (
            run_id TEXT, question_id TEXT, model_name TEXT, month_index INTEGER,
            bucket_index INTEGER, probability DOUBLE, ok BOOLEAN,
            elapsed_ms INTEGER, cost_usd DOUBLE, prompt_tokens INTEGER,
            completion_tokens INTEGER, total_tokens INTEGER, status TEXT,
            spd_json TEXT, human_explanation TEXT, reasoning_trace_json TEXT,
            is_test BOOLEAN DEFAULT FALSE
        )
        """
    )
    trace_json = json.dumps(TRACE)
    # Member rows duplicated across (month, bucket) as in production.
    for month in (1, 2):
        for bucket in range(1, 8):
            con.execute(
                "INSERT INTO forecasts_raw (run_id, question_id, model_name, month_index, "
                "bucket_index, probability, ok, status, spd_json, human_explanation, "
                "reasoning_trace_json, cost_usd, total_tokens) VALUES "
                f"('{FC_RUN}', '{QID_SPD}', 'model-a', {month}, {bucket}, 0.14, TRUE, 'ok', "
                "'{\"spds\": {}}', 'the model explains itself', ?, 0.01, 1000)",
                [trace_json],
            )

    con.execute(
        """
        CREATE TABLE forecasts_ensemble (
            run_id TEXT, question_id TEXT, iso3 TEXT, hazard_code TEXT,
            metric TEXT, model_name TEXT, month_index INTEGER,
            bucket_index INTEGER, probability DOUBLE, ev_value DOUBLE,
            weights_profile TEXT, created_at TIMESTAMP DEFAULT now(),
            status TEXT, human_explanation TEXT, reasoning_trace_json TEXT,
            is_test BOOLEAN DEFAULT FALSE
        )
        """
    )
    # FATALITIES has 7 buckets; realized 40.0 → bucket 3 (25–<100), 700 → bucket 6.
    probs_h1 = [0.05, 0.15, 0.40, 0.20, 0.10, 0.05, 0.05]
    probs_h2 = [0.05, 0.15, 0.40, 0.20, 0.10, 0.05, 0.05]
    for month, probs in ((1, probs_h1), (2, probs_h2)):
        for bucket, p in enumerate(probs, start=1):
            con.execute(
                "INSERT INTO forecasts_ensemble (run_id, question_id, iso3, hazard_code, "
                "metric, model_name, month_index, bucket_index, probability, ev_value) VALUES "
                f"('{FC_RUN}', '{QID_SPD}', 'ETH', 'ACE', 'FATALITIES', 'ensemble_mean_v2', "
                f"{month}, {bucket}, {p}, 55.0)"
            )
    # Binary: bucket_1 = P(yes).
    con.execute(
        "INSERT INTO forecasts_ensemble (run_id, question_id, iso3, hazard_code, metric, "
        "model_name, month_index, bucket_index, probability) VALUES "
        f"('{FC_RUN}', '{QID_BIN}', 'ETH', 'FL', 'EVENT_OCCURRENCE', 'ensemble_mean_v2', 1, 1, 0.2),"
        f"('{FC_RUN}', '{QID_BIN}', 'ETH', 'FL', 'EVENT_OCCURRENCE', 'ensemble_mean_v2', 1, 2, 0.8)"
    )

    con.execute(
        """
        CREATE TABLE calibration_weights (
            as_of_month TEXT, hazard_code TEXT, metric TEXT, model_name TEXT,
            weight DOUBLE, n_questions INTEGER, n_samples INTEGER,
            avg_brier DOUBLE, avg_log DOUBLE, avg_crps DOUBLE,
            train_cutoff_month TEXT, created_at TIMESTAMP DEFAULT now()
        )
        """
    )
    con.execute(
        "INSERT INTO calibration_weights (as_of_month, hazard_code, metric, model_name, weight, n_questions, avg_brier) VALUES "
        "('2026-05', 'ACE', 'FATALITIES', 'model-a', 0.90, 10, 0.6),"
        "('2026-06', 'ACE', 'FATALITIES', 'model-a', 1.10, 12, 0.5)"
    )

    con.execute(
        """
        CREATE TABLE calibration_advice (
            as_of_month TEXT, hazard_code TEXT, metric TEXT,
            model_name TEXT DEFAULT '__shared__', advice TEXT,
            findings_json TEXT, advice_version TEXT, created_at TIMESTAMP DEFAULT now()
        )
        """
    )
    con.execute(
        "INSERT INTO calibration_advice (as_of_month, hazard_code, metric, model_name, advice) VALUES "
        "('2026-06', 'ACE', 'FATALITIES', '__shared__', 'Anchor priors closer to the trailing base rate.')"
    )

    con.execute(
        """
        CREATE TABLE sibyl_forecasts (
            sibyl_run_id TEXT, run_id TEXT, question_id TEXT, iso3 TEXT,
            hazard_code TEXT, metric TEXT, track TEXT, status TEXT,
            skip_reason TEXT, k INTEGER, aggregation TEXT,
            volatility_score DOUBLE, triage_score DOUBLE,
            pooled_quantiles_json TEXT, trials_json TEXT, bucket_probs_json TEXT,
            js_divergence_vs_standard DOUBLE, js_divergence_inter_trial DOUBLE,
            cost_usd DOUBLE, opus_cost_usd DOUBLE, brave_cost_usd DOUBLE,
            leakage_json TEXT, created_at TIMESTAMP DEFAULT now(),
            is_test BOOLEAN DEFAULT FALSE
        )
        """
    )
    trials = json.dumps([{"trial": 1, "belief_trace": [{"step": 1, "belief": "initial"}]}])
    con.execute(
        "INSERT INTO sibyl_forecasts (sibyl_run_id, run_id, question_id, status, k, "
        "aggregation, js_divergence_vs_standard, cost_usd, trials_json) VALUES "
        f"('sib1', '{FC_RUN}', '{QID_SPD}', 'ok', 3, 'linear_pool', 0.12, 4.2, ?)",
        [trials],
    )

    con.close()
    return f"duckdb:///{db_path}"


@pytest.fixture
def built_bundle(mini_db: str, tmp_path: Path) -> Path:
    out_dir = tmp_path / "out"
    zip_path = build_bundle(mini_db, out_dir, months_back=0, n_case_studies=2)
    assert zip_path is not None and zip_path.exists()
    extract_dir = tmp_path / "extracted"
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    return extract_dir


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_zip_structure(built_bundle: Path) -> None:
    for name in (
        "ANALYST_GUIDE.md",
        "manifest.json",
        "digest.md",
        "questions_index.csv",
        "scores_flat.csv",
        "forecast_vs_outcome.csv",
        "rollups.csv",
        "calibration_weights.csv",
        "calibration_advice.md",
        "briefing/01_run_briefing.md",
        "briefing/02_case_studies.md",
        f"questions/{QID_SPD}.json",
        f"questions/{QID_BIN}.json",
    ):
        assert (built_bundle / name).exists(), f"missing {name}"


def test_only_scored_non_test_questions(built_bundle: Path) -> None:
    q_dir = built_bundle / "questions"
    names = {p.name for p in q_dir.glob("*.json")}
    assert f"{QID_SPD}.json" in names
    assert f"{QID_BIN}.json" in names
    assert f"{QID_UNSCORED}.json" not in names, "unscored question must be excluded"
    assert f"{QID_TEST}.json" not in names, "is_test question must be excluded by default"


def test_record_carries_exact_prompt_and_raw_response(built_bundle: Path) -> None:
    record = _read_json(built_bundle / "questions" / f"{QID_SPD}.json")
    assert record["spd_prompt"] == SPD_PROMPT
    members = {m["model_name"]: m for m in record["members"]}
    assert members["model-a"]["response_text"] == RAW_RESPONSE
    assert members["model-a"]["human_explanation"] == "the model explains itself"


def test_grounding_untruncated_and_rc_rationale(built_bundle: Path) -> None:
    record = _read_json(built_bundle / "questions" / f"{QID_SPD}.json")
    packs = {g["kind"]: g for g in record["grounding"]}
    assert packs["rc_grounding"]["report_markdown"] == GROUNDING_MD
    assert record["regime_change"]["rationale_bullets"] == [
        "escalating offensives",
        "peace talks collapsed",
    ]
    assert record["adversarial"]["net_assessment"] == "partially_supported"


def test_trace_quality_recomputed(built_bundle: Path) -> None:
    record = _read_json(built_bundle / "questions" / f"{QID_SPD}.json")
    member = record["members"][0]
    tq = member["trace_quality"]
    # prior 0.7 (no base rate at bundle time) + delta 1.0 + magnitude 1.0
    assert tq["trace_quality_score"] == pytest.approx(0.88)
    assert tq["has_trace"] is True


def test_outcome_and_unresolved_horizons(built_bundle: Path) -> None:
    record = _read_json(built_bundle / "questions" / f"{QID_SPD}.json")
    outcome = record["outcome"]
    assert len(outcome["resolutions"]) == 2
    assert outcome["unresolved_horizons"] == [3, 4, 5, 6]
    by_h = {r["horizon_m"]: r for r in outcome["resolutions"]}
    # FATALITIES buckets: 0, 1-<5, 5-<25, 25-<100, 100-<500, 500-<1000, >=1000
    assert by_h[1]["realized_bucket"] == 4  # 40 → 25-<100
    assert by_h[2]["realized_bucket"] == 6  # 700 → 500-<1000
    assert by_h[1]["source_desc"] == "facts_resolved:ACLED:2026-02"


def test_binary_realized_bucket(built_bundle: Path) -> None:
    record = _read_json(built_bundle / "questions" / f"{QID_BIN}.json")
    res = record["outcome"]["resolutions"][0]
    assert res["realized_bucket"] == 2  # value 0.0 → "no event" bucket


def test_rollups_never_mix_score_families(built_bundle: Path) -> None:
    with (built_bundle / "rollups.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        expected = "binary" if row["metric"] == "EVENT_OCCURRENCE" else "spd"
        assert row["score_family"] == expected
    families = {(r["score_family"], r["metric"]) for r in rows}
    assert ("binary", "EVENT_OCCURRENCE") in families
    assert ("spd", "FATALITIES") in families


def test_case_studies_inline_reasoning_and_sibyl_trials(built_bundle: Path) -> None:
    case_files = list((built_bundle / "case_studies").glob("*.json"))
    assert case_files, "case studies must exist"
    spd_case = built_bundle / "case_studies" / f"{QID_SPD}.json"
    assert spd_case.exists()
    record = _read_json(spd_case)
    assert record["members"][0]["reasoning_trace"]["prior"]["rationale"] == "base rate anchored"
    assert record["scores"], "case study must inline scores"
    assert record["sibyl"]["trials"], "case study must inline sibyl trials"


def test_weight_movement_computed(built_bundle: Path) -> None:
    with (built_bundle / "calibration_weights.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    row = next(r for r in rows if r["model_name"] == "model-a")
    assert row["as_of_month"] == "2026-06"
    assert row["previous_as_of_month"] == "2026-05"
    assert float(row["weight_delta"]) == pytest.approx(0.2)


def test_forecast_vs_outcome_p_realized(built_bundle: Path) -> None:
    with (built_bundle / "forecast_vs_outcome.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    spd_h1 = next(
        r for r in rows if r["question_id"] == QID_SPD and r["horizon_m"] == "1"
    )
    assert spd_h1["realized_bucket"] == "4"
    assert float(spd_h1["p_realized_bucket"]) == pytest.approx(0.20)
    assert spd_h1["modal_bucket"] == "3"


def test_digest_under_budget_and_guide_buckets(built_bundle: Path) -> None:
    digest = built_bundle / "digest.md"
    assert digest.stat().st_size < 200 * 1024
    guide = (built_bundle / "ANALYST_GUIDE.md").read_text(encoding="utf-8")
    # Drift trap: the guide's bucket table must come from pythia.buckets.
    from pythia.buckets import labels_for

    for label in labels_for("FATALITIES"):
        assert label in guide, f"guide missing FATALITIES bucket label {label}"
    assert "never average" in guide.lower() or "never blend" in guide.lower()


def test_include_test_flag(mini_db: str, tmp_path: Path) -> None:
    out_dir = tmp_path / "out_test"
    zip_path = build_bundle(
        mini_db, out_dir, months_back=0, n_case_studies=2, include_test=True
    )
    assert zip_path is not None
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    assert f"questions/{QID_TEST}.json" in names


def test_sparse_db_degrades_gracefully(tmp_path: Path) -> None:
    """A DB with only questions+scores (no HS/sibyl/calibration tables) works."""
    db_path = tmp_path / "sparse.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute(
        "CREATE TABLE questions (question_id TEXT, hs_run_id TEXT, iso3 TEXT, "
        "hazard_code TEXT, metric TEXT, target_month TEXT, window_start_date DATE, "
        "window_end_date DATE, wording TEXT, status TEXT, track INTEGER, "
        "pythia_metadata_json TEXT)"
    )
    con.execute(
        "INSERT INTO questions VALUES ('Q1', 'hs1', 'ETH', 'ACE', 'FATALITIES', "
        "'2026-06', DATE '2026-01-01', DATE '2026-06-30', 'w', 'active', 1, NULL)"
    )
    con.execute(
        "CREATE TABLE scores (question_id TEXT, horizon_m INTEGER, metric TEXT, "
        "score_type TEXT, model_name TEXT, value DOUBLE)"
    )
    con.execute(
        "INSERT INTO scores VALUES ('Q1', 1, 'FATALITIES', 'brier', 'ensemble_mean_v2', 0.5)"
    )
    # resolutions WITHOUT source_desc (pre-migration shape).
    con.execute(
        "CREATE TABLE resolutions (question_id TEXT, horizon_m INTEGER, "
        "observed_month TEXT, value DOUBLE, source_snapshot_ym TEXT)"
    )
    con.execute("INSERT INTO resolutions VALUES ('Q1', 1, '2026-01', 10.0, '2026-02')")
    con.close()

    out_dir = tmp_path / "out_sparse"
    zip_path = build_bundle(f"duckdb:///{db_path}", out_dir, months_back=0)
    assert zip_path is not None and zip_path.exists()
    with zipfile.ZipFile(zip_path) as zf:
        record = json.loads(zf.read("questions/Q1.json"))
    assert record["outcome"]["resolutions"][0]["value"] == 10.0
    assert "grounding" not in record


def test_empty_db_returns_none(tmp_path: Path) -> None:
    db_path = tmp_path / "empty.duckdb"
    duckdb.connect(str(db_path)).close()
    assert build_bundle(f"duckdb:///{db_path}", tmp_path / "out_empty") is None
