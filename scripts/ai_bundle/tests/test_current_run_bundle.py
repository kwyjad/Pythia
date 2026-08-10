# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Tests for the current-run bundle builder (the interpreter's input pack).

Fixture: two forecaster runs (fc_1 previous, fc_2 current) over epoch-shifted
questions matched on (iso3, hazard_code, metric), with forecast_deviation
rows from the Phase 1 tool's shape.
"""

from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb")

from scripts.ai_bundle.build_current_run_bundle import build_bundle

HS_PREV, HS_CUR = "hs_1", "hs_2"
FC_PREV, FC_CUR = "fc_1000", "fc_2000"

# Current epoch (2026-08).
CQ_ACE = "ETH_ACE_FATALITIES_2026-08"     # high deviation, matched to previous
CQ_FL = "ETH_FL_PA_2026-08"               # mid deviation
CQ_NOBASE = "KEN_FL_PA_2026-08"           # no forecast_deviation row
CQ_BIN = "PHL_TC_EVENT_OCCURRENCE_2026-08"  # binary, high RC disagreement
# Previous epoch (2026-07).
PQ_ACE = "ETH_ACE_FATALITIES_2026-07"     # matched to CQ_ACE
PQ_SDN = "SDN_ACE_FATALITIES_2026-07"     # previous top risk, exits


def _insert_spd(con, run_id, qid, model, k, top_heavy=False):
    for month in range(1, 7):
        for bucket in range(1, k + 1):
            if top_heavy:
                p = 0.6 if bucket == k else 0.4 / (k - 1)
            else:
                p = 1.0 / k
            con.execute(
                "INSERT INTO forecasts_raw (run_id, question_id, model_name, "
                "month_index, bucket_index, probability) VALUES (?, ?, ?, ?, ?, ?)",
                [run_id, qid, model, month, bucket, p],
            )


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
    rows = [
        (CQ_ACE, HS_CUR, "ETH", "ACE", "FATALITIES", "2027-01", "2026-08-01"),
        (CQ_FL, HS_CUR, "ETH", "FL", "PA", "2027-01", "2026-08-01"),
        (CQ_NOBASE, HS_CUR, "KEN", "FL", "PA", "2027-01", "2026-08-01"),
        (CQ_BIN, HS_CUR, "PHL", "TC", "EVENT_OCCURRENCE", "2027-01", "2026-08-01"),
        (PQ_ACE, HS_PREV, "ETH", "ACE", "FATALITIES", "2026-12", "2026-07-01"),
        (PQ_SDN, HS_PREV, "SDN", "ACE", "FATALITIES", "2026-12", "2026-07-01"),
    ]
    for qid, hs, iso3, hz, metric, target, start in rows:
        con.execute(
            "INSERT INTO questions (question_id, hs_run_id, iso3, hazard_code, "
            "metric, target_month, window_start_date, wording, status, track) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 'wording…', 'active', 1)",
            [qid, hs, iso3, hz, metric, target, start],
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
    _insert_spd(con, FC_CUR, CQ_ACE, "ensemble_mean_v2", 7, top_heavy=True)
    _insert_spd(con, FC_CUR, CQ_FL, "ensemble_mean_v2", 6)
    _insert_spd(con, FC_CUR, CQ_NOBASE, "ensemble_mean_v2", 6)
    for month in range(1, 7):
        for bucket, p in ((1, 0.35), (2, 0.65)):
            con.execute(
                "INSERT INTO forecasts_raw (run_id, question_id, model_name, "
                "month_index, bucket_index, probability) VALUES (?, ?, ?, ?, ?, ?)",
                [FC_CUR, CQ_BIN, "ensemble_mean_v2", month, bucket, p],
            )
    _insert_spd(con, FC_PREV, PQ_ACE, "ensemble_mean_v2", 7)  # uniform → moved
    _insert_spd(con, FC_PREV, PQ_SDN, "ensemble_mean_v2", 7)

    con.execute(
        """
        CREATE TABLE forecast_deviation (
            run_id TEXT, question_id TEXT, model_name TEXT, iso3 TEXT,
            hazard_code TEXT, metric TEXT, score_family TEXT,
            js_vs_baserate DOUBLE, log_ev_ratio DOUBLE, eiv_nominal DOUBLE,
            eiv_per_100k DOUBLE, baserate_source TEXT, baserate_json TEXT,
            is_test BOOLEAN DEFAULT FALSE, created_at TIMESTAMP DEFAULT now()
        )
        """
    )
    dev_rows = [
        (FC_CUR, CQ_ACE, "ensemble_mean_v2", "ETH", "ACE", "FATALITIES", "spd",
         0.50, 1.2, 200_000.0, 180.0, "acled_monthly_fatalities:6m",
         '{"probs": [0.4, 0.3, 0.15, 0.1, 0.03, 0.01, 0.01], "detail": {}}'),
        (FC_CUR, CQ_FL, "ensemble_mean_v2", "ETH", "FL", "PA", "spd",
         0.30, 0.5, 50_000.0, 40.0, "facts_resolved:pa+gdacs_occurrence", "{}"),
        (FC_CUR, CQ_BIN, "ensemble_mean_v2", "PHL", "TC", "EVENT_OCCURRENCE",
         "binary", 0.05, 0.1, None, None, "facts_resolved:event_occurrence", "{}"),
        (FC_PREV, PQ_ACE, "ensemble_mean_v2", "ETH", "ACE", "FATALITIES", "spd",
         0.20, 0.4, 90_000.0, 80.0, "acled_monthly_fatalities:6m", "{}"),
        (FC_PREV, PQ_SDN, "ensemble_mean_v2", "SDN", "ACE", "FATALITIES", "spd",
         0.60, 1.5, 400_000.0, 800.0, "acled_monthly_fatalities:6m", "{}"),
    ]
    for r in dev_rows:
        con.execute(
            "INSERT INTO forecast_deviation (run_id, question_id, model_name, "
            "iso3, hazard_code, metric, score_family, js_vs_baserate, "
            "log_ev_ratio, eiv_nominal, eiv_per_100k, baserate_source, "
            "baserate_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            list(r),
        )

    con.execute(
        """
        CREATE TABLE hs_triage (
            run_id TEXT, iso3 TEXT, hazard_code TEXT, tier TEXT,
            triage_score DOUBLE, need_full_spd BOOLEAN, drivers_json TEXT,
            data_quality_json TEXT, scenario_stub TEXT,
            regime_change_likelihood DOUBLE, regime_change_magnitude DOUBLE,
            regime_change_score DOUBLE, regime_change_level INTEGER,
            regime_change_direction TEXT, regime_change_window TEXT,
            regime_change_json TEXT, track INTEGER, is_test BOOLEAN DEFAULT FALSE
        )
        """
    )
    for iso3, hz, rc_score, rc_level in [
        ("ETH", "ACE", 0.24, 2), ("ETH", "FL", 0.0, 0),
        ("KEN", "FL", 0.0, 0), ("PHL", "TC", 0.60, 2),
    ]:
        con.execute(
            "INSERT INTO hs_triage (run_id, iso3, hazard_code, tier, triage_score, "
            "need_full_spd, regime_change_score, regime_change_level, track) "
            "VALUES (?, ?, ?, 'priority', 0.7, TRUE, ?, ?, 1)",
            [HS_CUR, iso3, hz, rc_score, rc_level],
        )

    con.execute(
        """
        CREATE TABLE sibyl_forecasts (
            sibyl_run_id TEXT, run_id TEXT, question_id TEXT, status TEXT,
            skip_reason TEXT, k INTEGER, aggregation TEXT,
            volatility_score DOUBLE, js_divergence_vs_standard DOUBLE,
            js_divergence_inter_trial DOUBLE, cost_usd DOUBLE,
            opus_cost_usd DOUBLE, brave_cost_usd DOUBLE, leakage_json TEXT,
            pooled_quantiles_json TEXT, trials_json TEXT,
            created_at TIMESTAMP DEFAULT now(), is_test BOOLEAN DEFAULT FALSE
        )
        """
    )
    con.execute(
        "INSERT INTO sibyl_forecasts (sibyl_run_id, run_id, question_id, status, k, "
        "aggregation, cost_usd) VALUES ('sib1', ?, ?, 'ok', 3, 'linear_pool', 4.0)",
        [FC_CUR, CQ_ACE],
    )

    con.execute(
        """
        CREATE TABLE scenarios (
            run_id TEXT, iso3 TEXT, hazard_code TEXT, metric TEXT,
            scenario_type TEXT, bucket_label TEXT, probability DOUBLE,
            text TEXT, created_at TIMESTAMP DEFAULT now(),
            is_test BOOLEAN DEFAULT FALSE
        )
        """
    )
    con.execute(
        "INSERT INTO scenarios (run_id, iso3, hazard_code, metric, scenario_type, "
        "bucket_label, probability, text) VALUES "
        f"('{FC_CUR}', 'ETH', 'ACE', 'FATALITIES', 'most_likely', '100-<500', 0.4, "
        "'escalation scenario text')"
    )

    con.close()
    return f"duckdb:///{db_path}"


def _extract(zip_path: Path, tmp_path: Path) -> Path:
    extract_dir = tmp_path / "extracted"
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    return extract_dir


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class TestCurrentRunBundle:
    def test_full_build(self, mini_db: str, tmp_path: Path) -> None:
        zip_path = build_bundle(mini_db, tmp_path / "out", top_n=2)
        assert zip_path is not None
        root = _extract(zip_path, tmp_path)

        for name in ("ANALYST_GUIDE.md", "manifest.json", "attention_index.csv",
                     "deltas.json", "blind_spots.json"):
            assert (root / name).exists(), name

        manifest = _read_json(root / "manifest.json")
        assert manifest["bundle_kind"] == "current_run_analysis"
        assert manifest["run_id"] == FC_CUR
        assert manifest["previous_run_id"] == FC_PREV
        assert manifest["n_questions"] == 4
        assert manifest["n_with_deviation"] == 3
        assert manifest["n_sibyl_covered"] == 1
        assert manifest["pack_tokens"] > 0
        assert manifest["truncated_question_ids"] == []
        assert manifest["window"]["start"].startswith("2026-08")

        # All four records present under the default budget.
        for qid in (CQ_ACE, CQ_FL, CQ_NOBASE, CQ_BIN):
            assert (root / "questions" / f"{qid}.json").exists(), qid

        record = _read_json(root / "questions" / f"{CQ_ACE}.json")
        assert record["forecast_run_id"] == FC_CUR
        assert record["deviation"]["ensemble_mean_v2"]["js_vs_baserate"] == 0.5
        assert record["baserate"]["probs"][0] == 0.4
        assert record["bucket_labels"]["1"] == "0"
        assert record["scenarios"][0]["scenario_type"] == "most_likely"
        # A current run has no outcome — the keys must be absent, not empty.
        assert "outcome" not in record
        assert "scores" not in record

        guide = (root / "ANALYST_GUIDE.md").read_text(encoding="utf-8")
        assert "js_vs_baserate" in guide
        assert "never" in guide.lower()

    def test_attention_ranks_and_floor(self, mini_db: str, tmp_path: Path) -> None:
        zip_path = build_bundle(
            mini_db, tmp_path / "out", top_n=2,
            per_capita_floor=100_000.0,  # only CQ_ACE (200k) clears it
        )
        root = _extract(zip_path, tmp_path)
        rows = {r["question_id"]: r for r in _read_csv(root / "attention_index.csv")}

        assert rows[CQ_ACE]["rank_deviation"] == "1"
        assert rows[CQ_FL]["rank_deviation"] == "2"
        assert rows[CQ_BIN]["rank_deviation"] == "3"
        assert rows[CQ_NOBASE]["rank_deviation"] == ""
        # Per-capita rank honors the absolute floor: CQ_FL (eiv 50k) excluded.
        assert rows[CQ_ACE]["rank_impact_per_capita"] == "1"
        assert rows[CQ_FL]["rank_impact_per_capita"] == ""
        # Binary question never enters the impact orderings.
        assert rows[CQ_BIN]["rank_impact_nominal"] == ""
        # RC-vs-deviation disagreement: PHL/TC (rc 0.60, tiny js) ranks first.
        assert rows[CQ_BIN]["rank_rc_disagreement"] == "1"
        # Blend rank present for every question with at least one ordering.
        assert rows[CQ_ACE]["attention_rank"] != ""
        assert rows[CQ_NOBASE]["attention_rank"] == ""

    def test_deltas(self, mini_db: str, tmp_path: Path) -> None:
        zip_path = build_bundle(mini_db, tmp_path / "out", top_n=2)
        root = _extract(zip_path, tmp_path)
        deltas = _read_json(root / "deltas.json")

        assert deltas["previous_run_id"] == FC_PREV
        entry_keys = {(e["iso3"], e["hazard_code"]) for e in deltas["attention_entries"]}
        exit_keys = {(e["iso3"], e["hazard_code"]) for e in deltas["attention_exits"]}
        # SDN was previous top-2 and has no current question → exit.
        assert ("SDN", "ACE") in exit_keys
        # SDN must never appear as an entry.
        assert ("SDN", "ACE") not in entry_keys

        tracking = {t["iso3"]: t for t in deltas["previous_flagged_tracking"]}
        assert tracking["SDN"]["still_in_current_run"] is False
        assert tracking["ETH"]["still_in_current_run"] is True
        assert tracking["ETH"]["current_js_vs_baserate"] == 0.5

        movements = deltas["largest_spd_movements"]
        if "movement_note" not in deltas:
            # ETH/ACE moved from uniform to top-heavy — must be reported.
            assert any(m["iso3"] == "ETH" and m["hazard_code"] == "ACE" for m in movements)
            assert all(m["js_current_vs_previous"] >= 0 for m in movements)

    def test_blind_spots(self, mini_db: str, tmp_path: Path) -> None:
        zip_path = build_bundle(mini_db, tmp_path / "out")
        root = _extract(zip_path, tmp_path)
        spots = _read_json(root / "blind_spots.json")
        no_base = {q["question_id"] for q in spots["no_baserate_questions"]}
        assert no_base == {CQ_NOBASE}
        assert spots["standing_caveats"]
        pairs = {(p["hazard_code"], p["metric"]) for p in spots["questions_by_pair"]}
        assert ("ACE", "FATALITIES") in pairs

    def test_token_budget_truncates_low_ranked_tail(
        self, mini_db: str, tmp_path: Path
    ) -> None:
        zip_path = build_bundle(mini_db, tmp_path / "out", max_pack_tokens=1000)
        root = _extract(zip_path, tmp_path)
        manifest = _read_json(root / "manifest.json")

        kept = manifest["n_question_records_kept"]
        truncated = manifest["truncated_question_ids"]
        assert kept >= 1, "the top-ranked question is never truncated"
        assert truncated, "a 1k-token budget must truncate something"
        assert kept + len(truncated) <= 4
        # Records are written in REPORT order, so the rows the report is
        # required to cover are the last thing a tight budget gives up. Under
        # a budget this small some of them do go, and when they do the
        # manifest says which: a categorised row without a record is the one
        # truncation that damages the report, so it is never just a count.
        rows = {r["question_id"]: r for r in _read_csv(root / "attention_index.csv")}
        categorised = [q for q, r in rows.items() if r["category"]]
        dropped = manifest["truncated_categorised_question_ids"]
        assert set(dropped) <= set(truncated)
        assert set(dropped) <= set(categorised)
        for qid in truncated:
            assert rows[qid]["record_path"] == ""
        # Whatever survived, it is a categorised row while any exist.
        if categorised:
            kept_ids = [q for q, r in rows.items() if r["record_path"]]
            assert all(q in categorised for q in kept_ids)

    def test_no_previous_run(self, mini_db: str, tmp_path: Path) -> None:
        # Point at the previous run itself: nothing earlier exists.
        zip_path = build_bundle(mini_db, tmp_path / "out", run_id=FC_PREV)
        root = _extract(zip_path, tmp_path)
        deltas = _read_json(root / "deltas.json")
        assert deltas["previous_run_id"] is None
        assert "first cycle" in deltas["note"]

    def test_empty_db_is_a_noop(self, tmp_path: Path) -> None:
        db = tmp_path / "empty.duckdb"
        duckdb.connect(str(db)).close()
        assert build_bundle(str(db), tmp_path / "out") is None
