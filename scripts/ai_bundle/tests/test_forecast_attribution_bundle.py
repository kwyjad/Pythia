# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Tests for the forecast attribution bundle builder.

Fixture: one forecaster run (fc_2000, current) and one before it (fc_1000)
over epoch-shifted questions, three ensemble members on the ACE question
(one with a full trace, one with a trace whose arithmetic fails, one with
no trace at all), a Track-2 flood question, a Sibyl row, a binary question
with no trace, forecast_deviation rows carrying a base-rate anchor, an
hs_triage RC flag, an SPD prompt in llm_calls and an HS grounding pack.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import zipfile
from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb")

from scripts.ai_bundle import build_forecast_attribution_bundle as fab

HS_PREV, HS_CUR = "hs_1", "hs_2"
FC_PREV, FC_CUR = "fc_1000", "fc_2000"
Q_ACE = "ETH_ACE_FATALITIES_2026-08"
Q_FL = "ETH_FL_PA_2026-08"
Q_BIN = "PHL_TC_EVENT_OCCURRENCE_2026-08"
PQ_ACE = "ETH_ACE_FATALITIES_2026-07"

K_FAT = 7
PRIOR_FAT = [0.30, 0.25, 0.20, 0.15, 0.06, 0.03, 0.01]
DELTA_1 = [-0.10, -0.05, 0.00, 0.05, 0.05, 0.03, 0.02]
POST_1 = [round(p + d, 4) for p, d in zip(PRIOR_FAT, DELTA_1)]
DELTA_2 = [0.02, 0.02, 0.01, -0.02, -0.02, -0.01, 0.00]
POST_2 = [round(p + d, 4) for p, d in zip(POST_1, DELTA_2)]

GOOD_TRACE = {
    "prior": {"spd": PRIOR_FAT, "rationale": "Anchored on the ACLED base rate for the trailing 36 months."},
    "updates": [
        {
            "signal": "Escalation of clashes and airstrikes around Mekelle reported by OCHA",
            "direction": "UP", "magnitude": "MODERATE", "months_affected": "all",
            "delta": DELTA_1, "post_update_spd": POST_1,
        },
        {
            "signal": "HS regime change flag partially accepted; ceasefire talks resumed",
            "direction": "DOWN", "magnitude": "SMALL", "months_affected": "1-2",
            "delta": DELTA_2, "post_update_spd": POST_2,
        },
    ],
    "point_estimate": "~120 fatalities",
    "point_estimate_bucket": 4,
    "rc_assessment": "partially_accepted",
}

# Delta does not sum to zero and the post SPD does not reconcile.
BAD_TRACE = {
    "prior": {"spd": PRIOR_FAT, "rationale": "Base rate."},
    "updates": [
        {
            "signal": "Funding shortfall and ration cuts announced by WFP",
            "direction": "UP", "magnitude": "LARGE",
            "delta": [0.30, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "post_update_spd": PRIOR_FAT,
        }
    ],
    "rc_assessment": "rebutted",
}

TRACK2_TRACE = {"prior": {"spd": [0.5, 0.3, 0.1, 0.05, 0.03, 0.02], "rationale": "Seasonal profile."},
                "updates": [], "rc_assessment": "accepted"}


def _insert_spd(con, run_id, qid, model, k, trace=None, top_heavy=False):
    trace_json = json.dumps(trace) if trace is not None else None
    for month in range(1, 7):
        for bucket in range(1, k + 1):
            p = (0.6 if bucket == k else 0.4 / (k - 1)) if top_heavy else 1.0 / k
            con.execute(
                "INSERT INTO forecasts_raw (run_id, question_id, model_name, month_index, "
                "bucket_index, probability, reasoning_trace_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                [run_id, qid, model, month, bucket, p, trace_json],
            )


@pytest.fixture
def mini_db(tmp_path: Path) -> str:
    db_path = tmp_path / "pythia.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute(
        """
        CREATE TABLE questions (
            question_id TEXT, hs_run_id TEXT, iso3 TEXT, hazard_code TEXT, metric TEXT,
            target_month TEXT, window_start_date DATE, window_end_date DATE, wording TEXT,
            status TEXT, track INTEGER, pythia_metadata_json TEXT, is_test BOOLEAN DEFAULT FALSE
        )
        """
    )
    rows = [
        (Q_ACE, HS_CUR, "ETH", "ACE", "FATALITIES", "2027-01", "2026-08-01", 1),
        (Q_FL, HS_CUR, "ETH", "FL", "PA", "2027-01", "2026-08-01", 2),
        (Q_BIN, HS_CUR, "PHL", "TC", "EVENT_OCCURRENCE", "2027-01", "2026-08-01", 1),
        (PQ_ACE, HS_PREV, "ETH", "ACE", "FATALITIES", "2026-12", "2026-07-01", 1),
    ]
    for qid, hs, iso3, hz, metric, target, start, track in rows:
        con.execute(
            "INSERT INTO questions (question_id, hs_run_id, iso3, hazard_code, metric, "
            "target_month, window_start_date, wording, status, track) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 'wording', 'active', ?)",
            [qid, hs, iso3, hz, metric, target, start, track],
        )
    con.execute(
        "INSERT INTO questions (question_id, hs_run_id, iso3, hazard_code, metric, target_month, "
        "window_start_date, wording, status, track, is_test) VALUES "
        "('SOM_ACE_PA_2026-08', 'hs_test', 'SOM', 'ACE', 'PA', '2027-01', '2026-08-01', 'w', 'active', 1, TRUE)"
    )

    con.execute(
        """
        CREATE TABLE forecasts_raw (
            run_id TEXT, question_id TEXT, model_name TEXT, month_index INTEGER,
            bucket_index INTEGER, probability DOUBLE, ok BOOLEAN, elapsed_ms INTEGER,
            cost_usd DOUBLE, prompt_tokens INTEGER, completion_tokens INTEGER,
            total_tokens INTEGER, status TEXT, spd_json TEXT, human_explanation TEXT,
            reasoning_trace_json TEXT, is_test BOOLEAN DEFAULT FALSE
        )
        """
    )
    _insert_spd(con, FC_CUR, Q_ACE, "gpt-5.6-sol", K_FAT, GOOD_TRACE, top_heavy=True)
    _insert_spd(con, FC_CUR, Q_ACE, "claude-opus-5", K_FAT, BAD_TRACE)
    _insert_spd(con, FC_CUR, Q_ACE, "gemini-3.5-flash", K_FAT, None)
    _insert_spd(con, FC_CUR, Q_ACE, "ensemble_mean_v2", K_FAT, None, top_heavy=True)
    _insert_spd(con, FC_CUR, Q_ACE, "sibyl", K_FAT, None)
    _insert_spd(con, FC_CUR, Q_FL, "track2_flash", 6, TRACK2_TRACE)
    for month in range(1, 7):
        for bucket, p in ((1, 0.35), (2, 0.65)):
            con.execute(
                "INSERT INTO forecasts_raw (run_id, question_id, model_name, month_index, "
                "bucket_index, probability) VALUES (?, ?, ?, ?, ?, ?)",
                [FC_CUR, Q_BIN, "gpt-5.6-sol", month, bucket, p],
            )
    _insert_spd(con, FC_PREV, PQ_ACE, "gpt-5.6-sol", K_FAT, GOOD_TRACE)
    _insert_spd(con, FC_PREV, PQ_ACE, "ensemble_mean_v2", K_FAT, None)
    _insert_spd(con, "fc_9999", "SOM_ACE_PA_2026-08", "gpt-5.6-sol", 6, None)  # test run

    con.execute(
        """
        CREATE TABLE forecasts_ensemble (
            run_id TEXT, question_id TEXT, iso3 TEXT, hazard_code TEXT, metric TEXT,
            model_name TEXT, month_index INTEGER, bucket_index INTEGER, probability DOUBLE,
            ev_value DOUBLE, weights_profile TEXT, created_at TIMESTAMP, status TEXT
        )
        """
    )
    con.execute(
        "INSERT INTO forecasts_ensemble VALUES (?, ?, 'ETH', 'ACE', 'FATALITIES', 'ensemble_mean_v2', "
        "1, 1, 0.1, 10.0, 'default', TIMESTAMP '2026-08-01 05:00:00', 'ok')",
        [FC_CUR, Q_ACE],
    )

    con.execute(
        """
        CREATE TABLE forecast_deviation (
            run_id TEXT, question_id TEXT, model_name TEXT, iso3 TEXT, hazard_code TEXT,
            metric TEXT, score_family TEXT, js_vs_baserate DOUBLE, log_ev_ratio DOUBLE,
            eiv_nominal DOUBLE, eiv_per_100k DOUBLE, baserate_source TEXT, baserate_json TEXT,
            baserate_n_obs INTEGER, is_test BOOLEAN DEFAULT FALSE, created_at TIMESTAMP DEFAULT now()
        )
        """
    )
    anchor = {"probs": [0.4, 0.3, 0.15, 0.1, 0.03, 0.01, 0.01], "detail": {"n_months_used": 36}}
    con.execute(
        "INSERT INTO forecast_deviation (run_id, question_id, model_name, iso3, hazard_code, "
        "metric, score_family, js_vs_baserate, log_ev_ratio, eiv_nominal, eiv_per_100k, "
        "baserate_source, baserate_json, baserate_n_obs) VALUES (?, ?, 'ensemble_mean_v2', 'ETH', "
        "'ACE', 'FATALITIES', 'spd', 0.5, 1.2, 200000.0, 180.0, 'acled_monthly_fatalities:36m', ?, 36)",
        [FC_CUR, Q_ACE, json.dumps(anchor)],
    )

    con.execute(
        """
        CREATE TABLE hs_triage (
            run_id TEXT, iso3 TEXT, hazard_code TEXT, tier TEXT, triage_score DOUBLE,
            need_full_spd BOOLEAN, drivers_json TEXT, data_quality_json TEXT, scenario_stub TEXT,
            regime_change_likelihood DOUBLE, regime_change_magnitude DOUBLE,
            regime_change_score DOUBLE, regime_change_level INTEGER,
            regime_change_direction TEXT, regime_change_window TEXT, regime_change_json TEXT,
            track INTEGER, is_test BOOLEAN DEFAULT FALSE
        )
        """
    )
    con.execute(
        "INSERT INTO hs_triage (run_id, iso3, hazard_code, tier, triage_score, need_full_spd, "
        "regime_change_likelihood, regime_change_magnitude, regime_change_score, "
        "regime_change_level, regime_change_direction, track) VALUES "
        "(?, 'ETH', 'ACE', 'priority', 0.8, TRUE, 0.4, 0.6, 0.24, 2, 'UP', 1)",
        [HS_CUR],
    )
    con.execute(
        """
        CREATE TABLE hs_hazard_tail_packs (
            hs_run_id TEXT, iso3 TEXT, hazard_code TEXT, rc_level INTEGER, rc_score DOUBLE,
            rc_direction TEXT, rc_window TEXT, query TEXT, report_markdown TEXT,
            sources_json TEXT, grounded BOOLEAN, grounding_debug_json TEXT,
            structural_context TEXT, recent_signals_json TEXT, created_at TIMESTAMP
        )
        """
    )
    sources = [
        {"title": "Airstrikes and clashes reported around Mekelle", "url": "https://example.org/mekelle",
         "snippet": "OCHA reported airstrikes and clashes near Mekelle this week.", "published": "2026-07-25"},
        {"title": "Coffee prices rise", "url": "https://example.org/coffee", "snippet": "Unrelated.",
         "published": "2026-06-01"},
    ]
    con.execute(
        "INSERT INTO hs_hazard_tail_packs (hs_run_id, iso3, hazard_code, query, sources_json, grounded) "
        "VALUES (?, 'ETH', 'ACE', 'rc_grounding: ETH ACE', ?, TRUE)",
        [HS_CUR, json.dumps(sources)],
    )

    con.execute(
        """
        CREATE TABLE llm_calls (
            call_id TEXT, run_id TEXT, hs_run_id TEXT, question_id TEXT, call_type TEXT, phase TEXT,
            model_name TEXT, provider TEXT, model_id TEXT, prompt_text TEXT, response_text TEXT,
            error_text TEXT, cost_usd DOUBLE, timestamp TIMESTAMP, is_test BOOLEAN DEFAULT FALSE
        )
        """
    )
    prompt = (
        "You are a forecaster.\n\nQUESTION DATA:\n- iso3: ETH\n\n"
        "REGIME CHANGE GUIDANCE (RC):\n- RC level: 2\n\n"
        "HS GROUNDING EVIDENCE:\nAirstrikes near Mekelle.\n\n"
        "STEP 1 — DECLARE YOUR PRIOR SPD\nState the prior.\n\n"
        "Output instructions:\n- Return JSON.\n"
    )
    con.execute(
        "INSERT INTO llm_calls (call_id, run_id, question_id, phase, model_name, prompt_text, "
        "cost_usd, timestamp) VALUES ('c1', ?, ?, 'spd_v2', 'gpt-5.6-sol', ?, 0.1, now())",
        [FC_CUR, Q_ACE, prompt],
    )
    con.execute(
        """
        CREATE TABLE crisiswatch_entries (
            iso3 TEXT, month INTEGER, year INTEGER, arrow TEXT, alert_type TEXT, summary TEXT,
            country_name TEXT, fetched_at TIMESTAMP
        )
        """
    )
    con.execute("INSERT INTO crisiswatch_entries VALUES ('ETH', 7, 2026, 'deteriorated', NULL, 's', 'Ethiopia', now())")
    con.close()
    return f"duckdb:///{db_path}"


def _read_csv_from_zip(zf: zipfile.ZipFile, name: str) -> list[dict]:
    with zf.open(name) as fh:
        return list(csv.DictReader(line.decode("utf-8") for line in fh))


class TestAttributionId:
    def test_recipe_is_pinned(self):
        # sha256("{run_id}|{question_id}|{model_name}|{update_index}")[:16].
        expected = hashlib.sha256(b"fc_2000|ETH_ACE_FATALITIES_2026-08|gpt-5.6-sol|0").hexdigest()[:16]
        assert fab.attribution_id("fc_2000", "ETH_ACE_FATALITIES_2026-08", "gpt-5.6-sol", 0) == expected
        assert len(expected) == 16

    def test_hardcoded_values(self):
        # Literal pins: a later refactor must not silently break the join
        # with the future resolutions bundle, which attaches outcomes to
        # this id. If these change, the recipe changed.
        assert fab.attribution_id("r", "q", "m", 0) == "bb272840bc1316e1"
        assert fab.attribution_id("fc_2000", "ETH_ACE_FATALITIES_2026-08", "gpt-5.6-sol", 0) == "2fdac7bfbd0a8cd7"
        assert fab.attribution_id("r", "q", "m", -1) != fab.attribution_id("r", "q", "m", 0)

    def test_evidence_id_recipe(self):
        assert fab.evidence_id("https://x", "t") == hashlib.sha256(b"https://x|t").hexdigest()[:16]
        assert fab.evidence_id(None, None) == hashlib.sha256(b"|").hexdigest()[:16]


class TestTaxonomy:
    def test_known_signal_classifies(self):
        tax = fab.load_taxonomy()
        klass, conf = fab.classify_signal("Escalation of clashes and airstrikes around Mekelle", tax)
        assert klass == "conflict_escalation"
        assert 0.5 <= conf <= 0.95
        assert fab.classify_signal("Ceasefire talks resumed in Nairobi", tax)[0] == "conflict_deescalation"
        assert fab.classify_signal("HS regime change flag", tax)[0] == "rc_flag"
        assert fab.classify_signal("IPC Phase 3+ population rose", tax)[0] == "food_security_phase"
        assert fab.classify_signal("", tax) == ("other", 0.0)
        assert fab.classify_signal("Nothing in particular.", tax) == ("other", 0.1)

    def test_priority_wins_over_order(self):
        tax = fab.load_taxonomy()
        # Mentions both a ceasefire (deescalation, 57) and the RC flag (100).
        assert fab.classify_signal("regime change flag despite ceasefire", tax)[0] == "rc_flag"

    def test_taxonomy_file_has_seed_classes(self):
        classes = {e["signal_class"] for e in fab.load_taxonomy()}
        for c in ("conflict_escalation", "conflict_deescalation", "political_transition",
                  "displacement", "humanitarian_access", "funding_shortfall", "seasonal_climate",
                  "meteorological_forecast", "hydrological_observation", "food_security_phase",
                  "epidemic", "economic_shock", "base_rate_reference", "data_gap", "rc_flag",
                  "prediction_market", "media_report_unspecified", "other", "no_trace"):
            assert c in classes


class TestPromptSections:
    def test_headings_split_and_remainder_is_unclassified(self):
        prompt = "preamble text\n\nREGIME CHANGE GUIDANCE (RC):\n- x\n\nSTEP 1 — DECLARE YOUR PRIOR SPD\nbody\n\nOutput instructions:\n- y\n"
        names = [n for n, _ in fab.parse_prompt_sections(prompt)]
        assert names[0] == "unclassified"
        assert "REGIME CHANGE GUIDANCE (RC)" in names
        assert any(n.startswith("STEP 1") for n in names)
        assert "Output instructions" in names
        assert "".join(t for _, t in fab.parse_prompt_sections(prompt)) == prompt


class TestBundle:
    def test_end_to_end(self, mini_db, tmp_path):
        out = tmp_path / "out"
        zip_path = fab.build_bundle(mini_db, out, keep_staging=False)
        assert zip_path is not None and zip_path.name == f"forecast_attribution__{FC_CUR}.zip"
        with zipfile.ZipFile(zip_path) as zf:
            names = set(zf.namelist())
            for expected in (
                "MANIFEST.json", "ANALYST_GUIDE.md", "LINKAGE.md",
                "attribution/signal_ledger.parquet", "attribution/signal_ledger_sample.csv",
                "attribution/prior_anchoring.csv", "attribution/rc_assessment.csv",
                "attribution/trace_quality.csv", "inputs/input_inventory.csv",
                "inputs/evidence_items.jsonl.gz", "inputs/evidence_to_signal.csv",
                "inputs/base_rates.csv", "prompts/prompt_sections.csv",
                "prompts/section_hashes.json", "prompts/token_share.csv",
                "contrasts/model_disagreement.csv", "contrasts/fred_vs_sibyl.csv",
                "contrasts/run_over_run.csv", "hazard/ACE.md", "hazard/DR.md", "hazard/FL.md",
                "hazard/TC.md", "hazard/HW.md", f"questions/{Q_ACE}.json",
                f"questions/{Q_FL}.json", f"questions/{Q_BIN}.json",
            ):
                assert expected in names, expected
            # Test rows are excluded by default.
            assert "questions/SOM_ACE_PA_2026-08.json" not in names

            manifest = json.loads(zf.read("MANIFEST.json"))
            assert manifest["bundle_kind"] == "forecast_attribution"
            assert manifest["run_id"] == FC_CUR
            assert manifest["hs_run_id"] == HS_CUR
            assert manifest["previous_run_id"] == FC_PREV
            assert manifest["taxonomy_version"] == fab.TAXONOMY_VERSION
            assert manifest["collector_failures"] == []
            assert manifest["counts"]["questions"] == 3
            assert manifest["linked_bundles"]["operational_debug_bundle"]["artifact"] == "pythia-debug-bundle"
            assert manifest["linked_bundles"]["resolutions_bundle"]["status"] == "not yet built"
            assert "attribution_id_recipe" in manifest
            assert manifest["split"]["moved_to_part2"] == []
            rollup = manifest["trace_quality_rollup"]["by_model"]
            assert rollup["gemini-3.5-flash"]["share_with_trace"] == 0.0
            assert rollup["gpt-5.6-sol"]["share_with_trace"] == 0.5  # ACE traced, binary not

            guide = zf.read("ANALYST_GUIDE.md").decode("utf-8")
            first_paragraphs = guide[:1500]
            assert "CLAIMED attribution" in first_paragraphs
            assert "ablation" in first_paragraphs.lower()
            linkage = zf.read("LINKAGE.md").decode("utf-8")
            assert "attribution_id" in linkage and "resolutions bundle" in linkage.lower()

            ledger = _read_csv_from_zip(zf, "attribution/signal_ledger_sample.csv")
            assert set(ledger[0].keys()) == set(fab.LEDGER_COLUMNS)
            by_model = {}
            for r in ledger:
                by_model.setdefault((r["question_id"], r["model_name"]), []).append(r)
            good = sorted(by_model[(Q_ACE, "gpt-5.6-sol")], key=lambda r: int(r["update_index"]))
            assert [int(r["update_index"]) for r in good] == [-1, 0, 1]
            assert good[0]["is_prior_row"] == "True" and good[0]["signal_class"] == "base_rate_reference"
            assert good[1]["signal_class"] == "conflict_escalation"
            assert good[2]["signal_class"] == "rc_flag"
            assert abs(float(good[1]["mass_moved_l1"]) - 0.15) < 1e-6
            assert float(good[1]["direction"]) > 0  # mass moved toward higher buckets
            assert good[1]["delta_sums_to_zero"] == "True" and good[1]["post_spd_reconciles"] == "True"
            assert good[1]["claimed_magnitude"] == "moderate"
            assert good[1]["attribution_id"] == fab.attribution_id(FC_CUR, Q_ACE, "gpt-5.6-sol", 0)
            bad = sorted(by_model[(Q_ACE, "claude-opus-5")], key=lambda r: int(r["update_index"]))
            assert bad[1]["delta_sums_to_zero"] == "False"
            assert bad[1]["signal_class"] == "funding_shortfall"
            none = by_model[(Q_ACE, "gemini-3.5-flash")]
            assert len(none) == 1 and none[0]["signal_class"] == "no_trace" and none[0]["delta_json"] == ""
            assert (Q_ACE, "ensemble_mean_v2") not in by_model
            assert by_model[(Q_ACE, "sibyl")][0]["signal_class"] == "no_trace"
            assert by_model[(Q_BIN, "gpt-5.6-sol")][0]["signal_class"] == "no_trace"
            t2 = by_model[(Q_FL, "track2_flash")]
            assert len(t2) == 1 and t2[0]["is_prior_row"] == "True"

            prior = _read_csv_from_zip(zf, "attribution/prior_anchoring.csv")
            gpt = next(r for r in prior if r["model_name"] == "gpt-5.6-sol" and r["question_id"] == Q_ACE)
            assert gpt["anchor_present"] == "True" and gpt["anchor_source"] == "acled_monthly_fatalities:36m"
            assert float(gpt["js_divergence"]) >= 0 and gpt["hazard_code"] == "ACE"
            fl = next(r for r in prior if r["question_id"] == Q_FL)
            assert fl["anchor_present"] == "False" and fl["js_divergence"] == ""

            rc = _read_csv_from_zip(zf, "attribution/rc_assessment.csv")
            gpt_rc = next(r for r in rc if r["model_name"] == "gpt-5.6-sol" and r["question_id"] == Q_ACE)
            assert gpt_rc["rc_assessment"] == "partial" and gpt_rc["hs_rc_level"] == "2"
            assert abs(float(gpt_rc["rc_flag_mass_moved_l1"]) - 0.05) < 1e-6
            claude_rc = next(r for r in rc if r["model_name"] == "claude-opus-5")
            assert claude_rc["rc_assessment"] == "rebutted"
            gem_rc = next(r for r in rc if r["model_name"] == "gemini-3.5-flash")
            assert gem_rc["rc_assessment"] == "absent"

            inv = {r["question_id"]: r for r in _read_csv_from_zip(zf, "inputs/input_inventory.csv")}
            assert inv[Q_ACE]["baserate_anchor_present"] == "True"
            assert inv[Q_ACE]["baserate_n_obs"] == "36"
            assert inv[Q_ACE]["crisiswatch_present"] == "True" and inv[Q_ACE]["crisiswatch_edition"] == "2026-07"
            assert inv[Q_ACE]["evidence_items"] == "2"
            assert inv[Q_ACE]["hs_rc_level"] == "2"
            assert inv[Q_ACE]["js_vs_baserate_mean_ensemble"] == "0.5"
            assert inv[Q_ACE]["resolver_history_months"] == ""  # no facts_resolved table

            with zf.open("inputs/evidence_items.jsonl.gz") as fh:
                items = [json.loads(l) for l in gzip.open(fh, "rt", encoding="utf-8")]
            assert len(items) == 2 and items[0]["host"] == "example.org"
            links = _read_csv_from_zip(zf, "inputs/evidence_to_signal.csv")
            assert links, "the Mekelle signal shares enough tokens with the Mekelle item"
            assert links[0]["attribution_id"] == fab.attribution_id(FC_CUR, Q_ACE, "gpt-5.6-sol", 0)
            assert links[0]["match_method"] == "token_containment"
            assert all(float(l["match_score"]) >= 0.5 for l in links)

            sections = _read_csv_from_zip(zf, "prompts/prompt_sections.csv")
            names_in = {s["section_name"] for s in sections}
            assert "REGIME CHANGE GUIDANCE (RC)" in names_in and "unclassified" in names_in
            hashes = json.loads(zf.read("prompts/section_hashes.json"))
            assert list(hashes) == ["spd_v2|ACE|FATALITIES|t1"]
            share = _read_csv_from_zip(zf, "prompts/token_share.csv")
            assert abs(sum(float(s["share"]) for s in share) - 1.0) < 0.01

            dis = _read_csv_from_zip(zf, "contrasts/model_disagreement.csv")
            pairs = {(d["model_a"], d["model_b"]) for d in dis}
            assert ("claude-opus-5", "gpt-5.6-sol") in pairs
            assert not any("sibyl" in p for p in pairs)
            row = next(d for d in dis if d["model_a"] == "claude-opus-5" and d["model_b"] == "gpt-5.6-sol")
            assert float(row["jsd_final"]) > 0 and float(row["jsd_prior"]) == 0.0

            fvs = _read_csv_from_zip(zf, "contrasts/fred_vs_sibyl.csv")
            assert len(fvs) == 1 and fvs[0]["fred_model"] == "ensemble_mean_v2"
            assert "conflict_escalation" in fvs[0]["fred_signal_classes"]

            rr = _read_csv_from_zip(zf, "contrasts/run_over_run.csv")
            assert len(rr) == 1 and rr[0]["previous_question_id"] == PQ_ACE
            assert rr[0]["ev_change"] != ""

            ace_md = zf.read("hazard/ACE.md").decode("utf-8")
            assert "conflict_escalation" in ace_md and Q_ACE in ace_md
            assert "No HW questions" in zf.read("hazard/HW.md").decode("utf-8")

            rec = json.loads(zf.read(f"questions/{Q_ACE}.json"))
            assert rec["forecast_run_id"] == FC_CUR
            assert len(rec["attribution"]["ledger"]) == 7  # gpt 3 + claude 2 + gemini 1 + sibyl 1
            assert rec["attribution"]["input_inventory"]["question_id"] == Q_ACE
            assert rec["attribution"]["evidence_ids"]
            assert "REGIME CHANGE GUIDANCE (RC)" in rec["attribution"]["prompt_section_fingerprint"]
            assert rec["members"][0]["model_name"]  # the shared record shape survived

        # The parquet is readable by DuckDB and matches the sample row count.
        with zipfile.ZipFile(zip_path) as zf:
            zf.extract("attribution/signal_ledger.parquet", tmp_path / "x")
        con = duckdb.connect()
        n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{(tmp_path / 'x' / 'attribution' / 'signal_ledger.parquet').as_posix()}')").fetchone()[0]
        con.close()
        assert n == 9  # 7 on the ACE question + track2 prior + binary no_trace

    def test_missing_forecast_deviation_table(self, mini_db, tmp_path):
        path = mini_db[len("duckdb:///"):]
        con = duckdb.connect(path)
        con.execute("DROP TABLE forecast_deviation")
        con.close()
        zip_path = fab.build_bundle(mini_db, tmp_path / "out")
        with zipfile.ZipFile(zip_path) as zf:
            manifest = json.loads(zf.read("MANIFEST.json"))
            assert manifest["collector_failures"] == []
            prior = _read_csv_from_zip(zf, "attribution/prior_anchoring.csv")
            assert all(r["anchor_present"] == "False" for r in prior)
            inv = _read_csv_from_zip(zf, "inputs/input_inventory.csv")
            assert all(r["baserate_anchor_present"] == "False" for r in inv)

    def test_split_moves_large_files_into_part2(self, mini_db, tmp_path):
        out = tmp_path / "out"
        zip_path = fab.build_bundle(mini_db, out, split_ceiling_mb=0.000001)
        part2 = out / f"forecast_attribution__{FC_CUR}__part2.zip"
        assert part2.exists()
        with zipfile.ZipFile(zip_path) as zf:
            names = set(zf.namelist())
            assert "attribution/signal_ledger.parquet" not in names
            manifest = json.loads(zf.read("MANIFEST.json"))
            assert "attribution/signal_ledger.parquet" in manifest["split"]["moved_to_part2"]
            assert manifest["split"]["part2_zip"] == part2.name
        with zipfile.ZipFile(part2) as zf:
            assert {"attribution/signal_ledger.parquet", "inputs/evidence_items.jsonl.gz", "MANIFEST.json"} <= set(zf.namelist())

    def test_main_never_fails(self, tmp_path):
        rc = fab.main(["--db", str(tmp_path / "does_not_exist.duckdb"), "--out-dir", str(tmp_path / "out")])
        assert rc == 0
        stubs = list((tmp_path / "out").glob("forecast_attribution__*.zip"))
        assert len(stubs) == 1
        with zipfile.ZipFile(stubs[0]) as zf:
            manifest = json.loads(zf.read("MANIFEST.json"))
            assert manifest["status"] == "failed" and manifest["collector_failures"]

    def test_include_test_flag(self, mini_db, tmp_path):
        zip_path = fab.build_bundle(mini_db, tmp_path / "out", run_id="fc_9999", include_test=True)
        with zipfile.ZipFile(zip_path) as zf:
            assert "questions/SOM_ACE_PA_2026-08.json" in zf.namelist()
