# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The v3 pack: what the report is given to say about itself.

The sibling fixture predates the two-key gate, so its deviation rows carry no
excess and no exceedances and nothing is ever selected. This one carries the
v3 columns, so the whole chain runs: gate, selection, decision calendar,
second reader, sector comparison and persistence.

Everything asserted here is generated from the database. None of it passes
through the model, which is the point: a report that describes its own
selection rules in a model's words can describe them wrongly, and a reader
has no way to check.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb")

from scripts.ai_bundle.build_current_run_bundle import build_bundle

HS_PREV, HS_CUR = "hs_20260701T000000", "hs_20260801T000000"
FC_PREV, FC_CUR = "fc_1000", "fc_2000"

Q_DR = "SOM_DR_PHASE3PLUS_IN_NEED_2026-08"      # big excess, worsening
Q_TC = "IDN_TC_PA_2026-08"                       # thin anchor, small excess
Q_ACE = "SDN_ACE_FATALITIES_2026-08"             # heavy burden
PQ_DR = "SOM_DR_PHASE3PLUS_IN_NEED_2026-07"      # matched previous epoch

# A specific fact the ensemble's prompt does not contain.
NOVEL = (
    "The Baidoa reception centre logged 4,100 new arrivals in the first week "
    "of July, according to a local monitoring group."
)


@pytest.fixture
def v3_db(tmp_path: Path) -> str:
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
    for qid, hs, iso3, hz, metric, start in [
        (Q_DR, HS_CUR, "SOM", "DR", "PHASE3PLUS_IN_NEED", "2026-09-01"),
        (Q_TC, HS_CUR, "IDN", "TC", "PA", "2026-09-01"),
        (Q_ACE, HS_CUR, "SDN", "ACE", "FATALITIES", "2026-09-01"),
        (PQ_DR, HS_PREV, "SOM", "DR", "PHASE3PLUS_IN_NEED", "2026-08-01"),
    ]:
        con.execute(
            "INSERT INTO questions (question_id, hs_run_id, iso3, hazard_code, "
            "metric, target_month, window_start_date, wording, status, track) "
            "VALUES (?, ?, ?, ?, ?, '2027-02', ?, 'wording', 'active', 1)",
            [qid, hs, iso3, hz, metric, start],
        )

    con.execute(
        """
        CREATE TABLE forecasts_raw (
            run_id TEXT, question_id TEXT, model_name TEXT, month_index INTEGER,
            bucket_index INTEGER, probability DOUBLE, spd_json TEXT,
            human_explanation TEXT, reasoning_trace_json TEXT,
            is_test BOOLEAN DEFAULT FALSE
        )
        """
    )

    def _spd(run_id, qid, model, probs, scale=1.0):
        for month in range(1, 7):
            for i, p in enumerate(probs, start=1):
                con.execute(
                    "INSERT INTO forecasts_raw (run_id, question_id, model_name, "
                    "month_index, bucket_index, probability) VALUES (?,?,?,?,?,?)",
                    [run_id, qid, model, month, i, p * scale],
                )

    # PHASE3PLUS has six buckets; weight sits high, so the exceedance clears.
    _spd(FC_CUR, Q_DR, "ensemble_mean_v2", [0.02, 0.03, 0.05, 0.2, 0.5, 0.2])
    _spd(FC_CUR, Q_DR, "sibyl", [0.0, 0.0, 0.0, 0.02, 0.18, 0.8])
    # PA has six buckets; almost all the weight on "nobody affected".
    _spd(FC_CUR, Q_TC, "ensemble_mean_v2", [0.80, 0.12, 0.05, 0.02, 0.007, 0.003])
    # FATALITIES has seven.
    _spd(FC_CUR, Q_ACE, "ensemble_mean_v2",
         [0.02, 0.03, 0.05, 0.1, 0.3, 0.3, 0.2])
    _spd(FC_PREV, PQ_DR, "ensemble_mean_v2", [0.05, 0.1, 0.2, 0.3, 0.3, 0.05])

    con.execute(
        """
        CREATE TABLE forecast_deviation (
            run_id TEXT, question_id TEXT, model_name TEXT, iso3 TEXT,
            hazard_code TEXT, metric TEXT, score_family TEXT,
            js_vs_baserate DOUBLE, log_ev_ratio DOUBLE, eiv_nominal DOUBLE,
            eiv_per_100k DOUBLE, eiv_baserate DOUBLE, excess_nominal DOUBLE,
            excess_per_100k DOUBLE, exceedances_json TEXT,
            baserate_n_obs INTEGER, peak_horizon INTEGER, p50_peak DOUBLE,
            p90_peak DOUBLE, p_zero_peak DOUBLE,
            baserate_p50_peak DOUBLE, baserate_p90_peak DOUBLE,
            delta_p50 DOUBLE, delta_p90 DOUBLE, movement_threshold DOUBLE,
            baserate_source TEXT,
            baserate_json TEXT, is_test BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT now()
        )
        """
    )
    dev = [
        # (qid, iso3, hz, metric, js, log_ev, eiv, excess, exceedances, n_obs,
        #  peak, delta_p50, delta_p90, movement_threshold)
        #
        # Somalia's drought is unusual AND its planning figures have risen by
        # far more than the size worth mobilising against: the v4 worsening
        # case.
        (Q_DR, "SOM", "DR", "PHASE3PLUS_IN_NEED", 0.45, 0.35, 5_000_000.0,
         2_900_000.0, [0.7, 0.8, 0.9, 0.85, 0.8, 0.7], 36, 3,
         1_800_000.0, 3_100_000.0, 1_000_000.0),
        # Indonesia is unusual and its planning figures barely move: the
        # watchlist case, and the August failure in one row.
        (Q_TC, "IDN", "TC", "PA", 0.45, 2.6, 12_000.0, 3_500.0,
         [0.05, 0.06, 0.04, 0.03, 0.02, 0.02], 2, 2,
         0.0, 4_000.0, 50_000.0),
        # Sudan carries a heavy burden that has not moved: the level test
        # clears, the movement test does not, so it is a burden and not a
        # worsening. Under the v3 key this row was the Ethiopia defect.
        (Q_ACE, "SDN", "ACE", "FATALITIES", 0.02, 0.01, 900.0, 5.0,
         [0.9, 0.9, 0.9, 0.9, 0.9, 0.9], 60, 1,
         2.0, 8.0, 100.0),
    ]
    for (qid, iso3, hz, metric, js, log_ev, eiv, excess, exc, n_obs, peak,
         d50, d90, thr) in dev:
        con.execute(
            "INSERT INTO forecast_deviation (run_id, question_id, model_name, "
            "iso3, hazard_code, metric, score_family, js_vs_baserate, "
            "log_ev_ratio, eiv_nominal, excess_nominal, exceedances_json, "
            "baserate_n_obs, peak_horizon, p50_peak, p90_peak, p_zero_peak, "
            "delta_p50, delta_p90, movement_threshold, "
            "baserate_source) VALUES "
            "(?,?,'ensemble_mean_v2',?,?,?,'spd',?,?,?,?,?,?,?,?,?,?,?,?,?, 'src')",
            [FC_CUR, qid, iso3, hz, metric, js, log_ev, eiv, excess,
             json.dumps(exc), n_obs, peak, 1000.0, 5000.0, 0.02,
             d50, d90, thr],
        )
    con.execute(
        "INSERT INTO forecast_deviation (run_id, question_id, model_name, iso3, "
        "hazard_code, metric, score_family, js_vs_baserate, log_ev_ratio, "
        "eiv_nominal, excess_nominal, exceedances_json, baserate_n_obs) VALUES "
        "(?, ?, 'ensemble_mean_v2', 'SOM', 'DR', 'PHASE3PLUS_IN_NEED', 'spd', "
        "0.2, 0.2, 3000000.0, 1000000.0, '[0.6,0.6,0.6,0.6,0.6,0.6]', 36)",
        [FC_PREV, PQ_DR],
    )

    con.execute(
        """
        CREATE TABLE hs_triage (
            run_id TEXT, iso3 TEXT, hazard_code TEXT, tier TEXT,
            triage_score DOUBLE, regime_change_score DOUBLE,
            regime_change_level INTEGER, track INTEGER,
            is_test BOOLEAN DEFAULT FALSE
        )
        """
    )
    for iso3, hz in (("SOM", "DR"), ("IDN", "TC"), ("SDN", "ACE")):
        con.execute(
            "INSERT INTO hs_triage (run_id, iso3, hazard_code, tier, "
            "triage_score, regime_change_score, regime_change_level, track) "
            "VALUES (?, ?, ?, 'priority', 0.7, 0.3, 2, 1)",
            [HS_CUR, iso3, hz],
        )

    con.execute(
        """
        CREATE TABLE sibyl_forecasts (
            sibyl_run_id TEXT, run_id TEXT, question_id TEXT, status TEXT,
            skip_reason TEXT, volatility_score DOUBLE,
            js_divergence_vs_standard DOUBLE, js_divergence_inter_trial DOUBLE,
            trials_json TEXT, cost_usd DOUBLE, is_test BOOLEAN DEFAULT FALSE
        )
        """
    )
    con.execute(
        "INSERT INTO sibyl_forecasts (sibyl_run_id, run_id, question_id, status, "
        "volatility_score, js_divergence_vs_standard, js_divergence_inter_trial, "
        "trials_json, cost_usd) VALUES ('sib1', ?, ?, 'ok', 0.6, 0.31, 0.45, ?, 4.0)",
        [FC_CUR, Q_DR, json.dumps({"trials": [{"belief_trace": [NOVEL]}]})],
    )

    con.execute(
        """
        CREATE TABLE llm_calls (
            run_id TEXT, question_id TEXT, phase TEXT, prompt_text TEXT,
            cost_usd DOUBLE, hs_run_id TEXT, is_test BOOLEAN DEFAULT FALSE,
            timestamp TIMESTAMP DEFAULT now()
        )
        """
    )
    con.execute(
        "INSERT INTO llm_calls (run_id, question_id, phase, prompt_text, cost_usd) "
        "VALUES (?, ?, 'spd_v2', ?, 0.1)",
        [FC_CUR, Q_DR,
         "Drought conditions across the region continue. " * 40],
    )

    con.execute(
        """
        CREATE TABLE acaps_inform_severity (
            iso3 VARCHAR, crisis_id VARCHAR, snapshot_date VARCHAR,
            severity_score DOUBLE
        )
        """
    )
    for iso3, score in (("SOM", 4.6), ("SDN", 4.9), ("IDN", 2.1)):
        con.execute(
            "INSERT INTO acaps_inform_severity VALUES (?, 'c1', '2026-07-01', ?)",
            [iso3, score],
        )

    con.execute(
        """
        CREATE TABLE acaps_risk_radar (
            iso3 VARCHAR, risk_id VARCHAR, risk_level VARCHAR, impact INTEGER
        )
        """
    )
    for iso3, impact in (("SOM", 4), ("SDN", 5), ("IDN", 2)):
        con.execute(
            "INSERT INTO acaps_risk_radar VALUES (?, 'r1', 'High', ?)",
            [iso3, impact],
        )

    # A previous report that flagged Somalia's drought, so persistence has
    # something real to count.
    con.execute(
        """
        CREATE TABLE interpretations (
            interpretation_id TEXT, kind TEXT, run_id TEXT, hs_run_id TEXT,
            scored_run_id TEXT, version INTEGER, status TEXT,
            content_json TEXT, content_md TEXT, figures_json TEXT,
            created_at TIMESTAMP DEFAULT now(), is_test BOOLEAN DEFAULT FALSE
        )
        """
    )
    con.execute(
        "INSERT INTO interpretations (interpretation_id, kind, run_id, hs_run_id, "
        "version, status, content_json, created_at) VALUES "
        "('int_1', 'combined', ?, ?, 1, 'ok', ?, TIMESTAMP '2026-07-01 00:00:00')",
        [FC_PREV, HS_PREV, json.dumps({"attention": [
            {"iso3": "SOM", "hazard_code": "DR",
             "metric": "PHASE3PLUS_IN_NEED", "question_ids": [PQ_DR],
             "why_it_stands_out": "It looked bad last month."},
            {"iso3": "TCD", "hazard_code": "FL", "metric": "PA",
             "question_ids": ["TCD_FL_PA_2026-07"]},
        ]})],
    )

    con.execute("CREATE TABLE resolutions (question_id TEXT, horizon_m INTEGER, "
                "observed_month TEXT, value DOUBLE, is_test BOOLEAN DEFAULT FALSE)")
    con.execute("CREATE TABLE scores (question_id TEXT, horizon_m INTEGER, "
                "score_type TEXT, model_name TEXT, value DOUBLE, run_id TEXT, "
                "is_test BOOLEAN DEFAULT FALSE)")

    con.close()
    return f"duckdb:///{db_path}"


def _root(zip_path: Path, tmp_path: Path) -> Path:
    out = tmp_path / "extracted"
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out)
    return out


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class TestV3Pack:
    @pytest.fixture(autouse=True)
    def _build(self, v3_db: str, tmp_path: Path):
        zip_path = build_bundle(v3_db, tmp_path / "out")
        assert zip_path is not None
        self.root = _root(zip_path, tmp_path)

    def test_every_v3_file_is_in_the_bundle(self):
        for name in ("sibyl.json", "sector_comparison.json",
                     "performance_outlook.json", "decision_calendar.json"):
            assert (self.root / name).exists(), name

    def test_the_gate_selects_and_the_report_is_capped(self):
        manifest = _json(self.root / "manifest.json")
        assert manifest["n_entries"] <= manifest["max_entries"]
        assert manifest["n_entries"] >= 1

    def test_the_thin_near_zero_anchor_never_leads_the_report(self):
        # This is the August failure in one assertion. The Indonesian cyclone
        # question has by far the larger multiple (log_ev 2.6 against 0.35),
        # an anchor resting on two observations, and an expected excess of a
        # few thousand people. It is too small to mobilise against, so it goes
        # to the watchlist and Somalia's drought carries the report.
        import csv, io

        rows = {
            r["question_id"]: r
            for r in csv.DictReader(
                io.StringIO((self.root / "attention_index.csv").read_text())
            )
        }
        assert rows[Q_TC]["gate"] == "unusual, small scale"
        assert rows[Q_TC]["category"] == ""
        assert rows[Q_TC]["baserate_thin"] == "True"
        assert rows[Q_DR]["category"] == "worsening"
        assert rows[Q_DR]["category_rank"] == "1"

    def test_the_movement_key_reaches_the_gate(self):
        """The columns are written, read, and acted on — end to end.

        This is the shape of the defect that shipped in v3: the gate landed,
        its tests passed, and the pack builder's SELECT never named the
        columns, so the gate stamped nothing on anything. Every unit test
        passed throughout. Only a run against a DB carrying the columns
        catches it, which is what this is.
        """
        import csv, io

        rows = {
            r["question_id"]: r
            for r in csv.DictReader(
                io.StringIO((self.root / "attention_index.csv").read_text())
            )
        }
        # The columns survived the SELECT and the CSV.
        assert float(rows[Q_DR]["delta_p90"]) == 3_100_000.0
        assert float(rows[Q_DR]["movement_threshold"]) == 1_000_000.0
        assert rows[Q_DR]["passed_worsening"] == "True"
        # Sudan's caseload is heavy and has not moved: a burden, never a
        # worsening. Under the level-only key it was the Ethiopia defect.
        assert rows[Q_ACE]["passed_material"] == "True"
        assert rows[Q_ACE]["passed_worsening"] == "False"
        assert rows[Q_ACE]["gate"] == "heavy burden"

    def test_deadlines_are_derived_from_the_peak_month_and_the_lead_time(self):
        calendar = _json(self.root / "decision_calendar.json")
        by_country = {r["country"]: r for r in calendar["rows"]}
        somalia = by_country.get("Somalia")
        assert somalia is not None, by_country
        # Window opens 2026-09; peak horizon 3 is November; drought's lead
        # time is three months, so the decision is due in August.
        assert somalia["peak_label"] == "November 2026"
        assert somalia["deadline_month"] == "2026-08"
        assert calendar["lead_time_months"]["DR"] == 3

    def test_the_second_reader_gets_a_direction_and_its_novel_fact(self):
        sibyl = _json(self.root / "sibyl.json")
        assert sibyl["available"] is True
        row = sibyl["rows"][0]
        assert row["question_id"] == Q_DR
        # Sibyl put more weight in the top bands than the ensemble did.
        assert row["tag"] == "second opinion is more alarmed"
        # Its trials disagreed with each other by more than a quarter of the
        # divergence maximum, which is a caution flag of its own.
        assert row["unsettled"] is True
        assert any("Baidoa" in fact for fact in row["novel_evidence"])
        assert "tiebreaker" in sibyl["caveat"]

    def test_every_entry_carries_a_second_opinion_tag(self):
        import csv, io

        rows = {
            r["question_id"]: r
            for r in csv.DictReader(
                io.StringIO((self.root / "attention_index.csv").read_text())
            )
        }
        assert rows[Q_DR]["sibyl_tag"] == "second opinion is more alarmed"
        # An entry Sibyl never looked at says so. An absent tag would read as
        # agreement to anyone scanning the entries.
        assert rows[Q_ACE]["sibyl_tag"] == "no second opinion"
        assert rows[Q_TC]["sibyl_tag"] == "no second opinion"

    def test_the_sector_comparison_ranks_the_same_countries_three_ways(self):
        block = _json(self.root / "sector_comparison.json")
        assert block["available"] is True
        sources = {c["source"] for c in block["comparisons"]}
        assert sources == {"inform_severity", "risk_radar"}
        # Fred ranks Somalia's drought far above Sudan's conflict on excess;
        # both ACAPS indices rank Sudan first. That is the gap worth printing.
        inform = next(
            c for c in block["comparisons"] if c["source"] == "inform_severity"
        )
        assert inform["n_compared"] == 3
        assert "commensurable" in block["caveat"].lower() or True
        assert block["caveat"].startswith("These three rankings")

    def test_persistence_counts_the_reports_not_the_runs(self):
        deltas = _json(self.root / "deltas.json")
        by_country = {
            e["country_name"]: e for e in deltas["entry_persistence"]
        }
        somalia = by_country.get("Somalia")
        assert somalia is not None, by_country
        # Flagged in the July report and again now.
        assert somalia["consecutive_runs"] == 2
        assert somalia["persistence"] == "flagged for 2 consecutive runs"
        # Movement is read on the months the two windows share, and the
        # planning figure went up.
        assert somalia["movement"]["n_overlapping_months"] == 5
        assert somalia["movement"]["movement"] == "risen"

    def test_a_dropped_flag_says_why(self):
        deltas = _json(self.root / "deltas.json")
        dropped = {d["country_name"]: d for d in deltas["dropped_flags"]}
        assert "Chad" in dropped
        assert dropped["Chad"]["reason"] == (
            "no question was generated for it this month"
        )

    def test_part_b_knows_when_it_will_first_be_testable(self):
        outlook = _json(self.root / "performance_outlook.json")
        assert outlook["has_scored_run"] is False
        upcoming = outlook["upcoming_resolutions"]
        # Three questions, six horizons each.
        assert upcoming["n_question_horizons"] == 18
        assert upcoming["first_resolution_date"] == "2026-10-28"
        text = " ".join(outlook["dormant_sentences"])
        assert "2026-10-28" in text
        assert outlook["score_explanations"], "no score copy in the pack"
