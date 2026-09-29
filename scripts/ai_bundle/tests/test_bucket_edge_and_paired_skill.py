# Pythia / Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
"""Two scored-bundle fixes and one addition.

* ``bucket_edge`` is judged against the nearest FINITE interior boundary
  (84 of 160 rows once read True because 5% of ``inf`` is ``inf``).
* ``skill_vs_climatology`` is computed on PAIRED (question, horizon) scores,
  per track: Track 2 DR/EVENT_OCCURRENCE read +0.80 unpaired against a paired
  +0.74 on the Sept 2026 data.
* The digest reports the primary aggregate's sharpness per (hazard, metric, track).
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from scripts.ai_bundle import build_scored_forecast_bundle as b


@pytest.mark.parametrize(
    "metric, value, edge, boundary",
    [
        ("FATALITIES", 210, False, 100.0),
        ("FATALITIES", 781, False, 1000.0),
        ("FATALITIES", 498, True, 500.0),
        ("FATALITIES", 0, False, 1.0),
        ("FATALITIES", 5000, False, 1000.0),
        ("PHASE3PLUS_IN_NEED", 1_000_000, True, 1_000_000.0),
        ("PA", 52_000, True, 50_000.0),
    ],
)
def test_bucket_edge(metric, value, edge, boundary):
    got_edge, got_boundary = b._bucket_edge(None, metric, value)
    assert got_edge is edge
    assert got_boundary == pytest.approx(boundary)


def test_bucket_edge_is_null_for_binary_and_missing():
    assert b._bucket_edge(None, "EVENT_OCCURRENCE", 1) == (None, None)
    assert b._bucket_edge(None, "FATALITIES", None) == (None, None)


def _db(tmp_path: Path) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(tmp_path / "s.duckdb"))
    con.execute("CREATE TABLE questions (question_id TEXT, hazard_code TEXT, metric TEXT, track INTEGER)")
    con.execute(
        "CREATE TABLE scores (question_id TEXT, horizon_m INTEGER, metric TEXT, score_type TEXT, "
        "model_name TEXT, value DOUBLE, run_id TEXT)"
    )
    return con


def test_skill_is_paired_and_split_by_track(tmp_path):
    con = _db(tmp_path)
    # Track 2: three questions. Climatology scored all three (easy ones cheap,
    # a hard one dear); the model scored only the two easy ones.
    # Track 1: one hard question climatology also scored.
    con.execute(
        "INSERT INTO questions VALUES "
        "('q1','DR','EVENT_OCCURRENCE',2),('q2','DR','EVENT_OCCURRENCE',2),"
        "('q3','DR','EVENT_OCCURRENCE',2),('q9','DR','EVENT_OCCURRENCE',1)"
    )
    con.execute(
        "INSERT INTO scores VALUES "
        "('q1',1,'EVENT_OCCURRENCE','brier','__ext_climatology',0.10,NULL),"
        "('q2',1,'EVENT_OCCURRENCE','brier','__ext_climatology',0.10,NULL),"
        "('q3',1,'EVENT_OCCURRENCE','brier','__ext_climatology',0.70,NULL),"
        "('q9',1,'EVENT_OCCURRENCE','brier','__ext_climatology',0.90,NULL),"
        "('q1',1,'EVENT_OCCURRENCE','brier','track2_flash',0.02,'r1'),"
        "('q2',1,'EVENT_OCCURRENCE','brier','track2_flash',0.03,'r1'),"
        "('q9',1,'EVENT_OCCURRENCE','brier','ensemble_mean_v2',0.45,'r1')"
    )
    rows = b._emit_rollups(con, tmp_path, ["q1", "q2", "q3", "q9"])

    def row(model, track):
        (r,) = [r for r in rows if r["model_name"] == model and r["track"] == track]
        return r

    t2 = row("track2_flash", 2)
    # Paired: climatology over q1,q2 only = 0.10 → skill = 1 − 0.025/0.10.
    assert t2["climatology_mean"] == pytest.approx(0.10)
    assert t2["skill_vs_climatology"] == pytest.approx(0.75)
    # The old unpaired figure would have divided by 0.4 (every question, both
    # tracks) and read +0.94.
    assert t2["n_questions"] == 2 and t2["n_paired"] == 2
    t1 = row("ensemble_mean_v2", 1)
    assert t1["climatology_mean"] == pytest.approx(0.90)
    assert t1["skill_vs_climatology"] == pytest.approx(0.5)
    # Climatology appears once per track and matches itself.
    assert row("__ext_climatology", 2)["skill_vs_climatology"] == pytest.approx(0.0)
    assert row("__ext_climatology", 2)["n_questions"] == 3
    # The csv carries the track column.
    header = (tmp_path / "rollups.csv").read_text().splitlines()[0]
    assert "track" in header.split(",") and "n_paired" in header


def test_digest_pools_paired_sums_and_reports_sharpness(tmp_path):
    rollups = [
        {"score_family": "spd", "track": 1, "model_name": "ensemble_mean_v2", "score_type": "brier",
         "n_samples": 4, "mean_value": 0.6, "median_value": 0.6,
         "n_paired": 4, "paired_model_mean": 0.6, "climatology_mean": 0.4},
        {"score_family": "spd", "track": 1, "model_name": "ensemble_mean_v2", "score_type": "brier",
         "n_samples": 1, "mean_value": 0.2, "median_value": 0.2,
         "n_paired": 1, "paired_model_mean": 0.2, "climatology_mean": 0.8},
    ]
    fvo = [
        {"question_id": "q1", "horizon_m": 1, "model_name": "ensemble_mean_v2", "hazard_code": "ACE",
         "metric": "FATALITIES", "track": 1, "p_modal_bucket": 0.5, "realized_bucket": 4,
         "p_realized_bucket": 0.2},
        {"question_id": "q1", "horizon_m": 1, "model_name": "ensemble_bayesmc_v2", "hazard_code": "ACE",
         "metric": "FATALITIES", "track": 1, "p_modal_bucket": 0.6, "realized_bucket": 4,
         "p_realized_bucket": 0.4},
        {"question_id": "q2", "horizon_m": 1, "model_name": "track2_flash", "hazard_code": "ACE",
         "metric": "FATALITIES", "track": 2, "p_modal_bucket": 0.7, "realized_bucket": 3,
         "p_realized_bucket": 0.7},
    ]
    b._write_digest(tmp_path, [], rollups, [], {}, months_back=12, fvo_rows=fvo)
    text = (tmp_path / "digest.md").read_text()
    # Pooled paired: model 2.6/5, clim 2.4/5 → skill = 1 − 2.6/2.4 = −0.083.
    assert "| spd | T1 | ensemble_mean_v2 | 5 |" in text
    assert "-0.083" in text
    assert "## Sharpness of the primary aggregate (SPD)" in text
    # bayesmc stands for the question, not mean.
    assert "| ACE | FATALITIES | T1 | 1 | 0.600 | 0.400 |" in text
    assert "| ACE | FATALITIES | T2 | 1 | 0.700 | 0.700 |" in text
