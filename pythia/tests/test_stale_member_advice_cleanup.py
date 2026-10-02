# Pythia / Copyright (c) 2025 Kevin Wyjad
"""A per-model advice row that no longer qualifies does not outlive its month.

The per-model threshold rose from ten score rows to twenty distinct
questions (2026-09-29), and family rows were added (2026-09-30). The
generator rebuilds a month's per-model and family rows from scratch, so a
model that qualified under the old rule earlier in the same month is left
with NO row rather than keeping one a prompt would still read. This pins
that clean-up, which until now had no test of its own.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb

from pythia.tools import generate_calibration_advice as gca


def _db(path: Path) -> None:
    con = duckdb.connect(str(path))
    con.execute(
        "CREATE TABLE questions (question_id TEXT, hazard_code TEXT, metric TEXT, "
        "is_test BOOLEAN DEFAULT FALSE)"
    )
    con.execute(
        "CREATE TABLE resolutions (question_id TEXT, horizon_m INTEGER, value DOUBLE, "
        "observed_month TEXT, source_desc TEXT)"
    )
    con.execute(
        "CREATE TABLE scores (question_id TEXT, horizon_m INTEGER, score_type TEXT, "
        "model_name TEXT, value DOUBLE, run_id TEXT)"
    )
    con.execute(
        "CREATE TABLE forecasts_ensemble (question_id TEXT, run_id TEXT, horizon_m INTEGER, "
        "class_bin TEXT, p DOUBLE)"
    )
    con.execute(
        "CREATE TABLE calibration_advice (as_of_month TEXT, hazard_code TEXT, metric TEXT, "
        "model_name TEXT, advice TEXT, findings_json TEXT, advice_version TEXT, "
        "created_at TIMESTAMP, PRIMARY KEY (as_of_month, hazard_code, metric, model_name))"
    )
    n_q = gca.MIN_QUESTIONS + 2
    for i in range(n_q):
        q = f"Q{i:02d}"
        con.execute("INSERT INTO questions VALUES (?, 'FL', 'PA', FALSE)", [q])
        con.execute(
            "INSERT INTO resolutions VALUES (?, 1, 1000, '2026-08', 'facts_resolved:IFRC:2026-08')",
            [q],
        )
        con.execute(
            "INSERT INTO scores VALUES (?, 1, 'brier', 'ensemble_mean_v2', 0.4, 'fc_1')", [q]
        )
        con.execute("INSERT INTO forecasts_ensemble VALUES (?, 'fc_1', 1, '0', 1.0)", [q])
        # model-a scored only a handful of questions: below the threshold.
        if i < 5:
            con.execute("INSERT INTO scores VALUES (?, 1, 'brier', 'model-a', 0.3, 'fc_1')", [q])
    # Rows written earlier in the same month under the old rules.
    con.execute(
        "INSERT INTO calibration_advice VALUES "
        "('2026-09','FL','PA','model-a','stale member advice',NULL,'v1',now()), "
        "('2026-09','FL','PA','family:gpt','stale family advice',NULL,'v1',now()), "
        "('2026-08','FL','PA','model-a','last month, untouched',NULL,'v1',now())"
    )
    con.close()


def test_a_member_that_no_longer_qualifies_is_left_with_no_row(tmp_path: Path) -> None:
    db = tmp_path / "advice.duckdb"
    _db(db)
    gca.generate_calibration_advice(f"duckdb:///{db}", as_of=date(2026, 9, 29))
    con = duckdb.connect(str(db))
    rows = con.execute(
        "SELECT as_of_month, model_name FROM calibration_advice "
        "WHERE hazard_code = 'FL' AND metric = 'PA' ORDER BY 1, 2"
    ).fetchall()
    con.close()
    month = [m for a, m in rows if a == "2026-09"]
    assert "model-a" not in month
    assert "family:gpt" not in month
    assert "__shared__" in month  # the group itself still gets its advice
    # Another month's rows are not this run's to remove.
    assert ("2026-08", "model-a") in rows
