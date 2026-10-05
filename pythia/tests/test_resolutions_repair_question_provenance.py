# Pythia / Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
"""compute_resolutions repairs question provenance before it resolves (Oct 2026).

The repair ran only at question creation, once a month, so the release of
2026-10-03 still held 28 production questions pointing at test scans.
"""

from __future__ import annotations

from datetime import date

import pytest

duckdb = pytest.importorskip("duckdb")

from pythia.db.schema import ensure_schema
from pythia.tools.compute_resolutions import compute_resolutions


def test_compute_resolutions_points_questions_back_at_production(tmp_path):
    path = tmp_path / "r.duckdb"
    con = duckdb.connect(str(path))
    ensure_schema(con)
    for hs, test in [("hs_20260915T130009", False), ("hs_20260917T104316", True)]:
        con.execute(
            "INSERT INTO hs_runs (hs_run_id, generated_at, git_sha, config_profile, countries_json, is_test) "
            "VALUES (?, CURRENT_TIMESTAMP, 'x', 'default', '[]', ?)",
            [hs, test],
        )
        con.execute(
            "INSERT INTO hs_triage (run_id, iso3, hazard_code, tier, triage_score, need_full_spd, "
            "drivers_json, regime_shifts_json, data_quality_json, scenario_stub) "
            "VALUES (?, 'SOM', 'ACE', 'priority', 0.8, TRUE, '[]', '[]', '{}', '')",
            [hs],
        )
    con.execute(
        "INSERT INTO questions (question_id, hs_run_id, iso3, hazard_code, metric, target_month, "
        "window_start_date, window_end_date, wording, status, pythia_metadata_json, is_test) "
        "VALUES ('SOM_ACE_PA_2026-10', 'hs_20260917T104316', 'SOM', 'ACE', 'PA', '2027-03', "
        "DATE '2026-10-01', DATE '2027-03-31', 'w', 'active', '{}', FALSE)"
    )
    con.close()
    compute_resolutions(f"duckdb:///{path}", today=date(2026, 10, 11))
    con = duckdb.connect(str(path))
    assert con.execute("SELECT hs_run_id FROM questions").fetchone()[0] == "hs_20260915T130009"
