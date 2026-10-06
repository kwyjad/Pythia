# Pythia / Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
"""ACE/PA resolves from IDMC conflict displacement, behind two gates (Oct 2026).

Every IDMC row was hazard IDU, all causes summed, while the resolver looked
for hazard ACE: 149 ACE/PA questions existed on the 3 October 2026 release
and none had a resolution.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from pythia.tools import base_rate_spd as brs
from pythia.tools import compute_resolutions as cr


def _db(path: Path):
    con = duckdb.connect(str(path))
    con.execute(
        """
        CREATE TABLE facts_resolved (
            ym TEXT, iso3 TEXT, hazard_code TEXT, metric TEXT,
            series_semantics TEXT, value DOUBLE, publisher TEXT, source_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    con.execute("CREATE TABLE hs_runs (hs_run_id TEXT PRIMARY KEY)")
    con.execute("INSERT INTO hs_runs VALUES ('run1')")
    con.execute(
        "CREATE TABLE questions (question_id TEXT, hs_run_id TEXT, iso3 TEXT, hazard_code TEXT, "
        "metric TEXT, target_month TEXT, window_start_date DATE, status TEXT, "
        "is_test BOOLEAN DEFAULT FALSE)"
    )
    return con


def _flow(con, ym, iso3, value, hazard="ACE"):
    con.execute(
        "INSERT INTO facts_resolved (ym, iso3, hazard_code, metric, series_semantics, value, "
        "publisher, source_id) VALUES (?, ?, ?, 'new_displacements', 'new', ?, 'IDMC', 'idmc')",
        [ym, iso3, hazard, value],
    )


@pytest.mark.db
def test_ace_pa_resolves_settled_months_and_bracketed_zeros_only(tmp_path, monkeypatch):
    db = tmp_path / "r.duckdb"
    db_url = f"duckdb:///{db}"
    monkeypatch.setattr(cr, "load_cfg", lambda: {"app": {"db_url": db_url}})
    con = _db(db)
    # SDN reports every month of 2024-02..2025-07 except 2025-02 and 2025-04
    # (a regular reporter); its last report is 2025-07.
    for m in ("2024-02", "2024-03", "2024-04", "2024-05", "2024-06", "2024-07", "2024-08", "2024-09", "2024-10", "2024-11", "2024-12", "2025-01",
              "2025-03", "2025-05", "2025-06", "2025-07"):
        _flow(con, m, "SDN", 9000)
    _flow(con, "2025-09", "SOM", 300)          # another country's later month
    _flow(con, "2025-08", "CHN", 7_305_385, hazard="IDU")  # all-cause: never read
    for qid, iso in (("SDN_ACE_PA_2025-02", "SDN"), ("CHN_ACE_PA_2025-02", "CHN")):
        con.execute(
            "INSERT INTO questions VALUES (?, 'run1', ?, 'ACE', 'PA', '2025-07', "
            "DATE '2025-02-01', 'active', FALSE)",
            [qid, iso],
        )
    con.execute(
        "CREATE TABLE resolutions (question_id TEXT, horizon_m INTEGER, observed_month TEXT, "
        "value DOUBLE, source_snapshot_ym TEXT, created_at TIMESTAMP, is_test BOOLEAN, "
        "source_desc TEXT, acled_snapshot_date DATE, PRIMARY KEY (question_id, horizon_m))"
    )
    # A zero an earlier rule wrote for a month the new rule cannot support.
    con.execute(
        "INSERT INTO resolutions (question_id, horizon_m, observed_month, value, source_desc) "
        "VALUES ('SDN_ACE_PA_2025-02', 6, '2025-07', 0.0, 'zero_default')"
    )
    con.close()

    # 2025-07 ended on 31 July; on 1 October it is unsettled (60 days in).
    outcome = cr.compute_resolutions(db_url=db_url, today=__import__("datetime").date(2025, 10, 1))

    con = duckdb.connect(str(db))
    rows = con.execute(
        "SELECT question_id, horizon_m, observed_month, value, source_desc "
        "FROM resolutions ORDER BY question_id, horizon_m"
    ).fetchall()
    con.close()
    series = brs.CONFLICT_DISPLACEMENT_SERIES
    assert rows == [
        ("SDN_ACE_PA_2025-02", 1, "2025-02", 0.0, "zero_default"),   # bracketed by 2025-03
        ("SDN_ACE_PA_2025-02", 2, "2025-03", 9000.0, series),
        ("SDN_ACE_PA_2025-02", 3, "2025-04", 0.0, "zero_default"),   # bracketed by 2025-05
        ("SDN_ACE_PA_2025-02", 4, "2025-05", 9000.0, series),
        ("SDN_ACE_PA_2025-02", 5, "2025-06", 9000.0, series),
        # 2025-07: not settled, and the stale zero for it was deleted.
    ]
    assert outcome["ACE/PA"]["sourced"] == 3
    assert outcome["ACE/PA"]["zero_default"] == 2
    assert outcome["ACE/PA"]["unresolved_for_lag"] == 2  # SDN and CHN 2025-07
    # China: no conflict series at all; its typhoon row is IDU.
    assert outcome["ACE/PA"]["unresolved_irregular_reporter"] == 5


def test_the_writer_and_the_reader_name_one_series():
    from resolver.ingestion import idmc_conflict as ic

    assert ic.HAZARD_CODE == brs.CONFLICT_DISPLACEMENT_HAZARD == "ACE"
    assert ic.METRIC == brs.CONFLICT_DISPLACEMENT_METRIC
    assert ic.SOURCE == brs.CONFLICT_DISPLACEMENT_PUBLISHER


def test_persistence_reads_the_newest_month_settled_when_the_forecast_was_made(tmp_path):
    from pythia.tools.score_baselines import PERSISTENCE_PAIRS, persistence_spd

    assert ("ACE", "PA") in PERSISTENCE_PAIRS
    con = _db(tmp_path / "p.duckdb")
    _flow(con, "2025-03", "SDN", 9000)
    _flow(con, "2025-07", "SDN", 500)
    _flow(con, "2025-07", "SOM", 300)
    # A window starting 2025-09 was forecast on 13 August: 2025-07 had not
    # settled, so the newest month the forecaster could read was 2025-03
    # (SDN reports too rarely for the months between to be zero).
    last = brs.last_observed_value(con, "SDN", "ACE", "PA", "2025-09")
    assert last == (9000.0, "2025-03", brs.CONFLICT_DISPLACEMENT_SERIES)
    # A window starting 2025-12 (forecast 13 November): 2025-07 has settled.
    assert brs.last_observed_value(con, "SDN", "ACE", "PA", "2025-12")[:2] == (500.0, "2025-07")
    vec = persistence_spd(0.0, "PA")
    assert vec and abs(sum(vec) - 1.0) < 1e-9 and vec[0] == max(vec)
