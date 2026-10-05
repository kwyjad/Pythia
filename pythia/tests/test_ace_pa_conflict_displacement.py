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
def test_ace_pa_resolves_from_the_conflict_series_with_both_gates(tmp_path, monkeypatch):
    db = tmp_path / "r.duckdb"
    db_url = f"duckdb:///{db}"
    monkeypatch.setattr(cr, "load_cfg", lambda: {"app": {"db_url": db_url}})
    con = _db(db)
    _flow(con, "2025-08", "SDN", 9000)
    _flow(con, "2025-09", "SOM", 300)          # makes September live
    _flow(con, "2025-08", "CHN", 7_305_385, hazard="IDU")  # all-cause: never read
    for qid, iso in (("SDN_ACE_PA_2025-08", "SDN"), ("CHN_ACE_PA_2025-08", "CHN")):
        con.execute(
            "INSERT INTO questions VALUES (?, 'run1', ?, 'ACE', 'PA', '2026-01', "
            "DATE '2025-08-01', 'active', FALSE)",
            [qid, iso],
        )
    con.close()

    cr.compute_resolutions(db_url=db_url)

    con = duckdb.connect(str(db))
    rows = con.execute(
        "SELECT question_id, horizon_m, observed_month, value, source_desc "
        "FROM resolutions ORDER BY question_id, horizon_m"
    ).fetchall()
    con.close()
    assert rows == [
        # A reported month: its value, from the one series.
        ("SDN_ACE_PA_2025-08", 1, "2025-08", 9000.0, brs.CONFLICT_DISPLACEMENT_SERIES),
        # A live month, a country IDMC reports on, no row: an observed zero.
        ("SDN_ACE_PA_2025-08", 2, "2025-09", 0.0, "zero_default"),
    ]
    # October onwards: no country reported, so not live; never a zero.
    # China: outside the conflict series' universe; its typhoon row is IDU.


def test_the_writer_and_the_reader_name_one_series():
    from resolver.ingestion import idmc_conflict as ic

    assert ic.HAZARD_CODE == brs.CONFLICT_DISPLACEMENT_HAZARD == "ACE"
    assert ic.METRIC == brs.CONFLICT_DISPLACEMENT_METRIC
    assert ic.SOURCE == brs.CONFLICT_DISPLACEMENT_PUBLISHER


def test_persistence_reads_the_month_before_the_window_as_the_resolver_would(tmp_path):
    from pythia.tools.score_baselines import PERSISTENCE_PAIRS, persistence_spd

    assert ("ACE", "PA") in PERSISTENCE_PAIRS
    con = _db(tmp_path / "p.duckdb")
    _flow(con, "2025-06", "SDN", 9000)
    _flow(con, "2025-07", "SOM", 300)
    # July is live and SDN is in the universe: an observed quiet month.
    last = brs.last_observed_value(con, "SDN", "ACE", "PA", "2025-08")
    assert last == (0.0, "2025-07", f"{brs.CONFLICT_DISPLACEMENT_SERIES}:quiet_month")
    # SOM reported July: its value.
    assert brs.last_observed_value(con, "SOM", "ACE", "PA", "2025-08")[:2] == (300.0, "2025-07")
    vec = persistence_spd(0.0, "PA")
    assert vec and abs(sum(vec) - 1.0) < 1e-9 and vec[0] == max(vec)
