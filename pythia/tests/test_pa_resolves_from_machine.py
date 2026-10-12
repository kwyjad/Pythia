# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""FL/PA and TC/PA resolve from the people-affected resolution machine.

Owner decision 2026-10-12: flood and cyclone people-affected questions read
``haz_resolutions``, not the IFRC GO rows in ``facts_resolved``. On the
release of 12 Oct 2026 the IFRC path resolved 9 of 66 due FL/PA horizon
months and 1 of 33 TC/PA; the machine held an answer for most of the rest.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb")

from pythia.tests.test_compute_resolutions import _setup_e2e_db
from pythia.tools.compute_resolutions import (
    compute_resolutions,
    haz_rung_and_value_type,
)
from pythia.tools.source_coverage import _COVERAGE_QUERIES
from resolver.hazard_resolution.schema import ensure_haz_schema

TODAY = date(2026, 10, 12)


def _db(tmp_path: Path, monkeypatch, *, with_machine: bool = True):
    db_path = tmp_path / "pa_machine.duckdb"
    db_url = f"duckdb:///{db_path}"
    monkeypatch.setattr(
        "pythia.tools.compute_resolutions.load_cfg",
        lambda: {"app": {"db_url": db_url}},
    )
    con = duckdb.connect(str(db_path))
    _setup_e2e_db(con)
    if with_machine:
        ensure_haz_schema(con)
    # One FL question whose window opens in August 2026: h1 = 2026-08,
    # h2 = 2026-09, everything later is after the calendar cutoff.
    con.execute(
        """
        INSERT INTO questions (question_id, hs_run_id, iso3, hazard_code, metric,
                               target_month, window_start_date, status)
        VALUES ('MLI_FL_PA_2026-08', 'run1', 'MLI', 'FL', 'PA', '2027-01',
                '2026-08-01', 'active')
        """
    )
    return con, db_path, db_url


def _haz(con, *, iso3="MLI", hazard="FL", year=2026, month=8, status, value=None,
         rule="ladder:reliefweb_extracted", rung="reliefweb_extracted",
         lower=False, provisional=True):
    prov = {"source": rung, "retrieved_at": "2026-10-11T03:00:00",
            "rule_fired": rule, "winning_rung": rung,
            "value_is_lower_bound": lower, "decision": {"flags": []}}
    if status != "RESOLVED_VALUE":
        prov = {"source": "detection:absence", "rule_fired": rule}
    con.execute(
        "DELETE FROM haz_resolutions WHERE iso3=? AND hazard=? AND year=? AND month=?",
        [iso3, hazard, year, month],
    )
    con.execute(
        """
        INSERT INTO haz_resolutions (iso3, year, month, hazard, status, value,
                                     provenance_json, rule_fired, flagged,
                                     provisional, run_type)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, FALSE, ?, 'live')
        """,
        [iso3, year, month, hazard, status, value, json.dumps(prov), rule, provisional],
    )


def _ifrc(con, ym, value):
    con.execute(
        "INSERT INTO facts_resolved (ym, iso3, hazard_code, metric, value, publisher) "
        "VALUES (?, 'MLI', 'FL', 'affected', ?, 'IFRC')",
        [ym, value],
    )


def _resolutions(db_path):
    con = duckdb.connect(str(db_path))
    try:
        return {
            h: (v, s)
            for h, v, s in con.execute(
                "SELECT horizon_m, value, source_desc FROM resolutions ORDER BY horizon_m"
            ).fetchall()
        }
    finally:
        con.close()


def _vintages(db_path):
    con = duckdb.connect(str(db_path))
    try:
        return con.execute(
            "SELECT horizon_m, vintage, value FROM resolution_vintages "
            "ORDER BY horizon_m, vintage"
        ).fetchall()
    finally:
        con.close()


@pytest.mark.db
def test_a_machine_value_resolves_and_names_its_rung(tmp_path, monkeypatch):
    con, path, url = _db(tmp_path, monkeypatch)
    _haz(con, status="RESOLVED_VALUE", value=42000.0)
    con.close()
    compute_resolutions(url, today=TODAY)
    assert _resolutions(path)[1] == (
        42000.0, "haz_resolutions:reliefweb_extracted:affected"
    )


@pytest.mark.db
def test_a_machine_zero_resolves_to_zero(tmp_path, monkeypatch):
    con, path, url = _db(tmp_path, monkeypatch)
    _haz(con, status="RESOLVED_ZERO", value=0.0,
         rule="flood_zero:no_gdacs_trigger+reliefweb_silent")
    con.close()
    compute_resolutions(url, today=TODAY)
    assert _resolutions(path)[1] == (0.0, "haz_resolutions:detection_absence:zero")


@pytest.mark.db
def test_no_data_and_no_row_stay_unresolved(tmp_path, monkeypatch):
    con, path, url = _db(tmp_path, monkeypatch)
    _haz(con, month=8, status="NO_DATA", rule="no_data:no_candidate_on_any_rung")
    # 2026-09 has no machine row at all (pending before the freeze deadline).
    con.close()
    compute_resolutions(url, today=TODAY)
    assert _resolutions(path) == {}


@pytest.mark.db
def test_machine_silence_never_falls_back_to_ifrc(tmp_path, monkeypatch):
    """The machine's ladder already read IFRC GO; its silence is the verdict."""
    con, path, url = _db(tmp_path, monkeypatch)
    _ifrc(con, "2026-08", 12000.0)
    _ifrc(con, "2026-09", 3000.0)
    _haz(con, month=8, status="NO_DATA", rule="no_data:no_candidate_on_any_rung")
    con.close()
    compute_resolutions(url, today=TODAY)
    assert _resolutions(path) == {}


@pytest.mark.db
def test_a_row_an_earlier_path_wrote_is_retracted(tmp_path, monkeypatch):
    con, path, url = _db(tmp_path, monkeypatch, with_machine=False)
    _ifrc(con, "2026-08", 12000.0)
    con.close()
    compute_resolutions(url, today=TODAY)
    assert _resolutions(path)[1][1].startswith("facts_resolved:IFRC")
    con = duckdb.connect(str(path))
    ensure_haz_schema(con)
    con.close()
    compute_resolutions(url, today=TODAY)
    assert _resolutions(path) == {}


@pytest.mark.db
def test_without_the_machine_table_the_old_cascade_runs(tmp_path, monkeypatch):
    con, path, url = _db(tmp_path, monkeypatch, with_machine=False)
    _ifrc(con, "2026-08", 12000.0)
    con.close()
    compute_resolutions(url, today=TODAY)
    assert _resolutions(path)[1] == (12000.0, "facts_resolved:IFRC:affected")


@pytest.mark.db
def test_a_provisional_revision_replaces_the_resolution_and_keeps_the_first(
    tmp_path, monkeypatch,
):
    con, path, url = _db(tmp_path, monkeypatch)
    _haz(con, status="RESOLVED_VALUE", value=10000.0, provisional=True)
    con.close()
    compute_resolutions(url, today=TODAY)
    con = duckdb.connect(str(path))
    _haz(con, status="RESOLVED_VALUE", value=25000.0, provisional=False)
    con.close()
    compute_resolutions(url, today=TODAY)
    assert _resolutions(path)[1][0] == 25000.0
    assert _vintages(path) == [(1, "at_freeze", 25000.0), (1, "first", 10000.0)]


@pytest.mark.db
def test_a_displacement_figure_is_scored_as_a_lower_bound(tmp_path, monkeypatch):
    con, path, url = _db(tmp_path, monkeypatch)
    _haz(con, status="RESOLVED_VALUE", value=1500.0, rung="idmc_idu",
         rule="ladder:idmc_idu:lower_bound", lower=True)
    con.close()
    compute_resolutions(url, today=TODAY)
    assert _resolutions(path)[1] == (
        1500.0, "haz_resolutions:idmc_idu:displaced_lower_bound"
    )


@pytest.mark.db
def test_cyclone_reads_the_machine_and_drought_does_not(tmp_path, monkeypatch):
    con, path, url = _db(tmp_path, monkeypatch)
    con.execute(
        """
        INSERT INTO questions (question_id, hs_run_id, iso3, hazard_code, metric,
                               target_month, window_start_date, status)
        VALUES ('MLI_TC_PA_2026-08', 'run1', 'MLI', 'TC', 'PA', '2027-01',
                '2026-08-01', 'active'),
               ('MLI_DR_PA_2026-08', 'run1', 'MLI', 'DR', 'PA', '2027-01',
                '2026-08-01', 'active')
        """
    )
    _haz(con, hazard="TC", status="RESOLVED_VALUE", value=900.0, rung="ifrc_go",
         rule="ladder:ifrc_go")
    _haz(con, hazard="DR", status="RESOLVED_VALUE", value=77.0, rung="ipc",
         rule="drought:ipc_delta")
    con.close()
    compute_resolutions(url, today=TODAY)
    con = duckdb.connect(str(path))
    rows = dict(con.execute(
        "SELECT question_id, source_desc FROM resolutions WHERE horizon_m = 1"
    ).fetchall())
    con.close()
    assert rows.get("MLI_TC_PA_2026-08") == "haz_resolutions:ifrc_go:affected"
    assert "MLI_DR_PA_2026-08" not in rows


def test_rung_falls_back_to_rule_fired():
    assert haz_rung_and_value_type("{}", "ladder:idmc_idu:lower_bound",
                                   "RESOLVED_VALUE") == ("idmc_idu", "displaced_lower_bound")
    assert haz_rung_and_value_type(None, "ladder:emdat", "RESOLVED_VALUE") == (
        "emdat", "affected")


def test_flood_and_cyclone_pa_never_zero_default():
    """FL/TC PA are absent from the zero-default coverage gates."""
    from pythia.tools.compute_resolutions import _should_default_to_zero

    for hz in ("FL", "TC"):
        assert not _should_default_to_zero("PA", hz)
    assert "PA" not in _COVERAGE_QUERIES
