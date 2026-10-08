# Pythia / Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
"""The report says where conflict displacement is not forecast (Oct 2026).

ACE/PA is asked only where IDMC reports regularly. A conflict country with no
displacement question in the run is either not forecast for the window, or
was not asked again because an earlier run of the epoch asked it. The report
says which, first among its caveats, and the sector comparison says a score
that leaves displacement out does so.
"""

from __future__ import annotations

import pytest

duckdb = pytest.importorskip("duckdb")

from interpreter.render import _sector_lines
from scripts.ai_bundle import build_current_run_bundle as b


@pytest.fixture()
def con():
    c = duckdb.connect(":memory:")
    c.execute("CREATE TABLE questions (question_id TEXT)")
    # Sudan's displacement question exists from an earlier run of the epoch.
    c.execute("INSERT INTO questions VALUES ('SDN_ACE_PA_2026-11')")
    yield c
    c.close()


ROWS = [
    {"question_id": "SOM_ACE_FATALITIES_2026-11", "hazard_code": "ACE", "metric": "FATALITIES"},
    {"question_id": "SOM_ACE_PA_2026-11", "hazard_code": "ACE", "metric": "PA"},
    {"question_id": "SDN_ACE_FATALITIES_2026-11", "hazard_code": "ACE", "metric": "FATALITIES"},
    {"question_id": "AFG_ACE_FATALITIES_2026-11", "hazard_code": "ACE", "metric": "FATALITIES"},
    {"question_id": "AFG_FL_PA_2026-11", "hazard_code": "FL", "metric": "PA"},
]


def test_conflict_countries_without_displacement_are_named(con):
    cov = b.displacement_coverage(con, ROWS)
    assert cov["not_forecast"] == ["AFG"]
    assert cov["not_asked_again"] == ["SDN"]
    assert "It is not forecast for Afghanistan" in cov["sentence"]
    assert "not asked again in this run for Sudan" in cov["sentence"]
    assert "not a forecast of no displacement" in cov["sentence"]


def test_the_caveat_leads_the_blind_spots(con):
    cov = b.displacement_coverage(con, ROWS)
    spots = b.build_blind_spots(ROWS, displacement=cov)
    assert spots["standing_caveats"][0] == cov["sentence"]
    assert spots["conflict_displacement"] == {"not_forecast": ["AFG"], "not_asked_again": ["SDN"]}
    # A run where every conflict country has its question adds nothing.
    full = b.displacement_coverage(con, ROWS[:2])
    assert b.build_blind_spots(ROWS[:2], displacement=full)["standing_caveats"] == b.STANDING_CAVEATS


def test_the_sector_section_prints_the_note():
    block = {"available": True, "comparisons": [],
             "displacement_note": "Fred's score for Sudan leaves out conflict displacement."}
    assert "Fred's score for Sudan leaves out conflict displacement." in _sector_lines({"sector": block})
