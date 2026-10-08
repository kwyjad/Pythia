# Pythia / Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
"""ACE/PA is asked only where IDMC reports regularly (owner decision 2026-10-08).

From the 13 Oct 2026 run an ACE country gets an ACE/PA question only when it
is a regular IDMC reporter (8 of the 12 months before the window), by the
resolver's own function. ACE/FATALITIES is unchanged. If the IDMC series is
missing, stale or admits nobody, the previous production run's list is used;
with no previous list every ACE country is asked. Existing questions are
left alone.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb")

from pythia.db.schema import ensure_schema
from pythia.tools import ace_pa_eligibility as elig
from pythia.tools.base_rate_spd import _add_months
from scripts.create_questions_from_triage import _compute_target_and_window, create_questions_from_triage

WINDOW = _compute_target_and_window(date.today())[0]


def _facts(con, months_by_iso: dict[str, list[str]]) -> None:
    con.execute(
        "CREATE TABLE IF NOT EXISTS facts_resolved (ym TEXT, iso3 TEXT, hazard_code TEXT, metric TEXT, "
        "series_semantics TEXT, publisher TEXT, value DOUBLE, created_at TIMESTAMP)"
    )
    for iso, months in months_by_iso.items():
        for ym in months:
            con.execute(
                "INSERT INTO facts_resolved VALUES (?, ?, 'ACE', 'new_displacements', 'new', 'IDMC', 500, "
                "'2026-10-11')", [ym, iso],
            )


def _months_before(n_back: int, count: int) -> list[str]:
    """``count`` consecutive months ending ``n_back`` months before WINDOW."""
    end = _add_months(WINDOW, -n_back)
    return [_add_months(end, -i) for i in range(count)]


def _setup(tmp_path: Path, months_by_iso: dict[str, list[str]] | None, ace: list[str],
           hs_run_id: str = "hs_new") -> str:
    db = tmp_path / "q.duckdb"
    con = duckdb.connect(str(db))
    ensure_schema(con)
    if months_by_iso is not None:
        _facts(con, months_by_iso)
    con.execute("INSERT INTO hs_runs (hs_run_id, generated_at, is_test) VALUES (?, now(), FALSE)",
                [hs_run_id])
    for iso in ace:
        con.execute(
            "INSERT INTO hs_triage (run_id, iso3, hazard_code, tier, triage_score, need_full_spd, "
            "drivers_json, regime_shifts_json, data_quality_json, scenario_stub) "
            "VALUES (?, ?, 'ACE', 'priority', 0.9, TRUE, '[]', '[]', '{}', '')", [hs_run_id, iso],
        )
    con.close()
    return str(db)


def _questions(db: str) -> set[tuple[str, str]]:
    con = duckdb.connect(db)
    try:
        return {(r[0], r[1]) for r in con.execute(
            "SELECT iso3, metric FROM questions WHERE hazard_code = 'ACE'").fetchall()}
    finally:
        con.close()


def _decision(db: str, hs: str = "hs_new") -> dict:
    con = duckdb.connect(db)
    try:
        return elig.latest_decision(con, hs)
    finally:
        con.close()


REGULAR = _months_before(2, 10)       # 10 of the 12 months before the window
IRREGULAR = _months_before(2, 3)      # 3 of 12


def test_pa_is_asked_only_for_regular_reporters(tmp_path):
    db = _setup(tmp_path, {"SOM": REGULAR, "SDN": IRREGULAR}, ["SOM", "SDN", "AFG"])
    create_questions_from_triage(f"duckdb:///{db}", hs_run_id="hs_new")
    assert _questions(db) == {
        ("SOM", "FATALITIES"), ("SOM", "PA"),
        ("SDN", "FATALITIES"),
        ("AFG", "FATALITIES"),
    }
    d = _decision(db)
    assert d["source"] == elig.SOURCE_RULE
    assert d["asked"] == ["SOM"] and d["not_asked"] == ["AFG", "SDN"]


def test_the_rule_is_the_resolvers_function(tmp_path):
    """8 of 12 admits, 7 of 12 does not: the scoring-class threshold."""
    db = _setup(tmp_path, {"AAA": _months_before(1, 8), "BBB": _months_before(1, 7)}, [])
    con = duckdb.connect(db)
    d = elig.decide(con, window_start=WINDOW, hs_run_id="hs_new")
    con.close()
    assert d.countries == {"AAA"}


def test_existing_irregular_questions_are_left_alone(tmp_path):
    """A question asked before the rule keeps its row and forecast; this run
    neither re-links nor re-points it, so its earlier forecast stands."""
    db = _setup(tmp_path, {"SOM": REGULAR, "SDN": IRREGULAR}, ["SOM", "SDN"])
    con = duckdb.connect(db)
    con.execute("INSERT INTO hs_runs (hs_run_id, generated_at, is_test) VALUES ('hs_old', now() - INTERVAL 12 DAY, FALSE)")
    con.execute(
        "INSERT INTO questions (question_id, hs_run_id, iso3, hazard_code, metric, target_month, "
        "window_start_date, window_end_date, wording, status, is_test) VALUES "
        f"('SDN_ACE_PA_{WINDOW}', 'hs_old', 'SDN', 'ACE', 'PA', '', DATE '{WINDOW}-01', "
        f"DATE '{WINDOW}-28', 'w', 'active', FALSE)"
    )
    con.close()
    create_questions_from_triage(f"duckdb:///{db}", hs_run_id="hs_new")
    con = duckdb.connect(db)
    row = con.execute("SELECT hs_run_id, status FROM questions WHERE question_id = ?",
                      [f"SDN_ACE_PA_{WINDOW}"]).fetchone()
    linked = con.execute("SELECT COUNT(*) FROM run_questions WHERE hs_run_id = 'hs_new' "
                         "AND question_id = ?", [f"SDN_ACE_PA_{WINDOW}"]).fetchone()[0]
    con.close()
    assert row == ("hs_old", "active")
    assert linked == 0


@pytest.mark.parametrize("series, why", [
    (None, "missing"),
    ({"SOM": _months_before(7, 10)}, "stale"),
    ({"SOM": IRREGULAR}, "admits no country"),
])
def test_the_guard_uses_the_previous_production_list(tmp_path, series, why):
    db = _setup(tmp_path, series, ["SOM", "SDN", "AFG"])
    con = duckdb.connect(db)
    con.execute("INSERT INTO hs_runs (hs_run_id, generated_at, is_test) VALUES ('hs_prev', now() - INTERVAL 12 DAY, FALSE)")
    con.execute("INSERT INTO hs_runs (hs_run_id, generated_at, is_test) VALUES ('hs_test', now() - INTERVAL 2 DAY, TRUE)")
    for hs, iso, test in [("hs_prev", "SDN", False), ("hs_prev", "AFG", False), ("hs_test", "SOM", True)]:
        con.execute("INSERT INTO run_questions (hs_run_id, question_id, iso3, hazard_code, metric, is_test) "
                    "VALUES (?, ?, ?, 'ACE', 'PA', ?)", [hs, f"{iso}_ACE_PA_old", iso, test])
    con.close()
    create_questions_from_triage(f"duckdb:///{db}", hs_run_id="hs_new")
    assert {iso for iso, m in _questions(db) if m == "PA"} == {"SDN", "AFG"}
    d = _decision(db)
    assert d["source"] == elig.SOURCE_PREVIOUS_RUN
    assert d["previous_hs_run_id"] == "hs_prev"
    assert why in d["reason"]


def test_with_no_previous_list_every_ace_country_is_asked(tmp_path, capsys):
    db = _setup(tmp_path, None, ["SOM", "SDN"])
    create_questions_from_triage(f"duckdb:///{db}", hs_run_id="hs_new")
    assert {iso for iso, m in _questions(db) if m == "PA"} == {"SOM", "SDN"}
    assert _decision(db)["source"] == elig.SOURCE_ALL_ACE
    assert "::warning title=ACE/PA eligibility guard::" in capsys.readouterr().out


def test_the_decision_reaches_the_step_summary(tmp_path, monkeypatch):
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    db = _setup(tmp_path, {"SOM": REGULAR}, ["SOM", "SDN"])
    create_questions_from_triage(f"duckdb:///{db}", hs_run_id="hs_new")
    text = summary.read_text()
    assert "ACE/PA eligibility" in text and "not asked for 1" in text
    assert json.loads(json.dumps(_decision(db)["countries"])) == ["SOM"]


def test_the_debug_bundle_warns_when_the_guard_fired(tmp_path):
    from scripts.dump_pythia_debug_bundle import _ace_pa_eligibility_check

    db = _setup(tmp_path, None, ["SOM"])
    create_questions_from_triage(f"duckdb:///{db}", hs_run_id="hs_new")
    check = _ace_pa_eligibility_check(_decision(db))
    assert check["status"] == "WARN" and "guard:" in check["detail"]

    (tmp_path / "b").mkdir()
    db2 = _setup(tmp_path / "b", {"SOM": REGULAR}, ["SOM", "SDN"])
    create_questions_from_triage(f"duckdb:///{db2}", hs_run_id="hs_new")
    check = _ace_pa_eligibility_check(_decision(db2))
    assert check["status"] == "OK"
    assert "not asked for 1 (displacement not forecast)" in check["detail"]
    assert _ace_pa_eligibility_check({"absent": True})["status"] == "WARN"
