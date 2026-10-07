# Pythia / Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
"""Conflict displacement questions that cannot be scored fairly (Oct 2026).

For a country IDMC does not report every month, a month resolves only when
IDMC reports it, so its resolved months are a selected sample. Each ACE/PA
resolution row is classed scored or indicative from the regular-reporter
rule as it stands for that month; later readings at ~180 and ~270 days are
kept as FATALITIES keeps 60 and 90.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
import pytest

from pythia.tools import compute_resolutions as cr
from pythia.tools.scoring_class import (
    INDICATIVE,
    SCORED,
    conflict_scoring_class,
    scored_only_clause,
)


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


def _flow(con, ym, iso3, value):
    con.execute(
        "INSERT INTO facts_resolved (ym, iso3, hazard_code, metric, series_semantics, value, "
        "publisher, source_id) VALUES (?, ?, 'ACE', 'new_displacements', 'new', ?, 'IDMC', 'idmc')",
        [ym, iso3, value],
    )


def test_the_class_follows_the_regular_reporter_rule():
    regular = {f"2025-{m:02d}": 1.0 for m in range(1, 13)}
    assert conflict_scoring_class(regular, "2026-01") == (SCORED, None)
    afg = {"2025-02": 7369.0, "2025-10": 160800.0}
    cls, reason = conflict_scoring_class(afg, "2025-11")
    assert cls == INDICATIVE and "fewer than 8 of the 12 months before 2025-11" in reason


@pytest.mark.db
def test_resolutions_carry_the_class_and_readers_can_leave_indicative_out(tmp_path, monkeypatch):
    db = tmp_path / "r.duckdb"
    db_url = f"duckdb:///{db}"
    monkeypatch.setattr(cr, "load_cfg", lambda: {"app": {"db_url": db_url}})
    con = _db(db)
    for y in (2024, 2025):
        for m in range(1, 13):
            _flow(con, f"{y}-{m:02d}", "SDN", 9000)          # regular
    for ym in ("2024-11", "2025-03", "2025-05"):
        _flow(con, ym, "AFG", 5000)                          # irregular
    for qid, iso in (("SDN_ACE_PA_2025-03", "SDN"), ("AFG_ACE_PA_2025-03", "AFG")):
        con.execute(
            "INSERT INTO questions VALUES (?, 'run1', ?, 'ACE', 'PA', '2025-08', "
            "DATE '2025-03-01', 'active', FALSE)",
            [qid, iso],
        )
    con.close()

    cr.compute_resolutions(db_url=db_url, today=date(2026, 3, 1))

    con = duckdb.connect(str(db))
    rows = con.execute(
        "SELECT question_id, horizon_m, scoring_class, scoring_class_reason FROM resolutions "
        "ORDER BY question_id, horizon_m"
    ).fetchall()
    assert {r[2] for r in rows if r[0].startswith("SDN")} == {SCORED}
    afg = [r for r in rows if r[0].startswith("AFG")]
    assert [r[1] for r in afg] == [1, 3]                     # only reported months resolve
    assert {r[2] for r in afg} == {INDICATIVE}
    assert all("not a regular IDMC reporter" in r[3] for r in afg)

    con.execute(
        "CREATE TABLE scores (question_id TEXT, horizon_m INTEGER, model_name TEXT, value DOUBLE)"
    )
    for qid, h in con.execute("SELECT question_id, horizon_m FROM resolutions").fetchall():
        con.execute("INSERT INTO scores VALUES (?, ?, 'm', 0.5)", [qid, h])
    kept = con.execute(
        f"SELECT DISTINCT split_part(question_id, '_', 1) FROM scores s "
        f"WHERE TRUE{scored_only_clause(con, 's')}"
    ).fetchall()
    con.close()
    assert kept == [("SDN",)]


def test_ace_pa_keeps_readings_at_first_180_and_270_days(tmp_path):
    con = duckdb.connect(str(tmp_path / "v.duckdb"))
    cr._ensure_vintage_table(con)
    milestones = cr.vintage_milestones("ACE", "PA")
    assert milestones == (("d180", 180), ("d270", 270))
    assert cr.vintage_milestones("ACE", "FATALITIES") == cr.VINTAGE_MILESTONES
    assert cr.vintage_milestones("FL", "PA") is None
    kw = dict(question_id="Q", horizon_m=1, observed_month="2025-01", source_desc="s",
              source_ts=None, is_test=False, milestones=milestones)
    assert cr.record_vintages(con, value=10.0, today=date(2025, 5, 1), **kw) == ["first"]
    assert cr.record_vintages(con, value=12.0, today=date(2025, 8, 1), **kw) == ["d180"]
    assert cr.record_vintages(con, value=13.0, today=date(2025, 11, 1), **kw) == ["d270"]
    values = dict(con.execute("SELECT vintage, value FROM resolution_vintages").fetchall())
    con.close()
    assert values == {"first": 10.0, "d180": 12.0, "d270": 13.0}
    assert cr.reading_label("ACE", "PA", "2025-01", date(2025, 5, 1)) == "first"
    assert cr.reading_label("ACE", "PA", "2025-01", date(2025, 11, 1)) == "d270"
    assert cr.reading_label("ACE", "FATALITIES", "2025-01", date(2025, 4, 10)) == "d60"


def test_the_report_and_the_page_say_an_indicative_forecast_cannot_be_marked():
    from interpreter import packs, render
    from pythia.api.routes.interpreter import _stamp_scoring_notes

    extras = {"scoring_notes": {"AFG_ACE_PA_2026-11": packs.INDICATIVE_NOTE}}
    resolver = render.FigureResolver(per_question={}, global_figures={})
    entry = {"rank": 1, "iso3": "AFG", "hazard_code": "ACE", "metric": "PA",
             "question_ids": ["AFG_ACE_PA_2026-11"]}
    text = "\n".join(render._render_entry(entry, resolver, extras=extras))
    assert "cannot be marked: IDMC does not report this country every month" in text
    other = dict(entry, iso3="SDN", question_ids=["SDN_ACE_PA_2026-11"])
    assert "cannot be marked" not in "\n".join(render._render_entry(other, resolver, extras=extras))
    content = {"attention": [dict(entry), dict(other)]}
    _stamp_scoring_notes(content, {"extras": extras})
    assert content["attention"][0]["scoring_note"] == packs.INDICATIVE_NOTE
    assert "scoring_note" not in content["attention"][1]
