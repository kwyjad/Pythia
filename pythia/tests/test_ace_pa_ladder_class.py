# Pythia / Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
"""ACE/PA under the ladder, Part H of the 13 Oct 2026 follow-up.

* A month answered by a DTM lower bound alone resolves but never counts
  toward the regular-reporter rule.
* A country regular only once the extracted OCHA/UN figures are counted is
  the ``ace_pa_ladder`` scoring class, kept out of every learning and skill
  reader until ``PYTHIA_ACE_PA_LADDER_MIN_RESOLVED`` (6) such horizon-months
  have resolved.
* The rung that admitted a country is stored and reported.
"""

from __future__ import annotations

from datetime import date

import pytest

from pythia.tools import scoring_class as sc
from pythia.tools.base_rate_spd import (
    CONFLICT_RUNG_DTM,
    CONFLICT_RUNG_EXTRACTED,
    CONFLICT_RUNG_IDMC,
    ConflictReading,
    conflict_admission_rung,
    conflict_regular_months,
    conflict_regular_reporter,
)


def _ex(value):
    return ConflictReading(CONFLICT_RUNG_EXTRACTED, value, date(2026, 1, 1), False, "rw:1")


def _dtm(value):
    return ConflictReading(CONFLICT_RUNG_DTM, value, date(2026, 1, 1), True, "dtm:r1")


def _months(start_y, start_m, n):
    out, y, m = [], start_y, start_m
    for _ in range(n):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return out


WINDOW = "2026-09"
LAST12 = _months(2025, 9, 12)  # 2025-09 .. 2026-08


def test_a_dtm_only_month_does_not_count_toward_regularity():
    idmc = {ym: 100.0 for ym in LAST12[:6]}
    lower = {ym: [_dtm(50.0)] for ym in LAST12[6:]}
    counted = conflict_regular_months(idmc, lower)
    assert set(counted) == set(LAST12[:6])
    assert not conflict_regular_reporter(counted, WINDOW)
    assert conflict_admission_rung(idmc, lower, WINDOW) is None


def test_the_admission_rung_says_which_rung_was_needed():
    idmc_regular = {ym: 100.0 for ym in LAST12[:9]}
    assert conflict_admission_rung(idmc_regular, {}, WINDOW) == CONFLICT_RUNG_IDMC
    idmc_short = {ym: 100.0 for ym in LAST12[:4]}
    lower = {ym: [_ex(70.0)] for ym in LAST12[4:9]}
    assert conflict_admission_rung(idmc_short, lower, WINDOW) == CONFLICT_RUNG_EXTRACTED


def test_the_scoring_class_has_three_values():
    idmc_short = {ym: 100.0 for ym in LAST12[:4]}
    lower = {ym: [_ex(70.0)] for ym in LAST12[4:9]}
    counted = conflict_regular_months(idmc_short, lower)
    assert sc.conflict_scoring_class(counted, WINDOW, idmc_reported=idmc_short)[0] == sc.LADDER
    assert sc.conflict_scoring_class(counted, WINDOW)[0] == sc.SCORED  # no IDMC series given
    regular = {ym: 100.0 for ym in LAST12}
    assert sc.conflict_scoring_class(regular, WINDOW, idmc_reported=regular)[0] == sc.SCORED
    assert sc.conflict_scoring_class(idmc_short, WINDOW, idmc_reported=idmc_short)[0] == sc.INDICATIVE


def _db(tmp_path, n_ladder):
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect(str(tmp_path / "s.duckdb"))
    con.execute("CREATE TABLE resolutions (question_id TEXT, horizon_m INTEGER, scoring_class TEXT)")
    con.execute("CREATE TABLE scores (question_id TEXT, horizon_m INTEGER, value DOUBLE)")
    rows = [("Q_SCORED", 1, "scored"), ("Q_IND", 1, "indicative")]
    rows += [("Q_LAD", h, sc.LADDER) for h in range(1, n_ladder + 1)]
    con.executemany("INSERT INTO resolutions VALUES (?,?,?)", rows)
    con.executemany("INSERT INTO scores VALUES (?,?,0.1)", [(q, h) for q, h, _ in rows])
    return con


def _kept(con):
    return sorted({q for (q,) in con.execute(
        f"SELECT question_id FROM scores s WHERE {sc.scored_only_sql('s')}").fetchall()})


def test_the_ladder_class_is_held_out_until_enough_have_resolved(tmp_path, monkeypatch):
    monkeypatch.delenv("PYTHIA_ACE_PA_LADDER_MIN_RESOLVED", raising=False)
    con = _db(tmp_path, 5)
    assert _kept(con) == ["Q_SCORED"]
    assert sc.excluded_classes(con) == {sc.INDICATIVE, sc.LADDER}
    con.execute("INSERT INTO resolutions VALUES ('Q_LAD', 6, 'ace_pa_ladder')")
    con.execute("INSERT INTO scores VALUES ('Q_LAD', 6, 0.1)")
    assert _kept(con) == ["Q_LAD", "Q_SCORED"]
    assert sc.excluded_classes(con) == {sc.INDICATIVE}


def test_the_threshold_is_env_overridable(tmp_path, monkeypatch):
    con = _db(tmp_path, 2)
    monkeypatch.setenv("PYTHIA_ACE_PA_LADDER_MIN_RESOLVED", "2")
    assert _kept(con) == ["Q_LAD", "Q_SCORED"]
    monkeypatch.setenv("PYTHIA_ACE_PA_LADDER_MIN_RESOLVED", "nonsense")
    assert sc.ladder_min_resolved() == 6


def test_the_interpreter_skill_split_leaves_the_ladder_class_out():
    from interpreter.performance import split_indicative

    kept, dropped = split_indicative([(0.1, 0.2, "scored"), (0.1, 0.2, "ace_pa_ladder"),
                                      (0.1, 0.2, "indicative")])
    assert kept == [(0.1, 0.2)] and dropped == 2


def test_the_eligibility_summary_names_the_rung():
    from pythia.tools.ace_pa_eligibility import admission_rung_summary

    text = admission_rung_summary({"SOM": "idmc", "COD": "reliefweb_extracted", "UKR": "idmc"})
    assert text == "; admitted on IDMC: SOM, UKR; on extracted OCHA/UN figures: COD"
    assert admission_rung_summary({}) == ""


def test_the_rung_table_renders_and_never_fails(tmp_path):
    from scripts.ci import ace_pa_rung_table as rt

    text = rt.render("2026-12", [
        {"iso3": "COD", "months_idmc": 2, "months_extracted_only": 7, "months_dtm_only": 2,
         "months_counted": 9, "regular_idmc_alone": False, "admission_rung": "reliefweb_extracted"},
        {"iso3": "SOM", "months_idmc": 10, "months_extracted_only": 0, "months_dtm_only": 0,
         "months_counted": 10, "regular_idmc_alone": True, "admission_rung": "idmc"},
    ], {"n_questions": 40, "n_revived": 3, "horizon_months_revived": 7, "by_rung": {}})
    assert "Admitted: 2 (1 on IDMC, 1 on extracted figures)." in text
    assert "Revived questions: 3 of 40" in text
    assert rt.main(["--db", str(tmp_path / "missing.duckdb"), "--window", "2026-12"]) == 0


def test_the_lower_rungs_read_on_a_read_only_connection(tmp_path):
    """The two readers ran ensure_schema (a CREATE) first, so on a read-only
    connection they failed and returned nothing: the ladder read empty for
    every read-only caller (found 2026-10-14 on the backfill's DB)."""
    import duckdb

    from pythia.tools.base_rate_spd import conflict_ladder_lower
    from resolver.ingestion import conflict_displacement_extract as cde
    from resolver.ingestion import dtm_conflict as dtm

    path = tmp_path / "ro.duckdb"
    con = duckdb.connect(str(path))
    cde.ensure_schema(con)
    dtm.ensure_schema(con)
    cols = [r[0] for r in con.execute("DESCRIBE conflict_displacement_candidates").fetchall()]
    row = {c: None for c in cols if c != "created_at"}
    row.update(iso3="YEM", ym="2026-05", source_ref="d1", preference_rank=0,
               value_type="new_displacements", value_persons=1200.0, quote="1,200 people")
    con.execute(
        f"INSERT INTO conflict_displacement_candidates ({', '.join(row)}) "
        f"VALUES ({', '.join('?' for _ in row)})", list(row.values()))
    con.execute(
        "INSERT INTO dtm_conflict_lower_bounds (iso3, ym, value, value_type, pairs_json, computed_at) "
        "VALUES ('SDN', '2026-04', 500, 'displaced_lower_bound', '[]', now())")
    con.close()

    ro = duckdb.connect(str(path), read_only=True)
    try:
        assert ("YEM", "2026-05") in cde.extracted_figures(ro)
        assert ("SDN", "2026-04") in dtm.dtm_lower_bounds(ro)
        low = conflict_ladder_lower(ro)
        assert "YEM" in low and "SDN" in low
    finally:
        ro.close()
