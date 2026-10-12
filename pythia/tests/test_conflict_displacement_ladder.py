# Pythia / Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
"""The ACE/PA conflict displacement ladder: IDMC, then an extracted
OCHA/UN figure, then the DTM increase as a lower bound (owner decision
2026-10-12). ``base_rate_spd.resolve_conflict_month_detail`` is the one rule."""

from __future__ import annotations

from datetime import date

from pythia.tools.base_rate_spd import (
    CONFLICT_FLAG_OOM,
    CONFLICT_RUNG_DTM,
    CONFLICT_RUNG_EXTRACTED,
    CONFLICT_RUNG_IDMC,
    CONFLICT_STATUS_HELD,
    CONFLICT_STATUS_IRREGULAR,
    CONFLICT_STATUS_QUIET,
    CONFLICT_STATUS_REPORTED,
    CONFLICT_STATUS_UNSETTLED,
    ConflictReading,
    conflict_ladder_months,
    conflict_regular_reporter,
    resolve_conflict_month,
    resolve_conflict_month_detail,
)

LATE = date(2026, 12, 31)  # 2026-08 settled for IDMC (90 days) and extraction


def _ex(value, avail=date(2026, 9, 30), ref="rw:1"):
    return ConflictReading(CONFLICT_RUNG_EXTRACTED, value, avail, False, ref)


def _dtm(value, avail=date(2026, 9, 1)):
    return ConflictReading(CONFLICT_RUNG_DTM, value, avail, True, "dtm:r1")


def test_idmc_wins_when_settled():
    r = resolve_conflict_month_detail({"2026-08": 5000}, "2026-08", LATE, lower=[_ex(6000), _dtm(900)])
    assert (r.value, r.status, r.rung, r.lower_bound) == (5000.0, CONFLICT_STATUS_REPORTED,
                                                          CONFLICT_RUNG_IDMC, False)
    assert r.source_desc.startswith("facts_resolved:IDMC")
    assert r.flags == ()


def test_extracted_answers_a_month_idmc_has_not_settled():
    today = date(2026, 10, 12)  # 2026-08 not settled for IDMC, extracted available
    r = resolve_conflict_month_detail({"2026-08": 5000}, "2026-08", today, lower=[_ex(6000)])
    assert (r.value, r.rung) == (6000.0, CONFLICT_RUNG_EXTRACTED)
    # once IDMC settles, it replaces the extracted figure
    r2 = resolve_conflict_month_detail({"2026-08": 5000}, "2026-08", LATE, lower=[_ex(6000)])
    assert r2.rung == CONFLICT_RUNG_IDMC


def test_extracted_is_not_read_before_it_is_available():
    today = date(2026, 9, 15)
    r = resolve_conflict_month_detail({}, "2026-08", today, lower=[_ex(6000)])
    assert r.status == CONFLICT_STATUS_UNSETTLED


def test_dtm_answers_a_held_out_idmc_month_and_is_a_lower_bound():
    r = resolve_conflict_month_detail({"2026-07": 10}, "2026-08", LATE, held={"2026-08"},
                                      lower=[_dtm(42000)])
    assert (r.value, r.rung, r.lower_bound) == (42000.0, CONFLICT_RUNG_DTM, True)
    assert "lower_bound" in r.source_desc
    assert resolve_conflict_month_detail({}, "2026-08", LATE, held={"2026-08"}).status == CONFLICT_STATUS_HELD


def test_extracted_outranks_dtm_and_disagreement_flags_without_substituting():
    r = resolve_conflict_month_detail({}, "2026-08", LATE, lower=[_dtm(500), _ex(80000)])
    assert r.rung == CONFLICT_RUNG_EXTRACTED and r.value == 80000.0
    assert r.flags == (CONFLICT_FLAG_OOM,)
    near = resolve_conflict_month_detail({}, "2026-08", LATE, lower=[_dtm(5000), _ex(8000)])
    assert near.flags == ()


def test_zero_stays_the_idmc_regular_reporter_bracket_rule():
    reported = {f"2025-{m:02d}": 100.0 for m in range(8, 13)}
    reported.update({f"2026-{m:02d}": 100.0 for m in range(1, 8)})
    reported["2026-09"] = 50.0
    assert conflict_regular_reporter(reported, "2026-08")
    assert resolve_conflict_month(reported, "2026-08", LATE) == (0.0, CONFLICT_STATUS_QUIET)
    # a lower-rung figure is a reported figure, never overwritten by a zero
    r = resolve_conflict_month_detail(reported, "2026-08", LATE, lower=[_ex(700)])
    assert (r.value, r.rung) == (700.0, CONFLICT_RUNG_EXTRACTED)
    # an irregular IDMC reporter's missing month stays unknown with no lower rung
    assert resolve_conflict_month({"2026-07": 1.0}, "2026-08", LATE)[1] == CONFLICT_STATUS_IRREGULAR


def test_regularity_under_the_ladder_counts_any_rung():
    idmc = {"2026-01": 1.0, "2026-02": 1.0}
    lower = {f"2025-{m:02d}": [_ex(10.0)] for m in range(9, 13)}
    lower.update({"2026-03": [_dtm(5.0)], "2026-04": [_dtm(5.0)]})
    union = conflict_ladder_months(idmc, lower)
    assert not conflict_regular_reporter(idmc, "2026-07")
    assert conflict_regular_reporter(union, "2026-07")
    assert union["2026-01"] == 1.0  # IDMC wins its own month


def test_negative_and_future_lower_readings_are_ignored():
    r = resolve_conflict_month_detail({}, "2026-08", LATE,
                                      lower=[_ex(-5.0), _dtm(100, avail=date(2027, 3, 1))])
    assert r.status == CONFLICT_STATUS_IRREGULAR


def test_the_resolver_writes_the_answering_rung_and_scores_under_the_ladder(tmp_path, monkeypatch):
    """compute_resolutions: an extracted figure resolves a month IDMC held
    out, its rung is in source_desc, and the scoring class counts any rung."""
    import duckdb

    from pythia.tests.test_ace_pa_conflict_displacement import _db, _flow
    from pythia.tools import compute_resolutions as cr
    from resolver.ingestion import conflict_displacement_extract as cde

    db = tmp_path / "ladder.duckdb"
    db_url = f"duckdb:///{db}"
    monkeypatch.setattr(cr, "load_cfg", lambda: {"app": {"db_url": db_url}})
    con = _db(db)
    _flow(con, "2025-01", "COD", 500)
    cde.ensure_schema(con)
    # COD: IDMC holds out its multi-month records; OCHA reported each month.
    for m in ("2024-01", "2024-02", "2024-03", "2024-04", "2024-05", "2024-06",
              "2024-07", "2024-08", "2024-09", "2024-10", "2024-11", "2024-12",
              "2025-02", "2025-03"):
        con.execute(
            "INSERT INTO conflict_displacement_candidates (iso3, ym, source_ref, "
            "preference_rank, value_type, value_persons, quote) VALUES "
            "('COD', ?, 'rw:' || ?, 0, 'new_displacements', 40000, 'q')",
            [m, m],
        )
    con.execute(
        "INSERT INTO questions VALUES ('COD_ACE_PA_2025-02', 'run1', 'COD', 'ACE', 'PA', "
        "'2025-07', DATE '2025-02-01', 'active', FALSE)"
    )
    con.close()
    cr.compute_resolutions(db_url=db_url, today=date(2025, 10, 1))
    con = duckdb.connect(str(db))
    rows = con.execute(
        "SELECT horizon_m, value, source_desc, scoring_class FROM resolutions "
        "ORDER BY horizon_m"
    ).fetchall()
    con.close()
    assert rows[0][:3] == (1, 40000.0, "conflict_displacement_candidates:reliefweb_extracted:new_displacements")
    assert rows[1][:2] == (2, 40000.0)
    # Regular under the ladder (12 of the 12 months before 2025-02 carry a figure).
    assert rows[0][3] == "scored"
