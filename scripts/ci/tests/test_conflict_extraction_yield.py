# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""scripts/ci/conflict_extraction_yield.py on a small fixture DB."""

from __future__ import annotations

import json
from datetime import date

import pytest

duckdb = pytest.importorskip("duckdb")

from resolver.ingestion import conflict_displacement_extract as cde  # noqa: E402
from scripts.ci import conflict_extraction_yield as cey  # noqa: E402

SETTINGS = {"max_docs_per_cell": 5, "publication_pad_days": 30, "prompt_version": "conflict_v1"}
TODAY = date(2026, 11, 27)


def _rej(reason, quote="", **kw):
    return {"reason": reason, "doc_id": "d", "value": 100, "unit": "people", "quote": quote, **kw}


def _ledger(con, iso3, ym, reason, *, docs=0, extracted=0, admitted=0, counts=None,
            rejected=None, fetch=None):
    cde.write_ledger(con, {
        "iso3": iso3, "ym": ym, "reason": reason, "docs_selected": docs,
        "figures_extracted": extracted, "figures_admitted": admitted,
        "rejected_counts_json": json.dumps(counts or {}),
        "rejected_json": json.dumps(rejected or []),
        "detail_json": json.dumps({"fetch": fetch} if fetch is not None else {}),
    })


def _extraction(con, doc_id, iso3, ym, status="ok"):
    con.execute(
        "INSERT INTO conflict_doc_extractions (doc_id, iso3, year, month, hazard, model, "
        "prompt_version, status, figures_json) VALUES (?, ?, ?, ?, 'ACE', 'm', 'conflict_v1', ?, '[]')",
        [doc_id, iso3, int(ym[:4]), int(ym[5:7]), status],
    )


@pytest.fixture()
def db(tmp_path):
    path = tmp_path / "fixture.duckdb"
    con = duckdb.connect(str(path))
    cde.ensure_schema(con)
    from resolver.hazard_resolution.schema import ensure_haz_schema

    ensure_haz_schema(con)
    # SDN 2025-01: 5 documents (the cap), 4 extracted ok, 1 error, 6 figures,
    # 1 admitted, rejections across the families.
    rejected = [
        _rej(cde.REJECT_NOT_NEW, "Since 3 January, 12,000 people have fled", cumulative_or_new="cumulative"),
        _rej(cde.REJECT_NOT_NEW, "A total of 40,000 people displaced to date", cumulative_or_new="cumulative"),
        _rej(cde.REJECT_NOT_NEW, "The camp hosts 80,000 IDPs", cumulative_or_new="stock"),
        _rej(cde.REJECT_OUTSIDE_MONTH, "In December, 5,000 fled",
             period_start="2024-12-01", period_end="2024-12-20", cell_ym="2025-01"),
        _rej(cde.REJECT_CROSSES_MONTH, "From 20 Jan to 5 Feb, 3,000 fled",
             period_start="2025-01-20", period_end="2025-02-05", cell_ym="2025-01"),
        _rej(cde.REJECT_LONGER_THAN_MONTH, "Since October, 90,000 fled",
             period_start="2024-10-01", period_end="2025-01-20", cell_ym="2025-01"),
        _rej(cde.REJECT_UNDATED, "Thousands fled the fighting", period_start="", period_end="",
             cell_ym="2025-01"),
        {"reason": "quote_not_found_in_document", "value": 7, "quote": "7 people fled"},
    ]
    counts = _reason_counts(rejected)
    counts[cde.REJECT_NOT_NEW] += 2  # two more counted than stored
    _ledger(con, "SDN", "2025-01", cde.CELL_ADMITTED, docs=5, extracted=6, admitted=1,
            counts=counts, rejected=rejected,
            fetch={"ok": True, "returned": 40, "total_available": 25, "selected": 5,
                   "not_about_cell": 3})
    for i in range(4):
        _extraction(con, f"d{i}", "SDN", "2025-01")
    _extraction(con, "d9", "SDN", "2025-01", status="error")
    # SDN 2025-02: no documents.
    _ledger(con, "SDN", "2025-02", cde.CELL_NO_DOCUMENTS)
    # COD 2025-01: 2 documents, figures rejected, none admitted.
    _ledger(con, "COD", "2025-01", cde.CELL_NO_ADMISSIBLE, docs=2, extracted=1,
            counts={cde.REJECT_NOT_CONFLICT: 1},
            rejected=[_rej(cde.REJECT_NOT_CONFLICT, "Floods displaced 900", cause="disaster")])
    # Outside the window: never counted.
    _ledger(con, "SDN", "2024-06", cde.CELL_NO_DOCUMENTS)
    # Admitted candidate for SDN 2025-01, published 12 days after month end.
    con.execute(
        "INSERT INTO conflict_displacement_candidates (iso3, ym, source_ref, preference_rank, "
        "value_type, value_persons, quote, doc_id, doc_date) VALUES "
        "('SDN', '2025-01', 'r', 0, 'new_displacements', 4000, 'q', 'd1', '2025-02-12')"
    )
    # Cached ACE documents: one fetched for SDN 2025-03 published 2025-03-15
    # (43 days after January ended), one for SDN 2025-01 itself.
    for rid, ym, created in (("rw-a", "2025-03", "2025-03-15T00:00:00+00:00"),
                             ("rw-b", "2025-01", "2025-01-10T00:00:00+00:00")):
        con.execute(
            "INSERT INTO haz_raw_reliefweb_docs (record_id, iso3, ym, hazard, payload_json, content_hash) "
            "VALUES (?, 'SDN', ?, 'ACE', ?, ?)",
            [rid, ym, json.dumps({"doc_id": rid, "date_created": created}), rid],
        )
    con.close()
    return path


def _reason_counts(rejected):
    from collections import Counter

    return Counter(r["reason"] for r in rejected)


def _report(path, **kw):
    con = cey.open_read_only(str(path))
    try:
        return cey.build_report(con, start="2024-10", end="2025-02", today=TODAY,
                                countries=["SDN", "COD", "AFG"], settings=SETTINGS, **kw)
    finally:
        con.close()


def test_funnel(db):
    rep = _report(db)
    f = rep["funnel"]
    assert f["cells_assessed"] == 3
    assert f["cells_with_documents"] == 2
    assert f["documents_selected"] == 7
    assert f["documents_extracted"] == 4
    assert f["documents_extraction_failed"] == 1
    assert f["figures_extracted"] == 7
    assert f["figures_admitted"] == 1
    assert f["cells_admitted"] == 1
    assert rep["funnel_by_country"]["AFG"] is None
    assert rep["funnel_by_country"]["SDN"]["cells_assessed"] == 2


def test_rejections_use_the_module_reason_codes(db):
    rj = _report(db)["rejections"]
    assert rj["counted"][cde.REJECT_NOT_NEW] == 5
    assert rj["stored"][cde.REJECT_NOT_NEW] == 3
    assert rj["counted"]["quote_not_found_in_document"] == 1
    assert rj["by_family"]["machine"] == 1
    assert rj["by_family"]["conflict_rule"] == rj["counted_total"] - 1
    assert rj["by_country"]["COD"] == {cde.REJECT_NOT_CONFLICT: 1}


def test_period_split(db):
    ps = _report(db)["period_split"]
    assert ps["split"] == {
        cey.SPLIT_SUB_MONTH_OTHER: 1, cey.SPLIT_CROSSES: 1,
        cey.SPLIT_LONGER: 1, cey.SPLIT_UNDATED: 1,
    }
    assert ps["sub_month_other_month_offset"] == {"-1": 1}


def test_not_new_split(db):
    nn = _report(db)["not_new_split"]
    assert nn["split"] == {cey.NOT_NEW_SINCE: 1, cey.NOT_NEW_CUMULATIVE: 1, cey.NOT_NEW_STOCK: 1}
    assert nn["counted"] == 5 and nn["stored"] == 3


def test_cap(db):
    cp = _report(db)["cap"]
    assert cp["cells_at_cap"] == 1
    assert cp["cells_with_match_count_stored"] == 1
    assert cp["more_matched_total"] == 20
    assert cp["not_about_cell_total"] == 3


def test_lag_and_late_documents(db):
    lg = _report(db)["lag"]
    assert lg["admitted_figures"] == 1
    assert lg["buckets"]["8-14"] == 1
    assert lg["median_days"] == 12
    ld = lg["late_documents"]
    assert ld["cells_with_late_documents"] == 1
    assert ld["late_documents"] == 1


def test_rungs_section_runs_on_a_db_without_the_series(db):
    rg = _report(db)["rungs"]
    assert not rg.get("stub"), rg
    assert {r["iso3"] for r in rg["rows"]} == {"SDN", "COD", "AFG"}
    # No IDMC series: only the extracted SDN 2025-01 figure resolves a month.
    assert rg["totals"].get("reliefweb_extracted") == 1
    assert sum(v for k, v in rg["totals"].items() if not k.startswith("unresolved")) == 1
    sdn = next(r for r in rg["rows"] if r["iso3"] == "SDN")
    assert sdn.get("reliefweb_extracted") == 1
    assert sum(rg["totals"].values()) == 3 * 5


def test_top_reasons_carry_scrubbed_truncated_quotes(db, monkeypatch):
    monkeypatch.setattr(cey, "QUOTE_MAX_CHARS", 20)
    tr = _report(db)["top_reasons"]
    assert tr[0]["reason"] == cde.REJECT_NOT_NEW
    assert 1 <= len(tr[0]["examples"]) <= 3
    assert all(len(q) <= 20 for t in tr for q in t["examples"])


def test_quote_is_scrubbed():
    q = cey._quote("key sk-ant-api03-" + "A" * 40 + " fled")
    assert "AAAAAAAAAAAA" not in q


def test_empty_ledger_says_no_extraction_rows_never_zeros(tmp_path):
    path = tmp_path / "empty.duckdb"
    con = duckdb.connect(str(path))
    cde.ensure_schema(con)
    con.close()
    rep = _report(path)
    assert rep["stub"] is True and rep["reason"].startswith("no extraction rows")
    assert "funnel" not in rep
    text = cey.render_markdown(rep)
    assert "no extraction rows" in text
    assert "| 0 |" not in text.split("## g.")[0]


def test_absent_tables_say_no_extraction_rows(tmp_path):
    path = tmp_path / "bare.duckdb"
    duckdb.connect(str(path)).close()
    rep = _report(path)
    assert rep["stub"] and "absent" in rep["reason"]


def test_the_script_opens_the_db_read_only(db, monkeypatch, tmp_path):
    seen = {}
    real = duckdb.connect

    def spy(path, *a, **kw):
        seen.update(kw)
        return real(path, *a, **kw)

    monkeypatch.setattr(duckdb, "connect", spy)
    assert cey.main(["--db", str(db), "--start", "2024-10", "--end", "2025-02",
                     "--out-dir", str(tmp_path / "out")]) == 0
    assert seen.get("read_only") is True
    assert (tmp_path / "out" / "conflict_extraction_yield.md").is_file()
    con = cey.open_read_only(str(db))
    with pytest.raises(Exception):
        con.execute("CREATE TABLE x (a INT)")
    con.close()


def test_the_report_creates_nothing(tmp_path):
    path = tmp_path / "bare.duckdb"
    duckdb.connect(str(path)).close()
    cey.main(["--db", str(path), "--out-dir", str(tmp_path / "out")])
    con = duckdb.connect(str(path))
    assert con.execute("SELECT COUNT(*) FROM information_schema.tables").fetchone()[0] == 0
    con.close()


def test_time_budget_skips_sections_by_name(db):
    rep = _report(db, max_seconds=1e-9)
    assert "period_split" in rep["sections_skipped"]
    assert "funnel" in rep
