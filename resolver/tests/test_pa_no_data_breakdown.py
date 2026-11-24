# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The NO_DATA breakdown: read-only, counted right, and honest about misses."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import pytest

from resolver.hazard_resolution.extract import ExtractedFigure
from resolver.hazard_resolution.rulebook import load_rulebook
from scripts.diagnostics import pa_no_data_breakdown as nd


@pytest.fixture(scope="module")
def rulebook():
    return load_rulebook()


def _fig(**overrides) -> ExtractedFigure:
    base = dict(
        value=5000.0, unit="people", quote="5,000 people were affected",
        stated_by="", area="", date="", cumulative_or_new="unstated",
        doc_id="d1", doc_url="https://reliefweb.int/node/1", doc_title="Floods",
        doc_date="2026-06-20T00:00:00+00:00", doc_source_rank=0, model="test",
        doc_date_original="2026-06-20T00:00:00+00:00", doc_primary_country="",
        doc_country_iso3s=(), doc_publisher="", doc_disaster_types=("Flood",),
    )
    base.update(overrides)
    return ExtractedFigure(**base)


def _doc(fig: ExtractedFigure, body: str) -> dict:
    return {
        "doc_id": fig.doc_id, "title": fig.doc_title, "url": fig.doc_url, "body": body,
        "date_created": fig.doc_date, "date_original": fig.doc_date_original,
        "primary_country_iso3": fig.doc_primary_country,
        "country_iso3s": [fig.doc_primary_country], "sources": [],
        "disaster_types": list(fig.doc_disaster_types), "source_rank": 0,
    }


def _no_data(con, iso3, ym, hazard="FL", *, trigger_source="gdacs", alert="Orange",
             triggered=True, unread=()):
    year, month = (int(p) for p in ym.split("-"))
    detail = {"events_qualifying": [{"event_id": 1, "alert_level": alert}],
              "events_below_threshold": [{"event_id": 2, "alert_level": "Green"}]}
    con.execute(
        "INSERT INTO haz_triggers (iso3, year, month, hazard, triggered, trigger_source, "
        "trigger_detail_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
        [iso3, year, month, hazard, triggered, trigger_source, json.dumps(detail)],
    )
    prov = {"rule_fired": "no_data:no_candidate_on_any_rung",
            "decision": {"rungs_unavailable": list(unread), "flags": ["no_candidate_past_freeze"]}}
    con.execute(
        "INSERT INTO haz_resolutions (iso3, year, month, hazard, status, value, provenance_json,"
        " rule_fired, flagged, provisional, frozen_at) VALUES (?, ?, ?, ?, 'NO_DATA', NULL, ?, "
        "'no_data:no_candidate_on_any_rung', TRUE, FALSE, ?)",
        [iso3, year, month, hazard, json.dumps(prov), dt.datetime(year, month, 28) + dt.timedelta(days=62)],
    )


def _cache(con, rulebook, *, iso3, ym, hazard, figs, doc, rejected=()):
    from resolver.hazard_resolution import extract as ex
    from resolver.hazard_resolution import reliefweb_docs as docs
    from resolver.hazard_resolution.sources import RawRecord, store_raw_records

    store_raw_records(con, docs.SOURCE, [RawRecord(
        record_id=f"rw-{doc['doc_id']}", payload=doc, iso3=iso3, ym=ym, hazard=hazard,
    )])
    ex.write_extraction(
        con, document=doc, iso3=iso3, ym=ym, hazard=hazard,
        model=ex.resolve_extraction_model(rulebook),
        prompt_version=str(rulebook.get("extraction.prompt_version")),
        status="ok", figures=figs, rejected=list(rejected), usage={}, cost_usd=0.0, error=None,
    )


@pytest.fixture()
def db(tmp_path, rulebook):
    import duckdb

    from resolver.hazard_resolution.schema import ensure_haz_schema
    from resolver.hazard_resolution.sources import RawRecord, store_raw_records

    path = tmp_path / "nd.duckdb"
    con = duckdb.connect(str(path))
    ensure_haz_schema(con)

    # AAA: one figure, rejected ONLY because it is dated outside the window.
    _no_data(con, "AAA", "2026-06")
    out_fig = _fig(doc_id="a1", quote="In March 2026, 5,000 people were affected",
                   date="2026-03-10", doc_primary_country="AAA")
    _cache(con, rulebook, iso3="AAA", ym="2026-06", hazard="FL", figs=[out_fig],
           doc=_doc(out_fig, out_fig.quote))

    # BBB: one figure dated outside the window AND about another country.
    _no_data(con, "BBB", "2026-06", alert="Red")
    two_fig = _fig(doc_id="b1", quote="In March 2026, 7,000 people were affected",
                   date="2026-03-10", doc_primary_country="ZZZ")
    _cache(con, rulebook, iso3="BBB", ym="2026-06", hazard="FL", figs=[two_fig],
           doc=_doc(two_fig, two_fig.quote))

    # CCC: no documents at all; IDMC holds a cyclone record for the month only,
    # and IFRC a flood record published after the cell froze.
    _no_data(con, "CCC", "2026-07", trigger_source="reliefweb_ladder", unread=("idmc_idu",))
    store_raw_records(con, "idmc_idu", [RawRecord(
        record_id="idu-9", iso3="CCC", ym="2026-07", hazard="TC",
        payload={"idu_id": "9", "hazard": "TC", "months_overlapped": ["2026-07"],
                 "start_date": "2026-07-03", "displaced": 400,
                 "raw": {"created_at": "2026-07-20"}},
    )])
    store_raw_records(con, "ifrc_go", [RawRecord(
        record_id="go-1-CCC", iso3="CCC", ym="2026-07", hazard="FL",
        payload={"report_id": "1", "hazard": "FL", "months_overlapped": ["2026-07"],
                 "report_date": "2026-11-20", "num_affected": 1200, "raw": {}},
    )])

    # DDD: a resolved value, which the breakdown must not count.
    con.execute(
        "INSERT INTO haz_triggers (iso3, year, month, hazard, triggered, trigger_source)"
        " VALUES ('DDD', 2026, 6, 'FL', TRUE, 'gdacs')")
    con.execute(
        "INSERT INTO haz_resolutions (iso3, year, month, hazard, status, value, "
        "provenance_json, rule_fired) VALUES ('DDD', 2026, 6, 'FL', 'RESOLVED_VALUE', 10, "
        "'{}', 'ladder:ifrc_go')")
    # EEE: assessed, no row written.
    con.execute(
        "INSERT INTO haz_triggers (iso3, year, month, hazard, triggered, trigger_source,"
        " trigger_detail_json) VALUES ('EEE', 2026, 7, 'FL', FALSE, 'none', ?)",
        [json.dumps({"no_row_reason": "sweep_hit_unconfirmed"})])
    con.execute("CHECKPOINT")
    con.close()
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_script_is_read_only(db, tmp_path):
    before = _sha(db)
    rc = nd.main(["--db", str(db), "--hazard", "flood", "--window", "2026-05:2026-08",
                  "--out-dir", str(tmp_path / "out")])
    assert rc == 0
    assert _sha(db) == before
    assert (tmp_path / "out" / "pa_no_data_breakdown.md").exists()
    assert (tmp_path / "out" / "pa_no_data_cells.csv").read_text().count("\n") == 4


def test_each_breakdown(db, tmp_path):
    summary = nd.run(str(db), hazards=["FL"], first="2026-05", last="2026-08",
                     out_dir=tmp_path / "o")
    b = summary["hazards"]["FL"]
    assert b["no_data"] == 3 and b["no_row"] == 1 and b["cells_assessed"] == 5
    assert b["no_row_reasons"] == {"sweep_hit_unconfirmed": 1}
    # 1. detection, with the strongest alert colour.
    assert b["detection"] == {"gdacs_orange": 1, "gdacs_red": 1, "reliefweb_ladder": 1}
    # 2. rungs.
    assert b["rungs"]["idmc_idu"] == {"unread": 1, "record_this_hazard": 0,
                                      "record_other_hazard_only": 1, "no_record": 2}
    assert b["rungs"]["ifrc_go"]["record_this_hazard"] == 1
    rw = b["rungs"]["reliefweb"]
    assert rw["no_documents"] == 1 and rw["figures_all_rejected"] == 2
    assert rw["rejection_reasons_current_rules"]["outside_reporting_window"] >= 1
    # 3. nearest miss: AAA by the window alone, BBB by several gates.
    assert b["nearest_miss_cells"]["outside_reporting_window"] == 1
    assert b["cells_resolved_by_lever"] == {"window": 1}
    ex = b["nearest_miss_examples"]["outside_reporting_window"][0]
    assert ex["iso3"] == "AAA" and "March 2026" in ex["quote"]
    # 4. concentration.
    assert b["by_month"] == {"2026-06": 2, "2026-07": 1}
    # 5. freeze: CCC's IFRC record arrived after the cell froze.
    assert b["freeze"]["cells_with_ifrc_record_after_freeze"] == 1
    assert b["freeze"]["cells_with_idmc_record_after_freeze"] == 0
    head = summary["headline"]["FL"]
    assert head["top_nearest_miss_reasons"] == {"outside_reporting_window": 1}


def test_a_figure_rejected_for_exactly_one_reason():
    rejected = [
        {"reason": "outside_reporting_window", "doc_id": "a", "quote": "x", "value": 1},
        {"reason": "document_primary_country_mismatch", "doc_id": "b", "quote": "y", "value": 2},
        {"reason": "unknown_unit", "doc_id": "c", "quote": "z", "value": 3},
        {"reason": "household_superseded_by_stated_people", "doc_id": "d", "quote": "w"},
    ]
    misses = nd.classify_nearest_miss(rejected, {"window": {("a", "x")}, "primary_country": set()})
    by_doc = {m["doc_id"]: m["only_reason"] for m in misses}
    assert by_doc == {"a": "outside_reporting_window", "b": "several", "c": "untested:unit"}


def test_detection_class_reads_the_strongest_colour():
    assert nd.detection_class(True, "gdacs", {"events_qualifying": [
        {"alert_level": "Orange"}, {"alert_level": "Red"}]}) == ("gdacs_red", "red")
    assert nd.detection_class(None, None, {}) == ("no_trigger_row", "")
    assert nd.detection_class(False, "none", {})[0] == "not_triggered_legacy"


def test_the_debug_bundle_carries_the_headline(db, tmp_path, monkeypatch):
    from resolver.hazard_resolution import acceptance
    from scripts import build_resolver_debug_bundle as bundle

    monkeypatch.setattr(acceptance, "months_window", lambda rb, n, today=None: ("2026-05", "2026-08"))
    bundle.build_bundle(
        out_path=tmp_path / "b.zip", db_path=db, diagnostics_dir=tmp_path / "missing",
        run_log_dir=None, max_bytes=bundle.DEFAULT_MAX_BYTES, staging=tmp_path / "staging",
        environ={},
    )
    payload = json.loads((tmp_path / "staging" / "hazard" / "pa_no_data_breakdown.json").read_text())
    assert payload["window"] == ["2026-05", "2026-08"]
    assert payload["FL"]["no_data_by_detection"]["gdacs_red"] == 1
    assert payload["FL"]["top_nearest_miss_reasons"] == {"outside_reporting_window": 1}
    assert payload["TC"]["no_data"] == 0
