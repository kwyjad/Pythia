# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The conflict displacement extraction source, network-free and model-free.

The ReliefWeb transport and the model are injected; everything else is the
real code: the PA machine's extraction loop and quote verification, the
admissibility rules, the tables and the budget.
"""

from __future__ import annotations

import copy
import datetime as dt
import json

import duckdb
import pytest

from resolver.hazard_resolution import extract as extract_mod
from resolver.hazard_resolution.rulebook import Rulebook, load_rulebook
from resolver.ingestion import conflict_displacement_extract as cde

MODEL = "anthropic:claude-haiku-test"
TODAY = dt.date(2026, 5, 20)

BODY_SDN = (
    "Fighting in El Fasher intensified in March. "
    "Between 3 and 18 March 2026, 12,500 people were newly displaced from El Fasher by clashes. "
    "Since April 2023, 8,900,000 people have been displaced across Sudan. "
    "In February, 4,000 families fled Tawila after attacks. "
    "Floods displaced 2,000 people in Kassala on 10 March 2026."
)


@pytest.fixture()
def con():
    c = duckdb.connect(":memory:")
    yield c
    c.close()


@pytest.fixture()
def rulebook():
    return load_rulebook()


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(cde.CAP_OVERRIDE_ENV, raising=False)
    monkeypatch.delenv("SIBYL_BACKTEST_MODE", raising=False)


def _item(doc_id, body=BODY_SDN, iso3="SDN", created="2026-03-25T00:00:00+00:00",
          source="OCHA"):
    return {
        "id": doc_id,
        "fields": {
            "title": f"Sudan flash update {doc_id}",
            "url": f"https://reliefweb.int/report/sudan/{doc_id}",
            "body": body,
            "date": {"created": created, "original": created},
            "source": [{"name": source, "shortname": source}],
            "primary_country": {"iso3": iso3},
            "country": [{"iso3": iso3}],
            "format": [{"name": "Situation Report"}],
        },
    }


def _post(items, seen=None):
    def post(url, payload, params, timeout):
        if seen is not None:
            seen.append(payload)
        return {"data": items, "totalCount": len(items)}
    return post


def _fig(**over):
    base = {
        "value": 12500, "unit": "people",
        "quote": "Between 3 and 18 March 2026, 12,500 people were newly displaced from El Fasher by clashes.",
        "stated_by": "", "area": "El Fasher", "date": "",
        "period_start": "2026-03-03", "period_end": "2026-03-18",
        "cumulative_or_new": "new", "cause": "conflict",
    }
    base.update(over)
    return base


def _call(figures, calls=None):
    def call(model_ref, prompt, rulebook):
        if calls is not None:
            calls.append(prompt)
        return json.dumps({"figures": figures}), {"prompt_tokens": 1000, "completion_tokens": 200}, ""
    return call


def _run(con, rulebook, items, figures, calls=None, iso3="SDN", ym="2026-03"):
    budget = cde.load_budget(con, rulebook, TODAY)
    return cde.run_cell(
        con, iso3, ym, rulebook, budget, post=_post(items), call=_call(figures, calls),
        model_ref=MODEL, country_name="Sudan", today=TODAY, run_id="t1",
    )


def _reasons(con, iso3="SDN", ym="2026-03"):
    raw = con.execute(
        "SELECT rejected_counts_json FROM conflict_displacement_cell_ledger WHERE iso3=? AND ym=?",
        [iso3, ym],
    ).fetchone()[0]
    return json.loads(raw)


# --- the happy path ---------------------------------------------------------


def test_a_new_conflict_figure_inside_the_month_is_admitted(con, rulebook):
    row = _run(con, rulebook, [_item("101")], [_fig()])
    assert row["reason"] == cde.CELL_ADMITTED
    assert row["best_value"] == 12500
    best = cde.extracted_figure(con, "SDN", "2026-03")
    assert best["value_persons"] == 12500
    assert best["value_type"] == "new_displacements"
    assert best["period_start"] == "2026-03-03" and best["period_end"] == "2026-03-18"
    assert best["doc_url"].endswith("/101")
    assert best["prompt_version"] == "conflict_v2"
    assert "12,500" in best["quote"]
    # The document went into the machine's shared cache, under ACE.
    n = con.execute(
        "SELECT COUNT(*) FROM haz_raw_reliefweb_docs WHERE hazard='ACE' AND iso3='SDN'"
    ).fetchone()[0]
    assert n == 1


def test_a_rerun_reads_the_cache_and_makes_no_call(con, rulebook):
    _run(con, rulebook, [_item("101")], [_fig()])
    calls: list[str] = []
    row = _run(con, rulebook, [_item("101")], [_fig()], calls=calls)
    assert calls == []
    assert row["docs_cached"] == 1
    assert row["reason"] == cde.CELL_ADMITTED


def test_the_prompt_names_the_country_the_month_and_asks_for_new_conflict_displacement(con, rulebook):
    calls: list[str] = []
    _run(con, rulebook, [_item("101")], [_fig()], calls=calls)
    prompt = calls[0]
    assert "Sudan" in prompt and "March 2026" in prompt
    assert "NEWLY DISPLACED by conflict or violence" in prompt


def test_the_query_scopes_on_the_primary_country_and_the_padded_window(con, rulebook):
    seen: list[dict] = []
    cde.fetch_documents(con, "SDN", "2026-03", rulebook, post=_post([_item("101")], seen))
    conds = {c["field"]: c["value"] for c in seen[0]["filter"]["conditions"]}
    assert conds["primary_country.iso3"] == "SDN"
    assert conds["date.created"]["to"].startswith("2026-04-30")
    assert "displaced" in seen[0]["query"]["value"]


# --- admissibility ------------------------------------------------------------


def test_a_cumulative_figure_is_rejected(con, rulebook):
    fig = _fig(value=8900000, quote="Since April 2023, 8,900,000 people have been displaced across Sudan.",
               cumulative_or_new="cumulative", period_start="2023-04-15", period_end="")
    row = _run(con, rulebook, [_item("101")], [fig])
    assert row["reason"] == cde.CELL_NO_ADMISSIBLE
    assert _reasons(con) == {cde.REJECT_NOT_NEW: 1}


@pytest.mark.parametrize("flag", ["", None, "unstated", "maybe"])
def test_a_blank_flag_is_flag_missing_never_not_new(con, rulebook, flag):
    fig = _fig()
    if flag is None:
        fig.pop("cumulative_or_new")
    else:
        fig["cumulative_or_new"] = flag
    row = _run(con, rulebook, [_item("101")], [fig])
    assert row["reason"] == cde.CELL_NO_ADMISSIBLE
    assert _reasons(con) == {cde.REJECT_FLAG_MISSING: 1}


def test_unclear_has_its_own_reason_and_current_stock_is_not_new(con, rulebook):
    fig_unclear = _fig(cumulative_or_new="unclear")
    stock = _fig(value=8900000, quote="Since April 2023, 8,900,000 people have been displaced across Sudan.",
                 cumulative_or_new="current_stock")
    _run(con, rulebook, [_item("101")], [fig_unclear, stock])
    assert _reasons(con) == {cde.REJECT_FLAG_UNCLEAR: 1, cde.REJECT_NOT_NEW: 1}


def test_the_prompt_requires_the_flag_from_the_enum(con, rulebook):
    calls: list[str] = []
    _run(con, rulebook, [_item("101")], [_fig()], calls=calls)
    assert '"new" | "cumulative" | "current_stock" | "unclear"' in calls[0]
    assert "REQUIRED" in calls[0] and "Never leave it empty" in calls[0]
    assert rulebook.get("conflict_displacement.extraction.prompt_version") == "conflict_v2"


def _seed_v1(con, rulebook, figures, doc="101", ym="2026-03"):
    """A cached v1 answer for one document, as the Oct 2026 backfill wrote it."""
    _run(con, rulebook, [_item(doc)], [], ym=ym)  # caches the document text
    con.execute("DELETE FROM conflict_doc_extractions")
    con.execute(
        "INSERT INTO conflict_doc_extractions (doc_id, iso3, year, month, hazard, model, "
        "prompt_version, status, figures_json, cost_usd) VALUES (?, 'SDN', ?, ?, 'ACE', ?, "
        "'conflict_v1', 'ok', ?, 0.002)",
        [doc, int(ym[:4]), int(ym[5:]), MODEL, json.dumps({"figures": figures, "rejected": []})],
    )


def test_only_flag_missing_documents_are_read_again(con, rulebook):
    blank = dict(_fig(cumulative_or_new="unstated"), doc_id="101")
    _seed_v1(con, rulebook, [blank])
    _seed_definite = dict(_fig(), doc_id="102")
    con.execute(
        "INSERT INTO conflict_doc_extractions (doc_id, iso3, year, month, hazard, model, "
        "prompt_version, status, figures_json) VALUES ('102', 'SDN', 2026, 3, 'ACE', ?, "
        "'conflict_v1', 'ok', ?)", [MODEL, json.dumps({"figures": [_seed_definite]})],
    )
    wanted = cde.flag_missing_documents(con, rulebook)
    assert [(w["doc_id"], w["n_figures_flag_missing"]) for w in wanted] == [("101", 1)]
    plan = cde.plan_reextract(con, rulebook, cde.load_budget(con, rulebook, TODAY))
    assert plan["expected_calls"] == 1 and plan["documents"] == 1
    calls: list[str] = []
    out = cde.reextract_flag_missing(con, rulebook, call=_call([_fig()], calls),
                                     model_ref=MODEL, today=TODAY)
    assert len(calls) == 1 and out["calls_made"] == 1
    # The document now has a v2 answer and is no longer wanted.
    assert cde.flag_missing_documents(con, rulebook) == []
    assert cde.extracted_figure(con, "SDN", "2026-03")["value_persons"] == 12500


def test_a_definite_v1_answer_is_reused_and_not_paid_for_again(con, rulebook):
    _seed_v1(con, rulebook, [dict(_fig(), doc_id="101")])
    calls: list[str] = []
    row = _run(con, rulebook, [_item("101")], [_fig()], calls=calls)
    assert calls == [] and row["docs_cached"] == 1
    assert row["reason"] == cde.CELL_ADMITTED


def test_a_figure_for_another_month_is_routed_not_rejected(con, rulebook):
    fig = _fig(value=4000, unit="households",
               quote="In February, 4,000 families fled Tawila after attacks.",
               period_start="2026-02-01", period_end="2026-02-28")
    # February not assessed yet: stored, not used, counted.
    row = _run(con, rulebook, [_item("101")], [fig])
    assert row["reason"] == cde.CELL_NO_ADMISSIBLE
    assert _reasons(con) == {}
    assert json.loads(row["detail_json"])["routing"]["routed_out_outside_window"] == 1
    assert cde.extracted_figure(con, "SDN", "2026-02") is None
    status = con.execute("SELECT status, routed_from_ym FROM conflict_displacement_candidates "
                         "WHERE ym='2026-02'").fetchone()
    assert status == (cde.STATUS_OUTSIDE_WINDOW, "2026-03")
    # Once February is assessed (no documents of its own), the figure is its.
    _run(con, rulebook, [], [], ym="2026-02")
    best = cde.extracted_figure(con, "SDN", "2026-02")
    assert best is not None and best["routed_from_ym"] == "2026-03"
    feb = con.execute("SELECT reason, figures_admitted FROM conflict_displacement_cell_ledger "
                      "WHERE iso3='SDN' AND ym='2026-02'").fetchone()
    assert feb == (cde.CELL_ADMITTED, 1)


def _figure(**over):
    base = dict(value=12500, unit="people", quote="q", stated_by="OCHA", area="", date="",
                cumulative_or_new="new", doc_id="9", doc_url="", doc_title="",
                doc_date="2026-03-25", doc_source_rank=0, model=MODEL,
                doc_primary_country="SDN", period_start="2026-03-03",
                period_end="2026-03-18", cause="conflict")
    base.update(over)
    return cde.ConflictFigure(**base)


def test_routing_to_the_next_and_previous_month(rulebook):
    nxt = _figure(quote="next", doc_id="1", period_start="2026-04-01", period_end="2026-04-10",
                  doc_date="2026-04-12")
    prev = _figure(quote="prev", doc_id="2", period_start="2026-02-20", period_end="2026-02-27",
                   doc_date="2026-03-05")
    out = cde.admit_country({"2026-03": [nxt, prev]}, "SDN", rulebook,
                            assessed=["2026-02", "2026-03", "2026-04"])
    assert [f.quote for f, _v, _i in out.admitted["2026-04"]] == ["next"]
    assert [f.quote for f, _v, _i in out.admitted["2026-02"]] == ["prev"]
    assert out.admitted.get("2026-03", []) == []
    assert out.routed_out == {"2026-03": 2}
    assert out.routed_in == {"2026-02": 1, "2026-04": 1}
    assert out.admitted["2026-04"][0][2]["routed_from_ym"] == "2026-03"


def test_one_document_read_by_both_cells_counts_once(rulebook):
    # A document published early April, read in the March cell (its window
    # runs to April 30) and in the April cell, states a March figure.
    in_march = _figure(doc_id="7", doc_date="2026-04-03")
    in_april = _figure(doc_id="7", doc_date="2026-04-03")
    out = cde.admit_country({"2026-03": [in_march], "2026-04": [in_april]}, "SDN", rulebook)
    assert len(out.admitted["2026-03"]) == 1
    assert out.admitted["2026-03"][0][2].get("routed_from_ym") is None  # the native reading
    assert [r["reason"] for r in out.rejected["2026-04"]] == [cde.REASON_DUPLICATE_ACROSS_CELLS]
    assert out.routed_in == {}


def test_a_routed_figure_is_not_available_before_its_document(rulebook):
    # A February figure stated in a document of 5 April: the rule alone
    # (February end + 30 days = 30 March) would read it before it existed.
    late = _figure(period_start="2026-02-10", period_end="2026-02-20", doc_date="2026-04-05")
    out = cde.admit_country({"2026-04": [late]}, "SDN", rulebook,
                            assessed=["2026-02", "2026-04"])
    info = out.admitted["2026-02"][0][2]
    assert info["available_on"] == "2026-04-05"
    native = cde.admit_country({"2026-03": [_figure()]}, "SDN", rulebook)
    assert native.admitted["2026-03"][0][2]["available_on"] == "2026-04-30"
    assert cde.available_on("2026-02", "2026-03-01") == dt.date(2026, 3, 30)


def test_the_ladder_reads_a_routed_figure_from_its_document_date(con, rulebook):
    from pythia.tools import base_rate_spd as brs

    late = _figure(period_start="2026-02-10", period_end="2026-02-20", doc_date="2026-04-05",
                   quote="Between 10 and 20 February 2026, 12,500 people fled.")
    out = cde.admit_country({"2026-04": [late]}, "SDN", rulebook,
                            assessed=["2026-02", "2026-04"])
    cde.write_country_candidates(con, "SDN", out, rulebook=rulebook, run_id="t")
    reading = brs.conflict_ladder_lower(con)["SDN"]["2026-02"][0]
    assert reading.available_on == dt.date(2026, 4, 5)


def test_a_period_longer_than_the_month_is_rejected_never_apportioned(con, rulebook):
    row = _run(con, rulebook, [_item("101")],
               [_fig(period_start="2026-02-10", period_end="2026-03-28")])
    assert row["reason"] == cde.CELL_NO_ADMISSIBLE
    assert _reasons(con) == {cde.REJECT_LONGER_THAN_MONTH: 1}
    assert cde.extracted_figure(con, "SDN", "2026-03") is None


def test_a_period_crossing_the_month_boundary_is_rejected(con, rulebook):
    row = _run(con, rulebook, [_item("101")],
               [_fig(period_start="2026-03-25", period_end="2026-04-05")])
    assert _reasons(con) == {cde.REJECT_CROSSES_MONTH: 1}
    assert row["figures_admitted"] == 0


def test_since_a_date_runs_to_the_publication_date(con, rulebook):
    # "since 20 March" in a document published 10 April crosses the boundary.
    row = _run(con, rulebook, [_item("101", created="2026-04-10T00:00:00+00:00")],
               [_fig(period_start="2026-03-20", period_end="")])
    assert _reasons(con) == {cde.REJECT_CROSSES_MONTH: 1}
    assert row["figures_admitted"] == 0


def test_an_undated_figure_is_rejected(con, rulebook):
    row = _run(con, rulebook, [_item("101")], [_fig(period_start="", period_end="", date="")])
    assert _reasons(con) == {cde.REJECT_UNDATED: 1}
    assert row["figures_admitted"] == 0


def test_a_figure_attributed_to_a_disaster_is_rejected(con, rulebook):
    fig = _fig(value=2000, quote="Floods displaced 2,000 people in Kassala on 10 March 2026.",
               period_start="", period_end="", date="2026-03-10", cause="disaster")
    row = _run(con, rulebook, [_item("101")], [fig])
    assert _reasons(con) == {cde.REJECT_NOT_CONFLICT: 1}
    assert row["figures_admitted"] == 0


def test_an_unverified_quote_is_rejected(con, rulebook):
    fig = _fig(value=30000, quote="30,000 people were newly displaced from El Fasher in March.")
    row = _run(con, rulebook, [_item("101")], [fig])
    assert row["reason"] == cde.CELL_NO_FIGURE_STATED
    assert _reasons(con) == {"quote_not_found_in_document": 1}


def test_a_document_about_another_country_supplies_nothing(rulebook):
    figure = cde.ConflictFigure(
        value=12500, unit="people", quote="q", stated_by="", area="", date="",
        cumulative_or_new="new", doc_id="9", doc_url="", doc_title="", doc_date="2026-03-25",
        doc_source_rank=0, model=MODEL, doc_primary_country="TCD",
        period_start="2026-03-03", period_end="2026-03-18", cause="conflict",
    )
    admitted, rejected = cde.admit_figures([figure], "SDN", "2026-03", rulebook)
    assert admitted == []
    assert [r["reason"] for r in rejected] == ["document_primary_country_mismatch"]


def test_a_document_about_another_country_is_not_read(con, rulebook):
    calls: list[str] = []
    row = _run(con, rulebook, [_item("201", iso3="TCD")], [_fig()], calls=calls)
    assert calls == []
    assert row["reason"] == cde.CELL_NO_DOCUMENTS


def test_households_convert_with_the_rulebook_multiplier(rulebook):
    figure = cde.ConflictFigure(
        value=1000, unit="households", quote="q", stated_by="OCHA", area="", date="2026-03-12",
        cumulative_or_new="new", doc_id="9", doc_url="", doc_title="", doc_date="2026-03-25",
        doc_source_rank=0, model=MODEL, doc_primary_country="SDN", cause="violence",
    )
    admitted, rejected = cde.admit_figures([figure], "SDN", "2026-03", rulebook)
    assert rejected == []
    _fig_out, persons, info = admitted[0]
    multiplier = rulebook.get("reliefweb.household_conversion.by_iso3", {}).get(
        "SDN", rulebook.get("reliefweb.household_conversion.default_multiplier"))
    assert persons == round(1000 * float(multiplier))
    assert info["conversion_factor"] == float(multiplier)


def test_restatements_collapse_and_the_best_figure_ranks_first(con, rulebook):
    items = [_item("101"), _item("102", created="2026-03-28T00:00:00+00:00")]
    row = _run(con, rulebook, items, [_fig()])
    assert row["figures_admitted"] == 1
    assert _reasons(con).get("duplicate_quote") == 1


# --- budget -------------------------------------------------------------------


def test_the_monthly_cap_binds_and_the_cell_says_so(con, rulebook, monkeypatch):
    monkeypatch.setenv(cde.CAP_OVERRIDE_ENV, "1")
    items = [_item(str(100 + i), body=BODY_SDN + f" Report {i}.") for i in range(3)]
    calls: list[str] = []
    row = _run(con, rulebook, items, [_fig()], calls=calls)
    assert len(calls) == 1
    assert row["budget_capped"] is True


def test_the_budget_is_separate_from_the_pa_machines(con, rulebook):
    _run(con, rulebook, [_item("101")], [_fig()])
    assert cde.calls_this_calendar_month(con) == 1
    # The PA machine's month count never sees a conflict call...
    assert extract_mod.calls_this_calendar_month(con) == 0
    # ...and PA calls never draw on the conflict cap.
    con.execute(
        "INSERT INTO haz_doc_extractions (doc_id, iso3, year, month, hazard, model, "
        "prompt_version, status, figures_json, prompt_tokens, completion_tokens) "
        "VALUES ('x', 'PHL', 2026, 3, 'FL', 'm', 'v2', 'ok', '{}', 10, 10)"
    )
    assert cde.calls_this_calendar_month(con) == 1
    budget = cde.load_budget(con, rulebook)
    assert budget.max_calls_per_month == 2000
    assert budget.live_reserve_calls == 0 and budget.run_type == "live"


def test_never_in_backtest(con, rulebook, monkeypatch):
    monkeypatch.setenv("SIBYL_BACKTEST_MODE", "1")
    calls: list[str] = []
    row = _run(con, rulebook, [_item("101")], [_fig()], calls=calls)
    assert row["reason"] == cde.CELL_DISABLED
    assert calls == []


def test_a_plan_prices_the_run_before_it_runs(con, rulebook):
    budget = cde.load_budget(con, rulebook, TODAY)
    info = cde.plan(con, ["SDN", "COD"], ["2026-03", "2026-02"], rulebook, budget)
    assert info["cells"] == 4
    assert info["max_calls_upper_bound"] == 4 * info["max_docs_per_cell"]
    assert info["max_cost_upper_bound_usd"] == round(
        info["max_calls_upper_bound"] * cde.DEFAULT_COST_PER_CALL_USD, 2)


def test_run_walks_newest_month_first_and_stops_on_the_clock(con, rulebook):
    ticks = iter([0.0, 0.0, 100.0, 100.0, 100.0, 100.0])
    summary = cde.run(
        con, ["SDN"], ["2026-02", "2026-03"], rulebook, post=_post([_item("101")]),
        call=_call([_fig()]), model_ref=MODEL, today=TODAY, max_minutes=1.0,
        clock=lambda: next(ticks),
    )
    assert summary["cells_by_reason"] == {cde.CELL_ADMITTED: 1}
    assert summary["not_reached"] == ["SDN/2026-02"]


def test_complete_months_are_newest_first_and_exclude_the_current_month():
    assert cde.complete_months(dt.date(2026, 1, 11), 3) == ["2025-12", "2025-11", "2025-10"]


# --- the rulebook fingerprint ---------------------------------------------------


def test_the_conflict_section_moves_no_hazard_fingerprint(rulebook):
    data = copy.deepcopy(rulebook.raw)
    assert "conflict_displacement" in data
    without = copy.deepcopy(data)
    without.pop("conflict_displacement")
    stripped = Rulebook(without, rulebook.path)
    changed = copy.deepcopy(data)
    changed["conflict_displacement"]["extraction"]["max_calls_per_month"] = 1
    changed["conflict_displacement"]["documents"]["query"] = "anything"
    edited = Rulebook(changed, rulebook.path)
    for hazard in ("flood", "cyclone", "drought"):
        assert rulebook.hazard_fingerprint(hazard) == stripped.hazard_fingerprint(hazard)
        assert rulebook.hazard_fingerprint(hazard) == edited.hazard_fingerprint(hazard)
        assert rulebook.legacy_hazard_fingerprint(hazard) == stripped.legacy_hazard_fingerprint(hazard)


def test_the_conflict_prompt_version_differs_from_the_pa_one(rulebook):
    assert (rulebook.get("conflict_displacement.extraction.prompt_version")
            != rulebook.get("extraction.prompt_version"))


# --- the workflow -----------------------------------------------------------------


def test_resolver_update_runs_both_sources_non_fatally_with_their_own_timeouts():
    from pathlib import Path

    import yaml

    path = Path(__file__).resolve().parents[2] / ".github/workflows/resolver_update.yml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    on = data.get("on") or data.get(True)
    # The dispatch-input cap: the backfill reuses only_connector.
    assert len(on["workflow_dispatch"]["inputs"]) <= 10
    steps = [s for job in data["jobs"].values() for s in job.get("steps", [])]
    phase = [s for s in steps if str(s.get("name", "")).startswith("Phase 2.6")]
    assert len(phase) == 3
    for step in phase:
        assert step.get("continue-on-error") is True, step["name"]
        assert int(step["timeout-minutes"]) <= 270, step["name"]
        assert "timeout " in step["run"] and "--summary-out" in step["run"], step["name"]
    backfill = next(s for s in phase if "BACKFILL" in s["name"])
    assert "displacement_backfill" in backfill["if"]
    assert backfill["env"][cde.CAP_OVERRIDE_ENV]
    assert "--plan-only" in backfill["run"]
    # The rulebook default is never what a backfill edits.
    assert load_rulebook().get("conflict_displacement.extraction.max_calls_per_month") == 2000


# --- the resolver debug bundle --------------------------------------------------


def test_the_debug_bundle_reports_and_checks_both_sources(tmp_path, rulebook):
    from resolver.ingestion import dtm_conflict as dtm
    from scripts import build_resolver_debug_bundle as bundle

    db = tmp_path / "r.duckdb"
    c = duckdb.connect(str(db))
    _run(c, rulebook, [_item("101")], [_fig()])
    series, _ = dtm.conflict_series([
        {"admin0Pcode": "SDN", "numPresentIdpInd": 100, "reportingDate": "2026-02-25",
         "displacementReason": "Conflict", "operation": "A"},
        {"admin0Pcode": "SDN", "numPresentIdpInd": 300, "reportingDate": "2026-03-10",
         "displacementReason": "Conflict", "operation": "A"},
    ], "SDN")
    bounds, _ = dtm.derive_lower_bounds(series)
    dtm.store_country(c, "SDN", series, bounds, max_gap_days=35)
    c.close()

    def _builder(name):
        return bundle.BundleBuilder(
            out_path=tmp_path / f"{name}.zip", db_path=db, diagnostics_dir=tmp_path / "diag",
            run_log_dir=tmp_path / "run_log", staging=tmp_path / name,
            max_bytes=bundle.DEFAULT_MAX_BYTES, environ={},
        )

    builder = _builder("s1")
    dest = builder.path("hazard")
    counts = builder._conflict_displacement(dest)
    assert counts["cells_by_reason"] == {cde.CELL_ADMITTED: 1}
    assert counts["admitted_figures"] == 1 and counts["dtm_lower_bound_months"] == 1
    assert (dest / "conflict_displacement" / "cell_ledger.csv").is_file()
    builder._check_conflict_displacement_inside_its_month()
    assert builder.checks[-1]["verdict"] == "PASS", builder.checks[-1]

    # A figure stored outside its month must fail the check.
    c = duckdb.connect(str(db))
    c.execute("UPDATE conflict_displacement_candidates SET period_end = '2026-04-02'")
    c.close()
    builder = _builder("s2")
    builder._check_conflict_displacement_inside_its_month()
    check = builder.checks[-1]
    assert check["verdict"] == "FAIL" and "2026-04-02" in check["detail"]


# --- the period against the calendar --------------------------------------------


D = dt.date


@pytest.mark.parametrize("start,end,ym,expected", [
    # "In March" read in a February cell: inside another single month, not
    # "longer than the month" (the pre-Nov-2026 label, which measured 31
    # days against February's 28).
    (D(2026, 3, 1), D(2026, 3, 31), "2026-02", cde.PERIOD_OTHER_MONTH),
    (D(2026, 3, 1), D(2026, 3, 31), "2026-03", cde.PERIOD_INSIDE),
    (D(2026, 2, 1), D(2026, 2, 28), "2026-03", cde.PERIOD_OTHER_MONTH),
    # A 31-day May read in a 30-day April cell, and the reverse.
    (D(2026, 5, 1), D(2026, 5, 31), "2026-04", cde.PERIOD_OTHER_MONTH),
    (D(2026, 4, 1), D(2026, 4, 30), "2026-05", cde.PERIOD_OTHER_MONTH),
    # Spans the April/May boundary in 31 days: crosses, never "longer".
    (D(2026, 4, 15), D(2026, 5, 15), "2026-04", cde.PERIOD_CROSSES),
    # 28 February to 1 March: crosses.
    (D(2026, 2, 28), D(2026, 3, 1), "2026-02", cde.PERIOD_CROSSES),
    # 32 days fits no month.
    (D(2026, 3, 1), D(2026, 4, 1), "2026-03", cde.PERIOD_LONGER),
    (None, None, "2026-03", cde.PERIOD_UNDATED),
    (D(2026, 3, 9), D(2026, 3, 2), "2026-03", cde.PERIOD_REVERSED),
])
def test_periods_are_classified_against_the_calendar(start, end, ym, expected):
    assert cde.classify_period(start, end, ym) == expected


def test_a_whole_march_figure_read_in_february_is_routed_to_march(rulebook):
    march = _figure(period_start="2026-03-01", period_end="2026-03-31", doc_date="2026-03-02")
    out = cde.admit_country({"2026-02": [march]}, "SDN", rulebook,
                            assessed=["2026-02", "2026-03"])
    assert out.rejected["2026-02"] == []
    assert len(out.admitted["2026-03"]) == 1
