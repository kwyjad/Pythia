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
    assert best["prompt_version"] == "conflict_v1"
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


def test_an_unstated_figure_is_rejected_as_not_new(con, rulebook):
    row = _run(con, rulebook, [_item("101")], [_fig(cumulative_or_new="unstated")])
    assert row["reason"] == cde.CELL_NO_ADMISSIBLE
    assert _reasons(con) == {cde.REJECT_NOT_NEW: 1}


def test_a_figure_for_another_month_is_rejected(con, rulebook):
    fig = _fig(value=4000, unit="households",
               quote="In February, 4,000 families fled Tawila after attacks.",
               period_start="2026-02-01", period_end="2026-02-28")
    row = _run(con, rulebook, [_item("101")], [fig])
    assert row["reason"] == cde.CELL_NO_ADMISSIBLE
    assert _reasons(con) == {cde.REJECT_OUTSIDE_MONTH: 1}


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
