# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Phase-3 acceptance tests: LLM extraction from ReliefWeb documents.

The Phase 3 acceptance list, one test group each:

- the GOLDEN SET — ten hand-picked documents (including one stating no
  figure, one in households, and one cumulative) reproduce the expected
  JSON exactly;
- EXTRACTION DISCIPLINE — the prompt forbids what it must, and the parser
  refuses a figure whose quote is not in the document, whose unit is not
  stated, or whose value is not a number;
- POST-PROCESSING — households convert deterministically, restatements
  collapse, attributed authority decides, and a figure above the GDACS
  ceiling is rejected with a reason;
- END TO END — extracted candidates flow through the ladder on a real
  flood month, and lose to EM-DAT while beating IFRC GO;
- NO SPEND WITHOUT A TRIGGER — no extraction call is made for a
  non-triggered country-month, for a cell a higher rung already answers,
  or once the monthly cap is reached;
- COST GUARD — the cap counts across runs, and a cached document is free.

Every test is network-free and model-free: the model seam replays recorded
responses, which is what lets the suite assert on the parts of extraction
that must be deterministic.
"""

from __future__ import annotations

import datetime as dt
import json

import duckdb
import pytest

from resolver.hazard_resolution import candidates as cand_mod
from resolver.hazard_resolution import extract as extract_mod
from resolver.hazard_resolution import figures as figures_mod
from resolver.hazard_resolution import impact as impact_mod
from resolver.hazard_resolution import reconcile as reconcile_mod
from resolver.hazard_resolution import reliefweb_docs as docs_mod
from resolver.hazard_resolution.schema import ensure_haz_schema
from resolver.tests.hazard_resolution_utils import (
    golden_call_fn,
    golden_documents,
    load_golden_set,
    make_candidate,
    make_rulebook,
    seed_gdacs_event,
    seed_population,
    seed_reliefweb_docs,
)

ISO3 = "PHL"
YM = "2024-03"
HAZARD = "FL"
CEILING = 500_000.0
MODEL = "anthropic:claude-haiku-4-5-20251001"
# 2024-03 froze at 2024-05-30, so these cells resolve final rather than
# provisional; the freeze behaviour itself is pinned by the ladder suite.
AFTER_FREEZE = dt.date(2024, 8, 1)


@pytest.fixture()
def rulebook():
    return make_rulebook()


@pytest.fixture()
def con():
    connection = duckdb.connect(":memory:")
    ensure_haz_schema(connection)
    return connection


def _case(name: str) -> dict:
    for case in load_golden_set()["documents"]:
        if case["name"] == name:
            return case
    raise AssertionError(f"golden set has no case named {name!r}")


def _extract_all(con, rulebook, *, documents=None, budget=None, call=None):
    """Run the whole cell through the seam and return (extraction, calls)."""
    fn, calls = golden_call_fn()
    extraction = extract_mod.extract_for_cell(
        con,
        iso3=ISO3,
        ym=YM,
        hazard=HAZARD,
        rulebook=rulebook,
        budget=budget or extract_mod.ExtractionBudget(max_calls_per_month=1000),
        documents=documents,
        call=call or fn,
        model_ref=MODEL,
    )
    return extraction, calls


# ---------------------------------------------------------------------------
# The golden set
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", load_golden_set()["documents"], ids=lambda c: c["name"])
def test_golden_set_reproduces_the_expected_json(case):
    """Each document produces exactly the figures it states — no more, no less."""

    figures, rejected = extract_mod.parse_response(case["response"], case["doc"], MODEL)

    assert [f.value for f in figures] == [e["value"] for e in case["expected"]]
    assert [f.unit for f in figures] == [e["unit"] for e in case["expected"]]
    assert [f.stated_by for f in figures] == [e["stated_by"] for e in case["expected"]]
    assert [f.date for f in figures] == [e["date"] for e in case["expected"]]
    for figure, expected in zip(figures, case["expected"]):
        if "cumulative_or_new" in expected:
            assert figure.cumulative_or_new == expected["cumulative_or_new"]
        # Hard rule 3's mechanical guarantee: the quote is IN the document.
        assert figure.quote in case["doc"]["body"]

    assert sorted(r["reason"] for r in rejected) == sorted(case["expected_rejected"])


def test_golden_set_covers_the_awkward_cases():
    """The fixture is only an acceptance test if it still has teeth."""

    cases = {c["name"]: c for c in load_golden_set()["documents"]}
    assert len(cases) == 10
    assert cases["no_qualifying_figure"]["expected"] == []
    assert cases["households_figure"]["expected"][0]["unit"] == "households"
    assert cases["cumulative_figure"]["expected"][0]["cumulative_or_new"] == "cumulative"
    # Three provincial figures, reported separately and never added up.
    assert len(cases["per_province_figures_not_summed"]["expected"]) == 3


# ---------------------------------------------------------------------------
# Extraction discipline
# ---------------------------------------------------------------------------


def test_prompt_forbids_estimation_conversion_and_summing():
    """The instructions the hard rule depends on are actually in the prompt."""

    prompt = extract_mod.build_prompt(_case("ocha_situation_report")["doc"], ISO3, HAZARD)
    lowered = prompt.lower()
    assert "never estimate" in lowered
    assert "never convert units" in lowered
    assert "never sum" in lowered
    assert "exact sentence" in lowered
    assert "empty list" in lowered
    # The document itself must reach the model, or there is nothing to read.
    assert "145,000 people have been affected" in prompt


def test_a_quote_the_document_does_not_contain_is_rejected():
    case = _case("quote_not_in_document")
    figures, rejected = extract_mod.parse_response(case["response"], case["doc"], MODEL)
    assert figures == []
    assert rejected[0]["reason"] == "quote_not_found_in_document"


def test_quote_matching_survives_whitespace_and_typography():
    """A quote must be found, not matched byte-for-byte after HTML mangling."""

    doc = {
        "doc_id": "5000",
        "body": "The ministry reports\n  that 4,000 people\twere affected — an increase.",
    }
    response = json.dumps(
        {
            "figures": [
                {
                    "value": 4000,
                    "unit": "people",
                    "quote": "The ministry reports that 4,000 people were affected - an increase.",
                    "stated_by": "ministry",
                    "area": "",
                    "date": "",
                    "cumulative_or_new": "unstated",
                }
            ]
        }
    )
    figures, rejected = extract_mod.parse_response(response, doc, MODEL)
    assert rejected == []
    assert figures[0].value == 4000.0


@pytest.mark.parametrize(
    "raw, reason",
    [
        ({"value": "not a number", "unit": "people", "quote": "q"}, "value_not_a_number"),
        ({"value": 10, "unit": "families", "quote": "q"}, "unknown_unit"),
        ({"value": 10, "unit": "people", "quote": ""}, "missing_quote"),
        ({"value": -5, "unit": "people", "quote": "q"}, "value_not_a_number"),
    ],
)
def test_malformed_figures_are_rejected_with_a_reason(raw, reason):
    """A dropped figure always says why — a silent drop reads as 'nothing stated'."""

    doc = {"doc_id": "5001", "body": "q"}
    figures, rejected = extract_mod.parse_response(
        json.dumps({"figures": [raw]}), doc, MODEL
    )
    assert figures == []
    assert rejected[0]["reason"] == reason


def test_an_unparseable_response_is_a_rejection_not_a_crash():
    figures, rejected = extract_mod.parse_response(
        "I could not find any figures, sorry.", {"doc_id": "5002", "body": ""}, MODEL
    )
    assert figures == []
    assert rejected[0]["reason"] == "unparseable_response"


def test_a_households_unit_is_never_silently_read_as_people():
    """The reason unknown_unit is a rejection rather than a default."""

    case = _case("households_figure")
    figures, _ = extract_mod.parse_response(case["response"], case["doc"], MODEL)
    assert figures[0].unit == extract_mod.UNIT_HOUSEHOLDS
    assert figures[0].value == 30_000.0  # still households at this stage


# ---------------------------------------------------------------------------
# Deterministic post-processing
# ---------------------------------------------------------------------------


def test_households_convert_with_a_recorded_multiplier(con, rulebook):
    extraction, _ = _extract_all(
        con, rulebook, documents=[_case("households_figure")["doc"]]
    )
    candidates, _ = figures_mod.build_candidates(
        extraction, iso3=ISO3, ym=YM, hazard=HAZARD, rulebook=rulebook,
        exposure_ceiling=CEILING,
    )
    assert len(candidates) == 1
    conversion = candidates[0].detail["household_conversion"]
    # PHL is in the rulebook's by_iso3 map at 4.1.
    assert conversion["multiplier"] == 4.1
    assert conversion["multiplier_origin"] == "by_iso3"
    assert conversion["stated_value"] == 30_000.0
    assert candidates[0].value == pytest.approx(123_000.0)


def test_household_multiplier_falls_back_to_the_declared_default(rulebook):
    value, origin = figures_mod.household_multiplier("XKX", rulebook)
    assert origin == "default"
    assert value == rulebook.get("reliefweb.household_conversion.default_multiplier")


def test_a_figure_stated_in_people_carries_no_conversion_record(con, rulebook):
    extraction, _ = _extract_all(
        con, rulebook, documents=[_case("ocha_situation_report")["doc"]]
    )
    candidates, _ = figures_mod.build_candidates(
        extraction, iso3=ISO3, ym=YM, hazard=HAZARD, rulebook=rulebook,
    )
    assert candidates[0].detail["household_conversion"] is None


def test_a_quote_repeated_across_documents_collapses_to_one_figure(con, rulebook):
    docs = [_case("ocha_situation_report")["doc"], _case("duplicate_of_sitrep_3")["doc"]]
    extraction, _ = _extract_all(con, rulebook, documents=docs)
    assert len(extraction.figures) == 2  # both documents state it

    candidates, rejected = figures_mod.build_candidates(
        extraction, iso3=ISO3, ym=YM, hazard=HAZARD, rulebook=rulebook,
        exposure_ceiling=CEILING,
    )
    assert len(candidates) == 1
    assert candidates[0].value == 145_000.0
    assert [r["reason"] for r in rejected] == ["duplicate_quote"]


def test_per_province_figures_are_kept_separate_never_summed(con, rulebook):
    extraction, _ = _extract_all(
        con, rulebook, documents=[_case("per_province_figures_not_summed")["doc"]]
    )
    candidates, _ = figures_mod.build_candidates(
        extraction, iso3=ISO3, ym=YM, hazard=HAZARD, rulebook=rulebook,
        exposure_ceiling=CEILING,
    )
    values = sorted(c.value for c in candidates)
    assert values == [8_500.0, 12_000.0, 21_300.0]
    assert 41_800.0 not in values  # the sum must never appear


def test_a_figure_above_the_gdacs_ceiling_is_rejected_with_a_reason(con, rulebook):
    extraction, _ = _extract_all(
        con, rulebook, documents=[_case("figure_above_exposure_ceiling")["doc"]]
    )
    candidates, rejected = figures_mod.build_candidates(
        extraction, iso3=ISO3, ym=YM, hazard=HAZARD, rulebook=rulebook,
        exposure_ceiling=CEILING,
    )
    assert candidates == []
    assert rejected[0]["reason"] == "exceeds_gdacs_exposure_ceiling"
    assert rejected[0]["exposed_population"] == CEILING


@pytest.mark.parametrize(
    "stated_by, tier",
    [
        ("NDRRMC", "government"),
        ("the Ministry of Health", "government"),
        ("OCHA", "un_agency"),
        ("Philippine Red Cross", "ifrc_ngo"),
        ("local media", "media"),
        ("", "unattributed"),
    ],
)
def test_authority_tiers_follow_the_rulebook(stated_by, tier, rulebook):
    _, resolved = figures_mod.authority_tier(stated_by, rulebook)
    assert resolved == tier


def test_the_most_authoritative_latest_figure_is_preferred(con, rulebook):
    """Government beats UN beats NGO beats media; latest wins inside a tier."""

    extraction, _ = _extract_all(con, rulebook, documents=golden_documents())
    candidates, _ = figures_mod.build_candidates(
        extraction, iso3=ISO3, ym=YM, hazard=HAZARD, rulebook=rulebook,
        exposure_ceiling=CEILING,
    )

    preferred = [c for c in candidates if c.preference_rank == 0]
    assert len(preferred) == 1
    # NDRRMC (government, 18 March) over the provincial reports (government,
    # 10 March) and everything in the tiers below.
    assert preferred[0].value == 152_340.0
    assert preferred[0].detail["authority_tier"] == "government"
    assert preferred[0].detail["quote"].startswith("The NDRRMC reports")
    # The quote, doc_url and extraction model are all on the candidate.
    assert preferred[0].doc_url.endswith("ndrrmc-sitrep-12")
    assert preferred[0].extraction_model == MODEL

    # ...and the largest figure available is NOT the answer, which is the
    # whole point of giving this rung its own precedence.
    assert max(c.value for c in candidates) > preferred[0].value


def test_post_processing_is_deterministic(con, rulebook):
    """Same inputs, same candidates — including the order they rank in."""

    extraction, _ = _extract_all(con, rulebook, documents=golden_documents())
    first, _ = figures_mod.build_candidates(
        extraction, iso3=ISO3, ym=YM, hazard=HAZARD, rulebook=rulebook,
        exposure_ceiling=CEILING,
    )
    second, _ = figures_mod.build_candidates(
        extraction, iso3=ISO3, ym=YM, hazard=HAZARD, rulebook=rulebook,
        exposure_ceiling=CEILING,
    )
    assert [(c.value, c.source_ref, c.preference_rank) for c in first] == [
        (c.value, c.source_ref, c.preference_rank) for c in second
    ]


# ---------------------------------------------------------------------------
# The ladder
# ---------------------------------------------------------------------------


def test_extracted_candidates_flow_through_the_ladder(con, rulebook):
    """A flood month with only ReliefWeb evidence resolves from rung 2."""

    seed_gdacs_event(con, exposed_population=CEILING)
    seed_population(con)
    seed_reliefweb_docs(con, golden_documents())

    extracted, provenance = impact_mod.extract_rung_for_cell(
        con,
        iso3=ISO3,
        ym=YM,
        hazard=HAZARD,
        rulebook=rulebook,
        budget=extract_mod.ExtractionBudget(max_calls_per_month=1000),
        fetch_documents=False,
        call=golden_call_fn()[0],
    )
    assert extracted, "the golden documents should yield candidates"

    found = cand_mod.build_candidates(
        con, ISO3, YM, HAZARD, rulebook, extracted=extracted
    )
    verdict = reconcile_mod.reconcile(
        iso3=ISO3, ym=YM, hazard=HAZARD, candidates=found, rulebook=rulebook,
        national_population=115_000_000.0, today=AFTER_FREEZE,
    )
    assert verdict.status == reconcile_mod.STATUS_RESOLVED_VALUE
    assert verdict.value == 152_340.0
    assert verdict.rule_fired == "ladder:reliefweb_extracted"
    assert verdict.provenance["source"] == extract_mod.SOURCE
    # Provenance carries the evidence the figure was transcribed, not made up.
    winner = verdict.provenance
    assert winner["decision"]["rungs_populated"][0] == "reliefweb_extracted"
    assert provenance["extraction"]["docs_read"] == len(golden_documents())


def test_emdat_still_outranks_an_extracted_figure(rulebook):
    extracted = make_candidate(
        "reliefweb_extracted", 152_340.0, preference_rank=0, detail={"quote": "…"}
    )
    verdict = reconcile_mod.reconcile(
        iso3=ISO3, ym=YM, hazard=HAZARD,
        candidates=[make_candidate("emdat", 90_000.0), extracted],
        rulebook=rulebook, today=AFTER_FREEZE,
    )
    assert verdict.value == 90_000.0
    assert verdict.rule_fired == "ladder:emdat"


def test_an_extracted_figure_outranks_ifrc_go(rulebook):
    extracted = make_candidate(
        "reliefweb_extracted", 152_340.0, preference_rank=0, detail={"quote": "…"}
    )
    verdict = reconcile_mod.reconcile(
        iso3=ISO3, ym=YM, hazard=HAZARD,
        candidates=[make_candidate("ifrc_go", 900_000.0), extracted],
        rulebook=rulebook, today=AFTER_FREEZE,
    )
    assert verdict.value == 152_340.0
    assert verdict.rule_fired == "ladder:reliefweb_extracted"


def test_preference_rank_beats_largest_wins_only_on_its_own_rung(rulebook):
    """Rungs without a preference rank keep the existing largest-wins rule."""

    ranked = [
        make_candidate("reliefweb_extracted", 10_000.0, source_ref="a", preference_rank=0),
        make_candidate("reliefweb_extracted", 99_000.0, source_ref="b", preference_rank=1),
    ]
    unranked = [
        make_candidate("ifrc_go", 10_000.0, source_ref="c"),
        make_candidate("ifrc_go", 99_000.0, source_ref="d"),
    ]
    assert reconcile_mod._best_on_rung(ranked, "reliefweb_extracted").value == 10_000.0
    assert reconcile_mod._best_on_rung(unranked, "ifrc_go").value == 99_000.0


def test_candidates_persist_the_quote_and_rank(con, rulebook):
    extracted = [
        make_candidate(
            "reliefweb_extracted", 152_340.0, preference_rank=0,
            extraction_model=MODEL, detail={"quote": "The NDRRMC reports …"},
        )
    ]
    cand_mod.write_candidates(con, extracted, ISO3, YM, HAZARD)
    row = con.execute(
        """
        SELECT preference_rank, detail_json, extraction_model, doc_url
        FROM haz_impact_candidates WHERE source = 'reliefweb_extracted'
        """
    ).fetchone()
    assert row[0] == 0
    assert "The NDRRMC reports" in json.loads(row[1])["quote"]
    assert row[2] == MODEL
    assert row[3]


# ---------------------------------------------------------------------------
# No spend without a trigger
# ---------------------------------------------------------------------------


def test_no_extraction_call_for_a_non_triggered_country_month(con, rulebook):
    """The acceptance criterion: a cell that resolved without reading pays nothing."""

    seed_gdacs_event(con, exposed_population=CEILING)
    seed_population(con)
    # Documents exist for BOTH countries; only one country is triggered.
    seed_reliefweb_docs(con, golden_documents(), iso3="PHL")
    seed_reliefweb_docs(con, golden_documents(), iso3="VNM")
    con.execute(
        """
        INSERT INTO haz_triggers (iso3, year, month, hazard, triggered, trigger_source)
        VALUES ('PHL', 2024, 3, 'FL', TRUE, 'gdacs'),
               ('VNM', 2024, 3, 'FL', FALSE, NULL)
        """
    )

    call, calls = golden_call_fn()
    run = impact_mod.resolve_triggered_cells(
        con,
        ym=YM,
        hazard=HAZARD,
        iso3s=impact_mod.triggered_iso3s(con, YM, HAZARD),
        rulebook=rulebook,
        fetches={},
        today=AFTER_FREEZE,
        fetch_documents=False,
        call=call,
    )

    assert run.cells == 1
    assert calls, "the triggered cell should have been read"
    assert all("VNM" not in prompt for prompt in calls)
    # And nothing was ever billed against the untriggered country.
    billed = [
        row[0]
        for row in con.execute("SELECT DISTINCT iso3 FROM haz_doc_extractions").fetchall()
    ]
    assert billed == ["PHL"]

    # The triggered cell resolved end to end, from the extracted rung, with
    # the extraction's own story on the row.
    status, value, rule, provenance = con.execute(
        """
        SELECT status, value, rule_fired, provenance_json FROM haz_resolutions
        WHERE iso3 = 'PHL' AND year = 2024 AND month = 3 AND hazard = 'FL'
        """
    ).fetchone()
    assert (status, value, rule) == ("RESOLVED_VALUE", 152_340.0, "ladder:reliefweb_extracted")
    extraction = json.loads(provenance)["reliefweb_extraction"]
    assert extraction["extraction"]["docs_read"] == len(golden_documents())
    assert extraction["extraction"]["budget_capped"] is False
    assert run.extraction_calls == len(golden_documents())


def test_extraction_is_skipped_when_a_higher_rung_already_has_a_figure(con, rulebook):
    """reliefweb_extracted sits below EM-DAT and cannot win those cells."""

    seed_reliefweb_docs(con, golden_documents())
    from resolver.hazard_resolution.sources import RawRecord, store_raw_records

    store_raw_records(
        con,
        "emdat",
        [
            RawRecord(
                record_id="emdat-2024-0001-PHL",
                payload={
                    "disno": "2024-0001-PHL",
                    "iso3": ISO3,
                    "hazard": HAZARD,
                    "total_affected": 90_000.0,
                    "start_date": "2024-03-05",
                    "end_date": "2024-03-09",
                },
                iso3=ISO3,
                ym=YM,
                hazard=HAZARD,
            )
        ],
    )

    call, calls = golden_call_fn()
    extracted, provenance = impact_mod.extract_rung_for_cell(
        con, iso3=ISO3, ym=YM, hazard=HAZARD, rulebook=rulebook,
        budget=extract_mod.ExtractionBudget(max_calls_per_month=1000),
        fetch_documents=False, call=call,
    )
    assert extracted == []
    assert calls == []
    assert provenance["skipped_reason"] == "higher_rung_populated"
    assert provenance["higher_rungs"] == ["emdat"]


def test_the_skip_can_be_turned_off_in_the_rulebook(con):
    """The trade — cheaper runs vs adjacent-rung conflict flags — is configurable."""

    rulebook = make_rulebook({"extraction": {"skip_when_higher_rung_populated": False}})
    seed_reliefweb_docs(con, [_case("ocha_situation_report")["doc"]])
    from resolver.hazard_resolution.sources import RawRecord, store_raw_records

    store_raw_records(
        con, "emdat",
        [RawRecord(
            record_id="emdat-x", iso3=ISO3, ym=YM, hazard=HAZARD,
            payload={"disno": "x", "iso3": ISO3, "hazard": HAZARD,
                     "total_affected": 90_000.0, "start_date": "2024-03-05",
                     "end_date": "2024-03-09"},
        )],
    )
    call, calls = golden_call_fn()
    extracted, _ = impact_mod.extract_rung_for_cell(
        con, iso3=ISO3, ym=YM, hazard=HAZARD, rulebook=rulebook,
        budget=extract_mod.ExtractionBudget(max_calls_per_month=1000),
        fetch_documents=False, call=call,
    )
    assert calls, "with the skip off, the documents should be read"
    assert extracted


def test_extraction_disabled_in_the_rulebook_makes_no_calls(con):
    rulebook = make_rulebook({"extraction": {"enabled": False}})
    seed_reliefweb_docs(con, golden_documents())
    call, calls = golden_call_fn()
    extracted, provenance = impact_mod.extract_rung_for_cell(
        con, iso3=ISO3, ym=YM, hazard=HAZARD, rulebook=rulebook,
        budget=extract_mod.ExtractionBudget(max_calls_per_month=1000),
        fetch_documents=False, call=call,
    )
    assert extracted == []
    assert calls == []
    assert provenance["skipped_reason"] == "extraction_disabled"


# ---------------------------------------------------------------------------
# The cost guard
# ---------------------------------------------------------------------------


def test_the_monthly_cap_stops_reading_and_records_the_cell(con, rulebook):
    budget = extract_mod.ExtractionBudget(max_calls_per_month=3)
    extraction, calls = _extract_all(
        con, rulebook, documents=golden_documents(), budget=budget
    )
    assert len(calls) == 3
    assert extraction.budget_capped is True
    assert budget.exhausted
    assert budget.capped_cells == [f"{ISO3}/{HAZARD}/{YM}"]


def test_the_cap_counts_calls_made_by_earlier_runs(con, rulebook):
    """Three CLI invocations in a month share one allowance, not three."""

    first = extract_mod.ExtractionBudget(max_calls_per_month=4)
    _extract_all(con, rulebook, documents=golden_documents(), budget=first)
    assert first.calls_this_run == 4

    reloaded = extract_mod.load_budget(con, make_rulebook(
        {"extraction": {"max_calls_per_month": 4}}
    ))
    assert reloaded.used_this_month == 4
    assert reloaded.exhausted


def test_a_cached_document_is_never_read_twice(con, rulebook):
    documents = golden_documents()[:3]
    budget = extract_mod.ExtractionBudget(max_calls_per_month=1000)
    first, first_calls = _extract_all(con, rulebook, documents=documents, budget=budget)
    assert len(first_calls) == 3

    second, second_calls = _extract_all(
        con, rulebook, documents=documents, budget=budget
    )
    assert second_calls == [], "a re-run must cost nothing"
    assert second.docs_cached == 3
    # ...and it must produce the same figures it did the first time.
    assert [f.value for f in second.figures] == [f.value for f in first.figures]
    assert [f.quote for f in second.figures] == [f.quote for f in first.figures]


def test_a_new_prompt_version_invalidates_the_cache(con, rulebook):
    documents = golden_documents()[:2]
    budget = extract_mod.ExtractionBudget(max_calls_per_month=1000)
    _extract_all(con, rulebook, documents=documents, budget=budget)

    current = str(rulebook.get("extraction.prompt_version"))
    bumped = make_rulebook({"extraction": {"prompt_version": current + "-bumped"}})
    _, calls = _extract_all(con, bumped, documents=documents, budget=budget)
    assert len(calls) == 2


def test_every_call_is_recorded_in_the_ledger(con, rulebook):
    _extract_all(con, rulebook, documents=golden_documents()[:2])
    rows = con.execute(
        """
        SELECT doc_id, model, prompt_version, status, n_figures, prompt_tokens,
               cost_usd, doc_url
        FROM haz_doc_extractions ORDER BY doc_id
        """
    ).fetchall()
    assert len(rows) == 2
    for row in rows:
        assert row[1] == MODEL
        assert row[2] == str(rulebook.get("extraction.prompt_version"))
        assert row[3] == "ok"
        assert row[5] == 6000  # the seam's reported prompt tokens
        assert row[7]


def test_a_failed_call_is_recorded_as_an_error_not_as_silence(con, rulebook):
    """A call that failed must not read downstream as 'the document said nothing'."""

    def failing(model_ref, prompt, rulebook_):
        return "", {"prompt_tokens": 100, "completion_tokens": 0}, "provider timeout"

    extraction = extract_mod.extract_for_cell(
        con, iso3=ISO3, ym=YM, hazard=HAZARD, rulebook=rulebook,
        budget=extract_mod.ExtractionBudget(max_calls_per_month=10),
        documents=golden_documents()[:1], call=failing, model_ref=MODEL,
    )
    assert extraction.docs_failed == 1
    assert extraction.figures == []
    row = con.execute(
        "SELECT status, error FROM haz_doc_extractions"
    ).fetchone()
    assert row[0] == "error"
    assert "timeout" in row[1]


def test_a_failed_call_is_retried_on_the_next_run(con, rulebook):
    """An error row is not a cache entry — the next run reads the document.

    This is the poisoned-cache regression: before the fix, a transient
    outage wrote status='error' rows that load_cached_extraction served
    forever, so a document that failed once was never read again.
    """

    documents = golden_documents()[:1]

    def failing(model_ref, prompt, rulebook_):
        return "", {"prompt_tokens": 100, "completion_tokens": 0}, "provider timeout"

    first = extract_mod.extract_for_cell(
        con, iso3=ISO3, ym=YM, hazard=HAZARD, rulebook=rulebook,
        budget=extract_mod.ExtractionBudget(max_calls_per_month=10),
        documents=documents, call=failing, model_ref=MODEL,
    )
    assert first.docs_failed == 1

    good, calls = golden_call_fn()
    second = extract_mod.extract_for_cell(
        con, iso3=ISO3, ym=YM, hazard=HAZARD, rulebook=rulebook,
        budget=extract_mod.ExtractionBudget(max_calls_per_month=10),
        documents=documents, call=good, model_ref=MODEL,
    )
    assert len(calls) == 1, "the errored document must be re-read, not served from cache"
    assert second.docs_cached == 0
    assert second.figures, "the retry's figures must land"
    row = con.execute("SELECT status FROM haz_doc_extractions").fetchone()
    assert row[0] == "ok", "the retry replaces the error row"


def test_a_call_that_never_reached_the_provider_costs_no_budget(con, rulebook):
    """Missing API key / breaker cooldown must not burn the monthly cap.

    Those calls return instantly with zero token usage; charging them wrote
    poison rows AND consumed the 1500-call allowance on calls that never
    left the process.
    """

    def unreached(model_ref, prompt, rulebook_):
        return "", {}, "missing ANTHROPIC_API_KEY"

    budget = extract_mod.ExtractionBudget(max_calls_per_month=10)
    extraction = extract_mod.extract_for_cell(
        con, iso3=ISO3, ym=YM, hazard=HAZARD, rulebook=rulebook,
        budget=budget, documents=golden_documents()[:3], call=unreached, model_ref=MODEL,
    )
    assert extraction.docs_failed == 3
    assert extraction.calls_made == 0
    assert budget.calls_this_run == 0
    # The error rows exist for the audit trail but count nothing toward the
    # calendar-month spend cap — no tokens were ever billed.
    assert extract_mod.calls_this_calendar_month(con) == 0


def test_the_extraction_cache_is_scoped_to_the_cell(con, rulebook):
    """A document shared across cells is answered per cell, never reused.

    The prompt names the country, the hazard and the target month, so a
    cached answer to one cell's question is not an answer to another's — a
    typhoon-plus-flooding sitrep must be read once per hazard.
    """

    documents = golden_documents()[:1]

    first_call, first_calls = golden_call_fn()
    extract_mod.extract_for_cell(
        con, iso3=ISO3, ym=YM, hazard="FL", rulebook=rulebook,
        budget=extract_mod.ExtractionBudget(max_calls_per_month=10),
        documents=documents, call=first_call, model_ref=MODEL,
    )
    assert len(first_calls) == 1

    second_call, second_calls = golden_call_fn()
    second = extract_mod.extract_for_cell(
        con, iso3=ISO3, ym=YM, hazard="TC", rulebook=rulebook,
        budget=extract_mod.ExtractionBudget(max_calls_per_month=10),
        documents=documents, call=second_call, model_ref=MODEL,
    )
    assert len(second_calls) == 1, "the other hazard's cell must make its own call"
    assert second.docs_cached == 0

    third_call, third_calls = golden_call_fn()
    third = extract_mod.extract_for_cell(
        con, iso3=ISO3, ym=YM, hazard="FL", rulebook=rulebook,
        budget=extract_mod.ExtractionBudget(max_calls_per_month=10),
        documents=documents, call=third_call, model_ref=MODEL,
    )
    assert len(third_calls) == 0, "the SAME cell's re-run is still free"
    assert third.docs_cached == 1


def test_a_fabricated_value_paired_with_a_real_sentence_is_rejected():
    """Quote-in-body alone is not verification — the value must be IN the quote."""

    case = _case("ocha_situation_report")
    doc = case["doc"]
    real_sentence = doc["body"].split(".")[0] + "."
    response = json.dumps(
        {
            "figures": [
                {
                    "value": 987_654,
                    "unit": "people",
                    "quote": real_sentence,
                    "stated_by": "",
                    "area": "",
                    "date": "",
                    "cumulative_or_new": "unstated",
                }
            ]
        }
    )
    figures, rejected = extract_mod.parse_response(response, doc, MODEL)
    if any(f.value == 987_654 for f in figures):  # the sentence really states it?
        raise AssertionError("test fixture invalid: pick a sentence without the value")
    assert figures == []
    assert any(r.get("reason") == "value_not_found_in_quote" for r in rejected)


def test_value_matching_reads_grouped_and_worded_numbers():
    assert extract_mod._value_stated_in_quote(12345, "some 12,345 people were affected")
    assert extract_mod._value_stated_in_quote(1_200_000, "about 1.2 million people affected")
    assert extract_mod._value_stated_in_quote(5000, "5 000 persons affected")
    assert not extract_mod._value_stated_in_quote(500_000, "heavy rains continued for days.")
    assert not extract_mod._value_stated_in_quote(500_000, "assistance reached 3,000 people")


def test_the_prompt_names_the_period_and_the_country():
    doc = _case("ocha_situation_report")["doc"]
    prompt = extract_mod.build_prompt(doc, ISO3, HAZARD, "the Philippines", ym="2024-03")
    assert "March 2024" in prompt
    assert "the Philippines" in prompt


def test_real_calls_write_llm_calls_telemetry_and_seam_calls_do_not(con, rulebook, monkeypatch):
    """The production seam writes cost telemetry; an injected test seam never does.

    The machine's spend must reach ``llm_calls`` (the Costs page reads
    nothing else), and network-free tests must not reach for the Pythia DB.
    """

    logged: list[dict] = []
    fake_call, _ = golden_call_fn()
    monkeypatch.setattr(extract_mod, "_default_call", fake_call)
    monkeypatch.setattr(
        extract_mod, "_log_llm_call", lambda **kw: logged.append(kw)
    )

    extract_mod.extract_for_cell(
        con, iso3=ISO3, ym=YM, hazard=HAZARD, rulebook=rulebook,
        budget=extract_mod.ExtractionBudget(max_calls_per_month=10),
        documents=golden_documents()[:2], call=None, model_ref=MODEL,
    )
    assert len(logged) == 2
    assert logged[0]["iso3"] == ISO3
    assert logged[0]["hazard"] == HAZARD

    logged.clear()
    seam, _ = golden_call_fn()
    extract_mod.extract_for_cell(
        con, iso3="VNM", ym=YM, hazard=HAZARD, rulebook=rulebook,
        budget=extract_mod.ExtractionBudget(max_calls_per_month=10),
        documents=golden_documents()[:2], call=seam, model_ref=MODEL,
    )
    assert logged == [], "an injected seam is not a billable call"


def test_the_telemetry_row_lands_in_llm_calls_with_the_resolution_phase(
    tmp_path, monkeypatch
):
    """End to end: _log_llm_call writes a costed row phase='hazard_extraction'.

    That phase maps to the Costs page's 'resolution' bucket
    (resolver/query/costs.py::phase_group) — the pairing that keeps the
    machine's spend out of 'other'.
    """

    pythia_db = tmp_path / "pythia.duckdb"
    monkeypatch.setenv("PYTHIA_DB_URL", f"duckdb:///{pythia_db}")

    from pythia.db import schema as pythia_schema

    con = pythia_schema.connect(read_only=False)
    pythia_schema.ensure_schema(con)

    extract_mod._log_llm_call(
        model_ref=MODEL,
        prompt="extract the figures",
        response='{"figures": []}',
        usage={"prompt_tokens": 6000, "completion_tokens": 200},
        error="",
        iso3=ISO3,
        hazard=HAZARD,
        ym=YM,
    )

    row = con.execute(
        """
        SELECT phase, iso3, hazard_code, model_id, cost_usd
        FROM llm_calls WHERE phase = 'hazard_extraction'
        """
    ).fetchone()
    assert row is not None, "the rich telemetry row must be written"
    assert row[1] == ISO3
    assert row[2] == HAZARD
    assert "haiku" in str(row[3])
    assert row[4] and row[4] > 0, "the row must carry a non-zero cost"

    from resolver.query.costs import phase_group

    assert phase_group(row[0]) == "resolution"


def test_the_cap_still_collects_already_paid_cached_figures(con, rulebook):
    """Budget exhaustion stops FRESH reads only — cached figures still land."""

    documents = golden_documents()[:3]
    call, _ = golden_call_fn()
    # Pay for all three documents first.
    first = extract_mod.extract_for_cell(
        con, iso3=ISO3, ym=YM, hazard=HAZARD, rulebook=rulebook,
        budget=extract_mod.ExtractionBudget(max_calls_per_month=10),
        documents=documents, call=call, model_ref=MODEL,
    )
    # Now the month's cap is spent: a re-run must still return every cached
    # figure rather than abandoning them at the first uncached document.
    capped = extract_mod.ExtractionBudget(max_calls_per_month=3, used_this_month=3)
    second = extract_mod.extract_for_cell(
        con, iso3=ISO3, ym=YM, hazard=HAZARD, rulebook=rulebook,
        budget=capped, documents=documents, call=call, model_ref=MODEL,
    )
    assert second.docs_cached == 3
    assert [f.value for f in second.figures] == [f.value for f in first.figures]


# ---------------------------------------------------------------------------
# Document selection
# ---------------------------------------------------------------------------


def test_documents_are_ranked_most_authoritative_then_most_recent(rulebook):
    items = [
        {"id": "1", "fields": {"title": "media", "body": "b",
                               "date": {"created": "2024-03-20T00:00:00+00:00"},
                               "source": [{"shortname": "Reuters", "name": "Reuters",
                                           "type": {"name": "Media"}}]}},
        {"id": "2", "fields": {"title": "ocha old", "body": "b",
                               "date": {"created": "2024-03-02T00:00:00+00:00"},
                               "source": [{"shortname": "OCHA", "name": "OCHA"}]}},
        {"id": "3", "fields": {"title": "ocha new", "body": "b",
                               "date": {"created": "2024-03-18T00:00:00+00:00"},
                               "source": [{"shortname": "OCHA", "name": "OCHA"}]}},
    ]
    selected, discarded = docs_mod.select_documents(items, ISO3, YM, HAZARD, rulebook)
    assert [d["doc_id"] for d in selected] == ["3", "2", "1"]
    assert discarded == 0


def test_the_document_cap_is_the_rulebooks(rulebook):
    capped = make_rulebook({"reliefweb": {"documents": {"max_docs_per_cell": 2}}})
    items = [
        {"id": str(i), "fields": {"title": f"t{i}", "body": "b",
                                  "date": {"created": f"2024-03-{i:02d}T00:00:00+00:00"},
                                  "source": [{"shortname": "OCHA"}]}}
        for i in range(1, 6)
    ]
    selected, discarded = docs_mod.select_documents(items, ISO3, YM, HAZARD, capped)
    assert len(selected) == 2
    assert discarded == 3


def test_a_document_body_is_stored_truncated_to_what_will_be_sent(rulebook):
    short = make_rulebook({"reliefweb": {"documents": {"body_char_limit": 50}}})
    items = [{"id": "9", "fields": {"title": "t", "body": "x" * 500,
                                    "date": {"created": "2024-03-05T00:00:00+00:00"},
                                    "source": [{"shortname": "OCHA"}]}}]
    selected, _ = docs_mod.select_documents(items, ISO3, YM, HAZARD, short)
    assert len(selected[0]["body"]) == 50
    assert selected[0]["body_truncated"] is True


def test_a_reliefweb_outage_is_unread_never_empty(con, rulebook):
    def failing_post(url, payload, params, timeout):
        raise RuntimeError("connection reset")

    outcome = docs_mod.fetch_documents(
        con, ISO3, YM, HAZARD, rulebook, post=failing_post
    )
    assert outcome.ok is False
    assert "connection reset" in outcome.error


def test_the_document_query_filters_country_hazard_window_and_format(rulebook):
    captured: dict = {}

    def capturing_post(url, payload, params, timeout):
        captured["payload"] = payload
        captured["params"] = params
        return {"data": [], "totalCount": 0}

    con = duckdb.connect(":memory:")
    ensure_haz_schema(con)
    docs_mod.fetch_documents(con, ISO3, YM, HAZARD, rulebook, post=capturing_post)

    conditions = captured["payload"]["filter"]["conditions"]
    fields = {c["field"]: c["value"] for c in conditions}
    assert fields["country.iso3"] == "PHL"
    assert "Flood" in fields["disaster_type.name"]
    assert "Situation Report" in fields["format.name"]
    assert fields["date.created"]["from"].startswith("2024-03-01")
    # Month end + the rulebook's publication pad (14 days).
    assert fields["date.created"]["to"].startswith("2024-04-14")
    assert captured["params"]["appname"]


def test_consecutive_document_fetches_are_paced_by_the_rulebook(con, monkeypatch):
    """The free ReliefWeb API is paced the same way the silence sweep paces it."""

    slept: list[float] = []
    monkeypatch.setattr("time.sleep", lambda s: slept.append(s))
    paced = make_rulebook({"reliefweb": {"documents": {"request_delay_sec": 5.0}}})
    impact_mod._LAST_DOC_FETCH.clear()

    impact_mod._pace_reliefweb(paced)
    assert slept == [], "the first request waits for nothing"
    impact_mod._pace_reliefweb(paced)
    assert slept and slept[0] > 4.0
