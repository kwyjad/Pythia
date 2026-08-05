# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Snapshot tests of the assembled SPD prompt for PA questions — Phase 6.

The unit-level behaviour of the block lives in
``resolver/tests/test_hazard_resolution_prompt_block.py``. What these tests
add is the seam: that a fully assembled ``build_spd_prompt_v2`` prompt for a
cyclone, a flood and a drought PA question actually CONTAINS the block, that
questions the machine does not resolve do not get one, and that a rulebook
change moves the assembled prompt text with no code change.

The distinction matters because the failure mode here is silent. A loader
that returns text nobody renders costs money and changes nothing — the repo
has already lost CrisisWatch and the HS grounding packs that way (see the
"loaded but never rendered" entry in CLAUDE.md). Asserting on the loader
alone would not have caught either.

Network-free and model-free: base rates are seeded into a temp DuckDB and
the prompt builder is pointed at it.
"""

from __future__ import annotations

from typing import Any

import duckdb
import pytest

import forecaster.prompts as prompts
from resolver.hazard_resolution.schema import ensure_haz_schema
from resolver.tests.hazard_resolution_utils import make_rulebook, seed_base_rates

RESEARCH: dict[str, Any] = {"sources": []}
TRIAGE = {"tier": "priority", "triage_score": 0.8}

#: (iso3, country, hazard, window start, forecast calendar months, backcast window)
CASES = {
    "cyclone": ("PHL", "Philippines", "TC", "2026-09-01", [9, 10, 11, 12, 1, 2],
                "2000-01..2026-05"),
    "flood": ("PAK", "Pakistan", "FL", "2026-06-01", [6, 7, 8, 9, 10, 11],
              "2010-01..2026-05"),
    "drought": ("ETH", "Ethiopia", "DR", "2026-03-01", [3, 4, 5, 6, 7, 8],
                "2017-01..2026-05"),
}

MIX = {
    "overall": {"emdat": 0.48, "ifrc_go": 0.29, "reliefweb_extracted": 0.14,
                "idmc_idu": 0.09},
    "lower_bound_share": 0.09,
}


def _question(iso3: str, hazard: str, window_start: str, metric: str = "PA") -> dict:
    year, month, _ = window_start.split("-")
    target_index = (int(year) * 12 + int(month) - 1) + 5
    return {
        "question_id": f"{iso3}_{hazard}_{metric}_{year}-{month}",
        "iso3": iso3,
        "hazard_code": hazard,
        "metric": metric,
        "resolution_source": "hazard_resolution",
        "window_start_date": window_start,
        "target_month": f"{target_index // 12:04d}-{target_index % 12 + 1:02d}",
    }


@pytest.fixture()
def seeded_db(tmp_path, monkeypatch):
    """A temp DuckDB carrying published base rates for all three cases.

    Returns the ``duckdb:///`` URL and points the prompt builder's DB
    resolution at it, so the prompt goes through the real
    ``duckdb_io.get_db``/``close_db`` path the pipeline uses.
    """
    db_path = tmp_path / "resolver.duckdb"
    con = duckdb.connect(str(db_path))
    ensure_haz_schema(con)
    for iso3, _name, hazard, _start, months, window in CASES.values():
        seed_base_rates(
            con,
            iso3=iso3,
            hazard=hazard,
            occurrence={m: (0.44 + 0.05 * i, 16) for i, m in enumerate(months)},
            quantiles=(12_400, 48_000, 210_000, 900_000, 3_100_000),
            n_events=21,
            window=window,
            window_start=2017 if hazard == "DR" else 2015,
            window_end=2026,
            provenance_mix=MIX,
        )
    con.close()

    url = f"duckdb:///{db_path}"
    monkeypatch.setattr(prompts, "_pythia_db_url_from_config", lambda: url)
    monkeypatch.setenv("RESOLVER_DB_URL", url)
    return url


def _build(question: dict, **kwargs) -> str:
    return prompts.build_spd_prompt_v2(
        question=question,
        history_summary={"type": "no_base_rate", "source": "NONE"},
        hs_triage_entry=TRIAGE,
        research_json=RESEARCH,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# The three snapshots the acceptance criteria name
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", sorted(CASES))
def test_assembled_pa_prompt_contains_the_base_rate_block(case, seeded_db):
    iso3, _name, hazard, window_start, months, window = CASES[case]
    prompt_text = _build(_question(iso3, hazard, window_start))

    assert "PA RESOLUTION BASE RATES" in prompt_text
    # The occurrence table, for the question's own forecast months.
    assert f"{window} backcast" in prompt_text
    # The severity quantiles, with their event count and window.
    assert "n=21 events" in prompt_text
    assert "q50 210k" in prompt_text
    # The provenance mix.
    assert "emdat 48%" in prompt_text
    # The generating process the forecast will be scored against.
    assert "HOW THIS RESOLVES" in prompt_text
    assert "freeze 60d after month end" in prompt_text


def test_cyclone_prompt_states_the_track_geometry_rule(seeded_db):
    iso3, _name, hazard, window_start, _months, _window = CASES["cyclone"]
    prompt_text = _build(_question(iso3, hazard, window_start))

    assert "a cyclone month = a best-track storm passing within 200 km" in prompt_text
    assert "34 kt+" in prompt_text
    assert "EM-DAT > ReliefWeb docs > IFRC GO > IDMC displacement (lower bound)" in prompt_text


def test_flood_prompt_states_the_gdacs_alert_rule(seeded_db):
    iso3, _name, hazard, window_start, _months, _window = CASES["flood"]
    prompt_text = _build(_question(iso3, hazard, window_start))

    assert "a flood month = a GDACS flood alert at orange+" in prompt_text
    assert "silent ReliefWeb sweep resolves ZERO, not missing" in prompt_text


def test_drought_prompt_states_the_ipc_rule_and_its_caveat(seeded_db):
    iso3, _name, hazard, window_start, _months, _window = CASES["drought"]
    prompt_text = _build(_question(iso3, hazard, window_start))

    assert "the rise in IPC/CH Phase 3+ population" in prompt_text
    # The explicit drought caveat: a short, partial record.
    assert "the drought record starts 2017" in prompt_text
    assert "IPC coverage is partial by country" in prompt_text
    # No ladder for drought, and no false claim of one.
    assert "a flood month" not in prompt_text
    assert "a cyclone month" not in prompt_text


def test_the_block_sits_next_to_the_resolver_history_it_supplements(seeded_db):
    iso3, _name, hazard, window_start, _months, _window = CASES["flood"]
    prompt_text = _build(_question(iso3, hazard, window_start))

    # Both are prior anchors; a forecaster reading one should meet the other
    # immediately, not several injects later.
    history_at = prompt_text.index("BASE RATE")
    block_at = prompt_text.index("PA RESOLUTION BASE RATES")
    assert history_at < block_at
    assert prompt_text.index("HS triage output") > block_at


# ---------------------------------------------------------------------------
# The rulebook is the source; the prompt cannot drift from it
# ---------------------------------------------------------------------------


def test_a_rulebook_change_moves_the_assembled_prompt_with_no_code_change(
    seeded_db, monkeypatch
):
    iso3, _name, hazard, window_start, _months, _window = CASES["flood"]
    question = _question(iso3, hazard, window_start)

    shipped = _build(question)
    assert "freeze 60d after month end" in shipped
    assert "GDACS flood alert at orange+" in shipped

    retuned = make_rulebook(
        {"freeze_days": 45, "flood": {"gdacs_trigger_level": "red"}}
    )
    monkeypatch.setattr(
        "resolver.hazard_resolution.rulebook.load_rulebook", lambda *a, **k: retuned
    )
    changed = _build(question)

    assert "freeze 45d after month end" in changed
    assert "GDACS flood alert at red+" in changed
    assert "orange+" not in changed


# ---------------------------------------------------------------------------
# Questions the machine does not resolve get nothing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "iso3,hazard,metric",
    [
        # Phase 3+ is a STOCK; the machine's drought value is the monthly
        # increase in it. Anchoring one on the other is out by the entire
        # pre-existing caseload.
        ("ETH", "DR", "PHASE3PLUS_IN_NEED"),
        # Resolved from GDACS alert levels, not from this machine's detection.
        ("PAK", "FL", "EVENT_OCCURRENCE"),
        # Not a hazard the machine resolves.
        ("ETH", "ACE", "PA"),
        ("ETH", "ACE", "FATALITIES"),
    ],
)
def test_ineligible_questions_get_no_block(iso3, hazard, metric, seeded_db):
    prompt_text = _build(_question(iso3, hazard, "2026-06-01", metric=metric))
    assert "PA RESOLUTION BASE RATES" not in prompt_text
    assert "HOW THIS RESOLVES" not in prompt_text


def test_a_country_with_no_published_base_rates_gets_no_block(seeded_db):
    prompt_text = _build(_question("VNM", "FL", "2026-06-01"))
    assert "PA RESOLUTION BASE RATES" not in prompt_text


def test_a_db_without_the_haz_tables_costs_the_prompt_nothing(tmp_path, monkeypatch):
    # No workflow populates haz_* yet, so this is today's production shape:
    # the prompt must build exactly as it did before Phase 6.
    db_path = tmp_path / "bare.duckdb"
    duckdb.connect(str(db_path)).close()
    url = f"duckdb:///{db_path}"
    monkeypatch.setattr(prompts, "_pythia_db_url_from_config", lambda: url)
    monkeypatch.setenv("RESOLVER_DB_URL", url)

    prompt_text = _build(_question("PAK", "FL", "2026-06-01"))
    assert "PA RESOLUTION BASE RATES" not in prompt_text
    assert "FORECASTING METHOD" in prompt_text


# ---------------------------------------------------------------------------
# Token budget, measured on the assembled prompt
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_block_costs_less_than_its_declared_budget(case, seeded_db):
    iso3, _name, hazard, window_start, _months, _window = CASES[case]
    question = _question(iso3, hazard, window_start)

    with_block = _build(question)
    start = with_block.index("PA RESOLUTION BASE RATES")
    end = with_block.index("\n\n", start)
    block = with_block[start:end]

    budget = int(make_rulebook().get("base_rates.prompt.max_chars"))
    assert len(block) <= budget
    assert len(block) / 4 < 300
