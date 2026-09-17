# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The two ACE injects the 2026-09-15 cycle proved were never sent.

Measured on the prompts as stored: all 120 ACE regime-change prompts said
"ACLED summary unavailable ... do not assume either" while RESOLVER
FEATURES carried the ACLED numbers a few lines below; and none of the 234
ACE SPD prompts mentioned ACLED CAST at all, while the closing paragraph
went on describing three forecasts to a model that had been handed two.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from resolver.db._duckdb_available import DUCKDB_AVAILABLE

pytestmark = pytest.mark.skipif(not DUCKDB_AVAILABLE, reason="duckdb not installed")


# ---------------------------------------------------------------------------
# The ACLED summary reaches the prompt
# ---------------------------------------------------------------------------

def _acled_db(tmp_path, monkeypatch):
    import duckdb

    from horizon_scanner import horizon_scanner as hs

    path = tmp_path / "acled.duckdb"
    con = duckdb.connect(str(path))
    con.execute(
        "CREATE TABLE acled_monthly_fatalities "
        "(iso3 VARCHAR, month DATE, fatalities BIGINT, source VARCHAR, "
        " updated_at TIMESTAMP)"
    )
    con.execute(
        "CREATE TABLE acled_political_events "
        "(iso3 VARCHAR, event_id VARCHAR, event_date VARCHAR, "
        " event_type VARCHAR)"
    )
    today = date.today()
    first_of_month = date(today.year, today.month, 1)
    # Twelve complete months, newest first: 100 a month for the recent three
    # and 50 for the three before, so the trend is unambiguous.
    month = first_of_month
    for i in range(1, 13):
        month = (month - timedelta(days=1)).replace(day=1)
        con.execute(
            "INSERT INTO acled_monthly_fatalities VALUES (?, ?, ?, 'ACLED', NULL)",
            ["SDN", month, 100 if i <= 3 else 50],
        )
    # And a partial CURRENT month, which must not be counted.
    con.execute(
        "INSERT INTO acled_monthly_fatalities VALUES (?, ?, ?, 'ACLED', NULL)",
        ["SDN", first_of_month, 7],
    )
    for i in range(5):
        con.execute(
            "INSERT INTO acled_political_events VALUES (?, ?, ?, ?)",
            ["SDN", f"e{i}", (first_of_month - timedelta(days=30)).isoformat(),
             "Battles"],
        )
    con.close()

    monkeypatch.setattr(hs, "pythia_connect", lambda read_only=True: duckdb.connect(
        str(path), read_only=True
    ))
    return hs


def test_the_summary_is_built_from_complete_months_only(tmp_path, monkeypatch):
    hs = _acled_db(tmp_path, monkeypatch)
    summary = hs._build_acled_summary_for_country("SDN")
    assert summary is not None
    # 3 x 100 + 9 x 50 = 750. The partial current month's 7 is excluded: a
    # partial month read as a complete one manufactures a de-escalating
    # trend every time.
    assert summary["fatalities_trailing_12m"] == 750
    assert summary["fatalities_trailing_3m"] == 300
    assert summary["trend_direction"] == "escalating"
    assert summary["trend_pct_change"] == pytest.approx(100.0)


def test_events_and_top_types_come_from_the_political_table(tmp_path, monkeypatch):
    hs = _acled_db(tmp_path, monkeypatch)
    summary = hs._build_acled_summary_for_country("SDN")
    assert summary["events_trailing_12m"] == 5
    assert summary["top_event_types"][0][0] == "Battles"


def test_a_country_acled_does_not_cover_still_reads_unavailable(tmp_path, monkeypatch):
    # The honest "unavailable" text must still fire where there is nothing.
    hs = _acled_db(tmp_path, monkeypatch)
    assert hs._build_acled_summary_for_country("ISL") is None


def test_the_ace_prompts_stop_saying_unavailable_once_it_is_passed():
    from horizon_scanner.hs_triage_prompts import build_triage_prompt
    from horizon_scanner.rc_prompts import build_rc_prompt

    summary = {
        "fatalities_trailing_12m": 750,
        "fatalities_trailing_3m": 300,
        "trend_direction": "escalating",
        "trend_pct_change": 100.0,
    }
    for build in (build_rc_prompt, build_triage_prompt):
        prompt = build(
            "ACE", country_name="Sudan", iso3="SDN",
            resolver_features={"iso3": "SDN"}, evidence_pack=None,
            acled_summary=summary,
        )
        assert "ACLED summary unavailable" not in prompt
        assert "750" in prompt


# ---------------------------------------------------------------------------
# ACLED CAST's absence is stated on the SPD path too
# ---------------------------------------------------------------------------

_REASON = (
    "ACLED CAST is UNAVAILABLE: the newest vintage this system holds was "
    "issued 2025-12-10 and the last month it forecasts is 2026-05, which is "
    "now in the past."
)


def test_the_spd_formatter_states_cast_absence():
    from horizon_scanner.conflict_forecasts import (
        format_conflict_forecasts_for_research,
    )

    out = format_conflict_forecasts_for_research({
        "views_fatalities": [{"lead_months": 1, "value": 12.0}],
        "cast_unavailable_reason": _REASON,
    })
    assert "ACLED CAST" in out
    assert "NO DATA" in out
    assert "2025-12-10" in out


def test_a_live_cast_gets_its_table_and_no_absence_note():
    from horizon_scanner.conflict_forecasts import (
        format_conflict_forecasts_for_research,
    )

    out = format_conflict_forecasts_for_research({
        "cast_total": [{"lead_months": 1, "value": 30.0}],
        "cast_unavailable_reason": _REASON,
    })
    assert "NO DATA" not in out
    assert "event count forecasts" in out


def test_both_formatters_agree_about_an_absent_cast():
    # The RC path has stated it since the vintage work; the SPD path did
    # not, and one prompt family telling the model a source is absent while
    # the other says nothing is the fault.
    from horizon_scanner.conflict_forecasts import (
        format_conflict_forecasts_for_prompt,
        format_conflict_forecasts_for_research,
    )

    forecasts = {
        "views_fatalities": [{"lead_months": 1, "value": 12.0}],
        "cast_unavailable_reason": _REASON,
    }
    for fn in (format_conflict_forecasts_for_prompt,
               format_conflict_forecasts_for_research):
        assert "NO DATA" in fn(forecasts)
