# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The GDACS history block read back out of a stored prompt.

The two legacy fixtures are copied from prompts the 16 September 2026 run
(fc_1789534892) actually sent: Nicaragua's timeline repeated 2026-08 five
times and 2026-09 seven, and Niger's showed "0 events in 4 months" with no
historical data for any forecast month.
"""

from __future__ import annotations

import duckdb
import pytest

from forecaster.gdacs_history import (
    MIN_HISTORY_MONTHS,
    gdacs_calendar_series,
    parse_base_rate_section,
    parse_gdacs_prompt_block,
    parse_recent_alert_months,
    recent_alert_months,
)

NIC_LEGACY = """\
HISTORICAL BASE RATE (GDACS, 2018-09 to 2026-09):
Nicaragua has had significant drought events in 78 of 127 months (61.4%).

CURRENT SITUATION

Recent significant GDACS event months for Nicaragua (Orange/Red alerts):
  - [Orange] 2026-09
  - [Orange] 2026-09
  - [Orange] 2026-09

GDACS EVENT HISTORY (NIC — drought):
Source: GDACS (Global Disaster Alert and Coordination System)
Coverage: 2018-09 to 2026-09 (127 months)
Overall event rate: 78 events in 127 months (61%)
Alert levels (event months only): Orange: 78

Seasonal event frequency (your forecast months):
  Oct: 2/2 years had events (100%)
  Nov: 2/2 years had events (100%)
  Feb: 2/12 years had events (17%)

Recent event timeline:
  2026-08: Orange | 2026-08: -- | 2026-08: Orange
  2026-08: Orange | 2026-08: Orange | 2026-09: Orange
  2026-09: Orange | 2026-09: Orange | 2026-09: Orange
  2026-09: Orange | 2026-09: Orange | 2026-09: Orange

INTERPRETATION: GDACS events indicate confirmed disaster activity.
"""

NER_LEGACY = """\
GDACS EVENT HISTORY (NER — drought):
Source: GDACS (Global Disaster Alert and Coordination System)
Coverage: 2026-06 to 2026-09 (4 months)
Overall event rate: 0 events in 4 months (0%)

Seasonal event frequency (your forecast months):
  Oct: no historical data
  Nov: no historical data

Recent event timeline:
  2026-06: -- | 2026-07: -- | 2026-08: --
  2026-09: --

INTERPRETATION: GDACS events indicate confirmed disaster activity.
END OF QUESTION DATA.
"""

UNAVAILABLE = """\
GDACS EVENT HISTORY (ERI — drought):
Source: GDACS (Global Disaster Alert and Coordination System)
History unavailable for ERI / drought: the database holds GDACS DR alerts for only 4 month(s) (2026-06 to 2026-09); fewer than 24. Do NOT read this as a 0% base rate; reason from seasonality, the climate outlook and current conditions instead.

REGIME CHANGE FLAG: ...
"""


def test_parses_the_legacy_nicaragua_block_and_finds_its_repeats():
    b = parse_gdacs_prompt_block(NIC_LEGACY)
    assert b["iso3"] == "NIC" and b["hazard_label"] == "drought"
    assert b["format"] == "legacy"
    assert (b["coverage_start"], b["coverage_end"], b["total_months"]) == ("2018-09", "2026-09", 127)
    assert b["event_months"] == 78 and b["event_rate_pct"] == 61.0
    assert b["seasonal"][10] == {"years_observed": 2, "years_with_event": 2}
    assert b["seasonal"][2] == {"years_observed": 12, "years_with_event": 2}
    assert len(b["timeline"]) == 12
    assert b["repeated_months"] == ["2026-08", "2026-09"]
    assert ("2026-08", None) in b["timeline"]
    assert b["timeline_gaps"] == []
    assert b["history_unavailable"] is False


def test_parses_the_legacy_niger_block_with_no_seasonal_data():
    b = parse_gdacs_prompt_block(NER_LEGACY)
    assert b["total_months"] == 4 and b["event_months"] == 0
    assert b["seasonal"][10] == {"years_observed": 0, "years_with_event": 0}
    assert [ym for ym, _ in b["timeline"]] == ["2026-06", "2026-07", "2026-08", "2026-09"]
    assert b["repeated_months"] == [] and b["timeline_gaps"] == []


def test_parses_the_unavailable_line():
    b = parse_gdacs_prompt_block(UNAVAILABLE)
    assert b["history_unavailable"] is True and b["format"] == "unavailable"
    assert b["total_months"] is None and b["timeline"] == []
    assert "fewer than 24" in b["unavailable_reason"]


def test_no_block_is_none():
    assert parse_gdacs_prompt_block(None) is None
    assert parse_gdacs_prompt_block("A prompt with no GDACS history at all.") is None


def test_timeline_gaps_are_named():
    text = NER_LEGACY.replace("2026-07: -- | ", "")
    b = parse_gdacs_prompt_block(text)
    assert b["timeline_gaps"] == ["2026-07"]


def _history_db(tmp_path):
    con = duckdb.connect(str(tmp_path / "g.duckdb"))
    con.execute(
        "CREATE TABLE facts_resolved (ym VARCHAR, iso3 VARCHAR, hazard_code VARCHAR, "
        "metric VARCHAR, value DOUBLE, alertlevel VARCHAR, created_at TIMESTAMP)"
    )
    rows = []
    # Another country gives the source a 2020-01 .. 2026-10 window.
    rows.append(("2020-01", "XXX", "DR", "event_occurrence", 0, "Green"))
    rows.append(("2026-10", "XXX", "DR", "event_occurrence", 0, "Green"))
    for ym in ("2025-03", "2026-08"):
        for _ in range(3):  # several rows per month, as GDACS writes them
            rows.append((ym, "NIC", "DR", "event_occurrence", 1, "Orange"))
    rows.append(("2026-09", "NIC", "DR", "event_occurrence", 1, "Red"))
    rows.append(("2026-09", "NIC", "DR", "event_occurrence", 1, "Orange"))
    rows.append(("2026-11", "NIC", "DR", "event_occurrence", 1, "Orange"))
    rows.append(("2026-11", "NIC", "DR", "event_occurrence", 1, "Orange"))
    for r in rows:
        con.execute(
            "INSERT INTO facts_resolved VALUES (?, ?, ?, ?, ?, ?, TIMESTAMP '2026-07-09')", list(r)
        )
    return con


def test_the_current_renderer_and_the_parser_agree(tmp_path):
    from datetime import date

    from forecaster.gdacs_history import seasonal_frequency
    from forecaster.prompts import _format_gdacs_event_history_for_prompt

    con = _history_db(tmp_path)
    series = gdacs_calendar_series(con, "NIC", "DR", today=date(2026, 11, 13))
    hist = {
        "hazard_code": "DR", "iso3": "NIC",
        "data_range": f"{series['window_start']} to {series['window_end']}",
        "total_months": series["total_months"], "event_months": series["event_months"],
        "event_rate_pct": round(series["event_months"] / series["total_months"] * 100, 1),
        "alert_distribution": {}, "seasonal": seasonal_frequency(series["months"]),
        "recent_12": series["months"][-12:], "history_available": series["history_available"],
        "unavailable_reason": series["unavailable_reason"],
    }
    block = _format_gdacs_event_history_for_prompt(hist, [12, 1, 2, 3, 4, 5])
    b = parse_gdacs_prompt_block("intro\n" + block + "\nEND OF QUESTION DATA.")
    assert b["format"] == "current"
    assert b["total_months"] == series["total_months"] >= MIN_HISTORY_MONTHS
    assert b["event_months"] == series["event_months"] == 3
    assert b["repeated_months"] == [] and b["timeline_gaps"] == []
    assert b["timeline"][-1] == ("2026-10", None)
    assert b["timeline"][-3] == ("2026-08", "Orange")
    assert set(b["seasonal"]) == {12, 1, 2, 3, 4, 5}


def test_rows_written_after_a_date_can_be_left_out(tmp_path):
    from datetime import date, datetime

    con = _history_db(tmp_path)
    con.execute(
        "INSERT INTO facts_resolved VALUES ('2024-05', 'NIC', 'DR', 'event_occurrence', 1, "
        "'Orange', TIMESTAMP '2026-11-15')"
    )
    full = gdacs_calendar_series(con, "NIC", "DR", today=date(2026, 11, 13))
    asof = gdacs_calendar_series(
        con, "NIC", "DR", today=date(2026, 11, 13), written_before=datetime(2026, 11, 13)
    )
    assert full["event_months"] == asof["event_months"] + 1


def test_recent_alert_months_lists_each_month_once(tmp_path):
    from datetime import date

    con = _history_db(tmp_path)
    got = recent_alert_months(con, "NIC", "DR", today=date(2026, 11, 13))
    assert [g["ym"] for g in got] == ["2026-11", "2026-09", "2026-08", "2025-03"]
    assert got[0]["in_progress"] is True and not got[1]["in_progress"]
    assert got[1]["alertlevel"] == "Red"  # the strongest level of the month


def test_the_binary_prompt_lists_recent_alert_months_once(tmp_path):
    from datetime import date

    from forecaster.binary_prompts import _section_current_situation

    con = _history_db(tmp_path)
    alerts = recent_alert_months(con, "NIC", "DR", today=date(2026, 11, 13))
    text = _section_current_situation(alerts, {}, None, "Nicaragua", "DR")
    months = parse_recent_alert_months(text)
    assert months == ["2026-11", "2026-09", "2026-08", "2025-03"]
    assert "2026-11 (month in progress" in text


def test_parses_the_legacy_recent_alert_list_with_its_repeats():
    assert parse_recent_alert_months(NIC_LEGACY) == ["2026-09", "2026-09", "2026-09"]


@pytest.mark.parametrize(
    "text, expected",
    [
        (NIC_LEGACY, ("2018-09", "2026-09", 78, 127, False)),
        (
            "HISTORICAL BASE RATE (GDACS, 2017-07 to 2026-10):\nNicaragua has had significant "
            "(Orange/Red) drought alerts in 26 of the 112 calendar months GDACS covers (23.2%). A month",
            ("2017-07", "2026-10", 26, 112, False),
        ),
        (
            "HISTORICAL BASE RATE (GDACS): History unavailable for Eritrea / drought: x. Do NOT",
            (None, None, None, None, True),
        ),
    ],
    ids=["legacy", "current", "unavailable"],
)
def test_parses_the_base_rate_section(text, expected):
    b = parse_base_rate_section(text)
    assert (
        b["coverage_start"], b["coverage_end"], b["event_months"], b["total_months"],
        b["history_unavailable"],
    ) == expected
