# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""scripts/ci/check_gdacs_prompt_blocks.py against a small DuckDB fixture.

Three prompts: one the current renderer would write, one whose timeline
repeats months (the pre-October shape), and one that prints a rate over a
four-month window. The checker must pass the first and name the faults in
the other two.
"""

from __future__ import annotations

import json
from datetime import datetime

import duckdb

from scripts.ci import check_gdacs_prompt_blocks as chk

RUN = "fc_1794543371"  # 2026-11-13 04:16 UTC: the last complete month is 2026-10


def _good_block() -> str:
    lines = [
        "HISTORICAL BASE RATE (GDACS, 2024-01 to 2026-10):",
        "Nicaragua has had significant (Orange/Red) drought alerts in 2 of the 34 calendar "
        "months GDACS covers (5.9%). A month with no alert naming Nicaragua counts as a month.",
        "",
        "CURRENT SITUATION",
        "",
        "Recent significant GDACS event months for Nicaragua (Orange/Red alerts):",
        "  - [Orange] 2026-09",
        "  - [Orange] 2025-06",
        "",
        "GDACS EVENT HISTORY (NIC — drought):",
        "Source: GDACS (Global Disaster Alert and Coordination System)",
        "Coverage: 2024-01 to 2026-10 (34 calendar months; a month with no alert counts as no event)",
        "Overall event rate: 2 events in 34 months (6%)",
        "",
        "Seasonal event frequency (your forecast months):",
        "  Dec: 0/2 years had events (0%)",
        "",
        "Recent event timeline:",
    ]
    months = [f"2025-{m:02d}" for m in range(11, 13)] + [f"2026-{m:02d}" for m in range(1, 11)]
    cells = [f"{ym}: {'Orange' if ym == '2026-09' else '--'}" for ym in months]
    for i in range(0, 12, 3):
        lines.append("  " + " | ".join(cells[i:i + 3]))
    lines += ["", "INTERPRETATION: GDACS events indicate confirmed disaster activity."]
    return "\n".join(lines)


REPEATED = """\
GDACS EVENT HISTORY (HND — drought):
Source: GDACS (Global Disaster Alert and Coordination System)
Coverage: 2024-01 to 2026-10 (34 months)
Overall event rate: 0 events in 34 months (0%)

Recent event timeline:
  2026-08: -- | 2026-08: -- | 2026-09: --
  2026-09: -- | 2026-10: -- | 2026-10: --

INTERPRETATION: x
"""

SHORT = """\
GDACS EVENT HISTORY (NER — drought):
Source: GDACS (Global Disaster Alert and Coordination System)
Coverage: 2026-07 to 2026-10 (4 months)
Overall event rate: 0 events in 4 months (0%)

Recent event timeline:
  2026-07: -- | 2026-08: -- | 2026-09: --
  2026-10: --

INTERPRETATION: x
"""


def _db(tmp_path):
    path = tmp_path / "c.duckdb"
    con = duckdb.connect(str(path))
    con.execute(
        "CREATE TABLE facts_resolved (ym VARCHAR, iso3 VARCHAR, hazard_code VARCHAR, "
        "metric VARCHAR, value DOUBLE, alertlevel VARCHAR, created_at TIMESTAMP)"
    )
    ins = "INSERT INTO facts_resolved VALUES (?, ?, 'DR', 'event_occurrence', ?, ?, ?)"
    old = datetime(2026, 7, 9)
    con.execute(ins, ["2024-01", "XXX", 0, "Green", old])
    con.execute(ins, ["2026-10", "XXX", 0, "Green", old])
    for ym in ("2025-06", "2026-09"):
        con.execute(ins, [ym, "NIC", 1, "Orange", old])
        con.execute(ins, [ym, "NIC", 1, "Orange", old])
    # Niger had events the short prompt hid, and one more arrived later.
    con.execute(ins, ["2024-08", "NER", 1, "Orange", old])
    con.execute(ins, ["2025-08", "NER", 1, "Orange", datetime(2026, 11, 15)])
    con.execute(
        "CREATE TABLE llm_calls (run_id VARCHAR, question_id VARCHAR, phase VARCHAR, "
        "hazard_code VARCHAR, iso3 VARCHAR, prompt_text VARCHAR, timestamp TIMESTAMP, "
        "is_test BOOLEAN)"
    )
    call = "INSERT INTO llm_calls VALUES (?, ?, ?, 'DR', ?, ?, TIMESTAMP '2026-11-14', FALSE)"
    con.execute(call, [RUN, "NIC_DR_EVENT_OCCURRENCE_2026-12", "binary_v2", "NIC", _good_block()])
    con.execute(call, [RUN, "NIC_DR_EVENT_OCCURRENCE_2026-12", "binary_v2", "NIC", _good_block()])
    con.execute(call, [RUN, "HND_DR_EVENT_OCCURRENCE_2026-12", "binary_v2", "HND", REPEATED])
    con.execute(call, [RUN, "NER_DR_PHASE3PLUS_IN_NEED_2026-12", "spd_v2", "NER", SHORT])
    con.close()
    return path


def test_the_good_prompt_passes_and_the_bad_ones_are_named(tmp_path):
    con = duckdb.connect(str(_db(tmp_path)), read_only=True)
    report = chk.check_run(con)
    assert report["run_id"] == RUN
    assert report["forecast_date"] == "2026-11-13" and report["cutoff_month"] == "2026-10"
    assert report["n_prompts"] == 3 and report["n_with_block"] == 3
    by_q: dict[str, set[str]] = {}
    for v in report["violations"]:
        by_q.setdefault(v["question_id"], set()).add(v["kind"])
    assert "NIC_DR_EVENT_OCCURRENCE_2026-12" not in by_q
    assert by_q["HND_DR_EVENT_OCCURRENCE_2026-12"] == {"repeated_month"}
    ner = by_q["NER_DR_PHASE3PLUS_IN_NEED_2026-12"]
    assert {"rate_under_min_months", "event_months_mismatch"} <= ner
    assert report["n_violating_prompts"] == 2
    # The 2025-08 Niger event arrived after the forecast: counted apart.
    assert report["by_hazard"]["DR"]["added_after_forecast"] == 1
    mismatch = [v for v in report["violations"] if v["kind"] == "event_months_mismatch"][0]
    assert "held 1 in" in mismatch["detail"]


def test_a_repeated_recent_alert_month_is_named(tmp_path):
    path = _db(tmp_path)
    con = duckdb.connect(str(path))
    bad = _good_block().replace("  - [Orange] 2025-06", "  - [Orange] 2026-09")
    con.execute("UPDATE llm_calls SET prompt_text = ? WHERE iso3 = 'NIC'", [bad])
    con.close()
    con = duckdb.connect(str(path), read_only=True)
    report = chk.check_run(con, RUN)
    kinds = {v["kind"] for v in report["violations"] if v["question_id"].startswith("NIC")}
    assert kinds == {"recent_alerts_repeated_month"}


def test_main_writes_the_report_and_an_issue_and_exits_zero(tmp_path):
    path = _db(tmp_path)
    out = tmp_path / "out"
    assert chk.main(["--db", str(path), "--out-dir", str(out)]) == 0
    report = json.loads((out / "gdacs_prompt_check.json").read_text())
    assert report["status"] == "violations"
    issues = json.loads((out / "issues.json").read_text())["issues"]
    assert issues[0]["id"] == "gdacs_prompt_block_invalid"
    assert issues[0]["severity"] == "degraded" and issues[0]["cost"] == 2
    md = (out / "gdacs_prompt_check.md").read_text()
    assert "HND_DR_EVENT_OCCURRENCE_2026-12" in md and "repeated_month" in md


def test_a_missing_database_still_exits_zero(tmp_path):
    out = tmp_path / "out"
    assert chk.main(["--db", str(tmp_path / "nope.duckdb"), "--out-dir", str(out)]) == 0
    assert json.loads((out / "issues.json").read_text())["issues"] == []
