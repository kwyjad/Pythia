# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Check the GDACS history blocks the forecaster actually SENT.

The November 2026 performance review read 84 stored drought prompts from
July to September 2026 and found the "GDACS EVENT HISTORY" block repeating
months in its timeline (Nicaragua: three quiet months read as twelve) and
printing "0 events" over a four-month window where the database holds
years of alerts (Niger). ``forecaster/gdacs_history.py`` fixed the renderer
on 29 September 2026; this script is how a run proves the fix reached the
prompts. The scored bundle's ``inject_status`` REBUILDS the history from
today's database, so it cannot see what a model was shown; this reads
``llm_calls.prompt_text`` instead, the source the debug bundle uses.

For every DR, FL and TC question of the run whose prompt carries a block:

1. ``repeated_month`` — a month listed more than once in the timeline.
2. ``timeline_gap`` / ``timeline_end`` — the timeline is not consecutive,
   or does not end at the last complete month before the forecast date.
3. ``rate_under_min_months`` — a rate printed over fewer than
   ``MIN_HISTORY_MONTHS`` months instead of the unavailable line.
4. ``event_months_mismatch`` — the block's event months differ from
   ``gdacs_calendar_series`` over the rows the database held on the forecast
   date. Rows written after it are counted apart (``added_after_forecast``)
   and are not a violation.
5. ``base_rate_mismatch`` (binary prompts) — the base-rate section and the
   history block describe different windows or rates.
6. ``recent_alerts_repeated_month`` (binary prompts) — the "Recent
   significant GDACS event months" list names a month more than once.

Writes ``gdacs_prompt_check.json`` and ``gdacs_prompt_check.md`` and always
exits 0: it is a diagnostic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from forecaster.gdacs_history import (  # noqa: E402
    MIN_HISTORY_MONTHS,
    gdacs_calendar_series,
    last_complete_month,
    parse_base_rate_section,
    parse_gdacs_prompt_block,
    parse_recent_alert_months,
)

HAZARDS = ("DR", "FL", "TC")
PHASES = ("spd_v2", "binary_v2")
ISSUE_ID = "gdacs_prompt_block_invalid"


def _connect(db: str):
    import duckdb

    path = db[len("duckdb:///"):] if db.startswith("duckdb:///") else db
    return duckdb.connect(path, read_only=True)


def _has_column(con, table: str, column: str) -> bool:
    try:
        rows = con.execute(f"PRAGMA table_info('{table}')").fetchall()
    except Exception:  # noqa: BLE001
        return False
    return any(str(r[1]).lower() == column.lower() for r in rows)


def latest_production_run(con) -> Optional[str]:
    """The newest forecaster run with SPD or binary calls and no test rows."""

    test_sql = ""
    if _has_column(con, "llm_calls", "is_test"):
        test_sql = "HAVING NOT bool_or(COALESCE(is_test, FALSE))"
    try:
        row = con.execute(
            f"""
            SELECT run_id FROM llm_calls
            WHERE phase IN ('spd_v2', 'binary_v2') AND run_id LIKE 'fc_%'
            GROUP BY run_id {test_sql}
            ORDER BY MIN(timestamp) DESC LIMIT 1
            """
        ).fetchone()
    except Exception:  # noqa: BLE001
        return None
    return row[0] if row else None


def forecast_timestamp(con, run_id: str) -> Optional[datetime]:
    """When the run built its prompts.

    A forecaster run id is ``fc_<epoch seconds>`` minted at submit, which is
    when the staged pipeline builds its prompts; the calls are logged at
    collect, possibly a day later. Falls back to the earliest call.
    """

    tail = run_id[3:] if run_id.startswith("fc_") else ""
    if tail.isdigit():
        return datetime.fromtimestamp(int(tail), tz=timezone.utc).replace(tzinfo=None)
    try:
        row = con.execute(
            "SELECT MIN(timestamp) FROM llm_calls WHERE run_id = ?", [run_id]
        ).fetchone()
    except Exception:  # noqa: BLE001
        return None
    return row[0] if row and row[0] else None


def load_prompts(con, run_id: str) -> list[dict[str, Any]]:
    """One prompt per (question, phase): the first logged. Members share it."""

    try:
        rows = con.execute(
            """
            SELECT question_id, phase, upper(hazard_code), iso3, prompt_text
            FROM llm_calls
            WHERE run_id = ? AND phase IN ('spd_v2', 'binary_v2')
              AND upper(COALESCE(hazard_code, '')) IN ('DR', 'FL', 'TC')
              AND prompt_text IS NOT NULL
            ORDER BY question_id, phase, timestamp
            """,
            [run_id],
        ).fetchall()
    except Exception:  # noqa: BLE001
        return []
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, Any]] = []
    for qid, phase, hz, iso3, text in rows:
        key = (qid, phase)
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "question_id": qid, "phase": phase, "hazard_code": hz,
            "iso3": (iso3 or str(qid or "")[:3]).upper(), "prompt_text": text,
        })
    return out


def _block_lines(text: str) -> list[str]:
    """The block's own lines, quoted in a violation."""

    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("GDACS EVENT HISTORY"):
            block = []
            for nxt in lines[i:]:
                if nxt.startswith("INTERPRETATION"):
                    break
                block.append(nxt)
            return block[:30]
    return []


def check_prompt(
    con,
    prompt: dict[str, Any],
    *,
    forecast_date: date,
    forecast_ts: Optional[datetime],
    has_created_at: bool,
    series_cache: dict,
) -> dict[str, Any]:
    """Check one stored prompt; returns a result with its violations."""

    text = prompt["prompt_text"] or ""
    block = parse_gdacs_prompt_block(text)
    res: dict[str, Any] = {
        "question_id": prompt["question_id"],
        "phase": prompt["phase"],
        "hazard_code": prompt["hazard_code"],
        "iso3": prompt["iso3"],
        "has_block": block is not None,
        "violations": [],
        "added_after_forecast": 0,
    }
    if block is None:
        return res
    res["block"] = {k: v for k, v in block.items() if k != "seasonal"}
    cutoff = last_complete_month(forecast_date)
    lines = _block_lines(text)

    def violate(kind: str, detail: str) -> None:
        res["violations"].append({"kind": kind, "detail": detail, "lines": lines})

    if block["repeated_months"]:
        violate("repeated_month", "timeline repeats " + ", ".join(block["repeated_months"]))
    if block["timeline"]:
        if block["timeline_gaps"]:
            violate("timeline_gap", "timeline skips " + ", ".join(block["timeline_gaps"]))
        last = max(ym for ym, _ in block["timeline"])
        if last != cutoff:
            violate(
                "timeline_end",
                f"timeline ends {last}; the last complete month before "
                f"{forecast_date.isoformat()} is {cutoff}",
            )
    if not block["history_unavailable"]:
        total = block.get("total_months")
        if total is None or total < MIN_HISTORY_MONTHS:
            violate(
                "rate_under_min_months",
                f"a rate printed over {total} months; below {MIN_HISTORY_MONTHS} the "
                "block must print the unavailable line",
            )

    # 4: the database as it stood on the forecast date.
    key = (prompt["iso3"], prompt["hazard_code"])
    if key not in series_cache:
        asof = gdacs_calendar_series(
            con, prompt["iso3"], prompt["hazard_code"], today=forecast_date,
            written_before=forecast_ts if has_created_at else None,
        )
        now = gdacs_calendar_series(con, prompt["iso3"], prompt["hazard_code"], today=forecast_date)
        series_cache[key] = (asof, now)
    asof, now = series_cache[key]
    res["db_event_months_asof"] = asof["event_months"]
    res["db_total_months_asof"] = asof["total_months"]
    res["added_after_forecast"] = max(0, int(now["event_months"]) - int(asof["event_months"]))
    if block["history_unavailable"]:
        if asof["history_available"]:
            violate(
                "event_months_mismatch",
                f"block says history unavailable; the database held "
                f"{asof['total_months']} months ({asof['event_months']} with events) on the forecast date",
            )
    elif block.get("event_months") is not None and block["event_months"] != asof["event_months"]:
        violate(
            "event_months_mismatch",
            f"block shows {block['event_months']} event months in {block.get('total_months')}; "
            f"the database held {asof['event_months']} in {asof['total_months']} on the forecast date",
        )

    if prompt["phase"] == "binary_v2":
        base = parse_base_rate_section(text)
        if base is not None:
            same = (
                base["history_unavailable"] == block["history_unavailable"]
                and (
                    block["history_unavailable"]
                    or (
                        base["coverage_start"] == block["coverage_start"]
                        and base["coverage_end"] == block["coverage_end"]
                        and base["event_months"] == block["event_months"]
                        and base["total_months"] == block["total_months"]
                    )
                )
            )
            if not same:
                violate(
                    "base_rate_mismatch",
                    f"base-rate section {base['coverage_start']}..{base['coverage_end']} "
                    f"{base['event_months']}/{base['total_months']} vs history block "
                    f"{block['coverage_start']}..{block['coverage_end']} "
                    f"{block['event_months']}/{block['total_months']}",
                )
        recent = parse_recent_alert_months(text)
        dup = sorted({m for m in recent if recent.count(m) > 1})
        if dup:
            violate("recent_alerts_repeated_month", "recent alert list repeats " + ", ".join(dup))
    return res


def check_run(con, run_id: Optional[str] = None) -> dict[str, Any]:
    """Check every GDACS block one run sent. Never raises."""

    run_id = run_id or latest_production_run(con)
    report: dict[str, Any] = {
        "run_id": run_id,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "min_history_months": MIN_HISTORY_MONTHS,
        "by_hazard": {},
        "violations": [],
        "n_prompts": 0,
        "n_with_block": 0,
        "n_violating_prompts": 0,
        "added_after_forecast_prompts": 0,
        "status": "ok",
        "error": None,
    }
    if not run_id:
        report["status"] = "no_run"
        report["error"] = "no forecaster run with SPD or binary calls in llm_calls"
        return report
    ts = forecast_timestamp(con, run_id)
    report["forecast_timestamp"] = ts.isoformat() if ts else None
    forecast_date = ts.date() if ts else datetime.now(timezone.utc).date()
    report["forecast_date"] = forecast_date.isoformat()
    report["cutoff_month"] = last_complete_month(forecast_date)
    has_created_at = _has_column(con, "facts_resolved", "created_at")
    report["rebuild_basis"] = (
        "rows created before the forecast" if has_created_at
        else "every row (facts_resolved has no created_at)"
    )
    prompts = load_prompts(con, run_id)
    report["n_prompts"] = len(prompts)
    cache: dict = {}
    for p in prompts:
        try:
            res = check_prompt(
                con, p, forecast_date=forecast_date, forecast_ts=ts,
                has_created_at=has_created_at, series_cache=cache,
            )
        except Exception as exc:  # noqa: BLE001
            res = {"question_id": p["question_id"], "phase": p["phase"],
                   "hazard_code": p["hazard_code"], "has_block": None,
                   "violations": [], "error": f"{type(exc).__name__}: {exc}"}
        hz = report["by_hazard"].setdefault(p["hazard_code"], {
            "prompts": 0, "with_block": 0, "violating_prompts": 0,
            "unavailable_line": 0, "added_after_forecast": 0, "by_kind": {},
        })
        hz["prompts"] += 1
        if res.get("has_block"):
            hz["with_block"] += 1
            report["n_with_block"] += 1
            if (res.get("block") or {}).get("history_unavailable"):
                hz["unavailable_line"] += 1
        if res.get("added_after_forecast"):
            hz["added_after_forecast"] += 1
            report["added_after_forecast_prompts"] += 1
        if res["violations"]:
            hz["violating_prompts"] += 1
            report["n_violating_prompts"] += 1
            for v in res["violations"]:
                hz["by_kind"][v["kind"]] = hz["by_kind"].get(v["kind"], 0) + 1
                report["violations"].append({
                    "question_id": res["question_id"], "phase": res["phase"],
                    "hazard_code": res["hazard_code"], **v,
                })
    if report["n_violating_prompts"]:
        report["status"] = "violations"
    return report


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# GDACS prompt-block check — {report.get('run_id')}",
        "",
        f"Forecast date {report.get('forecast_date')}, last complete month "
        f"{report.get('cutoff_month')}. Database rebuilt from "
        f"{report.get('rebuild_basis')}.",
        "",
        f"**{report.get('n_violating_prompts', 0)} of {report.get('n_with_block', 0)} prompts "
        f"with a GDACS block break a rule** ({report.get('n_prompts', 0)} DR/FL/TC prompts read).",
        "",
    ]
    if report.get("error"):
        lines += [f"Not checked: {report['error']}", ""]
    lines += [
        "| Hazard | Prompts | With block | Unavailable line | Violating | Added after forecast | Violations by kind |",
        "|---|---|---|---|---|---|---|",
    ]
    for hz, row in sorted((report.get("by_hazard") or {}).items()):
        kinds = ", ".join(f"{k} {n}" for k, n in sorted(row["by_kind"].items())) or "none"
        lines.append(
            f"| {hz} | {row['prompts']} | {row['with_block']} | {row['unavailable_line']} | "
            f"{row['violating_prompts']} | {row['added_after_forecast']} | {kinds} |"
        )
    lines.append("")
    lines.append(
        "\"Added after forecast\" counts prompts whose country gained GDACS event "
        "months after the forecast date; that is the database moving on, not a fault."
    )
    viol = report.get("violations") or []
    if viol:
        lines += ["", "## Violations", ""]
        for v in viol[:200]:
            lines.append(f"### {v['question_id']} ({v['phase']}): {v['kind']}")
            lines.append("")
            lines.append(v["detail"])
            lines.append("")
            if v.get("lines"):
                lines.append("```")
                lines.extend(v["lines"])
                lines.append("```")
                lines.append("")
        if len(viol) > 200:
            lines.append(f"... and {len(viol) - 200} more in gdacs_prompt_check.json")
    return "\n".join(lines) + "\n"


def issue_record(report: dict[str, Any]) -> Optional[dict[str, Any]]:
    """The chain-diagnostics issue a violation raises, or None."""

    n = int(report.get("n_violating_prompts") or 0)
    if not n:
        return None
    return {
        "id": ISSUE_ID,
        "severity": "degraded",
        "title": (
            f"{n} of {report.get('n_with_block')} GDACS history blocks in run "
            f"{report.get('run_id')}'s forecast prompts break a rule"
        ),
        "evidence": (
            "a repeated month, a timeline that skips or ends on the wrong month, a rate "
            "over too few months, event months the database did not hold, or a base-rate "
            "section that disagrees; see gdacs_prompt_check.md"
        ),
        "cost": n,
        "cost_unit": "prompts",
        "owner": "pythia",
        "recovers_on_rerun": False,
        "source": "scripts/ci/check_gdacs_prompt_blocks.py",
    }


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", required=True)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--out-dir", default="diagnostics")
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    try:
        out.mkdir(parents=True, exist_ok=True)
        con = _connect(args.db)
        try:
            report = check_run(con, args.run_id)
        finally:
            con.close()
    except Exception as exc:  # noqa: BLE001
        report = {"run_id": args.run_id, "status": "error",
                  "error": f"{type(exc).__name__}: {exc}", "by_hazard": {},
                  "violations": [], "n_prompts": 0, "n_with_block": 0,
                  "n_violating_prompts": 0}
    issue = issue_record(report)
    report["issue"] = issue
    try:
        (out / "gdacs_prompt_check.json").write_text(json.dumps(report, indent=2, default=str))
        (out / "gdacs_prompt_check.md").write_text(render_markdown(report))
        # The cycle diagnostics zip reads every issues.json in a run's
        # artifacts into CYCLE_SUMMARY.md (build_cycle_diagnostics.folder_issues).
        (out / "issues.json").write_text(json.dumps(
            {"source": "gdacs_prompt_check", "issues": [issue] if issue else []},
            indent=2, default=str,
        ))
        print(render_markdown(report).split("## Violations")[0])
        if issue:
            print(f"::warning title=GDACS prompt block::{issue['title']}")
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] could not write the GDACS prompt check: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
