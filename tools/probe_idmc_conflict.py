# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Read-only probe: what does IDMC's conflict displacement feed hold?

ACE/PA questions resolve against IDMC conflict displacement summed per
country and start month. Three things about the feed decide whether that sum
can be read as a monthly count of people, and none can be checked from the
build sandbox (helix-tools-api.idmcdb.org is denied there):

1. How late a record arrives. Each record is compared with its own creation
   or update stamp, and the distribution of (stamp - displacement date) is
   reported, so the settle period is a measurement rather than a guess.
2. Whether a record is a figure IDMC recommends for totals or a
   triangulation / duplicate (the ``role`` field), and how many records span
   more than one month.
3. What the largest country-months are made of: PSE 2023-10, IRN 2025-06,
   IRN 2026-02 and LBN's largest month.

It writes ``report.json`` and ``report.md`` and always exits 0. The client id
travels in the query string, so every error text is scrubbed before it is
printed or written (CLAUDE.md, Security invariants).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

DEFAULT_URL = "https://helix-tools-api.idmcdb.org/external-api/idus/all/"
TIMESTAMP_FIELDS = (
    "created_at", "updated_at", "modified_at", "last_modified", "created",
    "modified", "publish_date", "published_at", "entry_date", "date_created",
)
COMPOSITION_TARGETS = (("PSE", "2023-10"), ("IRN", "2025-06"), ("IRN", "2026-02"))


def _scrub(text: str, secret: str) -> str:
    return text.replace(secret, "REDACTED") if secret else text


def _date(value: Any) -> dt.date | None:
    if not value:
        return None
    text = str(value).strip()[:10]
    try:
        return dt.date.fromisoformat(text)
    except ValueError:
        return None


def _figure(record: Mapping[str, Any]) -> float | None:
    for field in ("figure", "total_figures", "displacement_figure"):
        raw = record.get(field)
        if raw is None or isinstance(raw, bool):
            continue
        try:
            return float(str(raw).replace(",", ""))
        except ValueError:
            continue
    return None


def _start(record: Mapping[str, Any]) -> dt.date | None:
    for field in ("displacement_start_date", "event_start_date", "displacement_date", "event_date"):
        got = _date(record.get(field))
        if got:
            return got
    return None


def _end(record: Mapping[str, Any]) -> dt.date | None:
    for field in ("displacement_end_date", "event_end_date"):
        got = _date(record.get(field))
        if got:
            return got
    return None


def _quantiles(values: list[float], qs: Iterable[float]) -> dict[str, float | None]:
    if not values:
        return {f"p{int(q * 100)}": None for q in qs}
    ordered = sorted(values)
    out: dict[str, float | None] = {}
    for q in qs:
        pos = q * (len(ordered) - 1)
        lo = int(pos)
        hi = min(lo + 1, len(ordered) - 1)
        out[f"p{int(q * 100)}"] = round(ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo), 1)
    return out


SETTLE_DAYS_GRID = (15, 30, 45, 60, 75, 90, 120, 180)


def _month_end(day: dt.date) -> dt.date:
    return (day.replace(day=28) + dt.timedelta(days=4)).replace(day=1) - dt.timedelta(days=1)


def settle_curve(
    conflict: list[Mapping[str, Any]], *, first_ym: str = "2024-01", last_ym: str = "2026-03",
) -> dict[str, Any]:
    """How much of a month's EVENTUAL recommended total had arrived N days
    after the month ended.

    Only recommended records count (a triangulation never enters a total),
    and only months from ``first_ym`` to ``last_ym``: old enough that their
    total has settled, recent enough that their records were created as
    reports rather than in a historical backfill. Two curves: the share of
    people (pooled over the window) and the share of country-months whose
    FIRST recommended record had arrived.
    """

    people: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for r in conflict:
        role = str(r.get("role") or "").lower()
        if role and not role.startswith("recommended"):
            continue
        start, stamp, fig = _start(r), _date(r.get("created_at")), _figure(r)
        if not (start and stamp and fig is not None):
            continue
        ym = start.strftime("%Y-%m")
        if not first_ym <= ym <= last_ym:
            continue
        people[(str(r.get("iso3") or "").upper(), ym)].append(
            (float((stamp - _month_end(start)).days), fig)
        )
    total = sum(f for rows in people.values() for _d, f in rows)
    by_people: dict[str, float | None] = {}
    by_first: dict[str, float | None] = {}
    for days in SETTLE_DAYS_GRID:
        arrived = sum(f for rows in people.values() for d, f in rows if d <= days)
        by_people[f"{days}d"] = round(arrived / total, 3) if total else None
        first = sum(1 for rows in people.values() if min(d for d, _f in rows) <= days)
        by_first[f"{days}d"] = round(first / len(people), 3) if people else None
    return {
        "months": f"{first_ym}..{last_ym}",
        "country_months": len(people),
        "share_of_people_arrived": by_people,
        "share_of_country_months_with_a_first_report": by_first,
    }


def analyse(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Everything the report says, from the records alone (pure; tested)."""

    conflict = [
        r for r in records
        if isinstance(r, Mapping)
        and str(r.get("displacement_type") or "").strip().lower().startswith("conflict")
    ]
    keys: Counter = Counter()
    for r in conflict:
        keys.update(r.keys())
    roles = Counter(str(r.get("role") or "<absent>") for r in conflict)
    stamp_fields = [f for f in TIMESTAMP_FIELDS if keys.get(f)]

    lags: dict[str, list[float]] = defaultdict(list)
    for field in stamp_fields:
        for r in conflict:
            stamp, start = _date(r.get(field)), _start(r)
            if stamp and start:
                month_end = (start.replace(day=28) + dt.timedelta(days=4)).replace(day=1) - dt.timedelta(days=1)
                lags[field].append(float((stamp - month_end).days))
    lag_summary = {
        field: {
            "n": len(vals),
            "days_after_month_end": _quantiles(vals, (0.5, 0.75, 0.8, 0.9, 0.95, 0.99)),
            "share_within_30d": round(sum(v <= 30 for v in vals) / len(vals), 3) if vals else None,
            "share_within_60d": round(sum(v <= 60 for v in vals) / len(vals), 3) if vals else None,
            "share_within_90d": round(sum(v <= 90 for v in vals) / len(vals), 3) if vals else None,
        }
        for field, vals in lags.items()
    }

    spans: list[float] = []
    multi_month = 0
    over_31 = 0
    for r in conflict:
        start, end = _start(r), _end(r)
        if start and end and end >= start:
            spans.append(float((end - start).days))
            if (end.year, end.month) != (start.year, start.month):
                multi_month += 1
            if (end - start).days > 31:
                over_31 += 1

    def _composition(iso3: str, ym: str) -> dict[str, Any]:
        rows = [
            r for r in conflict
            if str(r.get("iso3") or "").upper() == iso3
            and (_start(r) or dt.date(1900, 1, 1)).strftime("%Y-%m") == ym
        ]
        by_role: Counter = Counter()
        for r in rows:
            by_role[str(r.get("role") or "<absent>")] += _figure(r) or 0.0
        biggest = sorted(rows, key=lambda r: -(_figure(r) or 0.0))[:8]
        spans = Counter(
            "over_31_days" if (_start(r) and _end(r) and (_end(r) - _start(r)).days > 31)
            else "within_31_days"
            for r in rows
        )
        return {
            "iso3": iso3,
            "ym": ym,
            "spans": dict(spans),
            "records": len(rows),
            "sum": sum(_figure(r) or 0.0 for r in rows),
            "people_by_role": dict(by_role),
            "largest_records": [
                {
                    "figure": _figure(r),
                    "role": r.get("role"),
                    "start": str(_start(r)),
                    "end": str(_end(r)),
                    "event_name": str(r.get("event_name") or "")[:120],
                    "qualifier": r.get("qualifier"),
                    **{f: r.get(f) for f in stamp_fields[:2]},
                }
                for r in biggest
            ],
        }

    lbn: dict[str, float] = defaultdict(float)
    for r in conflict:
        if str(r.get("iso3") or "").upper() == "LBN" and _start(r):
            lbn[_start(r).strftime("%Y-%m")] += _figure(r) or 0.0
    targets = list(COMPOSITION_TARGETS)
    if lbn:
        targets.append(("LBN", max(lbn, key=lbn.get)))

    settle = settle_curve(conflict)

    return {
        "settle_curve": settle,
        "records": len(records),
        "conflict_records": len(conflict),
        "conflict_keys": dict(keys.most_common()),
        "roles": dict(roles),
        "timestamp_fields": stamp_fields,
        "lag": lag_summary,
        "span_days": _quantiles(spans, (0.5, 0.9, 0.99)) | {"n": len(spans)},
        "multi_month_records": multi_month,
        "records_over_31_days": over_31,
        "composition": [_composition(i, m) for i, m in targets],
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = ["# IDMC conflict displacement probe", ""]
    lines.append(f"Records: {report['records']}; conflict: {report['conflict_records']}")
    lines.append(f"Roles: {report['roles']}")
    lines.append(f"Timestamp fields present: {report['timestamp_fields']}")
    lines.append(f"Multi-month records: {report['multi_month_records']}; over 31 days: {report['records_over_31_days']}")
    lines.append(f"Span days: {report['span_days']}")
    lines.append("")
    lines.append("## Settle curve (recommended figures)")
    lines.append(str(report.get("settle_curve")))
    lines.append("")
    lines.append("## Lag after month end, by stamp field")
    for field, info in report["lag"].items():
        lines.append(f"- {field}: {info}")
    lines.append("")
    lines.append("## Composition")
    for comp in report["composition"]:
        lines.append(f"### {comp['iso3']} {comp['ym']}: {comp['records']} records, sum {comp['sum']:,.0f}")
        lines.append(f"by role: {comp['people_by_role']}; spans: {comp.get('spans')}")
        for rec in comp["largest_records"]:
            lines.append(f"- {rec}")
    lines.append("")
    lines.append(f"Keys: {report['conflict_keys']}")
    rule = report.get("ingest_rule")
    if rule:
        lines.append("")
        lines.append("## The series under the Oct 2026 ingest rules")
        lines.append(json.dumps({k: v for k, v in rule.items() if k != "held_months"}, default=str))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", default="diagnostics/idmc_conflict_probe")
    args = parser.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    key = (os.getenv("IDMC_API_KEY") or os.getenv("IDMC_HELIX_CLIENT_ID") or "").strip()
    if not key:
        (out / "report.md").write_text("no IDMC_API_KEY or IDMC_HELIX_CLIENT_ID; nothing probed\n")
        print("::warning::no IDMC credential; probe skipped")
        return 0
    try:
        import requests

        resp = requests.get(
            os.getenv("IDMC_IDU_ALL_URL", "").strip() or DEFAULT_URL,
            params={"client_id": key},
            timeout=600,
            headers={"Accept": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict):
            records = data.get("results") or data.get("data") or []
        else:
            records = data
    except Exception as exc:  # noqa: BLE001 - a probe reports, never raises
        message = _scrub(f"{type(exc).__name__}: {exc}", key)
        (out / "report.md").write_text(f"fetch failed: {message}\n")
        print(f"::warning::IDMC probe fetch failed: {message}")
        return 0
    report = analyse(list(records or []))
    try:
        # The series the ingest would write under the Oct 2026 rules
        # (recommended figures only, long spans and above-population months
        # held out), so the resolution counts can be measured offline.
        from resolver.ingestion import idmc_conflict as ic

        first, last = ic.month_window(dt.date.today(), 36)
        flows, flow_report = ic.conflict_monthly_flows(
            records or [], first, last, population=ic.load_population(),
        )
        flows.to_csv(out / "flows.csv", index=False)
        report["ingest_rule"] = {
            k: flow_report.get(k)
            for k in ("rows", "rows_held", "countries", "conflict_roles",
                      "conflict_people_not_recommended", "conflict_records_dropped",
                      "over_population", "months_per_country")
        }
        report["ingest_rule"]["held_months"] = flow_report.get("held_months", [])[:80]
    except Exception as exc:  # noqa: BLE001
        report["ingest_rule"] = {"error": _scrub(f"{type(exc).__name__}: {exc}", key)}
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    text = _markdown(report)
    (out / "report.md").write_text(text)
    print(text)
    summary = os.getenv("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(text + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
