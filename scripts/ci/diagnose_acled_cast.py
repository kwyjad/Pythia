# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Why does ACLED CAST stop at December 2025? One call, three verdicts.

``conflict_forecasts`` has held a 2025-12-01 CAST vintage since the 2026-01
cycle. The transport is healthy — the connector authenticates and pulls
9,879 records — so this is not an outage. Something about what we ask for,
or what our account is allowed, stops at December.

Three explanations fit what we can see, and they call for three different
fixes. This decides between them from evidence rather than from reasoning:

**Account tier.** ACLED's own page states that free CAST access includes ten
logins or downloads. A spent quota is reported in the JSON envelope's
``messages`` and ``data_query_restrictions`` fields, which the connector
discards — it reads ``data`` and nothing else. Fix: contact
access@acleddata.com, the route ACLED publishes for access beyond the free
tier.

**Schema change.** The current CAST methodology page states the temporal
unit is a four-week rolling period ending on Fridays, that the model
forecasts the next six periods, and that new predictions publish weekly. The
API documentation, revised September 2025, still describes a ``month``
string and a ``year`` integer, and the platform's last monthly artefact is
"CAST Monthly Report: December 2025". If CAST moved from monthly vintages to
weekly rolling periods at the end of 2025, our connector — which derives its
issue date from ``timestamp`` but filters to leads 1..6 computed from
``month``/``year`` — would see exactly what it sees. Fix: re-key on the
rolling period and map each to the calendar month containing its end date.

**Upstream stop.** ACLED stopped publishing to this endpoint. Fix:
escalate, and fall back to a baseline built from the ACLED history we
already hold.

Read-only: one authenticated GET and nothing else. Never writes to the
database, and always exits 0 — "the endpoint is quiet" is a finding, not a
build failure.

    python -m scripts.ci.diagnose_acled_cast
    python -m scripts.ci.diagnose_acled_cast --limit 200 --out diagnostics/cast.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Any

API_URL = "https://acleddata.com/api/cast/read"
TIMEOUT = 90

#: The schema the September 2025 API documentation describes. A row carrying
#: none of these, or carrying period-shaped fields instead, is the schema
#: change rather than a quota problem.
_DOCUMENTED_FIELDS = (
    "country", "iso", "admin1", "month", "year", "timestamp",
    "total_forecast", "battles_forecast", "erv_forecast", "vac_forecast",
)

#: Field names a four-week rolling period would plausibly arrive under.
_PERIOD_FIELD_HINTS = (
    "period", "period_start", "period_end", "window", "window_start",
    "window_end", "forecast_period", "week", "week_ending", "end_date",
    "start_date", "date",
)

#: Envelope fields that carry ACLED's own account for the response. The
#: connector reads `data` and drops every one of these, which is why two
#: separate investigations could not tell a quota limit from a schema change.
_ENVELOPE_FIELDS = (
    "status", "success", "count", "last_update", "messages", "error",
    "data_query_restrictions", "filename",
)

#: Words in the envelope that name a recency or quota restriction.
_RESTRICTION_WORDS = (
    "limit", "quota", "download", "login", "subscription", "expired",
    "restrict", "access", "upgrade", "exceed",
)


def _fetch(limit: int) -> tuple[int, dict[str, Any], str]:
    """(status_code, parsed body or {}, raw text). Never raises."""

    import requests

    from resolver.ingestion.acled_auth import get_auth_header

    headers = get_auth_header()
    headers["Accept"] = "application/json"
    resp = requests.get(
        API_URL, params={"limit": limit, "page": 1}, headers=headers, timeout=TIMEOUT
    )
    text = resp.text or ""
    try:
        body = resp.json()
    except Exception:  # noqa: BLE001 - a non-JSON body is itself the finding
        body = {}
    return resp.status_code, (body if isinstance(body, dict) else {}), text


def _rows(body: dict[str, Any]) -> list[dict[str, Any]]:
    data = body.get("data")
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    return []


def _max_timestamp(rows: list[dict[str, Any]]) -> str | None:
    """The newest ``timestamp`` across rows, as an ISO date.

    This is the field the connector derives its issue date from, so if it is
    recent while ``month``/``year`` stop in December, the shape has changed
    and the vintage is stale by construction rather than upstream.
    """

    best: dt.datetime | None = None
    for row in rows:
        raw = row.get("timestamp")
        if raw is None:
            continue
        moment: dt.datetime | None = None
        try:
            if isinstance(raw, (int, float)):
                moment = dt.datetime.fromtimestamp(float(raw))
            elif isinstance(raw, str):
                try:
                    moment = dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
                except ValueError:
                    moment = dt.datetime.fromtimestamp(float(raw))
        except (ValueError, TypeError, OSError):
            continue
        if moment is not None and (best is None or moment > best):
            best = moment
    return best.date().isoformat() if best else None


def _restriction_note(body: dict[str, Any]) -> str | None:
    """A quota or recency limit stated by ACLED itself, or None."""

    blob = " ".join(
        json.dumps(body.get(field), default=str)
        for field in ("messages", "data_query_restrictions", "error", "status")
        if body.get(field) not in (None, "", [], {})
    ).lower()
    if not blob.strip():
        return None
    if any(word in blob for word in _RESTRICTION_WORDS):
        return blob[:600]
    return None


def diagnose(limit: int = 200) -> dict[str, Any]:
    """Everything one call can tell us, plus the verdict it supports."""

    report: dict[str, Any] = {
        "endpoint": API_URL,
        "checked_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "limit": limit,
    }

    try:
        status, body, text = _fetch(limit)
    except Exception as exc:  # noqa: BLE001 - a dead endpoint is a finding
        report["http_error"] = f"{type(exc).__name__}: {exc}"
        report["verdict"] = "unreachable"
        report["next_step"] = (
            "The endpoint could not be reached at all. Check the ACLED "
            "credentials first (ACLED_USERNAME/ACLED_PASSWORD or "
            "ACLED_REFRESH_TOKEN), then escalate to access@acleddata.com."
        )
        return report

    report["http_status"] = status
    # The envelope, which the connector discards. This is the half of the
    # response that says WHY, and dropping it is why the December 2025 stall
    # survived two investigations.
    report["envelope"] = {
        field: body.get(field) for field in _ENVELOPE_FIELDS if field in body
    }
    report["envelope_keys"] = sorted(body.keys())
    if not body:
        report["raw_body_head"] = text[:500]

    rows = _rows(body)
    report["rows_returned"] = len(rows)

    months = sorted(
        {
            f"{row.get('year')}-{str(row.get('month')).strip().lower()}"
            for row in rows
            if row.get("year") is not None and row.get("month") is not None
        }
    )
    report["distinct_year_month"] = months[:24]
    report["distinct_year_month_count"] = len(months)
    years = sorted({int(r["year"]) for r in rows if str(r.get("year", "")).isdigit()})
    report["years_present"] = years
    report["max_year"] = years[-1] if years else None
    report["max_timestamp_date"] = _max_timestamp(rows)

    observed = sorted({key for row in rows for key in row})
    report["columns_observed"] = observed
    report["columns_undocumented"] = [c for c in observed if c not in _DOCUMENTED_FIELDS]
    report["columns_documented_missing"] = [
        c for c in _DOCUMENTED_FIELDS if c not in observed
    ]
    period_fields = [
        c for c in observed
        if any(hint in c.lower() for hint in _PERIOD_FIELD_HINTS)
        and c not in ("month", "year", "timestamp")
    ]
    report["period_shaped_columns"] = period_fields
    if rows:
        report["sample_row"] = rows[0]

    report.update(_verdict(report))
    return report


def _verdict(report: dict[str, Any]) -> dict[str, str]:
    """Which of the three explanations the evidence supports."""

    restriction = _restriction_note({"messages": report["envelope"].get("messages"),
                                     "data_query_restrictions":
                                         report["envelope"].get("data_query_restrictions"),
                                     "error": report["envelope"].get("error"),
                                     "status": report["envelope"].get("status")})
    if restriction:
        return {
            "verdict": "account_tier",
            "evidence": restriction,
            "next_step": (
                "ACLED states a recency or quota limit in its own response. "
                "Contact access@acleddata.com, the route ACLED publishes for "
                "access beyond the free tier; UNICEF should qualify for a "
                "research or humanitarian tier. Do not change the connector."
            ),
        }

    if report.get("period_shaped_columns"):
        return {
            "verdict": "schema_change",
            "evidence": (
                "rows carry period-shaped columns "
                f"{report['period_shaped_columns']} — CAST's current "
                "methodology page describes a four-week rolling period, and "
                "the API documentation has not caught up"
            ),
            "next_step": (
                "Re-key the connector on the rolling four-week period, map "
                "each period to the calendar month containing its END date "
                "for our conflict_forecasts schema, and set "
                "forecast_issue_date from timestamp rather than month/year."
            ),
        }

    max_ts = report.get("max_timestamp_date") or ""
    max_year = report.get("max_year")
    if max_ts >= "2026-01-01" and max_year is not None and int(max_year) <= 2025:
        return {
            "verdict": "schema_change",
            "evidence": (
                f"timestamp is recent ({max_ts}) while month/year stop at "
                f"{max_year} — the connector's lead-month arithmetic reads "
                "month/year, so a moved temporal unit produces exactly this"
            ),
            "next_step": (
                "Re-key the connector on whatever field now carries the "
                "forecast period, and set forecast_issue_date from timestamp."
            ),
        }

    if max_ts and max_ts < "2026-01-01":
        return {
            "verdict": "upstream_stopped",
            "evidence": (
                f"the newest row timestamp is {max_ts}; nothing in the "
                "response points at a quota limit or a changed shape"
            ),
            "next_step": (
                "ACLED has stopped publishing to this endpoint. Escalate, and "
                "meanwhile mark the vintage's age in the ACE prompt and build "
                "a baseline from the ACLED history already in the DB "
                "(acled_monthly_fatalities, acled_political_events)."
            ),
        }

    return {
        "verdict": "inconclusive",
        "evidence": (
            f"rows={report.get('rows_returned')}, max_year={max_year}, "
            f"max_timestamp={max_ts or 'none'}"
        ),
        "next_step": (
            "Nothing in one call decided it. Re-run with a larger --limit, "
            "and ask ACLED directly — nine months is long enough that a "
            "support email costs nothing and may answer faster."
        ),
    }


def render(report: dict[str, Any]) -> str:
    lines = [
        "=" * 72,
        "ACLED CAST diagnostic",
        "=" * 72,
        f"endpoint      : {report['endpoint']}",
        f"checked at    : {report['checked_at']}",
        f"http status   : {report.get('http_status', report.get('http_error'))}",
        f"rows returned : {report.get('rows_returned', 0)}",
        f"years present : {report.get('years_present')}",
        f"max timestamp : {report.get('max_timestamp_date')}",
        "",
        "--- envelope (the half the connector discards) ---",
        json.dumps(report.get("envelope", {}), indent=2, default=str)[:2000],
        "",
        "--- columns ---",
        f"undocumented  : {report.get('columns_undocumented')}",
        f"documented but missing: {report.get('columns_documented_missing')}",
        f"period-shaped : {report.get('period_shaped_columns')}",
        "",
        "=" * 72,
        f"VERDICT: {report.get('verdict')}",
        f"evidence: {report.get('evidence', '')}",
        "",
        f"next step: {report.get('next_step', '')}",
        "=" * 72,
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="diagnose-acled-cast", description=__doc__)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--out", default=os.getenv("CAST_DIAGNOSTIC_PATH", ""))
    args = parser.parse_args(argv)

    report = diagnose(limit=args.limit)
    print(render(report))

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"\nwrote {path}")

    # Always 0. "The endpoint is quiet" is a finding, not a build failure —
    # and a diagnostic that can fail a run is a diagnostic people switch off.
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
