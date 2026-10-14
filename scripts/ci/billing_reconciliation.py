# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""What each provider billed for a month, beside what our ledger recorded.

Every cost view in Pythia reads ``llm_calls.cost_usd``, priced from
``pythia/model_costs.json``. A wrong price, a missed call path or a provider
redirect shows up nowhere until the invoice arrives, so once a month this
compares the ledger with the provider's own cost report:

* Anthropic: the Admin API cost report
  (``GET /v1/organizations/cost_report``), with ``ANTHROPIC_ADMIN_API_KEY``.
  Amounts are reported in cents as decimal strings (``ANTHROPIC_AMOUNT_DIVISOR``).
* OpenAI: the organization costs endpoint
  (``GET /v1/organization/costs``), with ``OPENAI_ADMIN_API_KEY``; amounts
  in USD.
* Google: "not available". Google Cloud Billing has no cost API for an API
  key; the figure lives in the BigQuery billing export, which this job does
  not read.

The ledger is every ``llm_calls`` row of the month, test runs included,
because the provider bills test runs too. A provider more than
``PYTHIA_BILLING_DIFF_WARN_PCT`` (15) per cent off is ``mismatch`` and gets
a ``::warning``. The result is written to ``billing_reconciliation`` (one
row per month and provider, rewritten for the month), to ``--out`` and to
the step summary. Neither endpoint could be reached from the build
sandbox, so their response shapes follow the providers' documentation;
the first live run is the check.

Always exits 0.

Usage:
    python -m scripts.ci.billing_reconciliation --db data/resolver.duckdb \
        [--month 2026-09] --out diagnostics/billing_reconciliation.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

PROVIDERS = ("anthropic", "openai", "google")
ANTHROPIC_COST_URL = "https://api.anthropic.com/v1/organizations/cost_report"
OPENAI_COST_URL = "https://api.openai.com/v1/organization/costs"
ANTHROPIC_AMOUNT_DIVISOR = 100.0  # cents -> dollars
MAX_PAGES = 20

STATUS_OK = "ok"
STATUS_MISMATCH = "mismatch"
STATUS_UNAVAILABLE = "unavailable"

DDL = """
CREATE TABLE IF NOT EXISTS billing_reconciliation (
    month TEXT,
    provider TEXT,
    ledger_usd DOUBLE,
    billed_usd DOUBLE,
    diff_usd DOUBLE,
    diff_pct DOUBLE,
    status TEXT,
    reason TEXT,
    checked_at TIMESTAMP,
    PRIMARY KEY (month, provider)
)
"""


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except ValueError:
        return default


def previous_month(today: Optional[date] = None) -> str:
    today = today or date.today()
    y, m = (today.year, today.month - 1) if today.month > 1 else (today.year - 1, 12)
    return f"{y:04d}-{m:02d}"


def month_bounds(month: str) -> Tuple[datetime, datetime]:
    y, m = (int(x) for x in month.split("-"))
    start = datetime(y, m, 1, tzinfo=timezone.utc)
    end = datetime(y + (m == 12), 1 if m == 12 else m + 1, 1, tzinfo=timezone.utc)
    return start, end


def ledger_by_provider(con, month: str) -> Dict[str, float]:
    start, end = month_bounds(month)
    rows = con.execute(
        "SELECT LOWER(COALESCE(provider, '')), SUM(COALESCE(cost_usd, 0)) FROM llm_calls "
        "WHERE timestamp >= ? AND timestamp < ? GROUP BY 1",
        [start.replace(tzinfo=None), end.replace(tzinfo=None)],
    ).fetchall()
    out: Dict[str, float] = {}
    for prov, cost in rows:
        prov = "google" if prov in ("gemini", "google") else prov
        out[prov] = out.get(prov, 0.0) + float(cost or 0.0)
    return out


def _get(url: str, headers: Dict[str, str], params: Dict[str, Any]) -> Dict[str, Any]:
    import requests  # noqa: PLC0415

    resp = requests.get(url, headers=headers, params=params, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
    return resp.json()


def anthropic_billed(month: str, key: str, get: Callable = _get) -> float:
    start, end = month_bounds(month)
    params: Dict[str, Any] = {
        "starting_at": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ending_at": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "bucket_width": "1d",
        "limit": 31,
    }
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
    total = 0.0
    for _ in range(MAX_PAGES):
        payload = get(ANTHROPIC_COST_URL, headers, params)
        for bucket in payload.get("data") or []:
            for item in bucket.get("results") or []:
                total += float(item.get("amount") or 0.0) / ANTHROPIC_AMOUNT_DIVISOR
        if not payload.get("has_more") or not payload.get("next_page"):
            break
        params = dict(params, page=payload["next_page"])
    return total


def openai_billed(month: str, key: str, get: Callable = _get) -> float:
    start, end = month_bounds(month)
    params: Dict[str, Any] = {
        "start_time": int(start.timestamp()),
        "end_time": int(end.timestamp()),
        "bucket_width": "1d",
        "limit": 31,
    }
    headers = {"Authorization": f"Bearer {key}"}
    total = 0.0
    for _ in range(MAX_PAGES):
        payload = get(OPENAI_COST_URL, headers, params)
        for bucket in payload.get("data") or []:
            for item in bucket.get("results") or []:
                amount = item.get("amount") or {}
                total += float((amount.get("value") if isinstance(amount, dict) else amount) or 0.0)
        if not payload.get("has_more") or not payload.get("next_page"):
            break
        params = dict(params, page=payload["next_page"])
    return total


def compare(ledger: Optional[float], billed: Optional[float], *, warn_pct: float) -> Dict[str, Any]:
    if billed is None:
        return {"diff_usd": None, "diff_pct": None, "status": STATUS_UNAVAILABLE}
    ledger = float(ledger or 0.0)
    diff = ledger - billed
    pct = (100.0 * diff / billed) if billed else (0.0 if ledger == 0 else 100.0)
    return {
        "diff_usd": round(diff, 4),
        "diff_pct": round(pct, 1),
        "status": STATUS_MISMATCH if abs(pct) > warn_pct else STATUS_OK,
    }


def reconcile(con, month: str, *, env: Optional[Dict[str, str]] = None,
              get: Callable = _get) -> List[Dict[str, Any]]:
    env = dict(os.environ if env is None else env)
    warn_pct = _env_float("PYTHIA_BILLING_DIFF_WARN_PCT", 15.0)
    ledger = ledger_by_provider(con, month)
    fetchers = {
        "anthropic": ("ANTHROPIC_ADMIN_API_KEY", anthropic_billed),
        "openai": ("OPENAI_ADMIN_API_KEY", openai_billed),
    }
    rows: List[Dict[str, Any]] = []
    for prov in PROVIDERS:
        billed: Optional[float] = None
        reason = ""
        if prov in fetchers:
            key_name, fn = fetchers[prov]
            key = (env.get(key_name) or "").strip()
            if not key:
                reason = f"{key_name} not set: the cost report needs an admin key"
            else:
                try:
                    billed = round(fn(month, key, get), 4)
                except Exception as exc:  # noqa: BLE001
                    reason = _scrub(f"{type(exc).__name__}: {exc}")[:300]
        else:
            reason = ("not available: Google Cloud Billing has no cost API for an API key; "
                      "the figure is in the BigQuery billing export")
        cmp = compare(ledger.get(prov, 0.0), billed, warn_pct=warn_pct)
        if cmp["status"] == STATUS_MISMATCH:
            reason = f"ledger is {cmp['diff_pct']:+.1f}% off the provider's figure (limit {warn_pct:.0f}%)"
        rows.append({
            "month": month, "provider": prov,
            "ledger_usd": round(ledger.get(prov, 0.0), 4), "billed_usd": billed,
            **cmp, "reason": reason,
        })
    return rows


def _scrub(text: str) -> str:
    try:
        from pythia.secret_scrub import scrub_text  # noqa: PLC0415

        return scrub_text(text)
    except Exception:  # noqa: BLE001
        return text


def store(con, rows: List[Dict[str, Any]]) -> None:
    con.execute(DDL)
    if not rows:
        return
    con.execute("DELETE FROM billing_reconciliation WHERE month = ?", [rows[0]["month"]])
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    for r in rows:
        con.execute(
            "INSERT INTO billing_reconciliation VALUES (?,?,?,?,?,?,?,?,?)",
            [r["month"], r["provider"], r["ledger_usd"], r["billed_usd"], r["diff_usd"],
             r["diff_pct"], r["status"], r["reason"], now],
        )


def render(rows: List[Dict[str, Any]]) -> str:
    out = ["## Billing reconciliation", "",
           "| month | provider | ledger | billed | diff | status | note |",
           "|---|---|---|---|---|---|---|"]
    for r in rows:
        billed = "" if r["billed_usd"] is None else f"${r['billed_usd']:.2f}"
        diff = "" if r["diff_pct"] is None else f"{r['diff_pct']:+.1f}%"
        out.append(f"| {r['month']} | {r['provider']} | ${r['ledger_usd']:.2f} | {billed} | "
                   f"{diff} | {r['status']} | {r['reason']} |")
    return "\n".join(out) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", required=True)
    ap.add_argument("--month", default="")
    ap.add_argument("--out", default="diagnostics/billing_reconciliation.json")
    args = ap.parse_args(argv)
    month = args.month or previous_month()
    path = args.db[len("duckdb:///"):] if args.db.startswith("duckdb:///") else args.db
    rows: List[Dict[str, Any]] = []
    try:
        import duckdb  # noqa: PLC0415

        con = duckdb.connect(path)
        try:
            rows = reconcile(con, month)
            store(con, rows)
        finally:
            con.close()
    except Exception as exc:  # noqa: BLE001
        print(f"::warning title=Billing reconciliation::{type(exc).__name__}: {_scrub(str(exc))[:300]}")
    for r in rows:
        if r["status"] == STATUS_MISMATCH:
            print(f"::warning title=Billing reconciliation::{r['provider']} {month}: {r['reason']}")
    text = render(rows)
    print(text)
    try:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"month": month, "rows": rows}, indent=2), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    summary = os.getenv("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
