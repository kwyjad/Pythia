# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""IOM DTM conflict IDP stock, and the lower bound on new displacement it gives.

DTM is not a flow source. The 7 Oct 2026 probe (run 37620577120) found every
admin0 series a STOCK (``numPresentIdpInd``: IDPs present at a reporting
date), 21 of 38 countries covered, and a median of median gaps between
reports of 122 days. So this module never reads a DTM figure as a month's
new displacement. What it does read: when two consecutive rounds are close
together, the stock can only have risen by people who arrived in between
(returns only reduce it), so ``max(0, stock_t - stock_{t-1})`` is a LOWER
BOUND on new displacement over that interval.

The rule, and nothing looser:

* only records whose displacement reason is conflict (the probe's
  ``reason_class``); a country whose records carry no reason field gives no
  bound, because an all-cause stock rising is not conflict displacement;
* one series per DTM operation (two operations in one country can overlap,
  so their stocks are never added and their bounds never summed: a month
  takes the largest operation's bound, named on the row);
* consecutive rounds at most ``conflict_displacement.dtm.max_gap_days`` (35)
  apart; never differenced across a longer gap;
* the increase attributed WHOLE to the month of the later round, never
  apportioned; several close pairs ending in one month add up, which is
  still a lower bound on the month's new displacement plus the days of the
  first interval that fell in the month before;
* labelled ``displaced_lower_bound`` on every row.

Part A (Oct 2026): recorded and reported only. Part B reads
:func:`dtm_lower_bound` / :func:`dtm_lower_bounds`.

The key travels in the ``Ocp-Apim-Subscription-Key`` header, read from
``DTM_API_KEY`` or ``DTM_API_PRIMARY_KEY``, never a URL; every error text is
scrubbed before it is logged or stored (as ``tools/probe_dtm_conflict.py``
does, whose response helpers this module imports).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping

from tools.probe_dtm_conflict import (
    DTM_BASE,
    USER_AGENT,
    _find_field,
    extract_records,
    months_back,
    reason_class,
    record_date,
    record_iso3,
    record_round,
    scrub,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import duckdb

LOG = logging.getLogger(__name__)

VALUE_TYPE = "displaced_lower_bound"
STOCK_FIELDS = ("numPresentIdpInd",)
REASON_TOKENS = ("reason", "cause", "trigger", "displacementtype", "displacement_type", "shock")
DEFAULT_MAX_GAP_DAYS = 35
REQUEST_TIMEOUT_SEC = 60.0

_DDL = (
    """
    CREATE TABLE IF NOT EXISTS dtm_conflict_rounds (
        iso3 TEXT NOT NULL,
        operation TEXT NOT NULL,
        reporting_date DATE NOT NULL,
        stock DOUBLE NOT NULL,
        n_records INTEGER,
        round_number TEXT,
        reasons_json TEXT,
        source_url TEXT,
        retrieved_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (iso3, operation, reporting_date)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS dtm_conflict_lower_bounds (
        iso3 TEXT NOT NULL,
        ym TEXT NOT NULL,
        value DOUBLE NOT NULL,
        value_type TEXT NOT NULL,
        operation TEXT,
        n_pairs INTEGER,
        pairs_json TEXT,
        operations_json TEXT,
        max_gap_days INTEGER,
        computed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (iso3, ym)
    )
    """,
)


def ensure_schema(con: "duckdb.DuckDBPyConnection") -> None:
    for ddl in _DDL:
        con.execute(ddl)


# ---------------------------------------------------------------------------
# Pure: records -> per-operation stock series -> monthly lower bounds
# ---------------------------------------------------------------------------


def _stock(record: Mapping[str, Any]) -> float | None:
    for name in STOCK_FIELDS:
        raw = record.get(name)
        if raw is None or isinstance(raw, bool):
            continue
        try:
            value = float(str(raw).replace(",", ""))
        except ValueError:
            continue
        if value >= 0:
            return value
    return None


def conflict_series(
    records: Iterable[Mapping[str, Any]], iso3: str
) -> tuple[dict[str, dict[dt.date, dict[str, Any]]], dict[str, int]]:
    """Per operation, the conflict IDP stock at each reporting date.

    Returns ``(series, excluded)`` where ``series[operation][date]`` holds
    ``stock`` (summed over that date's conflict-reason records), ``n_records``,
    ``round`` and ``reasons``; ``excluded`` counts every record left out by
    reason (``no_reason_field``, ``not_conflict``, ``no_stock``, ``undated``,
    ``other_country``).
    """

    iso3 = iso3.upper()
    series: dict[str, dict[dt.date, dict[str, Any]]] = defaultdict(dict)
    excluded: dict[str, int] = defaultdict(int)
    for record in records:
        if not isinstance(record, Mapping):
            continue
        if record_iso3(record) not in (None, iso3):
            excluded["other_country"] += 1
            continue
        key = _find_field(record, REASON_TOKENS)
        if not key:
            excluded["no_reason_field"] += 1
            continue
        if reason_class(record.get(key)) != "conflict":
            excluded["not_conflict"] += 1
            continue
        stock = _stock(record)
        if stock is None:
            excluded["no_stock"] += 1
            continue
        day = record_date(record)
        if day is None:
            excluded["undated"] += 1
            continue
        op_key = _find_field(record, ("operation",), exclude=("id", "status", "type"))
        operation = str(record.get(op_key) or "").strip() if op_key else ""
        slot = series[operation].setdefault(
            day, {"stock": 0.0, "n_records": 0, "round": None, "reasons": []}
        )
        slot["stock"] += stock
        slot["n_records"] += 1
        slot["round"] = slot["round"] or record_round(record)
        reason = str(record.get(key))
        if reason not in slot["reasons"]:
            slot["reasons"].append(reason)
    return dict(series), dict(excluded)


def derive_lower_bounds(
    series: Mapping[str, Mapping[dt.date, Mapping[str, Any]]],
    *,
    max_gap_days: int = DEFAULT_MAX_GAP_DAYS,
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    """Per month, the lower bound on new conflict displacement. Pure.

    For each operation, consecutive rounds at most ``max_gap_days`` apart
    give ``max(0, stock_t - stock_{t-1})``, attributed to the month of the
    later round; pairs ending in the same month add up. A month takes the
    largest operation's bound (operations may overlap, so they are never
    added). Returns ``(bounds, counts)``; ``counts`` has ``pairs_used`` and
    ``pairs_skipped_gap``.
    """

    per_op_month: dict[tuple[str, str], dict[str, Any]] = {}
    counts = {"pairs_used": 0, "pairs_skipped_gap": 0}
    for operation, by_date in series.items():
        dates = sorted(by_date)
        for prev, cur in zip(dates, dates[1:]):
            gap = (cur - prev).days
            if gap > max_gap_days:
                counts["pairs_skipped_gap"] += 1
                continue
            counts["pairs_used"] += 1
            before = float(by_date[prev]["stock"])
            after = float(by_date[cur]["stock"])
            increase = max(0.0, after - before)
            ym = cur.strftime("%Y-%m")
            slot = per_op_month.setdefault((operation, ym), {"value": 0.0, "pairs": []})
            slot["value"] += increase
            slot["pairs"].append({
                "from": prev.isoformat(), "to": cur.isoformat(), "gap_days": gap,
                "stock_from": before, "stock_to": after, "increase": increase,
            })
    bounds: dict[str, dict[str, Any]] = {}
    for (operation, ym), slot in sorted(per_op_month.items()):
        current = bounds.get(ym)
        operations = (current or {}).get("operations", {})
        operations[operation] = slot["value"]
        if current is None or slot["value"] > current["value"]:
            bounds[ym] = {
                "value": slot["value"], "operation": operation,
                "pairs": slot["pairs"], "operations": operations,
            }
        else:
            current["operations"] = operations
    return bounds, counts


# ---------------------------------------------------------------------------
# Transport (injectable)
# ---------------------------------------------------------------------------

#: (url, params, headers, timeout) -> (status, parsed json or None, text)
GetFn = Callable[[str, dict, dict, float], "tuple[int, Any, str]"]


def default_get(url: str, params: dict, headers: dict, timeout: float) -> tuple[int, Any, str]:
    import requests

    resp = requests.get(url, params=params, headers=headers, timeout=timeout)
    try:
        payload = resp.json()
    except ValueError:
        payload = None
    return resp.status_code, payload, resp.text[:400]


def read_key() -> str:
    return (os.getenv("DTM_API_KEY") or os.getenv("DTM_API_PRIMARY_KEY") or "").strip()


def fetch_admin0(
    iso3: str, start: dt.date, end: dt.date, key: str, *, get: GetFn | None = None
) -> tuple[list[dict], str | None]:
    """The admin0 records for one country. Never raises; errors are scrubbed."""

    get = get or default_get
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json",
               "Ocp-Apim-Subscription-Key": key}
    params = {"Admin0Pcode": iso3.upper(), "FromReportingDate": start.isoformat(),
              "ToReportingDate": end.isoformat()}
    try:
        status, payload, text = get(f"{DTM_BASE}/admin0", params, headers, REQUEST_TIMEOUT_SEC)
    except Exception as exc:  # noqa: BLE001
        return [], scrub(f"{type(exc).__name__}: {exc}", key)[:300]
    if status != 200:
        return [], scrub(f"HTTP {status}: {text}", key)[:300]
    if payload is None:
        return [], scrub(f"HTTP 200 body is not JSON: {text}", key)[:300]
    return extract_records(payload), None


# ---------------------------------------------------------------------------
# Store and read
# ---------------------------------------------------------------------------


def store_country(
    con: "duckdb.DuckDBPyConnection",
    iso3: str,
    series: Mapping[str, Mapping[dt.date, Mapping[str, Any]]],
    bounds: Mapping[str, Mapping[str, Any]],
    *,
    max_gap_days: int,
) -> dict[str, int]:
    """Upsert the rounds and REWRITE the country's lower bounds."""

    ensure_schema(con)
    iso3 = iso3.upper()
    rounds = 0
    for operation, by_date in series.items():
        for day, slot in by_date.items():
            con.execute(
                """
                INSERT OR REPLACE INTO dtm_conflict_rounds
                    (iso3, operation, reporting_date, stock, n_records, round_number,
                     reasons_json, source_url, retrieved_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                [iso3, operation, day, float(slot["stock"]), int(slot["n_records"]),
                 None if slot.get("round") is None else str(slot["round"]),
                 json.dumps(slot.get("reasons") or []), f"{DTM_BASE}/admin0"],
            )
            rounds += 1
    con.execute("DELETE FROM dtm_conflict_lower_bounds WHERE iso3 = ?", [iso3])
    for ym, b in bounds.items():
        con.execute(
            """
            INSERT INTO dtm_conflict_lower_bounds
                (iso3, ym, value, value_type, operation, n_pairs, pairs_json,
                 operations_json, max_gap_days, computed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            [iso3, ym, float(b["value"]), VALUE_TYPE, b.get("operation"),
             len(b.get("pairs") or []), json.dumps(b.get("pairs") or []),
             json.dumps(b.get("operations") or {}), int(max_gap_days)],
        )
    return {"rounds": rounds, "lower_bound_months": len(bounds)}


def dtm_lower_bounds(
    con: "duckdb.DuckDBPyConnection", iso3: str | None = None
) -> dict[tuple[str, str], dict[str, Any]]:
    """Per (iso3, ym), the DTM lower bound row. The API Part B reads.

    ``value`` is a LOWER BOUND on people newly displaced by conflict in the
    month (``value_type == 'displaced_lower_bound'``), ``pairs_json`` the
    rounds it came from. Absent where no two close rounds bracket the month.
    """

    try:
        ensure_schema(con)
        sql = "SELECT * FROM dtm_conflict_lower_bounds"
        params: list[Any] = []
        if iso3:
            sql += " WHERE iso3 = ?"
            params.append(iso3.upper())
        cur = con.execute(sql + " ORDER BY iso3, ym", params)
        names = [d[0] for d in cur.description]
        rows = [dict(zip(names, r)) for r in cur.fetchall()]
    except Exception as exc:  # noqa: BLE001 - a reader must not raise
        LOG.warning("[dtm_conflict] lower bounds unreadable: %s", exc)
        return {}
    return {(r["iso3"], r["ym"]): r for r in rows}


def dtm_lower_bound(
    con: "duckdb.DuckDBPyConnection", iso3: str, ym: str
) -> dict[str, Any] | None:
    return dtm_lower_bounds(con, iso3).get((iso3.upper(), ym))


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def run(
    con: "duckdb.DuckDBPyConnection",
    countries: list[str],
    *,
    key: str,
    months: int = 36,
    max_gap_days: int = DEFAULT_MAX_GAP_DAYS,
    today: dt.date | None = None,
    get: GetFn | None = None,
    delay: float = 0.5,
) -> dict[str, Any]:
    """Fetch, derive and store every country. One failure costs one country."""

    ensure_schema(con)
    today = today or dt.date.today()
    start = months_back(today, months)
    summary: dict[str, Any] = {
        "window": f"{start.isoformat()}..{today.isoformat()}", "max_gap_days": max_gap_days,
        "per_country": {}, "countries_with_rounds": [], "countries_with_bounds": [],
        "fetch_errors": {},
    }
    for iso3 in countries:
        records, error = fetch_admin0(iso3, start, today, key, get=get)
        if error:
            summary["fetch_errors"][iso3] = error
            continue
        series, excluded = conflict_series(records, iso3)
        bounds, counts = derive_lower_bounds(series, max_gap_days=max_gap_days)
        stored = store_country(con, iso3, series, bounds, max_gap_days=max_gap_days)
        summary["per_country"][iso3] = {
            "records": len(records), "excluded": excluded,
            "operations": sorted(series), **counts, **stored,
        }
        if stored["rounds"]:
            summary["countries_with_rounds"].append(iso3)
        if stored["lower_bound_months"]:
            summary["countries_with_bounds"].append(iso3)
        if delay:
            time.sleep(delay)
    summary["lower_bound_rows"] = sum(
        v.get("lower_bound_months", 0) for v in summary["per_country"].values()
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", default=None, help="DuckDB path or duckdb:/// URL")
    parser.add_argument("--countries", default="", help="ISO3s (default: ACE countries)")
    parser.add_argument("--all-listed", action="store_true")
    parser.add_argument("--months", type=int, default=None)
    parser.add_argument("--summary-out", default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s [%(name)s] %(message)s")

    from resolver.db.duckdb_io import close_db, get_db
    from resolver.hazard_resolution.rulebook import load_rulebook
    from resolver.ingestion.conflict_displacement_extract import plan_countries

    key = read_key()
    summary: dict[str, Any] = {"key_present": bool(key)}
    exit_code = 0
    con = None
    try:
        rulebook = load_rulebook()
        months = args.months or int(rulebook.get("conflict_displacement.dtm.months"))
        gap = int(rulebook.get("conflict_displacement.dtm.max_gap_days"))
        db = args.db or os.getenv("RESOLVER_DB_URL") or None
        if db and "://" not in db:
            db = f"duckdb:///{Path(db).resolve()}"
        con = get_db(db)
        countries = [c.strip().upper() for c in args.countries.replace(",", " ").split() if c.strip()]
        countries = countries or plan_countries(con, all_listed=args.all_listed)
        summary["countries"] = countries
        if not key:
            print("::warning::no DTM_API_KEY or DTM_API_PRIMARY_KEY; DTM conflict "
                  "lower bounds not refreshed")
            summary["status"] = "no_credentials"
        else:
            summary.update(run(con, countries, key=key, months=months, max_gap_days=gap))
            summary["status"] = "ok"
            print(f"[dtm_conflict] {len(summary['countries_with_rounds'])} countries with "
                  f"conflict rounds, {len(summary['countries_with_bounds'])} with lower "
                  f"bounds, {summary['lower_bound_rows']} month rows; "
                  f"{len(summary['fetch_errors'])} fetch errors", flush=True)
    except Exception as exc:  # noqa: BLE001
        LOG.exception("[dtm_conflict] run failed")
        summary["error"] = scrub(f"{type(exc).__name__}: {exc}", key)[:500]
        exit_code = 1
    finally:
        close_db(con)
    if args.summary_out:
        out = Path(args.summary_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(scrub(json.dumps(summary, indent=2, default=str), key), encoding="utf-8")
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
