# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Read-only probe: what does IOM's DTM serve for conflict displacement?

ACE/PA questions resolve on IDMC conflict displacement, and IDMC reports late
and irregularly: only a handful of the countries with a current ACE/PA
question report in eight of every twelve months. Before anything is built on
IOM's Displacement Tracking Matrix as a second source, five facts about it
have to be measured rather than assumed, per country:

1. coverage: does DTM serve conflict displacement for the country at all;
2. cadence: the dates of its rounds/reports over the last 36 months and the
   median gap between them;
3. lag: reporting date against the period the figure describes, where a
   record carries both;
4. flow or stock: new displacements in a period, or IDPs present at a date,
   and which fields say so;
5. reason: whether a displacement-reason field exists and whether conflict
   can be told from disaster.

Routes: the DTM API v3 (``/v3/displacement/admin0`` and the admin1/admin2
routes; its key travels in the ``Ocp-Apim-Subscription-Key`` HEADER, read
from ``DTM_API_KEY`` or ``DTM_API_PRIMARY_KEY``, never a URL), the routes it
may expose without a key, and HDX's CKAN search for IOM DTM datasets. Every
request is recorded with its URL, status and a redacted body snippet; one
failure costs that route only. Field detection is tolerant: the JSON is
walked for keys naming a date, round, IDP, displacement, reason, cause,
admin0 or ISO code.

The build sandbox cannot reach these hosts; the workflow
``probe_dtm_conflict.yml`` is how this runs. It writes ``report.json`` and
``report.md``, writes nothing else, and always exits 0.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

DEFAULT_COUNTRIES = (
    "AFG BDI BFA CAF CMR COD COL ETH HTI IND IRQ ISR LBN LBY MLI MMR MOZ NER "
    "NGA PAK PER PHL PSE RUS SDN SOM SSD SYR TCD UKR VEN YEM MEX KEN UGA ECU "
    "BGD IDN"
)
DTM_BASE = "https://dtmapi.iom.int/v3/displacement"
HDX_SEARCH = "https://data.humdata.org/api/3/action/package_search"
IOM_ORG = "international-organization-for-migration"
USER_AGENT = "Mozilla/5.0 (compatible; PythiaProbe/1.0; +https://fredforecaster.org)"
SNIPPET_CHARS = 400
KEY_TOKENS = ("date", "round", "idp", "displac", "reason", "cause", "admin0", "iso",
              "period", "present", "new", "arriv", "flow", "stock")

# A field is about WHEN a figure was reported, or about the period it describes.
REPORT_DATE_TOKENS = ("reportingdate", "reporting_date", "reportdate", "report_date",
                      "publicationdate", "publication_date", "published")
PERIOD_DATE_TOKENS = ("period", "collection", "survey", "assessment", "roundend",
                      "round_end", "enddate", "end_date", "reference", "asof", "as_of")

STOCK_TOKENS = ("present", "stock", "numidp", "num_idp", "total_idp", "idps")
FLOW_TOKENS = ("new", "arrival", "arriv", "flow", "newly", "movement", "displacedin",
               "displaced_in", "inflow", "outflow")

CONFLICT_WORDS = ("conflict", "violence", "insecurity", "attack", "armed", "war",
                  "clash", "fighting", "military", "terror", "crime", "political",
                  "communal", "intercommunal", "banditry")
DISASTER_WORDS = ("disaster", "flood", "drought", "cyclone", "storm", "earthquake",
                  "natural", "climate", "fire", "landslide", "hurricane", "typhoon",
                  "volcan", "rain", "tsunami", "environment")


# --------------------------------------------------------------------------
# Pure helpers (tested in resolver/tests/test_probe_dtm_conflict.py)
# --------------------------------------------------------------------------

def scrub(text: str, secret: str) -> str:
    """Remove the key from any text before it is printed or written."""
    return text.replace(secret, "REDACTED") if secret else text


def _norm(key: str) -> str:
    return re.sub(r"[^a-z0-9_]", "", str(key).lower())


def walk_keys(obj: Any, *, limit: int = 20000) -> Counter:
    """Every dict key anywhere in a parsed JSON value, with how often it occurs."""
    seen: Counter = Counter()
    stack = [obj]
    visited = 0
    while stack and visited < limit:
        node = stack.pop()
        visited += 1
        if isinstance(node, Mapping):
            for k, v in node.items():
                seen[str(k)] += 1
                stack.append(v)
        elif isinstance(node, list):
            stack.extend(node)
    return seen


def interesting_keys(keys: Iterable[str]) -> list[str]:
    """Keys whose name mentions a token this probe cares about."""
    return sorted({k for k in keys if any(t in _norm(k) for t in KEY_TOKENS)})


def extract_records(payload: Any) -> list[dict]:
    """The list of records in a response, whatever envelope it came in.

    DTM v3 answers ``{"isSuccess": ..., "result": [...]}``; other routes use
    ``data``, ``results``, ``value`` or a bare list. Falls back to the largest
    list of dicts found anywhere in the payload.
    """
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    if not isinstance(payload, Mapping):
        return []
    for field in ("result", "results", "data", "value", "items", "records"):
        got = payload.get(field)
        if isinstance(got, list):
            return [r for r in got if isinstance(r, dict)]
        if isinstance(got, Mapping):
            inner = extract_records(got)
            if inner:
                return inner
    best: list[dict] = []
    stack: list[Any] = list(payload.values())
    while stack:
        node = stack.pop()
        if isinstance(node, list):
            # A record carries scalar values; a wrapper holding lists does not.
            dicts = [r for r in node if isinstance(r, dict)
                     and any(not isinstance(v, (dict, list)) for v in r.values())]
            if len(dicts) > len(best):
                best = dicts
            stack.extend(node)
        elif isinstance(node, Mapping):
            stack.extend(node.values())
    return best


def parse_date(value: Any) -> dt.date | None:
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return dt.date.fromisoformat(text[:10])
    except ValueError:
        pass
    for fmt in ("%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d", "%d-%m-%Y", "%Y-%m", "%b %Y", "%B %Y"):
        try:
            return dt.datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _find_field(record: Mapping[str, Any], tokens: Iterable[str],
                exclude: Iterable[str] = ()) -> str | None:
    excl = tuple(exclude)
    for key in record:
        n = _norm(key)
        if any(t in n for t in tokens) and not any(e in n for e in excl):
            return key
    return None


def report_date_field(record: Mapping[str, Any]) -> str | None:
    key = _find_field(record, REPORT_DATE_TOKENS, exclude=("year", "month"))
    if key and parse_date(record.get(key)):
        return key
    return None


def period_date_field(record: Mapping[str, Any]) -> str | None:
    """A date field describing the period, distinct from the reporting date."""
    rep = report_date_field(record)
    for key in record:
        if key == rep:
            continue
        n = _norm(key)
        if "date" not in n and "asof" not in n and "as_of" not in n:
            continue
        if any(t in n for t in PERIOD_DATE_TOKENS) and parse_date(record.get(key)):
            return key
    return None


def record_date(record: Mapping[str, Any]) -> dt.date | None:
    """The date a record was reported, from the reporting-date field, else
    year+month fields, else any field naming a date."""
    key = report_date_field(record)
    if key:
        return parse_date(record.get(key))
    year = month = None
    for k, v in record.items():
        n = _norm(k)
        if "year" in n and year is None:
            try:
                year = int(v)
            except (TypeError, ValueError):
                pass
        elif "month" in n and month is None:
            try:
                month = int(v)
            except (TypeError, ValueError):
                pass
    if year and month and 1 <= month <= 12:
        return dt.date(year, month, 1)
    for k, v in record.items():
        if "date" in _norm(k):
            got = parse_date(v)
            if got:
                return got
    return None


def record_round(record: Mapping[str, Any]) -> Any:
    key = _find_field(record, ("round",), exclude=("date",))
    return record.get(key) if key else None


def record_iso3(record: Mapping[str, Any]) -> str | None:
    for key in record:
        n = _norm(key)
        if ("admin0pcode" in n or "iso3" in n or n in ("iso", "countrycode", "country_code")):
            val = str(record.get(key) or "").strip().upper()
            if len(val) == 3 and val.isalpha():
                return val
    return None


def months_back(today: dt.date, months: int) -> dt.date:
    y, m = today.year, today.month - months
    while m <= 0:
        m += 12
        y -= 1
    return dt.date(y, m, 1)


def cadence(dates: Iterable[dt.date], *, today: dt.date, months: int = 36) -> dict[str, Any]:
    """Distinct report dates inside the window, the gaps between them, and the
    median gap in days."""
    start = months_back(today, months)
    distinct = sorted({d for d in dates if d and start <= d <= today})
    gaps = [(b - a).days for a, b in zip(distinct, distinct[1:])]
    months_covered = sorted({d.strftime("%Y-%m") for d in distinct})
    return {
        "window": f"{start.isoformat()}..{today.isoformat()}",
        "n_reports": len(distinct),
        "first": distinct[0].isoformat() if distinct else None,
        "last": distinct[-1].isoformat() if distinct else None,
        "days_since_last": (today - distinct[-1]).days if distinct else None,
        "median_gap_days": statistics.median(gaps) if gaps else None,
        "max_gap_days": max(gaps) if gaps else None,
        "months_with_a_report": len(months_covered),
        "dates": [d.isoformat() for d in distinct],
    }


def lag_days(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Reporting date minus the period date, where a record carries both."""
    lags: list[int] = []
    pairs: Counter = Counter()
    for r in records:
        rep_key, per_key = report_date_field(r), period_date_field(r)
        if not (rep_key and per_key):
            continue
        rep, per = parse_date(r.get(rep_key)), parse_date(r.get(per_key))
        if rep and per:
            lags.append((rep - per).days)
            pairs[f"{rep_key} - {per_key}"] += 1
    return {
        "n": len(lags),
        "field_pairs": dict(pairs),
        "median_days": statistics.median(lags) if lags else None,
        "min_days": min(lags) if lags else None,
        "max_days": max(lags) if lags else None,
    }


def classify_flow_stock(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Are the numeric figures IDPs present at a date (stock) or new
    displacements in a period (flow)? Decided from the field names that carry
    numbers, and named so a reader can check."""
    stock_fields: Counter = Counter()
    flow_fields: Counter = Counter()
    for r in records:
        for k, v in r.items():
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                try:
                    float(str(v).replace(",", ""))
                except (TypeError, ValueError):
                    continue
                if not str(v).strip():
                    continue
            n = _norm(k)
            if not any(t in n for t in ("idp", "displac", "ind", "hh", "household",
                                         "individual", "people", "persons", "figure")):
                continue
            identifier = n == "id" or n.endswith("_id") or n.endswith("id") and not n.endswith("idp")
            if (identifier or any(t in n for t in ("round", "year", "month", "pcode"))) and not any(
                t in n for t in ("idp", "displac")
            ):
                continue
            if any(t in n for t in FLOW_TOKENS):
                flow_fields[k] += 1
            elif any(t in n for t in STOCK_TOKENS):
                stock_fields[k] += 1
    if stock_fields and flow_fields:
        verdict = "both"
    elif stock_fields:
        verdict = "stock"
    elif flow_fields:
        verdict = "flow"
    else:
        verdict = "unknown"
    return {"verdict": verdict, "stock_fields": dict(stock_fields), "flow_fields": dict(flow_fields)}


def reason_class(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return "blank"
    has_c = any(w in text for w in CONFLICT_WORDS)
    has_d = any(w in text for w in DISASTER_WORDS)
    if has_c and has_d:
        return "mixed"
    if has_c:
        return "conflict"
    if has_d:
        return "disaster"
    return "other"


def reason_split(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """The displacement-reason field, its values, and whether conflict can be
    separated from disaster."""
    fields: Counter = Counter()
    values: Counter = Counter()
    classes: Counter = Counter()
    n = 0
    for r in records:
        n += 1
        key = _find_field(r, ("reason", "cause", "trigger", "displacementtype",
                              "displacement_type", "shock"))
        if not key:
            continue
        fields[key] += 1
        values[str(r.get(key))] += 1
        classes[reason_class(r.get(key))] += 1
    if not fields:
        separable = "no_reason_field"
    elif classes.get("conflict") and classes.get("disaster"):
        separable = "yes"
    elif classes.get("conflict") and not classes.get("mixed"):
        separable = "conflict_only_values"
    elif classes.get("mixed"):
        separable = "mixed_values"
    else:
        separable = "no_conflict_values"
    return {
        "records": n,
        "fields": dict(fields),
        "values": dict(values.most_common(30)),
        "classes": dict(classes),
        "separable": separable,
    }


def conflict_records(records: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Records whose reason names conflict; every record when no reason field
    exists (the report says which case applied)."""
    rows = list(records)
    if not any(_find_field(r, ("reason", "cause", "trigger", "displacementtype",
                               "displacement_type", "shock")) for r in rows):
        return rows
    out = []
    for r in rows:
        key = _find_field(r, ("reason", "cause", "trigger", "displacementtype",
                              "displacement_type", "shock"))
        if key and reason_class(r.get(key)) in ("conflict", "mixed"):
            out.append(r)
    return out


def analyse_country(records: list[Mapping[str, Any]], *, today: dt.date,
                    months: int = 36) -> dict[str, Any]:
    """Everything the report says about one country, from its records alone."""
    keys: Counter = Counter()
    for r in records:
        keys.update(r.keys())
    conflict = conflict_records(records)
    reason = reason_split(records)
    dates = [d for d in (record_date(r) for r in conflict) if d]
    rounds = sorted({str(x) for x in (record_round(r) for r in conflict) if x not in (None, "")})
    return {
        "records": len(records),
        "conflict_records": len(conflict),
        "conflict_filter": "by_reason" if reason["fields"] else "no_reason_field_all_records",
        "covered": bool(conflict),
        "keys": dict(keys.most_common()),
        "interesting_keys": interesting_keys(keys),
        "cadence": cadence(dates, today=today, months=months),
        "rounds": rounds[-40:],
        "lag": lag_days(conflict),
        "flow_or_stock": classify_flow_stock(conflict),
        "reason": reason,
    }


def summarise(per_country: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """The cross-country headline: who is covered, how often, flow or stock."""
    covered = sorted(k for k, v in per_country.items() if v.get("covered"))
    gaps = [v["cadence"]["median_gap_days"] for v in per_country.values()
            if v.get("covered") and v["cadence"].get("median_gap_days") is not None]
    verdicts = Counter(v["flow_or_stock"]["verdict"] for v in per_country.values() if v.get("covered"))
    separable = Counter(v["reason"]["separable"] for v in per_country.values() if v.get("records"))
    regular = sorted(
        k for k, v in per_country.items()
        if v.get("covered") and (v["cadence"].get("months_with_a_report") or 0) >= 24
    )
    return {
        "countries_asked": len(per_country),
        "countries_covered": covered,
        "countries_not_covered": sorted(set(per_country) - set(covered)),
        "countries_with_24_of_36_months": regular,
        "median_of_median_gap_days": statistics.median(gaps) if gaps else None,
        "flow_or_stock_verdicts": dict(verdicts),
        "reason_separable": dict(separable),
    }


def hdx_summary(packages: list[Mapping[str, Any]]) -> dict[str, Any]:
    """What an HDX package search returned: names, orgs, countries, dates."""
    rows = []
    countries: Counter = Counter()
    for p in packages:
        groups = [str(g.get("name") or "").upper() for g in (p.get("groups") or [])
                  if isinstance(g, Mapping)]
        countries.update(groups)
        rows.append({
            "name": p.get("name"),
            "title": str(p.get("title") or "")[:160],
            "organization": (p.get("organization") or {}).get("name")
            if isinstance(p.get("organization"), Mapping) else p.get("organization"),
            "groups": groups,
            "dataset_date": p.get("dataset_date"),
            "last_modified": p.get("metadata_modified") or p.get("last_modified"),
            "update_frequency": p.get("data_update_frequency"),
            "formats": sorted({str(r.get("format") or "") for r in (p.get("resources") or [])
                               if isinstance(r, Mapping)}),
        })
    return {"n": len(rows), "countries": dict(countries.most_common()), "packages": rows}


# --------------------------------------------------------------------------
# Network (not unit-tested; the workflow runs it)
# --------------------------------------------------------------------------

class Recorder:
    def __init__(self, key: str, delay: float = 0.5) -> None:
        self.key = key
        self.delay = delay
        self.log: list[dict[str, Any]] = []

    def get(self, url: str, *, params: Mapping[str, Any] | None = None,
            auth: bool = False, timeout: float = 90.0) -> Any:
        """GET and parse JSON; record the call; return None on any failure."""
        import requests

        headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
        if auth and self.key:
            headers["Ocp-Apim-Subscription-Key"] = self.key
        entry: dict[str, Any] = {"url": url, "params": dict(params or {}), "auth": bool(auth and self.key)}
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
            entry["final_url"] = scrub(resp.url, self.key)
            entry["status"] = resp.status_code
            entry["content_type"] = resp.headers.get("Content-Type", "")
            entry["body_snippet"] = scrub(resp.text[:SNIPPET_CHARS], self.key)
            if resp.status_code != 200:
                return None
            try:
                return resp.json()
            except ValueError:
                entry["error"] = "body is not JSON"
                return None
        except Exception as exc:  # noqa: BLE001 - a probe reports, never raises
            entry["error"] = scrub(f"{type(exc).__name__}: {exc}", self.key)
            return None
        finally:
            self.log.append(entry)
            time.sleep(self.delay)


def probe_dtm(rec: Recorder, countries: list[str], *, today: dt.date, months: int,
              sample_admin_levels: int) -> dict[str, Any]:
    authed = bool(rec.key)
    start = months_back(today, months)
    out: dict[str, Any] = {"authenticated": authed, "routes": {}}
    for route in ("country-list", "operation-list"):
        payload = rec.get(f"{DTM_BASE}/{route}", auth=True)
        records = extract_records(payload) if payload is not None else []
        out["routes"][route] = {
            "records": len(records),
            "keys": interesting_keys(walk_keys(records[:200])),
            "sample": records[:5],
        }
    per_country_records: dict[str, list[dict]] = {}
    for iso3 in countries:
        payload = rec.get(
            f"{DTM_BASE}/admin0",
            params={"Admin0Pcode": iso3, "FromReportingDate": start.isoformat(),
                    "ToReportingDate": today.isoformat()},
            auth=True,
        )
        records = extract_records(payload) if payload is not None else []
        if payload is not None and not records:
            # A 200 with no records: the country is not served, or the route
            # names its filter differently. Listed so a reader can tell.
            out.setdefault("empty_admin0", []).append(iso3)
        per_country_records[iso3] = [
            r for r in records if record_iso3(r) in (None, iso3)
        ]
    for iso3 in countries[:sample_admin_levels]:
        for level in ("admin1", "admin2"):
            payload = rec.get(
                f"{DTM_BASE}/{level}",
                params={"Admin0Pcode": iso3, "FromReportingDate": start.isoformat(),
                        "ToReportingDate": today.isoformat()},
                auth=True,
            )
            records = extract_records(payload) if payload is not None else []
            out["routes"][f"{level}:{iso3}"] = {
                "records": len(records),
                "keys": dict(walk_keys(records[:500]).most_common(60)),
                "reason": reason_split(records),
                "flow_or_stock": classify_flow_stock(records),
            }
    out["per_country"] = {
        iso3: analyse_country(rows, today=today, months=months)
        for iso3, rows in per_country_records.items()
    }
    out["summary"] = summarise(out["per_country"])
    out["samples"] = {iso3: rows[:3] for iso3, rows in per_country_records.items() if rows}
    return out


def probe_hdx(rec: Recorder, countries: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    payload = rec.get(HDX_SEARCH, params={"q": "dtm displacement", "rows": 50})
    pkgs = ((payload or {}).get("result") or {}).get("results") or [] if isinstance(payload, Mapping) else []
    out["search"] = hdx_summary(pkgs)
    per_country: dict[str, Any] = {}
    for iso3 in countries:
        payload = rec.get(HDX_SEARCH, params={
            "q": "dtm",
            "fq": f"organization:{IOM_ORG} groups:{iso3.lower()}",
            "rows": 50,
        })
        pkgs = ((payload or {}).get("result") or {}).get("results") or [] if isinstance(payload, Mapping) else []
        summary = hdx_summary(pkgs)
        per_country[iso3] = {
            "n_packages": summary["n"],
            "packages": [
                {k: p[k] for k in ("name", "title", "dataset_date", "last_modified",
                                   "update_frequency", "formats")}
                for p in summary["packages"][:15]
            ],
        }
    out["per_country"] = per_country
    out["countries_with_iom_dtm_datasets"] = sorted(k for k, v in per_country.items() if v["n_packages"])
    return out


def _markdown(report: dict[str, Any]) -> str:
    lines = ["# DTM conflict displacement probe", ""]
    lines.append(f"Run: {report['today']}; key present: {report['key_present']}")
    dtm = report.get("dtm") or {}
    summ = dtm.get("summary") or {}
    lines.append("")
    lines.append("## DTM API v3 headline")
    for k, v in summ.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Per country (DTM API)")
    lines.append("| ISO3 | records | conflict | filter | reports (36m) | months | median gap d | last | lag d | flow/stock | reason |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for iso3, c in (dtm.get("per_country") or {}).items():
        cad = c["cadence"]
        lines.append(
            f"| {iso3} | {c['records']} | {c['conflict_records']} | {c['conflict_filter']} | "
            f"{cad['n_reports']} | {cad['months_with_a_report']} | {cad['median_gap_days']} | "
            f"{cad['last']} | {c['lag']['median_days']} | {c['flow_or_stock']['verdict']} | "
            f"{c['reason']['separable']} |"
        )
    lines.append("")
    lines.append("## Field evidence")
    fields: Counter = Counter()
    for c in (dtm.get("per_country") or {}).values():
        fields.update(c.get("interesting_keys") or [])
    lines.append(f"Interesting keys seen (countries): {dict(fields.most_common(40))}")
    for name, info in (dtm.get("routes") or {}).items():
        lines.append(f"- route {name}: {json.dumps(info, default=str)[:600]}")
    hdx = report.get("hdx") or {}
    lines.append("")
    lines.append("## HDX")
    lines.append(f"Search 'dtm displacement': {(hdx.get('search') or {}).get('n')} packages; "
                 f"countries {(hdx.get('search') or {}).get('countries')}")
    lines.append(f"Countries with IOM DTM datasets: {hdx.get('countries_with_iom_dtm_datasets')}")
    lines.append("")
    lines.append("## Requests")
    lines.append("| status | auth | url | params | error |")
    lines.append("|---|---|---|---|---|")
    for e in report.get("requests") or []:
        lines.append(f"| {e.get('status')} | {e.get('auth')} | {e.get('final_url') or e.get('url')} | "
                     f"{e.get('params')} | {(e.get('error') or '')[:120]} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", default="diagnostics/dtm_conflict_probe")
    parser.add_argument("--countries", default=os.getenv("DTM_PROBE_COUNTRIES") or DEFAULT_COUNTRIES)
    parser.add_argument("--months", type=int, default=36)
    parser.add_argument("--sample-admin-levels", type=int, default=3,
                        help="countries whose admin1/admin2 routes are also asked")
    args = parser.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    countries = [c.strip().upper() for c in re.split(r"[\s,]+", args.countries) if c.strip()]
    key = (os.getenv("DTM_API_KEY") or os.getenv("DTM_API_PRIMARY_KEY") or "").strip()
    today = dt.date.today()
    report: dict[str, Any] = {
        "today": today.isoformat(),
        "countries": countries,
        "n_countries": len(countries),
        "months": args.months,
        "key_present": bool(key),
    }
    if not key:
        print("::warning::no DTM_API_KEY; asking the DTM routes unauthenticated and HDX")
    rec = Recorder(key)
    try:
        report["dtm"] = probe_dtm(rec, countries, today=today, months=args.months,
                                  sample_admin_levels=args.sample_admin_levels)
    except Exception as exc:  # noqa: BLE001
        report["dtm"] = {"error": scrub(f"{type(exc).__name__}: {exc}", key)}
    try:
        report["hdx"] = probe_hdx(rec, countries)
    except Exception as exc:  # noqa: BLE001
        report["hdx"] = {"error": scrub(f"{type(exc).__name__}: {exc}", key)}
    report["requests"] = rec.log
    report["request_status_counts"] = dict(Counter(str(e.get("status")) for e in rec.log))
    text_json = scrub(json.dumps(report, indent=2, default=str), key)
    (out / "report.json").write_text(text_json, encoding="utf-8")
    try:
        text = scrub(_markdown(report), key)
    except Exception as exc:  # noqa: BLE001
        text = f"report rendering failed: {type(exc).__name__}: {exc}"
    (out / "report.md").write_text(text, encoding="utf-8")
    print(text)
    summary = os.getenv("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(text + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
