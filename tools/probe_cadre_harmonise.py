# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Read-only probe: where can Pythia read Cadre Harmonisé (CH) Phase 3+ figures?

FEWS NET stopped publishing Phase 3+ numbers for Mauritania in November 2024,
and DR Phase 3+ questions for Sahel and West African countries depend on
whoever still publishes. CH is the IPC-compatible analysis those countries run
twice a year. Two candidate routes:

1. The IPC API (api.ipcinfo.org ``/population``), which the ``ipc_api``
   connector already reads. Does it carry CH analyses for these countries,
   and how recent?
2. The HDX Cadre Harmonisé dataset, found through CKAN ``package_search``.
   Which resource, which columns, how recent?

For each target country (MRT, NER, MLI, BFA, TCD, NGA by default) the report
gives, per route: the newest analysis period, the field that carries the
Phase 3+ population, and the lag in months from that period to this month.
It writes nothing but its report, holds no DB and always exits 0. Nothing here
wires CH into resolution; the report decides a follow-up.

The transport is injected (``Transport``) so the gate logic is tested without
a network: the build sandbox cannot reach either host.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

IPC_POPULATION = "https://api.ipcinfo.org/population"
HDX_SEARCH = "https://data.humdata.org/api/3/action/package_search"
HDX_QUERY = "cadre harmonise"

DEFAULT_COUNTRIES = ("MRT", "NER", "MLI", "BFA", "TCD", "NGA")
ISO3_TO_ISO2 = {"MRT": "MR", "NER": "NE", "MLI": "ML", "BFA": "BF", "TCD": "TD", "NGA": "NG",
                "SEN": "SN", "GMB": "GM", "GNB": "GW", "GIN": "GN", "SLE": "SL", "LBR": "LR",
                "CIV": "CI", "GHA": "GH", "TGO": "TG", "BEN": "BJ", "CMR": "CM", "CPV": "CV"}

#: A route is "current" for a country when its newest analysis period ends no
#: more than this many months before the month the probe runs. Same bound as
#: question eligibility (pythia/tools/phase3_eligibility.py).
CURRENT_LAG_MONTHS = 9

#: Largest HDX resource the probe will read.
MAX_RESOURCE_BYTES = 50_000_000

#: (status, content_type, body). A network failure is status 0 with the
#: error text as the body.
Response = Tuple[int, str, bytes]
Transport = Callable[[str, Dict[str, Any]], Response]

_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}

ISO3_COLUMNS = ("adm0_pcod3", "adm0_iso3", "iso3", "country_iso3", "adm0_code_iso3")
COUNTRY_COLUMNS = ("adm0_name", "country", "country_name")
PHASE3_COLUMNS = ("phase35", "phase3_5", "phase3plus", "phase_3plus", "p3plus", "phase3_plus")
TYPE_COLUMNS = ("chtype", "ch_type", "analysis_type", "period_type")
PERIOD_COLUMNS = ("reference_label", "exercise_label", "period", "reference_period")
YEAR_COLUMNS = ("reference_year", "exercise_year", "year")


def _scrub(text: str) -> str:
    key = os.environ.get("IPC_API_KEY", "")
    out = text.replace(key, "<redacted>") if key else text
    return re.sub(r"(key=)[^&\s]+", r"\1<redacted>", out)


def default_transport(url: str, params: Dict[str, Any]) -> Response:  # pragma: no cover - network
    import requests

    try:
        resp = requests.get(url, params=params, timeout=60,
                            headers={"Accept": "application/json, text/csv, */*",
                                     "User-Agent": "Pythia-probe/1.0 (+https://fredforecaster.org)"})
        body = resp.content[: MAX_RESOURCE_BYTES + 1]
        return resp.status_code, resp.headers.get("Content-Type", ""), body
    except Exception as exc:  # noqa: BLE001 - a failure is a finding
        return 0, "", _scrub(str(exc)).encode()


def months_between(later: str, earlier: str) -> int:
    ly, lm = (int(x) for x in later[:7].split("-"))
    ey, em = (int(x) for x in earlier[:7].split("-"))
    return (ly - ey) * 12 + (lm - em)


def period_end_month(text: str, year_hint: Optional[int] = None) -> Optional[str]:
    """The LAST month a period string names, as ``YYYY-MM``.

    Accepts "Jun 2025 - Aug 2025", "June - August 2025", "2025-06-01 - 2025-08-31",
    "Oct-Dec" with a year hint. None when nothing parses.
    """
    s = str(text or "").strip()
    if not s:
        return None
    iso = re.findall(r"(\d{4})-(\d{2})(?:-\d{2})?", s)
    if iso:
        y, m = iso[-1]
        return f"{int(y):04d}-{int(m):02d}"
    tokens = re.findall(r"[A-Za-z]+|\d{4}", s)
    last_month = None
    year = None
    for tok in tokens:
        if tok.isdigit():
            year = int(tok)
        elif tok[:3].lower() in _MONTHS:
            last_month = _MONTHS[tok[:3].lower()]
    if year is None:
        year = year_hint
    if last_month is None or year is None:
        return None
    return f"{year:04d}-{last_month:02d}"


@dataclass
class RouteFinding:
    route: str
    iso3: str
    status: str  # current | stale | absent | unreadable
    newest_period: Optional[str] = None
    lag_months: Optional[int] = None
    phase3_field: Optional[str] = None
    phase3_value: Optional[float] = None
    n_records: int = 0
    note: str = ""


def _grade(route: str, iso3: str, newest: Optional[str], today_ym: str, **kw) -> RouteFinding:
    if newest is None:
        return RouteFinding(route, iso3, "absent", **kw)
    lag = months_between(today_ym, newest)
    status = "current" if lag <= CURRENT_LAG_MONTHS else "stale"
    return RouteFinding(route, iso3, status, newest, lag, **kw)


def _first_key(row: Dict[str, Any], candidates) -> Optional[str]:
    lower = {str(k).lower(): k for k in row}
    for c in candidates:
        if c in lower:
            return lower[c]
    return None


def _num(raw: Any) -> Optional[float]:
    try:
        text = str(raw).replace(",", "").strip()
        return float(text) if text else None
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- IPC API

def ipc_api_findings(records: List[Dict[str, Any]], countries, today_ym: str) -> List[RouteFinding]:
    """Grade the IPC API ``/population`` records per target country (current period only)."""
    out: List[RouteFinding] = []
    for iso3 in countries:
        iso2 = ISO3_TO_ISO2.get(iso3, "")
        mine = []
        for rec in records:
            if not isinstance(rec, dict):
                continue
            code = rec.get("country") or rec.get("country_code") or rec.get("iso2") or ""
            if isinstance(code, dict):
                code = code.get("iso2") or code.get("code") or ""
            code = str(code).strip().upper()
            if code in (iso2, iso3):
                mine.append(rec)
        best: Optional[Tuple[str, Dict[str, Any]]] = None
        for rec in mine:
            end = period_end_month(rec.get("current_period_dates") or "")
            if end and (best is None or end > best[0]):
                best = (end, rec)
        if best is None:
            out.append(_grade("ipc_api", iso3, None, today_ym, n_records=len(mine),
                              note="no record with a parseable current period" if mine else "no record"))
            continue
        rec = best[1]
        field_name = "p3plus" if rec.get("p3plus") is not None else None
        title = str(rec.get("title") or rec.get("analysis_name") or "")
        out.append(_grade("ipc_api", iso3, best[0], today_ym, n_records=len(mine),
                          phase3_field=field_name, phase3_value=_num(rec.get("p3plus")),
                          note=f"title={title[:80]!r}" if title else ""))
    return out


def probe_ipc_api(transport: Transport, countries, today: date, api_key: str) -> Dict[str, Any]:
    today_ym = f"{today.year:04d}-{today.month:02d}"
    if not api_key:
        return {"status": "skipped", "reason": "IPC_API_KEY not set",
                "findings": [RouteFinding("ipc_api", c, "unreadable", note="no key").__dict__ for c in countries]}
    status, ctype, body = transport(IPC_POPULATION, {"key": api_key, "start": today.year - 3,
                                                     "end": today.year, "format": "json"})
    if status != 200:
        return {"status": "unreadable", "reason": f"HTTP {status}: {_scrub(body[:300].decode('utf-8', 'replace'))}",
                "findings": [RouteFinding("ipc_api", c, "unreadable").__dict__ for c in countries]}
    try:
        payload = json.loads(body.decode("utf-8"))
    except ValueError:
        return {"status": "unreadable", "reason": f"body is not JSON (content type {ctype!r})",
                "findings": [RouteFinding("ipc_api", c, "unreadable").__dict__ for c in countries]}
    records = payload if isinstance(payload, list) else [payload]
    keys = sorted(records[0].keys()) if records and isinstance(records[0], dict) else []
    return {"status": "ok", "n_records": len(records), "record_keys": keys,
            "findings": [f.__dict__ for f in ipc_api_findings(records, countries, today_ym)]}


# --------------------------------------------------------------------------- HDX

def pick_hdx_resources(search_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """CSV resources of packages that look like the Cadre Harmonisé dataset."""
    out: List[Dict[str, Any]] = []
    results = ((search_payload or {}).get("result") or {}).get("results") or []
    for pkg in results:
        title = f"{pkg.get('title', '')} {pkg.get('name', '')}".lower()
        if "cadre" not in title and "harmon" not in title:
            continue
        for res in pkg.get("resources") or []:
            fmt = str(res.get("format") or "").lower()
            url = str(res.get("url") or "")
            if fmt == "csv" or url.lower().endswith(".csv"):
                out.append({"package": pkg.get("name"), "resource": res.get("name"), "url": url,
                            "last_modified": res.get("last_modified")})
    return out


def hdx_findings(csv_text: str, countries, today_ym: str, source: str) -> Tuple[List[RouteFinding], Dict[str, Any]]:
    """Grade one HDX CH CSV per target country (current analyses only where typed)."""
    sample = csv_text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        reader = csv.DictReader(io.StringIO(csv_text), dialect=dialect)
    except csv.Error:
        reader = csv.DictReader(io.StringIO(csv_text))
    rows = [r for r in reader if r]
    # HXL files carry a tag row second; drop it.
    if rows and all(str(v).startswith("#") for v in rows[0].values() if v):
        rows = rows[1:]
    meta: Dict[str, Any] = {"source": source, "n_rows": len(rows),
                            "columns": list(rows[0].keys()) if rows else []}
    if not rows:
        return [_grade("hdx", c, None, today_ym, note="empty resource") for c in countries], meta
    iso_col = _first_key(rows[0], ISO3_COLUMNS)
    name_col = _first_key(rows[0], COUNTRY_COLUMNS)
    p3_col = _first_key(rows[0], PHASE3_COLUMNS)
    type_col = _first_key(rows[0], TYPE_COLUMNS)
    period_col = _first_key(rows[0], PERIOD_COLUMNS)
    year_col = _first_key(rows[0], YEAR_COLUMNS)
    meta.update({"iso3_column": iso_col, "phase3_column": p3_col, "type_column": type_col,
                 "period_column": period_col, "year_column": year_col})
    out: List[RouteFinding] = []
    for iso3 in countries:
        newest: Optional[str] = None
        total: Dict[str, float] = {}
        n = 0
        for r in rows:
            code = str(r.get(iso_col) or "").strip().upper() if iso_col else ""
            if code != iso3:
                continue
            n += 1
            if type_col and "current" not in str(r.get(type_col) or "current").lower():
                continue
            year = None
            if year_col:
                try:
                    year = int(str(r.get(year_col) or "").strip()[:4])
                except ValueError:
                    year = None
            end = period_end_month(str(r.get(period_col) or ""), year) if period_col else (
                f"{year:04d}-12" if year else None)
            if not end:
                continue
            value = _num(r.get(p3_col)) if p3_col else None
            if value is not None:
                total[end] = total.get(end, 0.0) + value
            if newest is None or end > newest:
                newest = end
        if iso_col is None:
            out.append(_grade("hdx", iso3, None, today_ym,
                              note=f"no ISO3 column (country column {name_col!r})"))
            continue
        out.append(_grade("hdx", iso3, newest, today_ym, n_records=n, phase3_field=p3_col,
                          phase3_value=total.get(newest) if newest else None))
    return out, meta


def probe_hdx(transport: Transport, countries, today: date) -> Dict[str, Any]:
    today_ym = f"{today.year:04d}-{today.month:02d}"
    status, _ctype, body = transport(HDX_SEARCH, {"q": HDX_QUERY, "rows": 20})
    if status != 200:
        return {"status": "unreadable", "reason": f"package_search HTTP {status}: {body[:200]!r}",
                "findings": [RouteFinding("hdx", c, "unreadable").__dict__ for c in countries]}
    try:
        payload = json.loads(body.decode("utf-8"))
    except ValueError:
        return {"status": "unreadable", "reason": "package_search body is not JSON",
                "findings": [RouteFinding("hdx", c, "unreadable").__dict__ for c in countries]}
    resources = pick_hdx_resources(payload)
    if not resources:
        return {"status": "no_dataset", "reason": f"no Cadre Harmonisé CSV for query {HDX_QUERY!r}",
                "findings": [RouteFinding("hdx", c, "absent", note="no dataset").__dict__ for c in countries]}
    tried: List[Dict[str, Any]] = []
    best: Dict[str, RouteFinding] = {}
    for res in resources[:5]:
        status, _ctype, body = transport(res["url"], {})
        if status != 200:
            tried.append({**res, "status": status})
            continue
        if len(body) > MAX_RESOURCE_BYTES:
            tried.append({**res, "status": status, "note": "over size cap"})
            continue
        findings, meta = hdx_findings(body.decode("utf-8-sig", "replace"), countries, today_ym, res["url"])
        tried.append({**res, "status": status, **meta})
        for f in findings:
            cur = best.get(f.iso3)
            if cur is None or (f.newest_period or "") > (cur.newest_period or ""):
                best[f.iso3] = f
    return {"status": "ok" if best else "unreadable", "resources": tried,
            "findings": [(best.get(c) or RouteFinding("hdx", c, "unreadable")).__dict__ for c in countries]}


# --------------------------------------------------------------------------- verdict

def verdict(ipc: Dict[str, Any], hdx: Dict[str, Any], countries) -> Dict[str, str]:
    """Per country: which route, if any, carries a current CH Phase 3+ figure."""
    by_route = {"ipc_api": {f["iso3"]: f for f in ipc.get("findings", [])},
                "hdx": {f["iso3"]: f for f in hdx.get("findings", [])}}
    out: Dict[str, str] = {}
    for iso3 in countries:
        i = by_route["ipc_api"].get(iso3, {})
        h = by_route["hdx"].get(iso3, {})
        if i.get("status") == "current":
            out[iso3] = "ipc_api_current"
        elif h.get("status") == "current" and h.get("phase3_field"):
            out[iso3] = "hdx_current"
        elif "unreadable" in (i.get("status"), h.get("status")) and not (
            i.get("status") in ("stale", "absent") and h.get("status") in ("stale", "absent")
        ):
            out[iso3] = "inconclusive"
        else:
            out[iso3] = "no_current_source"
    return out


def run(transport: Transport, countries, today: date, api_key: str) -> Dict[str, Any]:
    ipc = probe_ipc_api(transport, countries, today, api_key)
    hdx = probe_hdx(transport, countries, today)
    return {"probe_date": today.isoformat(), "current_lag_months": CURRENT_LAG_MONTHS,
            "countries": list(countries), "ipc_api": ipc, "hdx": hdx,
            "verdict": verdict(ipc, hdx, countries)}


def render_markdown(report: Dict[str, Any]) -> str:
    lines = [f"# Cadre Harmonisé probe ({report['probe_date']})", "",
             f"A route is current when its newest analysis period ends within "
             f"{report['current_lag_months']} months of this month.", "",
             f"IPC API: {report['ipc_api'].get('status')} {report['ipc_api'].get('reason', '')}",
             f"HDX: {report['hdx'].get('status')} {report['hdx'].get('reason', '')}", "",
             "| Country | Verdict | Route | Status | Newest period | Lag (months) | Phase 3+ field | Value |",
             "|---|---|---|---|---|---|---|---|"]
    for iso3 in report["countries"]:
        for route in ("ipc_api", "hdx"):
            f = next((x for x in report[route].get("findings", []) if x["iso3"] == iso3), {})
            lines.append(f"| {iso3} | {report['verdict'].get(iso3)} | {route} | {f.get('status')} | "
                         f"{f.get('newest_period') or ''} | {'' if f.get('lag_months') is None else f['lag_months']} | "
                         f"{f.get('phase3_field') or ''} | {'' if f.get('phase3_value') is None else int(f['phase3_value'])} |")
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", default="diagnostics/cadre_harmonise")
    parser.add_argument("--countries", default=",".join(DEFAULT_COUNTRIES))
    args = parser.parse_args(argv)
    countries = [c.strip().upper() for c in args.countries.split(",") if c.strip()]
    try:
        report = run(default_transport, countries, date.today(), os.environ.get("IPC_API_KEY", "").strip())
    except Exception as exc:  # noqa: BLE001 - always exit 0
        report = {"error": _scrub(str(exc))}
        print(f"probe failed: {report['error']}", file=sys.stderr)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cadre_harmonise_probe.json").write_text(json.dumps(report, indent=2, default=str))
    if "verdict" in report:
        md = render_markdown(report)
        (out / "cadre_harmonise_probe.md").write_text(md)
        print(md)
        summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary:
            with open(summary, "a", encoding="utf-8") as fh:
                fh.write(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
