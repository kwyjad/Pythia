# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Probe the Montandon (Monty) STAC API and decide whether Pythia can use it.

Pythia's `ifrc_montandon` connector does not talk to Montandon — it polls the
IFRC GO operational API. Montandon is a separate STAC-based Global Crisis Data
Bank that aggregates GDACS / EM-DAT / GLIDE / USGS / IBTrACS / PDC for events
and EM-DAT / IDMC / IFRC field reports / DesInventar for impacts. It has never
been evaluated. See docs/montandon_assessment.md.

This answers eight gates, ordered so the cheapest disqualifier runs first, and
writes both a machine-readable verdict and a human summary.

    G0  Does a production endpoint exist and respond?
    G1  Can impact records be filtered by upstream source?   <- KILL GATE
    G2  Is there a typed impact taxonomy with units?
    G3  How much country-month coverage does it add over our IFRC ingest?
    G4  How long after an event does an impact figure appear?
    G5  Where both exist, do Montandon and IFRC agree?
    G6  Are figures revised, and is the revision visible?
    G7  How far back does it go?

G1 is the kill gate and the reason this script exists in this shape. Montandon
ingests GDACS, whose impact figures are MODELLED POPULATION EXPOSURE (hazard
footprint x population) — orders of magnitude larger than a reported "people
affected" figure and a different quantity in kind. Ingesting Montandon without
being able to exclude those records would import that blend into a single
column, which is exactly the failure Pythia has just finished removing from its
own base rates. If G1 fails, the answer is no, regardless of every other gate.

G4 decides resolution-vs-base-rates. Pythia resolves against the previous
complete month, so an impact figure that lands 90 days after the event is
useless for scoring even if it is excellent history. Montandon leans on EM-DAT,
which is slow, so this gate failing is a likely and legitimate outcome — it
means "use it for base rates" rather than "reject it".

Usage
-----
    python -m tools.probe_montandon_stac \\
        [--endpoint https://montandon-eoapi.ifrc.org/stac] \\
        [--db data/resolver.duckdb] \\
        [--months-back 36] [--hazards FL,DR,TC] \\
        [--out-dir diagnostics]

Network access to *.ifrc.org is required, and is NOT available from the Claude
Code sandbox (the egress proxy denies it) — run this in CI or locally.
"""

from __future__ import annotations

import argparse
import json
import os
import logging
import statistics
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

LOG = logging.getLogger("probe_montandon")

PROD_ENDPOINT = "https://montandon-eoapi.ifrc.org/stac"
STAGE_ENDPOINT = "https://montandon-eoapi-stage.ifrc.org/stac"

# Upstream sources whose impact figures are REPORTED impact. Only these belong
# in a PA_REPORTED series.
REPORTED_IMPACT_SOURCES = {"emdat", "desinventar", "ifrc", "idmc", "glide"}
# Upstream sources whose figures are modelled exposure. These must never enter
# a PA series — see the module docstring.
MODELLED_EXPOSURE_SOURCES = {"gdacs", "pdc", "usgs", "ibtracs"}

# G4: our resolution cutoff is the previous complete month, so a figure that
# arrives later than roughly a month after month-end cannot score a horizon.
RESOLUTION_LATENCY_DAYS = 45
# G3: below this multiple of our current IFRC coverage, a new connector is not
# worth its maintenance cost.
COVERAGE_MULTIPLE_MIN = 1.3
COVERAGE_MULTIPLE_GOOD = 2.0

HTTP_TIMEOUT = 60


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


class Stac:
    """Minimal STAC client. Deliberately plain `requests`.

    `pystac_client` would be nicer, but a spike should not add a hard
    dependency to python_library_requirements before we know the answer.
    """

    def __init__(self, root: str, token: str = "", scheme: str = "bearer") -> None:
        self.root = root.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update(
            {"Accept": "application/json", "User-Agent": "pythia-montandon-probe/1.0"}
        )
        # The service's auth scheme is undocumented in anything we can reach,
        # so make it selectable rather than guessing one and reporting the
        # resulting 401 as if it were the service's fault.
        if token:
            if scheme == "basic":
                self.session.headers["Authorization"] = f"Basic {token}"
            elif scheme == "token":
                self.session.headers["Authorization"] = f"Token {token}"
            elif scheme == "header":
                self.session.headers["X-API-Key"] = token
            else:
                self.session.headers["Authorization"] = f"Bearer {token}"

    def get(self, path: str, **params: Any) -> tuple[int, Any]:
        url = path if path.startswith("http") else f"{self.root}/{path.lstrip('/')}"
        try:
            resp = self.session.get(url, params=params or None, timeout=HTTP_TIMEOUT)
        except requests.RequestException as exc:
            return 0, {"error": f"{type(exc).__name__}: {exc}"}
        try:
            return resp.status_code, resp.json()
        except ValueError:
            return resp.status_code, {"error": "non-JSON response", "text": resp.text[:500]}

    def post(self, path: str, body: dict) -> tuple[int, Any]:
        url = f"{self.root}/{path.lstrip('/')}"
        try:
            resp = self.session.post(url, json=body, timeout=HTTP_TIMEOUT)
        except requests.RequestException as exc:
            return 0, {"error": f"{type(exc).__name__}: {exc}"}
        try:
            return resp.status_code, resp.json()
        except ValueError:
            return resp.status_code, {"error": "non-JSON response", "text": resp.text[:500]}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _walk(obj: Any, depth: int = 0):
    """Yield (dotted_key, value) for every scalar in a nested dict/list."""
    if depth > 6:
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, (dict, list)):
                for sub_key, sub_val in _walk(value, depth + 1):
                    yield f"{key}.{sub_key}", sub_val
            else:
                yield key, value
    elif isinstance(obj, list):
        for item in obj[:20]:
            for sub_key, sub_val in _walk(item, depth + 1):
                yield sub_key, sub_val


def _classify_source(text: str) -> str:
    low = (text or "").lower()
    for name in MODELLED_EXPOSURE_SOURCES:
        if name in low:
            return "modelled_exposure"
    for name in REPORTED_IMPACT_SOURCES:
        if name in low:
            return "reported_impact"
    return "unknown"


def _parse_date(value: Any) -> date | None:
    if not value:
        return None
    text = str(value)[:10]
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------


def _classify_probe(status: int, body: Any) -> str:
    """Turn an HTTP outcome into the distinction that changes what we do next.

    "Did not return 200" is not one condition. A DNS failure, a 401 and a 404
    call for three different next steps — get network access, get credentials,
    fix the path — and collapsing them into "unreachable" produces a verdict
    that is not merely vague but wrong.
    """
    if status == 0:
        return "unreachable"
    if status == 200:
        return "ok"
    if status in (401, 403):
        return "auth_required"
    if status == 404:
        return "not_found"
    return "http_error"


def gate_g0(endpoint: str, token: str = "", scheme: str = "bearer") -> dict:
    """Does an endpoint exist, and can we actually get in?"""
    result: dict[str, Any] = {"gate": "G0", "question": "endpoint responds and admits us"}
    result["authenticated"] = bool(token)
    probes = {}
    for label, root in (("requested", endpoint), ("prod", PROD_ENDPOINT), ("stage", STAGE_ENDPOINT)):
        status, body = Stac(root, token=token, scheme=scheme).get("/")
        kind = _classify_probe(status, body)
        # Keep a snippet of whatever the server said. On a 401 this is the only
        # place the reason can come from, and it is what tells us which auth
        # scheme the service wants.
        detail = None
        if isinstance(body, dict):
            detail = body.get("error") or body.get("detail") or body.get("message")
            if detail is None and kind != "ok":
                detail = json.dumps(body)[:200]
        probes[label] = {
            "url": root,
            "status": status,
            "kind": kind,
            "title": (body or {}).get("title") if isinstance(body, dict) else None,
            "detail": detail,
        }
    result["probes"] = probes

    kinds = {label: p["kind"] for label, p in probes.items()}
    any_ok = "ok" in kinds.values()
    any_auth = "auth_required" in kinds.values()

    result["pass"] = bool(any_ok)
    result["blocked_by_auth"] = bool(any_auth and not any_ok)

    if probes["prod"]["kind"] == "ok":
        result["verdict"] = "production endpoint responds"
    elif any_ok:
        result["verdict"] = (
            "only the staging endpoint responds. Usable for a base-rate "
            "evaluation; must not become a production resolution dependency."
        )
    elif any_auth:
        result["verdict"] = (
            "endpoints are UP but require credentials (HTTP "
            f"{probes['prod']['status'] or probes['stage']['status']}). This is "
            "an access problem, not an availability one — supply a token via "
            "MONTANDON_TOKEN and re-run."
        )
    elif "not_found" in kinds.values():
        result["verdict"] = (
            "host responds but the STAC root is not at this path — check the "
            "endpoint URL before concluding anything about the service"
        )
    else:
        result["verdict"] = "no endpoint reachable — every later gate is unanswerable"
    return result


def gate_collections(stac: Stac) -> dict:
    status, body = stac.get("/collections")
    collections = []
    if status == 200 and isinstance(body, dict):
        for coll in body.get("collections", []) or []:
            collections.append(
                {
                    "id": coll.get("id"),
                    "title": coll.get("title"),
                    "description": (coll.get("description") or "")[:300],
                    "extent_temporal": (coll.get("extent", {}) or {})
                    .get("temporal", {})
                    .get("interval"),
                    "likely_kind": _classify_source(
                        f"{coll.get('id')} {coll.get('title')} {coll.get('description')}"
                    ),
                }
            )
    return {
        "status": status,
        "n_collections": len(collections),
        "collections": collections,
    }


def _search(stac: Stac, *, collections: list[str] | None = None, limit: int = 50,
            datetime_range: str | None = None, extra: dict | None = None) -> tuple[int, list[dict]]:
    body: dict[str, Any] = {"limit": limit}
    if collections:
        body["collections"] = collections
    if datetime_range:
        body["datetime"] = datetime_range
    if extra:
        body.update(extra)
    status, payload = stac.post("/search", body)
    if status != 200:
        # Some deployments only expose GET /search.
        params: dict[str, Any] = {"limit": limit}
        if collections:
            params["collections"] = ",".join(collections)
        if datetime_range:
            params["datetime"] = datetime_range
        status, payload = stac.get("/search", **params)
    features = []
    if isinstance(payload, dict):
        features = payload.get("features") or []
    return status, features


def gate_g1(stac: Stac, impact_collections: list[str]) -> dict:
    """KILL GATE: can impact records be filtered by upstream source?"""
    result: dict[str, Any] = {
        "gate": "G1",
        "question": "impact records are attributable to an upstream source",
        "kill_gate": True,
    }
    status, features = _search(stac, collections=impact_collections or None, limit=50)
    result["search_status"] = status
    result["n_sampled"] = len(features)

    source_fields: dict[str, set[str]] = {}
    kinds: dict[str, int] = {}
    for feat in features:
        for key, value in _walk(feat):
            if not isinstance(value, str):
                continue
            low_key = key.lower()
            if any(tok in low_key for tok in ("source", "provider", "attribution", "collection")):
                source_fields.setdefault(key, set()).add(value[:80])
        kind = _classify_source(json.dumps(feat)[:4000])
        kinds[kind] = kinds.get(kind, 0) + 1

    result["candidate_source_fields"] = {
        key: sorted(vals)[:10] for key, vals in sorted(source_fields.items())[:25]
    }
    result["sampled_kind_counts"] = kinds
    result["pass"] = bool(source_fields) and kinds.get("unknown", 0) < max(1, len(features))
    result["verdict"] = (
        "impact records carry an attributable source field — a reported-impact "
        "filter is constructible"
        if result["pass"]
        else "NO usable source attribution found. Montandon ingests GDACS, whose "
        "figures are modelled exposure; without a filter, consuming it would "
        "blend exposure and reported impact in one column. Reject."
    )
    return result


def gate_g2(stac: Stac, impact_collections: list[str]) -> dict:
    """Is there a typed impact taxonomy with units?"""
    result: dict[str, Any] = {"gate": "G2", "question": "typed impact taxonomy with units"}
    status, features = _search(stac, collections=impact_collections or None, limit=100)
    triples: dict[str, int] = {}
    for feat in features:
        props = feat.get("properties", {}) if isinstance(feat, dict) else {}
        category = props.get("monty:impact_category") or props.get("impact_category")
        itype = props.get("monty:impact_type") or props.get("impact_type")
        unit = props.get("monty:unit") or props.get("unit")
        if category or itype:
            triples[f"{category}|{itype}|{unit}"] = triples.get(f"{category}|{itype}|{unit}", 0) + 1
    result["search_status"] = status
    result["impact_triples"] = dict(sorted(triples.items(), key=lambda kv: -kv[1])[:40])
    result["pass"] = bool(triples)
    result["verdict"] = (
        "impact records are typed; map the whitelist onto affected/displaced/fatalities"
        if triples
        else "no typed impact category/type found on sampled items — a metric "
        "mapping cannot be built safely"
    )
    return result


def gate_g4(stac: Stac, impact_collections: list[str], months_back: int) -> dict:
    """How long after an event does an impact figure appear?"""
    result: dict[str, Any] = {
        "gate": "G4",
        "question": "impact latency vs the resolution cutoff",
        "threshold_days": RESOLUTION_LATENCY_DAYS,
    }
    since = (datetime.now(timezone.utc) - timedelta(days=30 * months_back)).date()
    status, features = _search(
        stac,
        collections=impact_collections or None,
        limit=200,
        datetime_range=f"{since.isoformat()}T00:00:00Z/..",
    )
    lags: list[int] = []
    for feat in features:
        props = feat.get("properties", {}) if isinstance(feat, dict) else {}
        event_dt = _parse_date(props.get("start_datetime") or props.get("datetime"))
        created = _parse_date(props.get("created") or props.get("published") or props.get("updated"))
        if event_dt and created and created >= event_dt:
            lags.append((created - event_dt).days)
    result["search_status"] = status
    result["n_with_both_dates"] = len(lags)
    if lags:
        lags.sort()
        result["median_lag_days"] = statistics.median(lags)
        result["p90_lag_days"] = lags[min(len(lags) - 1, int(len(lags) * 0.9))]
        result["pass"] = result["median_lag_days"] <= RESOLUTION_LATENCY_DAYS
        result["verdict"] = (
            "fast enough to resolve horizons"
            if result["pass"]
            else "too slow to resolve a horizon — usable for BASE RATES only, "
            "with IFRC GO remaining the resolver"
        )
    else:
        result["pass"] = None
        result["verdict"] = (
            "could not measure: sampled items lacked both an event date and a "
            "created/published date"
        )
    return result


def gate_g7(collections_report: dict) -> dict:
    """How far back does it go?"""
    result: dict[str, Any] = {"gate": "G7", "question": "historical depth", "threshold_years": 10}
    earliest = None
    for coll in collections_report.get("collections", []):
        interval = coll.get("extent_temporal") or []
        for pair in interval:
            if isinstance(pair, list) and pair and pair[0]:
                parsed = _parse_date(pair[0])
                if parsed and (earliest is None or parsed < earliest):
                    earliest = parsed
    result["earliest_item"] = earliest.isoformat() if earliest else None
    if earliest:
        years = (date.today() - earliest).days / 365.25
        result["years_available"] = round(years, 1)
        result["pass"] = years >= 10
        result["verdict"] = (
            f"{years:.0f} years of history"
            if result["pass"]
            else f"only {years:.0f} years — thin for base rates"
        )
    else:
        result["pass"] = None
        result["verdict"] = "no temporal extent advertised on any collection"
    return result


def gate_g3_g5_local(db_path: Path | None, months_back: int, hazards: list[str]) -> dict:
    """Our side of the coverage and agreement comparison.

    Reports what OUR facts_resolved currently holds, so the Montandon numbers
    have something to be compared against. Without a DB this reports
    `available: false` rather than an invented baseline.
    """
    out: dict[str, Any] = {
        "gate": "G3/G5 baseline",
        "question": "what our IFRC ingest currently covers",
        "coverage_multiple_min": COVERAGE_MULTIPLE_MIN,
        "coverage_multiple_good": COVERAGE_MULTIPLE_GOOD,
    }
    if not db_path or not db_path.exists():
        out["available"] = False
        out["note"] = (
            "no resolver DB supplied — run with --db to get the coverage "
            "baseline the G3 gate compares against"
        )
        out["verdict"] = "no DB supplied — baseline not computed"
        return out
    try:
        import duckdb
    except ImportError:
        out["available"] = False
        out["note"] = "duckdb not installed"
        out["verdict"] = "duckdb not installed — baseline not computed"
        return out

    since_ym = (date.today() - timedelta(days=30 * months_back)).strftime("%Y-%m")
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        hz_list = ",".join(f"'{h}'" for h in hazards)
        rows = con.execute(
            f"""
            SELECT hazard_code,
                   COUNT(DISTINCT iso3 || '|' || ym) AS country_months,
                   COUNT(DISTINCT iso3)              AS countries,
                   MIN(ym) AS first_ym, MAX(ym) AS last_ym
            FROM facts_resolved
            WHERE hazard_code IN ({hz_list})
              AND lower(metric) IN ('affected','people_affected','pa','displaced')
              AND lower(COALESCE(publisher,'')) IN ('ifrc','ifrc_go','ifrc_montandon')
              AND ym >= ?
            GROUP BY hazard_code ORDER BY hazard_code
            """,
            [since_ym],
        ).fetchall()
    except Exception as exc:  # pragma: no cover - defensive
        out["available"] = False
        out["note"] = f"query failed: {type(exc).__name__}: {exc}"
        out["verdict"] = f"baseline query failed: {type(exc).__name__}"
        return out
    finally:
        con.close()

    out["available"] = True
    out["since_ym"] = since_ym
    out["ifrc_country_months_by_hazard"] = {
        r[0]: {"country_months": r[1], "countries": r[2], "first_ym": r[3], "last_ym": r[4]}
        for r in rows
    }
    total = sum(r[1] for r in rows)
    out["ifrc_country_months_total"] = total
    out["verdict"] = (
        f"{total} IFRC PA country-months since {since_ym} across "
        f"{', '.join(r[0] for r in rows) or 'no hazards'}"
    )
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def render_markdown(report: dict) -> str:
    lines = ["# Montandon STAC probe", ""]
    lines.append(f"- Endpoint requested: `{report['endpoint']}`")
    lines.append(f"- Run at: {report['generated_at']}")
    lines.append("")

    lines.append("## Verdict")
    lines.append("")
    lines.append(report["overall"]["verdict"])
    lines.append("")
    lines.append(f"**Recommended outcome: {report['overall']['outcome']}**")
    lines.append("")

    lines.append("## Gates")
    lines.append("")
    lines.append("| Gate | Question | Result | Notes |")
    lines.append("|---|---|---|---|")
    for gate in report["gates"]:
        passed = gate.get("pass")
        mark = "PASS" if passed is True else ("FAIL" if passed is False else "n/a")
        lines.append(
            f"| {gate.get('gate')} | {gate.get('question')} | {mark} | "
            f"{gate.get('verdict', '').replace('|', '/')} |"
        )
    lines.append("")

    # What each endpoint actually said. Without this the summary cannot
    # distinguish "the service is down" from "we are not authorised", which
    # are entirely different problems with entirely different next steps.
    g0 = next((g for g in report["gates"] if g.get("gate") == "G0"), None)
    if g0 and g0.get("probes"):
        lines.append("### Endpoint probes")
        lines.append("")
        lines.append(f"_Token supplied: {'yes' if g0.get('authenticated') else 'no'}_")
        lines.append("")
        lines.append("| Endpoint | URL | HTTP | Interpretation | Server said |")
        lines.append("|---|---|---|---|---|")
        for label, probe in g0["probes"].items():
            status = probe.get("status")
            lines.append(
                f"| {label} | `{probe.get('url')}` | "
                f"{status if status else 'no response'} | {probe.get('kind')} | "
                f"{str(probe.get('detail') or '—').replace('|', '/')[:120]} |"
            )
        lines.append("")

    baseline = next(
        (g for g in report["gates"] if str(g.get("gate", "")).startswith("G3/G5")), None
    )
    if baseline:
        lines.append("### Our current IFRC coverage (the G3 comparison baseline)")
        lines.append("")
        if not baseline.get("available"):
            lines.append(f"Unavailable — {baseline.get('note', 'no reason recorded')}.")
        else:
            lines.append(
                f"IFRC PA country-months since {baseline.get('since_ym')}: "
                f"**{baseline.get('ifrc_country_months_total')}**"
            )
            lines.append("")
            lines.append("| Hazard | Country-months | Countries | First | Last |")
            lines.append("|---|---|---|---|---|")
            for hz, stats in (baseline.get("ifrc_country_months_by_hazard") or {}).items():
                lines.append(
                    f"| {hz} | {stats.get('country_months')} | {stats.get('countries')} | "
                    f"{stats.get('first_ym')} | {stats.get('last_ym')} |"
                )
            lines.append("")
            lines.append(
                f"Montandon must beat this by ≥{COVERAGE_MULTIPLE_MIN}× to be worth a "
                f"connector, and ≥{COVERAGE_MULTIPLE_GOOD}× to be a clear win."
            )
        lines.append("")

    colls = report.get("collections", {})
    if colls.get("collections"):
        lines.append("## Collections")
        lines.append("")
        lines.append("| id | likely kind | temporal extent |")
        lines.append("|---|---|---|")
        for coll in colls["collections"]:
            lines.append(
                f"| `{coll.get('id')}` | {coll.get('likely_kind')} | "
                f"{coll.get('extent_temporal')} |"
            )
        lines.append("")

    lines.append("## What to do with this")
    lines.append("")
    lines.append(
        "G1 is the kill gate. Montandon ingests GDACS, whose impact figures are "
        "modelled population exposure rather than reported impact. If those "
        "records cannot be excluded, consuming Montandon would blend two "
        "different quantities into one column — the failure Pythia removed from "
        "its own base rates in August 2026. See `docs/montandon_assessment.md`."
    )
    lines.append("")
    lines.append(
        "G4 decides resolution versus base rates. Pythia resolves against the "
        "previous complete month, so an impact figure arriving 90 days after an "
        "event cannot score a horizon even if it is excellent history."
    )
    return "\n".join(lines) + "\n"


def decide(gates: list[dict]) -> dict:
    by_id = {g.get("gate"): g for g in gates}
    g0, g1 = by_id.get("G0", {}), by_id.get("G1", {})
    g2, g4 = by_id.get("G2", {}), by_id.get("G4", {})

    if g0.get("blocked_by_auth"):
        who = "with the token supplied" if g0.get("authenticated") else "and we have no token"
        return {
            "outcome": "BLOCKED — credentials required",
            "verdict": (
                f"The Montandon STAC endpoints are UP but returned 401/403 {who}. "
                "This is an access problem, not an availability one, and nothing "
                "below G0 is answerable until it is resolved. Next step is to ask "
                "IFRC for API access and the auth scheme — not to draw any "
                "conclusion about the data itself."
            ),
        }
    if g0.get("pass") is False:
        return {
            "outcome": "UNKNOWN — endpoint unreachable",
            "verdict": (
                "No Montandon endpoint responded, so no gate below G0 was "
                "answerable. Re-run from an environment with egress to "
                "*.ifrc.org before drawing any conclusion."
            ),
        }
    if g1.get("pass") is False:
        return {
            "outcome": "D — reject",
            "verdict": (
                "Impact records cannot be attributed to an upstream source. "
                "Montandon ingests GDACS modelled exposure alongside reported "
                "impact, so without a filter it would import both into one "
                "column. Keep IFRC GO and send the improvement list instead."
            ),
        }
    if g2.get("pass") is False:
        return {
            "outcome": "D — reject (for now)",
            "verdict": (
                "No typed impact taxonomy was found on sampled items, so "
                "records cannot be mapped onto affected/displaced/fatalities "
                "safely."
            ),
        }
    if g4.get("pass") is False:
        return {
            "outcome": "C — base rates only",
            "verdict": (
                "Montandon is attributable and typed but too slow to resolve a "
                "horizon. Use it for history and base rates; IFRC GO stays the "
                "resolver. This is only safe if the G5 agreement check confirms "
                "the two are the same definitional class."
            ),
        }
    return {
        "outcome": "A/B — candidate for PA_REPORTED",
        "verdict": (
            "Attributable, typed and timely. Compare the G3 coverage delta "
            "against our IFRC baseline to choose between replacing IFRC GO "
            "(outcome A) and supplementing it within PA_REPORTED (outcome B)."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--endpoint", default=PROD_ENDPOINT)
    parser.add_argument("--db", default="", help="resolver.duckdb for the G3 baseline")
    parser.add_argument("--months-back", type=int, default=36)
    parser.add_argument("--hazards", default="FL,DR,TC")
    parser.add_argument("--out-dir", default="diagnostics")
    parser.add_argument(
        "--auth-scheme",
        default="bearer",
        choices=["bearer", "token", "basic", "header"],
        help="How to present MONTANDON_TOKEN (the service's scheme is undocumented)",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    # Never a CLI flag: a token on the command line lands in the process table
    # and in CI logs.
    token = os.environ.get("MONTANDON_TOKEN", "").strip()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    hazards = [h.strip().upper() for h in args.hazards.split(",") if h.strip()]
    gates: list[dict] = []

    LOG.info("G0: probing endpoints (token supplied: %s)", "yes" if token else "no")
    g0 = gate_g0(args.endpoint, token=token, scheme=args.auth_scheme)
    gates.append(g0)

    endpoint = args.endpoint
    probes = g0.get("probes", {})
    if probes.get("requested", {}).get("status") != 200:
        for label in ("prod", "stage"):
            if probes.get(label, {}).get("status") == 200:
                endpoint = probes[label]["url"]
                LOG.warning("falling back to the %s endpoint: %s", label, endpoint)
                break

    stac = Stac(endpoint, token=token, scheme=args.auth_scheme)
    collections_report: dict = {"status": None, "n_collections": 0, "collections": []}
    impact_collections: list[str] = []

    if g0.get("pass"):
        LOG.info("listing collections")
        collections_report = gate_collections(stac)
        impact_collections = [
            c["id"]
            for c in collections_report["collections"]
            if c.get("id") and "impact" in str(c["id"]).lower()
        ]
        LOG.info(
            "found %s collections (%s look like impact collections)",
            collections_report["n_collections"],
            len(impact_collections),
        )

        LOG.info("G1: source attribution (kill gate)")
        gates.append(gate_g1(stac, impact_collections))
        LOG.info("G2: impact taxonomy")
        gates.append(gate_g2(stac, impact_collections))
        LOG.info("G4: impact latency")
        gates.append(gate_g4(stac, impact_collections, args.months_back))
        LOG.info("G7: historical depth")
        gates.append(gate_g7(collections_report))
    else:
        LOG.error("no endpoint reachable; skipping G1-G7")

    LOG.info("G3/G5 baseline from our own DB")
    gates.append(gate_g3_g5_local(Path(args.db) if args.db else None, args.months_back, hazards))

    report = {
        "endpoint": endpoint,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "months_back": args.months_back,
        "hazards": hazards,
        "collections": collections_report,
        "gates": gates,
        "overall": decide(gates),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "montandon_probe.json").write_text(json.dumps(report, indent=2, default=str))
    (out_dir / "montandon_probe.md").write_text(render_markdown(report))

    LOG.info("outcome: %s", report["overall"]["outcome"])
    LOG.info("wrote %s and %s", out_dir / "montandon_probe.json", out_dir / "montandon_probe.md")

    # Always exit 0: this is a fact-finding spike, and "the endpoint is down"
    # is a finding, not a build failure.
    return 0


if __name__ == "__main__":
    sys.exit(main())
