# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""ReliefWeb silence check for non-triggered country-months.

Before a country-month resolves to zero, a keyword sweep of the
ReliefWeb API must confirm that nobody reported the hazard either.
Two queries per country-month, both recorded verbatim in the evidence:

1. taxonomy sweep — reports tagged with the hazard's ReliefWeb disaster
   types (``cyclone.reliefweb_sweep.disaster_types``);
2. keyword sweep — full-text search over title/body for the hazard's
   keyword list (only run when the taxonomy sweep is silent).

Both queries filter on ``country.iso3`` (any country tag, deliberately
broader than ``primary_country``) and a publication window of month
start .. month end + ``publication_pad_days``.

Fail-closed: an API failure makes the sweep INCONCLUSIVE, never silent —
a zero is only ever written on positive evidence of silence.

Uses the free public ReliefWeb API; ``RELIEFWEB_APPNAME`` (env) is sent
as the required ``appname`` parameter.  No key material is involved.
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import time
from datetime import timedelta
from typing import Any, Callable

import requests

from resolver.hazard_resolution.detect import month_bounds
from resolver.hazard_resolution.rulebook import Rulebook

LOG = logging.getLogger(__name__)

_DEFAULT_APPNAME = "pythia-resolution-machine"

# Injectable POST seam for tests: (url, json_payload, params, timeout) -> dict
PostFn = Callable[[str, dict, dict, float], dict]


def _utcnow_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def appname() -> str:
    """The required ReliefWeb ``appname`` parameter, from the environment.

    Public because :mod:`reliefweb_docs` sends the same parameter to the
    same API — one identity for the machine's whole ReliefWeb footprint.
    """

    configured = os.getenv("RELIEFWEB_APPNAME", "").strip()
    if not configured:
        LOG.warning(
            "[reliefweb_sweep] RELIEFWEB_APPNAME not set — using default '%s'",
            _DEFAULT_APPNAME,
        )
        return _DEFAULT_APPNAME
    return configured


def default_post(url: str, payload: dict, params: dict, timeout: float) -> dict:
    """POST with one 429 retry, mirroring horizon_scanner/reliefweb.py.

    Shared with :mod:`reliefweb_docs`: both talk to the same API and must
    back off the same way.
    """
    for attempt in range(2):
        try:
            resp = requests.post(
                url, json=payload, params=params, timeout=timeout,
                headers={"Accept": "application/json"},
            )
            if resp.status_code == 429 and attempt == 0:
                LOG.warning("[reliefweb_sweep] 429 — retrying after 2 s …")
                time.sleep(2)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            if attempt == 0:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status == 429:
                    LOG.warning("[reliefweb_sweep] 429 — retrying after 2 s …")
                    time.sleep(2)
                    continue
            raise
    raise requests.RequestException("ReliefWeb sweep POST exhausted retries")


def _quote_term(term: str) -> str:
    term = term.strip()
    return f'"{term}"' if " " in term else term


def _base_filter(iso3: str, ym: str, pad_days: int) -> dict:
    start, end = month_bounds(ym)
    window_end = end + timedelta(days=pad_days)
    return {
        "operator": "AND",
        "conditions": [
            {"field": "country.iso3", "value": iso3.upper()},
            {
                "field": "date.created",
                "value": {
                    "from": f"{start.isoformat()}T00:00:00+00:00",
                    "to": f"{window_end.isoformat()}T23:59:59+00:00",
                },
            },
        ],
    }


def sweep_country_month(
    iso3: str,
    ym: str,
    rulebook: Rulebook,
    *,
    hazard_key: str = "cyclone",
    post: PostFn | None = None,
) -> dict[str, Any]:
    """Run the silence sweep for one country-month.

    Returns an evidence dict::

        {
          "iso3", "ym", "hazard_key",
          "silent": bool,          # positively confirmed silence
          "inconclusive": bool,    # API failure — never treat as silent
          "total_hits": int,
          "queries": [ {kind, url, payload, hits, sample:[{title,url,date}]} ],
          "retrieved_at": iso, "error": str|None,
        }
    """
    cfg = f"{hazard_key}.reliefweb_sweep"
    disaster_types = list(rulebook.get(f"{cfg}.disaster_types"))
    keywords = [str(k) for k in rulebook.get(f"{cfg}.keywords")]
    pad_days = int(rulebook.get(f"{cfg}.publication_pad_days"))
    max_hits = int(rulebook.get(f"{cfg}.max_hits_for_silence"))
    sample_size = int(rulebook.get(f"{cfg}.sample_size"))
    timeout = float(rulebook.get(f"{cfg}.request_timeout_sec"))
    api_url = str(rulebook.get("reliefweb.api_base_url")).rstrip("/") + "/reports"

    post = post or default_post
    params = {"appname": appname()}
    fields = {"include": ["title", "url", "date.created", "disaster_type.name"]}

    evidence: dict[str, Any] = {
        "iso3": iso3.upper(),
        "ym": ym,
        "hazard_key": hazard_key,
        "silent": False,
        "inconclusive": False,
        "total_hits": 0,
        "queries": [],
        "retrieved_at": _utcnow_iso(),
        "error": None,
    }

    def _run(kind: str, payload: dict) -> int | None:
        try:
            data = post(api_url, payload, params, timeout)
        except Exception as exc:  # requests errors, JSON errors
            LOG.warning(
                "[reliefweb_sweep] %s query failed for %s %s: %s", kind, iso3, ym, exc
            )
            evidence["inconclusive"] = True
            evidence["error"] = f"{kind}: {exc}"
            evidence["queries"].append(
                {"kind": kind, "url": api_url, "payload": payload, "hits": None,
                 "sample": [], "error": str(exc)}
            )
            return None
        total = int((data.get("totalCount") or 0))
        sample = []
        for item in (data.get("data") or [])[:sample_size]:
            f = item.get("fields", {})
            date_obj = f.get("date") or {}
            sample.append(
                {
                    "title": f.get("title", ""),
                    "url": f.get("url", ""),
                    "date": date_obj.get("created", "") if isinstance(date_obj, dict) else "",
                }
            )
        evidence["queries"].append(
            {"kind": kind, "url": api_url, "payload": payload, "hits": total,
             "sample": sample}
        )
        return total

    # Query 1: taxonomy sweep.
    filt = _base_filter(iso3, ym, pad_days)
    taxonomy_payload = {
        "filter": {
            "operator": "AND",
            "conditions": filt["conditions"]
            + [{"field": "disaster_type.name", "value": disaster_types}],
        },
        "fields": fields,
        "sort": ["date.created:desc"],
        "limit": sample_size,
    }
    taxonomy_hits = _run("disaster_type", taxonomy_payload)
    if taxonomy_hits is None:
        return evidence
    evidence["total_hits"] = taxonomy_hits

    # Query 2: keyword sweep (only when taxonomy is silent).
    if taxonomy_hits <= max_hits:
        keyword_payload = {
            "filter": filt,
            "query": {
                "value": " OR ".join(_quote_term(k) for k in keywords),
                "fields": ["title", "body"],
                "operator": "OR",
            },
            "fields": fields,
            "sort": ["date.created:desc"],
            "limit": sample_size,
        }
        keyword_hits = _run("keywords", keyword_payload)
        if keyword_hits is None:
            return evidence
        evidence["total_hits"] += keyword_hits

    evidence["silent"] = evidence["total_hits"] <= max_hits
    return evidence
