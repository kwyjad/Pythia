# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Why does EM-DAT answer HTTP 500? One key, two query shapes, one verdict.

``api.emdat.be`` has returned 500 on every call the PA resolution machine
has ever made, and ``haz_raw_emdat`` holds no rows, so the ladder's top rung
has never existed. The connector kept only the status line, and only CI
holds ``EMDAT_API_KEY``, so the question of WHY could be settled nowhere.

Three explanations fit, with three different fixes:

**Query shape.** The connector passes ``from``/``to``/``classif`` as
top-level arguments of ``public_emdat`` and selects ``data`` with no field
set. EM-DAT's published examples nest the filters under ``filters:`` and
select columns inside ``data``. A GraphQL server that rejects the shape
should say so with a 400 and an ``errors`` list; some answer 500. Fix: the
query text in ``resolver/hazard_resolution/emdat.py``.

**Account.** An expired or unregistered key. The body says so in words.
Fix: the secret.

**Upstream.** The endpoint is down for everyone. Fix: wait, and the cache
fallback the connector already has.

This sends the connector's exact query, then the ``filters:`` variant, then
a schema introspection, and records status and body for each. Read-only,
never writes to the database, always exits 0 — "the endpoint answers 500"
is a finding, not a build failure.

    python -m scripts.ci.diagnose_emdat
    python -m scripts.ci.diagnose_emdat --out diagnostics/emdat_diagnostic.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

API_URL = "https://api.emdat.be/v1"
API_KEY_ENV = "EMDAT_API_KEY"
TIMEOUT = 60
BODY_CHARS = 2000

#: The connector's own query, verbatim from resolver/hazard_resolution/emdat.py.
QUERY_CONNECTOR = """
query PublicEmdat($limit: Int!, $offset: Int!, $from: Int!, $to: Int!, $classif: [String!]) {
  public_emdat(
    cursor: {limit: $limit, offset: $offset}
    from: $from
    to: $to
    classif: $classif
    include_hist: false
  ) {
    total_available
    data
  }
}
"""

#: The shape EM-DAT's documentation gives: filters nested, columns named.
QUERY_FILTERS = """
query PublicEmdat($limit: Int!, $offset: Int!, $from: Int!, $to: Int!, $classif: [String!]) {
  public_emdat(
    cursor: {limit: $limit, offset: $offset}
    filters: {from: $from, to: $to, classif: $classif, include_hist: false}
  ) {
    total_available
    data {
      disno
      classif_key
      iso
      country
      start_year
      start_month
      end_year
      end_month
      total_affected
    }
  }
}
"""

QUERY_INTROSPECT = """
query Introspect {
  __type(name: "Query") {
    fields {
      name
      args { name type { name kind ofType { name kind } } }
    }
  }
}
"""

VERDICT_QUERY_SHAPE = "query_shape"
VERDICT_ACCOUNT = "account"
VERDICT_UPSTREAM = "upstream"
VERDICT_OK = "connector_query_works"
VERDICT_INCONCLUSIVE = "inconclusive"

_ACCOUNT_WORDS = ("unauthori", "forbidden", "invalid token", "expired", "api key", "apikey", "not allowed")

PostFn = Callable[[str, dict[str, Any], dict[str, str], float], tuple[int, str]]


def _default_post(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: float) -> tuple[int, str]:
    import requests

    resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    return resp.status_code, resp.text or ""


def _probe(post: PostFn, name: str, query: str, variables: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    payload = {"query": query, "variables": variables}
    try:
        status, text = post(API_URL, payload, headers, TIMEOUT)
    except Exception as exc:  # noqa: BLE001 - a transport failure is the finding
        return {"name": name, "status": None, "error": f"{type(exc).__name__}: {exc}"}
    parsed: Any = None
    try:
        parsed = json.loads(text) if text else None
    except ValueError:
        parsed = None
    out: dict[str, Any] = {
        "name": name,
        "status": status,
        "body_head": text[:BODY_CHARS],
        "json": isinstance(parsed, dict),
    }
    if isinstance(parsed, dict):
        out["graphql_errors"] = parsed.get("errors")
        data = (parsed.get("data") or {}).get("public_emdat") if isinstance(parsed.get("data"), dict) else None
        if isinstance(data, dict):
            out["total_available"] = data.get("total_available")
            rows = data.get("data")
            out["rows_returned"] = len(rows) if isinstance(rows, list) else None
    return out


def decide(probes: dict[str, dict[str, Any]], key_present: bool) -> tuple[str, str]:
    """(verdict, reason) from the three probes. Pure."""

    if not key_present:
        return VERDICT_INCONCLUSIVE, f"{API_KEY_ENV} is not set; nothing was asked"
    connector = probes.get("connector", {})
    filters = probes.get("filters", {})
    introspect = probes.get("introspect", {})

    def _ok(p: dict[str, Any]) -> bool:
        return p.get("status") == 200 and not p.get("graphql_errors") and p.get("total_available") is not None

    if _ok(connector):
        return VERDICT_OK, "the connector's own query returned rows — the 500s were upstream and have cleared"

    bodies = " ".join(str(p.get("body_head") or "").lower() for p in probes.values())
    if any(word in bodies for word in _ACCOUNT_WORDS) or any(
        p.get("status") in (401, 403) for p in probes.values()
    ):
        return VERDICT_ACCOUNT, "the API names the key or the account in its refusal"

    if _ok(filters):
        return (
            VERDICT_QUERY_SHAPE,
            "the filters: variant returns rows while the connector's shape does not — "
            "rewrite the query in resolver/hazard_resolution/emdat.py",
        )
    if connector.get("graphql_errors") and not filters.get("graphql_errors"):
        return VERDICT_QUERY_SHAPE, f"GraphQL rejected the connector's shape: {connector.get('graphql_errors')}"
    if introspect.get("status") == 200 and introspect.get("json"):
        return (
            VERDICT_QUERY_SHAPE,
            "the schema is readable but neither query shape returned rows — compare the "
            "introspection block against the query text",
        )
    statuses = {p.get("status") for p in probes.values()}
    if statuses and statuses <= {500, 502, 503, 504, None}:
        return VERDICT_UPSTREAM, f"every probe failed the same way (statuses {sorted(s for s in statuses if s)}) — the endpoint is not answering anyone"
    return VERDICT_INCONCLUSIVE, f"mixed answers: {sorted(str(s) for s in statuses)}"


def run(post: PostFn | None = None, *, key: str | None = None, year: int | None = None) -> dict[str, Any]:
    post = post or _default_post
    key = key if key is not None else os.environ.get(API_KEY_ENV, "").strip()
    year = year or dt.date.today().year
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    variables = {"limit": 5, "offset": 0, "from": year - 1, "to": year, "classif": ["nat-hyd-flo"]}

    probes: dict[str, dict[str, Any]] = {}
    if key:
        probes["connector"] = _probe(post, "connector", QUERY_CONNECTOR, variables, headers)
        probes["filters"] = _probe(post, "filters", QUERY_FILTERS, variables, headers)
        probes["introspect"] = _probe(post, "introspect", QUERY_INTROSPECT, {}, headers)
    verdict, reason = decide(probes, bool(key))
    return {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "api_url": API_URL,
        "key_present": bool(key),
        "verdict": verdict,
        "reason": reason,
        "probes": probes,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnose the EM-DAT API from CI")
    parser.add_argument("--out", default=None, help="Write the JSON report here")
    args = parser.parse_args(argv)

    report = run()
    print(f"[emdat] verdict: {report['verdict']} — {report['reason']}")
    for name, probe in report["probes"].items():
        print(f"[emdat] probe {name}: status={probe.get('status')} json={probe.get('json')} "
              f"errors={probe.get('graphql_errors')} rows={probe.get('rows_returned')}")
        head = str(probe.get("body_head") or probe.get("error") or "")[:300]
        if head:
            print(f"[emdat]   body: {head}")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[emdat] report written to {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI shim
    sys.exit(main())
