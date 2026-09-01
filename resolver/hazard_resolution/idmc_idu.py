# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""IDMC IDU into ``haz_raw_idmc_idu`` — the ladder's bottom rung, a floor.

IDU (Internal Displacement Updates) reports people DISPLACED by a
disaster. Displaced people are a strict subset of affected people: a flood
that displaces 10,000 has affected at least 10,000, and almost certainly
more. So this rung resolves as a **lower bound**, never as a point
estimate — ``lower_bound_rungs`` in the rulebook says so, the candidate
carries ``value_type='displaced_lower_bound'``, and the resolution's
``rule_fired`` and provenance both record that the number is a floor.
Losing that label would silently turn a floor into an answer, which is the
single most dangerous thing this rung can do.

It sits at the bottom of the ladder for the same reason: any rung above it
states the quantity we actually want.

**Not reused:** :mod:`resolver.ingestion.idmc` is a large, heavily
configured client (PostgREST paging, HDX fallback, Helix schema probing)
built to produce canonical ``facts_resolved`` rows on the last-180-days
endpoint. The machine needs a much smaller thing — the ``all`` endpoint
IDMC supplies with the key, records kept intact — and bending that client
to it would couple two very different jobs. The endpoint URL is therefore
a rulebook value (IDMC supplies it alongside the key) and the key itself
is ``IDMC_API_KEY`` from the environment, sent as ``client_id``.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Callable

import requests

from resolver.hazard_resolution.rulebook import Rulebook
from resolver.hazard_resolution.rules import event_months
from resolver.hazard_resolution.sources import (
    FetchOutcome,
    RawRecord,
    fetch_window,
    load_raw_records,
    parse_date,
    parse_number,
    store_raw_records,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import duckdb

LOG = logging.getLogger(__name__)

SOURCE = "idmc_idu"
API_KEY_ENV = "IDMC_API_KEY"

#: The SAME credential under the name the rest of the repo already uses.
#: ``resolver/ingestion/idmc`` and ``scripts/ci/probe_idmc_reachability.py``
#: send ``IDMC_HELIX_CLIENT_ID`` as the ``client_id`` query param against
#: ``helix-tools-api.idmcdb.org/external-api/idus/...`` — the same host, the
#: same API family and the same parameter this module uses, differing only in
#: endpoint (``all`` here, ``last-180-days`` there). Accepting it as a
#: fallback means one IDMC client id serves both consumers instead of the
#: operator maintaining two secrets that hold one value and can drift apart.
#:
#: NOT a fallback, deliberately: ``IDMC_API_TOKEN``. That is a different
#: credential — a bearer token for ``backend.idmcdb.org`` — and its mere
#: presence is a feature flag (``scripts/ci/run_connectors.py`` sets
#: ``RESOLVER_SKIP_IDMC`` from it), so it must never be repurposed here.
FALLBACK_KEY_ENV = "IDMC_HELIX_CLIENT_ID"

#: Injectable transport seam for tests: (url, params, timeout) -> list[dict].
GetFn = Callable[[str, dict, float], list]

#: IDU figure fields, best first. ``figure`` is the headline displacement
#: count; the alternatives appear in older records.
_FIGURE_FIELDS = ("figure", "total_figures", "displacement_figure")


def _api_key() -> tuple[str, str] | None:
    """The IDU client id and the env var it came from, or None.

    ``IDMC_API_KEY`` wins when both are set, so an operator can point this
    rung at a different client id than the ingestion path without touching
    the shared one.
    """

    for env_name in (API_KEY_ENV, FALLBACK_KEY_ENV):
        key = os.getenv(env_name, "").strip()
        if key:
            if env_name != API_KEY_ENV:
                LOG.info(
                    "[idmc_idu] %s is not set — using %s (the same IDMC client "
                    "id, sent as client_id to the same external-api host)",
                    API_KEY_ENV, env_name,
                )
            return key, env_name
    LOG.warning(
        "[idmc_idu] neither %s nor %s is set — the ladder's lower-bound rung "
        "will contribute no candidates this run",
        API_KEY_ENV, FALLBACK_KEY_ENV,
    )
    return None


def _default_get(url: str, params: dict, timeout: float) -> list:
    resp = requests.get(
        url,
        params=params,
        timeout=timeout,
        headers={"Accept": "application/json"},
    )
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict):
        return data.get("results") or data.get("data") or []
    return data or []


def hazard_for_record(record: dict[str, Any], rulebook: Rulebook) -> str | None:
    """Map an IDU record's hazard text to a repo hazard code, or None.

    Keyword lists live in ``idmc_idu.hazard_keywords``. Matching is
    longest-keyword-first so "flash flood" cannot be claimed by a shorter
    entry belonging to another hazard, and a record matching nothing is
    skipped rather than guessed at.
    """

    blob = " ".join(
        str(record.get(key) or "")
        for key in (
            "type_name",
            "subtype",
            "hazard_type_name",
            "hazard_sub_type",
            "event_name",
            "standard_popup_text",
        )
    ).lower()
    if not blob.strip():
        return None

    best: tuple[int, str] | None = None
    for hazard, keywords in rulebook.get("idmc_idu.hazard_keywords").items():
        for keyword in keywords:
            key = str(keyword).lower()
            if key in blob and (best is None or len(key) > best[0]):
                best = (len(key), hazard)
    return best[1] if best else None


def displacement_from_record(record: dict[str, Any]) -> float | None:
    """The displacement figure an IDU record states, or None."""

    for field in _FIGURE_FIELDS:
        value = parse_number(record.get(field))
        if value is not None:
            return value
    return None


def _record(record: dict[str, Any], hazard: str) -> RawRecord | None:
    record_id = str(record.get("id") or record.get("event_id") or "").strip()
    iso3 = str(record.get("iso3") or record.get("iso") or "").strip().upper()
    start = parse_date(record.get("displacement_start_date")) or parse_date(
        record.get("event_start_date")
    )
    end = (
        parse_date(record.get("displacement_end_date"))
        or parse_date(record.get("event_end_date"))
        or start
    )
    if not record_id or len(iso3) != 3 or start is None:
        return None
    if end < start:
        end = start

    payload = {
        "idu_id": record_id,
        "iso3": iso3,
        "hazard": hazard,
        "displacement_type": record.get("displacement_type"),
        "event_name": record.get("event_name") or "",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "months_overlapped": event_months(start, end),
        # Named "displaced", never "affected": this is a floor for a
        # different quantity and the payload must not blur the two.
        "displaced": displacement_from_record(record),
        "raw": record,
    }
    return RawRecord(
        record_id=f"idu-{record_id}",
        payload=payload,
        iso3=iso3,
        ym=start.strftime("%Y-%m"),
        hazard=hazard,
        source_url=str(record.get("source_url") or record.get("url") or "") or None,
    )


def fetch_idmc_idu(
    con: "duckdb.DuckDBPyConnection",
    ym: str,
    hazard: str,
    rulebook: Rulebook,
    *,
    get: GetFn | None = None,
) -> FetchOutcome:
    """Fetch IDU disaster displacement around ``ym`` into ``haz_raw_idmc_idu``."""

    url = str(rulebook.get("idmc_idu.api_url"))
    timeout = float(rulebook.get("idmc_idu.request_timeout_sec"))
    wanted_types = {str(t).strip().lower() for t in rulebook.get("idmc_idu.displacement_types")}

    outcome = FetchOutcome(source=SOURCE, ok=False, source_urls=[url])
    resolved = _api_key()
    if not resolved:
        # UNAVAILABLE, not empty: the row records that this rung could not be
        # read rather than that it had nothing to say.
        outcome.error = f"neither {API_KEY_ENV} nor {FALLBACK_KEY_ENV} is set"
        return outcome
    key, key_env = resolved
    outcome.detail["credential_env"] = key_env

    start, end = fetch_window(ym, rulebook, "idmc_idu")
    get = get or _default_get

    try:
        rows = get(url, {"client_id": key}, timeout)
    except Exception as exc:
        LOG.error("[idmc_idu] fetch failed for %s %s: %s", hazard, ym, exc)
        outcome.error = str(exc)
        return outcome

    wanted_months = set(event_months(start, end))
    records: list[RawRecord] = []
    skipped_type = skipped_hazard = 0
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        dtype = str(row.get("displacement_type") or "").strip().lower()
        if wanted_types and dtype not in wanted_types:
            skipped_type += 1
            continue
        row_hazard = hazard_for_record(row, rulebook)
        if row_hazard != hazard:
            skipped_hazard += 1
            continue
        built = _record(row, hazard)
        if built is None:
            continue
        if not wanted_months.intersection(built.payload["months_overlapped"]):
            continue
        records.append(built)

    stored = store_raw_records(con, SOURCE, records)
    outcome.ok = True
    outcome.records = stored["records"]
    outcome.inserted = stored["inserted"]
    outcome.detail = {
        # Which env var supplied the client id — provenance for a rung whose
        # credential now has two accepted names.
        "credential_env": key_env,
        "window": {"from": start.isoformat(), "to": end.isoformat()},
        "rows_returned": len(rows or []),
        "skipped_displacement_type": skipped_type,
        "skipped_other_hazard": skipped_hazard,
    }
    LOG.info(
        "[idmc_idu] %s %s: %d rows returned, %d stored (%d new)",
        hazard, ym, len(rows or []), stored["records"], stored["inserted"],
    )
    return outcome


def records_for_country_month(
    con: "duckdb.DuckDBPyConnection", iso3: str, ym: str, hazard: str
) -> list[dict[str, Any]]:
    """Cached IDU records stating a displacement figure for this cell."""

    return [
        record
        for record in load_raw_records(
            con, SOURCE, iso3=iso3.upper(), hazard=hazard, ym=ym
        )
        if record.get("displaced") is not None
    ]
