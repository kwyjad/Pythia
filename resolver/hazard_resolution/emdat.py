# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""EM-DAT into ``haz_raw_emdat`` — the ladder's top rung.

EM-DAT (CRED's International Disaster Database) is the ladder's highest
rung because its figures are curated and definitionally stable: "total
affected" means injured + affected + homeless, consistently, across
decades. The price is latency — EM-DAT entries appear weeks to months
after an event and are revised afterwards — which is exactly why the
machine freezes a cell at month-end + ``freeze_days`` and audits later
revisions rather than chasing them.

The public API is GraphQL over a single endpoint (``emdat.api_url``),
authenticated with ``EMDAT_API_KEY`` from the environment. Never from the
rulebook: the loader rejects credential-shaped keys outright.

The response shape is deliberately handled loosely. EM-DAT's field naming
has changed across API versions (``total_affected`` vs ``no_affected``
plus components), so :func:`affected_from_record` prefers the explicit
total and falls back to summing the components EM-DAT defines it from —
and returns None rather than guessing when neither is present. A record
with no stated figure is not a zero; it is an absent rung.
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
    parse_number,
    store_raw_records,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import duckdb

LOG = logging.getLogger(__name__)

SOURCE = "emdat"
API_KEY_ENV = "EMDAT_API_KEY"

#: Injectable transport seam for tests: (url, payload, headers, timeout) -> dict.
PostFn = Callable[[str, dict, dict, float], dict]

#: EM-DAT's own definition of "total affected" — the components it sums.
_AFFECTED_COMPONENTS = ("no_injured", "no_affected", "no_homeless")

_QUERY = """
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


def _api_key() -> str | None:
    key = os.getenv(API_KEY_ENV, "").strip()
    if not key:
        LOG.warning(
            "[emdat] %s is not set — EM-DAT is the ladder's top rung and will "
            "contribute no candidates this run",
            API_KEY_ENV,
        )
        return None
    return key


def _default_post(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _record_dates(record: dict[str, Any]) -> tuple[Any, Any]:
    """EM-DAT's split year/month/day fields as (start, end) dates.

    EM-DAT routinely omits the day, and sometimes the month, of an event.
    A missing day means the 1st; a missing month makes the record
    unattributable to a month at all, and it is dropped rather than
    silently parked in January.
    """

    import datetime as dt

    def _build(prefix: str, day_default: int) -> dt.date | None:
        year = parse_number(record.get(f"{prefix}_year"))
        month = parse_number(record.get(f"{prefix}_month"))
        day = parse_number(record.get(f"{prefix}_day"))
        if year is None or month is None:
            return None
        try:
            return dt.date(int(year), int(month), int(day or day_default))
        except ValueError:
            try:
                return dt.date(int(year), int(month), 1)
            except ValueError:
                return None

    start = _build("start", 1)
    end = _build("end", 1) or start
    if start is not None and end is not None and end < start:
        end = start
    return start, end


def affected_from_record(record: dict[str, Any]) -> float | None:
    """The people-affected figure an EM-DAT record STATES, or None.

    Prefers the explicit total; falls back to summing EM-DAT's own
    components when the total is absent but at least one component is
    present. Never invents a figure — an absent rung is absent, not zero.
    """

    total = parse_number(record.get("total_affected"))
    if total is not None:
        return total

    components = [parse_number(record.get(field)) for field in _AFFECTED_COMPONENTS]
    present = [c for c in components if c is not None]
    if not present:
        return None
    return float(sum(present))


def _iso3(record: dict[str, Any]) -> str | None:
    for key in ("iso", "iso3", "ISO", "country_iso3"):
        value = str(record.get(key) or "").strip().upper()
        if len(value) == 3 and value.isalpha():
            return value
    return None


def _record(record: dict[str, Any], hazard: str) -> RawRecord | None:
    disno = str(record.get("disno") or record.get("disasterno") or "").strip()
    iso3 = _iso3(record)
    start, end = _record_dates(record)
    if not disno or not iso3 or start is None or end is None:
        return None

    payload = {
        "disno": disno,
        "iso3": iso3,
        "hazard": hazard,
        "classif_key": record.get("classif_key"),
        "event_name": record.get("event_name") or "",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "months_overlapped": event_months(start, end),
        "total_affected": affected_from_record(record),
        "total_deaths": parse_number(record.get("total_deaths")),
        "raw": record,
    }
    return RawRecord(
        record_id=f"emdat-{disno}-{iso3}",
        payload=payload,
        iso3=iso3,
        ym=start.strftime("%Y-%m"),
        hazard=hazard,
        # EM-DAT records are addressed by DisNo. in the public database.
        source_url=f"https://public.emdat.be/data?disno={disno}",
    )


def fetch_emdat(
    con: "duckdb.DuckDBPyConnection",
    ym: str,
    hazard: str,
    rulebook: Rulebook,
    *,
    post: PostFn | None = None,
) -> FetchOutcome:
    """Fetch EM-DAT records around ``ym`` for ``hazard`` into ``haz_raw_emdat``.

    Returns ``ok=False`` on a missing key or a failed call — never raises,
    and never lets "we could not ask" look like "there is nothing there".
    """

    url = str(rulebook.get("emdat.api_url"))
    timeout = float(rulebook.get("emdat.request_timeout_sec"))
    limit = int(rulebook.get("emdat.page_limit"))
    max_pages = int(rulebook.get("emdat.max_pages"))
    classif = list(rulebook.get("emdat.classif_keys").get(hazard, []))

    outcome = FetchOutcome(source=SOURCE, ok=False, source_urls=[url])
    if not classif:
        outcome.error = f"no emdat.classif_keys entry for hazard {hazard}"
        LOG.warning("[emdat] %s", outcome.error)
        return outcome

    key = _api_key()
    if not key:
        outcome.error = f"{API_KEY_ENV} not set"
        return outcome

    start, end = fetch_window(ym, rulebook, "emdat")
    post = post or _default_post
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    rows: list[dict[str, Any]] = []
    offset = 0
    try:
        for page in range(max_pages):
            payload = {
                "query": _QUERY,
                "variables": {
                    "limit": limit,
                    "offset": offset,
                    # EM-DAT's from/to are YEARS, not dates; the month-level
                    # filtering happens on the records themselves.
                    "from": start.year,
                    "to": end.year,
                    "classif": classif,
                },
            }
            data = post(url, payload, headers, timeout)
            errors = data.get("errors")
            if errors:
                raise RuntimeError(f"EM-DAT GraphQL errors: {errors}")
            block = ((data.get("data") or {}).get("public_emdat") or {})
            batch = block.get("data") or []
            rows.extend(batch)
            if len(batch) < limit:
                break
            offset += limit
        else:
            LOG.warning(
                "[emdat] stopped at emdat.max_pages=%d with %d rows — the window "
                "may be truncated",
                max_pages, len(rows),
            )
    except Exception as exc:
        LOG.error("[emdat] fetch failed for %s %s: %s", hazard, ym, exc)
        outcome.error = str(exc)
        return outcome

    # EM-DAT filters by year, so trim to the month window we asked for.
    wanted_months = set(event_months(start, end))
    records = []
    for row in rows:
        record = _record(row, hazard)
        if record is None:
            continue
        if not wanted_months.intersection(record.payload["months_overlapped"]):
            continue
        records.append(record)

    stored = store_raw_records(con, SOURCE, records)
    outcome.ok = True
    outcome.records = stored["records"]
    outcome.inserted = stored["inserted"]
    outcome.detail = {
        "window": {"from": start.isoformat(), "to": end.isoformat()},
        "rows_returned": len(rows),
        "classif": classif,
    }
    LOG.info(
        "[emdat] %s %s: %d rows returned, %d in window (%d new)",
        hazard, ym, len(rows), stored["records"], stored["inserted"],
    )
    return outcome


def records_for_country_month(
    con: "duckdb.DuckDBPyConnection", iso3: str, ym: str, hazard: str
) -> list[dict[str, Any]]:
    """Cached EM-DAT records whose START month is ``ym`` for this country.

    Attribution scope, not detection scope: a figure belongs wholly to the
    month its event started (``event_attribution.figure = start_month``),
    so this filters on the stored ``ym`` rather than the overlap list.
    """

    return [
        record
        for record in load_raw_records(
            con, SOURCE, iso3=iso3.upper(), hazard=hazard, ym=ym
        )
        if record.get("total_affected") is not None
    ]
