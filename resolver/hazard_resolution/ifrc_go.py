# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""IFRC GO field reports into ``haz_raw_ifrc_go`` — the ladder's third rung.

This is the source the machine exists to supplement, not replace. IFRC GO
resolves roughly 4% of people-affected questions per month on its own
(see ``docs/montandon_assessment.md``); it sits below EM-DAT and extracted
ReliefWeb figures because its coverage is thin and its early field reports
are revised upward — but where it does report, it reports a real, stated
``num_affected``, which is exactly what the ladder wants.

**Reuse.** Paging, retry/backoff and the two country shapes GO uses come
from :mod:`resolver.ingestion.ifrc_go_client` (``req_json``,
``iso3_pairs_from_go``), the repo's existing GO client. What is NOT reused
is that module's hazard detection and metric extraction: those produce
canonical ``facts_resolved`` rows with their own precedence semantics,
whereas the machine needs the raw field report kept intact so
reconciliation can cite the exact record and field a figure came from.

A GO field report can name several countries. The report's
``num_affected`` describes the operation as a whole, so it is recorded
once per named country WITHOUT splitting — splitting would invent a
per-country breakdown GO never stated. The reconciler treats it as a
country-level candidate and the sanity ceiling catches the case where
that over-attributes.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

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
    utcnow_iso,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import duckdb

LOG = logging.getLogger(__name__)

SOURCE = "ifrc_go"

#: Injectable transport seam for tests: (base, path, params, headers) -> dict.
GetFn = Callable[[str, str, dict, dict], dict]

_USER_AGENT = "pythia-resolution-machine/1.0"


def _default_get(base: str, path: str, params: dict, headers: dict) -> dict:
    from resolver.ingestion.ifrc_go_client import req_json

    return req_json(base, path, params, headers)


def _hazard_matches(record: dict[str, Any], hazard: str, rulebook: Rulebook) -> bool:
    """Does this field report concern ``hazard``?

    GO's structured ``dtype`` is authoritative when present; the report's
    free text is the fallback. Both go through the SAME keyword lists the
    IDU connector uses, so one rulebook edit changes both.
    """

    keywords = [k.lower() for k in rulebook.get("idmc_idu.hazard_keywords").get(hazard, [])]
    if not keywords:
        return False

    dtype = record.get("dtype")
    dtype_name = ""
    if isinstance(dtype, dict):
        dtype_name = str(dtype.get("name") or "")
    elif dtype is not None:
        dtype_name = str(dtype)

    haystacks = [
        dtype_name,
        str(record.get("summary") or ""),
        str(record.get("title") or ""),
        str(record.get("description") or ""),
    ]
    blob = " ".join(haystacks).lower()
    return any(keyword in blob for keyword in keywords)


def affected_from_record(
    record: dict[str, Any], rulebook: Rulebook
) -> tuple[float | None, str | None]:
    """The stated people-affected figure and the field it came from.

    Walks ``ifrc_go.affected_fields`` in order and takes the first field
    that states a number. Returns ``(None, None)`` when the report states
    none — the machine does not derive a figure from casualty components
    here, because a derived floor and a reported total are different
    quantities and the ladder must be able to tell them apart.
    """

    for field in rulebook.get("ifrc_go.affected_fields"):
        value = parse_number(record.get(field))
        if value is not None:
            return value, str(field)
    return None, None


def _iso3s(record: dict[str, Any]) -> list[str]:
    """Every ISO3 a field report names, via the shared GO country resolver."""

    try:
        import pandas as pd

        from resolver.ingestion.ifrc_go_client import COUNTRIES, iso3_pairs_from_go

        countries = pd.read_csv(COUNTRIES, encoding="utf-8-sig")
        return sorted({iso for _, iso in iso3_pairs_from_go(countries, record)})
    except Exception as exc:  # registry unavailable — fall back to the record
        LOG.debug("[ifrc_go] country registry lookup failed (%s); using record", exc)
        out: list[str] = []
        details = record.get("countries_details")
        if isinstance(details, list):
            out.extend(
                str(c.get("iso3", "")).strip().upper()
                for c in details
                if isinstance(c, dict)
            )
        single = record.get("country_details") or record.get("country")
        if isinstance(single, dict):
            out.append(str(single.get("iso3", "")).strip().upper())
        return sorted({c for c in out if len(c) == 3})


def _record(
    record: dict[str, Any], hazard: str, iso3: str, rulebook: Rulebook
) -> RawRecord | None:
    report_id = str(record.get("id") or "").strip()
    if not report_id:
        return None
    reported = (
        parse_date(record.get("report_date"))
        or parse_date(record.get("start_date"))
        or parse_date(record.get("created_at"))
    )
    if reported is None:
        return None

    value, field = affected_from_record(record, rulebook)
    payload = {
        "report_id": report_id,
        "iso3": iso3,
        "hazard": hazard,
        "title": record.get("summary") or record.get("title") or "",
        "report_date": reported.isoformat(),
        "start_date": reported.isoformat(),
        "end_date": reported.isoformat(),
        "months_overlapped": event_months(reported, reported),
        "num_affected": value,
        "affected_field": field,
        "raw": record,
        "fetched_at": utcnow_iso(),
    }
    return RawRecord(
        record_id=f"go-{report_id}-{iso3}",
        payload=payload,
        iso3=iso3,
        ym=reported.strftime("%Y-%m"),
        hazard=hazard,
        source_url=f"https://go.ifrc.org/reports/{report_id}",
    )


def fetch_ifrc_go(
    con: "duckdb.DuckDBPyConnection",
    ym: str,
    hazard: str,
    rulebook: Rulebook,
    *,
    get: GetFn | None = None,
) -> FetchOutcome:
    """Fetch GO field reports around ``ym`` into ``haz_raw_ifrc_go``."""

    base = str(rulebook.get("ifrc_go.api_base_url"))
    path = str(rulebook.get("ifrc_go.field_report_path"))
    page_size = int(rulebook.get("ifrc_go.page_size"))
    max_pages = int(rulebook.get("ifrc_go.max_pages"))

    outcome = FetchOutcome(source=SOURCE, ok=False, source_urls=[base + path])
    start, end = fetch_window(ym, rulebook, "ifrc_go")
    get = get or _default_get
    headers = {"Accept": "application/json", "User-Agent": _USER_AGENT}

    rows: list[dict[str, Any]] = []
    try:
        for page in range(max_pages):
            params = {
                "limit": page_size,
                "offset": page * page_size,
                "report_date__gte": start.isoformat(),
                "report_date__lte": end.isoformat(),
                "ordering": "-report_date",
            }
            data = get(base, path, params, headers)
            batch = (data or {}).get("results") or []
            rows.extend(batch)
            if len(batch) < page_size or not (data or {}).get("next"):
                break
        else:
            LOG.warning(
                "[ifrc_go] stopped at ifrc_go.max_pages=%d with %d rows — the "
                "window may be truncated",
                max_pages, len(rows),
            )
    except Exception as exc:
        LOG.error("[ifrc_go] fetch failed for %s %s: %s", hazard, ym, exc)
        outcome.error = str(exc)
        return outcome

    records: list[RawRecord] = []
    for row in rows:
        if not _hazard_matches(row, hazard, rulebook):
            continue
        for iso3 in _iso3s(row):
            built = _record(row, hazard, iso3, rulebook)
            if built is not None:
                records.append(built)

    stored = store_raw_records(con, SOURCE, records)
    outcome.ok = True
    outcome.records = stored["records"]
    outcome.inserted = stored["inserted"]
    outcome.detail = {
        "window": {"from": start.isoformat(), "to": end.isoformat()},
        "reports_returned": len(rows),
    }
    LOG.info(
        "[ifrc_go] %s %s: %d reports returned, %d country-records stored (%d new)",
        hazard, ym, len(rows), stored["records"], stored["inserted"],
    )
    return outcome


def records_for_country_month(
    con: "duckdb.DuckDBPyConnection", iso3: str, ym: str, hazard: str
) -> list[dict[str, Any]]:
    """Cached GO records stating a figure for this country-month."""

    return [
        record
        for record in load_raw_records(
            con, SOURCE, iso3=iso3.upper(), hazard=hazard, ym=ym
        )
        if record.get("num_affected") is not None
    ]
