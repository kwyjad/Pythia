# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""ReliefWeb documents into ``haz_raw_reliefweb_docs`` — rung 2's input.

Rung 2 of the impact ladder reads what humanitarian reporting SAYS, in
prose, and the only way to get a number out of prose is to read it. This
module does the half of that which needs no model: it decides WHICH
documents are worth reading and caches them.

Selection, in order:

1. **Filter.** One query per country-month: the country tag, the hazard's
   ReliefWeb disaster types (the same taxonomy the silence sweep uses, so
   detection and extraction cannot disagree about what a flood is), the
   content formats in ``reliefweb.documents.formats``, and a publication
   window of month start .. month end + ``publication_pad_days``.
2. **Rank.** ``reliefweb.documents.source_priority`` orders publishers
   best-first — OCHA, then governments, then UN agencies, and so on. A
   document whose publisher matches nothing ranks after every listed one.
   Ties break on recency, then document id, so the same month always
   selects the same documents.
3. **Cap.** The top ``max_docs_per_cell`` are stored; the rest are
   discarded with a count. This cap IS the per-cell extraction spend, so
   it is the first thing to lower if the monthly call cap starts binding.

Documents are cached under the shared ``haz_raw_*`` idiom: identical
content on a re-fetch is a no-op, a revised document appends a row, and
readers take the newest revision. That is what makes a re-run free — the
extractor keys its own cache on the document id.

This module is fetch-only and **never calls a model**. It also never
raises: a ReliefWeb outage returns ``ok=False`` so the rung is recorded as
UNREAD rather than empty.
"""

from __future__ import annotations

import datetime as dt
import logging
from datetime import timedelta
from typing import TYPE_CHECKING, Any

from resolver.hazard_resolution.reliefweb_sweep import PostFn, appname, default_post
from resolver.hazard_resolution.rulebook import (
    RULEBOOK_NAME_BY_HAZARD_CODE,
    Rulebook,
)
from resolver.hazard_resolution.sources import (
    FetchOutcome,
    RawRecord,
    load_raw_records,
    month_bounds,
    store_raw_records,
    utcnow_iso,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import duckdb

LOG = logging.getLogger(__name__)

SOURCE = "reliefweb_docs"

#: Fields requested from the API. ``body`` is the expensive one and the
#: reason this is a POST: it is also the only field a figure can be
#: extracted from.
_FIELDS = [
    "title",
    "url",
    "body",
    "date.created",
    "date.original",
    "source.name",
    "source.shortname",
    "source.type.name",
    "format.name",
    "disaster_type.name",
    "primary_country.iso3",
]


def _text_of(item: dict[str, Any], key: str) -> list[str]:
    """The ``name``/``shortname`` strings of a ReliefWeb list-of-dicts field."""

    values: list[str] = []
    for entry in item.get(key) or []:
        if isinstance(entry, dict):
            for sub in ("shortname", "name"):
                value = str(entry.get(sub) or "").strip()
                if value:
                    values.append(value)
            type_entry = entry.get("type")
            if isinstance(type_entry, dict):
                value = str(type_entry.get("name") or "").strip()
                if value:
                    values.append(value)
        elif isinstance(entry, str) and entry.strip():
            values.append(entry.strip())
    return values


def source_rank(publishers: list[str], rulebook: Rulebook) -> int:
    """Rank a document's publishers against ``documents.source_priority``.

    0 is the most preferred entry; an unmatched publisher ranks after every
    listed one. Matching is a case-insensitive substring test in both
    directions, because ReliefWeb writes the same publisher as "OCHA", "UN
    Office for the Coordination of Humanitarian Affairs" and "United
    Nations" depending on which field is read.
    """

    priority = [str(p).strip().lower() for p in rulebook.get("reliefweb.documents.source_priority")]
    haystack = [str(p).strip().lower() for p in publishers if str(p).strip()]
    for rank, wanted in enumerate(priority):
        for value in haystack:
            if wanted in value or value in wanted:
                return rank
    return len(priority)


def _created(item: dict[str, Any]) -> str:
    date_obj = item.get("date") or {}
    if isinstance(date_obj, dict):
        return str(date_obj.get("created") or date_obj.get("original") or "")
    return ""


def _document(
    item: dict[str, Any], iso3: str, ym: str, hazard: str, rulebook: Rulebook
) -> dict[str, Any] | None:
    """Normalise one API hit into the payload cached for extraction."""

    fields = item.get("fields") or {}
    doc_id = str(item.get("id") or fields.get("id") or "").strip()
    if not doc_id:
        return None

    body = str(fields.get("body") or "")
    limit = int(rulebook.get("reliefweb.documents.body_char_limit"))
    truncated = len(body) > limit

    publishers = _text_of(fields, "source")
    return {
        "doc_id": doc_id,
        "iso3": iso3,
        "ym": ym,
        "hazard": hazard,
        "title": str(fields.get("title") or ""),
        "url": str(fields.get("url") or ""),
        # The body is stored ALREADY truncated: the extractor's quote
        # verifier checks a quote against the text it was sent, and storing
        # more than was sent would let an unverifiable quote pass later.
        "body": body[:limit],
        "body_truncated": truncated,
        "body_chars": min(len(body), limit),
        "date_created": _created(fields),
        "sources": publishers,
        "formats": _text_of(fields, "format"),
        "disaster_types": _text_of(fields, "disaster_type"),
        "source_rank": source_rank(publishers, rulebook),
        "fetched_at": utcnow_iso(),
    }


def sort_documents(documents: list[dict[str, Any]]) -> None:
    """Order documents in place: authority, then recency, then id.

    Three stable passes rather than one composite key, because the middle
    term sorts DESCENDING (newest first) while the others ascend — and
    Python's sort being stable makes least-significant-key-first exactly
    equivalent, without a reversed-comparison wrapper type.
    """

    documents.sort(key=lambda d: str(d.get("doc_id") or ""))
    documents.sort(key=lambda d: str(d.get("date_created") or ""), reverse=True)
    documents.sort(key=lambda d: int(d.get("source_rank") or 0))


def select_documents(
    items: list[dict[str, Any]], iso3: str, ym: str, hazard: str, rulebook: Rulebook
) -> tuple[list[dict[str, Any]], int]:
    """Rank and cap the API's hits. Returns (selected, discarded_count).

    Pure function: the ranking is the rulebook's, and the same pool always
    yields the same selection.
    """

    documents = [
        doc
        for doc in (_document(item, iso3, ym, hazard, rulebook) for item in items)
        if doc is not None
    ]
    sort_documents(documents)
    cap = int(rulebook.get("reliefweb.documents.max_docs_per_cell"))
    return documents[:cap], max(0, len(documents) - cap)


def _payload(iso3: str, ym: str, hazard: str, rulebook: Rulebook) -> dict[str, Any]:
    hazard_key = RULEBOOK_NAME_BY_HAZARD_CODE[hazard]
    disaster_types = list(rulebook.get(f"{hazard_key}.reliefweb_sweep.disaster_types"))
    formats = list(rulebook.get("reliefweb.documents.formats"))
    pad_days = int(rulebook.get("reliefweb.documents.publication_pad_days"))
    pool = int(rulebook.get("reliefweb.documents.candidate_pool_size"))

    start, end = month_bounds(ym)
    window_end = end + timedelta(days=pad_days)
    return {
        "filter": {
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
                {"field": "disaster_type.name", "value": disaster_types},
                {"field": "format.name", "value": formats},
            ],
        },
        "fields": {"include": _FIELDS},
        "sort": ["date.created:desc"],
        "limit": pool,
    }


def fetch_documents(
    con: "duckdb.DuckDBPyConnection",
    iso3: str,
    ym: str,
    hazard: str,
    rulebook: Rulebook,
    *,
    post: PostFn | None = None,
) -> FetchOutcome:
    """Fetch and cache the documents for ONE triggered country-month.

    Per-cell by design: a non-triggered month has already resolved without
    reading anything, and this is the step that costs money downstream.
    Never raises — ``ok=False`` means the rung is unread, which is not the
    same fact as the rung being empty.
    """

    iso3 = iso3.upper()
    api_url = str(rulebook.get("reliefweb.api_base_url")).rstrip("/") + "/reports"
    timeout = float(rulebook.get("reliefweb.documents.request_timeout_sec"))

    outcome = FetchOutcome(source=SOURCE, ok=False, source_urls=[api_url])
    if hazard not in RULEBOOK_NAME_BY_HAZARD_CODE:
        outcome.error = f"no rulebook hazard name for hazard code {hazard!r}"
        LOG.warning("[reliefweb_docs] %s", outcome.error)
        return outcome

    payload = _payload(iso3, ym, hazard, rulebook)
    post = post or default_post
    try:
        data = post(api_url, payload, {"appname": appname()}, timeout)
    except Exception as exc:  # requests errors, JSON errors
        LOG.warning(
            "[reliefweb_docs] fetch failed for %s %s %s: %s", iso3, hazard, ym, exc
        )
        outcome.error = str(exc)
        return outcome

    items = list(data.get("data") or [])
    selected, discarded = select_documents(items, iso3, ym, hazard, rulebook)

    records = [
        RawRecord(
            record_id=f"rw-{doc['doc_id']}",
            payload=doc,
            iso3=iso3,
            ym=ym,
            hazard=hazard,
            source_url=doc.get("url") or None,
        )
        for doc in selected
    ]
    stored = store_raw_records(con, SOURCE, records)

    outcome.ok = True
    outcome.records = stored["records"]
    outcome.inserted = stored["inserted"]
    outcome.detail = {
        "total_available": int(data.get("totalCount") or len(items)),
        "returned": len(items),
        "selected": len(selected),
        "discarded_over_cap": discarded,
        "max_docs_per_cell": int(rulebook.get("reliefweb.documents.max_docs_per_cell")),
    }
    LOG.info(
        "[reliefweb_docs] %s %s %s: %d returned, %d selected (%d over cap, %d new)",
        iso3, hazard, ym, len(items), len(selected), discarded, stored["inserted"],
    )
    return outcome


def documents_for_country_month(
    con: "duckdb.DuckDBPyConnection",
    iso3: str,
    ym: str,
    hazard: str,
    rulebook: Rulebook | None = None,
) -> list[dict[str, Any]]:
    """Cached documents for a cell, in the order they should be read.

    Attribution scope: a document is stored against the month whose figures
    it was fetched for, so this filters on the stored ``ym`` exactly as the
    record-based rungs do.

    When a ``rulebook`` is given, the result is re-capped to
    ``max_docs_per_cell`` AFTER ranking. The cap at selection time is not
    enough on its own: successive fetches of a still-publishing emergency
    select different top-N sets that accumulate in the cache, and without a
    read-time cap the per-cell spend bound quietly stops being one.
    """

    documents = load_raw_records(
        con, SOURCE, iso3=iso3.upper(), hazard=hazard, ym=ym
    )
    sort_documents(documents)
    if rulebook is not None:
        cap = int(rulebook.get("reliefweb.documents.max_docs_per_cell"))
        if len(documents) > cap:
            LOG.info(
                "[reliefweb_docs] %s %s %s: %d cached documents exceed "
                "max_docs_per_cell=%d — reading the top %d by authority/recency",
                iso3.upper(), hazard, ym, len(documents), cap, cap,
            )
            documents = documents[:cap]
    return documents


def document_date(document: dict[str, Any]) -> dt.date | None:
    """The document's publication date, or None when it is unparseable."""

    from resolver.hazard_resolution.sources import parse_date

    return parse_date(document.get("date_created"))
