# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Shared raw-cache plumbing for the resolution machine's connectors.

Every ladder source stores what it fetched in its own thin ``haz_raw_*``
cache before anything interprets it, using the one storage idiom
``ibtracs.py`` established in Phase 1:

* one row per source record, ``record_id`` = the source-native identifier;
* the record verbatim in ``payload_json``;
* ``content_hash`` over that payload, so a re-fetch of identical content is
  a no-op (``UNIQUE (record_id, content_hash)``) while an upstream revision
  APPENDS a new row;
* ``retrieved_at`` on every row, because provenance is a hard rule and a
  figure without a retrieval time cannot be audited later.

Readers take the newest revision per ``record_id``
(:func:`load_raw_records`), so a revised record supersedes its predecessor
without the older evidence being destroyed.

**The hash covers CONTENT ONLY.** :data:`VOLATILE_PAYLOAD_KEYS` is stripped
from the payload before it is serialised and hashed, because a wall-clock
stamp inside the hashed bytes silently converts this idempotent cache into
an append-only log: the hash differs on every fetch, ``INSERT OR IGNORE``
never ignores, and each run appends a full duplicate of every record it
touched. Seven connectors did exactly that (a ``"fetched_at": utcnow_iso()``
copied from one to the next) and it grew the canonical DB from 3.0 GB to
17.7 GB in 26 nights before anyone read the "N new revision rows" this
module has been logging all along. The retrieval time is not lost — it is
the ``retrieved_at`` COLUMN, which :func:`load_raw_records` re-injects as
``_retrieved_at``.

One consequence to know: with dedup working, ``retrieved_at`` is when this
content was FIRST seen, not when it was last checked. The did-we-check fact
lives at run level in :meth:`FetchOutcome.as_provenance`. Do not add a
``last_seen_at`` column updated on every collision — a DuckDB UPDATE is
internally delete+insert, so ~100k of them a night would reintroduce the
bloat in a subtler form.
"""

from __future__ import annotations

import calendar
import datetime as dt
import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable

from resolver.hazard_resolution.schema import ensure_haz_schema, raw_table_name

if TYPE_CHECKING:  # pragma: no cover - typing only
    import duckdb

LOG = logging.getLogger(__name__)


@dataclass
class RawRecord:
    """One upstream record on its way into a ``haz_raw_*`` cache."""

    record_id: str
    payload: dict[str, Any]
    iso3: str | None = None
    ym: str | None = None
    hazard: str | None = None
    source_url: str | None = None


@dataclass
class FetchOutcome:
    """What a connector run did, for logging and provenance.

    ``ok=False`` means the fetch failed. Callers must treat that as
    "unknown", never as "nothing happened" — the difference decides
    whether a zero or a NO_DATA is safe to write.
    """

    source: str
    ok: bool
    records: int = 0
    inserted: int = 0
    source_urls: list[str] = field(default_factory=list)
    error: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def as_provenance(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "ok": self.ok,
            "records": self.records,
            "source_urls": list(self.source_urls),
            "error": self.error,
            **({"detail": self.detail} if self.detail else {}),
        }


def utcnow_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def month_bounds(ym: str) -> tuple[dt.date, dt.date]:
    """(first day, last day) of a ``YYYY-MM`` month."""

    year, month = (int(p) for p in ym.split("-"))
    return dt.date(year, month, 1), dt.date(
        year, month, calendar.monthrange(year, month)[1]
    )


def shift_month(ym: str, months: int) -> str:
    """``YYYY-MM`` shifted by ``months`` (may be negative)."""

    year, month = (int(p) for p in ym.split("-"))
    index = (year * 12 + month - 1) + months
    return f"{index // 12:04d}-{index % 12 + 1:02d}"


def fetch_window(ym: str, rulebook, prefix: str) -> tuple[dt.date, dt.date]:
    """The date window a source fetches around a target month.

    Sources publish late and events start before the month they land in,
    so each connector reads a rulebook-configured ``lookback_months`` /
    ``lookahead_months`` around the target rather than the bare month.
    """

    lookback = int(rulebook.get(f"{prefix}.lookback_months"))
    lookahead = int(rulebook.get(f"{prefix}.lookahead_months"))
    start, _ = month_bounds(shift_month(ym, -lookback))
    _, end = month_bounds(shift_month(ym, lookahead))
    return start, end


# Keys that describe WHEN we fetched, never WHAT was fetched. They are
# stripped before serialising so an unchanged record hashes identically on
# every fetch. See the module docstring: leaving a wall clock in the hashed
# payload is what turned this cache into an append-only log.
VOLATILE_PAYLOAD_KEYS = frozenset({"fetched_at", "retrieved_at", "_retrieved_at"})


def stable_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """The payload as it is stored and hashed: content, no wall clocks."""

    return {k: v for k, v in payload.items() if k not in VOLATILE_PAYLOAD_KEYS}


def content_hash(payload_json: str) -> str:
    return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()


def store_raw_records(
    con: "duckdb.DuckDBPyConnection",
    source: str,
    records: Iterable[RawRecord],
) -> dict[str, int]:
    """Upsert records into ``haz_raw_<source>``; return counts.

    Idempotent by construction: identical content collides on
    ``UNIQUE (record_id, content_hash)`` and is ignored, while revised
    content appends a new row.
    """

    table = raw_table_name(source)
    ensure_haz_schema(con)

    before = int(con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    seen = 0
    for record in records:
        # Strip volatile keys from the STORED payload, not just from the hash
        # input: hashing a stripped copy while storing an unstripped one would
        # make payload_json and content_hash disagree, and the surviving row's
        # fetched_at would become a permanently frozen lie.
        payload_json = json.dumps(
            stable_payload(record.payload), separators=(",", ":"), sort_keys=True
        )
        con.execute(
            f"""
            INSERT OR IGNORE INTO {table}
                (record_id, iso3, ym, hazard, payload_json, content_hash, source_url)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                str(record.record_id),
                record.iso3,
                record.ym,
                record.hazard,
                payload_json,
                content_hash(payload_json),
                record.source_url,
            ],
        )
        seen += 1

    total = int(con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    inserted = total - before
    LOG.info(
        "[%s] stored %d records (%d new revision rows; %s now %d rows)",
        source,
        seen,
        inserted,
        table,
        total,
    )
    return {"records": seen, "inserted": inserted, "rows_total": total}


def load_raw_records(
    con: "duckdb.DuckDBPyConnection",
    source: str,
    *,
    iso3: str | None = None,
    hazard: str | None = None,
    ym: str | None = None,
) -> list[dict[str, Any]]:
    """Newest revision of every cached record, optionally filtered.

    Each returned dict is the stored payload plus the cache's own
    provenance fields under ``_record_id`` / ``_source_url`` /
    ``_retrieved_at`` (underscore-prefixed so they cannot collide with a
    source's own field names).
    """

    table = raw_table_name(source)
    ensure_haz_schema(con)

    where: list[str] = []
    params: list[Any] = []
    if iso3:
        where.append("iso3 = ?")
        params.append(iso3.upper())
    if hazard:
        where.append("hazard = ?")
        params.append(hazard)
    if ym:
        where.append("ym = ?")
        params.append(ym)
    clause = f"WHERE {' AND '.join(where)}" if where else ""

    rows = con.execute(
        f"""
        WITH latest AS (
            SELECT record_id, payload_json, source_url, retrieved_at,
                   ROW_NUMBER() OVER (
                       PARTITION BY record_id ORDER BY retrieved_at DESC
                   ) AS rn
            FROM {table}
            {clause}
        )
        SELECT record_id, payload_json, source_url, retrieved_at
        FROM latest WHERE rn = 1
        ORDER BY record_id
        """,
        params,
    ).fetchall()

    out: list[dict[str, Any]] = []
    for record_id, payload_json, source_url, retrieved_at in rows:
        payload = json.loads(payload_json)
        payload["_record_id"] = record_id
        payload["_source_url"] = source_url
        payload["_retrieved_at"] = (
            retrieved_at.isoformat()
            if isinstance(retrieved_at, dt.datetime)
            else str(retrieved_at)
        )
        out.append(payload)
    return out


def raw_store_summary(
    con: "duckdb.DuckDBPyConnection", source: str, hazard: str | None = None
) -> dict[str, Any]:
    """Provenance summary of one raw cache (for evidence records)."""

    table = raw_table_name(source)
    ensure_haz_schema(con)
    clause, params = ("WHERE hazard = ?", [hazard]) if hazard else ("", [])
    row = con.execute(
        f"""
        SELECT COUNT(DISTINCT record_id), COUNT(*), MAX(ym),
               CAST(MAX(retrieved_at) AS VARCHAR)
        FROM {table} {clause}
        """,
        params,
    ).fetchone()
    urls = [
        r[0]
        for r in con.execute(
            f"SELECT DISTINCT source_url FROM {table} {clause} ORDER BY 1", params
        ).fetchall()
        if r[0]
    ]
    return {
        "source": source,
        "total_records": int(row[0] or 0),
        "total_revision_rows": int(row[1] or 0),
        "max_ym": row[2],
        "last_retrieved_at": row[3],
        "source_urls": urls,
    }


def parse_number(value: Any) -> float | None:
    """Parse a source-stated count into a float, or None.

    Deliberately strict about sign: a negative people-affected count is
    malformed, not a small one, and must not enter the ladder.
    """

    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        value = value.strip().replace(",", "").replace(" ", "")
        if not value:
            return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):  # NaN / inf
        return None
    return None if number < 0 else number


def parse_date(value: Any) -> dt.date | None:
    """Parse the date shapes the ladder sources use, or None."""

    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("Z", "+00:00")
    try:
        return dt.datetime.fromisoformat(text).date()
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%Y-%m-%dT%H:%M:%S"):
        try:
            return dt.datetime.strptime(text[: len(fmt) + 4], fmt).date()
        except ValueError:
            continue
    return None
