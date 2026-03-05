# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""ACLED political events connector.

Fetches event-level political data (strategic developments, protests, riots)
from the ACLED API and stores them in DuckDB for prompt injection.  This is
complementary to the existing aggregate ``acled_summary`` — it provides the
specific events behind those trailing statistics.

Public API
----------
- :func:`fetch_acled_political_events` — fetch from ACLED API
- :func:`store_acled_political_events` — upsert into DuckDB
- :func:`load_acled_political_events` — load from DuckDB
- :func:`format_political_events_for_prompt` — RC / triage prompt block
- :func:`format_political_events_for_spd` — compact SPD prompt block
"""

from __future__ import annotations

import logging
import os
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

import requests

log = logging.getLogger(__name__)

ACLED_API_BASE_URL = "https://acleddata.com/api/acled/read"

_POLITICAL_EVENT_TYPES = "Strategic developments|Protests|Riots"

_ACLED_FIELDS = (
    "event_id_cnty|event_date|event_type|sub_event_type"
    "|actor1|actor2|admin1|location|notes|fatalities|source"
)

_ALWAYS_INCLUDE_SUB_TYPES = frozenset({
    "Peace agreement",
    "Arrests",
    "Change to group/activity",
    "Headquarters or base established",
    "Non-violent transfer of territory",
    "Other",
})

_NOTES_MAX_LEN = 300
_MAX_RETRIES = 2


# ---------------------------------------------------------------------------
# DB URL resolution (mirrors horizon_scanner/conflict_forecasts.py)
# ---------------------------------------------------------------------------

def _db_url() -> str:
    """Resolve the Pythia DuckDB URL."""
    url = os.getenv("PYTHIA_DB_URL", "").strip()
    if url:
        return url
    try:
        from pythia.config import load as load_config
        cfg = load_config()
        url = str((cfg.get("app") or {}).get("db_url", "")).strip()
        if url:
            return url
    except Exception:
        pass
    from resolver.db.duckdb_io import DEFAULT_DB_URL
    return DEFAULT_DB_URL


# ---------------------------------------------------------------------------
# Significance filtering
# ---------------------------------------------------------------------------

def _apply_significance_filter(events: list[dict]) -> list[dict]:
    """Filter ACLED events to those most relevant for political context.

    Parameters
    ----------
    events
        Raw event dicts from the ACLED API.

    Returns
    -------
    list[dict]
        Filtered subset of significant events.
    """
    protest_count = sum(
        1 for e in events if e.get("event_type") == "Protests"
    )
    include_protests = protest_count >= 5

    result: list[dict] = []
    for ev in events:
        sub = ev.get("sub_event_type", "")
        etype = ev.get("event_type", "")
        fatalities = _safe_int(ev.get("fatalities"))

        # Always include high-signal sub-event types
        if sub in _ALWAYS_INCLUDE_SUB_TYPES:
            result.append(ev)
            continue

        # Always include anything with fatalities
        if fatalities > 0:
            result.append(ev)
            continue

        # Protests: only if protest wave (5+ in period)
        if etype == "Protests" and include_protests:
            result.append(ev)
            continue

        # Riots: only with fatalities (already handled above)
        # Skip routine small protests and non-fatal riots

    return result


# ---------------------------------------------------------------------------
# Fetcher
# ---------------------------------------------------------------------------

def fetch_acled_political_events(
    iso3: str,
    days_back: int = 60,
    max_events: int = 50,
) -> list[dict]:
    """Fetch recent political events from the ACLED API.

    Parameters
    ----------
    iso3
        ISO 3166-1 alpha-3 country code (e.g. ``"SDN"``).
    days_back
        How many days of history to fetch.
    max_events
        Maximum number of events to return from the API.

    Returns
    -------
    list[dict]
        Filtered, sorted event dicts ready for storage.  Returns an empty
        list on auth or network errors.
    """
    try:
        from resolver.ingestion.acled_auth import get_access_token
        token = get_access_token()
    except Exception as exc:
        log.warning(
            "ACLED auth unavailable — skipping political events fetch: %s", exc,
        )
        return []

    end_date = date.today()
    start_date = end_date - timedelta(days=days_back)

    params: dict[str, Any] = {
        "iso3": iso3.upper(),
        "event_date": f"{start_date:%Y-%m-%d}|{end_date:%Y-%m-%d}",
        "event_date_where": "BETWEEN",
        "event_type": _POLITICAL_EVENT_TYPES,
        "fields": _ACLED_FIELDS,
        "limit": max_events,
        "_format": "json",
    }

    headers = {"Authorization": f"Bearer {token}"}

    raw_events = _fetch_with_retry(params, headers)
    if not raw_events:
        return []

    filtered = _apply_significance_filter(raw_events)

    # Sort by date descending
    filtered.sort(key=lambda e: e.get("event_date", ""), reverse=True)

    # Normalise into storage-ready dicts
    result: list[dict] = []
    for ev in filtered:
        notes = str(ev.get("notes") or "")
        if len(notes) > _NOTES_MAX_LEN:
            notes = notes[:_NOTES_MAX_LEN].rsplit(" ", 1)[0] + "..."

        result.append({
            "event_id": ev.get("event_id_cnty", ""),
            "event_date": ev.get("event_date", ""),
            "event_type": ev.get("event_type", ""),
            "sub_event_type": ev.get("sub_event_type", ""),
            "actor1": ev.get("actor1", ""),
            "actor2": ev.get("actor2", ""),
            "admin1": ev.get("admin1", ""),
            "location": ev.get("location", ""),
            "notes_excerpt": notes,
            "fatalities": _safe_int(ev.get("fatalities")),
        })

    return result


def _fetch_with_retry(
    params: dict[str, Any],
    headers: dict[str, str],
) -> list[dict]:
    """GET the ACLED API with simple retry on transient errors."""
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = requests.get(
                ACLED_API_BASE_URL,
                params=params,
                headers=headers,
                timeout=60,
            )
        except requests.RequestException as exc:
            log.warning(
                "ACLED political events network error (attempt %d/%d): %s",
                attempt, _MAX_RETRIES, exc,
            )
            if attempt < _MAX_RETRIES:
                time.sleep(2 ** attempt)
                continue
            return []

        if resp.status_code == 429 or resp.status_code >= 500:
            log.warning(
                "ACLED political events HTTP %d (attempt %d/%d)",
                resp.status_code, attempt, _MAX_RETRIES,
            )
            if attempt < _MAX_RETRIES:
                time.sleep(2 ** attempt)
                continue
            return []

        if resp.status_code != 200:
            log.warning("ACLED political events HTTP %d", resp.status_code)
            return []

        try:
            payload = resp.json()
        except ValueError:
            log.warning("ACLED political events response was not valid JSON")
            return []

        data = payload.get("data") or payload.get("results") or []
        return data if isinstance(data, list) else []

    return []


# ---------------------------------------------------------------------------
# DuckDB storage
# ---------------------------------------------------------------------------

def store_acled_political_events(
    iso3: str,
    events: list[dict],
    db_url: str | None = None,
) -> None:
    """Upsert political events into DuckDB.

    Parameters
    ----------
    iso3
        ISO 3166-1 alpha-3 country code.
    events
        Event dicts as returned by :func:`fetch_acled_political_events`.
    db_url
        Optional DuckDB URL override.
    """
    if not events:
        return

    try:
        from pythia.db.schema import connect, ensure_schema
    except Exception:
        log.debug("Pythia DB helpers unavailable — skipping store.")
        return

    try:
        con = connect(read_only=False)
    except Exception:
        log.warning("Could not connect to DuckDB for political events store.")
        return

    try:
        ensure_schema(con)
        now = datetime.now(timezone.utc).isoformat()

        for ev in events:
            con.execute(
                """
                INSERT OR REPLACE INTO acled_political_events
                    (iso3, event_id, event_date, event_type, sub_event_type,
                     actor1, actor2, admin1, location, notes_excerpt,
                     fatalities, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    iso3.upper(),
                    ev.get("event_id", ""),
                    ev.get("event_date", ""),
                    ev.get("event_type", ""),
                    ev.get("sub_event_type", ""),
                    ev.get("actor1", ""),
                    ev.get("actor2", ""),
                    ev.get("admin1", ""),
                    ev.get("location", ""),
                    ev.get("notes_excerpt", ""),
                    ev.get("fatalities", 0),
                    now,
                ],
            )
    except Exception as exc:
        log.warning("Failed to store political events for %s: %s", iso3, exc)
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_acled_political_events(
    iso3: str,
    max_events: int = 30,
    db_url: str | None = None,
) -> list[dict] | None:
    """Load political events from DuckDB.

    Parameters
    ----------
    iso3
        ISO 3166-1 alpha-3 country code.
    max_events
        Maximum events to return.
    db_url
        Optional DuckDB URL override.

    Returns
    -------
    list[dict] | None
        Event dicts ordered by date descending, or ``None`` if no data.
    """
    try:
        from resolver.db.duckdb_io import get_db, close_db
    except Exception:
        log.debug("DuckDB helpers unavailable — skipping political events load.")
        return None

    db_url = db_url or _db_url()

    try:
        con = get_db(db_url)
    except Exception:
        log.debug("Could not connect to DuckDB at %s", db_url)
        return None

    try:
        tables = [
            r[0]
            for r in con.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'main'"
            ).fetchall()
        ]
        if "acled_political_events" not in tables:
            return None

        rows = con.execute(
            """
            SELECT event_id, event_date, event_type, sub_event_type,
                   actor1, actor2, admin1, location, notes_excerpt, fatalities
            FROM acled_political_events
            WHERE iso3 = ?
            ORDER BY event_date DESC
            LIMIT ?
            """,
            [iso3.upper(), max_events],
        ).fetchall()
    except Exception as exc:
        log.warning("Failed to load political events for %s: %s", iso3, exc)
        return None
    finally:
        close_db(con)

    if not rows:
        return None

    columns = [
        "event_id", "event_date", "event_type", "sub_event_type",
        "actor1", "actor2", "admin1", "location", "notes_excerpt", "fatalities",
    ]
    return [dict(zip(columns, row)) for row in rows]


# ---------------------------------------------------------------------------
# Prompt formatters
# ---------------------------------------------------------------------------

def format_political_events_for_prompt(
    events: list[dict] | None,
    iso3: str = "",
) -> str:
    """Format political events as a text block for RC / triage prompts.

    Parameters
    ----------
    events
        Event dicts from :func:`load_acled_political_events`.
    iso3
        Country code for the header.

    Returns
    -------
    str
        Formatted prompt section, or empty string if no data.
    """
    if not events:
        return ""

    total = len(events)
    display = events[:15]

    strategic = [e for e in display if e.get("event_type") == "Strategic developments"]
    protests_riots = [e for e in display if e.get("event_type") in ("Protests", "Riots")]

    parts: list[str] = []
    iso_label = f" ({iso3})" if iso3 else ""
    parts.append(f"RECENT POLITICAL EVENTS{iso_label}, last 60 days):")
    parts.append(f"Source: ACLED. {total} significant events identified.")

    if strategic:
        parts.append("\nStrategic developments:")
        for ev in strategic:
            actors = _format_actors(ev)
            actor_str = f" (Actors: {actors})" if actors else ""
            parts.append(
                f"- [{ev.get('event_date', '?')}] "
                f"{ev.get('sub_event_type', 'Unknown')}: "
                f"{ev.get('notes_excerpt', '')}{actor_str}"
            )

    if protests_riots:
        parts.append("\nProtests/Riots:")
        for ev in protests_riots:
            fat = ev.get("fatalities", 0)
            fat_str = f" ({fat} killed)" if fat else ""
            loc = ev.get("location") or ev.get("admin1") or ""
            loc_str = f" in {loc}" if loc else ""
            parts.append(
                f"- [{ev.get('event_date', '?')}] "
                f"{ev.get('event_type', 'Protest')}{loc_str}: "
                f"{ev.get('notes_excerpt', '')}{fat_str}"
            )

    if total > 15:
        parts.append(f"\n({total - 15} additional events not shown.)")

    parts.append(
        "\nThese events supplement the aggregate ACLED statistics. Use them to "
        "identify specific triggers, political transitions, or "
        "escalation/de-escalation signals."
    )

    return "\n".join(parts)


def format_political_events_for_spd(
    events: list[dict] | None,
    iso3: str = "",
) -> str:
    """Compact political events block for SPD prompts.

    Parameters
    ----------
    events
        Event dicts from :func:`load_acled_political_events`.
    iso3
        Country code for the header.

    Returns
    -------
    str
        Compact prompt section, or empty string if no data.
    """
    if not events:
        return ""

    display = events[:10]
    iso_label = f" ({iso3})" if iso3 else ""
    parts: list[str] = [f"KEY POLITICAL EVENTS{iso_label}:"]

    for ev in display:
        fat = ev.get("fatalities", 0)
        fat_str = f" [{fat} killed]" if fat else ""
        notes = ev.get("notes_excerpt", "")
        # Shorten for SPD — first 120 chars
        if len(notes) > 120:
            notes = notes[:120].rsplit(" ", 1)[0] + "..."
        parts.append(
            f"- {ev.get('event_date', '?')}: "
            f"{ev.get('sub_event_type', ev.get('event_type', '?'))} "
            f"— {notes}{fat_str}"
        )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_int(value: Any) -> int:
    """Convert a value to int, defaulting to 0."""
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _format_actors(ev: dict) -> str:
    """Build an actor string like 'Actor1 vs Actor2'."""
    a1 = (ev.get("actor1") or "").strip()
    a2 = (ev.get("actor2") or "").strip()
    if a1 and a2:
        return f"{a1} vs {a2}"
    return a1 or a2
