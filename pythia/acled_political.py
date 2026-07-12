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
    "|iso3|country"
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


def _iso_numeric(iso3: str) -> Optional[int]:
    """Numeric ISO 3166-1 code for the ACLED API's documented `iso` filter."""
    try:
        import pycountry

        country = pycountry.countries.get(alpha_3=(iso3 or "").upper())
        return int(country.numeric) if country else None
    except Exception:
        return None


# Candidate keys the ACLED API uses for the ISO3 code, beyond the ones
# `resolve_iso3` already understands. The live API does NOT reliably return
# the code under the literal key ``iso3`` — it may use the HXL tag
# ``#country+code`` or the underscore variant ``country_iso3`` — so a single
# ``ev.get("iso3")`` read attributes nothing and every event is discarded.
# Mirror the robust extraction the fatalities connector uses
# (resolver/ingestion/acled_client.py).
_ISO3_EXTRA_KEYS = ("iso3", "country_iso3", "#country+code")


def _event_iso3(ev: dict) -> str:
    """Best-effort ISO3 for one raw ACLED event.

    Tries the explicit ISO3 response keys, then falls back to resolving from
    the ``country`` name (which the request already asks for in ``fields``).
    Returns "" when nothing resolves.
    """
    from resolver.ingestion.utils.iso_normalize import resolve_iso3, to_iso3

    for key in _ISO3_EXTRA_KEYS:
        raw = ev.get(key)
        if raw:
            code = to_iso3(str(raw))
            if code:
                return code
    # resolve_iso3 also covers ISO3/CountryISO3 and the country-name fallback.
    resolved, _reason = resolve_iso3(ev, name_keys=("country",))
    return (resolved or "").strip().upper()


def _filter_events_to_country(events: list[dict], iso3: str) -> list[dict]:
    """Keep only events whose OWN returned country matches the requested one.

    The ACLED API silently ignores unsupported filter params: a previous
    version passed `iso3=` (not a filter) and received the same ~50 GLOBAL
    events for every country, which were then stored stamped with the
    requested iso3 — injecting Iranian/Ecuadorian events into e.g. Somalia's
    prompts. Never trust the request; attribute by the event's OWN country,
    resolved robustly (the live API's ISO3 key is not always ``iso3``).
    """
    iso3_up = (iso3 or "").upper()
    matched = [
        ev for ev in events
        if _event_iso3(ev) == iso3_up
    ]
    if events and not matched:
        log.warning(
            "ACLED political: 0/%d returned events resolve to iso3=%s — "
            "storing nothing rather than misattributed events.",
            len(events), iso3_up,
        )
    elif len(matched) < len(events):
        log.info(
            "ACLED political: kept %d/%d events matching iso3=%s",
            len(matched), len(events), iso3_up,
        )
    return matched


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
        # `iso` (numeric ISO 3166-1) is ACLED's documented country filter.
        # `iso3` is a RESPONSE field only — passing it as a filter is silently
        # ignored and returns global events (see _filter_events_to_country).
        "event_date": f"{start_date:%Y-%m-%d}|{end_date:%Y-%m-%d}",
        "event_date_where": "BETWEEN",
        "event_type": _POLITICAL_EVENT_TYPES,
        "fields": _ACLED_FIELDS,
        "limit": max_events,
        "_format": "json",
    }
    iso_num = _iso_numeric(iso3)
    if iso_num is not None:
        params["iso"] = iso_num
    else:
        log.warning(
            "ACLED political: could not resolve numeric ISO code for %s — "
            "no server-side country filter; relying on client-side "
            "attribution only.",
            iso3,
        )

    headers = {"Authorization": f"Bearer {token}"}

    raw_events = _fetch_with_retry(params, headers)
    if not raw_events:
        return []

    # Never trust the request-side filter — attribute by returned iso3.
    raw_events = _filter_events_to_country(raw_events, iso3)
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
            "iso3": _event_iso3(ev),
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

        iso3_up = iso3.upper()
        for ev in events:
            # Guard against misattribution: never store an event under a
            # country its own returned iso3 contradicts.
            ev_iso3 = str(ev.get("iso3") or "").strip().upper()
            if ev_iso3 and ev_iso3 != iso3_up:
                log.warning(
                    "ACLED political: dropping event %s (iso3=%s) requested "
                    "under %s — misattribution guard.",
                    ev.get("event_id"), ev_iso3, iso3_up,
                )
                continue
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


# ---------------------------------------------------------------------------
# Contamination purge (one-time self-heal)
# ---------------------------------------------------------------------------

def purge_contaminated_events() -> int:
    """Self-heal the table if cross-country contamination is detected.

    The pre-July-2026 fetcher passed ``iso3`` as an (unsupported, silently
    ignored) ACLED filter param, so every country stored the SAME global
    events stamped with its own iso3 — the signature is identical event_ids
    appearing under multiple iso3 values. When detected, the whole table is
    wiped so the fixed fetcher can repopulate it with correctly attributed
    events. Returns the number of rows deleted (0 when clean or absent).
    """
    try:
        from pythia.db.schema import connect
    except Exception:
        return 0

    try:
        con = connect(read_only=False)
    except Exception:
        return 0

    try:
        try:
            dup = con.execute(
                """
                SELECT COUNT(*) FROM (
                    SELECT event_id
                    FROM acled_political_events
                    WHERE event_id IS NOT NULL AND event_id != ''
                    GROUP BY event_id
                    HAVING COUNT(DISTINCT iso3) > 1
                )
                """
            ).fetchone()
        except Exception:
            return 0  # table missing — nothing to heal
        if not dup or int(dup[0] or 0) == 0:
            return 0

        total = con.execute(
            "SELECT COUNT(*) FROM acled_political_events"
        ).fetchone()
        n_rows = int(total[0] or 0) if total else 0
        con.execute("DELETE FROM acled_political_events")
        log.warning(
            "ACLED political: purged %d rows — %d event_ids were stored under "
            "multiple countries (pre-fix global-fetch contamination).",
            n_rows, int(dup[0]),
        )
        return n_rows
    except Exception as exc:  # noqa: BLE001
        log.warning("ACLED political contamination purge failed: %s", exc)
        return 0
    finally:
        con.close()
