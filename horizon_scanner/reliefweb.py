# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Fetch, store, load, and format ReliefWeb humanitarian reports.

Provides a self-contained data connector that:

1. :func:`fetch_reliefweb_reports` — pulls recent reports from the ReliefWeb
   API for a given country (by ISO3 code).
2. :func:`store_reliefweb_reports` — upserts them into the Pythia DuckDB
   ``reliefweb_reports`` table.
3. :func:`load_reliefweb_reports` — reads them back from DuckDB.
4. :func:`format_reliefweb_for_prompt` — renders them as a text block for
   RC / Triage prompt injection.
5. :func:`format_reliefweb_for_spd` — compact format for SPD prompts.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import date, datetime, timedelta, timezone
from html.parser import HTMLParser
from typing import Any, Optional

import requests

log = logging.getLogger(__name__)

_STALENESS_DAYS = 60  # days_back (45) + 15-day grace
_DEFAULT_DAYS_BACK = 45
_DEFAULT_MAX_REPORTS = 15
_BODY_EXCERPT_LENGTH = 500
_API_BASE_URL = "https://api.reliefweb.int/v2"
_MAX_PROMPT_REPORTS = 10


# ---------------------------------------------------------------------------
# HTML stripping helper
# ---------------------------------------------------------------------------

class _HTMLStripper(HTMLParser):
    """Minimal HTML tag stripper using the stdlib parser."""

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self._parts.append(data)

    def get_text(self) -> str:
        return "".join(self._parts)


def _strip_html(html: str) -> str:
    """Remove HTML tags and return plain text."""
    stripper = _HTMLStripper()
    stripper.feed(html)
    return stripper.get_text()


# ---------------------------------------------------------------------------
# DB URL resolution (same pattern as conflict_forecasts.py)
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
# API fetcher
# ---------------------------------------------------------------------------

def fetch_reliefweb_reports(
    iso3: str,
    days_back: int = _DEFAULT_DAYS_BACK,
    max_reports: int = _DEFAULT_MAX_REPORTS,
    appname: str = "UNICEF-Resolver-P1L1T6",
) -> list[dict[str, Any]]:
    """Fetch recent humanitarian reports from the ReliefWeb API.

    Parameters
    ----------
    iso3 : str
        ISO 3166-1 alpha-3 country code (e.g. ``"SDN"``).
    days_back : int
        How many days of history to retrieve (default 45).
    max_reports : int
        Maximum number of reports to return (default 15).
    appname : str
        Application name sent in the ``appname`` query parameter.

    Returns
    -------
    list[dict]
        Normalized report dicts ready for :func:`store_reliefweb_reports`,
        or an empty list on failure.
    """
    since = datetime.now(timezone.utc) - timedelta(days=days_back)
    since_iso = since.strftime("%Y-%m-%dT00:00:00+00:00")

    payload = {
        "filter": {
            "operator": "AND",
            "conditions": [
                {"field": "primary_country.iso3", "value": iso3.upper()},
                {"field": "date.created", "value": {"from": since_iso}},
            ],
        },
        "fields": {
            "include": [
                "title",
                "date.created",
                "source.name",
                "primary_country.iso3",
                "disaster_type.name",
                "theme.name",
                "body",
                "url",
            ],
        },
        "sort": ["date.created:desc"],
        "limit": max_reports,
    }

    url = f"{_API_BASE_URL}/reports"
    headers = {"Accept": "application/json"}
    params = {"appname": appname}

    for attempt in range(2):
        try:
            resp = requests.post(
                url, json=payload, headers=headers, params=params, timeout=30,
            )
            if resp.status_code == 429 and attempt == 0:
                log.warning("ReliefWeb 429 — retrying after 2 s …")
                time.sleep(2)
                continue
            resp.raise_for_status()
            break
        except requests.RequestException as exc:
            if attempt == 0 and isinstance(exc, requests.HTTPError) and exc.response is not None and exc.response.status_code == 429:
                log.warning("ReliefWeb 429 — retrying after 2 s …")
                time.sleep(2)
                continue
            log.warning("ReliefWeb API request failed for %s: %s", iso3, exc)
            return []
    else:
        return []

    data = resp.json().get("data", [])
    now_iso = datetime.now(timezone.utc).isoformat()
    reports: list[dict[str, Any]] = []

    for item in data:
        fields = item.get("fields", {})

        # Strip HTML and truncate body.
        raw_body = fields.get("body", "") or ""
        plain_body = _strip_html(raw_body).strip()
        if len(plain_body) > _BODY_EXCERPT_LENGTH:
            plain_body = plain_body[:_BODY_EXCERPT_LENGTH].rsplit(" ", 1)[0] + " …"

        sources = [s.get("name", "") for s in (fields.get("source") or [])]
        disaster_types = [d.get("name", "") for d in (fields.get("disaster_type") or [])]
        themes = [t.get("name", "") for t in (fields.get("theme") or [])]

        # date.created is nested: fields -> date -> created
        date_obj = fields.get("date") or {}
        published = date_obj.get("created", "") if isinstance(date_obj, dict) else ""

        reports.append({
            "report_id": item.get("id"),
            "iso3": iso3.upper(),
            "title": fields.get("title", ""),
            "published_date": published,
            "sources": json.dumps(sources),
            "disaster_types": json.dumps(disaster_types),
            "themes": json.dumps(themes),
            "body_excerpt": plain_body,
            "url": fields.get("url", ""),
            "fetched_at": now_iso,
        })

    log.info("Fetched %d ReliefWeb reports for %s", len(reports), iso3)
    return reports


# ---------------------------------------------------------------------------
# DuckDB storage
# ---------------------------------------------------------------------------

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS reliefweb_reports (
    iso3           VARCHAR NOT NULL,
    report_id      INTEGER NOT NULL,
    title          VARCHAR,
    published_date VARCHAR,
    sources        VARCHAR,
    disaster_types VARCHAR,
    themes         VARCHAR,
    body_excerpt   VARCHAR,
    url            VARCHAR,
    fetched_at     VARCHAR,
    PRIMARY KEY (iso3, report_id)
);
"""


def store_reliefweb_reports(
    iso3: str,
    reports: list[dict[str, Any]],
    db_url: str | None = None,
) -> None:
    """Upsert ReliefWeb reports into the Pythia DuckDB database.

    Parameters
    ----------
    iso3 : str
        Country ISO3 code (for logging only — the actual iso3 is in each
        report dict).
    reports : list[dict]
        Report dicts as returned by :func:`fetch_reliefweb_reports`.
    db_url : str, optional
        DuckDB connection URL. Resolved via :func:`_db_url` if not given.
    """
    if not reports:
        return

    try:
        from resolver.db.duckdb_io import get_db
    except Exception:
        log.debug("DuckDB helpers unavailable — skipping ReliefWeb store.")
        return

    db_url = db_url or _db_url()

    try:
        con = get_db(db_url)
    except Exception:
        log.debug("Could not connect to DuckDB at %s", db_url)
        return

    try:
        con.execute(_CREATE_TABLE_SQL)

        for r in reports:
            con.execute(
                """
                INSERT OR REPLACE INTO reliefweb_reports
                    (iso3, report_id, title, published_date, sources,
                     disaster_types, themes, body_excerpt, url, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    r["iso3"],
                    r["report_id"],
                    r["title"],
                    r["published_date"],
                    r["sources"],
                    r["disaster_types"],
                    r["themes"],
                    r["body_excerpt"],
                    r["url"],
                    r["fetched_at"],
                ],
            )

        log.info("Stored %d ReliefWeb reports for %s", len(reports), iso3)
    except Exception as exc:
        log.warning("Failed to store ReliefWeb reports for %s: %s", iso3, exc)
    finally:
        pass  # Let the resolver connection cache manage lifecycle.


# ---------------------------------------------------------------------------
# DuckDB loader
# ---------------------------------------------------------------------------

def load_reliefweb_reports(
    iso3: str,
    max_reports: int = _DEFAULT_MAX_REPORTS,
    db_url: str | None = None,
) -> list[dict[str, Any]] | None:
    """Load recent ReliefWeb reports for a country from DuckDB.

    Parameters
    ----------
    iso3 : str
        ISO 3166-1 alpha-3 country code.
    max_reports : int
        Maximum reports to return (default 15).
    db_url : str, optional
        DuckDB connection URL. Resolved via :func:`_db_url` if not given.

    Returns
    -------
    list[dict] or None
        List of report dicts ordered by ``published_date`` descending,
        or *None* if no data is available.
    """
    try:
        from resolver.db.duckdb_io import get_db
    except Exception:
        log.debug("DuckDB helpers unavailable — skipping ReliefWeb load.")
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
        if "reliefweb_reports" not in tables:
            return None

        rows = con.execute(
            """
            SELECT iso3, report_id, title, published_date, sources,
                   disaster_types, themes, body_excerpt, url, fetched_at
            FROM reliefweb_reports
            WHERE iso3 = ?
            ORDER BY published_date DESC
            LIMIT ?
            """,
            [iso3.upper(), max_reports],
        ).fetchall()

        if not rows:
            return None

        columns = [
            "iso3", "report_id", "title", "published_date", "sources",
            "disaster_types", "themes", "body_excerpt", "url", "fetched_at",
        ]
        return [dict(zip(columns, row)) for row in rows]

    except Exception as exc:
        log.warning("Failed to load ReliefWeb reports for %s: %s", iso3, exc)
        return None
    finally:
        pass  # Let the resolver connection cache manage lifecycle.


# ---------------------------------------------------------------------------
# Prompt formatters
# ---------------------------------------------------------------------------

def _parse_date_short(iso_str: str) -> str:
    """Extract YYYY-MM-DD from an ISO datetime string."""
    if not iso_str:
        return "unknown"
    return iso_str[:10]


def _top_themes(reports: list[dict[str, Any]], n: int = 5) -> list[str]:
    """Return the *n* most frequent themes across all reports."""
    counts: dict[str, int] = {}
    for r in reports:
        try:
            themes = json.loads(r.get("themes", "[]"))
        except (json.JSONDecodeError, TypeError):
            themes = []
        for t in themes:
            if t:
                counts[t] = counts.get(t, 0) + 1
    return [t for t, _ in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:n]]


def _is_stale(reports: list[dict[str, Any]]) -> bool:
    """Return True if the newest report is older than ``_STALENESS_DAYS``."""
    if not reports:
        return True
    newest = reports[0].get("published_date", "")
    if not newest:
        return True
    try:
        pub = date.fromisoformat(_parse_date_short(newest))
        return (date.today() - pub).days > _STALENESS_DAYS
    except (ValueError, TypeError):
        return True


def format_reliefweb_for_prompt(
    reports: list[dict[str, Any]] | None,
) -> str:
    """Format ReliefWeb reports as a text block for RC / Triage prompts.

    Returns an empty string if no data is available.
    """
    if not reports:
        return ""

    iso3 = reports[0].get("iso3", "???")
    themes = _top_themes(reports)
    stale_note = " [WARNING: DATA MAY BE STALE]" if _is_stale(reports) else ""

    parts: list[str] = []
    parts.append(
        f"RECENT RELIEFWEB REPORTS ({iso3}, last {_DEFAULT_DAYS_BACK} days):{stale_note}"
    )

    themes_str = ", ".join(themes) if themes else "none identified"
    parts.append(f"{len(reports)} reports found. Key themes: {themes_str}.")
    parts.append("")

    display_reports = reports[:_MAX_PROMPT_REPORTS]
    for idx, r in enumerate(display_reports, 1):
        dt = _parse_date_short(r.get("published_date", ""))
        title = r.get("title", "Untitled")
        try:
            sources = ", ".join(json.loads(r.get("sources", "[]")))
        except (json.JSONDecodeError, TypeError):
            sources = ""
        try:
            dtypes = ", ".join(json.loads(r.get("disaster_types", "[]")))
        except (json.JSONDecodeError, TypeError):
            dtypes = ""

        parts.append(f"{idx}. [{dt}] {title}")
        detail_parts = []
        if sources:
            detail_parts.append(f"Sources: {sources}")
        if dtypes:
            detail_parts.append(f"Types: {dtypes}")
        if detail_parts:
            parts.append(f"   {'. '.join(detail_parts)}.")

        excerpt = r.get("body_excerpt", "")
        if excerpt:
            parts.append(f"   {excerpt}")
        parts.append("")

    remainder = len(reports) - _MAX_PROMPT_REPORTS
    if remainder > 0:
        parts.append(f"(+{remainder} additional reports not shown)")
        parts.append("")

    parts.append(
        "These reports provide recent situational context from humanitarian "
        "organizations. Use them to ground your assessment in current events, "
        "not training-data memory."
    )

    return "\n".join(parts)


def format_reliefweb_for_spd(
    reports: list[dict[str, Any]] | None,
) -> str:
    """Format ReliefWeb reports compactly for SPD prompts.

    Returns an empty string if no data is available.
    """
    if not reports:
        return ""

    iso3 = reports[0].get("iso3", "???")
    stale_note = " [WARNING: DATA MAY BE STALE]" if _is_stale(reports) else ""

    parts: list[str] = [f"RECENT REPORTS ({iso3}):{stale_note}"]
    for r in reports:
        dt = _parse_date_short(r.get("published_date", ""))
        title = r.get("title", "Untitled")
        try:
            dtypes = ", ".join(json.loads(r.get("disaster_types", "[]")))
        except (json.JSONDecodeError, TypeError):
            dtypes = ""
        type_note = f" ({dtypes})" if dtypes else ""
        parts.append(f"- {dt}: {title}{type_note}")

    return "\n".join(parts)
