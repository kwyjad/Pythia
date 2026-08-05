# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""IBTrACS connector — NOAA best-track archive into ``haz_raw_ibtracs``.

Fetches the IBTrACS v04r01 CSV ("last3years" rolling file by default,
"ALL" for backcast — see ``cyclone.ibtracs`` in the rulebook) and stores
it in the generic raw cache shape from ``schema.py``: **one row per
storm**, ``record_id`` = the IBTrACS storm id (SID), the full track as
``payload_json``, and ``content_hash`` over the payload so a re-fetch of
identical content is a no-op while a revised best track appends a new
row (the revision trail ``haz_revisions`` audits). Detection reads the
newest revision per storm.

The payload keeps the columns detection needs, values unmodified apart
from longitude normalisation to [-180, 180] (IBTrACS mixes conventions
across basins). Wind is kept as BOTH ``usa_wind_kt`` (1-minute
sustained) and ``wmo_wind_kt`` (reporting agency, often 10-minute);
which one the trigger trusts is a rulebook decision
(``cyclone.wind_source_priority``), not an ingestion decision.

No authentication required.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import tempfile
from pathlib import Path
from typing import IO, TYPE_CHECKING

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from resolver.hazard_resolution.rulebook import (
    HAZARD_CODE_BY_RULEBOOK_NAME,
    Rulebook,
)
from resolver.hazard_resolution.schema import ensure_haz_schema

if TYPE_CHECKING:  # pragma: no cover - typing only
    import duckdb

LOG = logging.getLogger(__name__)

_HAZARD = HAZARD_CODE_BY_RULEBOOK_NAME["cyclone"]

# IBTrACS CSV columns we consume (the file has ~174; the rest are
# per-agency detail Phase 1 does not need).
_USECOLS = [
    "SID",
    "SEASON",
    "BASIN",
    "NAME",
    "ISO_TIME",
    "LAT",
    "LON",
    "WMO_WIND",
    "USA_WIND",
    "NATURE",
]

#: SQL snippet: newest revision per storm (content_hash revisions append).
_LATEST_REVISION_CTE = """
    WITH latest AS (
        SELECT record_id, payload_json, source_url, retrieved_at,
               ROW_NUMBER() OVER (
                   PARTITION BY record_id ORDER BY retrieved_at DESC
               ) AS rn
        FROM haz_raw_ibtracs
        WHERE hazard = ?
    )
    SELECT record_id, payload_json, source_url, retrieved_at
    FROM latest WHERE rn = 1
"""


def _build_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=2.0,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def ibtracs_url(rulebook: Rulebook, scope: str) -> str:
    """Resolve the IBTrACS CSV URL for a fetch scope."""
    template = str(rulebook.get("cyclone.ibtracs.url_template"))
    return template.format(scope=scope)


def parse_ibtracs_csv(source: str | Path | IO) -> pd.DataFrame:
    """Parse an IBTrACS v04 CSV into a typed track-point frame.

    Handles the file's two header rows (row 2 is units), coerces
    numerics, normalises longitude to [-180, 180], and drops rows
    missing the (sid, iso_time, lat, lon) essentials.
    """
    df = pd.read_csv(
        source,
        usecols=lambda c: c in _USECOLS,
        skiprows=[1],  # units row
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )
    missing = [c for c in _USECOLS if c not in df.columns]
    if missing:
        raise ValueError(f"IBTrACS CSV missing expected columns: {missing}")

    out = pd.DataFrame(
        {
            "sid": df["SID"].str.strip(),
            "season": pd.to_numeric(df["SEASON"], errors="coerce"),
            "basin": df["BASIN"].str.strip(),
            "storm_name": df["NAME"].str.strip(),
            "iso_time": pd.to_datetime(df["ISO_TIME"], errors="coerce"),
            "lat": pd.to_numeric(df["LAT"], errors="coerce"),
            "lon": pd.to_numeric(df["LON"], errors="coerce"),
            "usa_wind_kt": pd.to_numeric(df["USA_WIND"], errors="coerce"),
            "wmo_wind_kt": pd.to_numeric(df["WMO_WIND"], errors="coerce"),
            "nature": df["NATURE"].str.strip(),
        }
    )

    n_raw = len(out)
    out = out.dropna(subset=["iso_time", "lat", "lon"])
    out = out[out["sid"] != ""]
    # Normalise longitude to [-180, 180] (some basins report 0..360).
    out["lon"] = ((out["lon"] + 180.0) % 360.0) - 180.0
    out = out.drop_duplicates(subset=["sid", "iso_time"], keep="first")
    dropped = n_raw - len(out)
    if dropped:
        LOG.info("[ibtracs] dropped %d rows missing essentials or duplicated", dropped)
    return out.sort_values(["sid", "iso_time"]).reset_index(drop=True)


def fetch_ibtracs(
    rulebook: Rulebook,
    scope: str | None = None,
    session: requests.Session | None = None,
) -> tuple[pd.DataFrame, str]:
    """Download and parse the IBTrACS CSV for ``scope``.

    Returns ``(frame, source_url)``.  Raises ``requests.RequestException``
    on download failure — the caller decides whether stale stored data
    suffices (see the CLI's coverage handling).
    """
    scope = scope or str(rulebook.get("cyclone.ibtracs.default_scope"))
    url = ibtracs_url(rulebook, scope)
    timeout = float(rulebook.get("cyclone.ibtracs.request_timeout_sec"))
    session = session or _build_session()

    LOG.info("[ibtracs] fetching %s", url)
    # The ALL archive is ~300 MB — stream to a temp file, never into RAM.
    with tempfile.NamedTemporaryFile(suffix=".csv") as tmp:
        with session.get(url, timeout=timeout, stream=True) as resp:
            resp.raise_for_status()
            for chunk in resp.iter_content(chunk_size=1 << 20):
                tmp.write(chunk)
        tmp.flush()
        tmp.seek(0)
        df = parse_ibtracs_csv(tmp.name)

    LOG.info("[ibtracs] parsed %d track points (scope=%s)", len(df), scope)
    return df, url


def _num(value) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _storm_payload(sid: str, group: pd.DataFrame, scope: str) -> dict:
    points = [
        {
            "t": row.iso_time.strftime("%Y-%m-%d %H:%M:%S"),
            "lat": float(row.lat),
            "lon": float(row.lon),
            "usa_wind_kt": _num(row.usa_wind_kt),
            "wmo_wind_kt": _num(row.wmo_wind_kt),
            "nature": str(row.nature or ""),
        }
        for row in group.itertuples()
    ]
    season = _num(group["season"].iloc[0])
    return {
        "sid": sid,
        "season": int(season) if season is not None else None,
        "basin": str(group["basin"].iloc[0] or ""),
        "name": str(group["storm_name"].iloc[0] or ""),
        "first_point_time": points[0]["t"],
        "last_point_time": points[-1]["t"],
        "n_points": len(points),
        "source_scope": scope,
        "points": points,
    }


def store_ibtracs(
    con: "duckdb.DuckDBPyConnection", df: pd.DataFrame, scope: str, source_url: str
) -> dict[str, int]:
    """Store track points as per-storm rows in ``haz_raw_ibtracs``.

    Idempotent: ``content_hash`` covers the whole payload, and the
    ``UNIQUE (record_id, content_hash)`` constraint makes a re-store of
    identical content a no-op, while a storm whose best track was
    revised upstream appends a new revision row.

    Returns ``{"storms": ..., "inserted": ..., "rows_total": ...}``.
    """
    ensure_haz_schema(con)
    if df.empty:
        LOG.warning("[ibtracs] nothing to store (empty frame)")
        total = int(con.execute("SELECT COUNT(*) FROM haz_raw_ibtracs").fetchone()[0])
        return {"storms": 0, "inserted": 0, "rows_total": total}

    before = int(con.execute("SELECT COUNT(*) FROM haz_raw_ibtracs").fetchone()[0])
    inserted = 0
    storms = 0
    for sid, group in df.groupby("sid", sort=True):
        payload = _storm_payload(str(sid), group, scope)
        payload_json = json.dumps(payload, separators=(",", ":"))
        content_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        # ym anchors the storm to the month of its LAST point (a month
        # key like everywhere else in the resolver); detection filters on
        # the exact first/last point times inside the payload.
        ym = payload["last_point_time"][:7]
        con.execute(
            """
            INSERT OR IGNORE INTO haz_raw_ibtracs
                (record_id, iso3, ym, hazard, payload_json, content_hash, source_url)
            VALUES (?, NULL, ?, ?, ?, ?, ?)
            """,
            [str(sid), ym, _HAZARD, payload_json, content_hash, source_url],
        )
        storms += 1

    total = int(con.execute("SELECT COUNT(*) FROM haz_raw_ibtracs").fetchone()[0])
    inserted = total - before
    LOG.info(
        "[ibtracs] stored %d storms (%d new revision rows; table now %d rows)",
        storms,
        inserted,
        total,
    )
    return {"storms": storms, "inserted": inserted, "rows_total": total}


def load_track_points(
    con: "duckdb.DuckDBPyConnection", month_start_iso: str, month_end_iso: str
) -> pd.DataFrame:
    """Track points from storms whose track intersects the given window.

    Reads the newest revision of every storm whose payload's
    first/last point times overlap ``[month_start_iso, month_end_iso]``
    and explodes the payload back into a typed point frame.  The exact
    per-point month filter is the caller's job (a storm can span a month
    boundary and belongs to both months).
    """
    rows = con.execute(
        f"""
        SELECT record_id, payload_json FROM ({_LATEST_REVISION_CTE})
        WHERE json_extract_string(payload_json, '$.first_point_time') <= ?
          AND json_extract_string(payload_json, '$.last_point_time') >= ?
        """,
        [_HAZARD, f"{month_end_iso} 23:59:59", f"{month_start_iso} 00:00:00"],
    ).fetchall()

    records: list[dict] = []
    for _, payload_json in rows:
        payload = json.loads(payload_json)
        for point in payload["points"]:
            records.append(
                {
                    "sid": payload["sid"],
                    "storm_name": payload["name"],
                    "basin": payload["basin"],
                    "iso_time": point["t"],
                    "lat": point["lat"],
                    "lon": point["lon"],
                    "usa_wind_kt": point["usa_wind_kt"],
                    "wmo_wind_kt": point["wmo_wind_kt"],
                }
            )
    columns = [
        "sid", "storm_name", "basin", "iso_time", "lat", "lon",
        "usa_wind_kt", "wmo_wind_kt",
    ]
    return pd.DataFrame(records, columns=columns)


def store_summary(con: "duckdb.DuckDBPyConnection") -> dict:
    """Provenance summary of the IBTrACS store (for evidence records)."""
    ensure_haz_schema(con)
    row = con.execute(
        f"""
        SELECT COUNT(*),
               MAX(json_extract_string(payload_json, '$.last_point_time')),
               CAST(MAX(retrieved_at) AS VARCHAR)
        FROM ({_LATEST_REVISION_CTE})
        """,
        [_HAZARD],
    ).fetchone()
    revisions = int(
        con.execute(
            "SELECT COUNT(*) FROM haz_raw_ibtracs WHERE hazard = ?", [_HAZARD]
        ).fetchone()[0]
    )
    scopes = [
        r[0]
        for r in con.execute(
            """
            SELECT DISTINCT json_extract_string(payload_json, '$.source_scope')
            FROM haz_raw_ibtracs WHERE hazard = ? ORDER BY 1
            """,
            [_HAZARD],
        ).fetchall()
        if r[0]
    ]
    urls = [
        r[0]
        for r in con.execute(
            "SELECT DISTINCT source_url FROM haz_raw_ibtracs WHERE hazard = ? ORDER BY 1",
            [_HAZARD],
        ).fetchall()
        if r[0]
    ]
    return {
        "total_storms": int(row[0] or 0),
        "total_revision_rows": revisions,
        "max_point_time": row[1],
        "last_retrieved_at": row[2],
        "source_scopes": scopes,
        "source_urls": urls,
    }
