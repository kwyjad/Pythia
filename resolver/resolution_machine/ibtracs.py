# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""IBTrACS connector — NOAA best-track archive into ``haz_raw_ibtracs``.

Fetches the IBTrACS v04r01 CSV ("last3years" rolling file by default,
"ALL" for backcast — see ``cyclone.ibtracs`` in the rulebook), keeps the
columns detection needs, and upserts on the (sid, iso_time) primary key
so re-fetches are idempotent.  No authentication required.

The raw table stores values unmodified apart from longitude
normalisation to [-180, 180] (IBTrACS mixes conventions across basins).
Wind is kept as BOTH ``usa_wind_kt`` (1-minute sustained) and
``wmo_wind_kt`` (reporting agency, often 10-minute); which one the
trigger trusts is a rulebook decision (``cyclone.wind_source_priority``),
not an ingestion decision.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import IO

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from resolver.resolution_machine.rulebook import Rulebook
from resolver.resolution_machine.schema import ensure_schema, utcnow_iso

LOG = logging.getLogger(__name__)

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
    template = str(rulebook["cyclone.ibtracs.url_template"])
    return template.format(scope=scope)


def parse_ibtracs_csv(source: str | Path | IO) -> pd.DataFrame:
    """Parse an IBTrACS v04 CSV into the ``haz_raw_ibtracs`` frame shape.

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
    # Protect the (sid, iso_time) primary key.
    out = out.drop_duplicates(subset=["sid", "iso_time"], keep="first")
    dropped = n_raw - len(out)
    if dropped:
        LOG.info("[ibtracs] dropped %d rows missing essentials or duplicated", dropped)
    return out.reset_index(drop=True)


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
    scope = scope or str(rulebook["cyclone.ibtracs.default_scope"])
    url = ibtracs_url(rulebook, scope)
    timeout = float(rulebook["cyclone.ibtracs.request_timeout_sec"])
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


def store_ibtracs(con, df: pd.DataFrame, scope: str, source_url: str) -> int:
    """Idempotently upsert track points into ``haz_raw_ibtracs``.

    Returns the number of rows in the store afterwards.  Re-storing the
    same frame leaves the row count unchanged (INSERT OR REPLACE on the
    (sid, iso_time) primary key).
    """
    ensure_schema(con)
    if df.empty:
        LOG.warning("[ibtracs] nothing to store (empty frame)")
        return int(con.execute("SELECT COUNT(*) FROM haz_raw_ibtracs").fetchone()[0])

    stage = df.copy()
    stage["source_scope"] = scope
    stage["source_url"] = source_url
    stage["fetched_at"] = utcnow_iso()

    con.register("ibtracs_stage", stage)
    try:
        con.execute(
            """
            INSERT OR REPLACE INTO haz_raw_ibtracs
                (sid, season, basin, storm_name, iso_time, lat, lon,
                 usa_wind_kt, wmo_wind_kt, nature,
                 source_scope, source_url, fetched_at)
            SELECT sid, season, basin, storm_name, iso_time, lat, lon,
                   usa_wind_kt, wmo_wind_kt, nature,
                   source_scope, source_url, fetched_at
            FROM ibtracs_stage
            """
        )
    finally:
        con.unregister("ibtracs_stage")

    total = int(con.execute("SELECT COUNT(*) FROM haz_raw_ibtracs").fetchone()[0])
    LOG.info("[ibtracs] stored %d points (table now %d rows)", len(stage), total)
    return total


def store_summary(con) -> dict:
    """Provenance summary of the IBTrACS store (for evidence records)."""
    ensure_schema(con)
    row = con.execute(
        """
        SELECT COUNT(*),
               CAST(MAX(iso_time) AS VARCHAR),
               MAX(fetched_at),
               COUNT(DISTINCT source_scope)
        FROM haz_raw_ibtracs
        """
    ).fetchone()
    scopes = [
        r[0]
        for r in con.execute(
            "SELECT DISTINCT source_scope FROM haz_raw_ibtracs ORDER BY 1"
        ).fetchall()
    ]
    urls = [
        r[0]
        for r in con.execute(
            "SELECT DISTINCT source_url FROM haz_raw_ibtracs ORDER BY 1"
        ).fetchall()
    ]
    return {
        "total_points": int(row[0] or 0),
        "max_iso_time": row[1],
        "last_fetched_at": row[2],
        "source_scopes": scopes,
        "source_urls": urls,
    }
