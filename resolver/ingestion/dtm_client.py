#!/usr/bin/env python3
"""DTM connector that converts stock or flow tables into monthly displacement flows."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
)

import pandas as pd
import requests
import time
import yaml

from resolver.ingestion._manifest import ensure_manifest_for_csv
from resolver.ingestion.dtm_auth import get_dtm_api_key, get_auth_headers
from resolver.ingestion._shared.date_utils import parse_dates, window_mask
from resolver.ingestion._shared.run_io import count_csv_rows, write_json
from resolver.ingestion._shared.validation import validate_required_fields
from resolver.ingestion.diagnostics_emitter import (
    append_jsonl as diagnostics_append_jsonl,
    finalize_run as diagnostics_finalize_run,
    start_run as diagnostics_start_run,
)
from resolver.ingestion.utils import (
    ensure_headers,
    flow_from_stock,
    month_start,
    stable_digest,
    to_iso3,
)
from resolver.ingestion.utils.io import resolve_ingestion_window, resolve_output_path

ROOT = Path(__file__).resolve().parents[1]
STAGING = ROOT / "staging"
CONFIG_PATH = ROOT / "ingestion" / "config" / "dtm.yml"
DEFAULT_OUTPUT = ROOT / "staging" / "dtm_displacement.csv"
OUT_PATH = resolve_output_path(DEFAULT_OUTPUT)
OUT_DIR = OUT_PATH.parent
OUTPUT_PATH = OUT_PATH  # backwards compatibility alias
META_PATH = OUT_PATH.with_suffix(OUT_PATH.suffix + ".meta.json")
DIAGNOSTICS_DIR = ROOT / "diagnostics" / "ingestion"
CONNECTORS_REPORT = DIAGNOSTICS_DIR / "connectors_report.jsonl"
CONFIG_ISSUES_PATH = DIAGNOSTICS_DIR / "dtm_config_issues.json"
RESOLVED_SOURCES_PATH = DIAGNOSTICS_DIR / "dtm_sources_resolved.json"
RUN_DETAILS_PATH = DIAGNOSTICS_DIR / "dtm_run.json"

LOG = logging.getLogger("resolver.ingestion.dtm")

COLUMNS = [
    "source",
    "country_iso3",
    "admin1",
    "event_id",
    "as_of",
    "month_start",
    "value_type",
    "value",
    "unit",
    "method",
    "confidence",
    "raw_event_id",
    "raw_fields_json",
]

DEFAULT_CAUSE = "unknown"

CANONICAL_HEADERS = COLUMNS


DATA_PATH = ROOT / "data"
COUNTRIES_PATH = DATA_PATH / "countries.csv"
SHOCKS_PATH = DATA_PATH / "shocks.csv"

SERIES_INCIDENT = "incident"
SERIES_CUMULATIVE = "cumulative"

HAZARD_KEY_TO_CODE = {
    "flood": "FL",
    "drought": "DR",
    "tropical_cyclone": "TC",
    "heat_wave": "HW",
    "armed_conflict_onset": "ACO",
    "armed_conflict_escalation": "ACE",
    "civil_unrest": "CU",
    "displacement_influx": "DI",
    "economic_crisis": "EC",
    "phe": "PHE",
}

MULTI_HAZARD = ("multi", "Multi-shock Displacement/Needs", "all")


@dataclass(frozen=True)
class Hazard:
    """Lightweight hazard tuple for legacy helpers."""

    code: str
    label: str
    hclass: str


@dataclass
class SourceResult:
    """Container describing the outcome of processing a single source entry."""

    source_name: str
    records: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "ok"
    skip_reason: Optional[str] = None
    error: Optional[str] = None
    http_counts: Dict[str, int] = field(
        default_factory=lambda: {"2xx": 0, "4xx": 0, "5xx": 0}
    )
    rows_before: int = 0
    rows_after: int = 0
    dropped: int = 0
    parse_errors: int = 0
    min_date: Optional[str] = None
    max_date: Optional[str] = None

    @property
    def rows(self) -> int:
        return len(self.records)


__all__ = [
    "SERIES_INCIDENT",
    "SERIES_CUMULATIVE",
    "Hazard",
    "SourceResult",
    "DTMApiClient",
    "load_registries",
    "infer_hazard",
    "rollup_subnational",
    "compute_monthly_deltas",
    "load_config",
    "ensure_header_only",
    "build_rows",
    "write_rows",
    "parse_args",
    "main",
]


class DTMApiClient:
    """Client for fetching data from DTM API v3."""

    def __init__(self, config: dict):
        """Initialize DTM API client.

        Args:
            config: Configuration dictionary with 'api' settings.
        """
        api_cfg = config.get("api", {})
        # CRITICAL FIX: Use the actual API gateway URL, not the portal URL
        # The portal (dtm-apim-portal.iom.int) is for registration only
        # The actual API gateway is at dtmapi.iom.int
        self.base_url = api_cfg.get("base_url", "https://dtmapi.iom.int").rstrip("/")
        self.timeout = api_cfg.get("timeout", 30)
        self.rate_limit_delay = api_cfg.get("rate_limit_delay", 1.0)

        try:
            self.api_key = get_dtm_api_key()
        except RuntimeError as exc:
            LOG.error("DTM API authentication failed: %s", exc)
            raise

        LOG.info("DTM API client initialized (base_url=%s)", self.base_url)

    def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        http_counts: Optional[Dict[str, int]] = None,
    ) -> List[Dict[str, Any]]:
        """Make authenticated request to DTM API.

        Args:
            endpoint: API endpoint (e.g., "v3/displacement/admin0")
            params: Query parameters
            http_counts: Dictionary to update with HTTP status counts

        Returns:
            List of data dictionaries from API response

        Raises:
            requests.HTTPError: If request fails
            ValueError: On unexpected responses or redirects
        """
        url = f"{self.base_url}/{endpoint}"
        headers = get_auth_headers()

        # Apply rate limiting
        if self.rate_limit_delay > 0:
            time.sleep(self.rate_limit_delay)

        LOG.debug("DTM API request: %s (params=%s)", endpoint, params)

        try:
            response = requests.get(
                url,
                headers=headers,
                params=params,
                timeout=self.timeout,
                allow_redirects=False,  # IMPORTANT: Detect redirects early
            )

            # Update HTTP counts
            if http_counts is not None:
                status_code = response.status_code
                if 200 <= status_code < 300:
                    http_counts["2xx"] = http_counts.get("2xx", 0) + 1
                elif 400 <= status_code < 500:
                    http_counts["4xx"] = http_counts.get("4xx", 0) + 1
                elif 500 <= status_code < 600:
                    http_counts["5xx"] = http_counts.get("5xx", 0) + 1
                http_counts["last_status"] = status_code

            # Check for redirects (302, 301, etc.)
            if 300 <= response.status_code < 400:
                redirect_location = response.headers.get("Location", "unknown")
                LOG.error(
                    "DTM API returned redirect %s to %s. "
                    "This usually means the base_url is incorrect. "
                    "Current base_url: %s. "
                    "Check https://dtm-apim-portal.iom.int/ for the correct API gateway URL.",
                    response.status_code,
                    redirect_location,
                    self.base_url,
                )
                raise ValueError(
                    f"API redirect detected ({response.status_code} -> {redirect_location}). "
                    f"The base_url may be incorrect. Current: {self.base_url}"
                )

            response.raise_for_status()

            # Try to parse JSON
            try:
                data = response.json()
            except json.JSONDecodeError as e:
                content_type = response.headers.get("Content-Type", "unknown")
                content_preview = response.text[:200] if response.text else "(empty)"
                LOG.error(
                    "Failed to parse API response as JSON. "
                    "Status: %s, Content-Type: %s, Preview: %s",
                    response.status_code,
                    content_type,
                    content_preview,
                )
                raise ValueError(f"Invalid JSON response from {url}: {e}") from e

            # Handle official DTM API response format: {"isSuccess": true, "result": [...]}
            if isinstance(data, dict) and "isSuccess" in data:
                if not data.get("isSuccess"):
                    error_messages = data.get("errorMessages", ["Unknown error"])
                    error_msg = "; ".join(str(e) for e in error_messages)
                    raise ValueError(f"API returned error: {error_msg}")
                result = data.get("result", [])
                if not isinstance(result, list):
                    result = [result] if result else []
            # Legacy format support: handle various response structures
            elif isinstance(data, list):
                result = data
            elif isinstance(data, dict) and "data" in data:
                result = data["data"]
                if not isinstance(result, list):
                    result = [result] if result else []
            elif isinstance(data, dict) and "value" in data:  # OData format
                result = data["value"]
                if not isinstance(result, list):
                    result = [result] if result else []
            elif isinstance(data, dict):
                # Single object or unknown format
                if "error" in data or "Error" in data:
                    error_msg = data.get("error") or data.get("Error", "Unknown error")
                    raise ValueError(f"API returned error: {error_msg}")
                result = [data]
            else:
                result = []

            LOG.debug("DTM API response: %s rows from %s", len(result), endpoint)
            return result

        except requests.exceptions.Timeout:
            LOG.error("DTM API timeout for %s after %ss", endpoint, self.timeout)
            raise
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response else "unknown"
            LOG.error(
                "DTM API HTTP error for %s: %s (status=%s)",
                endpoint,
                exc,
                status,
            )
            raise
        except requests.exceptions.ConnectionError as e:
            LOG.error("Failed to connect to DTM API at %s: %s", url, e)
            raise
        except requests.exceptions.RequestException as exc:
            LOG.error("DTM API request failed for %s: %s", endpoint, exc)
            raise

    def get_countries(
        self, http_counts: Optional[Dict[str, int]] = None
    ) -> pd.DataFrame:
        """Fetch list of all available countries.

        Args:
            http_counts: Dictionary to update with HTTP status counts

        Returns:
            DataFrame with country information
        """
        data = self._make_request("v3/displacement/country-list", http_counts=http_counts)
        return pd.DataFrame(data)

    def get_idp_admin0(
        self,
        country: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        from_round: Optional[int] = None,
        to_round: Optional[int] = None,
        http_counts: Optional[Dict[str, int]] = None,
    ) -> pd.DataFrame:
        """Fetch IDP data at country level (Admin 0).

        Args:
            country: Filter by country name
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            from_round: Start round number
            to_round: End round number
            http_counts: Dictionary to update with HTTP status counts

        Returns:
            DataFrame with Admin 0 IDP data
        """
        params: Dict[str, Any] = {}
        if country:
            params["CountryName"] = country
        if from_date:
            params["FromReportingDate"] = from_date
        if to_date:
            params["ToReportingDate"] = to_date
        if from_round is not None:
            params["FromRoundNumber"] = from_round
        if to_round is not None:
            params["ToRoundNumber"] = to_round

        data = self._make_request("v3/displacement/admin0", params, http_counts=http_counts)
        return pd.DataFrame(data)

    def get_idp_admin1(
        self,
        country: Optional[str] = None,
        admin1: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        http_counts: Optional[Dict[str, int]] = None,
    ) -> pd.DataFrame:
        """Fetch IDP data at state/province level (Admin 1).

        Args:
            country: Filter by country name
            admin1: Filter by Admin 1 name
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            http_counts: Dictionary to update with HTTP status counts

        Returns:
            DataFrame with Admin 1 IDP data
        """
        params: Dict[str, Any] = {}
        if country:
            params["CountryName"] = country
        if admin1:
            params["Admin1Name"] = admin1
        if from_date:
            params["FromReportingDate"] = from_date
        if to_date:
            params["ToReportingDate"] = to_date

        data = self._make_request("v3/displacement/admin1", params, http_counts=http_counts)
        return pd.DataFrame(data)

    def get_idp_admin2(
        self,
        country: Optional[str] = None,
        operation: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        http_counts: Optional[Dict[str, int]] = None,
    ) -> pd.DataFrame:
        """Fetch IDP data at district level (Admin 2).

        Args:
            country: Filter by country name
            operation: Filter by operation type
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            http_counts: Dictionary to update with HTTP status counts

        Returns:
            DataFrame with Admin 2 IDP data
        """
        params: Dict[str, Any] = {}
        if country:
            params["CountryName"] = country
        if operation:
            params["Operation"] = operation
        if from_date:
            params["FromReportingDate"] = from_date
        if to_date:
            params["ToReportingDate"] = to_date

        data = self._make_request("v3/displacement/admin2", params, http_counts=http_counts)
        return pd.DataFrame(data)


def _env_bool(name: str, default: bool) -> bool:
    """Parse boolean feature flags from the environment."""

    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "y", "yes", "on"}


def _normalize_month(value: object) -> Optional[str]:
    """Return ``YYYY-MM`` strings when ``value`` parses as a date."""

    bucket = month_start(value)
    if not bucket:
        return None
    return bucket.strftime("%Y-%m")


def _is_subnational(record: Mapping[str, Any]) -> bool:
    for key in ("admin1", "admin2", "admin_pcode", "admin_name"):
        if str(record.get(key) or "").strip():
            return True
    return False


def load_registries() -> Tuple["pd.DataFrame", "pd.DataFrame"]:
    """Return the canonical country and shock registries used by legacy tests."""

    import pandas as pd  # local import to keep module import-side-effect free

    countries = pd.read_csv(COUNTRIES_PATH, dtype=str).fillna("")
    shocks = pd.read_csv(SHOCKS_PATH, dtype=str).fillna("")
    return countries, shocks


def _hazard_from_code(code: str, shocks: "pd.DataFrame") -> Hazard:
    if not code:
        return Hazard(*MULTI_HAZARD)
    if str(code).strip().lower() == "multi":
        return Hazard(*MULTI_HAZARD)
    match = shocks[shocks["hazard_code"].str.upper() == str(code).strip().upper()]
    if match.empty:
        return Hazard(*MULTI_HAZARD)
    row = match.iloc[0]
    return Hazard(row["hazard_code"], row["hazard_label"], row["hazard_class"])


def infer_hazard(
    texts: Iterable[str],
    shocks: Optional["pd.DataFrame"] = None,
    keywords_cfg: Optional[Mapping[str, Iterable[str]]] = None,
    *,
    default_key: Optional[str] = None,
) -> Hazard:
    """Map dataset text snippets to a hazard triple.

    This mirrors the legacy helper relied on by resolver tests and keeps the
    behaviour stable for CI while the production connector evolves.
    """

    if shocks is None:
        _, shocks = load_registries()
    if keywords_cfg is None:
        keywords_cfg = load_config().get("shock_keywords", {})
    if default_key is None:
        default_key = os.getenv(
            "DTM_DEFAULT_HAZARD",
            load_config().get("default_hazard", "displacement_influx"),
        )

    text_blob = " ".join(str(t).lower() for t in texts if t)
    matches: list[str] = []
    for key, keywords in (keywords_cfg or {}).items():
        for kw in keywords:
            if str(kw).lower() in text_blob:
                matches.append(str(key).strip().lower())
                break

    if not matches and shocks is not None:
        for _, row in shocks.iterrows():
            label = str(row.get("hazard_label", "")).strip().lower()
            if label and label in text_blob:
                matches.append(label)

    unique = sorted({m for m in matches if m})
    if not unique:
        mapped = HAZARD_KEY_TO_CODE.get(str(default_key or "").strip().lower())
        if not mapped:
            return Hazard(*MULTI_HAZARD)
        return _hazard_from_code(mapped, shocks)
    if len(unique) > 1:
        return Hazard(*MULTI_HAZARD)

    mapped = HAZARD_KEY_TO_CODE.get(unique[0], unique[0])
    return _hazard_from_code(mapped, shocks)


def rollup_subnational(
    records: Sequence[MutableMapping[str, Any]]
) -> List[MutableMapping[str, Any]]:
    """Aggregate subnational rows into national totals per month and source."""

    grouped: Dict[
        Tuple[str, str, str, str, str, str], List[MutableMapping[str, Any]]
    ] = defaultdict(list)
    for rec in records:
        as_of = _normalize_month(rec.get("as_of_date")) or ""
        key = (
            str(rec.get("iso3", "")),
            str(rec.get("hazard_code", "")),
            str(rec.get("metric", "")),
            as_of,
            str(rec.get("source_id", "")),
            str(rec.get("series_type", SERIES_INCIDENT)),
        )
        rec_copy = dict(rec)
        rec_copy["as_of_date"] = as_of
        grouped[key].append(rec_copy)

    rolled: List[MutableMapping[str, Any]] = []
    for key, rows in grouped.items():
        nationals = [r for r in rows if not _is_subnational(r)]
        if nationals:
            nationals.sort(key=lambda r: r.get("as_of_date", ""))
            rolled.extend(nationals)
            continue
        total = 0.0
        template = dict(rows[0])
        for row in rows:
            try:
                total += float(row.get("value", 0) or 0)
            except Exception:
                continue
        template["value"] = max(total, 0.0)
        for drop_key in ("admin1", "admin2", "admin_pcode", "admin_name"):
            template.pop(drop_key, None)
        rolled.append(template)

    rolled.sort(
        key=lambda r: (
            str(r.get("iso3", "")),
            str(r.get("hazard_code", "")),
            str(r.get("metric", "")),
            str(r.get("as_of_date", "")),
        )
    )
    return rolled


def compute_monthly_deltas(
    records: Sequence[MutableMapping[str, Any]],
    *,
    allow_first_month: Optional[bool] = None,
) -> List[MutableMapping[str, Any]]:
    """Convert cumulative series to month-over-month deltas.

    Incident series are passed through, cumulative series become non-negative
    monthly flows. This mirrors the legacy helper that powers resolver tests.
    """

    if allow_first_month is None:
        cfg = load_config()
        allow_first_month = _env_bool(
            "DTM_ALLOW_FIRST_MONTH",
            bool(cfg.get("allow_first_month_delta", False)),
        )

    grouped: Dict[Tuple[str, str, str, str], List[MutableMapping[str, Any]]] = defaultdict(list)
    for rec in records:
        as_of = _normalize_month(rec.get("as_of_date")) or ""
        rec_copy = dict(rec)
        rec_copy["as_of_date"] = as_of
        key = (
            str(rec.get("iso3", "")),
            str(rec.get("hazard_code", "")),
            str(rec.get("metric", "")),
            str(rec.get("source_id", "")),
        )
        grouped[key].append(rec_copy)

    output: List[MutableMapping[str, Any]] = []
    for rows in grouped.values():
        rows.sort(key=lambda r: r.get("as_of_date", ""))
        series_type = str(rows[0].get("series_type", SERIES_INCIDENT)).strip().lower()
        prev_value: Optional[float] = None
        for row in rows:
            value = row.get("value", 0)
            try:
                value_num = float(value)
            except Exception:
                value_num = 0.0
            if series_type != SERIES_CUMULATIVE:
                new_val = max(value_num, 0.0)
            else:
                if prev_value is None:
                    if allow_first_month:
                        new_val = max(value_num, 0.0)
                        prev_value = value_num
                    else:
                        prev_value = value_num
                        continue
                else:
                    delta = value_num - prev_value
                    if delta < 0:
                        delta = 0.0
                    new_val = delta
                    prev_value = value_num
            out_row = dict(row)
            out_row["value"] = new_val
            out_row["series_type"] = SERIES_INCIDENT
            output.append(out_row)

    output.sort(
        key=lambda r: (
            str(r.get("iso3", "")),
            str(r.get("hazard_code", "")),
            str(r.get("metric", "")),
            str(r.get("as_of_date", "")),
        )
    )
    return output


_CANDIDATE_DATE_FIELDS = [
    "as_of",
    "updated_at",
    "last_updated",
    "update_date",
    "report_date",
    "reporting_date",
    "date",
    "dtm_date",
]


def _parse_iso_date_or_none(s: str):
    # Fast path: YYYY-MM-DD or YYYY-MM
    if not s:
        return None
    s = str(s).strip()
    # Normalize common formats
    try:
        # Try YYYY-MM-DD
        if len(s) >= 10 and s[4] == "-" and s[7] == "-":
            return s[:10]
        # Try YYYY/MM/DD
        if len(s) >= 10 and s[4] in "/." and s[7] in "/.":
            return f"{s[:4]}-{s[5:7]}-{s[8:10]}"
        # Try YYYY-MM
        if len(s) >= 7 and s[4] == "-":
            return f"{s[:7]}-01"
    except Exception:
        return None
    return None


_FILENAME_DATE_REGEX = r"(20\d{2})[-_]?([01]\d)(?:[-_]?([0-3]\d))?"


def _asof_from_filename(fname: str):
    import re

    m = re.search(_FILENAME_DATE_REGEX, fname or "")
    if not m:
        return None
    yyyy, mm, dd = m.group(1), m.group(2), m.group(3)
    if not dd:
        dd = "01"
    return f"{yyyy}-{mm}-{dd}"


def _file_mtime_iso(path: str):
    import os

    try:
        ts = os.path.getmtime(path)
        d = datetime.utcfromtimestamp(ts).date().isoformat()
        return d
    except Exception:
        return None


_AS_OF_FALLBACK_COUNTS: dict[str, int] = {"filename": 0, "mtime": 0, "run": 0}


def _extract_record_as_of(row: dict, file_ctx: dict) -> str:
    """
    Choose the best available per-record as_of, in order of preference:
      1) Any known per-row timestamp field (normalized to YYYY-MM-DD or YYYY-MM-01)
      2) File name embedded date (e.g., 2024-07-15 or 202407)
      3) File modified time (UTC date)
      4) Run date (UTC today) as last resort
    """

    # 1) row fields
    for k in _CANDIDATE_DATE_FIELDS:
        if k in row and row[k]:
            iso = _parse_iso_date_or_none(str(row[k]))
            if iso:
                return iso
    # 2) file name date
    fname_iso = _asof_from_filename(file_ctx.get("filename") or "")
    if fname_iso:
        _AS_OF_FALLBACK_COUNTS["filename"] += 1
        return fname_iso
    # 3) file mtime
    mtime_iso = _file_mtime_iso(file_ctx.get("path") or "")
    if mtime_iso:
        _AS_OF_FALLBACK_COUNTS["mtime"] += 1
        return mtime_iso
    # 4) run date fallback
    _AS_OF_FALLBACK_COUNTS["run"] += 1
    return datetime.now(timezone.utc).date().isoformat()


def _is_candidate_newer(existing_iso: str, candidate_iso: str) -> bool:
    """Return True when the candidate `as_of` timestamp is strictly newer."""

    if not candidate_iso:
        return False
    if not existing_iso:
        return True
    return candidate_iso > existing_iso


def load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {}
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def ensure_header_only() -> None:
    ensure_headers(OUT_PATH, COLUMNS)
    ensure_manifest_for_csv(OUT_PATH, schema_version="dtm_displacement.v1", source_id="dtm")


def _parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


def _resolve_cause(row: Mapping[str, Any], cause_map: Mapping[str, str]) -> str:
    for key in ("cause", "cause_category", "reason"):
        value = row.get(key)
        if value:
            norm = str(value).strip().lower()
            mapped = cause_map.get(norm)
            if mapped:
                return mapped
            return norm
    return DEFAULT_CAUSE


def _source_label(entry: Mapping[str, Any]) -> str:
    return str(entry.get("id") or entry.get("name") or entry.get("id_or_path") or "dtm_source")


def _column(row: Mapping[str, Any], *candidates: str) -> Optional[str]:
    lowered = {col.lower(): col for col in row.keys()}
    for candidate in candidates:
        key = candidate.lower()
        if key in lowered:
            return lowered[key]
    for candidate in candidates:
        for col in row.keys():
            if col.lower().replace(" ", "") == candidate.lower().replace(" ", ""):
                return col
    return None


def _read_source(
    entry: Mapping[str, Any],
    cfg: Mapping[str, Any],
    *,
    no_date_filter: bool,
    window_start: Optional[str],
    window_end: Optional[str],
) -> SourceResult:
    source_label = _source_label(entry)
    LOG.debug("dtm: begin source %s", source_label)
    source_type = str(entry.get("type") or "file").strip().lower()
    if source_type != "file":
        raise ValueError("DTM connector currently supports file sources only")
    path = entry.get("id_or_path")
    if not path:
        LOG.warning("dtm: skipping source %s (missing id_or_path)", source_label)
        return SourceResult(
            source_name=source_label,
            status="skipped",
            skip_reason="missing id_or_path",
        )
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"DTM source not found: {csv_path}")
    LOG.debug("dtm: resolved source %s path=%s", source_label, csv_path)
    file_ctx = {"filename": csv_path.name, "path": str(csv_path)}
    aliases = cfg.get("country_aliases") or {}
    measure = str(entry.get("measure") or "stock").strip().lower()
    cause_map = {str(k).strip().lower(): str(v) for k, v in (cfg.get("cause_map") or {}).items()}

    try:
        frame = pd.read_csv(csv_path, dtype=object, keep_default_na=False)
    except pd.errors.EmptyDataError:
        frame = pd.DataFrame()

    rows_before = int(frame.shape[0])
    header_probe: MutableMapping[str, Any] = {str(col): None for col in frame.columns}

    if rows_before == 0 or not header_probe:
        LOG.debug("dtm: end source %s rows=0", source_label)
        return SourceResult(
            source_name=source_label,
            records=[],
            status="ok",
            rows_before=rows_before,
            rows_after=0,
            dropped=rows_before,
            parse_errors=0,
        )

    def _resolve_column(*candidates: Optional[str]) -> Optional[str]:
        explicit = candidates[0] if candidates else None
        if explicit:
            resolved_explicit = _column(header_probe, str(explicit))
            if resolved_explicit:
                return resolved_explicit
            return None
        fallbacks = [str(name) for name in candidates[1:] if name]
        if not fallbacks:
            return None
        return _column(header_probe, *fallbacks)

    country_column = _resolve_column(entry.get("country_column"), "country_iso3", "iso3", "country")
    admin_column = _resolve_column(entry.get("admin1_column"), "admin1", "adm1", "province", "state")
    date_column = _resolve_column(entry.get("date_column"), "date", "month", "period")
    value_column = _resolve_column(entry.get("value_column"), "value", "count", "population", "total")
    cause_column = _resolve_column(entry.get("cause_column"), "cause", "cause_category", "reason")

    if not date_column:
        LOG.warning("dtm: skipping source %s (missing date_column)", source_label)
        return SourceResult(
            source_name=source_label,
            status="invalid",
            skip_reason="missing date_column",
            rows_before=rows_before,
            rows_after=0,
            dropped=rows_before,
            parse_errors=0,
        )

    if not value_column or not country_column:
        raise ValueError("DTM source missing required columns")

    fmt = str(entry.get("date_format")) if entry.get("date_format") else None
    parsed_dates, parse_errors = parse_dates(frame[date_column], fmt)
    valid_dates = parsed_dates.notna()
    parsed_valid = parsed_dates[valid_dates]

    min_iso: Optional[str] = None
    max_iso: Optional[str] = None
    if not parsed_valid.empty:
        min_candidate = parsed_valid.min()
        max_candidate = parsed_valid.max()
        if pd.notna(min_candidate):
            min_iso = min_candidate.date().isoformat()
        if pd.notna(max_candidate):
            max_iso = max_candidate.date().isoformat()

    window_enabled = bool(not no_date_filter and window_start and window_end)
    if window_enabled:
        window_series = window_mask(parsed_dates, str(window_start), str(window_end))
    else:
        window_series = pd.Series(True, index=frame.index)

    final_mask = valid_dates & window_series
    rows_after = int(final_mask.sum())
    dropped = rows_before - rows_after

    if rows_after == 0:
        LOG.debug(
            "dtm: end source %s rows=0 (kept=%s dropped=%s parse_errors=%s)",
            source_label,
            rows_after,
            dropped,
            parse_errors,
        )
        return SourceResult(
            source_name=source_label,
            records=[],
            status="ok",
            rows_before=rows_before,
            rows_after=rows_after,
            dropped=dropped,
            parse_errors=parse_errors,
            min_date=min_iso,
            max_date=max_iso,
        )

    filtered = frame.loc[final_mask].copy()
    filtered = filtered.where(pd.notna(filtered), None)
    rows = filtered.to_dict(orient="records")

    per_admin: dict[tuple[str, str], dict[datetime, float]] = defaultdict(dict)
    per_admin_asof: dict[tuple[str, str], dict[datetime, str]] = defaultdict(dict)
    causes: dict[tuple[str, str], str] = {}

    for row in rows:
        iso = to_iso3(row.get(country_column), aliases)
        if not iso:
            continue
        bucket = month_start(row.get(date_column))
        if not bucket:
            continue
        admin1 = str(row.get(admin_column) or "").strip() if admin_column else ""
        value = _parse_float(row.get(value_column))
        if value is None or value < 0:
            continue
        per_admin[(iso, admin1)][bucket] = value
        as_of_value = _extract_record_as_of(row, file_ctx)
        existing_asof = per_admin_asof[(iso, admin1)].get(bucket)
        if not existing_asof or _is_candidate_newer(existing_asof, as_of_value):
            per_admin_asof[(iso, admin1)][bucket] = as_of_value
        if cause_column and row.get(cause_column):
            causes[(iso, admin1)] = _resolve_cause({cause_column: row.get(cause_column)}, cause_map)
        else:
            causes[(iso, admin1)] = _resolve_cause(row, cause_map)

    records: List[Dict[str, Any]] = []
    for key, series in per_admin.items():
        iso, admin1 = key
        if measure == "stock":
            flows = flow_from_stock(series)
        else:
            flows = {month_start(k): float(v) for k, v in series.items() if month_start(k)}
        cause = causes.get(key, DEFAULT_CAUSE)
        for bucket, value in flows.items():
            if not bucket or value is None or value <= 0:
                continue
            record_as_of = (
                per_admin_asof.get(key, {}).get(bucket)
                or datetime.now(timezone.utc).date().isoformat()
            )
            records.append(
                {
                    "iso3": iso,
                    "admin1": admin1,
                    "month": bucket,
                    "value": float(value),
                    "cause": cause,
                    "measure": measure,
                    "source_id": source_label,
                    "as_of": record_as_of,
                }
            )

    LOG.debug(
        "dtm: end source %s rows=%s (kept=%s dropped=%s parse_errors=%s)",
        source_label,
        len(records),
        rows_after,
        dropped,
        parse_errors,
    )
    return SourceResult(
        source_name=source_label,
        records=records,
        status="ok",
        rows_before=rows_before,
        rows_after=rows_after,
        dropped=dropped,
        parse_errors=parse_errors,
        min_date=min_iso,
        max_date=max_iso,
    )


def _fetch_api_data(
    cfg: Mapping[str, Any],
    *,
    results: Optional[List[SourceResult]] = None,
    no_date_filter: bool = False,
    window_start: Optional[str] = None,
    window_end: Optional[str] = None,
    http_counts: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    """Fetch IDP data from DTM API for all configured admin levels.

    Args:
        cfg: Configuration dictionary
        results: List to append SourceResult objects to
        no_date_filter: Whether to disable date filtering
        window_start: Start date for filtering (YYYY-MM-DD)
        window_end: End date for filtering (YYYY-MM-DD)
        http_counts: Dictionary to update with HTTP status counts

    Returns:
        List of records with standardized fields
    """
    from resolver.ingestion.dtm_auth import check_api_key_configured

    if results is None:
        results = []
    if http_counts is None:
        http_counts = {"2xx": 0, "4xx": 0, "5xx": 0}

    # Check if API key is configured before attempting to fetch
    if not check_api_key_configured():
        LOG.warning(
            "DTM API mode requested but DTM_API_KEY not configured. "
            "Skipping API data fetch. Set DTM_API_KEY to fetch live data."
        )
        results.append(
            SourceResult(
                source_name="dtm_api",
                status="skipped",
                skip_reason="api-key-not-configured",
            )
        )
        return []

    all_records: List[Dict[str, Any]] = []

    try:
        client = DTMApiClient(cfg)
    except RuntimeError as exc:
        LOG.error("Failed to initialize DTM API client: %s", exc)
        results.append(
            SourceResult(
                source_name="dtm_api",
                status="error",
                error=str(exc),
            )
        )
        return []

    admin_levels = cfg.get("admin_levels", ["admin0", "admin1", "admin2"])
    countries = cfg.get("countries")  # None = all countries
    operations = cfg.get("operations")  # None = all operations

    # Format dates for API (YYYY-MM-DD)
    from_date = window_start if not no_date_filter and window_start else None
    to_date = window_end if not no_date_filter and window_end else None

    LOG.info(
        "Fetching DTM data: levels=%s, countries=%s, date_range=%s to %s",
        admin_levels,
        countries or "all",
        from_date or "—",
        to_date or "—",
    )

    field_mapping = cfg.get("field_mapping", {})
    field_aliases = cfg.get("field_aliases", {})
    country_column = field_mapping.get("country_column", "CountryName")
    admin1_column = field_mapping.get("admin1_column", "Admin1Name")
    admin2_column = field_mapping.get("admin2_column", "Admin2Name")
    date_column = field_mapping.get("date_column", "ReportingDate")
    round_column = field_mapping.get("round_column", "RoundNumber")

    # Get list of IDP count field candidates
    idp_field_candidates = field_aliases.get("idp_count", ["TotalIDPs", "IDPTotal"])
    if field_mapping.get("idp_column"):
        idp_field_candidates.insert(0, field_mapping["idp_column"])

    aliases = cfg.get("country_aliases") or {}
    measure = cfg.get("output", {}).get("measure", "stock")
    cause_map = {
        str(k).strip().lower(): str(v)
        for k, v in (cfg.get("cause_map") or {}).items()
    }

    # Iterate over admin levels
    for admin_level in admin_levels:
        level_name = admin_level.strip().lower()

        # Determine which countries to fetch
        country_list = countries if countries else [None]

        for country in country_list:
            source_label = f"dtm_api_{level_name}"
            if country:
                source_label = f"{source_label}_{country}"

            try:
                if level_name == "admin0":
                    df = client.get_idp_admin0(
                        country=country,
                        from_date=from_date,
                        to_date=to_date,
                        http_counts=http_counts,
                    )
                elif level_name == "admin1":
                    df = client.get_idp_admin1(
                        country=country,
                        from_date=from_date,
                        to_date=to_date,
                        http_counts=http_counts,
                    )
                elif level_name == "admin2":
                    # Admin2 supports operation filtering
                    for operation in (operations or [None]):
                        df = client.get_idp_admin2(
                            country=country,
                            operation=operation,
                            from_date=from_date,
                            to_date=to_date,
                            http_counts=http_counts,
                        )
                        if not df.empty:
                            break
                else:
                    LOG.warning("Unknown admin level: %s", level_name)
                    continue

                if df.empty:
                    LOG.debug("No data for %s", source_label)
                    results.append(
                        SourceResult(
                            source_name=source_label,
                            status="ok",
                            rows_before=0,
                            rows_after=0,
                        )
                    )
                    continue

                rows_before = len(df)
                LOG.debug("Fetched %s rows for %s", rows_before, source_label)

                # Find the IDP count column
                idp_column = None
                for candidate in idp_field_candidates:
                    if candidate in df.columns:
                        idp_column = candidate
                        break

                if not idp_column:
                    LOG.warning(
                        "No IDP count column found for %s (tried: %s)",
                        source_label,
                        idp_field_candidates,
                    )
                    results.append(
                        SourceResult(
                            source_name=source_label,
                            status="invalid",
                            skip_reason="missing IDP count column",
                            rows_before=rows_before,
                        )
                    )
                    continue

                # Convert DataFrame to records
                per_admin: dict[tuple[str, str], dict[datetime, float]] = defaultdict(
                    dict
                )
                per_admin_asof: dict[tuple[str, str], dict[datetime, str]] = defaultdict(
                    dict
                )
                causes: dict[tuple[str, str], str] = {}

                for _, row in df.iterrows():
                    iso = to_iso3(row.get(country_column), aliases)
                    if not iso:
                        continue

                    bucket = month_start(row.get(date_column))
                    if not bucket:
                        continue

                    admin1 = ""
                    if admin1_column in df.columns:
                        admin1 = str(row.get(admin1_column) or "").strip()
                    if level_name == "admin2" and admin2_column in df.columns:
                        # For admin2, concatenate admin1 and admin2
                        admin2 = str(row.get(admin2_column) or "").strip()
                        if admin2:
                            admin1 = f"{admin1}/{admin2}" if admin1 else admin2

                    value = _parse_float(row.get(idp_column))
                    if value is None or value < 0:
                        continue

                    per_admin[(iso, admin1)][bucket] = value

                    # Use reporting date as as_of
                    as_of_value = str(row.get(date_column) or "")
                    if as_of_value:
                        as_of_value = _parse_iso_date_or_none(as_of_value) or as_of_value
                    if not as_of_value:
                        as_of_value = datetime.now(timezone.utc).date().isoformat()

                    existing_asof = per_admin_asof[(iso, admin1)].get(bucket)
                    if not existing_asof or _is_candidate_newer(
                        existing_asof, as_of_value
                    ):
                        per_admin_asof[(iso, admin1)][bucket] = as_of_value

                    # Set default cause
                    causes[(iso, admin1)] = _resolve_cause(row.to_dict(), cause_map)

                # Convert to standard record format
                for key, series in per_admin.items():
                    iso, admin1 = key
                    if measure == "stock":
                        flows = flow_from_stock(series)
                    else:
                        flows = {
                            month_start(k): float(v)
                            for k, v in series.items()
                            if month_start(k)
                        }
                    cause = causes.get(key, DEFAULT_CAUSE)
                    for bucket, value in flows.items():
                        if not bucket or value is None or value <= 0:
                            continue
                        record_as_of = (
                            per_admin_asof.get(key, {}).get(bucket)
                            or datetime.now(timezone.utc).date().isoformat()
                        )
                        all_records.append(
                            {
                                "iso3": iso,
                                "admin1": admin1,
                                "month": bucket,
                                "value": float(value),
                                "cause": cause,
                                "measure": measure,
                                "source_id": source_label,
                                "as_of": record_as_of,
                            }
                        )

                rows_after = len(
                    [r for r in all_records if r["source_id"] == source_label]
                )
                LOG.info(
                    "Processed %s: fetched=%s, kept=%s",
                    source_label,
                    rows_before,
                    rows_after,
                )

                results.append(
                    SourceResult(
                        source_name=source_label,
                        status="ok",
                        rows_before=rows_before,
                        rows_after=rows_after,
                        http_counts=dict(http_counts),
                    )
                )

            except Exception as exc:
                LOG.error("Failed to fetch %s: %s", source_label, exc, exc_info=True)
                results.append(
                    SourceResult(
                        source_name=source_label,
                        status="error",
                        error=str(exc),
                        http_counts=dict(http_counts),
                    )
                )

    LOG.info("Total records fetched from API: %s", len(all_records))
    return all_records


def build_rows(
    cfg: Mapping[str, Any],
    *,
    results: Optional[List[SourceResult]] = None,
    no_date_filter: bool = False,
    window_start: Optional[str] = None,
    window_end: Optional[str] = None,
    http_counts: Optional[Dict[str, int]] = None,
) -> List[List[Any]]:
    sources = cfg.get("sources") or []
    admin_mode = str(cfg.get("admin_agg") or "both").strip().lower()
    all_records: List[Dict[str, Any]] = []
    collected = results if results is not None else []

    # Determine if we should use API mode or file mode
    use_api_mode = "api" in cfg and not sources

    if use_api_mode:
        # Use API to fetch data
        LOG.info("Using DTM API mode")
        api_records = _fetch_api_data(
            cfg,
            results=collected,
            no_date_filter=no_date_filter,
            window_start=window_start,
            window_end=window_end,
            http_counts=http_counts,
        )
        all_records.extend(api_records)
    else:
        # Use file-based sources
        LOG.info("Using DTM file mode (%s sources)", len(sources))
        for entry in sources:
            if not isinstance(entry, Mapping):
                continue
            result = _read_source(
                entry,
                cfg,
                no_date_filter=no_date_filter,
                window_start=window_start,
                window_end=window_end,
            )
            collected.append(result)
            if result.records:
                all_records.extend(result.records)
    if not all_records:
        _log_as_of_fallbacks()
        return []
    run_date = datetime.now(timezone.utc).date().isoformat()
    method = "dtm_stock_to_flow" if any(rec.get("measure") == "stock" for rec in all_records) else "dtm_flow"
    dedup: dict[tuple[str, str, str, str], Dict[str, Any]] = {}
    country_totals: dict[tuple[str, str, str], float] = defaultdict(float)
    country_as_of: dict[tuple[str, str, str], str] = {}
    for rec in all_records:
        iso3 = rec["iso3"]
        admin1 = rec.get("admin1") or ""
        month = rec["month"]
        month_iso = month.isoformat()
        value = float(rec.get("value", 0.0))
        record_as_of = rec.get("as_of") or run_date
        if admin_mode in {"admin1", "both"} and admin1:
            key = (iso3, admin1, month_iso, rec["source_id"])
            event_id = f"{iso3}-displacement-{month.strftime('%Y%m')}-{stable_digest(key)}"
            record = {
                "source": "dtm",
                "country_iso3": iso3,
                "admin1": admin1,
                "event_id": event_id,
                "as_of": record_as_of,
                "month_start": month_iso,
                "value_type": "new_displaced",
                "value": int(round(value)),
                "unit": "people",
                "method": method,
                "confidence": rec.get("cause", DEFAULT_CAUSE),
                "raw_event_id": f"{rec['source_id']}::{admin1 or 'national'}::{month.strftime('%Y%m')}",
                "raw_fields_json": json.dumps(
                    {
                        "source_id": rec["source_id"],
                        "admin1": admin1,
                        "cause": rec.get("cause", DEFAULT_CAUSE),
                        "measure": rec.get("measure"),
                    },
                    ensure_ascii=False,
                ),
            }
            existing = dedup.get(key)
            if existing and not _is_candidate_newer(existing["as_of"], record["as_of"]):
                continue
            dedup[key] = record
        if admin_mode in {"country", "both"}:
            country_key = (iso3, month_iso, rec["source_id"])
            country_totals[country_key] += value
            existing_country_asof = country_as_of.get(country_key)
            if not existing_country_asof or _is_candidate_newer(existing_country_asof, record_as_of):
                country_as_of[country_key] = record_as_of
    rows = list(dedup.values())
    if admin_mode in {"country", "both"}:
        for (iso3, month_iso, source_id), total in country_totals.items():
            if total <= 0:
                continue
            month = datetime.strptime(month_iso, "%Y-%m-%d").date()
            event_id = f"{iso3}-displacement-{month.strftime('%Y%m')}-{stable_digest([iso3, month_iso, source_id])}"
            rows.append(
                {
                    "source": "dtm",
                    "country_iso3": iso3,
                    "admin1": "",
                    "event_id": event_id,
                    "as_of": country_as_of.get((iso3, month_iso, source_id), run_date),
                    "month_start": month_iso,
                    "value_type": "new_displaced",
                    "value": int(round(total)),
                    "unit": "people",
                    "method": method,
                    "confidence": DEFAULT_CAUSE,
                    "raw_event_id": f"{source_id}::country::{month.strftime('%Y%m')}",
                    "raw_fields_json": json.dumps(
                        {
                            "source_id": source_id,
                            "aggregation": "country",
                            "total_value": total,
                        },
                        ensure_ascii=False,
                    ),
                }
            )
    formatted = [
        [
            rec["source"],
            rec["country_iso3"],
            rec.get("admin1", ""),
            rec["event_id"],
            rec["as_of"],
            rec["month_start"],
            rec["value_type"],
            rec["value"],
            rec["unit"],
            rec["method"],
            rec["confidence"],
            rec["raw_event_id"],
            rec["raw_fields_json"],
        ]
        for rec in rows
    ]
    formatted.sort(key=lambda row: (row[1], row[2], row[5], row[3]))
    _log_as_of_fallbacks()
    return formatted


def _log_as_of_fallbacks() -> None:
    filename_count = _AS_OF_FALLBACK_COUNTS.get("filename", 0)
    if filename_count:
        LOG.info("dtm: as_of from filename for %s records", filename_count)
    mtime_count = _AS_OF_FALLBACK_COUNTS.get("mtime", 0)
    if mtime_count:
        LOG.info("dtm: as_of from file mtime for %s records", mtime_count)
    run_count = _AS_OF_FALLBACK_COUNTS.get("run", 0)
    if run_count:
        LOG.warning("dtm: as_of from run date fallback for %s records", run_count)
    for key in _AS_OF_FALLBACK_COUNTS:
        _AS_OF_FALLBACK_COUNTS[key] = 0


def _pluralise_sources(count: int) -> str:
    return "source" if count == 1 else "sources"


def _header_only_reason(invalid_count: int, kept: int) -> str:
    parts = ["header-only"]
    if invalid_count:
        parts.append(f"{invalid_count} {_pluralise_sources(invalid_count)} invalid")
    parts.append(f"kept={kept}")
    return "; ".join(parts)


def _nonempty_reason(kept: int, dropped: int, parse_errors: int, invalid_count: int) -> str:
    core = f"kept={kept}, dropped={dropped}, parse_errors={parse_errors}"
    if invalid_count:
        return f"{core}; invalid_sources={invalid_count}"
    return core


def write_rows(rows: List[List[Any]]) -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        writer.writerows(rows)
    ensure_manifest_for_csv(OUT_PATH, schema_version="dtm_displacement.v1", source_id="dtm")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fail-on-missing-config",
        action="store_true",
        help="Exit with code 2 when any DTM source entry lacks id_or_path.",
    )
    parser.add_argument(
        "--strict-empty",
        action="store_true",
        help="Exit with code 3 when no rows are written (after diagnostics).",
    )
    parser.add_argument(
        "--no-date-filter",
        action="store_true",
        help="Disable the ingestion window filter for DTM sources.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    from resolver.ingestion.dtm_auth import check_api_key_configured

    args = parse_args(argv or ())
    level_name = str(os.getenv("LOG_LEVEL") or "INFO").upper()
    log_level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=log_level, format="[%(levelname)s] %(message)s")
    LOG.setLevel(log_level)

    # Check API key configuration early for diagnostics
    api_key_configured = check_api_key_configured()
    if api_key_configured:
        LOG.info("DTM API key is configured")
    else:
        LOG.warning(
            "DTM API key NOT configured - connector will run in limited mode. "
            "Set DTM_API_KEY environment variable to fetch live data from DTM API."
        )

    diagnostics_ctx = diagnostics_start_run("dtm_client", "real")
    http_stats: Dict[str, Any] = {
        "2xx": 0,
        "4xx": 0,
        "5xx": 0,
        "retries": 0,
        "rate_limit_remaining": None,
        "last_status": None,
    }
    counts: Dict[str, int] = {"fetched": 0, "normalized": 0, "written": 0}
    extras: Dict[str, Any] = {
        "status_raw": "ok",
        "attempts": 1,
        "rows_total": 0,
        "api_key_configured": api_key_configured,
    }

    status_raw = "ok"
    reason: Optional[str] = None
    exit_code = 0
    strict = args.fail_on_missing_config or _env_bool("DTM_STRICT", False)
    strict_empty = args.strict_empty or _env_bool("DTM_STRICT_EMPTY", False)
    no_date_filter = args.no_date_filter or _env_bool("DTM_NO_DATE_FILTER", False)
    window_start_dt, window_end_dt = resolve_ingestion_window()
    window_start_iso = window_start_dt.isoformat() if window_start_dt else None
    window_end_iso = window_end_dt.isoformat() if window_end_dt else None
    window_disabled = no_date_filter or not (window_start_iso and window_end_iso)
    extras["strict_mode"] = strict
    extras["strict_empty"] = strict_empty
    extras["no_date_filter"] = no_date_filter
    if window_start_iso:
        extras["window_start"] = window_start_iso
    if window_end_iso:
        extras["window_end"] = window_end_iso
    extras["window_disabled"] = window_disabled

    source_results: List[SourceResult] = []
    invalid_entries: list[dict[str, Any]] = []
    valid_sources: list[dict[str, Any]] = []
    invalid_names: list[str] = []
    rows_written = 0
    config_invalid_count = 0
    runtime_invalid: list[dict[str, str]] = []
    sources_raw: List[Mapping[str, Any]] = []
    config_timestamp: Optional[str] = None

    try:
        if os.getenv("RESOLVER_SKIP_DTM"):
            status_raw = "skipped"
            reason = "disabled via RESOLVER_SKIP_DTM"
            extras["status_raw"] = status_raw
            LOG.info("dtm: skipped via RESOLVER_SKIP_DTM")
            ensure_header_only()
        else:
            cfg = load_config()
            if not cfg.get("enabled"):
                status_raw = "skipped"
                reason = "disabled: config"
                extras["status_raw"] = status_raw
                LOG.info("dtm: disabled via config; writing header only")
                ensure_header_only()
            else:
                sources_raw = cfg.get("sources") or []
                valid_sources, invalid_entries = validate_required_fields(
                    sources_raw, required=("id_or_path",)
                )
                config_invalid_count = len(invalid_entries)
                valid_count = len(valid_sources)
                config_timestamp = (
                    datetime.now(timezone.utc)
                    .replace(microsecond=0)
                    .isoformat()
                    .replace("+00:00", "Z")
                )
                issues_payload: Dict[str, Any] = {
                    "generated_at": config_timestamp,
                    "summary": {"invalid": config_invalid_count, "valid": valid_count},
                    "invalid": [],
                }
                for entry in invalid_entries:
                    record = dict(entry)
                    missing = list(record.pop("_missing_required", []))
                    invalid_name = _source_label(record)
                    if invalid_name:
                        invalid_names.append(invalid_name)
                    issues_payload["invalid"].append(
                        {**record, "error": "missing id_or_path", "missing": missing}
                    )
                write_json(CONFIG_ISSUES_PATH, issues_payload)
                extras["config_issues_path"] = str(CONFIG_ISSUES_PATH)
                extras.update(
                    {
                        "invalid_sources": config_invalid_count,
                        "valid_sources": valid_count,
                    }
                )
                if invalid_names:
                    extras["invalid_source_names"] = invalid_names
                if config_invalid_count and strict:
                    status_raw = "error"
                    reason = "missing id_or_path"
                    extras["status_raw"] = status_raw
                    exit_code = 2
                    LOG.error(
                        "dtm: missing id_or_path for %s source(s); strict mode aborting",
                        config_invalid_count,
                    )
                    ensure_header_only()
                else:
                    filtered_cfg = dict(cfg)
                    filtered_cfg["sources"] = valid_sources
                    if LOG.isEnabledFor(logging.DEBUG):
                        resolved_payload = {
                            "generated_at": config_timestamp,
                            "invalid_sources": config_invalid_count,
                            "valid_sources": valid_count,
                            "sources": valid_sources,
                        }
                        write_json(RESOLVED_SOURCES_PATH, resolved_payload)
                        extras["sources_resolved_path"] = str(RESOLVED_SOURCES_PATH)
                    start_count = len(source_results)
                    rows = build_rows(
                        filtered_cfg,
                        results=source_results,
                        no_date_filter=no_date_filter,
                        window_start=window_start_iso,
                        window_end=window_end_iso,
                        http_counts=http_stats,
                    )
                    new_results = source_results[start_count:]
                    for result in new_results:
                        if result.status == "invalid" and result.skip_reason:
                            runtime_invalid.append(
                                {
                                    "name": result.source_name,
                                    "reason": result.skip_reason,
                                    "rows_before": result.rows_before,
                                    "parse_errors": result.parse_errors,
                                }
                            )
                            invalid_names.append(result.source_name)
                    write_rows(rows)
                    rows_written = len(rows)
                    counts.update(
                        {
                            "fetched": rows_written,
                            "normalized": rows_written,
                            "written": rows_written,
                        }
                    )
                    extras["rows_total"] = rows_written
                    if source_results:
                        extras["sources"] = [
                            {
                                "name": result.source_name,
                                "status": result.status,
                                "rows": result.rows,
                                "skip_reason": result.skip_reason,
                                "http_counts": dict(result.http_counts),
                            }
                            for result in source_results
                        ]
                    skipped_from_run = [
                        {"name": res.source_name, "reason": res.skip_reason}
                        for res in source_results
                        if res.status == "skipped" and res.skip_reason
                    ]
                    if invalid_names or skipped_from_run:
                        extras.setdefault("skipped_sources", [])
                        extras["skipped_sources"].extend(
                            {"name": name, "reason": "missing id_or_path"}
                            for name in invalid_names
                        )
                        extras["skipped_sources"].extend(skip for skip in skipped_from_run if skip)
                    if rows_written:
                        if config_invalid_count or runtime_invalid:
                            LOG.warning(
                                "dtm: wrote %s rows with %s invalid source(s)",
                                rows_written,
                                config_invalid_count + len(runtime_invalid),
                            )
                        else:
                            LOG.info("dtm: wrote %s rows", rows_written)
                    else:
                        total_invalid = config_invalid_count + len(runtime_invalid)
                        if total_invalid:
                            LOG.warning(
                                "dtm: header-only output; %s invalid source(s)",
                                total_invalid,
                            )
                        else:
                            LOG.info("dtm: wrote header only (no rows)")
    except Exception as exc:  # pragma: no cover - defensive guard
        status_raw = "error"
        reason = str(exc)
        extras["status_raw"] = status_raw
        extras["exception"] = str(exc)
        exit_code = 1
        LOG.exception("dtm: run failed: %s", exc)
        try:
            ensure_header_only()
        except Exception:
            pass
    finally:
        actual_rows = count_csv_rows(OUT_PATH)
        rows_written = actual_rows
        counts.update(
            {
                "fetched": max(counts.get("fetched", 0), actual_rows),
                "normalized": max(counts.get("normalized", 0), actual_rows),
                "written": actual_rows,
            }
        )

        rows_before_total = sum(result.rows_before for result in source_results)
        rows_after_total = sum(result.rows_after for result in source_results)
        dropped_total = sum(result.dropped for result in source_results)
        parse_errors_total = sum(result.parse_errors for result in source_results)
        min_dates = [result.min_date for result in source_results if result.min_date]
        max_dates = [result.max_date for result in source_results if result.max_date]
        min_overall = min(min_dates) if min_dates else None
        max_overall = max(max_dates) if max_dates else None
        runtime_invalid_count = len(runtime_invalid)
        total_invalid = config_invalid_count + runtime_invalid_count

        if LOG.isEnabledFor(logging.DEBUG):
            LOG.debug(
                "dtm: window=%s→%s (disabled=%s), parsed min=%s, max=%s, kept=%s, dropped=%s (parse_errors=%s), invalid_sources=%s",
                window_start_iso or "—",
                window_end_iso or "—",
                window_disabled,
                min_overall or "—",
                max_overall or "—",
                rows_after_total,
                dropped_total,
                parse_errors_total,
                total_invalid,
            )

        extras.setdefault("invalid_sources_config", config_invalid_count)
        extras.setdefault("valid_sources", len(valid_sources))
        extras["invalid_sources"] = total_invalid
        extras["rows_total"] = actual_rows
        extras["rows_written"] = actual_rows
        extras["missing_sources"] = total_invalid
        extras["kept_rows"] = rows_after_total
        extras["dropped_rows"] = dropped_total
        extras["parse_errors"] = parse_errors_total
        if min_overall:
            extras["parsed_min"] = min_overall
        if max_overall:
            extras["parsed_max"] = max_overall

        meta_payload: Dict[str, Any] = {}
        try:
            if META_PATH.exists():
                meta_payload = json.loads(META_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            meta_payload = {}
        meta_payload["row_count"] = actual_rows
        if window_start_iso:
            meta_payload["backfill_start"] = window_start_iso
        if window_end_iso:
            meta_payload["backfill_end"] = window_end_iso
        meta_payload["sources_total"] = len(sources_raw)
        meta_payload["sources_valid"] = max(len(valid_sources) - runtime_invalid_count, 0)
        meta_payload["sources_invalid"] = total_invalid
        write_json(META_PATH, meta_payload)
        extras["meta_path"] = str(META_PATH)

        valid_payload: List[Dict[str, Any]] = []
        invalid_payload: List[Dict[str, Any]] = []
        for result in source_results:
            payload: Dict[str, Any] = {
                "name": result.source_name,
                "status": result.status,
                "rows": result.rows,
                "rows_before": result.rows_before,
                "rows_after": result.rows_after,
                "dropped": result.dropped,
                "parse_errors": result.parse_errors,
                "min": result.min_date,
                "max": result.max_date,
            }
            if result.skip_reason:
                payload["skip_reason"] = result.skip_reason
                payload.setdefault("reason", result.skip_reason)
            if result.error:
                payload["error"] = result.error
            if result.http_counts:
                payload["http_counts"] = dict(result.http_counts)
            if result.status == "invalid":
                invalid_payload.append(payload)
            else:
                valid_payload.append(payload)

        for entry in invalid_entries:
            record = dict(entry)
            missing_fields = list(record.pop("_missing_required", []))
            invalid_payload.append(
                {
                    "name": _source_label(record),
                    "reason": "missing id_or_path",
                    "missing": missing_fields,
                    "details": record,
                }
            )

        run_payload: Dict[str, Any] = {
            "generated_at": datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
            "window": {
                "start": window_start_iso,
                "end": window_end_iso,
                "disabled": window_disabled,
            },
            "rows_written": actual_rows,
            "missing_id_or_path": config_invalid_count,
            "outputs": {
                "csv": str(OUT_PATH),
                "meta": str(META_PATH),
            },
            "sources": {"valid": valid_payload, "invalid": invalid_payload},
            "totals": {
                "rows_before": rows_before_total,
                "rows_after": rows_after_total,
                "rows_written": actual_rows,
                "kept": rows_after_total,
                "dropped": dropped_total,
                "parse_errors": parse_errors_total,
                "invalid_sources": total_invalid,
                "min": min_overall,
                "max": max_overall,
            },
        }
        write_json(RUN_DETAILS_PATH, run_payload)
        extras["run_details_path"] = str(RUN_DETAILS_PATH)

        final_status = status_raw
        final_reason = reason
        if final_status not in {"error", "skipped"}:
            if actual_rows == 0:
                final_status = "ok-empty"
                final_reason = _header_only_reason(total_invalid, rows_after_total)
            else:
                final_status = "ok"
                final_reason = _nonempty_reason(
                    rows_after_total,
                    dropped_total,
                    parse_errors_total,
                    total_invalid,
                )

        if strict_empty and final_status == "ok-empty" and exit_code == 0:
            exit_code = 3
            LOG.error("dtm: strict empty mode aborting (no rows written)")

        status_raw = final_status
        reason = final_reason
        extras["status_raw"] = status_raw
        extras["exit_code"] = exit_code

        diagnostics_result = diagnostics_finalize_run(
            diagnostics_ctx,
            status=status_raw,
            reason=reason,
            http=http_stats,
            counts=counts,
            extras=extras,
        )
        diagnostics_append_jsonl(CONNECTORS_REPORT, diagnostics_result)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
