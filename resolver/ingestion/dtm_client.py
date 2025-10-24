#!/usr/bin/env python3
"""DTM connector that converts stock or flow tables into monthly displacement flows."""

from __future__ import annotations

import argparse
import csv
import json
import re
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
import time
import yaml

from resolver.ingestion._manifest import ensure_manifest_for_csv
from resolver.ingestion.dtm_auth import get_dtm_api_key
from resolver.ingestion._shared.run_io import count_csv_rows, write_json
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
API_REQUEST_PATH = DIAGNOSTICS_DIR / "dtm_api_request.json"
API_RESPONSE_SAMPLE_PATH = DIAGNOSTICS_DIR / "dtm_api_response_sample.json"

DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)

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


HTTP_COUNT_KEYS = ("2xx", "4xx", "5xx", "timeout", "error")
ROW_COUNT_KEYS = ("admin0", "admin1", "admin2", "total")


def _empty_http_counts() -> Dict[str, int]:
    return {key: 0 for key in HTTP_COUNT_KEYS}


def _ensure_http_count_keys(counts: Optional[Dict[str, int]]) -> Dict[str, int]:
    if counts is None:
        counts = {}
    for key in HTTP_COUNT_KEYS:
        counts.setdefault(key, 0)
    return counts


def _empty_row_counts() -> Dict[str, int]:
    return {key: 0 for key in ROW_COUNT_KEYS}


def _extract_status_code(error: BaseException) -> Optional[int]:
    """Best-effort parse of an HTTP status code from an exception."""

    message = str(error)
    for token in re.findall(r"\b(\d{3})\b", message):
        try:
            code = int(token)
        except ValueError:
            continue
        if 400 <= code < 600:
            return code
    return None


def _is_auth_status(code: Optional[int]) -> bool:
    return code in {401, 403}

def _record_http_failure(exc: Exception, http_counts: Optional[Dict[str, int]]) -> None:
    """Update HTTP counters and raise for actionable API failures."""

    if http_counts is not None:
        message = str(exc).lower()
        if "timeout" in message:
            http_counts["timeout"] = http_counts.get("timeout", 0) + 1
        elif "404" in message or "not found" in message:
            http_counts["4xx"] = http_counts.get("4xx", 0) + 1
        elif "401" in message or "403" in message:
            http_counts["4xx"] = http_counts.get("4xx", 0) + 1
        elif "500" in message or "server" in message:
            http_counts["5xx"] = http_counts.get("5xx", 0) + 1
        else:
            http_counts["error"] = http_counts.get("error", 0) + 1

    status = _extract_status_code(exc)
    if status is not None and 400 <= status < 600:
        message = str(exc)
        if _is_auth_status(status):
            raise DTMUnauthorizedError(status, message) from exc
        raise DTMHttpError(status, message) from exc


class DTMHttpError(RuntimeError):
    """Error raised when the DTM API returns an HTTP error status."""

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


class DTMUnauthorizedError(DTMHttpError):
    """Error raised specifically for 401/403 authentication failures."""

    def __init__(self, status_code: int, message: str):
        super().__init__(status_code, message)


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
    http_counts: Dict[str, int] = field(default_factory=_empty_http_counts)
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
    """Wrapper around official dtmapi.DTMApi client."""

    def __init__(self, config: dict, *, subscription_key: Optional[str] = None):
        """Initialize DTM API client using official dtmapi package.

        Args:
            config: Configuration dictionary with 'api' settings.
            subscription_key: Optional explicit API key to use.
        """
        try:
            from dtmapi import DTMApi
        except ImportError as exc:
            LOG.error(
                "Failed to import dtmapi package. Install with: pip install dtmapi>=0.1.5"
            )
            raise RuntimeError(
                "dtmapi package not installed. Run: pip install dtmapi>=0.1.5"
            ) from exc

        if subscription_key is None:
            api_key = get_dtm_api_key()
        else:
            api_key = subscription_key.strip()
            if not api_key:
                api_key = None
        if not api_key:
            raise ValueError("DTM API key not configured")

        self.subscription_key = api_key

        # Initialize official client
        self.client = DTMApi(subscription_key=api_key)

        # Store config for reference
        self.config = config
        api_cfg = config.get("api", {})
        self.rate_limit_delay = api_cfg.get("rate_limit_delay", 1.0)
        self.timeout = api_cfg.get("timeout", 60)

        # Log package version if available
        try:
            import dtmapi
            version = getattr(dtmapi, "__version__", "unknown")
            LOG.info("DTM API client initialized using dtmapi package v%s", version)
        except Exception:
            LOG.info("DTM API client initialized using official dtmapi package")

        # Test connection by fetching countries
        try:
            LOG.info("Testing DTM API connection...")
            countries = self.get_countries()
            LOG.info(
                "✓ DTM API connection successful (%d countries available)",
                len(countries),
            )
        except Exception as e:
            LOG.warning("✗ DTM API connection test failed: %s", e)
            LOG.warning("Will attempt to fetch data anyway...")

    def get_countries(
        self, http_counts: Optional[Dict[str, int]] = None
    ) -> pd.DataFrame:
        """Get list of all available countries.

        Args:
            http_counts: Dictionary to update with HTTP status counts

        Returns:
            DataFrame with country information
        """
        try:
            LOG.debug("Fetching countries list")
            df = self.client.get_all_countries()

            # Track success
            if http_counts is not None:
                http_counts["2xx"] = http_counts.get("2xx", 0) + 1

            LOG.info("Successfully fetched %d countries", len(df))
            return df
        except Exception as e:
            LOG.error("Failed to fetch countries: %s", e)
            _record_http_failure(e, http_counts)
            return pd.DataFrame()

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

        Uses official dtmapi.DTMApi.get_idp_admin0_data() method.

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
        try:
            LOG.debug(
                "Fetching Admin0 data: country=%s, from_date=%s, to_date=%s",
                country,
                from_date,
                to_date,
            )

            # Call official API method
            df = self.client.get_idp_admin0_data(
                CountryName=country,
                FromReportingDate=from_date,
                ToReportingDate=to_date,
                FromRoundNumber=from_round,
                ToRoundNumber=to_round,
            )

            # Track success
            if http_counts is not None:
                http_counts["2xx"] = http_counts.get("2xx", 0) + 1

            LOG.info("Successfully fetched %d Admin0 records", len(df))

            # Rate limiting
            if self.rate_limit_delay > 0:
                time.sleep(self.rate_limit_delay)

            return df

        except Exception as e:
            LOG.error("Failed to fetch Admin0 data: %s", e)
            _record_http_failure(e, http_counts)
            return pd.DataFrame()

    def get_idp_admin1(
        self,
        country: Optional[str] = None,
        admin1: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        http_counts: Optional[Dict[str, int]] = None,
    ) -> pd.DataFrame:
        """Fetch IDP data at state/province level (Admin 1).

        Uses official dtmapi.DTMApi.get_idp_admin1_data() method.

        Args:
            country: Filter by country name
            admin1: Filter by Admin 1 name
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            http_counts: Dictionary to update with HTTP status counts

        Returns:
            DataFrame with Admin 1 IDP data
        """
        try:
            LOG.debug(
                "Fetching Admin1 data: country=%s, admin1=%s, from_date=%s, to_date=%s",
                country,
                admin1,
                from_date,
                to_date,
            )

            # Call official API method
            df = self.client.get_idp_admin1_data(
                CountryName=country,
                Admin1Name=admin1,
                FromReportingDate=from_date,
                ToReportingDate=to_date,
            )

            # Track success
            if http_counts is not None:
                http_counts["2xx"] = http_counts.get("2xx", 0) + 1

            LOG.info("Successfully fetched %d Admin1 records", len(df))

            # Rate limiting
            if self.rate_limit_delay > 0:
                time.sleep(self.rate_limit_delay)

            return df

        except Exception as e:
            LOG.error("Failed to fetch Admin1 data: %s", e)
            _record_http_failure(e, http_counts)
            return pd.DataFrame()

    def get_idp_admin2(
        self,
        country: Optional[str] = None,
        operation: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        http_counts: Optional[Dict[str, int]] = None,
    ) -> pd.DataFrame:
        """Fetch IDP data at district level (Admin 2).

        Uses official dtmapi.DTMApi.get_idp_admin2_data() method.

        Args:
            country: Filter by country name
            operation: Filter by operation type
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            http_counts: Dictionary to update with HTTP status counts

        Returns:
            DataFrame with Admin 2 IDP data
        """
        try:
            LOG.debug(
                "Fetching Admin2 data: country=%s, operation=%s, from_date=%s, to_date=%s",
                country,
                operation,
                from_date,
                to_date,
            )

            # Call official API method
            df = self.client.get_idp_admin2_data(
                CountryName=country,
                Operation=operation,
                FromReportingDate=from_date,
                ToReportingDate=to_date,
            )

            # Track success
            if http_counts is not None:
                http_counts["2xx"] = http_counts.get("2xx", 0) + 1

            LOG.info("Successfully fetched %d Admin2 records", len(df))

            # Rate limiting
            if self.rate_limit_delay > 0:
                time.sleep(self.rate_limit_delay)

            return df

        except Exception as e:
            LOG.error("Failed to fetch Admin2 data: %s", e)
            _record_http_failure(e, http_counts)
            return pd.DataFrame()


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






def _fetch_api_data(
    cfg: Mapping[str, Any],
    *,
    results: Optional[List[SourceResult]] = None,
    no_date_filter: bool = False,
    window_start: Optional[str] = None,
    window_end: Optional[str] = None,
    http_counts: Optional[Dict[str, int]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Fetch IDP data from DTM API for all configured admin levels."""

    from resolver.ingestion.dtm_auth import check_api_key_configured

    if results is None:
        results = []
    http_counter = _ensure_http_count_keys(http_counts)
    summary: Dict[str, Any] = {
        "row_counts": _empty_row_counts(),
        "http_counts": _empty_http_counts(),
    }

    if not check_api_key_configured():
        raise RuntimeError("DTM API key not configured")

    primary_key = os.environ.get("DTM_API_KEY", "").strip()
    secondary_key = os.environ.get("DTM_API_SECONDARY_KEY", "").strip() or None

    all_records: List[Dict[str, Any]] = []

    try:
        client = DTMApiClient(cfg, subscription_key=primary_key)
    except (RuntimeError, ValueError) as exc:
        LOG.error("Failed to initialize DTM API client: %s", exc)
        results.append(
            SourceResult(
                source_name="dtm_api",
                status="error",
                error=str(exc),
            )
        )
        raise

    admin_levels_cfg = cfg.get("admin_levels", ["admin0", "admin1", "admin2"])
    admin_levels = [str(level).strip().lower() for level in admin_levels_cfg]

    countries_cfg = cfg.get("countries")
    if countries_cfg is None or countries_cfg == []:
        country_targets: List[Optional[str]] = [None]
    elif isinstance(countries_cfg, str):
        country_targets = [countries_cfg]
    else:
        country_targets = list(countries_cfg) or [None]

    operations_cfg = cfg.get("operations")
    if operations_cfg is None or operations_cfg == []:
        operation_targets: List[Optional[str]] = [None]
    elif isinstance(operations_cfg, str):
        operation_targets = [operations_cfg]
    else:
        operation_targets = list(operations_cfg)

    from_date = window_start if not no_date_filter and window_start else None
    to_date = window_end if not no_date_filter and window_end else None

    request_payload = {
        "admin_levels": admin_levels,
        "countries": None
        if country_targets == [None]
        else [str(c) for c in country_targets],
        "operations": None
        if operation_targets == [None]
        else [str(o) for o in operation_targets],
        "window_start": from_date,
        "window_end": to_date,
    }
    write_json(API_REQUEST_PATH, request_payload)

    LOG.info(
        "Fetching DTM data: levels=%s, countries=%s, date_range=%s to %s",
        admin_levels,
        request_payload["countries"] or "all",
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

    idp_field_candidates = field_aliases.get("idp_count", ["TotalIDPs", "IDPTotal"])
    if field_mapping.get("idp_column"):
        idp_field_candidates.insert(0, field_mapping["idp_column"])

    aliases = cfg.get("country_aliases") or {}
    measure = cfg.get("output", {}).get("measure", "stock")
    cause_map = {
        str(k).strip().lower(): str(v)
        for k, v in (cfg.get("cause_map") or {}).items()
    }

    used_secondary = False

    def _fetch_level(
        level_name: str,
        *,
        country: Optional[str],
        operation: Optional[str],
    ) -> pd.DataFrame:
        nonlocal client, used_secondary
        while True:
            try:
                if level_name == "admin0":
                    return client.get_idp_admin0(
                        country=country,
                        from_date=from_date,
                        to_date=to_date,
                        http_counts=http_counter,
                    )
                if level_name == "admin1":
                    return client.get_idp_admin1(
                        country=country,
                        from_date=from_date,
                        to_date=to_date,
                        http_counts=http_counter,
                    )
                if level_name == "admin2":
                    return client.get_idp_admin2(
                        country=country,
                        operation=operation,
                        from_date=from_date,
                        to_date=to_date,
                        http_counts=http_counter,
                    )
                raise ValueError(f"Unsupported admin level: {level_name}")
            except DTMUnauthorizedError as exc:
                if secondary_key and not used_secondary:
                    LOG.warning(
                        "Primary DTM API key rejected (%s); retrying with secondary key",
                        exc.status_code,
                    )
                    used_secondary = True
                    http_counter["retries"] = http_counter.get("retries", 0) + 1
                    client = DTMApiClient(cfg, subscription_key=secondary_key)
                    continue
                raise

    for admin_level in admin_levels:
        level_name = admin_level.strip().lower()

        for country in country_targets:
            operation_list = operation_targets if level_name == "admin2" else [None]
            for operation in operation_list:
                source_label_parts = ["dtm_api", level_name]
                if country:
                    source_label_parts.append(str(country))
                if operation:
                    source_label_parts.append(str(operation))
                source_label = "_".join(source_label_parts)

                try:
                    df = _fetch_level(level_name, country=country, operation=operation)
                except DTMHttpError as exc:
                    LOG.error("Failed to fetch %s: %s", source_label, exc)
                    results.append(
                        SourceResult(
                            source_name=source_label,
                            status="error",
                            error=str(exc),
                            http_counts=dict(http_counter),
                        )
                    )
                    raise

                if round_column in df.columns:
                    df = df.sort_values(
                        by=[date_column, round_column], ascending=[True, True]
                    )

                rows_before = int(df.shape[0])
                if rows_before == 0:
                    LOG.debug("No data for %s", source_label)
                    results.append(
                        SourceResult(
                            source_name=source_label,
                            status="ok",
                            rows_before=0,
                            rows_after=0,
                            http_counts=dict(http_counter),
                        )
                    )
                    continue

                LOG.debug("Fetched %s rows for %s", rows_before, source_label)

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
                            http_counts=dict(http_counter),
                        )
                    )
                    continue

                per_admin: dict[tuple[str, str], dict[datetime, float]] = defaultdict(dict)
                per_admin_asof: dict[tuple[str, str], dict[datetime, str]] = defaultdict(dict)
                causes: dict[tuple[str, str], str] = {}

                new_rows = 0
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
                        admin2 = str(row.get(admin2_column) or "").strip()
                        if admin2:
                            admin1 = f"{admin1}/{admin2}" if admin1 else admin2

                    value = _parse_float(row.get(idp_column))
                    if value is None or value < 0:
                        continue

                    per_admin[(iso, admin1)][bucket] = value

                    as_of_value = str(row.get(date_column) or "")
                    if as_of_value:
                        as_of_value = _parse_iso_date_or_none(as_of_value) or as_of_value
                    if not as_of_value:
                        as_of_value = datetime.now(timezone.utc).date().isoformat()

                    existing_asof = per_admin_asof[(iso, admin1)].get(bucket)
                    if not existing_asof or _is_candidate_newer(existing_asof, as_of_value):
                        per_admin_asof[(iso, admin1)][bucket] = as_of_value

                    causes[(iso, admin1)] = _resolve_cause(row.to_dict(), cause_map)

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
                        new_rows += 1

                summary["row_counts"].setdefault(level_name, 0)
                summary["row_counts"][level_name] += new_rows
                summary["row_counts"]["total"] += new_rows

                range_label = f"{from_date or '—'}→{to_date or '—'}"
                if country:
                    iso_candidate = to_iso3(country, aliases)
                    country_label = iso_candidate or str(country)
                else:
                    country_label = "all countries"
                if level_name == "admin2" and operation:
                    country_label = f"{country_label} (operation={operation})"
                LOG.info(
                    "Fetched %s rows (%s) for %s %s",
                    f"{new_rows:,}",
                    level_name,
                    country_label,
                    range_label,
                )

                results.append(
                    SourceResult(
                        source_name=source_label,
                        status="ok",
                        rows_before=rows_before,
                        rows_after=new_rows,
                        http_counts=dict(http_counter),
                    )
                )

    summary["http_counts"] = {
        key: int(http_counter.get(key, 0)) for key in HTTP_COUNT_KEYS
    }
    LOG.info("Total records fetched from API: %s", len(all_records))
    return all_records, summary



def build_rows(
    cfg: Mapping[str, Any],
    *,
    results: Optional[List[SourceResult]] = None,
    no_date_filter: bool = False,
    window_start: Optional[str] = None,
    window_end: Optional[str] = None,
    http_counts: Optional[Dict[str, int]] = None,
    diagnostics: Optional[MutableMapping[str, Any]] = None,
) -> List[List[Any]]:
    if "api" not in cfg:
        raise ValueError(
            "Config error: DTM is API-only; provide an 'api:' block in "
            "resolver/ingestion/config/dtm.yml"
        )

    sources = cfg.get("sources") or []
    if sources:
        LOG.warning("DTM is API-only; 'sources' section is ignored.")

    http_stats = _ensure_http_count_keys(http_counts)
    admin_mode = str(cfg.get("admin_agg") or "both").strip().lower()
    all_records: List[Dict[str, Any]] = []
    collected = results if results is not None else []

    diag_target = diagnostics if diagnostics is not None else None
    if diag_target is not None:
        admin_levels_cfg = cfg.get("admin_levels") or ["admin0", "admin1", "admin2"]
        diag_target["admin_levels"] = [
            str(level).strip().lower() for level in admin_levels_cfg
        ]
        countries_cfg = cfg.get("countries")
        if isinstance(countries_cfg, str):
            diag_target["countries"] = [countries_cfg]
        else:
            diag_target["countries"] = list(countries_cfg) if countries_cfg else None
        operations_cfg = cfg.get("operations")
        if isinstance(operations_cfg, str):
            diag_target["operations"] = [operations_cfg]
        else:
            diag_target["operations"] = (
                list(operations_cfg) if operations_cfg else None
            )
        diag_target["window_start"] = None if no_date_filter else window_start
        diag_target["window_end"] = None if no_date_filter else window_end
        diag_target["mode"] = "api"
        diag_target["trigger"] = "api-only"

    LOG.info("Using DTM API mode (api-only)")
    api_records, api_summary = _fetch_api_data(
        cfg,
        results=collected,
        no_date_filter=no_date_filter,
        window_start=window_start,
        window_end=window_end,
        http_counts=http_stats,
    )
    all_records.extend(api_records)
    if diag_target is not None:
        diag_target["row_counts"] = api_summary.get("row_counts", _empty_row_counts())
        diag_target["http_counts"] = api_summary.get(
            "http_counts", _empty_http_counts()
        )

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
                "confidence": DEFAULT_CAUSE,
                "raw_event_id": f"{rec['source_id']}::{admin1 or 'country'}::{month.strftime('%Y%m')}",
                "raw_fields_json": json.dumps(
                    {
                        "source_id": rec["source_id"],
                        "admin1": admin1,
                        "month": month_iso,
                        "value": value,
                        "cause": rec.get("cause", DEFAULT_CAUSE),
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
    try:
        write_json(API_RESPONSE_SAMPLE_PATH, formatted[:100])
    except Exception as exc:
        LOG.warning('Failed to write DTM API response sample: %s', exc)
    if diag_target is not None:
        diag_rows = diag_target.get("row_counts") or _empty_row_counts()
        diag_rows = {**_empty_row_counts(), **diag_rows}
        diag_rows["total"] = len(formatted)
        diag_target["row_counts"] = diag_rows
        if "http_counts" not in diag_target:
            diag_target["http_counts"] = {
                key: int(http_stats.get(key, 0)) for key in HTTP_COUNT_KEYS
            }
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

    api_key_configured = check_api_key_configured()
    if api_key_configured:
        LOG.info("DTM API key is configured")
    else:
        LOG.error(
            "DTM API key NOT configured - connector cannot run. Set DTM_API_KEY to continue."
        )

    diagnostics_ctx = diagnostics_start_run("dtm_client", "real")
    http_stats: Dict[str, Any] = {
        "2xx": 0,
        "4xx": 0,
        "5xx": 0,
        "timeout": 0,
        "error": 0,
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
    strict_empty = args.strict_empty or _env_bool("DTM_STRICT_EMPTY", False)
    no_date_filter = args.no_date_filter or _env_bool("DTM_NO_DATE_FILTER", False)
    window_start_dt, window_end_dt = resolve_ingestion_window()
    window_start_iso = window_start_dt.isoformat() if window_start_dt else None
    window_end_iso = window_end_dt.isoformat() if window_end_dt else None
    window_disabled = no_date_filter or not (window_start_iso and window_end_iso)
    extras["strict_empty"] = strict_empty
    extras["no_date_filter"] = no_date_filter
    if window_start_iso:
        extras["window_start"] = window_start_iso
    if window_end_iso:
        extras["window_end"] = window_end_iso
    extras["window_disabled"] = window_disabled

    source_results: List[SourceResult] = []
    run_diagnostics: Dict[str, Any] = {}
    rows_written = 0
    cfg: Mapping[str, Any] = {}

    try:
        if os.getenv("RESOLVER_SKIP_DTM"):
            status_raw = "skipped"
            reason = "disabled via RESOLVER_SKIP_DTM"
            LOG.info("dtm: skipped via RESOLVER_SKIP_DTM")
            ensure_header_only()
        else:
            cfg = load_config()
            if not cfg.get("enabled"):
                status_raw = "skipped"
                reason = "disabled: config"
                LOG.info("dtm: disabled via config; writing header only")
                ensure_header_only()
            else:
                filtered_cfg = dict(cfg)
                sources_cfg = filtered_cfg.pop("sources", None)
                if sources_cfg:
                    LOG.warning("DTM is API-only; 'sources' section is ignored.")
                rows = build_rows(
                    filtered_cfg,
                    results=source_results,
                    no_date_filter=no_date_filter,
                    window_start=window_start_iso,
                    window_end=window_end_iso,
                    http_counts=http_stats,
                    diagnostics=run_diagnostics,
                )
                rows_written = len(rows)
                counts["fetched"] = rows_written
                counts["normalized"] = rows_written
                extras["rows_total"] = rows_written
                if rows:
                    write_rows(rows)
                    counts["written"] = rows_written
                    LOG.info("dtm: wrote %s rows to %s", rows_written, OUT_PATH)
                else:
                    write_json(API_RESPONSE_SAMPLE_PATH, [])
                    ensure_header_only()
                    LOG.info("dtm: no rows returned; wrote header only")
    except ValueError as exc:
        LOG.error("dtm: %s", exc)
        status_raw = "error"
        reason = str(exc)
        exit_code = 2
    except DTMHttpError as exc:
        LOG.error("dtm: API error %s", exc)
        status_raw = "error"
        reason = str(exc)
        exit_code = 2
    except RuntimeError as exc:
        LOG.error("dtm: runtime error %s", exc)
        status_raw = "error"
        reason = str(exc)
        exit_code = 2
    except Exception as exc:  # pragma: no cover - defensive
        LOG.exception("dtm: unexpected error")
        status_raw = "error"
        reason = str(exc)
        exit_code = 2

    actual_rows = rows_written
    if rows_written and exit_code == 0:
        extras["output_path"] = str(OUT_PATH)
    if not rows_written and OUT_PATH.exists():
        try:
            actual_rows = count_csv_rows(OUT_PATH)
        except OSError:
            actual_rows = 0

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
    meta_payload["sources_total"] = len(source_results)
    meta_payload["sources_valid"] = sum(1 for result in source_results if result.status == "ok")
    meta_payload["sources_invalid"] = sum(
        1 for result in source_results if result.status not in {"ok", "delegated"}
    )
    write_json(META_PATH, meta_payload)
    extras["meta_path"] = str(META_PATH)

    if status_raw not in {"error", "skipped"}:
        if actual_rows == 0:
            status_raw = "ok-empty"
            reason = _header_only_reason(0, 0)
        else:
            status_raw = "ok"
            reason = _nonempty_reason(actual_rows, 0, 0, 0)

    if strict_empty and status_raw == "ok-empty" and exit_code == 0:
        exit_code = 3
        LOG.error("dtm: strict empty mode aborting (no rows written)")

    extras["status_raw"] = status_raw
    extras["exit_code"] = exit_code

    diag_admin_levels = run_diagnostics.get("admin_levels") or [
        str(level).strip().lower()
        for level in (cfg.get("admin_levels") if 'cfg' in locals() and isinstance(cfg, Mapping) else ["admin0", "admin1", "admin2"])
    ]
    diag_countries = run_diagnostics.get("countries")
    diag_operations = run_diagnostics.get("operations")
    diag_http = _ensure_http_count_keys(run_diagnostics.get("http_counts"))
    diag_rows = run_diagnostics.get("row_counts") or _empty_row_counts()
    diag_rows = {**_empty_row_counts(), **diag_rows}
    if not diag_rows.get("total"):
        diag_rows["total"] = actual_rows
    run_payload: Dict[str, Any] = {
        "mode": run_diagnostics.get("mode", "api"),
        "trigger": run_diagnostics.get("trigger"),
        "admin_levels": diag_admin_levels,
        "countries": diag_countries,
        "operations": diag_operations,
        "window_start": None if window_disabled else window_start_iso,
        "window_end": None if window_disabled else window_end_iso,
        "http_counts": {key: int(diag_http.get(key, 0)) for key in HTTP_COUNT_KEYS},
        "row_counts": {key: int(diag_rows.get(key, 0)) for key in ROW_COUNT_KEYS},
    }
    write_json(RUN_DETAILS_PATH, run_payload)
    extras["run_details_path"] = str(RUN_DETAILS_PATH)

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
