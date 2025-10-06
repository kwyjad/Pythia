#!/usr/bin/env python3
"""
ReliefWeb API → staging/reliefweb.csv

- Queries recent reports (last N days) with basic filters
- Paginates with retries and backoff
- Maps to (iso3, hazard_code) via keyword heuristics
- Extracts PIN/PA (or cases) from title/summary using regex
- Writes canonical staging CSV expected by our exporter/validator

Usage:
  python resolver/ingestion/reliefweb_client.py
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
import yaml
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from resolver.ingestion._manifest import ensure_manifest_for_csv
from resolver.ingestion import _pdf_text as pdf_text_mod
from resolver.ingestion._pdf_text import smart_extract
from resolver.ingestion.utils.id_digest import stable_digest
from resolver.ingestion.utils.io import ensure_headers
from resolver.ingestion.utils.month_bucket import month_start

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
STAGING = ROOT / "staging"
CONFIG = ROOT / "ingestion" / "config" / "reliefweb.yml"
REFERENCE = ROOT / "reference"
PDF_CACHE = STAGING / ".cache" / "reliefweb" / "pdf"
LEVEL_CACHE = STAGING / ".cache" / "reliefweb" / "levels"

DEBUG = os.getenv("RESOLVER_DEBUG", "0") == "1"

LOGGER = logging.getLogger("resolver.ingestion.reliefweb")

COUNTRIES = DATA / "countries.csv"
SHOCKS = DATA / "shocks.csv"

# Canonical output columns
COLUMNS = [
    "event_id",
    "country_name",
    "iso3",
    "hazard_code",
    "hazard_label",
    "hazard_class",
    "metric",
    "series_semantics",
    "value",
    "unit",
    "as_of_date",
    "publication_date",
    "publisher",
    "source_type",
    "source_url",
    "doc_title",
    "definition_text",
    "method",
    "confidence",
    "revision",
    "ingested_at",
]

PDF_COLUMNS = [
    "event_id",
    "country_name",
    "iso3",
    "hazard_code",
    "hazard_label",
    "hazard_class",
    "metric",
    "series_semantics",
    "value",
    "unit",
    "value_level",
    "as_of_date",
    "month_start",
    "publication_date",
    "publisher",
    "source_type",
    "source_url",
    "resource_url",
    "doc_title",
    "definition_text",
    "method",
    "method_value",
    "method_details",
    "extraction_layer",
    "matched_phrase",
    "tier",
    "confidence",
    "revision",
    "ingested_at",
]

PDF_OUTPUT = STAGING / "reliefweb_pdf.csv"

DEFAULT_PPH = 4.5
PPH_TABLE = REFERENCE / "avg_household_size.csv"
PPH_OVERRIDES = REFERENCE / "overrides" / "avg_household_size_overrides.yml"

TABLE_METRIC_SYNONYMS = {
    "people_in_need": [
        "people in need",
        "in need",
        "total in need",
        "pin",
    ],
    "people_affected": [
        "people affected",
        "affected people",
        "population affected",
    ],
    "idps": [
        "internally displaced",
        "idp",
        "displaced (internal)",
        "displaced persons",
    ],
    "households_in_need": [
        "households in need",
        "hh in need",
    ],
    "households_affected": [
        "households affected",
        "hh affected",
    ],
}

INFOGRAPHIC_HEADINGS = [
    "key figures",
    "key figure",
    "at a glance",
    "snapshot",
]

NARRATIVE_PATTERNS = {
    "people_in_need": re.compile(
        r"(?P<value>[0-9][0-9.,\s]*\+?(?:\s*(?:k|m|million))?)\s+people\s+in\s+need",
        re.I,
    ),
    "people_affected": re.compile(
        r"(?P<value>[0-9][0-9.,\s]*\+?(?:\s*(?:k|m|million))?)\s+(?:people|persons)\s+affected",
        re.I,
    ),
    "idps": re.compile(
        r"(?P<value>[0-9][0-9.,\s]*\+?(?:\s*(?:k|m|million))?)\s+(?:idps?|internally displaced)",
        re.I,
    ),
}

HOUSEHOLD_TO_PEOPLE = {
    "households_in_need": "people_in_need",
    "households_affected": "people_affected",
}

LAYER_PRIORITY = {"table": 0, "infographic": 1, "narrative": 2}


@dataclass
class PPHEntry:
    iso3: str
    people_per_household: float
    source: str
    year: Optional[int]
    notes: str = ""

    def describe(self) -> str:
        bits = [f"PPH={self.people_per_household:.2f}"]
        if self.source:
            bits.append(f"source={self.source}")
        if self.year:
            bits.append(f"year={self.year}")
        if self.notes:
            bits.append(self.notes)
        return ", ".join(bits)


@dataclass
class MetricCandidate:
    metric: str
    value: int
    unit: str
    layer: str
    matched_phrase: str
    method_value: str = "reported"
    method_details: str = ""
    country_iso3: Optional[str] = None
    extraction_meta: Dict[str, Any] = field(default_factory=dict)


class PPHLookup:
    """Load people-per-household values with overrides."""

    def __init__(self) -> None:
        self._table: Dict[str, PPHEntry] = {}
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        table: Dict[str, PPHEntry] = {}
        if PPH_TABLE.exists():
            df = pd.read_csv(PPH_TABLE)
            for _, row in df.iterrows():
                try:
                    value = float(row["people_per_household"])
                except (KeyError, ValueError, TypeError):
                    continue
                iso3 = str(row.get("iso3", "")).strip().upper()
                if not iso3:
                    continue
                table[iso3] = PPHEntry(
                    iso3=iso3,
                    people_per_household=value,
                    source=str(row.get("source", "")),
                    year=int(row["year"]) if not pd.isna(row.get("year")) else None,
                    notes=str(row.get("notes", "")),
                )
        if PPH_OVERRIDES.exists():
            with open(PPH_OVERRIDES, "r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
            overrides = data.get("overrides", {}) or {}
            for iso3, payload in overrides.items():
                iso3_norm = str(iso3).strip().upper()
                if not iso3_norm:
                    continue
                value = payload.get("people_per_household")
                if value is None:
                    continue
                entry = table.get(iso3_norm)
                year = payload.get("year")
                table[iso3_norm] = PPHEntry(
                    iso3=iso3_norm,
                    people_per_household=float(value),
                    source=str(payload.get("source", getattr(entry, "source", ""))),
                    year=int(year) if year is not None else getattr(entry, "year", None),
                    notes=str(payload.get("notes", getattr(entry, "notes", ""))),
                )
        self._table = table
        self._loaded = True

    def lookup(self, iso3: str) -> PPHEntry:
        self._load()
        key = (iso3 or "").strip().upper()
        entry = self._table.get(key)
        if entry:
            return entry
        return PPHEntry(iso3=key or "GLOBAL", people_per_household=DEFAULT_PPH, source="default", year=None)


PPH = PPHLookup()


def _clean_text(text: str) -> str:
    return (text or "").replace("\u202f", " ").replace("\xa0", " ")


def _parse_number(token: str) -> Optional[int]:
    if token is None:
        return None
    cleaned = _clean_text(token).strip().lower()
    if not cleaned:
        return None
    multiplier = 1.0
    if cleaned.endswith("million"):
        multiplier = 1_000_000.0
        cleaned = cleaned[:-7].strip()
    elif cleaned.endswith("m"):
        multiplier = 1_000_000.0
        cleaned = cleaned[:-1].strip()
    elif cleaned.endswith("k"):
        multiplier = 1_000.0
        cleaned = cleaned[:-1].strip()
    cleaned = cleaned.replace(",", "")
    try:
        value = float(cleaned)
    except ValueError:
        return None
    normalized = int(round(value * multiplier))
    if normalized < 0:
        return None
    return normalized


def _extract_value_from_line(line: str) -> Optional[int]:
    match = re.search(r"([0-9][0-9.,\s]*(?:k|m|million)?)", line, re.I)
    if not match:
        return None
    return _parse_number(match.group(1))


def select_best_pdf(resources: List[Dict[str, Any]], preferred_titles: Iterable[str]) -> Optional[Dict[str, Any]]:
    if not resources:
        return None
    prefs = [p.lower() for p in preferred_titles]

    def score(resource: Dict[str, Any]) -> Tuple[int, dt.datetime, int]:
        name = _clean_text(
            resource.get("name")
            or resource.get("description")
            or resource.get("title")
            or ""
        ).lower()
        matches = 0
        for pref in prefs:
            if pref and pref in name:
                matches = max(matches, len(pref))
        created_text = resource.get("date", {}).get("created") or resource.get("created")
        try:
            created = dt.datetime.fromisoformat(created_text.replace("Z", "+00:00"))
        except Exception:
            created = dt.datetime.fromtimestamp(0, tz=dt.timezone.utc)
        pages = int(resource.get("page_count") or 0)
        return matches, created, pages

    ranked = sorted(resources, key=score, reverse=True)
    return ranked[0]


def _split_table_line(line: str) -> Optional[List[str]]:
    if "|" in line:
        return [cell.strip() for cell in line.split("|") if cell.strip()]
    if "\t" in line:
        return [cell.strip() for cell in line.split("\t") if cell.strip()]
    if re.search(r"\s{2,}", line):
        return [cell.strip() for cell in re.split(r"\s{2,}", line) if cell.strip()]
    return None


def _find_table_candidates(text: str, country_names: Dict[str, str]) -> List[MetricCandidate]:
    lines = [_clean_text(line) for line in text.splitlines()]
    split_lines = [(_split_table_line(line), line) for line in lines]
    header_idx = None
    header_map: Dict[str, int] = {}
    for idx, (cells, raw) in enumerate(split_lines):
        if not cells or len(cells) < 2:
            continue
        lowered = [cell.lower() for cell in cells]
        for metric, synonyms in TABLE_METRIC_SYNONYMS.items():
            for synonym in synonyms:
                if synonym in lowered:
                    header_map[metric] = lowered.index(synonym)
        if header_map:
            header_idx = idx
            break
    candidates: List[MetricCandidate] = []
    if header_idx is None:
        # Try line level detection without header (key figures style)
        for _, raw_line in split_lines:
            if not raw_line:
                continue
            low = raw_line.lower()
            for metric, synonyms in TABLE_METRIC_SYNONYMS.items():
                if any(s in low for s in synonyms):
                    value = _extract_value_from_line(raw_line)
                    if value is None:
                        continue
                    candidates.append(
                        MetricCandidate(
                            metric=metric,
                            value=value,
                            unit="persons" if not metric.startswith("household") else "households",
                            layer="infographic",
                            matched_phrase=raw_line.strip(),
                        )
                    )
        return candidates

    # Parse rows after header
    for idx in range(header_idx + 1, len(split_lines)):
        cells, raw = split_lines[idx]
        if not cells or len(cells) < 2:
            continue
        first_cell = cells[0].lower()
        iso_match = None
        for iso3, name in country_names.items():
            if name.lower() in first_cell or iso3.lower() in first_cell:
                iso_match = iso3
                break
        for metric, col_idx in header_map.items():
            if col_idx >= len(cells):
                continue
            value = _parse_number(cells[col_idx])
            if value is None:
                continue
            candidates.append(
                MetricCandidate(
                    metric=metric,
                    value=value,
                    unit="persons" if not metric.startswith("household") else "households",
                    layer="table",
                    matched_phrase=raw.strip(),
                    country_iso3=iso_match,
                )
            )
    return candidates


def _find_infographic_candidates(text: str) -> List[MetricCandidate]:
    candidates: List[MetricCandidate] = []
    lowered = _clean_text(text).lower()
    for heading in INFOGRAPHIC_HEADINGS:
        idx = lowered.find(heading)
        if idx == -1:
            continue
        window = text[idx : idx + 400]
        for metric, synonyms in TABLE_METRIC_SYNONYMS.items():
            for synonym in synonyms:
                pattern = re.compile(rf"{re.escape(synonym)}[\s:]+([^\n]+)", re.I)
                match = pattern.search(window)
                if not match:
                    continue
                value = _parse_number(match.group(1))
                if value is None:
                    continue
                candidates.append(
                    MetricCandidate(
                        metric=metric,
                        value=value,
                        unit="persons" if not metric.startswith("household") else "households",
                        layer="infographic",
                        matched_phrase=match.group(0).strip(),
                    )
                )
    return candidates


def _find_narrative_candidates(text: str) -> List[MetricCandidate]:
    candidates: List[MetricCandidate] = []
    for metric, pattern in NARRATIVE_PATTERNS.items():
        for match in pattern.finditer(text):
            value = _parse_number(match.group("value"))
            if value is None:
                continue
            candidates.append(
                MetricCandidate(
                    metric=metric,
                    value=value,
                    unit="persons",
                    layer="narrative",
                    matched_phrase=match.group(0).strip(),
                )
            )
    return candidates


def _gather_candidates(text: str, country_names: Dict[str, str]) -> List[MetricCandidate]:
    combined = []
    combined.extend(_find_table_candidates(text, country_names))
    combined.extend(_find_infographic_candidates(text))
    combined.extend(_find_narrative_candidates(text))
    return combined


def _choose_best_candidates(candidates: List[MetricCandidate]) -> Dict[Tuple[Optional[str], str], MetricCandidate]:
    best: Dict[Tuple[Optional[str], str], MetricCandidate] = {}
    for candidate in candidates:
        key = (candidate.country_iso3, candidate.metric)
        current = best.get(key)
        if current is None:
            best[key] = candidate
            continue
        if LAYER_PRIORITY[candidate.layer] < LAYER_PRIORITY.get(current.layer, 99):
            best[key] = candidate
    return best


def _apply_household_conversions(
    best: Dict[Tuple[Optional[str], str], MetricCandidate],
    iso_list: List[str],
) -> None:
    if not best:
        return
    for key, candidate in list(best.items()):
        metric = candidate.metric
        if metric not in HOUSEHOLD_TO_PEOPLE:
            continue
        target_metric = HOUSEHOLD_TO_PEOPLE[metric]
        iso = candidate.country_iso3
        if (iso, target_metric) in best:
            continue
        if iso is None:
            if len(iso_list) != 1:
                continue
            iso = iso_list[0]
        entry = PPH.lookup(iso)
        derived_value = int(round(candidate.value * entry.people_per_household))
        best[(iso, target_metric)] = MetricCandidate(
            metric=target_metric,
            value=derived_value,
            unit="persons",
            layer=candidate.layer,
            matched_phrase=candidate.matched_phrase,
            method_value="derived_from_households",
            method_details=entry.describe(),
            country_iso3=iso,
            extraction_meta={"pph": entry.people_per_household},
        )


DATE_FORMATS = [
    "%d %b %Y",
    "%d %B %Y",
    "%d %b %y",
    "%d %B %y",
    "%Y-%m-%d",
    "%Y/%m/%d",
]


def _parse_date_token(token: str) -> Optional[dt.date]:
    if not token:
        return None
    cleaned = _clean_text(token).replace("–", "-").replace("/", "/").strip()
    cleaned = re.sub(r"(\d)(st|nd|rd|th)", r"\1", cleaned)
    for fmt in DATE_FORMATS:
        try:
            return dt.datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue
    try:
        return dt.datetime.fromisoformat(cleaned.replace("Z", "")).date()
    except Exception:
        return None


def _pick_pdf_dates(text: str, fallback_created: str, fallback_changed: str) -> Tuple[str, str]:
    coverage_line = re.search(
        r"(?:reporting|coverage)\s+period:?([^\n]+)", text, flags=re.IGNORECASE
    )
    if coverage_line:
        parts = re.split(r"[–-]", coverage_line.group(1))
        end_token = parts[-1] if parts else ""
        end = _parse_date_token(end_token)
        if end:
            return end.isoformat(), end.isoformat()
    report_match = re.search(r"report\s+date[:\s]+([0-9a-zA-Z\s,/-]+)", text, flags=re.IGNORECASE)
    if report_match:
        parsed = _parse_date_token(report_match.group(1))
        if parsed:
            return parsed.isoformat(), parsed.isoformat()
    fallback = fallback_created or fallback_changed
    if fallback:
        return fallback.split("T")[0], (fallback_changed or fallback).split("T")[0]
    today = dt.datetime.now(dt.timezone.utc).date()
    return today.isoformat(), today.isoformat()


LEVEL_CACHE_FILE = LEVEL_CACHE / "levels.json"
_LEVEL_STATE: Optional[Dict[str, Dict[str, Any]]] = None


def _load_level_state() -> Dict[str, Dict[str, Any]]:
    global _LEVEL_STATE
    if _LEVEL_STATE is not None:
        return _LEVEL_STATE
    if LEVEL_CACHE_FILE.exists():
        try:
            with LEVEL_CACHE_FILE.open("r", encoding="utf-8") as handle:
                _LEVEL_STATE = json.load(handle)
        except Exception:
            _LEVEL_STATE = {}
    else:
        _LEVEL_STATE = {}
    return _LEVEL_STATE


def load_last_level_for_lineage(lineage: str) -> Optional[int]:
    state = _load_level_state()
    payload = state.get(lineage)
    if not payload:
        return None
    return int(payload.get("value", 0))


def store_level_for_lineage(lineage: str, as_of: str, value: int) -> None:
    state = _load_level_state()
    state[lineage] = {"as_of": as_of, "value": int(value)}
    LEVEL_CACHE.mkdir(parents=True, exist_ok=True)
    with LEVEL_CACHE_FILE.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, sort_keys=True)


def compute_monthly_delta(lineage: str, level_value: int, as_of: str) -> Tuple[int, Optional[int], bool]:
    previous = load_last_level_for_lineage(lineage)
    if previous is None:
        delta = level_value
        first = True
    else:
        delta = max(level_value - previous, 0)
        first = False
    store_level_for_lineage(lineage, as_of, level_value)
    return delta, previous, first


def _canonical_metric(metric: str) -> str:
    mapping = {
        "people_in_need": "in_need",
        "people_affected": "affected",
        "idps": "displaced",
    }
    return mapping.get(metric, metric)


def _download_pdf(session: requests.Session, url: str, timeout: float = 30.0) -> bytes:
    PDF_CACHE.mkdir(parents=True, exist_ok=True)
    digest = stable_digest([url], length=32, algorithm="sha256")
    path = PDF_CACHE / f"{digest}.pdf"
    if path.exists():
        return path.read_bytes()
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    content = response.content
    path.write_bytes(content)
    return content


def build_pdf_rows_for_item(
    cfg: Dict[str, Any],
    session: requests.Session,
    fields: Dict[str, Any],
    hazard_code: str,
    hazard_label: str,
    hazard_class: str,
    iso_pairs: List[Tuple[str, str]],
    source_name: str,
    source_type: str,
    report_title: str,
    source_url: str,
) -> List[List[str]]:
    if not cfg.get("enable_pdfs", True):
        return []
    raw_resources = fields.get("file") or fields.get("files") or []
    pdf_resources = []
    for res in raw_resources:
        mime = (res.get("mime_type") or res.get("mimetype") or res.get("content_type") or "").lower()
        if "pdf" in mime:
            pdf_resources.append(res)
    if not pdf_resources:
        return []
    preferred = cfg.get("preferred_pdf_titles", [])
    best = select_best_pdf(pdf_resources, preferred)
    if not best:
        return []
    pdf_url = best.get("url") or best.get("href")
    if not pdf_url:
        return []
    timeout = float(cfg.get("timeout_seconds", 30))
    content = _download_pdf(session, pdf_url, timeout=timeout)
    min_chars = int(cfg.get("min_text_chars_before_ocr", 1500))
    previous_mode = pdf_text_mod.PDF_TEXT_TEST_MODE
    if not cfg.get("enable_ocr", True):
        pdf_text_mod.PDF_TEXT_TEST_MODE = True
    try:
        text, text_meta = smart_extract(content, min_chars)
    finally:
        pdf_text_mod.PDF_TEXT_TEST_MODE = previous_mode
    if not text.strip():
        return []

    country_map = {iso: name for name, iso in iso_pairs}
    candidates = _gather_candidates(text, country_map)
    if not candidates:
        return []
    best_map = _choose_best_candidates(candidates)
    _apply_household_conversions(best_map, [iso for _, iso in iso_pairs])

    created = fields.get("date", {}).get("created") or fields.get("date.created") or ""
    changed = fields.get("date", {}).get("changed") or fields.get("date.changed") or ""
    as_of, publication = _pick_pdf_dates(text, created or "", changed or "")
    month = month_start(as_of)
    month_str = month.isoformat() if month else ""

    rows: List[List[str]] = []
    pdf_rows: List[List[str]] = []
    ingested_at = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for (iso_override, metric_key), candidate in best_map.items():
        iso3 = iso_override or (iso_pairs[0][1] if len(iso_pairs) == 1 else None)
        if iso3 is None:
            # Ambiguous allocation in multi-country report.
            continue
        country_name = next((name for name, iso in iso_pairs if iso == iso3), "")
        canonical_metric = _canonical_metric(metric_key)
        if canonical_metric not in {"in_need", "affected", "displaced"}:
            continue
        lineage = "|".join([iso3, hazard_code, canonical_metric, "reliefweb_pdf"])
        delta, previous_level, first_observation = compute_monthly_delta(lineage, candidate.value, as_of)
        method_details = candidate.method_details or f"extracted_from_{candidate.layer}"
        method_details = f"{method_details}; text_extraction={text_meta.get('method', 'native')}"
        if first_observation:
            method_details = f"{method_details}; delta_from_level(first_observation)"
        definition_text = (
            f"ReliefWeb PDF extraction ({candidate.layer}) for {canonical_metric}"
        )
        event_id = stable_digest(
            [iso3, hazard_code, canonical_metric, as_of, candidate.value, pdf_url],
            length=16,
            algorithm="sha256",
        )
        rows.append(
            [
                event_id,
                country_name,
                iso3,
                hazard_code,
                hazard_label,
                hazard_class,
                canonical_metric,
                "new",
                str(delta),
                "persons",
                str(candidate.value),
                as_of,
                month_str,
                publication,
                source_name,
                source_type,
                source_url,
                pdf_url,
                report_title,
                definition_text,
                "pdf",
                candidate.method_value,
                method_details,
                candidate.layer,
                candidate.matched_phrase,
                "2",
                "med",
                1,
                ingested_at,
            ]
        )
    return rows

NUM_RE = re.compile(
    r"\b(?:about|approx\.?|around)?\s*([0-9][0-9., ]{0,15})(?:\s*(?:people|persons|individuals))?\b",
    re.I,
)

# Future fallback consideration: https://reliefweb.int/updates/rss


def load_cfg() -> Dict[str, Any]:
    with open(CONFIG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_registries() -> Tuple[pd.DataFrame, pd.DataFrame]:
    countries = pd.read_csv(COUNTRIES, dtype=str).fillna("")
    shocks = pd.read_csv(SHOCKS, dtype=str).fillna("")
    countries["country_norm"] = countries["country_name"].str.strip().str.lower()
    return countries, shocks


def norm(text: str) -> str:
    return (text or "").strip().lower()


def detect_hazard(text: str, cfg: Dict[str, Any]) -> Optional[str]:
    sample = norm(text)
    for code, keywords in cfg["hazard_keywords"].items():
        for keyword in keywords:
            if keyword in sample:
                return code
    return None


def extract_metric_value(text: str, cfg: Dict[str, Any]) -> Optional[Tuple[str, int, str, str]]:
    """Return (metric, value, unit, matched_phrase)."""

    combined = text or ""
    lowered = combined.lower()

    for pattern_cfg in cfg["metric_patterns"]:
        metric = pattern_cfg["metric"]
        unit = pattern_cfg["unit"]
        for phrase in pattern_cfg["patterns"]:
            idx = lowered.find(phrase)
            if idx == -1:
                continue
            window_start = max(0, idx - 80)
            window_end = min(len(combined), idx + len(phrase) + 80)
            window = combined[window_start:window_end]
            match = NUM_RE.search(window)
            if not match:
                continue
            raw_value = match.group(1)
            cleaned = raw_value.replace(",", "").replace(" ", "")
            cleaned = re.sub(r"[^0-9]", "", cleaned)
            try:
                value = int(cleaned)
            except ValueError:
                continue
            if value < 0:
                continue
            return metric, value, unit, phrase
    return None


def iso3_from_reliefweb_countries(
    countries_df: pd.DataFrame, rw_countries: List[Dict[str, Any]]
) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    for country in rw_countries or []:
        name = country.get("name") or country.get("shortname") or ""
        if not name:
            continue
        match = countries_df[countries_df["country_name"].str.lower() == name.lower()]
        if match.empty:
            continue
        rows.append((match.iloc[0]["country_name"], match.iloc[0]["iso3"]))
    return rows


def _dump(resp: requests.Response) -> str:
    try:
        body = resp.text[:500]
    except Exception:  # pragma: no cover - defensive
        body = "<no-body>"
    try:
        hdrs = dict(resp.headers)
    except Exception:  # pragma: no cover - defensive
        hdrs = {}
    return f"HTTP {resp.status_code}, headers={hdrs}, body[0:500]={body}"


def _is_waf_challenge(resp: requests.Response) -> bool:
    return (
        resp.status_code == 202
        and resp.headers.get("x-amzn-waf-action", "").lower() == "challenge"
    )


def rw_request(
    session: requests.Session,
    url: str,
    payload: Dict[str, Any],
    since: str,
    max_retries: int,
    retry_backoff: float,
    timeout: float,
    challenge_tracker: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], str]:
    last_err: Optional[str] = None

    # 0) Connectivity probe (optional; ignore body)
    for attempt in range(1, max_retries + 1):
        try:
            probe = session.get(url, params={"limit": 1}, timeout=timeout)
            if DEBUG:
                print(f"[reliefweb] GET probe status={probe.status_code}")
            if _is_waf_challenge(probe):
                challenge_tracker["count"] += 1
                if DEBUG:
                    print("[reliefweb] GET WAF challenge:", _dump(probe))
                if attempt >= max_retries:
                    challenge_tracker["persisted"] = True
                    return None, "empty"
                time.sleep(retry_backoff * attempt + random.uniform(0, 0.5))
                continue
            if probe.status_code in (429, 502, 503):
                if DEBUG:
                    print(
                        f"[reliefweb] GET probe rate limit status={probe.status_code}"
                    )
                last_err = _dump(probe)
                if attempt >= max_retries:
                    break
                time.sleep((1.5 ** attempt) + random.uniform(0, 0.5))
                continue
            if probe.status_code == 200:
                try:
                    probe.json()
                except ValueError:
                    if DEBUG:
                        print("[reliefweb] GET probe invalid JSON")
                break
            break
        except Exception as exc:
            if DEBUG:
                print("[reliefweb] GET probe exception:", str(exc))
            break

    # 1) Real request with filters via POST
    for attempt in range(1, max_retries + 1):
        try:
            response = session.post(url, json=payload, timeout=timeout)
            if response.status_code == 200:
                return response.json(), "post"
            if response.status_code in (429, 502, 503):
                if DEBUG:
                    print(
                        f"[reliefweb] POST attempt {attempt} backoff; status={response.status_code}"
                    )
                time.sleep((1.5 ** attempt) + random.uniform(0, 0.5))
                continue
            if _is_waf_challenge(response):
                challenge_tracker["count"] += 1
                if DEBUG:
                    print("[reliefweb] POST WAF challenge:", _dump(response))
                if attempt >= max_retries:
                    challenge_tracker["persisted"] = True
                    return None, "empty"
                time.sleep(retry_backoff * attempt + random.uniform(0, 0.5))
                continue
            last_err = _dump(response)
            break
        except Exception as exc:  # pragma: no cover - network failure paths
            last_err = str(exc)
            if DEBUG:
                print(f"[reliefweb] POST exception attempt {attempt}: {last_err}")
            if attempt >= max_retries:
                break
            time.sleep((1.5 ** attempt) + random.uniform(0, 0.5))

    if challenge_tracker.get("persisted"):
        return None, "empty"

    # 2) GET fallback (single flow with retry loop)
    offset = int(payload.get("offset", 0))
    get_params: List[Tuple[str, str]] = []
    for field in payload.get("fields", {}).get("include", []):
        get_params.append(("fields[include][]", field))

    # Date filter
    get_params.append(("filter[conditions][0][field]", "date.created"))
    get_params.append(("filter[conditions][0][value][from]", since))
    # Language filter
    get_params.append(("filter[conditions][1][field]", "language"))
    get_params.append(("filter[conditions][1][value]", "en"))
    # Format filter(s)
    formats = payload.get("filter", {}).get("conditions", [])
    format_values: List[str] = []
    if len(formats) >= 3:
        format_entry = formats[2]
        value = format_entry.get("value", []) if isinstance(format_entry, dict) else []
        if isinstance(value, list):
            format_values = [str(v) for v in value]
    get_params.append(("filter[conditions][2][field]", "format"))
    for fmt in format_values:
        get_params.append(("filter[conditions][2][value][]", fmt))

    get_params.append(("sort[]", "date.created:desc"))
    get_params.append(("limit", str(payload.get("limit", 100))))
    get_params.append(("offset", str(offset)))

    for attempt in range(1, max_retries + 1):
        try:
            response = session.get(url, params=get_params, timeout=timeout)
        except Exception as exc:  # pragma: no cover - defensive network handling
            last_err = str(exc)
            if attempt >= max_retries:
                break
            time.sleep((1.5 ** attempt) + random.uniform(0, 0.5))
            continue

        if response.status_code == 200:
            try:
                return response.json(), "get"
            except ValueError as exc:  # pragma: no cover - malformed payload
                last_err = str(exc)
                break
        if _is_waf_challenge(response):
            challenge_tracker["count"] += 1
            if DEBUG:
                print("[reliefweb] GET fallback WAF challenge:", _dump(response))
            if attempt >= max_retries:
                challenge_tracker["persisted"] = True
                return None, "empty"
            time.sleep(retry_backoff * attempt + random.uniform(0, 0.5))
            continue
        if response.status_code in (429, 502, 503):
            if DEBUG:
                print(
                    f"[reliefweb] GET fallback rate limit attempt {attempt}; status={response.status_code}"
                )
            if attempt >= max_retries:
                last_err = _dump(response)
                break
            time.sleep((1.5 ** attempt) + random.uniform(0, 0.5))
            continue
        last_err = _dump(response)
        break

    if challenge_tracker.get("persisted"):
        return None, "empty"

    raise RuntimeError(f"ReliefWeb API error: {last_err or 'no 200 after retries'}")


def build_payload(cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    since = (
        dt.datetime.now(dt.UTC) - dt.timedelta(days=int(cfg["window_days"]))
    ).strftime("%Y-%m-%dT00:00:00Z")
    fields = [
        "id",
        "title",
        "body",
        "date.created",
        "date.original",
        "date.changed",
        "url",
        "source",
        "format",
        "type",
        "disaster_type",
        "country",
    ]
    payload = {
        "appname": cfg["appname"],
        "filter": {
            "operator": "AND",
            "conditions": [
                {"field": "date.created", "value": {"from": since}},
                {"field": "language", "value": "en"},
                {"field": "format", "value": ["Report", "Appeal", "Update"]},
            ],
        },
        "fields": {"include": fields},
        "sort": ["date.created:desc"],
        "limit": int(cfg["page_size"]),
    }
    return payload, since


def map_source_type(rw_type: str, cfg: Dict[str, Any]) -> str:
    return cfg["source_type_map"].get(str(rw_type).lower(), "sitrep")


def pick_dates(rec: Dict[str, Any]) -> Tuple[str, str]:
    dates = rec.get("date", {}) or {}
    created = (dates.get("created") or "").split("T")[0]
    original = (dates.get("original") or "").split("T")[0]
    as_of = original or created or ""
    publication = created or as_of
    return as_of, publication


def make_rows() -> Tuple[List[List[str]], Dict[str, Any]]:
    if os.getenv("RESOLVER_SKIP_RELIEFWEB", "") == "1":
        return [], {"count": 0, "persisted": False, "mode": "empty"}

    cfg = load_cfg()
    countries, shocks = load_registries()
    iso_exclude = {code.upper() for code in cfg.get("iso3_exclude", [])}

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": cfg.get(
                "user_agent",
                "spagbot-resolver/1.0 (+https://github.com/kwyjad/Spagbot_metac-bot)",
            ),
            "Content-Type": "application/json",
            "Accept": cfg.get("accept_header", "application/json"),
        }
    )
    adapter = HTTPAdapter(
        max_retries=Retry(total=0, connect=4, read=4, backoff_factor=0)
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    base_url = cfg["base_url"]
    appname_cfg = cfg.get("appname", "spagbot-resolver")
    appname = os.getenv("RELIEFWEB_APPNAME", appname_cfg)
    url = f"{base_url}?appname={appname}"
    payload, since = build_payload(cfg)
    timeout = float(cfg.get("timeout_seconds", 30))
    max_retries = int(cfg.get("max_retries", 6))
    retry_backoff = float(cfg.get("retry_backoff_seconds", 2))
    challenge_tracker: Dict[str, Any] = {"count": 0, "persisted": False}
    page_pause = float(cfg.get("min_page_pause_seconds", 0.6))

    rows: List[List[str]] = []
    offset = 0
    total = None
    mode_used = "post"

    while True:
        payload["offset"] = offset
        data, mode = rw_request(
            session,
            url,
            payload,
            since,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
            timeout=timeout,
            challenge_tracker=challenge_tracker,
        )
        if data is None:
            challenge_tracker["mode"] = mode
            return rows, pdf_rows, challenge_tracker
        if mode == "get":
            mode_used = "get"
        total = total or data.get("totalCount", 0)
        items = data.get("data", [])
        if not items:
            break

        for item in items:
            report_id = str(item.get("id"))
            fields = item.get("fields", {}) or {}
            title = fields.get("title", "")
            body = fields.get("body", "")
            report_type_entries = fields.get("type") or [{}]
            report_type = report_type_entries[0].get("name", "report")
            sources = fields.get("source") or []
            source_name = sources[0].get("shortname") if sources else "OCHA"

            iso_pairs = iso3_from_reliefweb_countries(
                countries, fields.get("country") or []
            )
            if not iso_pairs:
                continue

            hazard_code = detect_hazard(f"{title} {body}", cfg)
            if not hazard_code:
                continue

            shock_row = shocks[shocks["hazard_code"] == hazard_code]
            if shock_row.empty:
                continue
            hazard_label = shock_row.iloc[0]["hazard_label"]
            hazard_class = shock_row.iloc[0]["hazard_class"]

            text_for_metrics = " ".join([title or "", body or ""])
            metric_info = extract_metric_value(text_for_metrics, cfg)
            if not metric_info:
                continue
            metric, value, unit, phrase = metric_info

            as_of, publication = pick_dates(fields)
            source_url = fields.get("url", "")
            doc_title = title

            ingested_at = dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

            for country_name, iso3 in iso_pairs:
                if iso3 in iso_exclude:
                    continue
                event_id = f"{iso3}-{hazard_code}-rw-{report_id}"
                rows.append(
                    [
                        event_id,
                        country_name,
                        iso3,
                        hazard_code,
                        hazard_label,
                        hazard_class,
                        metric,
                        "stock",
                        str(value),
                        unit,
                        as_of,
                        publication,
                        source_name or "OCHA",
                        map_source_type(report_type, cfg),
                        source_url,
                        doc_title,
                        f"Extracted {metric} via phrase '{phrase}' in ReliefWeb report.",
                        "api",
                        "med",
                        1,
                        ingested_at,
                    ]
                )

            pdf_rows.extend(
                build_pdf_rows_for_item(
                    cfg,
                    session,
                    fields,
                    hazard_code,
                    hazard_label,
                    hazard_class,
                    iso_pairs,
                    source_name or "OCHA",
                    map_source_type(report_type, cfg),
                    doc_title,
                    source_url,
                )
            )

        offset += len(items)
        if offset >= total:
            break
        time.sleep(page_pause)

    challenge_tracker["mode"] = mode_used

    return rows, pdf_rows, challenge_tracker


def main() -> None:
    if os.getenv("RESOLVER_SKIP_RELIEFWEB", "0") == "1":
        print("ReliefWeb connector skipped due to RESOLVER_SKIP_RELIEFWEB=1")
        STAGING.mkdir(parents=True, exist_ok=True)
        output = STAGING / "reliefweb.csv"
        ensure_headers(output, COLUMNS)
        ensure_headers(PDF_OUTPUT, PDF_COLUMNS)
        ensure_manifest_for_csv(output, source_id="reliefweb_api")
        ensure_manifest_for_csv(PDF_OUTPUT, source_id="reliefweb_pdf")
        print("[reliefweb] rows=0 challenged=0 mode=empty pdf_rows=0")
        return

    STAGING.mkdir(parents=True, exist_ok=True)
    output = STAGING / "reliefweb.csv"
    try:
        rows, pdf_rows, challenge_tracker = make_rows()
    except RuntimeError as exc:
        message = str(exc)
        if "WAF_CHALLENGE" in message:
            print(
                "ReliefWeb blocked by AWS WAF challenge (202 + x-amzn-waf-action=challenge). "
                "Writing empty CSV and continuing."
            )
            ensure_headers(output, COLUMNS)
            ensure_headers(PDF_OUTPUT, PDF_COLUMNS)
            ensure_manifest_for_csv(output, source_id="reliefweb_api")
            ensure_manifest_for_csv(PDF_OUTPUT, source_id="reliefweb_pdf")
            print("[reliefweb] rows=0 challenged=0 mode=empty pdf_rows=0")
            return
        raise
    challenged = int(challenge_tracker.get("count", 0))
    mode = challenge_tracker.get("mode", "post")
    if challenge_tracker.get("persisted"):
        print("ReliefWeb WAF challenge persisted; writing empty CSV this run")
        ensure_headers(output, COLUMNS)
        ensure_headers(PDF_OUTPUT, PDF_COLUMNS)
        ensure_manifest_for_csv(output, source_id="reliefweb_api")
        ensure_manifest_for_csv(PDF_OUTPUT, source_id="reliefweb_pdf")
        print(f"[reliefweb] rows=0 challenged={challenged} mode=empty pdf_rows=0")
        return

    if not rows:
        ensure_headers(output, COLUMNS)
        print(f"[reliefweb] rows=0 challenged={challenged} mode={mode}")
    else:
        df = pd.DataFrame(rows, columns=COLUMNS)
        df.to_csv(output, index=False)
        print(f"[reliefweb] rows={len(df)} challenged={challenged} mode={mode}")

    if not pdf_rows:
        ensure_headers(PDF_OUTPUT, PDF_COLUMNS)
        pdf_count = 0
    else:
        pdf_df = pd.DataFrame(pdf_rows, columns=PDF_COLUMNS)
        pdf_df.to_csv(PDF_OUTPUT, index=False)
        pdf_count = len(pdf_df)

    ensure_manifest_for_csv(output, source_id="reliefweb_api")
    ensure_manifest_for_csv(PDF_OUTPUT, source_id="reliefweb_pdf")
    print(f"[reliefweb_pdf] rows={pdf_count}")


if __name__ == "__main__":
    main()
