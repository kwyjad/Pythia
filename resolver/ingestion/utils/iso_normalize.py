"""Helpers for normalising country identifiers to ISO3 codes."""

from __future__ import annotations

import csv
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
COUNTRY_CSV = ROOT / "data" / "countries.csv"

DEFAULT_ALIAS_SOURCE: Mapping[str, str] = {
    "Democratic Republic of the Congo": "COD",
    "DR Congo": "COD",
    "DRC": "COD",
    "Congo, Democratic Republic of": "COD",
    "Congo, The Democratic Republic of": "COD",
    "Côte d'Ivoire": "CIV",
    "Cote d'Ivoire": "CIV",
    "Ivory Coast": "CIV",
    "State of Palestine": "PSE",
    "Palestine, State of": "PSE",
    "Syrian Arab Republic": "SYR",
    "Syria": "SYR",
    "Iran (Islamic Republic of)": "IRN",
    "Iran, Islamic Republic of": "IRN",
    "Venezuela (Bolivarian Republic of)": "VEN",
    "Venezuela, Bolivarian Republic of": "VEN",
    "United Republic of Tanzania": "TZA",
    "Tanzania, United Republic of": "TZA",
    "Lao People's Democratic Republic": "LAO",
    "Lao Peoples Democratic Republic": "LAO",
    "Laos": "LAO",
}


def _normalise_token(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    lowered = decomposed.lower()
    return "".join(ch for ch in lowered if ch.isalnum())


@lru_cache(maxsize=1)
def _load_country_lookup() -> tuple[Mapping[str, str], Mapping[str, str]]:
    iso_to_name: dict[str, str] = {}
    token_to_iso: dict[str, str] = {}
    if not COUNTRY_CSV.exists():
        return iso_to_name, token_to_iso
    with COUNTRY_CSV.open("r", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            iso = (row.get("iso3") or "").strip().upper()
            name = (row.get("country_name") or "").strip()
            if not iso:
                continue
            iso_to_name[iso] = name
            if name:
                token = _normalise_token(name)
                if token:
                    token_to_iso.setdefault(token, iso)
    return iso_to_name, token_to_iso


def _normalised_aliases(aliases: Mapping[str, str] | None) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not aliases:
        return mapping
    for raw_key, raw_value in aliases.items():
        key = _normalise_token(str(raw_key))
        if not key:
            continue
        iso = str(raw_value or "").strip().upper()
        if len(iso) == 3:
            mapping[key] = iso
    return mapping


@lru_cache(maxsize=1)
def _default_aliases() -> Mapping[str, str]:
    return _normalised_aliases(DEFAULT_ALIAS_SOURCE)


def _build_alias_map(aliases: Mapping[str, str] | None) -> dict[str, str]:
    mapping = dict(_default_aliases())
    if aliases:
        mapping.update(_normalised_aliases(aliases))
    return mapping


def to_iso3(name: str | None, aliases: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """Normalise ``name`` to an ISO3 code if possible."""

    if not name:
        return None
    text = str(name).strip()
    if not text:
        return None
    iso_to_name, token_lookup = _load_country_lookup()
    candidate = text.upper()
    if len(candidate) == 3 and candidate.isalpha():
        if candidate in iso_to_name:
            return candidate
    alias_map = _build_alias_map(aliases)
    if alias_map:
        token = _normalise_token(text)
        alias_iso = alias_map.get(token)
        if alias_iso:
            return alias_iso
    token = _normalise_token(text)
    if token and token in token_lookup:
        return token_lookup[token]
    for iso, country in iso_to_name.items():
        if text.lower() == country.lower():
            return iso
    return None


def resolve_iso3(
    fields: Mapping[str, Any] | None,
    aliases: Optional[Mapping[str, str]] = None,
    *,
    name_keys: Sequence[str] | None = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Resolve ISO3 value by preferring explicit ISO fields over names.

    Parameters
    ----------
    fields:
        Mapping of raw values (such as a Pandas Series representing a row).
    aliases:
        Optional alias overrides supplied by the caller.
    name_keys:
        Optional ordered list of field names to try when falling back to
        country labels. The provided order is preserved.

    Returns
    -------
    tuple
        A ``(iso3, reason)`` tuple where *iso3* is the resolved code (or
        ``None``) and *reason* carries a short label describing why no ISO was
        found.
    """

    if fields is None:
        return None, "no_iso3"

    iso_fields = ("admin0Pcode", "CountryPcode", "CountryISO3", "ISO3", "iso3")
    bad_iso_detected = False

    for key in iso_fields:
        if key not in fields:
            continue
        raw = fields.get(key)
        if raw is None:
            continue
        text = str(raw).strip()
        if not text:
            continue
        iso_candidate = to_iso3(text, aliases)
        if iso_candidate:
            iso_upper = iso_candidate.strip().upper()
            if text.strip().upper() == iso_upper:
                return iso_upper, None
            return iso_upper, "bad_iso"
        bad_iso_detected = True

    ordered_name_keys = list(name_keys or [])
    for fallback_key in ("CountryName", "Country", "country", "admin0Name"):
        if fallback_key not in ordered_name_keys:
            ordered_name_keys.append(fallback_key)

    for key in ordered_name_keys:
        if key not in fields:
            continue
        value = fields.get(key)
        if value is None:
            continue
        iso_candidate = to_iso3(value, aliases)
        if iso_candidate:
            iso_upper = iso_candidate.strip().upper()
            return iso_upper, None

    if bad_iso_detected:
        return None, "bad_iso"
    return None, "no_iso3"


__all__ = ["to_iso3", "resolve_iso3"]
