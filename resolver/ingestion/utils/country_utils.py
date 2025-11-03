"""Helpers for working with ISO3 country codes."""
from __future__ import annotations

import csv
import logging
import os
import re
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

LOGGER = logging.getLogger(__name__)

_MASTER_PATH = Path("resolver/ingestion/static/iso3_master.csv")
_TOKEN_SPLIT = re.compile(r"[,\s]+")


def _ensure_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        return (Path.cwd() / candidate).resolve()
    return candidate.resolve()


def normalise_token(raw: object) -> str:
    """Normalise free-form country input to an upper-case comparison token."""

    if raw is None:
        return ""
    text = unicodedata.normalize("NFKC", str(raw))
    text = text.replace("’", "'").replace("`", "'").replace("´", "'")
    text = text.strip()
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.upper()


def _load_master_data(path: str | Path = _MASTER_PATH) -> Tuple[List[str], Dict[str, str]]:
    resolved = _ensure_path(path)
    codes: List[str] = []
    name_map: Dict[str, str] = {}
    try:
        with resolved.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames and {"admin0Pcode", "admin0Name"}.issubset(reader.fieldnames):
                for row in reader:
                    iso_code = normalise_token(row.get("admin0Pcode"))
                    if not iso_code:
                        continue
                    if iso_code not in codes:
                        codes.append(iso_code)
                    name_token = normalise_token(row.get("admin0Name"))
                    if name_token and name_token not in name_map:
                        name_map[name_token] = iso_code
            else:
                handle.seek(0)
                simple_reader = csv.reader(handle)
                for row in simple_reader:
                    if not row:
                        continue
                    iso_code = normalise_token(row[0])
                    if not iso_code:
                        continue
                    if iso_code not in codes:
                        codes.append(iso_code)
    except FileNotFoundError:
        LOGGER.warning("ISO3 master list missing at %s", resolved)
    except OSError as exc:  # pragma: no cover - defensive logging
        LOGGER.error("Failed to read ISO3 master list at %s: %s", resolved, exc)
    return codes, name_map


def load_all_iso3(path: str | Path = _MASTER_PATH) -> List[str]:
    """Return the ordered list of ISO3 codes from ``path``."""

    codes, _ = _load_master_data(path)
    return codes


def _dedupe_normalised(tokens: Iterable[object]) -> List[str]:
    deduped: List[str] = []
    seen: set[str] = set()
    for token in tokens:
        normalised = normalise_token(token)
        if normalised and normalised not in seen:
            deduped.append(normalised)
            seen.add(normalised)
    return deduped


def _resolve_tokens(
    tokens: Iterable[object],
    master_codes: set[str],
    name_map: Dict[str, str],
) -> Tuple[List[str], List[str]]:
    resolved: List[str] = []
    unknown: List[str] = []
    seen: set[str] = set()
    for token in tokens:
        normalised = normalise_token(token)
        if not normalised:
            continue
        iso = normalised if normalised in master_codes else name_map.get(normalised)
        if iso and iso not in seen:
            resolved.append(iso)
            seen.add(iso)
        elif not iso:
            unknown.append(normalised)
    return resolved, unknown


def parse_countries_env(
    env_value: Optional[str],
    master_codes: set[str],
    name_map: Dict[str, str],
) -> List[str]:
    """Parse an environment override string into ISO3 codes."""

    if not env_value:
        return []
    tokens = [part for part in _TOKEN_SPLIT.split(env_value) if part and part.strip()]
    if not tokens:
        return []
    resolved, unknown = _resolve_tokens(tokens, master_codes, name_map)
    if unknown:
        LOGGER.warning(
            "Ignoring %d unrecognised country tokens from environment override: %s",
            len(unknown),
            ", ".join(unknown[:10]),
        )
    if resolved:
        LOGGER.debug(
            "Resolved %d countries from environment override (sample=%s)",
            len(resolved),
            ", ".join(resolved[:10]),
        )
    else:
        if not master_codes and tokens:
            fallback = _dedupe_normalised(tokens)
            if fallback:
                LOGGER.warning(
                    "Environment override retained %d codes because ISO3 master list was empty",
                    len(fallback),
                )
                return fallback
        LOGGER.warning("Environment override did not match any ISO3 codes")
    return resolved


def resolve_countries(
    cfg_list: Optional[Sequence[object]],
    env_value: Optional[str] = None,
    *,
    path: str | Path = _MASTER_PATH,
) -> List[str]:
    """Return validated ISO3 codes, allowing env overrides and config fallbacks."""

    master_list, name_map = _load_master_data(path)
    master_codes = set(master_list)

    env_value = env_value.strip() if env_value else None
    if env_value:
        resolved_env = parse_countries_env(env_value, master_codes, name_map)
        if resolved_env:
            return resolved_env
        LOGGER.warning("Falling back to config after empty environment override")

    requested = list(cfg_list or [])
    if not requested:
        LOGGER.debug(
            "No countries configured; falling back to ISO3 master list (%d entries)",
            len(master_list),
        )
        return master_list

    if not master_list:
        fallback = _dedupe_normalised(requested)
        LOGGER.warning(
            "ISO3 master list empty; using configured countries (%d entries) without validation",
            len(fallback),
        )
        return fallback

    resolved_cfg, unknown = _resolve_tokens(requested, master_codes, name_map)
    if unknown:
        LOGGER.warning(
            "Ignoring %d unrecognised country tokens from config: %s",
            len(unknown),
            ", ".join(unknown[:10]),
        )
    if resolved_cfg:
        LOGGER.debug(
            "Resolved countries: requested=%d valid=%d sample=%s",
            len(requested),
            len(resolved_cfg),
            ", ".join(resolved_cfg[:10]),
        )
        return resolved_cfg

    LOGGER.warning(
        "Configured countries resolved to zero valid codes; using ISO3 master list (%d entries)",
        len(master_list),
    )
    return master_list


def read_countries_override_from_env(
    env_var: str = "IDMC_COUNTRIES",
    file_var: str = "IDMC_COUNTRIES_FILE",
) -> Optional[str]:
    """Return the raw countries override string from environment or file."""

    raw_value = os.getenv(env_var)
    file_path = os.getenv(file_var)
    if file_path:
        try:
            file_text = Path(file_path).read_text(encoding="utf-8")
        except OSError as exc:  # pragma: no cover - defensive logging
            LOGGER.warning("Failed to read %s=%s: %s", file_var, file_path, exc)
        else:
            if file_text.strip():
                raw_value = file_text
    if raw_value is None:
        return None
    trimmed = raw_value.strip()
    return trimmed or None
