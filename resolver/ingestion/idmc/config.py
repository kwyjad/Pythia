# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Configuration helpers for the IDMC connector skeleton."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from resolver.ingestion._shared.config_loader import (
    get_config_details,
    load_connector_config,
)

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "idmc.yml"


@dataclass
class DateWindow:
    """Optional date filter bounds."""

    start: Optional[str] = None  # "YYYY-MM-DD"
    end: Optional[str] = None


def _default_cache_dir() -> str:
    return os.getenv("IDMC_CACHE_DIR", ".cache/idmc")


def _default_cache_ttl() -> int:
    raw = os.getenv("IDMC_CACHE_TTL_S", "86400")
    try:
        return int(raw)
    except ValueError:  # pragma: no cover - defensive
        return 86400


def _default_force_cache_only() -> bool:
    raw = os.getenv("IDMC_FORCE_CACHE_ONLY", "0").strip().lower()
    return raw not in {"", "0", "false", "no"}


def _csv_env(name: str, *, transform) -> Optional[List[str]]:
    raw = os.getenv(name)
    if raw is None:
        return None
    parts = []
    for piece in raw.split(","):
        text = piece.strip()
        if not text:
            continue
        parts.append(transform(text))
    return parts


def _default_base_url() -> str:
    return os.getenv("IDMC_BASE_URL", "https://backend.idmcdb.org")


def _default_alt_base_url() -> Optional[str]:
    value = os.getenv("IDMC_BASE_URL_ALT")
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def _default_endpoints() -> Dict[str, str]:
    return {
        "idus_json": "/data/idus_view_flat",
        "idus_geo": "/api/idus-view-flat-geojson",
        "monthly_flow": "",
        "stock": "",
    }


@dataclass
class CacheCfg:
    """Caching controls for the IDU connector."""

    dir: str = field(default_factory=_default_cache_dir)
    ttl_seconds: int = field(default_factory=_default_cache_ttl)
    force_cache_only: bool = field(default_factory=_default_force_cache_only)


@dataclass
class ApiCfg:
    """Parameters for the IDMC API."""

    base_url: str = field(default_factory=_default_base_url)
    alternate_base_url: Optional[str] = field(default_factory=_default_alt_base_url)
    endpoints: Dict[str, str] = field(default_factory=_default_endpoints)
    countries: List[str] = field(default_factory=list)
    series: List[str] = field(default_factory=lambda: ["stock", "flow"])
    date_window: DateWindow = field(default_factory=DateWindow)
    token_env: str = "IDMC_API_TOKEN"


@dataclass
class FieldAliases:
    """Column aliases for the CSV fixtures or future payloads."""

    value_flow: List[str] = field(
        default_factory=lambda: ["new_displacements", "New displacements"]
    )
    value_stock: List[str] = field(
        default_factory=lambda: ["idps", "IDPs", "Total IDPs"]
    )
    date: List[str] = field(
        default_factory=lambda: ["date", "Date", "month", "Month", "ReportingDate"]
    )
    iso3: List[str] = field(
        default_factory=lambda: ["iso3", "ISO3", "Country ISO3", "CountryISO3"]
    )


@dataclass
class HdxCfg:
    """Configuration for the HDX fallback."""

    package_id: Optional[str] = None
    base_url: Optional[str] = None


@dataclass
class IdmcConfig:
    """Top-level configuration object."""

    enabled: bool = True
    api: ApiCfg = field(default_factory=ApiCfg)
    cache: CacheCfg = field(default_factory=CacheCfg)
    field_aliases: FieldAliases = field(default_factory=FieldAliases)
    hdx: HdxCfg = field(default_factory=HdxCfg)


def _coerce_bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no"}
    return default


def _resolve_custom_path(raw_path: str | Path) -> Path:
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        return (Path.cwd() / candidate).resolve()
    return candidate.resolve()


def load(
    path: str | os.PathLike[str] | None = None, *, strict_loader: bool = False
) -> IdmcConfig:
    """Load the IDMC configuration from disk."""

    loader_warnings: Tuple[str, ...] = ()
    source_label = "resolver"
    resolved_path: Optional[Path] = None

    if path is not None:
        resolved_path = _resolve_custom_path(path)
        with resolved_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        source_label = "custom"
    else:
        (
            data,
            source_label,
            resolved_path,
            loader_warning_list,
        ) = load_connector_config("idmc", strict_mismatch=strict_loader)
        details = get_config_details("idmc")
        if details is not None:
            resolved_path = details.path
            loader_warnings = details.warnings
        else:
            resolved_path = resolved_path or DEFAULT_PATH
            loader_warnings = tuple(loader_warning_list)
        data = dict(data or {})

    if not isinstance(data, dict):
        data = {}

    api_block = data.get("api", {})
    alias_block = data.get("field_aliases", {})
    cache_block = data.get("cache", {})
    hdx_block = data.get("hdx", {}) if isinstance(data.get("hdx"), dict) else {}
    ttl_raw = cache_block.get("ttl_seconds", _default_cache_ttl())
    try:
        ttl_seconds = int(ttl_raw)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        ttl_seconds = _default_cache_ttl()

    env_countries = _csv_env("IDMC_ONLY_COUNTRIES", transform=str.upper)
    env_series = _csv_env("IDMC_SERIES", transform=lambda value: value.lower())
    env_force_cache_only = os.getenv("IDMC_FORCE_CACHE_ONLY")

    config_countries = [
        str(value).strip().upper()
        for value in api_block.get("countries", [])
        if str(value).strip()
    ]
    config_series = [
        str(value).strip().lower()
        for value in api_block.get("series", [])
        if str(value).strip()
    ] or ["stock", "flow"]

    countries = env_countries if env_countries is not None else config_countries
    series = env_series if env_series is not None else config_series
    if not series:
        series = ["stock", "flow"]

    cache_force_cache_only = _coerce_bool(
        cache_block.get("force_cache_only"), _default_force_cache_only()
    )
    if env_force_cache_only is not None:
        cache_force_cache_only = _coerce_bool(env_force_cache_only, cache_force_cache_only)

    hdx_package_id = str(hdx_block.get("package_id", "")).strip() or None
    hdx_base_url = str(hdx_block.get("base_url", "")).strip() or None

    alt_base_config = api_block.get("alternate_base_url") or api_block.get("base_url_alt")
    if isinstance(alt_base_config, str):
        alt_base_config = alt_base_config.strip() or None
    env_alt_base = _default_alt_base_url()
    alternate_base_url = env_alt_base if env_alt_base is not None else alt_base_config

    config = IdmcConfig(
        enabled=bool(data.get("enabled", True)),
        api=ApiCfg(
            base_url=api_block.get("base_url", _default_base_url()),
            alternate_base_url=alternate_base_url,
            endpoints={
                **_default_endpoints(),
                **api_block.get("endpoints", {}),
            },
            countries=countries,
            series=series,
            date_window=DateWindow(
                start=api_block.get("date_window", {}).get("start"),
                end=api_block.get("date_window", {}).get("end"),
            ),
            token_env=api_block.get("token_env", "IDMC_API_TOKEN"),
        ),
        cache=CacheCfg(
            dir=cache_block.get("dir", _default_cache_dir()),
            ttl_seconds=ttl_seconds,
            force_cache_only=cache_force_cache_only,
        ),
        field_aliases=FieldAliases(
            value_flow=alias_block.get(
                "value_flow", ["new_displacements", "New displacements"]
            ),
            value_stock=alias_block.get(
                "value_stock", ["idps", "IDPs", "Total IDPs"]
            ),
            date=alias_block.get(
                "date", ["date", "Date", "month", "Month", "ReportingDate"]
            ),
            iso3=alias_block.get(
                "iso3", ["iso3", "ISO3", "Country ISO3", "CountryISO3"]
            ),
        ),
        hdx=HdxCfg(package_id=hdx_package_id, base_url=hdx_base_url),
    )

    config._config_source = source_label  # type: ignore[attr-defined]
    config._config_path = (
        resolved_path.as_posix() if isinstance(resolved_path, Path) else None
    )  # type: ignore[attr-defined]
    config._config_warnings = tuple(loader_warnings)  # type: ignore[attr-defined]
    config._config_details = get_config_details("idmc")  # type: ignore[attr-defined]
    return config
