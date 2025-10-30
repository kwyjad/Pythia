"""Configuration helpers for the IDMC connector skeleton."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import yaml

DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "idmc.yml")
DEFAULT_PATH = os.path.abspath(DEFAULT_PATH)


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


def _default_base_url() -> str:
    return os.getenv("IDMC_BASE_URL", "https://backend.idmcdb.org")


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
class IdmcConfig:
    """Top-level configuration object."""

    enabled: bool = True
    api: ApiCfg = field(default_factory=ApiCfg)
    cache: CacheCfg = field(default_factory=CacheCfg)
    field_aliases: FieldAliases = field(default_factory=FieldAliases)


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


def load(path: str | None = None) -> IdmcConfig:
    """Load the IDMC configuration from disk."""

    cfg_path = path or DEFAULT_PATH
    with open(cfg_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    api_block = data.get("api", {})
    alias_block = data.get("field_aliases", {})
    cache_block = data.get("cache", {})
    ttl_raw = cache_block.get("ttl_seconds", _default_cache_ttl())
    try:
        ttl_seconds = int(ttl_raw)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        ttl_seconds = _default_cache_ttl()

    return IdmcConfig(
        enabled=bool(data.get("enabled", True)),
        api=ApiCfg(
            base_url=api_block.get("base_url", _default_base_url()),
            endpoints={
                **_default_endpoints(),
                **api_block.get("endpoints", {}),
            },
            countries=api_block.get("countries", []),
            series=api_block.get("series", ["stock", "flow"]),
            date_window=DateWindow(
                start=api_block.get("date_window", {}).get("start"),
                end=api_block.get("date_window", {}).get("end"),
            ),
            token_env=api_block.get("token_env", "IDMC_API_TOKEN"),
        ),
        cache=CacheCfg(
            dir=cache_block.get("dir", _default_cache_dir()),
            ttl_seconds=ttl_seconds,
            force_cache_only=_coerce_bool(
                cache_block.get("force_cache_only"), _default_force_cache_only()
            ),
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
    )
