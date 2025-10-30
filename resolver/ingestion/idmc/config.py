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


@dataclass
class ApiCfg:
    """Parameters for the (future) IDMC API."""

    countries: List[str] = field(default_factory=list)
    series: List[str] = field(default_factory=lambda: ["stock", "flow"])
    date_window: DateWindow = field(default_factory=DateWindow)
    endpoints: Dict[str, str] = field(
        default_factory=lambda: {"monthly_flow": "", "stock": ""}
    )


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
    field_aliases: FieldAliases = field(default_factory=FieldAliases)


def load(path: str | None = None) -> IdmcConfig:
    """Load the IDMC configuration from disk."""

    cfg_path = path or DEFAULT_PATH
    with open(cfg_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    api_block = data.get("api", {})
    alias_block = data.get("field_aliases", {})
    return IdmcConfig(
        enabled=bool(data.get("enabled", True)),
        api=ApiCfg(
            countries=api_block.get("countries", []),
            series=api_block.get("series", ["stock", "flow"]),
            date_window=DateWindow(
                start=api_block.get("date_window", {}).get("start"),
                end=api_block.get("date_window", {}).get("end"),
            ),
            endpoints=api_block.get(
                "endpoints", {"monthly_flow": "", "stock": ""}
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
