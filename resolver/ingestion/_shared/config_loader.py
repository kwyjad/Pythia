# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Shared helpers for connector configuration discovery."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import yaml

log = logging.getLogger(__name__)

RESOLVER_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = RESOLVER_ROOT.parent
INGESTION_CONFIG_ROOT = RESOLVER_ROOT / "ingestion" / "config"
LEGACY_CONFIG_ROOT = RESOLVER_ROOT / "config"


@dataclass(frozen=True)
class ConfigResolution:
    """Metadata about the most recent config lookup."""

    name: str
    source: str
    path: Path
    ingestion_path: Path
    fallback_path: Path
    warnings: Tuple[str, ...]
    duplicates_equal: bool = False
    fallback_used: bool = False


_LAST_RESULTS: Dict[str, ConfigResolution] = {}


def _relpath(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
    except FileNotFoundError:
        return {}
    except Exception as exc:  # pragma: no cover - defensive guard
        log.warning("Failed to load config from %s: %s", path, exc)
        return {}
    if isinstance(loaded, Mapping):
        return dict(loaded)
    return {}


def _diff_keys(left: Mapping[str, Any], right: Mapping[str, Any]) -> Sequence[str]:
    keys = set(left.keys()) | set(right.keys())
    differing = [
        key
        for key in sorted(keys)
        if left.get(key) != right.get(key)
    ]
    return differing or ["<structure>"]


def load_connector_config(
    name: str, *, strict_mismatch: bool = False
) -> Tuple[Dict[str, Any], str, Path, Tuple[str, ...]]:
    """Return a connector config payload plus provenance details."""

    normalized = str(name or "").strip()
    if not normalized:
        raise ValueError("Connector name must be provided")

    ingestion_path = (INGESTION_CONFIG_ROOT / f"{normalized}.yml").resolve()
    legacy_path = (LEGACY_CONFIG_ROOT / f"{normalized}.yml").resolve()

    ingestion_exists = ingestion_path.is_file()
    legacy_exists = legacy_path.is_file()

    warnings: list[str] = []
    duplicates_equal = False
    fallback_used = False
    source_label = "resolver"
    chosen_path: Optional[Path] = None
    payload: Dict[str, Any] = {}

    if legacy_exists:
        payload = _load_yaml(legacy_path)
        chosen_path = legacy_path
    if not legacy_exists and ingestion_exists:
        payload = _load_yaml(ingestion_path)
        chosen_path = ingestion_path
        source_label = "ingestion"
        fallback_used = True
        warning = (
            f"Using {_relpath(ingestion_path)}; please move to "
            f"{_relpath(legacy_path)}"
        )
        warnings.append(warning)
        log.warning("%s", warning)
    elif legacy_exists and ingestion_exists:
        ingestion_payload = _load_yaml(ingestion_path)
        if payload == ingestion_payload:
            duplicates_equal = True
            source_label = "resolver (dup-equal)"
        else:
            differing = _diff_keys(payload, ingestion_payload)
            mismatch = (
                f"Duplicate configs for '{normalized}' differ: "
                f"{_relpath(legacy_path)} vs {_relpath(ingestion_path)} "
                f"(keys: {', '.join(differing)})"
            )
            if strict_mismatch:
                raise ValueError(mismatch)
            warnings.append(
                f"duplicate-mismatch: resolver preferred over ingestion; {mismatch}"
            )
            log.warning("%s", warnings[-1])
    elif not legacy_exists and not ingestion_exists:
        raise FileNotFoundError(
            f"No config found for '{normalized}' in "
            f"{_relpath(legacy_path)} or {_relpath(ingestion_path)}"
        )

    if chosen_path is None:
        # This should only happen if legacy exists without ingestion.
        chosen_path = legacy_path if legacy_exists else ingestion_path

    payload = dict(payload or {})
    result = ConfigResolution(
        name=normalized,
        source=source_label,
        path=chosen_path,
        ingestion_path=ingestion_path,
        fallback_path=legacy_path,
        warnings=tuple(warnings),
        duplicates_equal=duplicates_equal,
        fallback_used=fallback_used,
    )
    _LAST_RESULTS[normalized] = result
    return payload, source_label, chosen_path, tuple(warnings)


def get_config_details(name: str) -> Optional[ConfigResolution]:
    """Return metadata from the most recent lookup for ``name``."""

    return _LAST_RESULTS.get(str(name or "").strip())


def get_config_warnings(name: str) -> Tuple[str, ...]:
    """Return warnings emitted during the last lookup for ``name``."""

    details = get_config_details(name)
    return details.warnings if details else ()


def emit_config_source(
    diag_writer: Callable[[Mapping[str, Any]], Any],
    connector_name: str,
    source_label: str,
    warnings: Sequence[str] | None = None,
) -> None:
    """Record a config source payload using ``diag_writer``."""

    payload: Dict[str, Any] = {
        "connector": str(connector_name),
        "config_source": str(source_label),
    }
    if warnings:
        payload["warnings"] = [str(item) for item in warnings if str(item)]
    try:
        diag_writer(payload)
    except Exception:  # pragma: no cover - diagnostics helper
        log.debug("Unable to emit config source diagnostics", exc_info=True)


__all__ = [
    "load_connector_config",
    "get_config_details",
    "get_config_warnings",
    "emit_config_source",
    "ConfigResolution",
    "INGESTION_CONFIG_ROOT",
    "LEGACY_CONFIG_ROOT",
]
