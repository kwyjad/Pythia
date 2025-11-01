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
) -> Tuple[Dict[str, Any], str]:
    """Return a connector config payload and a label describing its source."""

    normalized = str(name or "").strip()
    if not normalized:
        raise ValueError("Connector name must be provided")

    ingestion_path = (INGESTION_CONFIG_ROOT / f"{normalized}.yml").resolve()
    fallback_path = (LEGACY_CONFIG_ROOT / f"{normalized}.yml").resolve()

    ingestion_exists = ingestion_path.is_file()
    fallback_exists = fallback_path.is_file()

    warnings: list[str] = []
    duplicates_equal = False
    fallback_used = False
    source_label = "ingestion"
    chosen_path: Optional[Path] = None
    payload: Dict[str, Any] = {}

    if ingestion_exists:
        payload = _load_yaml(ingestion_path)
        chosen_path = ingestion_path
    if not ingestion_exists and fallback_exists:
        payload = _load_yaml(fallback_path)
        chosen_path = fallback_path
        source_label = "resolver"
        fallback_used = True
        warning = (
            f"Using { _relpath(fallback_path) }; please move to "
            f"{_relpath(ingestion_path)}"
        )
        warnings.append(warning)
        log.warning("%s", warning)
    elif ingestion_exists and fallback_exists:
        fallback_payload = _load_yaml(fallback_path)
        if payload == fallback_payload:
            duplicates_equal = True
            source_label = "ingestion (dup-equal)"
        else:
            differing = _diff_keys(payload, fallback_payload)
            mismatch = (
                f"Duplicate configs for '{normalized}' differ: "
                f"{_relpath(ingestion_path)} vs {_relpath(fallback_path)} "
                f"(keys: {', '.join(differing)})"
            )
            if strict_mismatch:
                raise ValueError(mismatch)
            warnings.append(mismatch)
            log.warning("%s", mismatch)
    elif not ingestion_exists and not fallback_exists:
        raise FileNotFoundError(
            f"No config found for '{normalized}' in "
            f"{_relpath(ingestion_path)} or {_relpath(fallback_path)}"
        )

    if chosen_path is None:
        # This should only happen if ingestion exists without fallback.
        chosen_path = ingestion_path if ingestion_exists else fallback_path

    payload = dict(payload or {})
    result = ConfigResolution(
        name=normalized,
        source=source_label,
        path=chosen_path,
        ingestion_path=ingestion_path,
        fallback_path=fallback_path,
        warnings=tuple(warnings),
        duplicates_equal=duplicates_equal,
        fallback_used=fallback_used,
    )
    _LAST_RESULTS[normalized] = result
    return payload, source_label


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
