#!/usr/bin/env python3
"""Run ingestion connectors and optional stubs to populate staging CSVs."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import fnmatch
from collections import Counter
import logging
import os
import re
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

import resolver.ingestion.feature_flags as ff
from resolver.ingestion._exit_policy import compute_exit_code
from resolver.ingestion.diagnostics_emitter import (
    append_jsonl as diagnostics_append_jsonl,
    finalize_run as diagnostics_finalize_run,
    start_run as diagnostics_start_run,
)

from resolver.ingestion._manifest import (
    count_csv_rows,
    ensure_manifest_for_csv,
    load_manifest,
    manifest_path_for,
)
from resolver.ingestion._retry import retry_call
from resolver.ingestion._runner_logging import (
    attach_connector_handler,
    child_logger,
    detach_connector_handler,
    init_logger,
    log_env_summary,
    redact,
)
from resolver.ingestion.utils.io import (
    resolve_output_path,
    resolve_period_label,
    resolve_staging_dir,
)

STAGING = ROOT.parent / "staging"
CONFIG_DIR = ROOT / "config"
LOGS_DIR = ROOT.parent / "logs" / "ingestion"
DIAGNOSTICS_DIR = PROJECT_ROOT / "diagnostics" / "ingestion"
DIAGNOSTICS_REPORT = DIAGNOSTICS_DIR / "connectors_report.jsonl"
RESOLVER_DEBUG = bool(int(os.getenv("RESOLVER_DEBUG", "0") or 0))

CONFIG_OVERRIDES = {
    "wfp_mvam": CONFIG_DIR / "wfp_mvam_sources.yml",
}

SMOKE_ENV_DEFAULTS = {
    "RESOLVER_MAX_PAGES": "2",
    "RESOLVER_MAX_RESULTS": "200",
    "RESOLVER_WINDOW_DAYS": "7",
    "RESOLVER_FAIL_ON_STUB_ERROR": "0",
}


def _repo_root() -> Path:
    """Return the repository root directory."""

    return Path(__file__).resolve().parents[2]


def _apply_smoke_env_defaults(
    logger: logging.LoggerAdapter | logging.Logger,
) -> Dict[str, str]:
    applied: Dict[str, str] = {}
    for key, value in SMOKE_ENV_DEFAULTS.items():
        if os.getenv(key):
            continue
        os.environ[key] = value
        applied[key] = value
    if applied:
        logger.info(
            "applied smoke defaults",
            extra={"event": "smoke_defaults", "values": applied},
        )
    return applied


def _module_name_from_path(py_path: Path) -> str:
    """Convert a Python file path to its dotted module path."""

    if not py_path.exists():
        raise FileNotFoundError(f"Connector file does not exist: {py_path}")

    no_ext = py_path.with_suffix("")
    parts = list(no_ext.parts)
    try:
        idx = parts.index("resolver")
    except ValueError as exc:
        raise RuntimeError(
            f"Connector path '{py_path}' does not include 'resolver' package root."
        ) from exc

    module = ".".join(parts[idx:])
    if not module:
        raise RuntimeError(f"Failed to derive module name from {py_path}")
    return module

INGESTION_MODE = (os.environ.get("RESOLVER_INGESTION_MODE") or "").strip().lower()
INCLUDE_STUBS = os.environ.get("RESOLVER_INCLUDE_STUBS", "0") == "1"
FAIL_ON_STUB_ERROR = os.environ.get("RESOLVER_FAIL_ON_STUB_ERROR", "0") == "1"
FORCE_DTM_STUB = os.environ.get("RESOLVER_FORCE_DTM_STUB", "0") == "1"

REAL = [
    "ifrc_go_client.py",
    "reliefweb_client.py",
    "unhcr_client.py",
    "unhcr_odp_client.py",
    "who_phe_client.py",
    "ipc_client.py",
    "wfp_mvam_client.py",
    "acled_client.py",
    "dtm_client.py",
    "hdx_client.py",
    "emdat_client.py",
    "gdacs_client.py",
    "worldpop_client.py",
]

CONNECTOR_OUTPUTS: Dict[str, str] = {
    "ifrc_go_client.py": "ifrc_go.csv",
    "reliefweb_client.py": "reliefweb.csv",
    "unhcr_client.py": "unhcr.csv",
    "unhcr_odp_client.py": "unhcr_odp.csv",
    "who_phe_client.py": "who_phe.csv",
    "ipc_client.py": "ipc.csv",
    "wfp_mvam_client.py": "wfp_mvam.csv",
    "acled_client.py": "acled.csv",
    "dtm_client.py": "dtm_displacement.csv",
    "hdx_client.py": "hdx.csv",
    "emdat_client.py": "emdat_pa.csv",
    "gdacs_client.py": "gdacs_signals.csv",
    "worldpop_client.py": "worldpop_denominators.csv",
    "ifrc_go_stub.py": "ifrc_go.csv",
    "reliefweb_stub.py": "reliefweb.csv",
    "unhcr_stub.py": "unhcr.csv",
    "hdx_stub.py": "hdx.csv",
    "who_stub.py": "who.csv",
    "ipc_stub.py": "ipc.csv",
    "emdat_stub.py": "emdat.csv",
    "gdacs_stub.py": "gdacs.csv",
    "copernicus_stub.py": "copernicus.csv",
    "unosat_stub.py": "unosat.csv",
    "acled_stub.py": "acled.csv",
    "ucdp_stub.py": "ucdp.csv",
    "fews_stub.py": "fews.csv",
    "wfp_mvam_stub.py": "wfp_mvam.csv",
    "gov_ndma_stub.py": "gov_ndma.csv",
    "dtm_stub.py": "dtm.csv",
}

SUMMARY_TARGETS = {
    "ifrc_go_client.py": {
        "label": "IFRC GO",
        "filename": CONNECTOR_OUTPUTS["ifrc_go_client.py"],
        "config": CONFIG_DIR / "ifrc_go.yml",
    },
    "reliefweb_client.py": {
        "label": "ReliefWeb",
        "filename": CONNECTOR_OUTPUTS["reliefweb_client.py"],
        "config": CONFIG_DIR / "reliefweb.yml",
    },
    "unhcr_client.py": {
        "label": "UNHCR",
        "filename": CONNECTOR_OUTPUTS["unhcr_client.py"],
        "config": CONFIG_DIR / "unhcr.yml",
    },
    "unhcr_odp_client.py": {
        "label": "UNHCR-ODP",
        "filename": CONNECTOR_OUTPUTS["unhcr_odp_client.py"],
        "config": None,
    },
    "who_phe_client.py": {
        "label": "WHO-PHE",
        "filename": CONNECTOR_OUTPUTS["who_phe_client.py"],
        "config": CONFIG_DIR / "who_phe.yml",
    },
    "ipc_client.py": {
        "label": "IPC",
        "filename": CONNECTOR_OUTPUTS["ipc_client.py"],
        "config": CONFIG_DIR / "ipc.yml",
    },
    "wfp_mvam_client.py": {
        "label": "WFP-mVAM",
        "filename": CONNECTOR_OUTPUTS["wfp_mvam_client.py"],
        "config": CONFIG_DIR / "wfp_mvam_sources.yml",
    },
    "acled_client.py": {
        "label": "ACLED",
        "filename": CONNECTOR_OUTPUTS["acled_client.py"],
        "config": CONFIG_DIR / "acled.yml",
    },
    "dtm_client.py": {
        "label": "DTM",
        "filename": CONNECTOR_OUTPUTS["dtm_client.py"],
        "config": CONFIG_DIR / "dtm.yml",
    },
    "gdacs_client.py": {
        "label": "GDACS",
        "filename": CONNECTOR_OUTPUTS["gdacs_client.py"],
        "config": CONFIG_DIR / "gdacs.yml",
    },
    "emdat_client.py": {
        "label": "EM-DAT",
        "filename": CONNECTOR_OUTPUTS["emdat_client.py"],
        "config": CONFIG_DIR / "emdat.yml",
    },
    "worldpop_client.py": {
        "label": "WorldPop",
        "filename": CONNECTOR_OUTPUTS["worldpop_client.py"],
        "config": CONFIG_DIR / "worldpop.yml",
    },
}

STUBS = [
    "ifrc_go_stub.py",
    "reliefweb_stub.py",
    "unhcr_stub.py",
    "hdx_stub.py",
    "who_stub.py",
    "ipc_stub.py",
    "emdat_stub.py",
    "gdacs_stub.py",
    "copernicus_stub.py",
    "unosat_stub.py",
    "acled_stub.py",
    "ucdp_stub.py",
    "fews_stub.py",
    "wfp_mvam_stub.py",
    "gov_ndma_stub.py",
]

if FORCE_DTM_STUB:
    REAL = [name for name in REAL if name != "dtm_client.py"]
    if "dtm_stub.py" not in STUBS:
        STUBS.insert(0, "dtm_stub.py")
else:
    STUBS = [name for name in STUBS if name != "dtm_stub.py"]

SKIP_ENVS = {
    "ifrc_go_client.py": ("RESOLVER_SKIP_IFRCGO", "IFRC GO connector"),
    "reliefweb_client.py": ("RESOLVER_SKIP_RELIEFWEB", "ReliefWeb connector"),
    "unhcr_client.py": ("RESOLVER_SKIP_UNHCR", "UNHCR connector"),
    "unhcr_odp_client.py": ("RESOLVER_SKIP_UNHCR_ODP", "UNHCR ODP connector"),
    "acled_client.py": ("RESOLVER_SKIP_ACLED", "ACLED connector"),
    "dtm_client.py": ("RESOLVER_SKIP_DTM", "DTM connector"),
    "emdat_client.py": ("RESOLVER_SKIP_EMDAT", "EM-DAT connector"),
    "gdacs_client.py": ("RESOLVER_SKIP_GDACS", "GDACS connector"),
    "who_phe_client.py": ("RESOLVER_SKIP_WHO", "WHO PHE connector"),
    "ipc_client.py": ("RESOLVER_SKIP_IPC", "IPC connector"),
    "hdx_client.py": ("RESOLVER_SKIP_HDX", "HDX connector"),
    "worldpop_client.py": ("RESOLVER_SKIP_WORLDPOP", "WorldPop connector"),
    "wfp_mvam_client.py": ("RESOLVER_SKIP_WFP_MVAM", "WFP mVAM connector"),
}

SECRET_GATES: Dict[str, Dict[str, object]] = {
    "acled_client.py": {
        "alternatives": [
            ("ACLED_REFRESH_TOKEN",),
            ("ACLED_ACCESS_TOKEN",),
            ("ACLED_TOKEN",),
            ("ACLED_USERNAME", "ACLED_PASSWORD"),
        ],
        "message": "missing ACLED_REFRESH_TOKEN/ACLED_TOKEN credentials",
    },
    "unhcr_odp_client.py": {
        "alternatives": [
            (
                "UNHCR_ODP_USERNAME",
                "UNHCR_ODP_PASSWORD",
                "UNHCR_ODP_CLIENT_ID",
                "UNHCR_ODP_CLIENT_SECRET",
            ),
        ],
        "message": (
            "missing UNHCR_ODP_USERNAME/UNHCR_ODP_PASSWORD/UNHCR_ODP_CLIENT_ID/"
            "UNHCR_ODP_CLIENT_SECRET"
        ),
    },
}


# Prefer known "fatal" / non-retryable exit codes (sysexits) when available
_EX_USAGE = getattr(os, "EX_USAGE", 64)
_EX_DATAERR = getattr(os, "EX_DATAERR", 65)
_EX_NOINPUT = getattr(os, "EX_NOINPUT", 66)
_EX_NOUSER = getattr(os, "EX_NOUSER", 67)
_EX_NOHOST = getattr(os, "EX_NOHOST", 68)
_EX_UNAVAILABLE = getattr(os, "EX_UNAVAILABLE", 69)
_EX_SOFTWARE = getattr(os, "EX_SOFTWARE", 70)
_EX_OSERR = getattr(os, "EX_OSERR", 71)
_EX_OSFILE = getattr(os, "EX_OSFILE", 72)
_EX_CANTCREAT = getattr(os, "EX_CANTCREAT", 73)
_EX_IOERR = getattr(os, "EX_IOERR", 74)
_EX_TEMPFAIL = getattr(os, "EX_TEMPFAIL", 75)
_EX_PROTOCOL = getattr(os, "EX_PROTOCOL", 76)
_EX_NOPERM = getattr(os, "EX_NOPERM", 77)
_EX_CONFIG = getattr(os, "EX_CONFIG", 78)

# Treat these as non-retryable (usage/config/software errors)
NON_RETRYABLE_EXIT_CODES = {
    2,  # bad CLI usage (common)
    _EX_USAGE,
    _EX_DATAERR,
    _EX_NOINPUT,
    _EX_NOUSER,
    _EX_NOHOST,
    _EX_UNAVAILABLE,
    _EX_SOFTWARE,
    _EX_OSERR,
    _EX_OSFILE,
    _EX_CANTCREAT,
    _EX_IOERR,
    _EX_PROTOCOL,
    _EX_NOPERM,
    _EX_CONFIG,
}

# Heuristics for transient errors in stderr/stdout
_TRANSIENT_PAT = re.compile(
    r"(timed out|timeout|temporar\w+ unavailable|connection reset|"
    r"connection aborted|connection refused|network is unreachable|"
    r"ECONNRESET|ETIMEDOUT|EHOSTUNREACH|ENETUNREACH|EAI_AGAIN|"
    r"rate limit|429|5\d{2}\b|service unavailable|try again)",
    re.IGNORECASE,
)

_MONTH_RE = re.compile(r"^(\d{4})[-/](\d{2})$")
_DATE_RE = re.compile(r"^(\d{4})[-/](\d{2})[-/](\d{2})$")
_DIGIT_MONTH_RE = re.compile(r"^(\d{4})(\d{2})$")
_DIGIT_DATE_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})$")


def _coerce_process_stream(stream: object | None) -> str:
    if stream is None:
        return ""
    if isinstance(stream, (bytes, bytearray)):
        try:
            return stream.decode(errors="ignore")
        except Exception:  # noqa: BLE001
            return ""
    return str(stream)


def _is_retryable_exception(
    exc: BaseException,
    *,
    exit_code: int | None = None,
    stderr: str | None = None,
    stdout: str | None = None,
) -> bool:
    # Network/timeouts raised as exceptions by helpers remain retryable
    transient_types = (TimeoutError, ConnectionError)
    if isinstance(exc, transient_types):
        return True

    # Subprocess script failures: retry unless clearly non-retryable
    if isinstance(exc, subprocess.CalledProcessError):
        code = exc.returncode if exit_code is None else exit_code
        if code in NON_RETRYABLE_EXIT_CODES:
            return False
        text_parts: list[str] = []
        if stderr is not None:
            text_parts.append(stderr)
        if stdout is not None:
            text_parts.append(stdout)
        if not text_parts:
            if hasattr(exc, "stderr") and exc.stderr:
                text_parts.append(_coerce_process_stream(exc.stderr))
            if hasattr(exc, "stdout") and exc.stdout:
                text_parts.append(_coerce_process_stream(exc.stdout))
        text = "\n".join(part for part in text_parts if part)
        # Retry on unknown codes if we see transient hints; otherwise still retry (assume transient)
        return True if not text else bool(_TRANSIENT_PAT.search(text))

    # Default to non-retryable
    exc_name = exc.__class__.__name__
    transient_names = (
        "Timeout",
        "ConnectionError",
        "SSLError",
        "ReadTimeout",
        "ChunkedEncodingError",
    )
    if any(name in exc_name for name in transient_names):
        return True
    message = str(exc).lower()
    transient_phrases = ("rate limit", "waf", "temporarily unavailable")
    return any(phrase in message for phrase in transient_phrases)


@dataclass
class ConnectorSpec:
    filename: str
    path: Path
    kind: str
    origin: str = "config"
    authoritatively_selected: bool = False
    output_path: Optional[Path] = None
    summary: Optional[str] = None
    skip_reason: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)
    config_path: Optional[Path] = None
    config: Dict[str, object] = field(default_factory=dict)
    canonical_name: str = ""
    ci_gate_reason: Optional[str] = None
    selected_by_only: bool = False
    matched_by_pattern: bool = False
    enable_decision: EnableDecision | None = None

    @property
    def name(self) -> str:
        return self.filename.rsplit(".", 1)[0]


@dataclass
class EnableDecision:
    should_run: bool
    gated_by: str
    forced_sources: tuple[str, ...] = ()
    config_enabled: bool = False
    has_config_flag: bool = False
    applied_skip_reason: Optional[str] = None
    ci_gate_reason: Optional[str] = None


def _load_yaml(path: Optional[Path]) -> dict:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data if isinstance(data, dict) else {}


def _coerce_config_flag(cfg: dict | None) -> tuple[bool, bool]:
    """Return ``(enabled, has_flag)`` for a connector config mapping."""

    if not isinstance(cfg, dict):
        return False, False
    raw = None
    has_flag = False
    if "enabled" in cfg:
        raw = cfg.get("enabled")
        has_flag = True
    elif "enable" in cfg:
        raw = cfg.get("enable")
        has_flag = True
    if raw is None:
        return False, has_flag
    if isinstance(raw, bool):
        return raw, has_flag
    if isinstance(raw, str):
        text = raw.strip().lower()
        if text in {"true", "1", "yes", "on"}:
            return True, has_flag
        if text in {"false", "0", "no", "off"}:
            return False, has_flag
    try:
        return bool(raw), has_flag
    except Exception:  # noqa: BLE001 - defensive cast
        return False, has_flag


def is_authoritatively_selected(spec: ConnectorSpec) -> bool:
    """Return ``True`` when the spec originated from an authoritative list."""

    if getattr(spec, "authoritatively_selected", None) is not None:
        return bool(spec.authoritatively_selected)
    return spec.origin in {"real_list", "stub_list"}


def _rows_and_method(path: Optional[Path]) -> tuple[int, str]:
    if path is None or not path.exists():
        return 0, "missing"

    manifest = load_manifest(manifest_path_for(path))
    manifest_rows: Optional[int]
    if manifest and isinstance(manifest.get("row_count"), int):
        manifest_rows = int(manifest["row_count"])
    else:
        manifest_rows = None

    rows_actual = count_csv_rows(path)
    if manifest_rows is None:
        return rows_actual, "recount"

    if rows_actual != manifest_rows:
        logging.getLogger(__name__).warning(
            "Manifest row_count mismatch; recounting",
            extra={
                "event": "manifest_mismatch",
                "path": redact(str(path)),
                "manifest_rows": manifest_rows,
                "recount_rows": rows_actual,
            },
        )
        try:
            ensure_manifest_for_csv(path)
        except FileNotFoundError:
            pass
        return rows_actual, "manifest+verified"

    return manifest_rows, "manifest"


def _normalise_month_value(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("/", "-")
    if "T" in text:
        text = text.split("T", 1)[0]
    match = _MONTH_RE.match(text)
    if match:
        return f"{match.group(1)}-{match.group(2)}"
    if len(text) >= 10:
        candidate = text[:10]
        match = _DATE_RE.match(candidate)
        if match:
            return f"{match.group(1)}-{match.group(2)}"
        try:
            parsed = dt.date.fromisoformat(candidate)
        except ValueError:
            parsed = None
        if parsed:
            return f"{parsed.year:04d}-{parsed.month:02d}"
    match = _DIGIT_MONTH_RE.match(text)
    if match:
        return f"{match.group(1)}-{match.group(2)}"
    match = _DIGIT_DATE_RE.match(text)
    if match:
        return f"{match.group(1)}-{match.group(2)}"
    return None


def _normalise_date_value(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("/", "-")
    if "T" in text:
        text = text.split("T", 1)[0]
    if len(text) >= 10:
        candidate = text[:10]
        try:
            dt.date.fromisoformat(candidate)
            return candidate
        except ValueError:
            match = _DATE_RE.match(candidate)
            if match:
                return "-".join(match.groups())
    match = _DIGIT_DATE_RE.match(text)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    return None


def _clean_iso_value(value: str) -> str:
    return str(value).strip().upper()


def _clean_hazard_value(value: str) -> str:
    return str(value).strip()


def _first_nonempty(row: Mapping[str, object], columns: Sequence[str]) -> Optional[str]:
    for column in columns:
        if column not in row:
            continue
        candidate = row.get(column)
        if candidate is None:
            continue
        text = str(candidate).strip()
        if text:
            return text
    return None


def _collect_output_samples(path: Optional[Path], limit: int = 5000) -> tuple[dict, dict]:
    coverage = {"ym_min": None, "ym_max": None, "as_of_min": None, "as_of_max": None}
    samples = {"top_iso3": [], "top_hazard": []}
    if path is None or not path.exists():
        return coverage, samples

    iso_counter: Counter[str] = Counter()
    hazard_counter: Counter[str] = Counter()
    ym_values: List[str] = []
    as_of_values: List[str] = []

    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            if not fieldnames:
                return coverage, samples
            iso_fields = [
                name
                for name in fieldnames
                if name
                and any(token in name.lower() for token in ("iso3", "country_iso3", "adm0_iso3", "iso"))
            ]
            if not iso_fields:
                iso_fields = [name for name in fieldnames if name and "country" in name.lower()]
            hazard_fields = [
                name
                for name in fieldnames
                if name and any(token in name.lower() for token in ("hazard", "shock", "event_type"))
            ]
            ym_fields = [
                name
                for name in fieldnames
                if name and any(token in name.lower() for token in ("ym", "yearmonth", "month", "period"))
            ]
            as_of_fields = [
                name
                for name in fieldnames
                if name and ("as_of" in name.lower() or "asof" in name.lower())
            ]
            if not as_of_fields:
                fallback_candidates = [
                    name
                    for name in fieldnames
                    if name
                    and (
                        name.lower().endswith("_date")
                        or name.lower()
                        in {"as_of_date", "publication_date", "report_date", "date"}
                    )
                ]
                for candidate in fallback_candidates:
                    if candidate not in as_of_fields:
                        as_of_fields.append(candidate)

            for idx, row in enumerate(reader):
                if limit and idx >= limit:
                    break
                iso_value = _first_nonempty(row, iso_fields)
                if iso_value:
                    iso_counter[_clean_iso_value(iso_value)] += 1
                hazard_value = _first_nonempty(row, hazard_fields)
                if hazard_value:
                    hazard_counter[_clean_hazard_value(hazard_value)] += 1
                ym_value = None
                for field in ym_fields:
                    candidate = _normalise_month_value(row.get(field))
                    if candidate:
                        ym_value = candidate
                        break
                if ym_value:
                    ym_values.append(ym_value)
                as_of_value = None
                for field in as_of_fields:
                    candidate = _normalise_date_value(row.get(field))
                    if candidate:
                        as_of_value = candidate
                        break
                if as_of_value:
                    as_of_values.append(as_of_value)
    except FileNotFoundError:
        return coverage, samples
    except Exception as exc:  # noqa: BLE001
        logging.getLogger(__name__).debug(
            "diagnostics sampling failed",
            extra={"event": "diagnostics_error", "path": str(path)},
            exc_info=exc,
        )
        return coverage, samples

    if ym_values:
        coverage["ym_min"] = min(ym_values)
        coverage["ym_max"] = max(ym_values)
    if as_of_values:
        coverage["as_of_min"] = min(as_of_values)
        coverage["as_of_max"] = max(as_of_values)
    samples["top_iso3"] = iso_counter.most_common(5)
    samples["top_hazard"] = hazard_counter.most_common(5)
    return coverage, samples


def _coerce_int_safe(value: object) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def _summarise_connector(name: str, output_path: Optional[Path]) -> str | None:
    meta = SUMMARY_TARGETS.get(name)
    if not meta or output_path is None:
        return None
    label = meta["label"]
    rows, method = _rows_and_method(output_path)
    parts = [f"[{label}] rows:{rows}"]
    if method in {"recount", "manifest+verified"}:
        parts.append(f"rows_method:{method}")
    cfg = _load_yaml(meta.get("config"))
    enabled_flag: Optional[bool] = None
    if cfg and isinstance(cfg.get("enabled"), bool):
        enabled_flag = bool(cfg.get("enabled"))
    added_enabled = False

    if name == "who_phe_client.py":
        enabled = bool(cfg.get("enabled", False))
        sources_cfg = cfg.get("sources", {})
        configured = 0
        if isinstance(sources_cfg, dict):
            for value in sources_cfg.values():
                if isinstance(value, dict):
                    url = str(value.get("url", "")).strip()
                else:
                    url = str(value).strip()
                if url:
                    configured += 1
        parts.append(f"enabled:{'yes' if enabled else 'no'}")
        added_enabled = True
        parts.append(f"sources:{configured}")
    elif name == "wfp_mvam_client.py":
        enabled = bool(cfg.get("enabled", False))
        sources = cfg.get("sources", [])
        count = 0
        if isinstance(sources, dict):
            sources = list(sources.values())
        if isinstance(sources, list):
            for entry in sources:
                if isinstance(entry, dict) and str(entry.get("url", "")).strip():
                    count += 1
                elif isinstance(entry, str) and entry.strip():
                    count += 1
        parts.append(f"enabled:{'yes' if enabled else 'no'}")
        added_enabled = True
        parts.append(f"sources:{count}")
    elif name == "ipc_client.py":
        enabled = bool(cfg.get("enabled", False))
        feeds = cfg.get("feeds", [])
        if isinstance(feeds, dict):
            feed_count = sum(1 for value in feeds.values() if value)
        elif isinstance(feeds, list):
            feed_count = len(feeds)
        else:
            feed_count = 0
        parts.append(f"enabled:{'yes' if enabled else 'no'}")
        added_enabled = True
        parts.append(f"feeds:{feed_count}")
    elif name == "unhcr_client.py":
        years = cfg.get("include_years") or []
        years_text: str
        if isinstance(years, list) and years:
            years_text = ",".join(str(y) for y in years)
        else:
            try:
                years_back = int(cfg.get("years_back", 3) or 0)
            except Exception:
                years_back = 3
            current = dt.date.today().year
            years_text = ",".join(str(current - offset) for offset in range(years_back + 1))
        parts.append(f"years:{years_text}")
    elif name == "acled_client.py":
        token_present = bool(os.getenv("ACLED_TOKEN") or str(cfg.get("token", "")).strip())
        parts.append(f"token:{'yes' if token_present else 'no'}")
    if not added_enabled and enabled_flag is not None:
        parts.append(f"enabled:{'yes' if enabled_flag else 'no'}")
    return " ".join(parts)


def _safe_summary(name: str, output_path: Optional[Path]) -> str | None:
    try:
        summary = _summarise_connector(name, output_path)
    except Exception as exc:  # noqa: BLE001
        logging.getLogger(__name__).warning(
            "failed to summarise connector %s", name, exc_info=exc
        )
        return None
    if summary:
        print(summary)
    return summary


def _should_skip(script: str) -> Optional[str]:
    env_name, label = SKIP_ENVS.get(script, (None, None))
    if env_name and os.environ.get(env_name) == "1":
        return f"{env_name}=1 — {label}"
    return None


def _check_secret_gate(filename: str) -> tuple[bool, Optional[str]]:
    gate = SECRET_GATES.get(filename)
    if not gate:
        return True, None
    alternatives = gate.get("alternatives", [])
    for combo in alternatives:
        if all(os.getenv(name, "").strip() for name in combo):
            return True, None
    message = gate.get("message")
    if isinstance(message, str) and message:
        return False, message
    formatted = " or ".join(
        " + ".join(names) if len(names) > 1 else names[0]
        for names in alternatives
    )
    if not formatted:
        formatted = "required credentials"
    return False, f"missing {formatted}"


def _normalise_name(name: str) -> str:
    name = name.strip()
    if not name:
        return name
    if not name.endswith(".py"):
        name = f"{name}.py"
    return name


def _filter_by_pattern(specs: Sequence[ConnectorSpec], pattern: str) -> List[ConnectorSpec]:
    pattern_lower = pattern.lower()
    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error:
        regex = None
    matched: List[ConnectorSpec] = []
    for spec in specs:
        identifiers = {spec.filename, spec.name, spec.canonical_name}
        if spec.config_path:
            identifiers.add(spec.config_path.stem)
        identifiers = {value for value in identifiers if value}
        fnmatch_hit = any(
            fnmatch.fnmatch(value.lower(), pattern_lower) for value in identifiers
        )
        regex_hit = bool(regex and any(regex.search(value) for value in identifiers))
        substr_hit = any(pattern_lower in value.lower() for value in identifiers)
        if fnmatch_hit or regex_hit or substr_hit:
            spec.matched_by_pattern = True
            matched.append(spec)
    return matched


def _discover_config_path(
    filename: str,
    canonical_name: str,
    explicit: Optional[Path],
) -> tuple[Optional[Path], List[Path]]:
    """Return the config path for a connector along with ignored candidates."""

    override_path = CONFIG_OVERRIDES.get(canonical_name)
    if override_path:
        return Path(override_path), []

    if explicit:
        explicit_path = Path(explicit)
        return explicit_path, []

    candidates: List[Path] = []
    if canonical_name:
        exact = CONFIG_DIR / f"{canonical_name}.yml"
        if exact.exists():
            candidates.append(exact)
        pattern = sorted(CONFIG_DIR.glob(f"{canonical_name}_*.yml"))
        for path in pattern:
            if path not in candidates:
                candidates.append(path)

    if not candidates:
        return None, []

    selected = candidates[0]
    ignored = candidates[1:]
    return selected, ignored


def _build_specs(
    real: Sequence[str],
    stubs: Sequence[str],
    selected: set[str] | None,
    run_real: bool,
    run_stubs: bool,
    *,
    real_authoritative: bool = False,
    stub_authoritative: bool = False,
) -> List[ConnectorSpec]:
    specs: List[ConnectorSpec] = []
    if run_real:
        authoritative_real = real_authoritative and bool(real)
        for filename in real:
            if selected and filename not in selected:
                continue
            specs.append(
                _create_spec(
                    filename,
                    "real",
                    origin="real_list",
                    authoritatively_selected=authoritative_real,
                )
            )
    if run_stubs:
        authoritative_stubs = stub_authoritative and bool(stubs)
        for filename in stubs:
            if selected and filename not in selected:
                continue
            specs.append(
                _create_spec(
                    filename,
                    "stub",
                    origin="stub_list",
                    authoritatively_selected=authoritative_stubs,
                )
            )
    return specs


def _create_spec(
    filename: str,
    kind: str,
    *,
    origin: str = "config",
    authoritatively_selected: bool = False,
) -> ConnectorSpec:
    path = ROOT / filename
    meta = SUMMARY_TARGETS.get(filename, {})
    default_filename = CONNECTOR_OUTPUTS.get(filename)
    output_path: Optional[Path] = None
    if default_filename:
        default_path = STAGING / default_filename
        output_path = resolve_output_path(default_path)
    summary = _safe_summary(filename, output_path)
    skip_reason = None
    ci_gate_reason = _should_skip(filename)
    if not path.exists():
        skip_reason = f"missing: {filename}"
    metadata: Dict[str, str] = {}
    if output_path:
        metadata["output_path"] = str(output_path)
    if default_filename:
        metadata["default_filename"] = default_filename
    canonical_name = ff.norm(filename)
    explicit_config = meta.get("config") if isinstance(meta, dict) else None
    explicit_path = explicit_config if isinstance(explicit_config, Path) else None
    config_path, ignored = _discover_config_path(filename, canonical_name, explicit_path)
    if ignored:
        logging.getLogger(__name__).warning(
            "::warning multiple config files found for %s; selected %s",
            filename,
            str(config_path) if config_path else "<none>",
            extra={
                "event": "config_discovery",
                "connector": canonical_name,
                "selected": str(config_path) if config_path else None,
                "ignored": [str(path) for path in ignored],
            },
        )
    cfg = _load_yaml(config_path)
    return ConnectorSpec(
        filename=filename,
        path=path,
        kind=kind,
        origin=origin,
        authoritatively_selected=authoritatively_selected,
        output_path=output_path,
        summary=summary,
        skip_reason=skip_reason,
        metadata=metadata,
        config_path=config_path if isinstance(config_path, Path) else None,
        config=cfg if isinstance(cfg, dict) else {},
        canonical_name=canonical_name,
        ci_gate_reason=ci_gate_reason,
    )


def _invoke_connector(path: Path, *, logger: logging.Logger | logging.LoggerAdapter | None = None) -> None:
    repo_root = _repo_root()
    module = _module_name_from_path(path)
    log = logger if logger is not None else logging.getLogger(__name__)
    log.info(
        "launching connector",
        extra={"event": "launch", "module": module, "cwd": str(repo_root)},
    )
    cmd = [sys.executable, "-m", module]
    try:
        proc = subprocess.run(cmd, cwd=repo_root)
    except OSError as exc:  # pragma: no cover - defensive
        log.error(
            "failed to start connector",
            extra={"event": "launch_error", "module": module, "error": str(exc)},
        )
        raise RuntimeError(f"{module} failed to start: {exc}") from exc
    if proc.returncode != 0:
        log.error(
            "connector exited with non-zero status",
            extra={
                "event": "launch_failed",
                "module": module,
                "returncode": proc.returncode,
            },
        )
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def _with_attempt_logger(
    base: logging.LoggerAdapter | logging.Logger, attempt: int
) -> logging.LoggerAdapter:
    if isinstance(base, logging.LoggerAdapter):
        extra = dict(base.extra)
        logger = base.logger
    else:
        extra = {}
        logger = base
    extra["attempt"] = attempt
    return logging.LoggerAdapter(logger, extra)


def _render_summary_table(rows: List[Dict[str, object]]) -> str:
    headers = [
        "Connector",
        "Mode",
        "Status",
        "Attempts",
        "Rows",
        "Duration(ms)",
        "Reason",
    ]
    table_rows = [
        [
            row.get("name", ""),
            row.get("mode", ""),
            row.get("status", ""),
            str(row.get("attempts", "")),
            str(row.get("rows", "")),
            str(row.get("duration_ms", "")),
            row.get("reason") or row.get("notes") or "",
        ]
        for row in rows
    ]
    if not table_rows:
        table_rows = [["-", "-", "-", "0", "0", "0", ""]]
    columns = list(zip(headers, *table_rows))
    widths = [max(len(str(value)) for value in column) for column in columns]
    lines = []
    header_line = " | ".join(header.ljust(width) for header, width in zip(headers, widths))
    lines.append(header_line)
    lines.append("-+-".join("-" * width for width in widths))
    for row in table_rows:
        lines.append(" | ".join(str(value).ljust(width) for value, width in zip(row, widths)))
    return "\n".join(lines)


def _render_output_overview(rows: List[Dict[str, object]]) -> str:
    headers = ["Output Path", "Rows", "Status"]
    table_rows: List[List[str]] = []
    for row in rows:
        path = row.get("output_path")
        if not path:
            continue
        table_rows.append(
            [
                str(path),
                str(row.get("rows", "")),
                str(row.get("status", "")),
            ]
        )
    if not table_rows:
        return ""
    columns = list(zip(headers, *table_rows))
    widths = [max(len(str(value)) for value in column) for column in columns]
    lines = []
    lines.append(" | ".join(header.ljust(width) for header, width in zip(headers, widths)))
    lines.append("-+-".join("-" * width for width in widths))
    for row in table_rows:
        lines.append(" | ".join(str(value).ljust(width) for value, width in zip(row, widths)))
    return "\n".join(lines)


def _rows_written(before: int, after: int) -> int:
    return max(0, after - before)


def _normalise_exit_status(status: str) -> str:
    status_lower = (status or "").lower()
    if status_lower.startswith("ok"):
        return "ok"
    if status_lower in {"error", "skipped"}:
        return status_lower
    if status_lower == "warning":
        return "ok"
    if not status_lower:
        return "skipped"
    return status_lower


def _format_skip_reason(reason: str) -> str:
    formatted = reason.strip().lower()
    if formatted == "disabled: config":
        return "config disables"
    if formatted == "disabled: ci":
        return "CI disables"
    return reason.strip()


def _describe_exit_decision(results: List[dict], exit_code: int) -> str:
    statuses = [str(entry.get("status") or "").lower() for entry in results]
    reasons = [str(entry.get("reason") or "") for entry in results if entry.get("reason")]
    unique_reasons = [
        _format_skip_reason(reason)
        for reason in sorted({reason for reason in reasons if reason})
    ]
    reason_text = ", ".join(unique_reasons)
    if exit_code == 0:
        if any(status == "ok" for status in statuses):
            return "at least one connector ran ⇒ success"
        if statuses and all(status == "skipped" for status in statuses):
            if reason_text:
                return f"all skipped due to {reason_text} ⇒ success"
            return "all connectors skipped for benign reasons ⇒ success"
        return "success"
    if any(status == "error" for status in statuses):
        return "one or more connectors failed ⇒ exit 1"
    if statuses and all(status == "skipped" for status in statuses):
        if reason_text:
            return f"all skipped due to {reason_text} ⇒ exit 1"
        return "all connectors skipped ⇒ exit 1"
    if not statuses:
        return "no connectors evaluated ⇒ exit 1"
    return "no connectors ran ⇒ exit 1"


def _resolve_enablement(
    spec: ConnectorSpec,
    *,
    forced_by_env: bool = False,
    forced_by_only: bool = False,
    forced_by_pattern: bool = False,
) -> EnableDecision:
    cfg = spec.config if isinstance(spec.config, dict) else {}
    config_enabled, has_flag = _coerce_config_flag(cfg)

    forced_sources: list[str] = []
    if forced_by_env:
        forced_sources.append("env")
    if forced_by_only:
        forced_sources.append("only")
    if forced_by_pattern:
        forced_sources.append("pattern")
    forced = bool(forced_sources)
    authoritative = is_authoritatively_selected(spec)
    requires_secret = spec.kind == "real"
    secrets_ok = True
    secret_reason: Optional[str] = None
    if requires_secret:
        secrets_ok, secret_reason = _check_secret_gate(spec.filename)

    if spec.skip_reason:
        return EnableDecision(
            should_run=False,
            gated_by="preexisting_skip",
            forced_sources=tuple(forced_sources),
            config_enabled=config_enabled,
            has_config_flag=has_flag,
            applied_skip_reason=None,
            ci_gate_reason=spec.ci_gate_reason,
        )

    if authoritative and not spec.ci_gate_reason:
        if requires_secret and not secrets_ok and not forced:
            return EnableDecision(
                should_run=False,
                gated_by="secret",
                forced_sources=tuple(forced_sources),
                config_enabled=config_enabled,
                has_config_flag=has_flag,
                applied_skip_reason=secret_reason,
                ci_gate_reason=spec.ci_gate_reason,
            )
        return EnableDecision(
            should_run=True,
            gated_by="selected:list",
            forced_sources=tuple(forced_sources),
            config_enabled=config_enabled,
            has_config_flag=has_flag,
            applied_skip_reason=None,
            ci_gate_reason=spec.ci_gate_reason,
        )

    if forced:
        return EnableDecision(
            should_run=True,
            gated_by="forced:" + "+".join(forced_sources),
            forced_sources=tuple(forced_sources),
            config_enabled=config_enabled,
            has_config_flag=has_flag,
            applied_skip_reason=None,
            ci_gate_reason=spec.ci_gate_reason,
        )

    if config_enabled and spec.ci_gate_reason:
        return EnableDecision(
            should_run=False,
            gated_by="ci_gate",
            forced_sources=tuple(forced_sources),
            config_enabled=config_enabled,
            has_config_flag=has_flag,
            applied_skip_reason=spec.ci_gate_reason,
            ci_gate_reason=spec.ci_gate_reason,
        )

    if not config_enabled:
        return EnableDecision(
            should_run=False,
            gated_by="config",
            forced_sources=tuple(forced_sources),
            config_enabled=False,
            has_config_flag=has_flag,
            applied_skip_reason="disabled: config",
            ci_gate_reason=spec.ci_gate_reason,
        )

    if spec.ci_gate_reason:
        return EnableDecision(
            should_run=False,
            gated_by="ci_gate",
            forced_sources=tuple(forced_sources),
            config_enabled=config_enabled,
            has_config_flag=has_flag,
            applied_skip_reason=spec.ci_gate_reason,
            ci_gate_reason=spec.ci_gate_reason,
        )

    if requires_secret and not secrets_ok and not forced:
        return EnableDecision(
            should_run=False,
            gated_by="secret",
            forced_sources=tuple(forced_sources),
            config_enabled=config_enabled,
            has_config_flag=has_flag,
            applied_skip_reason=secret_reason,
            ci_gate_reason=spec.ci_gate_reason,
        )

    return EnableDecision(
        should_run=True,
        gated_by="config",
        forced_sources=tuple(forced_sources),
        config_enabled=config_enabled,
        has_config_flag=has_flag,
        applied_skip_reason=None,
        ci_gate_reason=spec.ci_gate_reason,
    )


def _run_connector(spec: ConnectorSpec, logger: logging.LoggerAdapter) -> Dict[str, object]:
    start = time.perf_counter()
    rows_before, method_before = _rows_and_method(spec.output_path)
    cfg_enabled, has_flag = _coerce_config_flag(spec.config if isinstance(spec.config, dict) else {})
    if has_flag and not cfg_enabled:
        logger.info(
            "%s: disabled in config (override active).",
            spec.name,
            extra={"event": "config_disabled", "connector": spec.name},
        )
    _invoke_connector(spec.path, logger=logger)
    rows_after, method_after = _rows_and_method(spec.output_path)
    duration_ms = int((time.perf_counter() - start) * 1000)
    rows_written = _rows_written(rows_before, rows_after)
    logger.info(
        "completed",
        extra={
            "event": "completed",
            "duration_ms": duration_ms,
            "rows_written": rows_written,
            "rows_total": rows_after,
            "rows_method_before": method_before,
            "rows_method_after": method_after,
        },
    )
    return {
        "status": "ok",
        "rows": rows_after,
        "duration_ms": duration_ms,
        "rows_method": method_after,
        "rows_method_before": method_before,
        "rows_before": rows_before,
        "rows_after": rows_after,
        "rows_written": rows_written,
    }


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="treat connector/stub failures as fatal (non-zero exit)",
    )
    parser.add_argument(
        "--mode",
        choices=("real", "stubs", "all"),
        default=None,
        help="override RESOLVER_INGESTION_MODE",
    )
    parser.add_argument(
        "--run-stubs",
        type=int,
        choices=(0, 1),
        default=None,
        help="force running stub connectors (1) or skip them (0)",
    )
    parser.add_argument("--retries", type=int, default=2, help="number of retries per connector")
    parser.add_argument("--retry-base", type=float, default=1.0, help="initial retry delay in seconds")
    parser.add_argument("--retry-max", type=float, default=30.0, help="maximum retry delay in seconds")
    parser.add_argument(
        "--retry-no-jitter",
        action="store_true",
        help="disable jitter for retry backoff",
    )
    parser.add_argument(
        "--connector",
        action="append",
        default=[],
        help="limit run to specific connector(s) (filename without .py or with)",
    )
    parser.add_argument(
        "--only",
        default=None,
        help=(
            "run a single connector by name (matches config/file stem); "
            "combine with RESOLVER_FORCE_ENABLE=<name> to override the enabled flag"
        ),
    )
    parser.add_argument(
        "--pattern",
        default=None,
        help=(
            "filter connectors by case-insensitive glob/regex/substring before "
            "enablement checks"
        ),
    )
    parser.add_argument(
        "--log-format",
        choices=("plain", "json"),
        default=None,
        help="console log format (defaults to RUNNER_LOG_FORMAT)",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help="log level (defaults to RUNNER_LOG_LEVEL or INFO)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "enable fast smoke mode with relaxed error handling and smaller"
            " fetch windows"
        ),
    )
    if argv is None:
        return parser.parse_args()
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    root_input = sys.argv if argv is None else ['<custom>'] + list(argv)
    requested = {_normalise_name(name) for name in args.connector if name}
    selected: Optional[set[str]] = requested or None
    only_target = ff.norm(args.only) if args.only else None
    pattern_text = args.pattern

    smoke_mode = bool(args.smoke)
    fail_on_stub_error = FAIL_ON_STUB_ERROR

    ingestion_mode = (args.mode or INGESTION_MODE).strip().lower()
    include_stubs = INCLUDE_STUBS if args.run_stubs is None else bool(args.run_stubs)

    log_dir_env = os.environ.get("RUNNER_LOG_DIR")
    if log_dir_env:
        effective_log_dir = Path(log_dir_env).expanduser()
    else:
        effective_log_dir = LOGS_DIR
    os.makedirs(effective_log_dir, exist_ok=True)

    run_id = dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    effective_level = args.log_level
    if RESOLVER_DEBUG and not effective_level:
        effective_level = "DEBUG"
    root = init_logger(
        run_id,
        level=effective_level,
        fmt=args.log_format,
        log_dir=effective_log_dir,
    )
    if RESOLVER_DEBUG:
        logging.getLogger().setLevel(logging.DEBUG)
        root.setLevel(logging.DEBUG)
    if smoke_mode:
        applied_defaults = _apply_smoke_env_defaults(root)
        fail_on_stub_error = bool(
            int(os.getenv("RESOLVER_FAIL_ON_STUB_ERROR", "0") or 0)
        )
        root.info(
            "smoke mode enabled",
            extra={
                "event": "smoke_mode",
                "defaults_applied": bool(applied_defaults),
                "overrides": applied_defaults,
            },
        )
    root.info(
        "initialised logging",
        extra={
            "event": "logging_setup",
            "log_dir": str(effective_log_dir),
            "run_id": run_id,
        },
    )
    resolved_output_dir = resolve_staging_dir(STAGING)
    if "RESOLVER_OUTPUT_DIR" not in os.environ:
        os.environ["RESOLVER_OUTPUT_DIR"] = str(resolved_output_dir)
    else:
        resolved_output_dir = Path(os.environ["RESOLVER_OUTPUT_DIR"]).expanduser()
    period_label = resolve_period_label()
    root.info(
        "resolved staging output",
        extra={
            "event": "staging_output",
            "path": str(resolved_output_dir),
            "period": period_label,
        },
    )
    log_env_summary(root)
    root.info(
        "parsed arguments",
        extra={
            "event": "args",
            "connector_args": list(args.connector),
            "raw_argv": root_input[1:],
            "mode": args.mode,
            "run_stubs_arg": args.run_stubs,
            "only": args.only,
            "pattern": args.pattern,
        },
    )

    warnings.simplefilter("default")

    def _warn_to_log(message, category, filename, lineno, file=None, line=None):  # type: ignore[override]
        root.warning(
            f"{category.__name__}: {message}",
            extra={"event": "warning", "filename": filename, "lineno": lineno},
        )

    warnings.showwarning = _warn_to_log  # type: ignore[assignment]

    if ingestion_mode and ingestion_mode not in {"real", "stubs", "all"}:
        root.error(
            "Unknown RESOLVER_INGESTION_MODE=%s; expected one of real|stubs|all",
            ingestion_mode,
            extra={"event": "config_error"},
        )
        return 0

    real_list = list(REAL)
    stub_list = list(STUBS)
    real_set = set(real_list)
    stub_set = set(stub_list)

    strict_mode = bool(args.strict)
    real_mode_selected = ingestion_mode == "real"
    stub_mode_selected = ingestion_mode == "stubs"

    if real_mode_selected:
        strict_mode = True

    run_real = True
    run_stubs = include_stubs
    if ingestion_mode:
        run_real = ingestion_mode in {"real", "all"}
        run_stubs = ingestion_mode in {"stubs", "all"}
    if FORCE_DTM_STUB:
        run_stubs = True

    if selected is not None:
        unknown = selected - real_set - stub_set
        if unknown:
            for name in sorted(unknown):
                root.warning("Requested connector %s is unknown", name, extra={"event": "unknown_connector"})
            selected -= unknown
        if not selected:
            root.info("No known connectors requested; exiting", extra={"event": "no_connectors"})
            return 0
        run_real = any(name in real_set for name in selected)
        run_stubs = any(name in stub_set for name in selected)

    specs = _build_specs(
        real_list,
        stub_list,
        selected,
        run_real,
        run_stubs,
        real_authoritative=real_mode_selected,
        stub_authoritative=stub_mode_selected,
    )
    root.info(
        "planning run",
        extra={
            "event": "plan",
            "requested": sorted(requested),
            "run_real": run_real,
            "run_stubs": run_stubs,
            "count": len(specs),
            "only": only_target,
            "pattern": pattern_text,
        },
    )
    if requested:
        specs = [spec for spec in specs if spec.filename in requested]
    if only_target:
        matched_specs: list[ConnectorSpec] = []
        for spec in specs:
            if spec.canonical_name == only_target:
                spec.selected_by_only = True
                matched_specs.append(spec)
        specs = matched_specs
        if not specs:
            root.error(
                "No connector matched --only %s",
                args.only,
                extra={"event": "no_only_match", "only": args.only},
            )
            return 1
    if pattern_text:
        specs = _filter_by_pattern(specs, pattern_text)
        if not specs:
            root.info(
                "No connectors matched pattern '%s'",
                pattern_text,
                extra={"event": "no_pattern_match", "pattern": pattern_text},
            )
            return 0
    if not specs:
        root.info("No connectors to run", extra={"event": "no_connectors"})
        return 0

    force_raw = (os.getenv("RESOLVER_FORCE_ENABLE", "") or "").strip()
    raw_force_entries = {entry.strip() for entry in force_raw.split(",") if entry.strip()}
    normalised_force = {ff.norm(entry) for entry in raw_force_entries if ff.norm(entry)}
    checked_specs: List[ConnectorSpec] = []
    for spec in specs:
        name_for_flags = spec.canonical_name or ff.norm(spec.name)
        filename_stem = Path(spec.filename).stem
        forced_by_env = (
            name_for_flags in normalised_force
            or spec.name in raw_force_entries
            or filename_stem in raw_force_entries
            or spec.filename in raw_force_entries
        )
        decision = _resolve_enablement(
            spec,
            forced_by_env=forced_by_env,
            forced_by_only=spec.selected_by_only,
            forced_by_pattern=spec.matched_by_pattern,
        )
        spec.enable_decision = decision
        if decision.applied_skip_reason and not spec.skip_reason:
            spec.skip_reason = decision.applied_skip_reason
        should_run = decision.should_run and not spec.skip_reason
        config_path_text = str(spec.config_path) if spec.config_path else "<none>"
        decision_text = "run" if should_run else "skip"
        forced_label = "env" if forced_by_env else "none"
        config_enabled_text = "yes" if decision.config_enabled else "no"
        origin_text = spec.origin or "config"
        root.info(
            (
                "connector=%s config_path=%s decision=%s gated_by=%s "
                "config_enabled=%s forced_by=%s origin=%s"
            ),
            name_for_flags,
            config_path_text,
            decision_text,
            decision.gated_by,
            config_enabled_text,
            forced_label,
            origin_text,
            extra={
                "event": "enable_check",
                "connector": name_for_flags,
                "decision": decision_text,
                "gated_by": decision.gated_by,
                "config_enabled": decision.config_enabled,
                "has_config_flag": decision.has_config_flag,
                "forced_sources": list(decision.forced_sources),
                "forced": bool(decision.forced_sources),
                "forced_by_env": forced_by_env,
                "selected_by_only": spec.selected_by_only,
                "matched_by_pattern": spec.matched_by_pattern,
                "ci_gate_reason": decision.ci_gate_reason,
                "config_path": str(spec.config_path) if spec.config_path else None,
                "skip_reason": redact(spec.skip_reason) if spec.skip_reason else None,
                "applied_skip_reason": decision.applied_skip_reason,
                "origin": spec.origin,
                "authoritatively_selected": is_authoritatively_selected(spec),
            },
        )
        legacy_gated_by = "forced_by_env" if forced_by_env else decision.gated_by
        root.info(
            "connector=%s config_path=%s enable=%s gated_by=%s",
            name_for_flags,
            config_path_text,
            "True" if should_run else "False",
            legacy_gated_by,
        )
        checked_specs.append(spec)

    specs = checked_specs

    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        DIAGNOSTICS_REPORT.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass

    connectors_summary: List[Dict[str, object]] = []
    exit_policy_inputs: List[dict] = []
    had_error = False
    total_start = time.perf_counter()

    retries = max(0, args.retries)
    retry_base = float(args.retry_base)
    retry_max = float(args.retry_max)
    if smoke_mode:
        retries = max(retries, 2)
        retry_base = max(retry_base, 5.0)

    for spec in specs:
        child = child_logger(spec.name)
        handler = attach_connector_handler(child, spec.name)
        diagnostics_ctx = diagnostics_start_run(spec.name, "stub" if spec.kind == "stub" else "real")
        diag_counts: Dict[str, int] = {"fetched": 0, "normalized": 0, "written": 0}
        diag_http: Dict[str, object] = {}
        diag_coverage: Optional[Dict[str, object]] = None
        diag_samples: Optional[Dict[str, object]] = None
        diag_extras: Dict[str, object] = {}
        attempts = 0
        rows = 0
        status = "skipped"
        notes: Optional[str] = None
        result: Dict[str, object] = {}
        connector_start = time.perf_counter()
        duration_ms = 0
        try:
            if spec.skip_reason:
                child.warning(
                    "skipped",
                    extra={"event": "skipped", "reason": redact(spec.skip_reason)},
                )
                notes = spec.skip_reason
                status = "skipped"
                attempts = 0
                rows = 0
                diag_extras["skip_reason"] = spec.skip_reason
            else:
                def _attempt(attempt: int) -> Dict[str, object]:
                    nonlocal attempts
                    attempts = max(attempts, attempt)
                    attempt_logger = _with_attempt_logger(child, attempt)
                    start_extra = {
                        "event": "start",
                        "attempt": attempt,
                        "kind": spec.kind,
                        "path": str(spec.path),
                    }
                    for key, value in spec.metadata.items():
                        start_extra[key] = redact(value)
                    if spec.summary:
                        start_extra["summary"] = redact(spec.summary)
                    attempt_logger.info("starting", extra=start_extra)
                    return _run_connector(spec, attempt_logger)

                def _should_retry(exc: BaseException) -> bool:
                    exit_code: int | None = None
                    stderr_text = ""
                    stdout_text = ""
                    if isinstance(exc, subprocess.CalledProcessError):
                        exit_code = exc.returncode
                        try:
                            if getattr(exc, "stderr", None):
                                stderr_text = _coerce_process_stream(exc.stderr)
                            if getattr(exc, "stdout", None):
                                stdout_text = _coerce_process_stream(exc.stdout)
                        except Exception:  # noqa: BLE001
                            stderr_text = ""
                            stdout_text = ""
                    return _is_retryable_exception(
                        exc,
                        exit_code=exit_code,
                        stderr=stderr_text or None,
                        stdout=stdout_text or None,
                    )

                result = retry_call(
                    _attempt,
                    retries=retries,
                    base_delay=retry_base,
                    max_delay=retry_max,
                    jitter=not args.retry_no_jitter,
                    logger=child,
                    connector=spec.name,
                    is_retryable=_should_retry,
                )
                rows = int(result.get("rows", 0))
                rows_method = str(result.get("rows_method") or "")
                duration_ms = int((time.perf_counter() - connector_start) * 1000)
                rows_before = _coerce_int_safe(result.get("rows_before"))
                rows_after = _coerce_int_safe(result.get("rows_after") or rows)
                rows_written = _coerce_int_safe(result.get("rows_written"))
                if rows_written == 0 and rows_after >= 0:
                    rows_written = max(0, rows_after - rows_before)
                diag_counts = {
                    "fetched": _coerce_int_safe(
                        result.get("fetched_rows")
                        or result.get("rows_fetched")
                        or result.get("rows")
                        or rows
                    ),
                    "normalized": _coerce_int_safe(result.get("normalized_rows") or result.get("rows") or rows),
                    "written": rows_written,
                }
                if rows_method:
                    diag_extras["rows_method"] = rows_method
                if result.get("rows_method_before"):
                    diag_extras["rows_method_before"] = result.get("rows_method_before")
                diag_extras["rows_total"] = rows_after
                diag_extras["rows_before"] = rows_before
                diag_extras["rows_written"] = rows_written
                maybe_http = result.get("http")
                if isinstance(maybe_http, Mapping):
                    diag_http = dict(maybe_http)
                diag_coverage, diag_samples = _collect_output_samples(spec.output_path)
                if spec.output_path and spec.output_path.exists() and rows == 0:
                    status = "ok-empty"
                    notes = "header-only"
                else:
                    status = "ok"
                if rows_method in {"recount", "manifest+verified"}:
                    method_note = f"rows:{rows_method}"
                    notes = f"{notes}; {method_note}" if notes else method_note
                child.info(
                    "finished",
                    extra={
                        "event": "finished",
                        "status": status,
                        "rows": rows,
                        "attempts": attempts,
                        "duration_ms": duration_ms,
                        "rows_method": rows_method or None,
                        "notes": redact(notes) if notes else None,
                    },
                )
        except Exception as exc:  # noqa: BLE001
            duration_ms = int((time.perf_counter() - connector_start) * 1000)
            status = "error"
            notes = redact(str(exc))
            rows = 0
            if attempts == 0:
                attempts = 1
            warn_only = False
            if smoke_mode:
                try:
                    warn_only = _should_retry(exc)
                except Exception:  # noqa: BLE001
                    warn_only = False
            log_extra = {
                "event": "error",
                "rows": rows,
                "attempts": attempts,
                "duration_ms": duration_ms,
            }
            diag_extras["exception_type"] = exc.__class__.__name__
            if warn_only:
                status = "warning"
                child.warning("failed (smoke warning)", exc_info=exc, extra=log_extra)
                if notes:
                    notes = f"smoke-warning: {notes}"
            else:
                child.error("failed", exc_info=exc, extra=log_extra)
                had_error = True
        finally:
            if spec.enable_decision and "env" in spec.enable_decision.forced_sources:
                if notes:
                    if "forced_by_env" not in notes:
                        notes = f"{notes}; forced_by_env"
                else:
                    notes = "forced_by_env"
            if is_authoritatively_selected(spec):
                if notes:
                    if "selected:list" not in notes:
                        notes = f"{notes}; selected:list"
                else:
                    notes = "selected:list"
            if diag_coverage is None and spec.output_path:
                diag_coverage, diag_samples = _collect_output_samples(spec.output_path)
            diag_http_payload = dict(diag_http)
            diag_http_payload["retries"] = max(0, attempts - 1)
            diag_extras.setdefault("duration_ms", duration_ms)
            diag_extras["attempts"] = attempts
            diag_extras["status_raw"] = status
            summary_notes = redact(notes) if notes else None
            diagnostic_reason = summary_notes or spec.skip_reason or None
            diagnostics_result = diagnostics_finalize_run(
                diagnostics_ctx,
                status=status,
                reason=diagnostic_reason,
                http=diag_http_payload,
                counts=diag_counts,
                coverage=diag_coverage,
                samples=diag_samples,
                extras={key: value for key, value in diag_extras.items() if value is not None},
            )
            diagnostics_append_jsonl(DIAGNOSTICS_REPORT, diagnostics_result)
            detach_connector_handler(child, handler)
            root.info(
                "connector summary",
                extra={
                    "event": "connector_summary",
                    "connector_name": spec.name,
                    "status": status,
                    "attempts": attempts,
                    "duration_ms": duration_ms,
                    "notes": summary_notes,
                    "reason": summary_notes,
                },
            )
            mode = "skipped" if status == "skipped" else spec.kind
            reason_text = summary_notes or "-"
            summary_line = (
                f"{spec.name}, mode={mode}, status={status}, attempts={attempts}, "
                f"duration_ms={duration_ms}, reason={reason_text}"
            )
            print(summary_line)
            output_path_text: Optional[str]
            if spec.output_path:
                try:
                    output_path_text = str(spec.output_path.resolve())
                except OSError:
                    output_path_text = str(spec.output_path)
            else:
                output_path_text = None
            connectors_summary.append(
                {
                    "name": spec.name,
                    "status": status,
                    "attempts": attempts,
                    "rows": rows,
                    "duration_ms": duration_ms,
                    "notes": summary_notes,
                    "kind": spec.kind,
                    "rows_method": result.get("rows_method") if result else None,
                    "output_path": output_path_text,
                    "mode": mode,
                    "reason": summary_notes,
                }
            )
            normalised_status = _normalise_exit_status(status)
            exit_reason = None
            if normalised_status == "skipped":
                exit_reason = spec.skip_reason or notes
            exit_policy_inputs.append({"status": normalised_status, "reason": exit_reason})

    total_duration_ms = int((time.perf_counter() - total_start) * 1000)
    table = _render_summary_table(connectors_summary)
    root.info("Connector summary\n%s", table, extra={"event": "summary_table"})
    output_table = _render_output_overview(connectors_summary)
    if output_table:
        root.info("Output files\n%s", output_table, extra={"event": "output_table"})
    root.info(
        "run complete",
        extra={
            "event": "run_summary",
            "run_id": run_id,
            "connectors": [
                {key: value for key, value in entry.items() if key != "kind"}
                for entry in connectors_summary
            ],
            "total_duration_ms": total_duration_ms,
        },
    )

    real_failures = sum(
        1
        for entry in connectors_summary
        if entry.get("kind") == "real" and entry.get("status") == "error"
    )
    stub_failures = sum(
        1
        for entry in connectors_summary
        if entry.get("kind") == "stub" and entry.get("status") == "error"
    )
    status_counts = Counter(entry.get("status") for entry in connectors_summary)
    normalised_counts = Counter(result.get("status") for result in exit_policy_inputs)

    root.info(
        "status counts",
        extra={
            "event": "connector_status_counts",
            "status_counts": dict(status_counts),
            "normalised_counts": dict(normalised_counts),
        },
    )

    policy_exit_code = compute_exit_code(exit_policy_inputs)
    policy_reason = _describe_exit_decision(exit_policy_inputs, policy_exit_code)
    root.info(
        "exit policy result",
        extra={
            "event": "exit_policy",
            "exit_code": policy_exit_code,
            "reason": policy_reason,
            "status_counts": dict(normalised_counts),
        },
    )

    exit_code = policy_exit_code
    exit_reason_text = policy_reason

    if strict_mode and had_error:
        exit_code = 1
        exit_reason_text = "strict mode: connector error forces failure"
    elif run_real and not run_stubs and real_failures:
        exit_code = 1
        exit_reason_text = "real connector failure forces non-zero exit"
    elif fail_on_stub_error and stub_failures:
        exit_code = 1
        exit_reason_text = "stub failure with FAIL_ON_STUB_ERROR=1"

    root.info(
        "final exit decision",
        extra={
            "event": "exit_decision",
            "exit_code": exit_code,
            "exit_reason": exit_reason_text,
            "policy_exit_code": policy_exit_code,
            "policy_reason": policy_reason,
            "status_counts": dict(status_counts),
            "normalised_counts": dict(normalised_counts),
            "strict_mode": strict_mode,
            "had_error": had_error,
            "real_failures": real_failures,
            "stub_failures": stub_failures,
            "run_real": run_real,
            "run_stubs": run_stubs,
        },
    )

    summary_counts_text = ", ".join(
        f"{status}={count}" for status, count in sorted(status_counts.items()) if status
    )
    if not summary_counts_text:
        summary_counts_text = "none"
    print(f"Connector totals: {summary_counts_text} | exit={exit_code} ({exit_reason_text})")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
