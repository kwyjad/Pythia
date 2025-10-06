"""Offline smoke tests for ingestion connectors."""
from __future__ import annotations

import csv
import os
import re
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Iterable, List

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
INGESTION_DIR = REPO_ROOT / "resolver" / "ingestion"
STAGING_DIR = REPO_ROOT / "resolver" / "staging"
LOG_DIR = REPO_ROOT / "logs" / "ingestion_test"
SENTINEL_ENV_PATH = Path(__file__).resolve().parent / "fixtures" / "offline_sentinel.env"

LOG_KEYWORDS = (
    "disabled",
    "placeholder",
    "header-only",
    "pdf disabled",
    "pdf placeholder",
    "pdf header-only",
)
AS_OF_ALIASES = {"as_of", "as_of_date", "as_of_month", "as_of_timestamp", "as_of_iso"}
SOURCE_ALIASES = {"source", "source_url", "source_type", "source_event_id", "publisher"}
DATE_ALIASES = {"date", "month", "as_of", "as_of_date", "publication_date", "download_date", "year"}
EXCLUDE = {"run_all_stubs"}
KNOWN_MIN_HEADERS: Dict[str, Iterable[str]] = {
    "unhcr_odp": [
        "source",
        "source_event_id",
        "as_of_date",
        "country_iso3",
        "metric_name",
        "metric_unit",
        "value",
    ],
    "worldpop": ["iso3", "year", "population", "as_of", "source", "method"],
}

SKIP_ENV_OVERRIDES: Dict[str, Dict[str, str]] = {
    "acled": {"RESOLVER_SKIP_ACLED": "1"},
    "dtm": {"RESOLVER_SKIP_DTM": "1"},
    "emdat": {"RESOLVER_SKIP_EMDAT": "1"},
    "gdacs": {"RESOLVER_SKIP_GDACS": "1"},
    "hdx": {"RESOLVER_SKIP_HDX": "1"},
    "ifrc_go": {"RESOLVER_SKIP_IFRCGO": "1"},
    "ipc": {"RESOLVER_SKIP_IPC": "1"},
    "reliefweb": {"RESOLVER_SKIP_RELIEFWEB": "1"},
    "unhcr": {"RESOLVER_SKIP_UNHCR": "1"},
    "unhcr_odp": {"RESOLVER_SKIP_UNHCR_ODP": "1"},
    "wfp_mvam": {"RESOLVER_SKIP_WFP_MVAM": "1"},
    "who_phe": {"RESOLVER_SKIP_WHO": "1"},
    "worldpop": {"RESOLVER_SKIP_WORLDPOP": "1"},
}

STAGING_FILE_OVERRIDES: Dict[str, str] = {
    "dtm": "dtm_displacement.csv",
    "emdat": "emdat_pa.csv",
    "gdacs": "gdacs_signals.csv",
    "worldpop": "worldpop_denominators.csv",
}

ALLOW_NONZERO_EXIT = {"emdat"}

RELIEFWEB_PDF_REQUIRED_EXACT = {"event_id", "metric", "value"}
RELIEFWEB_PDF_REQUIRED_ALIASES = {
    "as_of": AS_OF_ALIASES,
    "source": SOURCE_ALIASES,
    "country_iso3": {"country_iso3", "iso3"},
    "month": {"month", "month_start"},
    "hazard_type": {"hazard_type", "hazard_code", "hazard_label", "hazard_class"},
}
RELIEFWEB_PDF_STAGING_NAME = "reliefweb_pdf.csv"
RELIEFWEB_PDF_LOG_KEYWORDS = (
    "pdf disabled",
    "pdf placeholder",
    "pdf header-only",
)

def _dataset_name(connector_path: Path) -> str:
    name = connector_path.stem
    if name.endswith("_client"):
        name = name[: -len("_client")]
    return name


def _load_sentinel_env() -> Dict[str, str]:
    if not SENTINEL_ENV_PATH.exists():
        return {}
    env: Dict[str, str] = {}
    for line in SENTINEL_ENV_PATH.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        env[key.strip()] = value.strip()
    return env


SENTINEL_ENV = _load_sentinel_env()
TOKEN_PATTERN = re.compile(r".*(_TOKEN|_KEY|_SECRET)$")


CONNECTOR_PATHS: List[Path] = []
for candidate in sorted(INGESTION_DIR.glob("*_client.py")):
    stem = candidate.stem
    if any(stem.endswith(suffix) for suffix in ("_stub", "_runner", "_util")):
        continue
    dataset = _dataset_name(candidate)
    if dataset in EXCLUDE:
        continue
    CONNECTOR_PATHS.append(candidate)

CONNECTOR_IDS = [_dataset_name(path) for path in CONNECTOR_PATHS]


def _build_child_env(dataset: str) -> Dict[str, str]:
    env = os.environ.copy()
    for key in list(env):
        if TOKEN_PATTERN.match(key):
            env.pop(key, None)
    env.update(SENTINEL_ENV)
    env["ENABLE"] = "0"
    env["RESOLVER_CI"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["RUNNER_LOG_DIR"] = str(LOG_DIR)
    pythonpath = env.get("PYTHONPATH", "")
    paths = [str(REPO_ROOT)]
    if pythonpath:
        paths.append(pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env.update(SKIP_ENV_OVERRIDES.get(dataset, {}))
    if dataset == "reliefweb":
        env.update(
            {
                "RELIEFWEB_ENABLE_PDF": "1",
                "RELIEFWEB_PDF_ALLOW_NETWORK": "0",
                "RELIEFWEB_PDF_ENABLE_OCR": "0",
            }
        )
    return env


def _read_header(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration as exc:  # pragma: no cover - defensive
            raise AssertionError(f"{path} is empty; expected header row") from exc
    return [col.strip().strip("'").strip('"') for col in header]


def _format_missing(dataset: str, missing: Iterable[str], header: Iterable[str]) -> str:
    actual = ", ".join(header)
    expected = ", ".join(missing)
    return f"{dataset}: missing required columns [{expected}] (header: [{actual}])"


def _assert_header(dataset: str, header: List[str]) -> None:
    header_lower = [col.strip().lower() for col in header if col.strip()]
    header_set = set(header_lower)
    if dataset == "reliefweb_pdf":
        missing_exact = [
            col for col in sorted(RELIEFWEB_PDF_REQUIRED_EXACT) if col not in header_set
        ]
        if missing_exact:
            pytest.fail(_format_missing(dataset, missing_exact, header))

        missing_aliases = []
        for column, aliases in RELIEFWEB_PDF_REQUIRED_ALIASES.items():
            lowered_aliases = {alias.lower() for alias in aliases}
            if not (header_set & lowered_aliases):
                missing_aliases.append(column)

        if missing_aliases:
            pytest.fail(_format_missing(dataset, missing_aliases, header))
        return

    if dataset in KNOWN_MIN_HEADERS:
        expected = {col.lower() for col in KNOWN_MIN_HEADERS[dataset]}
        missing = [col for col in expected if col not in header_set]
        if missing:
            pytest.fail(_format_missing(dataset, missing, header))
        return

    required = {"event_id"}
    missing_required = [col for col in required if col not in header_set]
    if missing_required:
        pytest.fail(_format_missing(dataset, missing_required, header))

    if not (header_set & AS_OF_ALIASES):
        pytest.fail(_format_missing(dataset, ["as_of"], header))

    if not (header_set & SOURCE_ALIASES):
        pytest.fail(_format_missing(dataset, ["source"], header))

    has_country_month = "country_iso3" in header_set and (
        "month" in header_set or header_set & DATE_ALIASES
    )
    has_iso_date = "iso3" in header_set and (header_set & DATE_ALIASES)

    if not (has_country_month or has_iso_date):
        pytest.fail(
            f"{dataset}: expected geographic/date columns; got [{', '.join(header)}]"
        )


def _collect_new_logs(start_time: float) -> List[Path]:
    if not LOG_DIR.exists():
        return []
    fresh: List[Path] = []
    for path in LOG_DIR.glob("ingest_*.log"):
        try:
            if path.stat().st_mtime >= start_time:
                fresh.append(path)
        except FileNotFoundError:
            continue
    return sorted(fresh, key=lambda p: p.stat().st_mtime)


def _log_has_keywords(content: str, keywords: Iterable[str] = LOG_KEYWORDS) -> bool:
    lowered = content.lower()
    return any(keyword in lowered for keyword in keywords)


def _search_outputs_for_keywords(
    logs: Iterable[Path], stdout: str, stderr: str, keywords: Iterable[str]
) -> bool:
    for log in logs:
        try:
            content = log.read_text(encoding="utf-8")
        except FileNotFoundError:
            continue
        if _log_has_keywords(content, keywords):
            return True
    combined = f"{stdout}\n{stderr}".strip()
    if combined and _log_has_keywords(combined, keywords):
        return True
    return False


@pytest.mark.parametrize("connector_path", CONNECTOR_PATHS, ids=CONNECTOR_IDS)
def test_ingestion_connector_offline_smoke(connector_path: Path) -> None:
    dataset = _dataset_name(connector_path)
    staging_name = STAGING_FILE_OVERRIDES.get(dataset, f"{dataset}.csv")
    staging_file = STAGING_DIR / staging_name
    staging_file.parent.mkdir(parents=True, exist_ok=True)
    if staging_file.exists():
        staging_file.unlink()

    reliefweb_pdf_file: Path | None = None
    if dataset == "reliefweb":
        reliefweb_pdf_file = STAGING_DIR / RELIEFWEB_PDF_STAGING_NAME
        if reliefweb_pdf_file.exists():
            reliefweb_pdf_file.unlink()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    module_name = f"resolver.ingestion.{connector_path.stem}"
    cmd = [sys.executable, "-m", module_name]
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=_build_child_env(dataset),
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
    )

    stdout_snippet = result.stdout.strip()[-400:]
    stderr_snippet = result.stderr.strip()[-400:]
    if result.returncode != 0:
        if dataset not in ALLOW_NONZERO_EXIT:
            pytest.fail(
                f"{dataset}: exited with {result.returncode}\nSTDOUT:{stdout_snippet}\nSTDERR:{stderr_snippet}"
            )
        warnings.warn(
            f"{dataset}: non-zero exit ({result.returncode}) tolerated for offline smoke; stdout={stdout_snippet} stderr={stderr_snippet}"
        )

    assert staging_file.exists(), f"{dataset}: expected staging file {staging_file}"

    header = _read_header(staging_file)
    _assert_header(dataset, header)

    if dataset == "reliefweb":
        assert (
            reliefweb_pdf_file is not None and reliefweb_pdf_file.exists()
        ), "reliefweb: expected staging file reliefweb_pdf.csv"
        pdf_header = _read_header(reliefweb_pdf_file)
        _assert_header("reliefweb_pdf", pdf_header)

    logs = _collect_new_logs(start_time)
    indicator_found = _search_outputs_for_keywords(
        logs, result.stdout, result.stderr, LOG_KEYWORDS
    )
    if dataset == "reliefweb":
        indicator_found = indicator_found or _search_outputs_for_keywords(
            logs, result.stdout, result.stderr, RELIEFWEB_PDF_LOG_KEYWORDS
        )

    if not indicator_found:
        warnings.warn(
            f"{dataset}: offline run completed without logging a disabled/placeholder/header-only marker"
        )
