# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Probe the IDMC Helix (GIDD) CSV export endpoint."""

from __future__ import annotations

import os
import time
from datetime import date
from pathlib import Path
from typing import Iterable, List, Tuple
from urllib.parse import urlencode

try:  # pragma: no cover - optional dependency guard
    import requests
except Exception:  # pragma: no cover - defensive fallback
    requests = None  # type: ignore[assignment]


SUMMARY_PATH = Path("diagnostics/ingestion/summary.md")
DEFAULT_BASE = "https://helix-tools-api.idmcdb.org"
USER_AGENT = "Pythia-IDMC/1.0"
TIMEOUT = 45.0


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _resolve_client_id() -> str | None:
    raw = os.getenv("IDMC_HELIX_CLIENT_ID")
    if raw is None:
        return None
    cleaned = str(raw).strip()
    return cleaned or None


def _base_url() -> str:
    base = os.getenv("IDMC_HELIX_BASE", DEFAULT_BASE)
    return str(base).strip().rstrip("/") or DEFAULT_BASE


def _render_block(lines: Iterable[str]) -> str:
    output = list(lines)
    if not output or output[-1] != "":
        output.append("")
    return "\n".join(output)


def probe() -> List[str]:
    lines = ["## Helix (GIDD) Reachability", ""]
    client_id = _resolve_client_id()
    if client_id is None:
        lines.append("- Skipped: `IDMC_HELIX_CLIENT_ID` not set")
        return lines
    if requests is None:  # pragma: no cover - defensive
        lines.append("- Skipped: `requests` module unavailable")
        return lines

    today = date.today()
    start_year = today.year - 1
    end_year = today.year
    base = _base_url()
    endpoint = f"{base}/external-api/gidd/displacements/displacement-export/"
    params: List[Tuple[str, str]] = [
        ("client_id", client_id),
        ("start_year", str(start_year)),
        ("end_year", str(end_year)),
        ("iso3__in", "FRA"),
        (
            "release_environment",
            str(os.getenv("IDMC_HELIX_ENV", "RELEASE") or "RELEASE").upper(),
        ),
    ]

    sanitized_params = [(key, value) for key, value in params if key != "client_id"]
    lines.append(f"- URL: `{endpoint}`")
    if sanitized_params:
        lines.append(
            f"- Request: `{endpoint}?{urlencode(sanitized_params, doseq=True)}`"
        )

    headers = {"Accept": "text/csv", "User-Agent": USER_AGENT}
    try:
        response = requests.get(endpoint, params=dict(params), headers=headers, timeout=TIMEOUT)
    except requests.RequestException as exc:  # pragma: no cover - network dependent
        lines.append(f"- Error: `{exc.__class__.__name__}`")
        lines.append(f"- Message: `{str(exc)}`")
        return lines

    lines.append(f"- Status: `{response.status_code}`")
    payload = response.content or b""
    lines.append(f"- Bytes: `{len(payload)}`")
    content_length = response.headers.get("Content-Length")
    if content_length:
        lines.append(f"- Content-Length: `{content_length}`")
    if response.ok and payload:
        try:
            snippet = payload.splitlines()[0].decode("utf-8", "ignore")
        except Exception:  # pragma: no cover - defensive
            snippet = ""
        if snippet:
            lines.append(f"- Header sample: `{snippet}`")
    return lines


def append_summary(lines: Iterable[str]) -> None:
    block = _render_block(lines)
    existing = SUMMARY_PATH.read_text(encoding="utf-8") if SUMMARY_PATH.exists() else ""
    if existing and not existing.endswith("\n"):
        existing += "\n"
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(existing + block, encoding="utf-8")


def main() -> int:
    lines = probe()
    lines.insert(1, f"- Checked: `{_now_iso()}`")
    append_summary(lines)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
