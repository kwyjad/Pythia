# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""IDMC reachability probe wrapper for CI diagnostics."""
from __future__ import annotations

import json
import os
from pathlib import Path

from .protocol_probe import ProbeResult, format_markdown_block, probe_host

DEFAULT_HOST = os.getenv("IDMC_PROBE_HOST", "backend.idmcdb.org").strip() or "backend.idmcdb.org"
DEFAULT_PORT = int(os.getenv("IDMC_PROBE_PORT", "443"))
DEFAULT_PATH = os.getenv(
    "IDMC_PROBE_PATH",
    "/data/idus_view_flat?select=id&limit=1",
)
DEFAULT_SCHEME = os.getenv("IDMC_PROBE_SCHEME", "https").strip() or "https"
CONNECT_TIMEOUT = float(os.getenv("IDMC_PROBE_CONNECT_TIMEOUT", "5"))
READ_TIMEOUT = float(os.getenv("IDMC_PROBE_READ_TIMEOUT", "5"))
VERIFY = os.getenv("IDMC_PROBE_VERIFY")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_json(target: Path, payload: ProbeResult) -> None:
    target.write_text(payload.to_json() + "\n", encoding="utf-8")


def _append_summary(target: Path, payload: ProbeResult) -> None:
    block = format_markdown_block("IDMC Reachability", payload)
    with target.open("a", encoding="utf-8") as handle:
        handle.write("\n\n" + block + "\n")


def main() -> int:
    probe = probe_host(
        DEFAULT_HOST,
        port=DEFAULT_PORT,
        scheme=DEFAULT_SCHEME,
        https_path=DEFAULT_PATH,
        connect_timeout=CONNECT_TIMEOUT,
        read_timeout=READ_TIMEOUT,
        verify=VERIFY if VERIFY is None else VERIFY,
    )
    print(probe.to_json())

    base_diag = Path("diagnostics/ingestion")
    _ensure_dir(base_diag)
    idmc_diag = base_diag / "idmc"
    _ensure_dir(idmc_diag)

    _write_json(idmc_diag / "probe.json", probe)
    _append_summary(base_diag / "summary.md", probe)
    return 0


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main())
