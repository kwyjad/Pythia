"""Collect reachability diagnostics for the IDMC connector."""

from __future__ import annotations

import json
import os

from resolver.ingestion.idmc.probe import ProbeOptions, probe_reachability


def main(argv: list[str] | None = None) -> int:
    base_url = os.getenv("IDMC_BASE_URL", "https://backend.idmcdb.org")
    timeout = float(os.getenv("IDMC_REACHABILITY_TIMEOUT", "3"))
    try:
        payload = probe_reachability(ProbeOptions(base_url=base_url, timeout=timeout))
    except Exception as exc:  # pragma: no cover - defensive
        payload = {"error": str(exc), "base_url": base_url}
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
