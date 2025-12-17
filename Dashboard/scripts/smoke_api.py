# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import sys
from typing import Any, Dict, List, Tuple

import requests

from Dashboard.lib import api_client


CHECKS: List[Tuple[str, Dict[str, Any]]] = [
    ("/v1/diagnostics/summary", {}),
    ("/v1/risk_index", {}),
    ("/v1/questions", {"latest_only": True}),
]


def _call(path: str, params: Dict[str, Any]) -> requests.Response:
    token = api_client.get_token()
    url = api_client.build_url(path)
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return requests.get(url, params=params, headers=headers, timeout=15)


def main() -> None:
    failures: List[str] = []
    for path, params in CHECKS:
        try:
            response = _call(path, params)
            response.raise_for_status()
            print(f"[ok] {path}: {len(response.content)} bytes")
        except Exception as exc:  # pragma: no cover - smoke script
            failures.append(path)
            print(f"[fail] {path}: {exc}", file=sys.stderr)

    if failures:
        sys.exit(1)

    print("All smoke checks passed.")


if __name__ == "__main__":
    main()
