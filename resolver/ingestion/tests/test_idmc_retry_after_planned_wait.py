# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Retry-After handling tests for the IDMC HTTP client."""
from __future__ import annotations

import urllib.error
import urllib.request
from typing import Dict

import pytest

from resolver.ingestion.idmc.http import http_get


class _FakeResponse:
    def __init__(self, body: bytes, *, status: int = 200, headers: Dict[str, str] | None = None) -> None:
        self._body = body
        self.status = status
        self.headers = headers or {}

    def read(self, size: int = -1) -> bytes:
        data, self._body = self._body, b""
        return data

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - defensive
        return None

    def getcode(self) -> int:
        return self.status


def test_retry_after_records_planned_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}

    def fake_urlopen(request, timeout: float):
        calls["count"] += 1
        if calls["count"] == 1:
            raise urllib.error.HTTPError(
                request.full_url,
                429,
                "Too Many Requests",
                hdrs={"Retry-After": "2"},
                fp=None,
            )
        return _FakeResponse(b"{}", status=200, headers={})

    monkeypatch.setenv("IDMC_TEST_NO_SLEEP", "1")
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    status, headers, body, diagnostics = http_get("https://example.com/data", retries=1, backoff_s=0.5)

    assert status == 200
    assert body == b"{}"
    assert diagnostics["retries"] == 1
    assert diagnostics["retry_after_s"] == [pytest.approx(2.0, rel=1e-3)]
    # planned sleeps include the retry-after wait and the exponential backoff
    assert diagnostics["planned_sleep_s"][0] == pytest.approx(2.0, rel=1e-3)
    assert diagnostics["planned_sleep_s"][1] == pytest.approx(0.5, rel=1e-3)
    assert diagnostics["backoff_s"] == pytest.approx(0.5, rel=1e-3)
    assert diagnostics["wire_bytes"] == len(b"{}")
    assert diagnostics["body_bytes"] == len(b"{}")

    monkeypatch.delenv("IDMC_TEST_NO_SLEEP", raising=False)
