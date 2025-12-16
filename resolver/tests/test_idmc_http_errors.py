# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

import pytest
from requests.exceptions import ConnectTimeout, SSLError

from resolver.ingestion.idmc import http


class _RaiseOnceSession:
    def __init__(self, exc: Exception):
        self._exc = exc
        self.closed = False

    def get(self, *_args, **_kwargs):
        raise self._exc

    def close(self):  # pragma: no cover - defensive cleanup
        self.closed = True


@pytest.mark.parametrize(
    "exc, expected_kind",
    [
        (ConnectTimeout("connect timed out"), "connect_timeout"),
        (SSLError("tls handshake failed"), "ssl_error"),
    ],
)
def test_http_error_kind_classification(monkeypatch, exc, expected_kind):
    """HttpRequestError.kind reflects the classified network failure."""

    fake_session = _RaiseOnceSession(exc)
    monkeypatch.setattr(http.requests, "Session", lambda: fake_session)

    with pytest.raises(http.HttpRequestError) as err:
        http.http_get(
            "https://backend.idmcdb.org/test",
            timeout=(0.1, 0.1),
            retries=0,
            verify=True,
        )

    error = err.value
    assert error.kind == expected_kind
    diagnostics = error.diagnostics
    assert diagnostics["exception_kind"] == expected_kind
    assert diagnostics["exceptions"], "exceptions list should contain attempt diagnostics"
    assert diagnostics["exceptions"][-1]["kind"] == expected_kind
