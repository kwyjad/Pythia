# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Sibyl's fetch_url reaches the public web and nothing else.

The model chooses every URL Sibyl fetches, and a page it read can steer that
choice. These tests hold the fetch to http(s), public addresses on every
redirect hop, and a bounded body. No test touches the network: name
resolution and the transport are both replaced.
"""

from __future__ import annotations

from datetime import date

import pytest

import sibyl.tools as sibyl_tools

TODAY = date(2026, 10, 2)


class _Resp:
    def __init__(self, status=200, headers=None, body=b""):
        self.status_code = status
        self.headers = headers or {"Content-Type": "text/plain; charset=utf-8"}
        self._body = body
        self.closed = False

    def iter_content(self, chunk_size=65536):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i:i + chunk_size]

    def close(self):
        self.closed = True


@pytest.fixture
def dns(monkeypatch):
    table = {
        "example.org": ["93.184.216.34"],
        "news.example.org": ["93.184.216.35"],
        "intranet.example.org": ["10.0.0.5"],
        "mixed.example.org": ["93.184.216.34", "127.0.0.1"],
    }

    def resolve(host):
        if host not in table:
            raise OSError("no such host")
        return table[host]

    monkeypatch.setattr(sibyl_tools, "_resolve", resolve)
    return table


@pytest.fixture
def transport(monkeypatch):
    calls = []
    routes = {}

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        return routes[url]()

    monkeypatch.setattr(sibyl_tools.requests, "get", fake_get)
    return calls, routes


@pytest.mark.parametrize(
    "url",
    [
        "http://169.254.169.254/latest/meta-data/",
        "http://127.0.0.1:8000/v1/health",
        "http://[::1]/",
        "http://[::ffff:127.0.0.1]/",
        "http://[::ffff:a9fe:a9fe]/",
        "http://100.64.0.1/",
        "http://10.1.2.3/",
        "http://192.168.0.1/",
        "http://intranet.example.org/",
        "http://mixed.example.org/",
        "file:///etc/passwd",
        "ftp://example.org/x",
        "gopher://example.org/",
        "http:///nohost",
        "http://unknown.invalid/",
    ],
)
def test_unsafe_urls_are_refused_without_a_request(url, dns, transport):
    calls, _ = transport
    result = sibyl_tools.fetch_url(url, TODAY)
    assert result.ok is False
    assert result.error == "unsafe_url"
    assert calls == []


def test_a_public_page_is_fetched_and_its_text_returned(dns, transport):
    calls, routes = transport
    routes["https://example.org/a"] = lambda: _Resp(body=b"floods in the north displaced thousands")
    result = sibyl_tools.fetch_url("https://example.org/a", TODAY)
    assert result.ok is True
    assert "displaced thousands" in result.text
    # Redirects are followed by hand, and the body is streamed.
    assert calls[0][1]["allow_redirects"] is False
    assert calls[0][1]["stream"] is True


def test_a_redirect_to_a_private_address_is_refused(dns, transport):
    calls, routes = transport
    routes["https://example.org/r"] = lambda: _Resp(302, {"Location": "http://169.254.169.254/"})
    result = sibyl_tools.fetch_url("https://example.org/r", TODAY)
    assert result.ok is False
    assert result.error == "unsafe_url"
    assert [c[0] for c in calls] == ["https://example.org/r"]


def test_a_public_redirect_is_followed(dns, transport):
    _, routes = transport
    routes["https://example.org/r"] = lambda: _Resp(301, {"Location": "https://news.example.org/story"})
    routes["https://news.example.org/story"] = lambda: _Resp(body=b"story text")
    result = sibyl_tools.fetch_url("https://example.org/r", TODAY)
    assert result.ok is True
    assert "story text" in result.text


def test_a_redirect_into_a_resolution_source_is_blocked(dns, transport, monkeypatch):
    _, routes = transport
    dns["go.ifrc.org"] = ["93.184.216.36"]
    routes["https://example.org/r"] = lambda: _Resp(302, {"Location": "https://go.ifrc.org/emergencies/1"})
    routes["https://go.ifrc.org/emergencies/1"] = lambda: _Resp(body=b"figures")
    result = sibyl_tools.fetch_url("https://example.org/r", TODAY)
    assert result.ok is False
    assert result.error == "blocked_domain"


def test_a_redirect_loop_is_cut_off(dns, transport):
    _, routes = transport
    routes["https://example.org/loop"] = lambda: _Resp(302, {"Location": "https://example.org/loop"})
    result = sibyl_tools.fetch_url("https://example.org/loop", TODAY)
    assert result.ok is False
    assert result.error == "unsafe_url"


def test_the_body_is_read_no_further_than_the_cap(dns, transport, monkeypatch):
    _, routes = transport
    monkeypatch.setattr(sibyl_tools, "FETCH_URL_MAX_BYTES", 1000)
    big = _Resp(body=b"x" * 50_000)
    routes["https://example.org/big"] = lambda: big
    resp, body, final = sibyl_tools._guarded_get("https://example.org/big")
    assert len(body) == 1000
    assert big.closed is True
