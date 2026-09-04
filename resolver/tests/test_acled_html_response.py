# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""An ACLED web page is a failure, never an empty result.

ACLED's gateway answers an unauthenticated API call with a Drupal page
titled "Unauthorized", with HTTP 200, even when the request sets
``Accept: application/json``. So a 401, a WAF interstitial and a session
expiry are indistinguishable by status code, and an HTML body already
killed one monthly ingest (2026-09-03). ``acled_auth.parse_json_response``
is now the single error path for every ACLED caller, and these tests pin
that each of them raises a NAMED failure carrying the status, the URL and
the first of the body, rather than returning zero records.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from resolver.ingestion import acled_auth

UNAUTHORIZED_PAGE = (
    "<!DOCTYPE html>\n<html lang=\"en\"><head><title>Unauthorized | ACLED</title>"
    "</head><body><h1>Unauthorized</h1><p>You are not authorized to access this page.</p>"
    "</body></html>"
)


class _Resp:
    def __init__(self, *, status=200, body="", content_type="text/html", url="https://acleddata.com/api/acled/read?x=1"):
        self.status_code = status
        self.text = body
        self.headers = {"Content-Type": content_type}
        self.url = url
        self.history = []

    def json(self):
        return json.loads(self.text)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


# ---------------------------------------------------------------------------
# The parser
# ---------------------------------------------------------------------------

def test_an_html_body_with_a_200_raises_a_named_failure_carrying_status_url_and_body():
    resp = _Resp(status=200, body=UNAUTHORIZED_PAGE, content_type="text/html; charset=UTF-8")
    with pytest.raises(acled_auth.AcledHtmlResponse) as excinfo:
        acled_auth.parse_json_response(resp, what="event read")
    err = excinfo.value
    assert err.status == 200
    assert "acleddata.com/api/acled/read" in err.url
    assert "Unauthorized" in err.snippet
    assert len(err.snippet) <= 200
    assert "status=200" in str(err) and "Unauthorized" in str(err)


def test_html_is_recognised_by_the_body_even_when_the_content_type_lies():
    resp = _Resp(status=200, body=UNAUTHORIZED_PAGE, content_type="application/json")
    with pytest.raises(acled_auth.AcledHtmlResponse):
        acled_auth.parse_json_response(resp, what="event read")


def test_a_non_200_json_refusal_is_described_not_bare():
    resp = _Resp(status=401, body='{"error":"invalid_token"}', content_type="application/json")
    with pytest.raises(acled_auth.AcledResponseError) as excinfo:
        acled_auth.parse_json_response(resp, what="CAST read")
    assert "status=401" in str(excinfo.value)
    assert "invalid_token" in str(excinfo.value)


def test_a_good_json_body_is_returned():
    resp = _Resp(status=200, body='{"status": 200, "data": [1, 2]}', content_type="application/json")
    assert acled_auth.parse_json_response(resp, what="read") == {"status": 200, "data": [1, 2]}


def test_body_shape_tolerates_objects_that_are_not_responses():
    class _Odd:
        status_code = object()
        headers = {}
        text = None

        def json(self):
            return {"data": []}

    assert acled_auth.body_shape(_Odd()) == "other"
    assert acled_auth.parse_json_response(_Odd(), what="read") == {"data": []}


# ---------------------------------------------------------------------------
# The three callers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _no_file_cache(monkeypatch):
    monkeypatch.setenv("ACLED_TOKEN_CACHE_PATH", "")
    acled_auth._CACHE.update({"access_token": None, "refresh_token": None, "expiry": None})
    yield
    acled_auth._CACHE.update({"access_token": None, "refresh_token": None, "expiry": None})


def test_acled_client_fetch_events_raises_on_a_web_page(tmp_path: Path, monkeypatch):
    from resolver.ingestion import acled_client

    diag_root = tmp_path / "diagnostics" / "ingestion"
    monkeypatch.setattr(acled_client, "ACLED_DIAGNOSTICS", diag_root / "acled")
    monkeypatch.setattr(acled_client, "ACLED_RUN_PATH", diag_root / "acled_client" / "run.json")
    monkeypatch.setattr(acled_client.acled_auth, "get_access_token", lambda: "tok")

    class _Session:
        def get(self, url, params=None, headers=None, timeout=None):
            return _Resp(status=200, body=UNAUTHORIZED_PAGE)

    monkeypatch.setattr(acled_client.requests, "Session", lambda: _Session())
    with pytest.raises(acled_auth.AcledHtmlResponse):
        acled_client.fetch_events({"base_url": "https://acleddata.com/api/acled/read"})


def test_acled_client_page_fetcher_raises_on_a_web_page(tmp_path: Path, monkeypatch):
    from resolver.ingestion import acled_client

    diag_root = tmp_path / "diagnostics" / "ingestion"
    monkeypatch.setattr(acled_client, "ACLED_DIAGNOSTICS", diag_root / "acled")
    monkeypatch.setattr(acled_client, "ACLED_RUN_PATH", diag_root / "acled_client" / "run.json")
    monkeypatch.setattr(acled_client.acled_auth, "get_access_token", lambda: "tok")

    class _Session:
        def get(self, url, params=None, headers=None, timeout=None):
            return _Resp(status=200, body=UNAUTHORIZED_PAGE)

    client = acled_client.ACLEDClient(session=_Session(), config={"base_url": "https://acleddata.com/api/acled/read"})
    with pytest.raises(acled_auth.AcledHtmlResponse):
        client._fetch_page({"limit": 10})


def test_acled_cast_raises_on_a_web_page_rather_than_reading_zero_records(monkeypatch):
    from resolver.connectors import acled_cast

    monkeypatch.setattr(
        "resolver.ingestion.acled_auth.get_auth_header",
        lambda: {"Authorization": "Bearer tok"},
    )
    monkeypatch.setattr(
        acled_cast.requests, "get",
        lambda *a, **k: _Resp(status=200, body=UNAUTHORIZED_PAGE, url="https://acleddata.com/api/cast/read"),
    )
    with pytest.raises(acled_auth.AcledHtmlResponse):
        acled_cast.AcledCastConnector()._fetch_all_records(year=2026)


def test_acled_political_raises_and_records_the_web_page(monkeypatch):
    from pythia import acled_political

    acled_political.reset_html_failure_stats()
    monkeypatch.setattr(
        acled_political.requests, "get",
        lambda *a, **k: _Resp(status=200, body=UNAUTHORIZED_PAGE, url="https://acleddata.com/api/acled/read"),
    )
    with pytest.raises(acled_auth.AcledHtmlResponse):
        acled_political._fetch_with_retry({"iso": "729"}, {"Authorization": "Bearer tok"})
    stats = acled_political.get_html_failure_stats()
    assert list(stats) == ["729"]
    assert "Unauthorized" in stats["729"]
    acled_political.reset_html_failure_stats()
    assert acled_political.get_html_failure_stats() == {}
