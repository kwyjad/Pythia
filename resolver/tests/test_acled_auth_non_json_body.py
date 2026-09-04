# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The 2026-09-03 Phase 1 failure: a 200 whose body is not JSON.

Both ACLED OAuth grants were answered ``200`` with a body ``json.loads``
refused. The status check passed, ``resp.json()`` raised, and the monthly
ingest died on ``Expecting value: line 1 column 1 (char 0)``, a message that
names neither the status, the content type, the URL reached after redirects,
nor one character of what was served. Four different causes produce that
identical message and each wants a different repair.

These tests pin the two halves of the fix: the error must say what came back,
and the request must ask for JSON.
"""

from __future__ import annotations

import pytest

from resolver.ingestion import acled_auth


class _Resp:
    def __init__(self, status=200, body="", content_type="text/html", url=None, history=()):
        self.status_code = status
        self.text = body
        self.headers = {"Content-Type": content_type}
        self.url = url or acled_auth.OAUTH_TOKEN_URL
        self.history = list(history)

    def json(self):
        import json

        return json.loads(self.text)


@pytest.fixture(autouse=True)
def _clear_cache():
    acled_auth._CACHE.update({"access_token": None, "refresh_token": None, "expiry": None})
    yield
    acled_auth._CACHE.update({"access_token": None, "refresh_token": None, "expiry": None})


def _capture(monkeypatch, resp):
    """Patch requests.post and return the dict the call is recorded into."""
    seen: dict = {}

    def fake_post(url, data=None, headers=None, timeout=None):
        seen["url"] = url
        seen["data"] = data
        seen["headers"] = headers or {}
        return resp

    monkeypatch.setattr(acled_auth.requests, "post", fake_post)
    return seen


# --- the error must describe the response -----------------------------------


def test_html_served_with_a_200_raises_a_message_that_names_the_body(monkeypatch):
    page = "<!DOCTYPE html><html><head><title>Just a moment...</title></head></html>"
    _capture(monkeypatch, _Resp(status=200, body=page, content_type="text/html; charset=UTF-8"))

    with pytest.raises(RuntimeError) as excinfo:
        acled_auth._password_grant("u@example.com", "pw")

    message = str(excinfo.value)
    # The regression: this used to be a bare JSONDecodeError.
    assert "not JSON" in message
    assert "status=200" in message
    assert "text/html" in message
    assert "body_shape=html" in message
    assert "Just a moment" in message


def test_a_redirect_away_from_the_token_route_is_named(monkeypatch):
    landing = "https://acleddata.com/data-export-tool/"
    resp = _Resp(
        status=200,
        body="<html><body>ACLED</body></html>",
        url=landing,
        history=[_Resp(status=301), _Resp(status=302)],
    )
    _capture(monkeypatch, resp)

    with pytest.raises(RuntimeError) as excinfo:
        acled_auth._refresh_grant("stale-token")

    message = str(excinfo.value)
    assert landing in message
    assert "redirects=301->302" in message


def test_a_non_200_still_names_the_status_and_body(monkeypatch):
    _capture(monkeypatch, _Resp(status=415, body="Unsupported Media Type", content_type="text/plain"))

    with pytest.raises(RuntimeError) as excinfo:
        acled_auth._password_grant("u@example.com", "pw")

    assert "status=415" in str(excinfo.value)
    assert "Unsupported Media Type" in str(excinfo.value)


def test_a_json_array_is_refused_rather_than_returned(monkeypatch):
    _capture(monkeypatch, _Resp(status=200, body="[]", content_type="application/json"))

    with pytest.raises(RuntimeError) as excinfo:
        acled_auth._password_grant("u@example.com", "pw")

    assert "not an object" in str(excinfo.value)


def test_the_body_snippet_is_redacted_against_the_environment(monkeypatch):
    monkeypatch.setenv("ACLED_PASSWORD", "sup3rsecret-password-value")
    body = "error: no account for password sup3rsecret-password-value"
    _capture(monkeypatch, _Resp(status=401, body=body, content_type="text/plain"))

    with pytest.raises(RuntimeError) as excinfo:
        acled_auth._password_grant("u@example.com", "sup3rsecret-password-value")

    assert "sup3rsecret-password-value" not in str(excinfo.value)


def test_a_good_response_still_returns_its_payload(monkeypatch):
    _capture(
        monkeypatch,
        _Resp(status=200, body='{"access_token": "abc", "refresh_token": "def"}',
              content_type="application/json"),
    )
    assert acled_auth._password_grant("u@example.com", "pw")["access_token"] == "abc"


# --- the request must ask for JSON ------------------------------------------


def test_both_grants_ask_for_json_and_send_a_real_user_agent(monkeypatch):
    good = '{"access_token": "abc"}'

    seen = _capture(monkeypatch, _Resp(status=200, body=good, content_type="application/json"))
    acled_auth._password_grant("u@example.com", "pw")
    assert seen["headers"]["Accept"] == "application/json"
    assert "python-requests" not in seen["headers"].get("User-Agent", "")
    assert seen["headers"]["User-Agent"]
    # The 415 fix of 2026-07-15 must survive this rewrite.
    assert seen["data"]["scope"] == "authenticated"

    seen = _capture(monkeypatch, _Resp(status=200, body=good, content_type="application/json"))
    acled_auth._refresh_grant("tok")
    assert seen["headers"]["Accept"] == "application/json"
    assert seen["headers"]["User-Agent"]


def test_the_user_agent_is_overridable(monkeypatch):
    monkeypatch.setenv("ACLED_USER_AGENT", "custom-agent/9")
    seen = _capture(
        monkeypatch,
        _Resp(status=200, body='{"access_token": "abc"}', content_type="application/json"),
    )
    acled_auth._password_grant("u@example.com", "pw")
    assert seen["headers"]["User-Agent"] == "custom-agent/9"
