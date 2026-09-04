# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The ACLED OAuth diagnostic must tell four look-alike failures apart.

A WAF page, a moved route, content negotiation and a dead credential all
surface as ``Expecting value: line 1 column 1 (char 0)`` and each wants a
different repair. The verdict logic is pure, so it is tested against recorded
response shapes with no network.
"""

from __future__ import annotations

import pytest

from scripts.ci import diagnose_acled_auth as diag


class _Resp:
    def __init__(self, status=200, body="", content_type="text/html", url="https://acleddata.com/oauth/token", history=()):
        self.status_code = status
        self.text = body
        self.headers = {"Content-Type": content_type}
        self.url = url
        self.history = list(history)

    def json(self):
        import json

        return json.loads(self.text)


CLOUDFLARE = (
    "<!DOCTYPE html><html><head><title>Just a moment...</title></head>"
    "<body>Checking your browser before accessing acleddata.com. Cloudflare Ray ID: abc</body></html>"
)
MARKETING = "<!DOCTYPE html><html><head><title>ACLED | Data Export Tool</title></head><body>Welcome</body></html>"
TOKEN_JSON = '{"access_token": "abc", "refresh_token": "def"}'


@pytest.fixture(autouse=True)
def _creds(monkeypatch):
    monkeypatch.setenv("ACLED_USERNAME", "probe@example.com")
    monkeypatch.setenv("ACLED_PASSWORD", "pw-value")
    monkeypatch.delenv("ACLED_REFRESH_TOKEN", raising=False)
    monkeypatch.delenv("ACLED_USER_AGENT", raising=False)


def _run(responder):
    def request(method, url, **kwargs):
        resp = responder(method, url, kwargs)
        if isinstance(resp, Exception):
            raise resp
        return resp

    return diag.run(request=request)


def test_a_working_endpoint_is_reported_ok():
    report = _run(lambda m, u, k: _Resp(status=200, body=TOKEN_JSON, content_type="application/json"))
    assert report["verdict"] == diag.VERDICT_OK
    # A healthy endpoint needs no search of the candidate routes.
    assert not [n for n in report["probes"] if n.startswith("candidate:")]


def test_a_cloudflare_page_is_named_as_a_challenge():
    report = _run(lambda m, u, k: _Resp(status=200, body=CLOUDFLARE))
    assert report["verdict"] == diag.VERDICT_WAF
    assert "WAF" in report["reason"]


def test_a_json_refusal_points_at_the_credentials():
    report = _run(
        lambda m, u, k: _Resp(
            status=401,
            body='{"error": "invalid_grant", "message": "credentials rejected"}',
            content_type="application/json",
        )
    )
    assert report["verdict"] == diag.VERDICT_CREDENTIALS


def test_an_alternate_route_answering_json_means_the_endpoint_moved():
    def responder(method, url, kwargs):
        if url == diag.acled_auth.OAUTH_TOKEN_URL:
            return _Resp(status=200, body=MARKETING)
        return _Resp(status=200, body='{"detail": "method not allowed"}', content_type="application/json")

    report = _run(responder)
    assert report["verdict"] == diag.VERDICT_MOVED
    assert "OAUTH_TOKEN_URL" in report["reason"]


def test_a_redirect_to_a_page_means_the_endpoint_moved():
    def responder(method, url, kwargs):
        if url == diag.acled_auth.OAUTH_TOKEN_URL:
            return _Resp(status=200, body=MARKETING, url="https://acleddata.com/", history=[_Resp(status=301)])
        return _Resp(status=404, body="not found", content_type="text/plain")

    report = _run(responder)
    assert report["verdict"] == diag.VERDICT_MOVED
    assert "redirect" in report["reason"]


def test_html_with_no_challenge_and_no_redirect_is_its_own_verdict():
    def responder(method, url, kwargs):
        if url == diag.acled_auth.OAUTH_TOKEN_URL:
            return _Resp(status=200, body=MARKETING)
        return _Resp(status=404, body="not found", content_type="text/plain")

    report = _run(responder)
    assert report["verdict"] == diag.VERDICT_HTML


def test_the_bare_header_probe_sends_the_pre_fix_request_shape():
    seen: dict = {}

    def responder(method, url, kwargs):
        seen[len(seen)] = kwargs.get("headers", {})
        return _Resp(status=200, body=MARKETING)

    _run(responder)
    shapes = list(seen.values())
    assert any("Accept" in h for h in shapes), "the current shape must be probed"
    assert any("Accept" not in h for h in shapes), "the pre-fix shape must be probed too"


def test_upstream_errors_are_not_blamed_on_the_credentials():
    report = _run(lambda m, u, k: _Resp(status=503, body="service unavailable", content_type="text/plain"))
    assert report["verdict"] == diag.VERDICT_UPSTREAM


def test_no_credentials_is_inconclusive_not_a_verdict(monkeypatch):
    monkeypatch.delenv("ACLED_USERNAME", raising=False)
    monkeypatch.delenv("ACLED_PASSWORD", raising=False)
    report = _run(lambda m, u, k: _Resp(status=200, body=MARKETING))
    assert report["verdict"] == diag.VERDICT_INCONCLUSIVE


def test_a_transport_failure_is_recorded_not_raised():
    report = _run(lambda m, u, k: OSError("connection reset"))
    assert report["verdict"] in {diag.VERDICT_INCONCLUSIVE, diag.VERDICT_UPSTREAM}
    assert "connection reset" in report["probes"]["get_token_url"]["error"]


def test_the_report_never_carries_a_credential(monkeypatch):
    monkeypatch.setenv("ACLED_PASSWORD", "sup3rsecret-password-value")
    body = "rejected password sup3rsecret-password-value for probe@example.com"
    report = _run(lambda m, u, k: _Resp(status=401, body=body, content_type="text/plain"))
    import json

    assert "sup3rsecret-password-value" not in json.dumps(report)


def test_main_always_exits_zero(capsys):
    import scripts.ci.diagnose_acled_auth as module

    module.run = lambda: {  # type: ignore[assignment]
        "verdict": diag.VERDICT_WAF,
        "reason": "challenged",
        "token_url": "u",
        "probes": {},
    }
    try:
        assert module.main([]) == 0
    finally:
        module.run = diag.run  # type: ignore[assignment]
    assert "verdict" in capsys.readouterr().out
