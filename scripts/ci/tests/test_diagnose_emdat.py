# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The EM-DAT diagnostic decides from evidence, never guesses, never raises."""

from __future__ import annotations

import json

from scripts.ci import diagnose_emdat as diag


def _post_factory(answers):
    """answers: {probe_name: (status, body_dict_or_text)}.

    Probes are told apart by what only they contain: the filters variant by
    its ``filters:`` argument, introspection by ``__type``, and the
    connector's own shape by neither.
    """

    def _name(q: str) -> str:
        if "introspect" in q:
            return "introspect"
        if "filters" in q:
            return "filters"
        return "connector"

    def post(url, payload, headers, timeout):
        name = _name(payload["query"])
        status, body = answers[name]
        return status, body if isinstance(body, str) else json.dumps(body)

    return post


def test_no_key_is_inconclusive_and_asks_nothing():
    calls = []

    def post(*args):
        calls.append(args)
        return 200, "{}"

    report = diag.run(post, key="")
    assert report["verdict"] == diag.VERDICT_INCONCLUSIVE
    assert calls == []


def test_connector_shape_working_is_reported_as_such():
    ok = {"data": {"public_emdat": {"total_available": 3, "data": [{}, {}, {}]}}}
    post = _post_factory({"connector": (200, ok), "filters": (200, ok), "introspect": (200, {"data": {}})})
    report = diag.run(post, key="k")
    assert report["verdict"] == diag.VERDICT_OK
    assert report["probes"]["connector"]["rows_returned"] == 3


def test_filters_shape_working_while_connector_500s_is_a_query_shape_verdict():
    ok = {"data": {"public_emdat": {"total_available": 3, "data": [{}, {}, {}]}}}
    post = _post_factory({
        "connector": (500, "Internal Server Error"),
        "filters": (200, ok),
        "introspect": (200, {"data": {}}),
    })
    report = diag.run(post, key="k")
    assert report["verdict"] == diag.VERDICT_QUERY_SHAPE
    assert "emdat.py" in report["reason"]


def test_refusal_naming_the_key_is_an_account_verdict():
    post = _post_factory({
        "connector": (403, '{"message": "invalid API key"}'),
        "filters": (403, '{"message": "invalid API key"}'),
        "introspect": (403, '{"message": "invalid API key"}'),
    })
    report = diag.run(post, key="k")
    assert report["verdict"] == diag.VERDICT_ACCOUNT


def test_every_probe_500_is_upstream():
    post = _post_factory({
        "connector": (500, "Internal Server Error"),
        "filters": (500, "Internal Server Error"),
        "introspect": (503, "Service Unavailable"),
    })
    report = diag.run(post, key="k")
    assert report["verdict"] == diag.VERDICT_UPSTREAM


def test_transport_failure_never_raises():
    def post(*args):
        raise ConnectionError("proxy refused")

    report = diag.run(post, key="k")
    assert report["verdict"] in (diag.VERDICT_UPSTREAM, diag.VERDICT_INCONCLUSIVE)
    assert all("error" in p for p in report["probes"].values())


def test_main_always_exits_zero(tmp_path, monkeypatch):
    monkeypatch.setenv(diag.API_KEY_ENV, "")
    out = tmp_path / "r.json"
    assert diag.main(["--out", str(out)]) == 0
    assert json.loads(out.read_text())["verdict"] == diag.VERDICT_INCONCLUSIVE
