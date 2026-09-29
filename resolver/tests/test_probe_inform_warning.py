# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The INFORM Warning probe decides what the service's answers MEAN.

The network half needs egress to JRC hosts, which the build sandbox does not
have, so these tests pin the half that reads responses: what kind of body came
back, which links are worth following, which API entries name a Warning
release, and whether a response looks like the data itself. A misread here
would send the connector after the wrong route.
"""

from __future__ import annotations

import json
from pathlib import Path

import requests

from tools.probe_inform_warning import (
    API_ROOT,
    classify_kind,
    extract_links,
    guide_lines,
    looks_like_warning_data,
    reachability,
    render_markdown,
    run,
    warning_entries,
    warning_identifiers,
    worth_following,
    Probe,
)


def test_the_body_decides_the_kind_before_the_header():
    # A JSON body served as text/html is still JSON: gateways mislabel.
    assert classify_kind("text/html", b'[{"a": 1}]') == "json"
    assert classify_kind("application/octet-stream", b"%PDF-1.7 ...") == "pdf"
    assert classify_kind("application/json", b"PK\x03\x04rest") == "zip"
    assert classify_kind("text/plain", b"<!DOCTYPE html><html>") == "html"
    assert classify_kind("text/csv", b"iso3,signal\nSOM,7") == "csv"
    assert classify_kind("", b"") == "empty"


def test_links_are_absolute_deduplicated_and_carry_their_anchor():
    page = (
        '<a href="/inform-index/API/InformAPI/Workflows">API &amp; data</a>'
        '<a href="/inform-index/API/InformAPI/Workflows">again</a>'
        '<a href="mailto:x@y.z">mail</a>'
        '<a href="javascript:void(0)">js</a>'
        '<script src="https://drmkc.jrc.ec.europa.eu/x/app.js"></script>'
    )
    links = extract_links(page, "https://drmkc.jrc.ec.europa.eu/inform-index/INFORM-Warning")
    urls = [u for u, _ in links]
    assert urls[0] == "https://drmkc.jrc.ec.europa.eu/inform-index/API/InformAPI/Workflows"
    assert urls.count(urls[0]) == 1
    assert links[0][1] == "API & data"
    assert not any(u.startswith(("mailto:", "javascript:")) for u in urls)


def test_only_data_looking_links_on_known_hosts_are_followed():
    assert worth_following("https://drmkc.jrc.ec.europa.eu/inform-index/API/x", "")
    assert worth_following("https://data.jrc.ec.europa.eu/dataset/abc", "INFORM Warning signals")
    assert worth_following("https://data.humdata.org/dataset/inform-warning", "")
    # A stylesheet named 'data' is not data, and an unrelated host is not asked.
    assert not worth_following("https://drmkc.jrc.ec.europa.eu/data/site.css", "")
    assert not worth_following("https://example.com/api/warning.json", "")
    # A link naming nothing about data is recorded, not fetched.
    assert not worth_following("https://drmkc.jrc.ec.europa.eu/inform-index/About", "About")


def test_guide_lines_quote_addresses_and_data_routes_only():
    text = "Introduction\nThe data are available through the Data Registry API at https://x.eu/api\n" \
           "Signals are normalised.\nPrevious releases are archived monthly.\n"
    lines = guide_lines(text)
    assert any("https://x.eu/api" in l for l in lines)
    assert any("archived" in l for l in lines)
    assert not any(l == "Introduction" for l in lines)


def test_workflow_entries_naming_warning_yield_ids_and_groups():
    payload = [
        {"WorkflowId": 101, "WorkflowGroupName": "INFORM2026", "Name": "INFORM Risk 2026"},
        {"WorkflowId": 202, "WorkflowGroupName": "INFORM Warning 2026-09", "Name": "Warning"},
        {"nested": [{"Id": 303, "System": "INFORM Warning"}]},
    ]
    entries = warning_entries(payload)
    ids = warning_identifiers(entries)
    assert 202 in ids["workflow_ids"] and 303 in ids["workflow_ids"]
    assert 101 not in ids["workflow_ids"]
    assert "INFORM Warning 2026-09" in ids["groups"]
    assert "INFORM Warning" in ids["groups"]


def test_warning_data_needs_country_codes_beside_warning_vocabulary():
    assert looks_like_warning_data("json", b'[{"Iso3": "SOM", "Signal": "Dry Conditions", "Value": 7.2}]')
    # Warning words alone are a landing page, not data.
    assert not looks_like_warning_data("json", b'{"title": "INFORM Warning signals"}')
    # Country codes alone are INFORM Risk, not Warning.
    assert not looks_like_warning_data("json", b'[{"Iso3": "SOM", "Score": 7.2}]')
    assert not looks_like_warning_data("html", b'<p>"SOM" warning</p>')


def test_reachability_keeps_the_best_outcome_per_host():
    probes = [
        Probe(label="a", url="https://h.eu/1", kind="error", error="ConnectionError"),
        Probe(label="b", url="https://h.eu/2", status=404, kind="html"),
        Probe(label="c", url="https://k.eu/1", status=200, kind="json"),
    ]
    assert reachability(probes) == {"h.eu": "http_error", "k.eu": "ok"}


class _Resp:
    def __init__(self, status: int, body: bytes, ctype: str, url: str):
        self.status_code = status
        self.content = body
        self.headers = {"Content-Type": ctype}
        self.url = url


def test_a_full_run_finds_a_warning_workflow_and_its_scores(tmp_path: Path):
    """End to end over a fake transport: a Warning workflow listed by the Risk
    API leads to a scores request whose body is flagged as candidate data."""
    workflows = json.dumps([
        {"WorkflowId": 7, "WorkflowGroupName": "INFORM Warning 2026-09"},
        {"WorkflowId": 1, "WorkflowGroupName": "INFORM2026"},
    ]).encode()
    scores = json.dumps([{"Iso3": "SOM", "IndicatorId": "Dry Conditions signal", "Score": 6.1}]).encode()
    landing = b'<html><a href="/inform-index/API/InformAPI/Workflows/WorkflowGroups">API</a></html>'

    def fake_get(url, timeout=None, allow_redirects=True):
        if "drmkc.jrc.ec.europa.eu" not in url and "humdata" not in url:
            raise requests.ConnectionError("denied")
        if url.endswith("Workflows/WorkflowGroups"):
            return _Resp(200, workflows, "application/json", url)
        if "Countries/Scores" in url and "WorkflowId=7" in url:
            return _Resp(200, scores, "application/json", url)
        if "INFORM-Warning" in url:
            return _Resp(200, landing, "text/html", url)
        return _Resp(404, b"<html>not found</html>", "text/html", url)

    report = run(tmp_path, max_follow=5, get=fake_get)
    ids = report["risk_api"]["identifiers"]
    assert 7 in ids["workflow_ids"] and 1 not in ids["workflow_ids"]
    cand_urls = [c["url"] for c in report["data_candidates"]]
    assert any("Countries/Scores" in u and "WorkflowId=7" in u for u in cand_urls)
    assert report["reachability"]["publications.jrc.ec.europa.eu"] == "unreachable"
    # Every successful response is saved raw so a person can read it later.
    saved = [p["saved_as"] for p in report["probes"] if p["saved_as"]]
    assert saved and all((tmp_path / s).exists() for s in saved)
    md = render_markdown(report)
    assert "INFORM Warning probe" in md and f"{API_ROOT}/Countries/Scores" in md


def test_main_never_fails_the_run(tmp_path: Path, monkeypatch):
    import tools.probe_inform_warning as mod

    def boom(*a, **k):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(mod, "run", boom)
    assert mod.main(["--out-dir", str(tmp_path)]) == 0
    assert "crashed" in json.loads((tmp_path / "inform_warning_probe.json").read_text())
