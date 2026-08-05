# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""ReliefWeb silence-sweep tests (no network — POST seam injected)."""

from __future__ import annotations

import pytest

from resolver.resolution_machine.reliefweb_sweep import sweep_country_month
from resolver.tests.resolution_machine_utils import make_rulebook


class _FakePost:
    """Records calls; serves canned totalCounts per query kind."""

    def __init__(self, taxonomy_hits: int, keyword_hits: int = 0, fail: bool = False):
        self.taxonomy_hits = taxonomy_hits
        self.keyword_hits = keyword_hits
        self.fail = fail
        self.calls: list[dict] = []

    def __call__(self, url, payload, params, timeout):
        self.calls.append({"url": url, "payload": payload, "params": params})
        if self.fail:
            raise RuntimeError("simulated API outage")
        if "query" in payload:
            total = self.keyword_hits
        else:
            total = self.taxonomy_hits
        data = [
            {
                "fields": {
                    "title": f"Report {i}",
                    "url": f"https://reliefweb.int/report/{i}",
                    "date": {"created": "2013-01-05T00:00:00+00:00"},
                }
            }
            for i in range(min(total, 2))
        ]
        return {"totalCount": total, "data": data}


def test_silent_sweep_runs_both_queries_and_confirms_silence():
    post = _FakePost(taxonomy_hits=0, keyword_hits=0)
    ev = sweep_country_month("MDG", "2013-01", make_rulebook(), post=post)
    assert ev["silent"] is True
    assert ev["inconclusive"] is False
    assert ev["total_hits"] == 0
    assert [q["kind"] for q in ev["queries"]] == ["disaster_type", "keywords"]
    # Evidence must record the queries verbatim and a retrieval timestamp.
    assert ev["queries"][0]["payload"]["filter"]["conditions"]
    assert ev["retrieved_at"]
    # The window and country reached the API.
    conditions = post.calls[0]["payload"]["filter"]["conditions"]
    assert {"field": "country.iso3", "value": "MDG"} in conditions


def test_taxonomy_hits_short_circuit_and_are_not_silent():
    post = _FakePost(taxonomy_hits=3)
    ev = sweep_country_month("PHL", "2013-11", make_rulebook(), post=post)
    assert ev["silent"] is False
    assert ev["total_hits"] == 3
    # Keyword query is skipped when the taxonomy sweep already has hits.
    assert [q["kind"] for q in ev["queries"]] == ["disaster_type"]
    assert len(ev["queries"][0]["sample"]) == 2


def test_keyword_hits_flip_silence_off():
    post = _FakePost(taxonomy_hits=0, keyword_hits=2)
    ev = sweep_country_month("PHL", "2013-11", make_rulebook(), post=post)
    assert ev["silent"] is False
    assert ev["total_hits"] == 2
    kinds = [q["kind"] for q in ev["queries"]]
    assert kinds == ["disaster_type", "keywords"]
    # Multi-word keywords are quoted in the query string.
    value = ev["queries"][1]["payload"]["query"]["value"]
    assert '"tropical storm"' in value


def test_api_failure_is_inconclusive_never_silent():
    post = _FakePost(taxonomy_hits=0, fail=True)
    ev = sweep_country_month("MDG", "2013-01", make_rulebook(), post=post)
    assert ev["inconclusive"] is True
    assert ev["silent"] is False  # fail-closed: no zero without evidence
    assert ev["error"]


def test_sweep_window_pads_publication_dates():
    rb = make_rulebook({"cyclone": {"reliefweb_sweep": {"publication_pad_days": 14}}})
    post = _FakePost(taxonomy_hits=0)
    sweep_country_month("PHL", "2013-11", rb, post=post)
    conditions = post.calls[0]["payload"]["filter"]["conditions"]
    date_cond = next(c for c in conditions if c["field"] == "date.created")
    assert date_cond["value"]["from"].startswith("2013-11-01")
    # Nov 30 + 14 days = Dec 14.
    assert date_cond["value"]["to"].startswith("2013-12-14")
