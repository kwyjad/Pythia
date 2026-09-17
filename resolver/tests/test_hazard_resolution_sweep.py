# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""ReliefWeb silence-sweep tests (no network — POST seam injected)."""

from __future__ import annotations

from resolver.hazard_resolution.reliefweb_sweep import sweep_country_month
from resolver.tests.hazard_resolution_utils import make_rulebook


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
    assert {"field": "primary_country.iso3", "value": "MDG"} in conditions


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


# ---------------------------------------------------------------------------
# The keyword query's scope (Sept 2026)
#
# The sweep became 95% of every trigger in the backcast: 62,304 cells
# against 3,499 from IBTrACS and GDACS. Landlocked Afghanistan "had" a
# cyclone in 305 of 321 months. The cause was in the keyword query, not the
# date window: it searched `body` over ANY `country.iso3` tag, so a passing
# mention inside a regional bulletin counted as a hazard report for every
# country the bulletin was tagged with, and one hit defeats silence.
# ---------------------------------------------------------------------------


def _keyword_payload(post: "_FakePost") -> dict:
    return next(c["payload"] for c in post.calls if "query" in c["payload"])


def _taxonomy_payload(post: "_FakePost") -> dict:
    return next(c["payload"] for c in post.calls if "query" not in c["payload"])


def _country_field(payload: dict) -> str:
    return next(
        c["field"] for c in payload["filter"]["conditions"]
        if c["field"].endswith("country.iso3")
    )


def test_the_keyword_query_searches_titles_not_bodies():
    post = _FakePost(taxonomy_hits=0, keyword_hits=0)
    sweep_country_month("AFG", "2013-01", make_rulebook(), post=post)
    assert _keyword_payload(post)["query"]["fields"] == ["title"]


def test_the_keyword_query_is_scoped_to_the_primary_country():
    post = _FakePost(taxonomy_hits=0, keyword_hits=0)
    sweep_country_month("AFG", "2013-01", make_rulebook(), post=post)
    assert _country_field(_keyword_payload(post)) == "primary_country.iso3"


def test_the_taxonomy_query_is_scoped_to_the_primary_country_too():
    # This test replaces one asserting the opposite, and the run that
    # refuted it is 35193605561. The first repair left the taxonomy query
    # filtering on any `country.iso3` tag, arguing that a curated
    # disaster-type tag is an editor's judgement about the report and so any
    # country on it is fair. Over nine re-walked months the keyword query
    # decided NOTHING and all 1,325 hits came from the taxonomy query: 154
    # of 233 countries had a cyclone-tagged report naming them in July 2023,
    # Afghanistan among them. The tag is a judgement about the report; the
    # country list beside it is not a judgement that the report is about
    # each of those countries.
    post = _FakePost(taxonomy_hits=0, keyword_hits=0)
    sweep_country_month("AFG", "2013-01", make_rulebook(), post=post)
    assert _country_field(_taxonomy_payload(post)) == "primary_country.iso3"


def test_the_sweep_records_the_bar_each_query_had_to_clear():
    # The scoping moved twice in one month. A stored sweep that does not say
    # which rule decided it cannot be read back against either.
    post = _FakePost(taxonomy_hits=0, keyword_hits=0)
    ev = sweep_country_month("AFG", "2013-01", make_rulebook(), post=post)
    assert ev["taxonomy_country_field"] == "primary_country.iso3"
    assert ev["keyword_country_field"] == "primary_country.iso3"
    assert ev["keyword_fields"] == ["title"]


def test_landslide_no_longer_defeats_flood_silence():
    # A landslide is a different hazard. It was in the flood keyword list.
    keywords = make_rulebook().get("flood.reliefweb_sweep.keywords")
    assert "landslide" not in keywords
    assert "flood" in keywords


def test_per_query_hits_are_recorded_apart():
    # The total alone cannot say whether the curated tag or the keyword
    # search defeated silence, and those are different claims about a cell.
    post = _FakePost(taxonomy_hits=0, keyword_hits=3)
    ev = sweep_country_month("AFG", "2013-01", make_rulebook(), post=post)
    assert ev["taxonomy_hits"] == 0
    assert ev["keyword_hits"] == 3
    assert ev["total_hits"] == 3
    assert ev["silent"] is False


def test_a_taxonomy_hit_skips_the_keyword_query_and_leaves_its_count_unset():
    post = _FakePost(taxonomy_hits=2, keyword_hits=99)
    ev = sweep_country_month("AFG", "2013-01", make_rulebook(), post=post)
    assert ev["taxonomy_hits"] == 2
    assert ev["keyword_hits"] is None
    assert ev["total_hits"] == 2


def test_the_country_field_is_validated_not_freely_configurable():
    # A ReliefWeb field name is a spelling the API either knows or refuses.
    # A typo here would widen the sweep silently, which is the whole fault.
    import yaml

    from resolver.hazard_resolution.rulebook import (
        DEFAULT_RULEBOOK_PATH,
        validate_rulebook,
    )

    with open(DEFAULT_RULEBOOK_PATH, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    for key in ("taxonomy_country_field", "keyword_country_field"):
        with open(DEFAULT_RULEBOOK_PATH, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        data["cyclone"]["reliefweb_sweep"][key] = "iso3"
        problems = validate_rulebook(data)
        assert any(key in p for p in problems), (key, problems)
