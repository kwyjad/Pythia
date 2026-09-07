# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Group D of the run-34124705852 repairs: a probe read as a connector,
and a page walk that stopped on our own boundary.

Two faults that look nothing alike and share one shape: a machine reading
its own limits as facts about an upstream.

``diagnose_acled_auth`` GETs a POST-only token route on purpose, to see
what the route says. The answer is a 405 web page, which is the probe
working. The bundle's ACLED check counted it and flagged all three ACLED
connectors on a run where every production grant returned 200 JSON. A
check that fails on a healthy run is one the reader learns to skip.

``_fetch_paginated_global`` stopped at a hardcoded page count with the
server's ``next`` link still in hand, and said nothing. At 100 records a
page and ``max_pages=30`` that is 3,000 records; the ACAPS INFORM country
log has more, so the trend built from it was built on our own ceiling. The
walk now follows ``next`` to exhaustion, and any early stop is announced
against the ``count`` the envelope states.

Network-free.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# D1: a diagnostic probe is not connector traffic
# ---------------------------------------------------------------------------


def test_the_recorder_labels_a_probe_and_forgets_it_afterwards():
    from resolver.diagnostics import http_recorder

    assert http_recorder._probe_label() is None
    with http_recorder.probing("diagnose_acled_auth:GET"):
        assert http_recorder._probe_label() == "diagnose_acled_auth:GET"
        with http_recorder.probing("nested"):
            assert http_recorder._probe_label() == "nested"
        assert http_recorder._probe_label() == "diagnose_acled_auth:GET"
    assert http_recorder._probe_label() is None, "the label must not outlive the block"


def test_the_probe_label_survives_an_exception():
    """A probe that raises must not leave every later call labelled."""

    from resolver.diagnostics import http_recorder

    with pytest.raises(ValueError):
        with http_recorder.probing("boom"):
            raise ValueError("boom")
    assert http_recorder._probe_label() is None


def test_the_acled_diagnostic_labels_its_own_requests(monkeypatch):
    """``_default_request`` is the one place the script touches the network."""

    from resolver.diagnostics import http_recorder
    from scripts.ci import diagnose_acled_auth

    seen: list[str | None] = []

    class _Resp:
        status_code = 405
        headers = {"Content-Type": "text/html"}
        text = "<html>Method Not Allowed</html>"
        history: list = []
        url = "https://acleddata.com/oauth/token"

    def _fake_request(method, url, **kwargs):
        seen.append(http_recorder._probe_label())
        return _Resp()

    monkeypatch.setattr("requests.request", _fake_request)
    diagnose_acled_auth._default_request("GET", "https://acleddata.com/oauth/token")

    assert seen == ["diagnose_acled_auth:GET"], (
        "the diagnostic's calls must be labelled, or the bundle reads a probe's "
        "405 web page as a connector that was handed the website"
    )


def test_the_recorded_call_carries_the_probe_label(monkeypatch):
    """The label must reach the row, not only the thread-local."""

    from resolver.diagnostics import http_recorder, run_log

    written: list[dict] = []

    def _capture(stream, payload):
        if stream == run_log.STREAM_HTTP:
            written.append(payload)

    monkeypatch.setattr(http_recorder.run_log, "record", _capture)

    with http_recorder.probing("diagnose_acled_auth:GET"):
        http_recorder._record(
            method="GET",
            url="https://acleddata.com/oauth/token",
            body=None,
            response=None,
            elapsed_ms=30.0,
            error=None,
        )
    http_recorder._record(
        method="GET",
        url="https://acleddata.com/api/acled/read",
        body=None,
        response=None,
        elapsed_ms=12.0,
        error=None,
    )

    assert [r.get("probe") for r in written] == ["diagnose_acled_auth:GET", None]


# ---------------------------------------------------------------------------
# D2: a page walk stops on the server's word, not on ours
# ---------------------------------------------------------------------------


class _Page:
    """One DRF page envelope."""

    status_code = 200

    def __init__(self, results, nxt, count):
        self._body = {"results": results, "next": nxt, "count": count}

    def json(self):
        return self._body


def _server(pages: int, per_page: int = 100):
    """A DRF endpoint holding ``pages`` pages, counting its own requests."""

    calls: list[str] = []
    total = pages * per_page

    def _get(url, params=None, headers=None, timeout=None):
        calls.append(url)
        # The page number rides in the url after the first request.
        n = 1
        if "page=" in url:
            n = int(url.rsplit("page=", 1)[1])
        results = [{"iso3": "SOM", "page": n, "i": i} for i in range(per_page)]
        nxt = f"https://acaps.test/next?page={n + 1}" if n < pages else None
        return _Page(results, nxt, total)

    return _get, calls


def _patch(monkeypatch, get_fn, budget=300.0):
    from pythia.tools import ingest_structured_data as ing

    monkeypatch.setattr(ing.requests, "get", get_fn)
    monkeypatch.setattr(ing, "_get_acaps_token", lambda force_refresh=False: "tok")
    monkeypatch.setattr(ing.time, "sleep", lambda _s: None)
    # Tolerant on purpose: against the pre-fix code these names are absent,
    # and a fixture that raises hides which assertion would have caught the
    # fault.
    if hasattr(ing, "_PAGE_BUDGET_SECONDS"):
        monkeypatch.setattr(ing, "_PAGE_BUDGET_SECONDS", budget)
    if hasattr(ing, "reset_paging_budget"):
        ing.reset_paging_budget()
    return ing


def test_the_walk_follows_next_past_the_old_thirty_page_ceiling(monkeypatch):
    """The country-log fault: 3,000 records was our boundary, not ACAPS'."""

    # 60 pages clears BOTH old boundaries: the country-log call site's
    # max_pages=30 and the function's own default of 50.
    get_fn, calls = _server(pages=60)
    ing = _patch(monkeypatch, get_fn)

    rows = ing._fetch_paginated_global(
        "/api/v1/inform-severity-index/country-log/", token="tok"
    )

    assert len(rows) == 6000, (
        "a 30-page cap stopped at 3,000 records with a next link still in "
        "hand, and the INFORM trend was built on that ceiling"
    )
    assert len(calls) == 60


def test_no_call_site_still_passes_a_guessed_page_cap():
    """Nine call sites carried a number that meant 'surely enough'."""

    import re
    from pathlib import Path

    src = Path("pythia/tools/ingest_structured_data.py").read_text()
    # The def's own default names the ceiling constant; a literal anywhere
    # else in a call is a caller re-guessing.
    literals = re.findall(r"^\s*max_pages=\d+,\s*$", src, flags=re.MULTILINE)
    assert literals == [], f"guessed page caps still in place: {literals}"


def test_a_bound_ceiling_is_announced_against_the_servers_own_count(
    monkeypatch, capsys
):
    """Silence and 'the source ran out' are the same thing to a reader."""

    get_fn, _calls = _server(pages=40)
    ing = _patch(monkeypatch, get_fn)

    rows = ing._fetch_paginated_global("/api/v1/x/", max_pages=3, token="tok")

    assert len(rows) == 300
    printed = capsys.readouterr().out
    assert "::warning" in printed and "truncated" in printed
    assert "4000" in printed, "the shortfall must be stated against count"


def test_an_exhausted_walk_says_nothing_alarming(monkeypatch, capsys):
    """A walk that finished is not a truncated one."""

    get_fn, _calls = _server(pages=4)
    ing = _patch(monkeypatch, get_fn)

    rows = ing._fetch_paginated_global("/api/v1/x/", token="tok")

    assert len(rows) == 400
    assert "::warning" not in capsys.readouterr().out


def test_a_wall_clock_budget_stops_the_walk_and_names_the_shortfall(
    monkeypatch, capsys
):
    """A pace polite enough to keep is slow enough to outrun a step."""

    get_fn, calls = _server(pages=500)
    ing = _patch(monkeypatch, get_fn, budget=0.0001)

    clock = iter([0.0, 0.0, 0.05, 0.10, 0.15, 0.20])

    def _now():
        try:
            return next(clock)
        except StopIteration:
            return 99.0

    monkeypatch.setattr(ing.time, "monotonic", _now)
    rows = ing._fetch_paginated_global("/api/v1/x/", token="tok")

    assert len(calls) < 500, "the budget must stop the walk"
    assert rows, "what was fetched before the budget bound is kept"
    printed = capsys.readouterr().out
    assert "::warning" in printed and "budget" in printed
    ing.reset_paging_budget()


def test_the_hard_ceiling_is_a_runaway_guard_not_a_data_bound():
    """500 pages is 50,000 records; a real archive says so through count."""

    from pythia.tools import ingest_structured_data as ing

    assert ing._PAGE_HARD_CEILING >= 500


def test_an_http_error_mid_walk_keeps_what_was_already_read(monkeypatch):
    """A page-3 outage must not discard pages 1 and 2."""

    class _Bad:
        status_code = 503

        def json(self):  # pragma: no cover - never reached
            raise AssertionError("must not parse a non-200")

    seen = {"n": 0}

    def _get(url, params=None, headers=None, timeout=None):
        seen["n"] += 1
        if seen["n"] >= 3:
            return _Bad()
        return _Page(
            [{"i": seen["n"]}], f"https://acaps.test/next?page={seen['n'] + 1}", 900
        )

    ing = _patch(monkeypatch, _get)
    rows = ing._fetch_paginated_global("/api/v1/x/", token="tok")
    assert len(rows) == 2


def test_the_budget_is_spent_across_the_process_not_per_walk(monkeypatch, capsys):
    """INFORM makes over a dozen walks; a per-walk budget multiplies by that.

    A 300s budget per walk against a 30-minute step is an hour of allowance
    on paper, which is how a step gets SIGKILLed with no summary written.
    """

    get_fn, calls = _server(pages=500)
    ing = _patch(monkeypatch, get_fn, budget=10.0)

    now = {"t": 0.0}

    def _tick():
        now["t"] += 4.0
        return now["t"]

    monkeypatch.setattr(ing.time, "monotonic", _tick)

    first = ing._fetch_paginated_global("/api/v1/a/", token="tok")
    calls_after_first = len(calls)
    second = ing._fetch_paginated_global("/api/v1/b/", token="tok")

    assert first, "the first walk gets the allowance"
    assert len(calls) == calls_after_first, (
        "the second walk must inherit the spent deadline, not a fresh one"
    )
    assert second == []
    printed = capsys.readouterr().out
    assert "/api/v1/b/" in printed, "a walk that got nothing must say why"
    ing.reset_paging_budget()
