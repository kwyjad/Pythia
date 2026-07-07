# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Sibyl leakage controls: date filter always on, backtest-mode gating of
the classifier, blocked resolution-source domains, residual-leak logging."""

from __future__ import annotations

from datetime import date

import pytest

from pythia.web_research.types import EvidencePack, EvidenceSource

import sibyl.tools as sibyl_tools
from sibyl.leakage import (
    LeakageStats,
    clamp_live_lookup_date,
    date_range_freshness,
    extract_dates,
    filter_sources,
    is_backtest,
    is_blocked_url,
    snippet_leaks,
)

TODAY = date(2026, 7, 7)
PAST_AS_OF = date(2025, 3, 1)


def _src(url: str, summary: str = "", date_str: str | None = None) -> EvidenceSource:
    return EvidenceSource(title="t", url=url, summary=summary, date=date_str)


# --- date filter is ALWAYS applied -------------------------------------------

def test_date_range_freshness_ends_at_as_of():
    fresh = date_range_freshness(PAST_AS_OF, window_days=60)
    assert fresh == "2024-12-31to2025-03-01"


def test_search_is_date_filtered_in_live_mode_too(monkeypatch):
    """brave_search must pass a freshness date range even when asOf=today —
    the filter is applied always (harmless live, load-bearing in backtest)."""
    captured: dict = {}

    def fake_fetch(query, **kwargs):
        captured.update(kwargs)
        return EvidencePack(query=query, backend="brave", grounded=False,
                            error={"type": "no_results"})

    monkeypatch.setattr(sibyl_tools, "fetch_via_brave_search", fake_fetch)
    sibyl_tools.brave_search("test query", date.today())
    assert "freshness_override" in captured
    assert captured["freshness_override"].endswith(f"to{date.today().isoformat()}")


# --- backtest gating ----------------------------------------------------------

def test_is_backtest_gate():
    assert is_backtest(PAST_AS_OF, today=TODAY) is True
    assert is_backtest(TODAY, today=TODAY) is False


def test_post_asof_sources_dropped_only_in_backtest():
    sources = [
        _src("https://news.example.com/a", date_str="2025-06-01"),  # after asOf
        _src("https://news.example.com/b", date_str="2025-02-01"),  # before asOf
    ]
    kept, stats = filter_sources(sources, PAST_AS_OF, today=TODAY)
    assert [s.url for s in kept] == ["https://news.example.com/b"]
    assert stats.dropped_post_asof == 1

    # Live mode (asOf = today): nothing to leak, classifier inert.
    live_sources = [_src("https://news.example.com/a", date_str=TODAY.isoformat())]
    kept_live, stats_live = filter_sources(live_sources, TODAY, today=TODAY)
    assert len(kept_live) == 1
    assert stats_live.dropped_post_asof == 0


def test_snippet_classifier_catches_post_asof_dates_in_text():
    sources = [
        _src("https://news.example.com/c", summary="Fighting escalated in June 2025 ..."),
    ]
    kept, stats = filter_sources(sources, PAST_AS_OF, today=TODAY)
    assert kept == []
    assert stats.dropped_post_asof == 1


def test_snippet_leak_detection_formats():
    assert snippet_leaks("as of 2025-06-15 the toll rose", PAST_AS_OF) is True
    assert snippet_leaks("reported on March 3, 2026", PAST_AS_OF) is True
    assert snippet_leaks("back in January 2024 the situation", PAST_AS_OF) is False
    assert snippet_leaks("no dates at all here", PAST_AS_OF) is False


def test_extract_dates_parses_iso_and_text_forms():
    dates = extract_dates("On 2025-05-02 and again on June 9, 2025, floods hit.")
    assert date(2025, 5, 2) in dates
    assert date(2025, 6, 9) in dates


# --- resolution-source blocking ------------------------------------------------

@pytest.mark.parametrize(
    "url",
    [
        "https://acleddata.com/dashboard",
        "https://www.acleddata.com/x",
        "https://go.ifrc.org/emergencies/1",
        "https://www.internal-displacement.org/countries/ETH",
        "https://gdacs.org/report",
        "https://fews.net/east-africa",
        "https://fdw.fews.net/api/x",
        "https://www.ipcinfo.org/analysis",
    ],
)
def test_resolution_source_urls_blocked(url):
    assert is_blocked_url(url) is True


def test_open_web_urls_not_blocked():
    assert is_blocked_url("https://reuters.com/africa/x") is False
    assert is_blocked_url("https://reliefweb.int/report/x") is False
    # Similar-but-different hostnames must not be caught by suffix matching.
    assert is_blocked_url("https://notacleddata.com/x") is False


def test_blocked_domains_dropped_in_live_mode_too():
    kept, stats = filter_sources(
        [_src("https://acleddata.com/data", date_str=TODAY.isoformat())],
        TODAY,
        today=TODAY,
    )
    assert kept == []
    assert stats.dropped_blocked_domain == 1


def test_fetch_url_refuses_blocked_domain():
    result = sibyl_tools.fetch_url("https://go.ifrc.org/emergencies/1", TODAY)
    assert result.ok is False
    assert result.error == "blocked_domain"
    assert result.leakage.dropped_blocked_domain == 1


# --- residual leak accounting ---------------------------------------------------

def test_residual_leak_rate_and_notes():
    sources = [
        _src("https://a.example.com", date_str="2025-06-01"),
        _src("https://b.example.com", date_str="2025-01-01"),
        _src("https://c.example.com", date_str="2025-07-01"),
        _src("https://d.example.com", date_str="2024-12-01"),
    ]
    kept, stats = filter_sources(sources, PAST_AS_OF, today=TODAY)
    assert len(kept) == 2
    assert stats.total_retrieved == 4
    assert stats.dropped_post_asof == 2
    assert stats.residual_leak_rate == pytest.approx(0.5)
    assert stats.notes, "leak drops must be logged for backtest audits"
    assert stats.to_dict()["residual_leak_rate"] == pytest.approx(0.5)


def test_leakage_stats_merge():
    a = LeakageStats(total_retrieved=3, dropped_post_asof=1)
    b = LeakageStats(total_retrieved=2, dropped_blocked_domain=1, notes=["x"])
    a.merge(b)
    assert a.total_retrieved == 5
    assert a.dropped_post_asof == 1
    assert a.dropped_blocked_domain == 1
    assert a.notes == ["x"]


# --- live-lookup clamp (extension point) ----------------------------------------

def test_clamp_live_lookup_date():
    assert clamp_live_lookup_date(date(2026, 1, 1), PAST_AS_OF) == PAST_AS_OF
    assert clamp_live_lookup_date(date(2025, 1, 1), PAST_AS_OF) == date(2025, 1, 1)


# --- relative page_age parsing + undatable-source policy ----------------------

def test_parse_source_date_relative_ages():
    from sibyl.leakage import _parse_source_date

    assert _parse_source_date("2 weeks ago", today=TODAY) == date(2026, 6, 23)
    assert _parse_source_date("3 days ago", today=TODAY) == date(2026, 7, 4)
    assert _parse_source_date("5 hours ago", today=TODAY) == TODAY
    assert _parse_source_date("1 month ago", today=TODAY) == date(2026, 6, 7)
    assert _parse_source_date("1 year ago", today=TODAY) == date(2025, 7, 7)
    assert _parse_source_date("2025-02-01", today=TODAY) == date(2025, 2, 1)
    assert _parse_source_date("gibberish", today=TODAY) is None
    assert _parse_source_date(None, today=TODAY) is None


def test_relative_page_age_dropped_in_backtest():
    """Brave often reports page_age as a relative form; '2 weeks ago' relative
    to the fetch time is post-asOf in a backtest and must be dropped."""
    sources = [
        _src("https://news.example.com/rel", date_str="2 weeks ago"),
        _src("https://news.example.com/old", date_str="2025-02-01"),
    ]
    kept, stats = filter_sources(sources, PAST_AS_OF, today=TODAY)
    assert [s.url for s in kept] == ["https://news.example.com/old"]
    assert stats.dropped_post_asof == 1


def test_undatable_source_dropped_in_backtest_only():
    """A populated-but-unparseable date field must not bypass the post-asOf
    filter in backtest; sources with NO date stay (freshness range + snippet
    classifier still guard them)."""
    sources = [
        _src("https://news.example.com/mystery", date_str="a while back"),
        _src("https://news.example.com/undated", date_str=None),
    ]
    kept, stats = filter_sources(sources, PAST_AS_OF, today=TODAY)
    assert [s.url for s in kept] == ["https://news.example.com/undated"]
    assert stats.dropped_undatable == 1

    # Live mode: unparseable dates are fine, nothing to leak.
    kept_live, stats_live = filter_sources(sources, TODAY, today=TODAY)
    assert len(kept_live) == 2
    assert stats_live.dropped_undatable == 0
