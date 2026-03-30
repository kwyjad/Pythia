# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Tests for Brave Search circuit breaker."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from pythia.web_research.brave_circuit_breaker import (
    BraveCircuitBreaker,
    get_breaker,
    is_tripped,
    reset,
)


# ---------------------------------------------------------------------------
# BraveCircuitBreaker unit tests
# ---------------------------------------------------------------------------

class TestBraveCircuitBreaker:
    """Core circuit breaker logic."""

    def test_not_tripped_initially(self):
        cb = BraveCircuitBreaker(threshold=3)
        assert cb.is_tripped() is False

    def test_trips_after_threshold_failures(self):
        cb = BraveCircuitBreaker(threshold=3)
        cb.record_failure(429, "rate limit")
        assert cb.is_tripped() is False
        cb.record_failure(429, "rate limit")
        assert cb.is_tripped() is False
        cb.record_failure(402, "payment required")
        assert cb.is_tripped() is True

    def test_success_resets_consecutive_count(self):
        cb = BraveCircuitBreaker(threshold=3)
        cb.record_failure(429, "error")
        cb.record_failure(429, "error")
        cb.record_success()
        # Counter reset — need 3 more consecutive failures
        cb.record_failure(429, "error")
        assert cb.is_tripped() is False
        cb.record_failure(429, "error")
        assert cb.is_tripped() is False

    def test_reset_clears_tripped(self):
        cb = BraveCircuitBreaker(threshold=2)
        cb.record_failure(429, "error")
        cb.record_failure(429, "error")
        assert cb.is_tripped() is True
        cb.reset()
        assert cb.is_tripped() is False

    def test_stats(self):
        cb = BraveCircuitBreaker(threshold=3)
        cb.record_success()
        cb.record_failure(429, "error")
        cb.record_failure(402, "error")
        stats = cb.stats()
        assert stats["tripped"] is False
        assert stats["consecutive_failures"] == 2
        assert stats["total_failures"] == 2
        assert stats["total_successes"] == 1
        assert stats["threshold"] == 3

    def test_stays_tripped_after_more_failures(self):
        cb = BraveCircuitBreaker(threshold=2)
        cb.record_failure(429, "e1")
        cb.record_failure(429, "e2")
        assert cb.is_tripped() is True
        cb.record_failure(429, "e3")
        assert cb.is_tripped() is True

    def test_threshold_of_one(self):
        cb = BraveCircuitBreaker(threshold=1)
        cb.record_failure(500, "server error")
        assert cb.is_tripped() is True


# ---------------------------------------------------------------------------
# Module-level singleton tests
# ---------------------------------------------------------------------------

class TestModuleSingleton:
    def setup_method(self):
        reset()

    def test_singleton_not_tripped(self):
        assert is_tripped() is False

    def test_singleton_trips_and_resets(self):
        breaker = get_breaker()
        breaker.record_failure(429, "e1")
        breaker.record_failure(429, "e2")
        breaker.record_failure(429, "e3")
        assert is_tripped() is True
        reset()
        assert is_tripped() is False


# ---------------------------------------------------------------------------
# brave_search.py integration tests
# ---------------------------------------------------------------------------

class TestBraveSearchIntegration:
    """Verify circuit breaker wiring in brave_search.py."""

    def setup_method(self):
        reset()

    def test_fetch_returns_immediately_when_tripped(self):
        """When breaker is tripped, fetch_via_brave_search should short-circuit."""
        breaker = get_breaker()
        # Trip the breaker
        for _ in range(3):
            breaker.record_failure(429, "budget exhausted")
        assert breaker.is_tripped() is True

        from pythia.web_research.backends.brave_search import fetch_via_brave_search
        pack = fetch_via_brave_search(
            "test query",
            recency_days=30,
            include_structural=True,
            timeout_sec=10,
            max_results=5,
        )
        assert pack.grounded is False
        assert pack.error is not None
        assert pack.error["type"] == "circuit_breaker_tripped"
        assert "brave_circuit_breaker_tripped" in pack.debug.get("grounding_backend", "")

    def test_fetch_proceeds_when_not_tripped(self):
        """When breaker is NOT tripped, normal flow should proceed (may fail on missing API key)."""
        assert is_tripped() is False
        from pythia.web_research.backends.brave_search import fetch_via_brave_search

        # Without API key, should get missing_api_key error (not circuit_breaker)
        with patch.dict("os.environ", {"BRAVE_SEARCH_API_KEY": ""}, clear=False):
            pack = fetch_via_brave_search(
                "test query",
                recency_days=30,
                include_structural=True,
                timeout_sec=10,
                max_results=5,
            )
        assert pack.error is not None
        assert pack.error["type"] == "missing_api_key"

    def test_run_single_query_records_failure(self):
        """_run_single_query should record failures to the breaker."""
        reset()
        breaker = get_breaker()
        assert breaker.stats()["total_failures"] == 0

        from pythia.web_research.backends.brave_search import _run_single_query

        # Mock requests.get to return a non-200 status
        mock_resp = MagicMock()
        mock_resp.status_code = 402
        with patch("pythia.web_research.backends.brave_search.requests.get", return_value=mock_resp), \
             patch("pythia.web_research.backends.brave_search._brave_limiter"):
            results, code = _run_single_query("test", "fake_key", "pm", 10)

        assert code == 402
        assert results == []
        assert breaker.stats()["total_failures"] == 1

    def test_run_single_query_records_success(self):
        """_run_single_query should record successes to the breaker."""
        reset()
        breaker = get_breaker()

        from pythia.web_research.backends.brave_search import _run_single_query

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"web": {"results": [{"url": "http://example.com", "title": "Test"}]}}
        with patch("pythia.web_research.backends.brave_search.requests.get", return_value=mock_resp), \
             patch("pythia.web_research.backends.brave_search._brave_limiter"):
            results, code = _run_single_query("test", "fake_key", "pm", 10)

        assert code == 200
        assert len(results) == 1
        assert breaker.stats()["total_successes"] == 1


# ---------------------------------------------------------------------------
# _write_hs_triage integration test
# ---------------------------------------------------------------------------

class TestWriteHsTriageBraveGate:
    """Verify that _write_hs_triage blocks ungrounded hazards when breaker is tripped."""

    def setup_method(self):
        reset()

    def test_blocks_ungrounded_when_tripped(self):
        """When breaker is tripped and hazard has no grounding, need_full_spd should be False."""
        breaker = get_breaker()
        for _ in range(3):
            breaker.record_failure(429, "budget")
        assert breaker.is_tripped() is True

        # Mock the DB calls to capture what gets written
        written_payloads = []

        def mock_execute(sql, params=None):
            if "INSERT INTO hs_triage" in sql and params:
                written_payloads.append(params)
            return MagicMock(fetchall=lambda: [], fetchone=lambda: None)

        mock_con = MagicMock()
        mock_con.execute = mock_execute

        from horizon_scanner.horizon_scanner import _write_hs_triage

        # Mock dependencies
        with patch("horizon_scanner.horizon_scanner.pythia_connect", return_value=mock_con), \
             patch("horizon_scanner.horizon_scanner._build_hazard_catalog", return_value={"ACE": {}, "DR": {}, "FL": {}, "TC": {}}), \
             patch("horizon_scanner.horizon_scanner.get_expected_hs_hazards", return_value=["ACE", "DR", "FL", "TC"]), \
             patch("horizon_scanner.horizon_scanner._hazard_has_grounding", return_value=False):

            triage = {
                "country": "IRQ",
                "hazards": {
                    "ACE": {
                        "triage_score": 0.8,
                        "drivers": [],
                        "regime_shifts": [],
                        "data_quality": {},
                        "scenario_stub": "",
                        "regime_change": {
                            "likelihood": 0.5,
                            "magnitude": 0.6,
                            "direction": "up",
                            "window": "3m",
                        },
                    },
                },
            }
            _write_hs_triage("test_run", "IRQ", triage, is_test=True)

        # Find the ACE row — need_full_spd should be False because no grounding + breaker tripped
        ace_rows = [p for p in written_payloads if len(p) >= 3 and p[2] == "ACE"]
        if ace_rows:
            # need_full_spd is at index 5 in the insert payload
            assert ace_rows[0][5] is False, "need_full_spd should be False for ungrounded hazard with tripped breaker"

    def test_allows_grounded_when_tripped(self):
        """When breaker is tripped but hazard HAS grounding, need_full_spd should stay True."""
        breaker = get_breaker()
        for _ in range(3):
            breaker.record_failure(429, "budget")
        assert breaker.is_tripped() is True

        written_payloads = []

        def mock_execute(sql, params=None):
            if "INSERT INTO hs_triage" in sql and params:
                written_payloads.append(params)
            return MagicMock(fetchall=lambda: [], fetchone=lambda: None)

        mock_con = MagicMock()
        mock_con.execute = mock_execute

        from horizon_scanner.horizon_scanner import _write_hs_triage

        with patch("horizon_scanner.horizon_scanner.pythia_connect", return_value=mock_con), \
             patch("horizon_scanner.horizon_scanner._build_hazard_catalog", return_value={"ACE": {}}), \
             patch("horizon_scanner.horizon_scanner.get_expected_hs_hazards", return_value=["ACE"]), \
             patch("horizon_scanner.horizon_scanner._hazard_has_grounding", return_value=True):

            triage = {
                "country": "IRQ",
                "hazards": {
                    "ACE": {
                        "triage_score": 0.8,
                        "drivers": [],
                        "regime_shifts": [],
                        "data_quality": {},
                        "scenario_stub": "",
                        "regime_change": {
                            "likelihood": 0.5,
                            "magnitude": 0.6,
                            "direction": "up",
                            "window": "3m",
                        },
                    },
                },
            }
            _write_hs_triage("test_run", "IRQ", triage, is_test=True)

        ace_rows = [p for p in written_payloads if len(p) >= 3 and p[2] == "ACE"]
        if ace_rows:
            # need_full_spd should remain True because grounding exists
            assert ace_rows[0][5] is True, "need_full_spd should be True for grounded hazard"
