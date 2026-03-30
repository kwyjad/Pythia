# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Tests for the LLM provider credit-retry wrapper."""

from __future__ import annotations

import asyncio
from unittest.mock import patch, MagicMock

import pytest

from forecaster.providers import (
    ModelSpec,
    ProviderBillingError,
    ProviderResult,
    _credit_retry_config_for,
    _is_billing_error,
    usage_to_dict,
)


# ---------------------------------------------------------------------------
# _is_billing_error tests
# ---------------------------------------------------------------------------

class TestIsBillingError:
    """Verify billing error detection per provider."""

    def test_openai_quota_429(self):
        assert _is_billing_error(
            "openai",
            "OpenAI HTTP 429: You exceeded your current quota",
            status_code=429,
        ) is True

    def test_openai_insufficient_quota_429(self):
        assert _is_billing_error(
            "openai",
            "OpenAI HTTP 429: insufficient_quota",
            status_code=429,
        ) is True

    def test_openai_billing_429(self):
        assert _is_billing_error(
            "openai",
            "OpenAI HTTP 429: billing issue on account",
            status_code=429,
        ) is True

    def test_openai_rate_limit_429_not_billing(self):
        """Rate limit 429s should NOT trigger credit retry."""
        assert _is_billing_error(
            "openai",
            "OpenAI HTTP 429: Rate limit exceeded",
            status_code=429,
        ) is False

    def test_openai_non_429(self):
        assert _is_billing_error(
            "openai",
            "OpenAI HTTP 500: server error",
            status_code=500,
        ) is False

    def test_anthropic_insufficient_credits(self):
        assert _is_billing_error(
            "anthropic",
            "Anthropic HTTP 400: insufficient credits",
            status_code=400,
        ) is True

    def test_anthropic_billing_403(self):
        assert _is_billing_error(
            "anthropic",
            "Anthropic HTTP 403: billing issue",
            status_code=403,
        ) is True

    def test_anthropic_blocked_403(self):
        assert _is_billing_error(
            "anthropic",
            "Anthropic HTTP 403: account blocked",
            status_code=403,
        ) is True

    def test_anthropic_rate_limit_429_not_billing(self):
        assert _is_billing_error(
            "anthropic",
            "Anthropic HTTP 429: rate limited",
            status_code=429,
        ) is False

    def test_google_resource_exhausted_billing(self):
        assert _is_billing_error(
            "google",
            "Google HTTP 429: RESOURCE_EXHAUSTED quota exceeded for billing account",
            status_code=429,
        ) is True

    def test_google_rate_limit_429_not_billing(self):
        """Google RPM/TPM rate limits should NOT trigger credit retry."""
        assert _is_billing_error(
            "google",
            "Google HTTP 429: Too Many Requests",
            status_code=429,
        ) is False

    def test_google_resource_exhausted_no_billing_keyword(self):
        """RESOURCE_EXHAUSTED without quota/billing keywords should not match."""
        assert _is_billing_error(
            "google",
            "Google HTTP 429: RESOURCE_EXHAUSTED rate limit",
            status_code=429,
        ) is False

    def test_kimi_not_supported(self):
        assert _is_billing_error(
            "kimi",
            "Kimi HTTP 429: quota exceeded",
            status_code=429,
        ) is False

    def test_deepseek_not_supported(self):
        assert _is_billing_error(
            "deepseek",
            "DeepSeek HTTP 429: billing error",
            status_code=429,
        ) is False

    def test_empty_error(self):
        assert _is_billing_error("openai", "", status_code=429) is False

    def test_none_error(self):
        assert _is_billing_error("openai", None, status_code=429) is False


# ---------------------------------------------------------------------------
# _credit_retry_config_for tests
# ---------------------------------------------------------------------------

class TestCreditRetryConfig:
    """Verify credit retry config lookup and env-var override."""

    def test_openai_defaults(self):
        cfg = _credit_retry_config_for("openai")
        assert cfg == (900, 3)

    def test_anthropic_defaults(self):
        cfg = _credit_retry_config_for("anthropic")
        assert cfg == (300, 3)

    def test_google_defaults(self):
        cfg = _credit_retry_config_for("google")
        assert cfg == (600, 3)

    def test_kimi_returns_none(self):
        assert _credit_retry_config_for("kimi") is None

    def test_deepseek_returns_none(self):
        assert _credit_retry_config_for("deepseek") is None

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("PYTHIA_CREDIT_RETRY_PAUSE_OPENAI", "600")
        monkeypatch.setenv("PYTHIA_CREDIT_RETRY_MAX_OPENAI", "5")
        cfg = _credit_retry_config_for("openai")
        assert cfg == (600, 5)


# ---------------------------------------------------------------------------
# ProviderBillingError tests
# ---------------------------------------------------------------------------

class TestProviderBillingError:
    def test_attributes(self):
        exc = ProviderBillingError("openai", "quota exceeded", status_code=429)
        assert exc.provider == "openai"
        assert exc.status_code == 429
        assert str(exc) == "quota exceeded"


# ---------------------------------------------------------------------------
# call_chat_ms credit retry integration tests
# ---------------------------------------------------------------------------

def _run_async(coro):
    """Run an async function synchronously for testing."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestCallChatMsCreditRetry:
    """Integration tests for credit-retry loop inside call_chat_ms."""

    def _make_model_spec(self, provider="openai", model_id="gpt-5.2"):
        return ModelSpec(
            name="test-model",
            provider=provider,
            model_id=model_id,
            weight=1.0,
            active=True,
        )

    def _make_billing_error_result(self, model_id="gpt-5.2"):
        return ProviderResult(
            text="",
            usage=usage_to_dict(None),
            cost_usd=0.0,
            model_id=model_id,
            error="OpenAI HTTP 429: You exceeded your current quota",
        )

    def _make_success_result(self, model_id="gpt-5.2"):
        return ProviderResult(
            text='{"buckets": [0.1, 0.2, 0.3, 0.2, 0.2]}',
            usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            cost_usd=0.01,
            model_id=model_id,
        )

    def test_credit_retry_recovery(self, monkeypatch):
        """Billing error on first attempt, success after credit retry."""
        monkeypatch.setenv("PYTHIA_CREDIT_RETRY_PAUSE_OPENAI", "0")
        monkeypatch.setenv("PYTHIA_LLM_RETRIES", "1")
        ms = self._make_model_spec()
        billing_result = self._make_billing_error_result()
        success_result = self._make_success_result()

        call_count = 0

        def mock_call_sync(provider, prompt, model, temp, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return billing_result
            return success_result

        from forecaster.providers import call_chat_ms

        with patch("forecaster.providers._call_provider_sync", side_effect=mock_call_sync), \
             patch("forecaster.providers._log_llm_call"):
            text, usage, error = _run_async(call_chat_ms(ms, "test prompt", run_id="test_run"))

        assert call_count == 2
        assert usage.get("credit_retries_used", 0) > 0
        assert usage.get("billing_error_detected") is True

    def test_credit_retry_exhausted(self, monkeypatch):
        """All credit retries exhausted — should give up."""
        monkeypatch.setenv("PYTHIA_CREDIT_RETRY_PAUSE_OPENAI", "0")
        monkeypatch.setenv("PYTHIA_CREDIT_RETRY_MAX_OPENAI", "1")
        monkeypatch.setenv("PYTHIA_LLM_RETRIES", "1")
        ms = self._make_model_spec()
        billing_result = self._make_billing_error_result()

        def mock_call_sync(provider, prompt, model, temp, **kwargs):
            return billing_result

        from forecaster.providers import call_chat_ms

        with patch("forecaster.providers._call_provider_sync", side_effect=mock_call_sync), \
             patch("forecaster.providers._log_llm_call"):
            text, usage, error = _run_async(call_chat_ms(ms, "test prompt", run_id="test_run"))

        assert error
        assert usage.get("billing_error_detected") is True
        assert usage.get("credit_retries_used", 0) == 1

    def test_kimi_no_credit_retry(self, monkeypatch):
        """Kimi errors should NOT trigger credit retry."""
        monkeypatch.setenv("PYTHIA_LLM_RETRIES", "1")
        ms = self._make_model_spec(provider="kimi", model_id="kimi-k2.5")
        billing_result = ProviderResult(
            text="",
            usage=usage_to_dict(None),
            cost_usd=0.0,
            model_id="kimi-k2.5",
            error="Kimi HTTP 429: quota exceeded billing",
        )

        call_count = 0

        def mock_call_sync(provider, prompt, model, temp, **kwargs):
            nonlocal call_count
            call_count += 1
            return billing_result

        from forecaster.providers import call_chat_ms

        with patch("forecaster.providers._call_provider_sync", side_effect=mock_call_sync), \
             patch("forecaster.providers._log_llm_call"):
            text, usage, error = _run_async(call_chat_ms(ms, "test prompt", run_id="test_run"))

        assert call_count == 1
        assert usage.get("credit_retries_used", 0) == 0

    def test_rate_limit_no_credit_retry(self, monkeypatch):
        """Rate-limit 429 should NOT trigger credit retry (only transient retry)."""
        monkeypatch.setenv("PYTHIA_CREDIT_RETRY_PAUSE_OPENAI", "0")
        monkeypatch.setenv("PYTHIA_LLM_RETRIES", "1")
        ms = self._make_model_spec()

        rate_limit_result = ProviderResult(
            text="",
            usage=usage_to_dict(None),
            cost_usd=0.0,
            model_id="gpt-5.2",
            error="OpenAI HTTP 429: Rate limit exceeded. Try again in 2s.",
        )

        call_count = 0

        def mock_call_sync(provider, prompt, model, temp, **kwargs):
            nonlocal call_count
            call_count += 1
            return rate_limit_result

        from forecaster.providers import call_chat_ms

        with patch("forecaster.providers._call_provider_sync", side_effect=mock_call_sync), \
             patch("forecaster.providers._log_llm_call"):
            text, usage, error = _run_async(call_chat_ms(ms, "test prompt", run_id="test_run"))

        assert usage.get("credit_retries_used", 0) == 0
