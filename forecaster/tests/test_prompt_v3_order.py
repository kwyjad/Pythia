# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Tests for the V3 (static-first) prompt ordering and cache plumbing.

Invariants pinned here:

* Legacy order (flag off) remains the default and keeps its historical
  structure (data first, method/schema last).
* V3 order contains exactly the same content lines as legacy (modulo the
  documented positional-referent flips and the two glue lines).
* Under V3, the prompt prefix is byte-identical across questions/countries
  of the same group — the property provider prefix caching depends on.
* Anthropic cache_control blocks are emitted only when
  PYTHIA_PROMPT_CACHE_ENABLED is on, at most 4, above the size guard, and
  their concatenation equals the plain prompt.
"""

from __future__ import annotations

import collections

import pytest


_Q_SOM = {
    "iso3": "SOM",
    "hazard_code": "ACE",
    "metric": "FATALITIES",
    "wording": "How many conflict fatalities will Somalia record?",
    "window_start_date": "2026-08-01",
    "target_month": "2027-01",
}
_Q_YEM = {
    "iso3": "YEM",
    "hazard_code": "ACE",
    "metric": "FATALITIES",
    "wording": "How many conflict fatalities will Yemen record?",
    "window_start_date": "2026-08-01",
    "target_month": "2027-01",
}
_HIST = {"source": "acled", "summary": "24 months of data"}
_TRIAGE = {"regime_change_level": 1, "triage_score": 0.6}


def _line_multiset(text: str, drop_substrings: tuple[str, ...]) -> collections.Counter:
    return collections.Counter(
        line
        for line in text.splitlines()
        if line.strip() and not any(d in line for d in drop_substrings)
    )


# The only allowed text differences between orders: positional referents and
# the V3 glue lines.
_SPD_ORDER_VARIANT_MARKERS = (
    "history summary above",
    "history summary below",
    "The QUESTION DATA to forecast follows below.",
    "END OF QUESTION DATA.",
    "Now apply the FORECASTING METHOD above",
)


def test_spd_v3_same_content_as_legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    from forecaster.prompts import build_spd_prompt_v2

    monkeypatch.delenv("PYTHIA_PROMPT_V3_ORDER", raising=False)
    legacy = build_spd_prompt_v2(_Q_SOM, _HIST, _TRIAGE, {})
    assert legacy.startswith("You are a careful probabilistic forecaster")
    assert legacy.rstrip().endswith("Do not include any text outside the JSON.")
    assert "history summary above" in legacy

    monkeypatch.setenv("PYTHIA_PROMPT_V3_ORDER", "1")
    v3 = build_spd_prompt_v2(_Q_SOM, _HIST, _TRIAGE, {})
    assert "history summary below" in v3
    assert (
        _line_multiset(legacy, _SPD_ORDER_VARIANT_MARKERS)
        == _line_multiset(v3, _SPD_ORDER_VARIANT_MARKERS)
    )


def test_spd_return_parts_joins_to_full_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    from forecaster.prompts import build_spd_prompt_v2

    monkeypatch.setenv("PYTHIA_PROMPT_V3_ORDER", "1")
    full = build_spd_prompt_v2(_Q_SOM, _HIST, _TRIAGE, {})
    prefix, suffix = build_spd_prompt_v2(_Q_SOM, _HIST, _TRIAGE, {}, return_parts=True)
    assert prefix + suffix == full
    assert len(prefix) > 4000

    # Legacy order: no usefully cacheable prefix.
    monkeypatch.delenv("PYTHIA_PROMPT_V3_ORDER", raising=False)
    prefix, suffix = build_spd_prompt_v2(_Q_SOM, _HIST, _TRIAGE, {}, return_parts=True)
    assert prefix == ""
    assert suffix == build_spd_prompt_v2(_Q_SOM, _HIST, _TRIAGE, {})


def test_spd_v3_prefix_stable_within_group(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prefix must be byte-identical across questions of one (hazard, metric, track)."""
    from forecaster.prompts import build_spd_prompt_v2

    monkeypatch.setenv("PYTHIA_PROMPT_V3_ORDER", "1")
    pre_a, _ = build_spd_prompt_v2(_Q_SOM, _HIST, _TRIAGE, {}, return_parts=True)
    pre_b, _ = build_spd_prompt_v2(
        _Q_YEM, {"source": "none"}, {"regime_change_level": 2}, {}, return_parts=True
    )
    assert pre_a == pre_b
    assert "SOM" not in pre_a and "Somalia" not in pre_a


def test_rc_v3_prefix_stable_across_countries(monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    from horizon_scanner.rc_prompts import build_rc_prompt_ace

    monkeypatch.delenv("PYTHIA_PROMPT_V3_ORDER", raising=False)
    legacy = build_rc_prompt_ace(
        country_name="Somalia", iso3="SOM", resolver_features={"x": 1}
    )
    assert legacy.startswith("You are assessing REGIME CHANGE for Somalia (SOM)")

    monkeypatch.setenv("PYTHIA_PROMPT_V3_ORDER", "1")
    a = build_rc_prompt_ace(country_name="Somalia", iso3="SOM", resolver_features={"x": 1})
    b = build_rc_prompt_ace(country_name="Yemen", iso3="YEM", resolver_features={"y": 2})
    common = len(os.path.commonprefix([a, b]))
    assert common > 5000
    assert "Somalia" not in a[:common]
    assert "COUNTRY: Somalia (SOM)" in a
    assert "END OF COUNTRY DATA" in a


def test_triage_v3_moves_rc_context_to_tail(monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    from horizon_scanner.hs_triage_prompts import build_triage_prompt_ace

    rc = {"likelihood": 0.4, "direction": "up", "magnitude": 0.3}
    monkeypatch.setenv("PYTHIA_PROMPT_V3_ORDER", "1")
    a = build_triage_prompt_ace(
        country_name="Somalia", iso3="SOM", resolver_features={}, rc_result=rc
    )
    b = build_triage_prompt_ace(
        country_name="Yemen", iso3="YEM", resolver_features={}, rc_result=None
    )
    common = len(os.path.commonprefix([a, b]))
    assert common > 4500
    assert "REGIME CHANGE CONTEXT" in a
    assert a.index("REGIME CHANGE CONTEXT") > common


def test_binary_v3_prefix_stable_across_countries(monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    from forecaster.binary_prompts import build_binary_event_prompt

    br = {
        "total_months": 120,
        "event_months": 12,
        "base_rate_pct": 10.0,
        "seasonal_pattern": {},
        "recent_12m_events": 1,
        "recent_12m_rate": 8.3,
        "trend": "stable",
    }
    monkeypatch.setenv("PYTHIA_PROMPT_V3_ORDER", "1")
    a = build_binary_event_prompt(
        question={"iso3": "MOZ", "hazard_code": "TC", "country_name": "Mozambique",
                  "window_start_date": "2026-08-01"},
        base_rate=br, current_alerts=[], structured_data={}, today="2026-07-27",
    )
    b = build_binary_event_prompt(
        question={"iso3": "PHL", "hazard_code": "TC", "country_name": "Philippines",
                  "window_start_date": "2026-08-01"},
        base_rate=br, current_alerts=[], structured_data={}, today="2026-07-27",
    )
    common = len(os.path.commonprefix([a, b]))
    assert common > 3000
    assert "Mozambique" not in a[:common]


def test_anthropic_cache_segments_gated_and_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from forecaster import providers

    big = "x" * 5000
    segments = [(big, False), ("q1" * 1000, True), ("trial", True), ("step", False)]
    prompt = "".join(t for t, _ in segments)

    # Flag off: plain string content, byte-identical to legacy.
    monkeypatch.delenv("PYTHIA_PROMPT_CACHE_ENABLED", raising=False)
    body = providers.build_anthropic_body(
        prompt, "claude-opus-5", 1.0, cache_segments=segments
    )
    assert body["messages"][0]["content"] == prompt

    # Flag on: content blocks, breakpoints marked, concatenation == prompt.
    monkeypatch.setenv("PYTHIA_PROMPT_CACHE_ENABLED", "1")
    body = providers.build_anthropic_body(
        prompt, "claude-opus-5", 1.0, cache_segments=segments
    )
    blocks = body["messages"][0]["content"]
    assert isinstance(blocks, list)
    assert "".join(b["text"] for b in blocks) == prompt
    marked = [b for b in blocks if "cache_control" in b]
    assert len(marked) == 2
    assert all(b["cache_control"] == {"type": "ephemeral"} for b in marked)

    # Size guard: a tiny prefix never gets a marker (below the cacheable min).
    small = [("tiny", True), ("rest", False)]
    body = providers.build_anthropic_body(
        "tinyrest", "claude-opus-5", 1.0, cache_segments=small
    )
    content = body["messages"][0]["content"]
    if isinstance(content, list):
        assert not any("cache_control" in b for b in content)

    # Max 4 breakpoints even if more are requested.
    many = [(f"seg{i}" * 2000, True) for i in range(8)]
    body = providers.build_anthropic_body(
        "".join(t for t, _ in many), "claude-opus-5", 1.0, cache_segments=many
    )
    marked = [b for b in body["messages"][0]["content"] if "cache_control" in b]
    assert len(marked) == 4


def test_openai_prompt_cache_key_gated(monkeypatch: pytest.MonkeyPatch) -> None:
    from forecaster import providers

    monkeypatch.delenv("PYTHIA_PROMPT_CACHE_ENABLED", raising=False)
    body = providers.build_openai_body(
        "p", "gpt-5.6-sol", 0.2, reasoning_effort="high",
        prompt_cache_key="pythia-spd-ACE-FATALITIES-t1",
    )
    assert "prompt_cache_key" not in body

    monkeypatch.setenv("PYTHIA_PROMPT_CACHE_ENABLED", "1")
    body = providers.build_openai_body(
        "p", "gpt-5.6-sol", 0.2, reasoning_effort="high",
        prompt_cache_key="pythia-spd-ACE-FATALITIES-t1",
    )
    assert body["prompt_cache_key"] == "pythia-spd-ACE-FATALITIES-t1"
