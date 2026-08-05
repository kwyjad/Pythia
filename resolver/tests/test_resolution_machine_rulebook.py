# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Rulebook loader tests — thresholds live in YAML, never in code."""

from __future__ import annotations

import pytest

from resolver.resolution_machine.rulebook import load_rulebook

# Every key Phase 1 code reads.  A missing key must raise at read time
# (no silent code-side defaults), so this list doubles as the contract.
_CONSUMED_KEYS = [
    "version",
    "freeze.days_after_month_end",
    "universe.countries_csv",
    "resolution.metric",
    "resolution.unit",
    "cyclone.hazard_code",
    "cyclone.min_wind_kt",
    "cyclone.buffer_km",
    "cyclone.wind_source_priority",
    "cyclone.ibtracs.url_template",
    "cyclone.ibtracs.default_scope",
    "cyclone.ibtracs.request_timeout_sec",
    "cyclone.ibtracs.coverage_grace_days",
    "cyclone.reliefweb_sweep.disaster_types",
    "cyclone.reliefweb_sweep.keywords",
    "cyclone.reliefweb_sweep.publication_pad_days",
    "cyclone.reliefweb_sweep.max_hits_for_silence",
    "cyclone.reliefweb_sweep.sample_size",
    "cyclone.reliefweb_sweep.request_timeout_sec",
    "cyclone.reliefweb_sweep.request_delay_sec",
    "reliefweb.api_base_url",
]


def test_rulebook_carries_every_consumed_key():
    rb = load_rulebook()
    for key in _CONSUMED_KEYS:
        assert rb[key] is not None, f"rulebook missing {key}"


def test_thresholds_are_sane_numbers():
    rb = load_rulebook()
    assert float(rb["cyclone.min_wind_kt"]) > 0
    assert float(rb["cyclone.buffer_km"]) > 0
    assert int(rb["freeze.days_after_month_end"]) > 0
    assert int(rb["cyclone.ibtracs.coverage_grace_days"]) >= 0
    assert rb["cyclone.wind_source_priority"], "wind priority must be non-empty"


def test_missing_key_raises_not_defaults():
    rb = load_rulebook()
    with pytest.raises(KeyError):
        rb["cyclone.no_such_threshold"]
    assert rb.get("cyclone.no_such_threshold") is None


def test_no_credentials_in_rulebook():
    # Hard rule 7: keys come from env vars, never config.
    import re

    from resolver.resolution_machine.rulebook import RULEBOOK_PATH

    text = RULEBOOK_PATH.read_text(encoding="utf-8").lower()
    assert not re.search(r"(api[_-]?key|token|password|secret)\s*:", text)
