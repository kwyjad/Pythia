# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Adversarial stats separation in the debug bundle (July 2026, run-2 review).

The ADVERSARIAL_SYNTH_{hz} generation rows (added so synthesis is visible in
telemetry) matched every ADVERSARIAL_% LIKE filter in the bundle, so an
otherwise clean run reported "Adversarial Checks | WARN | 8 calls, 2 errors"
— the two synthesis calls were counted as 0-source failed searches. These
tests pin: search stats exclude SYNTH rows; synthesis is reported by its own
loader.
"""

from __future__ import annotations

import json

import pytest

duckdb = pytest.importorskip("duckdb")

from scripts.dump_pythia_debug_bundle import (
    _load_adversarial_synth_stats,
    _load_grounding_call_stats,
    _load_grounding_subsystem_stats,
)

HS_RUN = "hs_test"


def _seed_llm_calls(con) -> None:
    con.execute(
        """
        CREATE TABLE llm_calls (
            call_id TEXT, run_id TEXT, hs_run_id TEXT, phase TEXT,
            provider TEXT, model_id TEXT, hazard_code TEXT, iso3 TEXT,
            prompt_text TEXT, response_text TEXT, error_text TEXT,
            cost_usd DOUBLE, elapsed_ms INTEGER,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    brave_pack = json.dumps({"query": "q", "grounded": True, "n_sources": 10, "source_urls": []})
    rows = [
        # Three Brave adversarial SEARCH calls — all with sources.
        ("c1", HS_RUN, "hs_triage", "brave", "brave-web-search", "ADVERSARIAL_ACE", brave_pack, "", 0.005),
        ("c2", HS_RUN, "hs_triage", "brave", "brave-web-search", "ADVERSARIAL_ACE", brave_pack, "", 0.005),
        ("c3", HS_RUN, "hs_triage", "brave", "brave-web-search", "ADVERSARIAL_ACE", brave_pack, "", 0.005),
        # One synthesis GENERATION call — no sources by nature.
        ("c4", HS_RUN, "hs_triage", "google", "gemini-3.5-flash", "ADVERSARIAL_SYNTH_ACE",
         '{"counter_evidence": []}', "", 0.0128),
    ]
    for call_id, hs_run_id, phase, provider, model_id, hz, resp, err, cost in rows:
        con.execute(
            "INSERT INTO llm_calls (call_id, hs_run_id, phase, provider, model_id, "
            "hazard_code, response_text, error_text, cost_usd) VALUES (?,?,?,?,?,?,?,?,?)",
            [call_id, hs_run_id, phase, provider, model_id, hz, resp, err, cost],
        )


def test_search_stats_exclude_synth_rows() -> None:
    con = duckdb.connect(":memory:")
    _seed_llm_calls(con)

    stats = _load_grounding_call_stats(con, "hs_triage", "ADVERSARIAL_%", None, HS_RUN)
    assert stats["n_calls"] == 3
    assert stats["n_errors"] == 0  # the 0-source synthesis row must not count

    rows = _load_grounding_subsystem_stats(con, "ADVERSARIAL_%", None, HS_RUN)
    assert sum(r["n_calls"] for r in rows) == 3
    assert all("gemini" not in (r["model_id"] or "") for r in rows)


def test_synth_stats_reported_separately() -> None:
    con = duckdb.connect(":memory:")
    _seed_llm_calls(con)

    synth = _load_adversarial_synth_stats(con, None, HS_RUN)
    assert synth["n_calls"] == 1
    assert synth["n_errors"] == 0
    assert synth["cost_usd"] == pytest.approx(0.0128)


def test_synth_stats_count_errors() -> None:
    con = duckdb.connect(":memory:")
    _seed_llm_calls(con)
    con.execute(
        "INSERT INTO llm_calls (call_id, hs_run_id, phase, provider, model_id, "
        "hazard_code, response_text, error_text, cost_usd) VALUES "
        "('c5', ?, 'hs_triage', 'google', 'gemini-3.5-flash', "
        "'ADVERSARIAL_SYNTH_DR', '', 'timeout after 60s', 0.0)",
        [HS_RUN],
    )
    synth = _load_adversarial_synth_stats(con, None, HS_RUN)
    assert synth["n_calls"] == 2
    assert synth["n_errors"] == 1
