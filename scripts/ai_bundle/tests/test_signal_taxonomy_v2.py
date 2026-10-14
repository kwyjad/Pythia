# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Signal taxonomy 2.0 (Oct 2026).

On the 13 Oct 2026 ledger 40% of update signals fell under "other". Half of
that was a bug: every pattern ended in ``\\b``, so a stem such as
``fatalit`` or ``escalat`` could never match a whole word. The rest needed
six new classes. Replayed on that ledger the share is now 2.8%.
"""

from __future__ import annotations

import pytest

from scripts.ai_bundle import build_forecast_attribution_bundle as fab


@pytest.fixture(scope="module")
def tax():
    return fab.load_taxonomy()


@pytest.mark.parametrize("text, klass", [
    # Real update signals from 13 Oct 2026 that read "other" before.
    ("Trailing 3-month fatality de-escalation (-30% relative to prior window)", "recent_observed_trend"),
    ("VIEWS model projections (125-189 deaths/mo) & conflictforecast.org risk score (0.606)", "quantitative_model_forecast"),
    ("GloFAS inactive status", "quantitative_model_forecast"),
    ("CrisisWatch deteriorated arrow for October", "crisiswatch_alert"),
    ("Proactive municipal drainage clearing and early warnings (Counter-evidence)", "adversarial_counterevidence"),
    ("Nonlethal containment and dialogue", "adversarial_counterevidence"),
    ("AU-led mediation and renewed dialogue between the parties", "diplomacy_mediation"),
    ("January: net application of the ranked signals, starting independently from the interpolated January prior",
     "prior_arithmetic_repair"),
    ("Displacement rising in the north", "displacement"),
    ("Climatology for November", "base_rate_reference"),
    ("FL RC upward flag, applied to November", "rc_flag"),
    ("ICPAC dry forecast", "meteorological_forecast"),
])
def test_signals_that_read_other_now_have_a_class(tax, text, klass):
    assert fab.classify_signal(text, tax)[0] == klass


def test_stems_match_whole_words(tax):
    # The trailing \b made every stem unmatchable.
    for word in ("fatalities", "escalation", "displaced", "climatology", "underfunded"):
        assert fab.classify_signal(f"{word} reported", tax)[0] != "other", word


def test_a_signal_carries_a_secondary_class(tax):
    klass, _conf, secondary = fab.classify_signal_full(
        "Adversarial counter-evidence (below-normal rainfall forecast)", tax)
    assert klass == "adversarial_counterevidence"
    assert secondary == "meteorological_forecast"
    assert fab.classify_signal_full("Ceasefire talks resumed", tax)[2] == "diplomacy_mediation"
    assert fab.classify_signal_full("Nothing in particular.", tax) == ("other", 0.1, None)


def test_ceasefire_stays_deescalation_and_the_old_seeds_still_hold(tax):
    assert fab.classify_signal("Ceasefire talks resumed in Nairobi", tax)[0] == "conflict_deescalation"
    assert fab.classify_signal("Escalation of clashes and airstrikes around Mekelle", tax)[0] == "conflict_escalation"
    assert fab.classify_signal("IPC Phase 3+ population rose", tax)[0] == "food_security_phase"


def _q(metric):
    return {"question_id": "Q", "iso3": "SOM", "hazard_code": "FL", "metric": metric}


def test_arithmetic_repair_moves_no_signal_mass(tax):
    trace = {"prior": {"spd": [0.5, 0.5], "rationale": "base rate"},
             "updates": [
                 {"signal": "VIEWS forecasts rising", "delta": [-0.1, 0.1]},
                 {"signal": "Interpolated January prior, renormalised", "delta": [0.05, -0.05]},
             ]}
    rows = fab.ledger_rows_for_member(run_id="fc", hs_run_id=None, q=_q("PA"), model_name="m",
                                      trace=trace, taxonomy=tax, created_at=None, n_buckets=2)
    upd = [r for r in rows if not r["is_prior_row"]]
    assert [r["signal_class"] for r in upd] == ["quantitative_model_forecast", "prior_arithmetic_repair"]
    assert [r["carries_signal_mass"] for r in upd] == [True, False]
    # The row keeps its mass; the class mix leaves it out.
    assert upd[1]["mass_moved_l1"] == pytest.approx(0.05)
    assert fab._class_mix(rows) == {"quantitative_model_forecast": 1.0}


def test_binary_calls_are_not_counted_as_missing_traces(tax):
    rows = fab.ledger_rows_for_member(run_id="fc", hs_run_id=None, q=_q("EVENT_OCCURRENCE"),
                                      model_name="m", trace=None, taxonomy=tax,
                                      created_at=None, n_buckets=None)
    assert [r["signal_class"] for r in rows] == [fab.BINARY_NO_TRACE_CLASS]
    rows = fab.ledger_rows_for_member(run_id="fc", hs_run_id=None, q=_q("PA"), model_name="m",
                                      trace=None, taxonomy=tax, created_at=None, n_buckets=None)
    assert [r["signal_class"] for r in rows] == ["no_trace"]


def test_the_version_moved():
    assert fab.TAXONOMY_VERSION == "2.0"
