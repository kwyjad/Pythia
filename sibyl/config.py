# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Sibyl configuration.

Every knob is env-overridable (``SIBYL_*``) following the repo's
``_env_float`` convention, with the spec defaults baked in.
"""

from __future__ import annotations

import os


def _env_float(name: str, default: float) -> float:
    try:
        raw = os.getenv(name)
        return float(raw) if raw not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        raw = os.getenv(name)
        return int(raw) if raw not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes")


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    return raw if raw not in (None, "") else default


# --- Model -----------------------------------------------------------------
# Opus-tier deep-research model. NOTE: claude-opus-4-8 rejects sampling
# params (temperature/top_p/top_k -> HTTP 400); trial diversity comes from
# per-trial perspective seeds in the prompt, not temperature.
MODEL = _env_str("SIBYL_MODEL", "claude-opus-4-8")

# Stable model_name under which Sibyl SPDs are written to forecasts_raw /
# forecasts_ensemble (and therefore scored). Analogous to `track2_flash`:
# a track marker, independent of the backing model id above.
SIBYL_MODEL_NAME = "sibyl"

# --- Trials ----------------------------------------------------------------
# Independent agentic trials per question. Reduced from the literature's 5-6
# sweet spot for budget - the first three trials capture most of the
# variance-reduction benefit.
K = _env_int("SIBYL_K", 3)

# Agent steps per trial (search/fetch/submit). Early submit is allowed.
MAX_STEPS = _env_int("SIBYL_MAX_STEPS", 10)

# Quantile levels each trial must report (discretized CDF).
QUANTILE_LEVELS = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]

# --- Aggregation -----------------------------------------------------------
# "linear_pool" (mean of CDFs = mixture; widens on disagreement) or
# "vincent" (per-level quantile averaging).
AGGREGATION = _env_str("SIBYL_AGGREGATION", "linear_pool")

# --- Calibration hook (deferred) --------------------------------------------
CALIBRATION_ENABLED = _env_bool("SIBYL_CALIBRATION_ENABLED", False)

# --- Budget ------------------------------------------------------------------
# Hard run cut-off: stop STARTING new questions/trials once cumulative run
# cost reaches this. The in-flight unit runs to completion, so realized
# spend can exceed the cap by roughly one question's cost. Load-bearing:
# Sibyl is expected to dominate Pythia's API spend. At K=3 an expected cycle
# is ~$12-15, so $40 is a tail backstop (~2.5-3x expected).
RUN_HARD_CAP_USD = _env_float("SIBYL_RUN_HARD_CAP_USD", 40.0)

# Optional secondary per-question guard; None/0 = unset.
_bpq = _env_float("SIBYL_BUDGET_USD_PER_QUESTION", 0.0)
BUDGET_USD_PER_QUESTION: float | None = _bpq if _bpq > 0 else None

# --- Question selection ------------------------------------------------------
N_QUESTIONS = _env_int("SIBYL_N_QUESTIONS", 10)

# Numeric affected/fatalities magnitude questions only (spec scope:
# "ACE fatalities; DR/FL/TC affected"). DR "affected" is represented as
# PHASE3PLUS_IN_NEED in this codebase (there are no DR/PA questions).
# EVENT_OCCURRENCE (binary) is excluded by construction.
ELIGIBLE_HAZARD_METRICS = frozenset(
    {
        ("ACE", "FATALITIES"),
        ("FL", "PA"),
        ("TC", "PA"),
        ("DR", "PHASE3PLUS_IN_NEED"),
    }
)

# Which standard-track aggregate a Sibyl SPD is compared against (and
# attached to), in preference order. Single-sourced here — the API route
# (pythia/api/routes/sibyl.py) and sibyl/spd.py both import it. This module
# must stay import-light (os only) so the API process can import it without
# pulling in the sibyl agent/provider tree.
STANDARD_MODEL_PREFERENCE = ("ensemble_bayesmc_v2", "ensemble_mean_v2", "track2_flash")

# --- Time / backtest ---------------------------------------------------------
# Live mode: asOf = now. Backtest mode: asOf = the question's window anchor
# (window_start_date), and the leakage controls in sibyl/leakage.py become
# active filters instead of no-ops.
BACKTEST_MODE = _env_bool("SIBYL_BACKTEST_MODE", False)

# --- Optional authoritative live lookups (extension point) -------------------
# Disabled by design: Sibyl's independence from the structured Pythia
# connectors is the point of the parallel track. When enabled (future), any
# lookup must be clamped to asOf by sibyl/leakage.py.
LIVE_LOOKUPS_ENABLED = _env_bool("SIBYL_LIVE_LOOKUPS_ENABLED", False)

# --- Search ------------------------------------------------------------------
BRAVE_MAX_RESULTS = _env_int("SIBYL_BRAVE_MAX_RESULTS", 8)
BRAVE_TIMEOUT_SEC = _env_int("SIBYL_BRAVE_TIMEOUT_SEC", 20)
# Default lookback window for date-filtered searches (days before asOf).
SEARCH_WINDOW_DAYS = _env_int("SIBYL_SEARCH_WINDOW_DAYS", 120)
FETCH_URL_TIMEOUT_SEC = _env_int("SIBYL_FETCH_URL_TIMEOUT_SEC", 20)
FETCH_URL_MAX_CHARS = _env_int("SIBYL_FETCH_URL_MAX_CHARS", 6000)

# --- LLM call limits ----------------------------------------------------------
ANTHROPIC_MAX_ATTEMPTS = _env_int("SIBYL_ANTHROPIC_MAX_ATTEMPTS", 3)
