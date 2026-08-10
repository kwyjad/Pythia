# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Interpreter configuration: model role, template version, caps, thresholds.

Env vars (see CLAUDE.md's table):
- PYTHIA_INTERPRETER_ENABLED (default 1) — kill switch.
- PYTHIA_INTERPRETER_MODEL_ID — purpose-specific override, wins over the
  config ``interpreter`` role (same precedence rule as every other role).
- PYTHIA_INTERPRETER_TEMPLATE_VERSION (default v1).
- PYTHIA_INTERPRETER_MAX_PACK_TOKENS (default 250000) — hard model-input cap.
- PYTHIA_INTERPRETER_TOP_N (default 8) — attention list length.
- PYTHIA_INTERPRETER_PER_CAPITA_FLOOR (default 10000).
- PYTHIA_INTERPRETER_THINKING (default high) — Anthropic output_config.effort
  level (low|medium|high|xhigh|max; emitted only for effort-capable models).
- PYTHIA_INTERPRETER_MAX_OUTPUT_TOKENS (default 32768) — thinking shares this
  budget, so it sits at the SPD ceiling, not the 16k general one.
- PYTHIA_INTERPRETER_STRICT_VALIDATION (default 0) — when 1, a validation
  failure suppresses publication (consumed by the Phase 4 validator).

v3 selection and report knobs (all with defaults below):
- PYTHIA_INTERPRETER_THRESHOLD_{PA,FATALITIES,PHASE3} and the two
  _POP_SHARE variants — what counts as an impact worth mobilising against.
- PYTHIA_INTERPRETER_MINPROB (and _MINPROB_{METRIC}) — how much probability
  has to sit above that threshold, in any one month of the window.
- PYTHIA_INTERPRETER_UNUSUAL_PERCENTILE — how far from its own history a
  forecast must sit, as a percentile of this run's own spread.
- PYTHIA_INTERPRETER_BASERATE_MIN_OBS — below this the anchor is called thin.
- PYTHIA_INTERPRETER_MAX_ENTRIES — how many entries the report carries.
- PYTHIA_INTERPRETER_LEAD_TIME_{HAZARD} — months between the decision and the
  month the plan has to cover (the decision calendar's derivation).
- PYTHIA_INTERPRETER_SIBYL_{DISAGREEMENT_RATIO,UNSETTLED_SHARE,NOVEL_FACTS}.
- PYTHIA_INTERPRETER_SECTOR_{LIST_SIZE,MIN_RANK_GAP}.
- PYTHIA_INTERPRETER_MIN_RESOLVED_FOR_SKILL and _SKILL_BOOTSTRAP — Part B's
  small-sample floor and the size of its bootstrap.
"""

from __future__ import annotations

import os

ROLE = "interpreter"
DEFAULT_TEMPLATE_VERSION = "v3"
SCHEMA_VERSION = "1"

# --- Attention selection (report sections) ---------------------------------
# Which box a row belongs in is the GATE's decision (interpreter/gating.py);
# selection only orders the gated rows by expected excess and cuts them to
# DEFAULT_MAX_ENTRIES. The old per-section knobs (a worsening MULTIPLE, a
# minimum and a maximum per box) were removed in v3: ranking by a multiple is
# what put a cyclone question with 87% of its weight on "nobody affected" at
# the top of a report whose second half was famine in Sudan.

# chars-per-token estimate shared with the pack builder.
CHARS_PER_TOKEN = 4.0


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name) or default)
    except ValueError:
        return default


def enabled() -> bool:
    return (os.getenv("PYTHIA_INTERPRETER_ENABLED", "1") or "1").strip().lower() in (
        "1", "true", "yes",
    )


def model_id_override() -> str | None:
    raw = (os.getenv("PYTHIA_INTERPRETER_MODEL_ID") or "").strip()
    return raw or None


def template_version() -> str:
    return (
        os.getenv("PYTHIA_INTERPRETER_TEMPLATE_VERSION") or DEFAULT_TEMPLATE_VERSION
    ).strip() or DEFAULT_TEMPLATE_VERSION


def max_pack_tokens() -> int:
    return _env_int("PYTHIA_INTERPRETER_MAX_PACK_TOKENS", 250_000)


def top_n() -> int:
    return _env_int("PYTHIA_INTERPRETER_TOP_N", 8)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name) or default)
    except ValueError:
        return default


# --------------------------------------------------------------------------
# Selection thresholds (interpreter v3)
#
# The PA figures are confirmed by the owner. The other two are provisional and
# deliberately trivial to change: every one is an env var as well as a default,
# because the first months of real use are what will settle them.
# --------------------------------------------------------------------------

# What counts as an impact worth mobilising against, per metric. The
# population share exists so small states are not silently excluded: a flat
# 50,000 floor means Vanuatu never appears in a cyclone report.
DEFAULT_THRESHOLD_PA = 50_000.0
DEFAULT_THRESHOLD_PA_POP_SHARE = 0.01
DEFAULT_THRESHOLD_FATALITIES = 100.0
DEFAULT_THRESHOLD_PHASE3 = 1_000_000.0
DEFAULT_THRESHOLD_PHASE3_POP_SHARE = 0.10

# How much probability must sit above that threshold, in any one horizon.
DEFAULT_MINPROB = 0.25

# How unusual a forecast must be to clear the first key: its divergence must
# sit at or above this percentile of the run's own distribution. A percentile
# rather than an absolute cut, because the spread of divergences varies by
# run and an absolute number would admit everything one month and nothing the
# next.
DEFAULT_UNUSUAL_PERCENTILE = 0.80

# Jeffreys smoothing on the empirical base rate. NOTE: this is the value
# base_rate_spd has always applied; lifting it here makes the strength
# tunable and auditable, but it does NOT change any ranking. The August
# report's 14x headline was never a zero-bucket artefact (the anchor read
# [0.9925, 0.0015, ...]); it was a ratio against a near-zero expectation,
# which only ranking by excess fixes.
DEFAULT_BASERATE_SMOOTHING = 0.5

# Below this many historical observations the anchor is called thin: the
# entry is demoted below every clear-anchor entry and says so in words.
DEFAULT_BASERATE_MIN_OBS = 12

DEFAULT_MAX_ENTRIES = 5

# Part B: the smallest resolved count that can support a skill claim.
DEFAULT_MIN_RESOLVED_FOR_SKILL = 100

# How many bootstrap resamples stand behind a printed skill interval. A point
# estimate off eighty question-horizons is a number with no error bar, and the
# first scored run will be exactly that size.
DEFAULT_SKILL_BOOTSTRAP_SAMPLES = 2000

# --- The decision calendar (task 5) ----------------------------------------
# How long before the month a plan has to cover the decision must be taken.
# Slow-onset hazards need the longest run-up: a drought response is a pipeline
# and a procurement cycle, a cyclone response is a warehouse that is already
# stocked or is not. These are the report's own planning assumptions, stated
# here so they can be argued with rather than buried in a sentence.
DEFAULT_LEAD_TIME_MONTHS = {
    "DR": 3,
    "FL": 1,
    "TC": 1,
    "ACE": 2,
}
DEFAULT_LEAD_TIME_FALLBACK = 1

# --- The second reader (task 4) --------------------------------------------
# How far Sibyl's expected value must sit from the main pipeline's before the
# report calls it a disagreement rather than agreement. A ratio, both ways.
DEFAULT_SIBYL_DISAGREEMENT_RATIO = 1.5
# How much inter-trial divergence means Sibyl's own research did not settle
# the question. js_divergence is on [0, ln 2]; this is the share of that.
DEFAULT_SIBYL_UNSETTLED_SHARE = 0.25
DEFAULT_SIBYL_NOVEL_FACTS = 2

# --- The sector comparison (task 6) ----------------------------------------
DEFAULT_SECTOR_LIST_SIZE = 5
# A country needs a rank gap of at least this many places before the report
# claims Fred and the sector disagree about it. Two adjacent ranks is noise.
DEFAULT_SECTOR_MIN_RANK_GAP = 5


def _metric_key(metric: str) -> str:
    m = (metric or "").upper()
    if m == "PHASE3PLUS_IN_NEED":
        return "PHASE3"
    return m


def threshold_for_metric(metric: str) -> float | None:
    """Absolute impact threshold for a metric, or None when it has none."""
    key = _metric_key(metric)
    defaults = {
        "PA": DEFAULT_THRESHOLD_PA,
        "FATALITIES": DEFAULT_THRESHOLD_FATALITIES,
        "PHASE3": DEFAULT_THRESHOLD_PHASE3,
    }
    if key not in defaults:
        return None
    return _env_float(f"PYTHIA_INTERPRETER_THRESHOLD_{key}", defaults[key])


def population_share_for_metric(metric: str) -> float | None:
    """Share-of-population alternative threshold, or None when it has none."""
    key = _metric_key(metric)
    defaults = {
        "PA": DEFAULT_THRESHOLD_PA_POP_SHARE,
        "PHASE3": DEFAULT_THRESHOLD_PHASE3_POP_SHARE,
    }
    if key not in defaults:
        return None
    return _env_float(f"PYTHIA_INTERPRETER_THRESHOLD_{key}_POP_SHARE", defaults[key])


def min_probability(metric: str | None = None) -> float:
    """Probability that must sit above the threshold in some horizon.

    Per-metric override first, then the shared default, so a metric can be
    tuned without moving the others.
    """
    if metric:
        key = _metric_key(metric)
        specific = os.environ.get(f"PYTHIA_INTERPRETER_MINPROB_{key}")
        if specific:
            try:
                return float(specific)
            except ValueError:
                pass
    return _env_float("PYTHIA_INTERPRETER_MINPROB", DEFAULT_MINPROB)


def unusual_percentile() -> float:
    return _env_float(
        "PYTHIA_INTERPRETER_UNUSUAL_PERCENTILE", DEFAULT_UNUSUAL_PERCENTILE
    )


def baserate_smoothing() -> float:
    return _env_float(
        "PYTHIA_INTERPRETER_BASERATE_SMOOTHING", DEFAULT_BASERATE_SMOOTHING
    )


def baserate_min_obs() -> int:
    return _env_int(
        "PYTHIA_INTERPRETER_BASERATE_MIN_OBS", DEFAULT_BASERATE_MIN_OBS
    )


def max_entries() -> int:
    return _env_int("PYTHIA_INTERPRETER_MAX_ENTRIES", DEFAULT_MAX_ENTRIES)


def min_resolved_for_skill() -> int:
    return _env_int(
        "PYTHIA_INTERPRETER_MIN_RESOLVED_FOR_SKILL", DEFAULT_MIN_RESOLVED_FOR_SKILL
    )


def skill_bootstrap_samples() -> int:
    return _env_int(
        "PYTHIA_INTERPRETER_SKILL_BOOTSTRAP", DEFAULT_SKILL_BOOTSTRAP_SAMPLES
    )


def lead_time_months(hazard_code: str | None) -> int:
    """Months between the decision and the month the plan has to cover.

    Per-hazard env override (``PYTHIA_INTERPRETER_LEAD_TIME_DR``) so the
    planning assumption can be retuned by whoever disagrees with it.
    """
    key = (hazard_code or "").strip().upper()
    default = DEFAULT_LEAD_TIME_MONTHS.get(key, DEFAULT_LEAD_TIME_FALLBACK)
    if not key:
        return default
    return _env_int(f"PYTHIA_INTERPRETER_LEAD_TIME_{key}", default)


def sibyl_disagreement_ratio() -> float:
    return _env_float(
        "PYTHIA_INTERPRETER_SIBYL_DISAGREEMENT_RATIO",
        DEFAULT_SIBYL_DISAGREEMENT_RATIO,
    )


def sibyl_unsettled_share() -> float:
    return _env_float(
        "PYTHIA_INTERPRETER_SIBYL_UNSETTLED_SHARE", DEFAULT_SIBYL_UNSETTLED_SHARE
    )


def sibyl_novel_facts() -> int:
    return _env_int("PYTHIA_INTERPRETER_SIBYL_NOVEL_FACTS", DEFAULT_SIBYL_NOVEL_FACTS)


def sector_list_size() -> int:
    return _env_int("PYTHIA_INTERPRETER_SECTOR_LIST_SIZE", DEFAULT_SECTOR_LIST_SIZE)


def sector_min_rank_gap() -> int:
    return _env_int(
        "PYTHIA_INTERPRETER_SECTOR_MIN_RANK_GAP", DEFAULT_SECTOR_MIN_RANK_GAP
    )


def public_base_url() -> str:
    """Site root for absolute question links in the PDF (relative hrefs in a
    downloaded PDF resolve against nothing)."""
    return (os.getenv("PYTHIA_PUBLIC_BASE_URL")
            or "https://fredforecaster.org").rstrip("/")


def thinking_level() -> str:
    return (os.getenv("PYTHIA_INTERPRETER_THINKING") or "high").strip() or "high"


def max_output_tokens() -> int:
    return _env_int("PYTHIA_INTERPRETER_MAX_OUTPUT_TOKENS", 32_768)


def validation_retries() -> int:
    """How many correction passes to allow after a failed validation.

    One by default. A second call costs about the same as the first, and the
    failures worth retrying are single-sentence slips; a model that misses
    twice is telling us something about the checks rather than about itself.
    """
    return _env_int("PYTHIA_INTERPRETER_VALIDATION_RETRIES", 1)


def strict_validation() -> bool:
    return (os.getenv("PYTHIA_INTERPRETER_STRICT_VALIDATION", "0") or "0").strip().lower() in (
        "1", "true", "yes",
    )


def call_timeout_sec() -> float:
    try:
        return float(os.getenv("PYTHIA_INTERPRETER_TIMEOUT_SEC") or 900.0)
    except ValueError:
        return 900.0


def estimate_tokens(text: str) -> int:
    import math

    return int(math.ceil(len(text) / CHARS_PER_TOKEN))
