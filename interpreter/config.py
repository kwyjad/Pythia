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
"""

from __future__ import annotations

import os

ROLE = "interpreter"
DEFAULT_TEMPLATE_VERSION = "v2"
SCHEMA_VERSION = "1"

# --- Attention selection (report sections) ---------------------------------
# A forecast BELOW its base rate is not news, so only forecasts above it can
# enter the "potentially worsening" sections. "Significantly above" is set as
# a multiple of the historical expectation rather than a divergence score,
# because a multiple is the one number a lay reader can check against the
# sentence it appears in: 2x means the system expects twice the impact
# history would suggest. Each section takes its top MIN entries whatever the
# multiple (so a quiet month still names the month's worst), plus any further
# entries that clear the multiple, capped at MAX so the report stays short.
DEFAULT_WORSENING_MULTIPLE = 2.0
DEFAULT_MIN_PER_CATEGORY = 3
DEFAULT_MAX_PER_CATEGORY = 6

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


def worsening_multiple() -> float:
    """How far above the base rate counts as "significantly above"."""
    return _env_float("PYTHIA_INTERPRETER_WORSENING_MULTIPLE",
                      DEFAULT_WORSENING_MULTIPLE)


def min_per_category() -> int:
    return _env_int("PYTHIA_INTERPRETER_MIN_PER_CATEGORY",
                    DEFAULT_MIN_PER_CATEGORY)


def max_per_category() -> int:
    return _env_int("PYTHIA_INTERPRETER_MAX_PER_CATEGORY",
                    DEFAULT_MAX_PER_CATEGORY)


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
