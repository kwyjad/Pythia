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
DEFAULT_TEMPLATE_VERSION = "v1"
SCHEMA_VERSION = "1"

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


def thinking_level() -> str:
    return (os.getenv("PYTHIA_INTERPRETER_THINKING") or "high").strip() or "high"


def max_output_tokens() -> int:
    return _env_int("PYTHIA_INTERPRETER_MAX_OUTPUT_TOKENS", 32_768)


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
