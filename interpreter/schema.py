# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The interpreter's structured-output contract (plan §7.3).

The model returns JSON only, validated against OUTPUT_SCHEMA (jsonschema).
Prose fields may contain ``{{fig:<key>}}`` placeholders that resolve against
the pack at render time — the deeper checks (bare-numeral lint, lexicon band
agreement, referential and numeric guards) are the Phase 4 validator's job;
this module owns the SHAPE.
"""

from __future__ import annotations

import json
from typing import Any

from interpreter.config import SCHEMA_VERSION

REASON_CODES = [
    "base_rate_deviation",
    "large_impact_nominal",
    "large_impact_per_capita",
    "rc_deviation_disagreement",
]

_PROSE = {"type": "string", "maxLength": 2000}
_PROSE_LIST = {"type": "array", "items": _PROSE}
_QUESTION_IDS = {"type": "array", "items": {"type": "string"}, "minItems": 1}
_FIGURE_REFS = {"type": "array", "items": {"type": "string"}}

CATEGORIES = ["worsening", "stable_major"]
HAZARD_FAMILIES = ["climate", "conflict", "other"]

# What a decision point may be anchored to. The runner overwrites
# ``deadline_month`` and ``basis`` with the pack's derived values wherever the
# pack has them (peak horizon minus the hazard's lead time), so the model's
# real job in this object is ``action``: naming the decision. An entry with no
# decision attached is an entry that should not be in the report, and asking
# for one is how the model is made to justify its inclusion.
DECISION_BASES = [
    "peak_horizon", "lean_season_start", "election_date", "seasonal_window",
]

_DECISION_POINT = {
    "type": "object",
    "required": ["action", "deadline_month", "basis"],
    "properties": {
        "action": _PROSE,
        "deadline_month": {"type": "string", "pattern": r"^\d{4}-\d{2}$"},
        "basis": {"type": "string", "enum": DECISION_BASES},
    },
    "additionalProperties": False,
}

_ATTENTION_ENTRY = {
    "type": "object",
    "required": [
        "rank", "reason_code", "iso3", "hazard_code", "metric",
        "category", "hazard_family", "question_ids", "why_it_stands_out",
        "decision_point",
    ],
    "properties": {
        "rank": {"type": "integer", "minimum": 1},
        # Which of the report's four boxes this belongs in. The pack decides
        # it (interpreter/selection.py); the model copies it across and may
        # not invent a placement.
        "category": {"type": "string", "enum": CATEGORIES},
        "hazard_family": {"type": "string", "enum": HAZARD_FAMILIES},
        "reason_code": {"type": "string", "enum": REASON_CODES},
        "iso3": {"type": "string", "minLength": 3, "maxLength": 3},
        "hazard_code": {"type": "string"},
        "metric": {"type": "string"},
        "question_ids": _QUESTION_IDS,
        "figure_refs": _FIGURE_REFS,
        "why_it_stands_out": _PROSE,
        "how_to_read_the_distribution": _PROSE,
        # Where the probability sits and how heavy the tail is, in words.
        # The frame the renderer fills with the two planning figures. Prose
        # with placeholders; the numbers are never the model's.
        "planning_sentence": _PROSE,
        "spd_shape": _PROSE,
        "what_the_model_was_reacting_to": _PROSE,
        "decision_point": _DECISION_POINT,
        "impacts": _PROSE_LIST,
        "operational_challenges": _PROSE_LIST,
        "lead_time_months": {"type": "integer", "minimum": 1, "maximum": 6},
    },
    "additionalProperties": False,
}

_CALL_ENTRY = {
    "type": "object",
    "required": ["question_ids"],
    "properties": {
        "question_ids": _QUESTION_IDS,
        "what_was_right": _PROSE,
        "what_went_wrong": _PROSE,
        "figure_refs": _FIGURE_REFS,
    },
    "additionalProperties": False,
}

_PERFORMANCE = {
    "type": "object",
    "required": ["plain_summary"],
    "properties": {
        "plain_summary": _PROSE,
        "skill_statement": _PROSE,
        "best_calls": {"type": "array", "items": _CALL_ENTRY},
        "worst_calls": {"type": "array", "items": _CALL_ENTRY},
        "track_comparison": _PROSE,
        "sibyl_comparison": _PROSE,
        "vs_system_average": _PROSE,
    },
    "additionalProperties": False,
}

OUTPUT_SCHEMA: dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["schema_version", "template_version", "kind", "headline"],
    "properties": {
        "schema_version": {"type": "string", "const": SCHEMA_VERSION},
        "template_version": {"type": "string"},
        "kind": {"type": "string", "enum": ["current", "scored", "combined"]},
        "headline": _PROSE,
        # One short paragraph of scan-scope counts, written from the pack's
        # run_summary figures (the model never counts anything itself).
        "run_summary": _PROSE,
        "attention": {"type": "array", "items": _ATTENTION_ENTRY},
        "performance": _PERFORMANCE,
        "changes_since_last_run": _PROSE_LIST,
        "blind_spots": _PROSE_LIST,
        "confidence_notes": _PROSE_LIST,
    },
    "additionalProperties": False,
}


def schema_json() -> str:
    """The schema as pretty JSON (embedded in the system prompt)."""
    return json.dumps(OUTPUT_SCHEMA, indent=1)


def validate_output(
    obj: Any,
    *,
    kind: str | None = None,
    require_performance: bool = True,
    require_attention: bool = True,
) -> list[str]:
    """Shape-validate a structured output. Returns a list of error strings
    (empty = valid).

    Kind-conditional requirements live here rather than in the jsonschema
    (draft-07 conditionals are write-only): ``attention`` is required for
    current/combined, ``performance`` for scored/combined.

    ``require_performance`` is lowered for a combined report with no scored
    run behind it. Demanding the block there would make the model write a
    summary of nothing, which the renderer then discards in favour of the
    generated dormant state; asking for prose we do not print is how a model
    learns to invent.

    ``require_attention`` is lowered when the gate admitted nothing at all.
    That is a finding about the month, and it should read as one: failing the
    whole report instead would turn a quiet month into a broken report, and
    the reader would be told nothing either way.
    """
    errors: list[str] = []
    try:
        import jsonschema
    except Exception:  # pragma: no cover - dependency is in requirements
        return ["jsonschema library unavailable — cannot validate"]

    validator = jsonschema.Draft7Validator(OUTPUT_SCHEMA)
    for err in sorted(validator.iter_errors(obj), key=lambda e: list(e.absolute_path)):
        path = "/".join(str(p) for p in err.absolute_path) or "<root>"
        errors.append(f"{path}: {err.message}")

    if isinstance(obj, dict):
        effective_kind = kind or obj.get("kind")
        if (
            require_attention
            and effective_kind in ("current", "combined")
            and not obj.get("attention")
        ):
            errors.append("<root>: 'attention' is required and non-empty for kind "
                          f"{effective_kind!r}")
        if (
            require_performance
            and effective_kind in ("scored", "combined")
            and not obj.get("performance")
        ):
            errors.append("<root>: 'performance' is required for kind "
                          f"{effective_kind!r}")
        if kind and obj.get("kind") != kind:
            errors.append(
                f"<root>: kind mismatch — output says {obj.get('kind')!r}, "
                f"run requested {kind!r}"
            )
    return errors
