# Pythia / Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
"""Guards for the SPD response JSON repair pass.

Regression cover for the 2026-08-01 production run, where 8 of 345 SPD member
calls succeeded, were billed, and were then discarded whole because the model
wrote an explicit unary plus in ``reasoning_trace.updates[].delta``
(``[-0.07, -0.13, +0.25]``). ``json.loads`` rejects a leading ``+``, the parse
failure was swallowed, and 8 Track-1 questions silently ran on 4 members
instead of 5 — while ``llm_calls`` recorded ``status='ok'`` for every one.
"""

from __future__ import annotations

import json

import pytest

from forecaster.cli import _safe_json_loads, strip_unary_plus_outside_strings


def _spd_payload(delta: str) -> str:
    """A response shaped like the real ones, fenced, with a full 6-month SPD."""
    months = ["2026-09", "2026-10", "2026-11", "2026-12", "2027-01", "2027-02"]
    spds = ",\n".join(
        f'    "{m}": {{"buckets": ["0", "1-<5", "5-<25", "25-<100", '
        f'"100-<500", "500-<1000", ">=1000"], '
        f'"probs": [0.03, 0.10, 0.30, 0.42, 0.10, 0.04, 0.01]}}'
        for m in months
    )
    return (
        "```json\n"
        "{\n"
        '  "reasoning_trace": {\n'
        '    "prior": {"spd": [0.02, 0.08, 0.25, 0.45, 0.15, 0.04, 0.01],\n'
        '              "rationale": "Base rate ~48 fatalities/month."},\n'
        '    "updates": [\n'
        "      {\n"
        '        "signal": "VIEWS projects a spike",\n'
        '        "direction": "UP",\n'
        '        "magnitude": "LARGE",\n'
        '        "months_affected": "2-4",\n'
        f'        "delta": {delta},\n'
        '        "post_update_spd": [0.01, 0.03, 0.10, 0.25, 0.40, 0.15, 0.06]\n'
        "      }\n"
        "    ],\n"
        '    "point_estimate": "~120 fatalities",\n'
        '    "point_estimate_bucket": 4,\n'
        '    "rc_assessment": "partially_accepted"\n'
        "  },\n"
        '  "spds": {\n' + spds + "\n  },\n"
        '  "human_explanation": "RC: accepted."\n'
        "}\n"
        "```"
    )


class TestStripUnaryPlus:
    def test_valid_json_is_left_byte_identical(self):
        s = json.dumps({"a": [1, -2.5, 0.3], "b": "x", "c": {"d": None}})
        assert strip_unary_plus_outside_strings(s) == s

    def test_strips_plus_in_arrays_and_after_colon(self):
        out = strip_unary_plus_outside_strings('{"d": [-0.07, +0.25], "e": +3}')
        assert json.loads(out) == {"d": [-0.07, 0.25], "e": 3}

    def test_bare_plus_decimal_becomes_leading_zero(self):
        # Dropping the '+' alone would leave '.5', which JSON also rejects.
        # (A bare '.5' / '-.5' with no plus is a different malformation and is
        # deliberately out of scope for this repair.)
        out = strip_unary_plus_outside_strings('{"d": [+.5, 1]}')
        assert json.loads(out)["d"] == [pytest.approx(0.5), 1]

    def test_preserves_plus_inside_string_values(self):
        s = '{"note": "shift of +0.25, i.e. +25%", "d": [+0.1]}'
        parsed = json.loads(strip_unary_plus_outside_strings(s))
        assert parsed["note"] == "shift of +0.25, i.e. +25%"
        assert parsed["d"] == [0.1]

    def test_preserves_plus_inside_escaped_string(self):
        s = r'{"note": "a \"quoted\" +0.5 delta", "d": [+1]}'
        parsed = json.loads(strip_unary_plus_outside_strings(s))
        assert parsed["note"] == 'a "quoted" +0.5 delta'
        assert parsed["d"] == [1]

    def test_does_not_touch_plus_in_operator_position(self):
        # Not valid JSON either way, but the scan must not silently rewrite it.
        assert strip_unary_plus_outside_strings("1 + 2") == "1 + 2"


class TestSafeJsonLoadsRepair:
    def test_clean_response_still_parses(self):
        obj = _safe_json_loads(_spd_payload("[-0.01, -0.05, -0.15, -0.20, 0.25, 0.11, 0.05]"))
        assert len(obj["spds"]) == 6

    def test_leading_plus_delta_is_recovered_with_full_spd(self):
        # This is the exact production failure: the SPD block was always fine.
        obj = _safe_json_loads(
            _spd_payload("[-0.01, -0.05, -0.15, -0.20, +0.25, +0.11, +0.05]")
        )
        assert sorted(obj["spds"]) == [
            "2026-09",
            "2026-10",
            "2026-11",
            "2026-12",
            "2027-01",
            "2027-02",
        ]
        assert all(len(v["probs"]) == 7 for v in obj["spds"].values())
        assert obj["reasoning_trace"]["updates"][0]["delta"][4] == pytest.approx(0.25)

    def test_repair_applies_to_the_embedded_object_fallback(self):
        text = 'here you go:\n{"d": [+0.25]}\nhope that helps'
        assert _safe_json_loads(text) == {"d": [0.25]}

    def test_genuinely_broken_json_still_raises(self):
        with pytest.raises(json.JSONDecodeError):
            _safe_json_loads('{"a": [1, 2,,]}')

    def test_none_still_raises(self):
        with pytest.raises(json.JSONDecodeError):
            _safe_json_loads(None)
