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


class TestSalvageSpdsBlock:
    """Regression cover for the 2026-09-01 run: three member responses were
    discarded whole for a broken prose field beside a valid ``spds`` block."""

    def _broken_prose_response(self) -> str:
        # The explanation carries an unescaped quote — json.loads fails at it —
        # while the spds block that precedes it is well formed.
        payload = _spd_payload("[0.0, -0.05, 0.05]")
        assert '"human_explanation": "RC: accepted."' in payload
        return payload.replace(
            '"human_explanation": "RC: accepted."',
            '"human_explanation": "RC: "partially" accepted."',
        )

    def test_broken_prose_no_longer_loses_the_forecast(self):
        from forecaster.cli import _salvage_spds_object

        text = self._broken_prose_response()
        with pytest.raises(json.JSONDecodeError):
            _safe_json_loads(text)
        salvaged = _salvage_spds_object(text)
        assert salvaged is not None and salvaged["_salvaged"] is True
        assert len(salvaged["spds"]) == 6
        assert salvaged["spds"]["2026-09"]["probs"][3] == 0.42

    def test_salvage_applies_the_unary_plus_repair_inside_the_block(self):
        from forecaster.cli import _salvage_spds_object

        text = self._broken_prose_response().replace("0.42", "+0.42")
        salvaged = _salvage_spds_object(text)
        assert salvaged is not None
        assert salvaged["spds"]["2026-10"]["probs"][3] == 0.42

    def test_no_spds_key_means_nothing_to_salvage(self):
        from forecaster.cli import _salvage_spds_object

        assert _salvage_spds_object('{"reasoning_trace": {"prior": [0.5, 0.5]}') is None
        assert _salvage_spds_object("") is None
        assert _salvage_spds_object(None) is None

    def test_broken_spds_block_is_not_salvaged(self):
        from forecaster.cli import _salvage_spds_object

        text = self._broken_prose_response().replace('"probs": [0.03,', '"probs": [0.03 0.03,', 1)
        assert _salvage_spds_object(text) is None

    def test_braces_inside_strings_do_not_confuse_the_scan(self):
        from forecaster.cli import _salvage_spds_object

        text = (
            '{"human_explanation": "see {this} and \\"that\\"", '
            '"spds": {"month_1": {"probs": [0.5, 0.5], "note": "a } brace"}}, '
            '"tail": broken'
        )
        salvaged = _salvage_spds_object(text)
        assert salvaged is not None
        assert salvaged["spds"]["month_1"]["probs"] == [0.5, 0.5]
