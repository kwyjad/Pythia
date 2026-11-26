# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The lineup probe's offline halves: the prompt it sends and how it judges
an answer. The network half runs only in probe_model_lineup.yml."""

from __future__ import annotations

from tools import probe_model_lineup as probe


def test_the_probe_sends_a_real_spd_prompt() -> None:
    prompt = probe.spd_prompt()
    assert '"spds"' in prompt and len(prompt) > 4000


def test_spd_validity() -> None:
    probs = "[0.5,0.2,0.1,0.1,0.05,0.03,0.02]"
    months = ",".join(f'"2027-0{m}": {{"probs": {probs}}}' for m in range(1, 7))
    assert probe.spd_valid("{\"spds\": {" + months + "}}")["valid"]
    assert not probe.spd_valid("not json")["valid"]
    assert not probe.spd_valid('{"spds": {"2027-01": {"probs": [1.0]}}}')["valid"]


def test_the_probe_uses_each_member_s_own_settings() -> None:
    argon = probe.ensemble_spec("google", "gemini-4-argon")
    assert argon.thinking == "medium" and argon.temperature is None
    assert argon.purpose == "spd_v2"
    assert probe.ensemble_spec("openai", "gpt-6.1-sol").thinking == "high"
