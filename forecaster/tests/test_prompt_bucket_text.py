# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Keep the hand-written SPD bucket prompt texts in sync with BUCKET_SPECS.

The SPD_BUCKET_TEXT_* constants must stay LITERAL triple-quoted strings —
the About page (web/src/lib/prompt_extractor.ts) regex-extracts them from
the prompts.py source — so single-sourcing is enforced here by CI instead
of by generating them at runtime.
"""

from __future__ import annotations

import re

import pytest

from forecaster import prompts
from pythia.buckets import get_bucket_specs, labels_for, n_buckets_for

_BUCKET_TEXTS = {
    "PA": prompts.SPD_BUCKET_TEXT_PA,
    "FATALITIES": prompts.SPD_BUCKET_TEXT_FATALITIES,
    "PHASE3PLUS_IN_NEED": prompts.SPD_BUCKET_TEXT_PHASE3,
}


@pytest.mark.parametrize("metric", sorted(_BUCKET_TEXTS))
def test_bucket_text_labels_match_bucket_specs(metric: str) -> None:
    text = _BUCKET_TEXTS[metric]
    labels_in_text = re.findall(r'\(label: "([^"]+)"\)', text)
    assert labels_in_text == labels_for(metric), (
        f"{metric} bucket prompt text labels drifted from BUCKET_SPECS — "
        "update the SPD_BUCKET_TEXT_* literal in forecaster/prompts.py"
    )


@pytest.mark.parametrize("metric", sorted(_BUCKET_TEXTS))
def test_bucket_text_has_one_line_per_bucket(metric: str) -> None:
    text = _BUCKET_TEXTS[metric]
    bucket_lines = re.findall(r"^- Bucket (\d+):", text, flags=re.M)
    assert [int(b) for b in bucket_lines] == list(
        range(1, n_buckets_for(metric) + 1)
    )


@pytest.mark.parametrize("metric", sorted(_BUCKET_TEXTS))
def test_bucket_text_boundaries_present(metric: str) -> None:
    """Every finite interior boundary value appears in the prompt text."""
    text = _BUCKET_TEXTS[metric].replace(",", "")
    for spec in get_bucket_specs(metric):
        if spec.upper is None or spec.upper == 1.0:
            continue  # open top bucket; the 0/1 boundary is prose ("exactly 0")
        token = f"{int(spec.upper):,}".replace(",", "")
        # Labels use k/M shorthand; the long-form boundary or the label must
        # appear once commas are stripped.
        assert token in text or spec.label in _BUCKET_TEXTS[metric], (
            f"{metric}: boundary {spec.upper} not reflected in prompt text"
        )


def test_v2_prompt_renders_all_labels_per_metric() -> None:
    """Smoke test: the live v2 prompt lists every bucket label and sizes
    its placeholders to the metric's bucket count."""
    cases = [
        ("ETH", "ACE", "FATALITIES", "ACLED"),
        ("ETH", "FL", "PA", "IFRC"),
        ("SOM", "DR", "PHASE3PLUS_IN_NEED", "FEWSNET"),
    ]
    for iso3, hz, metric, src in cases:
        question = {
            "question_id": f"{iso3}_{hz}_{metric}_2026-08",
            "iso3": iso3,
            "hazard_code": hz,
            "metric": metric,
            "resolution_source": src,
            "wording": "test wording",
            "window_start_date": "2026-08-01",
            "target_month": "2027-01",
        }
        text = prompts.build_spd_prompt_v2(
            question=question,
            history_summary={},
            hs_triage_entry={},
            research_json={},
        )
        k = n_buckets_for(metric)
        for label in labels_for(metric):
            assert f'"{label}"' in text, f"{metric}: label {label} missing"
        placeholder = "[" + ", ".join(f"p{i}" for i in range(1, k + 1)) + "]"
        assert placeholder in text
        # No stale 5-bucket phrasing anywhere in the rendered prompt.
        assert "five impact buckets" not in text
        assert "[p1, p2, p3, p4, p5]" not in text or k == 5
