# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Tests that the PRODUCTION precedence config actually drives the
run_pipeline precedence path.

Historical bug: the shipped precedence_config.yml listed the tier-0 source
as "ifrc_montandon" while the IFRC connector writes publisher="IFRC", so
IFRC rows silently fell to the default (worst) tier; and the YAML had no
"tiebreak"/"defaults" keys, so within-tier winners were chosen
alphabetically with no recency or non-null preference.
"""

from __future__ import annotations

import pandas as pd
import pytest

from resolver.tools.precedence_engine import resolve_facts_frame
from resolver.tools.run_pipeline import _load_precedence_config


@pytest.fixture(scope="module")
def prod_config() -> dict:
    cfg = _load_precedence_config()
    assert cfg, "production precedence_config.yml must load"
    return cfg


def _frame(rows: list[dict]) -> pd.DataFrame:
    base = {
        "country_iso3": "ETH",
        "hazard_type": "FL",
        "month": "2026-05",
        "metric": "affected",
        "value": "100",
        "as_of": "2026-05-10",
        "source": "IFRC",
        "run_id": "r1",
    }
    return pd.DataFrame([{**base, **r} for r in rows])


def test_ifrc_publisher_maps_to_tier_zero(prod_config: dict) -> None:
    tier0 = prod_config["tiers"][0]["sources"]
    assert "ifrc" in [s.lower() for s in tier0], (
        "tier 0 must contain the lowercased publisher value the IFRC "
        "connector writes ('IFRC'), not just the legacy slug"
    )


def test_prod_config_has_tiebreak_and_defaults(prod_config: dict) -> None:
    assert prod_config.get("tiebreak"), "production config must define tiebreak rules"
    assert prod_config.get("defaults", {}).get("prefer_full_row_coverage") is True


def test_ifrc_beats_idmc_for_same_key(prod_config: dict) -> None:
    df = _frame(
        [
            {"source": "IDMC", "value": "999", "as_of": "2026-05-20"},
            {"source": "IFRC", "value": "100", "as_of": "2026-05-10"},
        ]
    )
    resolved = resolve_facts_frame(df, prod_config)
    assert len(resolved) == 1
    row = resolved.iloc[0]
    assert row["selected_source"] == "IFRC"
    assert row["value"] == 100
    # Tier 0, not the unlisted default tier.
    assert int(row["selected_tier_index"]) == 0 if "selected_tier_index" in resolved.columns else True


def test_newer_as_of_wins_within_tier(prod_config: dict) -> None:
    df = _frame(
        [
            {"source": "IFRC", "value": "100", "as_of": "2026-05-01", "run_id": "r1"},
            {"source": "IFRC", "value": "250", "as_of": "2026-05-20", "run_id": "r2"},
        ]
    )
    resolved = resolve_facts_frame(df, prod_config)
    assert len(resolved) == 1
    assert resolved.iloc[0]["value"] == 250


def test_nonnull_value_beats_null(prod_config: dict) -> None:
    df = _frame(
        [
            # NULL-value row from an alphabetically-earlier source, newer as_of
            {"source": "ACLED", "value": "", "as_of": "2026-05-25"},
            {"source": "IFRC", "value": "180", "as_of": "2026-05-10"},
        ]
    )
    resolved = resolve_facts_frame(df, prod_config)
    assert len(resolved) == 1
    row = resolved.iloc[0]
    assert row["value"] == 180
    assert row["selected_source"] == "IFRC"


def test_engine_defaults_tiebreak_when_config_omits_it() -> None:
    """Belt-and-braces: even with a bare config, recency must win."""
    bare_config = {
        "tiers": [{"name": "Tier 0", "sources": ["ifrc", "acled"]}],
    }
    df = _frame(
        [
            {"source": "IFRC", "value": "100", "as_of": "2026-05-01", "run_id": "r1"},
            {"source": "IFRC", "value": "250", "as_of": "2026-05-20", "run_id": "r2"},
        ]
    )
    resolved = resolve_facts_frame(df, bare_config)
    assert len(resolved) == 1
    assert resolved.iloc[0]["value"] == 250
