from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from resolver.tools.precedence_engine import resolve_facts_frame

FIXTURES = Path(__file__).parent / "fixtures" / "precedence"


@pytest.fixture(scope="module")
def precedence_config() -> dict:
    with open(FIXTURES / "config_min.yml", "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@pytest.fixture(scope="module")
def facts_frame() -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "facts_sources.csv", dtype=str)


@pytest.fixture(scope="module")
def resolved_frame(precedence_config: dict, facts_frame: pd.DataFrame) -> pd.DataFrame:
    return resolve_facts_frame(facts_frame, precedence_config)


@pytest.fixture(scope="module")
def overrides_frame() -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "review_overrides.csv", dtype=str)


@pytest.fixture(scope="module")
def resolved_with_override(
    precedence_config: dict,
    facts_frame: pd.DataFrame,
    overrides_frame: pd.DataFrame,
) -> pd.DataFrame:
    return resolve_facts_frame(facts_frame, precedence_config, overrides_frame)


def _get_row(df: pd.DataFrame, country: str, hazard: str, month: str, metric: str) -> pd.Series:
    mask = (
        (df["country_iso3"] == country)
        & (df["hazard_type"] == hazard)
        & (df["month"] == month)
        & (df["metric"] == metric)
    )
    rows = df.loc[mask]
    assert len(rows) == 1, f"Expected one row for {(country, hazard, month, metric)}, found {len(rows)}"
    return rows.iloc[0]


def test_one_row_per_key(resolved_frame: pd.DataFrame) -> None:
    grouped = resolved_frame.groupby(["country_iso3", "hazard_type", "month", "metric"])  # type: ignore[arg-type]
    counts = grouped.size()
    assert (counts == 1).all(), "precedence engine must return exactly one record per key"
    expected_columns = {
        "country_iso3",
        "hazard_type",
        "month",
        "metric",
        "value",
        "selected_source",
        "selected_as_of",
        "selected_run_id",
        "selected_tier",
    }
    assert expected_columns.issubset(set(resolved_frame.columns))


def test_tier_preference_ifrc_overrides_recency(resolved_frame: pd.DataFrame) -> None:
    row = _get_row(resolved_frame, "ETH", "drought", "2025-03", "pin_new")
    assert row["selected_source"] == "ifrc_go"
    assert row["value"] == 10000


def test_as_of_recency_within_same_tier(resolved_frame: pd.DataFrame) -> None:
    row = _get_row(resolved_frame, "NGA", "flood", "2025-04", "pin_new")
    assert row["selected_source"] == "gdacs"
    assert row["selected_as_of"] == "2025-04-10"
    assert row["value"] == 2000


def test_null_handling_prefers_complete_rows(resolved_frame: pd.DataFrame) -> None:
    row = _get_row(resolved_frame, "SSD", "conflict_escalation", "2025-01", "pa_new")
    assert row["selected_source"] == "dtm"
    assert row["value"] == 7000


def test_within_tier_recency_breaks_tie(resolved_frame: pd.DataFrame) -> None:
    row = _get_row(resolved_frame, "SSD", "conflict_escalation", "2025-01", "pin_new")
    assert row["selected_source"] == "acled"
    assert row["selected_as_of"] == "2025-02-10"
    assert row["value"] == 5200


def test_stable_tiebreaker_source_alpha(resolved_frame: pd.DataFrame) -> None:
    row = _get_row(resolved_frame, "NGA", "flood", "2025-04", "pin_new")
    assert row["selected_source"] == "gdacs", "alphabetical fallback must be deterministic"


def test_intra_source_recency(resolved_frame: pd.DataFrame) -> None:
    row = _get_row(resolved_frame, "COL", "conflict_escalation", "2025-02", "pin_new")
    assert row["selected_source"] == "acled"
    assert row["selected_as_of"] == "2025-02-11"
    assert row["value"] == 280


def test_manual_override_supersedes_engine(resolved_with_override: pd.DataFrame) -> None:
    row = _get_row(resolved_with_override, "ETH", "drought", "2025-03", "pin_new")
    assert row["value"] == 9800
    assert row["selected_source"] == "review_override"
    assert row.get("override_note") == "Manual review: harmonized to IPC 3.26 bulletin"
