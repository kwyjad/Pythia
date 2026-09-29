# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Tests for pythia.tools.score_views — point-to-SPD conversion and scoring."""

from __future__ import annotations

import math

import pytest

from pythia.tools.score_views import (
    LOGNORMAL_SIGMA,
    MIN_POINT_FOR_LOGNORMAL,
    point_to_spd_fatalities,
)


class TestPointToSpdFatalities:
    """Tests for the log-normal point-to-SPD conversion."""

    def test_sums_to_one_typical(self):
        """SPD sums to 1.0 for typical forecasts."""
        for pf in [0.1, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0]:
            spd = point_to_spd_fatalities(pf)
            assert abs(sum(spd) - 1.0) < 1e-9, f"SPD for pf={pf} sums to {sum(spd)}"

    def test_sums_to_one_various_sigmas(self):
        """SPD sums to 1.0 for different sigma values."""
        for sigma in [0.3, 0.5, 1.0, 1.5, 2.0, 2.5]:
            spd = point_to_spd_fatalities(50.0, sigma=sigma)
            assert abs(sum(spd) - 1.0) < 1e-9, f"SPD for sigma={sigma} sums to {sum(spd)}"

    def test_bucket_count_matches_canonical(self):
        """SPD always has exactly K elements (K from pythia.buckets)."""
        from pythia.buckets import n_buckets_for

        k = n_buckets_for("FATALITIES")
        for pf in [0.0, 0.1, 10.0, 500.0, 10000.0]:
            spd = point_to_spd_fatalities(pf)
            assert len(spd) == k, f"Expected {k} buckets, got {len(spd)} for pf={pf}"

    def test_all_probabilities_positive(self):
        """All bucket probabilities are strictly positive."""
        for pf in [0.0, 0.1, 10.0, 500.0, 10000.0]:
            spd = point_to_spd_fatalities(pf)
            for i, p in enumerate(spd):
                assert p > 0, f"Bucket {i} has zero probability for pf={pf}"

    def test_near_zero_zero_bucket_dominates(self):
        """Near-zero forecasts should have the "0" bucket dominating."""
        spd = point_to_spd_fatalities(0.1)
        assert spd[0] > 0.80, f"Expected '0' bucket > 80%, got {spd[0]*100:.1f}%"
        for i in range(1, len(spd)):
            assert spd[i] <= spd[i - 1] + 1e-12, (
                "Near-zero should be (non-strictly) decreasing across buckets"
            )

    def test_zero_forecast(self):
        """Zero forecast should behave like near-zero."""
        spd = point_to_spd_fatalities(0.0)
        assert spd[0] > 0.80
        assert abs(sum(spd) - 1.0) < 1e-9

    def test_negative_forecast_clamped(self):
        """Negative forecast should be clamped to zero."""
        spd = point_to_spd_fatalities(-5.0)
        assert spd[0] > 0.80
        assert abs(sum(spd) - 1.0) < 1e-9

    def test_low_forecast_bottom_buckets_favored(self):
        """A forecast of 2.0 should favor the 1-<5 bucket over the top bucket."""
        spd = point_to_spd_fatalities(2.0)
        assert spd[1] > spd[-1], (
            "1-<5 bucket should be higher than >=1000 for low forecast"
        )

    def test_mid_forecast_spread(self):
        """A forecast of 50 should spread across middle buckets."""
        spd = point_to_spd_fatalities(50.0)
        # With sigma=1.0, forecast of 50 should have meaningful mass in the
        # 5-<25, 25-<100, and 100-<500 buckets (indices 2-4).
        assert spd[2] > 0.05, "5-<25 bucket should have some mass"
        assert spd[3] > 0.05, "25-<100 bucket should have some mass"
        assert spd[4] > 0.05, "100-<500 bucket should have some mass"

    def test_high_forecast_top_bucket_dominates(self):
        """A very high forecast (1000+) should favor the >=1000 bucket."""
        spd = point_to_spd_fatalities(1000.0)
        assert spd[-1] > spd[0], "Top bucket should dominate the '0' bucket"

    def test_very_high_forecast(self):
        """Extreme forecast (10000) should heavily weight the >=1000 bucket."""
        spd = point_to_spd_fatalities(10000.0)
        assert spd[-1] > 0.3, f"Expected >=1000 bucket > 30%, got {spd[-1]*100:.1f}%"
        assert abs(sum(spd) - 1.0) < 1e-9

    def test_monotonicity_with_increasing_forecast(self):
        """As forecast increases, top-bucket probability should generally increase."""
        prev_top = 0.0
        for pf in [1.0, 10.0, 50.0, 200.0, 1000.0]:
            spd = point_to_spd_fatalities(pf)
            # Top bucket should generally increase with forecast
            # (not strictly monotonic at every step due to log-normal shape,
            # but should increase across this range)
            if pf >= 50.0:
                assert spd[-1] >= prev_top - 0.01, (
                    f"Top bucket should not decrease significantly: "
                    f"pf={pf}, top={spd[-1]:.4f}, prev={prev_top:.4f}"
                )
            prev_top = spd[-1]

    def test_narrow_sigma_more_concentrated(self):
        """Narrower sigma should produce more concentrated SPDs."""
        narrow = point_to_spd_fatalities(50.0, sigma=0.3)
        wide = point_to_spd_fatalities(50.0, sigma=2.0)
        # Max probability should be higher with narrow sigma
        assert max(narrow) > max(wide), (
            "Narrow sigma should produce more concentrated distribution"
        )

    def test_threshold_boundary_below_min(self):
        """Forecast just below MIN_POINT_FOR_LOGNORMAL uses spike distribution."""
        spd = point_to_spd_fatalities(MIN_POINT_FOR_LOGNORMAL - 0.01)
        assert spd[0] == 0.90, "Below threshold should use spike distribution"

    def test_threshold_boundary_at_min(self):
        """Forecast at MIN_POINT_FOR_LOGNORMAL uses log-normal."""
        spd = point_to_spd_fatalities(MIN_POINT_FOR_LOGNORMAL)
        # Should use log-normal, not spike
        assert spd[0] != 0.90, "At threshold should use log-normal, not spike"
        assert abs(sum(spd) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# The reader must match what the ViEWS connector actually writes.
#
# On 2026-09-28 the scoring chain resolved 32 ACE/FATALITIES questions for
# August 2026 and score_views logged "Found 0 ViEWS<>Pythia matched forecast
# pairs": it filtered ``metric = 'FATALITIES'`` (a Pythia question metric)
# while the connector writes ``views_predicted_fatalities``. The source
# literal beside it had been fixed a month earlier for the same reason. These
# tests build the ViEWS rows with the connector's own transform, so a renamed
# literal on either side fails here instead of in a green scoring run.
# ---------------------------------------------------------------------------

from datetime import date  # noqa: E402

duckdb = pytest.importorskip("duckdb")

from pythia.tools import score_views  # noqa: E402
from resolver.connectors.views import ViewsConnector  # noqa: E402


def _connector_rows(iso3: str, issue: date, horizons: dict[int, float]) -> list[dict]:
    """ViEWS rows exactly as the connector emits them, one per lead month."""
    records = []
    for lead, value in horizons.items():
        month_index = issue.month + lead
        year = issue.year + (month_index - 1) // 12
        month = (month_index - 1) % 12 + 1
        records.append(
            {"isoab": iso3, "year": year, "month": month, "main_mean": value, "main_dich": 0.4}
        )
    return ViewsConnector()._transform(records, issue, "fatalities003_test")


def _build_db(path: str, cf_rows: list[dict]) -> None:
    con = duckdb.connect(path)
    con.execute(
        """
        CREATE TABLE conflict_forecasts (
            source VARCHAR, iso3 VARCHAR, hazard_code VARCHAR, metric VARCHAR,
            lead_months INTEGER, value DOUBLE, forecast_issue_date DATE,
            target_month DATE, model_version VARCHAR
        )
        """
    )
    for r in cf_rows:
        con.execute(
            "INSERT INTO conflict_forecasts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                r["source"], r["iso3"], r["hazard_code"], r["metric"], r["lead_months"],
                r["value"], r["forecast_issue_date"], r["target_month"], r["model_version"],
            ],
        )
    con.execute(
        "CREATE TABLE questions (question_id VARCHAR, iso3 VARCHAR, hazard_code VARCHAR, "
        "metric VARCHAR, hs_run_id VARCHAR)"
    )
    # No hs_runs table at all: the reader must not need one.
    con.execute(
        "INSERT INTO questions VALUES "
        "('NGA_ACE_FATALITIES_2026-08', 'NGA', 'ACE', 'FATALITIES', 'hs_orphan')"
    )
    con.execute(
        "CREATE TABLE resolutions (question_id VARCHAR, horizon_m INTEGER, "
        "observed_month VARCHAR, value DOUBLE)"
    )
    con.execute(
        "INSERT INTO resolutions VALUES ('NGA_ACE_FATALITIES_2026-08', 1, '2026-08', 800.0)"
    )
    con.close()


def test_the_literals_match_what_the_connector_writes() -> None:
    rows = _connector_rows("NGA", date(2026, 7, 1), {1: 650.0})
    fatality_rows = [r for r in rows if r["metric"] == score_views.VIEWS_FATALITIES_METRIC]
    assert fatality_rows, (
        f"the connector wrote metrics {sorted({r['metric'] for r in rows})}; "
        f"score_views reads {score_views.VIEWS_FATALITIES_METRIC!r}"
    )
    assert {r["source"] for r in fatality_rows} == {score_views.VIEWS_SOURCE}


def test_a_resolved_question_meets_its_lead_one_vintage(tmp_path) -> None:
    db = tmp_path / "views.duckdb"
    _build_db(str(db), _connector_rows("NGA", date(2026, 7, 1), {1: 650.0, 2: 700.0}))

    con = duckdb.connect(str(db))
    pairs = score_views._load_views_forecast_pairs(con)
    con.close()

    assert len(pairs) == 1
    pair = pairs[0]
    assert pair["question_id"] == "NGA_ACE_FATALITIES_2026-08"
    assert pair["horizon_m"] == 1
    assert pair["views_value"] == pytest.approx(650.0)
    assert pair["resolved_value"] == pytest.approx(800.0)


def test_scoring_writes_the_benchmark_rows(tmp_path) -> None:
    db = tmp_path / "views.duckdb"
    _build_db(str(db), _connector_rows("NGA", date(2026, 7, 1), {1: 650.0}))

    score_views.score_views(f"duckdb:///{db}")

    con = duckdb.connect(str(db))
    audit = con.execute("SELECT COUNT(*) FROM views_scored_forecasts").fetchone()[0]
    score_types = {
        r[0]
        for r in con.execute(
            "SELECT score_type FROM scores WHERE model_name = ?", [score_views.VIEWS_MODEL_NAME]
        ).fetchall()
    }
    con.close()
    assert audit == 1
    assert score_types == {"brier", "log", "crps"}


def test_the_probability_metric_is_never_scored_as_fatalities(tmp_path) -> None:
    db = tmp_path / "views.duckdb"
    rows = [
        r
        for r in _connector_rows("NGA", date(2026, 7, 1), {1: 650.0})
        if r["metric"] != score_views.VIEWS_FATALITIES_METRIC
    ]
    assert rows, "the connector should also emit the P(>=25 BRD) metric"
    _build_db(str(db), rows)

    con = duckdb.connect(str(db))
    assert score_views._load_views_forecast_pairs(con) == []
    con.close()


def test_retention_outlives_the_sixth_horizon_of_a_window() -> None:
    """The vintage that forecast a window must survive until its h6 is scored.

    Vintage 2026-07 forecasts the window starting 2026-08; horizon 6 is
    2027-01, resolved and scored around 2027-02-28, when the newest vintage
    is about 2027-02. The old rule kept two vintages and deleted 2026-07
    before horizon 2 could be scored.
    """
    from resolver.tools.fetch_conflict_forecasts import (
        KEEP_VINTAGES_PER_SOURCE,
        prune_old_vintages,
    )

    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE conflict_forecasts (source VARCHAR, forecast_issue_date DATE)")
    issues = [date(2026, 7, 1)] + [
        date(2026 + (7 + k - 1) // 12, (7 + k - 1) % 12 + 1, 1) for k in range(1, 8)
    ]  # 2026-07 .. 2027-02
    for d in issues:
        con.execute("INSERT INTO conflict_forecasts VALUES ('VIEWS', ?)", [d])
    prune_old_vintages(con, ["VIEWS"])
    kept = {r[0] for r in con.execute("SELECT forecast_issue_date FROM conflict_forecasts").fetchall()}
    assert date(2026, 7, 1) in kept
    assert KEEP_VINTAGES_PER_SOURCE >= 8

    for extra in range(20):
        con.execute("INSERT INTO conflict_forecasts VALUES ('VIEWS', ?)", [date(2030, 1, 1 + extra)])
    prune_old_vintages(con, ["VIEWS"])
    n = con.execute("SELECT COUNT(DISTINCT forecast_issue_date) FROM conflict_forecasts").fetchone()[0]
    assert n == KEEP_VINTAGES_PER_SOURCE
    con.close()
