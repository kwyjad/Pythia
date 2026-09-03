# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Three seasonal-TC outlooks that were stored as something they were not.

From the 2026-08 run: both BoM outlooks filed under season 1980-81, the TSR
August Atlantic update filed under the May issue date with the May figures,
and NOAA contributing zero forecasts with no comment. Network-free — each
test drives the parser over text in the shape the source publishes.
"""

from __future__ import annotations

import datetime as dt

import pytest


# ---------------------------------------------------------------------------
# BoM: a climatology baseline is not a season
# ---------------------------------------------------------------------------

class TestBomSeason:
    TODAY = dt.date(2026, 8, 1)

    def test_a_climatology_baseline_is_not_read_as_the_season(self):
        """The 2026-08 defect, exactly: both outlooks stored as 1980-81.

        BoM quotes its averages "since 1980-81" in the same shape a season is
        written, and a first-match regex over the page took it.
        """

        from horizon_scanner.seasonal_tc.bom_scraper import extract_season

        text = (
            "Averages are calculated from records since 1980-81. "
            "The 2026-27 Australian tropical cyclone season is expected to be "
            "below average."
        )
        assert extract_season(text, self.TODAY) == ("2026-27", 2026)

    def test_a_season_phrase_wins_over_an_earlier_pair(self):
        from horizon_scanner.seasonal_tc.bom_scraper import extract_season

        text = "The 1981-2010 climatology applies. The 2026-27 season outlook follows."
        assert extract_season(text, self.TODAY) == ("2026-27", 2026)

    def test_a_plausible_pair_is_used_when_no_season_phrase_exists(self):
        from horizon_scanner.seasonal_tc.bom_scraper import extract_season

        text = "Records since 1980-81 show that activity in 2026-27 will be near normal."
        assert extract_season(text, self.TODAY) == ("2026-27", 2026)

    def test_a_page_with_only_a_baseline_yields_nothing(self):
        """Nothing is better than 1980-81.

        A stored outlook labelled with a season nobody forecast is worse than
        an absent one: it looks like data.
        """

        from horizon_scanner.seasonal_tc.bom_scraper import extract_season

        assert extract_season("Averages since 1980-81.", self.TODAY) is None

    @pytest.mark.parametrize("written", ["2026-27", "2026/27", "2026–27"])
    def test_the_separators_bom_actually_uses(self, written):
        from horizon_scanner.seasonal_tc.bom_scraper import extract_season

        text = f"The {written} Australian tropical cyclone season outlook."
        assert extract_season(text, self.TODAY) == ("2026-27", 2026)


# ---------------------------------------------------------------------------
# TSR: an update quotes the forecast it supersedes
# ---------------------------------------------------------------------------

class TestTsrIssueDateAndType:
    AUGUST_PDF = (
        "TSR ATLANTIC HURRICANE FORECAST AUGUST UPDATE\n"
        "Issued: 5th August 2026\n\n"
        "Summary\n"
        "Our pre-season forecast, Issued: 21st May 2026, called for an "
        "above-normal season.\n"
    )

    def test_the_headers_own_date_wins_over_a_quoted_one(self):
        """The August update was stored under the May date, which collided
        with the real May forecast on the dedup key and replaced it."""

        from horizon_scanner.seasonal_tc.tsr_seasonal_extractor import (
            extract_issue_date,
        )

        assert extract_issue_date(self.AUGUST_PDF) == "2026-08-05"

    def test_the_latest_date_wins_when_the_header_carries_none(self):
        """A document cannot be issued before the forecasts it discusses."""

        from horizon_scanner.seasonal_tc.tsr_seasonal_extractor import (
            extract_issue_date,
        )

        text = "\n".join(
            ["filler"] * 400
            + ["Issued: 21st May 2026", "Issued: 5th August 2026"]
        )
        assert extract_issue_date(text) == "2026-08-05"

    def test_the_url_decides_the_forecast_type(self):
        """TSR names its own products in its filenames, and a filename is not
        prose that can be confused by a summary paragraph."""

        from horizon_scanner.seasonal_tc.tsr_seasonal_extractor import (
            extract_forecast_type,
        )

        assert extract_forecast_type(
            self.AUGUST_PDF, "2026-08-05", issue_month="August"
        ) == "august_update"

    def test_text_sniffing_alone_reproduces_the_defect(self):
        """Kept as evidence: this is what the run actually did."""

        from horizon_scanner.seasonal_tc.tsr_seasonal_extractor import (
            extract_forecast_type,
        )

        assert extract_forecast_type(self.AUGUST_PDF, "2026-08-05") == "pre_season"

    def test_two_outlooks_for_one_basin_and_season_cannot_share_a_date(self):
        from horizon_scanner.seasonal_tc.tsr_seasonal_extractor import (
            SeasonalForecast,
            _warn_on_colliding_issue_dates,
        )

        problems = _warn_on_colliding_issue_dates([
            SeasonalForecast(
                basin="ATL", season_year=2026, issue_date="2026-05-21",
                forecast_type="pre_season",
            ),
            SeasonalForecast(
                basin="ATL", season_year=2026, issue_date="2026-05-21",
                forecast_type="august_update",
            ),
        ])
        assert len(problems) == 1
        assert "misread" in problems[0]

    def test_distinct_dates_raise_nothing(self):
        from horizon_scanner.seasonal_tc.tsr_seasonal_extractor import (
            SeasonalForecast,
            _warn_on_colliding_issue_dates,
        )

        assert _warn_on_colliding_issue_dates([
            SeasonalForecast(
                basin="ATL", season_year=2026, issue_date="2026-05-21",
                forecast_type="pre_season",
            ),
            SeasonalForecast(
                basin="ATL", season_year=2026, issue_date="2026-08-05",
                forecast_type="august_update",
            ),
        ]) == []


# ---------------------------------------------------------------------------
# NOAA: a hand-maintained URL list that stopped at 2025
# ---------------------------------------------------------------------------

class TestNoaaCandidates:
    def test_a_year_with_no_curated_entry_still_has_candidates(self):
        """The actual cause of "NOAA returned zero forecasts with no comment":
        KNOWN_URLS had 2025 and nothing else, so a 2026 run asked for nothing."""

        from horizon_scanner.seasonal_tc.noaa_cpc_scraper import (
            KNOWN_URLS,
            candidate_urls,
        )

        assert 2026 not in KNOWN_URLS
        urls = candidate_urls(2026)
        assert urls
        assert all("2026" in url for url in urls.values())

    def test_all_three_activity_wordings_are_tried(self):
        """NOAA titles the release after the forecast, so the slug cannot be
        templated exactly — only generated."""

        from horizon_scanner.seasonal_tc.noaa_cpc_scraper import candidate_urls

        joined = " ".join(candidate_urls(2026).values())
        for word in ("above-normal", "near-normal", "below-normal"):
            assert word in joined

    def test_curated_entries_are_kept_and_not_duplicated(self):
        from horizon_scanner.seasonal_tc.noaa_cpc_scraper import (
            KNOWN_URLS,
            candidate_urls,
        )

        urls = candidate_urls(2025)
        assert urls["ATL_initial"] == KNOWN_URLS[2025]["ATL_initial"]
        values = list(urls.values())
        assert len(values) == len(set(values))

    def test_every_basin_is_dispatchable_by_prefix(self):
        """Generated keys are "CP_generated_0"; an equality test on "CP"
        silently dropped all of them."""

        from horizon_scanner.seasonal_tc.noaa_cpc_scraper import candidate_urls

        prefixes = {key.split("_")[0] for key in candidate_urls(2026)}
        assert {"ATL", "CP", "ENP"} <= prefixes
