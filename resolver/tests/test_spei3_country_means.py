# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The SPEI-3 country-mean builder, and the feed it writes.

The drought path has no observation reaching back before the HDX and NMME
ingests began, so the backcast resolves INCONCLUSIVE for most of its
window. SPEI-3 is that observation, but turning a raster into a country
number is zonal statistics — which the rulebook says does not belong in a
resolution run. So it is computed once by hand and committed, and the
existing ``tabular`` provider reads the file.

Two things are worth testing and both are here. The zonal reduction is a
pure function over a plain grid, so CI can cover the rule that decides
which countries get a value WITHOUT xarray, netCDF or a CDS account. And
the provider must be able to read a committed path at all.

The rule that matters: a country smaller than a grid cell contains no cell
CENTRE and would come back empty from an ordinary mask — invisible in every
skip counter, which is how NMME reported "175 countries produced, 1
skipped" against 252 in the run. It is sampled from the nearest cell and
labelled, so a reader can tell a sampled country from a covered one.
"""

from __future__ import annotations

import csv
import datetime as dt
from dataclasses import dataclass

import pytest

from scripts import build_spei3_country_means as spei


@dataclass
class _Country:
    """A rectangular territory: the fields ``country_mean`` reads.

    Deliberately not a shapely box. The reduction takes its containment
    predicate as an argument precisely so these tests need no geometry
    stack — a file that skips wholesale on a missing dependency is a file
    whose failures nobody sees until CI. ``shapely_contains`` is the real
    predicate and is exercised by the boundary loader's own suite.
    """

    iso3: str
    bounds: tuple[float, float, float, float]


def _box(iso3: str, minx: float, miny: float, maxx: float, maxy: float) -> _Country:
    return _Country(iso3=iso3, bounds=(minx, miny, maxx, maxy))


def _contains(country: _Country, lon: float, lat: float) -> bool:
    minx, miny, maxx, maxy = country.bounds
    return minx <= lon <= maxx and miny <= lat <= maxy


def _grid(lats, lons, values) -> spei.Grid:
    return spei.Grid(lats=lats, lons=lons, values=values)


# ---------------------------------------------------------------------------
# Month arithmetic
# ---------------------------------------------------------------------------


def test_months_are_stepped_in_calendar_months():
    """Thirty-day jumps are how February goes unrequested."""

    months = spei.month_range("2016-11", "2017-02")
    assert months == ["2016-11", "2016-12", "2017-01", "2017-02"]


def test_a_single_month_range_is_that_month():
    assert spei.month_range("2016-01", "2016-01") == ["2016-01"]


def test_the_series_starts_where_the_hole_does():
    assert spei.DEFAULT_START_YM == "2016-01"


def test_the_newest_month_is_the_last_complete_one():
    """A partial month's anomaly is not the month's anomaly."""

    assert spei.previous_complete_month(dt.date(2026, 9, 8)) == "2026-08"
    assert spei.previous_complete_month(dt.date(2026, 1, 3)) == "2025-12"


# ---------------------------------------------------------------------------
# The zonal mean
# ---------------------------------------------------------------------------


def test_a_country_covering_several_cells_gets_their_mean():
    grid = _grid(
        lats=[0.0, 1.0],
        lons=[0.0, 1.0],
        values=[[-2.0, -1.0], [-2.0, -1.0]],
    )
    out = spei.country_mean(grid, _box("AAA", -0.5, -0.5, 1.5, 1.5), contains=_contains)

    assert out.coverage == spei.COVERAGE_CELLS
    assert out.n_cells == 4
    assert out.value == pytest.approx(-1.5, abs=1e-6)


def test_cells_are_weighted_by_the_cosine_of_their_latitude():
    """A degree of longitude is shorter near the poles; an unweighted mean
    overstates the high-latitude end of a country."""

    grid = _grid(lats=[0.0, 60.0], lons=[0.0], values=[[-2.0], [0.0]])
    out = spei.country_mean(grid, _box("AAA", -1.0, -1.0, 1.0, 61.0), contains=_contains)

    # cos(0) = 1, cos(60) = 0.5 -> (-2*1 + 0*0.5) / 1.5
    assert out.value == pytest.approx(-2.0 / 1.5, abs=1e-6)
    assert out.value < -1.0, "the equatorial cell must outweigh the polar one"


def test_a_missing_cell_is_skipped_not_read_as_zero():
    grid = _grid(lats=[0.0], lons=[0.0, 1.0], values=[[None, -2.0]])
    out = spei.country_mean(grid, _box("AAA", -0.5, -0.5, 1.5, 0.5), contains=_contains)

    assert (out.n_cells, out.value) == (1, pytest.approx(-2.0))


def test_a_nan_cell_is_missing_too():
    grid = _grid(lats=[0.0], lons=[0.0, 1.0], values=[[float("nan"), -2.0]])
    out = spei.country_mean(grid, _box("AAA", -0.5, -0.5, 1.5, 0.5), contains=_contains)

    assert (out.n_cells, out.value) == (1, pytest.approx(-2.0))


def test_cells_outside_the_territory_are_not_counted():
    grid = _grid(lats=[0.0, 50.0], lons=[0.0], values=[[-2.0], [5.0]])
    out = spei.country_mean(grid, _box("AAA", -1.0, -1.0, 1.0, 1.0), contains=_contains)

    assert (out.n_cells, out.value) == (1, pytest.approx(-2.0))


# ---------------------------------------------------------------------------
# The rule that matters: a country smaller than a cell
# ---------------------------------------------------------------------------


def test_a_country_smaller_than_a_cell_is_sampled_not_dropped():
    """The NMME fault, in the module that would otherwise repeat it.

    A mask built from cell CENTRES omits such a country entirely rather
    than skipping it, so it is invisible in every skip counter.
    """

    grid = _grid(lats=[0.0, 1.0], lons=[0.0, 1.0], values=[[-1.8, 0.0], [0.0, 0.0]])
    # A tenth of a degree wide, sitting between cell centres.
    out = spei.country_mean(grid, _box("MLT", 0.05, 0.05, 0.15, 0.15), contains=_contains)

    assert out.coverage == spei.COVERAGE_NEAREST
    assert out.n_cells == 0
    assert out.value == pytest.approx(-1.8)


def test_a_sampled_country_says_it_was_sampled():
    """Labelled, so a reader can tell it from a country of 200 cells."""

    grid = _grid(lats=[0.0], lons=[0.0], values=[[-1.0]])
    tiny = spei.country_mean(grid, _box("MLT", 0.4, 0.4, 0.5, 0.5), contains=_contains)
    big = spei.country_mean(grid, _box("BRA", -5.0, -5.0, 5.0, 5.0), contains=_contains)

    assert (tiny.coverage, big.coverage) == (spei.COVERAGE_NEAREST, spei.COVERAGE_CELLS)


def test_the_nearest_cell_is_the_nearest_one():
    # The marker is -3.5 rather than a round -9: since `SATURATION_ABS` a
    # value that extreme is treated as the index saturating rather than
    # measuring, so the nearest-cell path would skip it and this test would
    # be asserting the opposite of what it means to. A distinctive number
    # inside the valid range does the same job.
    grid = _grid(lats=[0.0], lons=[0.0, 10.0], values=[[-1.0, -3.5]])
    out = spei.country_mean(grid, _box("MLT", 9.9, -0.05, 10.0, 0.05), contains=_contains)

    assert out.value == pytest.approx(-3.5)


def test_sampling_can_be_switched_off():
    grid = _grid(lats=[0.0], lons=[0.0], values=[[-1.0]])
    out = spei.country_mean(
        grid, _box("MLT", 5.4, 5.4, 5.5, 5.5),
        nearest_when_uncovered=False, contains=_contains,
    )
    assert (out.value, out.coverage) == (None, spei.COVERAGE_NONE)


def test_an_all_missing_grid_yields_no_value_never_a_zero():
    grid = _grid(lats=[0.0], lons=[0.0], values=[[None]])
    out = spei.country_mean(grid, _box("AAA", -1.0, -1.0, 1.0, 1.0), contains=_contains)

    assert (out.value, out.coverage) == (None, spei.COVERAGE_NONE)


# ---------------------------------------------------------------------------
# Reduce and write
# ---------------------------------------------------------------------------


def test_reduce_writes_one_row_per_country_month():
    grids = {
        "2016-01": _grid([0.0], [0.0], [[-1.5]]),
        "2016-02": _grid([0.0], [0.0], [[-0.5]]),
    }
    countries = {"AAA": _box("AAA", -1.0, -1.0, 1.0, 1.0)}
    rows, report = spei.reduce_grids(grids, countries, contains=_contains, iso3s=["AAA"])

    assert report.rows == 2
    assert {(r["ym"], r["value"]) for r in rows} == {("2016-01", -1.5), ("2016-02", -0.5)}


def test_a_country_with_no_value_writes_no_row():
    """Absence is unknown. The entry is absence_means_no_drought: false
    precisely so it cannot be read as a quiet month."""

    grids = {"2016-01": _grid([0.0], [0.0], [[None]])}
    countries = {"AAA": _box("AAA", -1.0, -1.0, 1.0, 1.0)}
    rows, report = spei.reduce_grids(grids, countries, contains=_contains, iso3s=["AAA"])

    assert rows == []
    assert report.countries_never_valued == ["AAA"]


def test_an_iso3_with_no_boundary_is_named_not_counted():
    grids = {"2016-01": _grid([0.0], [0.0], [[-1.0]])}
    countries = {"AAA": _box("AAA", -1.0, -1.0, 1.0, 1.0)}
    _rows, report = spei.reduce_grids(grids, countries, contains=_contains, iso3s=["AAA", "ZZZ"])

    assert report.countries_without_boundary == ["ZZZ"]


def test_the_csv_carries_the_columns_the_tabular_provider_reads(tmp_path):
    rows = [
        {"iso3": "AAA", "ym": "2016-02", "value": -1.5, "coverage": "cells",
         "n_cells": 4},
        {"iso3": "AAA", "ym": "2016-01", "value": -0.5, "coverage": "cells",
         "n_cells": 4},
    ]
    out = tmp_path / "spei3.csv"
    spei.write_csv(rows, out)

    with open(out, encoding="utf-8", newline="") as fh:
        read = list(csv.DictReader(fh))
    assert [r["ym"] for r in read] == ["2016-01", "2016-02"], "sorted by month"
    assert set(read[0]) >= {"iso3", "ym", "value"}


def test_the_target_countries_come_from_countries_csv():
    """One country list, read with the BOM its file carries."""

    iso3s = spei.load_target_iso3s()
    assert "SOM" in iso3s and "ETH" in iso3s
    assert all(len(i) == 3 and i.isupper() for i in iso3s)


def test_nothing_on_the_resolution_path_imports_this_script():
    """It is a one-off. A resolution run must never call it."""

    import subprocess
    import sys

    # Only Python, and only an import or a module run. The rulebook names
    # the script in a COMMENT, on purpose — that is how a maintainer knows
    # which file writes the feed, and it executes nothing.
    found = subprocess.run(
        [
            "grep", "-rnE",
            r"(import|python -m scripts\.|from scripts)[^#]*build_spei3_country_means",
            "--include=*.py",
            "resolver/hazard_resolution", "resolver/tools", "resolver/connectors",
        ],
        capture_output=True, text=True, cwd=spei.REPO_ROOT,
    )
    assert found.stdout.strip() == "", (
        f"the resolution path imports the one-off builder:\n{found.stdout}"
    )


# ---------------------------------------------------------------------------
# The tabular provider reads the committed file
# ---------------------------------------------------------------------------


def test_a_repo_relative_path_is_a_valid_feed_address():
    """The rulebook must accept the path, or the entry can never be wired."""

    from resolver.hazard_resolution.rulebook import _is_feed_address

    assert _is_feed_address("resolver/data/spei3_country_means.csv")
    assert _is_feed_address("https://example.test/feed.csv")
    assert _is_feed_address("hdx-ckan://asap-hotspots-monthly")


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "/etc/passwd",
        "../../secrets.csv",
        "spei3",
        "ftp://example.test/feed.csv",
    ],
)
def test_a_path_that_is_not_a_committed_feed_is_refused(bad):
    """Narrow on purpose: a bare word here is a typo, and reading it as a
    path turns the loud validation failure it deserves into a quiet
    FileNotFoundError at fetch time."""

    from resolver.hazard_resolution.rulebook import _is_feed_address

    assert not _is_feed_address(bad)


def test_a_local_path_resolves_against_the_repo_not_the_cwd():
    """The pipeline runs these from several directories."""

    from resolver.hazard_resolution import drought_indicators as di

    resolved = di._local_path("resolver/data/countries.csv")
    assert resolved is not None and resolved.is_absolute()
    assert resolved == di._REPO_ROOT / "resolver" / "data" / "countries.csv"


def test_a_remote_address_is_not_treated_as_a_path():
    from resolver.hazard_resolution import drought_indicators as di

    assert di._local_path("https://example.test/x.csv") is None
    assert di._local_path("hdx-ckan://dataset") is None


def test_the_provider_reads_a_committed_csv(tmp_path):
    from resolver.hazard_resolution import drought_indicators as di

    feed = tmp_path / "spei3.csv"
    feed.write_text("iso3,ym,value\nSOM,2016-01,-1.4\n", encoding="utf-8")

    body, content_type = di._default_get(str(feed), 60)
    assert b"SOM" in body
    assert content_type == "text/csv"


def test_a_missing_committed_file_names_the_path_and_the_repair():
    """A feed pointed at a file nobody generated is the hand-maintained
    list failure; only the path says which script writes it."""

    from resolver.hazard_resolution import drought_indicators as di

    with pytest.raises(FileNotFoundError) as excinfo:
        di._default_get("resolver/data/does_not_exist.csv", 60)
    message = str(excinfo.value)
    assert "does_not_exist.csv" in message
    assert "build_spei3_country_means" in message


def test_the_spei3_entry_is_wired_as_the_task_specifies():
    from resolver.hazard_resolution.rulebook import load_rulebook

    entries = load_rulebook().get("drought.indicators.entries")
    spei3 = next(e for e in entries if e["name"] == "spei3")

    assert spei3["provider"] == "tabular"
    assert spei3["url"] == "resolver/data/spei3_country_means.csv"
    assert spei3["threshold"] == -1.0
    assert spei3["direction"] == "below"
    assert spei3["required"] is False
    # An anomaly feed, not an alerting one: a country absent from it is a
    # country nobody measured, and reading that as "no drought" is how an
    # ingestion gap becomes a quiet month.
    assert spei3["absence_means_no_drought"] is False


def test_a_missing_spei3_file_costs_only_that_entry():
    """`required: false` is what makes committing the wiring before the
    CSV safe: the entry reports unavailable and the others still answer."""

    from resolver.hazard_resolution.rulebook import load_rulebook

    entries = load_rulebook().get("drought.indicators.entries")
    spei3 = next(e for e in entries if e["name"] == "spei3")
    assert spei3["required"] is False
    assert load_rulebook().get("drought.indicators.min_available") >= 1


# ---------------------------------------------------------------------------
# The counts this feed is meant to move
# ---------------------------------------------------------------------------


def _seed_triggers(con, rows):
    import json as _json

    from resolver.hazard_resolution.schema import ensure_haz_schema

    ensure_haz_schema(con)
    for iso3, year, month, hazard, reason, run_type in rows:
        detail = {} if reason is None else {"no_row_reason": reason, "assessed": False}
        con.execute(
            "INSERT INTO haz_triggers (iso3, year, month, hazard, triggered,"
            " trigger_source, trigger_detail_json, run_type)"
            " VALUES (?, ?, ?, ?, FALSE, 'x', ?, ?)",
            [iso3, year, month, hazard, _json.dumps(detail), run_type],
        )


def test_coverage_counts_the_two_reason_codes_by_run_type():
    """The before/after measurement, split live from backcast: the hole is
    in the backcast, so a fall concentrated there is the feed working."""

    import duckdb

    con = duckdb.connect(":memory:")
    _seed_triggers(con, [
        ("SOM", 2018, 3, "DR", "indicator_no_coverage", "backcast"),
        ("ETH", 2018, 3, "DR", "indicator_no_coverage", "backcast"),
        ("KEN", 2020, 5, "DR", "indicator_too_few_feeds_for_zero", "backcast"),
        ("SDN", 2026, 7, "DR", "indicator_too_few_feeds_for_zero", "live"),
    ])

    assert spei.coverage_counts(con) == [
        {"reason_code": "indicator_no_coverage", "run_type": "backcast", "cells": 2},
        {"reason_code": "indicator_too_few_feeds_for_zero",
         "run_type": "backcast", "cells": 1},
        {"reason_code": "indicator_too_few_feeds_for_zero",
         "run_type": "live", "cells": 1},
    ]


def test_coverage_ignores_other_hazards_and_decided_cells():
    import duckdb

    con = duckdb.connect(":memory:")
    _seed_triggers(con, [
        ("PHL", 2019, 2, "FL", "indicator_no_coverage", "backcast"),
        ("TCD", 2019, 2, "DR", None, "backcast"),
        ("TCD", 2019, 3, "DR", "pending_before_freeze", "backcast"),
    ])

    assert spei.coverage_counts(con) == []


def test_the_two_reason_codes_are_the_ones_the_machine_writes():
    """Pinned against the ledger's own constants, so a rename is caught
    here rather than by a report that silently counts nothing."""

    from resolver.hazard_resolution import cell_ledger

    assert set(spei.COVERAGE_REASONS) == {
        cell_ledger.REASON_NO_COVERAGE,
        cell_ledger.REASON_TOO_FEW_FEEDS,
    }
