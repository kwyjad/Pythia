# Pythia / Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.
"""Rainfall is shown as the chance of a dry and of a wet month (Oct 2026).

The outlook used to call a month below-normal when the ensemble mean fell
0.5 mm/day under the model climatology. That cut reads a different thing in
the Sahel dry season, where the whole month's rain is a few millimetres, and
in the Central American wet season, where 0.5 mm/day is noise. CPC's tercile
probabilities are relative in every climate: an ordinary month is 1 in 3 for
each end. The block shows those, per lead, and says when they are missing.
"""

from __future__ import annotations

import pytest

duckdb = pytest.importorskip("duckdb")

from horizon_scanner import seasonal_context as sc


@pytest.fixture()
def db(tmp_path):
    path = tmp_path / "p.duckdb"
    con = duckdb.connect(str(path))
    con.execute(
        "CREATE TABLE seasonal_forecasts (iso3 TEXT, variable TEXT, lead_months INTEGER, "
        "anomaly_value DOUBLE, tercile_category TEXT, forecast_issue_date DATE, units TEXT)"
    )
    rows = []
    # Niger: wet-season leads carry probabilities; lead 3 is CPC's dry-season
    # mask, so it carries the anomaly alone.
    for lead, anom, below, above in ((1, -0.6, 0.52, 0.14), (2, -0.2, 0.41, 0.25), (3, -0.05, None, None)):
        rows.append(f"('NER','prate',{lead},{anom},'below_normal',DATE '2026-09-08','mm/day')")
        if below is not None:
            rows.append(f"('NER','prate_prob_below',{lead},{below},'below_normal',DATE '2026-09-08','probability')")
            rows.append(f"('NER','prate_prob_above',{lead},{above},'near_normal',DATE '2026-09-08','probability')")
    # Guatemala: a large mm/day anomaly that is an ordinary wet-season month.
    rows.append("('GTM','prate',1,-0.9,'below_normal',DATE '2026-09-08','mm/day')")
    rows.append("('GTM','prate_prob_below',1,0.34,'near_normal',DATE '2026-09-08','probability')")
    rows.append("('GTM','prate_prob_above',1,0.31,'near_normal',DATE '2026-09-08','probability')")
    # Mali: no probability at all in this issue.
    rows.append("('MLI','prate',1,0.1,'near_normal',DATE '2026-09-08','mm/day')")
    con.execute("INSERT INTO seasonal_forecasts VALUES " + ",".join(rows))
    con.close()
    return f"duckdb:///{path}"


def test_each_lead_shows_dry_and_wet_chance_beside_one_in_three(db):
    out = sc.load_seasonal_forecasts("NER", db_url=db)
    detail = out["nmme_precip_detail"]
    assert "Lead 1: dry 52%, wet 14% (ordinary 33% each)" in detail
    assert "Lead 2: dry 41%, wet 25% (ordinary 33% each)" in detail
    assert "ordinary month is 33% for each" in out["nmme_precip_outlook"]
    assert "averaged over leads 1-2" in out["nmme_precip_outlook"]


def test_no_category_is_cut_from_the_mm_per_day_anomaly(db):
    out = sc.load_seasonal_forecasts("GTM", db_url=db)
    rain = out["nmme_precip_outlook"] + out["nmme_precip_detail"]
    # -0.9 mm/day crossed the old 0.5 cut and read "below-normal"; the
    # probabilities say this month is ordinary, and the block says so.
    assert "below-normal" not in rain.lower()
    assert "near-normal" not in rain.lower()
    assert "dry 34%, wet 31%" in rain
    assert "-0.90 mm/day" in rain


def test_a_lead_with_no_probability_says_so(db):
    out = sc.load_seasonal_forecasts("NER", db_url=db)
    assert "Lead 3: no tercile probability" in out["nmme_precip_detail"]
    assert "Lead 3 carries no probability" in out["nmme_precip_outlook"]


def test_the_reason_for_a_missing_probability_is_stated_once(db):
    # Niger, 8 Oct 2026 issue: all seven leads masked put the full sentence in
    # the prompt seven times.
    out = sc.load_seasonal_forecasts("MLI", db_url=db)
    rain = out["nmme_precip_outlook"] + out["nmme_precip_detail"]
    assert rain.count("dry-season or arid mask") == 1


def test_a_country_with_no_probability_says_so_in_the_outlook(db):
    out = sc.load_seasonal_forecasts("MLI", db_url=db)
    assert out["nmme_precip_outlook"].startswith("Rainfall tercile probabilities unavailable")
    assert "dry-season or arid mask" in out["nmme_precip_outlook"]


def test_format_names_a_missing_end_as_unavailable():
    outlook, detail = sc.format_rainfall_block(
        [], [{"lead_months": 1, "anomaly_value": 0.6}], [],
    )
    assert "dry 60%, wet unavailable" in detail
