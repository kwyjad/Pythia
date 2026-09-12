# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Closing the loop: a committed feed extends, and history is re-walked.

This is the part that decides whether extending SPEI-3 changes anything at
all. Committing a CSV moves no rulebook key, so the drought fingerprint does
not move, so ``completed_months`` returns every month already marked ``ok``
and the backcast skips them. The file lands and nothing re-walks: the
``indicator_no_coverage`` cells the feed exists to decide stay exactly as
they are.

So the producer names the months that gained coverage in the committed status
file, and the nightly backcast frees exactly those ledger rows — once, by the
request's own token, because a restale that reapplies every night is a
treadmill rather than a fix.

Also here: the staleness reader. The rulebook entry stays ``required: false``
with ``absence_means_no_drought: false``, so a missing or stale file
suppresses nothing and changes no verdict — which is exactly why nothing else
would notice the producer had stopped.
"""

from __future__ import annotations

import datetime as dt
import json

import duckdb
import pytest

from resolver.diagnostics import feed_status as fs
from resolver.hazard_resolution.backcast import apply_feed_restale
from resolver.hazard_resolution.schema import ensure_haz_schema


@pytest.fixture()
def con():
    connection = duckdb.connect(":memory:")
    ensure_haz_schema(connection)
    try:
        yield connection
    finally:
        connection.close()


def _ledger(con, hazard: str, *months: str) -> None:
    for ym in months:
        con.execute(
            "INSERT INTO haz_backcast_progress (hazard, ym, status, cells) "
            "VALUES (?, ?, 'ok', 12)",
            [hazard, ym],
        )


def _status_file(
    tmp_path, *, months, token="3m-abc", hazard="DR", newest="2026-08",
    dataset_type=None,
):
    path = tmp_path / "spei3_status.json"
    payload = {
        "feed": "spei3_country_means",
        "status": "ok",
        "newest_month": newest,
        "oldest_month": "2016-01",
        "months": 128,
        "rows": 30000,
        "countries": 246,
        "coverage": {"cells": 29000, "nearest_cell": 1000},
        "restale": {
            "hazard": hazard, "token": token, "months": list(months),
            "requested_at": "2026-09-10T00:00:00+00:00",
        },
    }
    if dataset_type is not None:
        payload["newest_month_dataset_type"] = dataset_type
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# The restale
# ---------------------------------------------------------------------------


def test_the_named_months_are_freed_from_the_resume_ledger(con, tmp_path):
    _ledger(con, "DR", "2016-01", "2016-02", "2016-03")
    path = _status_file(tmp_path, months=["2016-01", "2016-02"])

    outcome = apply_feed_restale(con, "DR", status_path=path)

    assert outcome["applied"] is True
    assert outcome["rows_deleted"] == 2
    remaining = {
        row[0] for row in con.execute(
            "SELECT ym FROM haz_backcast_progress WHERE hazard = 'DR'"
        ).fetchall()
    }
    assert remaining == {"2016-03"}


def test_a_month_the_request_does_not_name_is_untouched(con, tmp_path):
    """A routine monthly append must not re-walk ten years."""

    _ledger(con, "DR", *[f"2016-{m:02d}" for m in range(1, 13)])
    path = _status_file(tmp_path, months=["2026-08"])

    apply_feed_restale(con, "DR", status_path=path)

    assert con.execute(
        "SELECT COUNT(*) FROM haz_backcast_progress WHERE hazard = 'DR'"
    ).fetchone()[0] == 12


def test_the_request_is_applied_exactly_once(con, tmp_path):
    """Without the token the nightly run would free the same months, re-walk
    them, and free them again forever."""

    _ledger(con, "DR", "2016-01")
    path = _status_file(tmp_path, months=["2016-01"], token="1m-deadbeef")

    first = apply_feed_restale(con, "DR", status_path=path)
    _ledger(con, "DR", "2016-01")  # the walk re-recorded it as complete
    second = apply_feed_restale(con, "DR", status_path=path)

    assert first["applied"] is True
    assert second["applied"] is False
    assert "already applied" in str(second["reason"])
    assert con.execute(
        "SELECT COUNT(*) FROM haz_backcast_progress WHERE hazard = 'DR'"
    ).fetchone()[0] == 1


def test_a_new_extension_is_applied_even_after_an_earlier_one(con, tmp_path):
    _ledger(con, "DR", "2016-01", "2026-08")
    apply_feed_restale(
        con, "DR", status_path=_status_file(tmp_path, months=["2016-01"], token="a"),
    )
    outcome = apply_feed_restale(
        con, "DR", status_path=_status_file(tmp_path, months=["2026-08"], token="b"),
    )
    assert outcome["applied"] is True
    assert outcome["rows_deleted"] == 1


def test_another_hazard_is_never_restaled_by_the_drought_feed(con, tmp_path):
    _ledger(con, "TC", "2016-01")
    _ledger(con, "DR", "2016-01")
    path = _status_file(tmp_path, months=["2016-01"])

    assert apply_feed_restale(con, "TC", status_path=path)["applied"] is False
    assert con.execute(
        "SELECT COUNT(*) FROM haz_backcast_progress WHERE hazard = 'TC'"
    ).fetchone()[0] == 1


def test_a_missing_status_file_is_a_state_not_a_raise(con, tmp_path):
    outcome = apply_feed_restale(con, "DR", status_path=tmp_path / "absent.json")
    assert outcome["applied"] is False
    assert "no pending restale" in str(outcome["reason"])


def test_an_unparseable_status_file_costs_the_restale_and_nothing_else(con, tmp_path):
    path = tmp_path / "spei3_status.json"
    path.write_text("{not json", encoding="utf-8")
    outcome = apply_feed_restale(con, "DR", status_path=path)
    assert outcome["applied"] is False


def test_a_request_naming_months_no_ledger_row_holds_is_still_recorded(con, tmp_path):
    """The initial build frees months the ledger may never have held.

    Recording it anyway is what stops the request being re-applied on every
    night thereafter.
    """

    path = _status_file(tmp_path, months=["2016-01"], token="fresh")
    first = apply_feed_restale(con, "DR", status_path=path)
    second = apply_feed_restale(con, "DR", status_path=path)
    assert first["applied"] is True and first["rows_deleted"] == 0
    assert second["applied"] is False


def test_the_restale_is_recorded_where_the_next_run_can_read_it(con, tmp_path):
    _ledger(con, "DR", "2016-01")
    apply_feed_restale(
        con, "DR", status_path=_status_file(tmp_path, months=["2016-01"], token="t1"),
    )
    row = con.execute(
        "SELECT feed, hazard, token, months, rows_deleted FROM haz_feed_restale"
    ).fetchone()
    assert row[0] == "spei3_country_means"
    assert row[1] == "DR"
    assert row[2] == "t1"
    assert row[3] == "2016-01"
    assert row[4] == 1


# ---------------------------------------------------------------------------
# The staleness reader
# ---------------------------------------------------------------------------


def test_a_current_feed_is_ok(tmp_path):
    path = _status_file(tmp_path, months=[], newest="2026-08")
    status = fs.read_feed_status(path, today=dt.date(2026, 9, 10))
    assert status.state == fs.STATE_OK
    assert status.months_behind == 0


def test_the_products_own_lag_is_not_a_fault(tmp_path):
    """The state a HEALTHY feed is in, and the regression this pins.

    Copernicus updates the consolidated ERA5-Drought product 2-3 months
    behind real time, and on 2026-09-10 the real feed sat at exactly 3
    behind: newest 2026-05 against a previous complete month of 2026-08.
    The threshold was 3 at the time, so `lag > 3` was False by precisely one
    month — an alarm about to fire on every run while nothing was wrong.
    """

    path = _status_file(tmp_path, months=[], newest="2026-05")
    status = fs.read_feed_status(path, today=dt.date(2026, 9, 10))
    assert status.months_behind == fs.PRODUCT_LAG_MONTHS == 3
    assert status.state == fs.STATE_OK


def test_one_missed_producer_cycle_is_not_yet_a_fault(tmp_path):
    """The product's lag plus one month. A queued CDS job, a runner outage or
    a gate that failed closed on one bad month all land here."""

    path = _status_file(tmp_path, months=[], newest="2026-04")
    status = fs.read_feed_status(path, today=dt.date(2026, 9, 10))
    assert status.months_behind == 4
    assert status.state == fs.STATE_OK


def test_two_missed_producer_cycles_are_stale(tmp_path):
    """Not an accident. The producer runs monthly and fails closed, so two
    consecutive silent failures mean nobody is watching it."""

    path = _status_file(tmp_path, months=[], newest="2026-03")
    status = fs.read_feed_status(path, today=dt.date(2026, 9, 10))
    assert status.months_behind == 5
    assert status.state == fs.STATE_STALE
    assert "5 month(s) behind" in status.detail


def test_the_threshold_states_the_products_lag_and_the_tolerance_apart():
    """One literal would hide the difference between "the upstream is slow"
    and "our producer has stopped", and those want different responses."""

    assert fs.MAX_LAG_MONTHS == fs.PRODUCT_LAG_MONTHS + fs.MISSED_CYCLE_TOLERANCE_MONTHS
    assert fs.PRODUCT_LAG_MONTHS > 0 and fs.MISSED_CYCLE_TOLERANCE_MONTHS > 0
    for release in ("consolidated_dataset", "intermediate_dataset"):
        assert fs.max_lag_for(release) == (
            fs.PRODUCT_LAG_BY_DATASET_TYPE[release] + fs.MISSED_CYCLE_TOLERANCE_MONTHS
        )


def test_the_threshold_follows_the_release_the_newest_month_came_from(tmp_path):
    """The lag is a property of the product the producer actually requests.

    The intermediate release runs a single month behind, so a feed serving
    it and sitting three months back is NOT healthy — while the identical
    lag on the consolidated release is exactly what a healthy feed looks
    like. A fixed literal cannot say both, and this producer asks for both.
    """

    today = dt.date(2026, 9, 10)
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    consolidated = fs.read_feed_status(
        _status_file(tmp_path / "a", months=[], newest="2026-05",
                     dataset_type="consolidated_dataset"),
        today=today,
    )
    intermediate = fs.read_feed_status(
        _status_file(tmp_path / "b", months=[], newest="2026-05",
                     dataset_type="intermediate_dataset"),
        today=today,
    )
    assert consolidated.months_behind == intermediate.months_behind == 3
    assert consolidated.state == fs.STATE_OK
    assert intermediate.state == fs.STATE_STALE
    assert "intermediate_dataset release" in intermediate.detail


def test_a_healthy_intermediate_feed_is_not_reported_stale(tmp_path):
    """One month behind is the intermediate release doing its job."""

    status = fs.read_feed_status(
        _status_file(tmp_path, months=[], newest="2026-07",
                     dataset_type="intermediate_dataset"),
        today=dt.date(2026, 9, 10),
    )
    assert status.months_behind == 1
    assert status.state == fs.STATE_OK
    assert status.max_lag_months == 2


def test_a_status_file_that_names_no_release_keeps_the_wider_bound(tmp_path):
    """Written before the producer asked for more than one release.

    Every month such a file covers came from the consolidated one, so the
    consolidated bound is the right one — and it is the WIDER of the two,
    which is what stops the switch itself reporting a fault nobody could act
    on any faster than the next producer run.
    """

    status = fs.read_feed_status(
        _status_file(tmp_path, months=[], newest="2026-05"),
        today=dt.date(2026, 9, 10),
    )
    assert status.newest_month_dataset_type == ""
    assert status.max_lag_months == fs.MAX_LAG_MONTHS == 4
    assert status.state == fs.STATE_OK


def test_a_feed_that_has_never_been_produced_says_absent_not_stale(tmp_path):
    """Different faults, different repairs. Calling both "stale" sends the
    reader to the wrong one."""

    status = fs.read_feed_status(tmp_path / "nothing.json")
    assert status.state == fs.STATE_ABSENT
    assert "does not exist" in status.detail


def test_an_unparseable_status_file_says_unreadable(tmp_path):
    path = tmp_path / "spei3_status.json"
    path.write_text("[]", encoding="utf-8")
    assert fs.read_feed_status(path).state == fs.STATE_UNREADABLE


def test_a_failed_producer_run_is_reported_as_failed(tmp_path):
    path = tmp_path / "spei3_status.json"
    path.write_text(json.dumps({
        "status": "failed", "newest_month": "2026-08",
        "last_failure": {"run_id": "9", "reason": "a gate failed"},
    }), encoding="utf-8")
    status = fs.read_feed_status(path, today=dt.date(2026, 9, 10))
    assert status.state == fs.STATE_FAILED
    assert "a gate failed" in status.detail


def test_months_still_owed_are_reported_as_incomplete(tmp_path):
    path = tmp_path / "spei3_status.json"
    path.write_text(json.dumps({
        "status": "incomplete", "newest_month": "2026-08",
        "months_owed": ["2019-01", "2019-02"],
    }), encoding="utf-8")
    status = fs.read_feed_status(path, today=dt.date(2026, 9, 10))
    assert status.state == fs.STATE_INCOMPLETE
    assert status.months_owed == ["2019-01", "2019-02"]


def test_a_month_lag_is_calendar_arithmetic_across_a_year_boundary(tmp_path):
    """Thirty-day jumps are how an ACAPS window asked for Mar, Jan, Dec, Dec."""

    assert fs.months_behind("2025-11", "2026-02") == 3
    assert fs.months_behind("2026-02", "2026-02") == 0
    assert fs.months_behind("nonsense", "2026-02") is None


def test_a_request_with_no_token_is_not_a_request(tmp_path):
    """A half-written status file must not free ten years of ledger."""

    path = tmp_path / "spei3_status.json"
    path.write_text(json.dumps({
        "newest_month": "2026-08",
        "restale": {"hazard": "DR", "months": ["2016-01"]},
    }), encoding="utf-8")
    assert fs.spei3_restale_request(path) is None


def test_the_rulebook_entry_still_refuses_to_suppress_a_zero():
    """Not up for negotiation, and the reason the staleness check exists.

    A missing or stale SPEI file must change no verdict — which is precisely
    why nothing but a diagnostic would complain about it.
    """

    from resolver.hazard_resolution.rulebook import load_rulebook

    entries = load_rulebook().get("drought.indicators.entries")
    spei3 = next(e for e in entries if e["name"] == "spei3")
    assert spei3["required"] is False
    assert spei3["absence_means_no_drought"] is False
