# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The CAST diagnostic must reach three different verdicts from evidence.

Account tier, schema change and upstream stop call for three different
fixes, and the whole point of the script is that it does not guess between
them. Network-free: the one HTTP seam is replaced.
"""

from __future__ import annotations

import pytest

from scripts.ci import diagnose_acled_cast as diag


def _patch(monkeypatch, body, status=200):
    monkeypatch.setattr(
        diag, "_fetch", lambda limit: (status, body, "")
    )


def _row(**overrides):
    row = {
        "country": "Somalia",
        "iso": 706,
        "admin1": "Banadir",
        "month": "december",
        "year": 2025,
        "timestamp": "2025-12-04T00:00:00",
        "total_forecast": 41.0,
        "battles_forecast": 12.0,
        "erv_forecast": 9.0,
        "vac_forecast": 20.0,
    }
    row.update(overrides)
    return row


# ---------------------------------------------------------------------------
# The three verdicts
# ---------------------------------------------------------------------------

def test_a_quota_limit_in_the_envelope_is_an_account_tier_problem(monkeypatch):
    """ACLED states it in the half of the response the connector discards.

    Free CAST access includes ten logins or downloads; a spent quota is
    reported in `messages` and `data_query_restrictions`, and the connector
    reads `data` and nothing else.
    """

    _patch(monkeypatch, {
        "status": 200,
        "success": True,
        "messages": ["Your download limit for this dataset has been reached."],
        "data_query_restrictions": {"downloads_remaining": 0},
        "data": [_row()],
    })

    report = diag.diagnose()
    assert report["verdict"] == "account_tier"
    assert "access@acleddata.com" in report["next_step"]
    # It must not also recommend a code change: the connector is fine.
    assert "Do not change the connector" in report["next_step"]


def test_period_shaped_columns_are_a_schema_change(monkeypatch):
    """CAST's methodology page describes a four-week rolling period ending on
    Fridays; the API documentation, revised Sept 2025, still says month/year."""

    _patch(monkeypatch, {
        "success": True,
        "data": [
            _row(period_start="2026-08-08", period_end="2026-09-04", month=None, year=None)
        ],
    })

    report = diag.diagnose()
    assert report["verdict"] == "schema_change"
    assert "period_start" in str(report["period_shaped_columns"])
    assert "rolling four-week period" in report["next_step"]


def test_a_recent_timestamp_beside_a_stale_month_is_also_a_schema_change(
    monkeypatch
):
    """The subtle form, and the one our own logs would show.

    The connector derives its issue date from `timestamp` but computes lead
    months from `month`/`year`, so a moved temporal unit produces a recent
    timestamp against a December vintage.
    """

    _patch(monkeypatch, {
        "success": True,
        "data": [_row(timestamp="2026-08-28T00:00:00")],
    })

    report = diag.diagnose()
    assert report["verdict"] == "schema_change"
    assert report["max_timestamp_date"] == "2026-08-28"
    assert report["max_year"] == 2025


def test_an_old_timestamp_with_no_other_signal_is_upstream(monkeypatch):
    """Nothing points at a quota or a changed shape, so ACLED simply stopped."""

    _patch(monkeypatch, {"success": True, "data": [_row()]})

    report = diag.diagnose()
    assert report["verdict"] == "upstream_stopped"
    assert "Escalate" in report["next_step"]
    # And it names the fallback we can build without ACLED.
    assert "acled_monthly_fatalities" in report["next_step"]


def test_no_rows_at_all_is_inconclusive_not_a_conclusion(monkeypatch):
    """One call did not decide it, and saying so is the honest answer."""

    _patch(monkeypatch, {"success": True, "data": []})

    report = diag.diagnose()
    assert report["verdict"] == "inconclusive"


# ---------------------------------------------------------------------------
# It must never take down a run
# ---------------------------------------------------------------------------

def test_an_unreachable_endpoint_is_a_finding_not_a_crash(monkeypatch):
    def boom(limit):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(diag, "_fetch", boom)
    report = diag.diagnose()
    assert report["verdict"] == "unreachable"
    assert "credentials" in report["next_step"]


def test_main_always_exits_zero(monkeypatch, tmp_path):
    """A diagnostic that can fail a run is a diagnostic people switch off."""

    def boom(limit):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(diag, "_fetch", boom)
    out = tmp_path / "cast.json"
    assert diag.main(["--out", str(out)]) == 0
    assert out.exists()


def test_the_envelope_is_recorded_whatever_the_verdict(monkeypatch):
    """It is the half of the response that says why, and dropping it is how
    the December 2025 stall survived two investigations."""

    _patch(monkeypatch, {
        "status": 200, "success": True, "count": 1, "last_update": "2025-12-04",
        "data": [_row()],
    })

    report = diag.diagnose()
    assert report["envelope"]["last_update"] == "2025-12-04"
    assert report["envelope"]["count"] == 1


def test_a_non_json_body_keeps_its_first_500_characters(monkeypatch):
    monkeypatch.setattr(
        diag, "_fetch", lambda limit: (503, {}, "<html>Service Unavailable</html>")
    )
    report = diag.diagnose()
    assert "Service Unavailable" in report["raw_body_head"]


@pytest.mark.parametrize("raw", ["1764806400", 1764806400, "2025-12-04T00:00:00Z"])
def test_timestamps_parse_in_every_shape_acled_has_used(raw):
    assert diag._max_timestamp([{"timestamp": raw}]) is not None
