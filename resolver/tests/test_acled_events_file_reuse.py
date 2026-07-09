# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Tests for the ACLED single-fetch reuse path.

The connector stages its raw events (``acled_events_raw.csv`` + meta
sidecar) and ``acled_to_duckdb --events-file`` aggregates from that file
instead of re-downloading the same window from the ACLED API. Falls back
to the API when the staged data is missing, truncated, or doesn't cover
the requested window.
"""

import json

import pandas as pd
import pytest

from resolver.cli import acled_to_duckdb
from resolver.db import duckdb_io
from resolver.ingestion import acled_client as acled_client_module
from resolver.ingestion.acled_client import ACLEDClient


def _write_staged(
    tmp_path,
    *,
    start="2024-01-01",
    end="2024-03-31",
    truncated=False,
    rows=None,
    frame=None,
):
    csv_path = tmp_path / "acled_events_raw.csv"
    if frame is None:
        frame = pd.DataFrame(
            {
                "event_date": ["2024-01-05", "2024-01-20", "2024-02-10"],
                "iso3": ["AFG", "AFG", "ALB"],
                "country": ["Afghanistan", "Afghanistan", "Albania"],
                "fatalities": [2, 4, 1],
            }
        )
    frame.to_csv(csv_path, index=False)
    meta = {
        "start": start,
        "end": end,
        "rows": len(frame) if rows is None else rows,
        "truncated_by_deadline": truncated,
        "fetched_at": "2024-04-01T00:00:00+00:00",
    }
    (tmp_path / "acled_events_raw.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return csv_path


# ---------------------------------------------------------------------------
# _load_staged_events guards
# ---------------------------------------------------------------------------


def test_load_staged_events_ok(tmp_path):
    csv_path = _write_staged(tmp_path)
    frame, reason = acled_to_duckdb._load_staged_events(str(csv_path), "2024-01-01", "2024-03-31")
    assert frame is not None
    assert len(frame) == 3
    assert "staged events reused" in reason


def test_load_staged_events_missing_file(tmp_path):
    frame, reason = acled_to_duckdb._load_staged_events(
        str(tmp_path / "nope.csv"), "2024-01-01", "2024-03-31"
    )
    assert frame is None
    assert "not found" in reason


def test_load_staged_events_missing_sidecar(tmp_path):
    csv_path = _write_staged(tmp_path)
    (tmp_path / "acled_events_raw.meta.json").unlink()
    frame, reason = acled_to_duckdb._load_staged_events(str(csv_path), "2024-01-01", "2024-03-31")
    assert frame is None
    assert "sidecar" in reason


def test_load_staged_events_truncated(tmp_path):
    csv_path = _write_staged(tmp_path, truncated=True)
    frame, reason = acled_to_duckdb._load_staged_events(str(csv_path), "2024-01-01", "2024-03-31")
    assert frame is None
    assert "truncated" in reason


def test_load_staged_events_window_not_covered(tmp_path):
    csv_path = _write_staged(tmp_path, start="2024-02-01", end="2024-03-31")
    frame, reason = acled_to_duckdb._load_staged_events(str(csv_path), "2024-01-01", "2024-03-31")
    assert frame is None
    assert "does not cover" in reason

    csv_path = _write_staged(tmp_path, start="2024-01-01", end="2024-02-29")
    frame, reason = acled_to_duckdb._load_staged_events(str(csv_path), "2024-01-01", "2024-03-31")
    assert frame is None
    assert "does not cover" in reason


def test_load_staged_events_zero_rows(tmp_path):
    csv_path = _write_staged(tmp_path, rows=0)
    frame, reason = acled_to_duckdb._load_staged_events(str(csv_path), "2024-01-01", "2024-03-31")
    assert frame is None
    assert "zero rows" in reason


# ---------------------------------------------------------------------------
# monthly_fatalities(events_frame=...) aggregation equivalence
# ---------------------------------------------------------------------------


def _make_client(monkeypatch):
    monkeypatch.setattr(
        acled_client_module.acled_auth,
        "get_access_token",
        lambda: "token",
    )
    return ACLEDClient()


def test_monthly_fatalities_from_events_frame(monkeypatch):
    client = _make_client(monkeypatch)

    def _boom(self, *_args, **_kwargs):  # pragma: no cover - guard
        raise AssertionError("fetch_events must not be called when events_frame is given")

    monkeypatch.setattr(client, "fetch_events", _boom.__get__(client, ACLEDClient))

    events = pd.DataFrame(
        {
            "event_date": ["2024-01-05", "2024-01-20", "2024-02-10"],
            "iso3": ["afg", "AFG", "ALB"],
            "country": ["Afghanistan", "Afghanistan", "Albania"],
            "fatalities": [2, 4, 1],
        }
    )
    frame = client.monthly_fatalities("2024-01-01", "2024-03-31", events_frame=events)

    assert set(frame.columns) == {"iso3", "month", "fatalities", "source", "updated_at"}
    afg = frame[frame["iso3"] == "AFG"].iloc[0]
    alb = frame[frame["iso3"] == "ALB"].iloc[0]
    assert afg["fatalities"] == 6
    assert afg["month"].strftime("%Y-%m-%d") == "2024-01-01"
    assert alb["fatalities"] == 1


def test_monthly_fatalities_events_frame_window_filter(monkeypatch):
    client = _make_client(monkeypatch)
    events = pd.DataFrame(
        {
            "event_date": ["2023-12-31", "2024-01-05", "2024-04-01"],
            "iso3": ["AFG", "AFG", "AFG"],
            "country": ["Afghanistan"] * 3,
            "fatalities": [100, 4, 200],
        }
    )
    frame = client.monthly_fatalities("2024-01-01", "2024-03-31", events_frame=events)

    # Only the in-window event survives — the staged file may cover a wider
    # window than the requested aggregation range.
    assert len(frame) == 1
    assert frame.iloc[0]["fatalities"] == 4


def test_monthly_fatalities_events_frame_iso3_fallback(monkeypatch):
    client = _make_client(monkeypatch)
    events = pd.DataFrame(
        {
            "event_date": ["2024-01-05"],
            "iso3": [None],
            "country": ["Kenya"],
            "fatalities": [3],
        }
    )
    frame = client.monthly_fatalities("2024-01-01", "2024-01-31", events_frame=events)

    assert len(frame) == 1
    assert frame.iloc[0]["iso3"] == "KEN"
    assert frame.iloc[0]["fatalities"] == 3


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def _run_cli_with_capture(tmp_path, monkeypatch, args):
    captured = {}
    result_frame = pd.DataFrame(
        {
            "iso3": ["KEN"],
            "month": pd.to_datetime(["2024-01-01"]),
            "fatalities": [1],
            "source": ["ACLED"],
            "updated_at": pd.to_datetime(["2024-01-01T00:00:00Z"], utc=True),
        }
    )

    class _CaptureClient:
        def monthly_fatalities(self, *_args, **kwargs):
            captured["events_frame"] = kwargs.get("events_frame")
            return result_frame.copy()

    monkeypatch.setattr(acled_to_duckdb, "ACLEDClient", _CaptureClient)
    diag_dir = tmp_path / "diagnostics" / "acled"
    monkeypatch.setattr(acled_to_duckdb, "ACLED_DIAGNOSTICS_DIR", diag_dir)
    monkeypatch.setattr(acled_to_duckdb, "ACLED_DUCKDB_SUMMARY_PATH", diag_dir / "duckdb_summary.md")

    exit_code = acled_to_duckdb.run(args)
    return exit_code, captured


@pytest.mark.duckdb
def test_cli_uses_staged_events(tmp_path, monkeypatch):
    pytest.importorskip("duckdb")
    if not duckdb_io.DUCKDB_AVAILABLE:
        pytest.skip("DuckDB module not available")

    csv_path = _write_staged(tmp_path)
    db_path = tmp_path / "acled.duckdb"
    exit_code, captured = _run_cli_with_capture(
        tmp_path,
        monkeypatch,
        [
            "--start", "2024-01-01",
            "--end", "2024-03-31",
            "--db", str(db_path),
            "--events-file", str(csv_path),
        ],
    )
    assert exit_code == 0
    assert captured["events_frame"] is not None
    assert len(captured["events_frame"]) == 3


@pytest.mark.duckdb
def test_cli_falls_back_when_staged_missing(tmp_path, monkeypatch):
    pytest.importorskip("duckdb")
    if not duckdb_io.DUCKDB_AVAILABLE:
        pytest.skip("DuckDB module not available")

    db_path = tmp_path / "acled.duckdb"
    exit_code, captured = _run_cli_with_capture(
        tmp_path,
        monkeypatch,
        [
            "--start", "2024-01-01",
            "--end", "2024-03-31",
            "--db", str(db_path),
            "--events-file", str(tmp_path / "missing.csv"),
        ],
    )
    assert exit_code == 0
    assert captured["events_frame"] is None


# ---------------------------------------------------------------------------
# Connector staging
# ---------------------------------------------------------------------------


def test_stage_raw_events_writes_csv_and_meta(tmp_path, monkeypatch):
    staging_path = tmp_path / "staging" / "acled_events_raw.csv"
    monkeypatch.setattr(acled_client_module, "EVENTS_STAGING_PATH", staging_path)

    records = [
        {
            "event_date": "2024-01-05",
            "iso3": "AFG",
            "country": "Afghanistan",
            "event_type": "Battles",
            "sub_event_type": "Armed clash",
            "fatalities": "2",
            "notes": "a very long note that must not be staged",
        },
        {
            "event_date": "2024-02-10",
            "iso3": "ALB",
            "country": "Albania",
            "event_type": "Protests",
            "sub_event_type": "Peaceful protest",
            "fatalities": "0",
            "notes": "another note",
        },
    ]
    meta = {
        "start": "2024-01-01",
        "end": "2024-03-31",
        "truncated_by_deadline": False,
    }
    acled_client_module._stage_raw_events(records, meta)

    staged = pd.read_csv(staging_path)
    assert list(staged.columns) == ["event_date", "iso3", "country", "fatalities"]
    assert len(staged) == 2

    sidecar = json.loads(
        (staging_path.parent / "acled_events_raw.meta.json").read_text(encoding="utf-8")
    )
    assert sidecar["start"] == "2024-01-01"
    assert sidecar["end"] == "2024-03-31"
    assert sidecar["rows"] == 2
    assert sidecar["truncated_by_deadline"] is False
    assert sidecar["fetched_at"]


def test_stage_raw_events_records_truncation(tmp_path, monkeypatch):
    staging_path = tmp_path / "acled_events_raw.csv"
    monkeypatch.setattr(acled_client_module, "EVENTS_STAGING_PATH", staging_path)

    acled_client_module._stage_raw_events(
        [],
        {"start": "2024-01-01", "end": "2024-03-31", "truncated_by_deadline": True},
    )

    sidecar = json.loads(
        (tmp_path / "acled_events_raw.meta.json").read_text(encoding="utf-8")
    )
    assert sidecar["truncated_by_deadline"] is True
    assert sidecar["rows"] == 0


def test_staged_roundtrip_matches_direct_aggregation(tmp_path, monkeypatch):
    """End-to-end equivalence: stage raw records → load → aggregate ==
    aggregating the same events directly."""
    staging_path = tmp_path / "acled_events_raw.csv"
    monkeypatch.setattr(acled_client_module, "EVENTS_STAGING_PATH", staging_path)

    records = [
        {"event_date": "2024-01-05", "iso3": "AFG", "country": "Afghanistan", "fatalities": 2},
        {"event_date": "2024-01-20", "iso3": "AFG", "country": "Afghanistan", "fatalities": 4},
        {"event_date": "2024-02-10", "iso3": "ALB", "country": "Albania", "fatalities": 1},
    ]
    acled_client_module._stage_raw_events(
        records,
        {"start": "2024-01-01", "end": "2024-03-31", "truncated_by_deadline": False},
    )

    loaded, _reason = acled_to_duckdb._load_staged_events(
        str(staging_path), "2024-01-01", "2024-03-31"
    )
    assert loaded is not None

    client = _make_client(monkeypatch)
    via_staged = client.monthly_fatalities(
        "2024-01-01", "2024-03-31", events_frame=loaded
    )
    via_direct = client.monthly_fatalities(
        "2024-01-01", "2024-03-31", events_frame=pd.DataFrame(records)
    )

    drop_cols = ["updated_at"]  # timestamps differ between the two calls
    pd.testing.assert_frame_equal(
        via_staged.drop(columns=drop_cols),
        via_direct.drop(columns=drop_cols),
    )
