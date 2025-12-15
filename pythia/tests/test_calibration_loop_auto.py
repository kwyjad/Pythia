# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

from pythia.tools import calibration_loop


class _Sample(SimpleNamespace):
    """Lightweight sample stand-in for calibration grouping."""


def test_maybe_run_calibration_skips_when_no_eligible(monkeypatch):
    calls = []

    def _fake_load_samples(conn, as_of_month):  # noqa: ARG001
        return []

    def _fake_group_by_hazard_metric(samples):  # noqa: ARG001
        return {}

    def _fake_compute_calibration_pythia(**kwargs):  # noqa: ARG001
        raise AssertionError("compute_calibration_pythia should not be called")

    class _Conn:
        pass

    monkeypatch.setattr(calibration_loop, "_load_samples", _fake_load_samples)
    monkeypatch.setattr(calibration_loop, "_group_by_hazard_metric", _fake_group_by_hazard_metric)
    monkeypatch.setattr(calibration_loop, "compute_calibration_pythia", _fake_compute_calibration_pythia)
    monkeypatch.setattr(calibration_loop.duckdb_io, "get_db", lambda db_url: calls.append(db_url) or _Conn())
    monkeypatch.setattr(calibration_loop.duckdb_io, "close_db", lambda conn: calls.append(conn))

    calibration_loop.maybe_run_calibration(
        db_url="duckdb:///dummy.duckdb", as_of=date(2025, 1, 15), min_questions=30
    )

    assert isinstance(calls[0], str)
    assert isinstance(calls[-1], _Conn)


def test_maybe_run_calibration_runs_when_eligible(monkeypatch):
    compute_calls = []

    n_samples = 35
    samples = [_Sample(question_key=f"q{i}", hazard_code="HZ", metric="M") for i in range(n_samples)]

    def _fake_load_samples(conn, as_of_month):  # noqa: ARG001
        return samples

    def _fake_group_by_hazard_metric(samples_in):  # noqa: ARG001
        return {("HZ", "M"): samples}

    def _fake_compute_calibration_pythia(**kwargs):
        compute_calls.append(kwargs)

    class _Conn:
        pass

    monkeypatch.setattr(calibration_loop, "_load_samples", _fake_load_samples)
    monkeypatch.setattr(calibration_loop, "_group_by_hazard_metric", _fake_group_by_hazard_metric)
    monkeypatch.setattr(calibration_loop, "compute_calibration_pythia", _fake_compute_calibration_pythia)
    monkeypatch.setattr(calibration_loop.duckdb_io, "get_db", lambda db_url: _Conn())
    monkeypatch.setattr(calibration_loop.duckdb_io, "close_db", lambda conn: None)

    as_of = date(2025, 1, 15)
    calibration_loop.maybe_run_calibration(
        db_url="duckdb:///dummy.duckdb", as_of=as_of, min_questions=30
    )

    assert len(compute_calls) == 1
    assert compute_calls[0]["as_of"] == as_of


def test_maybe_run_calibration_uses_default_db_url(monkeypatch):
    compute_calls = []
    seen_db_urls = []

    default_db_url = "duckdb:///default.duckdb"

    samples = [_Sample(question_key=f"q{i}", hazard_code="HZ", metric="M") for i in range(3)]

    def _fake_load_samples(conn, as_of_month):  # noqa: ARG001
        return samples

    def _fake_group_by_hazard_metric(samples_in):  # noqa: ARG001
        return {("HZ", "M"): samples}

    def _fake_compute_calibration_pythia(**kwargs):
        compute_calls.append(kwargs)

    class _Conn:
        pass

    monkeypatch.setattr(calibration_loop.duckdb_io, "DEFAULT_DB_URL", default_db_url)
    monkeypatch.setattr(calibration_loop, "_load_samples", _fake_load_samples)
    monkeypatch.setattr(calibration_loop, "_group_by_hazard_metric", _fake_group_by_hazard_metric)
    monkeypatch.setattr(calibration_loop, "compute_calibration_pythia", _fake_compute_calibration_pythia)
    monkeypatch.setattr(calibration_loop.duckdb_io, "get_db", lambda db_url: seen_db_urls.append(db_url) or _Conn())
    monkeypatch.setattr(calibration_loop.duckdb_io, "close_db", lambda conn: None)

    calibration_loop.maybe_run_calibration(as_of=date(2025, 1, 15), min_questions=3)

    assert seen_db_urls == [default_db_url]
    assert compute_calls and compute_calls[0]["db_url"] == default_db_url
