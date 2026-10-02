# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The publish -> dashboard check (Oct 2026). Network is stubbed throughout."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from scripts.ci import check_dashboard_sync as cds

SHA_NEW = "c" * 64
SHA_OLD = "a" * 64


def _manifest(sha=SHA_NEW, minutes_old=600):
    created = (datetime.now(timezone.utc) - timedelta(minutes=minutes_old)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    return {"db_sha256": sha, "latest_hs_run_id": "hs_20261001T045127", "created_utc": created}


def _version(key, error=None):
    return {"db_sha256": key, "sync_status": {"downloaded_key": key, "last_error": error}}


def _args(mode="watchdog", grace=90):
    return cds.argparse.Namespace(
        mode=mode, api_base="https://api.example", repo="o/r", tag="t",
        wait_min=0, poll_sec=0, grace_min=grace,
    )


def _patch(monkeypatch, manifest, versions):
    seq = list(versions)
    monkeypatch.setattr(cds, "release_manifest", lambda repo, tag: manifest)

    def fake_version(base):
        item = seq.pop(0) if len(seq) > 1 else seq[0]
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(cds, "api_version", fake_version)


def test_in_sync_passes(monkeypatch):
    _patch(monkeypatch, _manifest(), [_version(SHA_NEW)])
    assert cds.run(_args()) == 0


def test_the_october_shape_fails_the_watchdog_and_names_the_error(monkeypatch, capsys):
    _patch(monkeypatch, _manifest(), [_version(SHA_OLD, error="Insufficient disk space")])
    assert cds.run(_args()) == 1
    out = capsys.readouterr().out
    assert "::error::" in out and "Insufficient disk space" in out
    assert "hs_20261001T045127" in out


def test_publish_mode_never_fails_the_run(monkeypatch, capsys):
    _patch(monkeypatch, _manifest(), [_version(SHA_OLD)])
    assert cds.run(_args(mode="publish")) == 0
    assert "::warning::" in capsys.readouterr().out


def test_a_fresh_release_is_inside_the_grace_window(monkeypatch):
    _patch(monkeypatch, _manifest(minutes_old=10), [_version(SHA_OLD)])
    assert cds.run(_args()) == 0


def test_an_unreachable_api_fails_the_watchdog(monkeypatch):
    _patch(monkeypatch, _manifest(), [OSError("connection refused")])
    assert cds.run(_args()) == 1


def test_it_polls_until_the_api_catches_up(monkeypatch):
    _patch(monkeypatch, _manifest(), [_version(SHA_OLD), _version(SHA_OLD), _version(SHA_NEW)])
    clock = iter(range(0, 1000, 10))
    args = _args()
    args.wait_min = 5
    assert cds.run(args, sleep=lambda s: None, now=lambda: next(clock)) == 0


def test_served_key_prefers_what_was_downloaded():
    assert cds.served_key({"db_sha256": "m", "sync_status": {"downloaded_key": "d"}}) == "d"
    assert cds.served_key({"db_sha256": "m"}) == "m"
    assert cds.served_key({}) is None
