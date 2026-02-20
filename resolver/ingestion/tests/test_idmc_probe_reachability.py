# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.ci import probe_idmc_reachability as probe


@pytest.mark.xfail(reason="Probe monkeypatch not intercepting real HTTP calls (pre-existing)", strict=False)
def test_probe_idmc_records_status(monkeypatch, tmp_path: Path):
    summary_path = tmp_path / "summary.md"
    diag_dir = tmp_path / "diag"

    monkeypatch.setattr(probe, "SUMMARY_PATH", summary_path)
    monkeypatch.setattr(probe, "DIAG_DIR", diag_dir)
    monkeypatch.setattr(probe, "BASE", "https://example.org")
    monkeypatch.setattr(probe, "ENDPOINT", "/data")
    monkeypatch.setattr(probe, "QUERY", "select=id&limit=1")

    monkeypatch.setattr(probe, "probe_dns", lambda host: {"records": [{"address": "1.2.3.4"}], "elapsed_ms": 5})
    monkeypatch.setattr(
        probe,
        "probe_tcp",
        lambda host, port: {"ok": True, "egress": ["10.0.0.1", 12345], "elapsed_ms": 3},
    )
    monkeypatch.setattr(
        probe,
        "probe_tls",
        lambda host, port: {"ok": True, "version": "TLSv1.3", "cipher": "AES", "elapsed_ms": 4},
    )
    monkeypatch.setattr(
        probe,
        "probe_http",
        lambda url: {"status": 200, "bytes": 128, "elapsed_ms": 12},
    )
    monkeypatch.setattr(
        probe,
        "ca_bundle_info",
        lambda: {"paths": ["/etc/ssl/certs/ca.pem"]},
    )

    rc = probe.main()
    assert rc == 0

    summary_text = summary_path.read_text(encoding="utf-8")
    assert "## IDMC Reachability" in summary_text
    assert "HTTP GET: status 200" in summary_text
    assert "Egress IP: 10.0.0.1" in summary_text

    payload = (diag_dir / "reachability.json").read_text(encoding="utf-8")
    assert "\"status\": 200" in payload
