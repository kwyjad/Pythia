"""Reachability probe tests that avoid hitting the network."""
from __future__ import annotations

from resolver.ingestion.idmc import probe as probe_mod


def test_probe_shape(monkeypatch):
    monkeypatch.setattr(
        probe_mod,
        "probe_dns",
        lambda host: {"ok": True, "records": [{"address": "192.0.2.1", "family": "AF_INET"}]},
    )
    monkeypatch.setattr(
        probe_mod,
        "probe_tcp",
        lambda host, port, timeout: {
            "ok": True,
            "peer": [host, port],
            "egress_ip": "203.0.113.10",
            "elapsed_ms": 12,
        },
    )
    monkeypatch.setattr(
        probe_mod,
        "probe_tls",
        lambda host, port, timeout: {"ok": True, "protocol": "TLSv1.3", "cipher": "TLS_AES_256_GCM_SHA384"},
    )
    monkeypatch.setattr(
        probe_mod,
        "probe_http_head",
        lambda url, timeout: {"ok": True, "status": 200, "headers": {"content-type": "application/json"}},
    )

    result = probe_mod.probe_reachability(probe_mod.ProbeOptions(base_url="https://example.test", timeout=1.0))

    assert set(result.keys()) >= {"base_url", "host", "dns", "tcp", "tls", "http_head", "egress_ip"}
    assert result["dns"]["records"][0]["address"] == "192.0.2.1"
    assert result["tcp"]["egress_ip"] == "203.0.113.10"
    assert result["tls"]["protocol"] == "TLSv1.3"
    assert result["http_head"]["status"] == 200
