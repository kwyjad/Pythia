# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Reachability probe utilities shared by the IDMC connector and CI."""
from __future__ import annotations

import socket
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Mapping, cast
from urllib.parse import urlparse

__all__ = ["ProbeOptions", "probe_reachability", "summarize_probe_outcome"]


@dataclass
class ProbeOptions:
    """Settings for probing an IDMC endpoint."""

    base_url: str = "https://backend.idmcdb.org"
    timeout: float = 3.0


def _elapsed_ms(started: float) -> int:
    return int(round((time.monotonic() - started) * 1000))


def probe_dns(host: str) -> Dict[str, Any]:
    started = time.monotonic()
    payload: Dict[str, Any] = {"records": []}
    try:
        results = socket.getaddrinfo(host, None)
        seen: set[tuple[str, str]] = set()
        for family, _, _, _, sockaddr in results:
            address = sockaddr[0] if isinstance(sockaddr, tuple) and sockaddr else None
            if not address:
                continue
            try:
                family_name = socket.AddressFamily(family).name  # type: ignore[arg-type]
            except Exception:  # pragma: no cover - platform dependent
                family_name = str(family)
            key = (address, family_name)
            if key in seen:
                continue
            seen.add(key)
            payload["records"].append({"address": address, "family": family_name})
        payload["ok"] = True
    except Exception as exc:  # pragma: no cover - network dependent
        payload["ok"] = False
        payload["error"] = str(exc)
    payload["elapsed_ms"] = _elapsed_ms(started)
    return payload


def probe_tcp(host: str, port: int, timeout: float) -> Dict[str, Any]:
    started = time.monotonic()
    payload: Dict[str, Any] = {"ok": False}
    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            payload["ok"] = True
            payload["peer"] = list(sock.getpeername())
            payload["egress_ip"] = sock.getsockname()[0]
    except Exception as exc:  # pragma: no cover - network dependent
        payload["error"] = str(exc)
    payload["elapsed_ms"] = _elapsed_ms(started)
    return payload


def probe_tls(host: str, port: int, timeout: float) -> Dict[str, Any]:
    started = time.monotonic()
    payload: Dict[str, Any] = {"ok": False}
    context = ssl.create_default_context()
    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            with context.wrap_socket(sock, server_hostname=host) as wrapped:
                payload["ok"] = True
                payload["protocol"] = wrapped.version()
                cipher = wrapped.cipher()
                if cipher:
                    payload["cipher"] = cipher[0]
    except Exception as exc:  # pragma: no cover - network dependent
        payload["error"] = str(exc)
    payload["elapsed_ms"] = _elapsed_ms(started)
    return payload


def probe_http_head(url: str, timeout: float) -> Dict[str, Any]:
    started = time.monotonic()
    payload: Dict[str, Any] = {"ok": False}
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload["ok"] = True
            payload["status"] = getattr(response, "status", None) or response.getcode()
            payload["headers"] = dict(response.headers.items())
    except urllib.error.URLError as exc:  # pragma: no cover - network dependent
        reason = getattr(exc, "reason", None)
        if reason is not None:
            try:
                payload["error"] = str(reason)
            except Exception:  # pragma: no cover - defensive
                payload["error"] = repr(reason)
        else:
            payload["error"] = str(exc)
    except Exception as exc:  # pragma: no cover - defensive
        payload["error"] = str(exc)
    payload["elapsed_ms"] = _elapsed_ms(started)
    return payload


def _skipped_payload() -> Dict[str, Any]:
    return {"ok": False, "skipped": True, "elapsed_ms": 0}


def probe_reachability(options: ProbeOptions | None = None) -> Dict[str, Any]:
    """Collect reachability diagnostics for an IDMC base URL."""

    opts = options or ProbeOptions()
    parsed = urlparse(opts.base_url)
    host = parsed.hostname or opts.base_url
    port = parsed.port or (443 if parsed.scheme == "https" else 80)

    sequence: list[Dict[str, Any]] = []

    dns = probe_dns(host)
    sequence.append({"step": "dns", "ok": bool(dns.get("ok")), "elapsed_ms": dns.get("elapsed_ms")})
    if not dns.get("ok"):
        return {
            "base_url": opts.base_url,
            "host": host,
            "dns": dns,
            "tcp": _skipped_payload(),
            "tls": _skipped_payload(),
            "http_head": _skipped_payload(),
            "sequence": sequence,
            "egress_ip": None,
        }

    tcp = probe_tcp(host, port, opts.timeout)
    sequence.append({"step": "tcp", "ok": bool(tcp.get("ok")), "elapsed_ms": tcp.get("elapsed_ms")})
    if not tcp.get("ok"):
        return {
            "base_url": opts.base_url,
            "host": host,
            "dns": dns,
            "tcp": tcp,
            "tls": _skipped_payload(),
            "http_head": _skipped_payload(),
            "sequence": sequence,
            "egress_ip": tcp.get("egress_ip"),
        }

    tls = probe_tls(host, port, opts.timeout)
    sequence.append({"step": "tls", "ok": bool(tls.get("ok")), "elapsed_ms": tls.get("elapsed_ms")})
    if not tls.get("ok"):
        return {
            "base_url": opts.base_url,
            "host": host,
            "dns": dns,
            "tcp": tcp,
            "tls": tls,
            "http_head": _skipped_payload(),
            "sequence": sequence,
            "egress_ip": tcp.get("egress_ip"),
        }

    http_head = probe_http_head(opts.base_url.rstrip("/") + "/", opts.timeout)
    sequence.append({"step": "http_head", "ok": bool(http_head.get("ok")), "elapsed_ms": http_head.get("elapsed_ms")})

    return {
        "base_url": opts.base_url,
        "host": host,
        "dns": dns,
        "tcp": tcp,
        "tls": tls,
        "http_head": http_head,
        "sequence": sequence,
        "egress_ip": tcp.get("egress_ip"),
    }


def summarize_probe_outcome(result: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a compact summary describing the probe outcome."""

    summary: Dict[str, Any] = {
        "base_url": str(result.get("base_url") or ""),
        "status": "unknown",
    }

    dns = cast(Dict[str, Any], result.get("dns") or {})
    if not dns.get("ok"):
        summary.update({
            "status": "fail",
            "stage": "dns",
            "reason": dns.get("error"),
        })
        return summary

    tcp = cast(Dict[str, Any], result.get("tcp") or {})
    if not tcp.get("ok"):
        summary.update({
            "status": "fail",
            "stage": "tcp",
            "reason": tcp.get("error"),
        })
        return summary

    tls = cast(Dict[str, Any], result.get("tls") or {})
    if not tls.get("ok"):
        summary.update({
            "status": "fail",
            "stage": "tls",
            "reason": tls.get("error"),
        })
        return summary

    http_head = cast(Dict[str, Any], result.get("http_head") or {})
    if http_head.get("ok"):
        summary.update({
            "status": "ok",
            "stage": "http",
            "status_code": http_head.get("status"),
        })
        return summary

    summary.update({
        "status": "warn",
        "stage": "http",
        "status_code": http_head.get("status"),
        "reason": http_head.get("error"),
    })
    return summary
