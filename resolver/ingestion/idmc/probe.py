"""Reachability probe utilities shared by the IDMC connector and CI."""
from __future__ import annotations

import socket
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict
from urllib.parse import urlparse

__all__ = ["ProbeOptions", "probe_reachability"]


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


def probe_reachability(options: ProbeOptions | None = None) -> Dict[str, Any]:
    """Collect reachability diagnostics for an IDMC base URL."""

    opts = options or ProbeOptions()
    parsed = urlparse(opts.base_url)
    host = parsed.hostname or opts.base_url
    port = parsed.port or (443 if parsed.scheme == "https" else 80)

    dns = probe_dns(host)
    tcp = probe_tcp(host, port, opts.timeout)
    tls = probe_tls(host, port, opts.timeout)
    http_head = probe_http_head(opts.base_url.rstrip("/") + "/", opts.timeout)

    return {
        "base_url": opts.base_url,
        "host": host,
        "dns": dns,
        "tcp": tcp,
        "tls": tls,
        "http_head": http_head,
        "egress_ip": tcp.get("egress_ip"),
    }
