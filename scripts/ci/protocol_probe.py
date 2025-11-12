"""Shared helpers for connector network reachability probes."""
from __future__ import annotations

import json
import os
import socket
import ssl
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Mapping, Optional

import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import RequestException

__all__ = [
    "ProbeResult",
    "probe_host",
    "format_markdown_block",
    "summarize_graphql_probe",
]


@dataclass
class ProbeResult:
    """Structured result for a protocol probe."""

    host: str
    port: int
    scheme: str
    path: str
    timestamp: str
    dns: Dict[str, Any]
    tcp: Dict[str, Any]
    tls: Dict[str, Any]
    http: Dict[str, Any]
    egress: Dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, sort_keys=True)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "scheme": self.scheme,
            "path": self.path,
            "timestamp": self.timestamp,
            "dns": self.dns,
            "tcp": self.tcp,
            "tls": self.tls,
            "http": self.http,
            "egress": self.egress,
        }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _user_agent() -> str:
    for name in ("IDMC_USER_AGENT", "RELIEFWEB_USER_AGENT", "RELIEFWEB_APPNAME"):
        value = os.getenv(name, "").strip()
        if value:
            return value
    return "Pythia-Resolver/IDMC-Probe"


def _elapsed_ms(started: float) -> int:
    return int(round((time.monotonic() - started) * 1000))


def _requests_session() -> requests.Session:
    session = requests.Session()
    adapter = HTTPAdapter(pool_connections=4, pool_maxsize=16)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _probe_dns(host: str) -> Dict[str, Any]:
    started = time.monotonic()
    payload: Dict[str, Any] = {"records": []}
    try:
        records = socket.getaddrinfo(host, None)
        seen: set[tuple[str, str]] = set()
        for family, _, _, _, sockaddr in records:
            address = sockaddr[0] if isinstance(sockaddr, tuple) and sockaddr else None
            if not address:
                continue
            family_name = (
                socket.AddressFamily(family).name
                if isinstance(family, int)
                else str(family)
            )
            key = (address, family_name)
            if key in seen:
                continue
            seen.add(key)
            payload["records"].append({"address": address, "family": family_name})
        payload["ok"] = bool(payload["records"])
    except Exception as exc:  # pragma: no cover - network dependent
        payload["error"] = str(exc)
    payload["elapsed_ms"] = _elapsed_ms(started)
    return payload


def _probe_tcp(host: str, port: int, timeout: float) -> Dict[str, Any]:
    started = time.monotonic()
    payload: Dict[str, Any] = {"ok": False}
    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            payload["ok"] = True
            try:
                peer = sock.getpeername()
                payload["peer"] = list(peer) if isinstance(peer, tuple) else peer
            except Exception:  # pragma: no cover - platform dependent
                payload["peer"] = None
    except Exception as exc:  # pragma: no cover - network dependent
        payload["error"] = str(exc)
    payload["elapsed_ms"] = _elapsed_ms(started)
    return payload


def _probe_tls(host: str, port: int, timeout: float) -> Dict[str, Any]:
    started = time.monotonic()
    payload: Dict[str, Any] = {"ok": False, "server_name": host}
    context = ssl.create_default_context()
    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            with context.wrap_socket(sock, server_hostname=host) as wrapped:
                payload["ok"] = True
                payload["version"] = wrapped.version()
                cipher = wrapped.cipher()
                if cipher:
                    payload["cipher"] = cipher[0]
                cert = wrapped.getpeercert()
                if cert:
                    payload["issuer"] = cert.get("issuer")
                    payload["subject"] = cert.get("subject")
                    payload["not_before"] = cert.get("notBefore")
                    payload["not_after"] = cert.get("notAfter")
    except Exception as exc:  # pragma: no cover - network dependent
        payload["error"] = str(exc)
    payload["elapsed_ms"] = _elapsed_ms(started)
    return payload


def _probe_http(
    url: str,
    *,
    session: requests.Session,
    timeout: tuple[float, float],
    verify: bool | str,
) -> Dict[str, Any]:
    started = time.monotonic()
    headers = {"User-Agent": _user_agent(), "Accept": "application/json"}
    payload: Dict[str, Any] = {"url": url, "verify": verify, "headers": headers}
    try:
        response = session.get(url, headers=headers, timeout=timeout, verify=verify)
        payload["status_code"] = response.status_code
        payload["reason"] = response.reason
        payload["elapsed_ms"] = _elapsed_ms(started)
        payload["content_type"] = response.headers.get("Content-Type")
        preview = response.text[:200]
        if preview:
            payload["preview"] = preview
    except RequestException as exc:  # pragma: no cover - network dependent
        payload["elapsed_ms"] = _elapsed_ms(started)
        payload["error"] = str(exc)
        payload["exception"] = exc.__class__.__name__
    except Exception as exc:  # pragma: no cover - defensive
        payload["elapsed_ms"] = _elapsed_ms(started)
        payload["error"] = str(exc)
        payload["exception"] = exc.__class__.__name__
    return payload


def _probe_egress(session: requests.Session, timeout: float) -> Dict[str, Any]:
    targets = {
        "ifconfig.me": "https://ifconfig.me/ip",
        "ipify": "https://api.ipify.org?format=text",
    }
    headers = {"User-Agent": _user_agent(), "Accept": "text/plain"}
    results: Dict[str, Any] = {}
    for label, url in targets.items():
        started = time.monotonic()
        entry: Dict[str, Any] = {"url": url}
        try:
            response = session.get(url, headers=headers, timeout=timeout)
            entry["status_code"] = response.status_code
            entry["elapsed_ms"] = _elapsed_ms(started)
            if response.ok:
                entry["ip"] = response.text.strip()[:128]
        except RequestException as exc:  # pragma: no cover - network dependent
            entry["elapsed_ms"] = _elapsed_ms(started)
            entry["error"] = str(exc)
            entry["exception"] = exc.__class__.__name__
        results[label] = entry
    return results


def probe_host(
    host: str,
    *,
    port: int = 443,
    scheme: str = "https",
    https_path: str = "/",
    connect_timeout: float = 5.0,
    read_timeout: float = 5.0,
    verify: bool | str | None = None,
) -> ProbeResult:
    """Probe DNS/TCP/TLS/HTTP for ``host`` and return a structured result."""

    path = https_path if https_path.startswith("/") else f"/{https_path}"
    base_url = f"{scheme}://{host}:{port}"
    url = f"{base_url}{path}"
    session = _requests_session()
    verify_setting = True if verify is None else verify
    dns = _probe_dns(host)
    tcp = _probe_tcp(host, port, connect_timeout)
    tls = _probe_tls(host, port, connect_timeout)
    http = _probe_http(
        url,
        session=session,
        timeout=(connect_timeout, read_timeout),
        verify=verify_setting,
    )
    egress = _probe_egress(session, read_timeout)
    return ProbeResult(
        host=host,
        port=port,
        scheme=scheme,
        path=path,
        timestamp=_now_iso(),
        dns=dns,
        tcp=tcp,
        tls=tls,
        http=http,
        egress=egress,
    )


def _status_line(entry: Dict[str, Any]) -> str:
    if entry.get("error"):
        return f"error: {entry['error']}"
    if entry.get("ok") is False:
        return "error"
    if "status_code" in entry:
        code = entry["status_code"]
        reason = entry.get("reason", "")
        return f"{code} {reason}".strip()
    return "ok"


def _shorten(value: Any, limit: int = 80) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def format_markdown_block(title: str, result: ProbeResult) -> str:
    """Render a Markdown block for the probe result."""

    lines = [f"## {title}", "", "| Probe | Status | Details |", "| --- | --- | --- |"]
    probes: Iterable[tuple[str, Dict[str, Any]]] = (
        ("DNS", result.dns),
        ("TCP", result.tcp),
        ("TLS", result.tls),
        ("HTTP", result.http),
    )
    for name, payload in probes:
        status = _status_line(payload)
        detail_parts = []
        if name == "DNS" and payload.get("records"):
            detail_parts.append(
                ", ".join(
                    f"{record.get('address')} ({record.get('family')})"
                    for record in payload.get("records", [])[:4]
                )
            )
        elif name == "HTTP" and payload.get("status_code"):
            detail_parts.append(payload.get("content_type", ""))
        elif payload.get("error"):
            detail_parts.append(payload["error"])
        details = _shorten("; ".join(part for part in detail_parts if part) or "")
        lines.append(f"| {name} | {status} | {details} |")
    egress_details = []
    for label, entry in result.egress.items():
        status = _status_line(entry)
        if entry.get("ip"):
            egress_details.append(f"{label}: {status} → {entry['ip']}")
        else:
            egress_details.append(f"{label}: {status}")
    if egress_details:
        lines.append(f"| Egress | ok | {_shorten('; '.join(egress_details))} |")
    lines.append("")
    lines.append("```json")
    lines.append(result.to_json())
    lines.append("```")
    return "\n".join(lines)


def summarize_graphql_probe(result: Mapping[str, Any]) -> list[str]:
    """Return bullet lines summarizing a GraphQL metadata probe outcome."""

    lines: list[str] = []
    if not isinstance(result, Mapping):
        return ["- probe.json: unexpected payload type"]

    ok = bool(result.get("ok"))
    status_text = "ok" if ok else "fail"
    details: list[str] = []
    http_status = result.get("http_status")
    if isinstance(http_status, int):
        details.append(f"HTTP {http_status}")
    elif http_status:
        details.append(f"http={http_status}")
    elapsed_ms = result.get("elapsed_ms")
    if isinstance(elapsed_ms, (int, float)):
        details.append(f"{int(round(elapsed_ms))} ms")
    elif elapsed_ms is not None:
        details.append(f"elapsed={elapsed_ms}")

    line = f"- status: {status_text}"
    if details:
        line += f" ({', '.join(details)})"
    lines.append(line)

    if result.get("skipped"):
        lines.append("- note: probe skipped (offline mode)")

    error = result.get("error")
    if error and not result.get("skipped"):
        lines.append(f"- error: {error}")

    api_version = result.get("api_version")
    if api_version:
        lines.append(f"- api_version: {api_version}")

    info = result.get("info")
    if isinstance(info, Mapping):
        version = info.get("version")
        if version:
            lines.append(f"- dataset version: {version}")
        timestamp = info.get("timestamp")
        if timestamp:
            lines.append(f"- metadata timestamp: {timestamp}")

    recorded_at = result.get("recorded_at")
    if recorded_at:
        lines.append(f"- recorded_at: {recorded_at}")

    total_available = result.get("total_available")
    if total_available is not None:
        lines.append(f"- total_available: {total_available}")

    return lines
