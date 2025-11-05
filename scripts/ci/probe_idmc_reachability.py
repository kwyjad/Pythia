"""Connectivity probe for the IDMC PostgREST endpoint."""
from __future__ import annotations

import json
import os
import socket
import ssl
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple
from urllib.error import URLError
from urllib.request import Request, urlopen

import certifi

try:  # pragma: no cover - optional dependency
    import requests
except Exception:  # pragma: no cover - requests may not be available
    requests = None  # type: ignore[assignment]

REQUESTS_VERSION = getattr(requests, "__version__", None) if requests else None

DIAG_DIR = Path("diagnostics/ingestion/idmc")
JSON_PATH = DIAG_DIR / "probe.json"
MARKDOWN_PATH = DIAG_DIR / "probe.md"
TARGET_HOST = os.getenv("IDMC_PROBE_HOST", "backend.idmcdb.org").strip() or "backend.idmcdb.org"
TARGET_PORT = int(os.getenv("IDMC_PROBE_PORT", "443"))
TARGET_PATH = os.getenv(
    "IDMC_PROBE_PATH", "data/idus_view_flat?select=id&limit=1"
).lstrip("/")
TARGET_SCHEME = os.getenv("IDMC_PROBE_SCHEME", "https").strip() or "https"
TARGET_URL = os.getenv(
    "IDMC_PROBE_URL",
    f"{TARGET_SCHEME}://{TARGET_HOST}:{TARGET_PORT}/{TARGET_PATH}",
)
CONNECT_TIMEOUT = float(os.getenv("IDMC_PROBE_CONNECT_TIMEOUT", "5"))
READ_TIMEOUT = float(os.getenv("IDMC_PROBE_READ_TIMEOUT", "10"))
EGRESS_TIMEOUT = float(os.getenv("IDMC_PROBE_EGRESS_TIMEOUT", "5"))
USER_AGENT = os.getenv("RELIEFWEB_USER_AGENT") or os.getenv("IDMC_USER_AGENT") or os.getenv("RELIEFWEB_APPNAME") or "PythiaResolver/1.0"
VERIFY_SETTING_RAW = os.getenv("IDMC_HTTP_VERIFY", "true").strip()


def _resolve_verify_setting(raw: str) -> bool | str:
    lowered = raw.lower()
    if lowered in {"", "default", "system"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    if lowered in {"1", "true", "yes", "on"}:
        return True
    return raw


VERIFY_SETTING = _resolve_verify_setting(VERIFY_SETTING_RAW)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _elapsed_ms(started: float) -> int:
    return int(round((time.monotonic() - started) * 1000))


def _truncate(value: str, limit: int = 512) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def probe_dns(host: str) -> Dict[str, Any]:
    started = time.monotonic()
    payload: Dict[str, Any] = {"records": []}
    try:
        results = socket.getaddrinfo(host, None)
        seen: set[Tuple[str, str]] = set()
        for family, _, _, _, sockaddr in results:
            address = sockaddr[0] if isinstance(sockaddr, tuple) and sockaddr else None
            if not address:
                continue
            if isinstance(family, int):
                try:
                    family_name = socket.AddressFamily(family).name
                except Exception:  # pragma: no cover - platform dependent
                    family_name = str(family)
            else:
                family_name = str(family)
            key = (address, family_name)
            if key in seen:
                continue
            seen.add(key)
            payload.setdefault("records", []).append({"address": address, "family": family_name})
    except Exception as exc:  # pragma: no cover - network dependent
        payload["error"] = str(exc)
    payload["elapsed_ms"] = _elapsed_ms(started)
    return payload


def probe_tcp(host: str, port: int, timeout: float) -> Dict[str, Any]:
    started = time.monotonic()
    payload: Dict[str, Any] = {"ok": False}
    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            payload["ok"] = True
            try:
                payload["peer"] = list(sock.getpeername())
            except Exception:  # pragma: no cover - platform dependent
                payload["peer"] = None
    except Exception as exc:  # pragma: no cover - network dependent
        payload["error"] = str(exc)
    payload["elapsed_ms"] = _elapsed_ms(started)
    return payload


def probe_tls(host: str, port: int, timeout: float) -> Dict[str, Any]:
    started = time.monotonic()
    payload: Dict[str, Any] = {"ok": False, "server_name": host}
    context = ssl.create_default_context()
    cafile = ssl.get_default_verify_paths().cafile or certifi.where()
    payload["ca_bundle"] = cafile
    try:
        if cafile:
            context.load_verify_locations(cafile=cafile)
    except Exception:  # pragma: no cover - defensive
        pass
    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            with context.wrap_socket(sock, server_hostname=host) as wrapped:
                payload["ok"] = True
                payload["version"] = wrapped.version()
                cipher = wrapped.cipher()
                payload["cipher"] = cipher[0] if cipher else None
                cert = wrapped.getpeercert()
                if cert:
                    payload["subject"] = cert.get("subject")
                    payload["issuer"] = cert.get("issuer")
                    payload["not_before"] = cert.get("notBefore")
                    payload["not_after"] = cert.get("notAfter")
    except Exception as exc:  # pragma: no cover - network dependent
        payload["error"] = str(exc)
    payload["elapsed_ms"] = _elapsed_ms(started)
    return payload


def probe_http(
    url: str, *, timeout: Tuple[float, float], verify: bool | str
) -> Dict[str, Any]:
    started = time.monotonic()
    payload: Dict[str, Any] = {
        "url": url,
        "verify": verify,
        "headers": {"User-Agent": USER_AGENT, "Accept": "application/json"},
    }
    if requests is None:
        payload["error"] = "requests-not-installed"
        payload["elapsed_ms"] = _elapsed_ms(started)
        return payload
    try:
        response = requests.get(
            url,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            timeout=timeout,
            verify=verify,
        )
        payload["status_code"] = response.status_code
        payload["reason"] = response.reason
        payload["elapsed_ms"] = _elapsed_ms(started)
        payload["content_type"] = response.headers.get("Content-Type")
        preview = response.text[:256]
        if preview:
            payload["preview"] = _truncate(preview)
    except requests.RequestException as exc:  # pragma: no cover - network dependent
        payload["elapsed_ms"] = _elapsed_ms(started)
        payload["error"] = str(exc)
        payload["exception"] = exc.__class__.__name__
        reason = getattr(exc, "__cause__", None) or getattr(exc, "args", [None])[0]
        if reason is not None:
            try:
                payload["reason"] = str(reason)
            except Exception:  # pragma: no cover - defensive
                payload["reason"] = repr(reason)
    except Exception as exc:  # pragma: no cover - defensive
        payload["elapsed_ms"] = _elapsed_ms(started)
        payload["error"] = str(exc)
        payload["exception"] = exc.__class__.__name__
    return payload


def probe_egress(timeout: float) -> Dict[str, Any]:
    targets = {
        "ifconfig.me": "https://ifconfig.me/ip",
        "ipify": "https://api.ipify.org?format=text",
    }
    results: Dict[str, Any] = {}
    for label, url in targets.items():
        started = time.monotonic()
        entry: Dict[str, Any] = {"url": url}
        try:
            req = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(req, timeout=timeout) as response:
                entry["status_code"] = response.status
                data = response.read(256).decode("utf-8", errors="replace").strip()
                if data:
                    entry["text"] = data
        except URLError as exc:  # pragma: no cover - network dependent
            entry["error"] = str(exc.reason)
        except Exception as exc:  # pragma: no cover - defensive
            entry["error"] = str(exc)
        entry["elapsed_ms"] = _elapsed_ms(started)
        results[label] = entry
    return results


def build_markdown(payload: Mapping[str, Any]) -> str:
    dns = payload.get("dns", {})
    tcp = payload.get("tcp", {})
    tls = payload.get("tls", {})
    http = payload.get("http", {})
    egress = payload.get("egress", {})
    lines: List[str] = ["## IDMC Reachability", ""]
    target = payload.get("target", {})
    host = target.get("host", TARGET_HOST)
    port = target.get("port", TARGET_PORT)
    lines.append(f"- **Target:** `{host}:{port}`")
    lines.append(f"- **Captured:** {payload.get('generated_at', 'unknown')}")
    dns_records = [
        f"{record.get('address')} ({record.get('family')})"
        for record in dns.get("records", [])
        if isinstance(record, Mapping)
    ]
    if dns.get("error"):
        lines.append(f"- **DNS:** error={dns.get('error')}")
    else:
        joined = ", ".join(dns_records) if dns_records else "—"
        lines.append(f"- **DNS:** {joined}")
    if dns.get("elapsed_ms") is not None:
        lines[-1] += f" ({dns.get('elapsed_ms')}ms)"
    if tcp:
        if tcp.get("ok"):
            peer = tcp.get("peer")
            peer_display = ":".join(str(part) for part in peer) if isinstance(peer, Iterable) else peer
            tcp_line = "- **TCP:** ok"
            if tcp.get("elapsed_ms") is not None:
                tcp_line += f" in {tcp.get('elapsed_ms')}ms"
            if peer_display:
                tcp_line += f" (peer={peer_display})"
        else:
            tcp_line = f"- **TCP:** error={tcp.get('error', 'unknown')}"
            if tcp.get("elapsed_ms") is not None:
                tcp_line += f" after {tcp.get('elapsed_ms')}ms"
        lines.append(tcp_line)
    if tls:
        if tls.get("ok"):
            tls_line = "- **TLS:** ok"
            if tls.get("version"):
                tls_line += f" version={tls.get('version')}"
            if tls.get("cipher"):
                tls_line += f" cipher={tls.get('cipher')}"
            if tls.get("elapsed_ms") is not None:
                tls_line += f" ({tls.get('elapsed_ms')}ms)"
        else:
            tls_line = f"- **TLS:** error={tls.get('error', 'unknown')}"
            if tls.get("elapsed_ms") is not None:
                tls_line += f" after {tls.get('elapsed_ms')}ms"
        lines.append(tls_line)
    if http:
        if http.get("status_code") is not None:
            line = f"- **HTTP GET:** status={http.get('status_code')}"
            if http.get("elapsed_ms") is not None:
                line += f" ({http.get('elapsed_ms')}ms)"
        else:
            error = http.get("exception") or http.get("error") or "unknown"
            line = f"- **HTTP GET:** error={error}"
            if http.get("elapsed_ms") is not None:
                line += f" after {http.get('elapsed_ms')}ms"
        lines.append(line)
    egress_ip = None
    for label in ("ifconfig.me", "ipify"):
        entry = egress.get(label)
        if isinstance(entry, Mapping) and entry.get("text"):
            egress_ip = entry.get("text")
            break
    if egress_ip:
        lines.append(f"- **Egress IP:** {egress_ip}")
    ca_bundle = payload.get("ca_bundle") or payload.get("tls", {}).get("ca_bundle")
    if ca_bundle:
        lines.append(f"- **CA bundle:** `{ca_bundle}`")
    verify_flag = http.get("verify") if isinstance(http, Mapping) else None
    if verify_flag is not None:
        lines.append(f"- **Verify TLS:** {verify_flag}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    started_iso = _now_iso()
    dns_info = probe_dns(TARGET_HOST)
    tcp_info = probe_tcp(TARGET_HOST, TARGET_PORT, CONNECT_TIMEOUT)
    tls_info = probe_tls(TARGET_HOST, TARGET_PORT, CONNECT_TIMEOUT)
    http_info = probe_http(TARGET_URL, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT), verify=VERIFY_SETTING)
    egress_info = probe_egress(EGRESS_TIMEOUT)
    payload: Dict[str, Any] = {
        "target": {"host": TARGET_HOST, "port": TARGET_PORT, "url": TARGET_URL},
        "generated_at": started_iso,
        "completed_at": _now_iso(),
        "dns": dns_info,
        "tcp": tcp_info,
        "tls": tls_info,
        "http": http_info,
        "egress": egress_info,
        "ca_bundle": ssl.get_default_verify_paths().cafile or certifi.where(),
        "python_version": sys.version,
        "requests_available": requests is not None,
        "requests_version": REQUESTS_VERSION,
        "env": {
            "HTTPS_PROXY": os.getenv("HTTPS_PROXY"),
            "HTTP_PROXY": os.getenv("HTTP_PROXY"),
            "IDMC_HTTP_VERIFY": os.getenv("IDMC_HTTP_VERIFY"),
        },
    }
    try:
        DIAG_DIR.mkdir(parents=True, exist_ok=True)
        JSON_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        markdown = build_markdown(payload)
        MARKDOWN_PATH.write_text(markdown, encoding="utf-8")
        print(markdown)
    except Exception:  # pragma: no cover - diagnostics must not break CI
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
