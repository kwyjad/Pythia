"""Reachability diagnostics for the IDMC IDU endpoint."""
from __future__ import annotations

import json
import os
import socket
import ssl
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

try:  # pragma: no cover - optional dependency
    import certifi  # type: ignore
except Exception:  # pragma: no cover - defensive
    certifi = None  # type: ignore

BASE = os.getenv("IDMC_BASE_URL", "https://backend.idmcdb.org").rstrip("/")
ENDPOINT = os.getenv("IDMC_PROBE_ENDPOINT", "/data/idus_view_flat")
QUERY = os.getenv("IDMC_PROBE_QUERY", "select=id&limit=1")
TIMEOUT = float(os.getenv("IDMC_PROBE_TIMEOUT", "5"))
DIAG_DIR = Path("diagnostics/ingestion/idmc")
SUMMARY_PATH = Path("diagnostics/ingestion/summary.md")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _elapsed(start: float) -> int:
    return int(round((time.monotonic() - start) * 1000))


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
            else:  # pragma: no cover - defensive fallback
                family_name = str(family)
            key = (address, family_name)
            if key in seen:
                continue
            seen.add(key)
            payload["records"].append({"address": address, "family": family_name})
    except Exception as exc:  # pragma: no cover - network dependent
        payload["error"] = str(exc)
    payload["elapsed_ms"] = _elapsed(started)
    return payload


def probe_tcp(host: str, port: int) -> Dict[str, Any]:
    started = time.monotonic()
    payload: Dict[str, Any] = {"ok": False}
    try:
        with socket.create_connection((host, port), timeout=TIMEOUT) as sock:
            payload["ok"] = True
            payload["peer"] = list(sock.getpeername())
            payload["egress"] = list(sock.getsockname())
    except Exception as exc:  # pragma: no cover - network dependent
        payload["error"] = str(exc)
    payload["elapsed_ms"] = _elapsed(started)
    return payload


def probe_tls(host: str, port: int) -> Dict[str, Any]:
    started = time.monotonic()
    payload: Dict[str, Any] = {"ok": False}
    context = ssl.create_default_context()
    try:
        with socket.create_connection((host, port), timeout=TIMEOUT) as sock:
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
    payload["elapsed_ms"] = _elapsed(started)
    return payload


def probe_http(url: str) -> Dict[str, Any]:
    started = time.monotonic()
    payload: Dict[str, Any] = {"status": None, "bytes": 0}
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "pythia-idmc-probe/1.0",
            "Accept": "application/json",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            body = response.read(2048)
            payload["status"] = getattr(response, "status", response.getcode())
            payload["bytes"] = len(body)
    except Exception as exc:  # pragma: no cover - network dependent
        payload["error"] = str(exc)
    payload["elapsed_ms"] = _elapsed(started)
    return payload


def ca_bundle_info() -> Dict[str, Any]:
    paths: List[str] = []
    default_paths = ssl.get_default_verify_paths()
    if getattr(default_paths, "cafile", None):
        paths.append(str(default_paths.cafile))
    if certifi is not None:  # pragma: no cover - optional dependency
        try:
            paths.append(certifi.where())
        except Exception:  # pragma: no cover - defensive
            pass
    return {"paths": paths}


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_summary(block: str) -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = ""
    if SUMMARY_PATH.exists():
        existing = SUMMARY_PATH.read_text(encoding="utf-8")
        if existing and not existing.endswith("\n"):
            existing += "\n"
    SUMMARY_PATH.write_text(existing + block, encoding="utf-8")


def build_summary(payload: Dict[str, Any]) -> str:
    lines = ["## IDMC reachability", f"Timestamp: {_now()}\n"]
    http = payload.get("http", {})
    lines.append(f"- Base URL: {payload.get('base_url')}")
    lines.append(f"- Endpoint: {payload.get('probe_url')}")
    status = http.get("status")
    if status is not None:
        lines.append(f"- HTTP status: {status} ({http.get('bytes', 0)} bytes)")
    if "error" in http:
        lines.append(f"- HTTP error: {http['error']}")
    tls = payload.get("tls", {})
    if tls.get("ok"):
        lines.append(f"- TLS: {tls.get('version')} cipher {tls.get('cipher')}")
    elif tls.get("error"):
        lines.append(f"- TLS error: {tls['error']}")
    dns = payload.get("dns", {})
    records = dns.get("records") or []
    if records:
        sample = ", ".join(sorted(record["address"] for record in records[:3]))
        lines.append(f"- DNS resolved: {sample}")
    tcp = payload.get("tcp", {})
    if tcp.get("egress"):
        lines.append(f"- Egress IP: {tcp['egress'][0]}")
    bundle = payload.get("ca_bundle", {})
    paths = bundle.get("paths") or []
    if paths:
        lines.append(f"- CA bundle: {paths[0]}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parsed = urlparse(BASE)
    host = parsed.hostname or BASE.split("//")[-1].split("/")[0]
    port = parsed.port or (443 if parsed.scheme == "https" else 80)

    probe_url = f"{BASE}{ENDPOINT}"
    if QUERY:
        probe_url = f"{probe_url}?{QUERY}" if "?" not in ENDPOINT else f"{BASE}{ENDPOINT}&{QUERY}"

    payload: Dict[str, Any] = {
        "base_url": BASE,
        "probe_url": probe_url,
        "host": host,
        "port": port,
        "timestamp": _now(),
    }

    payload["dns"] = probe_dns(host)
    payload["tcp"] = probe_tcp(host, port)
    payload["tls"] = probe_tls(host, port)
    payload["http"] = probe_http(probe_url)
    payload["ca_bundle"] = ca_bundle_info()

    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    write_json(DIAG_DIR / "reachability.json", payload)
    append_summary(build_summary(payload))

    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - entrypoint
    raise SystemExit(main())
