#!/usr/bin/env python3
"""Collect reachability diagnostics for the DTM API."""

from __future__ import annotations

import json
import os
import socket
import ssl
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple
from urllib.error import URLError
from urllib.request import urlopen

import certifi

try:  # pragma: no cover - informational only
    import platform
except ImportError:  # pragma: no cover - defensive
    platform = None  # type: ignore[assignment]

try:  # pragma: no cover - informational only
    import requests
except Exception:  # pragma: no cover - optional dependency
    requests = None  # type: ignore[assignment]

DIAG_DIR = Path("diagnostics/ingestion/dtm")
JSON_PATH = DIAG_DIR / "reachability.json"
DNS_TXT = DIAG_DIR / "reachability_dns.txt"
TCP_TXT = DIAG_DIR / "reachability_tcp.txt"
TLS_TXT = DIAG_DIR / "reachability_tls.txt"
CURL_TXT = DIAG_DIR / "reachability_curl.txt"
EGRESS_TXT = DIAG_DIR / "reachability_egress.txt"
TEXT_LIMIT = 2048


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _elapsed_ms(start: float) -> int:
    return int(round((time.monotonic() - start) * 1000))


def _truncate(value: str, limit: int = TEXT_LIMIT) -> str:
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
            payload["peer"] = list(sock.getpeername())
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
                payload["version"] = wrapped.version()
                payload["cipher"] = wrapped.cipher()[0] if wrapped.cipher() else None
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


def probe_curl_head(url: str, timeout: int) -> Dict[str, Any]:
    started = time.monotonic()
    payload: Dict[str, Any] = {"exit_code": None}
    cmd = [
        "curl",
        "-I",
        "-L",
        "--max-time",
        str(timeout),
        "--silent",
        "--show-error",
        url,
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout + 5,
        )
        payload["exit_code"] = result.returncode
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        if stdout:
            first_line = stdout.splitlines()[0]
            payload["status_line"] = first_line
            payload["headers"] = _truncate(stdout)
        if stderr:
            payload["stderr"] = _truncate(stderr)
        if result.returncode != 0 and not stderr:
            payload["error"] = f"curl exited {result.returncode}"
    except FileNotFoundError:
        payload["error"] = "curl not available"
    except Exception as exc:  # pragma: no cover - defensive
        payload["error"] = str(exc)
    payload["elapsed_ms"] = _elapsed_ms(started)
    return payload


def probe_egress(url: str, timeout: float) -> Dict[str, Any]:
    started = time.monotonic()
    payload: Dict[str, Any] = {}
    try:
        with urlopen(url, timeout=timeout) as response:
            data = response.read(TEXT_LIMIT).decode("utf-8", errors="replace").strip()
            payload["status_code"] = response.status
            payload["text"] = _truncate(data)
    except URLError as exc:  # pragma: no cover - network dependent
        payload["error"] = str(exc.reason)
    except Exception as exc:  # pragma: no cover - defensive
        payload["error"] = str(exc)
    payload["elapsed_ms"] = _elapsed_ms(started)
    return payload


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_text_summary(label: str, payload: Dict[str, Any]) -> str:
    lines = [label]
    for key, value in sorted(payload.items()):
        if isinstance(value, (dict, list)):
            text = json.dumps(value, ensure_ascii=False)
        else:
            text = str(value)
        lines.append(f"- {key}: {text}")
    return "\n".join(lines) + "\n"


def main() -> int:
    host = os.getenv("DTM_REACHABILITY_HOST", "dtmapi.iom.int").strip() or "dtmapi.iom.int"
    port = int(os.getenv("DTM_REACHABILITY_PORT", "443"))
    timeout = float(os.getenv("DTM_REACHABILITY_TIMEOUT", "5"))
    curl_timeout = int(max(timeout, 5))

    result: Dict[str, Any] = {
        "target_host": host,
        "target_port": port,
        "generated_at": _now_iso(),
        "ca_bundle": ssl.get_default_verify_paths().cafile or certifi.where(),
        "python_version": platform.python_version() if platform else sys.version,
        "requests_version": getattr(requests, "__version__", "not-installed"),
    }

    dns_info = probe_dns(host)
    tcp_info = probe_tcp(host, port, timeout)
    tls_info = probe_tls(host, port, timeout)
    scheme = "https" if port == 443 else "http"
    curl_url = f"{scheme}://{host}:{port}/"
    curl_info = probe_curl_head(curl_url, curl_timeout)
    egress_info = {
        "ifconfig_me": probe_egress("https://ifconfig.me/ip", timeout),
        "ipify": probe_egress("https://api.ipify.org?format=text", timeout),
    }

    result.update(
        {
            "dns": dns_info,
            "tcp": tcp_info,
            "tls": tls_info,
            "curl_head": curl_info,
            "egress": egress_info,
            "completed_at": _now_iso(),
        }
    )

    try:
        write_json(JSON_PATH, result)
        write_text(DNS_TXT, build_text_summary("DNS", dns_info))
        write_text(TCP_TXT, build_text_summary("TCP", tcp_info))
        write_text(TLS_TXT, build_text_summary("TLS", tls_info))
        write_text(CURL_TXT, build_text_summary("HTTP HEAD", curl_info))
        write_text(EGRESS_TXT, build_text_summary("Egress", egress_info))
    except Exception:
        # Diagnostics should not fail the pipeline; swallow errors.
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
