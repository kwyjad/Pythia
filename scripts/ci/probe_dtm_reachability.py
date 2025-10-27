#!/usr/bin/env python3
"""Capture reachability diagnostics for the public DTM API."""

from __future__ import annotations

import json
import logging
import socket
import ssl
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import certifi
import requests

TARGET_HOST = "dtmapi.iom.int"
TARGET_PORT = 443
TARGET_URL = f"https://{TARGET_HOST}"

LOGGER = logging.getLogger("probe_dtm_reachability")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _diagnostics_path() -> Path:
    return _repo_root() / "diagnostics" / "ingestion" / "dtm" / "reachability.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_dns_lookup(host: str) -> Dict[str, Any]:
    started = time.perf_counter()
    result: Dict[str, Any] = {"host": host, "records": []}
    try:
        infos = socket.getaddrinfo(host, None)
    except Exception as exc:  # pragma: no cover - network dependent
        result["error"] = str(exc)
    else:
        seen: List[str] = []
        for family, _, _, _, sockaddr in infos:
            if not sockaddr:
                continue
            address = sockaddr[0]
            entry = {"address": address, "family": socket.AddressFamily(family).name}
            if address not in seen:
                result["records"].append(entry)
                seen.append(address)
    finally:
        result["elapsed_ms"] = int((time.perf_counter() - started) * 1000)
    return result


def _safe_tcp_probe(host: str, port: int, timeout: float = 5.0) -> Dict[str, Any]:
    started = time.perf_counter()
    payload: Dict[str, Any] = {"host": host, "port": port, "timeout_s": timeout}
    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            elapsed = int((time.perf_counter() - started) * 1000)
            peer = sock.getpeername()
            payload.update({"ok": True, "elapsed_ms": elapsed, "peer": list(peer)})
            return payload
    except Exception as exc:  # pragma: no cover - network dependent
        payload.update({"ok": False, "error": str(exc)})
    payload.setdefault("elapsed_ms", int((time.perf_counter() - started) * 1000))
    return payload


def _safe_tls_probe(host: str, port: int, timeout: float = 5.0) -> Dict[str, Any]:
    started = time.perf_counter()
    payload: Dict[str, Any] = {"host": host, "port": port}
    context = ssl.create_default_context(cafile=certifi.where())
    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            with context.wrap_socket(sock, server_hostname=host) as tls_sock:
                cert = tls_sock.getpeercert()
                payload["ok"] = True
                payload["elapsed_ms"] = int((time.perf_counter() - started) * 1000)
                payload["subject"] = cert.get("subject")
                payload["issuer"] = cert.get("issuer")
                san = []
                for entry in cert.get("subjectAltName", []):
                    if isinstance(entry, tuple) and len(entry) >= 2:
                        san.append({"type": entry[0], "value": entry[1]})
                payload["subject_alt_name"] = san
                payload["not_before"] = cert.get("notBefore")
                payload["not_after"] = cert.get("notAfter")
                return payload
    except Exception as exc:  # pragma: no cover - network dependent
        payload.update({"ok": False, "error": str(exc)})
    payload.setdefault("elapsed_ms", int((time.perf_counter() - started) * 1000))
    return payload


def _safe_curl_probe(url: str, timeout: int = 10) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"url": url, "timeout_s": timeout}
    try:
        result = subprocess.run(
            ["curl", "-I", "--max-time", str(timeout), url],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        payload["error"] = "curl-not-found"
        return payload
    except Exception as exc:  # pragma: no cover - environment dependent
        payload["error"] = str(exc)
        return payload
    payload["exit_code"] = result.returncode
    stdout = (result.stdout or "").strip().splitlines()
    stderr = (result.stderr or "").strip()
    if stdout:
        payload["status_line"] = stdout[0]
    if stderr:
        payload["stderr"] = stderr
    return payload


def _safe_request(url: str, timeout: float = 5.0) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"url": url}
    try:
        response = requests.get(url, timeout=timeout)
    except Exception as exc:  # pragma: no cover - network dependent
        payload["error"] = str(exc)
        return payload
    payload["status_code"] = response.status_code
    payload["text"] = (response.text or "").strip()
    return payload


def _gather_reachability() -> Dict[str, Any]:
    started = _now_iso()
    payload: Dict[str, Any] = {
        "generated_at": started,
        "target_host": TARGET_HOST,
        "target_port": TARGET_PORT,
        "python_version": sys.version.split()[0],
        "requests_version": getattr(requests, "__version__", "unknown"),
        "ca_bundle": certifi.where(),
    }
    payload["dns"] = _safe_dns_lookup(TARGET_HOST)
    payload["tcp"] = _safe_tcp_probe(TARGET_HOST, TARGET_PORT)
    payload["tls"] = _safe_tls_probe(TARGET_HOST, TARGET_PORT)
    payload["curl_head"] = _safe_curl_probe(f"{TARGET_URL}/")
    payload["egress"] = {
        "ifconfig_me": _safe_request("https://ifconfig.me/ip"),
        "api_ipify_org": _safe_request("https://api.ipify.org"),
    }
    payload["completed_at"] = _now_iso()
    return payload


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    diagnostics_path = _diagnostics_path()
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = _gather_reachability()
        diagnostics_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        LOGGER.info("Reachability diagnostics written to %s", diagnostics_path)
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.error("Failed to gather reachability diagnostics: %s", exc)
        fallback = {
            "generated_at": _now_iso(),
            "error": str(exc),
            "target_host": TARGET_HOST,
        }
        diagnostics_path.write_text(json.dumps(fallback, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
