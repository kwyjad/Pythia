#!/usr/bin/env python3
"""Lightweight reachability probe for the IDMC (IDU) API."""
from __future__ import annotations

import contextlib
import json
import os
import socket
import ssl
import time
import urllib.request
from urllib.parse import urlparse

BASE = os.getenv("IDMC_BASE_URL", "https://backend.idmcdb.org")
PROBE_URL = os.getenv("IDMC_PROBE_URL", f"{BASE}/")
PARSED = urlparse(PROBE_URL)
HOST = PARSED.hostname or PARSED.netloc.split(":")[0]
PORT = PARSED.port or (443 if PARSED.scheme == "https" else 80)


def main() -> int:
    payload: dict[str, object | None] = {
        "system": "IDMC",
        "base": BASE,
        "host": HOST,
        "port": PORT,
        "dns": None,
        "tcp_ms": None,
        "tls_ok": None,
        "http": None,
        "error": None,
    }
    start = time.time()

    raw_sock: socket.socket | None = None
    tls_sock: ssl.SSLSocket | None = None
    try:
        ip = socket.gethostbyname(HOST)
        payload["dns"] = ip

        tcp_start = time.time()
        raw_sock = socket.create_connection((HOST, PORT), timeout=5)
        payload["tcp_ms"] = int((time.time() - tcp_start) * 1000)

        if PARSED.scheme == "https":
            ctx = ssl.create_default_context()
            tls_sock = ctx.wrap_socket(raw_sock, server_hostname=HOST)
            tls_sock.do_handshake()
            payload["tls_ok"] = True
            raw_sock = None
        else:
            payload["tls_ok"] = False

        method = "HEAD"
        request = urllib.request.Request(PROBE_URL, method=method)
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                payload["http"] = {"status": response.status}
        except Exception as http_error:  # pragma: no cover - network dependent
            payload["http"] = {"status": None, "note": str(http_error)}
    except Exception as exc:  # pragma: no cover - defensive
        payload["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        for candidate in (tls_sock, raw_sock):
            if candidate is not None:
                with contextlib.suppress(Exception):
                    candidate.close()
        payload["elapsed_ms"] = int((time.time() - start) * 1000)
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
