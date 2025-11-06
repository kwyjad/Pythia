"""Connectivity probe for the IDMC PostgREST endpoint."""
from __future__ import annotations

import calendar
import json
import os
import socket
import ssl
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple
from urllib.error import URLError
from urllib.request import Request, urlopen
from urllib.parse import urlencode, urlparse, urlunparse

import certifi

try:  # pragma: no cover - optional dependency
    import requests
except Exception:  # pragma: no cover - requests may not be available
    requests = None  # type: ignore[assignment]

REQUESTS_VERSION = getattr(requests, "__version__", None) if requests else None

DIAG_DIR = Path("diagnostics/ingestion/idmc")
JSON_PATH = DIAG_DIR / "probe.json"
MARKDOWN_PATH = DIAG_DIR / "probe.md"
SUMMARY_PATH = MARKDOWN_PATH
TARGET_HOST = os.getenv("IDMC_PROBE_HOST", "backend.idmcdb.org").strip() or "backend.idmcdb.org"
TARGET_PORT = int(os.getenv("IDMC_PROBE_PORT", "443"))
TARGET_PATH = os.getenv(
    "IDMC_PROBE_PATH", "external-api/gidd/displacements"
).lstrip("/")
TARGET_SCHEME = os.getenv("IDMC_PROBE_SCHEME", "https").strip() or "https"
TARGET_URL = os.getenv(
    "IDMC_PROBE_URL",
    f"{TARGET_SCHEME}://{TARGET_HOST}:{TARGET_PORT}/{TARGET_PATH}",
)
HELIX_BASE_URL = (
    os.getenv("IDMC_HELIX_BASE_URL", "https://helix-tools-api.idmcdb.org").strip()
    or "https://helix-tools-api.idmcdb.org"
)
HELIX_DISPLACEMENTS_PATH = "/external-api/gidd/displacements"
HELIX_CLIENT_ID = os.getenv("IDMC_HELIX_CLIENT_ID", "").strip()
HELIX_ENABLED = bool(HELIX_CLIENT_ID)
SUMMARY_TITLE = "IDMC Reachability"
DISPLAY_URL = TARGET_URL


def _env_truthy(value: str | None) -> bool:
    if value is None:
        return False
    lowered = value.strip().lower()
    return lowered in {"1", "true", "yes", "on"}


ALLOW_HDX_FALLBACK = _env_truthy(os.getenv("IDMC_ALLOW_HDX_FALLBACK"))
HDX_PACKAGE_ID = (os.getenv("IDMC_HDX_PACKAGE_ID") or "").strip()
HDX_BASE_URL = (
    os.getenv("IDMC_HDX_BASE_URL")
    or os.getenv("HDX_BASE")
    or "https://data.humdata.org"
).strip() or "https://data.humdata.org"

if HELIX_ENABLED:
    helix_base = HELIX_BASE_URL.rstrip("/")
    today = datetime.now(timezone.utc).date()
    month_start = today.replace(day=1)
    month_end = today.replace(day=calendar.monthrange(today.year, today.month)[1])
    iso_candidate = os.getenv("IDMC_PROBE_ISO3", "FRA").strip() or "FRA"
    release_env = (
        os.getenv("IDMC_HELIX_VERSION", "").strip()
        or os.getenv("IDMC_HELIX_ENV", "").strip()
        or "RELEASE"
    )
    helix_params = {
        "start": month_start.isoformat(),
        "end": month_end.isoformat(),
        "iso3": iso_candidate,
        "limit": "1",
        "release_environment": release_env,
    }
    if HELIX_CLIENT_ID:
        helix_params["client_id"] = HELIX_CLIENT_ID
    helix_query = urlencode({key: value for key, value in helix_params.items() if value})
    helix_url = f"{helix_base}{HELIX_DISPLACEMENTS_PATH}?{helix_query}"
    parsed = urlparse(helix_url)
    if parsed.scheme:
        TARGET_SCHEME = parsed.scheme
    if parsed.hostname:
        TARGET_HOST = parsed.hostname
    TARGET_PORT = parsed.port or (443 if TARGET_SCHEME == "https" else 80)
    helix_path = parsed.path.lstrip("/")
    if parsed.query:
        helix_path = f"{helix_path}?{parsed.query}"
    TARGET_PATH = helix_path
    TARGET_URL = urlunparse(
        (
            parsed.scheme or TARGET_SCHEME,
            parsed.netloc or f"{TARGET_HOST}:{TARGET_PORT}",
            parsed.path,
            "",
            parsed.query,
            "",
        )
    )
    DISPLAY_URL = TARGET_URL
    if HELIX_CLIENT_ID:
        DISPLAY_URL = DISPLAY_URL.replace(HELIX_CLIENT_ID, "REDACTED")
    SUMMARY_TITLE = "Helix (GIDD) Reachability"

_path_split = TARGET_PATH.split("?", 1)
_path_only = _path_split[0]
_path_query = _path_split[1] if len(_path_split) > 1 else ""
BASE = f"{TARGET_SCHEME}://{TARGET_HOST}:{TARGET_PORT}"
ENDPOINT = "/" + _path_only.lstrip("/")
QUERY = _path_query
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


def ca_bundle_info() -> Dict[str, Any]:
    cafile = ssl.get_default_verify_paths().cafile or certifi.where()
    payload: Dict[str, Any] = {"paths": []}
    if cafile:
        payload["paths"].append(cafile)
    return payload


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


def probe_tcp(host: str, port: int, timeout: float | None = None) -> Dict[str, Any]:
    started = time.monotonic()
    payload: Dict[str, Any] = {"ok": False}
    timeout_value = CONNECT_TIMEOUT if timeout is None else timeout
    try:
        with socket.create_connection((host, port), timeout=timeout_value) as sock:
            payload["ok"] = True
            try:
                payload["peer"] = list(sock.getpeername())
            except Exception:  # pragma: no cover - platform dependent
                payload["peer"] = None
    except Exception as exc:  # pragma: no cover - network dependent
        payload["error"] = str(exc)
    payload["elapsed_ms"] = _elapsed_ms(started)
    return payload


def probe_tls(host: str, port: int, timeout: float | None = None) -> Dict[str, Any]:
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
        timeout_value = CONNECT_TIMEOUT if timeout is None else timeout
        with socket.create_connection((host, port), timeout=timeout_value) as sock:
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
    url: str,
    *,
    timeout: Tuple[float, float] | None = None,
    verify: bool | str | None = None,
    display_url: str | None = None,
) -> Dict[str, Any]:
    started = time.monotonic()
    timeout_value = timeout or (CONNECT_TIMEOUT, READ_TIMEOUT)
    verify_value = VERIFY_SETTING if verify is None else verify
    payload: Dict[str, Any] = {
        "url": display_url or url,
        "verify": verify_value,
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
            timeout=timeout_value,
            verify=verify_value,
        )
        payload["status_code"] = response.status_code
        payload["reason"] = response.reason
        payload["elapsed_ms"] = _elapsed_ms(started)
        payload["content_type"] = response.headers.get("Content-Type")
        try:
            payload["bytes"] = len(response.content)
        except Exception:
            payload["bytes"] = None
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


def probe_hdx(
    package_id: str,
    *,
    base_url: str,
    timeout: Tuple[float, float],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "package_id": package_id,
        "base_url": base_url,
    }
    if not requests:
        payload["error"] = "requests-not-installed"
        return payload
    if not package_id:
        payload["error"] = "missing_package_id"
        return payload

    base = base_url.rstrip("/")
    query = urlencode({"id": package_id})
    package_url = f"{base}/api/3/action/package_show?{query}"
    payload["package_url"] = package_url

    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    started = time.monotonic()
    try:
        response = requests.get(package_url, headers=headers, timeout=timeout)
        payload["package_status_code"] = response.status_code
        payload["package_elapsed_ms"] = _elapsed_ms(started)
        response.raise_for_status()
        body = response.json()
    except requests.RequestException as exc:  # pragma: no cover - network dependent
        payload["package_elapsed_ms"] = _elapsed_ms(started)
        payload["error"] = f"package_request_error:{exc.__class__.__name__}"
        payload["exception"] = exc.__class__.__name__
        payload["package_status_code"] = getattr(exc.response, "status_code", None)
        return payload
    except ValueError as exc:  # pragma: no cover - defensive
        payload["package_elapsed_ms"] = _elapsed_ms(started)
        payload["error"] = f"package_decode_error:{exc.__class__.__name__}"
        payload["exception"] = exc.__class__.__name__
        return payload

    if not isinstance(body, Mapping) or not body.get("success"):
        payload["error"] = "package_show_unsuccessful"
        return payload

    resources = body.get("result", {}).get("resources", [])
    payload["resource_candidates"] = len(resources)
    chosen: Optional[Mapping[str, Any]] = None

    for resource in resources:
        if not isinstance(resource, Mapping):
            continue
        fmt = str(resource.get("format") or "").strip().lower()
        if fmt != "csv":
            continue
        text = " ".join(
            [str(resource.get("name") or ""), str(resource.get("description") or "")]
        ).lower()
        if "displacement" in text or "disaggreg" in text:
            chosen = resource
            payload["resource_selection"] = "keyword"
            break

    if chosen is None:
        for resource in resources:
            if not isinstance(resource, Mapping):
                continue
            fmt = str(resource.get("format") or "").strip().lower()
            if fmt != "csv":
                continue
            size_raw = resource.get("size")
            try:
                size_value = int(size_raw)
            except (TypeError, ValueError):
                size_value = None
            if size_value is not None and size_value > 50_000:
                chosen = resource
                payload["resource_selection"] = "size_threshold"
                break

    if chosen is None:
        for resource in resources:
            if not isinstance(resource, Mapping):
                continue
            fmt = str(resource.get("format") or "").strip().lower()
            if fmt == "csv":
                chosen = resource
                payload["resource_selection"] = "first_csv"
                break

    if chosen is None:
        payload["error"] = "no_csv_resource"
        return payload

    payload["resource_id"] = chosen.get("id")
    payload["resource_name"] = chosen.get("name")
    url = (
        str(chosen.get("url") or "").strip()
        or str(chosen.get("download_url") or "").strip()
    )
    payload["resource_url"] = url or None
    if not url:
        payload["error"] = "resource_missing_url"
        return payload

    headers = {"User-Agent": USER_AGENT, "Accept": "text/csv"}
    started = time.monotonic()
    try:
        response = requests.head(
            url, headers=headers, timeout=timeout, allow_redirects=True
        )
        payload["resource_status_code"] = response.status_code
        payload["resource_elapsed_ms"] = _elapsed_ms(started)
        response.close()
        if response.status_code in {405, 403}:
            raise requests.RequestException()
    except requests.RequestException:  # pragma: no cover - network dependent
        started = time.monotonic()
        response = None
        try:
            response = requests.get(
                url,
                headers=headers,
                timeout=timeout,
                stream=True,
                allow_redirects=True,
            )
            payload["resource_status_code"] = response.status_code
            payload["resource_elapsed_ms"] = _elapsed_ms(started)
        except requests.RequestException as exc:  # pragma: no cover - network dependent
            payload["resource_elapsed_ms"] = _elapsed_ms(started)
            payload["error"] = f"resource_request_error:{exc.__class__.__name__}"
            payload["resource_exception"] = exc.__class__.__name__
            payload["resource_status_code"] = getattr(exc.response, "status_code", None)
        finally:
            if response is not None:
                try:
                    response.close()
                except Exception:  # pragma: no cover - defensive cleanup
                    pass

    return payload


def build_markdown(payload: Mapping[str, Any]) -> str:
    dns = payload.get("dns", {})
    tcp = payload.get("tcp", {})
    tls = payload.get("tls", {})
    http = payload.get("http", {})
    egress = payload.get("egress", {})
    lines: List[str] = [f"## {SUMMARY_TITLE}", ""]
    target = payload.get("target", {})
    host = target.get("host", TARGET_HOST)
    port = target.get("port", TARGET_PORT)
    url_display = target.get("url")
    lines.append(f"- **Target:** `{host}:{port}`")
    if url_display:
        lines.append(f"- **URL:** {url_display}")
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
        status_value = http.get("status_code")
        if status_value is None:
            status_value = http.get("status")
        if status_value is not None:
            line = f"- HTTP GET: status {status_value}"
            if http.get("elapsed_ms") is not None:
                line += f" ({http.get('elapsed_ms')}ms)"
            if http.get("bytes"):
                line += f" bytes={http.get('bytes')}"
        else:
            error = http.get("exception") or http.get("error") or "unknown"
            line = f"- HTTP GET: error {error}"
            if http.get("elapsed_ms") is not None:
                line += f" after {http.get('elapsed_ms')}ms"
        lines.append(line)
    egress_ip = None
    if isinstance(tcp, Mapping):
        egress_entry = tcp.get("egress")
        if isinstance(egress_entry, (list, tuple)) and egress_entry:
            egress_ip = egress_entry[0]
    if egress_ip is None:
        for label in ("ifconfig.me", "ipify"):
            entry = egress.get(label)
            if isinstance(entry, Mapping) and entry.get("text"):
                egress_ip = entry.get("text")
                break
    if egress_ip:
        lines.append(f"- Egress IP: {egress_ip}")
    ca_bundle = payload.get("ca_bundle") or payload.get("tls", {}).get("ca_bundle")
    if ca_bundle:
        lines.append(f"- **CA bundle:** `{ca_bundle}`")
    verify_flag = http.get("verify") if isinstance(http, Mapping) else None
    if verify_flag is not None:
        lines.append(f"- **Verify TLS:** {verify_flag}")
    hdx = payload.get("hdx", {})
    if hdx:
        lines.append("")
        lines.append("### HDX")
        package_status = hdx.get("package_status_code")
        package_url = hdx.get("package_url") or hdx.get("package_id")
        if package_status is not None:
            lines.append(
                f"- **Package:** status={package_status} — {package_url or 'n/a'}"
            )
        else:
            lines.append(f"- **Package:** {hdx.get('error', 'unavailable')}")
        resource_url = hdx.get("resource_url")
        resource_status = hdx.get("resource_status_code")
        if resource_url:
            detail = f"status={resource_status}" if resource_status is not None else "unavailable"
            lines.append(f"- **Resource:** {detail} — {resource_url}")
        elif hdx.get("error"):
            lines.append(f"- **Resource:** {hdx.get('error')}")

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    base_url = BASE or f"{TARGET_SCHEME}://{TARGET_HOST}:{TARGET_PORT}"
    endpoint = ENDPOINT or "/" + TARGET_PATH.lstrip("/")
    query = QUERY or ""
    runtime_url = base_url + endpoint
    if query:
        runtime_url = f"{runtime_url}?{query}"
    parsed_runtime = urlparse(runtime_url)
    target_host = parsed_runtime.hostname or TARGET_HOST
    if parsed_runtime.port:
        target_port = parsed_runtime.port
    else:
        target_port = TARGET_PORT or (443 if parsed_runtime.scheme == "https" else 80)
    display_url = runtime_url
    if HELIX_CLIENT_ID:
        display_url = display_url.replace(HELIX_CLIENT_ID, "REDACTED")

    started_iso = _now_iso()
    dns_info = probe_dns(target_host)
    tcp_info = probe_tcp(target_host, target_port)
    tls_info = probe_tls(target_host, target_port)
    http_info = probe_http(runtime_url)
    http_info["url"] = display_url
    egress_info = probe_egress(EGRESS_TIMEOUT)
    hdx_info: Optional[Dict[str, Any]] = None
    if ALLOW_HDX_FALLBACK:
        hdx_info = probe_hdx(
            HDX_PACKAGE_ID,
            base_url=HDX_BASE_URL,
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
        )
    payload: Dict[str, Any] = {
        "target": {"host": target_host, "port": target_port, "url": display_url},
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
        "mode": "helix" if HELIX_ENABLED else "backend",
        "summary_title": SUMMARY_TITLE,
        "env": {
            "HTTPS_PROXY": os.getenv("HTTPS_PROXY"),
            "HTTP_PROXY": os.getenv("HTTP_PROXY"),
            "IDMC_HTTP_VERIFY": os.getenv("IDMC_HTTP_VERIFY"),
        },
    }
    if hdx_info is not None:
        payload["hdx"] = hdx_info
    if HELIX_ENABLED:
        parsed_display = urlparse(display_url)
        helix_path = parsed_display.path or ""
        if parsed_display.query:
            helix_path = f"{helix_path}?{parsed_display.query}"
        payload["helix"] = {
            "url": display_url,
            "status": http_info.get("status_code"),
            "bytes": http_info.get("bytes"),
            "path": helix_path,
        }
    try:
        DIAG_DIR.mkdir(parents=True, exist_ok=True)
        json_path = DIAG_DIR / "reachability.json"
        json_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        markdown = build_markdown(payload)
        SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        SUMMARY_PATH.write_text(markdown, encoding="utf-8")
        print(markdown)
    except Exception:  # pragma: no cover - diagnostics must not break CI
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
