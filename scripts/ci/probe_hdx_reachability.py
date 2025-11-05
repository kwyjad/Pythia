"""Reachability probe for the HDX IDMC dataset."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

try:  # pragma: no cover - optional dependency guard
    import requests
except Exception:  # pragma: no cover - defensive fallback
    requests = None  # type: ignore[assignment]


DATASET_DEFAULT = "preliminary-internal-displacement-updates"
RESOURCE_DEFAULT = "1ace9c2a-7daf-4563-ac15-f2aa5071cd40"
USER_AGENT = "Pythia-IDMC/1.0"
TIMEOUT = 30.0
DIAG_DIR = Path("diagnostics/ingestion/idmc")
JSON_PATH = DIAG_DIR / "hdx_probe.json"
MARKDOWN_PATH = DIAG_DIR / "hdx_probe.md"
MIN_BYTES = 50_000


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _resolve_dataset() -> str:
    return os.getenv("IDMC_HDX_DATASET", DATASET_DEFAULT).strip() or DATASET_DEFAULT


def _resolve_resource_id() -> str:
    for name in ("IDMC_HDX_RESOURCE_ID", "IDMC_HDX_RESOURCE"):
        raw = os.getenv(name)
        if raw:
            value = str(raw).strip()
            if value:
                return value
    return RESOURCE_DEFAULT


def _resolve_package_url(base: str, dataset: str) -> str:
    base_clean = base.rstrip("/")
    return f"{base_clean}/api/3/action/package_show?id={dataset}"


def _select_resource(
    result: Mapping[str, Any], target_id: Optional[str] = None
) -> Optional[Mapping[str, Any]]:
    resources = result.get("resources")
    if not isinstance(resources, list):
        return None
    candidates: list[tuple[int, str, Mapping[str, Any]]] = []
    for entry in resources:
        if not isinstance(entry, Mapping):
            continue
        fmt = str(entry.get("format", "")).lower()
        if fmt != "csv":
            continue
        identifier = str(entry.get("id", "")).strip()
        if target_id and identifier == target_id:
            return entry
        name = str(entry.get("name", "")).lower()
        url = str(entry.get("url", "")).lower()
        preferred = 1 if "idus_view_flat" in name or "idus_view_flat" in url else 0
        stamp = str(entry.get("last_modified") or entry.get("created") or "")
        candidates.append((preferred, stamp, entry))
    if not candidates:
        return None
    candidates.sort()
    # Prefer explicit id match above; otherwise choose last preferred by timestamp.
    preferred_candidates = [item for item in candidates if item[0] == 1]
    if preferred_candidates:
        return preferred_candidates[-1][2]
    return candidates[-1][2]


def _probe_package(package_url: str) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"package_url": package_url}
    if requests is None:
        payload["error"] = "requests-not-installed"
        return payload
    headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
    try:
        response = requests.get(package_url, headers=headers, timeout=TIMEOUT)
        payload["package_status_code"] = response.status_code
        response.raise_for_status()
        payload["package_response"] = response.json()
    except requests.RequestException as exc:  # pragma: no cover - network dependent
        payload["error"] = str(exc)
        payload["exception"] = exc.__class__.__name__
        status = getattr(exc, "response", None)
        if status is not None:
            payload["package_status_code"] = getattr(status, "status_code", None)
    return payload


def _probe_resource(url: str) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"resource_url": url}
    if requests is None:
        payload["error"] = "requests-not-installed"
        return payload
    headers = {"User-Agent": USER_AGENT, "Accept": "text/csv"}
    try:
        response = requests.get(url, headers=headers, timeout=TIMEOUT, allow_redirects=True)
        payload["resource_status_code"] = response.status_code
        response.raise_for_status()
        content = response.content or b""
        payload["resource_bytes"] = len(content)
        payload["content_length"] = response.headers.get("Content-Length")
        payload["bytes_ok"] = payload["resource_bytes"] >= MIN_BYTES
        try:
            header_line = content.splitlines()[0].decode("utf-8", "ignore") if content else ""
        except Exception:  # pragma: no cover - defensive decode
            header_line = ""
        payload["header_line"] = header_line
        header_lower = header_line.lower()
        if header_lower:
            payload["header_has_iso3"] = "iso3" in header_lower
            payload["header_has_value"] = any(
                key in header_lower for key in ("figure", "new_displacements")
            )
    except requests.RequestException as exc:  # pragma: no cover - network dependent
        payload["error"] = str(exc)
        payload["exception"] = exc.__class__.__name__
        status = getattr(exc, "response", None)
        if status is not None and "resource_status_code" not in payload:
            payload["resource_status_code"] = getattr(status, "status_code", None)
    return payload


def _build_markdown(payload: Mapping[str, Any]) -> str:
    lines = ["## HDX Reachability", ""]
    dataset = payload.get("dataset") or DATASET_DEFAULT
    lines.append(f"- **Dataset:** `{dataset}`")

    package_status = payload.get("package_status_code")
    if package_status is not None:
        lines.append(f"- **package_show:** status={package_status}")
    package_error = payload.get("package_error") or payload.get("error")
    if package_error:
        lines.append(f"- **package_show error:** {package_error}")

    resource_id = payload.get("resource_id") or payload.get("target", {}).get(
        "resource_id"
    )
    if resource_id:
        lines.append(f"- **Resource id:** `{resource_id}`")

    resource_status = payload.get("resource_status_code")
    resource_url = payload.get("resource_url")
    if resource_status is not None or resource_url:
        status_text = (
            f"status={resource_status}" if resource_status is not None else "status=unknown"
        )
        if resource_url:
            status_text += f" url={resource_url}"
        lines.append(f"- **Resource:** {status_text}")

    threshold = payload.get("bytes_threshold") or MIN_BYTES
    resource_bytes = payload.get("resource_bytes")
    bytes_ok = payload.get("bytes_ok")
    if resource_bytes is not None:
        suffix = " (ok)" if bytes_ok else " (below minimum)"
        lines.append(
            f"- **Bytes downloaded:** {resource_bytes} / min {threshold}{suffix}"
        )

    header_line = payload.get("header_line")
    if header_line:
        lines.append(f"- **Header sample:** `{header_line}`")
        header_checks = []
        if payload.get("header_has_iso3"):
            header_checks.append("iso3")
        if payload.get("header_has_value"):
            header_checks.append("figure/new_displacements")
        if header_checks:
            lines.append(f"- **Header contains:** {', '.join(header_checks)}")

    resource_error = payload.get("resource_error")
    if resource_error:
        lines.append(f"- **Resource error:** {resource_error}")

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    dataset = _resolve_dataset()
    resource_target = _resolve_resource_id()
    base_url = os.getenv("HDX_BASE", "https://data.humdata.org")
    package_url = _resolve_package_url(base_url, dataset)
    diagnostics: Dict[str, Any] = {
        "dataset": dataset,
        "generated_at": _now_iso(),
        "package_url": package_url,
        "base_url": base_url,
        "target": {"dataset": dataset, "resource_id": resource_target},
        "bytes_threshold": MIN_BYTES,
    }

    package_diag = _probe_package(package_url)
    diagnostics.update({
        "package_status_code": package_diag.get("package_status_code"),
    })
    if package_diag.get("error"):
        diagnostics["package_error"] = package_diag.get("error")
        diagnostics["exception"] = package_diag.get("exception")
    result = package_diag.get("package_response")
    resource_diag: Dict[str, Any] = {}
    if isinstance(result, Mapping) and result.get("success"):
        chosen = _select_resource(result.get("result") or {}, resource_target)
        if chosen:
            resource_url = str(chosen.get("url", ""))
            diagnostics["resource_url"] = resource_url
            diagnostics["resource_name"] = chosen.get("name")
            diagnostics["resource_id"] = chosen.get("id")
            if resource_url:
                resource_diag = _probe_resource(resource_url)
        else:
            diagnostics["package_error"] = "no_csv_resource"
            diagnostics.setdefault("resource_id", resource_target)
    else:
        diagnostics.setdefault("resource_id", resource_target)
    diagnostics.update({
        "resource_status_code": resource_diag.get("resource_status_code"),
        "resource_bytes": resource_diag.get("resource_bytes"),
        "bytes_ok": resource_diag.get("bytes_ok"),
        "header_line": resource_diag.get("header_line"),
        "header_has_iso3": resource_diag.get("header_has_iso3"),
        "header_has_value": resource_diag.get("header_has_value"),
        "content_length": resource_diag.get("content_length"),
    })
    if resource_diag.get("error"):
        diagnostics["resource_error"] = resource_diag.get("error")
        diagnostics.setdefault("exception", resource_diag.get("exception"))
    elif (
        resource_diag.get("resource_bytes") is not None
        and not resource_diag.get("bytes_ok")
    ):
        diagnostics.setdefault("resource_error", "bytes_below_minimum")

    try:
        DIAG_DIR.mkdir(parents=True, exist_ok=True)
        JSON_PATH.write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        markdown = _build_markdown(diagnostics)
        MARKDOWN_PATH.write_text(markdown, encoding="utf-8")
        print(markdown)
    except Exception:  # pragma: no cover - diagnostics should not fail the job
        return 0
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())

