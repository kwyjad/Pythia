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
USER_AGENT = "Pythia-IDMC/1.0"
TIMEOUT = 30.0
DIAG_DIR = Path("diagnostics/ingestion/idmc")
JSON_PATH = DIAG_DIR / "hdx_probe.json"
MARKDOWN_PATH = DIAG_DIR / "hdx_probe.md"


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _resolve_dataset() -> str:
    return os.getenv("IDMC_HDX_DATASET", DATASET_DEFAULT).strip() or DATASET_DEFAULT


def _resolve_package_url(base: str, dataset: str) -> str:
    base_clean = base.rstrip("/")
    return f"{base_clean}/api/3/action/package_show?id={dataset}"


def _select_resource(result: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
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
        name = str(entry.get("name", "")).lower()
        url = str(entry.get("url", "")).lower()
        preferred = 1 if "idus_view_flat" in name or "idus_view_flat" in url else 0
        stamp = str(entry.get("last_modified") or entry.get("created") or "")
        candidates.append((preferred, stamp, entry))
    if not candidates:
        return None
    candidates.sort()
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
        response = requests.head(url, headers=headers, timeout=TIMEOUT, allow_redirects=True)
        payload["resource_status_code"] = response.status_code
        if response.status_code in {405, 403}:
            response = requests.get(url, headers=headers, timeout=TIMEOUT, stream=True)
            payload["resource_status_code"] = response.status_code
        response.raise_for_status()
        length = response.headers.get("Content-Length")
        if length is not None:
            try:
                payload["resource_bytes"] = int(length)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                payload["resource_bytes"] = length
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
    resource_status = payload.get("resource_status_code")
    resource_url = payload.get("resource_url")
    if resource_status is not None or resource_url:
        status_text = (
            f"status={resource_status}" if resource_status is not None else "status=unknown"
        )
        if resource_url:
            status_text += f" url={resource_url}"
        lines.append(f"- **Resource:** {status_text}")
    resource_bytes = payload.get("resource_bytes")
    if resource_bytes is not None:
        lines.append(f"- **Bytes reported:** {resource_bytes}")
    resource_error = payload.get("resource_error")
    if resource_error:
        lines.append(f"- **Resource error:** {resource_error}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    dataset = _resolve_dataset()
    base_url = os.getenv("HDX_BASE", "https://data.humdata.org")
    package_url = _resolve_package_url(base_url, dataset)
    diagnostics: Dict[str, Any] = {
        "dataset": dataset,
        "generated_at": _now_iso(),
        "package_url": package_url,
        "base_url": base_url,
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
        chosen = _select_resource(result.get("result") or {})
        if chosen:
            resource_url = str(chosen.get("url", ""))
            diagnostics["resource_url"] = resource_url
            diagnostics["resource_name"] = chosen.get("name")
            if resource_url:
                resource_diag = _probe_resource(resource_url)
        else:
            diagnostics["package_error"] = "no_csv_resource"
    diagnostics.update({
        "resource_status_code": resource_diag.get("resource_status_code"),
        "resource_bytes": resource_diag.get("resource_bytes"),
    })
    if resource_diag.get("error"):
        diagnostics["resource_error"] = resource_diag.get("error")
        diagnostics.setdefault("exception", resource_diag.get("exception"))

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

