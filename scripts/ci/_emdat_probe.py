# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def _load_json(path: Path) -> Mapping[str, Any] | None:
    if not path.exists():
        return None
    raw = path.read_text(encoding="utf-8", errors="replace")
    data = json.loads(raw)
    if isinstance(data, Mapping):
        return data
    raise TypeError(f"expected mapping in {path}, found {type(data).__name__}")


def load_probe_payload(path: Path) -> Mapping[str, Any] | None:
    try:
        return _load_json(path)
    except FileNotFoundError:
        return None
    except TypeError:
        raise


def load_effective_payload(path: Path) -> Mapping[str, Any] | None:
    try:
        return _load_json(path)
    except FileNotFoundError:
        return None
    except TypeError:
        raise


def summarize_effective(payload: Mapping[str, Any] | None) -> list[str]:
    if payload is None:
        return ["- effective.json: missing"]
    if not isinstance(payload, Mapping):
        return [
            f"- effective.json: unexpected payload type ({type(payload).__name__})"
        ]

    def _bool_label(flag: Any, *, true: str = "on", false: str = "off") -> str:
        return true if bool(flag) else false

    lines: list[str] = []
    source = str(payload.get("source_type") or "unknown")
    override = str(payload.get("source_override") or "").strip()
    if override:
        lines.append(f"- source: {source} (override={override})")
    else:
        lines.append(f"- source: {source}")

    network_env = payload.get("network_env")
    network_line = f"- network: {_bool_label(payload.get('network'))}"
    if network_env:
        network_line += f" (env={network_env})"
    lines.append(network_line)

    key_label = "present" if payload.get("api_key_present") else "absent"
    lines.append(f"- api key: {key_label}")

    from_year = payload.get("default_from_year")
    to_year = payload.get("default_to_year")
    include_hist = bool(payload.get("include_hist"))
    if from_year is not None and to_year is not None:
        lines.append(
            f"- default window: {from_year}–{to_year} (include_hist={str(include_hist).lower()})"
        )

    classif_count = payload.get("classif_count")
    iso_count = payload.get("iso_count")
    if isinstance(classif_count, int):
        classif_part = str(classif_count)
    else:
        classif_part = "unknown"
    if isinstance(iso_count, int) and iso_count > 0:
        iso_part = str(iso_count)
    else:
        iso_part = "none"
    lines.append(f"- classif: {classif_part}  iso filter: {iso_part}")

    recorded = payload.get("recorded_at") or payload.get("ts")
    if recorded:
        lines.append(f"- recorded_at: {recorded}")

    lines.append("- effective file: diagnostics/ingestion/emdat/effective.json")
    return lines


def summarize_probe(payload: Mapping[str, Any] | None) -> list[str]:
    if payload is None:
        return ["- probe.json: missing"]
    if not isinstance(payload, Mapping):
        return [f"- probe.json: unexpected payload type ({type(payload).__name__})"]

    lines: list[str] = []
    status = payload.get("status")
    if isinstance(status, int):
        status_text = str(status)
    elif status:
        status_text = str(status)
    else:
        status_text = "unknown"
    state = "ok" if payload.get("ok") else "fail"
    lines.append(f"- status: {status_text} ({state})")

    api_version = payload.get("api_version") or "?"
    table_version = payload.get("table_version") or "?"
    lines.append(f"- api_version: {api_version}   table_version: {table_version}")

    latency = payload.get("latency_ms")
    if isinstance(latency, (int, float)):
        latency_line = f"- latency: {int(round(latency))} ms"
    else:
        latency_line = "- latency: n/a"
    lines.append(latency_line)

    metadata_ts = payload.get("metadata_timestamp") or payload.get("timestamp")
    if metadata_ts:
        lines.append(f"- metadata timestamp: {metadata_ts}")

    requests_block = payload.get("requests")
    if isinstance(requests_block, Mapping) and requests_block:
        total = requests_block.get("total", 0)
        ok_count = requests_block.get("2xx", 0)
        four = requests_block.get("4xx", 0)
        five = requests_block.get("5xx", 0)
        lines.append(
            f"- requests: total={total}  2xx={ok_count}  4xx={four}  5xx={five}"
        )

    if payload.get("error"):
        lines.append(f"- error: {payload['error']}")

    recorded = payload.get("recorded_at") or payload.get("ts")
    if recorded:
        lines.append(f"- recorded_at: {recorded}")

    lines.append("- probe file: diagnostics/ingestion/emdat/probe.json")
    return lines
