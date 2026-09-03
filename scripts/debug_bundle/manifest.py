# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""What is in the bundle, and what each file holds.

A zip of thirty files with run-id suffixes is not self-describing, and the
reader is usually somebody who did not build it. The manifest names every
file, says how many rows or records it carries, and gives one line on what
question it answers.

The schema version is bumped whenever the file SET changes, so a consumer
can tell a bundle missing a file it expects from a bundle built before that
file existed.
"""

from __future__ import annotations

import csv
import gzip
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# 1 — the original twelve files.
# 2 — batch lifecycle, provider objects, workflow logs, env/config, code
#     snapshot, connector freshness, prompt cache, retries, model
#     completeness, anomalies, run comparison, coverage detail, prompt
#     prefixes, and this manifest.
BUNDLE_SCHEMA_VERSION = 2

DESCRIPTIONS: dict[str, str] = {
    "anomalies": "Everything that failed or looks wrong, with severity and where the evidence is. Read this first.",
    "executive_summary": "One-page health report: anomalies, status, cost, cache, connector freshness, coverage.",
    "coverage_detail": "The long enumerations moved out of the summary: seasonal screen-outs, quiet pairs, full question tables.",
    "health_report": "Machine-readable health checks and subsystem statistics.",
    "question_metrics": "Per-question timings, cost, ensemble completeness, triage and RC fields.",
    "question_evidence": "Per-question web-research evidence packs.",
    "hs_country_evidence": "Per-country HS grounding evidence packs.",
    "llm_calls_detail": "Every LLM call: prompt tail, full response, usage, cost, error.",
    "prompt_prefixes": "Each distinct static prompt prefix, once, keyed by hash and referenced from llm_calls_detail.",
    "spd_tables": "Per-question SPD bucket probabilities by model and month.",
    "rc_triage_summary": "Per-country-hazard RC level, triage tier and track.",
    "rc_pass_detail": "Per-RC-pass model output detail.",
    "data_inject_inventory": "Which structured data source was available for each country.",
    "timing_breakdown": "Per-country wall time by pipeline stage, with batched-call counts.",
    "model_config": "The model lineup and the flags that shaped it (superseded in detail by env_and_config).",
    "env_and_config": "Every resolved PYTHIA_* variable, git SHA, workflow hashes, runner, pip freeze, LLM profile and prices.",
    "grounding_detail": "Per-grounding-call stage, backend, source count and error.",
    "batch_lifecycle": "Every provider batch: ids, timings, terminal counts, provider error, and the cost at each tier.",
    "provider_batch_objects": "The raw provider batch objects and their files, captured before they expire upstream.",
    "connector_freshness": "Per-source row count, observation age, staleness verdict and country coverage.",
    "crisiswatch_inject": "CrisisWatch editions in the table, arrow and alert counts, and the ACE questions forecast with no arrow.",
    "model_completeness_rollup": "The question-months aggregated from fewer members than expected, and which member was missing.",
    "prompt_cache_report": "Per phase/provider/model cache reads, writes, hit rate and dollars saved.",
    "retry_report": "Per phase/provider/model attempts, backoff seconds, error classes and breaker trips.",
    "model_completeness": "Per (question, model, month): did the forecast land, and with how many buckets.",
    "run_comparison": "This run's key metrics beside the previous two production runs.",
    "workflow_logs": "Actions logs for the whole cycle: this run, the pipeline stages, and the poller ticks.",
    "code_snapshot": "Verbatim copies of the files most often implicated in a bad run, plus the commit range since the last production run.",
}


def _stem(name: str) -> str:
    """``question_metrics__fc_123.csv`` -> ``question_metrics``."""

    base = name.split("/")[-1]
    for suffix in (".jsonl.gz", ".json", ".csv", ".md", ".txt", ".jsonl"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return base.split("__")[0]


def _count_records(path: Path) -> int | str:
    """Rows for a CSV, records for JSONL, entries for a list-shaped JSON."""

    try:
        name = path.name
        if name.endswith(".csv"):
            with path.open("r", encoding="utf-8", newline="") as f:
                return max(0, sum(1 for _ in csv.reader(f)) - 1)
        if name.endswith(".jsonl.gz"):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                return sum(1 for line in f if line.strip())
        if name.endswith(".jsonl"):
            with path.open("r", encoding="utf-8") as f:
                return sum(1 for line in f if line.strip())
        if name.endswith(".json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                return len(payload)
            if isinstance(payload, dict):
                for key in ("batches", "objects", "runs", "entries", "files"):
                    if isinstance(payload.get(key), list):
                        return len(payload[key])
                return len(payload)
        if name.endswith(".md") or name.endswith(".txt"):
            return sum(1 for _ in path.read_text(encoding="utf-8", errors="replace").splitlines())
    except Exception as exc:  # noqa: BLE001
        return f"unreadable: {exc}"
    return ""


def build(
    out_dir: Path,
    *,
    hs_run_id: str | None,
    forecaster_run_id: str | None,
    pipeline_id: str | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe every file currently in ``out_dir``."""

    files: list[dict[str, Any]] = []
    for path in sorted(out_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix in {".duckdb", ".wal", ".pyc"} or path.suffix == ".zip":
            continue
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(out_dir).as_posix()
        if rel == "BUNDLE_MANIFEST.json":
            continue
        stem = _stem(rel)
        # A directory's files share one description; the directory name is
        # the key so workflow_logs/*.txt do not each need an entry here.
        top = rel.split("/")[0] if "/" in rel else stem
        description = DESCRIPTIONS.get(stem) or DESCRIPTIONS.get(top) or ""
        files.append(
            {
                "name": rel,
                "records": _count_records(path),
                "bytes": path.stat().st_size,
                "description": description,
            }
        )

    manifest: dict[str, Any] = {
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hs_run_id": hs_run_id,
        "forecaster_run_id": forecaster_run_id,
        "pipeline_id": pipeline_id,
        "files": files,
        "n_files": len(files),
        "total_bytes": sum(int(f["bytes"]) for f in files),
    }
    if extra:
        manifest.update(extra)
    return manifest


def annotate_archives(manifest: dict[str, Any], packaging: dict[str, Any]) -> dict[str, Any]:
    """Say which zip each file is in.

    Without this a split bundle names files in the manifest that are not in
    the main zip, and a reader looking for a workflow log concludes the
    collector failed rather than looking in the second artifact.
    """

    main_zip = str(packaging.get("bundle_zip") or "")
    logs_zip = str(packaging.get("workflow_logs_zip") or "")
    split = bool(packaging.get("split"))
    for entry in manifest.get("files") or []:
        name = str(entry.get("name") or "")
        if split and logs_zip and name.startswith("workflow_logs/"):
            entry["archive"] = logs_zip
        else:
            entry["archive"] = main_zip
    manifest["packaging"] = packaging
    return manifest


def write(out_dir: Path, manifest: dict[str, Any]) -> Path:
    path = out_dir / "BUNDLE_MANIFEST.json"
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return path
