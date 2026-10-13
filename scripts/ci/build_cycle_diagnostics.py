# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Every diagnostic of one cycle, gathered into one zip on the publish run.

A cycle's evidence used to sit on five run pages: Resolver Update's debug
bundle, the chained workflows' diagnostics, the weekly ingest's and the
CrisisWatch refresh's artifacts, and Publish's inspection. Diagnosing the
11 October 2026 refusal meant visiting each and reading the Actions API by
hand. This step, run by ``publish_latest_data.yml`` after the release upload,
walks the chain that produced the published DB and puts everything into
``cycle_diagnostics__<YYYY-MM-DD>__<publish_run_id>.zip``.

How the cycle is found:

1. From the source run Publish resolved, follow each run's ``lineage.json``
   (written by ``scripts/ci/chain_diagnostics.py``) back through
   ``trigger_run`` until a run with no trigger. A run with no lineage record
   (older runs, or a workflow that writes none) falls back to the Actions API:
   the newest successful run of its expected upstream workflow created before
   it, within ``UPSTREAM_MAX_GAP_DAYS``.
2. A Pythia Pipeline Stage run brings its pipeline's other stages, found by
   the pipeline id in their run names.
3. Ingest Structured Data and Refresh CrisisWatch runs created from seven
   days before the root run to the publish run, and the canonical run any
   link started from, join by time rather than lineage.

Every run kept is one of ours: event in schedule / workflow_dispatch /
workflow_run / push, from this repository, on ``main`` where a branch is
named. Artifacts are downloaded by RUN id, never searched by name across the
repository, and no DB artifact (``pythia-resolver-db*``) is ever included.

The zip holds ``README.md``, ``CYCLE_SUMMARY.md``, ``manifest.json`` and one
folder per run in time order. It is scanned for every secret in the
environment before it is written (a hit exits 2 and deletes the zip; the
workflow step is non-fatal), and kept under ``--max-bytes`` by dropping, in
order, workflow logs, code snapshots, zips nested from Sibyl's artifacts and
then the largest CSVs cut to head and tail; every drop is in the manifest.
A missing artifact is a line in the README and manifest, never a failure.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - import convenience
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ci.chain_diagnostics import (  # noqa: E402
    ARTIFACT_PREFIX,
    SLUG_TO_NAME,
    WORKFLOWS,
)

TRUSTED_EVENTS = frozenset({"schedule", "workflow_dispatch", "workflow_run", "push"})
DB_ARTIFACT_PREFIX = "pythia-resolver-db"
DEFAULT_MAX_BYTES = 200 * 1024 * 1024
UPSTREAM_MAX_GAP_DAYS = 3
WINDOW_DAYS = 7
MAX_CHAIN_DEPTH = 12

#: Where each workflow's chain continues when it carries no lineage record.
EXPECTED_UPSTREAM: dict[str, tuple[str, ...]] = {
    "Publish Latest Data (Release)": (
        "Pythia — Compute Calibration Weights & Advice", "Sibyl Deep Research",
    ),
    "Pythia — Compute Calibration Weights & Advice": ("Pythia — Compute SPD Scores",),
    "Pythia — Compute SPD Scores": ("Pythia — Compute Resolutions",),
    "Pythia — Compute Resolutions": ("Resolver Update",),
    "Sibyl Deep Research": ("Pythia Pipeline Stage", "Horizon Scanner Triage"),
}

#: Artifacts collected per workflow, besides every run's chain diagnostics.
ARTIFACT_PATTERNS: dict[str, tuple[str, ...]] = {
    "Resolver Update": ("resolver-debug-bundle-*", "backfill-diagnostics"),
    "Pythia — Compute Calibration Weights & Advice": (
        "pythia-ai-analysis-bundle", "pythia-interpreter-scored", "calibration-summary",
    ),
    "Sibyl Deep Research": (
        "pythia-debug-bundle", "pythia-debug-bundle-workflow-logs", "pythia-health-report",
        "pythia-current-run-bundle", "pythia-forecast-attribution-bundle",
        "pythia-stage-diagnostics*", "pythia-interpreter-report",
    ),
    "Pythia Pipeline Stage": ("pythia-stage-diagnostics", "pythia-stage-context"),
    "Ingest Structured Data": ("ingest-diagnostics",),
    "Refresh CrisisWatch Data": ("crisiswatch-data-*",),
    "Publish Latest Data (Release)": ("resolver-inspect",),
}

#: Files the size policy never drops.
PROTECTED_NAMES = frozenset({
    "README.md", "CYCLE_SUMMARY.md", "manifest.json", "summary.md", "issues.json",
    "issues.md", "run.json", "lineage.json", "MISSING.json", "api_run.json",
})

SEVERITY_ORDER = ("blocking", "degraded", "known", "info")


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------


class Gh:
    """The Actions API and artifact downloads through ``gh``. Injectable."""

    def __init__(self, repo: str, *, attempts: int = 3, sleep: Callable[[float], None] = time.sleep):
        self.repo = repo
        self.attempts = attempts
        self.sleep = sleep

    def api(self, path: str) -> Any:
        last: Exception | None = None
        for i in range(self.attempts):
            try:
                out = subprocess.run(["gh", "api", path], capture_output=True, text=True,
                                     check=True, timeout=120).stdout
                return json.loads(out or "null")
            except Exception as exc:  # noqa: BLE001
                last = exc
                if i + 1 < self.attempts:
                    self.sleep(5 * (2 ** i))
        raise RuntimeError(f"gh api {path} failed: {last}")

    def download(self, run_id: str, name: str, dest: Path) -> tuple[bool, str]:
        dest.mkdir(parents=True, exist_ok=True)
        try:
            proc = subprocess.run(
                ["gh", "run", "download", str(run_id), "-n", name, "-D", str(dest), "-R", self.repo],
                capture_output=True, text=True, timeout=900,
            )
        except Exception as exc:  # noqa: BLE001
            return False, f"download error: {exc}"
        if proc.returncode != 0:
            return False, f"download error: {(proc.stderr or proc.stdout).strip()[:300]}"
        return True, "ok"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@dataclass
class RunRecord:
    run_id: str
    info: dict[str, Any]
    reason: str  # why it is in the cycle: source, trigger, fallback, stage, window, canonical
    lineage: dict[str, Any] | None = None
    folder: str = ""
    artifacts: list[dict[str, Any]] = field(default_factory=list)

    @property
    def workflow(self) -> str:
        return str(self.info.get("name") or "")

    @property
    def slug(self) -> str:
        return WORKFLOWS.get(self.workflow, (re.sub(r"[^a-z0-9]+", "_", self.workflow.lower()).strip("_") or "run", ""))[0]

    @property
    def created_at(self) -> str:
        return str(self.info.get("created_at") or "")


def _ts(raw: Any) -> dt.datetime | None:
    if not raw:
        return None
    try:
        t = dt.datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)
    except ValueError:
        return None


def trusted(info: Mapping[str, Any], repo: str, *, require_main: bool = True) -> bool:
    """A run our own triggers started, in this repository (CLAUDE.md, Security)."""
    if str(info.get("event") or "") not in TRUSTED_EVENTS:
        return False
    head = info.get("head_repository") or {}
    if head and str(head.get("full_name") or "") != repo:
        return False
    if require_main and info.get("head_branch") not in (None, "", "main"):
        return False
    return True


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------


class Collector:
    def __init__(self, gh: Any, repo: str, staging: Path, *, wait_sec: float = 600,
                 poll_sec: float = 30, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.gh = gh
        self.repo = repo
        self.staging = staging
        self.wait_sec = wait_sec
        self.poll_sec = poll_sec
        self.sleep = sleep
        self.clock = clock
        self.runs: dict[str, RunRecord] = {}
        self.missing: list[dict[str, Any]] = []
        self.notes: list[str] = []
        self._cache = staging / ".lineage_cache"

    # -- API helpers -----------------------------------------------------------

    def run_info(self, run_id: str) -> dict[str, Any] | None:
        try:
            return self.gh.api(f"repos/{self.repo}/actions/runs/{run_id}") or None
        except Exception as exc:  # noqa: BLE001
            self.missing.append({"run_id": run_id, "artifact": None, "reason": f"run unreadable: {exc}"})
            return None

    def artifacts_of(self, run_id: str) -> list[dict[str, Any]]:
        try:
            listing = self.gh.api(f"repos/{self.repo}/actions/runs/{run_id}/artifacts?per_page=100") or {}
            return list(listing.get("artifacts") or [])
        except Exception as exc:  # noqa: BLE001
            self.missing.append({"run_id": run_id, "artifact": None, "reason": f"artifact listing failed: {exc}"})
            return []

    def workflow_runs(self, workflow_name: str, per_page: int = 50) -> list[dict[str, Any]]:
        file = WORKFLOWS.get(workflow_name, ("", ""))[1]
        if not file:
            return []
        try:
            listing = self.gh.api(
                f"repos/{self.repo}/actions/workflows/{file}/runs?per_page={per_page}") or {}
        except Exception as exc:  # noqa: BLE001
            self.notes.append(f"could not list {workflow_name} runs: {exc}")
            return []
        return [r for r in listing.get("workflow_runs") or [] if trusted(r, self.repo)]

    # -- lineage ---------------------------------------------------------------

    def lineage_of(self, run_id: str, info: Mapping[str, Any]) -> dict[str, Any] | None:
        slug = WORKFLOWS.get(str(info.get("name") or ""), ("", ""))[0]
        if not slug:
            return None
        name = f"{ARTIFACT_PREFIX}-{slug}-{run_id}"
        dest = self._cache / str(run_id)
        if not (dest / "lineage.json").is_file():
            arts = self._wait_for(run_id, name, info)
            if not any(a.get("name") == name and not a.get("expired") for a in arts):
                return None
            ok, _ = self.gh.download(str(run_id), name, dest)
            if not ok:
                return None
        try:
            return json.loads((dest / "lineage.json").read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return None

    def _wait_for(self, run_id: str, name: str, info: Mapping[str, Any]) -> list[dict[str, Any]]:
        """The run's artifacts, waiting while it is still running and ``name`` is absent.

        Calibration dispatches Publish and then writes its own diagnostics, so
        the source run can still be finishing when this looks."""
        deadline = self.clock() + self.wait_sec
        arts = self.artifacts_of(run_id)
        status = str(info.get("status") or "completed")
        while (status != "completed" and not any(a.get("name") == name for a in arts)
               and self.clock() < deadline):
            self.sleep(self.poll_sec)
            fresh = self.run_info(run_id) or {}
            status = str(fresh.get("status") or "completed")
            arts = self.artifacts_of(run_id)
        return arts

    def fallback_upstream(self, info: Mapping[str, Any]) -> str | None:
        """Newest successful run of an expected upstream created before this one."""
        created = _ts(info.get("created_at"))
        best: tuple[dt.datetime, str] | None = None
        for upstream in EXPECTED_UPSTREAM.get(str(info.get("name") or ""), ()):
            for r in self.workflow_runs(upstream):
                if r.get("conclusion") != "success":
                    continue
                t = _ts(r.get("created_at"))
                if t is None or created is None or t >= created:
                    continue
                if (created - t).total_seconds() > UPSTREAM_MAX_GAP_DAYS * 86400:
                    continue
                if upstream == "Pythia Pipeline Stage" and "fc_collect_finalize" not in str(
                        r.get("display_title") or ""):
                    continue
                if best is None or t > best[0]:
                    best = (t, str(r.get("id")))
        return best[1] if best else None

    def add(self, run_id: str, reason: str, info: dict[str, Any] | None = None) -> RunRecord | None:
        run_id = str(run_id)
        if run_id in self.runs:
            return self.runs[run_id]
        info = info or self.run_info(run_id)
        if not info:
            return None
        if not trusted(info, self.repo, require_main=False):
            self.notes.append(f"run {run_id} skipped: not one of ours (event {info.get('event')})")
            return None
        rec = RunRecord(run_id=run_id, info=info, reason=reason)
        self.runs[run_id] = rec
        return rec

    def walk_chain(self, source_run_id: str) -> list[RunRecord]:
        chain: list[RunRecord] = []
        current, reason = str(source_run_id), "source"
        for _ in range(MAX_CHAIN_DEPTH):
            rec = self.add(current, reason)
            if rec is None:
                break
            chain.append(rec)
            rec.lineage = self.lineage_of(rec.run_id, rec.info)
            actor = (rec.info.get("actor") or {}).get("login") if isinstance(
                rec.info.get("actor"), dict) else None
            bot_dispatch = (rec.info.get("event") == "workflow_dispatch"
                            and actor == "github-actions[bot]")
            if rec.lineage is not None and (rec.lineage.get("trigger_run") or not bot_dispatch):
                nxt = rec.lineage.get("trigger_run")
                reason = "trigger (lineage.json)"
            elif rec.lineage is not None:
                # A link the Actions bot dispatched ("Continue the chain") was
                # given no trigger id; its upstream is found by the API.
                nxt = self.fallback_upstream(rec.info)
                reason = "upstream (Actions API: bot-dispatched link)"
            else:
                nxt = self.fallback_upstream(rec.info)
                reason = "upstream (Actions API fallback: no lineage.json)"
            if not nxt or str(nxt) in self.runs:
                break
            current = str(nxt)
        return chain

    def add_pipeline_stages(self) -> None:
        for rec in list(self.runs.values()):
            if rec.workflow != "Pythia Pipeline Stage":
                continue
            m = re.search(r"Pythia Pipeline Stage:\s*(\S+)\s*—", str(rec.info.get("display_title") or ""))
            if not m:
                continue
            pid = m.group(1)
            for r in self.workflow_runs("Pythia Pipeline Stage", per_page=100):
                if pid in str(r.get("display_title") or ""):
                    self.add(str(r.get("id")), f"stage of pipeline {pid}", r)

    def add_window(self, start: dt.datetime | None, end: dt.datetime | None,
                   window_days: int = WINDOW_DAYS) -> None:
        if start is None or end is None:
            return
        lo = start - dt.timedelta(days=window_days)
        for name in ("Ingest Structured Data", "Refresh CrisisWatch Data"):
            for r in self.workflow_runs(name, per_page=60):
                t = _ts(r.get("created_at"))
                if t is not None and lo <= t <= end:
                    self.add(str(r.get("id")), "time window", r)

    def add_canonical_sources(self) -> None:
        for rec in list(self.runs.values()):
            src = (rec.lineage or {}).get("canonical_db_run_downloaded")
            if src and str(src) not in self.runs:
                self.add(str(src), f"canonical DB run {rec.workflow} started from")

    # -- artifacts -------------------------------------------------------------

    def wanted(self, rec: RunRecord, name: str) -> bool:
        if name.startswith(DB_ARTIFACT_PREFIX):
            return False
        if name.startswith(f"{ARTIFACT_PREFIX}-"):
            return True
        return any(fnmatch.fnmatch(name, pat) for pat in ARTIFACT_PATTERNS.get(rec.workflow, ()))

    def download_all(self, ordered: list[RunRecord]) -> None:
        for i, rec in enumerate(ordered, start=1):
            rec.folder = f"{i:02d}_{rec.slug}__{rec.run_id}"
            folder = self.staging / rec.folder
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "api_run.json").write_text(json.dumps(_run_summary(rec.info), indent=2) + "\n",
                                                 encoding="utf-8")
            arts = self.artifacts_of(rec.run_id)
            rec.artifacts = arts
            names = {a.get("name") for a in arts}
            chain_name = f"{ARTIFACT_PREFIX}-{rec.slug}-{rec.run_id}"
            own = rec.reason == "this publish run"  # its diagnostics are built in this job
            if (not own and rec.workflow in SLUG_TO_NAME.values() and chain_name not in names
                    and rec.slug in _WRITERS):
                self.missing.append({"run_id": rec.run_id, "artifact": chain_name,
                                     "reason": "absent (the run predates chain diagnostics, or died before writing them)"})
            for a in arts:
                name = str(a.get("name") or "")
                if not self.wanted(rec, name):
                    continue
                if a.get("expired"):
                    self.missing.append({"run_id": rec.run_id, "artifact": name, "reason": "expired"})
                    continue
                cached = self._cache / rec.run_id
                dest = folder / name
                if name == chain_name and (cached / "lineage.json").is_file():
                    shutil.copytree(cached, dest, dirs_exist_ok=True)
                    continue
                ok, why = self.gh.download(rec.run_id, name, dest)
                if not ok:
                    self.missing.append({"run_id": rec.run_id, "artifact": name, "reason": why})
            for pat in ARTIFACT_PATTERNS.get(rec.workflow, ()):
                if own:
                    continue
                if not any(fnmatch.fnmatch(str(n), pat) for n in names):
                    self.missing.append({"run_id": rec.run_id, "artifact": pat, "reason": "absent"})


#: Workflows that write chain diagnostics (an absent artifact is worth a line).
_WRITERS = frozenset({
    "compute_resolutions", "compute_scores", "compute_calibration", "ingest_structured_data",
    "refresh_crisiswatch", "publish_latest_data",
})


def _run_summary(info: Mapping[str, Any]) -> dict[str, Any]:
    keys = ("id", "name", "display_title", "event", "status", "conclusion", "created_at",
            "updated_at", "run_started_at", "html_url", "head_branch", "head_sha", "run_attempt")
    out = {k: info.get(k) for k in keys}
    out["actor"] = (info.get("actor") or {}).get("login") if isinstance(info.get("actor"), dict) else None
    return out


def _duration(info: Mapping[str, Any]) -> str:
    a, b = _ts(info.get("run_started_at") or info.get("created_at")), _ts(info.get("updated_at"))
    if a is None or b is None:
        return ""
    minutes = (b - a).total_seconds() / 60
    return f"{minutes:.0f} min"


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def folder_issues(folder: Path) -> list[dict[str, Any]]:
    """Every issue in every ``issues.json`` under a run's folder (not inside zips)."""
    found: list[dict[str, Any]] = []
    for path in sorted(folder.rglob("issues.json")):
        payload = _read_json(path) or {}
        for raw in payload.get("issues") or []:
            if isinstance(raw, dict):
                item = dict(raw)
                item["_file"] = path.relative_to(folder.parent).as_posix()
                found.append(item)
    return found


def _rank(issue: Mapping[str, Any]) -> int:
    sev = str(issue.get("severity") or "")
    return SEVERITY_ORDER.index(sev) if sev in SEVERITY_ORDER else len(SEVERITY_ORDER)


def merge_issues(per_run: Iterable[tuple[RunRecord, list[dict[str, Any]]]]) -> list[dict[str, Any]]:
    """All issues, de-duplicated by id (worst severity kept, runs listed), worst first."""
    merged: dict[str, dict[str, Any]] = {}
    for rec, issues in per_run:
        for issue in issues:
            iid = str(issue.get("id") or issue.get("title") or "?")
            cur = merged.get(iid)
            if cur is None:
                cur = dict(issue)
                cur["runs"] = []
                merged[iid] = cur
            elif _rank(issue) < _rank(cur):
                runs = cur["runs"]
                cur.update(issue)
                cur["runs"] = runs
            label = f"{rec.slug} {rec.run_id}"
            if label not in cur["runs"]:
                cur["runs"].append(label)
    return sorted(merged.values(), key=lambda i: (_rank(i), str(i.get("id"))))


def _cell(text: Any) -> str:
    return str(text if text not in (None, "") else "—").replace("|", "/").replace("\n", " ")


def cycle_summary(ordered: list[RunRecord], staging: Path, repo: str) -> tuple[str, list[dict[str, Any]]]:
    per_run = [(rec, folder_issues(staging / rec.folder)) for rec in ordered]
    lines = ["# Cycle summary", "",
             "| workflow | run | conclusion | did work | DB downloaded from | DB uploaded | worst issue |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for rec, issues in per_run:
        chain_run = _read_json(staging / rec.folder / f"{ARTIFACT_PREFIX}-{rec.slug}-{rec.run_id}" / "run.json") or {}
        lineage = rec.lineage or {}
        uploaded = lineage.get("canonical_db_run_uploaded")
        if uploaded is None:
            uploaded = "yes" if any(str(a.get("name")) == DB_ARTIFACT_PREFIX for a in rec.artifacts) else "no"
        else:
            uploaded = "yes"
        worst = min(issues, key=_rank) if issues else None
        url = rec.info.get("html_url") or f"https://github.com/{repo}/actions/runs/{rec.run_id}"
        lines.append(
            f"| {_cell(rec.workflow)} | [{rec.run_id}]({url}) | {_cell(rec.info.get('conclusion') or rec.info.get('status'))} "
            f"| {_cell(chain_run.get('trigger_did_work'))} | {_cell(lineage.get('canonical_db_run_downloaded'))} "
            f"| {uploaded} | {_cell((worst or {}).get('severity', '') + ': ' + (worst or {}).get('title', '') if worst else '')} |"
        )
    merged = merge_issues(per_run)
    lines += ["", "## Every issue, worst first", ""]
    if not merged:
        lines.append("No run in this cycle recorded an issue.")
    for issue in merged:
        lines.append(f"- **{issue.get('severity')}** `{issue.get('id')}`: {issue.get('title')} "
                     f"(runs: {', '.join(issue.get('runs') or [])})")
    return "\n".join(lines) + "\n", merged


def readme(ordered: list[RunRecord], collector: Collector, publish_run_id: str, repo: str,
           merged: list[dict[str, Any]]) -> str:
    chain = [r for r in ordered if r.reason == "source" or r.reason.startswith(("trigger", "upstream"))]
    lines = ["# Cycle diagnostics", "",
             f"Built by Publish run [{publish_run_id}](https://github.com/{repo}/actions/runs/{publish_run_id}) "
             f"on {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}.", "",
             "## What the cycle was", ""]
    if chain:
        source = next(r for r in chain if r.reason == "source") if any(
            r.reason == "source" for r in chain) else chain[-1]
        root = chain[0]
        lines.append(f"The published DB came from {source.workflow} run {source.run_id}; "
                     f"following each run's trigger back reaches {root.workflow} run {root.run_id} "
                     f"({root.created_at}).")
    lines += ["", "## Runs included", "",
              "| folder | workflow | run | conclusion | duration | why it is here |",
              "| --- | --- | --- | --- | --- | --- |"]
    for rec in ordered:
        url = rec.info.get("html_url") or f"https://github.com/{repo}/actions/runs/{rec.run_id}"
        lines.append(f"| `{rec.folder}` | {_cell(rec.workflow)} | [{rec.run_id}]({url}) | "
                     f"{_cell(rec.info.get('conclusion') or rec.info.get('status'))} | "
                     f"{_cell(_duration(rec.info))} | {_cell(rec.reason)} |")
    lines += ["", "## Where to look first", ""]
    worst = [i for i in merged if i.get("severity") in ("blocking", "degraded")][:5]
    if worst:
        for issue in worst:
            lines.append(f"- {issue.get('severity')}: {issue.get('title')}: see `{issue.get('_file', '')}`")
    else:
        lines.append("- Nothing blocking or degraded. Start with `CYCLE_SUMMARY.md`, then each "
                     "folder's `summary.md`.")
    lines.append("- `CYCLE_SUMMARY.md` has one row per run and every issue, worst first.")
    if collector.missing:
        lines += ["", "## Artifacts that could not be fetched", ""]
        for m in collector.missing:
            lines.append(f"- run {m.get('run_id')}: `{m.get('artifact') or '(listing)'}`: {m.get('reason')}")
    if collector.notes:
        lines += ["", "## Notes", ""] + [f"- {n}" for n in collector.notes]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Assembly: redaction, size, zip
# ---------------------------------------------------------------------------


class SecretLeak(RuntimeError):
    def __init__(self, hits: dict[str, list[str]]):
        super().__init__("redaction scan found secret values in: " + ", ".join(sorted(hits)))
        self.hits = hits


def _files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and ".lineage_cache" not in p.parts)


def scan_for_secrets(staging: Path, env: Mapping[str, str]) -> None:
    """Every file checked for every credential value in the environment (fingerprints only)."""
    from resolver.diagnostics.redaction import find_secrets, secret_values

    secrets = secret_values(env)
    if not secrets:
        return
    hits: dict[str, list[str]] = {}
    for path in _files(staging):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            continue
        found = find_secrets(text, secrets)
        if found:
            hits[path.relative_to(staging).as_posix()] = found
    if hits:
        raise SecretLeak(hits)


def zip_tree(staging: Path, out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in _files(staging):
            zf.write(path, path.relative_to(staging).as_posix())
    return out_path.stat().st_size


def _protected(path: Path) -> bool:
    return path.name in PROTECTED_NAMES


def _drop_stage_candidates(staging: Path, stage: str, folders: Mapping[str, str]) -> list[Path]:
    out: list[Path] = []
    for path in _files(staging):
        if _protected(path):
            continue
        rel = path.relative_to(staging).as_posix().lower()
        parts = rel.split("/")
        if stage == "workflow_logs" and any(("workflow_logs" in p or "workflow-logs" in p) for p in parts):
            out.append(path)
        elif stage == "code_snapshots" and any(p in ("code_snapshot", "code") for p in parts[:-1]):
            out.append(path)
        elif stage == "sibyl_nested_zips" and path.suffix == ".zip" and folders.get(parts[0]) == "sibyl":
            out.append(path)
    return sorted(out, key=lambda p: p.stat().st_size, reverse=True)


def enforce_size(staging: Path, out_path: Path, max_bytes: int,
                 folders: Mapping[str, str]) -> tuple[int, list[dict[str, Any]]]:
    """Drop in the stated order until the zip fits; every drop recorded."""
    dropped: list[dict[str, Any]] = []
    size = zip_tree(staging, out_path)
    for stage in ("workflow_logs", "code_snapshots", "sibyl_nested_zips"):
        if size <= max_bytes:
            return size, dropped
        for path in _drop_stage_candidates(staging, stage, folders):
            dropped.append({"path": path.relative_to(staging).as_posix(), "stage": stage,
                            "original_bytes": path.stat().st_size, "action": "dropped"})
            path.unlink()
        size = zip_tree(staging, out_path)
    while size > max_bytes:
        csvs = sorted((p for p in _files(staging) if p.suffix == ".csv" and not _protected(p)
                       and not any(d["path"] == p.relative_to(staging).as_posix() for d in dropped)),
                      key=lambda p: p.stat().st_size, reverse=True)
        if not csvs:
            break
        path = csvs[0]
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        original = path.stat().st_size
        if len(lines) > 2001:
            kept = lines[:1001] + [f"# ... {len(lines) - 2001} line(s) cut by the size ceiling ..."] + lines[-1000:]
        else:
            kept = lines[:200]
        path.write_text("\n".join(kept) + "\n", encoding="utf-8")
        dropped.append({"path": path.relative_to(staging).as_posix(), "stage": "largest_csv",
                        "original_bytes": original, "action": "cut to head and tail"})
        size = zip_tree(staging, out_path)
    return size, dropped


def manifest_files(staging: Path, folder_runs: Mapping[str, RunRecord]) -> list[dict[str, Any]]:
    rows = []
    for path in _files(staging):
        rel = path.relative_to(staging).as_posix()
        parts = rel.split("/")
        rec = folder_runs.get(parts[0])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append({
            "path": rel,
            "source_run": rec.run_id if rec else None,
            "workflow": rec.workflow if rec else None,
            "artifact": parts[1] if rec and len(parts) > 2 else None,
            "bytes": path.stat().st_size,
            "sha256": digest,
        })
    return rows


def build(*, gh: Any, repo: str, source_run_id: str, publish_run_id: str, out_dir: Path,
          own_dir: Path | None = None, max_bytes: int = DEFAULT_MAX_BYTES,
          wait_sec: float = 600, env: Mapping[str, str] | None = None,
          staging: Path | None = None, today: dt.date | None = None,
          sleep: Callable[[float], None] = time.sleep) -> dict[str, Any]:
    env = dict(os.environ if env is None else env)
    tmp = staging or Path(tempfile.mkdtemp(prefix="cycle_diag_"))
    tmp.mkdir(parents=True, exist_ok=True)
    collector = Collector(gh, repo, tmp, wait_sec=wait_sec, sleep=sleep)
    chain = collector.walk_chain(source_run_id)
    collector.add_pipeline_stages()
    root = chain[-1] if chain else None
    publish_info = collector.run_info(publish_run_id) if publish_run_id else None
    end = _ts((publish_info or {}).get("created_at")) or dt.datetime.now(dt.timezone.utc)
    collector.add_window(_ts(root.created_at) if root else None, end)
    collector.add_canonical_sources()
    ordered = sorted(collector.runs.values(), key=lambda r: (r.created_at, r.run_id))
    if publish_info and str(publish_run_id) not in collector.runs:
        own = RunRecord(run_id=str(publish_run_id), info=publish_info, reason="this publish run")
        ordered.append(own)
    collector.download_all(ordered)
    if own_dir is not None and own_dir.is_dir() and ordered and ordered[-1].run_id == str(publish_run_id):
        shutil.copytree(own_dir, tmp / ordered[-1].folder / f"{ARTIFACT_PREFIX}-{ordered[-1].slug}-{publish_run_id}",
                        dirs_exist_ok=True)
        ordered[-1].lineage = _read_json(own_dir / "lineage.json")
    summary, merged = cycle_summary(ordered, tmp, repo)
    (tmp / "CYCLE_SUMMARY.md").write_text(summary, encoding="utf-8")
    (tmp / "README.md").write_text(readme(ordered, collector, publish_run_id, repo, merged),
                                   encoding="utf-8")
    shutil.rmtree(tmp / ".lineage_cache", ignore_errors=True)
    stamp = (today or dt.date.today()).isoformat()
    out_path = out_dir / f"cycle_diagnostics__{stamp}__{publish_run_id or 'local'}.zip"
    folders = {r.folder: r.slug for r in ordered}
    manifest: dict[str, Any] = {
        "built_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "repository": repo,
        "source_run": source_run_id,
        "publish_run": publish_run_id,
        "runs": [{"folder": r.folder, "run_id": r.run_id, "workflow": r.workflow, "reason": r.reason,
                  "conclusion": r.info.get("conclusion"), "created_at": r.created_at,
                  "lineage": r.lineage} for r in ordered],
        "missing": collector.missing,
        "notes": collector.notes,
        "max_bytes": max_bytes,
    }
    # Size first, then the manifest (it names the drops), then the scan over
    # everything that will be in the zip, then the final zip.
    (tmp / "manifest.json").write_text("{}\n", encoding="utf-8")
    size, dropped = enforce_size(tmp, out_path, max_bytes, folders)
    manifest["dropped"] = dropped
    manifest["files"] = manifest_files(tmp, {r.folder: r for r in ordered})
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    try:
        scan_for_secrets(tmp, env)
    except SecretLeak:
        if out_path.exists():
            out_path.unlink()
        if staging is None:
            shutil.rmtree(tmp, ignore_errors=True)
        raise
    manifest["compressed_bytes"] = zip_tree(tmp, out_path)
    manifest["zip"] = str(out_path)
    manifest["summary_md"] = summary
    if staging is None:
        shutil.rmtree(tmp, ignore_errors=True)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-run-id", default=os.environ.get("SOURCE_RUN_ID", ""))
    parser.add_argument("--publish-run-id", default=os.environ.get("GITHUB_RUN_ID", ""))
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", ""))
    parser.add_argument("--own-dir", default="", help="this publish run's chain-diagnostics directory")
    parser.add_argument("--out-dir", default="cycle_out")
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    parser.add_argument("--wait-sec", type=float, default=600)
    args = parser.parse_args(argv)
    if not args.source_run_id or not args.repo:
        print("::warning title=Cycle diagnostics::no source run or repository; nothing to gather")
        return 0
    try:
        manifest = build(gh=Gh(args.repo), repo=args.repo, source_run_id=args.source_run_id,
                         publish_run_id=args.publish_run_id, out_dir=Path(args.out_dir),
                         own_dir=Path(args.own_dir) if args.own_dir else None,
                         max_bytes=args.max_bytes, wait_sec=args.wait_sec)
    except SecretLeak as exc:
        print(f"::error title=Cycle diagnostics redaction failure::{exc}")
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"::warning title=Cycle diagnostics failed::{type(exc).__name__}: {exc}")
        return 1
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        try:
            with open(summary, "a", encoding="utf-8") as fh:
                fh.write(manifest["summary_md"] + "\n")
        except OSError:
            pass
    print(manifest["summary_md"])
    print(f"Wrote {manifest['zip']} ({manifest['compressed_bytes']} bytes, "
          f"{len(manifest['files'])} files, {len(manifest['missing'])} missing, "
          f"{len(manifest['dropped'])} dropped)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
