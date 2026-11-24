# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The forecast pipeline review's inputs, read out of a forecast chain's staging tree.

The forecast pipeline review is written by a scheduled Claude session after
every forecast chain, following ``forecast_review_template.md`` in Drive. It
reads Drive through the connector, which reads text only, cannot open a zip
and refuses any file over 10 MB. The 13 November 2026 chain zip was 39 MB and
its debug bundle a 20 MB zip inside it. So the Drive archive writes
``forecast_review/`` beside each forecast chain zip, and this module builds the
file that folder leads with, ``forecast_review_inputs.json``: one key per
section of the template's Part B, each count naming the file it came from in a
``source`` field beside it, and the figures the review needs from files too
large to lift (``llm_calls_detail``, ``spd_tables``) already computed.

A pure function of the staging tree (``build_cycle_diagnostics`` lays it out;
an unpacked chain zip has the same shape) plus, optionally, the previous
production forecast chain's own ``forecast_review_inputs.json``. No DB, no
network, stdlib only. Every section degrades to
``{"stub": true, "reason": ..., "fallback_file": ...}``; nothing raises.

Run alone::

    python -m scripts.ci.forecast_review_inputs --staging DIR --out FILE \
        [--previous PREVIOUS_INPUTS.json]

The keys, the caps and the backfill dispatch: ``docs/forecast_review_inputs.md``.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import gzip
import json
import math
import re
import shutil
import statistics
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - import convenience
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ci import resolver_review_inputs as rri  # noqa: E402

INPUTS_VERSION = 1

DEBUG_BUNDLE_GLOB = "pythia_debug_bundle__*.zip"
ATTRIBUTION_BUNDLE_GLOB = "forecast_attribution__*.zip"
CURRENT_RUN_BUNDLE_GLOB = "current_run_analysis__*.zip"
#: Written by the archive job into the staging tree before this module runs.
WORKFLOW_LOGS_INDEX = "workflow_logs_index.json"
#: Commits between this run's commit and the previous run's that touched a
#: prompt file, written by the archive job (``drive_archive.write_prompt_commits``).
PROMPT_COMMITS = "prompt_commits.json"
#: Paths whose edits change what the forecaster is shown. A trailing slash is a
#: directory prefix.
PROMPT_PATHS = ("forecaster/prompts.py", "forecaster/hazard_prompts.py", "pythia/prompts/",
                "forecaster/history_loaders.py", "forecaster/binary_prompts.py",
                "forecaster/gdacs_history.py")


def touches_prompts(filenames: Iterable[str]) -> list[str]:
    """The filenames among ``filenames`` that are prompt files."""
    hits = []
    for f in filenames:
        for p in PROMPT_PATHS:
            if (p.endswith("/") and f.startswith(p)) or f == p:
                hits.append(f)
                break
    return hits


def run_commit(staging: Path) -> str | None:
    """The commit the forecast ran at: the newest head_sha of a pipeline stage
    or Sibyl run folder (the debug bundle's own record is read later, by
    ``section_meta``)."""
    commit = None
    for slug, folder in rri._run_folders(staging):
        if slug in ("pipeline_stage", "sibyl"):
            commit = (_read_json(folder / "api_run.json") or {}).get("head_sha") or commit
    return commit

#: Section keys, in the template's order. ``previous_run`` embeds these of the
#: previous production chain.
SECTION_KEYS = (
    "meta", "chain", "issues", "inputs", "horizon_scanner", "triage_questions",
    "evidence", "ensemble", "track2", "experiments", "sibyl", "outputs", "cost",
    "run_over_run", "known_issues",
)

#: Marks the adversarial block in an SPD prompt (``pythia.adversarial_check``).
ADVERSARIAL_MARKER = "ADVERSARIAL EVIDENCE CHECK"
#: Aggregate and derived model names that are not ensemble members.
NON_MEMBER = ("ensemble_mean_v2", "ensemble_bayesmc_v2", "track2_flash", "sibyl")
TRACK2_MODEL_NAME = "track2_flash"

#: Design shares of the experiments when the stage context does not say
#: (the production workflows' settings, CLAUDE.md "Per-horizon architecture").
DEFAULT_SHARES = {
    "advice": ("PYTHIA_ADVICE_EXPERIMENT_SHARE", 0.5),
    "rc_shift": ("PYTHIA_RC_SHIFT_SHARE", 0.5),
    "adversarial": ("PYTHIA_ADVERSARIAL_WITHHELD_SHARE", 0.5),
}

#: Words that mark a health check passing because its subject is off (A4.4).
NARROW_OK_WORDS = ("disabled", "skipped", "absent", "not run", "off")

stub = rri.stub
_read_json = rri._read_json
read_csv = rri.read_csv
iter_csv = rri.iter_csv
md_tables = rri.md_tables
_num = rri._num
_ts = rri._ts


def stub_with(reason: str, fallback: str | None = None) -> dict[str, Any]:
    out = stub(reason)
    if fallback:
        out["fallback_file"] = fallback
    return out


def _median(values: Iterable[float | None]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return round(statistics.median(vals), 4) if vals else None


def _rel(path: Path | None, root: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.relative_to(root).as_posix() if root else path.name
    except ValueError:
        return path.name


# ---------------------------------------------------------------------------
# Where things are
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Sources:
    """Paths into a forecast chain's staging tree, nested zips unpacked."""

    staging: Path
    run_folders: list[tuple[str, Path]]     # (slug, folder) in chain order
    debug_zip: Path | None
    debug: Path | None                      # debug bundle, unpacked
    attribution_zip: Path | None
    attribution: Path | None
    current_zip: Path | None
    current: Path | None

    def folders(self, slug: str) -> list[Path]:
        return [p for s, p in self.run_folders if s == slug]

    def find(self, pattern: str) -> list[Path]:
        """Files matching ``pattern`` anywhere in the staging tree (not inside the unpacked bundles)."""
        return sorted(p for p in self.staging.rglob(pattern) if p.is_file())

    def first(self, pattern: str) -> Path | None:
        found = self.find(pattern)
        return found[-1] if found else None

    def dbg(self, stem: str) -> Path | None:
        """``<stem>__<run>.<ext>`` at the debug bundle's top, else ``stem`` itself."""
        if self.debug is None:
            return None
        direct = self.debug / stem
        if direct.is_file():
            return direct
        found = sorted(self.debug.glob(f"{stem}__*"))
        return found[0] if found else None

    def att(self, rel: str) -> Path | None:
        if self.attribution is None:
            return None
        p = self.attribution / rel
        return p if p.is_file() else None

    def cur(self, rel: str) -> Path | None:
        if self.current is None:
            return None
        p = self.current / rel
        return p if p.is_file() else None

    @property
    # The bundle a source path is named after in the JSON.
    def label(self) -> dict[str, str | None]:
        return {"debug_bundle": self.debug_zip.name if self.debug_zip else None,
                "attribution_bundle": self.attribution_zip.name if self.attribution_zip else None,
                "current_run_bundle": self.current_zip.name if self.current_zip else None}


def _unpack(zip_path: Path | None, dest: Path) -> Path | None:
    if zip_path is None:
        return None
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    try:
        with zipfile.ZipFile(zip_path) as zf:
            rri._safe_extract(zf, dest)
    except (zipfile.BadZipFile, OSError):
        return None
    entries = list(dest.iterdir())
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return dest


def _newest(staging: Path, pattern: str) -> Path | None:
    found = sorted((p for p in staging.rglob(pattern) if p.is_file() and "__part" not in p.name),
                   key=lambda p: p.name)
    return found[-1] if found else None


def locate(staging: Path, work: Path) -> Sources:
    """Find the run folders and unpack the three nested bundles into ``work``."""
    folders = rri._run_folders(staging)
    debug_zip = _newest(staging, DEBUG_BUNDLE_GLOB)
    att_zip = _newest(staging, ATTRIBUTION_BUNDLE_GLOB)
    cur_zip = _newest(staging, CURRENT_RUN_BUNDLE_GLOB)
    work.mkdir(parents=True, exist_ok=True)
    return Sources(
        staging=staging, run_folders=folders,
        debug_zip=debug_zip, debug=_unpack(debug_zip, work / "debug_bundle"),
        attribution_zip=att_zip, attribution=_unpack(att_zip, work / "attribution_bundle"),
        current_zip=cur_zip, current=_unpack(cur_zip, work / "current_run_bundle"),
    )


# ---------------------------------------------------------------------------
# Shared readers
# ---------------------------------------------------------------------------


def _health(src: Sources) -> dict[str, Any]:
    return _read_json(src.dbg("health_report")) or _read_json(src.first("health_report__*.json")) or {}


def _health_checks(src: Sources) -> list[dict[str, Any]]:
    return list(_health(src).get("health_checks") or [])


def _check(src: Sources, subsystem: str) -> dict[str, Any] | None:
    for c in _health_checks(src):
        if str(c.get("subsystem")) == subsystem:
            return c
    return None


def _question_metrics(src: Sources) -> list[dict[str, str]]:
    return read_csv(src.dbg("question_metrics") or src.first("question_metrics__*.csv"))


def _stage_health(src: Sources) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for p in src.find("stage_health__*.json"):
        payload = _read_json(p) or {}
        stage = payload.get("stage") or p.stem.split("__", 1)[-1]
        out[str(stage)] = payload | {"_file": _rel(p, src.staging)}
    return out


def _stage_context_env(src: Sources) -> dict[str, str]:
    for p in [src.debug / "stage_context" / "stage_context.json" if src.debug else None,
              *src.find("stage_context.json")]:
        payload = _read_json(p) if p else None
        if not payload:
            continue
        env = payload.get("env_snapshot") or {}
        env = env.get("env", env) if isinstance(env, dict) else {}
        if env:
            return {str(k): str(v) for k, v in env.items()}
    return {}


def _llm_calls(src: Sources) -> Iterable[dict[str, Any]]:
    path = src.dbg("llm_calls_detail")
    if path is None:
        return iter(())

    def gen() -> Iterable[dict[str, Any]]:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    return gen()


def _usage(row: Mapping[str, Any]) -> dict[str, Any]:
    raw = row.get("usage_json")
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw or "{}") or {}
    except (TypeError, json.JSONDecodeError):
        return {}


def _int(value: Any) -> int | None:
    n = _num(value)
    return int(n) if n is not None else None


def _qkey(row: Mapping[str, Any]) -> str:
    return f"{str(row.get('iso3') or '').upper()}|{str(row.get('hazard_code') or '').upper()}|" \
           f"{str(row.get('metric') or '').upper()}"


# ---------------------------------------------------------------------------
# meta (header)
# ---------------------------------------------------------------------------


def section_meta(src: Sources) -> dict[str, Any]:
    manifest = _read_json(src.staging / "manifest.json") or {}
    chain = manifest.get("chain") or {}
    publish = manifest.get("publish_run") or manifest.get("source_run")
    commit = None
    for slug, folder in src.run_folders:
        if slug in ("pipeline_stage", "sibyl"):
            commit = (_read_json(folder / "api_run.json") or {}).get("head_sha") or commit
    env_cfg = _read_json(src.dbg("env_and_config")) or {}
    commit = ((env_cfg.get("git") or {}).get("commit_sha")) or commit
    qm = _question_metrics(src)
    starts = sorted({r.get("window_start_date", "")[:7] for r in qm if r.get("window_start_date")})
    ends = sorted({r.get("horizon_end_month", "") for r in qm if r.get("horizon_end_month")})
    sibyl_run = ((_stage_health(src).get("sibyl") or {}).get("sibyl") or {}).get("sibyl_run_id")
    runs = [{"slug": s, "run_id": rri._FOLDER_RE.match(p.name).group("run")} for s, p in src.run_folders]
    return {
        "chain_folder": manifest.get("zip_name") or None,
        "kind": chain.get("kind"), "start_date": chain.get("start_date"),
        "started_at": chain.get("started_at"),
        "hs_run_id": chain.get("hs_run_id"), "fc_run_id": chain.get("fc_run_id"),
        "pipeline_id": chain.get("pipeline_id"),
        "sibyl_run_id": sibyl_run, "sibyl_gh_run_id": chain.get("sibyl_run_id"),
        "publish_run_id": str(publish.get("run_id") if isinstance(publish, dict) else publish or "") or None,
        "is_test": chain.get("is_test"),
        "commit": commit,
        "forecast_window": {"first_month": starts[0] if starts else None,
                            "last_month": ends[-1] if ends else None},
        "gh_runs": runs,
        "inputs_version": INPUTS_VERSION,
        "source": "manifest.json chain; env_and_config git; question_metrics window",
    }


# ---------------------------------------------------------------------------
# 2. chain
# ---------------------------------------------------------------------------


def _duration_s(api: Mapping[str, Any]) -> float | None:
    start, end = _ts(api.get("run_started_at") or api.get("created_at")), _ts(api.get("updated_at"))
    return round((end - start).total_seconds(), 1) if start and end else None


def _stage_name(api: Mapping[str, Any]) -> str | None:
    title = str(api.get("display_title") or api.get("name") or "")
    m = re.search(r"—\s*([a-z_]+)\s*$", title)
    return m.group(1) if m else None


def _poller_summary(src: Sources) -> dict[str, Any]:
    """Ticks, and ticks that found work, from the poller logs the debug bundle
    holds, and from the chain's own poller run folders (every tick between the
    first stage and Sibyl, Nov 2026), which outrank the logs."""
    logs: list[Path] = []
    if src.debug is not None and (src.debug / "workflow_logs").is_dir():
        logs = sorted((src.debug / "workflow_logs").glob("poll_llm_batches__*.txt"))
    folders = src.folders("poll_llm_batches")
    index = _read_json(src.debug / "workflow_logs" / "INDEX.json") if src.debug else None
    capped = any("capped" in str(p) for p in (index or {}).get("problems") or [])
    ticks = max(len(logs), len(folders))
    with_work = 0
    empty_logs = 0
    for p in logs:
        text = p.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            empty_logs += 1
            continue
        if re.search(r"=dispatched|— dispatched|Dispatched |dispatching stage", text):
            with_work += 1
    folder_work = 0
    for folder in folders:
        d = _read_json(next(iter(sorted(folder.rglob("poll_decision.json"))), None)) or {}
        for p in d.get("pipelines") or []:
            if str(p.get("action") or "").startswith("dispatch"):
                folder_work += 1
                break
    if folders:
        # The chain's own poller runs (every tick, Nov 2026) outrank the debug
        # bundle's logs, which stop at 40.
        with_work = folder_work
    lifecycle = _read_json(src.dbg("batch_lifecycle")) or {}
    by_provider: dict[str, list[tuple[Any, Any]]] = defaultdict(list)
    for b in lifecycle.get("batches") or []:
        by_provider[str(b.get("provider") or "?")].append(
            (_ts(b.get("submitted_at")), _ts(b.get("ended_at") or b.get("completed_at"))))
    submit_to_last_end = {}
    for prov, pairs in sorted(by_provider.items()):
        subs = [a for a, _ in pairs if a]
        ends = [b for _, b in pairs if b]
        submit_to_last_end[prov] = (round((max(ends) - min(subs)).total_seconds(), 1)
                                    if subs and ends else None)
    return {"ticks": ticks, "ticks_with_a_dispatch": with_work, "empty_logs": empty_logs,
            "submit_to_last_batch_ended_s": submit_to_last_end,
            "submit_to_last_batch_ended_source": "debug bundle batch_lifecycle (first submitted_at to "
                                                 "last ended_at, per provider)",
            "poller_run_folders": len(folders), "debug_bundle_poller_logs": len(logs),
            "capped_at_40_in_debug_bundle": capped or len(logs) == 40,
            "source": "debug bundle workflow_logs/poll_llm_batches__*.txt"
                      + ("; poll_llm_batches run folders" if folders else "")}


def _cell(text: Any) -> str | None:
    """A CYCLE_SUMMARY cell; a dash means the run wrote no chain diagnostics."""
    t = str(text or "").strip()
    return None if t in ("", "—", "-") else t


def section_chain(src: Sources) -> dict[str, Any]:
    manifest = _read_json(src.staging / "manifest.json") or {}
    summary_rows: dict[str, dict[str, str]] = {}
    text = (src.staging / "CYCLE_SUMMARY.md").read_text(encoding="utf-8") \
        if (src.staging / "CYCLE_SUMMARY.md").is_file() else ""
    for table in md_tables(text):
        for row in table["rows"]:
            m = re.search(r"\[(\d+)\]", row.get("run", ""))
            if m:
                summary_rows[m.group(1)] = row
    runs = []
    starts, ends = [], []
    for slug, folder in src.run_folders:
        run_id = rri._FOLDER_RE.match(folder.name).group("run")
        api = _read_json(folder / "api_run.json") or {}
        row = summary_rows.get(run_id, {})
        start = api.get("run_started_at") or api.get("created_at")
        if _ts(start):
            starts.append(_ts(start))
        if _ts(api.get("updated_at")):
            ends.append(_ts(api.get("updated_at")))
        runs.append({
            "workflow": api.get("name", "").split(":")[0] if api.get("name") else row.get("workflow"),
            "slug": slug, "run_id": run_id, "stage": _stage_name(api),
            "started": start, "duration_s": _duration_s(api),
            "conclusion": api.get("conclusion") or row.get("conclusion"),
            "did_work": _cell(row.get("did work")),
            "db_downloaded_from": _cell(row.get("DB downloaded from")),
            "db_uploaded": _cell(row.get("DB uploaded")),
            "worst_issue": _cell(row.get("worst issue")),
        })
    lifecycle = _read_json(src.dbg("batch_lifecycle")) or {}
    queue = [b.get("queue_wall_seconds") for b in lifecycle.get("batches") or []
             if b.get("queue_wall_seconds") is not None]
    publish = [r for r in runs if r["slug"] == "publish_latest_data"]
    release_end = None
    if publish:
        folder = dict((r["run_id"], p) for (s, p), r in zip(src.run_folders, runs)).get(publish[-1]["run_id"])
        release_end = _ts((_read_json(folder / "api_run.json") or {}).get("updated_at")) if folder else None
    pipeline_starts = [_ts(r["started"]) for r in runs if r["slug"] in ("pipeline_stage", "poll_llm_batches")
                       and _ts(r["started"])]
    t0 = min(pipeline_starts) if pipeline_starts else (min(starts) if starts else None)
    t1 = release_end or (max(ends) if ends else None)
    logs_index = _read_json(src.staging / WORKFLOW_LOGS_INDEX)
    if logs_index is None:
        errored = stub_with("workflow_logs_index.json not in this chain (written by the archive job)",
                            "the chain zip's workflow logs")
    else:
        errored = []
        for run in logs_index.get("runs") or []:
            for step in run.get("steps") or []:
                if step.get("conclusion") == "success" and (step.get("n_traceback") or step.get("n_error")):
                    errored.append({"run_id": run.get("run_id"), "workflow": run.get("workflow"),
                                    "job": step.get("job"), "step": step.get("name"),
                                    "continue_on_error": step.get("continue_on_error"),
                                    "n_traceback": step.get("n_traceback"), "n_error": step.get("n_error"),
                                    "first_traceback": step.get("first_traceback"),
                                    "first_error": step.get("first_error")})
    return {
        "runs": runs, "source_runs": "api_run.json per run folder; CYCLE_SUMMARY.md "
                                     "(did_work null: the run wrote no chain diagnostics)",
        "chain_start_to_release_s": round((t1 - t0).total_seconds(), 1) if t0 and t1 else None,
        "chain_start_to_release_note": "first pipeline stage (or first run) start to Publish's last update",
        "batch_queue_s": {"n_batches": len(queue), "median": _median(queue),
                          "max": max(queue) if queue else None,
                          "source": "debug bundle batch_lifecycle (queue_wall_seconds)"},
        "poller": _poller_summary(src),
        "missing_artifacts": manifest.get("missing") or [],
        "missing_source": "manifest.json missing",
        "steps_succeeded_with_traceback_or_error": errored,
        "steps_source": WORKFLOW_LOGS_INDEX,
    }


# ---------------------------------------------------------------------------
# 3. issues
# ---------------------------------------------------------------------------


def _known_ids() -> list[str]:
    path = REPO_ROOT / "resolver" / "config" / "known_issues.yml"
    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return [str(e.get("id")) for e in data.get("issues") or [] if e.get("id")]
    except Exception:  # noqa: BLE001
        ids = re.findall(r"^\s*-\s*id:\s*([A-Za-z0-9_]+)", path.read_text(encoding="utf-8")
                         if path.is_file() else "", re.M)
        return ids


def _is_known(text: str, ids: list[str]) -> bool:
    low = text.lower()
    if any(i.lower() in low for i in ids):
        return True
    # The three acknowledged external issues by their words (template A4.13).
    return any(w in low for w in ("em-dat", "emdat", "acled cast", "views vintage", "views_stale"))


def section_issues(src: Sources) -> dict[str, Any]:
    ids = _known_ids()
    items: list[dict[str, Any]] = []
    anom = _read_json(src.dbg("anomalies")) or {}
    for a in anom.get("anomalies") or []:
        detail = str(a.get("description") or "")
        items.append({"severity": a.get("severity"), "subsystem": a.get("subsystem"), "detail": detail,
                      "evidence_file": a.get("evidence_file"), "known": _is_known(detail, ids),
                      "source": "debug bundle anomalies"})
    checks = _health_checks(src)
    seen = {str(i["detail"]) for i in items}
    for c in checks:
        status = str(c.get("status") or "").upper()
        detail = str(c.get("detail") or "")
        if status in ("OK", "N/A") or detail in seen:
            continue
        items.append({"severity": status.lower(), "subsystem": c.get("subsystem"), "detail": detail,
                      "known": _is_known(detail, ids), "source": "debug bundle health_report health_checks"})
    for slug, folder in src.run_folders:
        for p in sorted(folder.glob("chain-diagnostics-*/issues.json")):
            for it in (_read_json(p) or {}).get("issues") or []:
                detail = str(it.get("title") or it.get("id") or "")
                items.append({"severity": it.get("severity"), "subsystem": f"{slug}:{it.get('id')}",
                              "detail": detail, "known": it.get("severity") == "known"
                              or _is_known(str(it.get("id")) + " " + detail, ids),
                              "source": _rel(p, src.staging)})
    counts = Counter(str(i["severity"]).lower() for i in items)
    narrow = []
    for c in checks:
        detail = str(c.get("detail") or "").lower()
        if str(c.get("status") or "").upper() == "OK" and any(w in detail for w in NARROW_OK_WORDS):
            narrow.append({"subsystem": c.get("subsystem"), "detail": c.get("detail")})
    return {"items": items, "counts_by_severity": dict(counts),
            "health_checks": [{"subsystem": c.get("subsystem"), "status": c.get("status"),
                               "detail": c.get("detail")} for c in checks],
            "health_checks_source": "debug bundle health_report health_checks",
            "narrow_ok_checks": narrow,
            "narrow_ok_rule": "an OK whose detail says " + ", ".join(NARROW_OK_WORDS) + " (rule A4.4)",
            "known_issue_ids": ids}


# ---------------------------------------------------------------------------
# 4. inputs
# ---------------------------------------------------------------------------


def _bool(value: Any) -> bool:
    return str(value or "").strip().lower() in ("true", "1", "yes")


#: inject -> (which questions should carry it, how the per-country inventory says it did).
INJECT_RULES: dict[str, tuple[Callable[[Mapping[str, str]], bool], Callable[[Mapping[str, str]], bool]]] = {
    "conflict_forecasts": (lambda q: q["hazard_code"] == "ACE",
                           lambda c: _bool(c.get("conflict_forecasts_available"))),
    "crisiswatch": (lambda q: q["hazard_code"] == "ACE",
                    lambda c: bool((c.get("crisiswatch_arrow") or "").strip())),
    "food_security": (lambda q: q["hazard_code"] == "DR" and q["metric"] == "PHASE3PLUS_IN_NEED",
                      lambda c: (c.get("food_security_source") or "").strip() not in ("", "none", "None")),
    "gdacs_event_history": (lambda q: q["hazard_code"] in ("FL", "DR", "TC"),
                            lambda c: (_num(c.get("gdacs_event_months")) or 0) > 0),
    "enso": (lambda q: q["hazard_code"] in ("FL", "DR", "TC"), lambda c: _bool(c.get("enso_loaded"))),
    "seasonal_tc": (lambda q: q["hazard_code"] == "TC", lambda c: _bool(c.get("seasonal_tc_loaded"))),
    "nmme": (lambda q: q["hazard_code"] in ("FL", "DR"), lambda c: _bool(c.get("nmme_available"))),
    "acled_fatalities_history": (lambda q: q["hazard_code"] == "ACE",
                                 lambda c: (_num(c.get("acled_fatalities_months")) or 0) > 0),
}


def section_inputs(src: Sources) -> dict[str, Any]:
    fresh_path = src.dbg("connector_freshness")
    fresh = read_csv(fresh_path)
    statuses = Counter(r.get("status") for r in fresh)
    flagged = [{k: r.get(k) for k in ("source", "table", "age_days_at_run_start", "status",
                                       "warn_threshold_days", "stale_threshold_days",
                                       "prompt_staleness_warning", "countries_covered",
                                       "countries_missing", "note")}
               for r in fresh if r.get("status") not in ("fresh",)]
    cw = _read_json(src.dbg("crisiswatch_inject")) or {}
    editions = cw.get("editions_present") or []
    older = [e for e in cw.get("ace_country_editions") or [] if e.get("older_than_expected")]
    qm = _question_metrics(src)
    inv_path = src.dbg("data_inject_inventory")
    inv = {r.get("iso3"): r for r in read_csv(inv_path)}
    injects = []
    for name, (should, did) in INJECT_RULES.items():
        want = [q for q in qm if should(q)]
        carried = [q for q in want if inv.get(q["iso3"]) and did(inv[q["iso3"]])]
        gap = [q["question_id"] for q in want if q not in carried]
        injects.append({"inject": name, "should_carry": len(want), "did_carry": len(carried),
                        "gap": len(gap), "gap_questions": gap[:30],
                        "source": "question_metrics x data_inject_inventory (country-level presence in the DB, "
                                  "not the prompt text)"})
    base = read_csv(src.att("inputs/base_rates.csv"))
    if base:
        missing = [r["question_id"] for r in base if not _bool(r.get("anchor_present"))]
        injects.append({"inject": "base_rates", "should_carry": len(base),
                        "did_carry": len(base) - len(missing), "gap": len(missing),
                        "gap_questions": missing[:30], "source": "attribution bundle inputs/base_rates.csv"})
    advice = Counter()
    for row in _llm_calls(src):
        if row.get("phase") in ("spd_v2", "binary_v2") and row.get("question_id"):
            arm = _usage(row).get("advice_arm")
            if arm:
                advice[arm] += 1
    if advice:
        injects.append({"inject": "calibration_advice", "member_calls_by_arm": dict(advice),
                        "note": "advice_empty: in the advice arm, no advice text existed for the group",
                        "source": "debug bundle llm_calls_detail usage_json advice_arm"})
    gp = _read_json(src.dbg("gdacs_prompt_check")) or _read_json(src.first("gdacs_prompt_check*.json"))
    return {
        "freshness": {"n_sources": len(fresh), "by_status": dict(statuses), "not_fresh": flagged,
                      "source": _rel(fresh_path, src.debug)} if fresh else
        stub_with("connector_freshness absent", "executive_summary connector freshness"),
        "crisiswatch": {
            "latest_edition": cw.get("latest_edition"),
            "countries_in_latest_edition": (editions[-1] or {}).get("n_countries") if editions else None,
            "latest_edition_age_months": cw.get("latest_edition_age_months"),
            "expected_edition": cw.get("expected_edition"),
            "expected_edition_held": cw.get("expected_edition_held"),
            "ace_question_countries_on_older_edition": [{"iso3": e.get("iso3"), "edition": e.get("edition")}
                                                        for e in older],
            "ace_questions_with_no_row": cw.get("n_ace_questions_without_crisiswatch"),
            "ace_countries_without_row": cw.get("ace_countries_without_crisiswatch_row"),
            "editions_missing_last_12": cw.get("editions_missing_last_12"),
            "editions_known_gaps": cw.get("editions_known_gaps"),
            "editions_present": editions,
            "source": "debug bundle crisiswatch_inject",
        } if cw else stub_with("crisiswatch_inject absent", "executive_summary"),
        "injects": injects,
        "gdacs_prompt_check": gp if gp else stub_with(
            "no gdacs_prompt_check file in this chain", "Sibyl stage diagnostics gdacs_prompt_check/"),
        "food_security_health": _health(src).get("food_security_health")
        or stub("health_report food_security_health absent"),
    }


# ---------------------------------------------------------------------------
# 5. Horizon Scanner
# ---------------------------------------------------------------------------


def _rc_pairs(rows: list[dict[str, str]]) -> dict[str, int | None]:
    """``ISO3|HZ`` -> RC level (None for a pair not assessed)."""
    out: dict[str, int | None] = {}
    for r in rows:
        key = f"{r.get('iso3')}|{r.get('hazard_code')}"
        if r.get("status") == "seasonal_skip":
            out[key] = None
        else:
            out[key] = _int(r.get("rc_level")) or 0
    return out


def section_horizon_scanner(src: Sources, previous: Mapping[str, Any] | None) -> dict[str, Any]:
    path = src.dbg("rc_triage_summary")
    rows = read_csv(path)
    if not rows:
        return stub_with("rc_triage_summary absent", "coverage_detail")
    pairs = _rc_pairs(rows)
    assessed = {k: v for k, v in pairs.items() if v is not None}
    by_hazard: dict[str, Counter] = defaultdict(Counter)
    for k, lvl in assessed.items():
        by_hazard[k.split("|")[1]][f"L{lvl}"] += 1
    passes = read_csv(src.dbg("rc_pass_detail"))
    per_pair: dict[str, list[dict[str, str]]] = defaultdict(list)
    for p in passes:
        hz = re.sub(r"^RC_|_PASS_\d+$", "", str(p.get("hazard_code") or ""))
        per_pair[f"{p.get('iso3')}|{hz}"].append(p)
    two = {k: v for k, v in per_pair.items() if len({x.get("pass_number") for x in v}) > 1}
    agree = sum(1 for v in two.values() if len({x.get("direction") for x in v}) == 1)
    failed = [{"pair": k, "error": x.get("error_text")[:200]} for k, v in per_pair.items() for x in v
              if (x.get("error_text") or "").strip()]
    rg = _health(src).get("rc_grounding_health") or {}
    stage = _stage_health(src)
    rc_levels = next((s.get("rc_levels") for s in stage.values()
                      if (s.get("rc_levels") or {}).get("n_rows")), None) or {}
    hs_stdout = "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in src.find("hs_stdout.txt"))
    dist_warn = sorted({ln.strip()[:240] for ln in hs_stdout.splitlines()
                        if "distribution" in ln.lower() and ("warn" in ln.lower() or "exceed" in ln.lower())})
    out: dict[str, Any] = {
        "pairs_expected": len(pairs),
        "pairs_screened_out_seasonally": sum(1 for v in pairs.values() if v is None),
        "pairs_assessed": len(assessed),
        "level_counts_assessed": dict(Counter(f"L{v}" for v in assessed.values())),
        "level_counts_by_hazard": {hz: dict(c) for hz, c in sorted(by_hazard.items())},
        "source": "debug bundle rc_triage_summary (blank rc_level on an assessed pair is L0)",
        "stage_health_rc_levels": rc_levels,
        "rc_passes": {"pairs_with_two_passes": len(two), "pairs_two_passes_same_direction": agree,
                      "single_pass_pairs": sum(1 for v in per_pair.values() if len(v) == 1),
                      "note": "the passes carry likelihood, not a level; agreement is by direction",
                      "source": "debug bundle rc_pass_detail"},
        "rc_calls_failed": failed,
        "rc_grounding": {k: rg.get(k) for k in ("n_calls", "n_errors", "n_verified_sources", "n_no_backend",
                                                 "no_backend_by_reason")} | {
            "avg_sources": round(rg["n_verified_sources"] / rg["n_calls"], 2) if rg.get("n_calls") else None,
            "source": "debug bundle health_report rc_grounding_health"},
        "distribution_warnings": dist_warn,
        "distribution_warnings_source": "hs_stdout.txt lines naming the distribution check",
        "rc_pairs": pairs,
    }
    prev = ((previous or {}).get("horizon_scanner") or {}).get("rc_pairs") if previous else None
    if isinstance(prev, dict):
        up, down, jumps = 0, 0, []
        for k, lvl in assessed.items():
            before = prev.get(k)
            if before is None:
                continue
            if lvl > before:
                up += 1
            elif lvl < before:
                down += 1
            if abs(lvl - before) >= 2:
                jumps.append({"pair": k, "before": before, "now": lvl})
        prev_l1 = Counter(k.split("|")[1] for k, v in prev.items() if v and v >= 1)
        out["changed_since_previous"] = {"up": up, "down": down, "jumped_two_or_more": jumps,
                                         "previous_l1_plus_by_hazard": dict(prev_l1),
                                         "source": "derived: rc_pairs against previous_run.horizon_scanner.rc_pairs"}
    else:
        out["changed_since_previous"] = stub("no previous run's rc_pairs to compare with")
    return out


# ---------------------------------------------------------------------------
# 6. Triage and the question set
# ---------------------------------------------------------------------------


def _parse_eligibility(detail: str) -> dict[str, Any]:
    out: dict[str, Any] = {"detail": detail}
    m = re.search(r"source=([a-z_]+)", detail)
    out["source"] = m.group(1) if m else None
    m = re.search(r"asked for (\d+)", detail)
    out["asked"] = int(m.group(1)) if m else None
    m = re.search(r"not asked for (\d+)", detail)
    out["not_asked"] = int(m.group(1)) if m else None
    m = re.search(r"dropped (\d+)", detail)
    out["dropped"] = int(m.group(1)) if m else None
    out["dropped_named"] = re.findall(r"([A-Z]{3}) \(newest", detail)
    return out


def section_triage_questions(src: Sources, previous: Mapping[str, Any] | None) -> dict[str, Any]:
    rows = read_csv(src.dbg("rc_triage_summary"))
    tiers = Counter(r.get("tier") for r in rows)
    status = Counter(r.get("status") for r in rows)
    tg = _health(src).get("triage_grounding_health") or {}
    failures = _read_json(src.first("hs_triage_failures__*.json"))
    coverage = read_csv(src.first("hs_triage_coverage__*.csv"))
    qm = _question_metrics(src)
    by_group: dict[str, Counter] = defaultdict(Counter)
    for q in qm:
        by_group[f"{q['hazard_code']}/{q['metric']}"][f"track{q.get('track') or '?'}"] += 1
    forecast = [q for q in qm if q.get("spd_status") == "ok"]
    keys = sorted({_qkey(q) for q in qm})
    ace_pa = _check(src, "ACE/PA Eligibility")
    p3 = _check(src, "Phase 3+ Eligibility")
    prov = _check(src, "Question Provenance")
    out: dict[str, Any] = {
        "triage": {"tiers": dict(tiers), "status_skip_patterns": dict(status),
                   "source": "debug bundle rc_triage_summary (tier, status)"},
        "triage_grounding": {k: tg.get(k) for k in ("n_calls", "n_errors", "n_verified_sources", "n_no_backend")}
        | {"source": "health_report triage_grounding_health"},
        "triage_failures": {"failures": failures if failures is not None else stub("hs_triage_failures absent"),
                            "coverage_final_status": dict(Counter(r.get("final_status") for r in coverage)),
                            "source": "hs_triage_failures__*.json; hs_triage_coverage__*.csv"},
        "ace_pa_eligibility": _parse_eligibility(ace_pa["detail"]) if ace_pa else stub("no ACE/PA health line"),
        "phase3_eligibility": _parse_eligibility(p3["detail"]) if p3 else stub("no Phase 3+ health line"),
        "eligibility_source": "debug bundle health_report health_checks",
        "questions": {"total": len(qm),
                      "track1": sum(1 for q in qm if q.get("track") == "1"),
                      "track2": sum(1 for q in qm if q.get("track") == "2"),
                      "with_forecast": len(forecast), "without_forecast": len(qm) - len(forecast),
                      "without_forecast_ids": [q["question_id"] for q in qm if q not in forecast],
                      "by_hazard_metric_track": {k: dict(v) for k, v in sorted(by_group.items())},
                      "source": "debug bundle question_metrics (spd_status ok = forecast)"},
        "question_provenance": prov or stub("no Question Provenance health line"),
        "question_keys": keys,
    }
    prev_keys = ((previous or {}).get("triage_questions") or {}).get("question_keys") if previous else None
    if isinstance(prev_keys, list):
        out["entered_since_previous"] = sorted(set(keys) - set(prev_keys))
        out["left_since_previous"] = sorted(set(prev_keys) - set(keys))
        out["set_change_source"] = "derived: question_keys against previous_run.triage_questions.question_keys"
    else:
        out["entered_since_previous"] = stub("no previous run's question_keys")
    return out


# ---------------------------------------------------------------------------
# 7. Evidence and prompts
# ---------------------------------------------------------------------------


def _experiment_arms_file(src: Sources) -> dict[str, Any] | None:
    p = src.first("experiment_arms__*.json")
    return (_read_json(p) or None) if p else None


def adversarial_sync(*, spd_questions: Mapping[str, Mapping[str, Any]], generated_pairs: set[str],
                     injected: set[str], withheld: set[str]) -> dict[str, Any]:
    """The RC sync constraint (template rule A4.7). ``spd_questions`` maps a
    question id to its row (iso3, hazard_code, rc_level)."""
    rc1 = {qid for qid, q in spd_questions.items() if (_int(q.get("rc_level")) or 0) >= 1}
    pair = {qid: f"{q.get('iso3')}|{q.get('hazard_code')}" for qid, q in spd_questions.items()}
    with_check = {qid for qid in rc1 if pair[qid] in generated_pairs}
    missed = sorted(with_check - injected - withheld)
    phantom = sorted(qid for qid in injected if pair.get(qid) not in generated_pairs)
    return {"spd_questions_rc1_plus": len(rc1),
            "rc1_plus_questions_with_a_generated_check": len(with_check),
            "checks_generated_pairs": len(generated_pairs),
            "injected_into_spd_prompts": len(injected),
            "eligible_withheld_by_experiment": len(withheld),
            "generated_not_injected_not_withheld": len(missed),
            "generated_not_injected_not_withheld_ids": missed[:30],
            "injected_with_no_stored_check": len(phantom),
            "injected_with_no_stored_check_ids": phantom[:30]}


#: A heading that names a country, a date or a count is a data heading.
_DATA_HEADING = re.compile(r"\([A-Z]{3}\b|\d")


def _prompt_commits(src: Sources) -> Any:
    """Commits between the two runs' commits that touched the prompt files,
    as the archive job wrote them (``prompt_commits.json``)."""
    payload = _read_json(src.staging / PROMPT_COMMITS)
    if not isinstance(payload, dict):
        return stub_with(f"{PROMPT_COMMITS} not in this chain (written by the archive job)",
                         "git log between the two runs' commits")
    if payload.get("commits") is None:
        return stub_with(str(payload.get("reason") or "commit lookup failed"), PROMPT_COMMITS)
    return {**payload, "source": PROMPT_COMMITS + " (GitHub compare API)"}


def section_evidence(src: Sources, previous: Mapping[str, Any] | None) -> dict[str, Any]:
    qm = {q["question_id"]: q for q in _question_metrics(src)}
    spd = {qid: q for qid, q in qm.items() if q.get("metric") != "EVENT_OCCURRENCE"}
    generated: set[str] = set()
    for r in read_csv(src.dbg("grounding_detail")):
        if str(r.get("stage") or r.get("call_type")) == "adversarial_search":
            generated.add(f"{r.get('iso3')}|{str(r.get('hazard_code') or '').replace('ADVERSARIAL_', '')}")
    injected: set[str] = set()
    synth_calls, synth_cost = 0, 0.0
    prefixes = (_read_json(src.dbg("prompt_prefixes")) or {}).get("prefixes") or {}
    if isinstance(prefixes, dict):
        prefixes = [dict(v or {}, prefix_sha256_16=k) for k, v in prefixes.items()]
    prefix_has = {str(p.get("prefix_sha256_16") or p.get("sha256") or ""): ADVERSARIAL_MARKER in str(p.get("text") or "")
                  for p in prefixes if isinstance(p, dict)}
    have_calls = False
    for row in _llm_calls(src):
        have_calls = True
        if row.get("call_type") == "adversarial_synthesis":
            synth_calls += 1
            synth_cost += _num(row.get("cost_usd")) or 0.0
        if row.get("phase") == "spd_v2" and row.get("question_id"):
            text = str(row.get("prompt_text") or "")
            sha = str(row.get("prompt_prefix_sha256") or "")
            if ADVERSARIAL_MARKER in text or prefix_has.get(sha):
                injected.add(str(row["question_id"]))
    arms = _experiment_arms_file(src) or {}
    withheld = set((arms.get("adversarial") or {}).get("withheld_question_ids") or [])
    if have_calls:
        sync = adversarial_sync(spd_questions=spd, generated_pairs=generated, injected=injected,
                                withheld=withheld)
        sync |= {"adversarial_synthesis_calls": synth_calls, "adversarial_synthesis_cost_usd": round(synth_cost, 4),
                 "withheld_source": "experiment_arms__*.json" if arms else "no experiment arms file: 0 withheld",
                 "source": "debug bundle question_metrics, grounding_detail (adversarial_search), "
                           "llm_calls_detail (spd_v2 prompts carrying '" + ADVERSARIAL_MARKER + "')"}
    else:
        sync = stub_with("llm_calls_detail absent: injection cannot be counted", "health_report Adversarial lines")
    hashes = _read_json(src.att("prompts/section_hashes.json")) or {}
    prev_hashes = ((previous or {}).get("evidence") or {}).get("section_hashes") if previous else None
    if isinstance(prev_hashes, dict) and hashes:
        template_q: dict[str, set] = defaultdict(set)
        lengths: dict[tuple[str, str], set] = defaultdict(set)
        for row in read_csv(src.att("prompts/token_share.csv")):
            template_q[row.get("template_id")].add(row.get("question_id"))
            lengths[(row.get("template_id"), row.get("section_name"))].add(row.get("char_count"))
        # A section whose length varies across a template's questions carries
        # per-question data; its hash moves every run and says nothing about
        # an edit to the prompt.
        per_question = {k for k, v in lengths.items() if len(v) > 1}
        changed = []
        data_sections_skipped = 0
        for tid, secs in hashes.items():
            before = prev_hashes.get(tid)
            if not isinstance(before, dict):
                continue
            for name, sha in secs.items():
                if (tid, name) in per_question:
                    data_sections_skipped += 1
                    continue
                if before.get(name) != sha:
                    changed.append({"template_id": tid, "section": name,
                                    "status": "new" if name not in before else "changed",
                                    "questions_affected": len(template_q.get(tid, ()))})
            for name in before:
                if name not in secs:
                    changed.append({"template_id": tid, "section": name, "status": "removed",
                                    "questions_affected": len(template_q.get(tid, ()))})
        for c in changed:
            c["likely_data"] = bool(_DATA_HEADING.search(c["section"])) or c["questions_affected"] < 2
        section_changes: Any = {"changes": [c for c in changed if not c["likely_data"]],
                                "likely_data_changes": [c for c in changed if c["likely_data"]],
                                "per_question_sections_skipped": data_sections_skipped,
                                "rule": "a section whose length varies across the template's questions in this "
                                        "run carries question data and is not compared; a heading naming a "
                                        "country, a date or a number, or a template with one question, is "
                                        "listed under likely_data_changes",
                                "source": "derived: attribution prompts/section_hashes.json against "
                                          "previous_run.evidence.section_hashes; questions from prompts/token_share.csv",
                                "commits": _prompt_commits(src)}
    else:
        section_changes = stub("no previous run's section_hashes to compare with")
    cache_rows = read_csv(src.dbg("prompt_cache_report"))
    by_provider: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for r in cache_rows:
        prov = r.get("provider") or "?"
        by_provider[prov]["prompt_tokens"] += _num(r.get("prompt_tokens")) or 0
        by_provider[prov]["cache_read_tokens"] += (_num(r.get("cache_read_input_tokens")) or 0) + \
            (_num(r.get("cached_tokens")) or 0)
        by_provider[prov]["estimated_saved_usd"] += _num(r.get("estimated_saved_usd")) or 0
        by_provider[prov]["calls"] += _num(r.get("n_calls")) or 0
    cache = []
    for prov, v in sorted(by_provider.items()):
        rate = (v["cache_read_tokens"] / v["prompt_tokens"] * 100) if v["prompt_tokens"] else None
        cache.append({"provider": prov, "prompt_tokens": int(v["prompt_tokens"]),
                      "cache_read_tokens": int(v["cache_read_tokens"]),
                      "hit_rate_pct": round(rate, 2) if rate is not None else None,
                      "estimated_saved_usd": round(v["estimated_saved_usd"], 4), "calls": int(v["calls"])})
    gd = read_csv(src.dbg("grounding_detail"))
    gstages: dict[str, Counter] = defaultdict(Counter)
    for r in gd:
        st = r.get("stage") or r.get("call_type") or "?"
        gstages[st]["calls"] += 1
        gstages[st]["grounded"] += 1 if _bool(r.get("grounded")) else 0
        gstages[st]["sources"] += _int(r.get("n_sources")) or 0
    return {
        "adversarial": sync,
        "prompt_section_changes": section_changes,
        "section_hashes": hashes or stub("attribution prompts/section_hashes.json absent"),
        "prompt_cache_by_provider": cache or stub("prompt_cache_report absent"),
        "prompt_cache_detail": cache_rows,
        "prompt_cache_source": "debug bundle prompt_cache_report (cache reads = cache_read_input_tokens + cached_tokens)",
        "grounding_by_stage": {k: dict(v) for k, v in gstages.items()},
        "grounding_source": "debug bundle grounding_detail",
    }


# ---------------------------------------------------------------------------
# 8 and 9. Ensemble and Track 2
# ---------------------------------------------------------------------------


def _expected_bucket(probs: Mapping[int, float]) -> float | None:
    total = sum(probs.values())
    if total <= 0:
        return None
    return sum(i * p for i, p in probs.items()) / total


def _parse_family_recal(detail: str) -> list[dict[str, Any]]:
    out = []
    for part in detail.split("·"):
        m = re.match(r"\s*([A-Z]{2,3}/[A-Z0-9_]+):\s*([a-z_]+)\s*(?:\((.*)\))?\s*$", part.strip())
        if m:
            out.append({"group": m.group(1), "status": m.group(2), "reason": m.group(3)})
    return out


def _member_table(src: Sources, models: Iterable[str]) -> list[dict[str, Any]]:
    health = (_health(src).get("llm_health") or {}).get("by_phase_provider_model") or []
    retries = read_csv(src.dbg("retry_report"))
    trace = read_csv(src.att("attribution/trace_quality.csv"))
    comp = read_csv(src.dbg("model_completeness"))
    out = []
    for m in models:
        calls = sum(int(r.get("n_calls") or 0) for r in health
                    if r.get("model_id") == m and r.get("phase") in ("spd_v2", "binary_v2", "track2_spd"))
        errors = sum(int(r.get("n_errors") or 0) for r in health
                     if r.get("model_id") == m and r.get("phase") in ("spd_v2", "binary_v2", "track2_spd"))
        rt = [r for r in retries if r.get("model_id") == m and r.get("phase") in ("spd_v2", "binary_v2")]
        parse = sum(int(_num(r.get("n_parse_failure")) or 0) for r in rt)
        extra = sum(int(_num(r.get("n_extra_attempts")) or 0) for r in rt)
        spd_ok: dict[str, bool] = {}
        bin_ok: dict[str, bool] = {}
        for r in comp:
            if r.get("model_name") != m:
                continue
            bucket = bin_ok if r.get("family") == "binary" else spd_ok
            ok = r.get("verdict") == "ok"
            bucket[r["question_id"]] = bucket.get(r["question_id"], True) and ok
        tq = [_num(r.get("trace_quality_score")) for r in trace if r.get("model_name") == m]
        out.append({"member": m, "calls": calls, "errors": errors, "extra_attempts": extra,
                    "parse_failures": parse,
                    "spd_questions_ok": sum(1 for v in spd_ok.values() if v),
                    "spd_questions_expected": len(spd_ok),
                    "binary_questions_ok": sum(1 for v in bin_ok.values() if v),
                    "binary_questions_expected": len(bin_ok),
                    "trace_quality_median": _median(tq)})
    return out


def section_ensemble(src: Sources) -> dict[str, Any]:
    cfg = _read_json(src.dbg("model_config")) or {}
    members = list((cfg.get("models") or {}).get("spd_track1_ensemble") or [])
    comp = read_csv(src.dbg("model_completeness"))
    rollup = _read_json(src.dbg("model_completeness_rollup")) or {}
    seen = sorted({r.get("model_name") for r in comp if r.get("model_name")})
    outside = [m for m in seen if m not in members and m not in NON_MEMBER
               and not m.endswith(("__raw", "__recal"))]
    fam = _check(src, "Family Recalibration")
    spd_line = _check(src, "SPD Ensemble")
    scen = _check(src, "Scenarios")
    salvage = re.search(r"(\d+)\s+salvaged", str((spd_line or {}).get("detail") or ""))
    # BayesMC against mean: month-1 expected bucket from spd_tables.
    spd_path = src.dbg("spd_tables")
    eb: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    for r in iter_csv(spd_path):
        if r.get("model_name") in ("ensemble_mean_v2", "ensemble_bayesmc_v2") and r.get("month_index") == "1":
            eb[(r["question_id"], r["model_name"])][int(_num(r.get("bucket_index")) or 0)] = \
                _num(r.get("probability")) or 0.0
    diffs = []
    for qid in sorted({k[0] for k in eb}):
        a = _expected_bucket(eb.get((qid, "ensemble_mean_v2"), {}))
        b = _expected_bucket(eb.get((qid, "ensemble_bayesmc_v2"), {}))
        if a is not None and b is not None and abs(a - b) > 0.10:
            diffs.append({"question_id": qid, "mean": round(a, 3), "bayesmc": round(b, 3)})
    dis = read_csv(src.att("contrasts/model_disagreement.csv"))
    per_q: dict[str, list[float]] = defaultdict(list)
    for r in dis:
        a, b = r.get("model_a") or "", r.get("model_b") or ""
        if a in members and b in members:
            v = _num(r.get("jsd_final"))
            if v is not None:
                per_q[r["question_id"]].append(v)
    spread = {q: sum(v) / len(v) for q, v in per_q.items() if v}
    if spread:
        vals = sorted(spread.values())
        p90 = vals[min(len(vals) - 1, int(math.ceil(0.9 * len(vals))) - 1)]
        top = sorted(((q, round(v, 4)) for q, v in spread.items() if v > p90), key=lambda x: -x[1])
    else:
        p90, top = None, []
    anchor = read_csv(src.att("attribution/prior_anchoring.csv"))
    copied = [r for r in anchor if r.get("model_name") in members and (_num(r.get("js_divergence")) or 1) < 0.01]
    trace = read_csv(src.att("attribution/trace_quality.csv"))
    no_update = [r for r in trace if r.get("model_name") in members and _bool(r.get("has_trace"))
                 and (_int(r.get("n_updates")) or 0) == 0]
    binary = rollup.get("binary") or {}
    scen_m = re.search(r"(\d+) generated, (\d+) failed parse, (\d+) skipped, (\d+) missing",
                       str((scen or {}).get("detail") or ""))
    return {
        "members": _member_table(src, members),
        "members_source": "health_report llm_health (spd_v2, binary_v2), retry_report, model_completeness, "
                          "attribution trace_quality.csv",
        "expected_members": members, "members_seen_in_completeness": [m for m in seen if m in members],
        "members_outside_profile": outside,
        "member_cells": {"spd_expected": rollup.get("n_cells_expected"), "spd_missing": rollup.get("n_cells_missing"),
                         "binary_expected": binary.get("n_cells_expected"),
                         "binary_missing": binary.get("n_cells_missing"),
                         "question_months_short": rollup.get("n_question_months_short"),
                         "short_question_months": rollup.get("short_question_months"),
                         "source": "debug bundle model_completeness_rollup"},
        "parse_salvaged": int(salvage.group(1)) if salvage else 0,
        "spd_ensemble_line": spd_line,
        "family_recalibration": _parse_family_recal(str((fam or {}).get("detail") or ""))
        or stub("no Family Recalibration health line"),
        "family_recalibration_source": "health_report Family Recalibration",
        "bayesmc_vs_mean": {"n_questions_differ_over_0_10": len(diffs), "questions": diffs[:30],
                            "source": "derived: month-1 expected bucket index from spd_tables"}
        if eb else stub_with("spd_tables absent", "spd_tables in the chain zip"),
        "member_spread": {"p90_mean_pairwise_jsd": round(p90, 4) if p90 is not None else None,
                          "questions_above_p90": [{"question_id": q, "mean_jsd": v} for q, v in top],
                          "source": "derived: mean jsd_final over member pairs, attribution model_disagreement.csv"},
        "prior_anchoring": {"member_priors_copied_from_anchor_js_lt_0_01": len(copied),
                            "member_traces_with_no_update": len(no_update),
                            "note": "a prior copied from the anchor and no update keeps the anchor unchanged",
                            "source": "attribution prior_anchoring.csv, trace_quality.csv"},
        "rc_sharpness": stub_with("rc_sharpness flags are on the stored traces, not in any lifted file",
                                  "the canonical DB snapshot (forecasts_raw.reasoning_trace_json)"),
        "binary": {"landed": (binary.get("n_cells_expected") or 0) - (binary.get("n_cells_missing") or 0)
                   if binary else None, "expected": binary.get("n_cells_expected"),
                   "source": "model_completeness_rollup binary"},
        "scenarios": ({"generated": int(scen_m.group(1)), "failed_parse": int(scen_m.group(2)),
                       "skipped": int(scen_m.group(3)), "missing": int(scen_m.group(4))} if scen_m
                      else stub("no Scenarios health line")) | ({"source": "health_report Scenarios"} if scen_m else {}),
    }


def section_track2(src: Sources) -> dict[str, Any]:
    qm = [q for q in _question_metrics(src) if q.get("track") == "2"]
    if not qm:
        return stub_with("no Track 2 questions in question_metrics", "question_metrics")
    trace = [r for r in read_csv(src.att("attribution/trace_quality.csv"))
             if r.get("model_name") == TRACK2_MODEL_NAME and r.get("metric") != "EVENT_OCCURRENCE"]
    no_prior = [r for r in trace if not _bool(r.get("has_trace")) or (_num(r.get("prior_quality_score")) or 0) == 0]
    model = None
    env = _stage_context_env(src)
    model = env.get("PYTHIA_TRACK2_MODEL_ID") or None
    costs = [_num(q.get("cost_usd")) for q in qm]
    retries = [r for r in read_csv(src.dbg("retry_report")) if r.get("phase") == "track2_spd"]
    return {
        "questions": len(qm), "ok": sum(1 for q in qm if q.get("spd_status") == "ok"),
        "failed": [q["question_id"] for q in qm if q.get("spd_status") != "ok"],
        "model": model or "track2_spd role (gemini_flash by config)",
        "calls": sum(_int(q.get("n_llm_calls")) or 0 for q in qm),
        "batched_calls": sum(_int(q.get("n_batched_calls")) or 0 for q in qm),
        "retries": retries,
        "traces": {"n": len(trace), "empty_updates": sum(1 for r in trace if (_int(r.get("n_updates")) or 0) == 0),
                   "no_prior": len(no_prior), "no_prior_ids": [r["question_id"] for r in no_prior][:30],
                   "source": "attribution trace_quality.csv (track2_flash, SPD only: binary returns no trace "
                             "by design)"},
        "cost_total_usd": round(sum(c for c in costs if c is not None), 4),
        "cost_per_question_usd": round(sum(c for c in costs if c is not None) / len(qm), 4),
        "source": "debug bundle question_metrics (track 2)",
    }


# ---------------------------------------------------------------------------
# 10. Experiments wiring
# ---------------------------------------------------------------------------


def binom_range(n: int, p: float, lo: float = 0.05, hi: float = 0.95) -> tuple[int, int]:
    """Exact 5th and 95th percentiles of Binomial(n, p)."""
    cdf, lower, upper = 0.0, None, None
    for k in range(n + 1):
        cdf += math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))
        if lower is None and cdf >= lo - 1e-12:
            lower = k
        if upper is None and cdf >= hi - 1e-12:
            upper = k
            break
    return (lower or 0, upper if upper is not None else n)


def arm_check(name: str, share: float | None, counts: Mapping[str, int], treated: str,
              unstamped: int | None, source: str, note: str = "") -> dict[str, Any]:
    """One experiment's wiring row: ``treated`` is the arm the share describes."""
    eligible = sum(counts.values())
    out: dict[str, Any] = {"experiment": name, "design_share": share, "eligible": eligible,
                           "arms": dict(counts), "rows_unstamped": unstamped, "source": source}
    if share is not None and eligible:
        lo, hi = binom_range(eligible, share)
        k = counts.get(treated, 0)
        out |= {"treated_arm": treated, "treated_count": k, "range_90pct": [lo, hi],
                "inside_90pct_range": lo <= k <= hi,
                "range_method": "exact binomial: 5th and 95th percentiles of Binomial(eligible, design_share)"}
    if note:
        out["note"] = note
    return out


def section_experiments(src: Sources) -> dict[str, Any]:
    env = _stage_context_env(src)

    def share(key: str) -> float | None:
        var, default = DEFAULT_SHARES[key]
        raw = env.get(var)
        v = _num(raw) if raw not in (None, "<unset>") else None
        return v if v is not None else default

    arms_file = _experiment_arms_file(src)

    def live(arm: str) -> bool:
        a = (arms_file or {}).get(arm)
        return isinstance(a, dict) and not a.get("off")

    def off_stub(arm: str) -> dict[str, Any] | None:
        a = (arms_file or {}).get(arm)
        if isinstance(a, dict) and a.get("off"):
            return stub_with(f"{arm} arm off on this run: {a.get('reason')}", "experiment_arms__*.json")
        return None

    out: list[dict[str, Any]] = []
    if off_stub("advice"):
        out.append(off_stub("advice"))
    elif live("advice"):
        a = arms_file["advice"]
        out.append(arm_check("advice", a.get("design_share", share("advice")), a.get("counts") or {},
                             "no_advice", a.get("unstamped"), "experiment_arms__*.json"))
    else:
        per_q: dict[str, str] = {}
        for row in _llm_calls(src):
            if row.get("phase") in ("spd_v2", "binary_v2") and row.get("question_id"):
                arm = _usage(row).get("advice_arm")
                if arm:
                    q = str(row["question_id"])
                    # A question is in the advice arm if any member was treated.
                    if per_q.get(q) != "advice":
                        per_q[q] = "advice" if arm == "advice" else ("no_advice" if arm == "no_advice"
                                                                      else per_q.get(q, arm))
        counts = Counter("no_advice" if v == "no_advice" else "advice_arm" for v in per_q.values())
        empty = sum(1 for v in per_q.values() if v == "advice_empty")
        out.append(arm_check("advice", share("advice"), {"advice_arm": counts.get("advice_arm", 0),
                                                         "no_advice": counts.get("no_advice", 0)},
                             "no_advice", None,
                             "derived: per question from llm_calls_detail usage_json advice_arm",
                             note=f"{empty} advice-arm question(s) had no advice text (advice_empty)"))
    rc = read_csv(src.att("attribution/rc_shift_arms.csv"))
    if off_stub("rc_shift"):
        out.append(off_stub("rc_shift"))
    elif live("rc_shift"):
        a = arms_file["rc_shift"]
        out.append(arm_check("rc_shift", a.get("design_share", share("rc_shift")), a.get("counts") or {},
                             "shift", a.get("unstamped"), "experiment_arms__*.json"))
    elif rc:
        counts = Counter()
        for r in rc:
            counts[r.get("rc_shift_arm")] += _int(r.get("n_questions")) or 0
        out.append(arm_check("rc_shift", share("rc_shift"), dict(counts), "shift", None,
                             "attribution attribution/rc_shift_arms.csv"))
    else:
        out.append(stub_with("no rc_shift arm data", "attribution rc_shift_arms.csv"))
    if off_stub("adversarial"):
        out.append(off_stub("adversarial"))
    elif live("adversarial"):
        a = arms_file["adversarial"]
        out.append(arm_check("adversarial", a.get("design_share", share("adversarial")),
                             {k: v for k, v in (a.get("counts") or {}).items() if k in ("shown", "withheld")},
                             "withheld", a.get("unstamped"), "experiment_arms__*.json"))
    else:
        text = "\n".join(p.read_text(encoding="utf-8", errors="replace")
                         for p in src.find("post_run__fc_collect_finalize.txt") + src.find("forecaster_stdout.txt"))
        m = re.search(r"\[adversarial-arm\] eligible=(\d+) shown=(\d+) withheld=(\d+)", text)
        if m:
            out.append(arm_check("adversarial", share("adversarial"),
                                 {"shown": int(m.group(2)), "withheld": int(m.group(3))}, "withheld", None,
                                 "forecaster stdout [adversarial-arm] line"))
        else:
            out.append(stub_with("no [adversarial-arm] line: the experiment was not live on this run "
                                 "(it starts with the 13 December 2026 run)", "post_run__fc_collect_finalize.txt"))
    sib = (arms_file or {}).get("sibyl") if arms_file else None
    out.append(sib if isinstance(sib, dict) else stub_with(
        "Sibyl variant arms are not recorded in any lifted file", "sibyl_run.log"))
    return {"experiments": out, "shares_source": "stage context env (fc_collect_finalize), else production defaults"}


# ---------------------------------------------------------------------------
# 11. Sibyl
# ---------------------------------------------------------------------------


def section_sibyl(src: Sources) -> dict[str, Any]:
    sh = (_stage_health(src).get("sibyl") or {}).get("sibyl") or {}
    log_path = src.first("sibyl_run.log")
    log = log_path.read_text(encoding="utf-8", errors="replace") if log_path else ""
    pool = None
    m = re.search(r"eligible pool by hazard and RC level[^:]*:\s*(\{.*\})", log)
    if m:
        try:
            pool = json.loads(m.group(1).replace("'", '"'))
        except json.JSONDecodeError:
            pool = m.group(1)
    picks = Counter(re.findall(r"selection_pass[=:\s'\"]+(floor|fill|control)", log))
    rule = re.search(r"selection rule[=:\s'\"]+([a-z_]+)", log, re.I)
    sj = _read_json(src.cur("sibyl.json")) or {}
    rows = sj.get("rows") or []
    fvs = {r.get("question_id"): r for r in read_csv(src.att("contrasts/fred_vs_sibyl.csv"))}
    diffs = []
    for r in rows:
        if r.get("fred_expected") is None or r.get("sibyl_expected") is None:
            continue
        diffs.append({"question_id": r.get("question_id"),
                      "sibyl_expected": r.get("sibyl_expected"), "ensemble_expected": r.get("fred_expected"),
                      "js_divergence_vs_standard": r.get("js_divergence_vs_standard"),
                      "tag": r.get("tag"),
                      "sibyl_claimed_drivers": (fvs.get(r.get("question_id")) or {}).get("sibyl_signal_classes")})
    diffs.sort(key=lambda d: -(d.get("js_divergence_vs_standard") or 0))
    check = _check(src, "Sibyl")
    ratio = re.search(r"\(([\d.]+)x the stated \$([\d.]+)", str((check or {}).get("detail") or ""))
    searches = Counter()
    for row in _llm_calls(src):
        if row.get("phase") == "sibyl" and "search" in str(row.get("call_type") or ""):
            searches[(row.get("provider"), bool((row.get("error_text") or "").strip()))] += 1
    if not sh and not sj:
        return stub_with("no Sibyl stage health or sibyl.json in this chain", "sibyl_run.log")
    tools = sh.get("tools") or {}
    depth = sh.get("research_depth") or {}
    return {
        "run": {k: sh.get(k) for k in ("sibyl_run_id", "model", "k", "aggregation", "run_hard_cap_usd",
                                         "budget_capped", "time_capped", "run_cost_usd", "opus_cost_usd",
                                         "brave_cost_usd", "n_selected", "n_forecast", "n_skipped", "by_status",
                                         "degraded")} | {"source": "stage_health__sibyl.json sibyl"},
        "selection": {"rule": rule.group(1) if rule else "rc_focus (CLAUDE.md default)",
                      "eligible_pool_by_hazard_and_rc_level": pool,
                      "picks_by_pass": dict(picks) or stub("selection_pass not in sibyl_run.log"),
                      "source": _rel(log_path, src.staging) if log_path else "sibyl_run.log absent"},
        "research": {"searches": tools.get("n_search_calls"), "searches_failed": tools.get("n_search_failed"),
                     "search_fail_share": tools.get("search_fail_share"),
                     "breaker_trips": tools.get("n_breaker_trips"), "documents_read": tools.get("n_docs_read"),
                     "search_calls_by_provider_and_error": {f"{p}:{'error' if e else 'ok'}": n
                                                             for (p, e), n in searches.items()},
                     "median_docs_per_trial": depth.get("median_docs_per_trial"),
                     "steps_per_trial": depth.get("steps_per_trial"),
                     "share_docs_wikipedia": depth.get("share_docs_wikipedia"),
                     "submit_gate_unmet": depth.get("n_submit_gate_unmet"),
                     "depth_warnings": sh.get("depth_warnings"),
                     "source": "stage_health__sibyl.json tools and research_depth; llm_calls_detail (sibyl search)"},
        "summary": (sj.get("summary") or {}) | {"source": "current-run bundle sibyl.json"},
        "carried_forward": sj.get("carried_forward"),
        "cost_gate": {"ratio": float(ratio.group(1)) if ratio else None,
                      "stated_per_question_usd": float(ratio.group(2)) if ratio else None,
                      "health_line": check, "source": "health_report Sibyl line"},
        "largest_differences": diffs[:5],
        "largest_differences_note": "claims, not causes (rule A4.10); expected values in the metric's units",
        "trials_reconciler_outliers": stub_with("trial-level counts are in the DB (sibyl_forecasts.trials_json)",
                                                "sibyl_run.log"),
    }


# ---------------------------------------------------------------------------
# 12. Outputs
# ---------------------------------------------------------------------------


def section_outputs(src: Sources) -> dict[str, Any]:
    interp = _check(src, "Interpreter")
    gate = src.first("gate_recalibration.md")
    gate_row = None
    if gate:
        text = gate.read_text(encoding="utf-8", errors="replace")
        meta = _read_json(src.staging / "manifest.json") or {}
        fc = ((meta.get("chain") or {}).get("fc_run_id")) or ""
        for line in text.splitlines():
            if fc and line.startswith(fc):
                parts = line.split()
                keys = ("run", "seen", "unusual", "level", "moved", "worsening", "burden", "watch", "near", "thin", "cut")
                gate_row = dict(zip(keys, parts))
                break
    anchor = _read_json(src.first("anchor_depth.json"))
    manifest = _read_json(src.staging / "manifest.json") or {}
    files = manifest.get("files") or []
    bundles = [{"file": f.get("path"), "bytes": f.get("bytes")} for f in files
               if re.search(r"(debug_bundle|forecast_attribution|current_run_analysis|llm_calls_detail|"
                            r"report__.*\.pdf|report_combined)", str(f.get("path") or ""))]
    cycle = section_chain(src)["runs"]
    sib = [r for r in cycle if r["slug"] == "sibyl"]
    pub = [r for r in cycle if r["slug"] == "publish_latest_data"]
    report = src.first("report_combined_v*.md")
    return {
        "interpreter": interp or stub("no Interpreter health line"),
        "interpreter_report": _rel(report, src.staging) if report else None,
        "gate": gate_row or stub_with("no gate row for this run", "gate_recalibration.md"),
        "gate_source": "gate_recalibration.md (this run's row)",
        "anchor_depth": (anchor or {}).get("summary") if anchor else stub("anchor_depth.json absent"),
        "pdf_built": bool(src.find("report__*.pdf")),
        "bundles": bundles, "bundles_source": "manifest.json files",
        "canonical_db_uploaded_by_sibyl": sib[-1]["db_uploaded"] if sib else None,
        "release": {"publish_conclusion": pub[-1]["conclusion"] if pub else None,
                    "source": "Publish api_run.json"},
        "archive": "this folder is written by the archive leg; Drive verification is in its leg report",
    }


# ---------------------------------------------------------------------------
# 13. Cost
# ---------------------------------------------------------------------------


def section_cost(src: Sources, previous: Mapping[str, Any] | None) -> dict[str, Any]:
    cs = _health(src).get("cost_summary") or {}
    qm = _question_metrics(src)
    n_q = len(qm) or None
    by_track: dict[str, list[float]] = defaultdict(list)
    for q in qm:
        if _num(q.get("cost_usd")) is not None:
            by_track[q.get("track") or "?"].append(_num(q.get("cost_usd")))
    prev_phases = (((previous or {}).get("cost") or {}).get("by_phase")) if previous else None
    phases = []
    for phase, v in sorted((cs.get("by_phase") or {}).items()):
        cost = v.get("total_cost_usd")
        row = {"phase": phase, "tokens": v.get("total_tokens"), "cost_usd": cost,
               "per_question_usd": round(cost / n_q, 4) if cost is not None and n_q else None}
        if isinstance(prev_phases, list):
            prev = next((p for p in prev_phases if p.get("phase") == phase), None)
            if prev and prev.get("per_question_usd") and row["per_question_usd"] is not None:
                row["ratio_to_previous_per_question"] = round(row["per_question_usd"] / prev["per_question_usd"], 3)
        phases.append(row)
    batch_line = _check(src, "Batch Economics")
    lifecycle = _read_json(src.dbg("batch_lifecycle")) or {}
    totals = lifecycle.get("totals") or {}
    cache = sum(_num(r.get("estimated_saved_usd")) or 0 for r in read_csv(src.dbg("prompt_cache_report")))
    lost = 0.0
    n_err = 0
    for row in _llm_calls(src):
        if (row.get("error_text") or "").strip():
            n_err += 1
            lost += _num(row.get("cost_usd")) or 0.0
    rc = read_csv(src.dbg("run_comparison"))
    return {
        "by_phase": phases, "total_usd": cs.get("total_cost_usd"), "total_tokens": cs.get("total_tokens"),
        "questions": n_q,
        "per_question_by_track": {f"track{k}": round(sum(v) / len(v), 4) for k, v in by_track.items() if v},
        "source": "health_report cost_summary; per question derived over question_metrics count",
        "per_track_source": "question_metrics cost_usd (forecast-phase spend per question)",
        "batch": {"health_line": batch_line, "lifecycle_totals": totals,
                  "source": "health_report Batch Economics; batch_lifecycle totals"},
        "prompt_cache_saving_usd": round(cache, 4),
        "cost_of_failed_calls_usd": {"value": round(lost, 4), "n_failed_calls": n_err,
                                     "source": "derived: sum of cost_usd on llm_calls_detail rows with error_text"},
        "run_comparison": rc or stub("run_comparison absent"),
        "run_comparison_source": "debug bundle run_comparison (this run beside the previous two)",
    }


# ---------------------------------------------------------------------------
# 14. Run over run
# ---------------------------------------------------------------------------


def _template_id(q: Mapping[str, Any]) -> str:
    phase = "binary_v2" if q.get("metric") == "EVENT_OCCURRENCE" else "spd_v2"
    return f"{phase}|{q.get('hazard_code')}|{q.get('metric')}|t{q.get('track')}"


_INVENTORY_KEYS = ("crisiswatch_edition", "baserate_n_obs", "acled_fatality_months", "ipc_phase_rows",
                   "evidence_items", "resolver_history_months", "seasonal_tc_outlook_present")


def section_run_over_run(src: Sources, previous: Mapping[str, Any] | None,
                         evidence: Mapping[str, Any]) -> dict[str, Any]:
    rows = read_csv(src.att("contrasts/run_over_run.csv"))
    if not rows:
        return stub_with("attribution contrasts/run_over_run.csv absent", "deltas.json")
    inv = {r["question_id"]: r for r in read_csv(src.att("inputs/input_inventory.csv"))}
    qm = {q["question_id"]: q for q in _question_metrics(src)}
    prev = previous or {}
    prev_inv = ((prev.get("run_over_run") or {}).get("inventory")) if isinstance(prev.get("run_over_run"), dict) else None
    changes = evidence.get("prompt_section_changes") if isinstance(evidence, dict) else None
    changed_templates = {c["template_id"] for c in (changes or {}).get("changes", [])} \
        if isinstance(changes, dict) and not changes.get("stub") else None

    def move(r: Mapping[str, str]) -> float:
        a, b = _num(r.get("ev_current")), _num(r.get("ev_previous"))
        if a is None or b is None:
            return 0.0
        return abs(math.log((a + 1) / (b + 1)))

    ranked = sorted(rows, key=move, reverse=True)[:25]
    out = []
    for r in ranked:
        qid = r.get("question_id")
        key = _qkey(r)
        now_inv = {k: inv.get(qid, {}).get(k) for k in _INVENTORY_KEYS}
        before_inv = (prev_inv or {}).get(key) if isinstance(prev_inv, dict) else None
        new_inputs = [k for k in _INVENTORY_KEYS if before_inv is not None and str(before_inv.get(k)) != str(now_inv.get(k))]
        tid = _template_id(qm.get(qid, {"metric": r.get("metric"), "hazard_code": r.get("hazard_code"), "track": "?"}))
        out.append({"question_id": qid, "previous_question_id": r.get("previous_question_id"),
                    "ev_previous": _num(r.get("ev_previous")), "ev_current": _num(r.get("ev_current")),
                    "log_ratio": round(move(r), 4),
                    "prior_ev_change": _num(r.get("prior_ev_change")),
                    "top_class_previous": r.get("top_class_previous"), "top_class_current": r.get("top_class_current"),
                    "prompt_section_hash_changed": (tid in changed_templates) if changed_templates is not None else None,
                    "new_input_in_inventory": new_inputs if before_inv is not None else None})
    inventory = {_qkey(r): {k: r.get(k) for k in _INVENTORY_KEYS} for r in inv.values()}
    deltas = _read_json(src.cur("deltas.json")) or {}
    return {"largest_moves": out,
            "ranking": "derived: |ln((ev_current+1)/(ev_previous+1))|, matched on (iso3, hazard, metric)",
            "source": "attribution contrasts/run_over_run.csv; inputs/input_inventory.csv",
            "largest_spd_movements": deltas.get("largest_spd_movements"),
            "deltas_source": "current-run bundle deltas.json",
            "inventory": inventory}


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def _guard(fn: Callable[[], Any]) -> Any:
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - one section never costs the others
        return stub(f"{type(exc).__name__}: {exc}")


def _run_date(src: Sources) -> dt.date | None:
    manifest = _read_json(src.staging / "manifest.json") or {}
    d = (manifest.get("chain") or {}).get("start_date")
    try:
        return dt.date.fromisoformat(str(d)[:10]) if d else None
    except ValueError:
        return None


def embed_previous(previous: Mapping[str, Any] | None) -> dict[str, Any]:
    if not previous:
        return stub("no earlier forecast_review")
    if previous.get("stub"):
        return dict(previous)
    return {k: previous.get(k) for k in SECTION_KEYS if k in previous}


def build_inputs(src: Sources, *, previous: Mapping[str, Any] | None = None) -> dict[str, Any]:
    prev = previous if previous and not previous.get("stub") else None
    run_date = _run_date(src)
    evidence = _guard(lambda: section_evidence(src, prev))
    return {
        "schema_version": INPUTS_VERSION,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "sources": src.label | {"workflow_logs_index": (src.staging / WORKFLOW_LOGS_INDEX).is_file()},
        "meta": _guard(lambda: section_meta(src)),
        "chain": _guard(lambda: section_chain(src)),
        "issues": _guard(lambda: section_issues(src)),
        "inputs": _guard(lambda: section_inputs(src)),
        "horizon_scanner": _guard(lambda: section_horizon_scanner(src, prev)),
        "triage_questions": _guard(lambda: section_triage_questions(src, prev)),
        "evidence": evidence,
        "ensemble": _guard(lambda: section_ensemble(src)),
        "track2": _guard(lambda: section_track2(src)),
        "experiments": _guard(lambda: section_experiments(src)),
        "sibyl": _guard(lambda: section_sibyl(src)),
        "outputs": _guard(lambda: section_outputs(src)),
        "cost": _guard(lambda: section_cost(src, prev)),
        "run_over_run": _guard(lambda: section_run_over_run(src, prev, evidence)),
        "known_issues": _guard(lambda: rri.section_known_issues(src, run_date)),
        "previous_run": embed_previous(previous),
    }


def stubs(payload: Mapping[str, Any]) -> list[str]:
    """Dotted paths of every stub in ``payload`` (for the PR and the review's Appendix 1)."""
    found: list[str] = []

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            if node.get("stub") is True:
                found.append(path)
                return
            for k, v in node.items():
                if k == "previous_run":
                    continue
                walk(v, f"{path}.{k}" if path else k)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")

    walk(payload, "")
    return found


def write_inputs_from(src: Sources, out: Path, *, previous: Mapping[str, Any] | None = None) -> dict[str, Any]:
    payload = build_inputs(src, previous=previous)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, default=str) + "\n", encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--staging", required=True, help="a forecast chain's staging tree, or an unpacked chain zip")
    p.add_argument("--out", default="forecast_review_inputs.json")
    p.add_argument("--work-dir", default="forecast_review_work")
    p.add_argument("--previous", default="", help="the previous chain's forecast_review_inputs.json")
    args = p.parse_args(argv)
    try:
        previous = _read_json(Path(args.previous)) if args.previous else None
        src = locate(Path(args.staging), Path(args.work_dir))
        payload = write_inputs_from(src, Path(args.out), previous=previous)
        print(json.dumps({"keys": [k for k in payload if k in SECTION_KEYS or k == "previous_run"],
                          "stubs": stubs(payload)}, indent=1))
    except Exception as exc:  # noqa: BLE001 - a diagnostic never fails its caller
        print(f"::warning title=forecast review inputs::{type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
