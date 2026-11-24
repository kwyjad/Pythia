# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The resolver review's inputs, read out of a chain's staging tree.

The monthly resolver-to-calibration review is written by a scheduled Claude
session that reads Drive through the connector: it reads text and cannot open
a zip. Its inputs sit in the scoring chain zip, most of them inside the
resolver debug bundle, a zip within that zip, and one of them
(``hazard/cell_ledger.csv``) was 29 MB in November 2026. So the Drive archive
writes ``resolver_review/`` beside the chain zip, and this module builds the
file that folder leads with, ``resolver_review_inputs.json``: one key per
section of the review template (``resolver_review_template.md`` in Drive,
``docs/resolver_review_inputs.md`` here), the counts the review needs from the
cell ledger, and the plausibility checks the review runs, already run.

A pure function of the staging tree (what ``build_cycle_diagnostics`` lays
out, or a chain zip unpacked): it opens no DB and asks no API. Every section
degrades to ``{"stub": true, "reason": ...}``; nothing here raises.

Run alone: ``python -m scripts.ci.resolver_review_inputs --staging DIR --out FILE``.
"""

from __future__ import annotations

import argparse
import calendar
import csv
import dataclasses
import datetime as dt
import io
import json
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - import convenience
    sys.path.insert(0, str(REPO_ROOT))

SCHEMA_VERSION = 1
DEBUG_BUNDLE_GLOB = "resolver-debug-bundle-*.zip"
SCORED_BUNDLE_GLOB = "scored_forecast_analysis__*.zip"
MAX_EXAMPLES = 10
#: A figure above this share of the national population is flagged.
POPULATION_SHARE_FLAG = 0.10
#: The four conflict displacement tables the review compares before and after.
CONFLICT_TABLES = (
    "conflict_displacement_candidates", "conflict_displacement_cell_ledger",
    "dtm_conflict_rounds", "dtm_conflict_lower_bounds",
)
#: Post-ingest stages whose chain diagnostics the review reads, in chain order.
POST_INGEST_SLUGS = ("compute_resolutions", "compute_scores", "compute_calibration",
                     "publish_latest_data")
EVENT_METRIC = "EVENT_OCCURRENCE"
#: Reconciliation verdicts that mean the connector is fine.
AGREEING = ("yes", "unchanged on purpose", "n/a")


def stub(reason: str) -> dict[str, Any]:
    return {"stub": True, "reason": reason}


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


def _read_json(path: Path | None) -> Any:
    if path is None or not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _csv_lines(handle: Iterable[str]) -> Iterator[str]:
    """Skip the ``#`` preamble the debug bundle writes above a CSV header."""
    started = False
    for line in handle:
        if not started and line.startswith("#"):
            continue
        started = True
        yield line


def iter_csv(path: Path | None) -> Iterator[dict[str, str]]:
    """Rows of a CSV one at a time (the cell ledger can be tens of MB)."""
    if path is None or not path.is_file():
        return iter(())

    def gen() -> Iterator[dict[str, str]]:
        with path.open(encoding="utf-8", newline="") as fh:
            yield from csv.DictReader(_csv_lines(fh))

    return gen()


def read_csv(path: Path | None) -> list[dict[str, str]]:
    return list(iter_csv(path))


def md_tables(text: str) -> list[dict[str, Any]]:
    """Every pipe table in a markdown document, with the heading above it."""
    tables: list[dict[str, Any]] = []
    heading = ""
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("#"):
            heading = line.lstrip("#").strip()
        if line.startswith("|") and i + 1 < len(lines) and re.match(r"^\|\s*:?-{2,}", lines[i + 1].strip()):
            header = [c.strip() for c in line.strip("|").split("|")]
            rows = []
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                rows.append(dict(zip(header, cells)))
                i += 1
            tables.append({"heading": heading, "header": header, "rows": rows})
            continue
        i += 1
    return tables


def _num(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).replace(",", "").replace("*", "").strip()
    if text in ("", "None", "nan", "—", "-"):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in ("true", "1", "yes", "t")


def _ym(value: Any) -> str:
    text = str(value or "").strip()
    m = re.match(r"^(\d{4})-(\d{2})", text)
    return f"{m.group(1)}-{m.group(2)}" if m else text


def _ts(value: Any) -> dt.datetime | None:
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def load_population(path: Path) -> dict[str, float]:
    """``resolver/data/population.csv``: the population column carries
    thousands separators and padding, so they are stripped before use."""
    out: dict[str, float] = {}
    for row in iter_csv(path):
        clean = {k.strip(): v for k, v in row.items() if k}
        iso = str(clean.get("iso3") or "").strip().upper()
        pop = _num(clean.get("population"))
        if iso and pop:
            out[iso] = pop
    return out


# ---------------------------------------------------------------------------
# Where things are in a staging tree
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Sources:
    """Paths into a chain's staging tree, and the debug bundle unpacked."""

    staging: Path
    run_folders: dict[str, Path]            # slug -> run folder (newest of a slug)
    all_run_folders: list[Path]
    debug_zip: Path | None
    debug: Path | None                      # the debug bundle, unpacked
    backfill: Path | None                   # backfill-diagnostics artifact
    chain_diag: dict[str, Path]             # slug -> chain-diagnostics dir
    scored_zip: Path | None
    resolver_inspect: Path | None

    @property
    def has_resolver_update(self) -> bool:
        return "resolver_update" in self.run_folders

    def bundle(self, rel: str) -> Path | None:
        if self.debug is None:
            return None
        p = self.debug / rel
        return p if p.exists() else None


_FOLDER_RE = re.compile(r"^(\d+)_(?P<slug>[a-z_]+?)__(?P<run>\d+)$")


def _run_folders(staging: Path) -> list[tuple[str, Path]]:
    out = []
    for p in sorted(staging.iterdir()) if staging.is_dir() else []:
        m = _FOLDER_RE.match(p.name)
        if p.is_dir() and m:
            out.append((m.group("slug"), p))
    return out


def _safe_extract(zf: zipfile.ZipFile, dest: Path) -> None:
    for info in zf.infolist():
        target = (dest / info.filename).resolve()
        if not str(target).startswith(str(dest.resolve())):
            continue  # unsafe path
        if info.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(info) as src, target.open("wb") as out:
            while chunk := src.read(1024 * 1024):
                out.write(chunk)


def locate(staging: Path, work: Path) -> Sources:
    """Find every input in ``staging`` and unpack the resolver debug bundle
    into ``work`` (the only nested zip that is unpacked to disk)."""
    folders = _run_folders(staging)
    by_slug: dict[str, Path] = {}
    for slug, path in folders:
        by_slug[slug] = path  # sorted by position: the last of a slug is the newest
    ru = by_slug.get("resolver_update")
    debug_zip = None
    if ru is not None:
        zips = sorted(ru.rglob(DEBUG_BUNDLE_GLOB))
        debug_zip = zips[-1] if zips else None
    debug = None
    if debug_zip is not None:
        debug = work / "debug_bundle"
        if debug.exists():
            import shutil

            shutil.rmtree(debug)
        debug.mkdir(parents=True)
        try:
            with zipfile.ZipFile(debug_zip) as zf:
                _safe_extract(zf, debug)
            # A bundle zipped with a top folder: step into it.
            entries = [p for p in debug.iterdir()]
            if len(entries) == 1 and entries[0].is_dir() and not (debug / "README.md").exists():
                debug = entries[0]
        except zipfile.BadZipFile:
            debug = None
    backfill = None
    if ru is not None:
        cands = [p for p in ru.rglob("backfill-diagnostics") if p.is_dir()]
        backfill = cands[0] if cands else None
    chain_diag: dict[str, Path] = {}
    for slug, path in folders:
        for d in sorted(path.glob(f"chain-diagnostics-{slug}-*")):
            if d.is_dir():
                chain_diag[slug] = d
    scored = sorted(p for p in staging.rglob(SCORED_BUNDLE_GLOB) if p.is_file())
    inspect = sorted(p for p in staging.rglob("resolver_inspect.md") if p.is_file())
    return Sources(staging=staging, run_folders=by_slug,
                   all_run_folders=[p for _, p in folders], debug_zip=debug_zip, debug=debug,
                   backfill=backfill, chain_diag=chain_diag,
                   scored_zip=scored[-1] if scored else None,
                   resolver_inspect=inspect[-1] if inspect else None)


# ---------------------------------------------------------------------------
# The scored bundle, read in place
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Scored:
    scores: list[dict[str, str]]            # scores_flat.csv
    records: list[dict[str, Any]]           # questions/*.json

    def latest_scores(self) -> list[dict[str, str]]:
        rows = [r for r in self.scores if str(r.get("is_latest_run", "")).lower() not in ("false", "0")]
        return rows

    def resolutions(self) -> dict[tuple[str, int], dict[str, Any]]:
        """(question_id, horizon) -> resolution with question context."""
        out: dict[tuple[str, int], dict[str, Any]] = {}
        for rec in self.records:
            q = rec.get("question") or {}
            for r in (rec.get("outcome") or {}).get("resolutions") or []:
                if r.get("horizon_m") is None:
                    continue
                out[(str(q.get("question_id")), int(r["horizon_m"]))] = {
                    "question_id": q.get("question_id"),
                    "iso3": str(q.get("iso3") or "").upper(),
                    "hazard_code": str(q.get("hazard_code") or "").upper(),
                    "metric": str(q.get("metric") or "").upper(),
                    "horizon_m": int(r["horizon_m"]),
                    "observed_month": _ym(r.get("observed_month")),
                    "value": r.get("value"),
                    "realized_bucket": r.get("realized_bucket"),
                    "source_desc": r.get("source_desc"),
                }
        if not out:  # no records: fall back to scores_flat
            for s in self.latest_scores():
                key = (str(s.get("question_id")), int(_num(s.get("horizon_m")) or 0))
                out.setdefault(key, {
                    "question_id": s.get("question_id"),
                    "iso3": str(s.get("iso3") or "").upper(),
                    "hazard_code": str(s.get("hazard_code") or "").upper(),
                    "metric": str(s.get("metric") or "").upper(),
                    "horizon_m": key[1],
                    "observed_month": _ym(s.get("observed_month")),
                    "value": _num(s.get("resolved_value")),
                    "realized_bucket": None,
                    "source_desc": s.get("source_desc"),
                })
        return out


def read_scored(path: Path | None) -> Scored | None:
    if path is None:
        return None
    try:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            scores: list[dict[str, str]] = []
            flat = [n for n in names if Path(n).name == "scores_flat.csv"]
            if flat:
                text = zf.read(flat[0]).decode("utf-8")
                scores = list(csv.DictReader(_csv_lines(io.StringIO(text))))
            records = []
            for n in names:
                parts = Path(n).parts
                if len(parts) >= 2 and parts[-2] == "questions" and n.endswith(".json"):
                    try:
                        records.append(json.loads(zf.read(n).decode("utf-8")))
                    except Exception:  # noqa: BLE001
                        continue
            return Scored(scores=scores, records=records)
    except (zipfile.BadZipFile, OSError):
        return None


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def _issues_from_json(path: Path) -> list[dict[str, Any]]:
    payload = _read_json(path) or {}
    out = []
    for it in payload.get("issues") or []:
        out.append({k: it.get(k) for k in ("id", "severity", "title", "first_seen", "runs_seen",
                                           "cost", "cost_unit", "owner", "note", "evidence")})
    return out


def _log_has_traceback(path: Path) -> bool:
    try:
        return "Traceback (most recent call last)" in path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False


def section_chain(src: Sources) -> dict[str, Any]:
    manifest = _read_json(src.staging / "manifest.json") or {}
    stages = []
    for folder in src.all_run_folders:
        api = _read_json(folder / "api_run.json") or {}
        m = _FOLDER_RE.match(folder.name)
        slug = m.group("slug") if m else folder.name
        diag = next(iter(sorted(folder.glob(f"chain-diagnostics-{slug}-*"))), None)
        lineage = _read_json(diag / "lineage.json") if diag else None
        start = _ts(api.get("run_started_at") or api.get("created_at"))
        end = _ts(api.get("updated_at"))
        stages.append({
            "folder": folder.name, "slug": slug, "workflow": api.get("name"),
            "run_id": api.get("id") or (m.group("run") if m else None),
            "event": api.get("event"), "conclusion": api.get("conclusion"),
            "started_at": api.get("run_started_at") or api.get("created_at"),
            "ended_at": api.get("updated_at"),
            "duration_min": round((end - start).total_seconds() / 60, 1) if start and end else None,
            "db_downloaded_from": (lineage or {}).get("canonical_db_run_downloaded"),
            "db_uploaded": (lineage or {}).get("canonical_db_run_uploaded"),
            "upload_guard": (lineage or {}).get("upload_guard_verdict"),
            "trigger_run": (lineage or {}).get("trigger_run"),
        })
    out: dict[str, Any] = {"stages": stages, "cycle_manifest_runs": manifest.get("runs")}
    if src.debug is None:
        out["resolver_update"] = stub("no resolver debug bundle in the chain")
        return out
    wr = _read_json(src.bundle("run/workflow_run.json"))
    steps = read_csv(src.bundle("run/step_timings.csv"))
    logs_dir = src.bundle("logs")
    logs = sorted(p for p in logs_dir.rglob("*.log")) if logs_dir else []
    zero = []
    for s in steps:
        dur = _num(s.get("duration_sec"))
        if dur is None or dur > 0 or s.get("status") != "completed":
            continue
        name = str(s.get("step") or "")
        m = re.search(r"phase\s*(\d+)(?:\.(\d+))?", name, re.I)
        matched = []
        if m:
            prefix = "phase" + m.group(1) + (m.group(2) or "")
            matched = [p for p in logs if p.name.lower().startswith(prefix)
                       and p.stat().st_size > 0]
        zero.append({"step": name, "conclusion": s.get("conclusion"),
                     "logs_with_content": [p.name for p in matched],
                     "logs_with_traceback": [p.name for p in matched if _log_has_traceback(p)]})
    out["resolver_update"] = {
        "workflow_run": wr if wr is not None else stub("run/workflow_run.json absent"),
        "step_timings": steps if steps else stub("run/step_timings.csv absent or empty"),
        "zero_second_steps": [z for z in zero if z["logs_with_content"]],
        "zero_second_steps_without_logs": [z["step"] for z in zero if not z["logs_with_content"]],
        "logs_with_traceback": [p.name for p in logs if _log_has_traceback(p)],
        "note": "a zero-second step is matched to its logs by the phase number in its name",
    }
    return out


def section_issues(src: Sources) -> dict[str, Any]:
    out: dict[str, Any] = {}
    path = src.bundle("checks/issues.json")
    if path is None:
        out["resolver_update"] = stub("checks/issues.json absent from the debug bundle")
    else:
        issues = _issues_from_json(path)
        out["resolver_update"] = {"counts": dict(Counter(i.get("severity") for i in issues)),
                                  "issues": issues}
    others: dict[str, Any] = {}
    for slug, d in src.chain_diag.items():
        if slug == "resolver_update":
            continue
        p = d / "issues.json"
        if p.is_file():
            others[slug] = _issues_from_json(p)
        elif (d / "issues.md").is_file():
            others[slug] = {"markdown": (d / "issues.md").read_text(encoding="utf-8")[:20000]}
    out["other_stages"] = others
    return out


def _crisiswatch_inspect_line(text: str) -> str | None:
    for line in text.splitlines():
        if "crisiswatch" in line.lower() and "|" in line:
            return line.strip()
    return None


def section_ingestion(src: Sources, stage_runs: list[dict[str, Any]]) -> dict[str, Any]:
    if src.debug is None:
        return stub("no resolver debug bundle in the chain")
    out: dict[str, Any] = {}
    rec = src.bundle("checks/reconciliation.md")
    if rec is None:
        out["reconciliation"] = stub("checks/reconciliation.md absent")
    else:
        tables = md_tables(rec.read_text(encoding="utf-8"))
        rows = tables[0]["rows"] if tables else []
        bad = []
        for r in rows:
            status = str(r.get("status") or "").lower()
            agrees = str(r.get("agrees") or "").strip().lower()
            fetched = _num(r.get("fetched"))
            if status.startswith("superseded"):
                continue
            if (status not in ("ok", "success") or fetched == 0
                    or (agrees not in ("", "—") and not agrees.startswith(AGREEING))):
                bad.append(r)
        out["reconciliation"] = {"connectors": len(rows), "not_ok_or_disagreeing": bad,
                                 "ok_and_agreeing": len(rows) - len(bad)}
    fresh = read_csv(src.bundle("db/freshness.csv"))
    out["freshness_past_threshold"] = ([r for r in fresh if r.get("verdict") == "stale"]
                                       if fresh else stub("db/freshness.csv absent"))
    vint = read_csv(src.bundle("db/queries/conflict_forecast_vintages.csv"))
    out["conflict_forecast_vintages"] = vint or stub("db/queries/conflict_forecast_vintages.csv absent")
    cw_rows = read_csv(src.bundle("db/queries/crisiswatch_editions.csv"))
    cw: dict[str, Any] = {}
    if cw_rows:
        latest = max(cw_rows, key=lambda r: (int(_num(r.get("year")) or 0), int(_num(r.get("month")) or 0)))
        cw["latest_edition"] = f"{int(_num(latest.get('year')) or 0)}-{int(_num(latest.get('month')) or 0):02d}"
        cw["entries"] = _num(latest.get("entries"))
        cw["countries"] = _num(latest.get("countries"))
        cw["deteriorated"] = _num(latest.get("deteriorated"))
        cw["fetched_at"] = latest.get("fetched_at")
        cw["editions_held"] = sorted(f"{int(_num(r.get('year')) or 0)}-{int(_num(r.get('month')) or 0):02d}"
                                     for r in cw_rows)
    else:
        cw["latest_edition"] = stub("db/queries/crisiswatch_editions.csv absent")
    if src.resolver_inspect is not None:
        cw["inspect_line"] = _crisiswatch_inspect_line(src.resolver_inspect.read_text(encoding="utf-8"))
    else:
        cw["inspect_line"] = stub("resolver_inspect.md not in the chain")
    refresh = [s for s in stage_runs if s.get("slug") == "refresh_crisiswatch"]
    cw["refresh_runs"] = {"ok": sum(1 for s in refresh if s.get("conclusion") == "success"),
                          "failed": sum(1 for s in refresh if s.get("conclusion") not in ("success", None)),
                          "runs": [s.get("run_id") for s in refresh]}
    out["crisiswatch"] = cw
    return out


def section_contradictions(src: Sources) -> Any:
    if src.debug is None:
        return stub("no resolver debug bundle in the chain")
    path = src.bundle("checks/contradictions.md")
    if path is None:
        return stub("checks/contradictions.md absent")
    tables = md_tables(path.read_text(encoding="utf-8"))
    rows = tables[0]["rows"] if tables else []
    checks = [{"check": r.get("check"), "verdict": r.get("verdict"), "observed": r.get("observed"),
               "expected": r.get("expected")} for r in rows]
    return {"counts": dict(Counter(c["verdict"] for c in checks)), "checks": checks}


def _month_end(ym: str) -> dt.date | None:
    m = re.match(r"^(\d{4})-(\d{2})$", ym)
    if not m:
        return None
    y, mo = int(m.group(1)), int(m.group(2))
    return dt.date(y, mo, calendar.monthrange(y, mo)[1])


def _freeze_days(src: Sources) -> int:
    for cand in (src.bundle("config/rulebook.yaml"),
                 REPO_ROOT / "resolver" / "hazard_resolution" / "rulebook.yaml"):
        if cand is None or not cand.is_file():
            continue
        m = re.search(r"^freeze_days:\s*(\d+)", cand.read_text(encoding="utf-8"), re.M)
        if m:
            return int(m.group(1))
    return 60


def cell_ledger_counts(path: Path, run_date: dt.date | None) -> dict[str, Any]:
    """Counts the review needs from the cell ledger, in one streaming pass."""
    status_hazard: Counter = Counter()
    reason_hazard: Counter = Counter()
    ceiling_basis: Counter = Counter()
    flags: Counter = Counter()
    open_months: Counter = Counter()
    provisional_months: Counter = Counter()
    rows = 0
    current = run_date.strftime("%Y-%m") if run_date else None
    for r in iter_csv(path):
        rows += 1
        hz = r.get("hazard") or "?"
        status = r.get("status") or ""
        ym = _ym(r.get("ym"))
        status_hazard[(status or "no_row", hz)] += 1
        if not status:
            reason_hazard[(r.get("reason_code") or "", hz)] += 1
        if status == "RESOLVED_VALUE":
            ceiling_basis[r.get("ceiling_basis") or "unrecorded"] += 1
        for f in str(r.get("flags") or "").split("|"):
            if f:
                flags[(f, hz)] += 1
        if current and ym >= current:
            open_months[(ym, hz, status or "no_row")] += 1
        if status and _truthy(r.get("provisional")):
            provisional_months[(ym, hz)] += 1

    def nest(counter: Counter, keys: tuple[str, str]) -> list[dict[str, Any]]:
        return [{keys[0]: a, keys[1]: b, "cells": n} for (a, b), n in sorted(counter.items())]

    return {
        "rows": rows,
        "by_status_hazard": nest(status_hazard, ("status", "hazard")),
        "no_row_by_reason_hazard": nest(reason_hazard, ("reason_code", "hazard")),
        "unexplained_no_row": sum(n for (reason, _), n in reason_hazard.items()
                                  if reason == "unexplained_no_row"),
        "resolved_value_by_ceiling_basis": dict(ceiling_basis),
        "flags_by_hazard": nest(flags, ("flag", "hazard")),
        "months_not_yet_ended": [{"ym": a, "hazard": b, "status": c, "cells": n}
                                 for (a, b, c), n in sorted(open_months.items())],
        "provisional_by_month": [{"ym": a, "hazard": b, "cells": n}
                                 for (a, b), n in sorted(provisional_months.items())],
    }


def section_pa_machine(src: Sources, run_date: dt.date | None) -> dict[str, Any]:
    if src.debug is None:
        return stub("no resolver debug bundle in the chain")
    out: dict[str, Any] = {}
    acc = src.bundle("hazard/acceptance_report.md")
    out["acceptance"] = (md_tables(acc.read_text(encoding="utf-8")) if acc
                         else stub("hazard/acceptance_report.md absent"))
    ledger = src.bundle("hazard/cell_ledger.csv")
    counts = cell_ledger_counts(ledger, run_date) if ledger else None
    out["cell_ledger"] = counts if counts else stub("hazard/cell_ledger.csv absent")
    budget = read_csv(src.bundle("hazard/extraction_budget.csv"))
    if budget and run_date:
        this_m = run_date.strftime("%Y-%m")
        last_m = (run_date.replace(day=1) - dt.timedelta(days=1)).strftime("%Y-%m")
        agg: dict[str, dict[str, Any]] = {}
        for r in budget:
            month = _ym(r.get("day"))
            if month not in (this_m, last_m):
                continue
            key = f"{month}:{r.get('caller')}"
            a = agg.setdefault(key, {"month": month, "caller": r.get("caller"), "calls": 0,
                                     "billed_calls": 0, "cost_usd": 0.0})
            a["calls"] += int(_num(r.get("calls")) or 0)
            a["billed_calls"] += int(_num(r.get("billed_calls")) or 0)
            a["cost_usd"] = round(a["cost_usd"] + (_num(r.get("cost_usd")) or 0.0), 4)
        out["extraction_budget"] = sorted(agg.values(), key=lambda a: (a["month"], str(a["caller"])))
    else:
        out["extraction_budget"] = stub("hazard/extraction_budget.csv absent or empty")
    prog = read_csv(src.bundle("hazard/backcast_progress.csv"))
    out["backcast_progress"] = ([{"hazard": h, "status": s, "months": n}
                                 for (h, s), n in sorted(Counter((r.get("hazard"), r.get("status"))
                                                                 for r in prog).items())]
                                if prog else stub("hazard/backcast_progress.csv absent"))
    days = _freeze_days(src)
    freezes = []
    for item in (counts or {}).get("provisional_by_month") or []:
        end = _month_end(item["ym"])
        if end is None:
            continue
        deadline = end + dt.timedelta(days=days)
        freezes.append({**item, "freezes_on": deadline.isoformat(),
                        "frozen_by_run_date": bool(run_date and deadline < run_date)})
    out["freeze_dates"] = {"freeze_days": days, "provisional_months": freezes}
    return out


def _signature_counts(path: Path | None) -> dict[str, Any]:
    sig = _read_json(path) or {}
    merged: dict[str, Any] = {}
    for key in ("required_counts", "optional_counts", "all_counts"):
        merged.update(sig.get(key) or {})
    return merged


def section_conflict(src: Sources) -> dict[str, Any]:
    if src.debug is None:
        return stub("no resolver debug bundle in the chain")
    before = _signature_counts(src.bundle("db/signature_before.json"))
    after = _signature_counts(src.bundle("db/signature_after.json"))
    tables = {t: {"before": before.get(t), "after": after.get(t),
                  "changed": (before.get(t) != after.get(t)) if t in before or t in after else None}
              for t in CONFLICT_TABLES}
    out: dict[str, Any] = {"tables": tables}
    summaries: dict[str, Any] = {}
    for name in ("conflict_displacement_extract.json", "conflict_displacement_extract_plan.json",
                 "dtm_conflict.json"):
        found = sorted(src.backfill.rglob(name)) if src.backfill else []
        summaries[name] = _read_json(found[0]) if found else stub(f"{name} absent: the step left no summary")
    out["phase26_summaries"] = summaries
    ledger = read_csv(src.bundle("hazard/conflict_displacement/cell_ledger.csv"))
    if ledger:
        yms = [_ym(r.get("ym") or r.get("month")) for r in ledger if (r.get("ym") or r.get("month"))]
        assessed = sorted(str(r.get("assessed_at") or "") for r in ledger if r.get("assessed_at"))
        out["ledger"] = {"rows": len(ledger), "latest_ym": max(yms) if yms else None,
                         "assessed_at_range": [assessed[0], assessed[-1]] if assessed else None,
                         "by_reason": dict(Counter(r.get("reason_code") for r in ledger))}
    else:
        out["ledger"] = stub("hazard/conflict_displacement/cell_ledger.csv absent or empty")
    idmc = sorted(src.backfill.rglob("ingestion/idmc/summary.json")) if src.backfill else []
    out["idmc_conflict"] = _read_json(idmc[0]) if idmc else stub("ingestion/idmc/summary.json absent")
    return out


def section_resolutions(src: Sources) -> dict[str, Any]:
    d = src.chain_diag.get("compute_resolutions")
    if d is None:
        return stub("no Compute Resolutions chain diagnostics in the chain")
    rep = _read_json(d / "reports" / "compute_resolutions.json")
    if rep is None:
        return stub("reports/compute_resolutions.json absent")
    keep = ("outcome_counts", "calendar_cutoff", "questions_considered", "rows_written",
            "resolved_from_source", "resolved_as_zero", "resolutions_before", "resolutions_after",
            "purged_stale_beyond_cutoff", "purged_non_series_fatalities", "machine_retracted",
            "vintages_written", "mostly_zero_default_groups")
    out = {k: rep.get(k) for k in keep if k in rep}
    unresolved = {}
    for group, counts in (rep.get("outcome_counts") or {}).items():
        unresolved[group] = {k: v for k, v in counts.items() if str(k).startswith("unresolved")}
    out["unresolved_split"] = unresolved
    out["by_source_desc"] = read_csv(d / "resolutions_by_source_desc.csv") or rep.get("by_source_desc")
    return out


# -- Plausibility ------------------------------------------------------------

_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})[a-z]?\b")
_PEOPLE_RE = re.compile(r"\b\d[\d,.]*\s*(?:million\s+)?(?:people|persons|individuals)\b", re.I)


@dataclasses.dataclass
class Cell:
    iso3: str
    hazard: str
    ym: str
    value: float | None
    status: str
    answering_rung: str


def _resolved_cells(ledger: Path | None) -> dict[tuple[str, str, str], Cell]:
    out = {}
    for r in iter_csv(ledger):
        if r.get("status") != "RESOLVED_VALUE":
            continue
        key = (str(r.get("iso3") or "").upper(), str(r.get("hazard") or "").upper(), _ym(r.get("ym")))
        out[key] = Cell(*key, value=_num(r.get("value")), status="RESOLVED_VALUE",
                        answering_rung=str(r.get("answering_rung") or ""))
    return out


def _winning_figures(figures: list[dict[str, str]], cells: Mapping[tuple[str, str, str], Cell]
                     ) -> dict[tuple[str, str, str], dict[str, str]]:
    """The accepted figure that set each resolved cell's value: the lowest
    preference rank among accepted figures whose persons value equals the
    cell's value (the first accepted figure when none equals it)."""
    by_cell: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for f in figures:
        if str(f.get("outcome") or "").lower() != "accepted":
            continue
        key = (str(f.get("iso3") or "").upper(), str(f.get("hazard") or "").upper(), _ym(f.get("ym")))
        by_cell[key].append(f)
    out = {}
    for key, cell in cells.items():
        cands = by_cell.get(key)
        if not cands:
            continue
        def val(f: dict[str, str]) -> float | None:
            return _num(f.get("value_persons")) or _num(f.get("value"))
        exact = [f for f in cands if cell.value is not None and val(f) is not None
                 and abs((val(f) or 0) - cell.value) < 0.5]
        pool = exact or ([] if cell.answering_rung and "reliefweb" not in cell.answering_rung else cands)
        if pool:
            out[key] = sorted(pool, key=lambda f: _num(f.get("preference_rank")) or 0)[0]
    return out


def _example(cell: tuple[str, str, str], value: Any, fig: Mapping[str, str] | None, **extra: Any) -> dict[str, Any]:
    ex = {"iso3": cell[0], "hazard": cell[1], "ym": cell[2], "value": value}
    if fig is not None:
        ex["quote"] = str(fig.get("quote") or "")[:400]
        ex["doc_id"] = fig.get("doc_id")
    ex.update(extra)
    return ex


def plausibility_checks(*, cells: Mapping[tuple[str, str, str], Cell], figures: list[dict[str, str]],
                        scored: Scored | None, population: Mapping[str, float],
                        outcome_counts: Mapping[str, Any] | None,
                        run_date: dt.date | None = None) -> dict[str, Any]:
    """The review's Section 8.4 checks, each with a count, up to ten examples
    and the scored resolutions (and forecaster rows) that rest on the cells.

    The cell ledger runs back to 2000, so ``count`` is taken over the cells a
    scored PA resolution reads (or, with no scored bundle, the twelve months
    before the run); ``count_last_12_months`` and ``count_all_months`` sit
    beside it, and examples list scored cells first.
    """
    winners = _winning_figures(figures, cells)
    pa_res = {k: v for k, v in (scored.resolutions() if scored else {}).items() if v["metric"] == "PA"}
    scores = scored.latest_scores() if scored else []
    scored_keys = {(v["iso3"], v["hazard_code"], v["observed_month"]) for v in pa_res.values()}
    if run_date is not None:
        hi = run_date.strftime("%Y-%m")
        lo = dt.date(run_date.year - 1, run_date.month, 1).strftime("%Y-%m")
    else:
        hi = lo = None

    def recent(key: tuple[str, str, str]) -> bool:
        return bool(lo and lo <= key[2] < hi)

    def in_scope(key: tuple[str, str, str]) -> bool:
        return key in scored_keys if scored is not None else recent(key)

    def reach(flagged: Iterable[tuple[str, str, str]]) -> dict[str, Any]:
        flagged = set(flagged)
        res = {k for k, v in pa_res.items() if (v["iso3"], v["hazard_code"], v["observed_month"]) in flagged}
        rows = {(s.get("question_id"), s.get("horizon_m"), s.get("model_name")) for s in scores
                if str(s.get("metric") or "").upper() == "PA"
                and (str(s.get("iso3") or "").upper(), str(s.get("hazard_code") or "").upper(),
                     _ym(s.get("observed_month"))) in flagged}
        return {"scored_resolutions": len(res), "scored_forecaster_rows": len(rows)}

    checks: dict[str, Any] = {}
    flagged_union: set[tuple[str, str, str]] = set()

    def order(ex: dict[str, Any]) -> tuple:
        key = (ex["iso3"], ex["hazard"].split("/")[0], ex["ym"])
        return (not in_scope(key), not recent(key), ex["ym"], ex["iso3"])

    def record(name: str, flagged: list[tuple[str, str, str]], examples: list[dict[str, Any]],
               **extra: Any) -> None:
        flagged_set = set(flagged)
        checks[name] = {"count": sum(1 for k in flagged_set if in_scope(k)),
                        "count_last_12_months": sum(1 for k in flagged_set if recent(k)),
                        "count_all_months": len(flagged_set),
                        "examples": sorted(examples, key=order)[:MAX_EXAMPLES],
                        **reach(flagged_set), **extra}

    # 1. Identical FL and TC PA values in one country-month.
    pairs: dict[tuple[str, str], dict[str, float | None]] = defaultdict(dict)
    for (iso, hz, ym), c in cells.items():
        if hz in ("FL", "TC") and c.value:
            pairs[(iso, ym)][hz] = c.value
    both = {k: v for k, v in pairs.items() if "FL" in v and "TC" in v}
    same = sorted(k for k, v in both.items() if v["FL"] == v["TC"])
    scored_both = {(iso, ym) for iso, _hz, ym in scored_keys
                   if (iso, "FL", ym) in scored_keys and (iso, "TC", ym) in scored_keys}
    flagged = [(iso, hz, ym) for iso, ym in same for hz in ("FL", "TC")]
    record("identical_fl_tc_pa_value", flagged,
           [_example((iso, "FL/TC", ym), both[(iso, ym)]["FL"], winners.get((iso, "FL", ym)))
            for iso, ym in same],
           scope_unit="cells (two per country-month)",
           country_months_both_resolved_in_scored_set=len(scored_both & set(both)),
           identical_country_months_in_scored_set=sum(1 for k in same if k in scored_both),
           identical_country_months_last_12_months=sum(1 for k in same if recent((k[0], "FL", k[1]))))
    flagged_union.update(flagged)

    # 2. The winning quote names only earlier years.
    flagged, ex = [], []
    for key, f in sorted(winners.items()):
        years = {int(y) for y in _YEAR_RE.findall(str(f.get("quote") or ""))}
        if years and max(years) < int(key[2][:4]):
            flagged.append(key)
            ex.append(_example(key, cells[key].value, f, years_named=sorted(years)))
    record("winning_quote_names_only_earlier_years", flagged, ex)
    flagged_union.update(flagged)

    # 3. Household conversion where the quote also states people.
    hh = [f for f in figures if str(f.get("outcome") or "").lower() == "accepted"
          and "household" in str(f.get("stated_unit") or f.get("unit") or "").lower()
          and _PEOPLE_RE.search(str(f.get("quote") or ""))]
    win_keys = {k for k, f in winners.items() if f in hh}
    record("household_conversion_where_quote_states_people", sorted(win_keys),
           [_example(k, cells[k].value, winners[k]) for k in sorted(win_keys)],
           accepted_figures=len(hh), set_a_cell_value=len(win_keys))
    flagged_union.update(win_keys)

    # 4. One figure winning two months.
    groups: dict[tuple, list[tuple[str, str, str]]] = defaultdict(list)
    for key, f in winners.items():
        sig = (key[0], key[1], f.get("doc_id"), str(f.get("quote") or "").strip()[:200],
               _num(f.get("value_persons")) or _num(f.get("value")))
        groups[sig].append(key)
    multi = {sig: sorted(keys) for sig, keys in groups.items() if len({k[2] for k in keys}) > 1}
    flagged = [k for keys in multi.values() for k in keys]
    record("one_figure_wins_two_months", flagged,
           [_example(keys[0], cells[keys[0]].value, winners[keys[0]], months=[k[2] for k in keys])
            for keys in multi.values()])
    flagged_union.update(flagged)

    # 5. Value above 10% of national population.
    over = []
    for key, c in sorted(cells.items()):
        pop = population.get(key[0])
        if pop and c.value and c.value / pop > POPULATION_SHARE_FLAG:
            over.append((key, c.value / pop))
    record("value_above_10pct_of_population", [k for k, _ in over],
           [_example(k, cells[k].value, winners.get(k), share=round(s, 3)) for k, s in over])
    flagged_union.update(k for k, _ in over)

    # 6. Round figures from extraction (exact multiples of 10,000).
    rnd = [k for k, c in sorted(cells.items()) if "reliefweb" in c.answering_rung
           and c.value and c.value >= 10000 and c.value % 10000 == 0]
    record("round_extracted_figure", rnd, [_example(k, cells[k].value, winners.get(k)) for k in rnd],
           note="a pointer to check, not a fault: round figures are often stated that way")

    # 7. Displacement lower bound used as people affected (scored set).
    lb = sorted({(v["iso3"], v["hazard_code"], v["observed_month"]) for v in pa_res.values()
                 if "lower_bound" in str(v.get("source_desc") or "") or "dtm" in str(v.get("source_desc") or "")})
    record("displacement_lower_bound_as_people_affected", lb,
           [_example(k, None, None) for k in lb],
           scored_pa_resolutions=len(pa_res),
           note="by design: the rung is scored as the answer and labelled a lower bound")

    # 8. Zero-default share by group.
    shares = []
    for group, c in sorted((outcome_counts or {}).items()):
        z, s = int(c.get("zero_default", 0) or 0), int(c.get("sourced", 0) or 0)
        shares.append({"group": group, "sourced": s, "zero_default": z,
                       "share": round(z / (z + s), 3) if z + s else None,
                       "exempt": EVENT_METRIC in str(group).upper(),
                       "above_half": bool(z + s and z / (z + s) > 0.5)})
    checks["zero_default_share_by_group"] = shares or stub("no outcome_counts")

    checks["union_of_flagged_cells"] = {"count": sum(1 for k in flagged_union if in_scope(k)),
                                        "count_all_months": len(flagged_union),
                                        "cells": sorted("/".join(k) for k in flagged_union if in_scope(k)),
                                        **reach(flagged_union),
                                        "note": "checks 1-5 together; round figures and lower bounds excluded"}
    checks["_flagged_cells"] = flagged_union  # internal, removed before writing
    return checks


def section_plausibility(src: Sources, scored: Scored | None, outcome_counts: Any,
                         population_path: Path, run_date: dt.date | None = None) -> dict[str, Any]:
    if src.debug is None:
        return stub("no resolver debug bundle in the chain")
    ledger = src.bundle("hazard/cell_ledger.csv")
    if ledger is None:
        return stub("hazard/cell_ledger.csv absent")
    cells = _resolved_cells(ledger)
    figures = read_csv(src.bundle("hazard/figures_ledger.csv"))
    out = plausibility_checks(cells=cells, figures=figures, scored=scored,
                              population=load_population(population_path),
                              outcome_counts=outcome_counts if isinstance(outcome_counts, Mapping) else None,
                              run_date=run_date)
    if scored is None:
        out["note_scored"] = "no scored bundle in the chain: scored reach is zero by absence, not by fact"
    if not figures:
        out["note_figures"] = "hazard/figures_ledger.csv absent or empty: quote checks found nothing to read"
    return out


def section_scoring(src: Sources, scored: Scored | None) -> dict[str, Any]:
    d = src.chain_diag.get("compute_scores")
    out: dict[str, Any] = {}
    if d is None:
        out["reports"] = stub("no Compute Scores chain diagnostics in the chain")
    else:
        out["reports"] = {name: (_read_json(d / "reports" / f"{name}.json")
                                 or stub(f"reports/{name}.json absent"))
                          for name in ("compute_scores", "score_baselines", "score_views")}
    if scored is None or not scored.records:
        out["hand_off"] = stub("no scored bundle question records in the chain")
        return out
    gaps, pairs, resolved = [], 0, 0
    for rec in scored.records:
        q = rec.get("question") or {}
        horizons = {int(r["horizon_m"]) for r in (rec.get("outcome") or {}).get("resolutions") or []
                    if r.get("horizon_m") is not None}
        resolved += len(horizons)
        have = {(int(s["horizon_m"]), s.get("model_name")) for s in rec.get("scores") or []
                if s.get("horizon_m") is not None}
        forecasters: dict[str, set[int]] = {}
        for name, grid in (rec.get("ensemble") or {}).items():
            forecasters[name] = {int(m) for m in (grid.get("months") or {}) if str(m).isdigit()}
        for m in rec.get("members") or []:
            if str(m.get("status") or "ok").lower() == "ok" and m.get("ok") not in (False, 0):
                forecasters.setdefault(m.get("model_name"), set(range(1, 7)))
        for name, months in forecasters.items():
            for h in horizons & months:
                pairs += 1
                if (h, name) not in have:
                    gaps.append({"question_id": q.get("question_id"), "horizon_m": h, "model_name": name})
    out["hand_off"] = {"resolved_question_horizons": resolved, "forecaster_pairs": pairs,
                       "pairs_with_forecast_but_no_score": len(gaps), "gaps": gaps[:50],
                       "note": "derived from the scored bundle's question records (latest run per question)"}
    return out


def centroid_counterfactuals(moves: list[dict[str, str]], scored: Scored | None,
                             flagged: set[tuple[str, str, str]]) -> list[dict[str, Any]]:
    res = scored.resolutions() if scored else {}
    out = []
    for m in moves:
        hz, metric = str(m.get("hazard_code") or "").upper(), str(m.get("metric") or "").upper()
        bucket = _num(m.get("bucket_index"))
        old, new, emp = _num(m.get("old")), _num(m.get("new")), _num(m.get("empirical"))
        item: dict[str, Any] = {"hazard_code": hz, "metric": metric, "bucket_index": bucket,
                                "old": old, "new": new, "empirical": emp,
                                "n_resolutions": _num(m.get("n_resolutions"))}
        members = [v for v in res.values() if v["hazard_code"] == hz and v["metric"] == metric
                   and bucket is not None and v.get("realized_bucket") is not None
                   and int(v["realized_bucket"]) == int(bucket) and v.get("value") is not None]
        if members:
            vals = [float(v["value"]) for v in members]
            item["bundle_resolutions"] = len(vals)
            item["bundle_mean"] = round(sum(vals) / len(vals), 1)
            clean = [float(v["value"]) for v in members
                     if (v["iso3"], v["hazard_code"], v["observed_month"]) not in flagged]
            item["flagged_resolutions"] = len(vals) - len(clean)
            if clean and item["flagged_resolutions"]:
                item["clean_mean"] = round(sum(clean) / len(clean), 1)
                if None not in (old, new, emp) and emp != old:
                    rate = (new - old) / (emp - old)
                    item["ema_rate"] = round(rate, 4)
                    item["counterfactual_new"] = round(old + rate * (item["clean_mean"] - old), 1)
        else:
            item["bundle_resolutions"] = 0
        out.append(item)
    return out


def section_calibration(src: Sources, scored: Scored | None, flagged: set) -> dict[str, Any]:
    d = src.chain_diag.get("compute_calibration")
    if d is None:
        return stub("no Compute Calibration chain diagnostics in the chain")
    weights = read_csv(d / "calibration_weights.csv")
    moves = read_csv(d / "centroid_ema_updates.csv")
    fam = read_csv(d / "family_recalibration.csv")
    return {
        "weights": weights or stub("calibration_weights.csv absent"),
        "groups_weighted": sum(1 for w in weights if (_num(w.get("weight_rows_written_this_run")) or 0) > 0),
        "groups": len(weights),
        "family_recalibration": fam or stub("family_recalibration.csv absent"),
        "centroid_moves": centroid_counterfactuals(moves, scored, flagged) if moves else
        stub("centroid_ema_updates.csv absent or empty"),
        "sibyl_calibration_steps": _read_json(d / "sibyl_calibration_steps.json")
        or stub("sibyl_calibration_steps.json absent"),
        "note": "counterfactual_new = old + ema_rate x (clean_mean - old); ema_rate backed out of this "
                "run's move; clean_mean leaves out resolutions on cells the plausibility checks flagged",
    }


def _known_issues_path(src: Sources) -> Path | None:
    if src.debug is not None:
        found = sorted((src.debug / "config").rglob("known_issues.yml")) if (src.debug / "config").is_dir() else []
        if found:
            return found[0]
    repo = REPO_ROOT / "resolver" / "config" / "known_issues.yml"
    return repo if repo.is_file() else None


def section_known_issues(src: Sources, run_date: dt.date | None) -> Any:
    path = _known_issues_path(src)
    if path is None:
        return stub("known_issues.yml not found")
    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # noqa: BLE001
        return stub(f"known_issues.yml unreadable: {type(exc).__name__}: {exc}")
    out = []
    for entry in data.get("issues") or []:
        rb = entry.get("review_by")
        rb_text = rb.isoformat() if isinstance(rb, dt.date) else (str(rb) if rb is not None else None)
        overdue: bool | None
        if rb_text in (None, ""):
            overdue = True
        elif rb_text == "never":
            overdue = False if entry.get("no_review_reason") else True
        else:
            try:
                overdue = bool(run_date and dt.date.fromisoformat(rb_text[:10]) < run_date)
            except ValueError:
                overdue = None
        out.append({"id": entry.get("id"), "owner": entry.get("owner"), "review_by": rb_text,
                    "overdue": overdue, "first_seen": str(entry.get("first_seen") or "") or None})
    return {"source": "debug bundle" if src.debug and str(path).startswith(str(src.debug)) else "repository",
            "run_date": run_date.isoformat() if run_date else None, "entries": out}


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def _run_date(src: Sources) -> dt.date | None:
    wr = _read_json(src.bundle("run/workflow_run.json")) or {}
    for key in ("run_started_at", "created_at", "started_at"):
        ts = _ts(wr.get(key))
        if ts:
            return ts.date()
    ru = src.run_folders.get("resolver_update")
    api = _read_json(ru / "api_run.json") if ru else {}
    ts = _ts((api or {}).get("run_started_at") or (api or {}).get("created_at"))
    return ts.date() if ts else None


def _guard(fn: Callable[[], Any]) -> Any:
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - one section never costs the others
        return stub(f"{type(exc).__name__}: {exc}")


def build_inputs(src: Sources, *, population_path: Path | None = None,
                 run_date: dt.date | None = None) -> dict[str, Any]:
    population_path = population_path or REPO_ROOT / "resolver" / "data" / "population.csv"
    run_date = run_date or _run_date(src)
    scored = _guard(lambda: read_scored(src.scored_zip))
    if isinstance(scored, dict):
        scored = None
    chain = _guard(lambda: section_chain(src))
    resolutions = _guard(lambda: section_resolutions(src))
    plaus = _guard(lambda: section_plausibility(
        src, scored, resolutions.get("outcome_counts") if isinstance(resolutions, dict) else None,
        population_path, run_date))
    flagged = plaus.pop("_flagged_cells", set()) if isinstance(plaus, dict) else set()
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "run_date": run_date.isoformat() if run_date else None,
        "sources": {
            "debug_bundle": src.debug_zip.name if src.debug_zip else None,
            "scored_bundle": src.scored_zip.name if src.scored_zip else None,
            "resolver_inspect": src.resolver_inspect.relative_to(src.staging).as_posix()
            if src.resolver_inspect else None,
            "chain_diagnostics": sorted(src.chain_diag),
            "backfill_diagnostics": bool(src.backfill),
        },
        "chain": chain,
        "issues": _guard(lambda: section_issues(src)),
        "ingestion": _guard(lambda: section_ingestion(
            src, chain.get("stages", []) if isinstance(chain, dict) else [])),
        "contradictions": _guard(lambda: section_contradictions(src)),
        "pa_machine": _guard(lambda: section_pa_machine(src, run_date)),
        "conflict_displacement": _guard(lambda: section_conflict(src)),
        "resolutions": resolutions,
        "plausibility": plaus,
        "scoring": _guard(lambda: section_scoring(src, scored)),
        "calibration": _guard(lambda: section_calibration(src, scored, flagged)),
        "known_issues": _guard(lambda: section_known_issues(src, run_date)),
    }


def write_inputs_from(src: Sources, out: Path, **kwargs: Any) -> dict[str, Any]:
    payload = build_inputs(src, **kwargs)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return payload


def write_inputs(staging: Path, out: Path, work: Path, **kwargs: Any) -> dict[str, Any]:
    src = locate(staging, work)
    payload = build_inputs(src, **kwargs)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--staging", required=True, help="a chain's staging tree, or an unpacked chain zip")
    p.add_argument("--out", default="resolver_review_inputs.json")
    p.add_argument("--work-dir", default="resolver_review_work")
    args = p.parse_args(argv)
    try:
        write_inputs(Path(args.staging), Path(args.out), Path(args.work_dir))
    except Exception as exc:  # noqa: BLE001 - a diagnostic never fails its caller
        print(f"::warning title=resolver review inputs::{type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
