# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""A permanent copy of every finished chain, in the owner's Google Drive.

GitHub deletes artifacts after 90 days at most, and some of ours after 14.
``archive_to_drive.yml`` runs this module to leave, per completed chain, one
self-describing zip (``build_cycle_diagnostics --full``: no size ceiling,
nothing dropped, Sibyl's large artifacts included, no DB inside) and, after a
forecast chain, a compressed and verified snapshot of the canonical DB.

Layout under ``$DRIVE_ROOT`` (``Fred_archive``)::

    runs/<YYYY>/<YYYY-MM-DD>__<chain>__<ids>/   the zip, plus loose README,
                                                CYCLE_SUMMARY, manifest,
                                                executive summary, health
                                                report and interpreter report;
                                                a scoring chain also gets
                                                scored_bundle/ (the scored
                                                bundle's report files, unzipped)
    db_snapshots/<YYYY>/                        pythia_resolver_db_*.duckdb.zst
                                                and its .db_snapshot.json

Subcommands, each a step of the workflow:

``plan``         which Publish runs need archiving (one, or every one in a
                 window), named and checked against Drive; writes the matrix.
``leg``          archive one: build the zip, (forecast) snapshot the DB, upload
                 with ``rclone copy --checksum`` and ``rclone check``.
``scored-bundle`` add only the missing ``scored_bundle/`` folder to a chain
                 already in Drive (read back from its zip; the zip and loose
                 files are never replaced).
``housekeeping`` thin old DB snapshots, check Drive space, merge the legs'
                 reports into ``archive_report.json`` for the chain diagnostics.

Names (``archive_names``) and thinning (``thinning_plan``) are pure functions
with tests; nothing else in the repository builds a name. Every rclone call
goes through ``Rclone``, so a local remote pointed at a directory runs the
whole thing end to end without Drive.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - import convenience
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_ROOT = "Fred_archive"
DEFAULT_REMOTE = "pythia_drive"
PUBLISH_WORKFLOW = "Publish Latest Data (Release)"
PUBLISH_FILE = "publish_latest_data.yml"
DB_ARTIFACT = "pythia-resolver-db"
DB_PREFIX = "pythia_resolver_db_"
DB_SUFFIX = ".duckdb.zst"
META_SUFFIX = ".db_snapshot.json"

#: Thinning (Part 5 of the brief): keep every snapshot of the last year, the
#: newest per calendar quarter before that, and never fewer than this many.
KEEP_ALL_DAYS = 365
MIN_SNAPSHOTS = 6

#: Drive space: warn and raise a blocking issue below either line.
MIN_FREE_BYTES = 20 * 1024 ** 3
MAX_USED_FRACTION = 0.85

#: Statuses a leg or the plan records per Publish run. The watchdog treats all
#: of ``ACCOUNTED`` as "this Publish run has been dealt with".
ACCOUNTED = frozenset({"archived", "already_present", "skipped_test", "skipped_not_published"})


# ---------------------------------------------------------------------------
# Names (pure)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ArchiveNames:
    kind: str
    year: str
    folder: str
    zip_name: str
    db_name: str | None
    db_meta_name: str | None

    @property
    def run_dir(self) -> str:
        return f"runs/{self.year}/{self.folder}"

    @property
    def db_dir(self) -> str:
        return f"db_snapshots/{self.year}"


_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def _clean(text: Any) -> str:
    return _SAFE.sub("_", str(text or "")).strip("_")


def archive_names(*, kind: str, start_date: str | dt.date, publish_run_id: str | int,
                  fc_run_id: str | None = None, sibyl_run_id: str | int | None = None,
                  is_test: bool = False) -> ArchiveNames:
    """Every name the archive writes for one chain. The only place names are made.

    ``start_date`` is the UTC date the chain's first run started, in full, so
    two runs in one month never collide; the run ids keep two chains started
    on one day apart. A test chain carries ``__TEST`` before the extension.
    """
    if isinstance(start_date, dt.date):
        date = start_date.isoformat()
    else:
        date = dt.date.fromisoformat(str(start_date)[:10]).isoformat()
    kind = kind if kind in {"forecast", "scoring"} else "other"
    gh = f"gh{_clean(publish_run_id)}"
    test = "__TEST" if is_test else ""
    if kind == "forecast":
        fc = _clean(fc_run_id) or "fc_unknown"
        ids = f"{fc}__{gh}"
    else:
        ids = gh
    folder = f"{date}__{kind}__{ids}{test}"
    zip_name = f"pythia_{date}__{kind}__{ids}{test}.zip"
    db_name = db_meta = None
    if kind == "forecast":
        fc = _clean(fc_run_id) or "fc_unknown"
        stem = f"{DB_PREFIX}{date}__{fc}__gh{_clean(sibyl_run_id) or 'unknown'}{test}"
        db_name = stem + DB_SUFFIX
        db_meta = stem + META_SUFFIX
    return ArchiveNames(kind=kind, year=date[:4], folder=folder, zip_name=zip_name,
                        db_name=db_name, db_meta_name=db_meta)


_SNAPSHOT_RE = re.compile(
    r"^pythia_resolver_db_(?P<date>\d{4}-\d{2}-\d{2})__(?P<fc>[^_].*?)__gh(?P<gh>[A-Za-z0-9]+)"
    r"(?P<test>__TEST)?\.duckdb\.zst$"
)


@dataclasses.dataclass(frozen=True)
class Snapshot:
    name: str
    date: dt.date
    gh: str
    is_test: bool

    @property
    def order(self) -> tuple:
        gh_num = int(self.gh) if self.gh.isdigit() else -1
        return (self.date, gh_num, self.name)

    @property
    def quarter(self) -> tuple[int, int]:
        return (self.date.year, (self.date.month - 1) // 3 + 1)


def parse_snapshot(name: str) -> Snapshot | None:
    m = _SNAPSHOT_RE.match(Path(name).name)
    if not m:
        return None
    try:
        date = dt.date.fromisoformat(m.group("date"))
    except ValueError:
        return None
    return Snapshot(name=Path(name).name, date=date, gh=m.group("gh"), is_test=bool(m.group("test")))


# ---------------------------------------------------------------------------
# Thinning (pure)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ThinningPlan:
    keep: list[str]
    delete: list[str]
    reasons: dict[str, str]
    unparsed: list[str]


def thinning_plan(names: Iterable[str], today: dt.date, *, keep_all_days: int = KEEP_ALL_DAYS,
                  min_snapshots: int = MIN_SNAPSHOTS) -> ThinningPlan:
    """Which DB snapshots to keep, by the date in the file name.

    * Every snapshot of the last ``keep_all_days`` days is kept (the boundary
      day itself counts as recent).
    * Older than that, only the newest snapshot of each calendar quarter is
      kept; two on one day are ordered by their GitHub run id.
    * A test snapshot (``__TEST``) is kept while recent and never stands for a
      quarter, and does not count towards the floor.
    * No deletion may leave fewer than ``min_snapshots`` production snapshots:
      the newest of the would-be deletions are kept back until the floor holds.
    * A name that does not parse is never deleted.
    """
    unparsed: list[str] = []
    snaps: list[Snapshot] = []
    for raw in names:
        s = parse_snapshot(raw)
        if s is None:
            unparsed.append(Path(raw).name)
        else:
            snaps.append(s)
    cutoff = today - dt.timedelta(days=keep_all_days)
    keep: dict[str, str] = {}
    delete: dict[str, str] = {}
    newest_in_quarter: dict[tuple[int, int], Snapshot] = {}
    for s in snaps:
        if s.date >= cutoff:
            keep[s.name] = f"within {keep_all_days} days"
        elif s.is_test:
            delete[s.name] = f"test snapshot older than {keep_all_days} days"
        else:
            cur = newest_in_quarter.get(s.quarter)
            if cur is None or s.order > cur.order:
                newest_in_quarter[s.quarter] = s
    for s in snaps:
        if s.date >= cutoff or s.is_test:
            continue
        q = f"{s.quarter[0]}-Q{s.quarter[1]}"
        if newest_in_quarter[s.quarter].name == s.name:
            keep[s.name] = f"newest of {q}"
        else:
            delete[s.name] = f"older than {keep_all_days} days and not the newest of {q}"
    by_name = {s.name: s for s in snaps}
    prod_kept = sum(1 for n in keep if not by_name[n].is_test)
    if prod_kept < min_snapshots:
        restorable = sorted((by_name[n] for n in delete if not by_name[n].is_test),
                            key=lambda s: s.order, reverse=True)
        for s in restorable[: min_snapshots - prod_kept]:
            delete.pop(s.name)
            keep[s.name] = f"kept back: the floor of {min_snapshots} snapshots"
    order = sorted(snaps, key=lambda s: s.order)
    return ThinningPlan(
        keep=[s.name for s in order if s.name in keep],
        delete=[s.name for s in order if s.name in delete],
        reasons={**keep, **delete},
        unparsed=unparsed,
    )


# ---------------------------------------------------------------------------
# rclone
# ---------------------------------------------------------------------------


class RcloneError(RuntimeError):
    pass


class Rclone:
    """Every Drive call. ``remote`` is the rclone remote name, ``root`` the folder."""

    def __init__(self, remote: str = DEFAULT_REMOTE, root: str = DEFAULT_ROOT, *,
                 binary: str = "rclone", extra: Sequence[str] = (),
                 runner: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> None:
        self.remote = remote
        self.root = root.strip("/")
        self.binary = binary
        self.extra = list(extra)
        self.runner = runner

    def path(self, rel: str = "") -> str:
        rel = rel.strip("/")
        base = f"{self.remote}:{self.root}"
        return f"{base}/{rel}" if rel else base

    def run(self, *args: str, check: bool = True, timeout: int = 7200) -> str:
        cmd = [self.binary, *args, *self.extra]
        proc = self.runner(cmd, capture_output=True, text=True, timeout=timeout)
        if check and proc.returncode != 0:
            raise RcloneError(f"rclone {args[0]} failed ({proc.returncode}): "
                              f"{(proc.stderr or proc.stdout or '').strip()[-600:]}")
        return proc.stdout or ""

    def root_folders(self) -> int:
        """How many folders named ``root`` rclone can see at the remote's top."""
        out = self.run("lsjson", f"{self.remote}:", "--dirs-only", "--max-depth", "1")
        try:
            entries = json.loads(out or "[]")
        except json.JSONDecodeError as exc:
            raise RcloneError(f"rclone lsjson returned no JSON: {exc}") from exc
        return sum(1 for e in entries if e.get("IsDir") and e.get("Name") == self.root)

    def assert_single_root(self) -> int:
        n = self.root_folders()
        if n > 1:
            raise RcloneError(
                f"{n} folders named '{self.root}' at the Drive root are visible to rclone. "
                "Uploading would split the archive between them; delete the extra one "
                "(keep the one holding runs/ and db_snapshots/) and run again.")
        return n

    def list_files(self, rel: str) -> list[str]:
        out = self.run("lsf", self.path(rel), "--files-only", "--recursive", check=False)
        return [ln.strip() for ln in out.splitlines() if ln.strip()]

    def exists(self, rel_file: str) -> bool:
        parent, _, name = rel_file.strip("/").rpartition("/")
        return name in {Path(f).name for f in self.list_files(parent) if "/" not in f}

    def copy_tree(self, local: Path) -> None:
        self.run("copy", str(local), self.path(), "--checksum")
        self.run("check", str(local), self.path(), "--one-way", "--checksum")

    def download(self, rel_file: str, local: Path) -> None:
        local.parent.mkdir(parents=True, exist_ok=True)
        self.run("copyto", self.path(rel_file), str(local), "--checksum")

    def upload_subtree(self, local_root: Path, rel_dir: str) -> None:
        """Copy ``local_root/rel_dir`` to the same place in Drive and check it,
        touching nothing outside ``rel_dir``."""
        src = local_root / rel_dir
        self.run("copy", str(src), self.path(rel_dir), "--checksum")
        self.run("check", str(src), self.path(rel_dir), "--one-way", "--checksum")

    def delete(self, rel_file: str) -> None:
        # Drive's default is to move to the trash (rclone drive use_trash=true),
        # so a mistake can be undone for 30 days.
        self.run("deletefile", self.path(rel_file))

    def about(self) -> dict[str, Any]:
        out = self.run("about", f"{self.remote}:", "--json")
        return json.loads(out or "{}")


# ---------------------------------------------------------------------------
# GitHub
# ---------------------------------------------------------------------------


def _gh_cls():
    from scripts.ci.build_cycle_diagnostics import Gh

    return Gh


def publish_run(gh: Any, repo: str, run_id: str) -> dict[str, Any] | None:
    try:
        return gh.api(f"repos/{repo}/actions/runs/{run_id}") or None
    except Exception:  # noqa: BLE001
        return None


def wait_until_complete(gh: Any, repo: str, run_id: str, info: dict[str, Any] | None, *,
                        wait_sec: float = 2700, poll_sec: float = 30,
                        sleep: Callable[[float], None] = time.sleep,
                        clock: Callable[[], float] = time.monotonic) -> dict[str, Any] | None:
    """The run's info once it has completed (Publish dispatches the archive as its last step)."""
    deadline = clock() + wait_sec
    while info and info.get("status") != "completed" and clock() < deadline:
        sleep(poll_sec)
        info = publish_run(gh, repo, run_id) or info
    return info


def recent_publish_runs(gh: Any, repo: str, days: int, now: dt.datetime) -> list[dict[str, Any]]:
    from scripts.ci.build_cycle_diagnostics import _ts, trusted

    out: list[dict[str, Any]] = []
    for page in range(1, 6):
        listing = gh.api(f"repos/{repo}/actions/workflows/{PUBLISH_FILE}/runs"
                         f"?per_page=100&page={page}") or {}
        runs = listing.get("workflow_runs") or []
        stop = False
        for r in runs:
            t = _ts(r.get("created_at"))
            if t is None:
                continue
            if (now - t).total_seconds() > days * 86400:
                stop = True
                continue
            if trusted(r, repo):
                out.append(r)
        if stop or len(runs) < 100:
            break
    return out


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


def resolve_chain(gh: Any, repo: str, publish_run_id: str, staging: Path) -> dict[str, Any]:
    """Chain facts for one Publish run: walk only (small lineage artifacts and
    the stage context), never the big bundles."""
    from scripts.ci import build_cycle_diagnostics as bcd

    staging.mkdir(parents=True, exist_ok=True)
    col = bcd.Collector(gh, repo, staging, wait_sec=0)
    chain = col.walk_chain(str(publish_run_id))
    col.add_pipeline_stages()
    ordered = sorted(col.runs.values(), key=lambda r: (r.created_at, r.run_id))
    ctx_dir = staging / "ctx"
    for rec in ordered:
        names: tuple[str, ...] = ()
        if rec.workflow == "Pythia Pipeline Stage" and "fc_collect_finalize" in str(
                rec.info.get("display_title") or ""):
            names = ("pythia-stage-context",)
        elif rec.workflow in ("Sibyl Deep Research", "Horizon Scanner Triage"):
            names = ("pythia-health-report", "pythia-stage-context")
        for name in names:
            arts = col.artifacts_of(rec.run_id)
            if any(a.get("name") == name and not a.get("expired") for a in arts):
                gh.download(rec.run_id, name, ctx_dir / rec.run_id / name)
    facts = bcd.chain_facts(ordered, staging)
    facts["runs"] = [{"run_id": r.run_id, "workflow": r.workflow, "reason": r.reason}
                     for r in ordered]
    facts["chain_length"] = len(chain)
    shutil.rmtree(staging, ignore_errors=True)
    return facts


def plan_one(gh: Any, repo: str, rclone: Rclone | None, publish_run_id: str, *,
             include_test: bool, work: Path, wait: bool = False,
             sleep: Callable[[float], None] = time.sleep) -> dict[str, Any]:
    """One Publish run: what it is, what it would be named, and whether to archive it."""
    rec: dict[str, Any] = {"publish_run_id": str(publish_run_id)}
    info = publish_run(gh, repo, str(publish_run_id))
    if not info:
        rec.update(status="failed", reason="Publish run unreadable")
        return rec
    if wait:
        info = wait_until_complete(gh, repo, str(publish_run_id), info, sleep=sleep) or info
    rec["publish_conclusion"] = info.get("conclusion")
    rec["publish_created_at"] = info.get("created_at")
    if info.get("status") == "completed" and info.get("conclusion") not in ("success", None):
        rec.update(status="skipped_not_published",
                   reason=f"Publish run concluded {info.get('conclusion')}")
        return rec
    facts = resolve_chain(gh, repo, str(publish_run_id), work / f"plan_{publish_run_id}")
    rec["chain"] = facts
    if not facts.get("start_date"):
        rec.update(status="failed", reason="no chain start date could be read")
        return rec
    is_test = bool(facts.get("is_test"))
    names = archive_names(kind=facts["kind"], start_date=facts["start_date"],
                          publish_run_id=publish_run_id, fc_run_id=facts.get("fc_run_id"),
                          sibyl_run_id=facts.get("sibyl_run_id"), is_test=is_test)
    rec["names"] = dataclasses.asdict(names)
    rec["is_test"] = is_test
    if is_test and not include_test:
        rec.update(status="skipped_test", reason="test-mode chain (include_test is false)")
        return rec
    if rclone is not None:
        wanted = [f"{names.run_dir}/{names.zip_name}"]
        if names.db_name:
            wanted.append(f"{names.db_dir}/{names.db_name}")
        present = [w for w in wanted if rclone.exists(w)]
        if len(present) == len(wanted):
            rec.update(status="already_present", reason="every file is already in Drive",
                       drive_paths=wanted)
            return rec
        rec["already_in_drive"] = present
    rec["status"] = "todo"
    return rec


def cmd_plan(args: argparse.Namespace, *, gh: Any = None, rclone: Rclone | None = None,
             now: dt.datetime | None = None) -> int:
    gh = gh or _gh_cls()(args.repo)
    if rclone is None and not args.no_drive:
        rclone = Rclone(args.remote, args.root)
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    if rclone is not None:
        rclone.assert_single_root()
    now = now or dt.datetime.now(dt.timezone.utc)
    if args.publish_run_id:
        ids = [str(args.publish_run_id)]
        wait = True
    else:
        runs = recent_publish_runs(gh, args.repo, int(args.backfill_days or 80), now)
        ids = [str(r["id"]) for r in sorted(runs, key=lambda r: str(r.get("created_at")))]
        wait = False
    records = []
    for pid in ids:
        try:
            records.append(plan_one(gh, args.repo, rclone, pid, include_test=args.include_test,
                                    work=work, wait=wait))
        except Exception as exc:  # noqa: BLE001 - one bad run never stops the rest
            records.append({"publish_run_id": pid, "status": "failed",
                            "reason": f"{type(exc).__name__}: {exc}"})
    todo = [r for r in records if r.get("status") == "todo"]
    matrix = {"include": [{"publish_run_id": r["publish_run_id"], "spec": json.dumps(r)}
                          for r in todo]}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"records": records}, indent=2, default=str) + "\n",
                              encoding="utf-8")
    _output("matrix", json.dumps(matrix))
    _output("count", str(len(todo)))
    for r in records:
        print(f"{r['publish_run_id']}: {r.get('status')} {r.get('reason') or ''}".rstrip())
    return 0


def _output(key: str, value: str) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(f"{key}={value}\n")


# ---------------------------------------------------------------------------
# Leg: one archive
# ---------------------------------------------------------------------------

#: Files lifted out of the zip and laid beside it, so a reader (or a Claude
#: session through the Drive connector) can read the archive without opening
#: zips. Top-level files by exact name; the rest by glob anywhere in the tree.
LOOSE_TOP = ("README.md", "CYCLE_SUMMARY.md", "manifest.json")
LOOSE_GLOBS = (
    "executive_summary__*.md", "health_report__*.json", "health_report__*.md",
    "report__*.md", "report_*_v*.md", "report__*.pdf", "test_report__*.pdf",
    "scored_report*.md", "scored_report*.pdf",
)


def loose_files(staging: Path) -> list[Path]:
    """The files to lay beside the zip. Interpreter reports come from any
    ``pythia-interpreter-*`` folder; ``interpreter_report_latest.pdf`` is a
    copy of a versioned PDF and is left out."""
    out: list[Path] = [staging / n for n in LOOSE_TOP if (staging / n).is_file()]
    seen = {p.resolve() for p in out}
    for pat in LOOSE_GLOBS:
        for p in sorted(staging.rglob(pat)):
            if p.is_file() and p.resolve() not in seen:
                out.append(p)
                seen.add(p.resolve())
    for p in sorted(staging.rglob("*")):
        if (p.is_file() and p.suffix in (".md", ".pdf")
                and any(part.startswith("pythia-interpreter-") for part in p.relative_to(staging).parts)
                and p.name != "interpreter_report_latest.pdf" and p.resolve() not in seen):
            out.append(p)
            seen.add(p.resolve())
    return out


def _loose_name(path: Path, staging: Path) -> str:
    """A flat, unique name for a loose file: the run folder's number prefixes it."""
    rel = path.relative_to(staging)
    if len(rel.parts) == 1:
        return rel.name
    prefix = rel.parts[0].split("_", 1)[0]
    return f"{prefix}__{rel.name}"


# ---------------------------------------------------------------------------
# The scored bundle, unzipped beside the scoring chain's zip
# ---------------------------------------------------------------------------

#: The monthly performance report is written by a Claude session that reads
#: Drive through the connector: it reads text, not a 15 MB zip nested inside
#: another zip. So the files the report reads are lifted out of the scored
#: bundle into ``scored_bundle/`` beside the chain zip.
SCORED_BUNDLE_GLOB = "scored_forecast_analysis__*.zip"
SCORED_BUNDLE_DIR = "scored_bundle"
SCORED_BUNDLE_FILES = (
    "digest.md", "headline.json", "manifest.json", "ANALYST_GUIDE.md",
    "monthly_report_inputs.json",
    "questions_index.csv", "skill_history.csv", "indicative_questions.csv",
    "resolution_sources.csv", "unasked_outcomes.csv", "binary_reliability.csv",
    "experiments.csv", "advice_experiment.csv", "recalibration_effect.csv",
    "calibration_status.csv", "calibration_weights.csv", "calibration_advice.md",
    "trace_stages_summary.csv", "update_value_summary.csv", "rc_outcomes_summary.csv",
)
SCORED_BUNDLE_GLOBS = ("case_studies_new/*.json",)
SCORED_FILE_CAP = 1024 * 1024
SCORED_TOTAL_CAP = 10 * 1024 * 1024
#: Files a bundle built before monthly_report_inputs (Nov 2026) cannot carry.
SCORED_BUNDLE_LATER_ADDITIONS = {
    "monthly_report_inputs.json": "added to the scored bundle after this bundle was built",
    "case_studies_new/*.json": "added to the scored bundle after this bundle was built",
}


def find_scored_bundle(staging: Path) -> Path | None:
    """The scored bundle zip inside a chain's staging tree (newest name), or None."""
    found = sorted(p for p in staging.rglob(SCORED_BUNDLE_GLOB) if p.is_file())
    return found[-1] if found else None


def _allowed(name: str) -> bool:
    import fnmatch

    return name in SCORED_BUNDLE_FILES or any(fnmatch.fnmatchcase(name, g) for g in SCORED_BUNDLE_GLOBS)


def extract_scored_bundle(bundle_zip: Path, dest: Path, *, file_cap: int = SCORED_FILE_CAP,
                          total_cap: int = SCORED_TOTAL_CAP,
                          env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Lift the allowlisted files out of ``bundle_zip`` into ``dest``.

    A file over ``file_cap``, or one that would take the folder past
    ``total_cap``, is skipped and named in ``SKIPPED.md`` with its size, as is
    any allowlisted file the bundle does not hold. The folder is scanned for
    every credential in ``env`` before it is returned (raises ``SecretLeak``).
    """
    import zipfile

    from scripts.ci.build_cycle_diagnostics import scan_for_secrets

    dest.mkdir(parents=True, exist_ok=True)
    written: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    total = 0
    with zipfile.ZipFile(bundle_zip) as zf:
        infos = {i.filename: i for i in zf.infolist() if not i.is_dir()}
        chosen = sorted((n for n in infos if _allowed(n)),
                        key=lambda n: (n not in SCORED_BUNDLE_FILES, n))
        for name in chosen:
            size = infos[name].file_size
            if size > file_cap:
                skipped.append({"file": name, "bytes": size, "reason": f"over the {file_cap} byte per-file cap"})
                continue
            if total + size > total_cap:
                skipped.append({"file": name, "bytes": size,
                                "reason": f"would take the folder past the {total_cap} byte cap"})
                continue
            target = dest / name
            if Path(name).is_absolute() or ".." in Path(name).parts:
                skipped.append({"file": name, "bytes": size, "reason": "unsafe path"})
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(name))
            total += size
            written.append({"file": name, "bytes": size})
        present = set(infos)
    absent = [n for n in SCORED_BUNDLE_FILES if n not in present]
    if not any(n.startswith("case_studies_new/") for n in present):
        absent.append("case_studies_new/*.json")
    for name in absent:
        skipped.append({"file": name, "bytes": None,
                        "reason": SCORED_BUNDLE_LATER_ADDITIONS.get(name, "not in this bundle")})
    if skipped:
        lines = ["# Files not copied out of the scored bundle", "",
                 f"Source: `{bundle_zip.name}`. Caps: {file_cap} bytes per file, "
                 f"{total_cap} bytes in all.", "",
                 "| File | Bytes | Why |", "|---|---|---|"]
        lines += [f"| `{s['file']}` | {s['bytes'] if s['bytes'] is not None else '-'} | {s['reason']} |"
                  for s in skipped]
        (dest / "SKIPPED.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    scan_for_secrets(dest, dict(os.environ if env is None else env))
    return {"folder": SCORED_BUNDLE_DIR, "source": bundle_zip.name, "files": written,
            "skipped": skipped, "total_bytes": total}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot_db(db_path: Path, out_zst: Path, *, level: int = 10,
                runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
                connect: Callable[[str], Any] | None = None,
                verify_dir: Path | None = None) -> dict[str, Any]:
    """Compress the DB with zstd, then prove the copy: decompress it, compare
    SHA-256, open it read-only and count three tables. The original is deleted
    once compressed, so the peak is the DB plus its compressed copy."""
    raw_bytes = db_path.stat().st_size
    digest = sha256_file(db_path)
    out_zst.parent.mkdir(parents=True, exist_ok=True)
    runner(["zstd", "-T0", f"-{level}", "-q", "-f", str(db_path), "-o", str(out_zst)],
           check=True, capture_output=True, text=True)
    db_path.unlink()
    check_path = Path(str(verify_dir or out_zst.parent)) / (out_zst.name[: -len(".zst")] + ".verify")
    try:
        runner(["zstd", "-d", "-q", "-f", str(out_zst), "-o", str(check_path)],
               check=True, capture_output=True, text=True)
        verify_digest = sha256_file(check_path)
        if verify_digest != digest:
            raise RuntimeError(f"decompressed copy does not match: {verify_digest} != {digest}")
        if connect is None:
            import duckdb

            def connect(p: str) -> Any:
                return duckdb.connect(p, read_only=True)
        con = connect(str(check_path))
        counts: dict[str, Any] = {}
        try:
            for table in ("forecasts_raw", "llm_calls", "scores"):
                try:
                    counts[table] = int(con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
                except Exception as exc:  # noqa: BLE001
                    counts[table] = f"unreadable: {type(exc).__name__}"
        finally:
            con.close()
    finally:
        if check_path.exists():
            check_path.unlink()
    return {
        "uncompressed_bytes": raw_bytes,
        "compressed_bytes": out_zst.stat().st_size,
        "sha256_uncompressed": digest,
        "sha256_compressed": sha256_file(out_zst),
        "zstd_level": level,
        "verified": True,
        "row_counts": counts,
    }


def _artifact_bytes(gh: Any, repo: str, run_id: str, name: str) -> int | None:
    try:
        listing = gh.api(f"repos/{repo}/actions/runs/{run_id}/artifacts?per_page=100") or {}
    except Exception:  # noqa: BLE001
        return None
    for a in listing.get("artifacts") or []:
        if a.get("name") == name and not a.get("expired"):
            return int(a.get("size_in_bytes") or 0)
    return None


def download_db(gh: Any, repo: str, sibyl_run_id: str, dest: Path,
                free_bytes: Callable[[str], int] | None = None) -> Path:
    """The Sibyl run's canonical DB, with the room checks db_growth uses."""
    from scripts.ci import db_growth

    free_bytes = free_bytes or db_growth.free_bytes
    dest.mkdir(parents=True, exist_ok=True)
    art = _artifact_bytes(gh, repo, sibyl_run_id, DB_ARTIFACT)
    if art is None:
        raise RuntimeError(f"run {sibyl_run_id} holds no unexpired {DB_ARTIFACT} artifact")
    ok, msg = db_growth.check_room(art, free_bytes(str(dest)), 1.0)
    print(f"[archive] before download: artifact {msg}")
    if not ok:
        raise RuntimeError(f"no room to download the DB artifact: {msg}")
    ok_dl, why = gh.download(str(sibyl_run_id), DB_ARTIFACT, dest)
    if not ok_dl:
        raise RuntimeError(f"DB download failed: {why}")
    found = sorted(dest.rglob("resolver.duckdb"))
    if not found:
        raise RuntimeError(f"no resolver.duckdb in the {DB_ARTIFACT} artifact")
    db = found[0]
    ok, msg = db_growth.check_room(db.stat().st_size, free_bytes(str(dest)), 0.5)
    print(f"[archive] after download: {msg}")
    if not ok:
        raise RuntimeError(f"no room to compress and verify the DB: {msg}")
    return db


def run_leg(spec: Mapping[str, Any], *, gh: Any, repo: str, rclone: Rclone, work: Path,
            out_dir: Path, env: Mapping[str, str] | None = None,
            snapshot: Callable[..., dict[str, Any]] = snapshot_db,
            fetch_db: Callable[..., Path] = download_db) -> dict[str, Any]:
    """Archive one chain. Returns the leg report; raises only on a credential leak."""
    from scripts.ci import build_cycle_diagnostics as bcd

    pid = str(spec["publish_run_id"])
    names = ArchiveNames(**spec["names"])
    report: dict[str, Any] = {"publish_run_id": pid, "names": dataclasses.asdict(names),
                              "chain": spec.get("chain"), "status": "failed"}
    rclone.assert_single_root()
    upload = work / "upload"
    shutil.rmtree(upload, ignore_errors=True)
    run_dir = upload / names.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = bcd.build(gh=gh, repo=repo, source_run_id=pid, publish_run_id=pid,
                         out_dir=out_dir, full=True, zip_name=names.zip_name, wait_sec=600,
                         env=env, keep_staging=True)
    staging = Path(manifest["staging"])
    zip_path = Path(manifest["zip"])
    report["zip"] = {"name": names.zip_name, "bytes": manifest["compressed_bytes"],
                     "uncompressed_bytes": sum(int(f["bytes"]) for f in manifest["files"]),
                     "files": len(manifest["files"]), "missing": manifest["missing"]}
    shutil.copy2(zip_path, run_dir / names.zip_name)
    loose = []
    for path in loose_files(staging):
        target = _loose_name(path, staging)
        shutil.copy2(path, run_dir / target)
        loose.append(target)
    report["loose_files"] = loose
    bundle = find_scored_bundle(staging)
    if bundle is not None:
        report["scored_bundle"] = extract_scored_bundle(bundle, run_dir / SCORED_BUNDLE_DIR, env=env)
    shutil.rmtree(staging, ignore_errors=True)
    if names.db_name and spec.get("chain", {}).get("sibyl_run_id"):
        sibyl = str(spec["chain"]["sibyl_run_id"])
        db_dir = upload / names.db_dir
        db = fetch_db(gh, repo, sibyl, work / "db")
        meta = snapshot(db, db_dir / names.db_name, verify_dir=work / "db")
        meta.update({"name": names.db_name, "sibyl_run_id": sibyl,
                     "fc_run_id": spec["chain"].get("fc_run_id"),
                     "publish_run_id": pid,
                     "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")})
        (db_dir / names.db_meta_name).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        report["db_snapshot"] = meta
        shutil.rmtree(work / "db", ignore_errors=True)
    elif names.db_name:
        report["db_snapshot"] = {"skipped": "no Sibyl run id in the chain"}
    rclone.copy_tree(upload)
    report["drive_paths"] = sorted(
        f"{rclone.root}/{p.relative_to(upload).as_posix()}" for p in upload.rglob("*") if p.is_file())
    report["status"] = "archived"
    shutil.rmtree(upload, ignore_errors=True)
    return report


def cmd_leg(args: argparse.Namespace, *, gh: Any = None, rclone: Rclone | None = None) -> int:
    spec = json.loads(args.spec or os.environ.get("ARCHIVE_SPEC", "{}"))
    gh = gh or _gh_cls()(args.repo)
    rclone = rclone or Rclone(args.remote, args.root)
    report: dict[str, Any]
    try:
        report = run_leg(spec, gh=gh, repo=args.repo, rclone=rclone, work=Path(args.work_dir),
                         out_dir=Path(args.out_dir))
        code = 0
    except Exception as exc:  # noqa: BLE001
        from scripts.ci.build_cycle_diagnostics import SecretLeak

        report = {"publish_run_id": spec.get("publish_run_id"), "names": spec.get("names"),
                  "chain": spec.get("chain"), "status": "failed",
                  "reason": f"{type(exc).__name__}: {exc}"}
        code = 2 if isinstance(exc, SecretLeak) else 1
        print(f"::error title=Drive archive failed::{report['reason']}")
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    zip_name = (report.get("names") or {}).get("zip_name") or ""
    _output("zip_name", zip_name)
    _output("artifact_name", "pythia-archive-" + Path(zip_name).stem if zip_name else "")
    print(json.dumps({k: report.get(k) for k in ("status", "zip", "db_snapshot", "drive_paths")},
                     indent=2, default=str))
    return code


def nested_scored_bundle(chain_zip: Path, dest_dir: Path) -> Path | None:
    """Copy the scored bundle zip out of an archived chain zip (where it sits
    as ``<nn>_compute_calibration__<run>/pythia-ai-analysis-bundle/...zip``)."""
    import fnmatch
    import zipfile

    with zipfile.ZipFile(chain_zip) as zf:
        names = sorted(n for n in zf.namelist()
                       if fnmatch.fnmatchcase(Path(n).name, SCORED_BUNDLE_GLOB))
        if not names:
            return None
        dest_dir.mkdir(parents=True, exist_ok=True)
        out = dest_dir / Path(names[-1]).name
        out.write_bytes(zf.read(names[-1]))
        return out


def backfill_scored_bundle(spec: Mapping[str, Any], *, rclone: Rclone, work: Path,
                           env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Add the missing ``scored_bundle/`` folder to a chain already in Drive.

    Reads the chain zip back from Drive (so an expired GitHub artifact does not
    matter), takes the scored bundle out of it, and uploads ONLY
    ``<run_dir>/scored_bundle/``. The zip and the loose files beside it are
    never replaced; a folder already in Drive is left alone.
    """
    names = ArchiveNames(**spec["names"])
    pid = str(spec.get("publish_run_id"))
    report: dict[str, Any] = {"publish_run_id": pid, "names": dataclasses.asdict(names),
                              "mode": "scored_bundle_only", "status": "failed"}
    rclone.assert_single_root()
    zip_rel = f"{names.run_dir}/{names.zip_name}"
    if not rclone.exists(zip_rel):
        report["reason"] = f"{zip_rel} is not in Drive; run the normal archive leg instead"
        return report
    folder_rel = f"{names.run_dir}/{SCORED_BUNDLE_DIR}"
    already = rclone.list_files(folder_rel)
    if already:
        report.update(status="already_present", reason=f"{folder_rel} already holds {len(already)} file(s)")
        return report
    shutil.rmtree(work, ignore_errors=True)
    local_zip = work / "chain" / names.zip_name
    rclone.download(zip_rel, local_zip)
    bundle = nested_scored_bundle(local_zip, work / "bundle")
    if bundle is None:
        report.update(status="no_scored_bundle", reason="the chain zip holds no scored bundle")
        shutil.rmtree(work, ignore_errors=True)
        return report
    upload = work / "upload"
    report["scored_bundle"] = extract_scored_bundle(bundle, upload / folder_rel, env=env)
    rclone.upload_subtree(upload, folder_rel)
    report["drive_paths"] = sorted(
        f"{rclone.root}/{p.relative_to(upload).as_posix()}" for p in upload.rglob("*") if p.is_file())
    report["status"] = "archived"
    shutil.rmtree(work, ignore_errors=True)
    return report


def cmd_scored_bundle(args: argparse.Namespace, *, gh: Any = None, rclone: Rclone | None = None) -> int:
    gh = gh or _gh_cls()(args.repo)
    rclone = rclone or Rclone(args.remote, args.root)
    work = Path(args.work_dir)
    try:
        spec = plan_one(gh, args.repo, None, str(args.publish_run_id), include_test=True, work=work)
        if spec.get("status") != "todo":
            report = {**spec, "mode": "scored_bundle_only"}
            code = 1 if spec.get("status") == "failed" else 0
        else:
            report = backfill_scored_bundle(spec, rclone=rclone, work=work / "scored")
            code = 1 if report.get("status") == "failed" else 0
    except Exception as exc:  # noqa: BLE001
        from scripts.ci.build_cycle_diagnostics import SecretLeak

        report = {"publish_run_id": str(args.publish_run_id), "mode": "scored_bundle_only",
                  "status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
        code = 2 if isinstance(exc, SecretLeak) else 1
    if report.get("status") == "failed":
        print(f"::error title=Drive scored-bundle backfill failed::{report.get('reason')}")
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: report.get(k) for k in ("status", "reason", "scored_bundle", "drive_paths")},
                     indent=2, default=str))
    return code


# ---------------------------------------------------------------------------
# Housekeeping: thinning, space, the merged report
# ---------------------------------------------------------------------------


def storage_verdict(about: Mapping[str, Any]) -> dict[str, Any]:
    used, total, free = about.get("used"), about.get("total"), about.get("free")
    if free is None and total is not None and used is not None:
        free = int(total) - int(used)
    frac = (int(used) / int(total)) if used is not None and total else None
    low = (free is not None and int(free) < MIN_FREE_BYTES) or (
        frac is not None and frac > MAX_USED_FRACTION)
    return {"used_bytes": used, "total_bytes": total, "free_bytes": free,
            "used_fraction": round(frac, 4) if frac is not None else None,
            "low": bool(low),
            "thresholds": {"min_free_bytes": MIN_FREE_BYTES, "max_used_fraction": MAX_USED_FRACTION}}


def thin_snapshots(rclone: Rclone, today: dt.date, *, apply: bool = True) -> dict[str, Any]:
    files = rclone.list_files("db_snapshots")
    by_name = {Path(f).name: f for f in files}
    plan = thinning_plan([n for n in by_name if n.endswith(DB_SUFFIX)], today)
    deleted = []
    for name in plan.delete:
        rel = f"db_snapshots/{by_name[name]}"
        meta_rel = rel[: -len(DB_SUFFIX)] + META_SUFFIX
        if apply:
            rclone.delete(rel)
            if Path(meta_rel).name in by_name:
                rclone.delete(meta_rel)
        deleted.append({"name": name, "reason": plan.reasons[name]})
    return {"kept": [{"name": n, "reason": plan.reasons[n]} for n in plan.keep],
            "deleted": deleted, "unparsed": plan.unparsed, "applied": apply}


def cmd_housekeeping(args: argparse.Namespace, *, rclone: Rclone | None = None,
                     today: dt.date | None = None) -> int:
    rclone = rclone or Rclone(args.remote, args.root)
    today = today or dt.date.today()
    report: dict[str, Any] = {"recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                              "drive_root": rclone.root}
    plan_records = []
    if args.plan and Path(args.plan).is_file():
        plan_records = (json.loads(Path(args.plan).read_text(encoding="utf-8")) or {}).get("records") or []
    legs: dict[str, dict[str, Any]] = {}
    for path in sorted(Path(args.legs_dir).rglob("*.json")) if args.legs_dir else []:
        try:
            leg = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if leg.get("publish_run_id"):
            legs[str(leg["publish_run_id"])] = leg
    entries = []
    for rec in plan_records:
        pid = str(rec.get("publish_run_id"))
        leg = legs.pop(pid, None)
        if rec.get("status") == "todo":
            entries.append(leg or {**rec, "status": "failed",
                                   "reason": "planned, but the archive leg left no report"})
        else:
            entries.append(rec)
    entries.extend(legs.values())
    report["publish_runs"] = entries
    try:
        rclone.assert_single_root()
        report["thinning"] = thin_snapshots(rclone, today, apply=not args.dry_run)
    except Exception as exc:  # noqa: BLE001
        report["thinning"] = {"error": f"{type(exc).__name__}: {exc}"}
    try:
        report["storage"] = storage_verdict(rclone.about())
    except Exception as exc:  # noqa: BLE001
        report["storage"] = {"error": f"{type(exc).__name__}: {exc}", "low": None}
    st = report["storage"]
    if st.get("low"):
        print(f"::warning title=Drive space low::{st.get('free_bytes')} bytes free, "
              f"{st.get('used_fraction')} used (limits: 20 GB free, 85% used)")
    for e in entries:
        if e.get("status") == "failed":
            print(f"::warning title=Drive archive::Publish run {e.get('publish_run_id')} "
                  f"not archived: {e.get('reason')}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"publish_runs": [(e.get("publish_run_id"), e.get("status")) for e in entries],
                      "thinning_deleted": [d["name"] for d in report["thinning"].get("deleted", [])],
                      "storage": st}, indent=2, default=str))
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", ""))
    parser.add_argument("--remote", default=os.environ.get("RCLONE_REMOTE", DEFAULT_REMOTE))
    parser.add_argument("--root", default=os.environ.get("DRIVE_ROOT", DEFAULT_ROOT))
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--publish-run-id", default="")
    p.add_argument("--backfill-days", default="")
    p.add_argument("--include-test", action="store_true")
    p.add_argument("--no-drive", action="store_true", help="do not ask Drive what is there")
    p.add_argument("--work-dir", default="archive_work")
    p.add_argument("--out", default="diagnostics/archive_plan.json")
    p = sub.add_parser("leg")
    p.add_argument("--spec", default="")
    p.add_argument("--work-dir", default="archive_work")
    p.add_argument("--out-dir", default="archive_out")
    p.add_argument("--report", default="archive_leg/leg.json")
    p = sub.add_parser("scored-bundle",
                       help="add only the missing scored_bundle/ folder to a chain already in Drive")
    p.add_argument("--publish-run-id", required=True)
    p.add_argument("--work-dir", default="archive_work")
    p.add_argument("--report", default="archive_leg/scored_bundle.json")
    p = sub.add_parser("housekeeping")
    p.add_argument("--plan", default="diagnostics/archive_plan.json")
    p.add_argument("--legs-dir", default="archive_legs")
    p.add_argument("--dry-run", action="store_true", help="plan the thinning, delete nothing")
    p.add_argument("--out", default="diagnostics/archive_report.json")
    args = parser.parse_args(argv)
    if args.cmd == "plan":
        return cmd_plan(args)
    if args.cmd == "leg":
        return cmd_leg(args)
    if args.cmd == "scored-bundle":
        return cmd_scored_bundle(args)
    return cmd_housekeeping(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
