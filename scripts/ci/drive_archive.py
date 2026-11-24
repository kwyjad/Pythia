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
                                                bundle's report files, unzipped);
                                                a chain with a Resolver Update
                                                also gets resolver_review/ (the
                                                resolver review's inputs file
                                                and the files it reads); a
                                                forecast chain also gets
                                                forecast_review/ (the forecast
                                                pipeline review's inputs file
                                                and the files it reads)
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
``resolver-review`` add only the missing ``resolver_review/`` folder the
                 same way.
``forecast-review`` add only the missing ``forecast_review/`` folder the
                 same way (and the workflow-log index, from the API).
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

    def purge_dir(self, rel_dir: str) -> None:
        """Remove ``rel_dir`` and everything in it (through the Drive trash)."""
        self.run("purge", self.path(rel_dir))

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


# ---------------------------------------------------------------------------
# The resolver review's inputs, unzipped beside a chain with a Resolver Update
# ---------------------------------------------------------------------------

#: The monthly resolver review is written by the same kind of session as the
#: performance report, reading Drive as text. Its inputs sit mostly in the
#: resolver debug bundle, a zip inside the chain zip; this folder holds a
#: summary of them (``resolver_review_inputs.json``) and the files it reads.
RESOLVER_REVIEW_DIR = "resolver_review"
RESOLVER_REVIEW_INPUTS = "resolver_review_inputs.json"
#: Lifted from the debug bundle, by path inside it. Never
#: ``hazard/cell_ledger.csv`` (29 MB in November 2026): its counts are in the
#: inputs file.
RESOLVER_REVIEW_BUNDLE_FILES = (
    "README.md", "checks/issues.md", "checks/contradictions.md", "checks/reconciliation.md",
    "logs/log_index.md", "db/freshness.csv", "db/signature_before.json",
    "db/signature_after.json", "db/queries/crisiswatch_editions.csv",
    "db/queries/conflict_forecast_vintages.csv", "hazard/acceptance_report.md",
    "hazard/figures_ledger.csv", "hazard/extraction_budget.csv",
    "hazard/backcast_progress.csv", "hazard/conflict_displacement/cell_ledger.csv",
    "hazard/pa_no_data_breakdown.json",
    "config/connectors/known_issues.yml", "run/step_timings.csv", "run/workflow_run.json",
)
RESOLVER_REVIEW_NEVER = ("hazard/cell_ledger.csv",)
#: Files lifted from the post-ingest stages' chain diagnostics, renamed
#: ``<stage>__<file>``. ``reports/*.json`` takes every report.
RESOLVER_REVIEW_STAGE_FILES: dict[str, tuple[str, ...]] = {
    "compute_resolutions": ("summary.md", "issues.md", "reports/compute_resolutions.json"),
    "compute_scores": ("summary.md", "issues.md", "reports/*.json"),
    "compute_calibration": ("summary.md", "issues.md", "centroid_ema_updates.csv",
                            "family_recalibration.csv", "calibration_weights.csv",
                            "sibyl_calibration_steps.json"),
    "publish_latest_data": ("summary.md", "issues.md"),
}


def _resolver_review_plan(src: Any) -> list[tuple[str, Path | None]]:
    """(name in the folder, source path or None when absent), in order."""
    plan: list[tuple[str, Path | None]] = []
    for rel in RESOLVER_REVIEW_BUNDLE_FILES:
        plan.append((rel.replace("/", "__"), src.bundle(rel)))
    for slug, patterns in RESOLVER_REVIEW_STAGE_FILES.items():
        d = src.chain_diag.get(slug)
        for pat in patterns:
            if d is not None and "*" in pat:
                found = sorted(d.glob(pat))
                if found:
                    plan += [(f"{slug}__{p.name}", p) for p in found]
                    continue
            path = (d / pat) if d is not None else None
            plan.append((f"{slug}__{Path(pat).name}", path if path is not None and path.is_file() else None))
    plan.append(("resolver_inspect.md", src.resolver_inspect))
    return plan


def extract_resolver_review(staging: Path, dest: Path, work: Path, *,
                            file_cap: int = SCORED_FILE_CAP, total_cap: int = SCORED_TOTAL_CAP,
                            env: Mapping[str, str] | None = None) -> dict[str, Any] | None:
    """Write ``resolver_review/`` for a chain holding a Resolver Update stage
    (None for one that holds none). Same caps as ``scored_bundle/``: a file
    over ``file_cap``, or one that would take the folder past ``total_cap``,
    is named in ``SKIPPED.md`` with its size, as is any listed file absent.
    The inputs file always goes in first. Scanned for credentials (raises
    ``SecretLeak``)."""
    from scripts.ci.build_cycle_diagnostics import scan_for_secrets
    from scripts.ci import resolver_review_inputs as rri

    src = rri.locate(staging, work)
    if not src.has_resolver_update:
        return None
    dest.mkdir(parents=True, exist_ok=True)
    rri.write_inputs_from(src, dest / RESOLVER_REVIEW_INPUTS)
    total = (dest / RESOLVER_REVIEW_INPUTS).stat().st_size
    written = [{"file": RESOLVER_REVIEW_INPUTS, "bytes": total}]
    skipped: list[dict[str, Any]] = []
    for name, path in _resolver_review_plan(src):
        if path is None or not Path(path).is_file():
            skipped.append({"file": name, "bytes": None, "reason": "not in this chain"})
            continue
        size = Path(path).stat().st_size
        if size > file_cap:
            skipped.append({"file": name, "bytes": size, "reason": f"over the {file_cap} byte per-file cap"})
            continue
        if total + size > total_cap:
            skipped.append({"file": name, "bytes": size,
                            "reason": f"would take the folder past the {total_cap} byte cap"})
            continue
        shutil.copy2(path, dest / name)
        total += size
        written.append({"file": name, "bytes": size})
    for rel in RESOLVER_REVIEW_NEVER:
        skipped.append({"file": rel.replace("/", "__"), "bytes": None,
                        "reason": f"never copied: its counts are in {RESOLVER_REVIEW_INPUTS}"})
    lines = ["# Files not copied into resolver_review/", "",
             f"Caps: {file_cap} bytes per file, {total_cap} bytes in all.", "",
             "| File | Bytes | Why |", "|---|---|---|"]
    lines += [f"| `{s['file']}` | {s['bytes'] if s['bytes'] is not None else '-'} | {s['reason']} |"
              for s in skipped]
    (dest / "SKIPPED.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    scan_for_secrets(dest, dict(os.environ if env is None else env))
    shutil.rmtree(work, ignore_errors=True)
    return {"folder": RESOLVER_REVIEW_DIR, "debug_bundle": src.debug_zip.name if src.debug_zip else None,
            "files": written, "skipped": skipped, "total_bytes": total}


# ---------------------------------------------------------------------------
# The forecast pipeline review's inputs, unzipped beside a forecast chain
# ---------------------------------------------------------------------------

#: The forecast pipeline review (``forecast_review_template.md`` in Drive) is
#: written after every forecast chain by a session reading Drive as text. Its
#: inputs sit in three zips nested in the chain zip, the debug bundle alone
#: 20 MB in November 2026; this folder holds a summary of them
#: (``forecast_review_inputs.json``) and the files the review reads.
FORECAST_REVIEW_DIR = "forecast_review"
FORECAST_REVIEW_INPUTS = "forecast_review_inputs.json"
#: Lifted from the debug bundle, by stem (``<stem>__<run>.<ext>`` at its top),
#: named ``debug__<file>``. Never ``llm_calls_detail``, ``spd_tables``,
#: ``provider_batch_objects`` or the workflow logs: their figures are in the
#: inputs file.
FORECAST_REVIEW_DEBUG_FILES = (
    "anomalies", "executive_summary", "health_report", "timing_breakdown", "batch_lifecycle",
    "retry_report", "model_completeness_rollup", "run_comparison", "connector_freshness",
    "crisiswatch_inject", "data_inject_inventory", "gdacs_prompt_check", "rc_triage_summary",
    "prompt_cache_report", "model_config", "env_and_config", "question_metrics", "coverage_detail",
    "provider_preflight", "rc_pass_detail", "grounding_detail", "model_completeness", "prompt_prefixes",
)
#: Lifted from the forecast attribution bundle, named ``attribution__<file>``.
FORECAST_REVIEW_ATTRIBUTION_FILES = (
    "ANALYST_GUIDE.md", "MANIFEST.json", "prompts/section_hashes.json", "prompts/prompt_sections.csv",
    "inputs/input_inventory.csv", "attribution/trace_quality.csv", "attribution/rc_assessment.csv",
    "attribution/rc_shift_arms.csv", "contrasts/run_over_run.csv", "contrasts/fred_vs_sibyl.csv",
    "inputs/base_rates.csv", "attribution/prior_anchoring.csv", "contrasts/model_disagreement.csv",
    "prompts/token_share.csv",
)
#: Lifted from the current-run bundle, named ``current__<file>``.
FORECAST_REVIEW_CURRENT_FILES = ("sibyl.json", "blind_spots.json", "deltas.json", "manifest.json")
#: Stage files anywhere in the chain, by glob, named ``<run folder slug>__<file>``.
FORECAST_REVIEW_STAGE_GLOBS = (
    "stage_health__*.json", "post_run__*.txt", "batch_economics__*.json", "hs_triage_failures__*.json",
    "hs_triage_coverage__*.csv", "provider_preflight.json", "grounding_citation.json", "sibyl_run.log",
    "experiment_arms__*.json", "gate_recalibration.md", "anchor_depth.json",
)
FORECAST_REVIEW_NEVER = ("debug__llm_calls_detail", "debug__spd_tables", "debug__provider_batch_objects",
                         "debug__workflow_logs/")


def _forecast_review_plan(src: Any) -> list[tuple[str, Path | None]]:
    """(name in the folder, source path or None when absent), in priority order."""
    plan: list[tuple[str, Path | None]] = [("workflow_logs_index.json",
                                            src.staging / "workflow_logs_index.json"
                                            if (src.staging / "workflow_logs_index.json").is_file() else None)]
    for stem in FORECAST_REVIEW_DEBUG_FILES:
        path = src.dbg(stem)
        plan.append((f"debug__{path.name if path else stem}", path))
    for rel in FORECAST_REVIEW_ATTRIBUTION_FILES:
        plan.append((f"attribution__{Path(rel).name}", src.att(rel)))
    for rel in FORECAST_REVIEW_CURRENT_FILES:
        plan.append((f"current__{rel}", src.cur(rel)))
    seen: set[Path] = set()
    for pattern in FORECAST_REVIEW_STAGE_GLOBS:
        hits = [p for p in src.find(pattern) if p.resolve() not in seen]
        if not hits:
            plan.append((pattern, None))
        for p in hits:
            seen.add(p.resolve())
            m = _rr_folder_re().match(p.relative_to(src.staging).parts[0])
            prefix = m.group("slug") if m else "chain"
            plan.append((f"{prefix}__{p.name}", p))
    return plan


def _rr_folder_re():
    from scripts.ci.resolver_review_inputs import _FOLDER_RE

    return _FOLDER_RE


def is_forecast_chain(staging: Path) -> bool:
    manifest = {}
    try:
        manifest = json.loads((staging / "manifest.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        pass
    if (manifest.get("chain") or {}).get("kind") == "forecast":
        return True
    return any(staging.rglob("pythia_debug_bundle__*.zip"))


def extract_forecast_review(staging: Path, dest: Path, work: Path, *,
                            previous: Mapping[str, Any] | None = None,
                            file_cap: int = SCORED_FILE_CAP, total_cap: int = SCORED_TOTAL_CAP,
                            env: Mapping[str, str] | None = None) -> dict[str, Any] | None:
    """Write ``forecast_review/`` for a forecast chain (None for any other).

    The inputs file goes in first; then the lifted files in priority order
    under the ``scored_bundle/`` caps, every listed file absent, too large or
    over the folder cap named in ``SKIPPED.md``. Scanned for credentials
    (raises ``SecretLeak``)."""
    from scripts.ci.build_cycle_diagnostics import scan_for_secrets
    from scripts.ci import forecast_review_inputs as fri

    if not is_forecast_chain(staging):
        return None
    src = fri.locate(staging, work)
    dest.mkdir(parents=True, exist_ok=True)
    payload = fri.write_inputs_from(src, dest / FORECAST_REVIEW_INPUTS, previous=previous)
    total = (dest / FORECAST_REVIEW_INPUTS).stat().st_size
    written = [{"file": FORECAST_REVIEW_INPUTS, "bytes": total}]
    skipped: list[dict[str, Any]] = []
    if total > file_cap:
        skipped.append({"file": FORECAST_REVIEW_INPUTS, "bytes": total,
                        "reason": f"over the {file_cap} byte per-file cap: written anyway, it leads the folder"})
    for name, path in _forecast_review_plan(src):
        if path is None or not Path(path).is_file():
            skipped.append({"file": name, "bytes": None, "reason": "not in this chain"})
            continue
        size = Path(path).stat().st_size
        if size > file_cap:
            skipped.append({"file": name, "bytes": size, "reason": f"over the {file_cap} byte per-file cap"})
            continue
        if total + size > total_cap:
            skipped.append({"file": name, "bytes": size,
                            "reason": f"would take the folder past the {total_cap} byte cap"})
            continue
        shutil.copy2(path, dest / name)
        total += size
        written.append({"file": name, "bytes": size})
    for name in FORECAST_REVIEW_NEVER:
        skipped.append({"file": name, "bytes": None,
                        "reason": f"never copied: its figures are in {FORECAST_REVIEW_INPUTS}"})
    lines = ["# Files not copied into forecast_review/", "",
             f"Caps: {file_cap} bytes per file, {total_cap} bytes in all.", "",
             "| File | Bytes | Why |", "|---|---|---|"]
    lines += [f"| `{s['file']}` | {s['bytes'] if s['bytes'] is not None else '-'} | {s['reason']} |"
              for s in skipped]
    (dest / "SKIPPED.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    scan_for_secrets(dest, dict(os.environ if env is None else env))
    shutil.rmtree(work, ignore_errors=True)
    return {"folder": FORECAST_REVIEW_DIR, "debug_bundle": src.debug_zip.name if src.debug_zip else None,
            "files": written, "skipped": skipped, "total_bytes": total,
            "stubs": fri.stubs(payload),
            "previous_run": None if not previous or previous.get("stub") else
            (previous.get("meta") or {}).get("fc_run_id")}


#: ``runs/<YYYY>/<YYYY-MM-DD>__forecast__...`` folders: date and test flag.
_RUN_FOLDER = re.compile(r"^(?P<date>\d{4}-\d{2}-\d{2})__forecast__(?P<rest>.+)$")


def find_previous_review(rclone: Rclone, names: "ArchiveNames", work: Path) -> dict[str, Any]:
    """The newest earlier production forecast chain's inputs file, from Drive.

    Lists ``runs/`` for ``*/forecast_review/forecast_review_inputs.json``,
    newest folder first, and takes the first whose ``meta.is_test`` is false
    and whose folder is older than this chain's. ``{"stub": ...}`` when none."""
    try:
        files = rclone.list_files("runs")
    except Exception as exc:  # noqa: BLE001
        return {"stub": True, "reason": f"Drive listing failed: {type(exc).__name__}: {exc}"}
    candidates = []
    for f in files:
        parts = f.split("/")
        if len(parts) < 4 or parts[-2] != FORECAST_REVIEW_DIR or parts[-1] != FORECAST_REVIEW_INPUTS:
            continue
        folder = parts[-3]
        m = _RUN_FOLDER.match(folder)
        if not m or "__TEST" in folder or folder >= names.folder:
            continue
        candidates.append((m.group("date"), folder, f))
    for _, folder, rel in sorted(candidates, reverse=True):
        local = work / "previous" / folder / FORECAST_REVIEW_INPUTS
        try:
            rclone.download(f"runs/{rel}", local)
            payload = json.loads(local.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if (payload.get("meta") or {}).get("is_test"):
            continue
        payload.pop("previous_run", None)
        return payload
    return {"stub": True, "reason": "no earlier forecast_review"}


_LOG_TS = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})")
#: A line the shell echoed as script source (``[36;1m``) is not an error.
_ECHOED_SOURCE = "[36;1m"


def _workflow_step_flags(gh: Any, repo: str, path: str, sha: str,
                         cache: dict[tuple[str, str], dict[str, bool] | None]) -> dict[str, bool] | None:
    """Step name -> continue-on-error, from the workflow file at ``sha``."""
    key = (path, sha)
    if key in cache:
        return cache[key]
    flags: dict[str, bool] | None = None
    try:
        import base64

        import yaml

        payload = gh.api(f"repos/{repo}/contents/{path}?ref={sha}") or {}
        doc = yaml.safe_load(base64.b64decode(payload.get("content") or "").decode("utf-8")) or {}
        flags = {}
        for job in (doc.get("jobs") or {}).values():
            for step in (job or {}).get("steps") or []:
                name = step.get("name") or (f"Run {step['uses']}" if step.get("uses") else None)
                if name:
                    coe = step.get("continue-on-error")
                    flags[str(name)] = flags.get(str(name), False) or coe is True or str(coe).lower() == "true"
    except Exception:  # noqa: BLE001
        flags = None
    cache[key] = flags
    return flags


def scan_job_log(text: str, steps: list[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Tracebacks and ``##[error]`` lines per step, attributed by timestamp.

    A ``::error`` a step echoes as script source is not counted: only lines
    GitHub rendered as an annotation (``##[error]``) and Python tracebacks."""
    windows = []
    for st in steps:
        a, b = str(st.get("started_at") or "")[:19], str(st.get("completed_at") or "")[:19]
        if a:
            windows.append((a, b or "9999", str(st.get("name"))))
    out: dict[str, dict[str, Any]] = {}
    for line in text.splitlines():
        if _ECHOED_SOURCE in line:
            continue
        is_tb = "Traceback (most recent call last)" in line
        is_err = "##[error]" in line
        if not (is_tb or is_err):
            continue
        m = _LOG_TS.match(line)
        ts = m.group(1) if m else ""
        step = next((n for a, b, n in windows if ts and a <= ts <= b), "(unattributed)")
        rec = out.setdefault(step, {"n_traceback": 0, "n_error": 0, "first_traceback": None, "first_error": None})
        body = line[29:].strip() if m else line.strip()
        if is_tb:
            rec["n_traceback"] += 1
            rec["first_traceback"] = rec["first_traceback"] or body[:300]
        if is_err:
            rec["n_error"] += 1
            rec["first_error"] = rec["first_error"] or body[:300]
    return out


def build_workflow_logs_index(gh: Any, repo: str, run_ids: Iterable[str]) -> dict[str, Any]:
    """One row per chain run: its steps, their conclusions, continue-on-error,
    and the Tracebacks and ``##[error]`` lines in their logs. Logs are read,
    never stored. Every failure is recorded on the row, never raised."""
    cache: dict[tuple[str, str], dict[str, bool] | None] = {}
    runs = []
    for run_id in run_ids:
        row: dict[str, Any] = {"run_id": str(run_id), "steps": [], "problems": []}
        try:
            info = gh.api(f"repos/{repo}/actions/runs/{run_id}") or {}
            row |= {"workflow": info.get("name"), "path": info.get("path"),
                    "head_sha": info.get("head_sha"), "conclusion": info.get("conclusion")}
            flags = _workflow_step_flags(gh, repo, str(info.get("path") or "").split("@")[0],
                                         str(info.get("head_sha") or ""), cache) if info.get("path") else None
            jobs = (gh.api(f"repos/{repo}/actions/runs/{run_id}/jobs?per_page=100") or {}).get("jobs") or []
        except Exception as exc:  # noqa: BLE001
            row["problems"].append(f"run unreadable: {type(exc).__name__}: {exc}")
            runs.append(row)
            continue
        for job in jobs:
            text = ""
            api_text = getattr(gh, "api_text", None)
            if api_text is not None:
                try:
                    text = api_text(f"repos/{repo}/actions/jobs/{job.get('id')}/logs") or ""
                except Exception as exc:  # noqa: BLE001
                    row["problems"].append(f"job {job.get('id')} log unreadable: {type(exc).__name__}")
            steps = job.get("steps") or []
            hits = scan_job_log(text, steps)
            for st in steps:
                name = str(st.get("name"))
                h = hits.get(name, {})
                row["steps"].append({"job": job.get("name"), "name": name, "conclusion": st.get("conclusion"),
                                     "continue_on_error": (flags or {}).get(name) if flags is not None else None,
                                     "n_traceback": h.get("n_traceback", 0), "n_error": h.get("n_error", 0),
                                     "first_traceback": h.get("first_traceback"),
                                     "first_error": h.get("first_error")})
            if "(unattributed)" in hits:
                row["steps"].append({"job": job.get("name"), "name": "(unattributed)", "conclusion": None,
                                     "continue_on_error": None, **hits["(unattributed)"]})
            row["log_read"] = bool(text) or row.get("log_read", False)
        runs.append(row)
    return {"generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "note": "counts Python tracebacks and lines GitHub rendered as ##[error]; a ::error the shell "
                    "echoed as script source is not counted. continue_on_error is read from the workflow "
                    "file at the run's head_sha (null when that could not be read).",
            "runs": runs}


def write_workflow_logs_index(gh: Any, repo: str, staging: Path) -> Path | None:
    """Write ``workflow_logs_index.json`` into ``staging`` for every run in its manifest."""
    try:
        manifest = json.loads((staging / "manifest.json").read_text(encoding="utf-8"))
        ids = [str(r.get("run_id")) for r in manifest.get("runs") or [] if r.get("run_id")]
        payload = build_workflow_logs_index(gh, repo, ids)
        out = staging / "workflow_logs_index.json"
        out.write_text(json.dumps(payload, indent=1, default=str) + "\n", encoding="utf-8")
        return out
    except Exception as exc:  # noqa: BLE001 - the index is a diagnostic
        print(f"::warning title=workflow logs index::{type(exc).__name__}: {exc}")
        return None


def build_prompt_commits(gh: Any, repo: str, base: str | None, head: str | None,
                         *, max_commits: int = 100) -> dict[str, Any]:
    """Commits between ``base`` and ``head`` that touched a prompt file
    (``forecast_review_inputs.PROMPT_PATHS``), from the GitHub compare API.
    ``commits`` is None with a ``reason`` when the lookup cannot be made."""
    from scripts.ci import forecast_review_inputs as fri

    out: dict[str, Any] = {"base": base, "head": head, "paths": list(fri.PROMPT_PATHS), "commits": None}
    if not base or not head:
        out["reason"] = "no previous run commit" if not base else "no commit for this run"
        return out
    if base == head:
        out.update(commits=[], n_commits_between=0)
        return out
    try:
        cmp_ = gh.api(f"repos/{repo}/compare/{base}...{head}") or {}
    except Exception as exc:  # noqa: BLE001
        out["reason"] = f"compare API failed: {type(exc).__name__}: {exc}"[:300]
        return out
    commits = list(cmp_.get("commits") or [])
    out["n_commits_between"] = cmp_.get("total_commits", len(commits))
    files = [str(f.get("filename") or "") for f in cmp_.get("files") or []]
    if not fri.touches_prompts(files):
        out["commits"] = []
        return out
    found = []
    for c in commits[-max_commits:]:
        sha = str(c.get("sha") or "")
        try:
            detail = gh.api(f"repos/{repo}/commits/{sha}") or {}
        except Exception as exc:  # noqa: BLE001
            found.append({"sha": sha, "error": f"{type(exc).__name__}: {exc}"[:200]})
            continue
        hits = fri.touches_prompts(str(f.get("filename") or "") for f in detail.get("files") or [])
        if hits:
            msg = str((c.get("commit") or {}).get("message") or "").splitlines()
            found.append({"sha": sha, "date": ((c.get("commit") or {}).get("committer") or {}).get("date"),
                          "subject": msg[0] if msg else "", "files": hits})
    out["commits"] = found
    if len(commits) > max_commits:
        out["note"] = f"only the newest {max_commits} of {len(commits)} commits were read"
    return out


def write_prompt_commits(gh: Any, repo: str, staging: Path, previous: Mapping[str, Any] | None) -> Path | None:
    """Write ``prompt_commits.json`` into ``staging``. Never raises."""
    from scripts.ci import forecast_review_inputs as fri

    try:
        base = None if not previous or previous.get("stub") else (previous.get("meta") or {}).get("commit")
        payload = build_prompt_commits(gh, repo, base, fri.run_commit(staging))
    except Exception as exc:  # noqa: BLE001
        payload = {"commits": None, "reason": f"{type(exc).__name__}: {exc}"[:300]}
    out = staging / fri.PROMPT_COMMITS
    out.write_text(json.dumps(payload, indent=1, default=str) + "\n", encoding="utf-8")
    return out


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
    review = extract_resolver_review(staging, run_dir / RESOLVER_REVIEW_DIR, work / "resolver_review",
                                     env=env)
    if review is not None:
        report["resolver_review"] = review
    if is_forecast_chain(staging):
        write_workflow_logs_index(gh, repo, staging)
        previous = find_previous_review(rclone, names, work)
        write_prompt_commits(gh, repo, staging, previous)
        report["forecast_review"] = extract_forecast_review(
            staging, run_dir / FORECAST_REVIEW_DIR, work / "forecast_review", previous=previous, env=env)
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


def unpack_chain_zip(chain_zip: Path, dest: Path) -> Path:
    """Unpack an archived chain zip into ``dest``: the result is laid out as
    the staging tree ``build_cycle_diagnostics`` built it from."""
    import zipfile

    from scripts.ci.resolver_review_inputs import _safe_extract

    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(chain_zip) as zf:
        _safe_extract(zf, dest)
    return dest


def backfill_resolver_review(spec: Mapping[str, Any], *, rclone: Rclone, work: Path,
                             env: Mapping[str, str] | None = None,
                             replace: bool = False) -> dict[str, Any]:
    """Add the missing ``resolver_review/`` folder to a chain already in Drive.

    Reads the chain zip back from Drive and uploads ONLY
    ``<run_dir>/resolver_review/``. The zip, the loose files and any other
    folder beside it are never replaced; a folder already in Drive is left
    alone unless ``replace`` is set, which removes ``resolver_review/`` (and
    nothing else) first and writes it again.
    """
    names = ArchiveNames(**spec["names"])
    pid = str(spec.get("publish_run_id"))
    report: dict[str, Any] = {"publish_run_id": pid, "names": dataclasses.asdict(names),
                              "mode": "resolver_review_only", "status": "failed"}
    rclone.assert_single_root()
    zip_rel = f"{names.run_dir}/{names.zip_name}"
    if not rclone.exists(zip_rel):
        report["reason"] = f"{zip_rel} is not in Drive; run the normal archive leg instead"
        return report
    folder_rel = f"{names.run_dir}/{RESOLVER_REVIEW_DIR}"
    already = rclone.list_files(folder_rel)
    if already and not replace:
        report.update(status="already_present", reason=f"{folder_rel} already holds {len(already)} file(s)")
        return report
    shutil.rmtree(work, ignore_errors=True)
    local_zip = work / "chain" / names.zip_name
    rclone.download(zip_rel, local_zip)
    staging = unpack_chain_zip(local_zip, work / "staging")
    upload = work / "upload"
    review = extract_resolver_review(staging, upload / folder_rel, work / "review_work", env=env)
    if review is None:
        report.update(status="no_resolver_update", reason="the chain holds no Resolver Update stage")
        shutil.rmtree(work, ignore_errors=True)
        return report
    report["resolver_review"] = review
    if already:
        # Only the folder this command writes, and only once its replacement
        # is built: a chain zip that cannot be read leaves the old folder.
        assert folder_rel.endswith("/" + RESOLVER_REVIEW_DIR)
        rclone.purge_dir(folder_rel)
        report["replaced_files"] = len(already)
    rclone.upload_subtree(upload, folder_rel)
    report["drive_paths"] = sorted(
        f"{rclone.root}/{p.relative_to(upload).as_posix()}" for p in upload.rglob("*") if p.is_file())
    report["status"] = "archived"
    shutil.rmtree(work, ignore_errors=True)
    return report


def cmd_resolver_review(args: argparse.Namespace, *, gh: Any = None, rclone: Rclone | None = None) -> int:
    gh = gh or _gh_cls()(args.repo)
    rclone = rclone or Rclone(args.remote, args.root)
    work = Path(args.work_dir)
    try:
        spec = plan_one(gh, args.repo, None, str(args.publish_run_id), include_test=True, work=work)
        if spec.get("status") != "todo":
            report = {**spec, "mode": "resolver_review_only"}
            code = 1 if spec.get("status") == "failed" else 0
        else:
            report = backfill_resolver_review(spec, rclone=rclone, work=work / "review",
                                              replace=bool(getattr(args, "replace", False)))
            code = 1 if report.get("status") == "failed" else 0
    except Exception as exc:  # noqa: BLE001
        from scripts.ci.build_cycle_diagnostics import SecretLeak

        report = {"publish_run_id": str(args.publish_run_id), "mode": "resolver_review_only",
                  "status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
        code = 2 if isinstance(exc, SecretLeak) else 1
    if report.get("status") == "failed":
        print(f"::error title=Drive resolver-review backfill failed::{report.get('reason')}")
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: report.get(k) for k in ("status", "reason", "replaced_files", "resolver_review",
                                                 "drive_paths")}, indent=2, default=str))
    return code


def backfill_forecast_review(spec: Mapping[str, Any], *, rclone: Rclone, work: Path,
                             gh: Any = None, repo: str = "",
                             env: Mapping[str, str] | None = None,
                             replace: bool = False) -> dict[str, Any]:
    """Add the missing ``forecast_review/`` folder to a forecast chain already in Drive.

    Reads the chain zip back from Drive, writes the workflow-log index from
    the API when ``gh`` is given, finds the previous chain's inputs in Drive,
    and uploads ONLY ``<run_dir>/forecast_review/``. The zip and everything
    beside it are never touched; an existing folder is left alone unless
    ``replace`` is set, which removes that folder alone once its replacement
    is built."""
    names = ArchiveNames(**spec["names"])
    pid = str(spec.get("publish_run_id"))
    report: dict[str, Any] = {"publish_run_id": pid, "names": dataclasses.asdict(names),
                              "mode": "forecast_review_only", "status": "failed"}
    rclone.assert_single_root()
    zip_rel = f"{names.run_dir}/{names.zip_name}"
    if not rclone.exists(zip_rel):
        report["reason"] = f"{zip_rel} is not in Drive; run the normal archive leg instead"
        return report
    folder_rel = f"{names.run_dir}/{FORECAST_REVIEW_DIR}"
    already = rclone.list_files(folder_rel)
    if already and not replace:
        report.update(status="already_present", reason=f"{folder_rel} already holds {len(already)} file(s)")
        return report
    shutil.rmtree(work, ignore_errors=True)
    local_zip = work / "chain" / names.zip_name
    rclone.download(zip_rel, local_zip)
    staging = unpack_chain_zip(local_zip, work / "staging")
    if not is_forecast_chain(staging):
        report.update(status="not_a_forecast_chain", reason="the chain holds no forecast pipeline")
        shutil.rmtree(work, ignore_errors=True)
        return report
    previous = find_previous_review(rclone, names, work)
    if gh is not None:
        write_workflow_logs_index(gh, repo, staging)
        write_prompt_commits(gh, repo, staging, previous)
    upload = work / "upload"
    review = extract_forecast_review(staging, upload / folder_rel, work / "review_work",
                                     previous=previous, env=env)
    report["forecast_review"] = review
    if already:
        assert folder_rel.endswith("/" + FORECAST_REVIEW_DIR)
        rclone.purge_dir(folder_rel)
        report["replaced_files"] = len(already)
    rclone.upload_subtree(upload, folder_rel)
    report["drive_paths"] = sorted(
        f"{rclone.root}/{p.relative_to(upload).as_posix()}" for p in upload.rglob("*") if p.is_file())
    report["status"] = "archived"
    shutil.rmtree(work, ignore_errors=True)
    return report


def cmd_forecast_review(args: argparse.Namespace, *, gh: Any = None, rclone: Rclone | None = None) -> int:
    gh = gh or _gh_cls()(args.repo)
    rclone = rclone or Rclone(args.remote, args.root)
    work = Path(args.work_dir)
    try:
        spec = plan_one(gh, args.repo, None, str(args.publish_run_id), include_test=True, work=work)
        if spec.get("status") != "todo":
            report = {**spec, "mode": "forecast_review_only"}
            code = 1 if spec.get("status") == "failed" else 0
        else:
            report = backfill_forecast_review(spec, rclone=rclone, work=work / "forecast", gh=gh,
                                              repo=args.repo, replace=bool(getattr(args, "replace", False)))
            code = 1 if report.get("status") == "failed" else 0
    except Exception as exc:  # noqa: BLE001
        from scripts.ci.build_cycle_diagnostics import SecretLeak

        report = {"publish_run_id": str(args.publish_run_id), "mode": "forecast_review_only",
                  "status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
        code = 2 if isinstance(exc, SecretLeak) else 1
    if report.get("status") == "failed":
        print(f"::error title=Drive forecast-review backfill failed::{report.get('reason')}")
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    fr = report.get("forecast_review") or {}
    print(json.dumps({"status": report.get("status"), "reason": report.get("reason"),
                      "replaced_files": report.get("replaced_files"),
                      "files": len(fr.get("files") or []), "total_bytes": fr.get("total_bytes"),
                      "stubs": fr.get("stubs"), "previous_run": fr.get("previous_run"),
                      "drive_paths": report.get("drive_paths")}, indent=2, default=str))
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
    p = sub.add_parser("resolver-review",
                       help="add only the missing resolver_review/ folder to a chain already in Drive")
    p.add_argument("--publish-run-id", required=True)
    p.add_argument("--work-dir", default="archive_work")
    p.add_argument("--report", default="archive_leg/resolver_review.json")
    p.add_argument("--replace", action="store_true",
                   help="remove an existing resolver_review/ folder (and nothing else) and write it again")
    p = sub.add_parser("forecast-review",
                       help="add only the missing forecast_review/ folder to a forecast chain already in Drive")
    p.add_argument("--publish-run-id", required=True)
    p.add_argument("--work-dir", default="archive_work")
    p.add_argument("--report", default="archive_leg/forecast_review.json")
    p.add_argument("--replace", action="store_true",
                   help="remove an existing forecast_review/ folder (and nothing else) and write it again")
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
    if args.cmd == "resolver-review":
        return cmd_resolver_review(args)
    if args.cmd == "forecast-review":
        return cmd_forecast_review(args)
    return cmd_housekeeping(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
