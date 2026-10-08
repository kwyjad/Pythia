# Pythia / Copyright (c) 2025 Kevin Wyjad
"""Every test file runs in some workflow, or says why it does not.

CI in this repository runs explicit test-file lists. A test file named in no
workflow's pytest command runs nowhere: it can go red for months and nobody
sees it. On 2026-10-07 fifteen such tests were failing, and the same finding
came up in three separate reviews. This check fixes the class rather than the
list.

How coverage is decided. Every ``run:`` block that mentions ``pytest`` is
read as text. Each token that names a file or directory in the repository
counts: a file covers itself, a directory covers every test file below it
(pytest recurses), and ``--ignore=PATH`` removes PATH again. That is a
deliberate over-approximation of what pytest collects: it cannot see a
``-k`` or marker filter. It catches the fault that actually happens, which is
a file nobody named.

A file that cannot run in CI goes in the quarantine file
(``.github/test_quarantine.txt``), one line each::

    path/to/test_x.py | 2026-12-31 | one-line reason

An entry past its review date is reported with a ``::warning`` and does NOT
fail the check: a CI check that turns red because the calendar moved fails
every unrelated pull request, and a check that fails for unrelated reasons is
one people learn to ignore. A malformed entry, an entry for a file that does
not exist, and an entry for a file that is in fact wired all fail, because
each of those is a quarantine that has stopped meaning anything.

Stdlib only: ci-lint installs pytest and nothing else.
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

QUARANTINE_PATH = Path(".github/test_quarantine.txt")
WORKFLOW_DIR = Path(".github/workflows")

# Directories that hold no Python tests of ours, or tests of vendored code.
EXCLUDED_DIRS = {".git", "node_modules", "web", "vendor", ".venv", "venv", "__pycache__"}

_TOKEN = re.compile(r"[A-Za-z0-9_./=-]+")


def is_test_file(path: Path) -> bool:
    name = path.name
    if not name.endswith(".py"):
        return False
    if name.startswith("test_"):
        return True
    # pytest's other default pattern, but only inside a tests directory: a
    # script that happens to end in _test.py is not a test.
    return name.endswith("_test.py") and any(p in ("tests", "test") for p in path.parts[:-1])


_DEFINES_TEST = re.compile(r"^\s*(?:async\s+)?def\s+test|^\s*class\s+Test", re.MULTILINE)


def _defines_tests(path: Path) -> bool:
    """A module named like a test that defines none is a module
    (``pythia/test_mode.py``), not a test pytest would collect."""
    try:
        return bool(_DEFINES_TEST.search(path.read_text(encoding="utf-8", errors="replace")))
    except OSError:
        return False


def find_test_files(root: Path) -> list[str]:
    out: list[str] = []
    for path in root.rglob("*.py"):
        rel = path.relative_to(root)
        if any(part in EXCLUDED_DIRS for part in rel.parts):
            continue
        if is_test_file(rel) and _defines_tests(path):
            out.append(rel.as_posix())
    return sorted(out)


def _run_blocks(text: str) -> list[str]:
    """The text of every ``run:`` value in a workflow (inline or block),
    read by indentation so no YAML parser is needed."""
    lines = text.splitlines()
    blocks: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"^(\s*)(?:-\s+)?run:\s*(.*)$", line)
        if not m:
            i += 1
            continue
        indent = len(m.group(1))
        rest = m.group(2).strip()
        if rest and rest[0] not in "|>":
            blocks.append(rest)
            i += 1
            continue
        body: list[str] = []
        i += 1
        while i < len(lines):
            nxt = lines[i]
            if nxt.strip() and (len(nxt) - len(nxt.lstrip())) <= indent:
                break
            body.append(nxt)
            i += 1
        blocks.append("\n".join(body))
    return blocks


@dataclass
class Coverage:
    files: set[str] = field(default_factory=set)
    dirs: set[str] = field(default_factory=set)
    ignored: set[str] = field(default_factory=set)
    by_workflow: dict[str, set[str]] = field(default_factory=dict)

    def covers(self, test_path: str) -> bool:
        if _under_any(test_path, self.ignored):
            return False
        return test_path in self.files or _under_any(test_path, self.dirs)


def _under_any(path: str, prefixes: set[str]) -> bool:
    for p in prefixes:
        p = p.rstrip("/")
        if path == p or path.startswith(p + "/"):
            return True
    return False


def collect_coverage(root: Path) -> Coverage:
    cov = Coverage()
    wf_dir = root / WORKFLOW_DIR
    for wf in sorted(list(wf_dir.glob("*.yml")) + list(wf_dir.glob("*.yaml"))):
        named: set[str] = set()
        for block in _run_blocks(wf.read_text(encoding="utf-8")):
            if "pytest" not in block:
                continue
            for line in block.splitlines():
                if line.lstrip().startswith("#"):
                    continue
                for tok in _TOKEN.findall(line):
                    if tok.startswith("--ignore="):
                        target = tok.split("=", 1)[1]
                        if target.startswith("./"):
                            target = target[2:]
                        cov.ignored.add(target.rstrip("/"))
                        continue
                    cand = tok[2:] if tok.startswith("./") else tok
                    if not cand or cand.startswith("-"):
                        continue
                    p = root / cand
                    if p.is_file() and cand.endswith(".py"):
                        cov.files.add(cand)
                        named.add(cand)
                    elif p.is_dir() and cand not in (".", ""):
                        cov.dirs.add(cand.rstrip("/"))
                        named.add(cand.rstrip("/") + "/")
        if named:
            cov.by_workflow[wf.name] = named
    return cov


@dataclass
class QuarantineEntry:
    path: str
    review_by: dt.date
    reason: str
    line_no: int


def read_quarantine(root: Path) -> tuple[list[QuarantineEntry], list[str]]:
    entries: list[QuarantineEntry] = []
    problems: list[str] = []
    qpath = root / QUARANTINE_PATH
    if not qpath.exists():
        return entries, problems
    for n, raw in enumerate(qpath.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 3 or not all(parts):
            problems.append(f"{QUARANTINE_PATH}:{n}: expected 'path | YYYY-MM-DD | reason', got {raw!r}")
            continue
        path, date_s, reason = parts
        try:
            review_by = dt.date.fromisoformat(date_s)
        except ValueError:
            problems.append(f"{QUARANTINE_PATH}:{n}: review date {date_s!r} is not YYYY-MM-DD")
            continue
        entries.append(QuarantineEntry(path, review_by, reason, n))
    return entries, problems


@dataclass
class Report:
    tests: list[str]
    wired: list[str]
    quarantined: list[str]
    orphans: list[str]
    problems: list[str]
    overdue: list[QuarantineEntry]

    @property
    def ok(self) -> bool:
        return not self.orphans and not self.problems


def check(root: Path, today: dt.date | None = None) -> Report:
    today = today or dt.date.today()
    tests = find_test_files(root)
    cov = collect_coverage(root)
    entries, problems = read_quarantine(root)
    quarantine = {e.path: e for e in entries}
    wired, quarantined, orphans = [], [], []
    for t in tests:
        if cov.covers(t):
            wired.append(t)
            if t in quarantine:
                problems.append(
                    f"{QUARANTINE_PATH}:{quarantine[t].line_no}: {t} is quarantined but a workflow runs it; remove the entry"
                )
        elif t in quarantine:
            quarantined.append(t)
        else:
            orphans.append(t)
    test_set = set(tests)
    for e in entries:
        if e.path not in test_set:
            problems.append(f"{QUARANTINE_PATH}:{e.line_no}: {e.path} is not a test file in the repository")
    overdue = [e for e in entries if e.review_by < today and e.path in test_set]
    return Report(tests, wired, quarantined, orphans, problems, overdue)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=".")
    ap.add_argument("--today", default=None, help="YYYY-MM-DD (tests only)")
    args = ap.parse_args(argv)
    today = dt.date.fromisoformat(args.today) if args.today else None
    rep = check(Path(args.root).resolve(), today)
    print(f"test files: {len(rep.tests)}  wired: {len(rep.wired)}  "
          f"quarantined: {len(rep.quarantined)}  orphaned: {len(rep.orphans)}")
    for e in rep.overdue:
        print(f"::warning::quarantined test {e.path} is past its review date {e.review_by}: {e.reason}")
    for p in rep.problems:
        print(f"::error::{p}")
    for o in rep.orphans:
        print(f"::error::{o} runs in no workflow. Name it in the pytest step of the workflow "
              f"that owns its code (and in that workflow's paths:), delete it, or quarantine it in "
              f"{QUARANTINE_PATH} with a review date and a reason.")
    return 0 if rep.ok else 1


if __name__ == "__main__":
    sys.exit(main())
