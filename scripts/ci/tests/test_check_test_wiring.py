# Pythia / Copyright (c) 2025 Kevin Wyjad
"""The test-wiring check: a test file in no workflow fails CI.

Stdlib + pytest only, because ci-lint installs nothing else.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

from scripts.ci import check_test_wiring as w

REPO = Path(__file__).resolve().parents[3]
TEST_BODY = "def test_x():\n    assert True\n"


def _repo(tmp_path: Path, workflow: str, files: dict[str, str], quarantine: str | None = None) -> Path:
    (tmp_path / ".github/workflows").mkdir(parents=True)
    (tmp_path / ".github/workflows/ci.yml").write_text(workflow)
    for rel, body in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
    if quarantine is not None:
        (tmp_path / ".github/test_quarantine.txt").write_text(quarantine)
    return tmp_path


WF = """
jobs:
  t:
    steps:
      - name: tests
        run: |
          pytest -vv \\
            pkg/tests/test_named.py \\
            whole/ --ignore=whole/test_skipped.py
      - run: echo pkg/tests/test_echoed.py
"""


def test_every_test_file_in_this_repository_runs_in_a_workflow():
    """The check itself, on the real repository. A failure lists the files."""
    rep = w.check(REPO)
    assert rep.ok, "\n".join(rep.orphans + rep.problems)


def test_a_file_named_nowhere_is_an_orphan(tmp_path):
    root = _repo(tmp_path, WF, {
        "pkg/tests/test_named.py": TEST_BODY,
        "pkg/tests/test_forgotten.py": TEST_BODY,
        # Named only in a step that never runs pytest: still an orphan.
        "pkg/tests/test_echoed.py": TEST_BODY,
    })
    rep = w.check(root)
    assert rep.orphans == ["pkg/tests/test_echoed.py", "pkg/tests/test_forgotten.py"]
    assert "pkg/tests/test_named.py" in rep.wired
    assert not rep.ok


def test_a_directory_covers_its_tests_and_ignore_removes_one(tmp_path):
    root = _repo(tmp_path, WF, {
        "pkg/tests/test_named.py": TEST_BODY,
        "whole/test_a.py": TEST_BODY,
        "whole/sub/test_b.py": TEST_BODY,
        "whole/test_skipped.py": TEST_BODY,
    })
    rep = w.check(root)
    assert rep.orphans == ["whole/test_skipped.py"]


def test_a_module_named_like_a_test_but_defining_none_is_not_a_test(tmp_path):
    root = _repo(tmp_path, WF, {
        "pkg/tests/test_named.py": TEST_BODY,
        "pkg/test_mode.py": "def is_test_mode():\n    return False\n",
        "scripts/reconcile_is_test.py": TEST_BODY,  # *_test.py outside a tests dir
    })
    assert w.check(root).ok


def test_quarantine_needs_a_reason_and_a_date_and_must_still_be_true(tmp_path):
    files = {
        "pkg/tests/test_named.py": TEST_BODY,
        "pkg/tests/test_parked.py": TEST_BODY,
    }
    q = (
        "# comment\n"
        "pkg/tests/test_parked.py | 2099-01-01 | needs a live provider\n"
        "pkg/tests/test_named.py | 2099-01-01 | wired anyway\n"
        "pkg/tests/test_gone.py | 2099-01-01 | file deleted\n"
        "pkg/tests/test_bad.py | soon | no date\n"
        "pkg/tests/test_short.py | 2099-01-01\n"
    )
    rep = w.check(_repo(tmp_path, WF, files, q))
    assert rep.quarantined == ["pkg/tests/test_parked.py"]
    assert rep.orphans == []
    text = "\n".join(rep.problems)
    assert "test_named.py is quarantined but a workflow runs it" in text
    assert "test_gone.py is not a test file" in text
    assert "'soon' is not YYYY-MM-DD" in text
    assert "expected 'path | YYYY-MM-DD | reason'" in text
    assert not rep.ok


def test_an_overdue_quarantine_warns_and_does_not_fail(tmp_path):
    files = {"pkg/tests/test_named.py": TEST_BODY, "pkg/tests/test_parked.py": TEST_BODY}
    q = "pkg/tests/test_parked.py | 2026-01-01 | needs a live provider\n"
    rep = w.check(_repo(tmp_path, WF, files, q), today=dt.date(2026, 10, 8))
    assert rep.ok
    assert [e.path for e in rep.overdue] == ["pkg/tests/test_parked.py"]


def test_main_exits_one_and_names_the_orphan(tmp_path, capsys):
    root = _repo(tmp_path, WF, {"pkg/tests/test_forgotten.py": TEST_BODY})
    assert w.main(["--root", str(root)]) == 1
    out = capsys.readouterr().out
    assert "::error::pkg/tests/test_forgotten.py runs in no workflow" in out
