# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

import pathlib
import re

WF_DIR = pathlib.Path(".github/workflows")
# The composite actions carry as much shell as a workflow (the canonical-DB
# discovery loop is ~165 lines of bash) and were outside every rule here.
ACTIONS_DIR = pathlib.Path(".github/actions")


def _workflow_texts():
    for path in WF_DIR.glob("*.y*ml"):
        yield path, path.read_text(encoding="utf-8", errors="replace")
    for path in sorted(ACTIONS_DIR.glob("*/action.y*ml")):
        yield path, path.read_text(encoding="utf-8", errors="replace")


def test_no_yaml_list_under_upload_artifact_path():
    offenders = []
    for path, text in _workflow_texts():
        if "uses: actions/upload-artifact@v4" in text:
            if re.search(r"with:\s*[\s\S]*?path:\s*\n\s*-\s", text):
                offenders.append(str(path))
    assert not offenders, (
        "upload-artifact path must be a newline scalar, not a YAML list: " + ", ".join(offenders)
    )


def test_bracket_test_spacing():
    offenders = []
    for path, text in _workflow_texts():
        # A POSIX character class ([[:space:]]) in sed is not a bash test.
        if re.search(r"\[\[(?!:)[^\s].*[^\s]\]\]", text):
            offenders.append(str(path))
    assert not offenders, "Missing spaces in '[[ ... ]]' tests: " + ", ".join(offenders)


def test_no_echo_escape_sequences():
    offenders = []
    for path, text in _workflow_texts():
        if re.search(r"echo\s+-e\b", text) or re.search(r'echo\s+".*\\n', text):
            offenders.append(str(path))
    assert not offenders, "Use printf for escapes instead of echo: " + ", ".join(offenders)


def test_no_and_and_or_as_if_else():
    """Reject the shell `A && B || C` if/else idiom (C also runs when B fails).

    Comment lines are skipped: a `#` line is never executed, so prose that
    happens to describe the idiom (or a GitHub-expression gotcha) is a false
    positive — one such comment in pythia_pipeline_stage.yml failed this test
    on main until the skip was added.
    """

    offenders = []
    for path, text in _workflow_texts():
        for line in text.splitlines():
            if line.lstrip().startswith("#"):
                continue
            if "&&" in line and "||" in line and "${{" not in line:
                offenders.append(f"{path}: {line.strip()}")
                break
    assert not offenders, "A && B || C found; replace with if/else: " + ", ".join(offenders)


# --------------------------------------------------------------------------
# Cross-package test wiring
#
# A test guards the directory it IMPORTS from, not the directory it lives in.
# resolver-ci-fast.yml runs all of resolver/tests/ but triggers only on
# resolver/**, so a resolver/tests file importing forecaster/, horizon_scanner/
# or pythia/ is re-run by NOTHING when the code it guards changes. That gap let
# test_idmc_history_conflict_pa.py sit red on main for days after #816.
#
# Wiring by hand is what failed; this makes it self-enforcing. Kept to stdlib
# only (no yaml) because ci-lint.yml installs just pytest and runs this file
# with --noconftest.
# --------------------------------------------------------------------------

REPO_PACKAGES = {"forecaster", "horizon_scanner", "pythia", "sibyl"}

_IMPORT_RE = re.compile(r"^\s*(?:from|import)\s+([a-zA-Z_][a-zA-Z0-9_]*)", re.MULTILINE)


def _foreign_packages(text):
    """Top-level repo packages other than `resolver` that this file imports."""
    return {m for m in _IMPORT_RE.findall(text) if m in REPO_PACKAGES}


def test_cross_package_resolver_tests_are_wired_into_a_workflow():
    workflows = list(_workflow_texts())
    offenders = []

    for test_path in sorted(pathlib.Path("resolver/tests").glob("test_*.py")):
        text = test_path.read_text(encoding="utf-8", errors="replace")
        packages = _foreign_packages(text)
        if not packages:
            continue

        rel = test_path.as_posix()
        # paths: entries are quoted list items; pytest arguments are bare. A
        # workflow only counts when it does BOTH — a trigger without an
        # invocation runs nothing, an invocation without a trigger never fires.
        triggered = {p.name for p, t in workflows if f'- "{rel}"' in t}
        invoked = {
            p.name
            for p, t in workflows
            if re.search(r"(?<![\"'])" + re.escape(rel) + r"(?![\"'])", t)
        }
        if not (triggered & invoked):
            offenders.append(
                f"{rel} imports {sorted(packages)} but no workflow both triggers on it "
                f"and runs it (triggered by: {sorted(triggered) or 'none'}; "
                f"invoked by: {sorted(invoked) or 'none'}). "
                "Add it to the paths: block AND the pytest call of the workflow that "
                "owns the code it guards — horizon_scanner -> horizon-scanner-ci.yml, "
                "forecaster/pythia -> forecaster-ci.yml."
            )

    assert not offenders, "Cross-package resolver/tests not wired into CI:\n" + "\n".join(
        offenders
    )
