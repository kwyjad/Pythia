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


# The operational debug bundle used to be dumped in two places (the staged
# pipeline's fc_collect_finalize and the legacy synchronous HS workflow) and
# was consolidated in the Sibyl job in Sept 2026 so a cycle produces one set
# of artifacts in one place. A second invoker quietly re-splits it.
_DEBUG_BUNDLE_INVOKER = pathlib.Path(".github/workflows/run_sibyl.yml")


# An INVOCATION, not a mention: forecaster-ci names the script in its
# paths: filter, which is a trigger, not a call.
_DEBUG_BUNDLE_INVOCATION = re.compile(
    r"python3?\s+(-m\s+scripts\.dump_pythia_debug_bundle|scripts/dump_pythia_debug_bundle\.py)\b"
)


def test_debug_bundle_is_dumped_from_exactly_one_workflow():
    invokers = []
    for path, text in _workflow_texts():
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if _DEBUG_BUNDLE_INVOCATION.search(stripped):
                invokers.append(pathlib.Path(path))
                break
    assert invokers == [_DEBUG_BUNDLE_INVOKER], (
        "scripts.dump_pythia_debug_bundle must be invoked from exactly one workflow "
        f"({_DEBUG_BUNDLE_INVOKER}); found: {sorted(str(p) for p in invokers)}"
    )


def test_the_gdacs_ingest_step_bounds_its_own_enrichment_pass():
    """A polite pace outruns a step budget, and the step must say so.

    The per-event GDACS pace is one request every two seconds, which is
    slow enough to matter: a reset run's 3,272 events would take 109
    minutes against a 60-minute step, and a killed step throws away every
    exposure the run had already fetched. The connector path takes its
    ceiling from the environment, so the workflow that owns the step
    budget is the thing that must set it.
    """

    body = pathlib.Path(".github/workflows/resolver_update.yml").read_text(
        encoding="utf-8"
    )
    marker = '- name: "Phase 2: Ingest GDACS'
    assert marker in body, "the GDACS ingest step was renamed; update this check"
    step = body.split(marker, 1)[1].split("- name:", 1)[0]
    assert "GDACS_ENRICH_MAX_SECONDS" in step, (
        "the GDACS ingest step must bound its enrichment pass — without a "
        "ceiling the pass outruns the step and the run loses the work"
    )
    budgets = [int(m) for m in re.findall(r"'(\d+)'", step)]
    assert budgets, "GDACS_ENRICH_MAX_SECONDS must be given a value"
    assert max(budgets) <= 55 * 60, (
        f"the budget must sit inside the reset step's 60 minutes, got "
        f"{max(budgets)}s"
    )
    # The step is 20 minutes off reset and 60 on it, so the two budgets are
    # not interchangeable: the larger one on a normal run would outlive the
    # step it is supposed to fit inside.
    assert min(budgets) <= 18 * 60, (
        f"the non-reset budget must sit inside the 20-minute step, got "
        f"{min(budgets)}s"
    )


def test_the_backcast_share_override_reaches_the_run_step():
    """A one-dispatch lever nobody can pull is not a lever.

    The input exists so a named backlog can be cleared without raising the
    rulebook's standing share, which governs every night after. It reaches
    the machine only as the env var ``load_budget`` reads, so both halves
    are asserted here — in the file ci-lint runs on every PR, because
    resolver-ci-fast has no ``.github/workflows/**`` trigger and a
    workflow-only edit would otherwise run nothing.
    """

    text = (WF_DIR / "haz_backcast.yml").read_text()
    assert "backcast_extraction_share:" in text, (
        "haz_backcast.yml needs the dispatch input"
    )
    assert re.search(
        r"PYTHIA_HAZ_BACKCAST_EXTRACTION_SHARE:\s*\$\{\{\s*inputs\.backcast_extraction_share",
        text,
    ), "the input must reach the run step as the env var load_budget reads"


# --------------------------------------------------------------------------
# Public-repo hardening (security PR 4, Oct 2026)
#
# Once the repository is public, anyone can open a pull request from a fork,
# name a branch `main`, and run workflows that upload artifacts. Three rules
# keep that from reaching the canonical DB, the release or a secret, and they
# are checked here because ci-lint runs this file on every PR.
# --------------------------------------------------------------------------

_RUN_KEY = re.compile(r"^(?P<indent>\s*)(?:-\s+)?run:\s*(?P<rest>.*)$")


def _run_blocks(text):
    """Yield (first line number, body) for every `run:` value in a YAML file.

    Stdlib only, like the rest of this file: a block scalar (`|`, `>`) runs
    until the first non-blank line indented no deeper than the key.
    """

    lines = text.splitlines()
    i = 0
    while i < len(lines):
        m = _RUN_KEY.match(lines[i])
        if not m or lines[i].lstrip().startswith("#"):
            i += 1
            continue
        key_indent = len(m.group("indent"))
        rest = m.group("rest").strip()
        start = i + 1
        if rest and rest[0] not in "|>":
            yield start, rest
            i += 1
            continue
        body = []
        i += 1
        while i < len(lines):
            line = lines[i]
            if line.strip() and len(line) - len(line.lstrip()) <= key_indent:
                break
            body.append(line)
            i += 1
        yield start, "\n".join(body)


# Values an attacker cannot choose. Everything else that reaches shell goes
# through env: so the shell sees a variable, never pasted text.
_SAFE_EXPR = re.compile(
    r"^(?:github\.(?:run_id|run_number|run_attempt|repository|repository_owner|sha|"
    r"workspace|server_url|api_url|job|workflow|ref_name|action_path)|"
    r"runner\.[a-z_]+|env\.[A-Za-z_][A-Za-z0-9_]*|secrets\.[A-Za-z_][A-Za-z0-9_]*|"
    r"matrix\.[A-Za-z_][A-Za-z0-9_.]*)$"
)


def test_no_untrusted_expression_is_pasted_into_shell():
    offenders = []
    for path, text in _workflow_texts():
        for lineno, body in _run_blocks(text):
            for expr in re.findall(r"\$\{\{\s*(.*?)\s*\}\}", body):
                if not _SAFE_EXPR.match(expr):
                    offenders.append(f"{path}:{lineno}: ${{{{ {expr} }}}}")
    assert not offenders, (
        "Move these expressions into the step's env: and read them as \"${VAR}\" "
        "in the script; a value pasted into run: is shell source:\n" + "\n".join(offenders)
    )


def test_every_workflow_declares_its_permissions():
    missing = [
        str(path)
        for path, text in _workflow_texts()
        if path.parent == WF_DIR and not re.search(r"(?m)^permissions:", text)
    ]
    assert not missing, (
        "Every workflow needs a top-level permissions: block (contents: read at "
        "least), or it inherits the repository default token: " + ", ".join(missing)
    )


def test_workflow_run_consumers_refuse_runs_from_forks():
    offenders = []
    for path, text in _workflow_texts():
        if path.parent != WF_DIR or not re.search(r"(?m)^\s{2}workflow_run:", text):
            continue
        trigger = text.split("workflow_run:", 1)[1].split("\n  workflow_dispatch", 1)[0]
        if not re.search(r"branches:\s*\[?\s*main", trigger):
            offenders.append(f"{path}: workflow_run trigger has no branches: [main]")
        if "github.event.workflow_run.head_repository.full_name == github.repository" not in text:
            offenders.append(f"{path}: no job checks the triggering run's head repository")
        if "github.event.workflow_run.event != 'pull_request'" not in text:
            offenders.append(f"{path}: no job refuses a triggering pull_request run")
    assert not offenders, "\n".join(offenders)


def test_third_party_actions_are_pinned_to_a_commit():
    offenders = []
    for path, text in _workflow_texts():
        for ref in re.findall(r"(?m)^\s*(?:-\s+)?uses:\s*([^\s#]+)", text):
            if ref.startswith(("actions/", "./", "docker://")):
                continue
            if not re.search(r"@[0-9a-f]{40}$", ref):
                offenders.append(f"{path}: {ref}")
    assert not offenders, "Pin third-party actions to a full commit SHA: " + ", ".join(offenders)


def test_run_discovery_never_accepts_a_pull_request_run():
    """Every place that picks a run for its artifact filters on event.

    A fork can name its branch `main`, so `--branch main` alone selects its
    runs. GitHub records those runs as event pull_request, which no producer
    of a trusted artifact ever is.
    """

    sources = [
        pathlib.Path(".github/actions/download-canonical-db/action.yml"),
        WF_DIR / "run_horizon_scanner.yml",
        WF_DIR / "publish_latest_data.yml",
        pathlib.Path("scripts/ci/poll_llm_batches.py"),
        pathlib.Path("scripts/ci/check_pipeline_active.py"),
    ]
    offenders = []
    for path in sources:
        text = path.read_text(encoding="utf-8")
        for m in re.finditer(r'(?m)^\s*gh run list[^\n]*|"run", "list"[^\n]*', text):
            window = text[m.start(): m.start() + 600]
            if "event" not in window:
                offenders.append(f"{path}: {m.group(0)[:80]}")
    assert not offenders, (
        "A `gh run list` that selects artifacts must also filter on event "
        "(TRUSTED_EVENTS): " + ", ".join(offenders)
    )
