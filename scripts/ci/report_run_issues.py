# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Print the run issue register where a human will actually see it.

Run 34222175003 finished green with two failed contradiction checks, an
unread EM-DAT rung and a broken flood ceiling. Every one of those was
recorded correctly; none of them reached anyone, because the only place
they existed was a 172-file zip.

This step is the answer. It builds the register, writes it beside the
diagnostics so the bundle can copy it verbatim, and puts it in the three
places a person looks:

1. the end of the run's stdout, framed so it survives scrolling;
2. the GitHub step summary, as a table;
3. `checks/issues.md` in the debug bundle (via `diagnostics/issues.json`).

**It always exits 0.** An exit code is a statement about whether the run
produced its output, and two failed checks must not throw away 2,887
successful ReliefWeb fetches. Blocking and degraded issues get `::error::`
annotations, which is loud enough; a registered known issue gets a
`::notice::`, because a register that does not quieten the noise it exists
to quieten is just a second place to read the same alarms.

Usage::

    python -m scripts.ci.report_run_issues --db data/resolver.duckdb \\
        --diagnostics-dir diagnostics --run-log-dir diagnostics/run_log
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - import convenience
    sys.path.insert(0, str(REPO_ROOT))

from resolver.diagnostics import issues as issues_mod  # noqa: E402
from resolver.diagnostics import run_log  # noqa: E402


def _emit_step_summary(text: str) -> bool:
    """Append to $GITHUB_STEP_SUMMARY. False when there is nowhere to write."""

    target = (os.environ.get("GITHUB_STEP_SUMMARY") or "").strip()
    if not target:
        return False
    try:
        with open(target, "a", encoding="utf-8") as handle:
            handle.write(text)
            handle.write("\n")
        return True
    except OSError:
        return False


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="report_run_issues",
        description="Build and print this run's issue register.",
    )
    parser.add_argument(
        "--db",
        default=os.environ.get("BACKFILL_DB_PATH") or os.environ.get("RESOLVER_DB_URL")
        or os.environ.get("PYTHIA_DB_URL") or "data/resolver.duckdb",
        help="DuckDB path or duckdb:/// URL",
    )
    parser.add_argument("--diagnostics-dir", default="diagnostics")
    parser.add_argument(
        "--run-log-dir", default=os.environ.get(run_log.ENV_DIR) or "",
    )
    parser.add_argument(
        "--no-history", action="store_true",
        help="Do not record this run's issues in the DB history table",
    )
    parser.add_argument(
        "--from-json", default="",
        help="Render an already-written issues.json instead of rebuilding",
    )
    args = parser.parse_args(argv)

    diagnostics = Path(args.diagnostics_dir)
    run_label = os.environ.get("GITHUB_RUN_ID", "")

    register = None
    if args.from_json:
        register = issues_mod.read_register(args.from_json)
    if register is None:
        # Imported here, not at module scope: this script must still be able
        # to render a written register on a machine with no duckdb.
        from scripts.build_resolver_debug_bundle import build_register, normalise_db_path

        run_dir = Path(args.run_log_dir) if args.run_log_dir else None
        try:
            register = build_register(
                db_path=normalise_db_path(args.db),
                diagnostics_dir=diagnostics,
                run_log_dir=run_dir if (run_dir and run_dir.is_dir()) else None,
                write_history=not args.no_history,
            )
        except Exception as exc:  # noqa: BLE001 - never fail the step
            print(f"::warning title=issue register::could not build the register: "
                  f"{type(exc).__name__}: {exc}")
            return 0

    try:
        issues_mod.write_outputs(
            register,
            json_path=diagnostics / "issues.json",
            markdown_path=diagnostics / "issues.md",
            run={"github_run_id": run_label},
            run_label=run_label,
        )
    except OSError as exc:
        print(f"::warning title=issue register::could not write issues.json: {exc}")

    # Annotations first, so they are attached before the wall of text.
    for annotation in issues_mod.render_annotations(register):
        print(annotation)

    _emit_step_summary(issues_mod.render_markdown(register, run_label=run_label))

    # Last thing on stdout, on purpose.
    sys.stdout.write(issues_mod.render_text(register, run_label=run_label))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
