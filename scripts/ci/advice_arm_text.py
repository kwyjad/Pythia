# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Does the advice arm of the experiment receive any advice text?

Half of all questions are held out with no calibration advice
(``PYTHIA_ADVICE_EXPERIMENT_SHARE``). The comparison means something only
where the other half actually got text: on 1 October 2026 several groups had
no shared advice and no member notes, so their "advice" arm was an untreated
arm wearing the treatment's label (``advice_empty`` since 2026-10-05).

Two views, both read-only:

* ``preview`` — what a question in the advice arm of each (hazard, metric)
  would be shown TODAY: the shared advice and each ensemble member's note,
  rendered by the forecaster's own loaders under the production flags.
* ``run`` — what a stored run recorded on ``forecasts_raw.advice_arm``,
  distinct questions per arm.

    python -m scripts.ci.advice_arm_text --db data/resolver.duckdb [--run-id fc_...]

Always exits 0; writes markdown to stdout and to ``--out`` when given.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

#: The groups the pipeline asks questions in (BLOCKED_HAZARDS excluded).
GROUPS: tuple[tuple[str, str], ...] = (
    ("ACE", "FATALITIES"), ("ACE", "PA"),
    ("DR", "PHASE3PLUS_IN_NEED"), ("DR", "EVENT_OCCURRENCE"),
    ("FL", "PA"), ("FL", "EVENT_OCCURRENCE"),
    ("TC", "PA"), ("TC", "EVENT_OCCURRENCE"),
)

#: The production workflows' settings that change what advice renders.
PRODUCTION_FLAGS = {
    "PYTHIA_ADVICE_FAMILY_CARRYOVER": "1",
    "PYTHIA_PRIOR_ANCHOR_SPD": "1",
    "PYTHIA_FAMILY_RECALIBRATION_MODE": "apply",
    "PYTHIA_MEMBER_ADVICE": "1",
}


def _member_names() -> list[str]:
    try:
        from pythia.llm_profiles import get_ensemble_resolved

        names = [m.get("model_id") or m.get("name") for m in get_ensemble_resolved()]
        names = [n for n in names if n]
    except Exception:  # noqa: BLE001
        names = []
    return names + ["track2_flash"]


def preview(groups=GROUPS, *, members: list[str] | None = None,
            db_url: str | None = None) -> list[dict]:
    """Per group: shared advice length, members with a note, and a verdict.

    The forecaster's loaders read ``PYTHIA_DB_URL`` (since Oct 2026), so
    ``db_url`` points them at the DB this report was asked about by setting
    it."""
    from forecaster import prompts

    if db_url:
        os.environ["PYTHIA_DB_URL"] = db_url

    members = members if members is not None else _member_names()
    out = []
    for hz, metric in groups:
        shared = prompts._load_calibration_advice_for_hazard(hz, metric) or ""
        noted = [
            m for m in members
            if prompts.load_member_calibration_advice(hz, metric, m)
        ]
        out.append({
            "hazard_code": hz, "metric": metric,
            "shared_chars": len(shared),
            "members_with_note": noted,
            "advice_arm_gets_text": bool(shared or noted),
        })
    return out


def run_arms(con, run_id: str) -> list[dict]:
    """Distinct questions per recorded arm for one run, per group."""
    try:
        rows = con.execute(
            """
            SELECT q.hazard_code, q.metric, COALESCE(r.advice_arm, '(none)'),
                   COUNT(DISTINCT r.question_id)
            FROM forecasts_raw r JOIN questions q USING (question_id)
            WHERE r.run_id = ?
            GROUP BY 1, 2, 3 ORDER BY 1, 2, 3
            """,
            [run_id],
        ).fetchall()
    except Exception as exc:  # noqa: BLE001
        return [{"error": str(exc)}]
    return [
        {"hazard_code": h, "metric": m, "arm": a, "questions": int(n)} for h, m, a, n in rows
    ]


def render(prev: list[dict], arms: list[dict] | None, run_id: str | None) -> str:
    lines = [
        "### Does the advice arm receive advice text?",
        "",
        "| Hazard | Metric | Shared advice (chars) | Members with a note | Advice arm gets text |",
        "|---|---|---:|---|---|",
    ]
    for r in prev:
        lines.append(
            f"| {r['hazard_code']} | {r['metric']} | {r['shared_chars']} | "
            f"{', '.join(r['members_with_note']) or 'none'} | "
            f"{'yes' if r['advice_arm_gets_text'] else '**no**'} |"
        )
    empty = [f"{r['hazard_code']}/{r['metric']}" for r in prev if not r["advice_arm_gets_text"]]
    lines.append("")
    lines.append(
        "Groups whose advice arm gets no text (their arm is stamped `advice_empty` "
        f"and compares nothing): {', '.join(empty) if empty else 'none'}."
    )
    if run_id is not None:
        lines += ["", f"Recorded arms on run `{run_id}` (distinct questions):", "",
                  "| Hazard | Metric | Arm | Questions |", "|---|---|---|---:|"]
        for a in arms or []:
            if "error" in a:
                lines.append(f"| error | {a['error']} | | |")
            else:
                lines.append(f"| {a['hazard_code']} | {a['metric']} | {a['arm']} | {a['questions']} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)
    db_url = f"duckdb:///{Path(args.db).resolve()}"
    os.environ["PYTHIA_DB_URL"] = db_url
    for key, value in PRODUCTION_FLAGS.items():
        os.environ.setdefault(key, value)
    try:
        prev = preview(db_url=db_url)
        arms = None
        if args.run_id:
            import duckdb

            con = duckdb.connect(args.db, read_only=True)
            try:
                arms = run_arms(con, args.run_id)
            finally:
                con.close()
        text = render(prev, arms, args.run_id)
    except Exception as exc:  # noqa: BLE001 - a report never fails a job
        text = f"advice_arm_text could not run: {exc}\n"
        prev, arms = [], None
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps({"preview": prev, "run": arms}, indent=2))
    summary = os.getenv("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
