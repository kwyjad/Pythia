# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The forecast run's experiment arms, as data.

Three experiments split the questions of every forecast run by a hash of the
question id (the advice arm, the RC-shift arm and, from 13 December 2026, the
adversarial counter-evidence arm). Their split lived only in the forecast
tables and, for the adversarial arm, one stdout line. The forecast pipeline
review checks the wiring of each (template rule A4.8): eligible counts, the
split, whether it sits inside a 90% binomial range around the design share,
and whether every row is stamped. This writes the counts it needs, read back
from what the forecaster stamped, at the end of ``fc_collect_finalize``::

    python -m scripts.ci.experiment_arms --db data/resolver.duckdb \
        --fc-run-id fc_... --out diagnostics/experiment_arms__fc_....json

Reads the rows the forecaster WROTE, so a question the stamp missed counts as
unstamped rather than disappearing. Never fails the stage: exits 0 always.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - import convenience
    sys.path.insert(0, str(REPO_ROOT))

#: arm -> (share variable, production default)
SHARES = {
    "advice": ("PYTHIA_ADVICE_EXPERIMENT_SHARE", 0.5),
    "rc_shift": ("PYTHIA_RC_SHIFT_SHARE", 0.5),
    "adversarial": ("PYTHIA_ADVERSARIAL_WITHHELD_SHARE", 0.5),
}


def share_of(arm: str, env: Mapping[str, str]) -> float | None:
    var, default = SHARES[arm]
    raw = env.get(var)
    if raw in (None, "", "<unset>"):
        return default
    try:
        return float(raw)
    except ValueError:
        return None


def _cols(con: Any, table: str) -> set[str]:
    try:
        return {str(r[1]).lower() for r in con.execute(f'PRAGMA table_info("{table}")').fetchall()}
    except Exception:  # noqa: BLE001
        return set()


def _breakdown(rows: list[tuple[str, str, str, str, str | None]]) -> dict[str, Any]:
    """rows: (question_id, track, hazard, metric, arm)."""
    counts = Counter(r[4] if r[4] else "(unstamped)" for r in rows)
    by: dict[str, Counter] = defaultdict(Counter)
    for qid, track, hz, metric, arm in rows:
        by[f"t{track}|{hz}/{metric}"][arm or "(unstamped)"] += 1
    return {"counts": {k: v for k, v in counts.items() if k != "(unstamped)"},
            "unstamped": counts.get("(unstamped)", 0),
            "by_track_hazard_metric": {k: dict(v) for k, v in sorted(by.items())}}


def collect(con: Any, fc_run_id: str, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    env = dict(os.environ if env is None else env)
    q = con.execute(
        """
        SELECT DISTINCT fe.question_id, CAST(q.track AS VARCHAR), upper(q.hazard_code), upper(q.metric)
        FROM forecasts_ensemble fe JOIN questions q ON q.question_id = fe.question_id
        WHERE fe.run_id = ?
        """, [fc_run_id]).fetchall()
    questions = {str(r[0]): (str(r[0]), str(r[1] or ""), str(r[2] or ""), str(r[3] or "")) for r in q}
    out: dict[str, Any] = {"fc_run_id": fc_run_id, "n_questions": len(questions),
                           "source": "forecasts_ensemble / forecasts_raw arm columns, as stamped"}
    ens_cols, raw_cols = _cols(con, "forecasts_ensemble"), _cols(con, "forecasts_raw")

    def per_question(table: str, column: str) -> dict[str, str | None]:
        rows = con.execute(
            f"SELECT question_id, max({column}) FROM {table} WHERE run_id = ? GROUP BY question_id",
            [fc_run_id]).fetchall()
        return {str(r[0]): (str(r[1]) if r[1] is not None else None) for r in rows}

    # Advice: every question; the ensemble row carries the applied arm.
    if "advice_arm" in ens_cols and share_of("advice", env):
        arms = per_question("forecasts_ensemble", "advice_arm")
        rows = [(*questions[qid], arms.get(qid)) for qid in questions]
        out["advice"] = {"design_share": share_of("advice", env), "treated_arm": "no_advice",
                         **_breakdown(rows)}
    else:
        out["advice"] = {"off": True, "reason": "no advice_arm column or the share is 0"}
    # RC shift: Track 1 SPD questions only; stamped on forecasts_raw.
    if "rc_shift_arm" in raw_cols and share_of("rc_shift", env):
        arms = per_question("forecasts_raw", "rc_shift_arm")
        rows = [(*questions[qid], arms.get(qid)) for qid, (_, track, _h, metric) in questions.items()
                if track == "1" and metric != "EVENT_OCCURRENCE"]
        out["rc_shift"] = {"design_share": share_of("rc_shift", env), "treated_arm": "shift",
                           "eligible_rule": "Track 1, not EVENT_OCCURRENCE", **_breakdown(rows)}
    else:
        out["rc_shift"] = {"off": True, "reason": "no rc_shift_arm column or the share is 0"}
    # Adversarial: eligible = stamped shown or withheld; n/a counted apart.
    if "adversarial_arm" in ens_cols and share_of("adversarial", env):
        arms = per_question("forecasts_ensemble", "adversarial_arm")
        rows = [(*questions[qid], arms.get(qid)) for qid in questions]
        b = _breakdown(rows)
        b["not_eligible"] = b["counts"].pop("n/a", 0)
        out["adversarial"] = {"design_share": share_of("adversarial", env), "treated_arm": "withheld",
                              "withheld_question_ids": sorted(q for q, a in arms.items() if a == "withheld"),
                              **b}
    else:
        out["adversarial"] = {"off": True, "reason": "no adversarial_arm column or the share is 0"}
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--db", required=True)
    p.add_argument("--fc-run-id", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    try:
        import duckdb

        path = args.db.replace("duckdb:///", "")
        con = duckdb.connect(path)  # read-write: a WAL left by an earlier step replays
        try:
            payload = collect(con, args.fc_run_id)
        finally:
            con.close()
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
        for arm in ("advice", "rc_shift", "adversarial"):
            a = payload.get(arm) or {}
            print(f"[experiment-arms] {arm}: {a.get('counts') or a.get('reason')} unstamped={a.get('unstamped')}")
    except Exception as exc:  # noqa: BLE001 - a diagnostic never fails the stage
        print(f"::warning title=experiment arms::{type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
