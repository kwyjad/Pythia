# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Rebuild trial.json, the Sibyl cost-gate fixture, from a release DB.

One recorded eight-step trial of the 13 October 2026 run: the model's
replies word for word, its completion tokens per step (thinking included),
and the size of each step's tool results. The tool results themselves are
NOT copied (they are third-party page text); the projection stands filler
of the same size in their place.

    python tests/fixtures/sibyl/cost_gate/build_fixture.py --db release.duckdb
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import duckdb

HERE = Path(__file__).resolve().parent


def build(db: str, hs_run_id: str, question_id: str, trial: int) -> dict:
    con = duckdb.connect(db, read_only=True)
    rows = con.execute(
        """
        SELECT call_type, prompt_text, response_text, completion_tokens
        FROM llm_calls
        WHERE phase = 'sibyl' AND provider = 'anthropic' AND hs_run_id = ?
          AND question_id = ? AND call_type LIKE ? AND COALESCE(error_text, '') = ''
        ORDER BY timestamp
        """,
        [hs_run_id, question_id, f"sibyl_trial{trial}_step%"],
    ).fetchall()
    steps = {}
    for ct, prompt, response, completion in rows:
        m = re.search(r"_step(\d+)$", ct)
        if m:
            steps[int(m.group(1))] = (prompt, response, int(completion or 0))
    n = max(steps)
    replies, completion, result_chars = [], [], []
    for s in range(1, n + 1):
        prompt, response, comp = steps[s]
        replies.append(response)
        completion.append(comp)
        if s < n:
            # The next step's logged prompt is the previous prefix (a marker
            # line) plus step s's transcript entry and the next tail. The
            # entry's tool results are everything after its RESULTS heading.
            nxt = steps[s + 1][0]
            head = f"=== STEP {s}: RESULTS ==="
            tail = f"=== STEP {s + 1} of "
            i, j = nxt.find(head), nxt.rfind(tail)
            body = nxt[i + len(head):j] if i >= 0 and j > i else ""
            parts = [p for p in re.split(r"\n\n(?=--- )", body.strip()) if p.strip()]
            result_chars.append([len(p) for p in parts])
        else:
            result_chars.append([])
    return {
        "source": {"hs_run_id": hs_run_id, "question_id": question_id, "trial_index": trial},
        "replies": replies,
        "completion_tokens": completion,
        "tool_result_chars": result_chars,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--hs-run-id", default="hs_20261013T041422")
    ap.add_argument("--question-id", default="YEM_ACE_FATALITIES_2026-11")
    ap.add_argument("--trial", type=int, default=1)
    args = ap.parse_args()
    out = build(args.db, args.hs_run_id, args.question_id, args.trial)
    (HERE / "trial.json").write_text(json.dumps(out, indent=1))
    print({k: (len(v) if isinstance(v, list) else v) for k, v in out.items()})


if __name__ == "__main__":
    main()
