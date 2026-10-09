# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""What a Sibyl run read and how it went about it (Oct 2026, Part 6).

* ``write_evidence`` stores one ``sibyl_evidence`` row per tool result a
  trial saw. The rows are built on the trial (``TrialResult.evidence_rows``)
  and written here, on the main thread, after the question's trials end.
* ``process_measures`` turns a run's outcomes into the process figures
  ``sibyl_runs`` carries. They describe HOW Sibyl researched, never whether
  it was right: accuracy is measured only against resolutions.
* ``reference_weight`` picks the weight of Sibyl's reference in the
  published pool: the fitted one (``sibyl_pool_weights``) or the fixed knob.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from sibyl import config as _cfg

logger = logging.getLogger(__name__)

_EVIDENCE_COLUMNS = (
    "sibyl_run_id", "question_id", "trial_index", "step", "call_index", "tool",
    "target", "lane", "retrieved_at", "http_status", "ok", "sha256",
    "shown_text", "doc_text", "doc_chars", "is_test", "role",
)


def write_evidence(
    con: Any,
    *,
    sibyl_run_id: str,
    question_id: str,
    trials: Iterable[Any],
    is_test: bool = False,
    replace: bool = True,
    role: Optional[str] = None,
) -> int:
    """Write every trial's evidence rows. Returns the rows written; never raises.

    *replace* deletes the question's earlier rows for this run first (the
    production write); the shadow phase appends with ``replace=False``. Each
    row carries the trial's role (*role* overrides it).
    """
    rows: List[Tuple[Any, ...]] = []
    for t in trials or []:
        for r in getattr(t, "evidence_rows", None) or []:
            rows.append((
                sibyl_run_id, question_id, int(t.trial_index), r.get("step"),
                r.get("call_index"), r.get("tool"), r.get("target"), r.get("lane"),
                r.get("retrieved_at"), r.get("http_status"), r.get("ok"),
                r.get("sha256"), r.get("shown_text"), r.get("doc_text"),
                r.get("doc_chars"), bool(is_test),
                role or getattr(t, "role", None) or "production",
            ))
    if not rows:
        return 0
    try:
        from pythia.db.schema import ensure_sibyl_measurement_tables  # noqa: PLC0415

        ensure_sibyl_measurement_tables(con)
        if replace:
            con.execute(
                "DELETE FROM sibyl_evidence WHERE sibyl_run_id = ? AND question_id = ?",
                [sibyl_run_id, question_id],
            )
        placeholders = ", ".join("?" for _ in _EVIDENCE_COLUMNS)
        con.executemany(
            f"INSERT INTO sibyl_evidence ({', '.join(_EVIDENCE_COLUMNS)}) "
            f"VALUES ({placeholders})",
            rows,
        )
        return len(rows)
    except Exception as exc:  # noqa: BLE001 - the evidence record never stops a run
        logger.warning("sibyl.measure: evidence write failed for %s: %s", question_id, exc)
        return 0


def _share(num: float, den: float) -> Optional[float]:
    return (num / den) if den else None


def _is_wikipedia(url: str) -> bool:
    from urllib.parse import urlparse  # noqa: PLC0415

    try:
        host = (urlparse(str(url)).hostname or "").lower()
    except ValueError:
        return False
    return host == "wikipedia.org" or host.endswith(".wikipedia.org")


def process_measures(outcomes: Sequence[Any]) -> Dict[str, Optional[float]]:
    """The process figures over a run's question outcomes.

    * share_resolver_done: trials whose resolver plan slot ended 'done'.
    * docs_per_trial: documents read per trial.
    * share_ledger_dated_figure: ledger items carrying a date AND a figure
      (a quote holding a digit).
    * share_forecasts_at_floor: written forecasts with any bucket, in any
      window month, at the bucket floor.
    * mean_jsd_from_reference: mean month-1 JSD of the raw pool from the
      reference, over forecasts that had both.

    Research depth (Oct 2026, review Part 1):

    * median_docs_per_trial: the median trial's documents read.
    * share_trials_under_doc_gate: trials that read fewer documents than
      SIBYL_SUBMIT_MIN_DOCS (they ended at the step limit).
    * steps_per_trial / tool_calls_per_trial: means over trials.
    * share_docs_wikipedia: documents read whose host ends in wikipedia.org.
    * n_submit_gate_unmet: trials the step limit ended with the submit gate
      unmet.
    """
    from sibyl.spd import _js_divergence  # noqa: PLC0415

    trials = [t for o in outcomes for t in (getattr(o, "trials", None) or [])]
    n_trials = len(trials)
    resolver_done = sum(1 for t in trials if getattr(t, "resolver_status", None) == "done")
    docs = sum(int(getattr(t, "n_docs_read", 0) or 0) for t in trials)
    items = [it for t in trials for it in (getattr(t, "ledger", None) or [])]
    dated_fig = sum(
        1 for it in items
        if it.get("date") and any(ch.isdigit() for ch in str(it.get("quote") or ""))
    )
    written = [o for o in outcomes if getattr(o, "final_by_month", None)]
    floor = float(_cfg.BUCKET_FLOOR)
    at_floor = sum(
        1 for o in written
        if any(min(v) <= floor + 1e-9 for v in o.final_by_month.values() if v)
    )
    jsds = [
        _js_divergence(o.raw_month1, o.reference_month1)
        for o in written
        if getattr(o, "raw_month1", None) and getattr(o, "reference_month1", None)
        and len(o.raw_month1) == len(o.reference_month1)
    ]
    doc_counts = sorted(int(getattr(t, "n_docs_read", 0) or 0) for t in trials)
    median_docs: Optional[float] = None
    if doc_counts:
        mid = len(doc_counts) // 2
        median_docs = (
            float(doc_counts[mid]) if len(doc_counts) % 2
            else (doc_counts[mid - 1] + doc_counts[mid]) / 2.0
        )
    under_gate = sum(1 for n in doc_counts if n < int(_cfg.SUBMIT_MIN_DOCS))
    steps = sum(int(getattr(t, "steps_used", 0) or 0) for t in trials)
    calls = sum(int(getattr(t, "n_tool_calls", 0) or 0) for t in trials)
    doc_urls = [u for t in trials for u in (getattr(t, "docs_read_urls", None) or [])]
    wiki = sum(1 for u in doc_urls if _is_wikipedia(u))
    gate_unmet = sum(1 for t in trials if getattr(t, "submit_gate_unmet", False))
    return {
        "share_resolver_done": _share(resolver_done, n_trials),
        "docs_per_trial": _share(docs, n_trials),
        "share_ledger_dated_figure": _share(dated_fig, len(items)),
        "share_forecasts_at_floor": _share(at_floor, len(written)),
        "mean_jsd_from_reference": (sum(jsds) / len(jsds)) if jsds else None,
        "median_docs_per_trial": median_docs,
        "share_trials_under_doc_gate": _share(under_gate, n_trials),
        "steps_per_trial": _share(steps, n_trials),
        "tool_calls_per_trial": _share(calls, n_trials),
        "share_docs_wikipedia": _share(wiki, len(doc_urls)),
        "n_submit_gate_unmet": (gate_unmet if n_trials else None),
    }


def reference_weight(con: Any, as_of_month: Optional[str] = None) -> Tuple[float, str]:
    """(weight, source) for the published pool.

    Backtest and ``SIBYL_REFERENCE_WEIGHT_MODE=fixed`` use
    ``SIBYL_REFERENCE_WEIGHT``. Otherwise the newest ``sibyl_pool_weights``
    row on or before *as_of_month* (status 'fitted' or 'prior'); with none,
    or on any read failure, the fixed knob.
    """
    fixed = float(_cfg.REFERENCE_WEIGHT)
    if _cfg.BACKTEST_MODE:
        return fixed, "backtest"
    if _cfg.REFERENCE_WEIGHT_MODE != "fitted":
        return fixed, "fixed"
    try:
        params: List[Any] = []
        where = "weight IS NOT NULL"
        if as_of_month:
            where += " AND as_of_month <= ?"
            params.append(as_of_month)
        row = con.execute(
            f"SELECT weight FROM sibyl_pool_weights WHERE {where} "
            "ORDER BY as_of_month DESC LIMIT 1",
            params,
        ).fetchone()
    except Exception:  # noqa: BLE001 - an older DB has no table
        return fixed, "fixed"
    if not row or row[0] is None:
        return fixed, "fixed"
    return float(row[0]), "fitted"
