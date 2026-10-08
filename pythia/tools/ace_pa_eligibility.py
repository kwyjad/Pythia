# Pythia / Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
"""Which countries are asked the ACE/PA (conflict displacement) question.

Owner decision (8 Oct 2026, from the 13 Oct 2026 run): ACE/PA is asked only
for a country IDMC reports regularly. ACE/FATALITIES is unchanged.

Why. ACE/PA resolves from IDMC conflict displacement, which IDMC records when
a report reaches it: a blank month is unknown, not zero. For a country IDMC
reports in at least 8 of the 12 months before (a regular reporter,
``base_rate_spd.conflict_regular_reporter``), a month resolves to its figure
or, bracketed by a later report, to zero. For every other country a month
resolves only when IDMC happens to publish, which samples the bad months, so
its questions are scored as indicative (``pythia/tools/scoring_class.py``)
and add nothing to weights, advice, recalibration or headline skill. Asking
them cost about 100 model calls and $2.50 a run (1 Oct 2026: 31 of 38
questions) for no return. ACLED fatalities still covers conflict in every
country.

The rule is the resolver's own: :func:`base_rate_spd.conflict_regular_reporter`
for the window's FIRST month, over the series as it stands on the forecast
day. A country that starts reporting regularly gets the question
automatically, and one that stops loses it. Existing questions are left
alone: an open question for an irregular country keeps its forecast and is
scored as indicative.

The guard. If the IDMC series is missing, stale (its newest month more than
``STALE_MONTHS`` before the window), or the rule admits no country, the rule
is not trusted to drop every ACE/PA question: the previous production run's
list is used instead, and if there is none every ACE country is asked. The
decision, its source and its reason are stored in ``ace_pa_eligibility``
(one row per HS run) and printed as a workflow warning; the debug bundle
reports any run that did not use the rule.

Revisit: at the first resolutions of the admitted questions (11 December
2026), and consider a quarterly or annual IDMC question for large irregular
cases such as SDN and COD.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, List, Optional, Set

LOG = logging.getLogger(__name__)

#: The series is stale when its newest reported month is more than this many
#: months before the window's first month. On a healthy cycle (ingest on the
#: 11th, forecast on the 13th) the newest month is the window's first month
#: less two; four allows two missed ingests.
STALE_MONTHS = 4

SOURCE_RULE = "rule"
SOURCE_PREVIOUS_RUN = "previous_run"
SOURCE_ALL_ACE = "all_ace_countries"

TABLE = "ace_pa_eligibility"


@dataclass
class Decision:
    """Which ACE countries get an ACE/PA question in one run, and why."""

    window_start: str
    source: str
    countries: Set[str]
    reason: str = ""
    newest_month: Optional[str] = None
    rule_countries: Set[str] = field(default_factory=set)
    previous_hs_run_id: Optional[str] = None

    @property
    def used_rule(self) -> bool:
        return self.source == SOURCE_RULE

    def admits(self, iso3: str) -> bool:
        if self.source == SOURCE_ALL_ACE:
            return True
        return str(iso3 or "").upper() in self.countries


def ensure_table(con: Any) -> None:
    con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE} (
            hs_run_id TEXT PRIMARY KEY,
            window_start TEXT,
            source TEXT,
            reason TEXT,
            newest_month TEXT,
            countries_json TEXT,
            n_countries INTEGER,
            rule_countries_json TEXT,
            asked_json TEXT,
            not_asked_json TEXT,
            previous_hs_run_id TEXT,
            is_test BOOLEAN DEFAULT FALSE,
            decided_at TIMESTAMP DEFAULT now()
        )
        """
    )


def _previous_production_list(con: Any, hs_run_id: str) -> tuple[Optional[str], Set[str]]:
    """The newest production HS run before this one that asked any ACE/PA
    question, and the countries it asked it for."""
    try:
        row = con.execute(
            """
            SELECT rq.hs_run_id
            FROM run_questions rq
            JOIN hs_runs h ON h.hs_run_id = rq.hs_run_id
            WHERE upper(rq.hazard_code) = 'ACE' AND upper(rq.metric) = 'PA'
              AND rq.hs_run_id <> ?
              AND NOT COALESCE(rq.is_test, FALSE) AND NOT COALESCE(h.is_test, FALSE)
            GROUP BY rq.hs_run_id
            ORDER BY MAX(h.generated_at) DESC NULLS LAST, rq.hs_run_id DESC
            LIMIT 1
            """,
            [hs_run_id],
        ).fetchone()
        if not row:
            return None, set()
        isos = con.execute(
            "SELECT DISTINCT upper(iso3) FROM run_questions WHERE hs_run_id = ? "
            "AND upper(hazard_code) = 'ACE' AND upper(metric) = 'PA' AND iso3 IS NOT NULL",
            [row[0]],
        ).fetchall()
        return str(row[0]), {str(r[0]) for r in isos if r[0]}
    except Exception as exc:  # noqa: BLE001 - no history is a state, not a failure
        LOG.warning("ace_pa_eligibility: previous run lookup failed: %s", exc)
        return None, set()


def decide(con: Any, *, window_start: str, hs_run_id: str) -> Decision:
    """The ACE/PA country list for a run whose window opens in ``window_start``
    (``YYYY-MM``). Never raises: a failure is a guard condition."""
    from pythia.tools.base_rate_spd import (  # noqa: PLC0415
        _add_months,
        conflict_displacement_series,
        conflict_regular_reporter,
    )

    reason = ""
    newest: Optional[str] = None
    rule: Set[str] = set()
    try:
        reported, _held = conflict_displacement_series(con, before_ym=window_start)
        months = [m for d in reported.values() for m in d]
        newest = max(months) if months else None
        if not reported:
            reason = "IDMC conflict displacement series is missing (no rows)"
        elif newest < _add_months(window_start, -STALE_MONTHS):
            reason = (
                f"IDMC conflict displacement series is stale: newest month {newest}, "
                f"more than {STALE_MONTHS} months before {window_start}"
            )
        else:
            rule = {iso for iso, d in reported.items() if conflict_regular_reporter(d, window_start)}
            if not rule:
                reason = f"the regular-reporter rule admits no country for {window_start}"
    except Exception as exc:  # noqa: BLE001
        reason = f"IDMC conflict displacement series unreadable: {exc}"

    if not reason:
        return Decision(window_start, SOURCE_RULE, set(rule), "", newest, set(rule))
    prev_run, prev = _previous_production_list(con, hs_run_id)
    if prev:
        return Decision(window_start, SOURCE_PREVIOUS_RUN, prev,
                        reason + f"; using the ACE/PA list of {prev_run}", newest, rule, prev_run)
    return Decision(window_start, SOURCE_ALL_ACE, set(),
                    reason + "; no previous production run asked ACE/PA, so every ACE country is asked",
                    newest, rule, None)


def record(con: Any, decision: Decision, *, hs_run_id: str, asked: List[str],
           not_asked: List[str], is_test: bool) -> None:
    """Store the decision (one row per HS run) and say it where a person looks."""
    ensure_table(con)
    con.execute(f"DELETE FROM {TABLE} WHERE hs_run_id = ?", [hs_run_id])
    con.execute(
        f"""
        INSERT INTO {TABLE} (hs_run_id, window_start, source, reason, newest_month,
            countries_json, n_countries, rule_countries_json, asked_json, not_asked_json,
            previous_hs_run_id, is_test)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [hs_run_id, decision.window_start, decision.source, decision.reason, decision.newest_month,
         json.dumps(sorted(decision.countries)), len(decision.countries),
         json.dumps(sorted(decision.rule_countries)), json.dumps(sorted(asked)),
         json.dumps(sorted(not_asked)), decision.previous_hs_run_id, bool(is_test)],
    )
    line = (
        f"ACE/PA questions for {window_label(decision)}: asked for {len(asked)} ACE "
        f"countr{'y' if len(asked) == 1 else 'ies'} ({', '.join(sorted(asked)) or 'none'}), "
        f"not asked for {len(not_asked)} (displacement not forecast); source: {decision.source}"
    )
    print(line)
    if not decision.used_rule:
        print(f"::warning title=ACE/PA eligibility guard::{decision.reason}")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        try:
            with open(summary, "a", encoding="utf-8") as fh:
                fh.write(f"\n#### ACE/PA eligibility\n\n{line}\n")
                if not decision.used_rule:
                    fh.write(f"\n**Guard:** {decision.reason}\n")
        except OSError:
            pass


def window_label(decision: Decision) -> str:
    return f"the window opening {decision.window_start}"


def latest_decision(con: Any, hs_run_id: Optional[str] = None) -> Optional[dict]:
    """The stored decision for ``hs_run_id`` (or the newest), else None."""
    try:
        where = "WHERE hs_run_id = ?" if hs_run_id else ""
        params = [hs_run_id] if hs_run_id else []
        cur = con.execute(
            f"SELECT * FROM {TABLE} {where} ORDER BY decided_at DESC LIMIT 1", params
        )
        row = cur.fetchone()
        if not row:
            return None
        out = dict(zip([d[0] for d in cur.description], row))
        for key in ("countries_json", "rule_countries_json", "asked_json", "not_asked_json"):
            try:
                out[key.replace("_json", "")] = json.loads(out.get(key) or "[]")
            except (TypeError, ValueError):
                out[key.replace("_json", "")] = []
        return out
    except Exception:  # noqa: BLE001 - an older DB has no table
        return None


__all__ = [
    "Decision",
    "SOURCE_ALL_ACE",
    "SOURCE_PREVIOUS_RUN",
    "SOURCE_RULE",
    "STALE_MONTHS",
    "decide",
    "ensure_table",
    "latest_decision",
    "record",
]
