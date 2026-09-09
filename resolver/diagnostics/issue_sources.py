# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Turn the evidence a run already produces into register issues.

There is no second detection path here and there must never be one. A run
already computes contradiction checks, records every source fetch it made,
reconciles what each connector claimed against what its table gained, and
collapses its ERROR and WARNING lines into a histogram. Each of those is a
statement that something is wrong; none of them was reaching a reader. So
these are readers over the artefacts that exist, not new checks.

Every collector is pure over plain data — lists of dicts and rows — so the
bundle can pass what it holds in memory and a test can pass a fixture.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, Sequence

from resolver.diagnostics.issues import (
    DEGRADED,
    INFO,
    OWNER_EXTERNAL,
    OWNER_PYTHIA,
    Issue,
)

#: Failure classes a re-run cannot clear. A rejected credential answers the
#: same way tomorrow; a timeout does not.
_PERMANENT_FAILURE_CLASSES = frozenset({"auth_rejected", "no_key"})

#: Check name -> issue id, where the fault has a name of its own that the
#: known-issues register can carry. Without this a registered issue and the
#: check that finds it would be two different ids and the suppression would
#: never match.
CHECK_ISSUE_IDS: dict[str, str] = {
    "emdat_is_read_when_a_key_is_configured": "emdat_auth_rejected",
}

#: Source name (as the PA machine's fetch stream spells it) -> issue id.
SOURCE_ISSUE_IDS: dict[str, str] = {
    "emdat": "emdat_auth_rejected",
}

#: Conflict-forecast source (as `conflict_forecasts.source` spells it,
#: lowercased) -> issue id. The vintages are per source and the register is
#: too: one live source must not stand for two dead ones.
VINTAGE_ISSUE_IDS: dict[str, str] = {
    "acled_cast": "acled_cast_stale_vintage",
    "acledcast": "acled_cast_stale_vintage",
    "views": "views_stale_vintage",
    "conflictforecast": "conflictforecast_stale_vintage",
    "conflictforecast_org": "conflictforecast_stale_vintage",
}


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(text).strip().lower()).strip("_")
    return slug or "unnamed"


def issues_from_checks(checks: Iterable[Mapping[str, Any]]) -> list[Issue]:
    """A FAIL or an ERROR in the contradiction suite is an issue.

    A check that carries its own ``issues`` list wins: the vintage check
    knows it found two stale sources and names both, where its own name
    could only ever say "some source is stale". Everything else becomes one
    issue named after the check, which is already a stable slug.
    """

    out: list[Issue] = []
    for check in checks or []:
        verdict = str(check.get("verdict") or "").upper()
        if verdict not in ("FAIL", "ERROR"):
            continue
        name = str(check.get("name") or "check")
        declared = check.get("issues")
        if isinstance(declared, list) and declared:
            for raw in declared:
                if not isinstance(raw, Mapping):
                    continue
                out.append(Issue(
                    id=str(raw.get("id") or _slug(name)),
                    severity=str(raw.get("severity") or DEGRADED),
                    title=str(raw.get("title") or check.get("detail") or name),
                    evidence=str(raw.get("evidence") or check.get("detail") or ""),
                    cost=raw.get("cost"),
                    cost_unit=str(raw.get("cost_unit") or ""),
                    owner=str(raw.get("owner") or OWNER_PYTHIA),
                    recovers_on_rerun=bool(raw.get("recovers_on_rerun", True)),
                    source=f"contradiction check {name}",
                ))
            continue
        detail = str(check.get("detail") or "").strip()
        out.append(Issue(
            id=CHECK_ISSUE_IDS.get(name, name),
            severity=DEGRADED,
            title=_title_from_check(name, verdict),
            evidence=f"observed {check.get('left')!r} against expected "
                     f"{check.get('right')!r}. {detail}".strip(),
            cost=_cost_from_detail(detail),
            cost_unit="rows" if _cost_from_detail(detail) is not None else "",
            owner=OWNER_PYTHIA,
            source=f"contradiction check {name}",
        ))
    return out


def _title_from_check(name: str, verdict: str) -> str:
    readable = name.replace("_", " ")
    if verdict == "ERROR":
        return f"The check `{name}` could not run, so nothing verified {readable}."
    return f"Check failed: {readable}."


_LEADING_COUNT = re.compile(r"^\s*([0-9][0-9,]*)\b")


def _cost_from_detail(detail: str) -> float | None:
    """A check detail that opens with a count is stating its own cost.

    Most of them do — "2,089 flagged rows name no flag", "118 revisions".
    Where a detail does not open with a number, the cost is genuinely not
    measured and the register says so rather than printing a zero.
    """

    match = _LEADING_COUNT.match(detail or "")
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", ""))
    except ValueError:
        return None


def issues_from_source_fetches(records: Iterable[Mapping[str, Any]]) -> list[Issue]:
    """A source the machine could not READ is an issue; a stale one is info.

    ``ok=false`` is an UNREAD rung: the ladder lost a rung and every cell
    under it resolved without it. ``ok=true`` with ``served_from_cache`` is
    a STALE rung, which still answers — worth recording, not worth alarming
    about.
    """

    unread: dict[str, dict[str, Any]] = {}
    cached: dict[str, int] = {}
    for record in records or []:
        source = str(record.get("source") or "unknown")
        if record.get("ok"):
            if record.get("served_from_cache"):
                cached[source] = cached.get(source, 0) + 1
            continue
        entry = unread.setdefault(source, {
            "n": 0, "classes": set(), "error": "",
        })
        entry["n"] += 1
        failure_class = str(record.get("failure_class") or "other")
        entry["classes"].add(failure_class)
        if not entry["error"] and record.get("error"):
            entry["error"] = str(record["error"])[:200]

    out: list[Issue] = []
    for source, entry in sorted(unread.items()):
        classes = sorted(entry["classes"])
        permanent = bool(set(classes) & _PERMANENT_FAILURE_CLASSES)
        out.append(Issue(
            id=SOURCE_ISSUE_IDS.get(source, f"source_unread_{_slug(source)}"),
            severity=DEGRADED,
            title=f"{source} could not be read, so every cell that wanted it "
                  f"resolved without that rung.",
            evidence=f"{entry['n']} failed fetch(es); failure class "
                     f"{', '.join(classes)}. {entry['error']}".strip(),
            cost=float(entry["n"]),
            cost_unit="failed fetches",
            owner=OWNER_EXTERNAL if permanent else OWNER_PYTHIA,
            recovers_on_rerun=not permanent,
            source="hazard/source_fetches.csv",
        ))
    for source, count in sorted(cached.items()):
        out.append(Issue(
            id=f"source_served_from_cache_{_slug(source)}",
            severity=INFO,
            title=f"{source} answered from the cache rather than live — a STALE "
                  f"rung, which still answers.",
            evidence=f"{count} fetch(es) served from cache",
            cost=float(count),
            cost_unit="cache-served fetches",
            source="hazard/source_fetches.csv",
        ))
    return out


def issues_from_reconciliation(rows: Iterable[Mapping[str, Any]]) -> list[Issue]:
    """A connector that claimed rows and stamped none is an issue.

    ``agrees`` comes from the reconciliation table: ``yes`` where the
    connector's own source filter gained rows carrying this run's write
    stamp, ``unchanged on purpose`` where a content-hash guarded writer
    correctly declined, and ``**NO**`` where a connector claimed rows the
    table cannot show.
    """

    out: list[Issue] = []
    for row in rows or []:
        agrees = str(row.get("agrees") or "").strip().strip("*").lower()
        connector = str(row.get("connector") or "unknown")
        status = str(row.get("status") or "")
        if agrees == "no":
            out.append(Issue(
                id=f"connector_wrote_nothing_{_slug(connector)}",
                severity=DEGRADED,
                title=f"{connector} reported writing rows its target table cannot show.",
                evidence=f"claimed {row.get('claimed')} row(s) into "
                         f"{row.get('table') or 'an unknown table'}; "
                         f"{row.get('touched')} carry this run's write stamp",
                cost=_as_float(row.get("claimed")),
                cost_unit="claimed rows unaccounted for",
                source="checks/reconciliation.md",
            ))
        elif status in ("error", "failed"):
            out.append(Issue(
                id=f"connector_failed_{_slug(connector)}",
                severity=DEGRADED,
                title=f"{connector} did not complete.",
                evidence=str(row.get("reason") or status),
                source="checks/reconciliation.md",
            ))
    return out


def _as_float(value: Any) -> float | None:
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def issues_from_log_histogram(
    shapes: Iterable[tuple[str, str, int]], *, min_count: int = 1
) -> list[Issue]:
    """The collapsed ERROR histogram, one issue per distinct shape.

    ``shapes`` is ``(log name, normalised line, count)`` — already collapsed
    by the bundle's log index, which replaces ISO3 codes, dates and numbers
    with placeholders so one fault repeated 252 times is one row. WARNINGs
    are deliberately excluded: a run legitimately warns dozens of times and
    an issue register that carries all of them is a log with a border round
    it. They stay in `logs/log_index.md`, where they belong.
    """

    out: list[Issue] = []
    for log_name, line, count in shapes or []:
        level, _, message = str(line).partition(":")
        if level.strip().upper() not in ("ERROR", "CRITICAL"):
            continue
        if int(count) < min_count:
            continue
        message = message.strip() or str(line)
        out.append(Issue(
            id=f"log_error_{_slug(log_name)}_{_slug(message)[:48]}",
            severity=DEGRADED,
            title=f"{log_name} logged an error {count}x: {message[:160]}",
            evidence=f"normalised line: {message[:300]}",
            cost=float(count),
            cost_unit="log lines",
            source=f"logs/{log_name}",
        ))
    return out


def issue_from_measurement(
    issue_id: str,
    title: str,
    *,
    severity: str = INFO,
    evidence: str = "",
    cost: float | None = None,
    cost_unit: str = "",
    owner: str = OWNER_PYTHIA,
    recovers_on_rerun: bool = True,
    source: str = "",
) -> Issue:
    """A measurement worth carrying: budget headroom, a refusal rate, a recovery rate."""

    return Issue(
        id=issue_id, severity=severity, title=title, evidence=evidence,
        cost=cost, cost_unit=cost_unit, owner=owner,
        recovers_on_rerun=recovers_on_rerun, source=source,
    )


__all__ = [
    "CHECK_ISSUE_IDS", "SOURCE_ISSUE_IDS", "VINTAGE_ISSUE_IDS",
    "issues_from_checks", "issues_from_source_fetches",
    "issues_from_reconciliation", "issues_from_log_histogram",
    "issue_from_measurement",
]
