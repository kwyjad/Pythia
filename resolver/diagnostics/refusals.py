# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Which resources a connector was refused, and whether they are the same ones.

A refusal rate says how often a source said no. It cannot say what kind of
no it was, and the two kinds want opposite responses.

If the SAME resources are refused every run, the refusal is a property of
those resources — a published report GDACS never wrote, an event withdrawn,
a route retired for that item. Asking once is right, and asking at all is
waste: the answer will not change tomorrow.

If DIFFERENT resources are refused each run, the refusal is a property of
the ASKING — throttling, a rate ceiling, a bot filter sampling traffic. The
documents are recoverable and asking once is costing evidence, because the
event refused tonight would have answered this morning.

Two measurements already refuted the throttling reading for GDACS: cutting
volume by 73% moved the refusal rate 79.6% to 76%, and 153 of 291 events
were served on the first request while 138 were refused on all four. Both
point the same way, and neither is the direct test. This is the direct
test, and it costs no request: the URL of every refusal is already in the
run's own HTTP stream.

Nothing here changes what the connector asks for. It reads the stream a run
already writes, keeps the last few runs' sets in the canonical database (the
only store that travels between runs), and reports which pattern the data
shows.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping
from urllib.parse import parse_qsl, urlsplit, urlunsplit

from resolver.diagnostics.issues import INFO, OWNER_EXTERNAL, Issue
from resolver.diagnostics.redaction import redact_text

#: Where the per-run sets live. The canonical DB is the only store that
#: travels between runs, for the same reason the issue history lives there.
REFUSAL_TABLE = "diagnostic_refused_resources"

#: Statuses that count as a refusal. A 404 is the source saying the thing
#: does not exist, which is a different statement from "not for you", and
#: mixing them would make a connector walking a sparse archive look throttled.
REFUSAL_STATUSES = frozenset({401, 403, 407, 429, 451})

#: Runs kept per connector. Enough to see whether an overlap is stable
#: rather than one night's coincidence, few enough that the table stays a
#: few thousand rows — this database has been ruined by an append-only
#: cache once already.
KEEP_RUNS = 5

#: Resources stored per connector per run. A connector that asks for
#: 40,000 URLs is not the shape this answers a question about, and the
#: truncation is recorded rather than silent.
MAX_RESOURCES_PER_RUN = 4000

#: Above this share of a run's refusals being refused again, the refusal
#: belongs to the resource.
SAME_RESOURCE_SHARE = 0.80

#: Below this share, it belongs to the asking.
DIFFERENT_RESOURCE_SHARE = 0.30

#: Below this many comparable resources the share is arithmetic on noise.
MIN_COMPARABLE = 20

VERDICT_SAME = "same resources refused"
VERDICT_DIFFERENT = "different resources refused"
VERDICT_MIXED = "mixed"
VERDICT_INCONCLUSIVE = "inconclusive"

#: Query parameters that vary per run and would make two identical asks
#: look like two different ones. Only names that are unambiguously a clock
#: or a nonce; a date window is deliberately NOT here, because a connector
#: whose window moves really is asking a different question.
_VOLATILE_QUERY_KEYS = frozenset({
    "_", "t", "ts", "timestamp", "cachebust", "cache_bust", "nocache", "rand",
})


def resource_key(url: str) -> str | None:
    """A stable identity for the thing a request asked for.

    The URL itself, minus the scheme, minus parameters that are a clock.
    Deliberately not a per-connector parsing rule: GDACS puts its event id
    in the path of one route and the query of another, and a rule written
    for those two would answer nothing about the third.
    """

    raw = str(url or "").strip()
    if not raw:
        return None
    try:
        parts = urlsplit(raw)
    except ValueError:
        return None
    if not parts.netloc:
        return None
    kept = [
        (k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if k.lower() not in _VOLATILE_QUERY_KEYS
    ]
    query = "&".join(f"{k}={v}" for k, v in sorted(kept))
    return urlunsplit(("", parts.netloc, parts.path, query, "")).lstrip("/")


@dataclass
class ConnectorRefusals:
    """One connector's asks and refusals in one run."""

    connector: str
    asked: set[str] = field(default_factory=set)
    refused: set[str] = field(default_factory=set)
    truncated: bool = False

    @property
    def refusal_share(self) -> float:
        return len(self.refused) / len(self.asked) if self.asked else 0.0


def refusals_from_records(
    records: Iterable[Mapping[str, Any]],
    *,
    connectors: Iterable[str] | None = None,
) -> dict[str, ConnectorRefusals]:
    """Read the run's own HTTP stream. No new detection path.

    A call a DIAGNOSTIC made is excluded: a probe asks questions no
    connector would, and one of this repository's checks has already been
    fooled by counting a probe's 405 as connector traffic.
    """

    wanted = {str(c) for c in connectors} if connectors is not None else None
    out: dict[str, ConnectorRefusals] = {}
    for record in records or []:
        if record.get("probe"):
            continue
        connector = str(record.get("connector") or "unknown")
        if wanted is not None and connector not in wanted:
            continue
        key = resource_key(str(record.get("url") or ""))
        if key is None:
            continue
        entry = out.setdefault(connector, ConnectorRefusals(connector=connector))
        if len(entry.asked) >= MAX_RESOURCES_PER_RUN and key not in entry.asked:
            entry.truncated = True
            continue
        entry.asked.add(key)
        try:
            status = int(record.get("status"))
        except (TypeError, ValueError):
            continue
        if status in REFUSAL_STATUSES:
            entry.refused.add(key)
    return out


@dataclass
class RefusalComparison:
    """This run's refusals against the previous run's."""

    connector: str
    previous_run_id: str
    n_refused_now: int
    n_refused_prev: int
    #: Refused now AND asked for by the previous run. The only resources
    #: about which the two runs can be compared at all: a resource the
    #: previous run never requested says nothing about whether it would
    #: have been refused.
    n_comparable: int
    n_refused_both: int
    verdict: str

    @property
    def overlap(self) -> float | None:
        if self.n_comparable <= 0:
            return None
        return self.n_refused_both / self.n_comparable


def compare(
    connector: str,
    now: ConnectorRefusals,
    previous_run_id: str,
    previous_asked: set[str],
    previous_refused: set[str],
) -> RefusalComparison:
    comparable = now.refused & previous_asked
    both = comparable & previous_refused
    n_comparable = len(comparable)
    share = (len(both) / n_comparable) if n_comparable else None
    if share is None or n_comparable < MIN_COMPARABLE:
        verdict = VERDICT_INCONCLUSIVE
    elif share >= SAME_RESOURCE_SHARE:
        verdict = VERDICT_SAME
    elif share <= DIFFERENT_RESOURCE_SHARE:
        verdict = VERDICT_DIFFERENT
    else:
        verdict = VERDICT_MIXED
    return RefusalComparison(
        connector=connector,
        previous_run_id=str(previous_run_id),
        n_refused_now=len(now.refused),
        n_refused_prev=len(previous_refused),
        n_comparable=n_comparable,
        n_refused_both=len(both),
        verdict=verdict,
    )


_VERDICT_MEANING = {
    VERDICT_SAME: (
        "the refusal belongs to those resources, not to the pace — asking "
        "once is right and asking at all is waste, because the answer will "
        "not change tomorrow"
    ),
    VERDICT_DIFFERENT: (
        "the refusal belongs to the asking rather than to the resources, "
        "which is what throttling looks like — the documents are "
        "recoverable and asking once is costing evidence. Bring the retry "
        "question back rather than acting on this alone"
    ),
    VERDICT_MIXED: (
        "neither pattern cleanly — some resources refuse every run and "
        "some do not, so both explanations are partly true"
    ),
    VERDICT_INCONCLUSIVE: (
        "too few resources both runs asked about to say. The share is not "
        "reported because arithmetic on a handful of rows is not evidence"
    ),
}


def issue_for(comparison: RefusalComparison) -> Issue:
    """The finding, at ``info``: nothing is broken, something is now known."""

    overlap = comparison.overlap
    share_text = "not comparable" if overlap is None else f"{overlap:.0%}"
    title = (
        f"{comparison.connector} refusals across runs: {comparison.verdict} "
        f"({share_text} of {comparison.n_comparable} comparable resources "
        f"refused in both runs)"
    )
    evidence = (
        f"{comparison.n_refused_now} resource(s) refused this run, "
        f"{comparison.n_refused_prev} in run {comparison.previous_run_id}; "
        f"{comparison.n_comparable} of this run's refusals were also asked "
        f"for then, and {comparison.n_refused_both} of those were refused "
        f"then too. Read as: {_VERDICT_MEANING[comparison.verdict]}. "
        "Derived from the run's own HTTP stream; no extra request was made."
    )
    return Issue(
        id=f"refusal_pattern_{_slugify(comparison.connector)}",
        severity=INFO,
        title=title,
        evidence=evidence,
        cost=float(comparison.n_refused_now),
        cost_unit="refused resources",
        owner=OWNER_EXTERNAL,
        recovers_on_rerun=(comparison.verdict == VERDICT_DIFFERENT),
        source="http/refused_resources.csv",
    )


def _slugify(text: str) -> str:
    out = re.sub(r"[^a-z0-9]+", "_", str(text).strip().lower()).strip("_")
    return out or "unnamed"


# -- state that travels between runs -------------------------------------


def ensure_refusal_table(con: Any) -> None:
    con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {REFUSAL_TABLE} (
            run_id TEXT NOT NULL,
            connector TEXT NOT NULL,
            resource TEXT NOT NULL,
            refused BOOLEAN NOT NULL,
            recorded_at DATE,
            PRIMARY KEY (run_id, connector, resource)
        )
        """
    )


def save_run(
    con: Any,
    run_id: str,
    entry: ConnectorRefusals,
    *,
    secrets: Iterable[str] | None = None,
    today: dt.date | None = None,
) -> int:
    """Record what this run asked for and what it was refused.

    URLs reach here already redacted by the recorder. They are redacted
    again on the way in, because capture-time redaction is a first line and
    never the guarantee — this table travels in a public artifact.
    """

    if not run_id or not entry.asked:
        return 0
    ensure_refusal_table(con)
    stamp = (today or dt.date.today()).isoformat()
    secret_list = list(secrets or [])
    rows = [
        (str(run_id), entry.connector, redact_text(resource, secret_list),
         resource in entry.refused, stamp)
        for resource in sorted(entry.asked)
    ]
    con.executemany(
        f"""
        INSERT INTO {REFUSAL_TABLE}
            (run_id, connector, resource, refused, recorded_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT (run_id, connector, resource) DO UPDATE SET
            refused = excluded.refused,
            recorded_at = excluded.recorded_at
        """,
        rows,
    )
    return len(rows)


def previous_run(con: Any, connector: str, current_run_id: str) -> tuple[str, set[str], set[str]] | None:
    """The most recent earlier run that recorded this connector.

    ``None`` when there is no earlier run — which is the honest answer on
    the first run after this shipped, and must not be rendered as an
    overlap of zero.
    """

    ensure_refusal_table(con)
    row = con.execute(
        f"""
        SELECT run_id FROM {REFUSAL_TABLE}
        WHERE connector = ? AND run_id <> ?
        GROUP BY run_id
        ORDER BY MAX(recorded_at) DESC, run_id DESC
        LIMIT 1
        """,
        [connector, str(current_run_id)],
    ).fetchone()
    if not row:
        return None
    previous_id = str(row[0])
    rows = con.execute(
        f"SELECT resource, refused FROM {REFUSAL_TABLE} "
        f"WHERE connector = ? AND run_id = ?",
        [connector, previous_id],
    ).fetchall()
    asked = {str(r[0]) for r in rows}
    refused = {str(r[0]) for r in rows if bool(r[1])}
    return previous_id, asked, refused


def prune(con: Any, *, keep_runs: int = KEEP_RUNS) -> int:
    """Keep the last few runs per connector and drop the rest."""

    ensure_refusal_table(con)
    deleted = con.execute(
        f"""
        DELETE FROM {REFUSAL_TABLE} WHERE (connector, run_id) IN (
            SELECT connector, run_id FROM (
                SELECT connector, run_id,
                       ROW_NUMBER() OVER (
                           PARTITION BY connector
                           ORDER BY MAX(recorded_at) DESC, run_id DESC
                       ) AS rn
                FROM {REFUSAL_TABLE}
                GROUP BY connector, run_id
            ) WHERE rn > ?
        )
        """,
        [int(keep_runs)],
    )
    try:
        return int(deleted.fetchone()[0])
    except Exception:  # noqa: BLE001 - DuckDB does not always return a count
        return 0
