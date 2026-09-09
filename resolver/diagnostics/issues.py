# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""One register of what went wrong in a run, printed where a human will see it.

Run 34222175003 finished green. Two contradiction checks had failed, EM-DAT
had contributed nothing all run, and the flood exposure ceiling was broken
across the whole series. None of that reached anybody without opening a
172-file zip.

Making a failed check fail the run is the wrong answer to that. A non-zero
exit throws away every connector's output for the sake of a fault the run
cannot repair, and several of these faults have owners outside this
repository — an expired EM-DAT tier and a frozen ACLED CAST vintage are not
things a re-run fixes. What is wanted is a report nobody can miss and an
exit code that still says whether the run produced its output.

So this module carries the model and the renderers, and nothing else
decides. Four severities:

``blocking``
    the run could not write its output. This is the only severity that
    belongs anywhere near an exit code.
``degraded``
    the run produced output and something in it is wrong or missing.
``known``
    the same, for a fault already registered in
    ``resolver/config/known_issues.yml`` with an owner and a review date.
    Reported once, as one line, and annotated as a notice rather than an
    error — a register that does not quieten the noise it is meant to
    quieten is just a second place to read the same alarms.
``info``
    a measurement worth carrying, not a fault.

A suppression that never expires is how a known issue becomes an invisible
one, so an entry past its ``review_by`` still reports at ``known`` and adds
one line saying the entry is overdue.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

try:  # pragma: no cover - PyYAML is a base dependency; the fallback is belt
    import yaml
except Exception:  # noqa: BLE001 - a missing parser costs the register, not the run
    yaml = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Where the register of already-known, already-owned faults lives.
KNOWN_ISSUES_PATH = REPO_ROOT / "resolver" / "config" / "known_issues.yml"

BLOCKING = "blocking"
DEGRADED = "degraded"
KNOWN = "known"
INFO = "info"

#: Worst first. The report is ordered by this and nothing else, so a
#: `blocking` issue can never sort below a chatty `info` one.
SEVERITY_ORDER: tuple[str, ...] = (BLOCKING, DEGRADED, KNOWN, INFO)
_SEVERITY_RANK = {name: i for i, name in enumerate(SEVERITY_ORDER)}

#: GitHub workflow-command level per severity. `known` is a NOTICE on
#: purpose: the whole point of registering an issue is that it stops
#: shouting.
_ANNOTATION = {
    BLOCKING: "error",
    DEGRADED: "error",
    KNOWN: "notice",
    INFO: "notice",
}

OWNER_PYTHIA = "pythia"
OWNER_EXTERNAL = "external"


def _today() -> dt.date:
    """Today, overridable so the tests are not a function of the calendar."""

    stamp = (os.getenv("PYTHIA_ISSUES_TODAY") or "").strip()
    if stamp:
        try:
            return dt.date.fromisoformat(stamp[:10])
        except ValueError:
            pass
    return dt.date.today()


@dataclass
class Issue:
    """One distinct fault, with what it cost and whether a re-run helps.

    ``cost`` is a number in whatever ``cost_unit`` names — rows, cells,
    figures. An issue with nothing measurable to report says so by leaving
    ``cost`` None, which renders as "not measured" rather than as zero: a
    zero there reads as "this cost nothing", which is a different claim.
    """

    id: str
    severity: str
    title: str
    evidence: str = ""
    cost: float | None = None
    cost_unit: str = ""
    owner: str = OWNER_PYTHIA
    recovers_on_rerun: bool = True
    first_seen: str = ""
    runs_seen: int = 1
    #: Set when a known-issues entry matched: who is chasing it and by when.
    note: str = ""
    review_by: str = ""
    overdue: bool = False
    #: Where the issue came from, so a reader can go to the evidence.
    source: str = ""

    @property
    def rank(self) -> int:
        return _SEVERITY_RANK.get(self.severity, len(SEVERITY_ORDER))

    def cost_text(self) -> str:
        if self.cost is None:
            return "not measured"
        number = int(self.cost) if float(self.cost).is_integer() else self.cost
        unit = f" {self.cost_unit}" if self.cost_unit else ""
        return f"{number:,}{unit}" if isinstance(number, int) else f"{number}{unit}"

    def age_text(self) -> str:
        if self.runs_seen > 1 and self.first_seen:
            return f"seen in {self.runs_seen} runs since {self.first_seen}"
        if self.first_seen:
            return f"first seen {self.first_seen}"
        return "first seen in this run"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "severity": self.severity,
            "title": self.title,
            "evidence": self.evidence,
            "cost": self.cost,
            "cost_unit": self.cost_unit,
            "owner": self.owner,
            "recovers_on_rerun": self.recovers_on_rerun,
            "first_seen": self.first_seen,
            "runs_seen": self.runs_seen,
            "note": self.note,
            "review_by": self.review_by,
            "overdue": self.overdue,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Issue":
        return cls(
            id=str(payload.get("id") or "unknown"),
            severity=str(payload.get("severity") or DEGRADED),
            title=str(payload.get("title") or ""),
            evidence=str(payload.get("evidence") or ""),
            cost=payload.get("cost"),
            cost_unit=str(payload.get("cost_unit") or ""),
            owner=str(payload.get("owner") or OWNER_PYTHIA),
            recovers_on_rerun=bool(payload.get("recovers_on_rerun", True)),
            first_seen=str(payload.get("first_seen") or ""),
            runs_seen=int(payload.get("runs_seen") or 1),
            note=str(payload.get("note") or ""),
            review_by=str(payload.get("review_by") or ""),
            overdue=bool(payload.get("overdue", False)),
            source=str(payload.get("source") or ""),
        )


@dataclass
class KnownIssues:
    """The register of faults already diagnosed, owned and being chased."""

    entries: dict[str, dict[str, Any]] = field(default_factory=dict)
    path: Path | None = None
    problems: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path | None = None) -> "KnownIssues":
        """Read the register. A missing or broken file costs suppression only.

        Failing to read it must never fail the report: the worst case is
        that three registered issues shout as `degraded` for one run, which
        is noisy and honest. The opposite failure — suppressing on a file
        nobody could parse — is the one that hides things.
        """

        target = Path(path) if path is not None else KNOWN_ISSUES_PATH
        register = cls(path=target)
        if not target.is_file():
            register.problems.append(f"no known-issues register at {target}")
            return register
        if yaml is None:  # pragma: no cover - PyYAML is installed everywhere
            register.problems.append("PyYAML is unavailable; nothing is suppressed")
            return register
        try:
            data = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
        except Exception as exc:  # noqa: BLE001
            register.problems.append(f"could not parse {target}: {exc}")
            return register
        raw = data.get("issues") if isinstance(data, Mapping) else None
        if not isinstance(raw, list):
            register.problems.append(f"{target} carries no `issues:` list")
            return register
        for item in raw:
            if not isinstance(item, Mapping):
                continue
            key = str(item.get("id") or "").strip()
            if not key:
                register.problems.append(f"{target} carries an entry with no id")
                continue
            register.entries[key] = dict(item)
        return register

    def __contains__(self, issue_id: str) -> bool:
        return str(issue_id) in self.entries

    def apply(self, issue: Issue, today: dt.date | None = None) -> Issue:
        """Demote a registered issue to `known` and attach its owner and note.

        `blocking` is deliberately NOT demotable. A run that could not write
        its output is not something a register entry can make acceptable.
        """

        entry = self.entries.get(issue.id)
        if entry is None:
            return issue
        if issue.severity == BLOCKING:
            issue.note = str(entry.get("note") or "")
            return issue
        issue.severity = KNOWN
        issue.owner = str(entry.get("owner") or issue.owner)
        issue.note = str(entry.get("note") or "")
        issue.review_by = str(entry.get("review_by") or "")
        if entry.get("first_seen"):
            issue.first_seen = str(entry["first_seen"])
        if entry.get("recovers_on_rerun") is not None:
            issue.recovers_on_rerun = bool(entry["recovers_on_rerun"])
        issue.overdue = self._is_overdue(issue.review_by, today)
        return issue

    @staticmethod
    def _is_overdue(review_by: str, today: dt.date | None = None) -> bool:
        if not review_by:
            # An entry with no review date is overdue by construction: a
            # suppression with no expiry is how a known issue becomes an
            # invisible one.
            return True
        try:
            due = dt.date.fromisoformat(str(review_by)[:10])
        except ValueError:
            return True
        return (today or _today()) > due


class IssueRegister:
    """Every issue this run found, deduplicated by id and ordered worst-first."""

    def __init__(self, known: KnownIssues | None = None) -> None:
        self.known = known if known is not None else KnownIssues.load()
        self._issues: dict[str, Issue] = {}
        self.notes: list[str] = list(self.known.problems)

    def add(self, issue: Issue) -> Issue:
        """Record one issue. A repeat id MERGES rather than appending.

        An issue is one fault, however many rows show it. Appending a second
        record for the same id is how a register turns back into a log.
        """

        issue = self.known.apply(issue)
        existing = self._issues.get(issue.id)
        if existing is None:
            self._issues[issue.id] = issue
            return issue
        if issue.rank < existing.rank:
            existing.severity = issue.severity
        self._merge_cost(existing, issue)
        if issue.evidence and issue.evidence not in existing.evidence:
            existing.evidence = f"{existing.evidence}; {issue.evidence}".strip("; ")
        return existing

    @staticmethod
    def _merge_cost(existing: Issue, incoming: Issue) -> None:
        """Add costs ONLY where the two are counting the same thing.

        One fault can be found by two collectors — EM-DAT's lockout shows up
        as six failed fetches AND as a failed contradiction check counting
        rows. Adding those gives 29,041 "failed fetches", which is a number
        in a unit belonging to a different measurement. The second cost is
        carried into the evidence instead, where it says what it is.
        """

        if incoming.cost is None:
            return
        if existing.cost is None:
            existing.cost, existing.cost_unit = incoming.cost, incoming.cost_unit
            return
        if (existing.cost_unit or "") == (incoming.cost_unit or ""):
            existing.cost += incoming.cost
            return
        note = f"also {incoming.cost_text()}"
        if note not in existing.evidence:
            existing.evidence = f"{existing.evidence}; {note}".strip("; ")

    def extend(self, issues: Iterable[Issue]) -> None:
        for issue in issues:
            self.add(issue)

    @property
    def issues(self) -> list[Issue]:
        return sorted(self._issues.values(), key=lambda i: (i.rank, i.id))

    def by_severity(self, severity: str) -> list[Issue]:
        return [i for i in self.issues if i.severity == severity]

    def counts(self) -> dict[str, int]:
        return {name: len(self.by_severity(name)) for name in SEVERITY_ORDER}

    def apply_history(self, history: Mapping[str, Mapping[str, Any]]) -> None:
        """Stamp `first_seen` / `runs_seen` from prior runs.

        A known-issues entry's own `first_seen` wins where it has one: it is
        an author's statement about when the fault started, which is better
        evidence than the first run that happened to record it.
        """

        for issue in self._issues.values():
            prior = history.get(issue.id)
            if not prior:
                continue
            if not issue.first_seen:
                issue.first_seen = str(prior.get("first_seen") or "")
            issue.runs_seen = int(prior.get("runs_seen") or 0) + 1

    def to_dict(self, *, run: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return {
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "run": dict(run or {}),
            "counts": self.counts(),
            "notes": self.notes,
            "issues": [issue.to_dict() for issue in self.issues],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "IssueRegister":
        """Rebuild a register from a written `issues.json`.

        The known-issues register is NOT re-applied: the severities in the
        file were decided when the run was assessed, and re-deciding them
        against a register that may have moved would make the printed report
        and the bundled one disagree.
        """

        register = cls(known=KnownIssues(entries={}))
        register.notes = [str(n) for n in (payload.get("notes") or [])]
        for raw in payload.get("issues") or []:
            if isinstance(raw, Mapping):
                issue = Issue.from_dict(raw)
                register._issues[issue.id] = issue
        return register


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

_BAR = "=" * 78


def render_text(register: IssueRegister, *, run_label: str = "") -> str:
    """The block printed last to stdout, framed so it survives scrolling."""

    counts = register.counts()
    lines = [
        "",
        _BAR,
        "  RUN ISSUE REGISTER" + (f" — {run_label}" if run_label else ""),
        "  "
        + ", ".join(f"{counts[name]} {name}" for name in SEVERITY_ORDER)
        + ".",
        _BAR,
    ]
    if not register.issues:
        lines += [
            "  Nothing to report: no check failed, no source was unread, and no",
            "  connector disagreed with its table.",
            _BAR,
            "",
        ]
        return "\n".join(lines)

    for severity in SEVERITY_ORDER:
        group = register.by_severity(severity)
        if not group:
            continue
        header = f"  -- {severity.upper()} ({len(group)}) "
        lines.append(header + "-" * max(4, len(_BAR) - len(header)))
        for issue in group:
            lines.append(f"  [{issue.id}] {_one_line(issue.title, 150)}")
            if severity == KNOWN:
                # One line, on purpose. A registered issue that reprints its
                # whole case every run is a registered issue nobody reads.
                # One LINE, not one paragraph. The full note is in
                # checks/issues.md; here it exists to remind the reader the
                # fault is owned, not to re-argue the case every run.
                tail = f"owner {issue.owner}"
                if issue.note:
                    tail += f"; {_one_line(issue.note, 140)}"
                if issue.review_by:
                    tail += f"; review by {issue.review_by}"
                lines.append(f"      cost {issue.cost_text()}; {tail}")
                if issue.overdue:
                    lines.append(
                        "      OVERDUE FOR REVIEW — a suppression with no expiry "
                        "is an invisible fault."
                    )
                continue
            if issue.evidence:
                lines.append(f"      {_one_line(issue.evidence, 160)}")
            lines.append(
                f"      cost {issue.cost_text()}; owner {issue.owner}; "
                + ("a re-run may clear it" if issue.recovers_on_rerun
                   else "a re-run will NOT clear it")
                + f"; {issue.age_text()}"
            )
        lines.append("")
    if register.notes:
        lines.append("  -- notes on the register itself " + "-" * 44)
        lines += [f"  {note}" for note in register.notes]
        lines.append("")
    lines += [
        "  Full evidence: checks/issues.md in the resolver debug bundle.",
        _BAR,
        "",
    ]
    return "\n".join(lines)


def render_markdown(register: IssueRegister, *, run_label: str = "") -> str:
    """The GitHub step summary and the bundle's `checks/issues.md`."""

    counts = register.counts()
    lines = [
        "## Run issue register" + (f" — {run_label}" if run_label else ""),
        "",
        "Every distinct fault this run found, worst first. A `known` row is an",
        "issue already registered in `resolver/config/known_issues.yml` with an",
        "owner and a review date; it is reported once and shouts at nobody.",
        "None of these change the run's exit code — only a run that could not",
        "write its output does.",
        "",
        "| severity | issue | cost | owner | re-run helps | age |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    if not register.issues:
        lines.append("| info | _nothing to report_ | — | — | — | — |")
    for issue in register.issues:
        lines.append(
            f"| {issue.severity} | **{issue.id}** — {_cell(issue.title)} | "
            f"{issue.cost_text()} | {issue.owner} | "
            f"{'yes' if issue.recovers_on_rerun else 'no'} | {issue.age_text()} |"
        )
    lines += ["", "_Counts: " + ", ".join(
        f"{counts[name]} {name}" for name in SEVERITY_ORDER) + "._", ""]

    for issue in register.issues:
        lines.append(f"### {issue.severity} — {issue.id}")
        lines.append("")
        lines.append(issue.title)
        lines.append("")
        if issue.evidence:
            lines.append(f"- evidence: {_cell(issue.evidence)}")
        lines.append(f"- cost: {issue.cost_text()}")
        lines.append(f"- owner: {issue.owner}")
        lines.append(
            "- a re-run clears it: "
            + ("yes" if issue.recovers_on_rerun else "no")
        )
        if issue.source:
            lines.append(f"- found by: {_cell(issue.source)}")
        if issue.note:
            lines.append(f"- note: {_cell(issue.note)}")
        if issue.review_by:
            lines.append(f"- review by: {issue.review_by}")
        if issue.overdue and issue.severity == KNOWN:
            lines.append(
                "- **this register entry is overdue for review.** A suppression "
                "that never expires is how a known issue becomes an invisible one."
            )
        lines.append("")
    if register.notes:
        lines.append("### notes on the register itself")
        lines.append("")
        lines += [f"- {_cell(note)}" for note in register.notes]
        lines.append("")
    return "\n".join(lines)


def render_annotations(register: IssueRegister) -> list[str]:
    """GitHub workflow commands, one per issue.

    `known` issues emit `::notice::`, which is what makes the register
    actually quieten the noise it exists to quieten.
    """

    out: list[str] = []
    for issue in register.issues:
        level = _ANNOTATION.get(issue.severity, "notice")
        title = f"{issue.severity}: {issue.id}"
        body = issue.title
        if issue.cost is not None:
            body += f" (cost: {issue.cost_text()})"
        if issue.severity == KNOWN and issue.note:
            body += f" — known: {issue.note}"
        if issue.overdue and issue.severity == KNOWN:
            # The annotations are where a reader looks first, and an
            # overdue entry is the one that most needs looking at: it is a
            # suppression nobody has re-read. Saying it in the stdout block
            # and not here leaves the quiet notice looking settled.
            title = f"{issue.severity} (overdue): {issue.id}"
            body += (
                f" — REGISTER ENTRY OVERDUE: due {issue.review_by}, "
                "chase it or move the date."
            )
        out.append(f"::{level} title={_command(title)}::{_command(body)}")
    return out


def _one_line(text: str, limit: int) -> str:
    """One line, cut on a word boundary — a wrapped note stops being one line."""

    flat = " ".join(str(text).split())
    if len(flat) <= limit:
        return flat
    return flat[: flat.rfind(" ", 0, limit - 1) or limit - 1].rstrip(" ,;.") + "…"


def _cell(text: str) -> str:
    """Markdown-table-safe: pipes break a row, newlines break the table."""

    return str(text).replace("|", "/").replace("\n", " ").strip()


def _command(text: str) -> str:
    """GitHub workflow commands are newline-delimited and take %0A escapes."""

    return (
        str(text)
        .replace("%", "%25")
        .replace("\r", "")
        .replace("\n", "%0A")
        .replace("::", ":")
    )


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

#: Where `first_seen` / `runs_seen` live. The canonical DB travels between
#: runs in the artifact, so it is the only durable store this pipeline has;
#: a JSON file on the runner dies with it and nothing may push to `main`.
HISTORY_TABLE = "diagnostic_issue_history"


def ensure_history_table(con: Any) -> None:
    con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {HISTORY_TABLE} (
            issue_id TEXT PRIMARY KEY,
            first_seen DATE,
            last_seen DATE,
            runs_seen INTEGER,
            last_severity TEXT,
            last_title TEXT
        )
        """
    )


def load_history(con: Any) -> dict[str, dict[str, Any]]:
    """What earlier runs recorded, keyed by issue id. Never raises."""

    if con is None:
        return {}
    try:
        ensure_history_table(con)
        rows = con.execute(
            f"SELECT issue_id, first_seen, runs_seen FROM {HISTORY_TABLE}"
        ).fetchall()
    except Exception:  # noqa: BLE001 - history is a nicety, never a blocker
        return {}
    out: dict[str, dict[str, Any]] = {}
    for issue_id, first_seen, runs_seen in rows:
        out[str(issue_id)] = {
            "first_seen": "" if first_seen is None else str(first_seen)[:10],
            "runs_seen": int(runs_seen or 0),
        }
    return out


def save_history(con: Any, register: IssueRegister, today: dt.date | None = None) -> int:
    """Record this run's issues so the next run can say how old each is.

    Returns the rows written. Never raises: a history write that fails must
    cost the counter and nothing else.
    """

    if con is None:
        return 0
    stamp = (today or _today()).isoformat()
    try:
        ensure_history_table(con)
    except Exception:  # noqa: BLE001
        return 0
    written = 0
    for issue in register.issues:
        try:
            con.execute(
                f"""
                INSERT INTO {HISTORY_TABLE}
                    (issue_id, first_seen, last_seen, runs_seen, last_severity, last_title)
                VALUES (?, CAST(? AS DATE), CAST(? AS DATE), 1, ?, ?)
                ON CONFLICT (issue_id) DO UPDATE SET
                    last_seen = CAST(excluded.last_seen AS DATE),
                    runs_seen = {HISTORY_TABLE}.runs_seen + 1,
                    last_severity = excluded.last_severity,
                    last_title = excluded.last_title
                """,
                [issue.id, issue.first_seen or stamp, stamp,
                 issue.severity, issue.title[:400]],
            )
            written += 1
        except Exception:  # noqa: BLE001
            continue
    return written


def write_outputs(
    register: IssueRegister,
    *,
    json_path: Path | None = None,
    markdown_path: Path | None = None,
    run: Mapping[str, Any] | None = None,
    run_label: str = "",
) -> None:
    """Write `issues.json` and `issues.md` side by side, creating parents."""

    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(register.to_dict(run=run), indent=2, sort_keys=False,
                       default=str) + "\n",
            encoding="utf-8",
        )
    if markdown_path is not None:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(
            render_markdown(register, run_label=run_label), encoding="utf-8"
        )


def read_register(path: str | os.PathLike[str]) -> IssueRegister | None:
    """Load a written `issues.json`, or None when it is absent or broken."""

    target = Path(path)
    if not target.is_file():
        return None
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(payload, Mapping):
        return None
    return IssueRegister.from_dict(payload)


__all__ = [
    "BLOCKING", "DEGRADED", "KNOWN", "INFO", "SEVERITY_ORDER",
    "OWNER_PYTHIA", "OWNER_EXTERNAL",
    "Issue", "KnownIssues", "IssueRegister",
    "render_text", "render_markdown", "render_annotations",
    "load_history", "save_history", "ensure_history_table",
    "write_outputs", "read_register", "KNOWN_ISSUES_PATH", "HISTORY_TABLE",
]
