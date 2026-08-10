# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Render the validated structured output to markdown.

The renderer substitutes every ``{{fig:<key>}}`` placeholder from the figure
maps built out of the pack — the model's own idea of a number is never
printed. An unresolvable placeholder renders as ``[figure unavailable]`` and
is counted (the Phase 4 validator turns that count into a failed check).
"""

from __future__ import annotations

import math
import re
from typing import Any

from interpreter import (
    charts,
    config,
    decisions,
    lexicon,
    names,
    panels,
    selection,
)

_PLACEHOLDER = re.compile(r"\{\{fig:([A-Za-z0-9_.-]+)\}\}")

UNAVAILABLE = "[figure unavailable]"

# ln 2 — js_vs_baserate's maximum, used for the percent rendering.
_LN2 = math.log(2.0)


def question_link(question_id: str) -> str:
    """A question id as a markdown link to its page on the dashboard.

    The PDF is read away from the dashboard, so a bare id there is a dead end.
    The link is absolute, built from ``PYTHIA_PUBLIC_BASE_URL``; the dashboard
    resolves the same markdown itself and can follow a relative path, but one
    form serving both is worth more than the shorter href.
    """
    qid = str(question_id)
    base = (config.public_base_url() or "").rstrip("/")
    if not base:
        return f"`{qid}`"
    return f"[{qid}]({base}/questions/{qid})"



_FRACTION_WORDS = {
    2: "half", 3: "third", 4: "quarter", 5: "fifth", 6: "sixth",
    7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth",
}


def _ordinal_word(inverse: float) -> str:
    """"one third of" reads; "1/3.2x" does not."""
    nearest = max(2, min(int(round(inverse)), 10))
    return _FRACTION_WORDS[nearest]


def _round_planning(value: float) -> float:
    """Round a planning figure to its own scale.

    150,000 rather than 149,997: the figure comes from interpolating inside a
    bucket whose width is tens of thousands, and printing every digit would
    claim a precision the bucket scheme does not have.
    """
    if value <= 0:
        return 0.0
    import math as _m

    magnitude = 10 ** max(0, int(_m.floor(_m.log10(value))) - 1)
    return round(value / magnitude) * magnitude


def format_figure(key: str, value: Any) -> str:
    """Human formatting per figure key. Deterministic; no rounding drama."""
    if value is None or value == "":
        return UNAVAILABLE
    try:
        if key in ("baserate_source", "modal_bucket_label"):
            return str(value)
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if key in ("rc_level",):
        return f"L{int(v)}"
    if key in ("attention_rank",):
        return f"#{v:.0f}" if v == int(v) else f"#{v:.1f}"
    if key.startswith("p_") or key in ("rc_score", "triage_score"):
        return f"{v * 100:.0f}%"
    if key.startswith("skill_"):
        return f"{v * 100:+.0f}%"
    if key == "js_vs_baserate":
        # "25% of maximum from its anchor" told readers nothing. Say how far
        # the forecast has moved from the usual pattern, in words, with the
        # share kept for anyone who wants it.
        share = min(v / _LN2, 1.0)
        if share >= 0.5:
            word = "a long way from its usual pattern"
        elif share >= 0.25:
            word = "well away from its usual pattern"
        elif share >= 0.1:
            word = "a little away from its usual pattern"
        else:
            word = "close to its usual pattern"
        return word
    if key in ("log_ev_ratio", "ev_multiple"):
        # A multiple a reader can check against the sentence around it;
        # "1/87.5x" meant nothing. Rendered as a NOUN PHRASE, not a clause,
        # because the model writes it inside a sentence of its own ("about
        # {{fig:ev_multiple}} the usual number of people affected"). Returning
        # "about 14.1 times the usual level" there produced "about about 14.1
        # times the usual level the usual number of people affected" in a
        # published report.
        ratio = math.exp(v) if key == "log_ev_ratio" else v
        if ratio >= 1.0:
            return f"{ratio:.1f} times"
        if ratio <= 0:
            return "a tiny fraction of"
        inverse = 1.0 / ratio
        if inverse >= 10:
            return "a small fraction of"
        return f"one {_ordinal_word(inverse)} of"
    if key == "p_zero_peak":
        return f"{v * 100:.0f}%"
    if key in ("p50_peak", "p90_peak"):
        # A planning figure. Rounded to something a planner would actually
        # write on a form, never to the false precision of an interpolation.
        return f"{_round_planning(v):,.0f} people" if v >= 1 else "almost nobody"
    if key == "excess_nominal":
        return f"{v:+,.0f} people"
    if key == "excess_per_100k":
        return f"{v:+,.0f} per 100,000"
    if key == "baserate_n_obs":
        return f"{int(v):,}"
    if key.startswith("eiv") or key.startswith("n_") or abs(v) >= 1000:
        return f"{v:,.0f}"
    if key.startswith("mean_brier") or key.startswith("climatology_brier"):
        return f"{v:.2f}"
    return f"{v:,.2f}" if v != int(v) else f"{int(v):,}"


class FigureResolver:
    """Resolves placeholder keys, tracking misses.

    ``per_question``: {question_id: {key: value}}; ``global_figures`` covers
    the pack-level performance keys. An entry resolves against its FIRST
    question_id's map, falling back to the global map.
    """

    def __init__(
        self,
        per_question: dict[str, dict[str, Any]] | None = None,
        global_figures: dict[str, Any] | None = None,
        spd_by_question: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.per_question = per_question or {}
        self.global_figures = global_figures or {}
        # {question_id: {"spd": [...], "bucket_labels": {...}}} for the charts.
        self.spd_by_question = spd_by_question or {}
        self.misses: list[str] = []

    def spd_for(self, question_ids: list[str] | None) -> tuple[Any, Any, bool]:
        """(spd, bucket_labels, is_binary) for the first cited question that
        has a distribution, else (None, None, False)."""
        for qid in question_ids or []:
            entry = self.spd_by_question.get(qid)
            if entry and entry.get("spd"):
                return (entry.get("spd"), entry.get("bucket_labels"),
                        bool(entry.get("binary")))
        return None, None, False

    def resolve_text(self, text: str, question_ids: list[str] | None = None) -> str:
        def _sub(match: re.Match[str]) -> str:
            key = match.group(1)
            for qid in question_ids or []:
                figs = self.per_question.get(qid)
                if figs and key in figs:
                    return format_figure(key, figs[key])
            if key in self.global_figures:
                return format_figure(key, self.global_figures[key])
            self.misses.append(key)
            return UNAVAILABLE

        return _PLACEHOLDER.sub(_sub, text or "")


def resolve_content(content: dict[str, Any], resolver: FigureResolver) -> dict[str, Any]:
    """A deep copy of content_json with every prose placeholder resolved.

    The API serves this so the dashboard renders structured content without
    re-implementing figure formatting in TypeScript — one formatting
    implementation, no drift. Traversal mirrors render_markdown's: attention
    entries and best/worst calls resolve in their question_ids context,
    everything else globally.
    """
    import copy

    out = copy.deepcopy(content)

    def _r(text: Any, qids: list[str] | None = None) -> Any:
        if isinstance(text, str):
            return resolver.resolve_text(text, qids)
        return text

    if out.get("headline"):
        out["headline"] = _r(out["headline"])
    if out.get("run_summary"):
        out["run_summary"] = _r(out["run_summary"])
    for entry in out.get("attention") or []:
        qids = [str(q) for q in entry.get("question_ids") or []]
        for name in ("why_it_stands_out", "how_to_read_the_distribution",
                     "planning_sentence", "spd_shape",
                     "what_the_model_was_reacting_to"):
            if entry.get(name):
                entry[name] = _r(entry[name], qids)
        for name in ("impacts", "operational_challenges"):
            if entry.get(name):
                entry[name] = [_r(item, qids) for item in entry[name]]
        decision = entry.get("decision_point")
        if isinstance(decision, dict) and decision.get("action"):
            decision["action"] = _r(decision["action"], qids)
    performance = out.get("performance")
    if isinstance(performance, dict):
        for name in ("plain_summary", "skill_statement", "track_comparison",
                     "sibyl_comparison", "vs_system_average"):
            if performance.get(name):
                performance[name] = _r(performance[name])
        for side in ("best_calls", "worst_calls"):
            for call in performance.get(side) or []:
                qids = [str(q) for q in call.get("question_ids") or []]
                for name in ("what_was_right", "what_went_wrong"):
                    if call.get(name):
                        call[name] = _r(call[name], qids)
    for name in ("changes_since_last_run", "blind_spots", "confidence_notes"):
        if out.get(name):
            out[name] = [_r(item) for item in out[name]]
    return out


_REASON_LABELS = {
    "base_rate_deviation": "far from its base rate",
    "large_impact_nominal": "large expected impact",
    "large_impact_per_capita": "large expected impact per capita",
    "rc_deviation_disagreement": "the system disagrees with itself here",
}



def _first_for(qids: list[str], table: dict[str, Any] | None) -> Any:
    """The first value in ``table`` any of an entry's questions carries."""
    for qid in qids:
        value = (table or {}).get(qid)
        if value:
            return value
    return None


def _render_entry(
    entry: dict[str, Any],
    resolver: FigureResolver,
    gates: dict[str, str] | None = None,
    extras: dict[str, Any] | None = None,
) -> list[str]:
    """One attention entry: a named heading and at most half a page under it."""
    qids = [str(q) for q in entry.get("question_ids") or []]
    extras = extras or {}

    def _r(field: str) -> str:
        return resolver.resolve_text(str(entry.get(field) or ""), qids)

    lines: list[str] = []
    # Full names, never codes: "Nicaragua, drought: major alert".
    lines.append(
        f"### {entry.get('rank')}. "
        f"{names.describe_pair(entry.get('iso3'), entry.get('hazard_code'), entry.get('metric'))}"
    )
    lines.append("")
    # One line of provenance for the entry: which test admitted it, how long
    # it has been flagged, and what the second reader made of it. All three
    # come from the pack's own stamps, never from the model, so a reader
    # asking "why is this here and not my country" gets a checkable answer.
    tags: list[str] = []
    gate = next((gates.get(q) for q in qids if (gates or {}).get(q)), None) if gates else None
    if gate:
        tags.append(f"selected because it is {gate}")
    persistence = _first_for(
        qids,
        {
            str(e.get("question_id")): e.get("persistence")
            for e in (extras.get("persistence") or {}).get("entries") or []
        },
    )
    if persistence:
        tags.append(str(persistence))
    sibyl_tag = _first_for(qids, extras.get("sibyl_tags"))
    if sibyl_tag:
        tags.append(str(sibyl_tag))
    if tags:
        lines.append(f"*{'; '.join(tags)}.*")
        lines.append("")
    if entry.get("why_it_stands_out"):
        lines.append(_r("why_it_stands_out"))
    if entry.get("planning_sentence"):
        lines.append("")
        lines.append(f"**What to plan against:** {_r('planning_sentence')}")
    if entry.get("spd_shape"):
        lines.append("")
        lines.append(f"*The shape of the forecast:* {_r('spd_shape')}")
    if entry.get("how_to_read_the_distribution"):
        lines.append("")
        lines.append(f"*Reading the forecast:* {_r('how_to_read_the_distribution')}")
    if entry.get("what_the_model_was_reacting_to"):
        lines.append("")
        lines.append(f"*What the system was reacting to:* {_r('what_the_model_was_reacting_to')}")
    for field, label in (("impacts", "Likely impacts"),
                         ("operational_challenges", "Operational challenges")):
        items = entry.get(field) or []
        if items:
            lines.append("")
            lines.append(f"*{label}:*")
            for item in items:
                lines.append(f"- {resolver.resolve_text(str(item), qids)}")
    decision = entry.get("decision_point") or {}
    if decision.get("action"):
        deadline = _first_for(
            qids,
            {
                str(r.get("question_id")): r.get("deadline_label")
                for r in (extras.get("decision_calendar") or {}).get("rows") or []
            },
        ) or decisions.month_label(decision.get("deadline_month")) or "no dated deadline"
        lines.append("")
        lines.append(
            f"**Decision by {deadline}:** "
            f"{resolver.resolve_text(str(decision['action']), qids)}"
        )
    # The bucket chart moved to the appendix (v3): a picture of uncertainty
    # is not something a response planner can act on, and it was crowding out
    # the two figures that are. The reader who wants the distribution finds
    # the full table at the back.
    if qids:
        lines += ["", "Questions: " + ", ".join(question_link(q) for q in qids)]
    lines.append("")
    return lines


def _selection_panel_lines(panel: dict[str, Any] | None) -> list[str]:
    """The boxed "How these entries were chosen" panel.

    Rendered from `panels.selection_panel`, never from model prose: a report
    that describes its own selection rules in the model's words can describe
    them wrongly, and the reader has no way to check.
    """
    if not panel:
        return []
    lines = ["", f"## {panel.get('title')}", ""]
    if panel.get("ordering"):
        lines += [str(panel["ordering"]), ""]
    tests = panel.get("tests") or []
    if tests:
        lines.append("An entry has to pass two tests:")
        lines.append("")
        for test in tests:
            lines.append(f"- {test}")
        lines.append("")
    thresholds = panel.get("thresholds") or []
    if thresholds:
        lines.append("Sizes worth mobilising for:")
        lines.append("")
        for item in thresholds:
            lines.append(f"- {item}")
        lines.append("")
    if panel.get("counts_sentence"):
        lines += [str(panel["counts_sentence"]), ""]
    if panel.get("thin_note"):
        lines += [str(panel["thin_note"]), ""]
    return lines


def _question_table_lines(rows: list[dict[str, Any]] | None) -> list[str]:
    """Every question considered, one row each: the report's audit trail."""
    if not rows:
        return []
    lines = [
        "",
        "### Every question considered",
        "",
        "This is the whole list the report was drawn from, so a reader can "
        "see what was weighed and left out as well as what was chosen.",
        "",
        "| Country | Hazard | Measure | Expected excess | Chance of passing the threshold | Selected as | Record |",
        "| --- | --- | --- | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('country', '')} | {row.get('hazard', '')} | "
            f"{row.get('metric', '')} | {row.get('excess', '')} | "
            f"{row.get('chance', '')} | {row.get('gate', '')} | "
            f"{row.get('anchor', '')} |"
        )
    lines.append("")
    return lines


def _bucket_table_lines(
    attention: list[dict[str, Any]], resolver: FigureResolver
) -> list[str]:
    """The full bucket distributions, in the appendix where they belong.

    They used to sit inside every entry, where they crowded out the two
    figures a planner can actually act on. A reader who wants the shape of
    the whole distribution still gets it, with the bands named in words.
    """
    blocks: list[str] = []
    for entry in attention:
        qids = [str(q) for q in entry.get("question_ids") or []]
        spd, _labels, is_binary = resolver.spd_for(qids)
        if not spd or is_binary:
            continue
        metric = str(entry.get("metric") or "")
        labels = panels.humanised_labels(metric)
        rows = []
        for i, prob in enumerate(spd):
            label = labels[i] if i < len(labels) else f"band {i + 1}"
            rows.append(f"| {label} | {float(prob) * 100:.0f}% |")
        if not rows:
            continue
        blocks += [
            "",
            f"**{names.describe_pair(entry.get('iso3'), entry.get('hazard_code'), metric)}**",
            "",
            "| How many | Chance |",
            "| --- | ---: |",
        ] + rows
    if not blocks:
        return []
    return [
        "",
        "### The full forecasts",
        "",
        "Each forecast is a set of chances across bands of size, averaged "
        "over the six months of the window.",
    ] + blocks + [""]


def _issue_line(extras: dict[str, Any]) -> list[str]:
    """Forecasts issued on X, covering Y to Z. One line, under the title."""
    line = extras.get("issue_line") or {}
    issued, covering = line.get("issued"), line.get("covering")
    if not issued and not covering:
        return []
    parts = []
    if issued:
        parts.append(f"Forecasts issued {issued}")
    if covering:
        parts.append(f"covering {covering}")
    return ["", f"*{', '.join(parts)}. Automated output, unreviewed.*", ""]


def _blind_spots_front(extras: dict[str, Any]) -> list[str]:
    """Three lines of what the system cannot see, on page one.

    This used to sit on page fourteen. That people-affected ground truth is
    thin, and that the machine which will fix it runs in shadow mode, changes
    how a reader should read every climate entry in the report. The full list
    stays in the appendix.
    """
    items = extras.get("blind_spots_short") or []
    if not items:
        return []
    lines = ["", "## What this report cannot see", ""]
    for item in items[:3]:
        lines.append(f"- {item}")
    lines.append("")
    return lines


def _decision_calendar_lines(
    content: dict[str, Any], extras: dict[str, Any], resolver: FigureResolver
) -> list[str]:
    """The dated page: what decision, for where, by when.

    Rows and dates come from the pack; the action is the model's sentence for
    that entry. A deadline is derived from the peak month less the hazard's
    lead time, so it cannot be a plausible date the system does not believe.
    """
    calendar = extras.get("decision_calendar") or {}
    rows = calendar.get("rows") or []
    if not rows:
        return []
    actions: dict[str, str] = {}
    for entry in content.get("attention") or []:
        action = (entry.get("decision_point") or {}).get("action")
        if not action:
            continue
        qids = [str(q) for q in entry.get("question_ids") or []]
        for qid in qids:
            actions[qid] = resolver.resolve_text(str(action), qids)

    lines = [
        "", "## The decision calendar", "",
        "What has to be decided, for where, and by when. Each deadline is the "
        "month with most expected impact, less the run-up that hazard needs. "
        "Where that run-up has already begun, the row says so instead of "
        "naming a month that has passed.",
        "",
        "| By | Country | Hazard | Decision |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        action = actions.get(str(row.get("question_id"))) or str(row.get("action") or "")
        lines.append(
            f"| {row.get('deadline_label', 'not dated')} | {row.get('country', '')} | "
            f"{row.get('hazard', '')} | {action or 'no decision was attached'} |"
        )
    lines.append("")
    return lines


def _sibyl_lines(extras: dict[str, Any]) -> list[str]:
    """The second reader's own section.

    Everything here is drawn from the pack, including the caveat: Sibyl has
    no scored track record, and a model asked to summarise a comparison it
    cannot check would eventually promote it into a tiebreaker.
    """
    block = extras.get("sibyl") or {}
    if not block.get("available"):
        return []
    rows = block.get("rows") or []
    if not rows:
        return []
    summary = block.get("summary") or {}
    lines = [
        "", "## A second reader", "",
        "Sibyl re-forecasts the most volatile questions by reading the open "
        "web with an agent, while the main system reads structured data "
        "feeds. Where the two disagree, something is worth a second look.",
        "",
        f"It covered {summary.get('n_covered', 0)} questions this month. It "
        f"was more alarmed than the main system on "
        f"{summary.get('n_more_alarmed', 0)}, more cautious on "
        f"{summary.get('n_more_cautious', 0)}, and broadly agreed on "
        f"{summary.get('n_agrees', 0)}.",
        "",
    ]
    chart = charts.second_opinion_chart(rows, title="Where the two readers differ")
    if chart:
        lines += [chart, "", f"*{charts.second_opinion_legend()}*", ""]

    lines += [
        "| Country | Hazard | Second opinion | Its own trials |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        settled = "did not settle it" if row.get("unsettled") else "agreed with each other"
        lines.append(
            f"| {row.get('country_name', '')} | {row.get('hazard_name', '')} | "
            f"{row.get('tag', '')} | {settled} |"
        )
    lines.append("")

    novel = [r for r in rows if r.get("novel_evidence")]
    if novel:
        lines += [
            "### What the second reader found that the main system did not",
            "",
            "These come from Sibyl's own research notes. Each one is absent "
            "from the evidence the main system was given, which is a reason "
            "to check it rather than a finding in itself.",
            "",
        ]
        for row in novel:
            lines.append(
                f"**{row.get('country_name', '')}, "
                f"{str(row.get('hazard_name') or '').lower()}**"
            )
            for fact in row.get("novel_evidence") or []:
                lines.append(f"- {fact}")
            lines.append("")

    if block.get("caveat"):
        lines += [f"*{block['caveat']}*", ""]
    return lines


def _sector_lines(extras: dict[str, Any]) -> list[str]:
    """Where Fred departs from the prevailing institutional view."""
    block = extras.get("sector") or {}
    if not block.get("available"):
        return []
    lines = ["", "## Fred against the sector", ""]
    for comparison in block.get("comparisons") or []:
        label = comparison.get("source_label") or comparison.get("source")
        lines.append(
            f"### Compared with {label} "
            f"({comparison.get('n_compared', 0)} countries in both)"
        )
        lines.append("")
        for side, heading in (
            ("more_worried", "Fred ranks these higher than the sector does"),
            ("less_worried", "Fred ranks these lower than the sector does"),
        ):
            entries = comparison.get(side) or []
            if not entries:
                continue
            lines += [f"*{heading}:*", ""]
            for entry in entries:
                gap = int(entry.get("rank_gap") or 0)
                lines.append(
                    f"- {entry.get('country_name') or entry.get('iso3')}: "
                    f"Fred places it {entry.get('fred_rank')} of "
                    f"{comparison.get('n_compared')}, {label} places it "
                    f"{entry.get('sector_rank')}, a gap of {abs(gap)} places."
                )
            lines.append("")
    if block.get("caveat"):
        lines += [f"*{block['caveat']}*", ""]
    return lines


def _changes_lines(
    content: dict[str, Any], extras: dict[str, Any], resolver: FigureResolver
) -> list[str]:
    """What changed since last month, with persistence and movement.

    The model's items describe what entered and left. The persistence table
    beneath them is generated, because how long something has been flagged
    and which way its planning figure has moved are facts about two runs, not
    a judgement.
    """
    items = content.get("changes_since_last_run") or []
    persistence = extras.get("persistence") or {}
    entries = [e for e in (persistence.get("entries") or []) if e.get("movement")]
    dropped = persistence.get("dropped") or []
    if not items and not entries and not dropped:
        return []

    lines = ["", "## What changed since last month", ""]
    for item in items:
        lines.append(f"- {resolver.resolve_text(str(item))}")
    if entries:
        lines += [
            "",
            "How this month's entries have moved, measured only on the months "
            "the two runs share:",
            "",
            "| Country | Hazard | How long flagged | Planning figure |",
            "| --- | --- | --- | --- |",
        ]
        for entry in entries:
            move = entry.get("movement") or {}
            lines.append(
                f"| {entry.get('country_name', '')} | "
                f"{entry.get('hazard_name', '')} | "
                f"{entry.get('persistence', '')} | "
                f"{move.get('movement', 'unchanged')} over "
                f"{move.get('n_overlapping_months', 0)} shared months |"
            )
        lines.append("")
    if dropped:
        lines += ["", "Flagged last month and not shown this month:", ""]
        for row in dropped:
            lines.append(
                f"- {row.get('country_name', '')}, "
                f"{str(row.get('hazard_name') or '').lower()}: "
                f"{row.get('reason', '')}."
            )
        lines.append("")
    return lines


def _performance_lines(
    content: dict[str, Any], extras: dict[str, Any], resolver: FigureResolver
) -> list[str]:
    """Part B, including its dormant state.

    A dormant Part B that says "there is nothing to score" and stops tells a
    reader nothing. This one says how many question-months are due, on what
    date, and which hazards they cover, so a reader finishes it knowing when
    the system will first be testable.
    """
    outlook = extras.get("performance_outlook") or {}
    performance = content.get("performance") or {}
    dormant = outlook.get("dormant_sentences") or []
    if not performance and not dormant:
        return []

    lines = ["", "## How well did we do", ""]
    if dormant:
        for sentence in dormant:
            lines.append(sentence)
            lines.append("")
        schedule = (outlook.get("upcoming_resolutions") or {}).get("schedule") or []
        if schedule:
            lines += [
                "| Outcomes land | For the month of | Question-months |",
                "| --- | --- | ---: |",
            ]
            for slot in schedule[:6]:
                lines.append(
                    f"| {slot.get('resolution_date', '')} | "
                    f"{slot.get('window_month_label', '')} | "
                    f"{slot.get('n_question_horizons', 0):,} |"
                )
            lines.append("")
    else:
        if performance.get("plain_summary"):
            lines.append(resolver.resolve_text(str(performance["plain_summary"])))
        if performance.get("skill_statement"):
            lines.append("")
            lines.append(resolver.resolve_text(str(performance["skill_statement"])))
        for side, title in (("best_calls", "Best calls"), ("worst_calls", "Worst calls")):
            calls = performance.get(side) or []
            if calls:
                lines += ["", f"### {title}", ""]
                for call in calls:
                    qids = list(call.get("question_ids") or [])
                    text = call.get("what_was_right") or call.get("what_went_wrong") or ""
                    lines.append(
                        f"- {resolver.resolve_text(str(text), qids)} "
                        f"({', '.join(question_link(q) for q in qids)})"
                    )
        for field, title in (
            ("track_comparison", "Track 1 against Track 2"),
            ("sibyl_comparison", "The main system against the second reader"),
            ("vs_system_average", "This run against the system average"),
        ):
            if performance.get(field):
                lines += ["", f"### {title}", ""]
                lines.append(resolver.resolve_text(str(performance[field])))

    diary = outlook.get("diary") or []
    if diary:
        lines += [
            "", "### The forecast diary", "",
            "Calls this report made in an earlier month, and what happened.",
            "",
        ]
        for row in diary:
            observed = ", ".join(
                f"{o.get('observed_month')}: {o.get('value')}"
                for o in row.get("observed") or []
            )
            lines.append(
                f"- {names.describe_pair(row.get('iso3'), row.get('hazard_code'), row.get('metric'))}, "
                f"flagged in the {row.get('report_month')} report. Recorded: {observed}."
            )
        lines.append("")
    return lines


def _score_explanation_lines(extras: dict[str, Any]) -> list[str]:
    """The plain-English score glossary, in the appendix.

    Single-sourced with the dashboard (web/src/lib/score_glossary.ts) so the
    two never explain the same score two different ways.
    """
    items = (extras.get("performance_outlook") or {}).get("score_explanations") or []
    if not items:
        return []
    lines = ["", "### What the scores mean", ""]
    for item in items:
        lines.append(f"- **{item.get('term')}**: {item.get('text')}")
    lines.append("")
    return lines


def render_markdown(
    content: dict[str, Any],
    resolver: FigureResolver,
    *,
    provenance: dict[str, Any] | None = None,
    extras: dict[str, Any] | None = None,
) -> str:
    """content_json -> the report markdown, placeholders resolved.

    ``extras`` is `packs.report_extras(pack)`: the selection panel, the gate
    tags and the appendix question table. All three are GENERATED from the
    gate's own counts, so a reader can check the report's account of itself
    against the report.
    """
    lines: list[str] = []
    kind = str(content.get("kind") or "")
    extras = extras or {}
    gates = extras.get("gates") or {}

    lines.append("# Fred's Monthly Risk Report")
    lines += _issue_line(extras)
    lines.append(f"**{resolver.resolve_text(str(content.get('headline') or ''))}**")

    # Page one, in the order a reader needs it: what happened, what we cannot
    # see, and how these entries were chosen. The blind spots come BEFORE the
    # entries because they change how every entry should be read.
    lines += _blind_spots_front(extras)
    lines += _selection_panel_lines(extras.get("selection_panel"))
    lines += _decision_calendar_lines(content, extras, resolver)

    if content.get("run_summary"):
        lines += ["", "## The scan this month", ""]
        lines.append(resolver.resolve_text(str(content["run_summary"])))

    attention = content.get("attention") or []
    if attention:
        # Grouped into the report's four boxes rather than one flat list, so
        # a reader can tell "this is getting worse" from "this is already
        # bad" without reading every entry.
        for category, family in selection.SECTION_ORDER:
            group = [
                e for e in attention
                if str(e.get("category")) == category
                and str(e.get("hazard_family")) == family
            ]
            if not group:
                continue
            lines += [
                "",
                f"## {selection.CATEGORY_LABELS[category]}: "
                f"{names.FAMILY_LABELS.get(family, family).lower()}",
                "",
            ]
            for entry in sorted(group, key=lambda e: int(e.get("rank") or 99)):
                lines += _render_entry(entry, resolver, gates, extras)

        # An entry the model failed to place must never disappear from the
        # report. Print it under its own heading instead, where it is
        # visible and obviously odd.
        placed = {
            id(e) for cat, fam in selection.SECTION_ORDER
            for e in attention
            if str(e.get("category")) == cat and str(e.get("hazard_family")) == fam
        }
        stragglers = [e for e in attention if id(e) not in placed]
        if stragglers:
            lines += ["", "## Other situations of note", ""]
            for entry in sorted(stragglers, key=lambda e: int(e.get("rank") or 99)):
                lines += _render_entry(entry, resolver, gates, extras)
    elif kind in ("current", "combined"):
        # A month in which nothing cleared the tests is a finding about the
        # month, and it has to read as one. An empty stretch of page where
        # the entries should be is indistinguishable from a broken report.
        lines += [
            "",
            "## Nothing cleared both tests this month",
            "",
            "No forecast this month was both unusual against its own history "
            "and large enough to mobilise against. That is a statement about "
            "the month, not a gap in the system. The full list of what was "
            "considered, with the figures behind each one, is in the appendix.",
            "",
        ]

    lines += _sibyl_lines(extras)
    lines += _changes_lines(content, extras, resolver)
    lines += _sector_lines(extras)
    lines += _performance_lines(content, extras, resolver)

    lines += ["", "## Appendix", "", "### Probability words", ""]
    lines.append(
        "The report uses these words with fixed meanings, so the reader can "
        "check the writer:"
    )
    lines.append("")
    lines.append(lexicon.markdown_table())
    lines += _score_explanation_lines(extras)
    lines += _question_table_lines(extras.get("question_table"))
    lines += _bucket_table_lines(list(attention), resolver)

    # The full blind-spot list and the confidence notes live at the back; the
    # three lines that change how the whole report reads are on page one.
    blind = content.get("blind_spots") or []
    if blind:
        lines += ["", "### What we cannot see, in full", ""]
        for item in blind:
            lines.append(f"- {resolver.resolve_text(str(item))}")
    notes = content.get("confidence_notes") or []
    if notes:
        lines += ["", "### Confidence notes", ""]
        for item in notes:
            lines.append(f"- {resolver.resolve_text(str(item))}")

    watchlist = extras.get("watchlist") or []
    if watchlist:
        lines += [
            "",
            "### Watchlist",
            "",
            "These forecasts are unusual against their own history but too "
            "small to mobilise against. They are tracked, not acted on.",
            "",
        ]
        for item in watchlist:
            lines.append(
                f"- {item.get('country')}, {item.get('hazard')}: "
                f"{item.get('metric')}"
            )
        lines.append("")
    if provenance:
        lines += ["", "### Run provenance", ""]
        for key, value in provenance.items():
            if value not in (None, ""):
                lines.append(f"- {key}: `{value}`")
    lines.append("")
    if resolver.misses:
        lines.append(
            f"_({len(resolver.misses)} figure reference(s) could not be "
            "resolved against the pack and render as "
            f"'{UNAVAILABLE}' above.)_"
        )
        lines.append("")
    lines.append(f"_Report kind: {kind}. Generated automatically; no numbers "
                 "in this report were written by the language model; every "
                 "figure is substituted from the system's own computed data._")
    return "\n".join(lines)
