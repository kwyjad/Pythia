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

from interpreter import charts, config, lexicon, names, panels, selection

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
                     "spd_shape", "what_the_model_was_reacting_to"):
            if entry.get(name):
                entry[name] = _r(entry[name], qids)
        for name in ("impacts", "operational_challenges"):
            if entry.get(name):
                entry[name] = [_r(item, qids) for item in entry[name]]
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



def _render_entry(
    entry: dict[str, Any],
    resolver: FigureResolver,
    gates: dict[str, str] | None = None,
) -> list[str]:
    """One attention entry: a named heading and at most half a page under it."""
    qids = [str(q) for q in entry.get("question_ids") or []]

    def _r(field: str) -> str:
        return resolver.resolve_text(str(entry.get(field) or ""), qids)

    lines: list[str] = []
    # Full names, never codes: "Nicaragua, drought: major alert".
    lines.append(
        f"### {entry.get('rank')}. "
        f"{names.describe_pair(entry.get('iso3'), entry.get('hazard_code'), entry.get('metric'))}"
    )
    lines.append("")
    # Which test admitted this entry, in the panel's own words. A reader who
    # asks "why is this here and not my country" gets the answer on the entry
    # rather than having to infer it from the section heading.
    gate = next((gates.get(q) for q in qids if (gates or {}).get(q)), None) if gates else None
    if gate:
        lines.append(f"*Selected because: {gate}.*")
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
    lines.append("")
    lines.append(f"**{resolver.resolve_text(str(content.get('headline') or ''))}**")

    lines += _selection_panel_lines(extras.get("selection_panel"))

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
                lines += _render_entry(entry, resolver, gates)

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
                lines += _render_entry(entry, resolver, gates)

    changes = content.get("changes_since_last_run") or []
    if changes:
        lines += ["", "## What changed since last month", ""]
        for item in changes:
            lines.append(f"- {resolver.resolve_text(str(item))}")

    performance = content.get("performance")
    if performance:
        lines += ["", "## How well did we do", ""]
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
                        f"({', '.join(f'`{q}`' for q in qids)})"
                    )
        for field, title in (
            ("track_comparison", "Track 1 vs Track 2"),
            ("sibyl_comparison", "Deep research (Sibyl) vs the standard track"),
            ("vs_system_average", "This run vs the system average"),
        ):
            if performance.get(field):
                lines += ["", f"### {title}", ""]
                lines.append(resolver.resolve_text(str(performance[field])))

    blind = content.get("blind_spots") or []
    if blind:
        lines += ["", "## What we cannot see", ""]
        for item in blind:
            lines.append(f"- {resolver.resolve_text(str(item))}")

    notes = content.get("confidence_notes") or []
    if notes:
        lines += ["", "## Confidence notes", ""]
        for item in notes:
            lines.append(f"- {resolver.resolve_text(str(item))}")

    lines += ["", "## Appendix", "", "### Probability words", ""]
    lines.append(
        "The report uses these words with fixed meanings, so the reader can "
        "check the writer:"
    )
    lines.append("")
    lines.append(lexicon.markdown_table())
    lines += _question_table_lines(extras.get("question_table"))
    lines += _bucket_table_lines(list(attention), resolver)
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
