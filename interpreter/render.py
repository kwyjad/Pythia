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

from interpreter import charts, lexicon, names, selection

_PLACEHOLDER = re.compile(r"\{\{fig:([A-Za-z0-9_.-]+)\}\}")

UNAVAILABLE = "[figure unavailable]"

# ln 2 — js_vs_baserate's maximum, used for the percent rendering.
_LN2 = math.log(2.0)



_FRACTION_WORDS = {
    2: "half", 3: "third", 4: "quarter", 5: "fifth", 6: "sixth",
    7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth",
}


def _ordinal_word(inverse: float) -> str:
    """"one third of" reads; "1/3.2x" does not."""
    nearest = max(2, min(int(round(inverse)), 10))
    return _FRACTION_WORDS[nearest]


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
        # A multiple a reader can check against the sentence around it.
        # "1/87.5x" meant nothing; "a fraction of the usual level" does.
        ratio = math.exp(v) if key == "log_ev_ratio" else v
        if ratio >= 1.0:
            return f"about {ratio:.1f} times the usual level"
        if ratio <= 0:
            return "far below the usual level"
        inverse = 1.0 / ratio
        if inverse >= 10:
            return "a small fraction of the usual level"
        return f"about one {_ordinal_word(inverse)} of the usual level"
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
    for entry in out.get("attention") or []:
        qids = [str(q) for q in entry.get("question_ids") or []]
        for name in ("why_it_stands_out", "how_to_read_the_distribution",
                     "what_the_model_was_reacting_to"):
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



def _render_entry(entry: dict[str, Any], resolver: FigureResolver) -> list[str]:
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
    if entry.get("why_it_stands_out"):
        lines.append(_r("why_it_stands_out"))
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
    # The distribution comes from the PACK, never from the model: the schema
    # forbids extra properties on an entry, and a chart drawn from model
    # output could disagree with the numbers printed beside it.
    spd, bucket_labels, is_binary = resolver.spd_for(qids)
    if is_binary:
        # A yes/no question has one number, not a distribution: bucket 1 is
        # P(yes). Six bars where five are empty would misrepresent it.
        chart = charts.probability_bar(spd[0] if spd else None)
    else:
        chart = charts.probability_chart(spd, bucket_labels)
    if chart:
        lines += ["", chart]
    if qids:
        lines += ["", "Questions: " + ", ".join(f"`{q}`" for q in qids)]
    lines.append("")
    return lines


def render_markdown(
    content: dict[str, Any],
    resolver: FigureResolver,
    *,
    provenance: dict[str, Any] | None = None,
) -> str:
    """content_json -> the report markdown, placeholders resolved."""
    lines: list[str] = []
    kind = str(content.get("kind") or "")

    lines.append("# Fred — this cycle, in plain language")
    lines.append("")
    lines.append(f"**{resolver.resolve_text(str(content.get('headline') or ''))}**")

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
                lines += _render_entry(entry, resolver)

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
                lines += _render_entry(entry, resolver)

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
                 "in this report were written by the language model — every "
                 "figure is substituted from the system's own computed data._")
    return "\n".join(lines)
