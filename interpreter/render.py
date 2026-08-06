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

from interpreter import lexicon

_PLACEHOLDER = re.compile(r"\{\{fig:([A-Za-z0-9_.-]+)\}\}")

UNAVAILABLE = "[figure unavailable]"

# ln 2 — js_vs_baserate's maximum, used for the percent rendering.
_LN2 = math.log(2.0)


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
        # Expressed as a share of the metric's maximum (ln 2) so the reader
        # gets "how far toward maximal disagreement", not nats.
        return f"{min(v / _LN2, 1.0) * 100:.0f}% of maximum"
    if key == "log_ev_ratio":
        ratio = math.exp(v)
        return f"{ratio:.1f}x" if ratio >= 1 else f"1/{1.0 / ratio:.1f}x"
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
    ) -> None:
        self.per_question = per_question or {}
        self.global_figures = global_figures or {}
        self.misses: list[str] = []

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


_REASON_LABELS = {
    "base_rate_deviation": "far from its base rate",
    "large_impact_nominal": "large expected impact",
    "large_impact_per_capita": "large expected impact per capita",
    "rc_deviation_disagreement": "the system disagrees with itself here",
}


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

    attention = content.get("attention") or []
    if attention:
        lines += ["", "## What to watch this month", ""]
        for entry in sorted(attention, key=lambda e: int(e.get("rank") or 99)):
            qids = list(entry.get("question_ids") or [])

            def _r(field: str) -> str:
                return resolver.resolve_text(str(entry.get(field) or ""), qids)

            reason = _REASON_LABELS.get(
                str(entry.get("reason_code") or ""), str(entry.get("reason_code") or "")
            )
            lines.append(
                f"### {entry.get('rank')}. {entry.get('iso3')} — "
                f"{entry.get('hazard_code')}/{entry.get('metric')} ({reason})"
            )
            lines.append("")
            if entry.get("why_it_stands_out"):
                lines.append(_r("why_it_stands_out"))
            if entry.get("how_to_read_the_distribution"):
                lines.append("")
                lines.append(f"*Reading the forecast:* {_r('how_to_read_the_distribution')}")
            if entry.get("what_the_model_was_reacting_to"):
                lines.append("")
                lines.append(f"*What the system was reacting to:* {_r('what_the_model_was_reacting_to')}")
            impacts = entry.get("impacts") or []
            if impacts:
                lines.append("")
                lines.append("*Likely impacts:*")
                for item in impacts:
                    lines.append(f"- {resolver.resolve_text(str(item), qids)}")
            challenges = entry.get("operational_challenges") or []
            if challenges:
                lines.append("")
                lines.append("*Operational challenges:*")
                for item in challenges:
                    lines.append(f"- {resolver.resolve_text(str(item), qids)}")
            if entry.get("lead_time_months") is not None:
                lines.append("")
                lines.append(f"*Lead time:* about {entry['lead_time_months']} month(s).")
            lines.append("")
            lines.append(
                "Questions: " + ", ".join(f"`{q}`" for q in qids)
            )
            lines.append("")

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
