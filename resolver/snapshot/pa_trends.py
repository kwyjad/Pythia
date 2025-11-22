from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


@dataclass
class PaTrendPoint:
    """Represents people-affected data for a given month."""

    ym: str  # YYYY-MM string
    pa_value: float  # people affected (or IDPs) for this month


# Minimal PA classification: patterns we consider as PA in facts_snapshot
_PA_PATTERNS: Sequence[Tuple[str, str]] = (
    ("emdat", "affected"),
    ("IOM DTM", "idp_displacement_stock_dtm"),
    ("IDMC", "new_displacements"),
)


def _pa_where_clause() -> str:
    """Build a WHERE clause to select PA metrics from facts_snapshot."""

    clauses = []
    for src, metric in _PA_PATTERNS:
        clauses.append(
            "(source = '{src}' AND metric = '{metric}')".format(src=src, metric=metric)
        )
    return " OR ".join(clauses)


def get_pa_trend(
    con,
    *,
    iso3: str,
    hazard_code: str,
    months: int = 36,
) -> List[PaTrendPoint]:
    """
    Return up to `months` months of PA values (one row per ym) for a given
    (iso3, hazard_code), ordered from oldest to newest.

    Data source: facts_snapshot table, which is expected to contain unified canonical
    rows from facts_resolved, facts_deltas, and acled_monthly_fatalities.
    """

    where_pa = _pa_where_clause()
    sql = """
        SELECT
            ym,
            SUM(value) AS pa_value
        FROM facts_snapshot
        WHERE iso3 = ?
          AND hazard_code = ?
          AND ({where_pa})
        GROUP BY ym
        ORDER BY ym ASC
        LIMIT ?
    """.format(where_pa=where_pa)

    rows = con.execute(sql, [iso3, hazard_code, months]).fetchall()
    return [PaTrendPoint(ym=row[0], pa_value=float(row[1])) for row in rows]


def render_pa_trend_markdown(
    trend: Iterable[PaTrendPoint],
    *,
    iso3: str,
    hazard_code: str,
    title: str | None = None,
) -> str:
    """
    Render a PA trend (as returned by get_pa_trend) as a compact Markdown table.
    """

    rows = list(trend)
    if not rows:
        return f"No PA history available for {iso3} / {hazard_code}.\n"

    title_line = title or f"### {iso3} — People Affected (last {len(rows)} months, hazard={hazard_code})"

    lines: List[str] = []
    lines.append(title_line)
    lines.append("")
    lines.append("| Month | People affected |")
    lines.append("|-------|-----------------|")
    for pt in rows:
        lines.append(f"| {pt.ym} | {int(round(pt.pa_value))} |")
    lines.append("")

    return "\n".join(lines)
