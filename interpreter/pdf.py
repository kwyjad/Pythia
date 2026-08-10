# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Render a stored interpretation to PDF (Phase 6).

Runs in CI, never in the API container. Reads the newest ``interpretations``
row (or an explicit ``--interpretation-id``), builds a self-contained HTML
document from its ``content_md`` — the markdown is our own renderer's output,
every figure already substituted — and hands it to WeasyPrint.

    python -m interpreter.pdf --db "$PYTHIA_DB_URL" \
        [--kind combined] [--interpretation-id int_...] [--include-test] \
        --out-dir interpreter_out

Outputs (in --out-dir):
- ``report__{YYYY-MM}__v{n}.pdf`` — the versioned report
- ``interpreter_report_latest.pdf`` — constant-name copy for the release

WeasyPrint was chosen over a headless browser deliberately (fewer moving
parts); the ``/interpreter/print`` route stays the always-available
``window.print()`` fallback when the release asset is missing. The markdown →
HTML conversion here is a small deterministic subset converter, NOT a general
markdown engine — it only needs to handle what ``render.render_markdown``
emits (headings, lists, pipe tables, bold/italic/code inlines).

Same contracts as the runner: ``main()`` returns 0 in every outcome, and
``PYTHIA_INTERPRETER_STRICT_VALIDATION=1`` suppresses the PDF for a
non-``ok`` row (the row itself stays stored and inspectable).
"""

from __future__ import annotations

import argparse
import html as _html
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

from interpreter import config, mapviz, store

LOGGER = logging.getLogger(__name__)

LATEST_PDF_NAME = "interpreter_report_latest.pdf"


# ---------------------------------------------------------------------------
# Markdown subset -> HTML
# ---------------------------------------------------------------------------

_CODE_SPAN = re.compile(r"`([^`]+)`")
_LINK = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")
_BOLD = re.compile(r"\*\*([^*]+)\*\*")
_STAR_ITAL = re.compile(r"\*([^*\s][^*]*)\*")
# Word-boundary underscores only: question ids like SOM_ACE_PA_2026-08 carry
# interior underscores that must never be read as emphasis (they arrive
# inside code spans, but the guard costs nothing and protects prose too).
_UNDER_ITAL = re.compile(r"(?<![\w])_([^_]+)_(?![\w])")


def _inline(text: str) -> str:
    """Escape, then convert the inline constructs render_markdown emits."""
    # Code spans first: their content is escaped but exempt from the
    # emphasis regexes (placeholder swap keeps `_` inside ids literal).
    placeholders: list[str] = []

    def _stash(match: re.Match[str]) -> str:
        placeholders.append(f"<code>{_html.escape(match.group(1))}</code>")
        return f"\x00{len(placeholders) - 1}\x00"

    def _stash_link(match: re.Match[str]) -> str:
        label = _html.escape(match.group(1))
        href = _html.escape(match.group(2), quote=True)
        placeholders.append(f'<a href="{href}">{label}</a>')
        return f"\x00{len(placeholders) - 1}\x00"

    working = _CODE_SPAN.sub(_stash, text)
    # Links after code spans, before escaping: a question id carries
    # underscores that the emphasis regexes must not touch inside an href.
    working = _LINK.sub(_stash_link, working)
    working = _html.escape(working, quote=False)
    working = _BOLD.sub(r"<strong>\1</strong>", working)
    working = _STAR_ITAL.sub(r"<em>\1</em>", working)
    working = _UNDER_ITAL.sub(r"<em>\1</em>", working)
    for idx, span in enumerate(placeholders):
        working = working.replace(f"\x00{idx}\x00", span)
    return working


_TABLE_SEP = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$")


def markdown_to_html(md: str) -> str:
    """The deterministic subset converter (see module docstring)."""
    out: list[str] = []
    lines = (md or "").split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        if stripped.startswith("<svg"):
            # Charts and the map are inline SVG our own renderer produced.
            # They pass through untouched; escaping them would print the
            # markup instead of the picture.
            svg = [stripped]
            while not svg[-1].rstrip().endswith("</svg>") and i + 1 < len(lines):
                i += 1
                svg.append(lines[i].strip())
            i += 1
            out.append('<div class="figure">' + "".join(svg) + "</div>")
            continue
        if stripped.startswith("### "):
            out.append(f"<h3>{_inline(stripped[4:])}</h3>")
            i += 1
            continue
        if stripped.startswith("## "):
            out.append(f"<h2>{_inline(stripped[3:])}</h2>")
            i += 1
            continue
        if stripped.startswith("# "):
            out.append(f"<h1>{_inline(stripped[2:])}</h1>")
            i += 1
            continue
        if stripped.startswith("- "):
            items = []
            while i < len(lines) and lines[i].strip().startswith("- "):
                items.append(f"<li>{_inline(lines[i].strip()[2:])}</li>")
                i += 1
            out.append("<ul>" + "".join(items) + "</ul>")
            continue
        if stripped.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(lines[i].strip())
                i += 1
            body_rows = []
            header_cells: list[str] | None = None
            for r_idx, row in enumerate(rows):
                if _TABLE_SEP.match(row):
                    continue
                cells = [c.strip() for c in row.strip("|").split("|")]
                if header_cells is None and r_idx == 0 and len(rows) > 1 and _TABLE_SEP.match(rows[1]):
                    header_cells = cells
                    continue
                body_rows.append(cells)
            parts = ["<table>"]
            if header_cells:
                parts.append(
                    "<thead><tr>"
                    + "".join(f"<th>{_inline(c)}</th>" for c in header_cells)
                    + "</tr></thead>"
                )
            parts.append("<tbody>")
            for cells in body_rows:
                parts.append(
                    "<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in cells) + "</tr>"
                )
            parts.append("</tbody></table>")
            out.append("".join(parts))
            continue
        # Paragraph: consume consecutive non-structural lines.
        para = [stripped]
        i += 1
        while i < len(lines):
            nxt = lines[i].strip()
            if not nxt or nxt.startswith(("#", "- ", "|", "<svg")):
                break
            para.append(nxt)
            i += 1
        out.append(f"<p>{_inline(' '.join(para))}</p>")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# HTML document
# ---------------------------------------------------------------------------

# Fred's palette (web/tailwind.config.ts), so the printed report and the
# dashboard are recognisably the same publication.
FRED_PRIMARY = "#156082"
FRED_SECONDARY = "#80350E"
FRED_TEXT = "#3A3A3A"
FRED_BORDER = "#D6D6D6"
FRED_MUTED = "#6B7280"
FRED_SURFACE = "#FFFFFF"
FRED_BG = "#F5F5F5"

_CSS = f"""
@page {{
  margin: 16mm 18mm 18mm;
  @bottom-center {{
    content: "Fred  ·  page " counter(page) " of " counter(pages);
    font-family: sans-serif; font-size: 8pt; color: {FRED_MUTED};
  }}
}}
body {{
  font-family: -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  color: {FRED_TEXT}; font-size: 10.5pt; line-height: 1.55;
}}
.masthead {{
  border-bottom: 3px solid {FRED_PRIMARY};
  padding-bottom: 6pt; margin-bottom: 12pt;
}}
.masthead .wordmark {{
  font-size: 22pt; font-weight: 700; color: {FRED_PRIMARY};
  letter-spacing: -0.5pt; line-height: 1;
}}
.masthead .strap {{
  font-size: 9.5pt; color: {FRED_MUTED}; margin-top: 3pt;
}}
.masthead .month {{
  float: right; font-size: 11pt; color: {FRED_SECONDARY}; font-weight: 600;
}}
h1 {{ font-size: 16pt; margin: 0 0 6pt; color: {FRED_TEXT}; }}
h2 {{ font-size: 12.5pt; margin: 16pt 0 5pt; page-break-after: avoid;
     color: {FRED_PRIMARY};
     border-bottom: 1px solid {FRED_BORDER}; padding-bottom: 2pt; }}
h2 + * {{ page-break-before: avoid; }}
h3 {{ font-size: 11pt; margin: 11pt 0 3pt; page-break-after: avoid;
     color: {FRED_SECONDARY}; }}
p {{ margin: 4pt 0; }}
ul {{ margin: 4pt 0 4pt 14pt; padding: 0; }}
li {{ margin: 2pt 0; }}
a {{ color: {FRED_PRIMARY}; text-decoration: none; }}
code {{ font-family: "SF Mono", Menlo, Consolas, monospace; font-size: 9pt;
       background: {FRED_BG}; padding: 0 2pt; border-radius: 2pt; }}
table {{ border-collapse: collapse; margin: 6pt 0; font-size: 9.5pt; }}
th, td {{ border: 1px solid {FRED_BORDER}; padding: 2pt 6pt; text-align: left; }}
th {{ background: {FRED_BG}; color: {FRED_TEXT}; }}
.figure {{ margin: 6pt 0 8pt; page-break-inside: avoid; }}
.figure svg {{ max-width: 100%; height: auto; }}
.mapblock {{ margin: 10pt 0 14pt; page-break-inside: avoid; }}
.mapblock .caption {{ font-size: 9pt; color: {FRED_MUTED}; margin-top: 3pt; }}
.meta {{ color: {FRED_MUTED}; font-size: 8.5pt; margin-bottom: 10pt; }}
.banner {{ border: 1px solid {FRED_SECONDARY}; background: #FBEDE6;
          color: {FRED_SECONDARY};
          padding: 6pt 8pt; margin-bottom: 10pt; font-size: 9.5pt; }}
"""

_STRAPLINE = "Monthly risk report from the Fred forecasting system"


def _masthead(row: dict[str, Any]) -> str:
    month = _html.escape(month_label(row))
    return (
        '<div class="masthead">'
        f'<div class="month">{month}</div>'
        '<div class="wordmark">Fred</div>'
        f'<div class="strap">{_STRAPLINE}</div>'
        "</div>"
    )


def _map_block(map_svg: str | None, captions: list[str] | None) -> str:
    """The attention map, printed once under the masthead.

    The dashboard draws this with JavaScript, which a PDF cannot run, so the
    printed report carries its own SVG. When the map is unavailable the block
    is simply absent; a missing picture must not fail a report.
    """
    if not map_svg:
        return ""
    caption = ""
    if captions:
        caption = (
            '<div class="caption">Darkest first: '
            + _html.escape(", ".join(captions))
            + ".</div>"
        )
    return f'<div class="mapblock">{map_svg}{caption}</div>'


def build_report_html(
    row: dict[str, Any],
    *,
    map_svg: str | None = None,
    map_captions: list[str] | None = None,
) -> str:
    """A self-contained HTML document from one interpretations row."""
    body = markdown_to_html(str(row.get("content_md") or ""))
    status = str(row.get("status") or "")
    banner = ""
    if status != "ok":
        banner = (
            '<div class="banner">This report failed automated validation '
            f"(status: {_html.escape(status)}) and is included for inspection "
            "only.</div>"
        )
    meta_bits = [
        f"kind: {_html.escape(str(row.get('kind') or ''))}",
        f"version: v{row.get('version')}",
        f"generated: {_html.escape(str(row.get('created_at') or '')[:19])}",
        f"id: {_html.escape(str(row.get('interpretation_id') or ''))}",
    ]
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>Fred - monthly risk report</title><style>{_CSS}</style>"
        "</head><body>"
        f"{_masthead(row)}"
        f"{banner}"
        f"{_map_block(map_svg, map_captions)}"
        f"<div class='meta'>{' · '.join(meta_bits)}</div>"
        f"{body}</body></html>"
    )


def _render_pdf(html: str, out_path: Path) -> None:
    """WeasyPrint seam (lazy import; tests monkeypatch this)."""
    from weasyprint import HTML  # noqa: PLC0415 - heavy optional dep

    HTML(string=html).write_pdf(str(out_path))


# ---------------------------------------------------------------------------
# Row selection + orchestration
# ---------------------------------------------------------------------------

_ROW_COLUMNS = (
    "interpretation_id", "kind", "run_id", "hs_run_id", "scored_run_id",
    "version", "status", "content_md", "created_at", "is_test",
)


def select_row(
    con,
    *,
    kind: str = "combined",
    interpretation_id: str | None = None,
    include_test: bool = False,
) -> dict[str, Any] | None:
    """Newest row of ``kind`` (or the explicit id), test-filtered by default."""
    store.ensure_table(con)
    cols = ", ".join(_ROW_COLUMNS)
    if interpretation_id:
        rows = con.execute(
            f"SELECT {cols} FROM interpretations WHERE interpretation_id = ?",
            [interpretation_id],
        ).fetchall()
    else:
        test_clause = "" if include_test else " AND COALESCE(is_test, FALSE) = FALSE"
        rows = con.execute(
            f"""
            SELECT {cols} FROM interpretations
            WHERE kind = ?{test_clause}
            ORDER BY created_at DESC, version DESC
            LIMIT 1
            """,
            [kind],
        ).fetchall()
    if not rows:
        return None
    return dict(zip(_ROW_COLUMNS, rows[0]))


_HS_RUN_MONTH = re.compile(r"^hs_(\d{4})(\d{2})\d{2}T")


def month_label(row: dict[str, Any]) -> str:
    """YYYY-MM for the versioned filename — the month the report is ABOUT.

    Scored rows use their round key. Current/combined rows take the month
    from ``hs_run_id`` rather than ``created_at``: for a normal cycle the two
    agree, but a backfill generates July's report today and naming that file
    for August would misfile it. Falls back to the creation month when the
    run id is missing or malformed.
    """
    if str(row.get("kind") or "") == "scored" and row.get("scored_run_id"):
        label = str(row["scored_run_id"])[:7]
        if re.match(r"^\d{4}-\d{2}$", label):
            return label
    match = _HS_RUN_MONTH.match(str(row.get("hs_run_id") or ""))
    if match:
        return f"{match.group(1)}-{match.group(2)}"
    return str(row.get("created_at") or "")[:7] or "unknown"


def pdf_filename(row: dict[str, Any]) -> str:
    return f"report__{month_label(row)}__v{int(row.get('version') or 0)}.pdf"


def generate_pdf(
    *,
    db: str,
    out_dir: str,
    kind: str = "combined",
    interpretation_id: str | None = None,
    include_test: bool = False,
) -> dict[str, Any]:
    """Select -> HTML -> WeasyPrint -> versioned PDF + latest copy."""
    result: dict[str, Any] = {"status": "skipped"}
    if not config.enabled():
        result["reason"] = "disabled"
        return result

    from resolver.db import duckdb_io

    con = duckdb_io.get_db(db or duckdb_io.DEFAULT_DB_URL)
    map_values: dict[str, float] = {}
    try:
        row = select_row(
            con, kind=kind, interpretation_id=interpretation_id,
            include_test=include_test,
        )
        if row is not None:
            map_values = mapviz.values_from_deviation(
                con, row.get("run_id"), include_test=include_test,
            )
    finally:
        duckdb_io.close_db(con)

    if row is None:
        LOGGER.info("[interpreter.pdf] no interpretation row found — nothing to render")
        result["reason"] = "no_row"
        return result
    if not row.get("content_md"):
        LOGGER.warning(
            "[interpreter.pdf] %s has no content_md (status=%s) — nothing to render",
            row.get("interpretation_id"), row.get("status"),
        )
        result.update({"reason": "no_content", "interpretation_id": row.get("interpretation_id")})
        return result
    if row.get("status") != "ok" and config.strict_validation():
        # Same rule as the runner's out-dir artifacts: strict mode keeps a
        # failed report out of publication while the row stays inspectable.
        LOGGER.warning(
            "[interpreter.pdf] strict validation: PDF suppressed for %s (status=%s)",
            row.get("interpretation_id"), row.get("status"),
        )
        result.update({
            "reason": "strict_suppressed",
            "interpretation_id": row.get("interpretation_id"),
            "row_status": row.get("status"),
        })
        return result

    map_svg = mapviz.attention_map_svg(
        map_values,
        title="Where this month's forecasts sit furthest from their usual pattern",
    )
    html = build_report_html(
        row,
        map_svg=map_svg or None,
        map_captions=mapviz.country_labels(map_values),
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    versioned = out / pdf_filename(row)
    try:
        _render_pdf(html, versioned)
    except Exception as exc:  # noqa: BLE001 - a PDF bug must never fail the run
        LOGGER.warning("[interpreter.pdf] WeasyPrint render failed: %s", exc)
        # The HTML itself is still useful evidence — keep it beside the row.
        (out / (versioned.stem + ".html")).write_text(html, encoding="utf-8")
        result.update({"status": "failed_render", "error": str(exc)})
        return result
    (out / LATEST_PDF_NAME).write_bytes(versioned.read_bytes())
    LOGGER.info("[interpreter.pdf] wrote %s (+ %s)", versioned, LATEST_PDF_NAME)
    result.update({
        "status": "ok",
        "interpretation_id": row.get("interpretation_id"),
        "row_status": row.get("status"),
        "pdf": str(versioned),
        "latest": str(out / LATEST_PDF_NAME),
    })
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, help="DuckDB URL or path")
    parser.add_argument("--kind", choices=["current", "scored", "combined"],
                        default="combined")
    parser.add_argument("--interpretation-id", default=None,
                        help="Render this exact row instead of the newest")
    parser.add_argument("--include-test", action="store_true")
    parser.add_argument("--out-dir", default="interpreter_out")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[interpreter.pdf] %(message)s")
    try:
        result = generate_pdf(
            db=args.db, out_dir=args.out_dir, kind=args.kind,
            interpretation_id=args.interpretation_id,
            include_test=args.include_test,
        )
    except Exception as exc:  # noqa: BLE001 - never fail the pipeline
        LOGGER.error("[interpreter.pdf] failed: %s", exc)
        result = {"status": "error", "error": str(exc)}
    print(f"[interpreter.pdf] RESULT={json.dumps(result, default=str)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
