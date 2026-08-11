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


def markdown_to_blocks(md: str) -> list[str]:
    """The deterministic subset converter, one HTML block per element.

    Returned as blocks rather than one string so the first page can be
    assembled properly: the title and the issue line come from the markdown,
    and the map and the metadata belong between them and the rest. The map
    used to be emitted before the whole body, which put its caption above the
    report's own title.
    """
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
    return out


def markdown_to_html(md: str) -> str:
    """The whole body as one HTML string (the blocks, joined)."""
    return "\n".join(markdown_to_blocks(md))


_H2 = re.compile(r"^<h2>(.*)</h2>$", re.S)


def _anchor_headings(blocks: list[str]) -> tuple[list[str], list[tuple[str, str]]]:
    """Give every top-level heading an id, and collect them for the contents.

    Returns (blocks with ids, [(id, title)]). Only h2 is collected: a table of
    contents that listed every sub-heading would be as long as the report it
    is meant to make navigable.
    """
    out: list[str] = []
    entries: list[tuple[str, str]] = []
    for block in blocks:
        match = _H2.match(block.strip())
        if match:
            anchor = f"sec-{len(entries)}"
            entries.append((anchor, match.group(1)))
            out.append(f'<h2 id="{anchor}">{match.group(1)}</h2>')
        else:
            out.append(block)
    return out, entries


def _contents_block(entries: list[tuple[str, str]]) -> str:
    """The table of contents, with real page numbers.

    ``target-counter`` is resolved by WeasyPrint at layout time, so the
    numbers are the printed ones rather than an estimate.
    """
    if len(entries) < 3:
        return ""  # a contents page for two sections helps nobody
    items = "".join(
        f'<li><a href="#{anchor}">{title}</a></li>' for anchor, title in entries
    )
    return f'<nav class="toc"><h2>Contents</h2><ul>{items}</ul></nav>'


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
  /* The bottom margin has to clear the running footer, or the page number
     lands on top of the last line of body text. It did. */
  margin: 16mm 18mm 24mm;
  @bottom-center {{
    content: "Fred  ·  page " counter(page) " of " counter(pages);
    font-family: sans-serif; font-size: 8pt; color: {FRED_MUTED};
    vertical-align: top; padding-top: 6mm;
  }}
}}
body {{
  font-family: -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  color: {FRED_TEXT}; font-size: 10.5pt; line-height: 1.55;
}}
h1 {{ font-size: 19pt; margin: 0 0 6pt; color: {FRED_PRIMARY};
     border-bottom: 3px solid {FRED_PRIMARY}; padding-bottom: 5pt; }}
h2 {{ font-size: 12.5pt; margin: 16pt 0 5pt; page-break-after: avoid;
     color: {FRED_PRIMARY};
     border-bottom: 1px solid {FRED_BORDER}; padding-bottom: 2pt; }}
h2 + * {{ page-break-before: avoid; }}
h3 {{ font-size: 11pt; margin: 11pt 0 3pt; page-break-after: avoid;
     color: {FRED_SECONDARY}; }}
p {{ margin: 4pt 0; }}
/* Indent with PADDING, not margin: with margin and zero padding the marker
   is laid out outside the box and drifts away from its own text. */
ul {{ margin: 4pt 0; padding-left: 16pt; list-style-position: outside; }}
li {{ margin: 2pt 0; padding-left: 2pt; }}
/* The map and the contents share page one, and the body starts on the
   next: the break belongs to the pair, not to the contents alone. */
.frontmatter {{ page-break-after: always; }}
.toc {{ margin-top: 10pt; }}
.toc h2 {{ margin-top: 6pt; }}
.toc ul {{ list-style: none; padding-left: 0; }}
.toc li {{ margin: 3pt 0; border-bottom: 1px dotted {FRED_BORDER}; }}
.toc li a {{ color: {FRED_TEXT}; }}
.toc li a::after {{ content: target-counter(attr(href), page); float: right;
                   color: {FRED_MUTED}; }}
a {{ color: {FRED_PRIMARY}; text-decoration: none; }}
code {{ font-family: "SF Mono", Menlo, Consolas, monospace; font-size: 9pt;
       background: {FRED_BG}; padding: 0 2pt; border-radius: 2pt; }}
table {{ border-collapse: collapse; margin: 6pt 0; font-size: 9.5pt; }}
th, td {{ border: 1px solid {FRED_BORDER}; padding: 2pt 6pt; text-align: left; }}
th {{ background: {FRED_BG}; color: {FRED_TEXT}; }}
.figure {{ margin: 6pt 0 8pt; page-break-inside: avoid; }}
.figure svg {{ max-width: 100%; height: auto; }}
.mapblock {{ margin: 10pt 0 14pt; page-break-inside: avoid; text-align: center; }}
.mapblock svg {{ max-width: 100%; height: auto; }}
.mapblock .caption {{ font-size: 9pt; color: {FRED_MUTED}; margin-top: 4pt;
                     text-align: left; }}
.meta {{ color: {FRED_MUTED}; font-size: 8.5pt; margin-bottom: 10pt; }}
.banner {{ border: 1px solid {FRED_SECONDARY}; background: #FBEDE6;
          color: {FRED_SECONDARY};
          padding: 6pt 8pt; margin-bottom: 10pt; font-size: 9.5pt; }}
"""

# The masthead block was cut in v4. Two titles at the top of page one
# ("Fred" plus a strapline, then the report's own h1) is one title too many,
# and the strapline said nothing the title did not. The month moved into the
# issue line, which already sits under the title and carries the unreviewed
# stamp beside it.


def _map_block(map_svg: str | None, captions: list[str] | None) -> str:
    """The attention map, centred, with its scale stated in words.

    The dashboard draws this with JavaScript, which a PDF cannot run, so the
    printed report carries its own SVG. When the map is unavailable the block
    is simply absent; a missing picture must not fail a report.

    The caption used to be the bare string "Darkest first: " followed by a
    list of countries, which read as a debugging artefact. It now says what
    the shading means and then names the countries.
    """
    if not map_svg:
        return ""
    text = (
        "Warm shading means a country expects more people affected than its "
        "history would suggest; cool shading means fewer. The depth is how "
        "far it has moved. Grey means no forecast this month."
    )
    if captions:
        text += " Moved most: " + _html.escape(", ".join(captions)) + "."
    return f'<div class="mapblock">{map_svg}<div class="caption">{text}</div></div>'


def build_report_html(
    row: dict[str, Any],
    *,
    map_svg: str | None = None,
    map_captions: list[str] | None = None,
) -> str:
    """A self-contained HTML document from one interpretations row.

    First page order: title, the issue line, then the map and the contents
    together, then the body. The map used to be emitted ahead of the whole
    body, so its caption printed above the report's own title; the contents
    then took a page of its own and the reader turned twice before the
    first entry.
    """
    blocks = markdown_to_blocks(str(row.get("content_md") or ""))
    blocks, headings = _anchor_headings(blocks)

    # The title (h1) and the issue line beneath it come from the markdown and
    # belong above the map; everything from the first section onward comes
    # after the contents page.
    lead: list[str] = []
    while blocks and not blocks[0].startswith(("<h2", "<h3")):
        lead.append(blocks.pop(0))
        if len(lead) >= 3:
            break

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
        f"<title>Fred's Monthly Forecast Report</title><style>{_CSS}</style>"
        "</head><body>"
        f"{banner}"
        f"{''.join(lead)}"
        # The map and the contents sit together on page one. They used to be
        # split by the metadata line, which pushed the contents onto a page
        # of its own and made the reader turn twice before the first entry.
        f"<div class='frontmatter'>"
        f"{_map_block(map_svg, map_captions)}"
        f"{_contents_block(headings)}"
        f"</div>"
        f"<div class='meta'>{' · '.join(meta_bits)}</div>"
        f"{''.join(blocks)}</body></html>"
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
        title="Where this month's forecasts sit against their usual level",
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
