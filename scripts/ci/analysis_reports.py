# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The analysis reports: parse them, render them, index them (Nov 2026).

Three kinds of report are written by scheduled Claude sessions as Markdown
into Google Drive, ``Fred_archive/Performance_Reports/<YYYY>/``:

    skill      fred_performance_<YYYY-MM>.md
    resolver   fred_resolver_review_<YYYY-MM>.md
    forecast   fred_forecast_review_<YYYY-MM-DD>__<fc_run_id>.md

any of them with an optional ``__rerun_<YYYY-MM-DD>`` before ``.md``. A rerun
is a newer version of the same report and never replaces the original.

The dashboard API reads GitHub release assets only, never Drive, so
``sync_analysis_reports.yml`` copies the reports into their own release
(``analysis-reports``, never the data release) with a PDF of each and an
``index.json``. This module is everything that workflow does apart from the
rclone and ``gh`` calls, which stay in the workflow so this file needs
nothing outside the standard library to parse and index (WeasyPrint only to
render, imported lazily).

    python -m scripts.ci.analysis_reports build \
        --src DIR --out DIR [--previous-index index.json] [--force-rebuild] \
        [--modtimes lsjson.json] [--summary summary.md]

reads every ``.md`` under ``--src`` (any depth; ``template/`` is skipped),
writes to ``--out`` the assets that changed (``<id>.md``, ``<id>.pdf``) and
``index.json``, and prints the counts. A report whose text matches a
credential in the environment is blocked and named; a PDF that fails is
logged and recorded as ``pdf: null``. ``main()`` returns 0 in every outcome
but a crash.

Docs: docs/analysis_reports.md.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - import convenience
    sys.path.insert(0, str(REPO_ROOT))

LOGGER = logging.getLogger("analysis_reports")

SCHEMA_VERSION = 1
SOURCE = "Fred_archive/Performance_Reports"
DEFAULT_TAG = "analysis-reports"
INDEX_NAME = "index.json"

DISPLAY_NAMES = {
    "skill": "Skill analysis",
    "resolver": "Resolver pipeline review",
    "forecast": "Forecast pipeline review",
}

_RERUN = r"(?:__rerun_(?P<rerun>\d{4}-\d{2}-\d{2}))?"
FILE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("skill", re.compile(rf"^fred_performance_(?P<period>\d{{4}}-\d{{2}}){_RERUN}\.md$")),
    ("resolver", re.compile(rf"^fred_resolver_review_(?P<period>\d{{4}}-\d{{2}}){_RERUN}\.md$")),
    ("forecast", re.compile(
        rf"^fred_forecast_review_(?P<date>\d{{4}}-\d{{2}}-\d{{2}})__(?P<fc>fc_\d+){_RERUN}\.md$")),
)

#: The chain a report covers, in either of the two forms the header table uses:
#: ``pythia_2026-11-11__scoring__gh43009722469.zip`` and
#: ``2026-10-13__forecast__fc_1791866322__gh38355281121``.
CHAIN_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2})__(scoring|forecast)__(?:(fc_\d+)__)?gh(\d+)")

SUMMARY_MAX = 320
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_ESCAPE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!|<>~])")


# ---------------------------------------------------------------------------
# Parsing (pure)
# ---------------------------------------------------------------------------


def parse_filename(name: str) -> dict[str, Any] | None:
    """Type, period, chain hints and rerun suffix from a file name, or None."""
    for kind, pattern in FILE_PATTERNS:
        m = pattern.match(name)
        if not m:
            continue
        stem = name[:-3]
        rerun = m.group("rerun")
        original = stem[: -len(f"__rerun_{rerun}")] if rerun else None
        if kind == "forecast":
            date = m.group("date")
            period = date[:7]
            fc = m.group("fc")
        else:
            date, fc = None, None
            period = m.group("period")
        return {
            "report_id": stem,
            "type": kind,
            "display_name": DISPLAY_NAMES[kind],
            "period": period,
            "file_date": date,
            "file_fc_run_id": fc,
            "rerun_of": original,
            "rerun_date": rerun,
        }
    return None


def _unescape(text: str) -> str:
    return _ESCAPE.sub(r"\1", text)


def _split_header(text: str) -> tuple[list[str], list[str]]:
    """Lines before the first ``## 1.`` heading, and the lines from it on."""
    lines = _unescape(text).replace("\r\n", "\n").split("\n")
    for i, line in enumerate(lines):
        if re.match(r"^##\s+1[.\s]", line.strip()):
            return lines[:i], lines[i:]
    return lines, []


def header_fields(header: Iterable[str]) -> dict[str, str]:
    """``| Field | Value |`` rows before section 1, keyed by the field name."""
    out: dict[str, str] = {}
    for raw in header:
        line = raw.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 2 or not cells[0] or set(cells[0]) <= set("-: "):
            continue
        if cells[0].lower() == "field":
            continue
        out.setdefault(cells[0], " | ".join(cells[1:]).strip())
    return out


def parse_chain(text: str) -> dict[str, Any] | None:
    m = CHAIN_RE.search(text or "")
    if not m:
        return None
    date, kind, fc, gh = m.groups()
    folder = f"{date}__{kind}__" + (f"{fc}__" if fc else "") + f"gh{gh}"
    return {"date": date, "kind": kind, "fc_run_id": fc, "publish_run_id": gh,
            "folder_name": folder}


_MD_INLINE = (
    (re.compile(r"\[([^\]]+)\]\([^)]*\)"), r"\1"),
    (re.compile(r"`([^`]*)`"), r"\1"),
    (re.compile(r"\*\*([^*]+)\*\*"), r"\1"),
    (re.compile(r"(?<![\w*])\*([^*\s][^*]*)\*(?![\w*])"), r"\1"),
    (re.compile(r"(?<![\w])_([^_]+)_(?![\w])"), r"\1"),
)


def strip_markdown(text: str) -> str:
    out = text
    for pattern, repl in _MD_INLINE:
        out = pattern.sub(repl, out)
    return re.sub(r"\s+", " ", out).strip()


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])")


def sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_END.split(text) if s]


def cut_summary(text: str, limit: int = SUMMARY_MAX) -> str:
    """Whole sentences up to ``limit`` characters; a first sentence longer
    than that is cut at a word with an ellipsis."""
    parts = sentences(text)
    out = ""
    for part in parts:
        candidate = f"{out} {part}".strip()
        if len(candidate) >= limit:
            break
        out = candidate
    if out:
        return out
    if not text:
        return ""
    clipped = text[: limit - 1].rsplit(" ", 1)[0].rstrip(",;:")
    return clipped + "…"


def first_summary_paragraph(body: list[str]) -> str:
    """The first paragraph under the ``## 1.`` heading, markdown stripped."""
    if not body:
        return ""
    para: list[str] = []
    for raw in body[1:]:
        line = raw.strip()
        if not line:
            if para:
                break
            continue
        if line.startswith(("#", "|", "- ", "* ")) or re.match(r"^\d+[.)]\s", line):
            if para:
                break
            if line.startswith("#"):
                break
            continue
        para.append(line)
    return strip_markdown(" ".join(para))


_VERDICTS = (
    ("worked with faults", re.compile(r"\bworked with faults\b", re.I)),
    ("failed", re.compile(r"\bfailed\b", re.I)),
    ("worked", re.compile(r"\bworked\b", re.I)),
)


def extract_verdict(summary: str) -> str | None:
    """The verdict named in the summary's first sentence (forecast reviews)."""
    parts = sentences(summary)
    if not parts:
        return None
    first = parts[0]
    for verdict, pattern in _VERDICTS:
        if pattern.search(first):
            return verdict
    return None


def _field(fields: Mapping[str, str], *names: str) -> str | None:
    lowered = {k.lower(): v for k, v in fields.items()}
    for name in names:
        if name.lower() in lowered and lowered[name.lower()].strip():
            return lowered[name.lower()].strip()
    return None


def parse_report(name: str, text: str, *, drive_modified: str | None = None) -> dict[str, Any] | None:
    """One index record, or None when the file name matches no pattern."""
    meta = parse_filename(name)
    if meta is None:
        return None
    raw = text.encode("utf-8")
    header, body = _split_header(text)
    fields = header_fields(header)
    chain_text = " ".join(fields.get(k, "") for k in fields
                          if "chain" in k.lower()) or "\n".join(header)
    chain = parse_chain(chain_text) or parse_chain("\n".join(header))
    if chain is None:
        LOGGER.warning("%s: no scoring or forecast chain found in the header table; "
                       "indexed under its file-name period %s", name, meta["period"])
    title = ""
    for line in header:
        if line.strip().startswith("# "):
            title = strip_markdown(line.strip()[2:])
            break
    prepared_raw = _field(fields, "Prepared")
    prepared_match = _DATE.search(prepared_raw or "")
    summary_full = first_summary_paragraph(body)
    record = {
        **meta,
        "title": title or meta["display_name"],
        "chain": chain,
        "dry_run": "DRY RUN" in "\n".join(header),
        "template_version": _field(fields, "Template version"),
        "prepared": prepared_match.group(0) if prepared_match else None,
        "summary": cut_summary(summary_full),
        "verdict": extract_verdict(summary_full) if meta["type"] == "forecast" else None,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "drive_modified": drive_modified,
        "md_asset": f"{meta['report_id']}.md",
        "pdf_asset": f"{meta['report_id']}.pdf",
        "removed_from_drive": False,
    }
    return record


# ---------------------------------------------------------------------------
# Index (pure)
# ---------------------------------------------------------------------------


def change_status(record: Mapping[str, Any], previous: Mapping[str, Mapping[str, Any]],
                  *, force: bool = False) -> str:
    """``new``, ``changed`` or ``unchanged`` against the previous index."""
    old = previous.get(record["report_id"])
    if old is None:
        return "new"
    if old.get("sha256") != record["sha256"]:
        return "changed"
    return "changed" if force else "unchanged"


def previous_records(index: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not index or not isinstance(index.get("reports"), list):
        return {}
    return {r["report_id"]: dict(r) for r in index["reports"]
            if isinstance(r, Mapping) and r.get("report_id")}


def build_index(records: Iterable[Mapping[str, Any]], previous: Mapping[str, Mapping[str, Any]],
                *, now: str | None = None) -> dict[str, Any]:
    """The new index: this run's records, plus any earlier report no longer in
    Drive, kept with ``removed_from_drive: true`` (assets are never deleted)."""
    current = {r["report_id"]: dict(r) for r in records}
    for rid, old in previous.items():
        if rid not in current:
            kept = dict(old)
            kept["removed_from_drive"] = True
            current[rid] = kept
    ordered = sorted(current.values(),
                     key=lambda r: (r.get("period") or "", r.get("report_id") or ""), reverse=True)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now or dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "source": SOURCE,
        "reports": ordered,
    }


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------


def report_html(md: str, title: str) -> str:
    """A self-contained HTML page for one report, with the interpreter PDF's
    converter, contents list and palette."""
    import html as _html  # noqa: PLC0415

    from interpreter.pdf import _CSS, _anchor_headings, _contents_block, markdown_to_blocks  # noqa: PLC0415

    blocks, headings = _anchor_headings(markdown_to_blocks(md))
    lead: list[str] = []
    while blocks and not blocks[0].startswith(("<h2", "<h3")) and len(lead) < 6:
        lead.append(blocks.pop(0))
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>{_html.escape(title)}</title><style>{_CSS}"
        ".frontmatter { page-break-after: avoid; } table { width: 100%; }"
        "</style></head><body>"
        f"{''.join(lead)}{_contents_block(headings)}{''.join(blocks)}</body></html>"
    )


def render_pdf(md: str, title: str, out_path: Path) -> None:
    """WeasyPrint with the interpreter's data-only URL fetcher (tests patch this)."""
    from interpreter.pdf import _render_pdf  # noqa: PLC0415

    _render_pdf(report_html(md, title), out_path)


# ---------------------------------------------------------------------------
# Build: Drive copy in, release assets and index out
# ---------------------------------------------------------------------------


def find_reports(src: Path) -> tuple[list[Path], list[str]]:
    """Every report file under ``src`` and every other file skipped (named).

    ``template/`` is never read: it holds the templates, not reports."""
    found: list[Path] = []
    skipped: list[str] = []
    for path in sorted(src.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(src).as_posix()
        if rel.split("/", 1)[0] == "template":
            continue
        if parse_filename(path.name) is None:
            skipped.append(rel)
            continue
        found.append(path)
    return found, skipped


def _load_modtimes(path: Path | None) -> dict[str, str]:
    """``{name: ModTime}`` from ``rclone lsjson -R`` output, if given."""
    if path is None or not path.exists():
        return {}
    try:
        entries = json.loads(path.read_text(encoding="utf-8") or "[]")
    except json.JSONDecodeError:
        return {}
    return {Path(str(e.get("Path") or e.get("Name") or "")).name: str(e.get("ModTime") or "")
            for e in entries if isinstance(e, Mapping)}


def _secret_blocked(path: Path, env: Mapping[str, str]) -> list[str]:
    """Credential fingerprints found in one report, or []. Uses the cycle
    bundle's scan so the two cannot disagree about what a credential is."""
    from scripts.ci.build_cycle_diagnostics import SecretLeak, scan_for_secrets  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy2(path, Path(tmp) / path.name)
        try:
            scan_for_secrets(Path(tmp), env)
        except SecretLeak as exc:
            hits = getattr(exc, "hits", None) or {}
            return sorted({h for found in hits.values() for h in found}) or ["credential"]
    return []


def build(src: Path, out: Path, *, previous_index: Mapping[str, Any] | None = None,
          force_rebuild: bool = False, modtimes: Mapping[str, str] | None = None,
          env: Mapping[str, str] | None = None, renderer=None) -> dict[str, Any]:
    """Parse, scan and render; write changed assets and ``index.json`` to ``out``.

    Returns a report of what happened, which the job summary prints."""
    env = os.environ if env is None else env
    renderer = renderer or render_pdf
    out.mkdir(parents=True, exist_ok=True)
    previous = previous_records(previous_index)
    modtimes = modtimes or {}
    paths, skipped = find_reports(src)
    report: dict[str, Any] = {"found": len(paths), "new": [], "changed": [], "unchanged": [],
                              "skipped": skipped, "blocked": {}, "pdf_failures": {},
                              "no_chain": [], "uploads": []}
    records: list[dict[str, Any]] = []
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        record = parse_report(path.name, text, drive_modified=modtimes.get(path.name))
        if record is None:  # pragma: no cover - find_reports filtered already
            continue
        if record["chain"] is None:
            report["no_chain"].append(record["report_id"])
        hits = _secret_blocked(path, env)
        if hits:
            LOGGER.error("%s: blocked, text matches a credential (%s)", path.name, ", ".join(hits))
            report["blocked"][record["report_id"]] = hits
            if record["report_id"] in previous:
                # The published copy stays as it was; this one is not uploaded.
                records.append(dict(previous[record["report_id"]]))
            continue
        status = change_status(record, previous, force=force_rebuild)
        report[status].append(record["report_id"])
        old = previous.get(record["report_id"], {})
        if status == "unchanged":
            record["pdf_asset"] = old.get("pdf_asset", record["pdf_asset"])
            records.append(record)
            continue
        if status != "unchanged" and old.get("sha256") != record["sha256"]:
            shutil.copy2(path, out / record["md_asset"])
            report["uploads"].append(record["md_asset"])
        try:
            renderer(text, record["title"], out / record["pdf_asset"])
            report["uploads"].append(record["pdf_asset"])
        except Exception as exc:  # noqa: BLE001 - one PDF must not stop the rest
            LOGGER.warning("%s: PDF failed: %s", path.name, exc)
            report["pdf_failures"][record["report_id"]] = str(exc)[:300]
            record["pdf_asset"] = None
        records.append(record)
    index = build_index(records, previous)
    (out / INDEX_NAME).write_text(json.dumps(index, indent=2, sort_keys=False) + "\n",
                                  encoding="utf-8")
    report["removed_from_drive"] = [r["report_id"] for r in index["reports"]
                                    if r.get("removed_from_drive")]
    report["indexed"] = len(index["reports"])
    return report


def summary_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "## Analysis reports sync",
        "",
        "| Item | Count |",
        "|---|---|",
        f"| Reports found in Drive | {report.get('found', 0)} |",
        f"| New | {len(report.get('new', []))} |",
        f"| Changed | {len(report.get('changed', []))} |",
        f"| Unchanged | {len(report.get('unchanged', []))} |",
        f"| Skipped (name matches no pattern) | {len(report.get('skipped', []))} |",
        f"| Blocked (credential found) | {len(report.get('blocked', {}))} |",
        f"| PDF failures | {len(report.get('pdf_failures', {}))} |",
        f"| No chain in header | {len(report.get('no_chain', []))} |",
        f"| Kept, gone from Drive | {len(report.get('removed_from_drive', []))} |",
        f"| In the index | {report.get('indexed', 0)} |",
        "",
    ]
    for key, label in (("new", "New"), ("changed", "Changed"), ("skipped", "Skipped"),
                       ("no_chain", "No chain found")):
        items = report.get(key) or []
        if items:
            lines.append(f"**{label}:** " + ", ".join(f"`{i}`" for i in items))
            lines.append("")
    for rid, hits in (report.get("blocked") or {}).items():
        lines.append(f"**Blocked:** `{rid}` ({', '.join(hits)})")
    for rid, err in (report.get("pdf_failures") or {}).items():
        lines.append(f"**PDF failed:** `{rid}`: {err}")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build")
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--previous-index")
    p.add_argument("--force-rebuild", action="store_true")
    p.add_argument("--modtimes")
    p.add_argument("--summary")
    p.add_argument("--report-json")
    args = parser.parse_args(argv)

    previous = None
    if args.previous_index and Path(args.previous_index).exists():
        try:
            previous = json.loads(Path(args.previous_index).read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            LOGGER.warning("previous index unreadable; every report counts as new")
    report = build(Path(args.src), Path(args.out), previous_index=previous,
                   force_rebuild=args.force_rebuild,
                   modtimes=_load_modtimes(Path(args.modtimes) if args.modtimes else None))
    text = summary_markdown(report)
    print(text)
    if args.summary:
        with open(args.summary, "a", encoding="utf-8") as fh:
            fh.write(text)
    if args.report_json:
        Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
