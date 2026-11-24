# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.
"""Apply text edits to a Fred slide deck, in place, keeping its layout.

The decks in docs/decks/ were built with PptxGenJS and the generator scripts
are not in this repository, so a deck is never rebuilt: its text is edited
where it stands, in ``ppt/slides/slideN.xml`` or in the notes slide that
slide links to. Nothing else in the file is touched.

Edits are a JSON list::

    [
      {"slide": 3, "old": "Gemini 3.5 Flash", "new": "Gemini 3.6 Flash"},
      {"slide": 3, "where": "notes", "old": "the 28th", "new": "the 11th"},
      {"slide": 9, "old": "$60", "new": "$25", "count": 2}
    ]

``where`` is ``slide`` (default) or ``notes``. ``count`` is how many
occurrences the edit must find on that slide (default 1); an edit finding
another number is reported and changes nothing.

How text is matched. A paragraph's text is the concatenation of its runs
(``<a:r><a:t>``), so a phrase may span several runs. The changed part of
an edit is found by dropping the prefix and suffix ``old`` and ``new``
share, so ``"Cost: $60 a run" -> "Cost: $25 a run"`` changes only ``60``.
When the changed part sits in one run it is replaced there, keeping that
run's formatting. When it crosses runs it is written into the first run
and removed from the others, but ONLY where every run it crosses carries
the same formatting: crossing a bold label into plain text is refused,
because merging them would make the label plain or the text bold. Split
such an edit at the boundary.

Exit status: 0 when every edit applied and the file reopens; 1 when an
edit matched nothing, matched the wrong number of times or was refused;
2 when the edited file will not open. On 1 or 2 the input file is left
as it was.

Usage::

    python -m scripts.decks.edit_deck DECK.pptx --edits edits.json [--out NEW.pptx]
    python -m scripts.decks.edit_deck DECK.pptx --dump [--slide 3]
    python -m scripts.decks.edit_deck DECK.pptx --edits e.json --render OUTDIR

``--render`` converts the edited deck to PDF with LibreOffice and writes a
PNG of each changed slide, so text overflowing its box can be checked by
eye. It is skipped with a warning when LibreOffice is not installed.
"""

from __future__ import annotations

import argparse
import json
import posixpath
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from lxml import etree

NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
}
NOTES_REL = ("http://schemas.openxmlformats.org/officeDocument/2006/relationships/"
             "notesSlide")


@dataclass
class Edit:
    slide: int
    old: str
    new: str
    where: str = "slide"
    count: int = 1

    @classmethod
    def from_dict(cls, d: dict) -> "Edit":
        where = str(d.get("where", "slide"))
        if where not in ("slide", "notes"):
            raise ValueError(f"where must be 'slide' or 'notes', not {where!r}")
        if not d.get("old"):
            raise ValueError("an edit needs a non-empty 'old'")
        return cls(slide=int(d["slide"]), old=str(d["old"]), new=str(d["new"]),
                   where=where, count=int(d.get("count", 1)))

    def label(self) -> str:
        return f"slide {self.slide} {self.where}: {self.old!r} -> {self.new!r}"


@dataclass
class Result:
    applied: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    changed_slides: set[int] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Locating parts
# ---------------------------------------------------------------------------


def slide_part(n: int) -> str:
    return f"ppt/slides/slide{n}.xml"


def notes_part(zf: zipfile.ZipFile, n: int) -> str | None:
    """The notes slide slide ``n`` links to, or None."""
    rels = f"ppt/slides/_rels/slide{n}.xml.rels"
    try:
        root = etree.fromstring(zf.read(rels))
    except KeyError:
        return None
    for rel in root.findall("rel:Relationship", NS):
        if rel.get("Type") == NOTES_REL:
            target = rel.get("Target", "")
            return posixpath.normpath(posixpath.join("ppt/slides", target))
    return None


# ---------------------------------------------------------------------------
# Text replacement inside one XML part
# ---------------------------------------------------------------------------


def _runs(paragraph) -> list:
    """The ``<a:t>`` elements of a paragraph's text runs, in order."""
    return [r.find("a:t", NS) for r in paragraph.findall("a:r", NS)
            if r.find("a:t", NS) is not None]


def _rpr_key(t_elem) -> bytes:
    rpr = t_elem.getparent().find("a:rPr", NS)
    return b"" if rpr is None else etree.tostring(rpr, method="c14n")


def _changed_span(old: str, new: str) -> tuple[int, int, str]:
    """(start, end) of the part of ``old`` that changes, and what replaces it."""
    p = 0
    while p < len(old) and p < len(new) and old[p] == new[p]:
        p += 1
    s = 0
    while (s < len(old) - p and s < len(new) - p
           and old[len(old) - 1 - s] == new[len(new) - 1 - s]):
        s += 1
    return p, len(old) - s, new[p:len(new) - s]


def _find_all(text: str, needle: str) -> list[int]:
    out, i = [], text.find(needle)
    while i != -1:
        out.append(i)
        i = text.find(needle, i + len(needle))
    return out


def _apply_in_paragraph(paragraph, edit: Edit, start: int) -> str | None:
    """Replace the occurrence of ``edit.old`` starting at ``start``.

    Returns None on success, or why the edit was refused."""
    ts = _runs(paragraph)
    offsets, pos = [], 0
    for t in ts:
        offsets.append(pos)
        pos += len(t.text or "")
    a, b, repl = _changed_span(edit.old, edit.new)
    a += start
    b += start

    def run_at(i: int, *, end: bool = False) -> int:
        for k in range(len(ts) - 1, -1, -1):
            if (offsets[k] < i) if end else (offsets[k] <= i):
                return k
        return 0

    if a == b:  # pure insertion: put it in the run holding the character before
        k = run_at(a, end=True) if a > 0 else 0
        t = ts[k]
        off = a - offsets[k]
        t.text = (t.text or "")[:off] + repl + (t.text or "")[off:]
        return None

    first, last = run_at(a), run_at(b, end=True)
    if first != last:
        keys = {_rpr_key(ts[k]) for k in range(first, last + 1)}
        if len(keys) > 1:
            return ("the changed text crosses runs with different formatting "
                    "(e.g. a bold label and plain text); split the edit at the "
                    "boundary")
    for k in range(first, last + 1):
        t = ts[k]
        txt = t.text or ""
        lo = max(a - offsets[k], 0)
        hi = min(b - offsets[k], len(txt))
        t.text = txt[:lo] + (repl if k == first else "") + txt[hi:]
    return None


def apply_to_xml(xml: bytes, edits: Iterable[Edit]) -> tuple[bytes, list[tuple[Edit, str | None]]]:
    """Apply edits to one part. Returns the new XML and, per edit, None or
    a problem."""
    root = etree.fromstring(xml)
    outcomes: list[tuple[Edit, str | None]] = []
    for edit in edits:
        hits = []
        for para in root.iter(f"{{{NS['a']}}}p"):
            text = "".join(t.text or "" for t in _runs(para))
            for i in _find_all(text, edit.old):
                hits.append((para, i))
        if not hits:
            outcomes.append((edit, "matched nothing"))
            continue
        if len(hits) != edit.count:
            outcomes.append((edit, f"matched {len(hits)} times, expected {edit.count}"))
            continue
        problem = None
        # Right to left within a paragraph, so earlier offsets stay valid.
        for para, i in sorted(hits, key=lambda h: -h[1]):
            problem = _apply_in_paragraph(para, edit, i)
            if problem:
                break
        outcomes.append((edit, problem))
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8",
                          standalone=True), outcomes


# ---------------------------------------------------------------------------
# Whole deck
# ---------------------------------------------------------------------------


def _check_opens(path: Path) -> str | None:
    try:
        from pptx import Presentation  # noqa: WPS433 (optional at import time)
    except ImportError:
        return None if zipfile.is_zipfile(path) else "not a zip file"
    try:
        prs = Presentation(str(path))
        for slide in prs.slides:
            for shape in slide.shapes:
                if shape.has_text_frame:
                    _ = shape.text_frame.text
            if slide.has_notes_slide:
                _ = slide.notes_slide.notes_text_frame
    except Exception as exc:  # noqa: BLE001 - any failure means it will not open
        return f"{type(exc).__name__}: {exc}"
    return None


def apply_edits(deck: Path, edits: list[Edit], out: Path | None = None) -> Result:
    """Apply ``edits`` to ``deck`` and write the result to ``out`` (default:
    ``deck`` itself). Nothing is written unless every edit applied and the
    result reopens."""
    out = out or deck
    result = Result()
    with zipfile.ZipFile(deck) as zf:
        infos = zf.infolist()
        data = {i.filename: zf.read(i.filename) for i in infos}
        by_part: dict[str, list[Edit]] = {}
        for e in edits:
            part = slide_part(e.slide) if e.where == "slide" else notes_part(zf, e.slide)
            if part is None or part not in data:
                result.problems.append(f"{e.label()}: no {e.where} part for that slide")
                continue
            by_part.setdefault(part, []).append(e)

    for part, part_edits in by_part.items():
        new_xml, outcomes = apply_to_xml(data[part], part_edits)
        ok = True
        for e, problem in outcomes:
            if problem:
                result.problems.append(f"{e.label()}: {problem}")
                ok = False
            else:
                result.applied.append(e.label())
                result.changed_slides.add(e.slide)
        if ok:
            data[part] = new_xml

    if result.problems:
        return result

    with tempfile.TemporaryDirectory() as tmp:
        tmp_out = Path(tmp) / "deck.pptx"
        with zipfile.ZipFile(tmp_out, "w") as zout:
            for info in infos:
                zout.writestr(info, data[info.filename])
        problem = _check_opens(tmp_out)
        if problem:
            result.problems.append(f"edited file will not open: {problem}")
            return result
        shutil.copyfile(tmp_out, out)
    return result


def dump(deck: Path, slide: int | None = None) -> str:
    """Every paragraph of every slide (and its notes), for writing edits."""
    lines = []
    with zipfile.ZipFile(deck) as zf:
        n = 1
        while slide_part(n) in zf.namelist():
            if slide is None or slide == n:
                for where, part in (("slide", slide_part(n)), ("notes", notes_part(zf, n))):
                    if not part:
                        continue
                    root = etree.fromstring(zf.read(part))
                    for para in root.iter(f"{{{NS['a']}}}p"):
                        text = "".join(t.text or "" for t in _runs(para))
                        if text.strip():
                            lines.append(f"{n:>3} {where:<5} {text}")
            n += 1
    return "\n".join(lines)


def render(deck: Path, slides: Iterable[int], outdir: Path) -> list[Path]:
    """PNG of each named slide via LibreOffice and pdftoppm. Empty when
    either tool is missing."""
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    pdftoppm = shutil.which("pdftoppm")
    if not soffice or not pdftoppm:
        print("::warning::LibreOffice or pdftoppm not installed; changed slides not rendered")
        return []
    outdir.mkdir(parents=True, exist_ok=True)
    subprocess.run([soffice, "--headless", "--convert-to", "pdf", "--outdir",
                    str(outdir), str(deck)], check=True, capture_output=True, timeout=600)
    pdf = outdir / (deck.stem + ".pdf")
    written = []
    for n in sorted(slides):
        prefix = outdir / f"{deck.stem}_slide{n}"
        subprocess.run([pdftoppm, "-png", "-r", "60", "-f", str(n), "-l", str(n),
                        "-singlefile", str(pdf), str(prefix)], check=True, timeout=120)
        written.append(prefix.with_suffix(".png"))
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("deck", type=Path)
    ap.add_argument("--edits", type=Path, help="JSON list of edits")
    ap.add_argument("--out", type=Path, help="write here instead of over the deck")
    ap.add_argument("--dump", action="store_true", help="print every paragraph")
    ap.add_argument("--slide", type=int, help="with --dump: one slide only")
    ap.add_argument("--render", type=Path, help="write PNGs of changed slides here")
    args = ap.parse_args(argv)

    if args.dump:
        print(dump(args.deck, args.slide))
        return 0
    if not args.edits:
        ap.error("--edits or --dump is required")
    edits = [Edit.from_dict(d) for d in json.loads(args.edits.read_text(encoding="utf-8"))]
    result = apply_edits(args.deck, edits, args.out)
    for line in result.applied:
        print(f"applied  {line}")
    for line in result.problems:
        print(f"PROBLEM  {line}")
    if result.problems:
        print("Nothing was written.")
        return 2 if any("will not open" in p for p in result.problems) else 1
    if args.render:
        for png in render(args.out or args.deck, result.changed_slides, args.render):
            print(f"rendered {png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
