# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Turn a fetched document into text worth reading (Oct 2026).

Until October 2026 ``fetch_url`` stripped a page's tags, kept the first 6,000
characters, and could not read a PDF at all: 43 of the 57 pages Sibyl read
in four runs were Wikipedia, and the situation reports and appeals that carry
the figures are PDFs.

* HTML: the main content (``article``, then ``main``, then the body), with
  navigation, headers, footers, forms and asides dropped and tables rendered
  as ``a | b | c`` rows.
* PDF (pdfplumber, already a dependency): the first two pages plus the pages
  that score highest on the country name and the question's terms, up to
  ``SIBYL_PDF_MAX_PAGES``, in page order with ``[page N]`` markers.
* Every document is capped at ``SIBYL_DOC_MAX_CHARS``.

Pure functions over bytes and text: the guarded fetch stays in
``sibyl.tools``.
"""

from __future__ import annotations

import io
import logging
import re
from typing import Iterable, List, Optional, Sequence

from sibyl import config as _cfg

logger = logging.getLogger(__name__)

_DROP_TAGS = ("script", "style", "noscript", "nav", "header", "footer", "form", "aside",
              "button", "svg", "iframe")


def is_pdf(body: bytes, content_type: str = "", url: str = "") -> bool:
    if body[:5] == b"%PDF-":
        return True
    ct = (content_type or "").lower()
    return "pdf" in ct or (url or "").lower().split("?", 1)[0].endswith(".pdf")


def _table_rows(table) -> List[str]:
    rows: List[str] = []
    for tr in table.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
        cells = [c for c in cells if c]
        if cells:
            rows.append(" | ".join(cells))
    return rows


def html_main_text(html: str) -> str:
    """Readable main-content text of an HTML page, tables as rows."""
    from bs4 import BeautifulSoup  # noqa: PLC0415

    soup = BeautifulSoup(html or "", "html.parser")
    for tag in soup(list(_DROP_TAGS)):
        tag.decompose()
    root = soup.find("article") or soup.find("main") or soup.body or soup
    for table in root.find_all("table"):
        rendered = "\n".join(_table_rows(table))
        table.replace_with(soup.new_string("\n" + rendered + "\n"))
    text = root.get_text(separator="\n")
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines()]
    out = "\n".join(ln for ln in lines if ln)
    return out[: _cfg.DOC_MAX_CHARS]


def _score(text: str, terms: Sequence[str]) -> int:
    low = text.lower()
    return sum(low.count(t.lower()) for t in terms if t)


def pick_pages(page_texts: Sequence[str], terms: Sequence[str], max_pages: int) -> List[int]:
    """Indices to keep: pages 0 and 1, then the best-scoring others."""
    n = len(page_texts)
    if n <= max_pages:
        return list(range(n))
    keep = set(range(min(2, n)))
    scored = sorted(
        (i for i in range(n) if i not in keep),
        key=lambda i: (-_score(page_texts[i], terms), i),
    )
    for i in scored:
        if len(keep) >= max_pages:
            break
        keep.add(i)
    return sorted(keep)


def pdf_text(body: bytes, terms: Iterable[str] = (), max_pages: Optional[int] = None) -> str:
    """Text of the pages worth reading. Raises ValueError when unreadable."""
    import pdfplumber  # noqa: PLC0415

    max_pages = max_pages or _cfg.PDF_MAX_PAGES
    try:
        with pdfplumber.open(io.BytesIO(body)) as pdf:
            texts = [(p.extract_text() or "") for p in pdf.pages]
    except Exception as exc:  # noqa: BLE001 - a broken PDF is a failed read
        raise ValueError(f"unreadable PDF: {type(exc).__name__}") from exc
    keep = pick_pages(texts, list(terms), max_pages)
    parts = [f"[page {i + 1} of {len(texts)}]\n{texts[i].strip()}" for i in keep if texts[i].strip()]
    out = "\n\n".join(parts)
    if not out.strip():
        raise ValueError("PDF has no extractable text (scanned image?)")
    return out[: _cfg.DOC_MAX_CHARS]


def document_text(body: bytes, *, content_type: str = "", url: str = "",
                  text: Optional[str] = None, terms: Iterable[str] = ()) -> str:
    """Dispatch on the bytes: PDF, HTML, or plain text."""
    if is_pdf(body, content_type, url):
        return pdf_text(body, terms)
    raw = text if text is not None else body.decode("utf-8", errors="replace")
    if "html" in (content_type or "").lower() or raw.lstrip()[:1] == "<":
        return html_main_text(raw)
    return raw.strip()[: _cfg.DOC_MAX_CHARS]
