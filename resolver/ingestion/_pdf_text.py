"""Utilities for extracting text from ReliefWeb PDF resources.

The real ReliefWeb connector relies on a combination of native PDF text
extraction and OCR fallbacks.  The implementation below keeps the production
behaviour lightweight while remaining extremely easy to monkeypatch inside the
unit test-suite.  The helpers expose a common API that mirrors the production
connector, but they avoid hard dependencies on heavy third-party libraries when
they are not available in the runtime environment.

The functions are intentionally small, feature measurable logging hooks and are
written in a way that allows tests to inject fake implementations.  When OCR is
not available, the helpers degrade gracefully by returning an empty string so
that callers can decide how to react.
"""

from __future__ import annotations

import hashlib
import io
import logging
import time
from typing import Callable, Dict, List, Optional, Tuple


LOGGER = logging.getLogger(__name__)


# Global switches used by the test-suite.  The connector imports this module
# directly which makes these variables trivial to monkeypatch using
# ``monkeypatch.setattr``.
PDF_TEXT_TEST_MODE = False
_NATIVE_EXTRACTOR: Optional[Callable[[bytes], str]] = None
_OCR_EXTRACTOR: Optional[
    Callable[[bytes, Optional[List[str]], Optional[List[int]]], str]
] = None


def _default_native_extractor(content_bytes: bytes) -> str:
    """Extract text using ``pdfminer.six`` when available."""

    try:  # Deferred import keeps the connector light when pdfminer is absent.
        from pdfminer.high_level import extract_text as pdfminer_extract_text
    except Exception:  # pragma: no cover - optional dependency
        LOGGER.debug("pdfminer not available; returning empty native text")
        return ""

    with io.BytesIO(content_bytes) as fh:
        try:
            text = pdfminer_extract_text(fh)
        except Exception as exc:  # pragma: no cover - defensive branch
            LOGGER.warning("pdfminer extraction failed: %s", exc)
            return ""
    return text or ""


def _default_ocr_extractor(
    content_bytes: bytes,
    languages: Optional[List[str]] = None,
    page_mask: Optional[List[int]] = None,
) -> str:
    """Perform OCR using ``pytesseract`` + ``pdf2image`` when available."""

    try:  # Deferred imports keep optional dependencies truly optional.
        from pdf2image import convert_from_bytes
        import pytesseract
    except Exception:  # pragma: no cover - optional dependency
        LOGGER.debug("OCR libraries not available; returning empty text")
        return ""

    pages = convert_from_bytes(content_bytes)
    if page_mask is not None:
        to_process = [idx for idx in page_mask if 0 <= idx < len(pages)]
    else:
        to_process = list(range(len(pages)))

    if not to_process:
        to_process = list(range(len(pages)))

    texts: List[str] = []
    for page_index in to_process:
        image = pages[page_index]
        try:
            page_text = pytesseract.image_to_string(
                image, lang="+".join(languages) if languages else None
            )
        except Exception as exc:  # pragma: no cover - defensive branch
            LOGGER.warning("OCR failed on page %s: %s", page_index, exc)
            page_text = ""
        texts.append(page_text)
    return "\n".join(texts)


def _get_native_extractor() -> Callable[[bytes], str]:
    return _NATIVE_EXTRACTOR or _default_native_extractor


def _get_ocr_extractor() -> Callable[[bytes, Optional[List[str]], Optional[List[int]]], str]:
    return _OCR_EXTRACTOR or _default_ocr_extractor


def extract_text_from_pdf(content_bytes: bytes) -> str:
    """Return the native text extracted from the PDF."""

    start = time.perf_counter()
    text = _get_native_extractor()(content_bytes)
    LOGGER.debug(
        "pdf.native chars=%s elapsed_ms=%0.2f",
        len(text),
        (time.perf_counter() - start) * 1000,
    )
    return text


def ocr_pdf_bytes(
    content_bytes: bytes,
    languages: Optional[List[str]] = None,
    page_mask: Optional[List[int]] = None,
) -> str:
    """Return OCR text extracted from the PDF."""

    start = time.perf_counter()
    text = _get_ocr_extractor()(content_bytes, languages, page_mask)
    LOGGER.debug(
        "pdf.ocr chars=%s pages=%s elapsed_ms=%0.2f",
        len(text),
        [] if page_mask is None else page_mask,
        (time.perf_counter() - start) * 1000,
    )
    return text


def smart_extract(content_bytes: bytes, min_chars: int = 1500) -> Tuple[str, Dict[str, object]]:
    """Extract text using native parsing with a selective OCR fallback."""

    native_text = extract_text_from_pdf(content_bytes)
    meta: Dict[str, object] = {
        "method": "native",
        "chars": len(native_text),
        "pages_ocrd": [],
        "digest": hashlib.sha256(content_bytes).hexdigest(),
    }
    if len(native_text) >= min_chars or not native_text.strip():
        return native_text, meta

    # OCR fallback.  We only attempt OCR when explicitly enabled outside of
    # tests.  During tests we keep the behaviour deterministic by respecting
    # ``PDF_TEXT_TEST_MODE`` which callers can toggle.
    if PDF_TEXT_TEST_MODE:
        return native_text, meta

    # Re-run native extraction for each page to identify candidates with little
    # text.  The simple heuristic below works well in practice and is cheap.
    try:
        from pdfminer.high_level import extract_text_to_fp
        from pdfminer.pdfpage import PDFPage
    except Exception:  # pragma: no cover - optional dependency
        LOGGER.debug("pdfminer not available for selective OCR; OCRing full doc")
        ocr_text = ocr_pdf_bytes(content_bytes)
        meta.update({"method": "ocr", "chars": len(ocr_text)})
        return ocr_text, meta

    pages_to_ocr: List[int] = []
    page_texts: List[str] = []
    with io.BytesIO(content_bytes) as fh:
        for page_index, page in enumerate(PDFPage.get_pages(fh)):
            # We need a fresh BytesIO for ``extract_text_to_fp`` because the
            # helper consumes from the beginning of the file.  ``PDFPage`` has
            # already moved ``fh`` to the start of the next page so we re-open
            # a dedicated buffer per iteration.
            sub_buffer = io.BytesIO(content_bytes)
            buffer = io.StringIO()
            try:
                extract_text_to_fp(sub_buffer, buffer, page_numbers=[page_index])
            except Exception:  # pragma: no cover - defensive branch
                page_texts.append("")
                pages_to_ocr.append(page_index)
                continue
            text = buffer.getvalue()
            page_texts.append(text)
            if len(text.strip()) < max(50, min_chars // 10):
                pages_to_ocr.append(page_index)

    ocr_text = ocr_pdf_bytes(content_bytes, page_mask=pages_to_ocr or None)
    if not ocr_text:
        return native_text, meta

    def _split_ocr_text(text: str, expected: int) -> Optional[List[str]]:
        """Split concatenated OCR output into page-aligned chunks."""

        if expected <= 1:
            return [text]

        # ``pytesseract`` terminates each page with a ``\f`` character.  We
        # leverage that to recover page boundaries.  When the delimiter is not
        # present (for example when using a custom extractor), we fall back to a
        # single chunk so that the caller can handle the mismatch gracefully.
        chunks = text.rstrip("\f").split("\f")
        if len(chunks) != expected:
            return None
        return chunks

    per_page_ocr: Dict[int, str] = {}
    ocr_chunks = _split_ocr_text(ocr_text, len(pages_to_ocr))
    if ocr_chunks is None:
        # Fallback: we could not confidently split the OCR output.  Re-run OCR
        # for each target page individually to avoid duplicating the combined
        # text while preserving correctness.  ``ocr_pdf_bytes`` already applies
        # logging and any monkeypatched behaviour, so we reuse it here.
        for page_idx in pages_to_ocr:
            per_page_ocr[page_idx] = ocr_pdf_bytes(content_bytes, page_mask=[page_idx])
    else:
        per_page_ocr = dict(zip(pages_to_ocr, ocr_chunks))

    combined = []
    for idx, page_text in enumerate(page_texts):
        combined.append(per_page_ocr.get(idx, page_text))
    merged_text = "\n".join(filter(None, combined)) or ocr_text

    meta.update({"method": "hybrid", "chars": len(merged_text), "pages_ocrd": pages_to_ocr})
    return merged_text, meta


def install_test_extractors(
    native: Optional[Callable[[bytes], str]] = None,
    ocr: Optional[Callable[[bytes, Optional[List[str]], Optional[List[int]]], str]] = None,
) -> None:
    """Install in-memory extractors used exclusively by the tests."""

    global _NATIVE_EXTRACTOR, _OCR_EXTRACTOR
    _NATIVE_EXTRACTOR = native
    _OCR_EXTRACTOR = ocr

