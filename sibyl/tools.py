# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Sibyl agent tools: open-web search and page fetch.

Exactly two tools are exposed to the agent — ``brave_search`` (via the
shared Brave wrapper, with its circuit breaker and rate limiting) and
``fetch_url``. The structured Pythia connectors are deliberately NOT
available: the independence of the two tracks is the point. A disabled,
config-gated extension point exists for authoritative live lookups
(``LIVE_LOOKUPS_ENABLED``, default off).
"""

from __future__ import annotations

import ipaddress
import logging
import socket
import threading
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import urljoin, urlsplit

import requests

from pythia.web_research.backends.brave_search import fetch_via_brave_search
from pythia.web_research.types import EvidenceSource

from sibyl import config as _cfg
from sibyl import reader
from sibyl.config import (
    BRAVE_MAX_RESULTS,
    BRAVE_TIMEOUT_SEC,
    FETCH_URL_MAX_BYTES,
    FETCH_URL_MAX_CHARS,
    FETCH_URL_MAX_REDIRECTS,
    FETCH_URL_TIMEOUT_SEC,
    SEARCH_WINDOW_DAYS,
)
from sibyl.leakage import (
    LeakageStats,
    date_range_freshness,
    filter_sources,
    is_backtest,
    is_blocked_for,
    snippet_leaks,
)

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Uniform result envelope handed back to the agent loop."""

    tool: str
    ok: bool
    text: str  # what the agent sees
    cost_usd: float = 0.0
    sources: List[EvidenceSource] = field(default_factory=list)
    leakage: LeakageStats = field(default_factory=LeakageStats)
    error: Optional[str] = None
    status_code: Optional[int] = None
    # A document read: its full text BEFORE the extraction model saw it.
    doc_text: Optional[str] = None
    url: Optional[str] = None

    @property
    def search_failed(self) -> bool:
        """A search the provider did not answer (not merely an empty one).

        "No results" with HTTP 200 is Brave saying nothing matched; every
        other error (a tripped breaker, a missing key, a non-200) is a failure.
        """
        if self.tool not in ("brave_search", "reliefweb_search") or self.ok:
            return False
        return not (self.error == "no_results" and self.status_code in (None, 200))


class RunCounters:
    """Run-scoped tool counters, thread-safe (trials may run concurrently).

    Reset at the start of every run by ``sibyl.run.run_sibyl`` and written
    to ``sibyl_runs`` at its end.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with getattr(self, "_lock", threading.Lock()):
            self.search_calls = 0
            self.search_failed = 0
            self.breaker_trips = 0
            self.breaker_resets = 0
            self.docs_read = 0

    def add(self, name: str, n: int = 1) -> None:
        with self._lock:
            setattr(self, name, getattr(self, name) + n)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "n_search_calls": self.search_calls,
                "n_search_failed": self.search_failed,
                "n_breaker_trips": self.breaker_trips,
                "n_breaker_resets": self.breaker_resets,
                "n_docs_read": self.docs_read,
            }


COUNTERS = RunCounters()


def reset_run_state() -> None:
    """Start-of-run reset: the shared Brave breaker and Sibyl's counters.

    The breaker is a module singleton shared with the HS grounding code; a
    trip left over from anything earlier in the process would otherwise blind
    every Sibyl search (the July 2026 run lost all 216 searches that way).
    """
    from pythia.web_research import brave_circuit_breaker  # noqa: PLC0415

    brave_circuit_breaker.reset()
    COUNTERS.reset()


def _sleep(seconds: float) -> None:
    """Seam for tests."""
    time.sleep(seconds)


LANES = ("news", "reference")


def lane_window_days(lane: str) -> int:
    """News: the last SEARCH_WINDOW_DAYS; reference: REFERENCE_WINDOW_DAYS."""
    return _cfg.REFERENCE_WINDOW_DAYS if lane == "reference" else SEARCH_WINDOW_DAYS


def _run_brave(query: str, freshness: str, *, window_days: int = SEARCH_WINDOW_DAYS,
               language: Optional[str] = None, country: Optional[str] = None):
    kwargs = {}
    if language:
        kwargs["search_lang"] = language
    if country:
        kwargs["country"] = country
    return fetch_via_brave_search(
        query,
        recency_days=window_days,
        include_structural=False,
        timeout_sec=BRAVE_TIMEOUT_SEC,
        max_results=BRAVE_MAX_RESULTS,
        freshness_override=freshness,
        **kwargs,
    )


def _pack_error_type(pack) -> Optional[str]:
    if pack.error and not pack.sources:
        return (pack.error or {}).get("type", "unknown")
    return None


def _maybe_reset_breaker() -> bool:
    """Wait, reset the tripped breaker and say whether a retry is allowed.

    At most ``BREAKER_MAX_RESETS`` resets a run, counted under the counter
    lock so concurrent trials cannot overspend it.
    """
    from pythia.web_research import brave_circuit_breaker  # noqa: PLC0415

    with COUNTERS._lock:
        if COUNTERS.breaker_resets >= _cfg.BREAKER_MAX_RESETS:
            return False
        COUNTERS.breaker_resets += 1
    logger.warning(
        "sibyl.tools: Brave circuit breaker tripped; waiting %.0fs, resetting "
        "and retrying once (%d of %d resets this run)",
        _cfg.BREAKER_COOLDOWN_SEC, COUNTERS.breaker_resets, _cfg.BREAKER_MAX_RESETS,
    )
    _sleep(_cfg.BREAKER_COOLDOWN_SEC)
    brave_circuit_breaker.reset()
    return True


def brave_search(
    query: str,
    as_of: date,
    *,
    today: Optional[date] = None,
    lane: str = "news",
    language: Optional[str] = None,
    country: Optional[str] = None,
) -> ToolResult:
    """Date-filtered web search through the shared Brave wrapper.

    Two lanes (Oct 2026): ``news`` (the last SEARCH_WINDOW_DAYS) and
    ``reference`` (ten years back), both ending at *as_of* (harmless live, a
    hard cap in backtest); optional language and country hints. Leakage
    post-filtering runs on the results. A tripped circuit breaker is waited
    out, reset and retried once (bounded per run).
    """
    COUNTERS.add("search_calls")
    lane = lane if lane in LANES else "news"
    window = lane_window_days(lane)
    freshness = date_range_freshness(as_of, window)
    hints = dict(window_days=window, language=language, country=country)
    pack = _run_brave(query, freshness, **hints)
    if _pack_error_type(pack) == "circuit_breaker_tripped":
        COUNTERS.add("breaker_trips")
        if _maybe_reset_breaker():
            pack = _run_brave(query, freshness, **hints)
            if _pack_error_type(pack) == "circuit_breaker_tripped":
                COUNTERS.add("breaker_trips")
    cost = 0.0
    try:
        cost = float((pack.debug.get("usage") or {}).get("cost_usd", 0.0))
    except (TypeError, ValueError):
        cost = 0.0
    status: Optional[int] = None
    try:
        raw_status = (pack.debug or {}).get("status_code")
        status = int(raw_status) if raw_status not in (None, "", 0) else None
    except (TypeError, ValueError):
        status = None

    err_type = _pack_error_type(pack)
    if err_type:
        status_txt = f" HTTP {status}" if status and status != 200 else ""
        result = ToolResult(
            tool="brave_search",
            ok=False,
            text=f"[search failed: {err_type}{status_txt}] No results for query: {query}",
            cost_usd=cost,
            error=err_type,
            status_code=status,
        )
        if result.search_failed:
            COUNTERS.add("search_failed")
        return result

    kept, stats = filter_sources(pack.sources, as_of, today=today)
    if not kept:
        return ToolResult(
            tool="brave_search",
            ok=True,
            text=(
                "No usable results (all results were filtered out by the "
                f"as-of/leakage controls). Query: {query}"
            ),
            cost_usd=cost,
            leakage=stats,
        )

    lines = [f"Search results ({lane} lane) for: {query} (window ending {as_of.isoformat()})"]
    for i, src in enumerate(kept, start=1):
        date_str = f" [{src.date}]" if src.date else ""
        summary = (src.summary or "").strip()
        lines.append(f"{i}. {src.title}{date_str}\n   {src.url}\n   {summary}")
    return ToolResult(
        tool="brave_search",
        ok=True,
        text="\n".join(lines),
        cost_usd=cost,
        sources=kept,
        leakage=stats,
        status_code=status,
    )


def _html_to_text(html: str) -> str:
    """Extract readable text from HTML via BeautifulSoup (repo-standard)."""
    from bs4 import BeautifulSoup  # deferred: not needed for search-only runs

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)


class UnsafeURL(ValueError):
    """The URL points somewhere a research fetch must never go."""


def _resolve(host: str) -> List[str]:
    """Every address ``host`` resolves to (seam for tests)."""

    infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    return sorted({info[4][0] for info in infos})


def check_public_url(url: str) -> None:
    """Raise ``UnsafeURL`` unless ``url`` is http(s) to public addresses only.

    The model chooses what Sibyl fetches, and a prompt can be steered by the
    pages it read. Without this check a fetch could reach the runner's own
    services, a cloud metadata endpoint (169.254.169.254) or anything else on
    a private network. Every address the host resolves to must be global; a
    name with one private answer among public ones is refused.
    """

    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise UnsafeURL(f"scheme {parts.scheme or '(none)'!r} is not http(s)")
    host = parts.hostname
    if not host:
        raise UnsafeURL("no host")
    try:
        literal = ipaddress.ip_address(host)
        addresses = [str(literal)]
    except ValueError:
        try:
            addresses = _resolve(host)
        except OSError as exc:
            raise UnsafeURL(f"host does not resolve ({type(exc).__name__})") from exc
    if not addresses:
        raise UnsafeURL("host resolves to nothing")
    for addr in addresses:
        ip = ipaddress.ip_address(addr.split("%", 1)[0])
        # ::ffff:127.0.0.1 is 127.0.0.1; judge the IPv4 address it carries,
        # since some Python releases call the mapped form global.
        if ip.version == 6 and ip.ipv4_mapped is not None:
            ip = ip.ipv4_mapped
        if not ip.is_global or ip.is_multicast:
            raise UnsafeURL(f"host resolves to a non-public address ({ip})")


def _guarded_get(url: str) -> Tuple[requests.Response, bytes, str]:
    """GET ``url``, re-checking every redirect hop and capping the body.

    Returns the final response, at most ``FETCH_URL_MAX_BYTES`` of its body
    (``FETCH_PDF_MAX_BYTES`` for a PDF), and the URL it came from. Redirects
    are followed by hand, because ``requests`` would follow one to a private
    address without asking.
    """

    current = url
    for _ in range(FETCH_URL_MAX_REDIRECTS + 1):
        check_public_url(current)
        resp = requests.get(
            current,
            timeout=FETCH_URL_TIMEOUT_SEC,
            headers={"User-Agent": "Mozilla/5.0 (compatible; PythiaSibyl/1.0)"},
            allow_redirects=False,
            stream=True,
        )
        if resp.status_code in (301, 302, 303, 307, 308) and resp.headers.get("Location"):
            nxt = urljoin(current, resp.headers["Location"])
            resp.close()
            current = nxt
            continue
        body = bytearray()
        ctype = (resp.headers.get("Content-Type") or "").lower()
        cap = (
            max(FETCH_URL_MAX_BYTES, _cfg.FETCH_PDF_MAX_BYTES)
            if ("pdf" in ctype or current.lower().split("?", 1)[0].endswith(".pdf"))
            else FETCH_URL_MAX_BYTES
        )
        try:
            for chunk in resp.iter_content(chunk_size=65536):
                if not chunk:
                    continue
                body.extend(chunk)
                if len(body) >= cap:
                    del body[cap:]
                    break
        finally:
            resp.close()
        return resp, bytes(body), current
    raise UnsafeURL(f"more than {FETCH_URL_MAX_REDIRECTS} redirects")


def _decode(resp: requests.Response, body: bytes) -> str:
    """The body as text: the charset the server named, else utf-8."""

    content_type = resp.headers.get("Content-Type") or ""
    encoding = "utf-8"
    if "charset=" in content_type.lower():
        encoding = content_type.lower().split("charset=", 1)[1].split(";", 1)[0].strip(" \"'") or "utf-8"
    try:
        return body.decode(encoding, errors="replace")
    except LookupError:
        return body.decode("utf-8", errors="replace")


def fetch_url(
    url: str,
    as_of: date,
    *,
    today: Optional[date] = None,
    terms: Sequence[str] = (),
) -> ToolResult:
    """Fetch a page the agent found via search and return readable text.

    Resolution-source URLs are refused in backtest only (live runs may read
    them, owner decision Oct 2026); in backtest the extracted text also goes
    through the snippet leak classifier before being returned.
    """
    stats = LeakageStats(total_retrieved=1)
    if _is_reliefweb_report(url):
        # reliefweb.int answers page fetches with HTTP 202; read via the API.
        return reliefweb_report(url, as_of, today=today, terms=terms)
    if is_blocked_for(url, as_of, today=today):
        stats.dropped_blocked_domain = 1
        return ToolResult(
            tool="fetch_url",
            ok=False,
            text=(
                "This URL belongs to a resolution data source and is blocked "
                "for Sibyl in backtest mode. Rely on open-web reporting instead."
            ),
            leakage=stats,
            error="blocked_domain",
        )

    try:
        resp, body, final_url = _guarded_get(url)
    except UnsafeURL as exc:
        logger.warning("sibyl.fetch_url: refused %s: %s", url, exc)
        return ToolResult(
            tool="fetch_url", ok=False,
            text=f"[fetch refused: {exc}] {url}",
            leakage=stats, error="unsafe_url",
        )
    except requests.RequestException as exc:
        return ToolResult(
            tool="fetch_url", ok=False,
            text=f"[fetch failed: {type(exc).__name__}] {url}",
            leakage=stats, error=type(exc).__name__,
        )
    if resp.status_code != 200:
        return ToolResult(
            tool="fetch_url", ok=False,
            text=f"[fetch failed: HTTP {resp.status_code}] {url}",
            leakage=stats, error=f"http_{resp.status_code}",
            status_code=resp.status_code,
        )

    # A redirect may have landed on a resolution source the first URL hid.
    if final_url != url and is_blocked_for(final_url, as_of, today=today):
        stats.dropped_blocked_domain = 1
        return ToolResult(
            tool="fetch_url", ok=False,
            text=(
                "This URL redirects to a resolution data source and is blocked "
                "for Sibyl in backtest mode. Rely on open-web reporting instead."
            ),
            leakage=stats, error="blocked_domain",
        )

    content_type = (resp.headers.get("Content-Type") or "").lower()
    try:
        text = reader.document_text(
            body, content_type=content_type, url=final_url,
            text=None if reader.is_pdf(body, content_type, final_url) else _decode(resp, body),
            terms=terms,
        )
    except ValueError as exc:
        return ToolResult(
            tool="fetch_url", ok=False,
            text=f"[fetch returned an unreadable document: {exc}] {url}",
            leakage=stats, error="unreadable",
        )
    text = (text or "").strip()
    if not text:
        return ToolResult(
            tool="fetch_url", ok=False,
            text=f"[fetch returned no readable text] {url}",
            leakage=stats, error="empty",
        )

    # Scan the SAME text that is returned to the model — a post-asOf date
    # after an arbitrary scan cutoff must not slip through to the prompt.
    if is_backtest(as_of, today=today) and snippet_leaks(text, as_of):
        stats.dropped_post_asof = 1
        stats.notes.append(f"fetched page dated after asOf dropped: {url}")
        logger.info("sibyl.leakage: dropped fetched page post-asOf: %s", url)
        return ToolResult(
            tool="fetch_url", ok=False,
            text=(
                "The fetched page contains material dated after the "
                f"forecast as-of date ({as_of.isoformat()}) and was withheld "
                "by the backtest leakage controls."
            ),
            leakage=stats, error="post_asof",
        )

    COUNTERS.add("docs_read")
    return ToolResult(
        tool="fetch_url", ok=True,
        text=f"Content of {url}:\n{text}",
        leakage=stats, doc_text=text, url=url,
    )


# --- ReliefWeb through its API (Oct 2026) -------------------------------------
#
# reliefweb.int answered all 45 page fetches of the Sept 2026 runs with HTTP
# 202. The API is open (the resolution machine already uses it, with the same
# RELIEFWEB_APPNAME). Without the name, or when the API fails, the tool says
# so and the agent carries on with Brave. In backtest mode only reports
# created on or before the as-of date are returned.

_RW_IDS: Dict[str, int] = {}
_RW_LOCK = threading.Lock()


def _is_reliefweb_report(url: str) -> bool:
    try:
        parts = urlsplit(url)
    except ValueError:
        return False
    host = (parts.hostname or "").lower()
    return (host == "reliefweb.int" or host.endswith(".reliefweb.int")) and parts.path.startswith(
        ("/report/", "/node/")
    )


def _rw_post(payload: dict) -> dict:
    """POST to the ReliefWeb reports endpoint (seam for tests)."""
    import os  # noqa: PLC0415

    from resolver.hazard_resolution.reliefweb_sweep import default_post  # noqa: PLC0415

    name = os.getenv("RELIEFWEB_APPNAME", "").strip()
    if not name:
        raise RuntimeError("RELIEFWEB_APPNAME is not set")
    url = _cfg.RELIEFWEB_API_BASE.rstrip("/") + "/reports"
    return default_post(url, payload, {"appname": name}, _cfg.RELIEFWEB_TIMEOUT_SEC)


def _rw_date_filter(as_of: date, today: Optional[date]) -> Optional[dict]:
    if not is_backtest(as_of, today=today):
        return None
    return {"field": "date.created", "value": {"to": f"{as_of.isoformat()}T23:59:59+00:00"}}


def reliefweb_search(
    query: str,
    as_of: date,
    *,
    today: Optional[date] = None,
    country_iso3: Optional[str] = None,
) -> ToolResult:
    """Search ReliefWeb reports: title, source, original date, format, URL."""
    COUNTERS.add("search_calls")
    conditions = []
    if country_iso3:
        conditions.append({"field": "primary_country.iso3", "value": country_iso3.upper()})
    date_cond = _rw_date_filter(as_of, today)
    if date_cond:
        conditions.append(date_cond)
    payload: Dict[str, object] = {
        "query": {"value": query},
        "fields": {"include": ["id", "title", "url", "url_alias", "source.shortname",
                               "date.original", "date.created", "format.name"]},
        "sort": ["date.created:desc"],
        "limit": _cfg.RELIEFWEB_MAX_RESULTS,
    }
    if conditions:
        payload["filter"] = {"operator": "AND", "conditions": conditions}
    try:
        data = _rw_post(payload)
    except Exception as exc:  # noqa: BLE001 - the agent carries on with Brave
        COUNTERS.add("search_failed")
        return ToolResult(
            tool="reliefweb_search", ok=False,
            text=f"[ReliefWeb search failed: {exc}] Use brave_search instead. Query: {query}",
            error="reliefweb_unavailable",
        )
    items = data.get("data") or []
    lines = [f"ReliefWeb reports for: {query} (created on or before {as_of.isoformat()})"]
    sources: List[EvidenceSource] = []
    for i, item in enumerate(items, start=1):
        f = item.get("fields") or {}
        url = f.get("url_alias") or f.get("url") or ""
        rid = item.get("id") or f.get("id")
        if url and rid:
            with _RW_LOCK:
                _RW_IDS[url] = int(rid)
                if f.get("url"):
                    _RW_IDS[f["url"]] = int(rid)
        src = ", ".join(s.get("shortname", "") for s in (f.get("source") or []) if isinstance(s, dict))
        fmt = ", ".join(x.get("name", "") for x in (f.get("format") or []) if isinstance(x, dict))
        when = (f.get("date") or {}).get("original") or (f.get("date") or {}).get("created") or ""
        lines.append(f"{i}. {f.get('title', '')} [{str(when)[:10]}] ({src}; {fmt})\n   {url}")
        sources.append(EvidenceSource(title=f.get("title", ""), url=url, publisher=src,
                                      date=str(when)[:10] or None, summary=fmt))
    if not sources:
        return ToolResult(tool="reliefweb_search", ok=False, text=lines[0] + "\n(no reports)",
                          error="no_results", status_code=200)
    return ToolResult(tool="reliefweb_search", ok=True, text="\n".join(lines), sources=sources)


def reliefweb_report(
    url: str,
    as_of: date,
    *,
    today: Optional[date] = None,
    terms: Sequence[str] = (),
) -> ToolResult:
    """Read one ReliefWeb report through the API, with its PDF attachment."""
    stats = LeakageStats(total_retrieved=1)
    with _RW_LOCK:
        rid = _RW_IDS.get(url)
    cond = {"field": "id", "value": rid} if rid else {"field": "url_alias", "value": url}
    conditions = [cond]
    date_cond = _rw_date_filter(as_of, today)
    if date_cond:
        conditions.append(date_cond)
    payload = {
        "filter": {"operator": "AND", "conditions": conditions},
        "fields": {"include": ["id", "title", "body", "url", "date.original",
                               "source.shortname", "file"]},
        "limit": 1,
    }
    try:
        data = _rw_post(payload)
    except Exception as exc:  # noqa: BLE001
        return ToolResult(tool="fetch_url", ok=False,
                          text=f"[ReliefWeb read failed: {exc}] {url}",
                          leakage=stats, error="reliefweb_unavailable")
    items = data.get("data") or []
    if not items:
        return ToolResult(tool="fetch_url", ok=False,
                          text=f"[ReliefWeb has no such report on or before {as_of.isoformat()}] {url}",
                          leakage=stats, error="not_found")
    f = items[0].get("fields") or {}
    parts = [str(f.get("title") or ""), str(f.get("body") or "")]
    for att in f.get("file") or []:
        if not isinstance(att, dict) or "pdf" not in str(att.get("mimetype", "")).lower():
            continue
        try:
            resp, body, final_url = _guarded_get(att.get("url", ""))
            if resp.status_code == 200:
                parts.append(reader.pdf_text(body, terms))
                break
        except (UnsafeURL, requests.RequestException, ValueError) as exc:
            parts.append(f"[attachment not read: {type(exc).__name__}]")
    text = "\n\n".join(p for p in parts if p.strip())[: _cfg.DOC_MAX_CHARS]
    if not text.strip():
        return ToolResult(tool="fetch_url", ok=False, text=f"[empty report] {url}",
                          leakage=stats, error="empty")
    COUNTERS.add("docs_read")
    return ToolResult(tool="fetch_url", ok=True, text=f"Content of {url} (via the ReliefWeb API):\n{text}",
                      leakage=stats, doc_text=text, url=url)
