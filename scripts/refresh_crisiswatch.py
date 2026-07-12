# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""ICG CrisisWatch scraper (two acquisition paths, one parser).

Fetches the current CrisisWatch edition from
https://www.crisisgroup.org/crisiswatch and parses the HTML with
BeautifulSoup.  Two acquisition paths:

- ``--source live`` (default): headless Chromium via Playwright against
  the live site.  Works locally (system Chrome passes Cloudflare); CI
  datacenter IPs are blocked by Cloudflare.
- ``--source wayback`` (used in CI): the Internet Archive's copy of the
  page.  The page is fully server-rendered, so a Wayback snapshot
  contains all country entries.  Flow: best-effort Save Page Now capture
  request -> CDX snapshot listing (newest first) -> download raw
  ``id_`` HTML (gzip-sniffed) -> validate (size + entry count, rejecting
  Cloudflare-challenged captures) -> parse.  No browser needed.

The /crisiswatch/print endpoint is BROKEN (serves stale October 2019
data).  The main page renders the current edition correctly — a single
long page with all country entries in the DOM.

Output is written to ``horizon_scanner/data/crisiswatch_latest.json`` in
a format compatible with ``crisiswatch._load_fallback_json()``.

Exit codes: 0 = wrote a newer edition, or skipped because no newer
edition exists yet (``--only-if-newer`` no-op); 1 = fetch/parse/
validation failure, zero ISO3 resolution, or the ``--max-age-days``
staleness alarm tripped.

Usage::

    # Local (live site; install Playwright first)
    pip install playwright && playwright install chromium
    python -m scripts.refresh_crisiswatch [--output PATH] [--verbose]

    # CI (Wayback Machine; only needs requests + beautifulsoup4 + PyYAML)
    python -m scripts.refresh_crisiswatch --source wayback \\
        --only-if-newer --max-age-days 60 --verbose
"""

from __future__ import annotations

import argparse
import calendar
import gzip
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

from bs4 import BeautifulSoup, Tag

# ---------------------------------------------------------------------------
# Country name → ISO3 mapping (imported from crisiswatch module)
# ---------------------------------------------------------------------------

try:
    from horizon_scanner.crisiswatch import _ICG_COUNTRY_ISO3, _resolve_iso3
except ImportError:
    # Fallback: if the import fails (e.g. running outside the project root),
    # provide a stub.  The script will still parse, but ISO3 resolution will
    # be limited.
    _ICG_COUNTRY_ISO3: dict[str, str] = {}  # type: ignore[no-redef]

    def _resolve_iso3(name: str) -> str | None:  # type: ignore[misc]
        return None


log = logging.getLogger(__name__)

_CRISISWATCH_URL = "https://www.crisisgroup.org/crisiswatch"

# Realistic Chrome user-agent (must match a real browser to pass CF).
_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

_DEFAULT_OUTPUT = Path("horizon_scanner/data/crisiswatch_latest.json")

# Minimum HTML length to consider a successful page load.
# Cloudflare challenge pages are typically ~6-8 KB; the real CrisisWatch
# page is ~400-600 KB.
_MIN_HTML_LENGTH = 20_000

# Max retry attempts when Cloudflare blocks the page.
_MAX_ATTEMPTS = 3

# Seconds to wait between retry attempts (allows CF challenge cookie to
# settle in the reused browser context).
_RETRY_DELAY_SEC = 15

# ---- Wayback Machine (--source wayback) constants ----

_WAYBACK_CDX_URL = "https://web.archive.org/cdx/search/cdx"
# The `id_` suffix returns the raw archived HTML without Wayback's
# rewriting (original URLs, no toolbar chrome).
_WAYBACK_SNAPSHOT_URL = (
    "https://web.archive.org/web/{timestamp}id_/" + _CRISISWATCH_URL
)
_WAYBACK_SPN_URL = "https://web.archive.org/save/" + _CRISISWATCH_URL
_WAYBACK_SPN_API_URL = "https://web.archive.org/save"

# Identify ourselves politely to archive.org (generic UAs are throttled
# harder).
_WAYBACK_USER_AGENT = (
    "PythiaCrisisWatchRefresh/1.0 (+https://github.com/kwyjad/Pythia)"
)

# The real page has ~70-85 c-crisiswatch-entry divs; Cloudflare challenge
# captures have 0.
_MIN_ENTRY_COUNT = 20

# How far back to walk the CDX index for a valid snapshot.
_DEFAULT_LOOKBACK_DAYS = 120

# Save Page Now: seconds between CDX polls while waiting for the fresh
# capture to be indexed.
_SPN_POLL_INTERVAL_SEC = 30

# SVG xlink:href value → (arrow, alert_type) mapping.
# The CrisisWatch page encodes status via SVG <use xlink:href="#...">
# icons inside the <h3> heading of each country entry.
_SVG_STATUS_MAP: dict[str, tuple[str, str]] = {
    "#deteriorated": ("deteriorated", ""),
    "#improved": ("improved", ""),
    "#unchanged": ("unchanged", ""),
    "#risk-alert": ("", "conflict_risk"),
    "#resolution": ("", "resolution_opportunity"),
}

# Regional CrisisWatch entries that map to multiple countries.
# Each entry is expanded into one output row per ISO3 code.
_REGIONAL_ENTRY_MAP: dict[str, list[tuple[str, str]]] = {
    "Amazon": [
        ("Brazil", "BRA"),
        ("Ecuador", "ECU"),
        ("Colombia", "COL"),
    ],
    "Nile Waters": [
        ("Ethiopia", "ETH"),
        ("Sudan", "SDN"),
        ("Egypt", "EGY"),
    ],
    "Korean Peninsula": [
        ("South Korea", "KOR"),
        ("North Korea", "PRK"),
    ],
}


# ---------------------------------------------------------------------------
# Playwright page fetch (with Cloudflare bypass)
# ---------------------------------------------------------------------------


def _fetch_page_html(
    *,
    timeout_sec: int = 90,
    verbose: bool = False,
    channel: str | None = None,
) -> str:
    """Launch headless Chromium, load CrisisWatch, and return page HTML.

    When *channel* is provided (e.g. ``"chrome"`` or ``"msedge"``), the
    system browser is used directly — no stealth patches or retry loop,
    since system Chrome passes Cloudflare on the first try.

    When *channel* is ``None`` (default, for GitHub Actions), uses
    ``playwright-stealth`` (or manual init-script fallback) and retries
    up to ``_MAX_ATTEMPTS`` times against the Cloudflare JS challenge.

    Scrolls to the bottom repeatedly to trigger any lazy-loaded content.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        log.error(
            "Playwright is not installed.  "
            "Run: pip install playwright && playwright install chromium"
        )
        sys.exit(1)

    debug_dir = Path("horizon_scanner/data")
    debug_dir.mkdir(parents=True, exist_ok=True)

    # ---- System Chrome path (--channel): no stealth, single attempt ----
    if channel is not None:
        log.info("Using system browser channel=%s (no stealth)", channel)
        return _fetch_with_system_chrome(
            channel=channel,
            timeout_sec=timeout_sec,
            verbose=verbose,
            debug_dir=debug_dir,
        )

    # ---- Bundled Chromium path (GitHub Actions): stealth + retries ----
    try:
        from playwright_stealth import stealth_sync
        _has_stealth = True
    except (ImportError, Exception) as exc:
        log.warning(
            "playwright-stealth not available (%s); using manual stealth patches",
            exc,
        )
        stealth_sync = None  # type: ignore[assignment]
        _has_stealth = False

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-web-security",
            ],
        )
        context = browser.new_context(
            user_agent=_USER_AGENT,
            viewport={"width": 1920, "height": 1080},
            locale="en-US",
            timezone_id="America/New_York",
            color_scheme="light",
        )

        html = ""

        for attempt in range(1, _MAX_ATTEMPTS + 1):
            # Reuse the same context across retries so the Cloudflare
            # challenge cookie persists.
            page = context.new_page()

            # Apply stealth patches BEFORE any navigation.
            if _has_stealth and stealth_sync is not None:
                stealth_sync(page)
            else:
                # Manual stealth: hide automation signals that Cloudflare
                # checks for.
                page.add_init_script("""
                    Object.defineProperty(navigator, 'webdriver', {
                        get: () => false,
                    });
                    Object.defineProperty(navigator, 'languages', {
                        get: () => ['en-US', 'en'],
                    });
                    Object.defineProperty(navigator, 'plugins', {
                        get: () => [1, 2, 3, 4, 5],
                    });
                    window.chrome = { runtime: {} };
                    const originalQuery = window.navigator.permissions.query;
                    window.navigator.permissions.query = (parameters) => (
                        parameters.name === 'notifications' ?
                        Promise.resolve({ state: Notification.permission }) :
                        originalQuery(parameters)
                    );
                """)

            log.info(
                "Attempt %d/%d: navigating to %s ...",
                attempt, _MAX_ATTEMPTS, _CRISISWATCH_URL,
            )

            try:
                page.goto(
                    _CRISISWATCH_URL,
                    wait_until="domcontentloaded",
                    timeout=timeout_sec * 1000,
                )
            except Exception as exc:
                log.warning("Navigation failed on attempt %d: %s", attempt, exc)
                page.close()
                if attempt < _MAX_ATTEMPTS:
                    log.info("Waiting %ds before retry ...", _RETRY_DELAY_SEC)
                    time.sleep(_RETRY_DELAY_SEC)
                continue

            # Wait for Cloudflare's JS challenge to resolve.
            # CF typically takes 3-8 seconds to verify the browser.
            # Try multiple selectors — whichever appears first means CF
            # passed and real content is loading.
            cf_passed = False
            try:
                page.wait_for_selector(
                    "text=Conflict Risk Alert", timeout=30_000,
                )
                cf_passed = True
                log.info("Cloudflare challenge passed, content detected")
            except Exception:
                # Fallback: try any <h3> (country entry headings).
                try:
                    page.wait_for_selector("h3", timeout=15_000)
                    cf_passed = True
                    log.info("Cloudflare challenge passed (<h3> detected)")
                except Exception:
                    log.warning(
                        "Content selectors not found after 45s on attempt %d",
                        attempt,
                    )

            if cf_passed:
                _scroll_to_bottom(page)

            # Extract HTML and page title.
            html = page.content()
            title = page.title()
            log.info(
                "Attempt %d: page title='%s', HTML length=%d chars",
                attempt, title, len(html),
            )

            if verbose or len(html) < _MIN_HTML_LENGTH:
                screenshot_path = debug_dir / "crisiswatch_debug.png"
                try:
                    page.screenshot(
                        path=str(screenshot_path), full_page=True,
                    )
                    log.info("Debug screenshot saved to %s", screenshot_path)
                except Exception as exc:
                    log.debug("Screenshot failed: %s", exc)

            page.close()

            if len(html) >= _MIN_HTML_LENGTH:
                log.info("Success on attempt %d", attempt)
                break

            # Cloudflare blocked — HTML is too small.
            log.warning(
                "Cloudflare block detected on attempt %d "
                "(HTML=%d chars < %d threshold)",
                attempt, len(html), _MIN_HTML_LENGTH,
            )
            if verbose:
                debug_html_path = (
                    debug_dir / f"crisiswatch_cf_block_attempt{attempt}.html"
                )
                debug_html_path.write_text(html, encoding="utf-8")
                log.info("Blocked HTML saved to %s", debug_html_path)

            if attempt < _MAX_ATTEMPTS:
                log.info("Waiting %ds before retry ...", _RETRY_DELAY_SEC)
                time.sleep(_RETRY_DELAY_SEC)

        browser.close()

    # Final check after all attempts.
    if len(html) < _MIN_HTML_LENGTH:
        log.error(
            "Cloudflare blocked all %d attempts (final HTML=%d chars).  "
            "The page structure or Cloudflare rules may have changed.  "
            "Consider running locally with --channel chrome.",
            _MAX_ATTEMPTS, len(html),
        )
        # Save the final blocked HTML for debugging.
        debug_html_path = debug_dir / "crisiswatch_debug.html"
        debug_html_path.write_text(html, encoding="utf-8")
        log.info("Final blocked HTML saved to %s", debug_html_path)
        sys.exit(1)

    log.info("Extracted HTML: %d chars", len(html))
    return html


def _scroll_to_bottom(page: Any) -> None:
    """Scroll to page bottom repeatedly to trigger lazy-loaded content."""
    # Wait for network to settle before scrolling.
    try:
        page.wait_for_load_state("networkidle", timeout=30_000)
    except Exception:
        log.debug("networkidle timed out, proceeding anyway")

    prev_height = 0
    scroll_rounds = 0
    max_rounds = 30
    while scroll_rounds < max_rounds:
        height = page.evaluate("document.body.scrollHeight")
        if height == prev_height:
            break
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        page.wait_for_timeout(2500)
        prev_height = height
        scroll_rounds += 1

    log.info("Scroll complete after %d rounds.", scroll_rounds)


def _fetch_with_system_chrome(
    *,
    channel: str,
    timeout_sec: int,
    verbose: bool,
    debug_dir: Path,
) -> str:
    """Fetch CrisisWatch using a system browser (Chrome/Edge).

    System browsers pass Cloudflare without stealth patches, so this
    path uses a single attempt with no retry loop.
    """
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, channel=channel)
        page = browser.new_page()

        log.info("Navigating to %s via %s ...", _CRISISWATCH_URL, channel)
        page.goto(
            _CRISISWATCH_URL,
            wait_until="domcontentloaded",
            timeout=timeout_sec * 1000,
        )

        # Wait for real content selectors.
        try:
            page.wait_for_selector(
                "text=Conflict Risk Alert", timeout=30_000,
            )
            log.info("Content detected (Conflict Risk Alert)")
        except Exception:
            try:
                page.wait_for_selector("h3", timeout=15_000)
                log.info("Content detected (<h3>)")
            except Exception:
                log.warning("Content selectors not found after 45s")

        _scroll_to_bottom(page)

        html = page.content()
        title = page.title()
        log.info(
            "Page title='%s', HTML length=%d chars", title, len(html),
        )

        if verbose:
            screenshot_path = debug_dir / "crisiswatch_debug.png"
            try:
                # Viewport-only screenshot (full_page crashes Chrome on
                # large pages like CrisisWatch ~3 MB HTML).
                page.screenshot(path=str(screenshot_path), full_page=False)
                log.info("Debug screenshot saved to %s", screenshot_path)
            except Exception as exc:
                log.debug("Screenshot failed: %s", exc)

        browser.close()

    if len(html) < _MIN_HTML_LENGTH:
        log.error(
            "Page too small (%d chars < %d threshold).  "
            "The page structure may have changed.",
            len(html), _MIN_HTML_LENGTH,
        )
        debug_html_path = debug_dir / "crisiswatch_debug.html"
        debug_html_path.write_text(html, encoding="utf-8")
        sys.exit(1)

    log.info("Extracted HTML: %d chars", len(html))
    return html


# ---------------------------------------------------------------------------
# Wayback Machine fetch (--source wayback; no browser needed)
# ---------------------------------------------------------------------------


def _get_with_retries(
    url: str,
    *,
    params: dict[str, str] | None = None,
    timeout_sec: int = 60,
    max_attempts: int = 3,
    backoff_sec: int = 15,
) -> requests.Response | None:
    """GET with a polite UA and retries on connection errors/429/5xx.

    Returns ``None`` when all attempts fail — callers decide fatality.
    """
    headers = {"User-Agent": _WAYBACK_USER_AGENT}
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.get(
                url, params=params, headers=headers, timeout=timeout_sec,
            )
        except requests.RequestException as exc:
            log.warning(
                "GET %s failed on attempt %d/%d: %s",
                url, attempt, max_attempts, exc,
            )
            resp = None
        if resp is not None:
            if resp.status_code < 400:
                return resp
            if resp.status_code not in (429,) and resp.status_code < 500:
                # Non-retryable client error (403, 404, ...).
                log.warning(
                    "GET %s returned HTTP %d (not retrying)",
                    url, resp.status_code,
                )
                return None
            log.warning(
                "GET %s returned HTTP %d on attempt %d/%d",
                url, resp.status_code, attempt, max_attempts,
            )
        if attempt < max_attempts:
            delay = backoff_sec
            if resp is not None and resp.headers.get("Retry-After"):
                try:
                    delay = min(120, int(resp.headers["Retry-After"]))
                except ValueError:
                    pass
            time.sleep(delay)
    return None


def _maybe_gunzip(raw: bytes) -> bytes:
    """Decompress gzip payloads (Wayback serves stored gzip bytes raw).

    The ``id_`` snapshot endpoint returns the stored gzip payload without
    a ``Content-Encoding`` header, so ``requests`` does not decompress
    it.  Sniff the magic bytes; a bounded loop covers accidental
    double-compression.
    """
    for _ in range(3):
        if raw[:2] != b"\x1f\x8b":
            break
        try:
            raw = gzip.decompress(raw)
        except OSError:
            # Corrupt gzip — return as-is; validation will reject it.
            break
    return raw


def _validate_snapshot_html(
    html: str,
    *,
    min_length: int | None = None,
    min_entries: int | None = None,
) -> str | None:
    """Return a rejection reason, or ``None`` if the HTML looks real.

    Cloudflare-challenged captures ("Just a moment...") fail both the
    size check (~6 KB) and the entry-count check (0 entry divs).

    Defaults are read from the module constants at call time so tests
    can tune them via monkeypatch.
    """
    if min_length is None:
        min_length = _MIN_HTML_LENGTH
    if min_entries is None:
        min_entries = _MIN_ENTRY_COUNT
    if len(html) < min_length:
        return f"too small ({len(html)} chars < {min_length})"
    n_entries = html.count("c-crisiswatch-entry")
    if n_entries < min_entries:
        return f"only {n_entries} entry divs (< {min_entries})"
    return None


def _list_wayback_snapshots(
    *,
    lookback_days: int = _DEFAULT_LOOKBACK_DAYS,
    timeout_sec: int = 30,
) -> list[str]:
    """Return snapshot timestamps (newest first) from the CDX index.

    ``limit=-100`` (the last 100 captures) plus the ``from=`` window
    sidesteps CDX pagination entirely — the page gets a handful of
    captures per month.
    """
    from_date = (
        datetime.now(timezone.utc) - timedelta(days=lookback_days)
    ).strftime("%Y%m%d")
    resp = _get_with_retries(
        _WAYBACK_CDX_URL,
        params={
            "url": "crisisgroup.org/crisiswatch",
            "output": "json",
            "from": from_date,
            "fl": "timestamp,statuscode,mimetype",
            "filter": "mimetype:text/html",
            "limit": "-100",
        },
        timeout_sec=timeout_sec,
    )
    if resp is None:
        log.warning("CDX snapshot listing failed")
        return []
    try:
        rows = resp.json()
    except ValueError as exc:
        log.warning("CDX response is not JSON: %s", exc)
        return []
    # Row 0 of output=json is the header row (["timestamp", ...]).
    timestamps = [
        row[0]
        for row in rows[1:]
        if len(row) >= 2 and row[1] == "200"
    ]
    # 14-digit timestamps sort chronologically as strings.
    timestamps.sort(reverse=True)
    log.info(
        "CDX: %d snapshots in the last %d days (newest: %s)",
        len(timestamps), lookback_days,
        timestamps[0] if timestamps else "none",
    )
    return timestamps


def _fetch_snapshot_html(
    timestamp: str, *, timeout_sec: int = 120,
) -> str | None:
    """Download and validate one raw snapshot; ``None`` on rejection."""
    url = _WAYBACK_SNAPSHOT_URL.format(timestamp=timestamp)
    resp = _get_with_retries(url, timeout_sec=timeout_sec)
    if resp is None:
        log.warning("Snapshot %s download failed", timestamp)
        return None
    html = _maybe_gunzip(resp.content).decode("utf-8", errors="replace")
    reason = _validate_snapshot_html(html)
    if reason:
        log.warning("Snapshot %s rejected: %s", timestamp, reason)
        return None
    log.info(
        "Snapshot %s valid: %d chars, %d entry divs",
        timestamp, len(html), html.count("c-crisiswatch-entry"),
    )
    return html


def _trigger_save_page_now(*, timeout_sec: int = 120) -> bool:
    """Ask archive.org to capture a fresh snapshot (best-effort).

    Uses the authenticated SPN2 API when ``IA_S3_ACCESS_KEY`` /
    ``IA_S3_SECRET_KEY`` are set (higher rate limits from shared CI
    IPs; free keys at https://archive.org/account/s3.php), else the
    anonymous endpoint.  Never raises — a failed capture request just
    means we fall back to existing snapshots.

    Note the anonymous endpoint renders the capture synchronously and
    often exceeds any client timeout even when the capture succeeds
    server-side — callers should still poll CDX afterwards regardless
    of the return value.
    """
    access = os.environ.get("IA_S3_ACCESS_KEY", "").strip()
    secret = os.environ.get("IA_S3_SECRET_KEY", "").strip()
    try:
        if access and secret:
            resp = requests.post(
                _WAYBACK_SPN_API_URL,
                data={"url": _CRISISWATCH_URL},
                headers={
                    "User-Agent": _WAYBACK_USER_AGENT,
                    "Accept": "application/json",
                    "Authorization": f"LOW {access}:{secret}",
                },
                timeout=timeout_sec,
            )
            mode = "authenticated"
        else:
            resp = requests.get(
                _WAYBACK_SPN_URL,
                headers={"User-Agent": _WAYBACK_USER_AGENT},
                timeout=timeout_sec,
            )
            mode = "anonymous"
    except requests.RequestException as exc:
        log.warning("Save Page Now request failed: %s", exc)
        return False
    ok = resp.status_code < 400
    log.log(
        logging.INFO if ok else logging.WARNING,
        "Save Page Now (%s): HTTP %d", mode, resp.status_code,
    )
    return ok


def _wait_for_new_snapshot(
    baseline: str | None,
    *,
    wait_sec: int,
    timeout_sec: int = 30,
) -> str | None:
    """Poll CDX for a snapshot newer than *baseline*; ``None`` on timeout."""
    deadline = time.monotonic() + wait_sec
    while time.monotonic() < deadline:
        time.sleep(_SPN_POLL_INTERVAL_SEC)
        snapshots = _list_wayback_snapshots(timeout_sec=timeout_sec)
        if snapshots and (baseline is None or snapshots[0] > baseline):
            log.info("Fresh snapshot indexed: %s", snapshots[0])
            return snapshots[0]
    log.info(
        "No fresh snapshot indexed within %ds — using existing captures",
        wait_sec,
    )
    return None


def _fetch_wayback_html(
    *,
    timeout_sec: int = 120,
    spn: bool = True,
    spn_wait_sec: int = 240,
    lookback_days: int = _DEFAULT_LOOKBACK_DAYS,
) -> tuple[str, str]:
    """Fetch CrisisWatch HTML from the Wayback Machine.

    Returns ``(html, snapshot_timestamp)`` or exits(1) when no valid
    snapshot exists in the lookback window.
    """
    snapshots = _list_wayback_snapshots(
        lookback_days=lookback_days, timeout_sec=timeout_sec,
    )
    baseline = snapshots[0] if snapshots else None

    if spn:
        # Poll CDX even when the SPN request itself failed or timed
        # out: the anonymous endpoint renders the capture synchronously,
        # so the capture frequently succeeds server-side after the
        # client has given up (verified 2026-07-10).
        if not _trigger_save_page_now():
            log.info(
                "SPN request did not confirm — polling CDX anyway in "
                "case the capture succeeded server-side",
            )
        fresh_ts = _wait_for_new_snapshot(baseline, wait_sec=spn_wait_sec)
        if fresh_ts:
            # The fresh capture goes through the same validation — SPN
            # captures can themselves be Cloudflare-challenged.
            html = _fetch_snapshot_html(fresh_ts, timeout_sec=timeout_sec)
            if html is not None:
                return html, fresh_ts
            log.warning(
                "Fresh SPN capture %s invalid — walking existing snapshots",
                fresh_ts,
            )

    for timestamp in snapshots:
        html = _fetch_snapshot_html(timestamp, timeout_sec=timeout_sec)
        if html is not None:
            return html, timestamp

    log.error(
        "No valid Wayback snapshot in the last %d days "
        "(%d candidates tried).  crisisgroup.org may be blocking "
        "archive.org's crawler; try a manual local run "
        "(--source live --channel chrome).",
        lookback_days, len(snapshots),
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Edition comparison / staleness (--only-if-newer, --max-age-days)
# ---------------------------------------------------------------------------


def _edition_key(month_str: str, year: int) -> tuple[int, int] | None:
    """Parse ``"April 2026"`` -> ``(2026, 4)``; ``None`` if unparseable."""
    month_str = (month_str or "").strip()
    for fmt in ("%B %Y", "%b %Y"):
        try:
            dt = datetime.strptime(month_str, fmt)
            return (dt.year, dt.month)
        except ValueError:
            pass
    # Fallback: scan month names within the string.
    lowered = month_str.lower()
    for idx, name in enumerate(calendar.month_name):
        if idx and name.lower() in lowered:
            m = re.search(r"(\d{4})", month_str)
            y = int(m.group(1)) if m else year
            if y:
                return (y, idx)
    return None


def _load_existing_edition(output_path: Path) -> tuple[int, int] | None:
    """Edition key of the JSON already on disk; ``None`` if absent/corrupt."""
    try:
        data = json.loads(output_path.read_text(encoding="utf-8"))
        return _edition_key(
            str(data.get("month", "")), int(data.get("year", 0) or 0),
        )
    except Exception as exc:
        log.debug("No existing edition at %s: %s", output_path, exc)
        return None


def _check_staleness(
    edition: tuple[int, int] | None,
    max_age_days: int,
    *,
    now: datetime | None = None,
) -> None:
    """exit(1) when *edition* ended more than *max_age_days* ago.

    Keyed on the edition month (not ``fetched_at``, which no longer
    advances on --only-if-newer no-op runs) so a rotting refresh
    eventually turns scheduled runs red instead of silently serving
    months-old data.
    """
    if max_age_days <= 0 or edition is None:
        return
    year, month = edition
    last_day = calendar.monthrange(year, month)[1]
    end_of_month = datetime(year, month, last_day, tzinfo=timezone.utc)
    now = now or datetime.now(timezone.utc)
    age_days = (now - end_of_month).days
    if age_days > max_age_days:
        log.error(
            "On-disk CrisisWatch edition %04d-%02d is %d days old "
            "(> %d) — the refresh is rotting.  Investigate Wayback "
            "snapshot availability or run a manual local refresh.",
            year, month, age_days, max_age_days,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# HTML parsing
# ---------------------------------------------------------------------------


def _parse_overview_section(
    soup: BeautifulSoup,
) -> dict[str, Any]:
    """Parse the Global Overview section at the top of the page.

    Returns a dict with keys:
        outlook_month, conflict_risk_alerts, resolution_opportunities,
        deteriorated, improved, report_month

    The overview lives inside ``<div class="c-in-focus">``.  It has two
    ``<section class="c-in-focus__state">`` blocks, each headed by an
    ``<h3>`` ("Outlook for This Month" / "Trends for Last Month") with
    the month in a child ``<span>``.  Category ``<h4>`` headings
    (Conflict Risk Alerts, etc.) are inside
    ``<div class="c-crisiswatch--regional__trend">`` blocks; country
    links sit in ``<p>`` siblings of the ``<h4>``.
    """
    result: dict[str, Any] = {
        "outlook_month": "",
        "report_month": "",
        "conflict_risk_alerts": [],
        "resolution_opportunities": [],
        "deteriorated": [],
        "improved": [],
    }

    # --- Extract report month from <h1> → <time> as primary source ---
    h1 = soup.find("h1")
    if h1:
        time_tag = h1.find("time")
        if time_tag:
            result["report_month"] = time_tag.get_text(strip=True)

    # --- Extract outlook/trends months from <h3> headings ---
    # The overview uses <h3> (not <h2>) for "Outlook for This Month" /
    # "Trends for Last Month".  The month is in a child <span>.
    in_focus = soup.find("div", class_="c-in-focus")
    if in_focus:
        for h3 in in_focus.find_all("h3"):
            text = h3.get_text(strip=True)
            text_lower = text.lower()

            # Month is in a child <span> of the <h3>.
            month_span = h3.find("span")
            month_text = month_span.get_text(strip=True) if month_span else ""

            if "outlook for this month" in text_lower:
                result["outlook_month"] = month_text or text.replace(
                    "Outlook for This Month", ""
                ).strip()
            elif "trends for last month" in text_lower:
                if month_text:
                    result["report_month"] = month_text

    # --- Parse <h4> category headings and collect country links ---
    category_map = {
        "conflict risk alert": "conflict_risk_alerts",
        "resolution opportunit": "resolution_opportunities",
        "deteriorated situation": "deteriorated",
        "improved situation": "improved",
    }

    # Scope to the overview container to avoid picking up stray <h4>s.
    search_root = in_focus or soup
    all_h4 = search_root.find_all("h4")
    for h4 in all_h4:
        h4_text = h4.get_text(strip=True).lower()
        target_key = None
        for pattern, key in category_map.items():
            if pattern in h4_text:
                target_key = key
                break
        if not target_key:
            continue

        # Country links are in <p> or <div> siblings within the same
        # <div class="c-crisiswatch--regional__trend"> parent.
        countries: list[str] = []
        parent = h4.parent
        if parent:
            for link in parent.find_all("a"):
                country_name = link.get_text(strip=True)
                if country_name:
                    iso3 = _resolve_iso3(country_name)
                    if iso3:
                        countries.append(iso3)
                    else:
                        log.warning(
                            "Overview: unmatched country '%s' in %s",
                            country_name, target_key,
                        )
        if not countries:
            # Fallback: scan siblings after the h4
            for sib in h4.find_next_siblings():
                if isinstance(sib, Tag) and sib.name in ("h4", "h2", "h3"):
                    break
                if isinstance(sib, Tag):
                    for link in sib.find_all("a"):
                        country_name = link.get_text(strip=True)
                        if country_name:
                            iso3 = _resolve_iso3(country_name)
                            if iso3:
                                countries.append(iso3)

        result[target_key] = countries

    return result


def _parse_country_entries(
    soup: BeautifulSoup,
) -> tuple[list[dict[str, Any]], str, int]:
    """Parse per-country entries from the Latest Updates section.

    Returns (entries, month_name, year).

    Each entry is a ``<div class="c-crisiswatch-entry"
    data-entry-country="..." id="...">`` containing:
    - ``<h3>`` with SVG status icons and country name text
    - ``<time>`` with the edition month (e.g. "February 2026")
    - ``<div class="o-crisis-states__detail">`` with narrative ``<p>`` tags
    """
    entries: list[dict[str, Any]] = []
    detected_month = ""
    detected_year = 0

    # Country entries use <div class="c-crisiswatch-entry" data-entry-country="...">.
    entry_divs = soup.find_all("div", class_="c-crisiswatch-entry")
    log.info("Found %d c-crisiswatch-entry divs", len(entry_divs))

    for entry_div in entry_divs:
        country_slug = entry_div.get("data-entry-country", "")
        if not country_slug:
            continue

        # Country name is the text content of the <h3> (stripping SVG icons).
        h3 = entry_div.find("h3")
        if not h3:
            continue
        country_name = h3.get_text(strip=True)
        if not country_name:
            continue

        iso3 = _resolve_iso3(country_name)
        if not iso3:
            # Try cleaning up: "Israel/Palestine", "India (Jammu and Kashmir)"
            clean_name = re.sub(r"\s*\(.*?\)\s*", "", country_name).strip()
            iso3 = _resolve_iso3(clean_name)
            if not iso3:
                log.warning(
                    "Unmatched country: '%s' (slug=%s)",
                    country_name, country_slug,
                )

        # --- Extract arrow/status from SVG icons in the <h3> ---
        # Icons: <span class="o-icon ..."><svg><use xlink:href="#deteriorated"></use></svg></span>
        arrow = "unchanged"
        alert_type = ""

        for use_tag in h3.find_all("use"):
            href = (
                use_tag.get("xlink:href")
                or use_tag.get("href")
                or ""
            )
            if href in _SVG_STATUS_MAP:
                a, at = _SVG_STATUS_MAP[href]
                if a:
                    arrow = a
                if at:
                    alert_type = at

        # --- Extract month from <time> tag ---
        time_tag = entry_div.find("time")
        if time_tag:
            time_text = time_tag.get_text(strip=True)
            if time_text and not detected_month:
                m = re.match(r"(\w+)\s+(\d{4})", time_text)
                if m:
                    detected_month = f"{m.group(1)} {m.group(2)}"
                    try:
                        detected_year = int(m.group(2))
                    except ValueError:
                        pass

        # --- Extract summary from narrative div ---
        detail_div = entry_div.find(
            "div", class_=re.compile(r"o-crisis-states__detail"),
        )
        summary_parts: list[str] = []
        headline = ""

        if detail_div:
            for p_tag in detail_div.find_all("p"):
                # Skip "Click here for past entries" links
                if p_tag.find("a", href=re.compile(r"/crisiswatch/database")):
                    continue
                text = p_tag.get_text(strip=True)
                if text:
                    summary_parts.append(text)

            # The first <strong> is typically the headline.
            strong = detail_div.find("strong")
            if strong:
                headline = strong.get_text(strip=True)

        # Build summary: prefer headline, fall back to first paragraph.
        summary = headline or (summary_parts[0] if summary_parts else "")
        # Truncate to ~500 chars.
        if len(summary) > 500:
            summary = summary[:497] + "..."

        # --- Expand regional entries into per-country rows ---
        if country_name in _REGIONAL_ENTRY_MAP:
            for sub_country, sub_iso3 in _REGIONAL_ENTRY_MAP[country_name]:
                entries.append({
                    "country": sub_country,
                    "iso3": sub_iso3,
                    "arrow": arrow,
                    "alert_type": alert_type,
                    "summary": summary,
                })
            log.info(
                "Expanded regional entry '%s' into %d countries",
                country_name, len(_REGIONAL_ENTRY_MAP[country_name]),
            )
        else:
            entries.append({
                "country": country_name,
                "iso3": iso3 or "",
                "arrow": arrow,
                "alert_type": alert_type,
                "summary": summary,
            })

    return entries, detected_month, detected_year


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(
    *,
    output_path: Path = _DEFAULT_OUTPUT,
    timeout_sec: int = 90,
    verbose: bool = False,
    channel: str | None = None,
    auto_push: bool = False,
    source: str = "live",
    only_if_newer: bool = False,
    force: bool = False,
    spn: bool = True,
    spn_wait_sec: int = 240,
    max_age_days: int = 0,
) -> dict[str, Any]:
    """Fetch and parse CrisisWatch, write JSON, return the data dict."""

    if source == "wayback":
        html, snapshot_ts = _fetch_wayback_html(
            timeout_sec=timeout_sec, spn=spn, spn_wait_sec=spn_wait_sec,
        )
        provenance = f"wayback:{snapshot_ts}"
    else:
        html = _fetch_page_html(
            timeout_sec=timeout_sec, verbose=verbose, channel=channel,
        )
        provenance = "live"

    if verbose:
        debug_html_path = Path("horizon_scanner/data/crisiswatch_debug.html")
        debug_html_path.parent.mkdir(parents=True, exist_ok=True)
        debug_html_path.write_text(html, encoding="utf-8")
        log.info("Raw HTML saved to %s (%d bytes)", debug_html_path, len(html))

    soup = BeautifulSoup(html, "html.parser")

    # Parse overview section.
    overview = _parse_overview_section(soup)
    log.info(
        "Overview: outlook=%s, report=%s, "
        "conflict_risk_alerts=%d, resolution_opportunities=%d, "
        "deteriorated=%d, improved=%d",
        overview.get("outlook_month"),
        overview.get("report_month"),
        len(overview.get("conflict_risk_alerts", [])),
        len(overview.get("resolution_opportunities", [])),
        len(overview.get("deteriorated", [])),
        len(overview.get("improved", [])),
    )

    # Parse country entries.
    entries, month_str, year = _parse_country_entries(soup)
    log.info("Parsed %d country entries (month=%s, year=%d)", len(entries), month_str, year)

    # Use overview report_month as the canonical month if available.
    if overview.get("report_month"):
        month_str = overview["report_month"]
        m = re.match(r"(\w+)\s+(\d{4})", month_str)
        if m:
            try:
                year = int(m.group(2))
            except ValueError:
                pass

    if not entries:
        log.error(
            "No country entries parsed!  The page structure may have changed.  "
            "Re-run with --verbose to inspect the debug HTML."
        )
        sys.exit(1)

    # Count entries with valid ISO3.
    n_resolved = sum(1 for e in entries if e.get("iso3"))
    n_unresolved = len(entries) - n_resolved
    if n_unresolved:
        log.warning("%d entries have unresolved ISO3 codes", n_unresolved)
    if n_resolved == 0:
        log.error(
            "0 of %d entries resolved to ISO3 — the "
            "horizon_scanner.crisiswatch import likely failed and the "
            "_resolve_iso3 stub is active (missing PyYAML?).  Refusing "
            "to write useless data.",
            len(entries),
        )
        sys.exit(1)

    # Only-if-newer gate: never regress to an older edition, and skip
    # the write (no fetched_at churn, no commit) when the parsed
    # edition is not strictly newer than what is already on disk.
    if only_if_newer and not force:
        candidate = _edition_key(month_str, year)
        existing = _load_existing_edition(output_path)
        if candidate is None:
            log.error(
                "Cannot parse candidate edition month %r — refusing "
                "--only-if-newer write (page structure may have changed).",
                month_str,
            )
            sys.exit(1)
        if existing is not None and candidate <= existing:
            log.info(
                "Candidate edition %04d-%02d is not newer than existing "
                "%04d-%02d — skipping write.",
                candidate[0], candidate[1], existing[0], existing[1],
            )
            _check_staleness(existing, max_age_days)
            return {}

    # Build output dict (compatible with _load_fallback_json).
    now = datetime.now(timezone.utc).isoformat()
    data: dict[str, Any] = {
        "month": month_str,
        "year": year,
        "fetched_at": now,
        "source": provenance,
        "outlook_month": overview.get("outlook_month", ""),
        "conflict_risk_alerts": overview.get("conflict_risk_alerts", []),
        "resolution_opportunities": overview.get("resolution_opportunities", []),
        "deteriorated": overview.get("deteriorated", []),
        "improved": overview.get("improved", []),
        "entries": entries,
    }

    # Write JSON.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    log.info("Wrote %d entries to %s", len(entries), output_path)

    # Summary statistics.
    arrows = [e.get("arrow", "") for e in entries]
    n_deteriorated = sum(1 for a in arrows if a == "deteriorated")
    n_improved = sum(1 for a in arrows if a == "improved")
    n_unchanged = sum(1 for a in arrows if a == "unchanged")
    n_alerts = sum(1 for e in entries if e.get("alert_type"))
    log.info(
        "Summary: total=%d, deteriorated=%d, improved=%d, "
        "unchanged=%d, alerts=%d, resolved_iso3=%d",
        len(entries), n_deteriorated, n_improved, n_unchanged,
        n_alerts, n_resolved,
    )

    # Staleness alarm (trivially passes right after a fresh write unless
    # even the newest available edition is old).
    _check_staleness(_edition_key(month_str, year), max_age_days)

    # Auto-push: git add, commit, push the output JSON.
    if auto_push:
        import subprocess

        subprocess.run(["git", "add", "-f", str(output_path)], check=True)
        result = subprocess.run(["git", "diff", "--cached", "--quiet"])
        if result.returncode != 0:
            subprocess.run([
                "git", "commit", "-m",
                "chore(crisiswatch): refresh monthly data",
            ], check=True)
            subprocess.run(["git", "push"], check=True)
            log.info("Committed and pushed updated JSON")
        else:
            log.info("No changes to commit")

    return data


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Fetch ICG CrisisWatch data and write JSON.",
    )
    parser.add_argument(
        "--output", type=str, default=str(_DEFAULT_OUTPUT),
        help="Output JSON path (default: %(default)s)",
    )
    parser.add_argument(
        "--timeout", type=int, default=90,
        help="Page load timeout in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable DEBUG logging and save debug artifacts",
    )
    parser.add_argument(
        "--channel", type=str, default=None,
        help="Use system browser: 'chrome' or 'msedge' (bypasses Cloudflare)",
    )
    parser.add_argument(
        "--auto-push", action="store_true",
        help="Git add, commit and push the output JSON after success",
    )
    parser.add_argument(
        "--source", choices=["live", "wayback"], default="live",
        help="Acquisition path: 'live' = Playwright against "
             "crisisgroup.org (default), 'wayback' = Internet Archive "
             "snapshots (no browser needed; used in CI)",
    )
    parser.add_argument(
        "--only-if-newer", action="store_true",
        help="Skip the write when the parsed edition is not strictly "
             "newer than the existing JSON",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Write even when --only-if-newer would skip",
    )
    parser.add_argument(
        "--no-spn", action="store_true",
        help="Wayback only: skip the Save Page Now capture trigger",
    )
    parser.add_argument(
        "--spn-wait", type=int, default=240,
        help="Wayback only: seconds to poll for a fresh snapshot after "
             "Save Page Now (default: %(default)s)",
    )
    parser.add_argument(
        "--max-age-days", type=int, default=0,
        help="Exit non-zero when the resulting on-disk edition is older "
             "than this many days (0 = disabled)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    run(
        output_path=Path(args.output),
        timeout_sec=args.timeout,
        verbose=args.verbose,
        channel=args.channel,
        auto_push=args.auto_push,
        source=args.source,
        only_if_newer=args.only_if_newer,
        force=args.force,
        spn=not args.no_spn,
        spn_wait_sec=args.spn_wait,
        max_age_days=args.max_age_days,
    )


if __name__ == "__main__":
    main()
