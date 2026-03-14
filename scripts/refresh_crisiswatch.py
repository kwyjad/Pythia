# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Playwright-based ICG CrisisWatch scraper.

Fetches the current CrisisWatch edition from the main page at
https://www.crisisgroup.org/crisiswatch using headless Chromium
(Playwright) and parses the HTML with BeautifulSoup.

The /crisiswatch/print endpoint is BROKEN (serves stale October 2019
data).  The main page renders the current edition correctly — a single
long page with all country entries in the DOM.

Output is written to ``horizon_scanner/data/crisiswatch_latest.json`` in
a format compatible with ``crisiswatch._load_fallback_json()``.

Usage::

    # Install Playwright first (one-time)
    pip install playwright && playwright install chromium

    # Run the scraper
    python -m scripts.refresh_crisiswatch [--output PATH] [--verbose]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
) -> dict[str, Any]:
    """Fetch and parse CrisisWatch, write JSON, return the data dict."""

    html = _fetch_page_html(
        timeout_sec=timeout_sec, verbose=verbose, channel=channel,
    )

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

    # Build output dict (compatible with _load_fallback_json).
    now = datetime.now(timezone.utc).isoformat()
    data: dict[str, Any] = {
        "month": month_str,
        "year": year,
        "fetched_at": now,
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
        description="Fetch ICG CrisisWatch data via Playwright and write JSON.",
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
    )


if __name__ == "__main__":
    main()
