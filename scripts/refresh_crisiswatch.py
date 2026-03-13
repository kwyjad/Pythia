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

Cloudflare protection: the scraper uses ``playwright-stealth`` and
realistic browser fingerprinting to bypass Cloudflare's JS challenge.
If the challenge is not resolved after 3 attempts, the scraper exits
with an error.

Usage::

    # Install Playwright first (one-time)
    pip install playwright playwright-stealth && playwright install chromium

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

# Status label text → (arrow, alert_type) mapping.
# The CrisisWatch page uses plain-text labels inside timeline <span>
# elements, not icons.
_STATUS_MAP: dict[str, tuple[str, str]] = {
    "deteriorated situation": ("deteriorated", ""),
    "improved situation": ("improved", ""),
    "unchanged situation": ("unchanged", ""),
    "conflict risk alert": ("unchanged", "conflict_risk"),
    "resolution opportunity": ("unchanged", "resolution_opportunity"),
}


# ---------------------------------------------------------------------------
# Playwright page fetch (with Cloudflare bypass)
# ---------------------------------------------------------------------------


def _fetch_page_html(*, timeout_sec: int = 90, verbose: bool = False) -> str:
    """Launch headless Chromium, load CrisisWatch, and return page HTML.

    Uses ``playwright-stealth`` and realistic browser fingerprinting to
    bypass Cloudflare's JS challenge.  Retries up to ``_MAX_ATTEMPTS``
    times if the page returns a Cloudflare challenge (< 20 KB HTML).

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

    try:
        from playwright_stealth import stealth_sync
    except ImportError:
        log.error(
            "playwright-stealth is not installed.  "
            "Run: pip install playwright-stealth"
        )
        sys.exit(1)

    debug_dir = Path("horizon_scanner/data")
    debug_dir.mkdir(parents=True, exist_ok=True)

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
            stealth_sync(page)

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
                # Wait for network to settle before scrolling.
                try:
                    page.wait_for_load_state(
                        "networkidle", timeout=30_000,
                    )
                except Exception:
                    log.debug("networkidle timed out, proceeding anyway")

                # Scroll to bottom repeatedly to trigger lazy-loaded content.
                prev_height = 0
                scroll_rounds = 0
                max_rounds = 30
                while scroll_rounds < max_rounds:
                    height = page.evaluate("document.body.scrollHeight")
                    if height == prev_height:
                        break
                    page.evaluate(
                        "window.scrollTo(0, document.body.scrollHeight)"
                    )
                    page.wait_for_timeout(2500)
                    prev_height = height
                    scroll_rounds += 1

                log.info("Scroll complete after %d rounds.", scroll_rounds)

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
            "Consider running locally or updating stealth settings.",
            _MAX_ATTEMPTS, len(html),
        )
        # Save the final blocked HTML for debugging.
        debug_html_path = debug_dir / "crisiswatch_debug.html"
        debug_html_path.write_text(html, encoding="utf-8")
        log.info("Final blocked HTML saved to %s", debug_html_path)
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
    """
    result: dict[str, Any] = {
        "outlook_month": "",
        "report_month": "",
        "conflict_risk_alerts": [],
        "resolution_opportunities": [],
        "deteriorated": [],
        "improved": [],
    }

    # The overview has two <h2> headings:
    #   "Outlook for This Month March 2026"
    #   "Trends for Last Month February 2026"
    # Under each, <h4> sub-headings label the categories.

    all_h2 = soup.find_all("h2")
    for h2 in all_h2:
        text = h2.get_text(strip=True)
        text_lower = text.lower()

        if "outlook for this month" in text_lower:
            # Extract month: everything after "Outlook for This Month "
            m = re.search(r"(?i)outlook for this month\s+(.+)", text)
            if m:
                result["outlook_month"] = m.group(1).strip()

        elif "trends for last month" in text_lower:
            m = re.search(r"(?i)trends for last month\s+(.+)", text)
            if m:
                result["report_month"] = m.group(1).strip()

    # Parse <h4> category headings and collect country links that follow.
    category_map = {
        "conflict risk alert": "conflict_risk_alerts",
        "resolution opportunit": "resolution_opportunities",
        "deteriorated situation": "deteriorated",
        "improved situation": "improved",
    }

    all_h4 = soup.find_all("h4")
    for h4 in all_h4:
        h4_text = h4.get_text(strip=True).lower()
        target_key = None
        for pattern, key in category_map.items():
            if pattern in h4_text:
                target_key = key
                break
        if not target_key:
            continue

        # Collect <a> tags between this <h4> and the next <h4>/<h2>.
        countries: list[str] = []
        # Look inside the h4's parent container for links
        parent = h4.parent
        if parent:
            # Find all links after this h4 within the same parent
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
    """
    entries: list[dict[str, Any]] = []
    detected_month = ""
    detected_year = 0

    # Country entries use <h3 id="country-slug"> tags.
    country_headings = soup.find_all("h3", id=True)
    log.info("Found %d country <h3> headings", len(country_headings))

    for h3 in country_headings:
        country_name = h3.get_text(strip=True)
        if not country_name:
            continue

        # Skip non-country headings (e.g. section titles)
        country_id = h3.get("id", "")
        if not country_id or country_id.startswith("block-"):
            continue

        iso3 = _resolve_iso3(country_name)
        if not iso3:
            # Try cleaning up: sometimes names have extra text
            # e.g. "Israel/Palestine" or "India (Jammu and Kashmir)"
            clean_name = re.sub(r"\s*\(.*?\)\s*", "", country_name).strip()
            iso3 = _resolve_iso3(clean_name)
            if not iso3:
                log.warning("Unmatched country: '%s' (id=%s)", country_name, country_id)

        # Collect all siblings until the next <h3> or <h2>.
        siblings: list[Tag] = []
        for sib in h3.find_next_siblings():
            if not isinstance(sib, Tag):
                continue
            if sib.name in ("h3", "h2"):
                break
            siblings.append(sib)

        # --- Extract arrow/status from the most recent timeline entry ---
        arrow = "unchanged"
        alert_type = ""
        entry_month = ""

        for sib in siblings:
            if sib.name != "a":
                continue
            href = sib.get("href", "")
            if "/crisiswatch" not in href:
                continue

            # Timeline <a> tags have two <span> children:
            #   <span>Deteriorated Situation</span>
            #   <span>February 2026</span>
            spans = sib.find_all("span")
            if len(spans) >= 2:
                status_text = spans[0].get_text(strip=True).lower()
                month_text = spans[1].get_text(strip=True)

                # We want the LAST (most recent) timeline entry.
                # The timeline goes chronologically, so keep overwriting.
                for pattern, (a, at) in _STATUS_MAP.items():
                    if pattern in status_text:
                        arrow = a
                        if at:
                            alert_type = at
                        entry_month = month_text
                        break
            elif len(spans) == 1:
                # Single span — try to parse status text
                status_text = spans[0].get_text(strip=True).lower()
                for pattern, (a, at) in _STATUS_MAP.items():
                    if pattern in status_text:
                        arrow = a
                        if at:
                            alert_type = at
                        break

        # Try to detect month/year from the most recent timeline entry.
        if entry_month and not detected_month:
            m = re.match(r"(\w+)\s+(\d{4})", entry_month)
            if m:
                detected_month = f"{m.group(1)} {m.group(2)}"
                try:
                    detected_year = int(m.group(2))
                except ValueError:
                    pass

        # --- Extract summary from narrative paragraphs ---
        summary_parts: list[str] = []
        for sib in siblings:
            if sib.name != "p":
                continue
            # Skip paragraphs that are just archive links
            if sib.find("a", href=re.compile(r"/crisiswatch/database")):
                continue
            text = sib.get_text(strip=True)
            if text:
                summary_parts.append(text)

        # The first <strong> in the first <p> is typically the headline.
        headline = ""
        for sib in siblings:
            if sib.name != "p":
                continue
            strong = sib.find("strong")
            if strong:
                headline = strong.get_text(strip=True)
                break

        # Build summary: prefer headline, fall back to first paragraph.
        summary = headline or (summary_parts[0] if summary_parts else "")
        # Truncate to ~500 chars.
        if len(summary) > 500:
            summary = summary[:497] + "..."

        entry: dict[str, Any] = {
            "country": country_name,
            "iso3": iso3 or "",
            "arrow": arrow,
            "alert_type": alert_type,
            "summary": summary,
        }
        entries.append(entry)

    return entries, detected_month, detected_year


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(
    *,
    output_path: Path = _DEFAULT_OUTPUT,
    timeout_sec: int = 90,
    verbose: bool = False,
) -> dict[str, Any]:
    """Fetch and parse CrisisWatch, write JSON, return the data dict."""

    html = _fetch_page_html(timeout_sec=timeout_sec, verbose=verbose)

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
    )


if __name__ == "__main__":
    main()
