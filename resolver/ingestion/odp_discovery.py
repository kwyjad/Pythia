# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Mapping, Optional

import requests
import yaml

LOGGER = logging.getLogger(__name__)


@dataclass
class OdpPage:
    page_id: str
    url: str


@dataclass
class DiscoveredLink:
    href: str
    text: str


@dataclass
class PageDiscovery:
    page_id: str
    page_url: str
    links: List[DiscoveredLink]


JSON_LINK_RE = re.compile(r"\bformat=json\b|_format=json", re.IGNORECASE)


def _default_fetch_html(url: str, timeout: int = 15) -> str:
    """Fetch HTML content from the given URL. In tests, this function is monkeypatched."""
    LOGGER.info("Fetching ODP page", extra={"url": url})
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def load_config(config: Mapping[str, Any]) -> List[OdpPage]:
    """Turn a raw config mapping into a list of OdpPage objects."""
    pages_cfg = config.get("pages", [])
    pages: List[OdpPage] = []
    for entry in pages_cfg:
        page_id = entry.get("id")
        url = entry.get("url")
        if not page_id or not url:
            continue
        pages.append(OdpPage(page_id=page_id, url=url))
    return pages


def discover_pages(
    config: Mapping[str, Any],
    fetch_html: Optional[Callable[[str], str]] = None,
) -> List[PageDiscovery]:
    """
    Given a config mapping with a 'pages' list, fetch each page and discover JSON links.

    A "JSON link" is defined as an <a> element whose href contains '?_format=json'
    or 'format=json', or whose visible text includes 'json' (case-insensitive).
    """
    pages = load_config(config)
    fetch_fn = fetch_html or _default_fetch_html
    results: List[PageDiscovery] = []

    for page in pages:
        html = fetch_fn(page.url)
        links: List[DiscoveredLink] = []
        anchor_re = re.compile(r'<a\s+[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.IGNORECASE | re.DOTALL)
        for match in anchor_re.finditer(html):
            href = match.group(1)
            text = re.sub(r"\s+", " ", match.group(2)).strip()
            if JSON_LINK_RE.search(href) or re.search(r"\bjson\b", text, re.IGNORECASE):
                links.append(DiscoveredLink(href=href, text=text))
        results.append(PageDiscovery(page_id=page.page_id, page_url=page.url, links=links))

    return results


def discover_to_disk(config_path: Path, out_dir: Path) -> List[PageDiscovery]:
    """
    Load config YAML from disk, perform discovery, and write a combined discovery.json
    into `out_dir`.
    """
    with config_path.open("r", encoding="utf-8") as fh:
        raw_cfg = yaml.safe_load(fh) or {}
    discoveries = discover_pages(raw_cfg)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "discovery.json"
    serialisable = [
        {
            "page_id": d.page_id,
            "page_url": d.page_url,
            "links": [{"href": l.href, "text": l.text} for l in d.links],
        }
        for d in discoveries
    ]
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(serialisable, fh, indent=2, ensure_ascii=False)

    LOGGER.info(
        "ODP discovery complete",
        extra={
            "pages": len(discoveries),
            "links_total": sum(len(d.links) for d in discoveries),
            "out": str(out_path),
        },
    )
    return discoveries


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Discover JSON endpoints on ODP pages.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to ODP discovery config YAML (with 'pages' list).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="diagnostics/ingestion/odp",
        help="Directory to write discovery.json into.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    discover_to_disk(Path(args.config), Path(args.out))


if __name__ == "__main__":
    main()
