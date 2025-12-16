# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

import json
import textwrap

from resolver.ingestion import odp_discovery


def _fake_fetch_html_factory(pages_html):
    """Return a fetch_html function that serves static HTML for given page URLs."""
    def fetch_html(url: str) -> str:
        return pages_html[url]

    return fetch_html


def test_discover_pages_finds_json_links(tmp_path):
    html_by_url = {
        "https://example.org/page1": textwrap.dedent(
            '''
            <html><body>
              <a href="https://api.example.org/foo?_format=json">Download JSON</a>
              <a href="https://api.example.org/bar?format=json">JSON</a>
              <a href="https://example.org/nonjson">Not JSON</a>
            </body></html>
            '''
        )
    }
    config = {
        "pages": [
            {"id": "p1", "url": "https://example.org/page1"},
        ]
    }
    fetch_html = _fake_fetch_html_factory(html_by_url)

    results = odp_discovery.discover_pages(config, fetch_html=fetch_html)
    assert len(results) == 1
    page = results[0]
    assert page.page_id == "p1"
    assert page.page_url == "https://example.org/page1"
    hrefs = {link.href for link in page.links}
    assert hrefs == {
        "https://api.example.org/foo?_format=json",
        "https://api.example.org/bar?format=json",
        "https://example.org/nonjson",
    }
    texts = {link.text for link in page.links}
    assert "Download JSON" in texts
    assert "JSON" in texts


def test_discover_to_disk_writes_discovery_json(tmp_path):
    cfg_path = tmp_path / "odp.yml"
    cfg_path.write_text(
        textwrap.dedent(
            """
            pages:
              - id: "demo"
                url: "https://example.org/demo"
            """
        ),
        encoding="utf-8",
    )

    html_by_url = {
        "https://example.org/demo": '<a href="https://api.example.org/demo?_format=json">JSON</a>'
    }

    def fake_fetch(url: str) -> str:
        return html_by_url[url]

    original_fetch = odp_discovery._default_fetch_html
    try:
        odp_discovery._default_fetch_html = fake_fetch  # type: ignore[assignment]
        out_dir = tmp_path / "out"
        results = odp_discovery.discover_to_disk(cfg_path, out_dir)
    finally:
        odp_discovery._default_fetch_html = original_fetch

    assert len(results) == 1
    out_file = out_dir / "discovery.json"
    assert out_file.exists()
    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert data[0]["page_id"] == "demo"
    assert data[0]["links"][0]["href"].endswith("?_format=json")
