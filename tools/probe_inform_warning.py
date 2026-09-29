# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Find out how INFORM Warning data can be read, by asking the service.

INFORM Warning (UNDP + the European Commission's JRC) publishes monthly anomaly
"Signals" per country, and Pythia wants to read them. Nobody here has seen the
route: the build sandbox cannot reach any JRC host, and web search returns the
landing pages but not what is on them. A connector written against a guessed
route fails every month, so this script asks first and writes nothing but a
report.

What it asks, cheapest first:

    1. The landing pages (INFORM Warning, the Warning tool, the API demo, the
       user guide's record page). Every link on them that names an API, a
       registry, a download or a data file is kept.
    2. The two PDFs that describe data access: the INFORM Warning user guide
       (JRC146982) and the INFORM GRI Web API documentation. With `pypdf`
       installed, their text is saved and every line naming an address, an
       API, a download or a historical release is quoted in the report.
    3. The documented INFORM Risk API (`/inform-index/API/InformAPI/`), whose
       releases are "workflows". If any workflow group or system names
       "Warning", the probe asks for its workflows and then for country scores
       from up to three of them — which is what would settle whether Warning
       shares the Risk API and whether past releases can be requested.
    4. The HDX and JRC catalogue searches for an "INFORM Warning" dataset.
    5. One hop along every link found above that looks like data, and any
       extra URLs the operator passes.

Every response is saved raw under ``<out-dir>/raw/`` and described in
``inform_warning_probe.json``; ``inform_warning_probe.md`` is the human
report, and the same text is printed to stdout so the job log carries it.
The probe never fails the run: an unreachable host is a finding.

Usage
-----
    python -m tools.probe_inform_warning [--out-dir diagnostics/inform_warning]
        [--max-follow 40] [--extra-urls URL,URL]
"""

from __future__ import annotations

import argparse
import html
import io
import json
import logging
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import urljoin, urlparse

import requests

LOG = logging.getLogger("probe_inform_warning")

BASE = "https://drmkc.jrc.ec.europa.eu/inform-index"
API_ROOT = f"{BASE}/API/InformAPI"

LANDING_PAGES = [
    ("warning_landing", f"{BASE}/INFORM-Warning"),
    ("warning_tool", f"{BASE}/inform-warning-tool/"),
    ("api_demo", f"{BASE}/In-depth/API-Demo"),
    ("guide_record", "https://publications.jrc.ec.europa.eu/repository/handle/JRC146982"),
]

PDFS = [
    ("user_guide_pdf",
     "https://publications.jrc.ec.europa.eu/repository/bitstream/JRC146982/JRC146982_01.pdf"),
    ("gri_api_doc_pdf",
     f"{BASE}/portals/0/INFORM%20GRI%20-%20New%20Web%20API%20Documentation.pdf"),
]

# The INFORM Risk API's Workflow controller is documented as offering Index,
# GetBySystem, GetByWorkflowGroup, GetByYear, WorkflowGroups, Systems and
# Default. Whether the controller is spelled Workflow or Workflows is not
# known from here, so both are asked; a 404 on one spelling is a finding.
WORKFLOW_LISTINGS = [
    "Workflows/WorkflowGroups",
    "Workflows/Systems",
    "Workflows/Default",
    "Workflows",
    "Workflow/WorkflowGroups",
    "Workflow/Systems",
    "Workflow/Default",
]

CATALOGUE_SEARCHES = [
    ("hdx_search",
     "https://data.humdata.org/api/3/action/package_search?q=%22INFORM%20Warning%22&rows=20"),
    ("jrc_catalogue_ckan_guess",
     "https://data.jrc.ec.europa.eu/api/3/action/package_search?q=%22INFORM%20Warning%22"),
    ("jrc_catalogue_search", "https://data.jrc.ec.europa.eu/search?q=INFORM+Warning"),
]

# Hosts a followed link may point at. Anything else is recorded, not fetched.
FOLLOW_HOST_SUFFIXES = ("europa.eu", "humdata.org", "undp.org")

# A link is worth following when its URL or anchor text names one of these.
LINK_KEYWORDS = re.compile(
    r"api|registry|download|dataset|data[-_/ ]|warning|signal|\.csv|\.xlsx?|\.json|\.zip",
    re.IGNORECASE,
)

# A line of guide text worth quoting names an address or a way of getting data.
GUIDE_LINE_KEYWORDS = re.compile(
    r"https?://|\bapi\b|registry|download|endpoint|\bjson\b|\bcsv\b|\bxlsx?\b|"
    r"archive|historical|previous release|vintage|licen[cs]e|cc by|workflow",
    re.IGNORECASE,
)

# A response that looks like the data itself carries ISO3 codes beside words
# the Warning product uses.
WARNING_DATA_WORDS = re.compile(
    r"signal|warning|reliab|tf3|tf6|tf12|dry conditions|wet conditions|river flow",
    re.IGNORECASE,
)
ISO3_TOKEN = re.compile(r"\"(?:AFG|SOM|SDN|ETH|YEM|SYR|COD|HTI|MLI|NER|SSD|MOZ)\"|\b(?:AFG|SOM|SDN|ETH|YEM|COD)\b")

MAX_SAVE_BYTES = 25 * 1024 * 1024
SNIPPET_CHARS = 400
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0 Safari/537.36 Pythia-inform-warning-probe"
)


@dataclass
class Probe:
    label: str
    url: str
    status: int | None = None
    final_url: str = ""
    content_type: str = ""
    kind: str = ""
    n_bytes: int = 0
    error: str = ""
    snippet: str = ""
    saved_as: str = ""
    looks_like_warning_data: bool = False
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pure helpers (tested)
# ---------------------------------------------------------------------------

def classify_kind(content_type: str, body: bytes) -> str:
    """Name what a body IS from its bytes first and its header second."""
    head = body[:8].lstrip()
    if body.startswith(b"%PDF"):
        return "pdf"
    if body.startswith(b"PK\x03\x04"):
        return "zip"
    if head[:1] in (b"{", b"["):
        return "json"
    ct = (content_type or "").lower()
    if "json" in ct:
        return "json"
    if "html" in ct or body[:200].lower().lstrip().startswith((b"<!doctype html", b"<html")):
        return "html"
    if "csv" in ct:
        return "csv"
    if "pdf" in ct:
        return "pdf"
    if "spreadsheet" in ct or "excel" in ct:
        return "xlsx"
    if "xml" in ct:
        return "xml"
    return "other" if body else "empty"


def extract_links(page_html: str, base_url: str) -> list[tuple[str, str]]:
    """Every href/src on a page as (absolute url, anchor text), in order, deduplicated."""
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for m in re.finditer(
        r"<a\b[^>]*?href\s*=\s*[\"']([^\"'#]+)[\"'][^>]*>(.*?)</a>",
        page_html, re.IGNORECASE | re.DOTALL,
    ):
        _add_link(out, seen, m.group(1), m.group(2), base_url)
    for m in re.finditer(r"(?:src|data-url|data-src)\s*=\s*[\"']([^\"'#]+)[\"']", page_html, re.IGNORECASE):
        _add_link(out, seen, m.group(1), "", base_url)
    for m in re.finditer(r"https?://[^\s\"'<>)]+", page_html):
        _add_link(out, seen, m.group(0), "", base_url)
    return out


def _add_link(out: list, seen: set, href: str, anchor: str, base_url: str) -> None:
    href = html.unescape(href.strip())
    if href.lower().startswith(("javascript:", "mailto:", "tel:")):
        return
    url = urljoin(base_url, href)
    if not url.startswith(("http://", "https://")) or url in seen:
        return
    seen.add(url)
    text = re.sub(r"<[^>]+>", " ", anchor or "")
    out.append((url, re.sub(r"\s+", " ", html.unescape(text)).strip()))


def worth_following(url: str, anchor: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    if not host.endswith(FOLLOW_HOST_SUFFIXES):
        return False
    path = urlparse(url).path.lower()
    if path.endswith((".png", ".jpg", ".jpeg", ".gif", ".svg", ".css", ".ico", ".woff", ".woff2", ".js")):
        return False
    return bool(LINK_KEYWORDS.search(url) or LINK_KEYWORDS.search(anchor or ""))


def guide_lines(text: str, limit: int = 200) -> list[str]:
    """Lines of extracted PDF text that name an address or a data route."""
    out: list[str] = []
    for raw in text.splitlines():
        line = re.sub(r"\s+", " ", raw).strip()
        if len(line) < 4:
            continue
        if GUIDE_LINE_KEYWORDS.search(line):
            out.append(line[:300])
            if len(out) >= limit:
                break
    return out


def warning_entries(payload: Any) -> list[dict]:
    """Dicts anywhere in a JSON payload whose string values name 'warning'."""
    found: list[dict] = []

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            if any(isinstance(v, str) and "warning" in v.lower() for v in obj.values()):
                found.append(obj)
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)

    walk(payload)
    return found


def warning_identifiers(entries: Iterable[dict]) -> dict[str, list]:
    """Workflow ids and group names from entries naming 'warning'."""
    ids: list = []
    groups: list[str] = []
    for e in entries:
        for k, v in e.items():
            kl = k.lower()
            if kl in ("workflowid", "id") and isinstance(v, (int, str)) and str(v).strip():
                if v not in ids:
                    ids.append(v)
            if kl in ("workflowgroupname", "groupname", "name", "system") and isinstance(v, str):
                if "warning" in v.lower() and v not in groups:
                    groups.append(v)
    return {"workflow_ids": ids, "groups": groups}


def looks_like_warning_data(kind: str, body: bytes) -> bool:
    if kind not in ("json", "csv", "other"):
        return False
    text = body[:200_000].decode("utf-8", errors="replace")
    return bool(ISO3_TOKEN.search(text) and WARNING_DATA_WORDS.search(text))


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")[:60] or "response"


# ---------------------------------------------------------------------------
# Network half
# ---------------------------------------------------------------------------

class Prober:
    def __init__(self, out_dir: Path, get: Callable[..., requests.Response] | None = None,
                 timeout: int = 45) -> None:
        self.out_dir = out_dir
        self.raw_dir = out_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": USER_AGENT,
            "Accept": "application/json, text/html;q=0.9, */*;q=0.8",
        })
        self._get = get or self.session.get
        self.probes: list[Probe] = []
        self.bodies: dict[str, bytes] = {}

    def fetch(self, label: str, url: str) -> Probe:
        p = Probe(label=label, url=url)
        self.probes.append(p)
        try:
            resp = self._get(url, timeout=self.timeout, allow_redirects=True)
        except requests.RequestException as exc:
            p.error = f"{type(exc).__name__}: {exc}"[:500]
            p.kind = "error"
            LOG.info("  %-28s %s -> %s", label, url, p.error)
            return p
        body = resp.content or b""
        p.status = resp.status_code
        p.final_url = resp.url if resp.url != url else ""
        p.content_type = resp.headers.get("Content-Type", "")
        p.n_bytes = len(body)
        p.kind = classify_kind(p.content_type, body)
        if p.kind not in ("pdf", "zip", "xlsx"):
            p.snippet = body[:SNIPPET_CHARS].decode("utf-8", errors="replace")
        p.looks_like_warning_data = looks_like_warning_data(p.kind, body)
        ext = {"json": "json", "html": "html", "pdf": "pdf", "csv": "csv", "zip": "zip",
               "xlsx": "xlsx", "xml": "xml"}.get(p.kind, "bin")
        name = f"{len(self.probes):03d}_{_slug(label)}.{ext}"
        (self.raw_dir / name).write_bytes(body[:MAX_SAVE_BYTES])
        if len(body) > MAX_SAVE_BYTES:
            p.notes.append(f"saved first {MAX_SAVE_BYTES} of {len(body)} bytes")
        p.saved_as = f"raw/{name}"
        self.bodies[url] = body
        LOG.info("  %-28s %s -> %s %s %d bytes", label, url, p.status, p.kind, p.n_bytes)
        return p

    def json_of(self, p: Probe) -> Any:
        if p.kind != "json":
            return None
        try:
            return json.loads(self.bodies.get(p.url, b"").decode("utf-8", errors="replace"))
        except ValueError:
            return None


def pdf_text(body: bytes) -> tuple[str, str]:
    """(text, note). pypdf is optional; without it the PDF is only saved."""
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError:
        return "", "pypdf not installed; PDF saved but not read"
    try:
        reader = PdfReader(io.BytesIO(body))
        return "\n".join((page.extract_text() or "") for page in reader.pages), ""
    except Exception as exc:  # a malformed PDF is a finding, not a crash
        return "", f"pypdf failed: {type(exc).__name__}: {exc}"


def run(out_dir: Path, max_follow: int = 40, extra_urls: list[str] | None = None,
        get: Callable[..., requests.Response] | None = None) -> dict:
    pr = Prober(out_dir, get=get)
    report: dict[str, Any] = {
        "api_key_configured": bool(os.environ.get("INFORM_WARNING_API_KEY")),
        "landing": [], "links": [], "pdfs": [], "risk_api": {}, "catalogues": [],
        "followed": [], "extra": [],
    }

    LOG.info("1. Landing pages")
    candidates: list[tuple[str, str, str]] = []
    for label, url in LANDING_PAGES:
        p = pr.fetch(label, url)
        report["landing"].append(p.label)
        if p.kind == "html":
            page = pr.bodies[url].decode("utf-8", errors="replace")
            for link, anchor in extract_links(page, p.final_url or url):
                report["links"].append({"from": label, "url": link, "anchor": anchor[:120],
                                        "follow": worth_following(link, anchor)})
                if worth_following(link, anchor):
                    candidates.append((label, link, anchor))

    LOG.info("2. PDFs")
    for label, url in PDFS:
        p = pr.fetch(label, url)
        entry: dict[str, Any] = {"label": label, "ok": p.kind == "pdf"}
        if p.kind == "pdf":
            text, note = pdf_text(pr.bodies[url])
            if note:
                entry["note"] = note
            if text:
                (out_dir / f"{label}.txt").write_text(text, encoding="utf-8")
                entry["text_file"] = f"{label}.txt"
                entry["n_chars"] = len(text)
                entry["lines"] = guide_lines(text)
                for u in re.findall(r"https?://[^\s\"'<>)\]]+", text):
                    u = u.rstrip(".,;")
                    if worth_following(u, ""):
                        candidates.append((label, u, ""))
        report["pdfs"].append(entry)

    LOG.info("3. INFORM Risk API workflows")
    api: dict[str, Any] = {"listings": [], "warning_entries": [], "identifiers": {}, "followups": []}
    all_entries: list[dict] = []
    for route in WORKFLOW_LISTINGS:
        p = pr.fetch(f"api_{route}", f"{API_ROOT}/{route}")
        payload = pr.json_of(p)
        n = len(payload) if isinstance(payload, list) else (1 if payload else 0)
        api["listings"].append({"route": route, "status": p.status, "kind": p.kind, "n_items": n})
        if payload is not None:
            all_entries.extend(warning_entries(payload))
    api["warning_entries"] = all_entries[:50]
    ids = warning_identifiers(all_entries)
    api["identifiers"] = ids
    for group in ids["groups"][:5]:
        for route in (f"Workflows/GetByWorkflowGroup/{group}",
                      f"Workflows/GetByWorkflowGroup?WorkflowGroupName={group}",
                      f"Workflows/GetBySystem/{group}"):
            p = pr.fetch("api_group", f"{API_ROOT}/{route}")
            payload = pr.json_of(p)
            if payload is not None:
                more = warning_identifiers(warning_entries(payload) or
                                           (payload if isinstance(payload, list) else []))
                for wid in more["workflow_ids"]:
                    if wid not in ids["workflow_ids"]:
                        ids["workflow_ids"].append(wid)
            api["followups"].append({"route": route, "status": p.status, "kind": p.kind})
    for wid in ids["workflow_ids"][:3]:
        for route in (f"Countries/Scores/?WorkflowId={wid}&Iso3=SOM",
                      f"Countries/Scores/?WorkflowId={wid}"):
            p = pr.fetch("api_scores", f"{API_ROOT}/{route}")
            api["followups"].append({"route": route, "status": p.status, "kind": p.kind,
                                     "looks_like_warning_data": p.looks_like_warning_data})
    report["risk_api"] = api

    LOG.info("4. Catalogues")
    for label, url in CATALOGUE_SEARCHES:
        p = pr.fetch(label, url)
        entry = {"label": label, "status": p.status, "kind": p.kind}
        payload = pr.json_of(p)
        if isinstance(payload, dict) and isinstance(payload.get("result"), dict):
            results = payload["result"].get("results") or []
            entry["datasets"] = [
                {"name": r.get("name"), "title": r.get("title"),
                 "resources": [res.get("url") for res in (r.get("resources") or [])][:10]}
                for r in results[:20]
            ]
            for r in results[:20]:
                if "warning" in (r.get("title") or "").lower():
                    for res in (r.get("resources") or [])[:5]:
                        if res.get("url"):
                            candidates.append((label, res["url"], r.get("title") or ""))
        report["catalogues"].append(entry)

    LOG.info("5. One hop along %d candidate link(s), max %d", len(candidates), max_follow)
    fetched = {p.url for p in pr.probes}
    n = 0
    for origin, url, anchor in candidates:
        if n >= max_follow:
            report["followed_truncated"] = len(candidates) - n
            break
        if url in fetched:
            continue
        fetched.add(url)
        n += 1
        p = pr.fetch(f"follow_{origin}", url)
        report["followed"].append(p.url)

    for url in extra_urls or []:
        url = url.strip()
        if url:
            pr.fetch("extra", url)
            report["extra"].append(url)

    report["probes"] = [asdict(p) for p in pr.probes]
    report["data_candidates"] = [
        {"url": p.url, "status": p.status, "kind": p.kind, "saved_as": p.saved_as}
        for p in pr.probes if p.looks_like_warning_data and (p.status or 0) < 400
    ]
    report["reachability"] = reachability(pr.probes)
    return report


def reachability(probes: list[Probe]) -> dict[str, str]:
    """Per host: the best outcome any request to it had."""
    out: dict[str, str] = {}
    rank = {"ok": 3, "http_error": 2, "unreachable": 1}
    for p in probes:
        host = urlparse(p.url).hostname or "?"
        if p.kind == "error":
            state = "unreachable"
        elif p.status and p.status < 400:
            state = "ok"
        else:
            state = "http_error"
        if rank[state] > rank.get(out.get(host, ""), 0):
            out[host] = state
    return out


def render_markdown(report: dict) -> str:
    L: list[str] = ["# INFORM Warning probe", ""]
    L.append(f"`INFORM_WARNING_API_KEY` configured: {'yes' if report.get('api_key_configured') else 'no'} "
             "(the probe sends no credential either way)")
    L += ["", "## Hosts", "", "| host | best outcome |", "|---|---|"]
    for host, state in sorted(report.get("reachability", {}).items()):
        L.append(f"| {host} | {state} |")

    L += ["", "## Every request", "", "| label | status | kind | bytes | url | saved |", "|---|---|---|---|---|---|"]
    for p in report.get("probes", []):
        status = p["status"] if p["status"] is not None else p["error"][:60]
        L.append(f"| {p['label']} | {status} | {p['kind']} | {p['n_bytes']} | {p['url']} | {p['saved_as']} |")

    cands = report.get("data_candidates") or []
    L += ["", "## Responses that look like Warning data", ""]
    if cands:
        for c in cands:
            L.append(f"- {c['url']} ({c['kind']}, HTTP {c['status']}, `{c['saved_as']}`)")
    else:
        L.append("None. No response carried ISO3 codes beside Warning vocabulary.")

    api = report.get("risk_api") or {}
    L += ["", "## INFORM Risk API", "", "| route | status | kind | items |", "|---|---|---|---|"]
    for row in api.get("listings", []):
        L.append(f"| {row['route']} | {row['status']} | {row['kind']} | {row['n_items']} |")
    ids = api.get("identifiers") or {}
    L.append("")
    L.append(f"Entries naming 'warning': {len(api.get('warning_entries') or [])}; "
             f"groups: {ids.get('groups') or 'none'}; workflow ids: {ids.get('workflow_ids') or 'none'}")
    for row in api.get("followups", []):
        L.append(f"- {row['route']} -> {row['status']} {row['kind']}")

    L += ["", "## PDFs", ""]
    for entry in report.get("pdfs", []):
        L.append(f"### {entry['label']}")
        if not entry.get("ok"):
            L.append("Not retrieved as a PDF.")
            continue
        if entry.get("note"):
            L.append(entry["note"])
        for line in entry.get("lines", []):
            L.append(f"> {line}")
        L.append("")

    L += ["## Catalogues", ""]
    for entry in report.get("catalogues", []):
        outcome = f"HTTP {entry['status']} {entry['kind']}" if entry["status"] else "unreachable"
        L.append(f"- {entry['label']}: {outcome}")
        for d in entry.get("datasets", []) or []:
            L.append(f"  - {d.get('title')} (`{d.get('name')}`)")

    links = [l for l in report.get("links", []) if l.get("follow")]
    L += ["", f"## Data-looking links on the landing pages ({len(links)})", ""]
    for l in links[:120]:
        L.append(f"- [{l['from']}] {l['url']}  {l['anchor']}")
    if report.get("followed_truncated"):
        L.append(f"\n{report['followed_truncated']} candidate link(s) not followed (cap reached).")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--out-dir", default="diagnostics/inform_warning")
    ap.add_argument("--max-follow", type=int, default=40)
    ap.add_argument("--extra-urls", default="")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(message)s", stream=sys.stderr)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        report = run(out_dir, max_follow=args.max_follow,
                     extra_urls=[u for u in args.extra_urls.split(",") if u.strip()])
    except Exception as exc:  # the probe must never fail the run
        LOG.exception("probe crashed")
        report = {"crashed": f"{type(exc).__name__}: {exc}"}
        (out_dir / "inform_warning_probe.json").write_text(json.dumps(report, indent=2))
        print(f"# INFORM Warning probe\n\nThe probe crashed: {report['crashed']}")
        return 0
    (out_dir / "inform_warning_probe.json").write_text(json.dumps(report, indent=2, default=str))
    md = render_markdown(report)
    (out_dir / "inform_warning_probe.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
