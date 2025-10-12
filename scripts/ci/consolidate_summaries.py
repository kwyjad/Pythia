#!/usr/bin/env python3
"""Merge per-job AI summaries into a single SUMMARY.md artifact."""
from __future__ import annotations

import datetime as _dt
import os
import sys
import tarfile
from pathlib import Path
from typing import List, Sequence, Tuple


def _read_summary_from_dir(directory: Path) -> List[Tuple[str, str]]:
    """Return (name, content) tuples discovered inside a job artifact dir."""
    summaries: List[Tuple[str, str]] = []
    diagnostics_path = directory / ".ci" / "diagnostics" / "SUMMARY.md"
    summary_file = directory / "SUMMARY.md"

    if diagnostics_path.exists():
        summaries.append((directory.name, diagnostics_path.read_text(encoding="utf-8", errors="replace")))
        return summaries

    if summary_file.exists():
        summaries.append((directory.name, summary_file.read_text(encoding="utf-8", errors="replace")))
        return summaries

    errors: List[str] = []
    for tar_path in sorted(directory.glob("diagnostics_*.tar.gz")):
        try:
            with tarfile.open(tar_path, "r:gz") as tar:
                member = tar.getmember(".ci/diagnostics/SUMMARY.md")
                extracted = tar.extractfile(member)
                if extracted is None:
                    continue
                content = extracted.read().decode("utf-8", errors="replace")
                summaries.append((directory.name, content))
                return summaries
        except KeyError:
            continue
        except Exception as exc:
            errors.append(f"Failed to read {tar_path.name}: {exc}")

    if errors:
        joined = "\n".join(errors)
        summaries.append((directory.name, f"No SUMMARY.md found; errors encountered while inspecting tarballs:\n\n````text\n{joined}\n````"))
    return summaries


def discover_summaries(downloads_dir: Path) -> Tuple[List[Tuple[str, str]], List[str]]:
    collected: List[Tuple[str, str]] = []
    visited: List[str] = []
    if not downloads_dir.exists():
        return collected, visited

    for child in sorted(downloads_dir.iterdir()):
        if child.is_file():
            continue
        visited.append(child.name)
        collected.extend(_read_summary_from_dir(child))
    return collected, visited


def build_summary(collected: Sequence[Tuple[str, str]], visited: Sequence[str]) -> str:
    now = _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z")
    env = os.environ
    run_id = env.get("GITHUB_RUN_ID", "unknown")
    attempt = env.get("GITHUB_RUN_ATTEMPT", "unknown")
    commit = env.get("GITHUB_SHA", "unknown")

    lines: List[str] = []
    lines.append("# Consolidated CI Diagnostics")
    lines.append("")
    lines.append("## Run Metadata")
    lines.append("")
    lines.append(f"- **Run ID:** {run_id}")
    lines.append(f"- **Attempt:** {attempt}")
    lines.append(f"- **Commit:** {commit}")
    lines.append(f"- **Generated:** {now}")
    lines.append("")

    if not collected:
        lines.append("No job summaries were discovered in downloaded artifacts.")
        if visited:
            lines.append("")
            lines.append("Visited artifact directories:")
            for name in visited:
                lines.append(f"- {name}")
        lines.append("")
        return "\n".join(lines)

    lines.append("## Table of Contents")
    lines.append("")
    for name, _ in collected:
        anchor = name.lower().replace(" ", "-").replace("/", "-").replace(":", "-")
        lines.append(f"- [{name}](#{anchor})")
    lines.append("")

    seen_names = set()
    for name, content in collected:
        anchor = name.lower().replace(" ", "-").replace("/", "-").replace(":", "-")
        lines.append(f"## {name}")
        lines.append("<a id=\"" + anchor + "\"></a>")
        lines.append("")
        lines.append(content.rstrip())
        lines.append("")
        seen_names.add(name.split(":", 1)[0])

    missing_dirs = [name for name in visited if name not in seen_names]
    if missing_dirs:
        lines.append("## Missing Summaries")
        lines.append("")
        for name in missing_dirs:
            lines.append(f"- {name}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    downloads_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./downloads")
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("./consolidated")
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        collected, visited = discover_summaries(downloads_dir)
    except Exception as exc:
        collected = [("collection-error", f"Failed to collect summaries: {exc}")]
        visited = []

    content = build_summary(collected, visited)
    output_path = out_dir / "SUMMARY.md"
    output_path.write_text(content, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
