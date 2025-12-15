#!/usr/bin/env python3
# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import os
from typing import List, Sequence, Tuple
import xml.etree.ElementTree as ET

_MAX_FAILURE_SNIPPET = 1200


def _format_failure_text(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    if len(text) <= _MAX_FAILURE_SNIPPET:
        return text
    head = text[:_MAX_FAILURE_SNIPPET]
    return head + "\n...[truncated]..."


def junit_reports_to_markdown(paths: Sequence[os.PathLike[str] | str]) -> Tuple[str, int, int, int]:
    """Return (markdown, total, failures, errors) for the provided JUnit report paths."""
    sections: List[str] = []
    total = 0
    failures = 0
    errors = 0

    for raw_path in paths:
        path = os.fspath(raw_path)
        if not os.path.exists(path):
            sections.append(f"- _missing_: `{os.path.basename(path)}`")
            continue
        try:
            tree = ET.parse(path)
            root = tree.getroot()
        except Exception as exc:  # pragma: no cover - defensive logging
            sections.append(f"- _error parsing `{os.path.basename(path)}`_: {exc}")
            continue

        report_total = int(root.attrib.get("tests", 0))
        report_failures = int(root.attrib.get("failures", 0))
        report_errors = int(root.attrib.get("errors", 0))
        total += report_total
        failures += report_failures
        errors += report_errors

        sections.append(
            f"- `{os.path.basename(path)}`: tests={report_total}, failures={report_failures}, errors={report_errors}"
        )

        for testcase in root.iter("testcase"):
            failing_nodes = list(testcase.iter("failure")) + list(testcase.iter("error"))
            if not failing_nodes:
                continue
            name = testcase.attrib.get("name", "?")
            classname = testcase.attrib.get("classname", "?")
            sections.append("")
            sections.append(f"**❌ {classname}::{name}**")
            for node in failing_nodes:
                message = (node.attrib.get("message") or "").strip()
                if message:
                    sections.append(f"- _message_: {message}")
                body = _format_failure_text(node.text or "")
                if body:
                    sections.append("\n```")
                    sections.append(body)
                    sections.append("```")

    if not sections:
        sections.append("_No JUnit reports found._")

    sections.append("")
    markdown = "\n".join(sections)
    return markdown, total, failures, errors


__all__ = ["junit_reports_to_markdown"]
