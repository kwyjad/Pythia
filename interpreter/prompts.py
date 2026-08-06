# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Prompt assembly from the versioned templates in interpreter/templates/."""

from __future__ import annotations

import hashlib
from pathlib import Path

from interpreter import config, lexicon, schema

_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


def _template(version: str, name: str) -> str:
    path = _TEMPLATES_DIR / version / name
    if not path.exists():
        raise FileNotFoundError(
            f"interpreter template {version}/{name} not found at {path}"
        )
    return path.read_text(encoding="utf-8")


def build_system_prompt(version: str | None = None) -> str:
    version = version or config.template_version()
    text = _template(version, "system.md")
    text = text.replace("{{LEXICON_TABLE}}", lexicon.markdown_table())
    text = text.replace("{{OUTPUT_SCHEMA}}", schema.schema_json())
    text = text.replace("{{REPORT_SKELETON}}", _template(version, "report_skeleton.md"))
    return text


def build_user_prompt(
    *,
    kind: str,
    pack_text: str,
    version: str | None = None,
    scored_section: str | None = None,
) -> str:
    """The kind-specific task prompt wrapped around the pack text.

    ``scored_section`` (combined kind only): the Part B material — either the
    latest scored interpretation's content, or the explicit
    no-scored-run-available instruction.
    """
    version = version or config.template_version()
    if kind == "scored":
        text = _template(version, "scored_run.md")
        return text.replace("{{PACK}}", pack_text)

    text = _template(version, "current_run.md")
    text = text.replace("{{KIND}}", kind)
    text = text.replace("{{TOP_N}}", str(config.top_n()))
    if kind == "combined":
        if scored_section:
            section = (
                "PART B (performance): a scored-run interpretation already "
                "exists; fold its substance into your `performance` section "
                "(you may restate its placeholders verbatim — they resolve "
                "against the scored pack):\n\n" + scored_section
            )
        else:
            section = (
                "PART B (performance): NO scored run exists yet. Set "
                "`performance.plain_summary` to a plain statement that no "
                "forecast window has resolved and been scored yet, so "
                "performance cannot be reported this cycle. Omit the other "
                "performance fields."
            )
    else:
        section = (
            "This is a current-run-only interpretation: omit the "
            "`performance` section entirely."
        )
    return text.replace("{{SCORED_SECTION}}", section).replace("{{PACK}}", pack_text)


def prompt_hash(system_prompt: str, user_prompt: str) -> str:
    h = hashlib.sha256()
    h.update(system_prompt.encode("utf-8"))
    h.update(b"\x00")
    h.update(user_prompt.encode("utf-8"))
    return h.hexdigest()[:16]
