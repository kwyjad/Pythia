# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Rulebook loader for the resolution machine.

All thresholds and parameters come from ``rulebook.yaml`` next to this
module — hard rule 5: never hard-code them.  The loader is a thin dict
wrapper with dotted-path access so call sites read like the rulebook::

    rb = load_rulebook()
    rb["cyclone.buffer_km"]        # KeyError if missing — no silent defaults
    rb.get("cyclone.buffer_km")    # None if missing

Missing keys raise (via ``[]``) rather than falling back to code-side
defaults: a threshold that silently reverts to a hard-coded value is
exactly what the rulebook exists to prevent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

RULEBOOK_PATH = Path(__file__).resolve().parent / "rulebook.yaml"


class Rulebook:
    """Read-only dotted-path view over the parsed rulebook YAML."""

    def __init__(self, data: dict[str, Any], path: Path) -> None:
        self._data = data
        self.path = path

    def __getitem__(self, dotted_key: str) -> Any:
        node: Any = self._data
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                raise KeyError(
                    f"rulebook key '{dotted_key}' missing from {self.path}"
                )
            node = node[part]
        return node

    def get(self, dotted_key: str, default: Any = None) -> Any:
        try:
            return self[dotted_key]
        except KeyError:
            return default

    @property
    def version(self) -> str:
        return str(self.get("version", "unversioned"))

    def as_dict(self) -> dict[str, Any]:
        return self._data


def load_rulebook(path: Path | str | None = None) -> Rulebook:
    """Load ``rulebook.yaml`` (or an explicit override path, for tests)."""
    rb_path = Path(path) if path else RULEBOOK_PATH
    with open(rb_path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"rulebook at {rb_path} did not parse to a mapping")
    return Rulebook(data, rb_path)
