# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import importlib


def test_import_summarizer_module() -> None:
    module = importlib.import_module("scripts.ci.summarize_connectors")
    assert hasattr(module, "render_summary_md"), "render_summary_md should exist on summarizer"
