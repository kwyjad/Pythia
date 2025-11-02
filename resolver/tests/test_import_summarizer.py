from __future__ import annotations

import importlib


def test_import_summarizer_module() -> None:
    module = importlib.import_module("scripts.ci.summarize_connectors")
    assert hasattr(module, "render_summary_md"), "render_summary_md should exist on summarizer"
