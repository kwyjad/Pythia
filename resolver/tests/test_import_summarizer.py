from __future__ import annotations

import importlib


def test_import_summarizer_module() -> None:
    module = importlib.import_module("scripts.ci.summarize_connectors")
    assert hasattr(module, "render_summary_md"), "render_summary_md should exist on summarizer"


def test_import_dtm_deep_dive_compat() -> None:
    module = importlib.import_module("scripts.ci.summarize_connectors")
    compat = getattr(module, "_render_dtm_deep_dive", None)
    assert callable(compat), "_render_dtm_deep_dive should remain callable"
