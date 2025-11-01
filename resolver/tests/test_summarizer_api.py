from scripts.ci.summarize_connectors import SUMMARY_TITLE, build_markdown, load_report


def test_public_api_present():
    assert callable(load_report)
    assert callable(build_markdown)
    assert isinstance(SUMMARY_TITLE, str)
    assert SUMMARY_TITLE.startswith("# ")


def test_fmt_count_backcompat():
    from scripts.ci.summary import _fmt_count

    assert _fmt_count(None) == "-"
    assert _fmt_count(0) == "-"
    assert _fmt_count(3) == "3"
