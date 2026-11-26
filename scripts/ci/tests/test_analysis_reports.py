# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""scripts/ci/analysis_reports.py: parsing, indexing and the build step.

The fixtures are the title, header table and Section 1 of the three reports
in Drive on 26 Nov 2026 (the rest of each report cut).
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import pytest

from scripts.ci import analysis_reports as ar

FIXTURES = Path(__file__).parent / "fixtures" / "analysis_reports"
SKILL = "fred_performance_2026-11.md"
RESOLVER = "fred_resolver_review_2026-11.md"
FORECAST = "fred_forecast_review_2026-10-13__fc_1791866322.md"


def _parse(name: str, text: str | None = None) -> dict:
    text = (FIXTURES / name).read_text(encoding="utf-8") if text is None else text
    rec = ar.parse_report(name, text)
    assert rec is not None
    return rec


# --- file names ---------------------------------------------------------------


@pytest.mark.parametrize("name,kind,period", [
    ("fred_performance_2026-11.md", "skill", "2026-11"),
    ("fred_resolver_review_2026-11.md", "resolver", "2026-11"),
    ("fred_forecast_review_2026-10-13__fc_1791866322.md", "forecast", "2026-10"),
])
def test_type_and_period_from_each_file_pattern(name, kind, period) -> None:
    meta = ar.parse_filename(name)
    assert meta["type"] == kind and meta["period"] == period
    assert meta["display_name"] == ar.DISPLAY_NAMES[kind]
    assert meta["rerun_of"] is None


def test_rerun_suffix_names_the_original() -> None:
    meta = ar.parse_filename("fred_forecast_review_2026-10-13__fc_1791866322__rerun_2026-11-30.md")
    assert meta["report_id"] == "fred_forecast_review_2026-10-13__fc_1791866322__rerun_2026-11-30"
    assert meta["rerun_of"] == "fred_forecast_review_2026-10-13__fc_1791866322"
    assert meta["rerun_date"] == "2026-11-30"
    meta = ar.parse_filename("fred_performance_2026-11__rerun_2026-12-02.md")
    assert meta["rerun_of"] == "fred_performance_2026-11" and meta["period"] == "2026-11"


@pytest.mark.parametrize("name", [
    "forecast_review_template.md", "fred_performance_2026-11.txt", "fred_performance_2026-1.md",
    "fred_forecast_review_2026-10-13.md", "notes.md",
])
def test_non_matching_names_are_skipped(name) -> None:
    assert ar.parse_filename(name) is None
    assert ar.parse_report(name, "# x") is None


# --- content ------------------------------------------------------------------


def test_scoring_chain_form() -> None:
    rec = _parse(SKILL)
    assert rec["chain"] == {"date": "2026-11-11", "kind": "scoring", "fc_run_id": None,
                            "publish_run_id": "43009722469",
                            "folder_name": "2026-11-11__scoring__gh43009722469"}
    assert _parse(RESOLVER)["chain"]["publish_run_id"] == "43009722469"


def test_forecast_chain_form() -> None:
    chain = _parse(FORECAST)["chain"]
    assert chain == {"date": "2026-10-13", "kind": "forecast", "fc_run_id": "fc_1791866322",
                     "publish_run_id": "38355281121",
                     "folder_name": "2026-10-13__forecast__fc_1791866322__gh38355281121"}


def test_backslash_escaped_markdown_still_parses() -> None:
    text = (FIXTURES / FORECAST).read_text(encoding="utf-8").replace("_", "\\_").replace("# ", "\\# ", 1)
    rec = _parse(FORECAST, text)
    assert rec["chain"]["fc_run_id"] == "fc_1791866322"
    assert rec["title"].startswith("Fred forecast pipeline review")


def test_dry_run_detection() -> None:
    assert _parse(SKILL)["dry_run"] is True
    assert _parse(RESOLVER)["dry_run"] is True
    assert _parse(FORECAST)["dry_run"] is False


def test_title_template_and_prepared() -> None:
    rec = _parse(FORECAST)
    assert rec["title"] == "Fred forecast pipeline review: 2026-10-13 (fc_1791866322)"
    assert rec["template_version"] == "1.0 (2026-11-24)"
    assert rec["prepared"] == "2026-11-25"
    assert _parse(SKILL)["prepared"] == "2026-11-20"


def test_summary_is_cut_at_a_sentence_under_320() -> None:
    rec = _parse(SKILL)
    assert len(rec["summary"]) < 320
    assert rec["summary"].startswith("Fred lost to climatology on conflict fatalities")
    assert rec["summary"].endswith(".")
    assert _parse(RESOLVER)["summary"] == (
        "The chain ran end to end and every hand-off held, but two faults reached the ground truth.")
    assert "`" not in _parse(FORECAST)["summary"]


def test_overlong_first_sentence_is_cut_with_an_ellipsis() -> None:
    long = "word " * 120
    out = ar.cut_summary(long.strip() + ".")
    assert out.endswith("…") and len(out) <= 320


@pytest.mark.parametrize("summary,verdict", [
    ("The run worked with faults. More.", "worked with faults"),
    ("The run worked. Nothing broke.", "worked"),
    ("The run failed at hs_submit. Nothing was published.", "failed"),
    ("The run was partial. It worked later.", None),
])
def test_verdict_from_the_first_sentence(summary, verdict) -> None:
    assert ar.extract_verdict(summary) == verdict


def test_verdict_only_on_forecast_reviews() -> None:
    assert _parse(FORECAST)["verdict"] == "worked with faults"
    assert _parse(SKILL)["verdict"] is None


def test_missing_header_table_is_still_indexed(caplog) -> None:
    text = "# Fred performance report\n\n## 1. Summary\n\nFred did fine. Really.\n"
    with caplog.at_level(logging.WARNING, logger="analysis_reports"):
        rec = ar.parse_report("fred_performance_2026-12.md", text)
    assert rec["chain"] is None and rec["period"] == "2026-12"
    assert rec["summary"] == "Fred did fine. Really."
    assert "no scoring or forecast chain" in caplog.text


# --- index and build ----------------------------------------------------------


def _src(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    (src / "2026").mkdir(parents=True)
    (src / "template").mkdir()
    for name in (SKILL, RESOLVER, FORECAST):
        shutil.copy2(FIXTURES / name, src / "2026" / name)
    (src / "template" / "fred_performance_2099-01.md").write_text("# template", encoding="utf-8")
    (src / "2026" / "scratch notes.md").write_text("x", encoding="utf-8")
    return src


def _fake_pdf(md, title, out):
    Path(out).write_bytes(b"%PDF-1.7 fake")


def test_build_writes_assets_and_index_and_skips_template(tmp_path) -> None:
    out = tmp_path / "out"
    report = ar.build(_src(tmp_path), out, env={}, renderer=_fake_pdf)
    assert report["found"] == 3 and len(report["new"]) == 3
    assert report["skipped"] == ["2026/scratch notes.md"]
    index = json.loads((out / "index.json").read_text())
    assert index["schema_version"] == 1 and index["source"] == ar.SOURCE
    assert {r["report_id"] for r in index["reports"]} == {
        "fred_performance_2026-11", "fred_resolver_review_2026-11",
        "fred_forecast_review_2026-10-13__fc_1791866322"}
    assert (out / "fred_performance_2026-11.pdf").exists()
    assert (out / "fred_performance_2026-11.md").exists()


def test_change_detection_by_sha256(tmp_path) -> None:
    src = _src(tmp_path)
    first = tmp_path / "a"
    ar.build(src, first, env={}, renderer=_fake_pdf)
    previous = json.loads((first / "index.json").read_text())
    (src / "2026" / RESOLVER).write_text(
        (src / "2026" / RESOLVER).read_text() + "\nOne more line.\n", encoding="utf-8")
    second = tmp_path / "b"
    report = ar.build(src, second, previous_index=previous, env={}, renderer=_fake_pdf)
    assert report["changed"] == ["fred_resolver_review_2026-11"]
    assert sorted(report["unchanged"]) == ["fred_forecast_review_2026-10-13__fc_1791866322",
                                           "fred_performance_2026-11"]
    assert sorted(report["uploads"]) == ["fred_resolver_review_2026-11.md",
                                         "fred_resolver_review_2026-11.pdf"]
    forced = ar.build(src, tmp_path / "c", previous_index=previous, env={},
                      renderer=_fake_pdf, force_rebuild=True)
    assert len(forced["changed"]) == 3


def test_report_gone_from_drive_is_kept(tmp_path) -> None:
    src = _src(tmp_path)
    first = tmp_path / "a"
    ar.build(src, first, env={}, renderer=_fake_pdf)
    previous = json.loads((first / "index.json").read_text())
    (src / "2026" / SKILL).unlink()
    report = ar.build(src, tmp_path / "b", previous_index=previous, env={}, renderer=_fake_pdf)
    assert report["removed_from_drive"] == ["fred_performance_2026-11"]
    index = json.loads((tmp_path / "b" / "index.json").read_text())
    gone = [r for r in index["reports"] if r["report_id"] == "fred_performance_2026-11"][0]
    assert gone["removed_from_drive"] is True


def test_a_pdf_failure_is_recorded_and_the_rest_continue(tmp_path) -> None:
    def flaky(md, title, out):
        if "resolver" in str(out):
            raise RuntimeError("pango missing")
        _fake_pdf(md, title, out)

    out = tmp_path / "out"
    report = ar.build(_src(tmp_path), out, env={}, renderer=flaky)
    assert list(report["pdf_failures"]) == ["fred_resolver_review_2026-11"]
    index = {r["report_id"]: r for r in json.loads((out / "index.json").read_text())["reports"]}
    assert index["fred_resolver_review_2026-11"]["pdf_asset"] is None
    assert index["fred_performance_2026-11"]["pdf_asset"] == "fred_performance_2026-11.pdf"


def test_a_report_carrying_a_credential_is_blocked(tmp_path) -> None:
    src = _src(tmp_path)
    secret = "ghp_" + "A" * 36
    (src / "2026" / RESOLVER).write_text(
        (src / "2026" / RESOLVER).read_text() + f"\nToken {secret}\n", encoding="utf-8")
    out = tmp_path / "out"
    report = ar.build(src, out, env={"GH_TOKEN": secret}, renderer=_fake_pdf)
    assert list(report["blocked"]) == ["fred_resolver_review_2026-11"]
    assert not (out / "fred_resolver_review_2026-11.md").exists()
    assert secret not in json.dumps(report)
    assert "fred_resolver_review_2026-11" in ar.summary_markdown(report)


def test_modtimes_from_rclone_lsjson(tmp_path) -> None:
    lsjson = tmp_path / "ls.json"
    lsjson.write_text(json.dumps([{"Path": f"2026/{FORECAST}", "Name": FORECAST,
                                   "ModTime": "2026-11-25T08:05:42Z"}]))
    out = tmp_path / "out"
    ar.build(_src(tmp_path), out, env={}, renderer=_fake_pdf, modtimes=ar._load_modtimes(lsjson))
    index = {r["report_id"]: r for r in json.loads((out / "index.json").read_text())["reports"]}
    assert index[FORECAST[:-3]]["drive_modified"] == "2026-11-25T08:05:42Z"


def test_report_html_renders_nested_lists_and_contents() -> None:
    pytest.importorskip("duckdb")  # interpreter.pdf's package imports
    html = ar.report_html((FIXTURES / RESOLVER).read_text(encoding="utf-8"), "t")
    assert "<ul><li><strong>Wrong figures" in html
    assert "<ul><li>Pakistan" in html
