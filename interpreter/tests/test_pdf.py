# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Phase 6: PDF rendering — markdown subset converter, row selection,
strict-mode suppression, and the generate seam (WeasyPrint monkeypatched)."""

from __future__ import annotations

import duckdb
import pytest

from interpreter import pdf, store


def _save(con, *, kind="combined", run_id="fc_1", scored_run_id=None,
          status="ok", is_test=None, content_md="# report\n\nBody."):
    return store.save_interpretation(
        con, kind=kind, run_id=run_id, hs_run_id="hs_1",
        scored_run_id=scored_run_id, template_version="v1",
        model_name="claude-opus-5", thinking_level="high",
        prompt_hash="abc", pack_hash="def",
        content={"kind": kind}, content_md=content_md,
        figures=None, status=status, validation={}, cost_usd=1.0,
        input_tokens=10, output_tokens=5, is_test=is_test,
    )


class TestMarkdownToHtml:
    def test_headings_lists_and_paragraphs(self):
        html = pdf.markdown_to_html(
            "# Title\n\n## Section\n\n### Sub\n\nA paragraph\nspanning lines.\n\n- one\n- two\n"
        )
        assert "<h1>Title</h1>" in html
        assert "<h2>Section</h2>" in html
        assert "<h3>Sub</h3>" in html
        assert "<p>A paragraph spanning lines.</p>" in html
        assert "<ul><li>one</li><li>two</li></ul>" in html

    def test_pipe_table_with_separator_row(self):
        html = pdf.markdown_to_html(
            "| word | band |\n|---|---|\n| likely | 66-90% |\n"
        )
        assert "<th>word</th>" in html
        assert "<td>likely</td>" in html
        assert "---" not in html

    def test_inline_code_protects_underscored_question_ids(self):
        # Question ids carry interior underscores that must never become <em>.
        html = pdf.markdown_to_html("Questions: `SOM_ACE_PA_2026-08`")
        assert "<code>SOM_ACE_PA_2026-08</code>" in html
        assert "<em>" not in html

    def test_bold_italic_and_escaping(self):
        html = pdf.markdown_to_html(
            "**Headline.** *Reading:* plain _aside_ and <script>alert(1)</script>"
        )
        assert "<strong>Headline.</strong>" in html
        assert "<em>Reading:</em>" in html
        assert "<em>aside</em>" in html
        assert "<script>" not in html
        assert "&lt;script&gt;" in html


class TestBuildReportHtml:
    def test_ok_row_has_no_banner(self):
        html = pdf.build_report_html(
            {"kind": "combined", "version": 1, "status": "ok",
             "content_md": "# r", "created_at": "2026-08-06 12:00:00",
             "interpretation_id": "int_x"}
        )
        # The CSS defines .banner; the banner ELEMENT must be absent.
        assert '<div class="banner">' not in html
        assert "<h1>r</h1>" in html
        assert "v1" in html

    def test_failed_row_banners_visibly(self):
        html = pdf.build_report_html(
            {"kind": "combined", "version": 2, "status": "failed_validation",
             "content_md": "# r", "created_at": "2026-08-06", "interpretation_id": "i"}
        )
        assert "failed automated validation" in html
        assert "failed_validation" in html


class TestRowSelection:
    def _con(self, tmp_path):
        return duckdb.connect(str(tmp_path / "t.duckdb"))

    def test_latest_row_test_filtered_both_senses(self, tmp_path, monkeypatch):
        monkeypatch.delenv("PYTHIA_TEST_MODE", raising=False)
        con = self._con(tmp_path)
        _save(con, run_id="fc_prod", is_test=False)
        _save(con, run_id="fc_test", is_test=True)
        row = pdf.select_row(con, kind="combined")
        assert row["run_id"] == "fc_prod"
        row = pdf.select_row(con, kind="combined", include_test=True)
        assert row["run_id"] == "fc_test"
        con.close()

    def test_explicit_id_wins(self, tmp_path):
        con = self._con(tmp_path)
        iid, _ = _save(con, run_id="fc_a")
        _save(con, run_id="fc_b")
        row = pdf.select_row(con, interpretation_id=iid)
        assert row["run_id"] == "fc_a"
        con.close()

    def test_pre_interpreter_db_returns_none(self, tmp_path):
        con = self._con(tmp_path)
        assert pdf.select_row(con, kind="combined") is None
        con.close()


class TestFilenames:
    def test_versioned_name_from_creation_month(self):
        row = {"kind": "combined", "version": 3,
               "created_at": "2026-08-06 12:00:00", "scored_run_id": None}
        assert pdf.pdf_filename(row) == "report__2026-08__v3.pdf"

    def test_backfilled_report_is_named_for_the_run_not_today(self):
        # A July run interpreted in August must not be filed under August.
        row = {"kind": "combined", "version": 1, "hs_run_id": "hs_20260715T103033",
               "created_at": "2026-08-06 12:00:00", "scored_run_id": None}
        assert pdf.pdf_filename(row) == "report__2026-07__v1.pdf"

    def test_malformed_run_id_falls_back_to_creation_month(self):
        row = {"kind": "combined", "version": 1, "hs_run_id": "not-a-run-id",
               "created_at": "2026-08-06 12:00:00", "scored_run_id": None}
        assert pdf.pdf_filename(row) == "report__2026-08__v1.pdf"

    def test_scored_rows_use_the_round_key(self):
        row = {"kind": "scored", "version": 1,
               "created_at": "2026-09-02 00:00:00", "scored_run_id": "2026-08"}
        assert pdf.pdf_filename(row) == "report__2026-08__v1.pdf"


class TestGenerate:
    def _db(self, tmp_path):
        path = tmp_path / "gen.duckdb"
        con = duckdb.connect(str(path))
        return path, con

    def test_end_to_end_writes_versioned_and_latest(self, tmp_path, monkeypatch):
        monkeypatch.delenv("PYTHIA_TEST_MODE", raising=False)
        path, con = self._db(tmp_path)
        _save(con, is_test=False)
        con.close()
        rendered = {}

        def fake_render(html, out_path):
            rendered["html"] = html
            out_path.write_bytes(b"%PDF-fake")

        monkeypatch.setattr(pdf, "_render_pdf", fake_render)
        result = pdf.generate_pdf(db=str(path), out_dir=str(tmp_path / "out"))
        assert result["status"] == "ok"
        assert (tmp_path / "out" / "interpreter_report_latest.pdf").read_bytes() == b"%PDF-fake"
        versioned = list((tmp_path / "out").glob("report__*__v1.pdf"))
        assert len(versioned) == 1
        assert "<h1>report</h1>" in rendered["html"]

    def test_strict_mode_suppresses_failed_rows(self, tmp_path, monkeypatch):
        monkeypatch.delenv("PYTHIA_TEST_MODE", raising=False)
        monkeypatch.setenv("PYTHIA_INTERPRETER_STRICT_VALIDATION", "1")
        path, con = self._db(tmp_path)
        _save(con, status="failed_validation", is_test=False)
        con.close()
        monkeypatch.setattr(
            pdf, "_render_pdf",
            lambda *_: (_ for _ in ()).throw(AssertionError("must not render")),
        )
        result = pdf.generate_pdf(db=str(path), out_dir=str(tmp_path / "out"))
        assert result["status"] == "skipped"
        assert result["reason"] == "strict_suppressed"
        assert not (tmp_path / "out").exists()

    def test_non_strict_failed_row_renders_with_banner(self, tmp_path, monkeypatch):
        monkeypatch.delenv("PYTHIA_TEST_MODE", raising=False)
        monkeypatch.delenv("PYTHIA_INTERPRETER_STRICT_VALIDATION", raising=False)
        path, con = self._db(tmp_path)
        _save(con, status="failed_validation", is_test=False)
        con.close()
        captured = {}

        def fake_render(html, out_path):
            captured["html"] = html
            out_path.write_bytes(b"%PDF-fake")

        monkeypatch.setattr(pdf, "_render_pdf", fake_render)
        result = pdf.generate_pdf(db=str(path), out_dir=str(tmp_path / "out"))
        assert result["status"] == "ok"
        assert result["row_status"] == "failed_validation"
        assert "failed automated validation" in captured["html"]

    def test_render_failure_keeps_html_evidence_and_returns_zero(self, tmp_path, monkeypatch):
        monkeypatch.delenv("PYTHIA_TEST_MODE", raising=False)
        path, con = self._db(tmp_path)
        _save(con, is_test=False)
        con.close()
        monkeypatch.setattr(
            pdf, "_render_pdf",
            lambda *_: (_ for _ in ()).throw(RuntimeError("weasyprint exploded")),
        )
        result = pdf.generate_pdf(db=str(path), out_dir=str(tmp_path / "out"))
        assert result["status"] == "failed_render"
        assert list((tmp_path / "out").glob("*.html"))
        # main() never fails the pipeline, whatever happened.
        assert pdf.main(["--db", str(path), "--out-dir", str(tmp_path / "out")]) == 0

    def test_no_row_is_a_skip_not_an_error(self, tmp_path):
        path, con = self._db(tmp_path)
        con.close()
        result = pdf.generate_pdf(db=str(path), out_dir=str(tmp_path / "out"))
        assert result == {"status": "skipped", "reason": "no_row"}

    @pytest.mark.slow
    def test_real_weasyprint_if_installed(self, tmp_path, monkeypatch):
        pytest.importorskip("weasyprint")
        monkeypatch.delenv("PYTHIA_TEST_MODE", raising=False)
        path, con = self._db(tmp_path)
        _save(con, is_test=False)
        con.close()
        result = pdf.generate_pdf(db=str(path), out_dir=str(tmp_path / "out"))
        assert result["status"] in ("ok", "failed_render")
        if result["status"] == "ok":
            head = (tmp_path / "out" / "interpreter_report_latest.pdf").read_bytes()[:5]
            assert head == b"%PDF-"


class TestMap:
    """The printed map. The dashboard draws its own with JavaScript, which a
    PDF cannot run, so this is the only map a printed report gets."""

    def test_countries_are_filled_by_value(self):
        from interpreter import mapviz

        svg = mapviz.attention_map_svg({"ETH": 0.9, "SOM": 0.05})
        assert svg.startswith("<svg")
        assert mapviz.SCALE[-1] in svg    # the dark end is used
        assert mapviz.NO_DATA in svg      # so is "no forecast"
        assert "no forecast" in svg       # and the legend says so

    def test_scale_bands(self):
        from interpreter import mapviz

        assert mapviz.colour_for(None) == mapviz.NO_DATA
        assert mapviz.colour_for("nonsense") == mapviz.NO_DATA
        assert mapviz.colour_for(0.0) == mapviz.SCALE[0]
        assert mapviz.colour_for(0.99) == mapviz.SCALE[-1]

    def test_missing_boundaries_degrade_rather_than_fail(self, monkeypatch):
        from interpreter import mapviz

        monkeypatch.setattr(mapviz, "_features", lambda: [])
        assert mapviz.attention_map_svg({"ETH": 0.5}) == ""

    def test_captions_name_countries_not_codes(self):
        from interpreter import mapviz

        assert mapviz.country_labels({"ETH": 0.9, "NIC": 0.2}) == [
            "Ethiopia", "Nicaragua",
        ]

    def test_values_from_deviation_without_the_table_returns_nothing(self, tmp_path):
        from interpreter import mapviz

        con = duckdb.connect(str(tmp_path / "m.duckdb"))
        assert mapviz.values_from_deviation(con, "fc_1", include_test=False) == {}
        con.close()


class TestBranding:
    def test_masthead_carries_fred_and_the_month(self):
        html = pdf.build_report_html(
            {"kind": "combined", "version": 1, "status": "ok",
             "hs_run_id": "hs_20260801T000000", "content_md": "# Report"}
        )
        assert "Fred" in html
        assert "2026-08" in html
        assert pdf.FRED_PRIMARY in html

    def test_map_is_included_when_supplied_and_absent_when_not(self):
        row = {"kind": "combined", "version": 1, "status": "ok",
               "content_md": "# Report"}
        with_map = pdf.build_report_html(
            row, map_svg="<svg id='m'></svg>", map_captions=["Ethiopia"]
        )
        assert "<svg id='m'></svg>" in with_map
        assert "Ethiopia" in with_map
        assert 'class="mapblock"' not in pdf.build_report_html(row)

    def test_inline_svg_passes_through_unescaped(self):
        html = pdf.markdown_to_html('Text.\n\n<svg width="10"><rect /></svg>\n\nMore.')
        assert '<svg width="10"><rect /></svg>' in html
        assert "&lt;svg" not in html

    def test_question_links_become_anchors(self):
        html = pdf.markdown_to_html(
            "Questions: [SOM_FL_PA_2026-08](https://example.org/questions/SOM_FL_PA_2026-08)"
        )
        assert 'href="https://example.org/questions/SOM_FL_PA_2026-08"' in html
        # The underscores in the id must survive the emphasis pass.
        assert ">SOM_FL_PA_2026-08</a>" in html
