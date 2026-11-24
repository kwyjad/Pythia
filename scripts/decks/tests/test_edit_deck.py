# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.
"""Tests for scripts/decks/edit_deck.py, on a deck built here."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pptx = pytest.importorskip("pptx")
from pptx import Presentation  # noqa: E402
from pptx.util import Inches  # noqa: E402

from scripts.decks.edit_deck import Edit, apply_edits, dump, main  # noqa: E402


def _build(path: Path) -> Path:
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    tf = slide.shapes.add_textbox(Inches(1), Inches(1), Inches(6), Inches(2)).text_frame
    p = tf.paragraphs[0]
    label = p.add_run()
    label.text = "Cost:"
    label.font.bold = True
    body = p.add_run()
    body.text = " $60 a run"
    p2 = tf.add_paragraph()
    a, b = p2.add_run(), p2.add_run()
    a.text, b.text = "Gemini 3.", "5 Flash writes it"
    p3 = tf.add_paragraph()
    p3.add_run().text = "Runs on the 28th and the 28th again"
    slide.notes_slide.notes_text_frame.text = "Say the cycle starts on the 28th."
    prs.save(str(path))
    return path


def _texts(path: Path) -> list[str]:
    prs = Presentation(str(path))
    tf = [s for s in prs.slides[0].shapes if s.has_text_frame][0].text_frame
    return [para.text for para in tf.paragraphs]


def _runs(path: Path, i: int) -> list[tuple[str, bool | None]]:
    prs = Presentation(str(path))
    tf = [s for s in prs.slides[0].shapes if s.has_text_frame][0].text_frame
    return [(r.text, r.font.bold) for r in tf.paragraphs[i].runs]


def test_edit_inside_one_run_keeps_the_label_bold(tmp_path):
    deck = _build(tmp_path / "d.pptx")
    res = apply_edits(deck, [Edit(1, "Cost: $60 a run", "Cost: $25 a run")])
    assert not res.problems
    assert _runs(deck, 0) == [("Cost:", True), (" $25 a run", None)]


def test_a_phrase_across_same_format_runs_is_replaced(tmp_path):
    deck = _build(tmp_path / "d.pptx")
    res = apply_edits(deck, [Edit(1, "Gemini 3.5 Flash", "Gemini 3.6 Flash")])
    assert not res.problems
    assert _texts(deck)[1] == "Gemini 3.6 Flash writes it"
    assert [t for t, _ in _runs(deck, 1)] == ["Gemini 3.", "6 Flash writes it"]


def test_a_change_across_a_bold_label_is_refused_and_nothing_written(tmp_path):
    deck = _build(tmp_path / "d.pptx")
    before = deck.read_bytes()
    res = apply_edits(deck, [Edit(1, "Cost: $60", "Total $25")])
    assert res.problems and "different formatting" in res.problems[0]
    assert deck.read_bytes() == before


def test_an_edit_that_matches_nothing_is_reported(tmp_path):
    deck = _build(tmp_path / "d.pptx")
    before = deck.read_bytes()
    res = apply_edits(deck, [Edit(1, "Opus 5", "Opus 5.5"),
                             Edit(1, "Cost: $60 a run", "Cost: $25 a run")])
    assert any("matched nothing" in p for p in res.problems)
    assert deck.read_bytes() == before


def test_count_must_match(tmp_path):
    deck = _build(tmp_path / "d.pptx")
    res = apply_edits(deck, [Edit(1, "the 28th", "the 11th")])
    assert any("matched 2 times" in p for p in res.problems)
    res = apply_edits(deck, [Edit(1, "the 28th", "the 11th", count=2)])
    assert not res.problems
    assert _texts(deck)[2] == "Runs on the 11th and the 11th again"


def test_notes_are_edited(tmp_path):
    deck = _build(tmp_path / "d.pptx")
    res = apply_edits(deck, [Edit(1, "the 28th", "the 11th", where="notes")])
    assert not res.problems
    notes = Presentation(str(deck)).slides[0].notes_slide.notes_text_frame.text
    assert notes == "Say the cycle starts on the 11th."
    assert "the 28th again" in _texts(deck)[2]


def test_out_leaves_the_input_alone_and_the_result_reopens(tmp_path):
    deck = _build(tmp_path / "d.pptx")
    before = deck.read_bytes()
    out = tmp_path / "out.pptx"
    res = apply_edits(deck, [Edit(1, "$60", "$25")], out)
    assert not res.problems
    assert deck.read_bytes() == before
    assert "$25" in _texts(out)[0]


def test_cli_exit_codes_and_dump(tmp_path, capsys):
    deck = _build(tmp_path / "d.pptx")
    edits = tmp_path / "e.json"
    edits.write_text(json.dumps([{"slide": 1, "old": "nope", "new": "x"}]))
    assert main([str(deck), "--edits", str(edits)]) == 1
    edits.write_text(json.dumps([{"slide": 1, "old": "$60", "new": "$25"}]))
    assert main([str(deck), "--edits", str(edits)]) == 0
    assert "$25 a run" in dump(deck)
    assert "notes Say the cycle" in dump(deck, 1)


def test_the_repository_decks_open_and_carry_text():
    root = Path(__file__).resolve().parents[3] / "docs" / "decks"
    manifest = json.loads((root / "manifest.json").read_text())
    assert len(manifest["decks"]) == 8
    for d in manifest["decks"]:
        assert d["file"].startswith(f"{d['number']}_")
        assert "Fred" in dump(root / d["file"], 1)
