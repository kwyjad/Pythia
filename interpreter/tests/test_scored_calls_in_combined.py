# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""A combined report may cite the scored report's earlier-epoch questions.

On 13 Oct 2026 the combined report (int_combined_fc_1791866322_v1) failed
validation twelve times over: its performance block copied the stored
scored report's best and worst calls, which name questions of the 2026-08
and 2026-09 epochs, and the referential check knew only the current pack.
The correction pass could not fix it, because the ids were right.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb")

import interpreter.run as run_mod
from interpreter import validate
from interpreter.run import run_interpreter
from interpreter.tests.test_run_end_to_end import (  # noqa: F401 — fixtures
    QID,
    _content,
    current_bundle,
    mock_model,
    scored_bundle,
)

EARLIER = "KEN_DR_PHASE3PLUS_IN_NEED_2026-09"


def _scored_bundle_with(root: Path, qid: str) -> Path:
    (root / "questions_index.csv").write_text(
        f"question_id,iso3,hazard_code,metric\n{qid},KEN,DR,PHASE3PLUS_IN_NEED\n", encoding="utf-8")
    return root


def test_a_performance_citation_of_an_earlier_epoch_question_passes(
    tmp_path, current_bundle, scored_bundle, mock_model  # noqa: F811
):
    db = tmp_path / "p.duckdb"
    scored = _content("scored")
    scored["performance"]["best_calls"] = [
        {"question_ids": [EARLIER], "what_was_right": "The drought call held."}]
    mock_model.response = json.dumps(scored)
    assert run_interpreter(db=str(db), kind="scored",
                           pack_path=str(_scored_bundle_with(scored_bundle, EARLIER)))["status"] == "ok"

    combined = _content("combined")
    combined["performance"]["best_calls"] = [
        {"question_ids": [EARLIER], "what_was_right": "The drought call held."}]
    mock_model.response = json.dumps(combined)
    result = run_interpreter(db=str(db), kind="combined", pack_path=str(current_bundle))
    assert result["status"] == "ok"


def test_an_attention_citation_must_still_be_in_the_current_pack():
    content = {"attention": [{"question_ids": [EARLIER]}],
               "performance": {"best_calls": [{"question_ids": [EARLIER]}]}}
    res = validate.check_referential(content, valid_question_ids={QID}, per_question={},
                                     global_figures={}, performance_question_ids={EARLIER})
    assert not res.passed
    assert res.errors == [f"attention[0]: question_id {EARLIER!r} is not in the pack"]


def test_an_id_the_scored_report_never_cited_still_fails():
    content = {"performance": {"worst_calls": [{"question_ids": ["XXX_ACE_PA_2026-01"]}]}}
    res = validate.check_referential(content, valid_question_ids={QID}, per_question={},
                                     global_figures={}, performance_question_ids={EARLIER})
    assert not res.passed


def test_scored_question_ids_reads_every_citation():
    content = {"performance": {"best_calls": [{"question_ids": ["A"]}],
                               "worst_calls": [{"question_ids": ["B", "C"]}]}}
    assert validate.scored_question_ids(content) == {"A", "B", "C"}
    assert validate.scored_question_ids(None) == set()
