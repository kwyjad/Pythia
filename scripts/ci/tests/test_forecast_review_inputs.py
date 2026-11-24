# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The forecast pipeline review's inputs, built from a forecast chain tree.

The fixture (``fixtures/forecast_review/``) is the 13 November 2026 forecast
chain trimmed to three countries (AFG, PER, SDN): the real file shapes, the
debug, attribution and current-run bundles laid out as directories that the
test zips, and no credential fingerprints. No network, no DB.
"""

from __future__ import annotations

import copy
import json
import shutil
import zipfile
from pathlib import Path

import pytest

from scripts.ci import forecast_review_inputs as fri

FIXTURE = Path(__file__).parent / "fixtures" / "forecast_review"
FC = "fc_1794543371"


def _zip_dir(src: Path, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(src.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(src).as_posix())


def build_chain(tmp: Path, *, drop: tuple[str, ...] = (), arms: dict | None = None,
                stdout: str | None = None) -> Path:
    """The fixture chain tree with its three bundles zipped in place."""
    staging = tmp / "chain"
    shutil.copytree(FIXTURE / "chain", staging)
    work = tmp / "bundles"
    shutil.copytree(FIXTURE, work, ignore=shutil.ignore_patterns("chain"))
    for rel in drop:
        target = work / rel
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
    sib = staging / "11_sibyl__43531717339"
    _zip_dir(work / "debug", sib / "pythia-debug-bundle" / f"pythia_debug_bundle__{FC}.zip")
    _zip_dir(work / "attribution", sib / "pythia-forecast-attribution-bundle" / f"forecast_attribution__{FC}.zip")
    _zip_dir(work / "current", sib / "pythia-current-run-bundle" / "current_run_analysis__2026-11.zip")
    if arms is not None:
        p = staging / "10_pipeline_stage__43530492374" / "pythia-stage-diagnostics" / f"experiment_arms__{FC}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(arms), encoding="utf-8")
    if stdout is not None:
        p = staging / "10_pipeline_stage__43530492374" / "pythia-stage-diagnostics" / "post_run__fc_collect_finalize.txt"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(stdout, encoding="utf-8")
    return staging


def build(tmp: Path, **kwargs) -> dict:
    previous = kwargs.pop("previous", None)
    staging = build_chain(tmp, **kwargs)
    src = fri.locate(staging, tmp / "work")
    return fri.build_inputs(src, previous=previous)


def test_every_key_is_present_and_named_in_order(tmp_path):
    payload = build(tmp_path)
    for key in fri.SECTION_KEYS + ("previous_run",):
        assert key in payload, key
    for key in fri.SECTION_KEYS:
        assert not (isinstance(payload[key], dict) and payload[key].get("stub")), (key, payload[key])
    assert payload["meta"]["fc_run_id"] == FC
    assert payload["meta"]["inputs_version"] == fri.INPUTS_VERSION
    assert payload["previous_run"] == {"stub": True, "reason": "no earlier forecast_review"}


def test_counts_name_their_source(tmp_path):
    payload = build(tmp_path)
    assert "source" in payload["inputs"]["crisiswatch"]
    assert "source" in payload["horizon_scanner"]
    assert "source" in payload["evidence"]["adversarial"]
    assert all("source" in i for i in payload["inputs"]["injects"])
    assert all("source" in e for e in payload["experiments"]["experiments"] if not e.get("stub"))


def test_the_real_shapes_are_read(tmp_path):
    p = build(tmp_path)
    hs = p["horizon_scanner"]
    # AFG, PER and SDN, 4 hazards each; seasonal skips are not assessed.
    assert hs["pairs_expected"] == 12
    assert hs["pairs_assessed"] + hs["pairs_screened_out_seasonally"] == 12
    q = p["triage_questions"]["questions"]
    assert q["total"] == 11 and q["track1"] + q["track2"] == 11
    cw = p["inputs"]["crisiswatch"]
    assert cw["latest_edition"] == "2026-10"
    assert {e["iso3"] for e in cw["ace_question_countries_on_older_edition"]} == {"BRA", "GTM", "MAR"}
    assert p["sibyl"]["run"]["sibyl_run_id"] == "sibyl_1794614792427"
    assert p["sibyl"]["selection"]["eligible_pool_by_hazard_and_rc_level"]["ACE"]["2"] == 4
    assert p["outputs"]["gate"]["run"] == FC or p["outputs"]["gate"].get("stub")
    assert p["ensemble"]["expected_members"] == ["claude-opus-5-5", "gemini-3.1-pro-preview",
                                                 "gemini-3.6-flash", "gpt-6-luna", "gpt-6-sol"]
    assert p["ensemble"]["member_cells"]["spd_missing"] == 0
    assert p["track2"]["questions"] == sum(1 for _ in range(q["track2"]))
    # Health checks whose OK only says the subject is off (rule A4.4).
    assert any(c["subsystem"] == "Research Grounding" for c in p["issues"]["narrow_ok_checks"])


def test_a_missing_file_gives_a_stub_not_an_exception(tmp_path):
    payload = build(tmp_path, drop=(f"debug/rc_triage_summary__{FC}.csv", f"debug/llm_calls_detail__{FC}.jsonl.gz",
                                    "attribution/contrasts/run_over_run.csv", "current/sibyl.json"))
    assert payload["horizon_scanner"]["stub"] is True
    assert payload["run_over_run"]["stub"] is True
    assert payload["evidence"]["adversarial"]["stub"] is True
    assert "fallback_file" in payload["horizon_scanner"]
    # The rest still builds.
    assert payload["ensemble"]["member_cells"]["spd_expected"]


def test_no_bundles_at_all_still_builds(tmp_path):
    staging = tmp_path / "chain"
    shutil.copytree(FIXTURE / "chain", staging)
    payload = fri.build_inputs(fri.locate(staging, tmp_path / "w"))
    assert payload["sources"]["debug_bundle"] is None
    assert payload["horizon_scanner"]["stub"] is True
    assert payload["meta"]["hs_run_id"] == "hs_20261113T035247"
    json.dumps(payload)  # serialisable


def test_previous_run_is_embedded_and_drives_the_diffs(tmp_path):
    first = build(tmp_path / "a")
    prev = copy.deepcopy(first)
    prev["meta"]["fc_run_id"] = "fc_1791866322"
    prev["horizon_scanner"]["rc_pairs"]["AFG|ACE"] = 0          # was L0, now L2: a two-level jump
    prev["triage_questions"]["question_keys"].append("SDN|FL|PA")  # left the set
    import csv
    per_t: dict[str, set] = {}
    for row in csv.DictReader(open(FIXTURE / "attribution" / "prompts" / "token_share.csv")):
        per_t.setdefault(row["template_id"], set()).add(row["question_id"])
    tid = next(t for t, qs in per_t.items() if len(qs) > 1 and t in prev["evidence"]["section_hashes"])
    static = "Buckets"
    prev["evidence"]["section_hashes"][tid][static] = "0" * 64
    prev["cost"]["by_phase"][0]["per_question_usd"] /= 2
    now = build(tmp_path / "b", previous=prev)
    assert now["previous_run"]["meta"]["fc_run_id"] == "fc_1791866322"
    assert "previous_run" not in now["previous_run"]
    jumps = now["horizon_scanner"]["changed_since_previous"]["jumped_two_or_more"]
    assert {"pair": "AFG|ACE", "before": 0, "now": 2} in jumps
    assert now["triage_questions"]["left_since_previous"] == ["SDN|FL|PA"]
    changes = now["evidence"]["prompt_section_changes"]["changes"]
    assert any(c["template_id"] == tid and c["section"] == static for c in changes)
    assert now["cost"]["by_phase"][0]["ratio_to_previous_per_question"] == pytest.approx(2.0, rel=0.01)
    moves = now["run_over_run"]["largest_moves"]
    assert moves and all(m["new_input_in_inventory"] is not None for m in moves)


def test_a_stub_previous_is_kept_as_its_reason(tmp_path):
    payload = build(tmp_path, previous={"stub": True, "reason": "no earlier forecast_review"})
    assert payload["previous_run"]["reason"] == "no earlier forecast_review"
    assert payload["horizon_scanner"]["changed_since_previous"]["stub"] is True


# ---------------------------------------------------------------------------
# Experiments: the binomial range
# ---------------------------------------------------------------------------


def test_binomial_range_on_a_known_split():
    lo, hi = fri.binom_range(100, 0.5)
    assert (lo, hi) == (42, 58)
    # CDF(1)=0.011, CDF(2)=0.055; CDF(7)=0.945, CDF(8)=0.989
    assert fri.binom_range(10, 0.5) == (2, 8)
    inside = fri.arm_check("x", 0.5, {"a": 50, "b": 50}, "b", 0, "t")
    assert inside["inside_90pct_range"] is True and inside["range_90pct"] == [42, 58]
    outside = fri.arm_check("x", 0.5, {"a": 70, "b": 30}, "b", 0, "t")
    assert outside["inside_90pct_range"] is False
    assert "exact binomial" in outside["range_method"]


def test_the_experiments_read_the_arms_file_first(tmp_path):
    arms = {"adversarial": {"design_share": 0.5, "counts": {"shown": 20, "withheld": 3, "n/a": 40},
                            "unstamped": 1, "withheld_question_ids": []},
            "advice": {"design_share": 0.5, "counts": {"advice": 30, "no_advice": 33}, "unstamped": 0}}
    p = build(tmp_path, arms=arms)
    rows = {e.get("experiment"): e for e in p["experiments"]["experiments"]}
    assert rows["adversarial"]["eligible"] == 23  # n/a is not eligible
    assert rows["adversarial"]["inside_90pct_range"] is False
    assert rows["adversarial"]["rows_unstamped"] == 1
    assert rows["advice"]["source"] == "experiment_arms__*.json"


def test_the_adversarial_arm_falls_back_to_the_stdout_line(tmp_path):
    p = build(tmp_path, stdout="[adversarial-arm] eligible=40 shown=21 withheld=19 not_eligible=60\n")
    row = next(e for e in p["experiments"]["experiments"] if e.get("experiment") == "adversarial")
    assert row["arms"] == {"shown": 21, "withheld": 19}
    assert row["inside_90pct_range"] is True


# ---------------------------------------------------------------------------
# The adversarial sync constraint
# ---------------------------------------------------------------------------


def test_adversarial_sync_counts_including_a_deliberate_mismatch():
    qs = {"A_ACE": {"iso3": "A", "hazard_code": "ACE", "rc_level": "2"},
          "B_ACE": {"iso3": "B", "hazard_code": "ACE", "rc_level": "1"},
          "C_FL": {"iso3": "C", "hazard_code": "FL", "rc_level": "1"},
          "D_DR": {"iso3": "D", "hazard_code": "DR", "rc_level": "0"},
          "E_DR": {"iso3": "E", "hazard_code": "DR", "rc_level": "1"}}
    out = fri.adversarial_sync(spd_questions=qs, generated_pairs={"A|ACE", "B|ACE", "C|FL"},
                               injected={"A_ACE", "D_DR"}, withheld={"C_FL"})
    assert out["spd_questions_rc1_plus"] == 4
    assert out["rc1_plus_questions_with_a_generated_check"] == 3
    assert out["injected_into_spd_prompts"] == 2
    assert out["eligible_withheld_by_experiment"] == 1
    # B was generated and neither injected nor withheld: a Forecast fault.
    assert out["generated_not_injected_not_withheld_ids"] == ["B_ACE"]
    # D carries a block with no stored check behind it.
    assert out["injected_with_no_stored_check_ids"] == ["D_DR"]


def test_the_fixture_sync_is_clean(tmp_path):
    adv = build(tmp_path)["evidence"]["adversarial"]
    assert adv["generated_not_injected_not_withheld"] == 0
    assert adv["injected_with_no_stored_check"] == 0
    assert adv["injected_into_spd_prompts"] >= 1


def test_stubs_lists_every_stub_path(tmp_path):
    payload = build(tmp_path)
    paths = fri.stubs(payload)
    assert "chain.steps_succeeded_with_traceback_or_error" in paths  # no workflow_logs_index here
    assert not any(p.startswith("previous_run") for p in paths)


def test_the_cli_writes_the_file_and_never_fails(tmp_path, capsys):
    staging = build_chain(tmp_path)
    out = tmp_path / "x.json"
    assert fri.main(["--staging", str(staging), "--out", str(out), "--work-dir", str(tmp_path / "w")]) == 0
    payload = json.loads(out.read_text())
    assert payload["meta"]["fc_run_id"] == FC
    assert fri.main(["--staging", str(tmp_path / "nowhere"), "--out", str(tmp_path / "y.json"),
                     "--work-dir", str(tmp_path / "w2")]) == 0
