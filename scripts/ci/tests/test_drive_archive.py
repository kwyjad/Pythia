# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The Drive archive: names, thinning, the full cycle zip, idempotency, the leg,
housekeeping, the chain-diagnostics section and the watchdog.

No network and no Drive: the Actions API goes through a fake ``gh`` and Drive
through a fake rclone that copies into a directory.
"""

from __future__ import annotations

import datetime as dt
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

from scripts.ci import build_cycle_diagnostics as bcd
from scripts.ci import chain_diagnostics as cd
from scripts.ci import check_drive_archive as cda
from scripts.ci import drive_archive as da

REPO = "owner/pythia"
SIX = ("pythia-llm-calls-detail", "pythia-spd-tables", "pythia-question-metrics",
       "pythia-evidence-packs", "pythia-llm-prompts", "pythia-interpreter-gate-diagnostics")


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------


def test_forecast_names_carry_the_date_the_fc_run_and_both_github_runs():
    n = da.archive_names(kind="forecast", start_date="2026-11-13T00:04:11+00:00",
                         publish_run_id=38900000000, fc_run_id="fc_1794614400",
                         sibyl_run_id=38800000000)
    assert n.zip_name == "pythia_2026-11-13__forecast__fc_1794614400__gh38900000000.zip"
    assert n.folder == "2026-11-13__forecast__fc_1794614400__gh38900000000"
    assert n.run_dir == "runs/2026/2026-11-13__forecast__fc_1794614400__gh38900000000"
    assert n.db_name == "pythia_resolver_db_2026-11-13__fc_1794614400__gh38800000000.duckdb.zst"
    assert n.db_meta_name.endswith(".db_snapshot.json")
    assert n.db_dir == "db_snapshots/2026"


def test_scoring_names_have_no_db_and_test_chains_say_so():
    n = da.archive_names(kind="scoring", start_date=dt.date(2026, 11, 11), publish_run_id="7")
    assert n.zip_name == "pythia_2026-11-11__scoring__gh7.zip" and n.db_name is None
    t = da.archive_names(kind="forecast", start_date="2026-11-13", publish_run_id="8",
                         fc_run_id="fc_1", sibyl_run_id="9", is_test=True)
    assert t.zip_name.endswith("__TEST.zip") and t.db_name.endswith("__TEST.duckdb.zst")
    assert t.folder.endswith("__TEST")


def test_two_chains_in_one_month_never_collide_and_unknowns_are_named():
    a = da.archive_names(kind="forecast", start_date="2026-10-01", publish_run_id="1", fc_run_id="fc_1")
    b = da.archive_names(kind="forecast", start_date="2026-10-13", publish_run_id="2", fc_run_id="fc_2")
    assert a.zip_name != b.zip_name and a.db_name != b.db_name
    u = da.archive_names(kind="weird", start_date="2026-10-13", publish_run_id="3")
    assert u.kind == "other" and u.zip_name == "pythia_2026-10-13__other__gh3.zip"
    f = da.archive_names(kind="forecast", start_date="2026-10-13", publish_run_id="3")
    assert "fc_unknown" in f.zip_name and "__ghunknown" in f.db_name


def test_snapshot_names_parse_back():
    n = da.archive_names(kind="forecast", start_date="2026-11-13", publish_run_id="1",
                         fc_run_id="fc_1794614400", sibyl_run_id="555")
    s = da.parse_snapshot(n.db_name)
    assert s.date == dt.date(2026, 11, 13) and s.gh == "555" and not s.is_test
    assert da.parse_snapshot("notes.txt") is None


# ---------------------------------------------------------------------------
# Thinning
# ---------------------------------------------------------------------------


def _snap(date, gh="100", test=False):
    return da.archive_names(kind="forecast", start_date=date, publish_run_id="1",
                            fc_run_id="fc_1", sibyl_run_id=gh, is_test=test).db_name


TODAY = dt.date(2027, 6, 30)


def test_everything_within_a_year_is_kept_and_the_boundary_day_counts():
    boundary = TODAY - dt.timedelta(days=365)
    names = [_snap(boundary), _snap(boundary - dt.timedelta(days=1)),
             _snap(boundary - dt.timedelta(days=2), gh="99")]
    # Two older ones in one quarter, plus padding so the floor does not bind.
    names += [_snap(TODAY - dt.timedelta(days=d)) for d in (1, 30, 60, 90, 120, 150)]
    plan = da.thinning_plan(names, TODAY)
    assert _snap(boundary) in plan.keep
    assert _snap(boundary - dt.timedelta(days=1)) in plan.keep  # newest of its quarter
    assert _snap(boundary - dt.timedelta(days=2), gh="99") in plan.delete


def test_quarter_edges_keep_one_per_quarter():
    old = ["2025-01-02", "2025-03-31", "2025-04-01", "2025-06-30", "2025-07-01"]
    names = [_snap(d) for d in old] + [_snap(TODAY - dt.timedelta(days=d)) for d in range(6)]
    plan = da.thinning_plan(names, TODAY)
    assert _snap("2025-03-31") in plan.keep and _snap("2025-01-02") in plan.delete
    assert _snap("2025-06-30") in plan.keep and _snap("2025-04-01") in plan.delete
    assert _snap("2025-07-01") in plan.keep
    assert "2025-Q1" in plan.reasons[_snap("2025-03-31")]


def test_two_snapshots_on_one_day_keep_the_later_run():
    names = [_snap("2025-02-10", gh="300"), _snap("2025-02-10", gh="400")]
    names += [_snap(TODAY - dt.timedelta(days=d)) for d in range(6)]
    plan = da.thinning_plan(names, TODAY)
    assert _snap("2025-02-10", gh="400") in plan.keep
    assert _snap("2025-02-10", gh="300") in plan.delete


def test_old_test_snapshots_go_and_never_count_towards_the_floor():
    names = [_snap("2025-02-10", test=True), _snap(TODAY, test=True)]
    names += [_snap(d) for d in ("2025-01-10", "2025-02-11")]
    plan = da.thinning_plan(names, TODAY)
    assert _snap("2025-02-10", test=True) in plan.delete
    assert _snap(TODAY, test=True) in plan.keep
    # Only two production snapshots exist, both kept by the floor.
    assert _snap("2025-01-10") in plan.keep and _snap("2025-02-11") in plan.keep


def test_the_floor_of_six_holds_back_the_newest_deletions():
    names = [_snap(f"2024-0{m}-0{d}") for m in (1, 2) for d in range(1, 6)]
    plan = da.thinning_plan(names, TODAY)
    assert len(plan.keep) == 6
    assert _snap("2024-02-05") in plan.keep and _snap("2024-01-01") in plan.delete
    assert any("floor" in plan.reasons[n] for n in plan.keep)


def test_an_unparsed_name_is_never_deleted():
    plan = da.thinning_plan(["pythia_resolver_db_garbage.duckdb.zst"], TODAY)
    assert plan.delete == [] and plan.unparsed == ["pythia_resolver_db_garbage.duckdb.zst"]


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _run(rid, name, created, *, event="workflow_dispatch", title=None, conclusion="success"):
    # The real API shape: ``name`` carries the run-name where a workflow has
    # one, ``path`` names the file.
    file = next((f for n, (_s, f) in cd.WORKFLOWS.items() if n == name), "x.yml")
    return {"id": int(rid), "name": title or name, "path": f".github/workflows/{file}",
            "event": event, "status": "completed",
            "conclusion": conclusion, "created_at": created, "updated_at": created,
            "run_started_at": created, "head_branch": "main",
            "head_repository": {"full_name": REPO}, "display_title": title or name,
            "html_url": f"https://github.com/{REPO}/actions/runs/{rid}"}


class FakeGh:
    def __init__(self):
        self.runs, self.artifacts, self.contents, self.downloads = {}, {}, {}, []

    def add(self, info, artifacts=None):
        rid = str(info["id"])
        self.runs[rid] = info
        self.artifacts[rid] = []
        for name, files in (artifacts or {}).items():
            self.artifacts[rid].append({"name": name, "expired": False, "size_in_bytes": 10})
            self.contents[(rid, name)] = files

    def api(self, path):
        parts = path.split("?")[0].split("/")
        if parts[-1] == "artifacts":
            return {"artifacts": list(self.artifacts.get(parts[-2], []))}
        if "workflows" in parts:
            file = parts[parts.index("workflows") + 1]
            name = next(n for n, (_s, f) in cd.WORKFLOWS.items() if f == file)
            runs = [r for r in self.runs.values() if bcd.workflow_name(r) == name]
            return {"workflow_runs": sorted(runs, key=lambda r: r["created_at"], reverse=True)}
        if parts[-1] not in self.runs:
            raise RuntimeError("404")
        return self.runs[parts[-1]]

    def download(self, run_id, name, dest):
        self.downloads.append((str(run_id), name))
        files = self.contents.get((str(run_id), name))
        if files is None:
            return False, "not found"
        for rel, body in files.items():
            p = Path(dest) / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(body) if isinstance(body, bytes) else p.write_text(body, encoding="utf-8")
        return True, "ok"


class FakeRclone(da.Rclone):
    """Drive as a directory: the real code paths, a local store."""

    def __init__(self, store: Path, root="Fred_archive", roots=1):
        super().__init__("pythia_drive", root)
        self.store = store
        self.roots = roots
        self.deleted: list[str] = []
        self.about_payload = {"total": 2 * 1024 ** 4, "used": 1024 ** 4, "free": 1024 ** 4}

    def root_folders(self):
        return self.roots

    def list_files(self, rel):
        base = self.store / self.root / rel
        if not base.is_dir():
            return []
        return sorted(p.relative_to(base).as_posix() for p in base.rglob("*") if p.is_file())

    def copy_tree(self, local):
        shutil.copytree(local, self.store / self.root, dirs_exist_ok=True)

    def delete(self, rel_file):
        (self.store / self.root / rel_file).unlink()
        self.deleted.append(rel_file)

    def about(self):
        return dict(self.about_payload)


STAGE_CTX = {"forecaster_run_id": "fc_1794614400", "hs_run_id": "hs_20261113T000411",
             "pipeline_id": "hs_20261113T000411",
             "env_snapshot": {"env": {"PYTHIA_TEST_MODE": "<unset>"}}}


def _forecast_chain(gh: FakeGh, *, test=False, large=b""):
    ctx = dict(STAGE_CTX)
    if test:
        ctx["env_snapshot"] = {"env": {"PYTHIA_TEST_MODE": "1"}}
    title = "Pythia Pipeline Stage: hs_20261113T000411 — {}"
    gh.add(_run(10, "Pythia Pipeline Stage", "2026-11-13T00:04:00Z", event="schedule",
                title=title.format("hs_submit")),
           {"pythia-stage-diagnostics": {"s.md": "ok"}})
    gh.add(_run(14, "Pythia Pipeline Stage", "2026-11-13T05:00:00Z",
                title=title.format("hs_rc_collect")),
           {"pythia-stage-diagnostics": {"s2.md": "ok"}})
    gh.add(_run(11, "Pythia Pipeline Stage", "2026-11-13T09:00:00Z",
                title=title.format("fc_collect_finalize")),
           {"pythia-stage-context": {"stage_context.json": json.dumps(ctx)},
            "pythia-resolver-db": {"resolver.duckdb": b"DB"}})
    sibyl = {"pythia-resolver-db": {"resolver.duckdb": b"DB"},
             "pythia-debug-bundle": {"pythia_debug_bundle__fc_1794614400.zip": b"PK\x03\x04debug" + large},
             "pythia-health-report": {"executive_summary__fc_1794614400.md": "# Summary\n",
                                      "health_report__fc_1794614400.json": "{}",
                                      "stage_context/stage_context.json": json.dumps(ctx)},
             "pythia-interpreter-report": {"report__2026-11__hs_x__v1.pdf": b"%PDF",
                                           "interpreter_report_latest.pdf": b"%PDF",
                                           "report_2026-11_v1.md": "# Report\n"}}
    for name in SIX:
        sibyl[name] = {f"{name}.txt": "x"}
    gh.add(_run(12, "Sibyl Deep Research", "2026-11-13T10:00:00Z"), sibyl)
    lineage = {"slug": "publish_latest_data", "this_run": "13", "trigger_run": "12"}
    gh.add(_run(13, "Publish Latest Data (Release)", "2026-11-13T12:00:00Z"),
           {"chain-diagnostics-publish_latest_data-13": {"lineage.json": json.dumps(lineage),
                                                          "issues.json": json.dumps({"issues": []})},
            "pythia-cycle-diagnostics": {"c.zip": b"PK"}})


# ---------------------------------------------------------------------------
# Full mode
# ---------------------------------------------------------------------------


def test_full_mode_collects_the_six_and_never_a_db_and_drops_nothing(tmp_path):
    gh = FakeGh()
    _forecast_chain(gh, large=b"\x01" * 50_000)
    m = bcd.build(gh=gh, repo=REPO, source_run_id="13", publish_run_id="13",
                  out_dir=tmp_path / "out", staging=tmp_path / "st", wait_sec=0, env={},
                  full=True, max_bytes=1000)
    names = zipfile.ZipFile(m["zip"]).namelist()
    for art in SIX:
        assert any(f"/{art}/" in n for n in names), art
    assert not any("pythia-resolver-db" in n or n.endswith("resolver.duckdb") for n in names)
    assert not any(name.startswith("pythia-resolver-db") for _r, name in gh.downloads)
    # The nested debug bundle is kept whole, and nothing is dropped for size.
    debug = next(n for n in names if n.endswith("pythia_debug_bundle__fc_1794614400.zip"))
    assert zipfile.ZipFile(m["zip"]).getinfo(debug).file_size > 50_000
    assert m["dropped"] == [] and m["mode"] == "full" and m["max_bytes"] is None
    assert {"README.md", "CYCLE_SUMMARY.md", "manifest.json"} <= set(names)
    man = json.loads(zipfile.ZipFile(m["zip"]).read("manifest.json"))
    assert all({"bytes", "sha256", "source_run", "workflow", "artifact"} <= set(f) for f in man["files"])
    assert "missing" in man
    # The cycle zip Publish already made is not folded into the archive.
    assert not any("pythia-cycle-diagnostics" in n for n in names)


def test_the_200_mb_mode_still_leaves_the_six_out(tmp_path):
    gh = FakeGh()
    _forecast_chain(gh)
    m = bcd.build(gh=gh, repo=REPO, source_run_id="13", publish_run_id="13",
                  out_dir=tmp_path / "out", staging=tmp_path / "st", wait_sec=0, env={})
    names = zipfile.ZipFile(m["zip"]).namelist()
    assert not any(f"/{art}/" in n for n in names for art in SIX)
    assert m["mode"] == "ceiling"


def test_full_mode_still_refuses_a_credential(tmp_path):
    gh = FakeGh()
    _forecast_chain(gh)
    secret = "ghs_" + "Z9y8X7w6V5u4T3s2R1q0" * 2
    gh.contents[("12", "pythia-llm-prompts")]["pythia-llm-prompts.txt"] = f"key {secret}"
    with pytest.raises(bcd.SecretLeak):
        bcd.build(gh=gh, repo=REPO, source_run_id="13", publish_run_id="13",
                  out_dir=tmp_path / "out", staging=tmp_path / "st", wait_sec=0,
                  env={"GH_TOKEN": secret}, full=True)
    assert not list((tmp_path / "out").glob("*.zip"))


def test_chain_facts_read_the_kind_start_and_identifiers(tmp_path):
    gh = FakeGh()
    _forecast_chain(gh)
    m = bcd.build(gh=gh, repo=REPO, source_run_id="13", publish_run_id="13",
                  out_dir=tmp_path / "out", staging=tmp_path / "st", wait_sec=0, env={}, full=True)
    facts = m["chain"]
    assert facts["kind"] == "forecast" and facts["start_date"] == "2026-11-13"
    assert facts["started_at"].startswith("2026-11-13T00:04")
    assert facts["fc_run_id"] == "fc_1794614400" and facts["sibyl_run_id"] == "12"
    assert facts["is_test"] is False
    # Every stage of the pipeline joins, though the API names each run by its
    # run-name ("Pythia Pipeline Stage: <pipeline> — <stage>").
    assert {"10", "11", "14"} <= {r["run_id"] for r in m["runs"]}


def test_the_workflow_is_read_from_the_path_not_the_run_name():
    info = {"name": "Pythia Pipeline Stage: hs_x — hs_submit",
            "path": ".github/workflows/pythia_pipeline_stage.yml"}
    assert bcd.workflow_name(info) == "Pythia Pipeline Stage"
    assert bcd.workflow_name({"name": "Something"}) == "Something"


# ---------------------------------------------------------------------------
# Plan and idempotency
# ---------------------------------------------------------------------------


def test_plan_names_a_forecast_chain_and_finds_nothing_in_drive(tmp_path):
    gh = FakeGh()
    _forecast_chain(gh)
    rc = FakeRclone(tmp_path / "drive")
    rec = da.plan_one(gh, REPO, rc, "13", include_test=False, work=tmp_path / "w")
    assert rec["status"] == "todo"
    assert rec["names"]["zip_name"] == "pythia_2026-11-13__forecast__fc_1794614400__gh13.zip"
    assert rec["names"]["db_name"] == "pythia_resolver_db_2026-11-13__fc_1794614400__gh12.duckdb.zst"
    # Planning never downloads the big bundles.
    assert not any(n in ("pythia-debug-bundle", "pythia-llm-calls-detail") for _r, n in gh.downloads)


def test_a_chain_already_in_drive_stops_and_a_partial_one_does_not(tmp_path):
    gh = FakeGh()
    _forecast_chain(gh)
    rc = FakeRclone(tmp_path / "drive")
    rec = da.plan_one(gh, REPO, rc, "13", include_test=False, work=tmp_path / "w")
    names = da.ArchiveNames(**rec["names"])
    zip_path = tmp_path / "drive" / "Fred_archive" / names.run_dir / names.zip_name
    zip_path.parent.mkdir(parents=True)
    zip_path.write_bytes(b"z")
    assert da.plan_one(gh, REPO, rc, "13", include_test=False, work=tmp_path / "w")["status"] == "todo"
    db = tmp_path / "drive" / "Fred_archive" / names.db_dir / names.db_name
    db.parent.mkdir(parents=True)
    db.write_bytes(b"d")
    again = da.plan_one(gh, REPO, rc, "13", include_test=False, work=tmp_path / "w")
    assert again["status"] == "already_present"


def test_test_chains_are_skipped_unless_asked(tmp_path):
    gh = FakeGh()
    _forecast_chain(gh, test=True)
    rec = da.plan_one(gh, REPO, None, "13", include_test=False, work=tmp_path / "w")
    assert rec["status"] == "skipped_test"
    rec = da.plan_one(gh, REPO, None, "13", include_test=True, work=tmp_path / "w")
    assert rec["status"] == "todo" and rec["names"]["zip_name"].endswith("__TEST.zip")


def test_a_failed_publish_run_is_not_archived(tmp_path):
    gh = FakeGh()
    gh.add(_run(13, "Publish Latest Data (Release)", "2026-11-13T12:00:00Z", conclusion="failure"))
    rec = da.plan_one(gh, REPO, None, "13", include_test=False, work=tmp_path / "w")
    assert rec["status"] == "skipped_not_published"


def test_two_drive_roots_stop_the_archive(tmp_path):
    with pytest.raises(da.RcloneError, match="2 folders named 'Fred_archive'"):
        FakeRclone(tmp_path, roots=2).assert_single_root()


# ---------------------------------------------------------------------------
# The leg
# ---------------------------------------------------------------------------


def test_a_leg_uploads_the_zip_loose_files_and_the_snapshot(tmp_path):
    gh = FakeGh()
    _forecast_chain(gh)
    rc = FakeRclone(tmp_path / "drive")
    spec = da.plan_one(gh, REPO, rc, "13", include_test=False, work=tmp_path / "w")

    def fetch_db(_gh, _repo, sibyl, dest):
        assert sibyl == "12"
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "resolver.duckdb").write_bytes(b"db")
        return dest / "resolver.duckdb"

    def snapshot(db, out, verify_dir=None):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"zst")
        return {"uncompressed_bytes": 2, "compressed_bytes": 3, "verified": True}

    report = da.run_leg(spec, gh=gh, repo=REPO, rclone=rc, work=tmp_path / "w",
                        out_dir=tmp_path / "out", env={}, snapshot=snapshot, fetch_db=fetch_db)
    assert report["status"] == "archived"
    names = da.ArchiveNames(**spec["names"])
    run_dir = tmp_path / "drive" / "Fred_archive" / names.run_dir
    files = {p.name for p in run_dir.iterdir()}
    assert names.zip_name in files
    assert {"README.md", "CYCLE_SUMMARY.md", "manifest.json"} <= files
    assert any(f.endswith("executive_summary__fc_1794614400.md") for f in files)
    assert any(f.endswith("health_report__fc_1794614400.json") for f in files)
    assert any(f.endswith("report__2026-11__hs_x__v1.pdf") for f in files)
    assert any(f.endswith("report_2026-11_v1.md") for f in files)
    assert not any(f.endswith("interpreter_report_latest.pdf") for f in files)
    db_dir = tmp_path / "drive" / "Fred_archive" / names.db_dir
    assert (db_dir / names.db_name).is_file() and (db_dir / names.db_meta_name).is_file()


@pytest.mark.skipif(shutil.which("zstd") is None, reason="zstd not installed")
def test_snapshot_db_compresses_and_proves_the_copy(tmp_path):
    duckdb = pytest.importorskip("duckdb")
    db = tmp_path / "w" / "resolver.duckdb"
    db.parent.mkdir()
    con = duckdb.connect(str(db))
    for t in ("forecasts_raw", "llm_calls", "scores"):
        con.execute(f"CREATE TABLE {t} AS SELECT range AS x FROM range(5)")
    con.close()
    raw_sha = da.sha256_file(db)
    meta = da.snapshot_db(db, tmp_path / "up" / "x.duckdb.zst", verify_dir=tmp_path / "w")
    assert meta["verified"] and meta["sha256_uncompressed"] == raw_sha
    assert meta["row_counts"] == {"forecasts_raw": 5, "llm_calls": 5, "scores": 5}
    assert not db.exists() and not list((tmp_path / "w").glob("*.verify"))
    out = tmp_path / "back.duckdb"
    subprocess.run(["zstd", "-d", "-q", str(tmp_path / "up" / "x.duckdb.zst"), "-o", str(out)], check=True)
    assert da.sha256_file(out) == raw_sha


# ---------------------------------------------------------------------------
# Housekeeping and the chain-diagnostics section
# ---------------------------------------------------------------------------


def _housekeeping_args(tmp_path, plan, legs):
    import argparse

    (tmp_path / "plan.json").write_text(json.dumps({"records": plan}))
    legs_dir = tmp_path / "legs"
    legs_dir.mkdir(exist_ok=True)
    for leg in legs:
        (legs_dir / f"leg_{leg['publish_run_id']}.json").write_text(json.dumps(leg))
    return argparse.Namespace(plan=str(tmp_path / "plan.json"), legs_dir=str(legs_dir),
                              dry_run=False, out=str(tmp_path / "diag" / "archive_report.json"))


def test_housekeeping_merges_legs_thins_and_flags_a_full_drive(tmp_path):
    rc = FakeRclone(tmp_path / "drive")
    snaps = tmp_path / "drive" / "Fred_archive" / "db_snapshots" / "2024"
    snaps.mkdir(parents=True)
    for d in (1, 2, 3, 4, 5, 6, 7, 8):
        name = _snap(f"2024-01-0{d}")
        (snaps / name).write_bytes(b"z")
        (snaps / (name[: -len(da.DB_SUFFIX)] + da.META_SUFFIX)).write_text("{}")
    rc.about_payload = {"total": 100 * 1024 ** 3, "used": 90 * 1024 ** 3, "free": 10 * 1024 ** 3}
    plan = [{"publish_run_id": "1", "status": "todo"}, {"publish_run_id": "2", "status": "todo"},
            {"publish_run_id": "3", "status": "already_present"}]
    legs = [{"publish_run_id": "1", "status": "archived"}]
    args = _housekeeping_args(tmp_path, plan, legs)
    da.cmd_housekeeping(args, rclone=rc, today=TODAY)
    report = json.loads(Path(args.out).read_text())
    status = {e["publish_run_id"]: e["status"] for e in report["publish_runs"]}
    assert status == {"1": "archived", "2": "failed", "3": "already_present"}
    assert len(report["thinning"]["deleted"]) == 2 and len(rc.deleted) == 4  # zst + json each
    assert report["storage"]["low"] is True

    out = tmp_path / "diag" / "archive_to_drive"
    w = cd.Writer(slug="archive_to_drive", out_dir=out, db_path=None,
                  diagnostics_dir=tmp_path / "diag", started_at="", env={})
    out.mkdir(parents=True)
    w.archive()
    ids = {i["id"]: i["severity"] for i in w.issues}
    assert ids["drive_space_low"] == "blocking" and ids["drive_archive_failed"] == "degraded"
    assert (out / "archive_report.json").is_file()


def test_storage_verdict_lines():
    gb = 1024 ** 3
    assert da.storage_verdict({"total": 2000 * gb, "used": 100 * gb, "free": 1900 * gb})["low"] is False
    assert da.storage_verdict({"total": 2000 * gb, "used": 1990 * gb, "free": 10 * gb})["low"] is True
    assert da.storage_verdict({"total": 100 * gb, "used": 86 * gb, "free": 14 * gb})["low"] is True


# ---------------------------------------------------------------------------
# Watchdog
# ---------------------------------------------------------------------------

NOW = dt.datetime(2026, 12, 30, 12, tzinfo=dt.timezone.utc)


def _pub(rid, days_ago):
    t = (NOW - dt.timedelta(days=days_ago)).isoformat().replace("+00:00", "Z")
    return _run(rid, "Publish Latest Data (Release)", t)


def test_watchdog_warns_then_errors_and_respects_the_grace_period():
    publish = [_pub(1, 1), _pub(2, 5), _pub(3, 70), _pub(4, 10)]
    reports = [({"id": 99}, {"publish_runs": [{"publish_run_id": "4", "status": "archived"}],
                             "storage": {"low": False}})]
    v = cda.evaluate(publish, reports, NOW)
    assert v["ok"] == ["4"]
    assert len(v["warnings"]) == 1 and "Publish run 2" in v["warnings"][0]
    assert len(v["errors"]) == 1 and "Publish run 3" in v["errors"][0]


def test_watchdog_counts_skipped_test_chains_and_flags_a_full_drive():
    publish = [_pub(5, 5)]
    reports = [({"id": 98}, {"publish_runs": [{"publish_run_id": "5", "status": "skipped_test"}],
                             "storage": {"low": True, "free_bytes": 1, "used_fraction": 0.99}})]
    v = cda.evaluate(publish, reports, NOW)
    assert v["warnings"] == [] and len(v["errors"]) == 1 and "nearly full" in v["errors"][0]


def test_watchdog_failed_entries_do_not_count():
    publish = [_pub(6, 5)]
    reports = [({"id": 97}, {"publish_runs": [{"publish_run_id": "6", "status": "failed"}]})]
    assert len(cda.evaluate(publish, reports, NOW)["warnings"]) == 1
