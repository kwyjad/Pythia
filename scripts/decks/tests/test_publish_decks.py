# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.
"""Tests for scripts/decks/publish_decks.py with a fake rclone."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from scripts.decks.publish_decks import DriveDecks, parse_link_dump, publish

GRANT = ('<< POST permissions {"role":"reader","type":"anyone"}\n'
         '>> {"kind":"drive#permission","id":"anyoneWithLink","type":"anyone","role":"reader"}')


def _manifest(tmp: Path) -> Path:
    d = tmp / "decks"
    d.mkdir()
    for n, name in ((0, "0_a.pptx"), (1, "1_b.pptx")):
        (d / name).write_bytes(b"PK\x03\x04" + bytes([n]) * 10)
    m = d / "manifest.json"
    m.write_text(json.dumps({"decks": [
        {"number": 1, "title": "B", "file": "1_b.pptx", "last_checked": "2026-11-24",
         "last_checked_commit": "43299ee"},
        {"number": 0, "title": "A", "file": "0_a.pptx", "last_checked": "2026-11-24",
         "last_checked_commit": "43299ee"}]}))
    return m


class FakeRclone:
    def __init__(self, ids=None, change_id=False, link_stderr=GRANT, fail_copy=False):
        self.ids = dict(ids or {})
        self.change_id, self.link_stderr, self.fail_copy = change_id, link_stderr, fail_copy
        self.calls = []

    def __call__(self, cmd, **kw):
        self.calls.append(cmd)
        op = cmd[1]
        if op == "lsjson":
            out = json.dumps([{"Name": n, "ID": i} for n, i in self.ids.items()])
            return subprocess.CompletedProcess(cmd, 0, out, "")
        if op == "copyto":
            if self.fail_copy:
                return subprocess.CompletedProcess(cmd, 1, "", "403 insufficient scope")
            name = Path(cmd[2]).name
            if name not in self.ids or self.change_id:
                self.ids[name] = f"id-{name}-{len(self.calls)}"
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if op == "link":
            name = cmd[2].rsplit("/", 1)[1]
            return subprocess.CompletedProcess(
                cmd, 0, f"https://drive.google.com/open?id={self.ids[name]}\n", self.link_stderr)
        raise AssertionError(cmd)


def test_publish_records_ids_links_and_order(tmp_path):
    m = _manifest(tmp_path)
    fake = FakeRclone(ids={"0_a.pptx": "keep0"})
    out = publish(DriveDecks("r", "Fred_archive", runner=fake), repo="o/r", ref="abc",
                  check_url=lambda u: True, manifest_path=m)
    assert out["drive"]["used"] and out["drive"]["id_changes"] == []
    assert [d["number"] for d in out["decks"]] == [0, 1]
    a = out["decks"][0]
    assert a["drive_file_id"] == "keep0"
    assert a["view_url"] == "https://drive.google.com/file/d/keep0/view?usp=sharing"
    assert a["sharing"]["view_only"] and a["signed_out_download_ok"]
    assert a["size_bytes"] == 14 and len(a["sha256"]) == 64
    assert any(c[1] == "copyto" and c[3] == "r:Fred_archive/decks/0_a.pptx" for c in fake.calls)


def test_a_changed_id_is_reported_and_the_new_one_used(tmp_path):
    m = _manifest(tmp_path)
    fake = FakeRclone(ids={"0_a.pptx": "old0", "1_b.pptx": "old1"}, change_id=True)
    out = publish(DriveDecks("r", "F", runner=fake), repo="o/r", ref="abc",
                  check_url=lambda u: True, manifest_path=m)
    changes = {c["file"]: c for c in out["drive"]["id_changes"]}
    assert changes["0_a.pptx"]["before"] == "old0"
    assert out["decks"][0]["drive_file_id"] == changes["0_a.pptx"]["after"]


def test_an_editable_link_falls_back_to_downloads(tmp_path):
    m = _manifest(tmp_path)
    fake = FakeRclone(link_stderr='{"type":"anyone","role":"writer"}')
    out = publish(DriveDecks("r", "F", runner=fake), repo="o/r", ref="abc",
                  check_url=lambda u: True, manifest_path=m)
    assert not out["drive"]["used"] and "view-only" in out["drive"]["reason"]
    for d in out["decks"]:
        assert d["view_url"] is None
        assert d["download_url"] == f"https://github.com/o/r/raw/abc/docs/decks/{d['file']}"


def test_a_refused_copy_falls_back(tmp_path):
    m = _manifest(tmp_path)
    out = publish(DriveDecks("r", "F", runner=FakeRclone(fail_copy=True)), repo="o/r",
                  ref="abc", check_url=lambda u: True, manifest_path=m)
    assert not out["drive"]["used"] and "insufficient scope" in out["drive"]["reason"]


def test_no_drive_gives_downloads_only(tmp_path):
    out = publish(None, repo="o/r", ref="abc", manifest_path=_manifest(tmp_path))
    assert out["drive"]["reason"] == "not configured"
    assert all(d["view_url"] is None for d in out["decks"])


def test_parse_link_dump():
    assert parse_link_dump(GRANT)["view_only"]
    g = parse_link_dump('{"role":"reader"} {"role":"writer","type":"anyone"}')
    assert not g["view_only"] and g["anyone_with_link"]
