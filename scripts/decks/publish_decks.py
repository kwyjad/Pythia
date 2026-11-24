# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.
"""Publish the Fred slide decks to Google Drive and write decks.json.

Each deck in docs/decks/ is copied to ``<root>/decks/<file>`` on the rclone
Drive remote (the same remote and ``Fred_archive`` root the run archive
uses, signed in with the drive.file scope, so rclone sees only files it
created). Then:

* **File ids are kept.** rclone's Drive backend updates an existing file in
  place, so a deck keeps its Drive id and its link across publishes. The id
  is read before and after the copy; a changed id is reported, and the new
  one is what decks.json records.
* **Link sharing.** ``rclone link`` grants "anyone with the link" the reader
  role. The publisher reads Drive's own answer to that request (rclone
  ``--dump bodies``, parsed here and never printed) and records whether
  the role is reader and not writer.
* **Signed-out check.** Each deck's direct-download URL is fetched with no
  cookies or credentials; a deck that answers with the file (a zip) rather
  than a Google sign-in page counts as open to a signed-out reader.

decks.json then holds, per deck: title, file, the Drive view and download
links, the file id, the date and commit it was last checked against the
code (from docs/decks/manifest.json), its size and sha256. The workflow
uploads it to the ``pythia-data-latest`` release beside the database, which
is where the dashboard's About page reads it.

When Drive cannot be used (no rclone config, a failed copy, or a refused
link) the decks are offered as direct downloads only: ``view_url`` is null
and ``download_url`` points at the file in this public repository at the
commit published, so no binary is stored twice. ``drive`` records why.

Usage::

    python -m scripts.decks.publish_decks --out decks.json [--no-drive]
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
DECKS_DIR = ROOT / "docs" / "decks"
MANIFEST = DECKS_DIR / "manifest.json"
DEFAULT_REPO = "kwyjad/pythia"


def view_url(file_id: str) -> str:
    return f"https://drive.google.com/file/d/{file_id}/view?usp=sharing"


def download_url(file_id: str) -> str:
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def github_url(repo: str, ref: str, file: str) -> str:
    return f"https://github.com/{repo}/raw/{ref}/docs/decks/{file}"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_link_dump(text: str) -> dict[str, Any]:
    """What Drive said it granted, from rclone's ``--dump bodies`` output of
    ``rclone link``: the roles and types in the permission it returned."""
    roles = set(re.findall(r'"role"\s*:\s*"([a-zA-Z]+)"', text))
    types = set(re.findall(r'"type"\s*:\s*"([a-zA-Z]+)"', text))
    return {
        "roles": sorted(roles),
        "types": sorted(types),
        "view_only": "reader" in roles and not (roles & {"writer", "owner", "organizer",
                                                          "fileOrganizer", "commenter"}),
        "anyone_with_link": "anyone" in types,
    }


def anonymous_fetch_ok(url: str, opener: Callable[..., Any] | None = None) -> bool:
    """True when ``url`` returns the deck itself to a client with no
    credentials: HTTP 200, not a Google sign-in page, and a zip body."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (deck-publish check)"})
    try:
        with (opener or urllib.request.urlopen)(req, timeout=60) as resp:
            final = resp.geturl()
            head = resp.read(4)
            return (getattr(resp, "status", 200) == 200
                    and "accounts.google.com" not in final and head.startswith(b"PK"))
    except Exception:  # noqa: BLE001 - any failure means "not shown open"
        return False


class DriveDecks:
    """The rclone calls the publisher makes."""

    def __init__(self, remote: str, root: str,
                 runner: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> None:
        self.remote, self.root, self.runner = remote, root.strip("/"), runner

    def path(self, rel: str = "") -> str:
        return f"{self.remote}:{self.root}/decks" + (f"/{rel}" if rel else "")

    def _run(self, *args: str) -> subprocess.CompletedProcess:
        proc = self.runner(["rclone", *args], capture_output=True, text=True, timeout=1800)
        if proc.returncode != 0:
            raise RuntimeError(f"rclone {args[0]} failed ({proc.returncode}): "
                               f"{(proc.stderr or '').strip()[-400:]}")
        return proc

    def ids(self) -> dict[str, str]:
        proc = self.runner(["rclone", "lsjson", self.path(), "--files-only"],
                           capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:  # the folder does not exist yet
            return {}
        return {e["Name"]: e.get("ID", "") for e in json.loads(proc.stdout or "[]")}

    def copy(self, local: Path) -> None:
        self._run("copyto", str(local), self.path(local.name), "--checksum")

    def link(self, name: str) -> tuple[str, dict[str, Any]]:
        proc = self._run("link", self.path(name), "-vv", "--dump", "bodies")
        return (proc.stdout or "").strip(), parse_link_dump(proc.stderr or "")


def publish(drive: DriveDecks | None, *, repo: str, ref: str,
            check_url: Callable[[str], bool] = anonymous_fetch_ok,
            manifest_path: Path = MANIFEST) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    decks_dir = manifest_path.parent
    rows = []
    for d in sorted(manifest["decks"], key=lambda d: d["number"]):
        path = decks_dir / d["file"]
        rows.append({
            "number": d["number"], "title": d["title"], "file": d["file"],
            "last_checked": d.get("last_checked"),
            "last_checked_commit": d.get("last_checked_commit"),
            "size_bytes": path.stat().st_size, "sha256": sha256(path),
            "view_url": None, "download_url": github_url(repo, ref, d["file"]),
            "drive_file_id": None,
        })
    drive_info: dict[str, Any] = {"used": False, "reason": "not configured", "id_changes": []}
    if drive is not None:
        try:
            before = drive.ids()
            for row in rows:
                drive.copy(decks_dir / row["file"])
            after = drive.ids()
            for row in rows:
                old, new = before.get(row["file"]), after.get(row["file"])
                if not new:
                    raise RuntimeError(f"{row['file']} has no Drive id after the copy")
                if old and old != new:
                    drive_info["id_changes"].append({"file": row["file"], "before": old, "after": new})
                link, grant = drive.link(row["file"])
                if not grant["view_only"] or not grant["anyone_with_link"]:
                    raise RuntimeError(f"{row['file']}: Drive did not confirm a view-only "
                                       f"link (roles {grant['roles']}, types {grant['types']})")
                row.update(drive_file_id=new, view_url=view_url(new),
                           download_url=download_url(new), drive_link=link,
                           sharing=grant, signed_out_download_ok=check_url(download_url(new)))
            drive_info = {"used": True, "reason": "", "id_changes": drive_info["id_changes"]}
        except Exception as exc:  # noqa: BLE001 - fall back to direct downloads
            for row in rows:
                for k in ("drive_file_id", "view_url", "drive_link", "sharing",
                          "signed_out_download_ok"):
                    row.pop(k, None)
                row.update(view_url=None, drive_file_id=None,
                           download_url=github_url(repo, ref, row["file"]))
            drive_info = {"used": False, "reason": str(exc)[:400], "id_changes": []}
    return {
        "generated_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "source_commit": ref,
        "drive": drive_info,
        "decks": rows,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-drive", action="store_true")
    ap.add_argument("--remote", default=os.environ.get("RCLONE_REMOTE", "pythia_drive"))
    ap.add_argument("--root", default=os.environ.get("DRIVE_ROOT", "Fred_archive"))
    ap.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", DEFAULT_REPO))
    ap.add_argument("--ref", default=os.environ.get("GITHUB_SHA", "main"))
    args = ap.parse_args(argv)

    drive = None if args.no_drive else DriveDecks(args.remote, args.root)
    out = publish(drive, repo=args.repo, ref=args.ref)
    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    d = out["drive"]
    if not args.no_drive and not d["used"]:
        print(f"::warning title=Decks published as downloads only::Drive was not used: {d['reason']}")
    for ch in d["id_changes"]:
        print(f"::warning title=Deck Drive id changed::{ch['file']}: {ch['before']} -> {ch['after']}")
    for row in out["decks"]:
        if d["used"] and not row.get("signed_out_download_ok"):
            print(f"::warning title=Deck not open to signed-out readers::{row['file']} did not "
                  "download without credentials; check its sharing in Drive.")
        print(f"{row['number']} {row['file']:<32} view={row['view_url']} download={row['download_url']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
