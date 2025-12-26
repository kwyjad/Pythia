#!/usr/bin/env python3
# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""
Robust zip packer for explicit file lists with integrity manifest.

Features:
- Reads files as raw bytes and writes them into a zip (no text mode surprises).
- Emits MANIFEST.json inside the zip with size and sha256 for each file.
- Warns on missing files, likely Git-LFS pointers, and heuristic "truncation" markers.
- Supports inline list or an external file-list (one path per line).
- Exits non-zero if no files were added.

Usage:
  # Inline list (edit DEFAULT_FILE_LIST below)
  python tools/make_zip_from_list.py

  # External list file (one path per line; '#' comments allowed)
  python tools/make_zip_from_list.py --file-list filelist.txt

  # Custom output path
  python tools/make_zip_from_list.py --file-list filelist.txt --out context_pack.zip
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

# --- Option 1: Inline list (edit these if you prefer not to use --file-list)
DEFAULT_FILE_LIST = [
    "pythia/web_research/web_research.py",
    "pythia/web_research/backends/gemini_grounding.py",
    "horizon_scanner/horizon_scanner.py",
    "horizon_scanner/prompts.py",
    "horizon_scanner/db_writer.py",
    "scripts/dump_pythia_debug_bundle.py",
]

LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1\n"


def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def read_list_file(path: Path) -> list[str]:
    items: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            items.append(s)
    return items


def suspicious_truncation_markers(b: bytes) -> list[str]:
    """
    Heuristic: look for common truncation/placeholder markers after a safe decode.
    We ignore decode errors to scan text-like content.
    """
    text = b.decode("utf-8", errors="ignore")
    markers = []
    candidates = [
        "<<truncated>>",
        "<<ImageDisplayed>>",
        "<<conversation too long; truncated>>",
        "… (truncated)",
        "... [truncated]",
        "[[TRUNCATED]]",
    ]
    for m in candidates:
        if m in text:
            markers.append(m)
    return markers


def looks_like_lfs_pointer(b: bytes) -> bool:
    return b.startswith(LFS_POINTER_PREFIX)


def add_file(zipf: ZipFile, base: Path, relpath: str, manifest: list[dict], warnings: list[str]) -> bool:
    candidate = relpath.strip()
    if not candidate:
        warnings.append("Skipping blank path entry")
        return False

    p = (base / candidate).resolve()
    try:
        p.relative_to(base)
    except ValueError:
        warnings.append(f"Outside base directory (use repo-relative paths): {candidate}")
        return False

    if not p.exists() or not p.is_file():
        warnings.append(f"Missing: {candidate}")
        return False

    data = p.read_bytes()
    size = len(data)
    digest = sha256_bytes(data)
    # Heuristics & warnings
    if size == 0:
        warnings.append(f"Empty file: {relpath}")
    if looks_like_lfs_pointer(data):
        warnings.append(f"Git-LFS pointer detected (not actual content): {relpath}")
    for m in suspicious_truncation_markers(data):
        warnings.append(f"Suspicious marker '{m}' found in: {relpath}")

    # Build ZipInfo to preserve a sane timestamp and unix perms (0644)
    arcname = Path(candidate).as_posix().lstrip("./")
    if not arcname:
        warnings.append(f"Unusable archive name for path: {candidate}")
        return False

    base_name = Path(arcname).name
    parent_name = Path(arcname).parent.name
    archive_name = base_name
    existing_names = {z.filename for z in zipf.filelist}
    if archive_name in existing_names:
        prefix = parent_name or ""
        if prefix:
            archive_name = f"{prefix}_{archive_name}"
        # Ensure uniqueness even if prefixed name still collides.
        counter = 1
        while archive_name in existing_names:
            suffix = f"{prefix + '_' if prefix else ''}{counter}_"
            archive_name = f"{suffix}{base_name}"
            counter += 1

    zi = ZipInfo(filename=archive_name)
    # MS-DOS date format (Y, M, D, h, m, s)
    now = datetime.now(timezone.utc)
    zi.date_time = now.utctimetuple()[:6]
    zi.compress_type = ZIP_DEFLATED
    # External attributes: set regular file with 0644 perms (unix)
    zi.external_attr = (0o100644 & 0xFFFF) << 16

    zipf.writestr(zi, data)

    manifest.append({
        "original_path": arcname,
        "archive_name": archive_name,
        "size": size,
        "sha256": digest,
    })
    print(f"Added: {arcname} → {archive_name} ({size} bytes)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Zip explicit file list with integrity manifest.")
    parser.add_argument("--file-list", type=str, default=None, help="Path to a text file with one relative path per line.")
    parser.add_argument("--out", type=str, default="context_pack.zip", help="Output zip filename.")
    args = parser.parse_args()

    base = Path(".").resolve()
    if args.file_list:
        files = read_list_file(Path(args.file_list))
    else:
        files = list(DEFAULT_FILE_LIST)

    files = [f.replace("\\", "/") for f in files]  # normalize
    if not files:
        print("No files specified. Use --file-list or edit DEFAULT_FILE_LIST.")
        raise SystemExit(2)

    out_path = (base / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    warnings: list[str] = []
    with ZipFile(out_path, "w", compression=ZIP_DEFLATED) as zipf:
        for rel in files:
            candidate = rel.strip()
            # Disallow absolute paths for safety
            if Path(candidate).is_absolute():
                warnings.append(f"Skipping absolute path (use relative): {candidate}")
                continue
            add_file(zipf, base, candidate, manifest, warnings)

        # Write the manifest at the end
        created = datetime.now(timezone.utc).replace(microsecond=0)
        manifest_bytes = json.dumps({
            "created_utc": created.isoformat().replace("+00:00", "Z"),
            "count": len(manifest),
            "entries": manifest,
        }, indent=2).encode("utf-8")

        zi = ZipInfo(filename="MANIFEST.json")
        now = datetime.now(timezone.utc)
        zi.date_time = now.utctimetuple()[:6]
        zi.compress_type = ZIP_DEFLATED
        zi.external_attr = (0o100644 & 0xFFFF) << 16
        zipf.writestr(zi, manifest_bytes)

    print(f"\n✅ Created: {out_path}")
    print(f"Files added: {len(manifest)}")
    if warnings:
        print("\n⚠️ Warnings:")
        for w in warnings:
            print(f" - {w}")

    if len(manifest) == 0:
        # Nothing added; make it obvious to callers/CI the pack failed.
        try:
            out_path.unlink()
        except FileNotFoundError:
            pass
        raise SystemExit(3)


if __name__ == "__main__":
    main()
