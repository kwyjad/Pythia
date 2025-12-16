# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

import argparse
import os
import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

HEADER_LINES = [
    "Pythia",
    "Copyright (c) 2025 Kevin Wyjad",
    "Licensed under the Pythia Non-Commercial Public License v1.0.",
    "See the LICENSE file in the project root for details.",
]

COMMENT_PREFIXES = {
    "python": "#",
    "shell": "#",
    "powershell": "#",
    "slash": "//",
    "yaml": "#",
    "sql": "--",
    "docker": "#",
    "toml": "#",
}

INCLUDE_EXTENSIONS = {
    ".py": "python",
    ".sh": "shell",
    ".bash": "shell",
    ".ps1": "powershell",
    ".js": "slash",
    ".ts": "slash",
    ".tsx": "slash",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".sql": "sql",
    ".toml": "toml",
}

SKIP_DIRS = {
    ".git",
    "vendor",
    "node_modules",
    ".venv",
    "dist",
    "build",
}

CODING_RE = re.compile(r"^#.*coding[:=]\\s*[-A-Za-z0-9_.]+")


def detect_comment_style(path: Path) -> Optional[str]:
    if path.name.startswith("Dockerfile"):
        return "docker"
    extension = path.suffix.lower()
    return INCLUDE_EXTENSIONS.get(extension)


def should_skip_dir(directory: str, root: Path) -> bool:
    if directory in SKIP_DIRS:
        return True
    if directory == "archive" and root.name == "docs":
        return True
    return False


def read_text(path: Path) -> Optional[str]:
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in raw:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


def has_header(lines: List[str], start: int, prefix: str) -> bool:
    if len(lines) - start < len(HEADER_LINES):
        return False
    for offset, text in enumerate(HEADER_LINES):
        line = lines[start + offset].lstrip()
        if not line.startswith(prefix):
            return False
        content = line[len(prefix):].lstrip()
        if content != text:
            return False
    return True


def insertion_index(lines: List[str], file_type: str) -> int:
    index = 0
    if lines and lines[0].startswith("#!"):
        index = 1
    if file_type == "python" and index < len(lines) and CODING_RE.match(lines[index]):
        index += 1
    return index


def file_missing_header(path: Path, text: str, file_type: str) -> bool:
    lines = text.splitlines()
    prefix = COMMENT_PREFIXES[file_type]
    insert_at = insertion_index(lines, file_type)
    return not has_header(lines, insert_at, prefix)


def iter_files(root: Path) -> Iterable[Path]:
    for current_root, dirs, files in os.walk(root):
        root_path = Path(current_root)
        dirs[:] = [d for d in dirs if not should_skip_dir(d, root_path)]
        for filename in files:
            yield root_path / filename


def main() -> None:
    parser = argparse.ArgumentParser(description="Check for copyright headers in source files.")
    parser.add_argument("root", nargs="?", default=".", help="Repository root to scan")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    missing: List[Path] = []
    skipped: List[Tuple[Path, str]] = []

    for path in iter_files(root):
        if path.is_dir():
            continue
        file_type = detect_comment_style(path)
        if not file_type:
            skipped.append((path, "unsupported type"))
            continue
        text = read_text(path)
        if text is None:
            skipped.append((path, "unreadable or binary"))
            continue
        if file_missing_header(path, text, file_type):
            missing.append(path)

    if missing:
        print("Files missing headers:")
        for path in missing:
            print(f"  - {path.relative_to(root)}")
        exit(1)

    print("All checked files contain the required header.")
    print(f"Skipped files: {len(skipped)}")
    for path, reason in skipped:
        print(f"  - {path.relative_to(root)} ({reason})")


if __name__ == "__main__":
    main()
