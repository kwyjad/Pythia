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


def detect_newline(text: str) -> str:
    if "\r\n" in text:
        return "\r\n"
    return "\n"


def read_text(path: Path) -> Tuple[Optional[str], Optional[str]]:
    try:
        raw = path.read_bytes()
    except OSError:
        return None, None
    if b"\x00" in raw:
        return None, None
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None, None
    newline = detect_newline(text)
    return text, newline


def header_lines(prefix: str) -> List[str]:
    return [f"{prefix} {line}" for line in HEADER_LINES]


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


def apply_header_to_text(lines: List[str], file_type: str, newline: str) -> Optional[str]:
    prefix = COMMENT_PREFIXES[file_type]
    insert_at = insertion_index(lines, file_type)
    if has_header(lines, insert_at, prefix):
        return None

    new_lines: List[str] = []
    new_lines.extend(lines[:insert_at])
    new_lines.extend(header_lines(prefix))
    new_lines.append("")
    new_lines.extend(lines[insert_at:])
    content = newline.join(new_lines)
    return content


def process_file(path: Path) -> Tuple[str, Optional[str]]:
    file_type = detect_comment_style(path)
    if not file_type:
        return "skipped", "unsupported type"

    text, newline = read_text(path)
    if text is None or newline is None:
        return "skipped", "unreadable or binary"

    lines = text.splitlines()
    had_trailing_newline = text.endswith("\n") or text.endswith("\r\n")
    updated_content = apply_header_to_text(lines, file_type, newline)
    if updated_content is None:
        return "skipped", "header present"

    if had_trailing_newline and not updated_content.endswith(newline):
        updated_content += newline
    if not had_trailing_newline and updated_content.endswith(newline):
        updated_content = updated_content[: -len(newline)]

    path.write_text(updated_content, encoding="utf-8", newline="")
    return "updated", None


def iter_files(root: Path) -> Iterable[Path]:
    for current_root, dirs, files in os.walk(root):
        root_path = Path(current_root)
        dirs[:] = [d for d in dirs if not should_skip_dir(d, root_path)]
        for filename in files:
            yield root_path / filename


def main() -> None:
    parser = argparse.ArgumentParser(description="Add copyright headers to source files.")
    parser.add_argument("root", nargs="?", default=".", help="Repository root to scan")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    updated = []
    skipped = []
    errors = []

    for path in iter_files(root):
        if path.is_dir():
            continue
        status, reason = process_file(path)
        if status == "updated":
            updated.append(path)
        elif status == "skipped":
            skipped.append((path, reason or ""))
        else:
            errors.append((path, reason or ""))

    print("Files updated:", len(updated))
    for path in updated:
        print(f"  + {path.relative_to(root)}")

    print("Files skipped:", len(skipped))
    for path, reason in skipped:
        print(f"  - {path.relative_to(root)} ({reason})")

    if errors:
        print("Errors:", len(errors))
        for path, reason in errors:
            print(f"  ! {path.relative_to(root)} ({reason})")


if __name__ == "__main__":
    main()
