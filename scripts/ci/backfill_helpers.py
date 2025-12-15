# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Helpers for resolver backfill CI workflows."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import zipfile
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ci.flatten_artifacts import flatten_zips


def _write_zip(zip_path: Path, sources: Sequence[Path]) -> bool:
    existing = [source for source in sources if source.exists()]
    if not existing:
        return False

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        wrote = False
        for source in existing:
            if source.is_file():
                zf.write(source, arcname=source.name)
                wrote = True
                continue

            base = source.name or source.as_posix().replace("/", "_") or "artifact"
            for rel_path in sorted(p for p in source.rglob("*")):
                if not rel_path.is_file():
                    continue
                arcname = Path(base) / rel_path.relative_to(source)
                zf.write(rel_path, arcname=str(arcname))
                wrote = True

        if not wrote:
            zf.writestr("EMPTY.txt", "no files were collected")

    return True


def build_flat_artifacts_bundle(out_path: Path) -> None:
    tasks: Sequence[tuple[Path, Sequence[Path]]] = (
        (
            Path("diagnostics/flatten_inputs/connector-logs.zip"),
            [Path("diagnostics/ingestion/logs")],
        ),
        (
            Path("diagnostics/flatten_inputs/connector-diagnostics.zip"),
            [Path("diagnostics/ingestion")],
        ),
        (
            Path("diagnostics/flatten_inputs/dtm-diagnostics-raw.zip"),
            [Path("diagnostics/ingestion/raw")],
        ),
        (
            Path("diagnostics/flatten_inputs/dtm-diagnostics-metrics.zip"),
            [Path("diagnostics/ingestion/metrics")],
        ),
        (
            Path("diagnostics/flatten_inputs/dtm-diagnostics-samples.zip"),
            [Path("diagnostics/ingestion/samples")],
        ),
        (
            Path("diagnostics/flatten_inputs/resolver-backfill-ingest.zip"),
            [
                Path("data"),
                Path("resolver/staging"),
                Path("resolver/logs/ingestion"),
            ],
        ),
    )

    produced: list[Path] = []
    for zip_path, sources in tasks:
        if _write_zip(zip_path, sources):
            produced.append(zip_path)

    if not produced:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED):
            pass
        return

    flatten_zips(out_path, produced)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build helper artifacts for CI backfill workflows")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("diagnostics/all-artifacts-flat.zip"),
        help="Output path for the flattened artifact bundle",
    )
    args = parser.parse_args()

    build_flat_artifacts_bundle(args.out)


if __name__ == "__main__":
    main()
