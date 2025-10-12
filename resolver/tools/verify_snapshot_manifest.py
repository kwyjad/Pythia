"""Validate snapshot manifests by recomputing file hashes and row counts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _count_rows(path: Path) -> int:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        with path.open("r", encoding="utf-8") as handle:
            total = sum(1 for _ in handle)
        return max(total - 1, 0)
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq  # type: ignore

            return int(pq.ParquetFile(path).metadata.num_rows)
        except Exception:  # pragma: no cover - fallbacks for minimal environments
            try:
                import duckdb  # type: ignore

                return int(
                    duckdb.sql("SELECT COUNT(*) FROM read_parquet(?)", [str(path)])
                    .fetchone()[0]
                )
            except Exception:
                import pandas as pd  # type: ignore

                return int(len(pd.read_parquet(path)))
    raise ValueError(f"Unsupported file extension for row counting: {path.suffix}")


def _resolve_path(base: Path, candidate: str) -> Path:
    candidate_path = Path(candidate)
    if candidate_path.is_absolute():
        return candidate_path
    return (base / candidate_path).resolve()


def verify_manifest(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = payload.get("files")
    if not isinstance(files, list):
        raise ValueError("Manifest does not contain a 'files' array")

    base_dir = manifest_path.parent
    errors: list[str] = []

    for entry in files:
        if not isinstance(entry, dict):
            errors.append(f"Invalid manifest file entry: {entry!r}")
            continue
        path_value = entry.get("path")
        sha_expected = str(entry.get("sha256") or "")
        rows_expected = entry.get("rows")
        if path_value is None:
            errors.append(f"Manifest entry missing 'path': {entry!r}")
            continue
        target = _resolve_path(base_dir, str(path_value))
        if not target.exists():
            errors.append(f"Snapshot artifact missing: {target}")
            continue
        actual_sha = _sha256(target)
        if sha_expected and actual_sha != sha_expected:
            errors.append(
                f"SHA256 mismatch for {target}: expected {sha_expected} got {actual_sha}"
            )
        if rows_expected is not None:
            actual_rows = _count_rows(target)
            if int(rows_expected) != actual_rows:
                errors.append(
                    f"Row count mismatch for {target}: expected {rows_expected} got {actual_rows}"
                )

    if errors:
        raise ValueError("\n".join(errors))

    if "schema_version" not in payload:
        raise ValueError("Manifest missing 'schema_version'")
    if "generated_at" not in payload:
        raise ValueError("Manifest missing 'generated_at'")

    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Path to manifest.json")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        verify_manifest(args.manifest)
    except Exception as exc:  # pragma: no cover - CLI wrapper
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
