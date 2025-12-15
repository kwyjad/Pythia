# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd

STANDARD_COLUMNS = [
    "iso3",
    "as_of_date",
    "metric",
    "value",
    "source_system",
    "collection_type",
    "coverage",
    "freshness_days",
    "origin_iso3",
    "destination_iso3",
    "method_note",
    "series",
    "indicator",
    "indicator_kind",
    "qa_rank",
]


def _read_candidate_file(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [column for column in STANDARD_COLUMNS if column not in frame.columns]
    for column in missing:
        frame[column] = pd.NA
    return frame[STANDARD_COLUMNS]


def _coerce_types(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    text_columns: Iterable[str] = [
        "iso3",
        "metric",
        "source_system",
        "collection_type",
        "coverage",
        "origin_iso3",
        "destination_iso3",
        "method_note",
        "series",
        "indicator",
        "indicator_kind",
    ]

    for column in text_columns:
        frame[column] = frame[column].astype("string")

    frame["as_of_date"] = pd.to_datetime(frame["as_of_date"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")

    freshness = pd.to_numeric(frame["freshness_days"], errors="coerce")
    frame["freshness_days"] = pd.Series(pd.array(freshness, dtype="Int64"), index=frame.index)

    qa_rank = pd.to_numeric(frame["qa_rank"], errors="coerce")
    frame["qa_rank"] = pd.Series(pd.array(qa_rank, dtype="Int64"), index=frame.index)

    return frame


def _build_summary(frame: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    return {
        "total_rows": int(len(frame)),
        "by_source_system": frame["source_system"].dropna().value_counts().sort_index().to_dict(),
        "by_metric": frame["metric"].dropna().value_counts().sort_index().to_dict(),
    }


def main() -> int:
    candidates_dir = Path(os.environ.get("CANDIDATES_DIR", "artifacts/precedence/candidates"))
    output_dir = Path(os.environ.get("DIAGNOSTICS_DIR", "diagnostics/precedence"))

    candidates_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_paths = sorted(candidates_dir.glob("*_candidates.csv"))

    frames = [_read_candidate_file(path) for path in candidate_paths]
    if frames:
        combined = pd.concat(frames, ignore_index=True)
    else:
        combined = pd.DataFrame(columns=STANDARD_COLUMNS)

    combined = combined.dropna(how="all")
    combined = _coerce_types(combined)

    union_path = output_dir / "union_candidates.csv"
    summary_path = output_dir / "union_summary.json"

    combined.to_csv(union_path, index=False)

    summary = _build_summary(combined)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if combined.empty:
        print(
            f"No candidates found in {candidates_dir}; wrote empty union to {union_path}",
            flush=True,
        )
    else:
        sources = ", ".join(f"{key}:{value}" for key, value in summary["by_source_system"].items()) or "none"
        metrics = ", ".join(f"{key}:{value}" for key, value in summary["by_metric"].items()) or "none"
        print(
            "Unioned"
            f" {len(candidate_paths)} files"
            f" → {len(combined)} rows"
            f" | sources: {sources}"
            f" | metrics: {metrics}",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
