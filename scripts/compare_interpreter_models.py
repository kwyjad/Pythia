# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Run one interpreter pack through two models and put the reports side by side.

    python -m scripts.compare_interpreter_models \
        --db "$PYTHIA_DB_URL" \
        --pack ai_bundle/current_run_analysis__2026-08.zip \
        --models anthropic:claude-opus-5 anthropic:claude-sonnet-5 \
        --out-dir interpreter_compare

Two decisions are pending on the interpreter's model: whether Opus earns its
cost, and whether the analytical fields added in v4 land or come back hollow.
Arguing about either from memory is worthless. This produces the artifacts to
read, plus the one number that can be compared mechanically: how much of each
report is written in the sentence that would fit any crisis
(``interpreter/validate.py::count_generic_phrases``).

Every run uses ``--force``, so each model gets its own stored version rather
than skipping because the other one already wrote a report for the run. The
reports are written to ``--out-dir`` for reading; the comparison table is
printed and written as JSON beside them.

**This does not change the production model.** It produces evidence. Deciding
from the artifacts is a person's job, and a phrase count is a proxy for
writing quality, never a verdict on it.

``main()`` returns 0 in every outcome.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

DEFAULT_MODELS = ("anthropic:claude-opus-5", "anthropic:claude-sonnet-5")


def _slug(model_ref: str) -> str:
    return model_ref.replace(":", "__").replace("/", "_")


def _substance(content: dict[str, Any]) -> dict[str, Any]:
    """How much of the v4 analytical work the model actually did.

    Counts, not judgements. A field that is present but empty is the failure
    mode worth catching: a model that writes ``"tensions": []`` for every
    entry has declined the task, and a model that writes one sentence of
    genuine reconciliation has done it. The counts say which happened; whether
    the reconciliations are any good is for a reader.
    """
    entries = content.get("attention") or []
    n = len(entries)
    tensions = sum(len(e.get("tensions") or []) for e in entries)
    challenges = sum(
        1 for e in entries
        if isinstance(e.get("challenge"), dict) and e["challenge"].get("verdict")
    )
    verdicts: dict[str, int] = {}
    for e in entries:
        verdict = (e.get("challenge") or {}).get("verdict")
        if verdict:
            verdicts[str(verdict)] = verdicts.get(str(verdict), 0) + 1
    falsifiers = sum(1 for e in entries if (e.get("falsifier") or "").strip())
    second = sum(
        1 for e in entries if (e.get("second_opinion_explanation") or "").strip()
    )

    def _chars(field: str) -> int:
        return sum(len(str(e.get(field) or "")) for e in entries)

    return {
        "n_entries": n,
        "n_tensions": tensions,
        "tensions_per_entry": round(tensions / n, 2) if n else 0.0,
        "n_challenges": challenges,
        "challenge_verdicts": verdicts,
        "n_falsifiers": falsifiers,
        "n_second_opinion_explanations": second,
        "has_cross_cutting": bool((content.get("cross_cutting") or "").strip()),
        "n_scan_disagreements": len(content.get("scan_forecast_disagreements") or []),
        "chars_why_it_stands_out": _chars("why_it_stands_out"),
        "chars_reacting_to": _chars("what_the_model_was_reacting_to"),
    }


def compare_contents(by_model: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The mechanical half of the comparison, for two or more model outputs."""
    from interpreter.validate import count_generic_phrases

    out: dict[str, Any] = {}
    for model_ref, content in by_model.items():
        counts = count_generic_phrases(content)
        out[model_ref] = {
            "generic_phrases": counts,
            "generic_phrase_total": sum(counts.values()),
            "generic_phrases_distinct": len(counts),
            "substance": _substance(content),
        }
    return out


def _table(comparison: dict[str, Any]) -> str:
    models = list(comparison)
    rows = [
        ("entries", lambda c: c["substance"]["n_entries"]),
        ("tensions", lambda c: c["substance"]["n_tensions"]),
        ("tensions/entry", lambda c: c["substance"]["tensions_per_entry"]),
        ("challenges", lambda c: c["substance"]["n_challenges"]),
        ("falsifiers", lambda c: c["substance"]["n_falsifiers"]),
        ("2nd-opinion notes", lambda c: c["substance"]["n_second_opinion_explanations"]),
        ("cross-cutting", lambda c: c["substance"]["has_cross_cutting"]),
        ("scan disagreements", lambda c: c["substance"]["n_scan_disagreements"]),
        ("stock phrases (uses)", lambda c: c["generic_phrase_total"]),
        ("stock phrases (distinct)", lambda c: c["generic_phrases_distinct"]),
        ("chars: stands out", lambda c: c["substance"]["chars_why_it_stands_out"]),
        ("chars: reacting to", lambda c: c["substance"]["chars_reacting_to"]),
    ]
    width = max(len(label) for label, _ in rows) + 2
    header = " " * width + "".join(f"{m[-28:]:>30}" for m in models)
    lines = [header, "-" * len(header)]
    for label, getter in rows:
        cells = "".join(f"{str(getter(comparison[m])):>30}" for m in models)
        lines.append(f"{label:<{width}}{cells}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, help="DuckDB URL or path")
    parser.add_argument("--pack", required=True, help="Current-run bundle zip/dir")
    parser.add_argument("--kind", choices=["current", "scored", "combined"],
                        default="combined")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--out-dir", default="interpreter_compare")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[compare] %(message)s")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from interpreter import run as interpreter_run
    from interpreter import store

    by_model: dict[str, dict[str, Any]] = {}
    results: dict[str, Any] = {}
    for model_ref in args.models:
        model_dir = out_dir / _slug(model_ref)
        model_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info("running %s", model_ref)
        try:
            result = interpreter_run.run_interpreter(
                db=args.db, kind=args.kind, pack_path=args.pack,
                # A new version per model, or the second run would skip
                # because the first already stored an ok report for the run.
                force=True,
                out_dir=str(model_dir),
                model_override=model_ref,
            )
        except Exception as exc:  # noqa: BLE001 - a harness never fails a run
            LOGGER.error("%s failed: %s", model_ref, exc)
            results[model_ref] = {"status": "error", "error": str(exc)}
            continue
        results[model_ref] = result
        interpretation_id = result.get("interpretation_id")
        if not interpretation_id:
            LOGGER.warning("%s produced no stored interpretation", model_ref)
            continue
        try:
            duckdb_io, con = interpreter_run._open_db(args.db)
            try:
                row = store.get_interpretation(con, interpretation_id)
            finally:
                duckdb_io.close_db(con)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("could not read back %s: %s", interpretation_id, exc)
            continue
        content = (row or {}).get("content")
        if isinstance(content, dict):
            by_model[model_ref] = content

    if not by_model:
        print("[compare] no model produced a readable report")
        return 0

    comparison = compare_contents(by_model)
    print(_table(comparison))
    print(
        "\n[compare] The counts are a proxy. Read both reports in "
        f"{out_dir} and judge the two facts each model chose per entry, and "
        "whether its tensions are genuine contradictions or restatements."
    )
    payload = {
        "pack": str(args.pack),
        "kind": args.kind,
        "runs": results,
        "comparison": comparison,
    }
    (out_dir / "comparison.json").write_text(
        json.dumps(payload, indent=1, default=str), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
