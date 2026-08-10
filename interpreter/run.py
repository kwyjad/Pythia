# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Interpreter runner: pack -> Opus 5 -> validated JSON -> markdown -> DB.

Usage:
    python -m interpreter.run \
        --db "$PYTHIA_DB_URL" \
        --kind combined \
        --pack ai_bundle/current_run_analysis__2026-08.zip \
        [--run-id fc_...] [--hs-run-id hs_...] [--scored-run-id 2026-08] \
        [--template-version v1] [--force] [--dry-run] [--out-dir ./out]

Kinds:
- ``scored``   — --pack is the scored bundle; writes the performance-side
  interpretation the combined report later folds in.
- ``current``  — --pack is the current-run bundle; Part A only.
- ``combined`` — --pack is the current-run bundle; Part B comes from the most
  recent stored ``scored`` interpretation (explicit "no scored run available
  yet" when none exists).

``--force`` produces a NEW version row for the same run instead of skipping.
``main()`` returns 0 in every outcome — a report bug must never fail the
pipeline; failures are stored with status ``failed_generation`` /
``failed_validation`` so they stay inspectable.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

from interpreter import config, packs, prompts, render, store, validate

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model call (module-level seam so tests monkeypatch interpreter.run._call_model)
# ---------------------------------------------------------------------------


def _resolve_model_ref() -> str:
    override = config.model_id_override()
    if override:
        return override if ":" in override else f"anthropic:{override}"
    from pythia.llm_profiles import get_role_model

    ref = get_role_model(config.ROLE)
    if not ref:
        raise RuntimeError(
            f"role {config.ROLE!r} does not resolve to a model; add it to "
            "pythia/config.yaml's roles block and pythia/llm_profiles._ROLE_FALLBACKS"
        )
    return ref


def _call_model(prompt: str, model_ref: str) -> tuple[str, dict, str]:
    """One synchronous interpreter call. Never raises.

    Goes through forecaster.providers.call_chat_ms — the repo's single
    provider path — with log_call=False (a rich llm_calls row is written by
    _log_llm_call instead; the generic row would double-count the spend).
    Thinking depth is requested via ModelSpec.thinking -> output_config.effort
    (emitted only for effort-capable models); temperature is never set at
    this call site (the opus-5 prefix guard would drop it anyway).
    """
    import asyncio

    from forecaster.providers import ModelSpec, call_chat_ms
    from pythia.llm_profiles import split_model_ref

    provider, model_id = split_model_ref(model_ref, default_provider="anthropic")
    spec = ModelSpec(
        name=model_id,
        provider=provider,
        model_id=model_id,
        purpose="interpreter",
        thinking=config.thinking_level(),
    )
    try:
        return asyncio.run(
            call_chat_ms(
                spec,
                prompt,
                prompt_key="interpreter.report",
                component="Interpreter",
                log_call=False,
                timeout_sec=config.call_timeout_sec(),
                # Thinking shares this budget — the SPD-class ceiling, not
                # the 16k general one.
                max_output_tokens=config.max_output_tokens(),
            )
        )
    except Exception as exc:  # noqa: BLE001 - network/runtime failures
        LOGGER.warning("[interpreter] model call failed: %s", exc)
        return "", {}, str(exc)


def _log_llm_call(
    *,
    model_ref: str,
    prompt: str,
    response: str,
    usage: dict[str, Any],
    error: str,
    run_id: str | None,
    hs_run_id: str | None,
    kind: str,
) -> None:
    """One rich llm_calls row per REAL interpreter call (phase='interpreter').

    Same contract as the PA machine's extraction logging: log_call=False at
    call_chat_ms, the rich row written here, telemetry failures never break
    the run. phase='interpreter' maps to the 'interpreter' bucket in
    resolver/query/costs.py::phase_group and the frontend PHASE_* maps.
    """
    try:
        import asyncio

        from forecaster.llm_logging import log_forecaster_llm_call
        from pythia.llm_profiles import split_model_ref

        provider, model_id = split_model_ref(model_ref, default_provider="anthropic")
        asyncio.run(
            log_forecaster_llm_call(
                run_id=run_id or "",
                question_id="",
                prompt_text=prompt,
                model_name=model_id,
                provider=provider,
                model_id=model_id,
                phase="interpreter",
                call_type=f"interpreter_{kind}",
                hs_run_id=hs_run_id,
                response_text=response,
                usage=usage or {},
                error_text=(error or None),
            )
        )
    except Exception as exc:  # noqa: BLE001 - telemetry must never break a run
        LOGGER.warning("[interpreter] llm_calls telemetry row not written: %s", exc)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse_model_json(text: str) -> dict[str, Any] | None:
    """Tolerant JSON extraction: strip code fences, then try the whole
    string, then the first balanced top-level object."""
    if not text:
        return None
    cleaned = _FENCE.sub("", text).strip()
    for candidate in (cleaned,):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:  # noqa: BLE001
            pass
    start = cleaned.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(cleaned[start : i + 1])
                    return obj if isinstance(obj, dict) else None
                except Exception:  # noqa: BLE001
                    return None
    return None


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _open_db(db: str):
    from resolver.db import duckdb_io

    return duckdb_io, duckdb_io.get_db(db or duckdb_io.DEFAULT_DB_URL)


def _error_count(report) -> list[str]:
    """Every error string in a validation report, flattened."""
    return [
        err for check in report.checks.values() if not check.passed
        for err in (check.errors or [])
    ]


def _repair_entry_identity(
    content: dict[str, Any], identity: dict[str, dict[str, str]]
) -> int:
    """Overwrite each attention entry's iso3/hazard_code/metric from the pack.

    Returns the number of fields corrected, logged so a persistently sloppy
    model is visible rather than silently patched.
    """
    if not isinstance(content, dict) or not identity:
        return 0
    fixed = 0
    for entry in content.get("attention") or []:
        if not isinstance(entry, dict):
            continue
        for qid in entry.get("question_ids") or []:
            truth = identity.get(str(qid))
            if not truth:
                continue
            for name, value in truth.items():
                if str(entry.get(name) or "") != value:
                    LOGGER.info(
                        "[interpreter] %s: %s %r corrected to %r from the pack",
                        qid, name, entry.get(name), value,
                    )
                    entry[name] = value
                    fixed += 1
            break
    return fixed


def _maybe_inherit_test_mode(con, hs_run_id: str | None) -> None:
    """Derive test mode from the interpreted run (the Sibyl pattern).

    The workflow step that invokes this runner cannot carry the upstream
    ``test_mode`` input, so a test-mode cycle would otherwise store its
    interpretation (and its llm_calls telemetry) stamped as production data.
    An explicit PYTHIA_TEST_MODE always wins; the derivation only ever turns
    the flag ON, never off.
    """
    import os

    if os.getenv("PYTHIA_TEST_MODE") or not hs_run_id:
        return
    try:
        row = con.execute(
            "SELECT COALESCE(is_test, FALSE) FROM hs_runs WHERE hs_run_id = ?",
            [hs_run_id],
        ).fetchone()
    except Exception as exc:  # noqa: BLE001 - probe only, never fatal
        LOGGER.warning("[interpreter] hs_runs is_test probe failed: %s", exc)
        return
    if row and bool(row[0]):
        os.environ["PYTHIA_TEST_MODE"] = "1"
        LOGGER.info(
            "[interpreter] hs_run %s is a test run — stamping outputs is_test",
            hs_run_id,
        )


def run_interpreter(
    *,
    db: str,
    kind: str,
    pack_path: str,
    run_id: str | None = None,
    hs_run_id: str | None = None,
    scored_run_id: str | None = None,
    template_version: str | None = None,
    force: bool = False,
    dry_run: bool = False,
    out_dir: str | None = None,
) -> dict[str, Any]:
    """Run one interpretation. Returns a small result dict (for tests/logs)."""
    template_version = template_version or config.template_version()
    result: dict[str, Any] = {"kind": kind, "status": "skipped"}

    if not config.enabled():
        LOGGER.info("[interpreter] PYTHIA_INTERPRETER_ENABLED=0 — skipping")
        result["reason"] = "disabled"
        return result

    pack = packs.load_pack(pack_path)
    if kind in ("current", "combined") and pack.kind != "current":
        raise ValueError(f"kind {kind} needs a current-run bundle, got {pack.kind}")
    if kind == "scored" and pack.kind != "scored":
        raise ValueError(f"kind scored needs a scored bundle, got {pack.kind}")

    run_id = run_id or (str(pack.manifest.get("run_id")) if pack.manifest.get("run_id") else None)
    hs_run_id = hs_run_id or (
        str(pack.manifest.get("hs_run_id")) if pack.manifest.get("hs_run_id") else None
    )
    if kind == "scored" and not scored_run_id:
        # The scored bundle has no single run id; its monthly label is the
        # stable key a re-run of the same scoring round shares.
        scored_run_id = str(
            pack.manifest.get("generated_at", "")
        )[:7] or None

    duckdb_io, con = _open_db(db)
    try:
        store.ensure_table(con)
        _maybe_inherit_test_mode(con, hs_run_id)

        if not force:
            existing = store.existing_ok_version(con, kind, run_id, scored_run_id)
            if existing is not None:
                LOGGER.info(
                    "[interpreter] %s interpretation v%d already exists for this "
                    "run — skipping (use --force for a new version)",
                    kind, existing,
                )
                result.update({"reason": "exists", "version": existing})
                return result

        # --- Figures + Part B material ------------------------------------
        per_question = packs.question_figures(pack) if pack.kind == "current" else {}
        global_figures: dict[str, Any] = (
            packs.performance_figures(pack) if pack.kind == "scored" else {}
        )
        if pack.kind == "current":
            # Scan-scope counts the report opens with. Without these in the
            # global map the run_summary placeholders would resolve to
            # "[figure unavailable]" and the opening paragraph would read as
            # a hole where the scale of the run should be.
            global_figures.update(packs.run_summary_figures(pack))
        scored_section: str | None = None
        if kind == "combined":
            latest = store.latest_scored(con)
            if latest and latest.get("content"):
                scored_section = json.dumps(latest["content"], indent=1, default=str)
                scored_run_id = scored_run_id or latest.get("scored_run_id")
                # The stored map is {"global": {...}, "per_question": {...}} —
                # merge both so the scored interpretation's placeholders
                # resolve here without reloading the scored pack.
                figs = latest.get("figures") or {}
                if isinstance(figs.get("global"), dict):
                    global_figures.update(figs["global"])
                if isinstance(figs.get("per_question"), dict):
                    for qid, fmap in figs["per_question"].items():
                        if isinstance(fmap, dict):
                            per_question.setdefault(qid, fmap)

        # --- Prompts -------------------------------------------------------
        pack_text, pack_stats = packs.assemble_input_text(
            pack,
            budget_tokens=config.max_pack_tokens(),
            estimate_tokens=config.estimate_tokens,
        )
        system_prompt = prompts.build_system_prompt(template_version)
        user_prompt = prompts.build_user_prompt(
            kind=kind, pack_text=pack_text,
            version=template_version, scored_section=scored_section,
        )
        full_prompt = system_prompt + "\n\n---\n\n" + user_prompt
        p_hash = prompts.prompt_hash(system_prompt, user_prompt)

        provenance = {
            "kind": kind,
            "run_id": run_id,
            "hs_run_id": hs_run_id,
            "scored_run_id": scored_run_id,
            "template_version": template_version,
            "prompt_hash": p_hash,
            "pack_hash": pack.pack_hash,
        }

        if dry_run:
            LOGGER.info(
                "[interpreter] dry run: prompt ~%d tokens (pack %d, records "
                "kept %d, truncated %d)",
                config.estimate_tokens(full_prompt),
                pack_stats["input_tokens_estimate"],
                len(pack_stats["records_kept"]),
                len(pack_stats["records_truncated"]),
            )
            if out_dir:
                out = Path(out_dir)
                out.mkdir(parents=True, exist_ok=True)
                (out / f"prompt_{kind}.txt").write_text(full_prompt, encoding="utf-8")
            result.update({"reason": "dry_run", "prompt_tokens": config.estimate_tokens(full_prompt)})
            return result

        # --- Model call ----------------------------------------------------
        model_ref = _resolve_model_ref()
        response_text, usage, error = _call_model(full_prompt, model_ref)
        _log_llm_call(
            model_ref=model_ref, prompt=full_prompt, response=response_text,
            usage=usage, error=error, run_id=run_id, hs_run_id=hs_run_id,
            kind=kind,
        )
        model_id = model_ref.split(":", 1)[-1]

        cost_usd = None
        try:
            from forecaster.providers import estimate_cost_usd

            cost_usd = estimate_cost_usd(model_id, usage or {})
        except Exception:  # noqa: BLE001
            pass
        input_tokens = int((usage or {}).get("prompt_tokens") or 0) or None
        output_tokens = int((usage or {}).get("completion_tokens") or 0) or None

        common_kwargs = dict(
            kind=kind, run_id=run_id, hs_run_id=hs_run_id,
            scored_run_id=scored_run_id, template_version=template_version,
            model_name=model_id, thinking_level=config.thinking_level(),
            prompt_hash=p_hash, pack_hash=pack.pack_hash,
            cost_usd=cost_usd, input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        content = parse_model_json(response_text) if not error else None
        if content is None:
            interpretation_id, version = store.save_interpretation(
                con, content=None, content_md=None, figures=None,
                status="failed_generation",
                validation={"error": error or "unparseable model output",
                            "response_chars": len(response_text or "")},
                **common_kwargs,
            )
            LOGGER.warning(
                "[interpreter] generation failed (%s) — stored %s v%d",
                error or "unparseable output", interpretation_id, version,
            )
            result.update({"status": "failed_generation", "version": version,
                           "interpretation_id": interpretation_id})
            return result

        # The pack owns each question's country, hazard and metric. Copying
        # them is a transcription step with no judgement in it, so a slip is
        # repaired here rather than published or made fatal: the August run
        # copied FATALITIES into hazard_code and the report printed "Mali,
        # fatalities: deaths". The judgement fields (category, hazard_family)
        # are NOT repaired — those the validator holds the model to.
        _repair_entry_identity(content, packs.pack_identity(pack))

        # --- Validation (schema, referential, numeric, prose, style, sections) ---
        def _validate(candidate: dict[str, Any]):
            return validate.validate_interpretation(
                candidate,
                kind=kind,
                valid_question_ids=packs.pack_question_ids(pack) | set(per_question),
                per_question=per_question,
                global_figures=global_figures,
                con=con,
                run_id=run_id,
                pack_categories=packs.pack_categories(pack),
            )

        report = _validate(content)

        # One correction pass. Most failures that survive the prompt are a
        # single slip in one field: a stray numeral, one banned construction.
        # Re-running the whole cycle by hand to fix one sentence costs a full
        # model call and leaves a banner on the dashboard in the meantime,
        # so the runner asks for the fix itself, quoting the exact complaints.
        # Capped at one attempt: a model that cannot satisfy the checks twice
        # is telling us something about the checks, and the report still
        # stores either way.
        retries = 0
        if not report.passed and config.validation_retries() > 0:
            complaints = [
                f"- {name}: {err}"
                for name, check in report.checks.items()
                if not check.passed
                for err in (check.errors or [])
            ]
            LOGGER.warning(
                "[interpreter] validation failed on %d point(s); asking the "
                "model to correct them", len(complaints),
            )
            fix_prompt = (
                full_prompt
                + "\n\n## Your previous answer failed these checks\n\n"
                + "\n".join(complaints[:60])
                + "\n\nReturn the WHOLE report again as JSON, identical "
                "except for the points above. Change nothing else: not the "
                "entries covered, not their categories, not their order. "
                "Fix only what is listed."
            )
            retry_text, retry_usage, retry_error = _call_model(fix_prompt, model_ref)
            _log_llm_call(
                model_ref=model_ref, prompt=fix_prompt, response=retry_text,
                usage=retry_usage, error=retry_error, run_id=run_id,
                hs_run_id=hs_run_id, kind=f"{kind}_retry",
            )
            retries = 1
            retry_content = parse_model_json(retry_text) if not retry_error else None
            if retry_content is not None:
                _repair_entry_identity(retry_content, packs.pack_identity(pack))
                retry_report = _validate(retry_content)
                if retry_report.passed or len(_error_count(retry_report)) < len(complaints):
                    # Keep the better answer, never the worse one.
                    content, report = retry_content, retry_report
                # The retry's tokens are part of this interpretation's cost.
                try:
                    from forecaster.providers import estimate_cost_usd

                    extra = estimate_cost_usd(model_id, retry_usage or {})
                    if extra:
                        cost_usd = (cost_usd or 0.0) + extra
                except Exception:  # noqa: BLE001
                    pass
                input_tokens = (input_tokens or 0) + int(
                    (retry_usage or {}).get("prompt_tokens") or 0
                ) or None
                output_tokens = (output_tokens or 0) + int(
                    (retry_usage or {}).get("completion_tokens") or 0
                ) or None
                common_kwargs.update(
                    cost_usd=cost_usd, input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
            else:
                LOGGER.warning(
                    "[interpreter] correction pass produced no usable JSON (%s); "
                    "keeping the first answer", retry_error or "unparseable",
                )

        status = "ok" if report.passed else "failed_validation"

        # --- Render (the report is still written on validation failure so it
        # can be inspected; the dashboard shows it behind a warning banner).
        resolver = render.FigureResolver(
            per_question, global_figures,
            spd_by_question=(
                packs.question_distributions(pack)
                if pack.kind == "current" else {}
            ),
        )
        content_md = render.render_markdown(content, resolver, provenance=provenance)
        figures = {"global": global_figures, "per_question": per_question}

        interpretation_id, version = store.save_interpretation(
            con, content=content, content_md=content_md, figures=figures,
            status=status,
            validation={**report.as_dict(),
                        "unresolved_figures": resolver.misses,
                        "strict": config.strict_validation(),
                        "pack_stats": pack_stats},
            **common_kwargs,
        )
        LOGGER.info(
            "[interpreter] stored %s v%d (status=%s, cost=%s, unresolved figs=%d)",
            interpretation_id, version, status, cost_usd, len(resolver.misses),
        )
        if status != "ok":
            for name, check in report.checks.items():
                for err in check.errors[:20]:
                    LOGGER.warning("[interpreter] %s check: %s", name, err)

        # Strict mode: a failed validation suppresses the publication
        # artifacts — the row is still stored (inspectable, never silent),
        # but nothing consumer-facing is written.
        suppressed = config.strict_validation() and status != "ok"
        if out_dir and not suppressed:
            out = Path(out_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / f"report_{kind}_v{version}.md").write_text(content_md, encoding="utf-8")
            (out / f"content_{kind}_v{version}.json").write_text(
                json.dumps(content, indent=1, default=str), encoding="utf-8"
            )
        elif suppressed:
            LOGGER.warning(
                "[interpreter] strict validation: publication artifacts "
                "suppressed for %s v%d", interpretation_id, version,
            )
        result.update({"status": status, "version": version,
                       "interpretation_id": interpretation_id,
                       "validation": report.as_dict(),
                       "published": not suppressed})
        return result
    finally:
        duckdb_io.close_db(con)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, help="DuckDB URL or path")
    parser.add_argument("--kind", choices=["current", "scored", "combined"],
                        default="combined")
    parser.add_argument("--pack", required=True,
                        help="Bundle zip/dir (current-run bundle for "
                             "current/combined, scored bundle for scored)")
    parser.add_argument("--run-id", default=None, help="Forecaster run id")
    parser.add_argument("--hs-run-id", default=None, help="HS run id")
    parser.add_argument("--scored-run-id", default=None,
                        help="Scored-round key (default: bundle month label)")
    parser.add_argument("--template-version", default=None)
    parser.add_argument("--force", action="store_true",
                        help="New version row even when one exists")
    parser.add_argument("--dry-run", action="store_true",
                        help="Assemble prompts, no model call, no store")
    parser.add_argument("--out-dir", default=None,
                        help="Also write report/content files here")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[interpreter] %(message)s")
    try:
        result = run_interpreter(
            db=args.db, kind=args.kind, pack_path=args.pack,
            run_id=args.run_id, hs_run_id=args.hs_run_id,
            scored_run_id=args.scored_run_id,
            template_version=args.template_version,
            force=args.force, dry_run=args.dry_run, out_dir=args.out_dir,
        )
        print(f"[interpreter] RESULT={json.dumps(result, default=str)}")
    except Exception as exc:  # noqa: BLE001 - never fail the pipeline
        LOGGER.error("[interpreter] failed: %s", exc)
        print(f"[interpreter] RESULT={json.dumps({'status': 'error', 'error': str(exc)})}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
