# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Exactly what this run was configured with, values and all.

``model_config.json`` said whether a handful of flags were set. That is
enough to know a flag existed and not enough to know what it did: the
prompt-cache flags were on in the workflow for a month while the batched
path had a hit rate of zero by construction, and a set/unset table cannot
tell those apart. So this file carries values.

Credentials are fingerprinted rather than dropped
(``<redacted:sha256:xxxxxxxx>``), because the question that comes up in an
incident is not "what is the key" but "is it the same key the last good
run used".

Beside the environment: the commit and branch, the hash of every workflow
file involved (a stage running an older main is a real failure mode — a
Resolver Update in August ran a commit that predated the on-runner
CrisisWatch refresh and nothing said so), the runner and interpreter, the
installed packages, and the LLM profile as RESOLVED — ensemble members with
their per-model thinking effort, every role, and the prices actually in
force for the models this run used.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from scripts.debug_bundle.redaction import redact_env_value, redact_text

# Workflow files whose content decides what this cycle did. Their hashes
# say whether every stage ran the same code.
WORKFLOW_FILES = (
    ".github/workflows/pythia_pipeline_stage.yml",
    ".github/workflows/poll_llm_batches.yml",
    ".github/workflows/run_sibyl.yml",
    ".github/workflows/run_horizon_scanner.yml",
    ".github/workflows/resolver_update.yml",
    ".github/workflows/ingest-structured-data.yml",
    ".github/workflows/publish_latest_data.yml",
)

# Non-PYTHIA_* variables that shape a run's behaviour or scope.
EXTRA_ENV_NAMES = (
    "GITHUB_REPOSITORY", "GITHUB_REF", "GITHUB_SHA", "GITHUB_RUN_ID",
    "GITHUB_RUN_ATTEMPT", "GITHUB_WORKFLOW", "GITHUB_JOB", "GITHUB_ACTOR",
    "RUNNER_OS", "RUNNER_ARCH", "RUNNER_NAME", "ImageOS", "ImageVersion",
    "GDACS_MONTHS", "FEWSNET_MONTHS", "IPC_API_MONTHS", "GDELT_EVENTS_DAYS",
    "ACLED_MAX_RUNTIME_SEC", "CONNECTOR_TIMEOUT",
    "POLLER_CHAIN_DEPTH", "POLLER_MAX_CHAIN",
    "CANONICAL_DB_RUN_ID", "CANONICAL_DB_WORKFLOW", "CANONICAL_DB_ARTIFACT_NAME",
    "SIBYL_MODEL", "SIBYL_K", "SIBYL_N_QUESTIONS", "SIBYL_RUN_HARD_CAP_USD",
    "HS_MAX_WORKERS", "FORECASTER_SPD_MAX_WORKERS", "FORECASTER_RESEARCH_MAX_WORKERS",
)


def _run(cmd: list[str], *, cwd: Path | None = None) -> str | None:
    try:
        out = subprocess.run(
            cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True, timeout=60
        )
    except Exception:
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def _git(repo_root: Path) -> dict[str, Any]:
    sha = _run(["git", "rev-parse", "HEAD"], cwd=repo_root)
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root)
    # In Actions the checkout is detached, so HEAD's symbolic name is
    # "HEAD" and GITHUB_REF_NAME is the honest answer.
    if branch == "HEAD":
        branch = os.getenv("GITHUB_REF_NAME") or branch
    return {
        "commit_sha": sha or os.getenv("GITHUB_SHA") or None,
        "branch": branch or None,
        "commit_subject": _run(["git", "log", "-1", "--pretty=%s"], cwd=repo_root),
        "commit_date": _run(["git", "log", "-1", "--pretty=%cI"], cwd=repo_root),
        "dirty": bool(_run(["git", "status", "--porcelain"], cwd=repo_root)),
    }


def _file_sha256(path: Path) -> str | None:
    import hashlib  # noqa: PLC0415

    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except Exception:
        return None


def _pip_freeze() -> list[str] | str:
    text = _run([sys.executable, "-m", "pip", "freeze", "--disable-pip-version-check"])
    if text is None:
        return "pip freeze unavailable"
    return sorted(line.strip() for line in text.splitlines() if line.strip())


def _llm_profile() -> dict[str, Any]:
    """The profile as the code resolves it, not as the YAML reads."""

    out: dict[str, Any] = {}
    try:
        from pythia import llm_profiles  # noqa: PLC0415

        out["profile_name"] = llm_profiles.get_current_profile()
        out["registry"] = llm_profiles.get_model_registry()
        out["ensemble"] = llm_profiles.get_ensemble_resolved()
        roles: dict[str, Any] = {}
        role_names = set(getattr(llm_profiles, "_ROLE_FALLBACKS", {}))
        try:
            profile_roles = llm_profiles._get_profile_data().get("roles") or {}
            role_names |= set(profile_roles)
        except Exception:
            pass
        for role in sorted(role_names):
            try:
                roles[role] = llm_profiles.get_role_model(role)
            except Exception as exc:  # noqa: BLE001
                roles[role] = f"<unresolved: {exc}>"
        out["roles"] = roles
    except Exception as exc:  # noqa: BLE001
        out["error"] = redact_text(f"{type(exc).__name__}: {exc}")
    return out


def _price_table(model_ids: list[str], repo_root: Path) -> dict[str, Any]:
    """The prices actually applied, for the models this run actually used.

    A missing entry means every call on that model was logged at $0, which
    also silently breaks any budget cap keyed on cost — so an absent price
    is a finding, not a blank.
    """

    table: dict[str, Any] = {}
    try:
        raw = json.loads((repo_root / "pythia" / "model_costs.json").read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"error": redact_text(str(exc))}
    for model_id in sorted({m for m in model_ids if m}):
        entry = raw.get(model_id)
        if entry is None:
            for key, value in raw.items():
                if key.split("/")[-1] == model_id:
                    entry = value
                    break
        table[model_id] = entry if entry is not None else "<no price entry — calls cost $0>"
    return table


def collect(
    *,
    repo_root: Path,
    models_used: list[str] | None = None,
    environ: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build the env_and_config.json payload. Never raises."""

    env = dict(environ if environ is not None else os.environ)
    names = sorted(
        {n for n in env if n.startswith("PYTHIA_")} | {n for n in EXTRA_ENV_NAMES}
    )
    resolved: dict[str, Any] = {}
    for name in names:
        value = env.get(name)
        resolved[name] = "<unset>" if value is None else redact_env_value(name, value)

    # Names present in the process that carry a credential, listed so a
    # reader can see WHICH secrets the job held without seeing any of them.
    credential_names = sorted(
        n
        for n in env
        if n not in names
        and any(p in n.lower() for p in ("key", "token", "secret", "password", "credential"))
    )

    payload: dict[str, Any] = {
        "env": resolved,
        "credentials_present": {
            name: redact_env_value(name, env.get(name)) for name in credential_names
        },
        "git": _git(repo_root),
        "workflow_file_sha256": {
            rel: _file_sha256(repo_root / rel) for rel in WORKFLOW_FILES
        },
        "runner": {
            "os": platform.platform(),
            "runner_os": env.get("RUNNER_OS"),
            "runner_arch": env.get("RUNNER_ARCH"),
            "image_os": env.get("ImageOS"),
            "image_version": env.get("ImageVersion"),
            "cpu_count": os.cpu_count(),
        },
        "python": {
            "version": sys.version,
            "version_info": list(sys.version_info[:3]),
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "pip_freeze": _pip_freeze(),
        "llm_profile": _llm_profile(),
    }
    ensemble_models = [
        str(e.get("model_id"))
        for e in (payload["llm_profile"].get("ensemble") or [])
        if isinstance(e, dict)
    ]
    payload["price_table_applied"] = _price_table(
        list(models_used or []) + ensemble_models, repo_root
    )
    return payload
