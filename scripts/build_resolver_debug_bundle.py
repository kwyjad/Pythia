# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""One zip that explains a Resolver Update run to someone with nothing else.

The `backfill-diagnostics` artifact this replaces was close but left gaps
that forced guesswork. Reviewing the 28 August 2026 run, none of these could
be settled from the artifact alone: which URL each connector actually
called, what the ACLED CAST response envelope said, what the rulebook
thresholds were, why 1,215 of 2,223 hazard cells produced no row, or which
code and config the run was executing. All of it had to be inferred from log
prose.

Four rules shape what goes in:

**Evidence over prose.** A log line saying "252 cells were inconclusive" is
not a substitute for 252 rows saying which cells and why. Every claim a log
makes in words should exist somewhere as a row.

**Self-describing.** ``README.md`` is written for a reader who has never
seen the repo: what ran, what it produced, what is wrong, where to look
first.

**Uploadable.** A hard ceiling (default 80 MB compressed) so the zip can go
straight into a chat. When it binds, ``code/`` is dropped before any
evidence is, and the manifest says so — truncation is loud or it is a lie.

**Redacted by allowlist, then verified.** Secrets are redacted at capture
and again at assembly, and the whole bundle is scanned afterwards for every
secret value in the environment. A hit fails the build. A bundle meant for a
chat window cannot leak.

Everything degrades: no database, no CI variables, no phase logs, no run
streams — each missing input costs its own section and is recorded as a
problem, never an exception. A diagnostic that fails when the run failed is
a diagnostic that is never there when it is needed.

Usage::

    python -m scripts.build_resolver_debug_bundle --db duckdb:///data/resolver.duckdb \\
        --out diagnostics/resolver-debug-bundle.zip
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - import convenience
    sys.path.insert(0, str(REPO_ROOT))

from resolver.diagnostics import run_log  # noqa: E402
from resolver.diagnostics.redaction import (  # noqa: E402
    find_secrets,
    is_secret_name,
    redact_obj,
    redact_text,
    secret_values,
)

#: Compressed ceiling. 80 MB uploads to a chat; 200 MB does not.
DEFAULT_MAX_BYTES = 80 * 1024 * 1024

#: Environment variable prefixes worth recording. Values are redacted where
#: the NAME says credential; everything else is the run's actual settings,
#: which is the point.
ENV_PREFIXES = ("PYTHIA_", "RESOLVER_", "HS_", "FORECASTER_", "GDACS_", "FEWSNET_",
                "IPC_", "ACLED_", "IDMC_", "GDELT_", "EMDAT_", "RELIEFWEB_",
                "SIGNATURE_", "BACKFILL_", "DIAGNOSTICS_", "ONLY_", "LOG_LEVEL",
                "EMPTY_POLICY")

#: Config files copied verbatim (redacted). Thresholds should never have to
#: be inferred from behaviour.
CONFIG_FILES = (
    "resolver/hazard_resolution/rulebook.yaml",
    "pythia/config.yaml",
    "resolver/tools/precedence_config.yml",
)
CONFIG_DIRS = ("resolver/config", "resolver/ingestion/config")
WORKFLOW_FILES = (
    ".github/workflows/resolver_update.yml",
    ".github/workflows/haz_backcast.yml",
    ".github/workflows/ingest-structured-data.yml",
)

#: A source file over this size is a data blob, not code a reader will read.
MAX_CODE_FILE_BYTES = 200 * 1024

#: Directories a participating module may live in.
CODE_ROOTS = ("resolver", "horizon_scanner", "pythia")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def normalise_db_path(raw: str | None) -> Path | None:
    """``duckdb:///abs/path`` or a bare path -> Path. None when unusable."""

    if not raw:
        return None
    text = str(raw).strip()
    if not text:
        return None
    # Same strip the rest of the repo uses (summarize_connectors et al.):
    # the workflow writes duckdb:///<workspace>/... where <workspace> is
    # already absolute, so the fourth slash is the path's own.
    if text.startswith("duckdb:///"):
        text = text[len("duckdb:///") :]
    elif text.startswith("duckdb://"):
        text = text[len("duckdb://") :]
    return Path(text) if text else None


def write_csv(path: Path, rows: Sequence[Sequence[Any]], header: Sequence[str],
              preamble: str | None = None) -> None:
    """Write a CSV, optionally led by a ``#``-commented preamble (the SQL)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        if preamble:
            for line in preamble.strip().splitlines():
                handle.write(f"# {line}\n")
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow(["" if v is None else v for v in row])


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _run_git(args: list[str]) -> str:
    try:
        out = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=30
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def _normalise_log_line(line: str) -> str:
    """Collapse a log line to its shape so recurring faults count as one.

    ISO3 codes, dates, times and numbers become placeholders. Without this
    a reader has to reconstruct "the same failure, 252 times" by eye, which
    is exactly the work the bundle exists to remove.
    """

    text = line.strip()
    text = re.sub(r"\d{4}-\d{2}-\d{2}[T ]?\d{2}:\d{2}:\d{2}(?:[.,]\d+)?", "<TS>", text)
    text = re.sub(r"\d{4}-\d{2}-\d{2}", "<DATE>", text)
    text = re.sub(r"\d{4}-\d{2}\b", "<YM>", text)
    text = re.sub(r"\b[A-Z]{3}\b(?!-)", "<ISO3>", text)
    text = re.sub(r"0x[0-9a-fA-F]+", "<HEX>", text)
    text = re.sub(r"\b\d[\d,._]*\b", "<N>", text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# The builder
# ---------------------------------------------------------------------------


class BundleBuilder:
    """Assembles the bundle. Every section is best-effort and records problems."""

    def __init__(
        self,
        *,
        out_path: Path,
        db_path: Path | None,
        diagnostics_dir: Path,
        run_log_dir: Path | None,
        staging: Path,
        max_bytes: int = DEFAULT_MAX_BYTES,
        environ: dict[str, str] | None = None,
    ) -> None:
        self.out_path = out_path
        self.db_path = db_path
        self.diagnostics_dir = diagnostics_dir
        self.run_log_dir = run_log_dir
        self.staging = staging
        self.max_bytes = max_bytes
        self.env = dict(os.environ if environ is None else environ)
        self.secrets = secret_values(self.env)
        self.problems: list[str] = []
        self.notes: list[str] = []
        self.sections: dict[str, dict[str, Any]] = {}
        self.checks: list[dict[str, Any]] = []
        self._con = None
        self._db_reported = False
        self._tables: set[str] | None = None
        self._modules_seen: set[str] = set()

    # -- infrastructure -----------------------------------------------------

    def problem(self, text: str) -> None:
        self.problems.append(text)

    def section(self, name: str, fn: Callable[[], Any]) -> None:
        """Run one section. A failure costs that section, never the bundle."""

        try:
            result = fn()
            self.sections[name] = {"ok": True, "detail": result or {}}
        except Exception as exc:  # noqa: BLE001 - a bundle must always be produced
            self.sections[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            self.problem(f"section {name} failed: {type(exc).__name__}: {exc}")

    def path(self, *parts: str) -> Path:
        return self.staging.joinpath(*parts)

    # -- database -----------------------------------------------------------

    @property
    def con(self):
        """A DuckDB connection, or None when there is no readable database."""

        if self._con is not None:
            return self._con
        if self.db_path is None or not self.db_path.is_file():
            if not self._db_reported:
                self._db_reported = True
                self.problem(
                    "no readable database at "
                    f"{self.db_path or '(no path given)'} — table counts, "
                    "freshness and every db/queries file are absent"
                )
            return None
        try:
            # A crashed writer leaves a WAL a read-only open refuses to
            # replay; this bundle runs after continue-on-error steps, so
            # that is the normal case rather than the exceptional one.
            wal = self.db_path.with_suffix(self.db_path.suffix + ".wal")
            from resolver.db import duckdb_io

            self._con = duckdb_io.get_db(str(self.db_path))
            _ = wal  # kept for the reader: get_db handles replay itself
        except Exception as exc:  # noqa: BLE001
            self.problem(f"could not open the database at {self.db_path}: {exc}")
            self._con = None
        return self._con

    def tables(self) -> set[str]:
        if self._tables is not None:
            return self._tables
        con = self.con
        if con is None:
            self._tables = set()
            return self._tables
        try:
            self._tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
        except Exception as exc:  # noqa: BLE001
            self.problem(f"could not list tables: {exc}")
            self._tables = set()
        return self._tables

    def query(self, sql: str, params: Sequence[Any] | None = None):
        """Run a query. Returns (columns, rows) or None, never raises."""

        con = self.con
        if con is None:
            return None
        try:
            cur = con.execute(sql, list(params or []))
            columns = [d[0] for d in (cur.description or [])]
            return columns, cur.fetchall()
        except Exception as exc:  # noqa: BLE001
            self.problem(f"query failed ({sql.strip().splitlines()[0][:80]}...): {exc}")
            return None

    def columns_of(self, table: str) -> list[tuple[str, str]]:
        """(name, type) per column. ``PRAGMA table_info`` yields (cid, name, type, ...)
        — the NAME is column 1, and reading column 0 makes every lookup answer
        'absent' with a perfectly well-formed result."""

        if table not in self.tables():
            return []
        result = self.query(f'PRAGMA table_info("{table}")')
        if result is None:
            return []
        return [(str(row[1]), str(row[2])) for row in result[1]]

    # -- run/ ---------------------------------------------------------------

    def build_run(self) -> dict[str, Any]:
        env = self.env
        in_ci = bool(env.get("GITHUB_RUN_ID"))

        repo = env.get("GITHUB_REPOSITORY", "")
        run_id = env.get("GITHUB_RUN_ID", "")
        workflow_run: dict[str, Any] = {
            "in_ci": in_ci,
            "run_id": run_id or None,
            "run_attempt": env.get("GITHUB_RUN_ATTEMPT") or None,
            "run_number": env.get("GITHUB_RUN_NUMBER") or None,
            "workflow": env.get("GITHUB_WORKFLOW") or None,
            "repository": repo or None,
            "run_url": (
                f"https://github.com/{repo}/actions/runs/{run_id}" if repo and run_id else None
            ),
            "trigger": env.get("GITHUB_EVENT_NAME") or None,
            "actor": env.get("GITHUB_ACTOR") or None,
            "ref": env.get("GITHUB_REF") or None,
            "runner_os": env.get("RUNNER_OS") or platform.system(),
            "started_at": env.get("PYTHIA_RUN_STARTED_AT") or None,
            "bundled_at": _utcnow(),
        }
        if not in_ci:
            workflow_run["unavailable"] = (
                "not a GitHub Actions run: run id, trigger, actor and step "
                "timings are unknown outside CI"
            )
        write_json(self.path("run", "workflow_run.json"), workflow_run)

        # Effective environment + the windows each phase resolved.
        recorded = {
            name: value
            for name, value in sorted(env.items())
            if any(name.startswith(p) for p in ENV_PREFIXES)
        }
        effective = {
            "variables": {
                name: ("<redacted:by-name>" if is_secret_name(name) else value)
                for name, value in recorded.items()
            },
            "credentials_present": sorted(
                name for name in env if is_secret_name(name) and env.get(name)
            ),
            "credentials_absent": sorted(
                name
                for name in (
                    "ACLED_USERNAME", "ACLED_PASSWORD", "ACLED_REFRESH_TOKEN",
                    "IDMC_API_TOKEN", "IDMC_HELIX_CLIENT_ID", "IDMC_API_KEY",
                    "EMDAT_API_KEY", "IPC_API_KEY", "ACAPS_USERNAME",
                    "ACAPS_PASSWORD", "ANTHROPIC_API_KEY", "GEMINI_API_KEY",
                    "RELIEFWEB_APPNAME", "GO_API_TOKEN",
                )
                if not env.get(name)
            ),
            "effective_windows": self._effective_windows(recorded),
        }
        effective["variables"] = redact_obj(effective["variables"], self.secrets)
        write_json(self.path("run", "env_effective.json"), effective)

        write_json(
            self.path("run", "git.json"),
            {
                "commit": _run_git(["rev-parse", "HEAD"]) or env.get("GITHUB_SHA") or None,
                "branch": _run_git(["rev-parse", "--abbrev-ref", "HEAD"]) or None,
                "dirty": bool(_run_git(["status", "--porcelain"])),
                "recent_commits": [
                    line
                    for line in _run_git(
                        [
                            "log", "-20", "--date=short",
                            "--pretty=%h|%ad|%s", "--", "resolver", "horizon_scanner",
                        ]
                    ).splitlines()
                    if line
                ],
            },
        )

        freeze = ""
        try:
            freeze = subprocess.run(
                [sys.executable, "-m", "pip", "freeze"],
                capture_output=True, text=True, timeout=120,
            ).stdout
        except Exception as exc:  # noqa: BLE001
            freeze = f"(pip freeze unavailable: {exc})\n"
        duck = "unavailable"
        try:
            import duckdb

            duck = duckdb.__version__
        except Exception:
            pass
        write_text(
            self.path("run", "versions.txt"),
            f"python: {sys.version}\nplatform: {platform.platform()}\n"
            f"duckdb: {duck}\n\n--- pip freeze ---\n{freeze}",
        )

        self._step_timings()
        return {"in_ci": in_ci, "env_vars_recorded": len(recorded)}

    def _effective_windows(self, recorded: dict[str, str]) -> dict[str, Any]:
        """What each phase actually resolved, not what the input said.

        The August run printed some of these into logs and the artifact
        carried none of them; a reader could not tell a short FEWS NET
        window (partial country coverage, by design) from an outage.
        """

        windows: dict[str, Any] = {
            "gdacs_months": recorded.get("GDACS_MONTHS"),
            "fewsnet_months": recorded.get("FEWSNET_MONTHS"),
            "ipc_api_months": recorded.get("IPC_API_MONTHS"),
            "acled_max_runtime_sec": recorded.get("ACLED_MAX_RUNTIME_SEC"),
            "only_connector": recorded.get("ONLY_CONNECTOR"),
            "empty_policy": recorded.get("EMPTY_POLICY"),
            "log_level": recorded.get("LOG_LEVEL"),
        }
        country_list = REPO_ROOT / "horizon_scanner" / "data" / "hs_country_list.txt"
        if country_list.is_file():
            live = [
                line.strip()
                for line in country_list.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]
            windows["hs_country_list"] = {"source": str(country_list.relative_to(REPO_ROOT)),
                                          "n_countries": len(live)}
        countries_csv = REPO_ROOT / "resolver" / "data" / "countries.csv"
        if countries_csv.is_file():
            try:
                with open(countries_csv, encoding="utf-8-sig", newline="") as handle:
                    windows["resolver_country_registry"] = {
                        "source": "resolver/data/countries.csv",
                        "n_countries": sum(1 for _ in csv.DictReader(handle)),
                    }
            except Exception:
                pass
        # The machine's own trailing window, read from the rulebook rather
        # than assumed: freeze_days decides what is still revisable.
        windows["hazard_machine"] = self._rulebook_windows()
        return windows

    def _rulebook_windows(self) -> dict[str, Any]:
        try:
            from resolver.hazard_resolution.rulebook import load_rulebook

            rb = load_rulebook()
            return {
                "freeze_days": rb.get("freeze_days"),
                "ladder": rb.get("ladder"),
                "sanity.ceiling_multiplier": rb.get("sanity.ceiling_multiplier"),
                "sanity.population_fallback_share": rb.get("sanity.population_fallback_share"),
                "extraction.max_calls_per_month": rb.get("extraction.max_calls_per_month"),
                "extraction.backcast_max_calls_per_month": rb.get(
                    "extraction.backcast_max_calls_per_month"
                ),
                "extraction.live_reserve_calls": rb.get("extraction.live_reserve_calls"),
                "cyclone.buffer_km": rb.get("cyclone.buffer_km"),
                "cyclone.min_wind_kt": rb.get("cyclone.min_wind_kt"),
                "flood.gdacs_trigger_level": rb.get("flood.gdacs_trigger_level"),
            }
        except Exception as exc:  # noqa: BLE001
            return {"unavailable": f"{type(exc).__name__}: {exc}"}

    def _step_timings(self) -> None:
        """Per-step name, status and duration — from CI when it is there."""

        rows: list[list[Any]] = []
        source = "unavailable"
        raw = self.env.get("PYTHIA_STEP_TIMINGS_JSON", "")
        if raw and Path(raw).is_file():
            try:
                payload = json.loads(Path(raw).read_text(encoding="utf-8"))
                source = raw
                for step in payload.get("steps", payload if isinstance(payload, list) else []):
                    rows.append([
                        step.get("name"), step.get("status"), step.get("conclusion"),
                        step.get("started_at"), step.get("completed_at"),
                        step.get("duration_sec"),
                    ])
            except Exception as exc:  # noqa: BLE001
                self.problem(f"step timings file unreadable: {exc}")
        write_csv(
            self.path("run", "step_timings.csv"), rows,
            ["step", "status", "conclusion", "started_at", "completed_at", "duration_sec"],
            preamble=(
                f"source: {source}\n"
                "Empty outside CI, and empty in CI unless the workflow writes a\n"
                "step-timing JSON and points PYTHIA_STEP_TIMINGS_JSON at it. The\n"
                "workflow's own step list is in config/workflows/resolver_update.yml."
            ),
        )

    # -- logs/ --------------------------------------------------------------

    def build_logs(self) -> dict[str, Any]:
        dest = self.path("logs")
        dest.mkdir(parents=True, exist_ok=True)
        copied: list[tuple[str, int]] = []
        if not self.diagnostics_dir.is_dir():
            write_text(
                dest / "log_index.md",
                "# Logs\n\nNo diagnostics directory was found at "
                f"`{self.diagnostics_dir}` — the run's phase logs are absent.\n",
            )
            self.problem(f"no diagnostics directory at {self.diagnostics_dir}")
            return {"logs": 0}

        for src in sorted(self.diagnostics_dir.rglob("*.log")):
            rel = src.relative_to(self.diagnostics_dir).as_posix().replace("/", "__")
            try:
                text = src.read_text(encoding="utf-8", errors="replace")
            except Exception as exc:  # noqa: BLE001
                self.problem(f"could not read log {src}: {exc}")
                continue
            write_text(dest / rel, redact_text(text, self.secrets))
            copied.append((rel, len(text.splitlines())))

        self._log_index(dest, copied)
        return {"logs": len(copied)}

    def _log_index(self, dest: Path, copied: list[tuple[str, int]]) -> None:
        lines = [
            "# Log index",
            "",
            "Per log: line counts by level, then the ERROR and WARNING lines with",
            "ISO3 codes, dates and numbers replaced by placeholders — so one fault",
            "repeated 252 times is one row with a count of 252, which is the shape",
            "a reader needs and the shape a raw log never has.",
            "",
        ]
        if not copied:
            lines.append("_No phase logs were found._")
            write_text(dest / "log_index.md", "\n".join(lines) + "\n")
            return

        for name, n_lines in sorted(copied):
            text = (dest / name).read_text(encoding="utf-8", errors="replace")
            levels: Counter[str] = Counter()
            shapes: Counter[str] = Counter()
            for line in text.splitlines():
                upper = line.upper()
                for level in ("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"):
                    if level in upper:
                        levels[level] += 1
                        if level in ("CRITICAL", "ERROR", "WARNING"):
                            shapes[f"{level}: {_normalise_log_line(line)[:200]}"] += 1
                        break
            lines.append(f"## {name}")
            lines.append("")
            lines.append(f"- lines: {n_lines}")
            if levels:
                lines.append(
                    "- levels: "
                    + ", ".join(f"{k}={v}" for k, v in sorted(levels.items()))
                )
            if shapes:
                lines.append("")
                lines.append("| count | normalised ERROR/WARNING line |")
                lines.append("| ---: | --- |")
                for shape, count in shapes.most_common(25):
                    lines.append(f"| {count} | {shape.replace('|', '/')} |")
            lines.append("")
        write_text(dest / "log_index.md", "\n".join(lines) + "\n")

    # -- http/ --------------------------------------------------------------

    def build_http(self) -> dict[str, Any]:
        dest = self.path("http")
        dest.mkdir(parents=True, exist_ok=True)
        requests_path = self._stream_file(run_log.STREAM_HTTP)
        records = list(run_log.read_stream(requests_path)) if requests_path else []

        if not records:
            write_text(
                dest / "README.md",
                "# HTTP\n\nNo outbound requests were recorded.\n\n"
                "The recorder is opt-in: it writes only when `PYTHIA_RUN_LOG_DIR`\n"
                "is set before the connectors run (see\n"
                "`resolver/diagnostics/http_recorder.py`). An empty section means\n"
                "the variable was unset, not that nothing was fetched.\n",
            )
            self.problem(
                "no HTTP requests recorded — PYTHIA_RUN_LOG_DIR was not set when "
                "the connectors ran, so URLs and response envelopes are unavailable"
            )
            return {"requests": 0}

        with open(dest / "requests.jsonl", "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(redact_obj(record, self.secrets), default=str) + "\n")

        # A per-connector roll-up, because 40,000 lines of JSONL is evidence
        # and not yet an answer.
        by_connector: dict[str, Counter[str]] = defaultdict(Counter)
        elapsed: dict[str, list[float]] = defaultdict(list)
        for record in records:
            key = str(record.get("connector") or "unknown")
            by_connector[key]["requests"] += 1
            status = record.get("status")
            bucket = "error" if status is None else f"{int(status) // 100}xx"
            by_connector[key][bucket] += 1
            by_connector[key]["bytes"] += int(record.get("response_bytes") or 0)
            try:
                elapsed[key].append(float(record.get("elapsed_ms") or 0.0))
            except Exception:
                pass
        rows = []
        for key in sorted(by_connector):
            counts = by_connector[key]
            times = sorted(elapsed[key]) or [0.0]
            rows.append([
                key, counts["requests"], counts["2xx"], counts["3xx"], counts["4xx"],
                counts["5xx"], counts["error"], counts["bytes"],
                round(times[len(times) // 2], 1), round(times[-1], 1),
            ])
        write_csv(
            dest / "requests_by_connector.csv", rows,
            ["connector", "requests", "n_2xx", "n_3xx", "n_4xx", "n_5xx",
             "n_transport_error", "response_bytes", "median_ms", "max_ms"],
            preamble="Derived from http/requests.jsonl.",
        )

        envelopes_path = self._stream_file(run_log.STREAM_ENVELOPE)
        envelopes = list(run_log.read_stream(envelopes_path)) if envelopes_path else []
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in envelopes:
            grouped[str(record.get("connector") or "unknown")].append(record)
        env_dir = dest / "envelopes"
        env_dir.mkdir(parents=True, exist_ok=True)
        for connector, entries in grouped.items():
            safe = re.sub(r"[^A-Za-z0-9._-]+", "_", connector)
            write_json(env_dir / f"{safe}.json", redact_obj(entries, self.secrets))
        return {"requests": len(records), "envelopes": len(envelopes),
                "connectors": len(by_connector)}

    def _stream_file(self, stream: str) -> Path | None:
        if self.run_log_dir is None:
            return None
        candidate = self.run_log_dir / f"{stream}.jsonl"
        return candidate if candidate.is_file() else None

    # -- db/ ----------------------------------------------------------------

    def build_db(self) -> dict[str, Any]:
        dest = self.path("db")
        dest.mkdir(parents=True, exist_ok=True)

        before = self._copy_signature("db_signature_before.json", dest / "signature_before.json")
        after = self._copy_signature("db_signature_after.json", dest / "signature_after.json")

        if self.con is None:
            write_text(
                dest / "README.md",
                "# Database\n\nNo readable database. Row counts, freshness and every\n"
                "`queries/` file are absent; the signature JSONs above (if present)\n"
                "are all that is known.\n",
            )
            return {"tables": 0}

        self._table_counts(dest, before)
        self._freshness(dest)
        self._queries(dest / "queries")
        return {"tables": len(self.tables())}

    def _copy_signature(self, name: str, dest: Path) -> dict[str, Any]:
        src = self.diagnostics_dir / name
        if not src.is_file():
            return {}
        try:
            payload = json.loads(src.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            self.problem(f"{name} unreadable: {exc}")
            return {}
        write_json(dest, payload)
        return payload if isinstance(payload, dict) else {}

    def _table_counts(self, dest: Path, before: dict[str, Any]) -> None:
        """Every table, before and after — not only the signature's subset.

        A gap in a table nobody thought to name in advance is exactly where
        an unexplained ingest failure sits.
        """

        baseline: dict[str, Any] = {}
        baseline.update(before.get("required_counts") or {})
        baseline.update(before.get("optional_counts") or {})
        baseline.update(before.get("all_counts") or {})

        rows = []
        for table in sorted(self.tables()):
            result = self.query(f'SELECT COUNT(*) FROM "{table}"')
            after = result[1][0][0] if result and result[1] else None
            was = baseline.get(table)
            delta = (after - was) if (after is not None and was is not None) else None
            rows.append([table, was, after, delta])
        write_csv(
            dest / "table_counts.csv", rows,
            ["table", "rows_before", "rows_after", "delta"],
            preamble=(
                "rows_before comes from diagnostics/db_signature_before.json.\n"
                "It is blank for any table that signature did not count — run\n"
                "`db_signature write --all-tables` to widen the baseline."
            ),
        )

    _DATE_HINTS = ("date", "time", "updated", "issued", "stamp", "_at", "period")

    #: Staleness thresholds in days, per table. A table with no entry still
    #: gets its age reported; it simply has no verdict to fail.
    _STALENESS_DAYS = {
        "conflict_forecasts": 45,
        "facts_resolved": 60,
        "facts_deltas": 60,
        "acled_monthly_fatalities": 60,
        "crisiswatch_entries": 70,
        "enso_state": 120,
        "hdx_signals": 180,
        "gdelt_conflict_indicators": 21,
        "reliefweb_reports": 30,
        "acaps_inform_severity": 60,
        "acaps_risk_radar": 60,
        "acaps_daily_monitoring": 21,
        "seasonal_forecasts": 60,
    }

    def _freshness(self, dest: Path) -> None:
        rows = []
        for table in sorted(self.tables()):
            columns = self.columns_of(table)
            date_columns = [
                name
                for name, typ in columns
                if typ.upper().startswith(("DATE", "TIMESTAMP"))
                or any(hint in name.lower() for hint in self._DATE_HINTS)
            ]
            if not date_columns:
                continue
            threshold = self._STALENESS_DAYS.get(table)
            for column in date_columns:
                result = self.query(
                    f'SELECT MAX(TRY_CAST("{column}" AS TIMESTAMP)), COUNT(*) FROM "{table}"'
                )
                if result is None or not result[1]:
                    continue
                newest, n_rows = result[1][0]
                age = None
                if newest is not None:
                    age_result = self.query(
                        "SELECT date_diff('day', CAST(? AS TIMESTAMP), CURRENT_TIMESTAMP)",
                        [newest],
                    )
                    if age_result and age_result[1]:
                        age = age_result[1][0][0]
                if n_rows == 0:
                    verdict = "absent"
                elif newest is None:
                    verdict = "no_dates"
                elif threshold is None or age is None:
                    verdict = "no_threshold"
                else:
                    verdict = "stale" if age > threshold else "fresh"
                rows.append([table, column, newest, age, threshold, n_rows, verdict])
        write_csv(
            dest / "freshness.csv", rows,
            ["table", "date_column", "max_value", "age_days",
             "staleness_threshold_days", "rows", "verdict"],
            preamble=(
                "Every date-typed or date-named column in every table, with its\n"
                "maximum and its age. `no_threshold` means nothing is configured\n"
                "for that table, not that the value is fine."
            ),
        )

    #: (filename, required tables, SQL). A query whose tables are absent is
    #: skipped with a note rather than raising — pre-machine and pre-restart
    #: databases must still produce a bundle.
    DB_QUERIES: tuple[tuple[str, tuple[str, ...], str], ...] = (
        (
            "haz_resolutions_by_hazard_month_status",
            ("haz_resolutions",),
            """
            SELECT hazard, year, month, status, run_type,
                   COUNT(*) AS n,
                   SUM(CASE WHEN flagged THEN 1 ELSE 0 END) AS n_flagged,
                   SUM(CASE WHEN provisional THEN 1 ELSE 0 END) AS n_provisional,
                   SUM(CASE WHEN value IS NOT NULL THEN 1 ELSE 0 END) AS n_with_value
            FROM haz_resolutions
            GROUP BY hazard, year, month, status, run_type
            ORDER BY hazard, year, month, status
            """,
        ),
        (
            "haz_triggers_by_hazard_month",
            ("haz_triggers",),
            """
            SELECT hazard, year, month, triggered, trigger_source, run_type,
                   COUNT(*) AS n
            FROM haz_triggers
            GROUP BY hazard, year, month, triggered, trigger_source, run_type
            ORDER BY hazard, year, month, triggered
            """,
        ),
        (
            "haz_impact_candidates_by_source",
            ("haz_impact_candidates",),
            """
            SELECT hazard, year, month, source, value_type, COUNT(*) AS n,
                   MIN(value) AS min_value, MAX(value) AS max_value
            FROM haz_impact_candidates
            GROUP BY hazard, year, month, source, value_type
            ORDER BY hazard, year, month, source
            """,
        ),
        (
            "haz_doc_extractions_by_status",
            ("haz_doc_extractions",),
            """
            SELECT hazard, year, month, model, prompt_version, status, run_type,
                   COUNT(*) AS n_calls,
                   SUM(COALESCE(n_figures, 0)) AS figures,
                   SUM(COALESCE(n_rejected, 0)) AS rejected,
                   SUM(COALESCE(cost_usd, 0)) AS cost_usd
            FROM haz_doc_extractions
            GROUP BY hazard, year, month, model, prompt_version, status, run_type
            ORDER BY hazard, year, month
            """,
        ),
        (
            "haz_base_rates_occurrence_coverage",
            ("haz_base_rates_occurrence",),
            """
            SELECT hazard, COUNT(*) AS rows,
                   COUNT(DISTINCT iso3) AS countries,
                   COUNT(DISTINCT calendar_month) AS months
            FROM haz_base_rates_occurrence
            GROUP BY hazard ORDER BY hazard
            """,
        ),
        (
            "haz_base_rates_severity_coverage",
            ("haz_base_rates_severity",),
            "SELECT * FROM haz_base_rates_severity ORDER BY hazard",
        ),
        (
            "conflict_forecast_vintages",
            ("conflict_forecasts",),
            """
            SELECT source, metric, COUNT(*) AS n,
                   COUNT(DISTINCT iso3) AS countries,
                   MAX(forecast_issue_date) AS latest_issue_date,
                   MIN(forecast_issue_date) AS earliest_issue_date
            FROM conflict_forecasts
            GROUP BY source, metric ORDER BY source, metric
            """,
        ),
        ("enso_state_full", ("enso_state",), "SELECT * FROM enso_state ORDER BY fetch_date DESC"),
        (
            "seasonal_tc_outlooks_full",
            ("seasonal_tc_outlooks",),
            """
            SELECT basin, source, forecast_season, named_storms_forecast, category,
                   fetched_at,
                   TRY_CAST(json_extract_string(raw_json, '$.issue_date') AS VARCHAR)
                       AS issue_date
            FROM seasonal_tc_outlooks
            ORDER BY basin, source, fetched_at DESC
            """,
        ),
        (
            "facts_resolved_by_source_metric",
            ("facts_resolved",),
            """
            SELECT publisher, hazard_code, metric, series_semantics,
                   COUNT(*) AS n, COUNT(DISTINCT iso3) AS countries,
                   MIN(ym) AS first_ym, MAX(ym) AS last_ym,
                   MAX(as_of_date) AS max_as_of
            FROM facts_resolved
            GROUP BY publisher, hazard_code, metric, series_semantics
            ORDER BY publisher, hazard_code, metric
            """,
        ),
        (
            "facts_deltas_by_source_metric",
            ("facts_deltas",),
            """
            SELECT hazard_code, metric, series_semantics, COUNT(*) AS n,
                   COUNT(DISTINCT iso3) AS countries,
                   MIN(ym) AS first_ym, MAX(ym) AS last_ym
            FROM facts_deltas
            GROUP BY hazard_code, metric, series_semantics
            ORDER BY hazard_code, metric
            """,
        ),
        (
            "crisiswatch_editions",
            ("crisiswatch_entries",),
            """
            SELECT year, month, COUNT(*) AS entries,
                   COUNT(DISTINCT iso3) AS countries,
                   SUM(CASE WHEN arrow = 'deteriorated' THEN 1 ELSE 0 END) AS deteriorated,
                   MAX(fetched_at) AS fetched_at
            FROM crisiswatch_entries
            GROUP BY year, month ORDER BY year DESC, month DESC
            """,
        ),
        (
            "llm_calls_recent_by_phase",
            ("llm_calls",),
            """
            SELECT CAST(timestamp AS DATE) AS day, phase, call_type, model_id,
                   COUNT(*) AS calls,
                   SUM(COALESCE(cost_usd, 0)) AS cost_usd,
                   SUM(COALESCE(total_tokens, 0)) AS tokens,
                   SUM(CASE WHEN error_text IS NOT NULL THEN 1 ELSE 0 END) AS errors
            FROM llm_calls
            WHERE timestamp >= CURRENT_TIMESTAMP - INTERVAL 14 DAY
            GROUP BY day, phase, call_type, model_id
            ORDER BY day DESC, cost_usd DESC
            """,
        ),
        (
            "acled_monthly_fatalities_coverage",
            ("acled_monthly_fatalities",),
            """
            SELECT MIN(ym) AS first_ym, MAX(ym) AS last_ym,
                   COUNT(*) AS rows, COUNT(DISTINCT iso3) AS countries
            FROM acled_monthly_fatalities
            """,
        ),
    )

    def _queries(self, dest: Path) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        skipped: list[str] = []
        for name, needs, sql in self.DB_QUERIES:
            missing = [t for t in needs if t not in self.tables()]
            if missing:
                skipped.append(f"{name}: missing {', '.join(missing)}")
                continue
            result = self.query(sql)
            if result is None:
                skipped.append(f"{name}: query failed (see problems)")
                continue
            columns, rows = result
            write_csv(dest / f"{name}.csv", rows, columns, preamble=f"SQL:\n{sql.strip()}")
        if skipped:
            write_text(
                dest / "_skipped.md",
                "# Queries not run\n\n"
                "Each line names a query and why it produced no file. A missing\n"
                "table is a fact about this database, not a failure of the bundle.\n\n"
                + "\n".join(f"- {line}" for line in skipped)
                + "\n",
            )

    # -- hazard/ ------------------------------------------------------------

    def build_hazard(self) -> dict[str, Any]:
        dest = self.path("hazard")
        dest.mkdir(parents=True, exist_ok=True)

        for name in sorted(self.diagnostics_dir.glob("haz_run_*.json")):
            try:
                payload = json.loads(name.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                self.problem(f"{name.name} unreadable: {exc}")
                continue
            write_json(dest / "run_summaries" / name.name, redact_obj(payload, self.secrets))

        report = self.diagnostics_dir / "haz_acceptance_report.md"
        if report.is_file():
            write_text(
                dest / "acceptance_report.md",
                redact_text(report.read_text(encoding="utf-8", errors="replace"), self.secrets),
            )

        cells = self._cell_ledger(dest)
        figures = self._figures_ledger(dest)
        self._extraction_budget(dest)
        self._backcast_progress(dest)
        return {"cells": cells, "figures": figures}

    def _cell_ledger(self, dest: Path) -> int:
        """One row per assessed cell, DB-first with the run stream layered on.

        The database says what every cell was decided; only the stream says
        why a cell got no row at all. Building from both means a bundle made
        from a database alone is still most of the picture.
        """

        merged: dict[tuple[str, str, str], dict[str, Any]] = {}

        result = self.query(
            """
            SELECT t.iso3, t.hazard,
                   printf('%04d-%02d', t.year, t.month) AS ym,
                   t.triggered, t.trigger_source, t.run_type,
                   t.evidence_of_absence_json IS NOT NULL AS has_absence_evidence,
                   r.status, r.value, r.rule_fired, r.flagged, r.provisional,
                   r.frozen_at IS NOT NULL AS frozen,
                   c.n_candidates, c.sources
            FROM haz_triggers t
            LEFT JOIN haz_resolutions r
              ON r.iso3 = t.iso3 AND r.year = t.year
             AND r.month = t.month AND r.hazard = t.hazard
            LEFT JOIN (
                SELECT iso3, year, month, hazard, COUNT(*) AS n_candidates,
                       string_agg(DISTINCT source, '|') AS sources
                FROM haz_impact_candidates
                GROUP BY iso3, year, month, hazard
            ) c ON c.iso3 = t.iso3 AND c.year = t.year
               AND c.month = t.month AND c.hazard = t.hazard
            ORDER BY t.hazard, t.year, t.month, t.iso3
            """
        ) if {"haz_triggers", "haz_resolutions", "haz_impact_candidates"} <= self.tables() else None

        if result is not None:
            for row in result[1]:
                key = (str(row[0]), str(row[1]), str(row[2]))
                merged[key] = {
                    "iso3": row[0], "hazard": row[1], "ym": row[2],
                    "triggered": row[3], "trigger_source": row[4], "run_type": row[5],
                    "has_absence_evidence": row[6], "status": row[7], "value": row[8],
                    "rule_fired": row[9], "flagged": row[10], "provisional": row[11],
                    "frozen": row[12], "n_candidates": row[13] or 0,
                    "candidate_sources": row[14] or "",
                    "stage": "", "write_outcome": "", "reason_code": "",
                    "answering_rung": "", "rungs_readable": "", "rungs_unavailable": "",
                    "extraction": "", "detail": "",
                }

        stream = self._stream_file(run_log.STREAM_CELLS)
        n_stream = 0
        for record in run_log.read_stream(stream) if stream else []:
            n_stream += 1
            key = (str(record.get("iso3")), str(record.get("hazard")), str(record.get("ym")))
            entry = merged.setdefault(
                key,
                {
                    "iso3": key[0], "hazard": key[1], "ym": key[2],
                    "triggered": record.get("triggered"),
                    "trigger_source": record.get("trigger_source"),
                    "run_type": record.get("run_type"),
                    "has_absence_evidence": None, "status": None, "value": None,
                    "rule_fired": None, "flagged": None, "provisional": None,
                    "frozen": None, "n_candidates": 0, "candidate_sources": "",
                    "stage": "", "write_outcome": "", "reason_code": "",
                    "answering_rung": "", "rungs_readable": "", "rungs_unavailable": "",
                    "extraction": "", "detail": "",
                },
            )
            entry["stage"] = record.get("stage") or entry["stage"]
            entry["write_outcome"] = record.get("write_outcome") or entry["write_outcome"]
            entry["reason_code"] = record.get("reason_code") or entry["reason_code"]
            entry["answering_rung"] = record.get("answering_rung") or entry["answering_rung"]
            entry["rungs_readable"] = "|".join(record.get("rungs_readable") or []) or entry["rungs_readable"]
            entry["rungs_unavailable"] = "|".join(record.get("rungs_unavailable") or []) or entry["rungs_unavailable"]
            if entry["status"] is None:
                entry["status"] = record.get("status")
                entry["value"] = record.get("value")
                entry["rule_fired"] = record.get("rule_fired")
            for field in ("extraction", "detail"):
                payload = record.get(field)
                if payload:
                    entry[field] = json.dumps(redact_obj(payload, self.secrets), default=str)[:2000]

        # A cell assessed with no row and no recorded reason is itself a
        # finding: it means some path writes nothing and says nothing.
        for entry in merged.values():
            if entry["status"] is None and not entry["reason_code"]:
                entry["reason_code"] = "unexplained_no_row"

        header = [
            "iso3", "hazard", "ym", "stage", "triggered", "trigger_source",
            "has_absence_evidence", "status", "value", "rule_fired", "flagged",
            "provisional", "frozen", "write_outcome", "reason_code",
            "answering_rung", "rungs_readable", "rungs_unavailable",
            "n_candidates", "candidate_sources", "extraction", "detail", "run_type",
        ]
        rows = [[entry.get(col) for col in header] for _, entry in sorted(merged.items())]
        write_csv(
            dest / "cell_ledger.csv", rows, header,
            preamble=(
                "One row per country-month-hazard cell the machine ASSESSED.\n"
                "status blank = no row in haz_resolutions; reason_code says why.\n"
                "reason_code=unexplained_no_row means neither the database nor the\n"
                "run stream accounted for the cell — that is a bug, not a silence.\n"
                f"Rows from the database: {len(result[1]) if result else 0}; "
                f"records from the run stream: {n_stream}."
            ),
        )
        if not merged:
            self.problem("cell ledger is empty: no haz_triggers rows and no cell stream")
        return len(rows)

    def _figures_ledger(self, dest: Path) -> int:
        stream = self._stream_file(run_log.STREAM_FIGURES)
        records = list(run_log.read_stream(stream)) if stream else []
        header = [
            "iso3", "hazard", "ym", "outcome", "doc_id", "value", "unit",
            "stated_value", "stated_unit", "value_persons", "conversion_factor",
            "figure_date", "doc_date", "doc_date_original", "doc_primary_country",
            "stated_by", "reason", "ceiling", "ceiling_multiplier",
            "ceiling_source", "ceiling_source_ref", "ceiling_field",
            "preference_rank", "quote", "doc_url",
        ]
        rows = [
            [redact_text(str(r.get(col) or ""), self.secrets) if col in ("quote", "doc_url")
             else r.get(col) for col in header]
            for r in records
        ]
        write_csv(
            dest / "figures_ledger.csv", rows, header,
            preamble=(
                "Every LLM-extracted figure and what became of it.\n"
                "stated_value/stated_unit are what the SOURCE said; value_persons is\n"
                "the whole-person count the ladder uses and conversion_factor joins\n"
                "them. value keeps the legacy meaning (people, as used).\n"
                "ceiling_field names the upstream field the ceiling came from: a\n"
                "ceiling of 2 against a reported 40,000 is a GDACS enrichment\n"
                "failure, not a mis-transcription, and only that column says which.\n"
                "Empty unless PYTHIA_RUN_LOG_DIR was set while the machine ran."
            ),
        )

        # The rejected figures are recoverable from the DB even without the
        # stream: haz_doc_extractions stores everything the model reported,
        # including what post-processing later discarded.
        if "haz_doc_extractions" in self.tables():
            result = self.query(
                """
                SELECT iso3, hazard, printf('%04d-%02d', year, month) AS ym,
                       doc_id, model, status, n_figures, n_rejected, cost_usd,
                       error, doc_url, created_at
                FROM haz_doc_extractions
                ORDER BY created_at DESC
                LIMIT 20000
                """
            )
            if result is not None:
                write_csv(
                    dest / "extraction_calls.csv", result[1], result[0],
                    preamble=(
                        "SQL: SELECT ... FROM haz_doc_extractions (newest 20,000).\n"
                        "The per-document cache AND cost ledger. n_rejected counts\n"
                        "figures the deterministic post-processing discarded."
                    ),
                )
        return len(rows)

    def _extraction_budget(self, dest: Path) -> None:
        if "haz_doc_extractions" not in self.tables():
            write_text(
                dest / "extraction_budget.csv",
                "# haz_doc_extractions is absent — the machine has never extracted\n"
                "caller,day,calls,billed_calls,cost_usd\n",
            )
            return
        sql = """
            SELECT COALESCE(run_type, 'backcast') AS caller,
                   CAST(created_at AS DATE) AS day,
                   COUNT(*) AS calls,
                   SUM(CASE WHEN COALESCE(prompt_tokens, 0)
                                 + COALESCE(completion_tokens, 0) > 0
                            THEN 1 ELSE 0 END) AS billed_calls,
                   SUM(COALESCE(cost_usd, 0)) AS cost_usd
            FROM haz_doc_extractions
            GROUP BY caller, day
            ORDER BY day DESC, caller
        """
        result = self.query(sql)
        caps = self._rulebook_windows()
        preamble = (
            f"SQL:\n{sql.strip()}\n\n"
            "caller is haz_doc_extractions.run_type; a NULL run_type counts as\n"
            "backcast, because every pre-split row was backcast-created.\n"
            "Only calls that BILLED tokens consume the cap — a call that never\n"
            "reached the provider spent nothing and must cost nothing.\n"
            f"Caps this run: total={caps.get('extraction.max_calls_per_month')}, "
            f"backcast share={caps.get('extraction.backcast_max_calls_per_month')}, "
            f"live reserve={caps.get('extraction.live_reserve_calls')}."
        )
        if result is None:
            write_csv(dest / "extraction_budget.csv", [],
                      ["caller", "day", "calls", "billed_calls", "cost_usd"],
                      preamble=preamble)
            return
        write_csv(dest / "extraction_budget.csv", result[1], result[0], preamble=preamble)

    def _backcast_progress(self, dest: Path) -> None:
        if "haz_backcast_progress" not in self.tables():
            return
        result = self.query("SELECT * FROM haz_backcast_progress ORDER BY hazard, ym")
        if result is None:
            return
        write_csv(
            dest / "backcast_progress.csv", result[1], result[0],
            preamble=(
                "SQL: SELECT * FROM haz_backcast_progress.\n"
                "status='ok' means the resume ledger will SKIP the month. Whether a\n"
                "failed month is marked complete decides whether fixing a feed\n"
                "recovers its history or leaves a permanent hole."
            ),
        )

    # -- config/ ------------------------------------------------------------

    def build_config(self) -> dict[str, Any]:
        dest = self.path("config")
        copied = 0
        for rel in CONFIG_FILES:
            copied += int(self._copy_config(REPO_ROOT / rel, dest / Path(rel).name))
        for rel in CONFIG_DIRS:
            source = REPO_ROOT / rel
            if not source.is_dir():
                continue
            for src in sorted(source.rglob("*")):
                if src.is_file() and src.suffix in {".yml", ".yaml", ".json", ".toml"}:
                    copied += int(
                        self._copy_config(src, dest / "connectors" / src.relative_to(source))
                    )
        for rel in WORKFLOW_FILES:
            copied += int(
                self._copy_config(REPO_ROOT / rel, dest / "workflows" / Path(rel).name)
            )
        return {"files": copied}

    def _copy_config(self, src: Path, dest: Path) -> bool:
        if not src.is_file():
            self.notes.append(f"config file absent: {src.relative_to(REPO_ROOT) if src.is_relative_to(REPO_ROOT) else src}")
            return False
        try:
            text = src.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001
            self.problem(f"could not read config {src}: {exc}")
            return False
        write_text(dest, redact_text(text, self.secrets))
        return True

    # -- code/ --------------------------------------------------------------

    def build_code(self) -> dict[str, Any]:
        """The source files that participated, verbatim.

        Selected from the module names the logs mention, then widened by one
        hop of their own repo-internal imports. A reader with the zip can
        then read the code the run was executing rather than the code on
        main today.
        """

        dest = self.path("code")
        modules = self._modules_from_logs()
        files = {m: p for m, p in ((m, self._module_path(m)) for m in modules) if p}
        for module in list(files):
            for extra in self._direct_imports(files[module]):
                if extra not in files:
                    path = self._module_path(extra)
                    if path:
                        files[extra] = path

        rows = []
        for module, src in sorted(files.items()):
            try:
                size = src.stat().st_size
            except Exception:
                continue
            if size > MAX_CODE_FILE_BYTES:
                rows.append([src.relative_to(REPO_ROOT).as_posix(), "", "", "",
                             f"skipped: {size} bytes > {MAX_CODE_FILE_BYTES}"])
                continue
            try:
                text = src.read_text(encoding="utf-8", errors="replace")
            except Exception as exc:  # noqa: BLE001
                rows.append([src.relative_to(REPO_ROOT).as_posix(), "", "", "",
                             f"unreadable: {exc}"])
                continue
            rel = src.relative_to(REPO_ROOT)
            write_text(dest / rel, redact_text(text, self.secrets))
            import hashlib

            rows.append([
                rel.as_posix(),
                hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:16],
                len(text.splitlines()),
                _run_git(["log", "-1", "--date=short", "--pretty=%ad", "--", str(rel)]),
                "",
            ])
        write_csv(
            dest / "file_index.csv", rows,
            ["path", "sha256_16", "lines", "last_commit_date", "note"],
            preamble=(
                "Files selected from the module names the run's logs mention, plus\n"
                "one hop of their repo-internal imports. Tests, web/ and anything\n"
                f"over {MAX_CODE_FILE_BYTES} bytes are excluded."
            ),
        )
        return {"files": len([r for r in rows if not r[4]])}

    _MODULE_RE = re.compile(r"\b((?:resolver|horizon_scanner|pythia)(?:\.[a-z_][a-z0-9_]*)+)\b")

    def _modules_from_logs(self) -> set[str]:
        modules: set[str] = set()
        logs = self.path("logs")
        if logs.is_dir():
            for log in logs.glob("*.log"):
                try:
                    text = log.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
                modules.update(self._MODULE_RE.findall(text))
        # A run whose logs named nothing still needs the machine in the zip:
        # "which code produced this run" must have an answer either way.
        modules.update(
            {
                "resolver.hazard_resolution.cli",
                "resolver.hazard_resolution.impact",
                "resolver.hazard_resolution.reconcile",
                "resolver.hazard_resolution.figures",
                "resolver.hazard_resolution.extract",
                "resolver.hazard_resolution.drought",
                "resolver.hazard_resolution.detect",
                "resolver.tools.run_pipeline",
            }
        )
        return {m for m in modules if m.split(".")[0] in CODE_ROOTS}

    def _module_path(self, module: str) -> Path | None:
        parts = module.split(".")
        if "tests" in parts or parts[0] not in CODE_ROOTS:
            return None
        candidate = REPO_ROOT.joinpath(*parts).with_suffix(".py")
        if candidate.is_file():
            return candidate
        package = REPO_ROOT.joinpath(*parts, "__init__.py")
        return package if package.is_file() else None

    _IMPORT_RE = re.compile(
        r"^\s*(?:from\s+((?:resolver|horizon_scanner|pythia)[\w.]*)\s+import\s+(.+)"
        r"|import\s+((?:resolver|horizon_scanner|pythia)[\w.]*))",
        re.MULTILINE,
    )

    def _direct_imports(self, path: Path) -> set[str]:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return set()
        out: set[str] = set()
        for base, names, plain in self._IMPORT_RE.findall(text):
            if plain:
                out.add(plain)
            if base:
                out.add(base)
                for name in names.split("#")[0].replace("(", "").replace(")", "").split(","):
                    name = name.split(" as ")[0].strip()
                    if name and name.islower():
                        out.add(f"{base}.{name}")
        return out

    # -- checks/ ------------------------------------------------------------

    def build_checks(self) -> dict[str, Any]:
        dest = self.path("checks")
        dest.mkdir(parents=True, exist_ok=True)
        for check in (
            self._check_enso_vs_tc_narrative,
            self._check_enso_phase_without_index,
            self._check_enso_two_ranks_alive,
            self._check_connector_rows_vs_table_delta,
            self._check_hazard_zero_failures,
            self._check_tc_duplicate_issue_dates,
            self._check_forecast_vintages,
            self._check_resolution_above_rejection_ceiling,
            self._check_acled_html_responses,
            self._check_no_past_target_month_served,
            self._check_figures_inside_their_document_window,
        ):
            try:
                check()
            except Exception as exc:  # noqa: BLE001
                self.checks.append({
                    "name": getattr(check, "__name__", "check"),
                    "verdict": "ERROR",
                    "detail": f"{type(exc).__name__}: {exc}",
                    "left": "", "right": "",
                })
        self._write_contradictions(dest / "contradictions.md")
        self._write_reconciliation(dest / "reconciliation.md")
        failed = sum(1 for c in self.checks if c["verdict"] == "FAIL")
        return {"checks": len(self.checks), "failed": failed}

    def _check(self, name: str, verdict: str, left: Any, right: Any, detail: str) -> None:
        self.checks.append({
            "name": name, "verdict": verdict,
            "left": str(left), "right": str(right), "detail": detail,
        })

    def _check_enso_vs_tc_narrative(self) -> None:
        name = "enso_phase_matches_the_tc_context_narrative"
        if not {"enso_state", "seasonal_tc_context_cache"} <= self.tables():
            return self._check(name, "SKIP", "", "", "one of the two tables is absent")
        columns = {c for c, _t in self.columns_of("enso_state")}
        kind = (
            "WHERE COALESCE(row_kind, 'live') <> 'historical' "
            if "row_kind" in columns else ""
        )
        stored = self.query(
            "SELECT enso_phase, oni, observation_date FROM enso_state "
            f"{kind}ORDER BY fetch_date DESC LIMIT 1"
        )
        if not stored or not stored[1]:
            return self._check(name, "SKIP", "", "", "enso_state is empty")
        phase = str(stored[1][0][0] or "").strip().lower()
        if not phase:
            return self._check(name, "SKIP", "", "", "no phase stored")
        cached = self.query(
            "SELECT context_text FROM seasonal_tc_context_cache "
            "WHERE context_text IS NOT NULL LIMIT 200"
        )
        words = {"el niño": "el niño", "el nino": "el niño",
                 "la niña": "la niña", "la nina": "la niña", "neutral": "neutral"}
        mentions: Counter[str] = Counter()
        for (text,) in (cached[1] if cached else []):
            lowered = str(text).lower()
            for needle, label in words.items():
                if needle in lowered:
                    mentions[label] += 1
        if not mentions:
            return self._check(name, "SKIP", phase, "", "no ENSO wording in the TC cache")
        normalised = "el niño" if "niño" in phase or "nino" in phase else phase
        top, _count = mentions.most_common(1)[0]
        verdict = "PASS" if top.startswith(normalised[:5]) or normalised.startswith(top[:5]) else "FAIL"
        self._check(
            name, verdict, f"enso_state.enso_phase={stored[1][0][0]}",
            f"tc_context most-mentioned={top} ({dict(mentions)})",
            "The August 2026 run stored 'Neutral' through a strong El Niño while "
            "the cached TC narratives said otherwise, and nothing compared them.",
        )

    def _check_enso_phase_without_index(self) -> None:
        name = "enso_phase_never_stated_without_a_measurement"
        if "enso_state" not in self.tables():
            return self._check(name, "SKIP", "", "", "enso_state is absent")
        result = self.query(
            "SELECT COUNT(*) FROM enso_state "
            "WHERE COALESCE(enso_phase, '') <> '' "
            "AND nino34_anomaly IS NULL AND oni IS NULL"
        )
        if result is None:
            return
        bad = int(result[1][0][0])
        self._check(
            name, "FAIL" if bad else "PASS", f"{bad} rows", "0",
            "A phase beside a null Niño 3.4 and a null ONI is a classification "
            "with nothing behind it. The absence of a measurement is never a "
            "default value.",
        )

    def _check_enso_two_ranks_alive(self) -> None:
        """At least two ranks of the ENSO ladder answered in the last 30 days.

        The continuity and corroboration checks compare one source against
        another, and on 2026-09-04 there was nothing to compare: ERDDAP
        answered HTTP 400, the CPC weekly file had frozen in January 2021,
        and only the ONI table worked. The ladder now reads every rank on
        every run and records which answered in ``index_evidence_json``;
        this check reads that record over the live rows of the last 30 days
        (anchored on the newest live row, so a bundle built later still
        judges the run it describes).
        """

        name = "enso_ladder_has_two_live_ranks_in_the_last_30_days"
        if "enso_state" not in self.tables():
            return self._check(name, "SKIP", "", "", "enso_state is absent")
        columns = {c for c, _t in self.columns_of("enso_state")}
        if "index_evidence_json" not in columns:
            return self._check(name, "SKIP", "", "", "no index_evidence_json column")
        kind = "COALESCE(row_kind, 'live')" if "row_kind" in columns else "'live'"
        result = self.query(
            f"SELECT fetch_date, index_evidence_json FROM enso_state "
            f"WHERE {kind} <> 'historical' AND index_evidence_json IS NOT NULL "
            f"AND fetch_date >= (SELECT MAX(fetch_date) FROM enso_state "
            f"WHERE {kind} <> 'historical') - INTERVAL 30 DAY "
            f"ORDER BY fetch_date DESC"
        )
        if result is None or not result[1]:
            return self._check(
                name, "SKIP", "", "",
                "no live enso_state row carries index evidence",
            )
        ranks_ok: set[int] = set()
        newest: dict[int, str] = {}
        for _fetch_date, evidence in result[1]:
            try:
                payload = json.loads(evidence)
            except Exception:
                continue
            for reading in payload.get("readings", []) or []:
                if reading.get("ok"):
                    rank = int(reading.get("rank", 0))
                    ranks_ok.add(rank)
                    newest.setdefault(rank, str(reading.get("newest_observation")))
        detail = (
            "The continuity and corroboration checks compare one numeric source "
            "against another. With one rank alive there is nothing to compare, "
            "and a changed column in that one source is indistinguishable from "
            "a changed ocean. Ranks answering, with each rank's newest "
            f"observation: {dict(sorted(newest.items()))}."
        )
        self._check(
            name, "PASS" if len(ranks_ok) >= 2 else "FAIL",
            f"{len(ranks_ok)} rank(s) usable: {sorted(ranks_ok)}", ">= 2 ranks", detail,
        )

    def _check_connector_rows_vs_table_delta(self) -> None:
        name = "connectors_claiming_rows_show_a_table_delta"
        report = self.diagnostics_dir / "ingestion" / "connectors_report.jsonl"
        counts = self._table_delta_map()
        if not report.is_file():
            return self._check(name, "SKIP", "", "", "no connectors_report.jsonl")
        try:
            from pythia.tools.summarize_all_phases import _CONNECTORS  # type: ignore
        except Exception:
            _CONNECTORS = {}
        offenders = []
        for record in run_log.read_stream(report):
            cid = str(record.get("connector_id") or record.get("connector") or "")
            written = _rows_written(record)
            if not cid or not written:
                continue
            table = (_CONNECTORS.get(cid) or (None, None, None, None))[1]
            if not table or table not in counts:
                continue
            delta = counts[table]
            if delta is not None and delta <= 0:
                offenders.append(f"{cid} claimed {written} rows; {table} delta={delta}")
        self._check(
            name, "FAIL" if offenders else "PASS",
            f"{len(offenders)} connectors", "0",
            "; ".join(offenders[:10])
            or "Every connector claiming rows moved its target table. A claim "
            "without a delta means the write went somewhere else, or nowhere.",
        )

    def _table_delta_map(self) -> dict[str, int | None]:
        path = self.path("db", "table_counts.csv")
        if not path.is_file():
            return {}
        out: dict[str, int | None] = {}
        with open(path, encoding="utf-8", newline="") as handle:
            rows = [r for r in handle if not r.startswith("#")]
        for record in csv.DictReader(io.StringIO("".join(rows))):
            raw = record.get("delta")
            try:
                out[str(record.get("table"))] = int(raw) if raw not in (None, "") else None
            except Exception:
                out[str(record.get("table"))] = None
        return out

    def _check_hazard_zero_failures(self) -> None:
        name = "no_hazard_reports_zero_failures_while_resolving_nothing"
        offenders = []
        checked = 0
        for path in sorted(self.path("hazard", "run_summaries").glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            checked += 1
            failures = _deep_int(payload, "failures")
            resolved = sum(
                _deep_int(payload, key) or 0
                for key in ("resolved_value", "resolved_zero", "zeros")
            )
            cells = _deep_int(payload, "cells") or 0
            if failures == 0 and cells > 0 and resolved == 0:
                offenders.append(f"{path.name}: {cells} cells assessed, 0 resolved, failures=0")
        if not checked:
            return self._check(name, "SKIP", "", "", "no hazard run summaries")
        self._check(
            name, "FAIL" if offenders else "PASS",
            f"{len(offenders)} of {checked} summaries", "0",
            "; ".join(offenders)
            or "A run that assessed cells and resolved none while reporting no "
            "failures is reporting a falsehood in the plain sense: the 2026-08 "
            "drought pass recorded failures=0 with all 756 cells inconclusive.",
        )

    def _check_tc_duplicate_issue_dates(self) -> None:
        name = "no_two_tc_outlooks_share_a_basin_season_and_issue_date"
        if "seasonal_tc_outlooks" not in self.tables():
            return self._check(name, "SKIP", "", "", "seasonal_tc_outlooks is absent")
        result = self.query(
            """
            SELECT basin, source, forecast_season,
                   json_extract_string(raw_json, '$.issue_date') AS issue_date,
                   COUNT(*) AS n
            FROM seasonal_tc_outlooks
            WHERE json_extract_string(raw_json, '$.issue_date') IS NOT NULL
            GROUP BY basin, source, forecast_season, issue_date
            HAVING COUNT(*) > 1
            """
        )
        if result is None:
            return
        rows = result[1]
        self._check(
            name, "FAIL" if rows else "PASS", f"{len(rows)} collisions", "0",
            "; ".join(f"{r[0]}/{r[1]}/{r[2]} @ {r[3]} x{r[4]}" for r in rows[:10])
            or "Two records of the same thing for the same period cannot share an "
            "issue date; when they do, one has been misread.",
        )

    def _check_forecast_vintages(self) -> None:
        name = "no_stored_forecast_vintage_past_its_threshold"
        if "conflict_forecasts" not in self.tables():
            return self._check(name, "SKIP", "", "", "conflict_forecasts is absent")
        result = self.query(
            """
            SELECT source, MAX(forecast_issue_date) AS latest,
                   date_diff('day', MAX(TRY_CAST(forecast_issue_date AS DATE)),
                             CURRENT_DATE) AS age_days
            FROM conflict_forecasts GROUP BY source ORDER BY source
            """
        )
        if result is None:
            return
        threshold = self._STALENESS_DAYS["conflict_forecasts"]
        stale = [r for r in result[1] if r[2] is not None and r[2] > threshold]
        self._check(
            name, "FAIL" if stale else "PASS",
            f"{len(stale)} of {len(result[1])} sources", f"at most {threshold} days",
            "; ".join(f"{r[0]} latest={r[1]} ({r[2]}d)" for r in stale)
            or "Every conflict-forecast source is inside its staleness threshold.",
        )

    #: The connectors whose calls go to acleddata.com, by the ids the
    #: connectors report uses for them.
    _ACLED_CONNECTOR_IDS = ("acled_client", "acled", "acledcast_forecasts", "acled_cast",
                            "acled_political_events", "acled_political")

    def _check_acled_html_responses(self) -> None:
        """Any ACLED call answered with a web page is recorded as a failure.

        ACLED's gateway serves its "Unauthorized" page with HTTP 200, even
        to a request that asked for JSON, so a 401, a WAF interstitial and
        a session expiry are indistinguishable by status code. A run must
        not report ok=N fail=0 while receiving web pages.
        """

        name = "acled_html_responses_are_recorded_as_connector_failures"
        stream = self._stream_file(run_log.STREAM_HTTP)
        if stream is None:
            return self._check(name, "SKIP", "", "", "no HTTP stream recorded")
        html_calls: list[str] = []
        for record in run_log.read_stream(stream):
            url = str(record.get("url") or "")
            if "acleddata.com" not in url:
                continue
            content_type = str(record.get("content_type") or "").lower()
            if "html" in content_type:
                html_calls.append(
                    f"{record.get('connector')} {record.get('status')} {url[:80]}"
                )
        if not html_calls:
            return self._check(
                name, "PASS", "0 HTML responses from acleddata.com", "0",
                "No ACLED call was answered with a web page.",
            )
        report = self.diagnostics_dir / "ingestion" / "connectors_report.jsonl"
        ok_connectors: list[str] = []
        if report.is_file():
            for record in run_log.read_stream(report):
                cid = str(record.get("connector_id") or record.get("connector") or "")
                if cid in self._ACLED_CONNECTOR_IDS and str(record.get("status")) == "ok" \
                        and not record.get("reason"):
                    ok_connectors.append(cid)
        verdict = "FAIL" if ok_connectors else "PASS"
        self._check(
            name, verdict,
            f"{len(html_calls)} HTML response(s); ACLED connectors reporting ok: {ok_connectors or 'none'}",
            "every ACLED connector that received a web page reports a failure",
            "HTML from acleddata.com is the shape of an unauthenticated call, a WAF "
            "challenge and a session expiry alike; it is never zero records. "
            "Calls: " + "; ".join(html_calls[:10]),
        )

    def _check_no_past_target_month_served(self) -> None:
        """No conflict_forecasts row a prompt reader serves has a past target month.

        Runs the SAME predicate the readers apply (imported, not copied), so
        the check fails the day someone replaces it with a lead-months
        filter — which on a frozen vintage served January under the name of
        "lead 1" in September.
        """

        name = "no_conflict_forecast_served_has_a_target_month_in_the_past"
        if "conflict_forecasts" not in self.tables():
            return self._check(name, "SKIP", "", "", "conflict_forecasts is absent")
        if "target_month" not in {c for c, _t in self.columns_of("conflict_forecasts")}:
            return self._check(name, "SKIP", "", "", "conflict_forecasts has no target_month column")
        try:
            from horizon_scanner.conflict_forecasts import SERVED_TARGET_FILTER_SQL
        except Exception as exc:  # noqa: BLE001
            return self._check(name, "SKIP", "", "", f"reader predicate unavailable: {exc}")
        result = self.query(
            f"SELECT source, COUNT(*) FROM conflict_forecasts "
            f"WHERE ({SERVED_TARGET_FILTER_SQL}) "
            f"AND target_month < date_trunc('month', CURRENT_DATE) GROUP BY source"
        )
        if result is None:
            return
        served_past = {str(r[0]): int(r[1]) for r in result[1]}
        stale = self.query(
            "SELECT source, COUNT(*) FROM conflict_forecasts "
            "WHERE target_month < date_trunc('month', CURRENT_DATE) GROUP BY source"
        )
        stale_rows = {str(r[0]): int(r[1]) for r in (stale[1] if stale else [])}
        self._check(
            name, "FAIL" if served_past else "PASS",
            f"{sum(served_past.values())} served rows with a past target month",
            "0",
            f"Reader predicate: {SERVED_TARGET_FILTER_SQL}. Rows in the table whose "
            f"target month has passed (stored, flagged, not served): {stale_rows or 'none'}.",
        )

    def _check_figures_inside_their_document_window(self) -> None:
        """No accepted figure is about a date outside its cell's reporting window.

        Documents 4222929 and 4222839 (floods between March and May) were
        filed under AFG / FL / 2026-07 in the September 2026 run. The
        ledger now carries the date each figure is about and the
        document's primary country; this check re-runs the window test
        over the accepted rows alone, so it fails the day the attribution
        stage stops rejecting them.
        """

        name = "no_accepted_figure_lies_outside_its_cell_reporting_window"
        ledger = self.path("hazard", "figures_ledger.csv")
        if not ledger.is_file():
            return self._check(name, "SKIP", "", "", "no figures ledger")
        try:
            from resolver.hazard_resolution.figures import reporting_window
            from resolver.hazard_resolution.rulebook import load_rulebook
            from resolver.hazard_resolution.sources import parse_date

            rulebook = load_rulebook()
        except Exception as exc:  # noqa: BLE001
            return self._check(name, "SKIP", "", "", f"rulebook unavailable: {exc}")
        with open(ledger, encoding="utf-8", newline="") as handle:
            body = "".join(line for line in handle if not line.startswith("#"))
        offenders: list[str] = []
        accepted = 0
        undated = 0
        for record in csv.DictReader(io.StringIO(body)):
            if str(record.get("outcome")) != "accepted":
                continue
            accepted += 1
            about = None
            for field in ("figure_date", "doc_date_original", "doc_date"):
                about = parse_date(record.get(field) or "")
                if about is not None:
                    break
            if about is None:
                undated += 1
                continue
            ym = str(record.get("ym") or "")
            try:
                start, end = reporting_window(ym, rulebook)
            except Exception:
                continue
            if not (start <= about <= end):
                offenders.append(
                    f"{record.get('iso3')}/{record.get('hazard')}/{ym} doc {record.get('doc_id')} "
                    f"about {about.isoformat()} (window {start}..{end})"
                )
            primary = str(record.get("doc_primary_country") or "").upper()
            if primary and primary != str(record.get("iso3") or "").upper():
                offenders.append(
                    f"{record.get('iso3')}/{record.get('hazard')}/{ym} doc {record.get('doc_id')} "
                    f"is about {primary}"
                )
        if not accepted:
            return self._check(name, "SKIP", "", "", "no accepted figures in the ledger")
        self._check(
            name, "FAIL" if offenders else "PASS",
            f"{len(offenders)} of {accepted} accepted figures ({undated} undated, not tested)",
            "0",
            "; ".join(offenders[:10])
            or "Every accepted figure with a date is about a date inside its cell's "
            "reporting window, and about the cell's own country.",
        )

    def _check_resolution_above_rejection_ceiling(self) -> None:
        name = "no_resolution_exceeds_a_ceiling_that_rejected_a_figure_in_the_same_cell"
        ledger = self.path("hazard", "figures_ledger.csv")
        if "haz_resolutions" not in self.tables() or not ledger.is_file():
            return self._check(name, "SKIP", "", "", "need haz_resolutions and a figures ledger")
        rejected: dict[tuple[str, str, str], float] = {}
        with open(ledger, encoding="utf-8", newline="") as handle:
            body = "".join(line for line in handle if not line.startswith("#"))
        for record in csv.DictReader(io.StringIO(body)):
            if str(record.get("outcome")) != "rejected":
                continue
            try:
                ceiling = float(record.get("ceiling") or "")
            except Exception:
                continue
            key = (str(record.get("iso3")), str(record.get("hazard")), str(record.get("ym")))
            rejected[key] = min(rejected.get(key, ceiling), ceiling)
        if not rejected:
            return self._check(name, "SKIP", "", "", "no rejected figures recorded")
        result = self.query(
            "SELECT iso3, hazard, printf('%04d-%02d', year, month), value "
            "FROM haz_resolutions WHERE value IS NOT NULL"
        )
        if result is None:
            return
        offenders = [
            f"{r[0]}/{r[1]}/{r[2]}: resolved {r[3]:.0f} > ceiling {rejected[(str(r[0]), str(r[1]), str(r[2]))]:.0f}"
            for r in result[1]
            if (str(r[0]), str(r[1]), str(r[2])) in rejected
            and float(r[3]) > rejected[(str(r[0]), str(r[1]), str(r[2]))]
        ]
        self._check(
            name, "FAIL" if offenders else "PASS",
            f"{len(offenders)} cells", "0",
            "; ".join(offenders[:10])
            or "No cell published a figure larger than the ceiling it used to "
            "reject another figure. When one does, the ceiling and the ladder "
            "disagree about the same cell and one of them is wrong.",
        )

    def _write_contradictions(self, path: Path) -> None:
        lines = [
            "# Contradictions",
            "",
            "Assertions run across the assembled bundle, each naming the two",
            "values compared. A FAIL is a disagreement between two things in this",
            "run that cannot both be true.",
            "",
            "| verdict | check | observed | expected |",
            "| --- | --- | --- | --- |",
        ]
        for check in self.checks:
            lines.append(
                f"| {check['verdict']} | {check['name']} | "
                f"{check['left'].replace('|', '/')} | {check['right'].replace('|', '/')} |"
            )
        lines.append("")
        for check in self.checks:
            lines.append(f"### {check['verdict']} — {check['name']}")
            lines.append("")
            lines.append(check["detail"] or "(no detail)")
            lines.append("")
        write_text(path, "\n".join(lines) + "\n")

    def _write_reconciliation(self, path: Path) -> None:
        """Rows fetched, normalised, claimed written, and the actual delta.

        The existing connector summary carries a caveat that its totals are
        not comparable across connectors. Naming the semantics of each
        counter makes them comparable instead.
        """

        report = self.diagnostics_dir / "ingestion" / "connectors_report.jsonl"
        deltas = self._table_delta_map()
        try:
            from pythia.tools.summarize_all_phases import _CONNECTORS  # type: ignore
        except Exception:
            _CONNECTORS = {}

        lines = [
            "# Reconciliation",
            "",
            "| connector | status | fetched | normalised | claimed written | target table | table delta | agrees |",
            "| --- | --- | ---: | ---: | ---: | --- | ---: | --- |",
        ]
        counts_meaning = (
            "\n**What each counter means.** `fetched` is upstream records the "
            "connector received; `normalised` is rows it produced in the canonical "
            "shape; `claimed written` is the connector's own report, which for "
            "per-country sources counts COUNTRIES and for self-storing globals "
            "counts ROWS — which is why the raw numbers were never comparable. "
            "`table delta` is the only figure measured the same way for every "
            "connector: rows_after minus rows_before in the target table.\n"
        )
        if not report.is_file():
            lines.append("| _(no connectors_report.jsonl)_ | | | | | | | |")
            write_text(path, "\n".join(lines) + "\n" + counts_meaning)
            return

        for record in run_log.read_stream(report):
            cid = str(record.get("connector_id") or record.get("connector") or "?")
            table = (_CONNECTORS.get(cid) or (None, None, None, None))[1] or ""
            delta = deltas.get(table)
            written = _rows_written(record)
            counts = record.get("counts") or {}
            agrees = "—"
            if written and delta is not None:
                agrees = "yes" if delta > 0 else "**NO**"
            lines.append(
                f"| {cid} | {record.get('status', '?')} | "
                f"{counts.get('fetched', '')} | {counts.get('normalized', counts.get('normalised', ''))} | "
                f"{written or ''} | {table} | {'' if delta is None else delta} | {agrees} |"
            )
        write_text(path, "\n".join(lines) + "\n" + counts_meaning)

    # -- README + manifest --------------------------------------------------

    def build_readme(self, manifest: dict[str, Any]) -> None:
        run = {}
        run_file = self.path("run", "workflow_run.json")
        if run_file.is_file():
            run = json.loads(run_file.read_text(encoding="utf-8"))

        failed = [c for c in self.checks if c["verdict"] == "FAIL"]
        lines = [
            "# Resolver debug bundle",
            "",
            "Everything needed to diagnose one Resolver Update run, without the",
            "repository, the workflow logs or the database. If you have never seen",
            "this system: it ingests humanitarian data from about twenty upstream",
            "sources into a DuckDB file, then resolves how many people a flood,",
            "cyclone or drought affected in each country and month.",
            "",
            "## What ran",
            "",
            f"- run: {run.get('run_url') or run.get('run_id') or 'not a CI run'}",
            f"- workflow: {run.get('workflow') or 'unknown'}",
            f"- trigger: {run.get('trigger') or 'unknown'}",
            f"- bundled at: {run.get('bundled_at')}",
            f"- database: {self.db_path if self.db_path else 'none'}"
            + ("" if self.con is not None else "  **(unreadable — db/ is mostly empty)**"),
            "",
            "## What is wrong",
            "",
        ]
        if failed:
            lines.append("Contradictions found in this run:")
            lines.append("")
            for check in failed:
                lines.append(f"- **{check['name']}** — {check['left']} (expected {check['right']})")
        else:
            lines.append("No contradiction check failed. That is not the same as")
            lines.append("everything being right; read `checks/reconciliation.md` next.")
        if self.problems:
            lines += ["", "Problems assembling this bundle:", ""]
            lines += [f"- {p}" for p in self.problems[:40]]
            if len(self.problems) > 40:
                lines.append(f"- …and {len(self.problems) - 40} more (see manifest.json)")
        lines += [
            "",
            "## Where to look first",
            "",
            "1. `checks/contradictions.md` — two things in this run that disagree.",
            "2. `logs/log_index.md` — every ERROR and WARNING, collapsed so one",
            "   fault repeated 252 times reads as one row with a count.",
            "3. `checks/reconciliation.md` — what each connector claimed against",
            "   what its table actually gained.",
            "4. `hazard/cell_ledger.csv` — for any unresolved hazard cell, the",
            "   reason code saying why it produced no row.",
            "5. `http/requests.jsonl` and `http/envelopes/` — the URL each",
            "   connector called and what came back, including the response",
            "   fields the connectors themselves discard.",
            "",
            "## What is in here",
            "",
            "| path | what it answers |",
            "| --- | --- |",
            "| `run/` | which code, config, environment and windows this run used |",
            "| `logs/` | the phase logs, plus a normalised error histogram |",
            "| `http/` | every outbound call: URL, status, timing, response envelope |",
            "| `db/` | table counts before and after, freshness per table, fixed queries |",
            "| `hazard/` | per-cell and per-figure ledgers, extraction spend, backcast state |",
            "| `config/` | the rulebook, connector configs and workflows, verbatim |",
            "| `code/` | the source files that participated, verbatim |",
            "| `checks/` | contradictions and connector reconciliation |",
            "",
            "## Reading the ledgers",
            "",
            "`hazard/cell_ledger.csv` has one row per country-month-hazard the",
            "machine assessed. A blank `status` means no row was written; the",
            "`reason_code` says why, and each reason names a different repair:",
            "",
            "- `pending_before_freeze` — the sources have not finished reporting.",
            "  Nothing is wrong; the calendar has not caught up.",
            "- `sweep_inconclusive` — ReliefWeb could not be read, so silence is",
            "  unproven and a zero would be an outage recorded as a fact.",
            "- `coverage_gate_suppressed_zero` — the detector cannot show it",
            "  covered the month, so an ingestion gap must not read as quiet.",
            "- `indicator_inconclusive` — a drought indicator was unreadable.",
            "- `cell_raised` — the walk threw. A re-run retries exactly this cell.",
            "- `unexplained_no_row` — nothing accounted for the cell. That is a bug.",
            "",
            "`hazard/figures_ledger.csv` carries every extracted figure and its",
            "fate. For a rejection, `ceiling`, `ceiling_source_ref` and",
            "`ceiling_field` say what it was measured against and where that",
            "number came from — a ceiling of 2 against a reported 40,000 is a",
            "GDACS enrichment failure, not a mis-transcription.",
            "",
            "## Redaction",
            "",
            "No credential is in this bundle. Values are replaced by",
            "`<redacted:sha256:xxxxxxxx>`, a fingerprint rather than a mask, so",
            "you can still tell whether this run's key is the one the last good",
            "run used. The whole zip was scanned for every secret in the build",
            "environment before it was written; a hit fails the build.",
            "",
            f"Bundle version {manifest.get('bundle_version')}. "
            f"{manifest.get('n_files')} files, "
            f"{manifest.get('compressed_bytes', 0) // 1024} KB compressed.",
            "",
        ]
        write_text(self.path("README.md"), "\n".join(lines) + "\n")


def _utcnow() -> str:
    import datetime as dt

    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _rows_written(record: dict[str, Any]) -> int:
    counts = record.get("counts") or {}
    for key in ("written", "rows_written", "stored"):
        try:
            value = int(counts.get(key) or record.get(key) or 0)
        except Exception:
            value = 0
        if value:
            return value
    return 0


def _deep_int(payload: Any, key: str) -> int | None:
    """Find ``key`` anywhere in a nested summary and return it as an int.

    Run summaries nest differently per hazard; a reader of the bundle should
    not have to know which shape this month's summary happened to use.
    """

    if isinstance(payload, dict):
        if key in payload:
            try:
                return int(payload[key])
            except Exception:
                return None
        for value in payload.values():
            found = _deep_int(value, key)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = _deep_int(item, key)
            if found is not None:
                return found
    return None


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

BUNDLE_VERSION = 1


def _iter_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path


def _zip_size(staging: Path, out_path: Path, drop: set[str]) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in _iter_files(staging):
            rel = path.relative_to(staging).as_posix()
            if any(rel == d or rel.startswith(d.rstrip("/") + "/") for d in drop):
                continue
            zf.write(path, rel)
    return out_path.stat().st_size


def _truncate_largest(staging: Path, skip: set[str], keep_lines: int = 2000) -> str | None:
    """Cut the largest remaining file down to its head and tail. Loudly."""

    candidates = [
        p for p in _iter_files(staging)
        if p.relative_to(staging).as_posix() not in skip
        and p.suffix in {".log", ".jsonl", ".csv", ".md", ".txt"}
    ]
    if not candidates:
        return None
    target = max(candidates, key=lambda p: p.stat().st_size)
    original = target.stat().st_size
    try:
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return None
    if len(lines) <= keep_lines * 2:
        return None
    half = keep_lines
    body = (
        lines[:half]
        + [
            "",
            f"### TRUNCATED — original file was {original} bytes / {len(lines)} lines;"
            f" {len(lines) - half * 2} lines were removed from the middle to keep the"
            " bundle under its size ceiling ###",
            "",
        ]
        + lines[-half:]
    )
    target.write_text("\n".join(body) + "\n", encoding="utf-8")
    return f"{target.relative_to(staging).as_posix()}: {original} bytes -> head+tail {half * 2} lines"


def build_bundle(
    *,
    out_path: Path,
    db_path: Path | None,
    diagnostics_dir: Path,
    run_log_dir: Path | None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    staging: Path | None = None,
    environ: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Assemble the bundle and return its manifest. Never raises on bad input."""

    temp_dir = None
    if staging is None:
        temp_dir = tempfile.mkdtemp(prefix="resolver-debug-bundle-")
        staging = Path(temp_dir) / "bundle"
    staging.mkdir(parents=True, exist_ok=True)

    builder = BundleBuilder(
        out_path=out_path, db_path=db_path, diagnostics_dir=diagnostics_dir,
        run_log_dir=run_log_dir, staging=staging, max_bytes=max_bytes, environ=environ,
    )
    try:
        # Order matters: logs before code (code is selected from the logs),
        # db before checks (checks read table_counts.csv), hazard before
        # checks (the ceiling check reads the figures ledger).
        builder.section("run", builder.build_run)
        builder.section("logs", builder.build_logs)
        builder.section("http", builder.build_http)
        builder.section("db", builder.build_db)
        builder.section("hazard", builder.build_hazard)
        builder.section("config", builder.build_config)
        builder.section("code", builder.build_code)
        builder.section("checks", builder.build_checks)

        manifest: dict[str, Any] = {
            "bundle_version": BUNDLE_VERSION,
            "generated_at": _utcnow(),
            "sections": builder.sections,
            "problems": builder.problems,
            "notes": builder.notes,
            "checks": builder.checks,
            "max_bytes": max_bytes,
            "dropped": [],
            "truncated": [],
        }
        write_json(staging / "manifest.json", manifest)
        builder.build_readme(manifest)

        # Redaction gate. Assembly is finished, so this sees everything that
        # would ship — including text a capture-time redaction missed.
        leaks = _scan_for_leaks(staging, builder.secrets)
        manifest["redaction_scan"] = {
            "secret_values_checked": len(builder.secrets),
            "files_with_hits": leaks,
        }
        if leaks:
            manifest["failed"] = "redaction scan found secret values in the bundle"
            write_json(staging / "manifest.json", manifest)
            raise SecretLeak(leaks)

        # Size ceiling. Code goes before evidence does, and every drop is
        # recorded — a bundle that silently loses half its content is worse
        # than one that says it had to.
        drop: set[str] = set()
        size = _zip_size(staging, out_path, drop)
        if size > max_bytes:
            drop.add("code")
            manifest["dropped"].append(
                f"code/ ({size} bytes exceeded the {max_bytes}-byte ceiling; "
                "code is dropped before any evidence is)"
            )
            size = _zip_size(staging, out_path, drop)
        guard = 0
        while size > max_bytes and guard < 12:
            guard += 1
            note = _truncate_largest(staging, skip=set(), keep_lines=1500)
            if note is None:
                manifest["dropped"].append(
                    f"still {size} bytes after truncation: nothing further could be cut"
                )
                break
            manifest["truncated"].append(note)
            size = _zip_size(staging, out_path, drop)

        manifest["compressed_bytes"] = size
        manifest["n_files"] = sum(
            1
            for p in _iter_files(staging)
            if not any(
                p.relative_to(staging).as_posix().startswith(d.rstrip("/") + "/") for d in drop
            )
        )
        write_json(staging / "manifest.json", manifest)
        builder.build_readme(manifest)
        manifest["compressed_bytes"] = _zip_size(staging, out_path, drop)
        return manifest
    finally:
        if builder._con is not None:
            try:
                from resolver.db import duckdb_io

                duckdb_io.close_db(builder._con)
            except Exception:
                pass
        if temp_dir and os.environ.get("PYTHIA_BUNDLE_KEEP_STAGING", "") not in {"1", "true"}:
            shutil.rmtree(temp_dir, ignore_errors=True)


class SecretLeak(RuntimeError):
    """A secret value reached the assembled bundle. The build must fail."""

    def __init__(self, hits: dict[str, list[str]]):
        super().__init__(
            "redaction scan found secret values in: " + ", ".join(sorted(hits))
        )
        self.hits = hits


def _scan_for_leaks(staging: Path, secrets: list[str]) -> dict[str, list[str]]:
    """Every file, checked for every known secret value. Fingerprints only."""

    hits: dict[str, list[str]] = {}
    if not secrets:
        return hits
    for path in _iter_files(staging):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        found = find_secrets(text, secrets)
        if found:
            hits[path.relative_to(staging).as_posix()] = found
    return hits


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="build_resolver_debug_bundle",
        description="Build one self-contained diagnostic zip for a Resolver Update run.",
    )
    parser.add_argument(
        "--db",
        default=os.environ.get("RESOLVER_DB_URL") or os.environ.get("PYTHIA_DB_URL")
        or os.environ.get("BACKFILL_DB_PATH") or "data/resolver.duckdb",
        help="DuckDB path or duckdb:/// URL (default: the run's DB env vars)",
    )
    parser.add_argument("--out", required=True, help="Where to write the zip")
    parser.add_argument(
        "--diagnostics-dir", default="diagnostics",
        help="Directory holding the run's phase logs and signatures",
    )
    parser.add_argument(
        "--run-log-dir", default=os.environ.get(run_log.ENV_DIR) or "",
        help="Directory holding this run's evidence streams (PYTHIA_RUN_LOG_DIR)",
    )
    parser.add_argument(
        "--max-bytes", type=int, default=DEFAULT_MAX_BYTES,
        help=f"Compressed ceiling in bytes (default {DEFAULT_MAX_BYTES})",
    )
    args = parser.parse_args(argv)

    db_path = normalise_db_path(args.db)
    run_dir = Path(args.run_log_dir) if args.run_log_dir else None
    try:
        manifest = build_bundle(
            out_path=Path(args.out),
            db_path=db_path,
            diagnostics_dir=Path(args.diagnostics_dir),
            run_log_dir=run_dir if (run_dir and run_dir.is_dir()) else None,
            max_bytes=args.max_bytes,
        )
    except SecretLeak as exc:
        sys.stderr.write(f"REDACTION FAILURE: {exc}\n")
        try:
            Path(args.out).unlink()
        except Exception:
            pass
        return 2

    print(f"Wrote {args.out} ({manifest.get('compressed_bytes', 0)} bytes, "
          f"{manifest.get('n_files', 0)} files)")
    for problem in manifest.get("problems", []):
        print(f"  problem: {problem}")
    failed = [c for c in manifest.get("checks", []) if c.get("verdict") == "FAIL"]
    for check in failed:
        print(f"  contradiction: {check['name']} — {check['left']}")
    # Exit 0 even with contradictions: the bundle's job is to REPORT them,
    # and a red diagnostics step at the end of a long ingest teaches people
    # to ignore the step rather than to read the bundle.
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
