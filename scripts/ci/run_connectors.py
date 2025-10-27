"""Run resolver connectors sequentially with per-connector log teeing."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from resolver.ingestion.diagnostics_emitter import (
    finalize_run as diagnostics_finalize_run,
    start_run as diagnostics_start_run,
)

DEFAULT_CONNECTORS: List[str] = [
    "acled_client",
    "dtm_client",
    "ifrc_go_client",
    "ipc_client",
    "reliefweb_client",
    "unhcr_client",
    "unhcr_odp_client",
    "wfp_mvam_client",
    "who_phe_client",
]

LOGS_DIR = Path("diagnostics") / "ingestion" / "logs"
REPORT_PATH = Path("diagnostics") / "ingestion" / "connectors_report.jsonl"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse only the arguments provided by our runner or tests.

    When invoked under pytest we pass an empty list so argparse does not try
    to consume pytest's own CLI flags from ``sys.argv``.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extra-args",
        action="append",
        default=[],
        metavar="CONNECTOR=ARGS",
        help=(
            "Optional connector-specific CLI arguments. Can be repeated, e.g. "
            "--extra-args 'dtm_client=--strict-empty'"
        ),
    )
    parser.add_argument(
        "--extra-env",
        action="append",
        default=[],
        metavar="CONNECTOR=KEY=VALUE",
        help=(
            "Optional connector-specific environment overrides. Can be repeated, e.g. "
            "--extra-env 'dtm_client=DTM_NO_DATE_FILTER=1'"
        ),
    )
    if argv is None:
        argv = []
    return parser.parse_args(list(argv))


def _parse_extra_args(values: Sequence[str]) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    for raw in values:
        text = (raw or "").strip()
        if not text or "=" not in text:
            continue
        connector, extra = text.split("=", 1)
        key = connector.strip()
        if not key:
            continue
        args = shlex.split(extra.strip()) if extra.strip() else []
        if key in mapping:
            mapping[key].extend(args)
        else:
            mapping[key] = list(args)
    return mapping


def _parse_extra_env(values: Sequence[str]) -> Dict[str, Dict[str, str]]:
    mapping: Dict[str, Dict[str, str]] = {}
    for raw in values:
        text = (raw or "").strip()
        if not text or "=" not in text:
            continue
        connector, rest = text.split("=", 1)
        key_value = rest.strip()
        if not connector or "=" not in key_value:
            continue
        key, value = key_value.split("=", 1)
        conn_key = connector.strip()
        env_key = key.strip()
        if not conn_key or not env_key:
            continue
        env_value = value.strip()
        bucket = mapping.setdefault(conn_key, {})
        bucket[env_key] = env_value
    return mapping


class TeeProcess:
    """Wrapper around ``subprocess.Popen`` exposing stdout iteration."""

    def __init__(self, cmd: Iterable[str], env: Dict[str, str]):
        self._proc = subprocess.Popen(  # noqa: S603, S607 - controlled command
            list(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            bufsize=1,
        )

    @property
    def stdout(self):
        return self._proc.stdout

    @property
    def returncode(self) -> int:
        return self._proc.returncode

    def wait(self) -> None:
        self._proc.wait()


def _quoted(cmd: Iterable[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def _as_jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _as_jsonable(item) for key, item in value.items()}
    if dataclasses.is_dataclass(value):  # pragma: no branch - trivial checks
        return _as_jsonable(dataclasses.asdict(value))
    if hasattr(value, "model_dump"):
        try:
            return _as_jsonable(value.model_dump())
        except Exception:  # pragma: no cover - defensive
            pass
    if hasattr(value, "dict"):
        try:
            return _as_jsonable(value.dict())  # type: ignore[call-arg]
        except Exception:  # pragma: no cover - defensive
            pass
    if hasattr(value, "_asdict"):
        try:
            return _as_jsonable(value._asdict())
        except Exception:  # pragma: no cover - defensive
            pass
    if isinstance(value, (list, tuple, set)):
        return [_as_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        return json.loads(json.dumps(value, default=str))
    except Exception:  # pragma: no cover - defensive
        return str(value)


def _write_report(path: Path, entries: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for entry in entries:
            payload: object = dict(entry) if isinstance(entry, Mapping) else entry
            handle.write(json.dumps(_as_jsonable(payload), default=str))
            handle.write("\n")


def tee_run(cmd: List[str], log_path: Path, env: Dict[str, str], *, append: bool = False) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append and log_path.exists() else "w"
    with log_path.open(mode, encoding="utf-8") as handle:
        if mode == "a":
            handle.write("\n")
        handle.write(f"$ {_quoted(cmd)}\n")
        handle.flush()
        proc = TeeProcess(cmd, env)
        stdout = proc.stdout
        assert stdout is not None
        for line in stdout:
            sys.stdout.write(line)
            handle.write(line)
        proc.wait()
        return proc.returncode or 0


def try_with_optional_debug(cmd: List[str], log_path: Path, env: Dict[str, str]) -> int:
    log_level = env.get("LOG_LEVEL", "INFO").upper()
    if log_path.exists():
        try:
            log_path.unlink()
        except OSError:
            pass
    if log_level == "DEBUG":
        debug_cmd = list(cmd) + ["--debug"]
        rc = tee_run(debug_cmd, log_path, env)
        if rc == 2:
            print(f"retrying {cmd[-1]} without --debug (rc=2)")
            rc = tee_run(cmd, log_path, env, append=True)
        return rc
    return tee_run(cmd, log_path, env)


def _resolve_connectors(env: Dict[str, str]) -> List[str]:
    only = env.get("ONLY_CONNECTOR", "").strip()
    if only:
        return [only]
    raw_list = [item.strip() for item in env.get("CONNECTOR_LIST", "").split(",") if item.strip()]
    return raw_list or list(DEFAULT_CONNECTORS)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else [])
    extra_args = _parse_extra_args(args.extra_args)
    extra_env = _parse_extra_env(args.extra_env)
    root = Path.cwd()
    logs_dir = root / LOGS_DIR
    report_path = root / REPORT_PATH

    diag_base = root / "diagnostics" / "ingestion"
    for sub in ("logs", "raw", "metrics", "samples", "dtm"):
        (diag_base / sub).mkdir(parents=True, exist_ok=True)
    for sub in ("raw", "metrics", "samples"):
        keep = diag_base / sub / ".keep"
        if not keep.exists():
            try:
                keep.touch()
            except OSError:
                pass

    logs_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    reset_message = "initialised"
    try:
        report_path.write_text("", encoding="utf-8")
        reset_message = "reset"
    except OSError:
        try:
            report_path.unlink()
            report_path.touch()
            reset_message = "reset"
        except OSError:
            pass
    print(f"{reset_message} connectors report at {report_path}")

    env = os.environ.copy()
    if env.get("LOG_LEVEL", "INFO").upper() == "DEBUG":
        env.setdefault("PYTHONDEVMODE", "1")
        env.setdefault("PYTHONWARNINGS", "default")

    connectors = _resolve_connectors(env)
    if not connectors:
        print("no connectors requested; exiting")
        return 0

    python_exe = sys.executable
    overall_rc = 0
    records: Dict[str, Mapping[str, Any]] = {}

    for connector in connectors:
        name = connector.strip()
        if not name:
            continue
        module = f"resolver.ingestion.{name}"
        cmd = [python_exe, "-m", module]
        additional = extra_args.get(name, [])
        if additional:
            cmd.extend(additional)
        log_path = logs_dir / f"{name}.log"
        print(f"=== RUN {name} → {log_path} ===")
        diagnostics_ctx = diagnostics_start_run(name, "real")
        connector_env = dict(env)
        overrides = extra_env.get(name)
        if overrides:
            connector_env.update(overrides)
        rc = try_with_optional_debug(cmd, log_path, connector_env)
        status = "ok" if rc == 0 else "error"
        reason = None if rc == 0 else f"exit code {rc}"
        result = diagnostics_finalize_run(
            diagnostics_ctx,
            status,
            reason=reason,
            extras={
                "exit_code": rc,
                "log_path": str(log_path),
            },
        )
        records[name] = result
        _write_report(report_path, [result])
        for sub in ("raw", "metrics", "samples"):
            keep = diag_base / sub / ".keep"
            try:
                keep.touch()
            except OSError:
                pass
        print(f"=== DONE {name} (rc={rc}) ===")
        if rc != 0 and overall_rc == 0:
            overall_rc = rc
    print(f"connectors_report entries written: {len(records)}")
    return overall_rc


if __name__ == "__main__":
    raise SystemExit(main())
