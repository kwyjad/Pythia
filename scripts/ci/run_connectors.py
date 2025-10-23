"""Run resolver connectors sequentially with per-connector log teeing."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List

from resolver.ingestion.diagnostics_emitter import (
    append_jsonl as diagnostics_append_jsonl,
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


def main() -> int:
    root = Path.cwd()
    logs_dir = root / LOGS_DIR
    report_path = root / REPORT_PATH

    logs_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        report_path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass

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

    for connector in connectors:
        name = connector.strip()
        if not name:
            continue
        module = f"resolver.ingestion.{name}"
        cmd = [python_exe, "-m", module]
        log_path = logs_dir / f"{name}.log"
        print(f"=== RUN {name} → {log_path} ===")
        diagnostics_ctx = diagnostics_start_run(name, "real")
        rc = try_with_optional_debug(cmd, log_path, env)
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
        diagnostics_append_jsonl(report_path, result)
        print(f"=== DONE {name} (rc={rc}) ===")
        if rc != 0 and overall_rc == 0:
            overall_rc = rc
    return overall_rc


if __name__ == "__main__":
    raise SystemExit(main())
