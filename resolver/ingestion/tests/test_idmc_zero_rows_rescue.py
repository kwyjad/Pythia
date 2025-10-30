import json
import os
import subprocess
import sys
from pathlib import Path


def test_idmc_zero_rows_rescue(tmp_path):
    env = os.environ.copy()
    env["IDMC_FORCE_CACHE_ONLY"] = "1"
    env["IDMC_CACHE_DIR"] = str(tmp_path / "cache")
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [env.get("PYTHONPATH"), str(Path.cwd())])
    )

    connectors_path = Path("diagnostics/ingestion/connectors.jsonl")
    connectors_path.parent.mkdir(parents=True, exist_ok=True)
    before = []
    if connectors_path.exists():
        before = connectors_path.read_text(encoding="utf-8").splitlines()

    subprocess.run(
        [
            sys.executable,
            "-m",
            "resolver.ingestion.idmc.cli",
            "--only-countries=ZZZ",
            "--window-days=7",
        ],
        check=True,
        env=env,
    )

    lines = connectors_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) >= len(before) + 1
    payload = json.loads(lines[-1])
    assert payload["status"] == "ok"
    assert payload["rows_normalized"] == 0
    assert payload["zero_rows"]
    assert payload["zero_rows"]["selectors"]["only_countries"] == ["ZZZ"]
    assert "notes" in payload["zero_rows"]
    assert payload["samples"]["drop_reasons"].endswith("drop_reasons.json")
