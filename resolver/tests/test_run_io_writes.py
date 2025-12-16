# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import json
from pathlib import Path

from resolver.ingestion._shared import run_io


def test_write_and_append_jsonl(tmp_path: Path) -> None:
    payload = {"a": 1, "b": "x"}
    target = tmp_path / "a" / "b" / "c.json"

    written = run_io.write_json(target, payload)
    assert written == target
    assert written.exists()
    assert json.loads(written.read_text(encoding="utf-8")) == payload

    jsonl_target = tmp_path / "a" / "log.jsonl"
    run_io.append_jsonl(jsonl_target, {"k": 1})
    run_io.append_jsonl(jsonl_target, {"k": 2})

    contents = [
        json.loads(line)
        for line in jsonl_target.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert contents == [{"k": 1}, {"k": 2}]
