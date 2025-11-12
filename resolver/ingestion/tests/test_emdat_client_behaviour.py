import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest
import yaml

from resolver.ingestion import emdat_client


@pytest.fixture(autouse=True)
def _clear_emdat_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RESOLVER_SKIP_EMDAT", raising=False)
    monkeypatch.delenv("EMDAT_NETWORK", raising=False)
    monkeypatch.delenv("EMPTY_POLICY", raising=False)


def _read_header(path: Path) -> List[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return next(csv.reader(handle))


def _load_manifest(path: Path) -> Dict[str, Any]:
    manifest_path = path.with_suffix(path.suffix + ".meta.json")
    assert manifest_path.exists(), "Expected manifest to be written"
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_offline_mode_writes_schema_header(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out_path = tmp_path / "emdat_pa.csv"
    monkeypatch.setattr(emdat_client, "OUT_PATH", out_path)
    monkeypatch.setattr(emdat_client, "OUT_DIR", out_path.parent)

    result = emdat_client.main()

    assert result is True
    assert _read_header(out_path) == emdat_client.CANONICAL_HEADERS

    manifest = _load_manifest(out_path)
    schema = yaml.safe_load(Path(emdat_client.SCHEMA_PATH).read_text(encoding="utf-8"))
    assert manifest.get("schema_version") == str(schema.get("version", "1"))
    assert manifest.get("source_id") == "emdat"
    assert manifest.get("row_count") == 0


def test_live_mode_empty_window_allowed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out_path = tmp_path / "emdat_pa.csv"
    monkeypatch.setattr(emdat_client, "OUT_PATH", out_path)
    monkeypatch.setattr(emdat_client, "OUT_DIR", out_path.parent)
    monkeypatch.setenv("EMDAT_NETWORK", "1")
    monkeypatch.setenv("EMPTY_POLICY", "1")

    monkeypatch.setattr(emdat_client, "load_config", lambda: {"sources": [object()]})
    monkeypatch.setattr(emdat_client, "collect_rows", lambda cfg: [])

    calls: Dict[str, Any] = {}

    def _start(connector_id: str, mode: str) -> Dict[str, Any]:
        calls.setdefault("start", []).append((connector_id, mode))
        return {"connector_id": connector_id, "mode": mode}

    def _finalize(context: Dict[str, Any], **kwargs: Any) -> Any:
        calls.setdefault("finalize", []).append({"context": context, "kwargs": kwargs})
        return {"status": kwargs.get("status")}

    monkeypatch.setattr(emdat_client.diagnostics_emitter, "start_run", _start)
    monkeypatch.setattr(emdat_client.diagnostics_emitter, "finalize_run", _finalize)

    result = emdat_client.main()

    assert result is True
    assert _read_header(out_path) == emdat_client.CANONICAL_HEADERS
    manifest = _load_manifest(out_path)
    assert manifest.get("row_count") == 0

    assert calls.get("start") == [("emdat_client", "live")]
    finalize = calls.get("finalize")
    assert finalize and finalize[0]["kwargs"]["status"] == "ok"
    assert finalize[0]["kwargs"].get("counts") == {
        "fetched": 0,
        "normalized": 0,
        "written": 0,
        "rows": 0,
    }


def test_live_mode_writes_rows_and_finalizes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out_path = tmp_path / "emdat_pa.csv"
    monkeypatch.setattr(emdat_client, "OUT_PATH", out_path)
    monkeypatch.setattr(emdat_client, "OUT_DIR", out_path.parent)
    monkeypatch.setenv("EMDAT_NETWORK", "1")

    monkeypatch.setattr(emdat_client, "load_config", lambda: {"sources": [object()]})

    row = {
        "event_id": "TST-001",
        "country_name": "Testland",
        "iso3": "TST",
        "hazard_code": "FL",
        "hazard_label": "Flood",
        "hazard_class": "natural",
        "metric": "total_affected",
        "series_semantics": "incident",
        "value": 42,
        "unit": "persons",
        "as_of_date": "2024-02",
        "publication_date": "2024-02-10",
        "publisher": "CRED/EM-DAT",
        "source_type": "other",
        "source_url": "https://example.test/emdat",
        "doc_title": "Event report",
        "definition_text": "Test definition",
        "method": "linear",
        "confidence": "med",
        "revision": 0,
        "ingested_at": "2024-02-15T00:00:00Z",
    }
    monkeypatch.setattr(emdat_client, "collect_rows", lambda cfg: [row])

    calls: Dict[str, Any] = {}

    def _start(connector_id: str, mode: str) -> Dict[str, Any]:
        calls.setdefault("start", []).append((connector_id, mode))
        return {"connector_id": connector_id, "mode": mode}

    def _finalize(context: Dict[str, Any], **kwargs: Any) -> Any:
        calls.setdefault("finalize", []).append({"context": context, "kwargs": kwargs})
        return {"status": kwargs.get("status")}

    monkeypatch.setattr(emdat_client.diagnostics_emitter, "start_run", _start)
    monkeypatch.setattr(emdat_client.diagnostics_emitter, "finalize_run", _finalize)

    result = emdat_client.main()

    assert result is True
    header = _read_header(out_path)
    assert header == emdat_client.CANONICAL_HEADERS
    rows = list(csv.DictReader(out_path.open(newline="", encoding="utf-8")))
    assert rows and rows[0]["event_id"] == "TST-001"

    manifest = _load_manifest(out_path)
    assert manifest.get("row_count") == 1

    assert calls.get("start") == [("emdat_client", "live")]
    finalize = calls.get("finalize")
    assert finalize and finalize[0]["kwargs"]["status"] == "ok"
    assert finalize[0]["kwargs"].get("counts") == {
        "fetched": 1,
        "normalized": 1,
        "written": 1,
        "rows": 1,
    }
