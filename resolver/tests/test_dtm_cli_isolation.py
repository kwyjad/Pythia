from pathlib import Path

from resolver.ingestion import dtm_client


def test_main_ignores_pytest_argv_and_writes_header_in_skip(tmp_path, monkeypatch):
    monkeypatch.setenv("RESOLVER_SKIP_DTM", "1")
    out_path = Path(tmp_path) / "dtm.csv"
    monkeypatch.setattr(dtm_client, "OUT_DIR", Path(tmp_path))
    monkeypatch.setattr(dtm_client, "OUT_PATH", out_path)
    monkeypatch.setattr(dtm_client, "DEFAULT_OUTPUT", out_path)
    monkeypatch.setattr(dtm_client, "OUTPUT_PATH", out_path)
    monkeypatch.setattr(
        dtm_client,
        "META_PATH",
        out_path.with_suffix(out_path.suffix + ".meta.json"),
    )
    monkeypatch.setattr(dtm_client, "CONNECTORS_REPORT", Path(tmp_path) / "connectors_report.jsonl")
    monkeypatch.setattr(dtm_client, "RUN_DETAILS_PATH", Path(tmp_path) / "dtm_run.json")
    rc = dtm_client.main()
    assert rc == 0
    text = out_path.read_text().strip().splitlines()
    assert len(text) == 1
    assert text[0] == ",".join(dtm_client.CANONICAL_HEADERS)
    monkeypatch.delenv("RESOLVER_SKIP_DTM", raising=False)
