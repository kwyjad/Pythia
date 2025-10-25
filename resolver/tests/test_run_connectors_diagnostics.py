from pathlib import Path


def test_run_connectors_bootstraps_diagnostics(tmp_path, monkeypatch):
    from scripts.ci import run_connectors

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_connectors, "_resolve_connectors", lambda env: [])

    rc = run_connectors.main([])
    assert rc == 0

    diag_base = Path(tmp_path) / "diagnostics" / "ingestion"
    expected_dirs = ["logs", "raw", "metrics", "samples", "dtm"]
    for sub in expected_dirs:
        assert (diag_base / sub).is_dir()

    for sub in ("raw", "metrics", "samples"):
        keep = diag_base / sub / ".keep"
        assert keep.is_file()
