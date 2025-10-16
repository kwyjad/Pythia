from __future__ import annotations

from pathlib import Path

import pytest

from resolver.ingestion import run_all_stubs


def _make_spec(*, tmp_path: Path, config: dict | None = None) -> run_all_stubs.ConnectorSpec:
    path = tmp_path / "demo_client.py"
    path.touch()
    config_path = tmp_path / "demo.yml"
    if config is None:
        config_path = None
    return run_all_stubs.ConnectorSpec(
        filename="demo_client.py",
        path=path,
        kind="real",
        output_path=None,
        summary=None,
        skip_reason=None,
        metadata={},
        config_path=config_path,
        config=config or {},
        canonical_name="demo",
        ci_gate_reason=None,
    )


def _prepare_runner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RESOLVER_STAGING_DIR", str(tmp_path / "staging"))
    monkeypatch.setenv("RUNNER_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.delenv("RESOLVER_FORCE_ENABLE", raising=False)


def _stub_specs(monkeypatch: pytest.MonkeyPatch, spec: run_all_stubs.ConnectorSpec) -> None:
    def fake_build_specs(real, stubs, selected, run_real, run_stubs, **kwargs):  # noqa: ANN001
        return [spec]

    monkeypatch.setattr(run_all_stubs, "_build_specs", fake_build_specs)


def test_enabled_flag_false_skips_connector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    spec = _make_spec(tmp_path=tmp_path, config={"enabled": False})
    _stub_specs(monkeypatch, spec)
    _prepare_runner(monkeypatch, tmp_path)

    call_count = 0

    def fake_run(connector, logger):  # noqa: ANN001
        nonlocal call_count
        call_count += 1
        return {"rows": 0, "rows_method": ""}

    monkeypatch.setattr(run_all_stubs, "_run_connector", fake_run)

    exit_code = run_all_stubs.main([])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "decision=skip" in captured.out
    assert "enable=False gated_by=config" in captured.out
    assert "status=skipped" in captured.out
    assert "notes=disabled: config" in captured.out
    assert call_count == 0


def test_legacy_enable_flag_false_skips_connector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    spec = _make_spec(tmp_path=tmp_path, config={"enable": False})
    _stub_specs(monkeypatch, spec)
    _prepare_runner(monkeypatch, tmp_path)
    call_count = 0

    def fake_run(connector, logger):  # noqa: ANN001
        nonlocal call_count
        call_count += 1
        return {"rows": 0, "rows_method": ""}

    monkeypatch.setattr(run_all_stubs, "_run_connector", fake_run)

    exit_code = run_all_stubs.main([])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "enable=False gated_by=config" in captured.out
    assert "status=skipped" in captured.out
    assert "notes=disabled: config" in captured.out
    assert call_count == 0


def test_missing_flag_defaults_to_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    spec = _make_spec(tmp_path=tmp_path, config={})
    _stub_specs(monkeypatch, spec)
    _prepare_runner(monkeypatch, tmp_path)
    call_count = 0

    def fake_run(connector, logger):  # noqa: ANN001
        nonlocal call_count
        call_count += 1
        return {"rows": 0, "rows_method": ""}

    monkeypatch.setattr(run_all_stubs, "_run_connector", fake_run)

    exit_code = run_all_stubs.main([])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "enable=False gated_by=config" in captured.out
    assert "status=skipped" in captured.out
    assert "notes=disabled: config" in captured.out
    assert call_count == 0


def test_force_enable_runs_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    spec = _make_spec(tmp_path=tmp_path, config={"enabled": False})
    _stub_specs(monkeypatch, spec)
    _prepare_runner(monkeypatch, tmp_path)
    monkeypatch.setenv("RESOLVER_FORCE_ENABLE", "demo")

    call_count = 0

    def fake_run(connector, logger):  # noqa: ANN001
        nonlocal call_count
        call_count += 1
        return {"rows": 5, "rows_method": ""}

    monkeypatch.setattr(run_all_stubs, "_run_connector", fake_run)

    exit_code = run_all_stubs.main([])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "enable=True gated_by=forced_by_env" in captured.out
    assert "forced_by=env" in captured.out
    assert "status=ok" in captured.out
    assert "notes=forced_by_env" in captured.out
    assert call_count == 1


def test_real_mode_list_is_authoritative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    spec = _make_spec(tmp_path=tmp_path, config={"enabled": False})
    spec.origin = "real_list"
    spec.authoritatively_selected = True
    _stub_specs(monkeypatch, spec)
    _prepare_runner(monkeypatch, tmp_path)

    call_count = 0

    def fake_run(connector, logger):  # noqa: ANN001
        nonlocal call_count
        call_count += 1
        return {"rows": 3, "rows_method": ""}

    monkeypatch.setattr(run_all_stubs, "_run_connector", fake_run)

    exit_code = run_all_stubs.main(["--mode", "real"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "decision=run" in captured.out
    assert "gated_by=selected:list" in captured.out
    assert "origin=real_list" in captured.out
    assert "enable=True gated_by=selected:list" in captured.out
    assert "notes=selected:list" in captured.out
    assert call_count == 1
