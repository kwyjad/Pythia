from __future__ import annotations

from pathlib import Path

import pytest

from resolver.ingestion import run_all_stubs


def _make_spec(
    *,
    tmp_path: Path,
    config: dict | None = None,
    skip_reason: str | None = None,
    ci_gate_reason: str | None = None,
) -> run_all_stubs.ConnectorSpec:
    path = tmp_path / "demo_client.py"
    path.touch()
    return run_all_stubs.ConnectorSpec(
        filename="demo_client.py",
        path=path,
        kind="real",
        output_path=None,
        summary=None,
        skip_reason=skip_reason,
        metadata={},
        config_path=None,
        config=config or {},
        canonical_name="demo",
        ci_gate_reason=ci_gate_reason,
    )


def test_resolve_enablement_precedence(tmp_path: Path) -> None:
    spec = _make_spec(tmp_path=tmp_path, config={"enabled": False})
    decision = run_all_stubs._resolve_enablement(spec)
    assert not decision.should_run
    assert decision.gated_by == "config_disabled"
    assert decision.applied_skip_reason == "disabled: config"

    forced = run_all_stubs._resolve_enablement(spec, forced_by_env=True)
    assert forced.should_run
    assert forced.gated_by.startswith("forced:")
    assert "env" in forced.forced_sources

    only_forced = run_all_stubs._resolve_enablement(spec, forced_by_only=True)
    assert only_forced.should_run
    assert "only" in only_forced.forced_sources

    pattern_forced = run_all_stubs._resolve_enablement(spec, forced_by_pattern=True)
    assert pattern_forced.should_run
    assert "pattern" in pattern_forced.forced_sources


def test_resolve_enablement_ci_gate_after_config(tmp_path: Path) -> None:
    reason = "RESOLVER_SKIP_DEMO=1 — Demo connector"
    spec = _make_spec(tmp_path=tmp_path, config={"enabled": True}, ci_gate_reason=reason)
    decision = run_all_stubs._resolve_enablement(spec)
    assert not decision.should_run
    assert decision.gated_by == "ci_gate"
    assert decision.applied_skip_reason == reason

    forced = run_all_stubs._resolve_enablement(spec, forced_by_env=True)
    assert forced.should_run
    assert forced.gated_by.startswith("forced:")


def test_enablement_logging_includes_reason(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec = _make_spec(tmp_path=tmp_path, config={"enabled": False})

    def fake_build_specs(real, stubs, selected, run_real, run_stubs):  # noqa: ANN001
        return [spec]

    monkeypatch.setattr(run_all_stubs, "_build_specs", fake_build_specs)
    monkeypatch.setattr(run_all_stubs, "_invoke_connector", lambda *_, **__: None)
    monkeypatch.setenv("RESOLVER_STAGING_DIR", str(tmp_path / "staging"))
    monkeypatch.setenv("RUNNER_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.delenv("RESOLVER_FORCE_ENABLE", raising=False)

    exit_code = run_all_stubs.main([])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "gated_by=config_disabled" in captured.out
    assert "decision=skip" in captured.out
    assert spec.skip_reason == "disabled: config"
