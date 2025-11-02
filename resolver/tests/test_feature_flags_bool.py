from resolver.ingestion._shared.feature_flags import as_bool, getenv_bool


def test_as_bool_variants():
    assert as_bool("1") is True
    assert as_bool("0") is False
    assert as_bool("TRUE") is True
    assert as_bool("False") is False
    assert as_bool("yes") is True
    assert as_bool("no") is False
    assert as_bool(None, default=True) is True
    assert as_bool("unexpected", default=True) is True
    assert as_bool("unexpected", default=False) is True


def test_getenv_bool(monkeypatch):
    monkeypatch.setenv("FLAGX", "0")
    assert getenv_bool("FLAGX", default=True) is False
    monkeypatch.setenv("FLAGX", "1")
    assert getenv_bool("FLAGX") is True
    monkeypatch.delenv("FLAGX", raising=False)
    assert getenv_bool("FLAGX", default=False) is False
