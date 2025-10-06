"""Fast smoke test to ensure CI always has at least one test to run."""

import importlib


def test_ci_smoke() -> None:
    """Verify the resolver package can be imported and exposes a package attribute."""
    module = importlib.import_module("resolver")
    assert hasattr(module, "__package__")
