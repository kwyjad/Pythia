"""Ensure the resolver package can be imported after editable installs."""

def test_import_resolver():
    import resolver  # noqa: F401
