from pathlib import Path


def test_about_page_exists_and_is_linked() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    about_page = repo_root / "web" / "src" / "app" / "about" / "page.tsx"
    nav_file = repo_root / "web" / "src" / "components" / "Nav.tsx"

    assert about_page.exists()
    about_contents = about_page.read_text(encoding="utf-8")
    assert "const ABOUT_MD" in about_contents
    assert "# Welcome!" in about_contents

    nav_contents = nav_file.read_text(encoding="utf-8")
    assert 'href="/about"' in nav_contents
