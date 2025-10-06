# ReliefWeb PDF fixtures

Binary PDFs are deliberately **not** stored in the repository.  The ReliefWeb
PDF ingestion tests monkeypatch `_pdf_text.smart_extract` to return deterministic
fixture strings so we can exercise parsing behaviour without bundling large
assets.  When adding new scenarios, keep the test strings in the test module or
store short text snippets in this folder.
