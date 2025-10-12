import pathlib
import re

WF_DIR = pathlib.Path(".github/workflows")


def _workflow_texts():
    for path in WF_DIR.glob("*.y*ml"):
        yield path, path.read_text(encoding="utf-8", errors="replace")


def test_no_yaml_list_under_upload_artifact_path():
    offenders = []
    for path, text in _workflow_texts():
        if "uses: actions/upload-artifact@v4" in text:
            if re.search(r"with:\s*[\s\S]*?path:\s*\n\s*-\s", text):
                offenders.append(str(path))
    assert not offenders, (
        "upload-artifact path must be a newline scalar, not a YAML list: " + ", ".join(offenders)
    )


def test_bracket_test_spacing():
    offenders = []
    for path, text in _workflow_texts():
        if re.search(r"\[\[[^\s].*[^\s]\]\]", text):
            offenders.append(str(path))
    assert not offenders, "Missing spaces in '[[ ... ]]' tests: " + ", ".join(offenders)


def test_no_echo_escape_sequences():
    offenders = []
    for path, text in _workflow_texts():
        if re.search(r"echo\s+-e\b", text) or re.search(r'echo\s+".*\\n', text):
            offenders.append(str(path))
    assert not offenders, "Use printf for escapes instead of echo: " + ", ".join(offenders)


def test_no_and_and_or_as_if_else():
    offenders = []
    for path, text in _workflow_texts():
        for line in text.splitlines():
            if "&&" in line and "||" in line and "${{" not in line:
                offenders.append(str(path))
                break
    assert not offenders, "A && B || C found; replace with if/else: " + ", ".join(offenders)
