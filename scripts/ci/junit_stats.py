#!/usr/bin/env python3
"""Extract high level statistics from a pytest JUnit XML report."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict
import xml.etree.ElementTree as ET


def _count_nodes(testcases: list[ET.Element]) -> Dict[str, int]:
    failures = errors = skipped = 0
    for case in testcases:
        failures += len(case.findall("failure"))
        errors += len(case.findall("error"))
        skipped += len(case.findall("skipped"))
    return {"failures": failures, "errors": errors, "skipped": skipped}


def parse_junit(path: Path) -> Dict[str, int]:
    tree = ET.parse(path)
    root = tree.getroot()

    if root.tag == "testsuite":
        suites = [root]
    elif root.tag == "testsuites":
        suites = [elem for elem in root if elem.tag == "testsuite"]
    else:
        suites = root.findall(".//testsuite") or [root]

    totals = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}

    for suite in suites:
        testcases = suite.findall("testcase")
        case_counts = _count_nodes(testcases)

        totals["tests"] += int(suite.attrib.get("tests", len(testcases)))
        totals["failures"] += int(suite.attrib.get("failures", case_counts["failures"]))
        totals["errors"] += int(suite.attrib.get("errors", case_counts["errors"]))
        totals["skipped"] += int(suite.attrib.get("skipped", case_counts["skipped"]))

    totals["passed"] = max(totals["tests"] - totals["failures"] - totals["errors"] - totals["skipped"], 0)
    return totals


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarise a pytest JUnit XML report")
    parser.add_argument("--in", dest="path", required=True, help="Path to the junit xml file")
    args = parser.parse_args(argv)

    junit_path = Path(args.path)
    if not junit_path.exists():
        raise SystemExit(f"JUnit file not found: {junit_path}")

    stats = parse_junit(junit_path)
    json.dump(stats, fp=sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main())
