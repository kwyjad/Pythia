from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

HEADER_TITLE_1 = "# Connector Diagnostics"
HEADER_TITLE_2 = "## Run Summary"

CONNECTOR_TABLE_HEADER = "| Connector | Mode | Status | Reason |\n|---|---|---|---|"
LOGS_META_HEADER = "| File | Lines | Bytes |\n|---|---:|---:|"


@dataclass
class ConnectorRow:
    name: str
    mode: str
    status: str
    reason: str


@dataclass
class LogsMetaRow:
    file: str
    lines: int
    bytes: int


def build_markdown(
    *,
    connector_rows: Iterable[ConnectorRow] = (),
    logs_meta: Iterable[LogsMetaRow] = (),
    notes: Optional[str] = None,
    dtm_config_path: Optional[str] = None,
) -> str:
    """Build the deterministic markdown summary expected by fast tests."""
    md: List[str] = []
    md.append(HEADER_TITLE_1)
    md.append(HEADER_TITLE_2)
    md.append("")

    if dtm_config_path:
        md.append(f"Config: {dtm_config_path}")
        md.append("")

    md.append("### Connector Diagnostics")
    md.append(CONNECTOR_TABLE_HEADER)
    connector_rows_list = list(connector_rows)
    if connector_rows_list:
        for row in connector_rows_list:
            md.append(f"| {row.name} | {row.mode} | {row.status} | {row.reason} |")
    else:
        md.append("| (none) | (none) | (none) | (none) |")
    md.append("")

    md.append("### Logs (meta)")
    md.append(LOGS_META_HEADER)
    logs_meta_list = list(logs_meta)
    if logs_meta_list:
        for meta in logs_meta_list:
            md.append(f"| {meta.file} | {meta.lines} | {meta.bytes} |")
    else:
        md.append("| (none) | 0 | 0 |")
    md.append("")

    if notes:
        md.append(notes.strip())
        md.append("")

    return "\n".join(md)


def write_summary_md(path: str, **kwargs) -> None:
    content = build_markdown(**kwargs)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)
