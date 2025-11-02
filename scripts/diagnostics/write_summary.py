from __future__ import annotations

import json
import os
import sys

from resolver.diagnostics.summarizer import ConnectorRow, LogsMetaRow, write_summary_md


def main() -> int:
    output_path = os.environ.get("RESOLVER_SUMMARY_PATH", "summary.md")
    dtm_config_path = os.environ.get("DTM_CONFIG_PATH")

    rows_env = os.environ.get("RESOLVER_CONNECTOR_ROWS_JSON", "[]")
    logs_env = os.environ.get("RESOLVER_LOGS_META_JSON", "[]")

    try:
        connector_rows = [ConnectorRow(**row) for row in json.loads(rows_env)]
        logs_meta = [LogsMetaRow(**row) for row in json.loads(logs_env)]
    except Exception:
        connector_rows = []
        logs_meta = []

    write_summary_md(
        output_path,
        connector_rows=connector_rows,
        logs_meta=logs_meta,
        dtm_config_path=dtm_config_path,
    )
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
