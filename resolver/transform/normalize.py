"""CLI entrypoint for canonical normalization."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

from .adapters.base import CANONICAL_COLUMNS, BaseAdapter
from .adapters.ifrc import IFRCAdapter

LOGGER = logging.getLogger(__name__)

ADAPTER_REGISTRY: dict[str, type[BaseAdapter]] = {
    "ifrc_go": IFRCAdapter,
    "ifrc": IFRCAdapter,
}


def _parse_sources(raw: str | Iterable[str] | None) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        items = [part.strip() for part in raw.split(",") if part.strip()]
        return items
    return [item.strip() for item in raw if item.strip()]


def _build_adapter(name: str) -> BaseAdapter:
    cls = ADAPTER_REGISTRY.get(name)
    if not cls:
        raise ValueError(f"Unknown source '{name}'. Available: {sorted(ADAPTER_REGISTRY)}")
    return cls(name)



def _write_canonical(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if frame.empty:
        LOGGER.info("%s: writing header-only canonical CSV", output_path.name)
        pd.DataFrame(columns=CANONICAL_COLUMNS).to_csv(output_path, index=False)
        return
    frame.loc[:, CANONICAL_COLUMNS].to_csv(output_path, index=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Normalize staging CSVs to the canonical schema")
    parser.add_argument("--in", dest="input_dir", required=True, help="Directory containing raw staging CSVs")
    parser.add_argument("--out", dest="output_dir", required=True, help="Directory for canonical CSV outputs")
    parser.add_argument("--period", dest="period", help="Optional period label for logging", default=None)
    parser.add_argument("--sources", dest="sources", required=True, help="Comma-separated list of sources to normalize")
    parser.add_argument("--log-level", dest="log_level", default="INFO", help="Python logging level (default: INFO)")

    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(name)s %(message)s")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    sources = _parse_sources(args.sources)

    if not sources:
        parser.error("--sources must contain at least one entry")

    if not input_dir.exists():
        parser.error(f"input directory {input_dir} does not exist")

    if args.period:
        LOGGER.info("Normalizing period %s", args.period)

    for source in sources:
        adapter = _build_adapter(source)
        try:
            canonical = adapter.normalize(input_dir)
        except Exception:
            LOGGER.exception("%s: normalization failed", source)
            raise

        output_path = output_dir / f"{source}.csv"
        LOGGER.info("%s: writing %s rows to %s", source, len(canonical), output_path)
        _write_canonical(canonical, output_path)

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
