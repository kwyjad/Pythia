"""Flatten multiple artifact archives into a single zip."""

from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path
from typing import Sequence


def flatten_zips(out_path: Path | str, inputs: Sequence[Path | str]) -> Path:
    """Flatten `inputs` archives into `out_path` and return the resolved path."""

    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for inp in inputs:
            inp_path = Path(inp)
            stem = inp_path.stem.replace(" ", "_")
            with zipfile.ZipFile(inp_path) as zin:
                for name in zin.namelist():
                    if name.endswith("/"):
                        continue

                    base = os.path.basename(name)
                    out_name = f"{stem}__{base}" if base else f"{stem}__file"
                    data = zin.read(name)
                    zout.writestr(out_name, data)

    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Output zip path")
    parser.add_argument("inputs", nargs="+", help="Input zip files")
    args = parser.parse_args()

    out_path = flatten_zips(args.out, args.inputs)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
