import argparse
import os
import zipfile
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output zip path")
    ap.add_argument("inputs", nargs="+", help="Input zip files")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for inp in args.inputs:
            stem = Path(inp).stem.replace(" ", "_")
            with zipfile.ZipFile(inp) as zin:
                for name in zin.namelist():
                    if name.endswith("/"):
                        continue

                    base = os.path.basename(name)
                    out_name = f"{stem}__{base}" if base else f"{stem}__file"
                    data = zin.read(name)
                    zout.writestr(out_name, data)

    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
