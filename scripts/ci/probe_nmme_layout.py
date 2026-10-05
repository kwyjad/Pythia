# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""What CPC serves beside the NMME ensemble-mean anomaly (read-only probe).

The drought gate and the prompts read NMME precipitation as an absolute
anomaly in mm/day, and an absolute anomaly cannot mean the same thing in the
Sahel dry season and a Central American wet season. A relative measure
(percent of normal, or a probability of below-normal) needs either the model
climatology or CPC's own probabilities, and the build sandbox cannot reach
ftp.cpc.ncep.noaa.gov to see which exist. This lists the NMME tree two
levels deep, names every file that looks like a climatology or a
probability product, and opens the first precipitation candidate of each
kind to print its variables, dimensions and attributes. It writes
``diagnostics/nmme_layout.json`` and always exits 0: "the route is not
there" is a finding, not a build failure.
"""

from __future__ import annotations

import json
import sys
import tempfile
from ftplib import FTP, error_perm
from pathlib import Path

HOST = "ftp.cpc.ncep.noaa.gov"
ROOT = "/NMME"
OUT = Path("diagnostics/nmme_layout.json")
KINDS = {
    "climatology": ("clim",),
    "probability": ("prob",),
    "full_field": ("fcst", "forecast", "ensmean.nc", "total"),
}
MAX_LIST = 400


def _nlst(ftp: FTP, path: str) -> list[str]:
    try:
        return sorted(ftp.nlst(path))
    except error_perm as exc:
        return [f"<error: {exc}>"]


def _is_dir(ftp: FTP, path: str) -> bool:
    here = ftp.pwd()
    try:
        ftp.cwd(path)
        return True
    except error_perm:
        return False
    finally:
        ftp.cwd(here)


def _describe(ftp: FTP, path: str) -> dict:
    """Download one NetCDF file and print what it carries."""
    out: dict = {"path": path}
    try:
        import xarray as xr
    except ImportError as exc:  # pragma: no cover - CI installs xarray
        out["error"] = f"xarray unavailable: {exc}"
        return out
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / Path(path).name
        try:
            with open(local, "wb") as fh:
                ftp.retrbinary(f"RETR {path}", fh.write)
            out["bytes"] = local.stat().st_size
            ds = xr.open_dataset(local, decode_times=False)
            out["dims"] = {k: int(v) for k, v in ds.sizes.items()}
            out["variables"] = {
                name: {
                    "dims": list(var.dims),
                    "attrs": {k: str(v) for k, v in var.attrs.items()},
                }
                for name, var in ds.variables.items()
            }
            out["global_attrs"] = {k: str(v) for k, v in ds.attrs.items()}
            for name, var in ds.data_vars.items():
                try:
                    out.setdefault("ranges", {})[name] = [
                        float(var.min()), float(var.max()), float(var.mean()),
                    ]
                except Exception:  # noqa: BLE001
                    pass
            ds.close()
        except Exception as exc:  # noqa: BLE001 - a finding, not a failure
            out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def main() -> int:
    report: dict = {"host": HOST, "root": ROOT, "tree": {}, "candidates": {}, "described": []}
    try:
        with FTP(HOST, timeout=120) as ftp:
            ftp.login()
            top = _nlst(ftp, ROOT)
            report["tree"][ROOT] = top
            for entry in top[:MAX_LIST]:
                path = entry if entry.startswith("/") else f"{ROOT}/{entry}"
                if not _is_dir(ftp, path):
                    continue
                children = _nlst(ftp, path)
                report["tree"][path] = children[:MAX_LIST]
                for child in children[:60]:
                    cpath = child if child.startswith("/") else f"{path}/{child}"
                    low = cpath.lower()
                    if any(k in low for k in ("clim", "prob")) and _is_dir(ftp, cpath):
                        report["tree"][cpath] = _nlst(ftp, cpath)[:MAX_LIST]
            for path, names in report["tree"].items():
                for name in names:
                    full = name if name.startswith("/") else f"{path}/{name}"
                    low = full.lower()
                    for kind, keys in KINDS.items():
                        if any(k in low for k in keys) and low.endswith((".nc", ".grb", ".grib", ".grb2")):
                            report["candidates"].setdefault(kind, []).append(full)
            for kind, files in report["candidates"].items():
                prate = [f for f in files if "prate" in f.lower() or "precip" in f.lower()]
                for path in (prate or files)[:2]:
                    if path.lower().endswith(".nc"):
                        report["described"].append({"kind": kind, **_describe(ftp, path)})
    except Exception as exc:  # noqa: BLE001 - the probe always reports
        report["error"] = f"{type(exc).__name__}: {exc}"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True)[:60000])
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
