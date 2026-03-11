# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""NMME seasonal forecast ingestion from CPC FTP.

Downloads ENSMEAN anomaly NetCDF files from the CPC NMME FTP server,
computes area-weighted country-level averages using regionmask, and
returns a DataFrame ready for DuckDB upsert.

Source: ftp://ftp.cpc.ncep.noaa.gov/NMME/realtime_anom/ENSMEAN/
Updated ~9th of each month with 7 lead months of temperature and
precipitation anomalies at 1° × 1° resolution.
"""

from __future__ import annotations

import logging
import tempfile
from datetime import date, datetime
from ftplib import FTP
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

FTP_HOST = "ftp.cpc.ncep.noaa.gov"
FTP_BASE = "/NMME/realtime_anom/ENSMEAN"

# Variable names in the NetCDF files and the CPC filename prefixes.
VARIABLES = {
    "tmp2m": "tmp2m_anom",   # 2-metre temperature anomaly
    "prate": "prate_anom",   # precipitation rate anomaly
}

# Maximum lead months available in the NMME ENSMEAN product.
MAX_LEAD_MONTHS = 7

# Tercile thresholds (σ).  ±0.43 σ ≈ 33rd / 67th percentile of a
# standard normal distribution, which is the conventional split for
# NMME anomaly terciles.
TERCILE_UPPER = 0.43
TERCILE_LOWER = -0.43


# ------------------------------------------------------------------
# FTP helpers
# ------------------------------------------------------------------

def _discover_issue_dir(ftp: FTP, year_month: Optional[str] = None) -> str:
    """Return the FTP directory name for the latest (or given) issue month.

    Parameters
    ----------
    ftp : connected FTP instance
    year_month : optional ``YYYYMM`` string.  If *None*, tries the
        current month first, then falls back to the previous month.

    Returns
    -------
    Directory name like ``"2026030800"``

    Raises
    ------
    FileNotFoundError
        If no matching directory is found.
    """
    ftp.cwd(FTP_BASE)
    available = sorted(ftp.nlst())

    if year_month:
        prefix = f"{year_month}0800"
        if prefix in available:
            return prefix
        raise FileNotFoundError(
            f"NMME directory {prefix}/ not found on CPC FTP.  "
            f"Available: {available[-5:]}"
        )

    # Auto-detect: try current month, then previous.
    today = date.today()
    for offset in (0, 1):
        m = today.month - offset
        y = today.year
        if m < 1:
            m += 12
            y -= 1
        prefix = f"{y}{m:02d}0800"
        if prefix in available:
            return prefix

    raise FileNotFoundError(
        f"No recent NMME directory found on CPC FTP.  "
        f"Latest available: {available[-3:]}"
    )


def _download_nc_files(
    issue_dir: str,
    dest: Path,
    *,
    variables: dict[str, str] | None = None,
    max_leads: int = MAX_LEAD_MONTHS,
) -> list[dict]:
    """Download ENSMEAN NetCDF files for all variables × leads.

    Returns a list of dicts ``{path, variable, lead}`` for each
    successfully downloaded file.
    """
    variables = variables or VARIABLES
    downloaded: list[dict] = []

    with FTP(FTP_HOST) as ftp:
        ftp.login()
        remote_dir = f"{FTP_BASE}/{issue_dir}"
        ftp.cwd(remote_dir)
        remote_files = set(ftp.nlst())

        logged_vars: set = set()
        for var_key, fname_prefix in variables.items():
            for lead in range(1, max_leads + 1):
                # CPC naming: e.g.  tmp2m_anom.ENSMEAN.202603.mon1.nc
                ym_part = issue_dir[:6]  # "202603" from "2026030800"
                fname = f"{fname_prefix}.ENSMEAN.{ym_part}.mon{lead}.nc"

                if fname not in remote_files:
                    if var_key not in logged_vars:
                        log.warning(
                            "NMME filename miss for %s — remote files: %s",
                            var_key, sorted(remote_files),
                        )
                        logged_vars.add(var_key)

                    # Fuzzy fallback: match on prefix + lead indicator
                    prefix_lower = fname_prefix.lower()
                    lead_patterns = [f"mon{lead}", f"lead{lead}"]
                    candidates = [
                        f for f in remote_files
                        if prefix_lower in f.lower()
                        and any(lp in f.lower() for lp in lead_patterns)
                    ]
                    if len(candidates) == 1:
                        fname = candidates[0]
                        log.info("Fuzzy-matched NMME file: %s", fname)
                    else:
                        log.warning("File not found on FTP: %s/%s", remote_dir, fname)
                        continue

                local_path = dest / fname
                with open(local_path, "wb") as fh:
                    ftp.retrbinary(f"RETR {fname}", fh.write)

                downloaded.append(
                    {"path": local_path, "variable": var_key, "lead": lead}
                )
                log.debug("Downloaded %s → %s", fname, local_path)

    log.info(
        "Downloaded %d NMME files from %s/%s",
        len(downloaded), FTP_BASE, issue_dir,
    )
    return downloaded


# ------------------------------------------------------------------
# Spatial aggregation
# ------------------------------------------------------------------

def _aggregate_nc_to_countries(
    nc_path: Path,
    variable: str,
) -> pd.DataFrame:
    """Compute area-weighted country averages from a single NetCDF file.

    Parameters
    ----------
    nc_path : path to the downloaded ``.nc`` file
    variable : variable key (``"tmp2m"`` or ``"prate"``)

    Returns
    -------
    DataFrame with columns ``[iso3, anomaly_value]``
    """
    import regionmask
    import xarray as xr

    ds = xr.open_dataset(nc_path)

    # The anomaly field name inside the file varies; pick the first
    # non-coordinate data variable.
    data_vars = list(ds.data_vars)
    if not data_vars:
        log.warning("No data variables in %s", nc_path)
        return pd.DataFrame(columns=["iso3", "anomaly_value"])
    field = data_vars[0]

    da = ds[field]

    # Squeeze out any singleton dimensions (time, lead, etc.)
    for dim in list(da.dims):
        if dim not in ("lat", "latitude", "lon", "longitude", "Y", "X"):
            if da.sizes[dim] == 1:
                da = da.squeeze(dim, drop=True)

    # Normalise coordinate names to lat/lon.
    rename = {}
    for name in da.dims:
        low = name.lower()
        if low in ("latitude", "y") and name != "lat":
            rename[name] = "lat"
        elif low in ("longitude", "x") and name != "lon":
            rename[name] = "lon"
    if rename:
        da = da.rename(rename)

    # Ensure longitude is in [-180, 180] for regionmask.
    if float(da.lon.max()) > 180:
        da = da.assign_coords(lon=(((da.lon + 180) % 360) - 180))
        da = da.sortby("lon")

    countries = regionmask.defined_regions.natural_earth_v5_0_0.countries_110
    mask = countries.mask(da)

    # Area weights based on latitude (cosine weighting).
    weights = np.cos(np.deg2rad(da.lat))

    rows: list[dict] = []
    for region_number in np.unique(mask.values[~np.isnan(mask.values)]):
        region_number = int(region_number)
        region_mask = mask == region_number
        region_data = da.where(region_mask)
        region_weights = weights.where(region_mask.any(dim="lon"))

        # Weighted mean: sum(data * weight) / sum(weight)
        weighted_sum = (region_data * weights).sum(skipna=True)
        weight_sum = (weights * region_mask.astype(float)).sum(skipna=True)

        if float(weight_sum) == 0:
            continue

        mean_val = float(weighted_sum / weight_sum)

        # Map region number → ISO3 via regionmask's abbreviation.
        try:
            region_obj = countries[region_number]
            iso3 = region_obj.abbrev
        except (KeyError, IndexError):
            continue

        if not iso3 or len(iso3) != 3:
            continue

        rows.append({"iso3": iso3.upper(), "anomaly_value": round(mean_val, 4)})

    ds.close()
    return pd.DataFrame(rows)


def _classify_tercile(anomaly: float) -> str:
    """Return tercile category from anomaly value (σ units)."""
    if anomaly > TERCILE_UPPER:
        return "above_normal"
    if anomaly < TERCILE_LOWER:
        return "below_normal"
    return "near_normal"


# ------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------

def fetch_and_process(
    *,
    year_month: Optional[str] = None,
    max_leads: int = MAX_LEAD_MONTHS,
    dest_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Fetch NMME ENSMEAN data and aggregate to country-level rows.

    Parameters
    ----------
    year_month : optional ``YYYYMM``.  Auto-detects if *None*.
    max_leads : number of lead months to fetch (default 7).
    dest_dir : directory for downloaded files.  Uses a temp dir if *None*.

    Returns
    -------
    DataFrame with columns:
        iso3, variable, lead_months, anomaly_value,
        tercile_category, forecast_issue_date
    """
    use_temp = dest_dir is None
    if use_temp:
        tmpdir = tempfile.mkdtemp(prefix="nmme_")
        dest_dir = Path(tmpdir)
    else:
        dest_dir.mkdir(parents=True, exist_ok=True)

    # 1. Discover the issue directory.
    with FTP(FTP_HOST) as ftp:
        ftp.login()
        issue_dir = _discover_issue_dir(ftp, year_month)
    log.info("Using NMME issue directory: %s", issue_dir)

    # Extract issue date from dir name (YYYYMM0800 → YYYY-MM-08).
    issue_ym = issue_dir[:6]
    forecast_issue_date = datetime.strptime(issue_ym, "%Y%m").date().replace(day=8)

    # 2. Download NetCDF files.
    files = _download_nc_files(issue_dir, dest_dir, max_leads=max_leads)
    if not files:
        log.warning("No NMME files downloaded.")
        return pd.DataFrame(
            columns=[
                "iso3", "variable", "lead_months", "anomaly_value",
                "tercile_category", "forecast_issue_date",
            ]
        )

    # 3. Aggregate each file to country-level averages.
    all_rows: list[pd.DataFrame] = []
    for entry in files:
        df = _aggregate_nc_to_countries(entry["path"], entry["variable"])
        if df.empty:
            continue
        df["variable"] = entry["variable"]
        df["lead_months"] = entry["lead"]
        all_rows.append(df)

    if not all_rows:
        log.warning("No country-level data produced from NMME files.")
        return pd.DataFrame(
            columns=[
                "iso3", "variable", "lead_months", "anomaly_value",
                "tercile_category", "forecast_issue_date",
            ]
        )

    result = pd.concat(all_rows, ignore_index=True)

    # 4. Derive tercile classification.
    result["tercile_category"] = result["anomaly_value"].apply(_classify_tercile)

    # 5. Add metadata.
    result["forecast_issue_date"] = forecast_issue_date

    # Ensure column order.
    result = result[
        [
            "iso3", "variable", "lead_months", "anomaly_value",
            "tercile_category", "forecast_issue_date",
        ]
    ]

    log.info(
        "NMME processing complete: %d rows, issue date %s, "
        "%d countries, %d variables × %d leads",
        len(result),
        forecast_issue_date,
        result["iso3"].nunique(),
        result["variable"].nunique(),
        result["lead_months"].nunique(),
    )
    return result
