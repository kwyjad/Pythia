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

# Natural Earth ADM0_A3 → ISO 3166-1 alpha-3 corrections.
# regionmask .abbrev uses Natural Earth abbreviations which differ from
# ISO3 for several countries.
_NE_TO_ISO3: dict[str, str] = {
    # Natural Earth ADM0_A3 mismatches
    "DRC": "COD",  # DR Congo
    "PAL": "PSE",  # Palestine
    "SLO": "SVN",  # Slovenia
    "BOS": "BIH",  # Bosnia (sometimes)
    "CIS": "CIV",  # Côte d'Ivoire (sometimes)
    "KOS": "XKX",  # Kosovo
    "SDS": "SSD",  # South Sudan
    "SOL": "SOM",  # Somaliland -> Somalia
    "CYN": "CYP",  # Northern Cyprus -> Cyprus
    "SAH": "ESH",  # Western Sahara
    "TAI": "TWN",  # Taiwan (sometimes TWN, sometimes TAI in NE)
    # regionmask short / non-standard abbreviations
    "INDO": "IDN",  # Indonesia
    "MY": "MYS",    # Malaysia
    "CL": "CHL",    # Chile
    "BO": "BOL",    # Bolivia
    "PE": "PER",    # Peru
    "AR": "ARG",    # Argentina
    "CY": "CYP",    # Cyprus
    "CN": "CHN",    # China
    "IS": "ISL",    # Iceland
    "LB": "LBN",    # Lebanon
    "ET": "ETH",    # Ethiopia
    "SS": "SSD",    # South Sudan
    "SO": "SOM",    # Somalia
    "KE": "KEN",    # Kenya
    "MW": "MWI",    # Malawi
    "TZ": "TZA",    # Tanzania
    "SL": "SLE",    # Sierra Leone
    "F": "FRA",     # France
    "SR": "SUR",    # Suriname
    "GY": "GUY",    # Guyana
}


@functools.lru_cache(maxsize=1)
def _load_name_to_iso3() -> dict[str, str]:
    """Load country name -> ISO3 mapping from countries.csv."""
    csv_path = Path(__file__).resolve().parents[1] / "data" / "countries.csv"
    mapping: dict[str, str] = {}
    try:
        with csv_path.open("r", encoding="utf-8-sig") as fh:
            for row in _csv.DictReader(fh):
                name = (row.get("country_name") or "").strip()
                iso3 = (row.get("iso3") or "").strip().upper()
                if name and iso3:
                    mapping[name.lower()] = iso3
    except Exception:
        log.warning("Failed to load countries.csv from %s", csv_path)
    return mapping


def _country_name_to_iso3(name: str) -> str:
    """Resolve a country name to ISO3 using countries.csv."""
    if not name:
        return ""
    mapping = _load_name_to_iso3()
    return mapping.get(name.lower().strip(), "")

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
    """Download ENSMEAN NetCDF files — one multi-lead file per variable.

    CPC now publishes a single file per variable containing all lead
    months as a dimension:
        ``NMME.tmp2m.{ym}.ENSMEAN.anom.nc``
        ``NMME.prate.{ym}.ENSMEAN.anom.nc``

    Returns a list of dicts ``{path, variable}`` for each successfully
    downloaded file.  (Lead months are inside the file, not separate.)
    """
    variables = variables or VARIABLES
    downloaded: list[dict] = []

    with FTP(FTP_HOST) as ftp:
        ftp.login()
        remote_dir = f"{FTP_BASE}/{issue_dir}"
        ftp.cwd(remote_dir)
        remote_files = set(ftp.nlst())

        ym_part = issue_dir[:6]  # "202603" from "2026030800"

        for var_key in variables:
            # New CPC naming: NMME.tmp2m.202603.ENSMEAN.anom.nc
            fname = f"NMME.{var_key}.{ym_part}.ENSMEAN.anom.nc"

            if fname not in remote_files:
                log.warning(
                    "NMME filename miss for %s (expected %s) — remote files: %s",
                    var_key, fname, sorted(remote_files),
                )

                # Fuzzy fallback: match on var_key + ENSMEAN + anom
                var_lower = var_key.lower()
                candidates = [
                    f for f in remote_files
                    if var_lower in f.lower()
                    and "ensmean" in f.lower()
                    and "anom" in f.lower()
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

            downloaded.append({"path": local_path, "variable": var_key})
            log.debug("Downloaded %s → %s", fname, local_path)

    log.info(
        "Downloaded %d NMME files from %s/%s",
        len(downloaded), FTP_BASE, issue_dir,
    )
    return downloaded


# ------------------------------------------------------------------
# Spatial aggregation
# ------------------------------------------------------------------

def _get_country_regions():
    """Return the highest-resolution regionmask country regions available."""
    import regionmask
    try:
        return regionmask.defined_regions.natural_earth_v5_1_2.countries_10
    except AttributeError:
        return regionmask.defined_regions.natural_earth_v5_0_0.countries_10


def _aggregate_2d_field_to_countries(da, countries, mask) -> pd.DataFrame:
    """Compute area-weighted country averages from a 2-D (lat × lon) field.

    Parameters
    ----------
    da : xarray.DataArray with only ``lat`` and ``lon`` dimensions
    countries : regionmask region object
    mask : pre-computed regionmask mask array

    Returns
    -------
    DataFrame with columns ``[iso3, anomaly_value]``
    """
    weights = np.cos(np.deg2rad(da.lat))

    rows: list[dict] = []
    for region_number in np.unique(mask.values[~np.isnan(mask.values)]):
        region_number = int(region_number)
        region_mask = mask == region_number
        region_data = da.where(region_mask)

        # Weighted mean: sum(data * weight) / sum(weight)
        weighted_sum = (region_data * weights).sum(skipna=True)
        weight_sum = (weights * region_mask.astype(float)).sum(skipna=True)

        if float(weight_sum) == 0:
            continue

        mean_val = float(weighted_sum / weight_sum)

        try:
            region_obj = countries[region_number]
            iso3 = region_obj.abbrev
        except (KeyError, IndexError):
            continue

        if not iso3 or len(iso3) != 3:
            continue

        rows.append({"iso3": iso3.upper(), "anomaly_value": round(mean_val, 4)})

    log.debug("[nmme] resolved %d countries from mask", len(rows))
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["iso3", "anomaly_value"])


def _prepare_data_array(ds):
    """Extract the data variable and normalise coordinates.

    Returns the prepared DataArray (may have a lead dimension).
    """
    data_vars = list(ds.data_vars)
    if not data_vars:
        return None
    da = ds[data_vars[0]]

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

    return da


def _find_lead_dim(da):
    """Identify the lead-time dimension name, if present."""
    spatial = {"lat", "lon"}
    for dim in da.dims:
        if dim not in spatial:
            return dim
    return None


def _aggregate_nc_to_countries(
    nc_path: Path,
    variable: str,
) -> pd.DataFrame:
    """Compute area-weighted country averages from a single NetCDF file.

    Handles both legacy single-lead files and new multi-lead files
    (squeezes singleton non-spatial dims).

    Returns DataFrame with columns ``[iso3, anomaly_value]``.
    """
    import xarray as xr

    ds = xr.open_dataset(nc_path, decode_times=False)
    da = _prepare_data_array(ds)
    if da is None:
        log.warning("No data variables in %s", nc_path)
        ds.close()
        return pd.DataFrame(columns=["iso3", "anomaly_value"])

    # Squeeze out any singleton non-spatial dimensions.
    for dim in list(da.dims):
        if dim not in ("lat", "lon") and da.sizes[dim] == 1:
            da = da.squeeze(dim, drop=True)

    countries = _get_country_regions()
    mask = countries.mask(da)
    result = _aggregate_2d_field_to_countries(da, countries, mask)
    ds.close()
    return result


def _aggregate_multi_lead_nc(
    nc_path: Path,
    variable: str,
    max_leads: int = MAX_LEAD_MONTHS,
) -> list[tuple[int, pd.DataFrame]]:
    """Aggregate a multi-lead NetCDF file to per-country, per-lead DataFrames.

    Parameters
    ----------
    nc_path : path to the multi-lead ``.nc`` file
    variable : variable key (``"tmp2m"`` or ``"prate"``)
    max_leads : maximum number of lead months to extract

    Returns
    -------
    List of ``(lead_month, DataFrame)`` tuples where each DataFrame has
    columns ``[iso3, anomaly_value]``.  ``lead_month`` is 1-based.
    """
    import xarray as xr

    ds = xr.open_dataset(nc_path, decode_times=False)
    da = _prepare_data_array(ds)
    if da is None:
        log.warning("No data variables in %s", nc_path)
        ds.close()
        return []

    # Squeeze out singleton time dimensions but keep the lead dimension.
    lead_dim = _find_lead_dim(da)
    for dim in list(da.dims):
        if dim not in ("lat", "lon") and dim != lead_dim and da.sizes[dim] == 1:
            da = da.squeeze(dim, drop=True)

    # Re-check for lead dim after squeezing.
    lead_dim = _find_lead_dim(da)

    if lead_dim is None:
        # Fallback: file has no lead dimension (single-lead legacy file).
        countries = _get_country_regions()
        mask = countries.mask(da)
        df = _aggregate_2d_field_to_countries(da, countries, mask)
        ds.close()
        return [(1, df)] if not df.empty else []

    n_leads = min(da.sizes[lead_dim], max_leads)
    log.info(
        "NMME multi-lead file %s: %d leads on dim '%s'",
        nc_path.name, da.sizes[lead_dim], lead_dim,
    )

    countries = _get_country_regions()

    # Build regionmask once from a single 2-D slice.
    da_slice0 = da.isel({lead_dim: 0})
    mask = countries.mask(da_slice0)

    results: list[tuple[int, pd.DataFrame]] = []
    for i in range(n_leads):
        da_slice = da.isel({lead_dim: i})
        df = _aggregate_2d_field_to_countries(da_slice, countries, mask)
        if not df.empty:
            results.append((i + 1, df))  # 1-based lead month

    ds.close()
    return results


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
    #    Each file may contain multiple lead months as a dimension.
    all_rows: list[pd.DataFrame] = []
    for entry in files:
        lead_results = _aggregate_multi_lead_nc(
            entry["path"], entry["variable"], max_leads=max_leads,
        )
        for lead_month, df in lead_results:
            if df.empty:
                continue
            df["variable"] = entry["variable"]
            df["lead_months"] = lead_month
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
