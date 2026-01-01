# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Build data exports for download endpoints."""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:  # pragma: no cover - silence library default
    LOGGER.addHandler(logging.NullHandler())


_WARNING_FLAGS: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    if key in _WARNING_FLAGS:
        return
    _WARNING_FLAGS.add(key)
    LOGGER.warning(message)


def _table_exists(conn, table: str) -> bool:
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE LOWER(table_name) = LOWER(?)",
            [table],
        ).fetchone()
        return bool(row and row[0])
    except Exception:
        pass

    try:
        df = conn.execute("PRAGMA show_tables").fetchdf()
    except Exception:
        return False
    if df.empty:
        return False
    first_col = df.columns[0]
    return df[first_col].astype(str).str.lower().eq(table.lower()).any()


def _table_columns(conn, table: str) -> set[str]:
    try:
        df = conn.execute(f"PRAGMA table_info('{table}')").fetchdf()
    except Exception:
        return set()
    if df.empty or "name" not in df.columns:
        return set()
    return set(df["name"].astype(str).str.lower().tolist())


def _pick_column(columns: set[str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate.lower() in columns:
            return candidate.lower()
    return None


def _load_country_registry() -> dict[str, str]:
    path = Path(__file__).resolve().parents[1] / "data" / "countries.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype=str).fillna("")
    if "iso3" not in df.columns or "country_name" not in df.columns:
        return {}
    return {
        str(iso3).strip().upper(): str(name).strip()
        for iso3, name in zip(df["iso3"], df["country_name"])
        if str(iso3).strip()
    }


def _add_months(ym: str | None, months: int) -> str | None:
    if not ym:
        return None
    try:
        year_str, month_str = ym.split("-")
        year = int(year_str)
        month = int(month_str)
    except Exception:
        return None
    total = year * 12 + (month - 1) + months
    new_year = total // 12
    new_month = (total % 12) + 1
    return f"{new_year:04d}-{new_month:02d}"


def _parse_run_id_date(run_id: str | None) -> datetime | None:
    if not run_id:
        return None
    match = re.search(r"\b(\d{8})T\d{6}\b", str(run_id))
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%d")
    except Exception:
        return None


def _load_triage_models(con) -> dict[str, str]:
    if con is None or not _table_exists(con, "llm_calls"):
        return {}
    cols = _table_columns(con, "llm_calls")
    if "hs_run_id" not in cols or "phase" not in cols:
        return {}
    model_col = _pick_column(cols, ["model_id", "model_name"])
    if not model_col:
        return {}

    df = con.execute(
        f"""
        SELECT hs_run_id, {model_col} AS model
        FROM llm_calls
        WHERE LOWER(CAST(phase AS VARCHAR)) = 'hs_triage'
          AND {model_col} IS NOT NULL
        """
    ).fetchdf()

    if df.empty:
        return {}

    df["model"] = df["model"].astype(str).str.strip()
    df = df[df["model"] != ""]
    if df.empty:
        return {}

    counts = (
        df.groupby(["hs_run_id", "model"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["hs_run_id", "n", "model"], ascending=[True, False, True])
    )
    top = counts.groupby("hs_run_id", as_index=False).head(1)
    return dict(zip(top["hs_run_id"], top["model"]))


def _standardize_forecasts(conn, table: str) -> pd.DataFrame:
    if not _table_exists(conn, table):
        return pd.DataFrame()

    cols = _table_columns(conn, table)
    question_col = _pick_column(cols, ["question_id"])
    model_col = _pick_column(cols, ["model_name", "model"])
    run_col = _pick_column(cols, ["run_id"])
    created_col = _pick_column(cols, ["created_at", "timestamp"])
    status_col = _pick_column(cols, ["status"])
    prob_col = _pick_column(cols, ["probability", "p", "prob"])
    month_col = _pick_column(cols, ["month_index", "horizon_m"])
    bucket_col = _pick_column(cols, ["bucket_index", "class_bin"])

    if not question_col or not model_col or not prob_col or not month_col or not bucket_col:
        _warn_once(
            f"missing_columns:{table}",
            f"Forecast export skipped for {table}; missing required columns.",
        )
        return pd.DataFrame()

    status_expr = f"{status_col} AS status" if status_col else "NULL AS status"
    run_expr = f"{run_col} AS run_id" if run_col else "NULL AS run_id"
    created_expr = f"{created_col} AS created_at" if created_col else "NULL AS created_at"

    filter_bits = [
        f"{prob_col} IS NOT NULL",
        f"{month_col} IS NOT NULL",
        f"{bucket_col} IS NOT NULL",
    ]
    if status_col:
        filter_bits.append(f"LOWER(CAST({status_col} AS VARCHAR)) = 'ok'")
    filter_expr = " AND ".join(filter_bits)

    sql = f"""
        SELECT
            {question_col} AS question_id,
            {model_col} AS model_name,
            {run_expr},
            {created_expr},
            {month_col} AS month_index,
            {bucket_col} AS bucket_index,
            {prob_col} AS probability,
            {status_expr}
        FROM {table}
        WHERE {filter_expr}
    """
    df = conn.execute(sql).fetchdf()

    if df.empty:
        return df

    if run_col and created_col:
        df["_created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
        latest = (
            df.groupby(["question_id", "model_name"], dropna=False)["_created_at"]
            .transform("max")
            .fillna(pd.NaT)
        )
        df = df[df["_created_at"] == latest].drop(columns=["_created_at"])
    else:
        _warn_once(
            f"latest_run:{table}",
            f"Forecast export for {table} missing run_id/created_at; keeping all rows.",
        )

    return df


def _latest_triage(conn) -> pd.DataFrame:
    if not _table_exists(conn, "hs_triage"):
        return pd.DataFrame()
    cols = _table_columns(conn, "hs_triage")
    if not {"run_id", "iso3", "hazard_code"}.issubset(cols):
        return pd.DataFrame()
    tier_col = _pick_column(cols, ["tier"])
    score_col = _pick_column(cols, ["triage_score"])
    created_col = _pick_column(cols, ["created_at"])

    tier_expr = f"{tier_col} AS tier" if tier_col else "NULL AS tier"
    score_expr = f"{score_col} AS triage_score" if score_col else "NULL AS triage_score"
    created_expr = f"{created_col} AS created_at" if created_col else "NULL AS created_at"

    df = conn.execute(
        f"""
        SELECT
            run_id,
            iso3,
            hazard_code,
            {tier_expr},
            {score_expr},
            {created_expr}
        FROM hs_triage
        """
    ).fetchdf()

    if df.empty:
        return df

    if created_col:
        df["_created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
        latest = (
            df.groupby(["run_id", "iso3", "hazard_code"], dropna=False)["_created_at"]
            .transform("max")
            .fillna(pd.NaT)
        )
        df = df[df["_created_at"] == latest].drop(columns=["_created_at"])

    return df


def build_forecast_spd_export(con) -> pd.DataFrame:
    columns = [
        "ISO",
        "country_name",
        "year",
        "month",
        "forecast_month",
        "metric",
        "hazard",
        "model",
        "SPD_1",
        "SPD_2",
        "SPD_3",
        "SPD_4",
        "SPD_5",
        "EIV",
        "triage_score",
        "triage_tier",
        "hs_run_ID",
    ]

    if con is None or not _table_exists(con, "questions"):
        return pd.DataFrame(columns=columns)

    q_cols = _table_columns(con, "questions")
    required_q = {"question_id", "iso3", "hazard_code", "metric", "target_month", "hs_run_id"}
    if not required_q.issubset(q_cols):
        _warn_once("questions_missing", "Questions table missing required columns for export.")
        return pd.DataFrame(columns=columns)

    questions = con.execute(
        """
        SELECT question_id, iso3, hazard_code, metric, target_month, hs_run_id
        FROM questions
        """
    ).fetchdf()

    forecasts = []
    for table in ("forecasts_ensemble", "forecasts_raw"):
        df = _standardize_forecasts(con, table)
        if not df.empty:
            forecasts.append(df)

    if not forecasts:
        return pd.DataFrame(columns=columns)

    forecasts_df = pd.concat(forecasts, ignore_index=True)

    merged = forecasts_df.merge(questions, on="question_id", how="inner")
    if merged.empty:
        return pd.DataFrame(columns=columns)

    merged["iso3"] = merged["iso3"].astype(str).str.upper()
    merged["hazard_code"] = merged["hazard_code"].astype(str).str.upper()
    merged["metric"] = merged["metric"].astype(str).str.upper()

    bucket_series = pd.to_numeric(merged["bucket_index"], errors="coerce")
    bucket_series = bucket_series.where(~bucket_series.isna(), None)
    if bucket_series.isna().any():
        class_bin = merged["bucket_index"].astype(str).str.strip().str.lower()
        pa_map = {
            "<10k": 1,
            "10k-<50k": 2,
            "50k-<250k": 3,
            "250k-<500k": 4,
            ">=500k": 5,
        }
        fatal_map = {
            "<5": 1,
            "5-<25": 2,
            "25-<100": 3,
            "100-<500": 4,
            ">=500": 5,
        }
        is_fatal = merged["metric"].eq("FATALITIES")
        mapped = class_bin.map(pa_map)
        mapped = mapped.where(~is_fatal, class_bin.map(fatal_map))
        bucket_series = bucket_series.fillna(mapped)

    merged["bucket"] = pd.to_numeric(bucket_series, errors="coerce")
    merged["month_index"] = pd.to_numeric(merged["month_index"], errors="coerce")
    merged = merged.dropna(subset=["bucket", "month_index", "probability", "target_month"])
    if merged.empty:
        return pd.DataFrame(columns=columns)

    merged["bucket"] = merged["bucket"].round().astype(int)
    merged["month_index"] = merged["month_index"].round().astype(int)
    merged = merged[(merged["bucket"] >= 1) & (merged["bucket"] <= 5)]
    merged = merged[merged["month_index"] >= 1]
    if merged.empty:
        return pd.DataFrame(columns=columns)

    merged["forecast_month"] = merged.apply(
        lambda row: _add_months(str(row["target_month"]), int(row["month_index"]) - 1),
        axis=1,
    )
    merged = merged.dropna(subset=["forecast_month"])
    if merged.empty:
        return pd.DataFrame(columns=columns)

    triage_df = _latest_triage(con)
    if not triage_df.empty:
        triage_df["iso3"] = triage_df["iso3"].astype(str).str.upper()
        triage_df["hazard_code"] = triage_df["hazard_code"].astype(str).str.upper()
        merged = merged.merge(
            triage_df,
            left_on=["hs_run_id", "iso3", "hazard_code"],
            right_on=["run_id", "iso3", "hazard_code"],
            how="left",
        )
    else:
        merged["tier"] = None
        merged["triage_score"] = None

    merged.rename(columns={"tier": "triage_tier"}, inplace=True)

    duplicate_mask = merged.duplicated(
        subset=[
            "iso3",
            "hazard_code",
            "metric",
            "model_name",
            "forecast_month",
            "bucket",
        ],
        keep=False,
    )
    if duplicate_mask.any():
        _warn_once("duplicate_buckets", "Duplicate bucket rows found; keeping first value per bucket.")

    merged["probability"] = pd.to_numeric(merged["probability"], errors="coerce")
    merged = merged.dropna(subset=["probability"])

    country_map = _load_country_registry()
    merged["country_name"] = merged["iso3"].map(country_map).fillna(merged["iso3"])

    pivot_index = [
        "iso3",
        "country_name",
        "hazard_code",
        "metric",
        "model_name",
        "forecast_month",
        "hs_run_id",
        "triage_score",
        "triage_tier",
    ]

    pivot = (
        merged.sort_values(pivot_index + ["bucket"])
        .drop_duplicates(subset=pivot_index + ["bucket"], keep="first")
        .pivot_table(
            index=pivot_index,
            columns="bucket",
            values="probability",
            aggfunc="first",
        )
    )

    pivot.columns = [f"SPD_{int(col)}" for col in pivot.columns]
    pivot = pivot.reset_index()

    for idx in range(1, 6):
        col = f"SPD_{idx}"
        if col not in pivot.columns:
            pivot[col] = 0.0
    pivot[[f"SPD_{idx}" for idx in range(1, 6)]] = pivot[
        [f"SPD_{idx}" for idx in range(1, 6)]
    ].fillna(0.0)

    pivot["forecast_month"] = pivot["forecast_month"].astype(str)
    pivot[["year", "month"]] = pivot["forecast_month"].str.split("-", expand=True)
    pivot["year"] = pd.to_numeric(pivot["year"], errors="coerce")
    pivot["month"] = pd.to_numeric(pivot["month"], errors="coerce")

    pa_centroids = pd.Series([0, 30000, 150000, 375000, 700000], index=[1, 2, 3, 4, 5])
    fatal_centroids = pd.Series([0, 15, 62, 300, 700], index=[1, 2, 3, 4, 5])

    def _compute_eiv(row: pd.Series) -> float:
        centroids = fatal_centroids if row["metric"] == "FATALITIES" else pa_centroids
        return float(
            row["SPD_1"] * centroids[1]
            + row["SPD_2"] * centroids[2]
            + row["SPD_3"] * centroids[3]
            + row["SPD_4"] * centroids[4]
            + row["SPD_5"] * centroids[5]
        )

    pivot["EIV"] = pivot.apply(_compute_eiv, axis=1)

    output = pd.DataFrame(
        {
            "ISO": pivot["iso3"],
            "country_name": pivot["country_name"],
            "year": pivot["year"],
            "month": pivot["month"],
            "forecast_month": pivot["forecast_month"],
            "metric": pivot["metric"],
            "hazard": pivot["hazard_code"],
            "model": pivot["model_name"],
            "SPD_1": pivot["SPD_1"],
            "SPD_2": pivot["SPD_2"],
            "SPD_3": pivot["SPD_3"],
            "SPD_4": pivot["SPD_4"],
            "SPD_5": pivot["SPD_5"],
            "EIV": pivot["EIV"],
            "triage_score": pivot.get("triage_score"),
            "triage_tier": pivot.get("triage_tier"),
            "hs_run_ID": pivot["hs_run_id"],
        }
    )

    return output[columns]


def build_triage_export(con) -> pd.DataFrame:
    columns = [
        "Triage Year",
        "Triage Month",
        "Triage Date",
        "Run ID",
        "Triage model",
        "ISO3",
        "Country",
        "Triage Score",
        "Triage Tier",
    ]

    if con is None or not _table_exists(con, "hs_triage"):
        return pd.DataFrame(columns=columns)

    triage_df = _latest_triage(con)
    if triage_df.empty:
        return pd.DataFrame(columns=columns)

    triage_df["iso3"] = triage_df["iso3"].astype(str).str.upper()
    triage_df["triage_score"] = pd.to_numeric(triage_df["triage_score"], errors="coerce")
    triage_df["_created_at"] = pd.to_datetime(triage_df["created_at"], errors="coerce")

    tier_order = {"quiet": 0, "watchlist": 1, "priority": 2}
    triage_df["_tier_sort"] = (
        triage_df["tier"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(tier_order)
        .fillna(-1)
    )
    triage_df["_tier_label"] = triage_df["tier"].astype(str).fillna("")
    triage_df.loc[triage_df["tier"].isna(), "_tier_label"] = ""

    score_df = (
        triage_df.groupby(["run_id", "iso3"], dropna=False)["triage_score"]
        .max()
        .reset_index()
    )
    date_df = (
        triage_df.groupby(["run_id", "iso3"], dropna=False)["_created_at"]
        .max()
        .reset_index()
    )
    tier_df = (
        triage_df.sort_values(["run_id", "iso3", "_tier_sort", "_tier_label"])
        .groupby(["run_id", "iso3"], as_index=False)
        .tail(1)[["run_id", "iso3", "tier"]]
    )

    collapsed = score_df.merge(date_df, on=["run_id", "iso3"], how="left").merge(
        tier_df, on=["run_id", "iso3"], how="left"
    )

    def _derive_date_fields(row: pd.Series) -> pd.Series:
        created_at = row["_created_at"]
        run_id = row["run_id"]
        parsed = created_at if pd.notna(created_at) else _parse_run_id_date(run_id)
        if not parsed:
            return pd.Series({"Triage Year": None, "Triage Month": None, "Triage Date": None})
        return pd.Series(
            {
                "Triage Year": parsed.year,
                "Triage Month": parsed.month,
                "Triage Date": parsed.strftime("%Y-%m-%d"),
            }
        )

    date_fields = collapsed.apply(_derive_date_fields, axis=1)
    collapsed = pd.concat([collapsed, date_fields], axis=1)

    country_map = _load_country_registry()
    collapsed["Country"] = collapsed["iso3"].map(country_map).fillna(collapsed["iso3"])

    model_map = _load_triage_models(con)
    collapsed["Triage model"] = collapsed["run_id"].map(model_map)

    output = pd.DataFrame(
        {
            "Triage Year": collapsed["Triage Year"],
            "Triage Month": collapsed["Triage Month"],
            "Triage Date": collapsed["Triage Date"],
            "Run ID": collapsed["run_id"],
            "Triage model": collapsed["Triage model"],
            "ISO3": collapsed["iso3"],
            "Country": collapsed["Country"],
            "Triage Score": collapsed["triage_score"],
            "Triage Tier": collapsed["tier"],
        }
    )

    return output.sort_values(["Run ID", "ISO3"]).reset_index(drop=True)[columns]
