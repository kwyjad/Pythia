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

from pythia.db.helpers import table_exists as _table_exists, table_columns as _table_columns, pick_column as _pick_column
from resolver.query import eiv_sql

LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:  # pragma: no cover - silence library default
    LOGGER.addHandler(logging.NullHandler())


_WARNING_FLAGS: set[str] = set()


def _extract_year_month(series: pd.Series) -> pd.Series:
    if series is None or series.empty:
        return pd.Series(dtype="object")
    raw = series.astype(str)
    extracted = raw.str.extract(r"^(\d{4}-\d{2})", expand=False)
    if LOGGER.isEnabledFor(logging.DEBUG):
        null_mask = extracted.isna()
        missing_count = int(null_mask.sum())
        if missing_count:
            sample_values = raw[null_mask].head(3).tolist()
            LOGGER.debug(
                "forecast_month parsing: %d rows missing YYYY-MM, sample=%s",
                missing_count,
                sample_values,
            )
    return extracted


def _warn_once(key: str, message: str) -> None:
    if key in _WARNING_FLAGS:
        return
    _WARNING_FLAGS.add(key)
    LOGGER.warning(message)


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
    rc_likelihood_col = _pick_column(cols, ["regime_change_likelihood"])
    rc_direction_col = _pick_column(cols, ["regime_change_direction"])
    rc_magnitude_col = _pick_column(cols, ["regime_change_magnitude"])
    rc_score_col = _pick_column(cols, ["regime_change_score"])

    tier_expr = f"{tier_col} AS tier" if tier_col else "NULL AS tier"
    score_expr = f"{score_col} AS triage_score" if score_col else "NULL AS triage_score"
    created_expr = f"{created_col} AS created_at" if created_col else "NULL AS created_at"
    rc_likelihood_expr = (
        f"{rc_likelihood_col} AS regime_change_likelihood"
        if rc_likelihood_col
        else "NULL AS regime_change_likelihood"
    )
    rc_direction_expr = (
        f"{rc_direction_col} AS regime_change_direction"
        if rc_direction_col
        else "NULL AS regime_change_direction"
    )
    rc_magnitude_expr = (
        f"{rc_magnitude_col} AS regime_change_magnitude"
        if rc_magnitude_col
        else "NULL AS regime_change_magnitude"
    )
    rc_score_expr = (
        f"{rc_score_col} AS regime_change_score"
        if rc_score_col
        else "NULL AS regime_change_score"
    )

    df = conn.execute(
        f"""
        SELECT
            run_id,
            iso3,
            hazard_code,
            {tier_expr},
            {score_expr},
            {rc_likelihood_expr},
            {rc_direction_expr},
            {rc_magnitude_expr},
            {rc_score_expr},
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
        "regime_change_likelihood",
        "regime_change_direction",
        "regime_change_magnitude",
        "regime_change_score",
        "hs_run_ID",
        "track",
    ]

    if con is None or not _table_exists(con, "questions"):
        return pd.DataFrame(columns=columns)

    q_cols = _table_columns(con, "questions")
    required_q = {"question_id", "iso3", "hazard_code", "metric", "target_month", "hs_run_id"}
    if not required_q.issubset(q_cols):
        _warn_once("questions_missing", "Questions table missing required columns for export.")
        return pd.DataFrame(columns=columns)

    has_track = "track" in q_cols
    track_expr = ", track" if has_track else ""
    questions = con.execute(
        f"""
        SELECT question_id, iso3, hazard_code, metric, target_month, hs_run_id{track_expr}
        FROM questions
        """
    ).fetchdf()
    if not has_track:
        questions["track"] = None

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
        merged["regime_change_likelihood"] = None
        merged["regime_change_direction"] = None
        merged["regime_change_magnitude"] = None
        merged["regime_change_score"] = None

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
        "track",
        "triage_score",
        "triage_tier",
    ]
    rc_columns = [
        "regime_change_likelihood",
        "regime_change_direction",
        "regime_change_magnitude",
        "regime_change_score",
    ]
    for rc_col in rc_columns:
        if rc_col in merged.columns:
            pivot_index.append(rc_col)

    merged["_created_at"] = pd.to_datetime(merged.get("created_at"), errors="coerce")
    if LOGGER.isEnabledFor(logging.DEBUG):
        null_created = merged["_created_at"].isna()
        null_count = int(null_created.sum())
        if null_count:
            sample_values = merged.loc[null_created, "created_at"].head(3).tolist()
            LOGGER.debug(
                "forecast export: %d rows missing created_at, sample=%s",
                null_count,
                sample_values,
            )

    sort_columns = pivot_index + ["bucket", "_created_at"]
    ascending = [True] * (len(pivot_index) + 1) + [False]
    if "run_id" in merged.columns:
        sort_columns.append("run_id")
        ascending.append(False)

    dedup = merged.sort_values(sort_columns, ascending=ascending).drop_duplicates(
        subset=pivot_index + ["bucket"], keep="first"
    )
    if LOGGER.isEnabledFor(logging.DEBUG):
        dropped = len(merged) - len(dedup)
        if dropped:
            LOGGER.debug("forecast export: dropped %d duplicate bucket rows", dropped)

    pivot = (
        dedup.set_index(pivot_index + ["bucket"])["probability"]
        .unstack("bucket")
        .reindex(columns=[1, 2, 3, 4, 5])
        .fillna(0.0)
    )
    pivot.columns = [f"SPD_{int(col)}" for col in pivot.columns]
    pivot = pivot.reset_index()

    raw_forecast_month = pivot["forecast_month"]
    pivot["forecast_month"] = _extract_year_month(raw_forecast_month).fillna(
        raw_forecast_month.astype(str).str.slice(0, 7)
    )
    pivot["year"] = pd.to_numeric(pivot["forecast_month"].str.slice(0, 4), errors="coerce")
    pivot["month"] = pd.to_numeric(pivot["forecast_month"].str.slice(5, 7), errors="coerce")

    hazard_centroids: dict[tuple[str, str, int], float] = {}
    wildcard_centroids: dict[tuple[str, int], float] = {}
    if _table_exists(con, "bucket_centroids"):
        bc_cols = _table_columns(con, "bucket_centroids")
        if {"hazard_code", "metric", "bucket_index", "centroid"}.issubset(bc_cols):
            bc_df = con.execute(
                """
                SELECT hazard_code, metric, bucket_index, centroid
                FROM bucket_centroids
                """
            ).fetchdf()
            if not bc_df.empty:
                bc_df["hazard_code"] = bc_df["hazard_code"].astype(str).str.upper()
                bc_df["metric"] = bc_df["metric"].astype(str).str.upper()
                bc_df["bucket_index"] = pd.to_numeric(bc_df["bucket_index"], errors="coerce")
                bc_df["centroid"] = pd.to_numeric(bc_df["centroid"], errors="coerce")
                for _, row in bc_df.dropna().iterrows():
                    hazard = str(row["hazard_code"]).upper()
                    metric = str(row["metric"]).upper()
                    bucket_index = int(row["bucket_index"])
                    centroid = float(row["centroid"])
                    if hazard == "*":
                        wildcard_centroids[(metric, bucket_index)] = centroid
                    else:
                        hazard_centroids[(hazard, metric, bucket_index)] = centroid

    def _resolve_centroid(hazard: str, metric: str, bucket_index: int) -> float | None:
        key = (hazard, metric, bucket_index)
        if key in hazard_centroids:
            return hazard_centroids[key]
        wildcard_key = (metric, bucket_index)
        if wildcard_key in wildcard_centroids:
            return wildcard_centroids[wildcard_key]
        return eiv_sql.centroid_from_defaults(metric, bucket_index)

    def _compute_eiv(row: pd.Series) -> float:
        hazard = str(row["hazard_code"]).upper()
        metric = str(row["metric"]).upper()
        total = 0.0
        for idx in range(1, 6):
            centroid = _resolve_centroid(hazard, metric, idx) or 0.0
            total += float(row[f"SPD_{idx}"]) * centroid
        return float(total)

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
            "regime_change_likelihood": pivot.get("regime_change_likelihood"),
            "regime_change_direction": pivot.get("regime_change_direction"),
            "regime_change_magnitude": pivot.get("regime_change_magnitude"),
            "regime_change_score": pivot.get("regime_change_score"),
            "hs_run_ID": pivot["hs_run_id"],
            "track": pivot.get("track"),
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

    tier_order = {"quiet": 0, "priority": 1}
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


def build_ensemble_scores_export(con, model_filter: str) -> pd.DataFrame:
    """Build per-question scores export for a specific ensemble model.

    Args:
        con: DuckDB connection.
        model_filter: substring to match model_name in scores table
            (e.g. ``'ensemble_mean'`` or ``'ensemble_bayesmc'``).

    Returns a DataFrame with 56 columns matching the Scores_ensemble template.
    """

    columns: list[str] = [
        "question_id",
        "run_id",
        "run_date",
        "country_iso3",
        "country_name",
        "hazard_type",
        "metric",
        "triage_score",
        "triage_class",
        "rc_score",
        "rc_class",
        "track",
    ]
    for m in range(1, 7):
        for b in range(1, 6):
            columns.append(f"month{m}_bin{b}_forecast")
    for m in range(1, 7):
        columns.append(f"eiv_month{m}")
    for m in range(1, 7):
        columns.append(f"resolution_month{m}")
    columns.extend(["brier", "log_loss", "crps"])

    empty = pd.DataFrame(columns=columns)
    if con is None:
        return empty

    for tbl in ("questions", "scores"):
        if not _table_exists(con, tbl):
            return empty

    # ── 1. Identify the score model name ──────────────────────────────
    model_names = con.execute(
        "SELECT DISTINCT model_name FROM scores "
        "WHERE LOWER(CAST(model_name AS VARCHAR)) LIKE ?",
        [f"%{model_filter.lower()}%"],
    ).fetchdf()
    if model_names.empty:
        _warn_once(
            f"no_score_model:{model_filter}",
            f"No scores found for model matching '{model_filter}'",
        )
        return empty
    score_model = str(model_names.iloc[0]["model_name"])

    # ── 2. Questions + scores (avg across horizons) ───────────────────
    q_cols = _table_columns(con, "questions")
    has_track = "track" in q_cols
    track_select = "q.track," if has_track else ""
    track_group = ", q.track" if has_track else ""

    base = con.execute(
        f"""
        SELECT
            q.question_id,
            q.hs_run_id,
            q.iso3,
            q.hazard_code,
            UPPER(q.metric) AS metric,
            q.target_month,
            {track_select}
            AVG(CASE WHEN s.score_type = 'brier' THEN s.value END) AS brier,
            AVG(CASE WHEN s.score_type = 'log'   THEN s.value END) AS log_loss,
            AVG(CASE WHEN s.score_type = 'crps'  THEN s.value END) AS crps
        FROM scores s
        JOIN questions q ON s.question_id = q.question_id
        WHERE s.model_name = ?
        GROUP BY q.question_id, q.hs_run_id, q.iso3,
                 q.hazard_code, UPPER(q.metric), q.target_month{track_group}
        """,
        [score_model],
    ).fetchdf()
    if not has_track:
        base["track"] = None
    if base.empty:
        return empty

    base["iso3"] = base["iso3"].astype(str).str.upper()
    base["hazard_code"] = base["hazard_code"].astype(str).str.upper()

    # ── 3. Run dates ──────────────────────────────────────────────────
    if _table_exists(con, "hs_runs"):
        runs = con.execute(
            "SELECT hs_run_id, STRFTIME(generated_at, '%Y-%m-%d') AS run_date "
            "FROM hs_runs"
        ).fetchdf()
        base = base.merge(runs, on="hs_run_id", how="left")
    else:
        base["run_date"] = None

    # ── 4. Triage & regime-change data ────────────────────────────────
    _add_triage_columns(con, base)

    # ── 5. Forecasts → month×bin columns + EIV ────────────────────────
    _add_forecast_columns(con, base, model_filter)

    # ── 6. Resolutions per horizon ────────────────────────────────────
    _add_resolution_columns(con, base)

    # ── 7. Country names ──────────────────────────────────────────────
    country_map = _load_country_registry()
    base["country_name"] = base["iso3"].map(country_map).fillna(base["iso3"])

    # ── 8. Final output ───────────────────────────────────────────────
    output = pd.DataFrame(
        {
            "question_id": base["question_id"],
            "run_id": base["hs_run_id"],
            "run_date": base.get("run_date"),
            "country_iso3": base["iso3"],
            "country_name": base["country_name"],
            "hazard_type": base["hazard_code"],
            "metric": base["metric"],
            "triage_score": base.get("triage_score"),
            "triage_class": base.get("triage_class"),
            "rc_score": base.get("rc_score"),
            "rc_class": base.get("rc_class"),
            "track": base.get("track"),
        }
    )
    for m in range(1, 7):
        for b in range(1, 6):
            col = f"month{m}_bin{b}_forecast"
            output[col] = base.get(col)
    for m in range(1, 7):
        output[f"eiv_month{m}"] = base.get(f"eiv_month{m}")
    for m in range(1, 7):
        output[f"resolution_month{m}"] = base.get(f"resolution_month{m}")
    output["brier"] = base["brier"]
    output["log_loss"] = base["log_loss"]
    output["crps"] = base["crps"]

    return (
        output.sort_values(["run_id", "country_iso3", "hazard_type"])
        .reset_index(drop=True)[columns]
    )


# ── helpers for build_ensemble_scores_export ──────────────────────────


def _add_triage_columns(con, df: pd.DataFrame) -> None:
    """Merge triage_score, triage_class, rc_score, rc_class into *df* in-place."""

    for col in ("triage_score", "triage_class", "rc_score", "rc_class"):
        df[col] = None

    if not _table_exists(con, "hs_triage"):
        return

    cols = _table_columns(con, "hs_triage")
    if not {"run_id", "iso3", "hazard_code"}.issubset(cols):
        return

    score_expr = "triage_score" if "triage_score" in cols else "NULL"
    tier_expr = "tier" if "tier" in cols else "NULL"
    rc_score_expr = "regime_change_score" if "regime_change_score" in cols else "NULL"
    rc_level_expr = "regime_change_level" if "regime_change_level" in cols else "NULL"
    created_expr = "created_at" if "created_at" in cols else "NULL"

    triage = con.execute(
        f"""
        SELECT
            run_id,
            iso3,
            hazard_code,
            {score_expr}   AS triage_score,
            {tier_expr}    AS triage_class,
            {rc_score_expr} AS rc_score,
            {rc_level_expr} AS rc_class,
            {created_expr}  AS created_at
        FROM hs_triage
        """
    ).fetchdf()

    if triage.empty:
        return

    triage["iso3"] = triage["iso3"].astype(str).str.upper()
    triage["hazard_code"] = triage["hazard_code"].astype(str).str.upper()

    # keep latest per (run_id, iso3, hazard_code)
    if "created_at" in triage.columns:
        triage["_ts"] = pd.to_datetime(triage["created_at"], errors="coerce")
        triage = triage.sort_values("_ts").drop_duplicates(
            subset=["run_id", "iso3", "hazard_code"], keep="last"
        )
        triage.drop(columns=["_ts", "created_at"], inplace=True)
    else:
        triage = triage.drop_duplicates(
            subset=["run_id", "iso3", "hazard_code"], keep="last"
        )

    merged = df.merge(
        triage,
        left_on=["hs_run_id", "iso3", "hazard_code"],
        right_on=["run_id", "iso3", "hazard_code"],
        how="left",
        suffixes=("", "_t"),
    )

    for col in ("triage_score", "triage_class", "rc_score", "rc_class"):
        src = f"{col}_t" if f"{col}_t" in merged.columns else col
        df[col] = merged[src].values

    return


def _add_forecast_columns(con, df: pd.DataFrame, model_filter: str) -> None:
    """Add month×bin forecast columns and EIV columns to *df* in-place."""

    # initialise columns
    for m in range(1, 7):
        for b in range(1, 6):
            df[f"month{m}_bin{b}_forecast"] = None
    for m in range(1, 7):
        df[f"eiv_month{m}"] = None

    if not _table_exists(con, "forecasts_ensemble"):
        return

    fe_cols = _table_columns(con, "forecasts_ensemble")
    model_col = _pick_column(fe_cols, ["model_name", "model"])
    month_col = _pick_column(fe_cols, ["month_index", "horizon_m"])
    bucket_col = _pick_column(fe_cols, ["bucket_index", "class_bin"])
    prob_col = _pick_column(fe_cols, ["probability", "p", "prob"])

    if not all([model_col, month_col, bucket_col, prob_col]):
        return

    # find model in forecasts_ensemble
    fc_models = con.execute(
        f"SELECT DISTINCT {model_col} FROM forecasts_ensemble "
        f"WHERE LOWER(CAST({model_col} AS VARCHAR)) LIKE ?",
        [f"%{model_filter.lower()}%"],
    ).fetchdf()
    if fc_models.empty:
        return
    fc_model = str(fc_models.iloc[0][model_col])

    status_col = _pick_column(fe_cols, ["status"])
    status_filter = (
        f"AND LOWER(CAST({status_col} AS VARCHAR)) = 'ok'" if status_col else ""
    )

    fc = con.execute(
        f"""
        SELECT
            question_id,
            {month_col}  AS month_idx,
            {bucket_col} AS bucket_idx,
            {prob_col}   AS prob
        FROM forecasts_ensemble
        WHERE {model_col} = ?
          AND {prob_col} IS NOT NULL
          AND {month_col} IS NOT NULL
          AND {bucket_col} IS NOT NULL
          {status_filter}
        """,
        [fc_model],
    ).fetchdf()

    if fc.empty:
        return

    fc["month_idx"] = pd.to_numeric(fc["month_idx"], errors="coerce")
    fc["prob"] = pd.to_numeric(fc["prob"], errors="coerce")

    bucket_numeric = pd.to_numeric(fc["bucket_idx"], errors="coerce")
    if bucket_numeric.isna().any():
        label = fc["bucket_idx"].astype(str).str.strip().str.lower()
        pa_map = {"<10k": 1, "10k-<50k": 2, "50k-<250k": 3, "250k-<500k": 4, ">=500k": 5}
        fatal_map = {"<5": 1, "5-<25": 2, "25-<100": 3, "100-<500": 4, ">=500": 5}
        bucket_numeric = bucket_numeric.fillna(label.map(pa_map)).fillna(label.map(fatal_map))
    fc["bucket_idx"] = pd.to_numeric(bucket_numeric, errors="coerce")

    fc = fc.dropna(subset=["month_idx", "bucket_idx", "prob"])
    fc["month_idx"] = fc["month_idx"].astype(int)
    fc["bucket_idx"] = fc["bucket_idx"].astype(int)
    fc = fc[(fc["month_idx"] >= 1) & (fc["month_idx"] <= 6)]
    fc = fc[(fc["bucket_idx"] >= 1) & (fc["bucket_idx"] <= 5)]

    fc = fc.drop_duplicates(
        subset=["question_id", "month_idx", "bucket_idx"], keep="first"
    )

    fc["col_name"] = (
        "month" + fc["month_idx"].astype(str)
        + "_bin" + fc["bucket_idx"].astype(str)
        + "_forecast"
    )
    pivoted = fc.pivot_table(
        index="question_id", columns="col_name", values="prob", aggfunc="first"
    ).reset_index()

    merged = df[["question_id"]].merge(pivoted, on="question_id", how="left")
    for m in range(1, 7):
        for b in range(1, 6):
            col = f"month{m}_bin{b}_forecast"
            if col in merged.columns:
                df[col] = merged[col].values

    # ── EIV computation ───────────────────────────────────────────────
    hazard_centroids: dict[tuple[str, str, int], float] = {}
    wildcard_centroids: dict[tuple[str, int], float] = {}
    if _table_exists(con, "bucket_centroids"):
        bc = con.execute(
            "SELECT hazard_code, metric, bucket_index, centroid FROM bucket_centroids"
        ).fetchdf()
        if not bc.empty:
            bc["hazard_code"] = bc["hazard_code"].astype(str).str.upper()
            bc["metric"] = bc["metric"].astype(str).str.upper()
            bc["bucket_index"] = pd.to_numeric(bc["bucket_index"], errors="coerce")
            bc["centroid"] = pd.to_numeric(bc["centroid"], errors="coerce")
            for _, row in bc.dropna().iterrows():
                h, met, bi, c = (
                    str(row["hazard_code"]),
                    str(row["metric"]),
                    int(row["bucket_index"]),
                    float(row["centroid"]),
                )
                if h == "*":
                    wildcard_centroids[(met, bi)] = c
                else:
                    hazard_centroids[(h, met, bi)] = c

    def _centroid(hazard: str, metric: str, bucket: int) -> float:
        key = (hazard, metric, bucket)
        if key in hazard_centroids:
            return hazard_centroids[key]
        wkey = (metric, bucket)
        if wkey in wildcard_centroids:
            return wildcard_centroids[wkey]
        return eiv_sql.centroid_from_defaults(metric, bucket) or 0.0

    for m in range(1, 7):
        eiv_vals = []
        for _, row in df.iterrows():
            hazard = str(row.get("hazard_code", "")).upper()
            metric = str(row.get("metric", "")).upper()
            total = 0.0
            any_missing = False
            for b in range(1, 6):
                prob = row.get(f"month{m}_bin{b}_forecast")
                if pd.isna(prob) or prob is None:
                    any_missing = True
                    break
                total += float(prob) * _centroid(hazard, metric, b)
            eiv_vals.append(None if any_missing else total)
        df[f"eiv_month{m}"] = eiv_vals


def _add_resolution_columns(con, df: pd.DataFrame) -> None:
    """Add resolution_month1..6 columns to *df* in-place."""

    for m in range(1, 7):
        df[f"resolution_month{m}"] = None

    if not _table_exists(con, "resolutions"):
        return

    res = con.execute(
        "SELECT question_id, horizon_m, value FROM resolutions"
    ).fetchdf()
    if res.empty:
        return

    res["horizon_m"] = pd.to_numeric(res["horizon_m"], errors="coerce")
    res["value"] = pd.to_numeric(res["value"], errors="coerce")
    res = res.dropna(subset=["horizon_m"])
    res["horizon_m"] = res["horizon_m"].astype(int)
    res = res[(res["horizon_m"] >= 1) & (res["horizon_m"] <= 6)]

    res["col_name"] = "resolution_month" + res["horizon_m"].astype(str)
    piv = res.pivot_table(
        index="question_id", columns="col_name", values="value", aggfunc="first"
    ).reset_index()

    merged = df[["question_id"]].merge(piv, on="question_id", how="left")
    for m in range(1, 7):
        col = f"resolution_month{m}"
        if col in merged.columns:
            df[col] = merged[col].values


def build_model_scores_export(con) -> pd.DataFrame:
    """Build aggregated model-level scores export.

    Returns a DataFrame with 16 columns matching the Scores_model template.
    """

    columns = [
        "forecast_model",
        "hazard",
        "metric",
        "track",
        "forecasts",
        "average_brier",
        "average_log_loss",
        "average_crps",
        "median_brier",
        "median_log_loss",
        "median_crps",
        "min_brier",
        "min_log_loss",
        "min_crps",
        "max_brier",
        "max_log_loss",
        "max_crps",
    ]

    empty = pd.DataFrame(columns=columns)
    if con is None:
        return empty

    for tbl in ("questions", "scores"):
        if not _table_exists(con, tbl):
            return empty

    q_cols = _table_columns(con, "questions")
    has_track = "track" in q_cols
    track_select = "q.track AS track," if has_track else ""
    track_group = ", q.track" if has_track else ""
    track_order = ", track" if has_track else ""

    df = con.execute(
        f"""
        SELECT
            COALESCE(s.model_name, 'ensemble') AS forecast_model,
            q.hazard_code                      AS hazard,
            UPPER(q.metric)                    AS metric,
            {track_select}
            COUNT(DISTINCT s.question_id)      AS forecasts,
            AVG(   CASE WHEN s.score_type = 'brier' THEN s.value END) AS average_brier,
            AVG(   CASE WHEN s.score_type = 'log'   THEN s.value END) AS average_log_loss,
            AVG(   CASE WHEN s.score_type = 'crps'  THEN s.value END) AS average_crps,
            MEDIAN(CASE WHEN s.score_type = 'brier' THEN s.value END) AS median_brier,
            MEDIAN(CASE WHEN s.score_type = 'log'   THEN s.value END) AS median_log_loss,
            MEDIAN(CASE WHEN s.score_type = 'crps'  THEN s.value END) AS median_crps,
            MIN(   CASE WHEN s.score_type = 'brier' THEN s.value END) AS min_brier,
            MIN(   CASE WHEN s.score_type = 'log'   THEN s.value END) AS min_log_loss,
            MIN(   CASE WHEN s.score_type = 'crps'  THEN s.value END) AS min_crps,
            MAX(   CASE WHEN s.score_type = 'brier' THEN s.value END) AS max_brier,
            MAX(   CASE WHEN s.score_type = 'log'   THEN s.value END) AS max_log_loss,
            MAX(   CASE WHEN s.score_type = 'crps'  THEN s.value END) AS max_crps
        FROM scores s
        JOIN questions q ON s.question_id = q.question_id
        GROUP BY s.model_name, q.hazard_code, UPPER(q.metric){track_group}
        ORDER BY forecast_model, hazard, metric{track_order}
        """
    ).fetchdf()

    if df.empty:
        return empty

    if not has_track:
        df["track"] = None

    return df[columns].reset_index(drop=True)


def build_rationale_export(
    con,
    hazard_code: str,
    model_name: str | None = None,
) -> pd.DataFrame:
    """Build deduplicated LLM rationale export filtered by hazard and optionally model.

    Returns one row per (question_id, model_name) containing the human_explanation
    text and contextual columns.  The same explanation is stored on every
    (month_index, bucket_index) row, so SELECT DISTINCT collapses the duplicates.
    """

    columns = [
        "question_id",
        "run_id",
        "hs_run_id",
        "iso3",
        "country_name",
        "hazard_code",
        "metric",
        "target_month",
        "model_name",
        "human_explanation",
    ]

    empty = pd.DataFrame(columns=columns)

    if con is None or not _table_exists(con, "questions"):
        return empty

    hazard_upper = hazard_code.strip().upper()

    sub_queries: list[str] = []
    params: list[str] = []

    # ── forecasts_raw (per-model forecasts) ──────────────────────────
    if _table_exists(con, "forecasts_raw"):
        raw_cols = _table_columns(con, "forecasts_raw")
        if "human_explanation" in raw_cols:
            raw_sql = (
                "SELECT DISTINCT"
                "  fr.question_id,"
                "  fr.run_id,"
                "  q.hs_run_id,"
                "  q.iso3,"
                "  q.hazard_code,"
                "  q.metric,"
                "  q.target_month,"
                "  fr.model_name,"
                "  fr.human_explanation"
                " FROM forecasts_raw fr"
                " JOIN questions q ON fr.question_id = q.question_id"
                " WHERE UPPER(q.hazard_code) = ?"
                "  AND fr.human_explanation IS NOT NULL"
                "  AND TRIM(CAST(fr.human_explanation AS VARCHAR)) != ''"
            )
            params.append(hazard_upper)
            if model_name:
                raw_sql += " AND LOWER(CAST(fr.model_name AS VARCHAR)) LIKE ?"
                params.append(f"%{model_name.lower()}%")
            sub_queries.append(raw_sql)

    # ── forecasts_ensemble (aggregated forecasts) ────────────────────
    if _table_exists(con, "forecasts_ensemble"):
        ens_cols = _table_columns(con, "forecasts_ensemble")
        if "human_explanation" in ens_cols:
            ens_sql = (
                "SELECT DISTINCT"
                "  fe.question_id,"
                "  fe.run_id,"
                "  q.hs_run_id,"
                "  fe.iso3,"
                "  fe.hazard_code,"
                "  fe.metric,"
                "  q.target_month,"
                "  fe.model_name,"
                "  fe.human_explanation"
                " FROM forecasts_ensemble fe"
                " JOIN questions q ON fe.question_id = q.question_id"
                " WHERE UPPER(fe.hazard_code) = ?"
                "  AND fe.human_explanation IS NOT NULL"
                "  AND TRIM(CAST(fe.human_explanation AS VARCHAR)) != ''"
            )
            params.append(hazard_upper)
            if model_name:
                ens_sql += " AND LOWER(CAST(fe.model_name AS VARCHAR)) LIKE ?"
                params.append(f"%{model_name.lower()}%")
            sub_queries.append(ens_sql)

    if not sub_queries:
        return empty

    full_sql = " UNION ALL ".join(sub_queries)
    full_sql += " ORDER BY iso3, metric, model_name, question_id"

    df = con.execute(full_sql, params).fetchdf()

    if df.empty:
        return empty

    df["iso3"] = df["iso3"].astype(str).str.upper()
    df["hazard_code"] = df["hazard_code"].astype(str).str.upper()
    df["metric"] = df["metric"].astype(str).str.upper()

    country_map = _load_country_registry()
    df["country_name"] = df["iso3"].map(country_map).fillna(df["iso3"])

    output = pd.DataFrame({col: df.get(col) for col in columns})
    return output.sort_values(
        ["iso3", "metric", "model_name", "question_id"]
    ).reset_index(drop=True)
