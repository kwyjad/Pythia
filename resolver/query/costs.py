# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Helpers for aggregating LLM cost and latency summaries."""

from __future__ import annotations

import pandas as pd

COST_COLUMNS = [
    "grain",
    "row_type",
    "year",
    "month",
    "run_id",
    "model",
    "phase",
    "total_cost_usd",
    "n_questions",
    "avg_cost_per_question",
    "median_cost_per_question",
    "n_countries",
    "avg_cost_per_country",
    "median_cost_per_country",
]

LATENCY_COLUMNS = [
    "run_id",
    "year",
    "month",
    "model",
    "phase",
    "n_calls",
    "p50_elapsed_ms",
    "p90_elapsed_ms",
]


def phase_group(phase: str | None) -> str:
    if not phase:
        return "other"
    value = str(phase).strip().lower()
    if not value:
        return "other"
    if "web" in value:
        return "web_search"
    if value.startswith("hs"):
        return "hs"
    if "research" in value:
        return "research"
    if "spd" in value or "forecast" in value:
        return "forecast"
    if "scenario" in value:
        return "scenario"
    return "other"


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


def _normalize_string(series: pd.Series) -> pd.Series:
    series = series.where(series.notna(), None)

    def _clean(value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text if text else None

    return series.map(_clean)


def _load_llm_calls(conn) -> pd.DataFrame:
    if not _table_exists(conn, "llm_calls"):
        return pd.DataFrame()

    cols = _table_columns(conn, "llm_calls")
    run_id_col = _pick_column(cols, ["run_id"])
    hs_run_id_col = _pick_column(cols, ["hs_run_id"])
    question_id_col = _pick_column(cols, ["question_id"])
    iso3_col = _pick_column(cols, ["iso3"])
    phase_col = _pick_column(cols, ["phase"])
    model_id_col = _pick_column(cols, ["model_id"])
    model_name_col = _pick_column(cols, ["model_name"])
    cost_col = _pick_column(cols, ["cost_usd"])
    elapsed_col = _pick_column(cols, ["elapsed_ms"])
    created_col = _pick_column(cols, ["created_at"])

    run_expr = f"CAST({run_id_col} AS VARCHAR) AS run_id" if run_id_col else "NULL AS run_id"
    hs_run_expr = (
        f"CAST({hs_run_id_col} AS VARCHAR) AS hs_run_id" if hs_run_id_col else "NULL AS hs_run_id"
    )
    question_expr = (
        f"CAST({question_id_col} AS VARCHAR) AS question_id"
        if question_id_col
        else "NULL AS question_id"
    )
    iso3_expr = f"CAST({iso3_col} AS VARCHAR) AS iso3" if iso3_col else "NULL AS iso3"
    phase_expr = f"CAST({phase_col} AS VARCHAR) AS phase" if phase_col else "NULL AS phase"
    model_id_expr = (
        f"CAST({model_id_col} AS VARCHAR) AS model_id" if model_id_col else "NULL AS model_id"
    )
    model_name_expr = (
        f"CAST({model_name_col} AS VARCHAR) AS model_name"
        if model_name_col
        else "NULL AS model_name"
    )
    cost_expr = f"CAST({cost_col} AS DOUBLE) AS cost_usd" if cost_col else "0.0 AS cost_usd"
    elapsed_expr = (
        f"CAST({elapsed_col} AS DOUBLE) AS elapsed_ms" if elapsed_col else "NULL AS elapsed_ms"
    )
    created_expr = f"{created_col} AS created_at" if created_col else "NULL AS created_at"

    sql = f"""
        SELECT
            {run_expr},
            {hs_run_expr},
            {question_expr},
            {iso3_expr},
            {phase_expr},
            {model_id_expr},
            {model_name_expr},
            {cost_expr},
            {elapsed_expr},
            {created_expr}
        FROM llm_calls
    """

    df = conn.execute(sql).fetchdf()
    if df.empty:
        return df

    for col in ["run_id", "hs_run_id", "question_id", "iso3", "phase", "model_id", "model_name"]:
        df[col] = _normalize_string(df[col])

    df["cost_usd"] = pd.to_numeric(df["cost_usd"], errors="coerce").fillna(0.0)
    df["elapsed_ms"] = pd.to_numeric(df["elapsed_ms"], errors="coerce")

    model = df["model_id"].where(df["model_id"].notna(), None)
    model = model.where(model.notna(), df["model_name"])
    model = model.where(model.notna(), None)
    model = model.fillna("unknown")
    df["model"] = model

    df["phase"] = df["phase"].map(phase_group)

    run_key = df["run_id"].where(df["run_id"].notna(), df["hs_run_id"])
    df["run_id"] = run_key

    created_at = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    df["year"] = created_at.dt.year.astype("Int64")
    df["month"] = created_at.dt.month.astype("Int64")
    df["created_at"] = created_at

    return df[
        [
            "run_id",
            "question_id",
            "iso3",
            "phase",
            "model",
            "cost_usd",
            "elapsed_ms",
            "year",
            "month",
            "created_at",
        ]
    ]


def _with_group(df: pd.DataFrame, group_cols: list[str]) -> tuple[pd.DataFrame, list[str], bool]:
    if group_cols:
        return df, group_cols, False
    df = df.copy()
    df["_all"] = "all"
    return df, ["_all"], True


def _attach_entity_stats(
    summary: pd.DataFrame,
    df: pd.DataFrame,
    group_cols: list[str],
    entity_col: str,
    n_col: str,
    avg_col: str,
    median_col: str,
) -> pd.DataFrame:
    if entity_col not in df.columns or df[entity_col].isna().all():
        summary[n_col] = 0
        summary[avg_col] = None
        summary[median_col] = None
        return summary

    entity_df = df.dropna(subset=[entity_col])
    if entity_df.empty:
        summary[n_col] = 0
        summary[avg_col] = None
        summary[median_col] = None
        return summary

    per_entity = (
        entity_df.groupby(group_cols + [entity_col], dropna=False)["cost_usd"]
        .sum()
        .reset_index()
    )
    stats = (
        per_entity.groupby(group_cols, dropna=False)["cost_usd"]
        .agg(["mean", "median", "count"])
        .reset_index()
        .rename(columns={"mean": avg_col, "median": median_col, "count": n_col})
    )
    merged = summary.merge(stats, on=group_cols, how="left")
    merged[n_col] = merged[n_col].fillna(0).astype(int)
    return merged


def _compute_summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=group_cols + ["total_cost_usd"])

    total = (
        df.groupby(group_cols, dropna=False)["cost_usd"]
        .sum()
        .reset_index()
        .rename(columns={"cost_usd": "total_cost_usd"})
    )
    summary = _attach_entity_stats(
        total, df, group_cols, "question_id", "n_questions", "avg_cost_per_question", "median_cost_per_question"
    )
    summary = _attach_entity_stats(
        summary, df, group_cols, "iso3", "n_countries", "avg_cost_per_country", "median_cost_per_country"
    )
    return summary


def _finalize_costs_frame(
    df: pd.DataFrame,
    grain: str,
    row_type: str,
    include_model: bool,
    include_phase: bool,
) -> pd.DataFrame:
    df = df.copy()
    df["grain"] = grain
    df["row_type"] = row_type
    if "year" not in df.columns:
        df["year"] = pd.NA
    if "month" not in df.columns:
        df["month"] = pd.NA
    if "run_id" not in df.columns:
        df["run_id"] = pd.NA
    if "model" not in df.columns:
        df["model"] = pd.NA
    if "phase" not in df.columns:
        df["phase"] = pd.NA
    if not include_model:
        df["model"] = pd.NA
    if not include_phase:
        df["phase"] = pd.NA
    for col in [
        "total_cost_usd",
        "n_questions",
        "avg_cost_per_question",
        "median_cost_per_question",
        "n_countries",
        "avg_cost_per_country",
        "median_cost_per_country",
    ]:
        if col not in df.columns:
            df[col] = pd.NA
    return df[COST_COLUMNS]


def _build_costs_grain(df: pd.DataFrame, grain: str, group_cols: list[str]) -> dict[str, pd.DataFrame]:
    if df.empty:
        empty = pd.DataFrame(columns=COST_COLUMNS)
        return {"summary": empty, "by_model": empty, "by_phase": empty}

    df_grouped, group_cols_use, drop_all = _with_group(df, group_cols)
    summary = _compute_summary(df_grouped, group_cols_use)
    by_model = (
        df_grouped.groupby(group_cols_use + ["model"], dropna=False)["cost_usd"]
        .sum()
        .reset_index()
        .rename(columns={"cost_usd": "total_cost_usd"})
    )
    by_model = by_model.merge(
        summary[
            group_cols_use
            + [
                "n_questions",
                "avg_cost_per_question",
                "median_cost_per_question",
                "n_countries",
                "avg_cost_per_country",
                "median_cost_per_country",
            ]
        ],
        on=group_cols_use,
        how="left",
    )

    by_phase = (
        df_grouped.groupby(group_cols_use + ["phase"], dropna=False)["cost_usd"]
        .sum()
        .reset_index()
        .rename(columns={"cost_usd": "total_cost_usd"})
    )
    by_phase = by_phase.merge(
        summary[
            group_cols_use
            + [
                "n_questions",
                "avg_cost_per_question",
                "median_cost_per_question",
                "n_countries",
                "avg_cost_per_country",
                "median_cost_per_country",
            ]
        ],
        on=group_cols_use,
        how="left",
    )

    if drop_all:
        summary = summary.drop(columns=group_cols_use)
        by_model = by_model.drop(columns=group_cols_use)
        by_phase = by_phase.drop(columns=group_cols_use)

    return {
        "summary": _finalize_costs_frame(summary, grain, "summary", include_model=False, include_phase=False),
        "by_model": _finalize_costs_frame(by_model, grain, "by_model", include_model=True, include_phase=False),
        "by_phase": _finalize_costs_frame(by_phase, grain, "by_phase", include_model=False, include_phase=True),
    }


def build_costs_total(conn) -> dict[str, pd.DataFrame]:
    df = _load_llm_calls(conn)
    return _build_costs_grain(df, "total", [])


def build_costs_monthly(conn) -> dict[str, pd.DataFrame]:
    df = _load_llm_calls(conn)
    return _build_costs_grain(df, "monthly", ["year", "month"])


def build_costs_runs(conn) -> dict[str, pd.DataFrame]:
    df = _load_llm_calls(conn)
    if df.empty:
        empty = pd.DataFrame(columns=COST_COLUMNS)
        return {"summary": empty, "by_model": empty, "by_phase": empty}

    metadata = (
        df.groupby(["run_id"], dropna=False)["created_at"]
        .min()
        .reset_index()
        .rename(columns={"created_at": "run_created_at"})
    )
    if not metadata.empty:
        metadata["year"] = metadata["run_created_at"].dt.year.astype("Int64")
        metadata["month"] = metadata["run_created_at"].dt.month.astype("Int64")
    else:
        metadata["year"] = pd.NA
        metadata["month"] = pd.NA

    tables = _build_costs_grain(df, "run", ["run_id"])
    for key, table in tables.items():
        if table.empty:
            continue
        table = table.merge(metadata[["run_id", "year", "month"]], on="run_id", how="left")
        table["year"] = table["year_y"].combine_first(table["year_x"])
        table["month"] = table["month_y"].combine_first(table["month_x"])
        table = table.drop(columns=["year_x", "year_y", "month_x", "month_y"])
        tables[key] = table[COST_COLUMNS]
    return tables


def build_latencies_runs(conn) -> pd.DataFrame:
    df = _load_llm_calls(conn)
    if df.empty or "elapsed_ms" not in df.columns:
        return pd.DataFrame(columns=LATENCY_COLUMNS)

    df = df[df["elapsed_ms"].notna()]
    if df.empty:
        return pd.DataFrame(columns=LATENCY_COLUMNS)

    group_cols = ["run_id", "year", "month", "model", "phase"]
    agg = (
        df.groupby(group_cols, dropna=False)["elapsed_ms"]
        .agg(
            n_calls="count",
            p50_elapsed_ms="median",
            p90_elapsed_ms=lambda s: s.quantile(0.9),
        )
        .reset_index()
    )
    agg["n_calls"] = agg["n_calls"].fillna(0).astype(int)
    return agg[LATENCY_COLUMNS]
