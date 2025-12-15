"""Evaluate SPD aggregation methods using resolved months in DuckDB."""
from __future__ import annotations

import argparse
import json
import os
from datetime import date, datetime
from typing import Dict, Iterable, List, NamedTuple, Optional, Tuple

import pandas as pd

from forecaster import scoring
from forecaster.cli import SPD_CLASS_BINS_FATALITIES, SPD_CLASS_BINS_PA
from resolver.db import duckdb_io


DEFAULT_MODELS = ("ensemble_mean_v2", "ensemble_bayesmc_v2")


class ForecastPoint(NamedTuple):
    question_id: int
    horizon_m: int
    model_name: str
    metric: str
    hazard_code: Optional[str]
    iso3: str
    target_month: str
    month_label: str
    probs: List[float]


class TruthPoint(NamedTuple):
    iso3: str
    hazard_code: Optional[str]
    metric: str
    month_label: str
    value: float


class ScoreRow(NamedTuple):
    question_id: int
    iso3: str
    hazard_code: Optional[str]
    metric: str
    month_label: str
    horizon_m: int
    model_name: str
    true_value: float
    true_bucket: int
    brier: float
    log_score: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", dest="db_url", default=duckdb_io.DEFAULT_DB_URL)
    parser.add_argument("--run-id", dest="run_id", type=int)
    parser.add_argument(
        "--model-names",
        default=",".join(DEFAULT_MODELS),
        help="Comma-delimited list of model names to compare.",
    )
    parser.add_argument("--metric")
    parser.add_argument("--hazard")
    parser.add_argument("--iso3")
    parser.add_argument("--out-dir", default=os.path.join("debug", "eval"))
    return parser.parse_args()


def _add_months(month: date, months: int) -> date:
    year = month.year + (month.month - 1 + months) // 12
    new_month = (month.month - 1 + months) % 12 + 1
    return date(year, new_month, 1)


def _parse_month(value: object) -> date:
    if isinstance(value, date):
        return date(value.year, value.month, 1)
    if isinstance(value, datetime):
        return date(value.year, value.month, 1)
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value)
        return date(parsed.year, parsed.month, 1)
    raise ValueError(f"Unrecognized month value: {value}")


def _resolve_run_id(conn, explicit_run_id: Optional[int]) -> int:
    if explicit_run_id is not None:
        return explicit_run_id

    table_info = conn.execute("PRAGMA table_info('forecasts_ensemble')").fetchall()
    columns = {row[1].lower() for row in table_info}
    timestamp_col = None
    for candidate in ("timestamp", "created_at"):
        if candidate in columns:
            timestamp_col = candidate
            break

    if timestamp_col:
        query = f"""
            SELECT run_id
            FROM forecasts_ensemble
            ORDER BY {timestamp_col} DESC
            LIMIT 1
        """
        run_id = conn.execute(query).fetchone()
        if run_id:
            return int(run_id[0])

    run_id = conn.execute(
        """
        SELECT run_id
        FROM forecasts_ensemble
        GROUP BY 1
        ORDER BY COUNT(*) DESC
        LIMIT 1
        """
    ).fetchone()
    if not run_id:
        raise RuntimeError("No forecasts_ensemble rows available to select a run_id.")
    return int(run_id[0])


def _load_forecasts(
    conn,
    run_id: int,
    model_names: Iterable[str],
    metric_filter: Optional[str],
    hazard_filter: Optional[str],
    iso3_filter: Optional[str],
) -> list[ForecastPoint]:
    models_sql = ",".join([f"'{m}'" for m in model_names])
    query = f"""
        SELECT
            f.question_id,
            f.model_name,
            f.horizon_m,
            f.class_bin,
            f.p,
            q.metric,
            COALESCE(f.hazard_code, q.hazard_code) AS hazard_code,
            q.iso3,
            q.target_month
        FROM forecasts_ensemble f
        JOIN questions q USING(question_id)
        WHERE f.status = 'ok'
          AND f.run_id = {run_id}
          AND f.model_name IN ({models_sql})
    """

    filters = []
    if metric_filter:
        filters.append("LOWER(q.metric) = LOWER(?)")
    if hazard_filter:
        filters.append("LOWER(COALESCE(f.hazard_code, q.hazard_code)) = LOWER(?)")
    if iso3_filter:
        filters.append("LOWER(q.iso3) = LOWER(?)")
    if filters:
        query += " AND " + " AND ".join(filters)

    params: list[object] = []
    if metric_filter:
        params.append(metric_filter)
    if hazard_filter:
        params.append(hazard_filter)
    if iso3_filter:
        params.append(iso3_filter)

    df = conn.execute(query, params).fetch_df()
    if df.empty:
        return []

    grouped: Dict[Tuple[int, str, int], Dict[str, object]] = {}
    bin_orders = {
        "PA": {label: idx for idx, label in enumerate(SPD_CLASS_BINS_PA)},
        "FATALITIES": {label: idx for idx, label in enumerate(SPD_CLASS_BINS_FATALITIES)},
    }

    for _, row in df.iterrows():
        metric = str(row["metric"]).upper()
        if metric not in bin_orders:
            continue
        key = (int(row["question_id"]), str(row["model_name"]), int(row["horizon_m"]))
        hazard_code = row["hazard_code"]
        hazard_code = str(hazard_code) if hazard_code is not None else None
        iso3 = str(row["iso3"])
        target_month = _parse_month(row["target_month"])
        month_label = _add_months(target_month, int(row["horizon_m"]) - 1).strftime("%Y-%m")

        if key not in grouped:
            grouped[key] = {
                "metric": metric,
                "hazard_code": hazard_code,
                "iso3": iso3,
                "target_month": target_month.strftime("%Y-%m"),
                "month_label": month_label,
                "probs": [0.0] * len(bin_orders[metric]),
            }

        class_bin = str(row["class_bin"])
        bin_index = bin_orders[metric].get(class_bin)
        if bin_index is not None:
            grouped[key]["probs"][bin_index] = float(row["p"])

    forecasts: list[ForecastPoint] = []
    for (question_id, model_name, horizon_m), payload in grouped.items():
        forecasts.append(
            ForecastPoint(
                question_id=question_id,
                horizon_m=horizon_m,
                model_name=model_name,
                metric=payload["metric"],
                hazard_code=payload["hazard_code"],
                iso3=payload["iso3"],
                target_month=payload["target_month"],
                month_label=payload["month_label"],
                probs=scoring.normalize_probs(payload["probs"]),
            )
        )
    return forecasts


def _detect_truth_table(conn) -> Optional[Tuple[str, dict]]:
    tables = [row[0] for row in conn.execute("SHOW TABLES").fetchall()]
    for table in tables:
        info = conn.execute(f"PRAGMA table_info('{table}')").fetchall()
        columns = [row[1].lower() for row in info]
        iso3_col = next((c for c in columns if c == "iso3"), None)
        hazard_col = next((c for c in columns if c in {"hazard_code", "hazard"}), None)
        metric_col = next((c for c in columns if c == "metric"), None)
        month_col = next((c for c in columns if c in {"ym", "month", "period"}), None)
        value_col = next((c for c in columns if c in {"value", "amount"}), None)
        if iso3_col and hazard_col and metric_col and month_col and value_col:
            return table, {
                "iso3": iso3_col,
                "hazard": hazard_col,
                "metric": metric_col,
                "month": month_col,
                "value": value_col,
                "tier": next((c for c in columns if c in {"source_tier", "tier"}), None),
                "as_of": next((c for c in columns if c in {"as_of", "updated_at"}), None),
            }
    return None


def _load_truth(conn, forecasts: Iterable[ForecastPoint]) -> dict[Tuple[str, Optional[str], str, str], TruthPoint]:
    forecasts_list = list(forecasts)
    if not forecasts_list:
        return {}

    detection = _detect_truth_table(conn)
    if not detection:
        return {}

    table, columns = detection
    keys_df = pd.DataFrame(
        {
            "iso3": [f.iso3 for f in forecasts_list],
            "hazard_code": [f.hazard_code or "" for f in forecasts_list],
            "metric": [f.metric for f in forecasts_list],
            "month_label": [f.month_label for f in forecasts_list],
        }
    ).drop_duplicates()

    conn.register("eval_keys", keys_df)
    hazard_expr = f"COALESCE(LOWER(CAST(t.{columns['hazard']} AS VARCHAR)), '')"
    join_query = f"""
        SELECT t.*, k.month_label, k.iso3 AS key_iso3, k.hazard_code AS key_hazard, k.metric AS key_metric
        FROM eval_keys k
        JOIN {table} t
          ON LOWER(CAST(t.{columns['iso3']} AS VARCHAR)) = LOWER(k.iso3)
         AND {hazard_expr} = COALESCE(LOWER(k.hazard_code), '')
         AND LOWER(CAST(t.{columns['metric']} AS VARCHAR)) = LOWER(k.metric)
         AND strftime(t.{columns['month']}, '%Y-%m') = k.month_label
    """

    truth_df = conn.execute(join_query).fetch_df()
    if truth_df.empty:
        return {}

    tier_col = columns.get("tier")
    as_of_col = columns.get("as_of")
    sort_fields: list[Tuple[str, bool]] = []
    if tier_col and tier_col in truth_df.columns:
        sort_fields.append((tier_col, True))
    if as_of_col and as_of_col in truth_df.columns:
        sort_fields.append((as_of_col, False))

    if sort_fields:
        truth_df = truth_df.sort_values(
            by=[name for name, _ in sort_fields],
            ascending=[asc for _, asc in sort_fields],
        )

    truth_map: dict[Tuple[str, Optional[str], str, str], TruthPoint] = {}
    value_col = columns["value"]
    for _, row in truth_df.iterrows():
        key = (
            str(row["key_iso3"]),
            str(row["key_hazard"]) if row["key_hazard"] else None,
            str(row["key_metric"]),
            str(row["month_label"]),
        )
        if key in truth_map:
            continue
        truth_map[key] = TruthPoint(
            iso3=key[0],
            hazard_code=key[1],
            metric=key[2],
            month_label=key[3],
            value=float(row[value_col]),
        )
    return truth_map


def _score(
    forecasts: Iterable[ForecastPoint],
    truth_map: dict[Tuple[str, Optional[str], str, str], TruthPoint],
) -> list[ScoreRow]:
    results: list[ScoreRow] = []
    for fc in forecasts:
        key = (fc.iso3, fc.hazard_code, fc.metric, fc.month_label)
        truth = truth_map.get(key)
        if not truth:
            continue
        bucket = scoring.bucket_index_from_value(fc.metric, truth.value)
        brier = scoring.multiclass_brier(fc.probs, bucket)
        log = scoring.log_score(fc.probs, bucket)
        results.append(
            ScoreRow(
                question_id=fc.question_id,
                iso3=fc.iso3,
                hazard_code=fc.hazard_code,
                metric=fc.metric,
                month_label=fc.month_label,
                horizon_m=fc.horizon_m,
                model_name=fc.model_name,
                true_value=truth.value,
                true_bucket=bucket,
                brier=brier,
                log_score=log,
            )
        )
    return results


def _summaries(rows: list[ScoreRow]) -> dict:
    df = pd.DataFrame(rows)
    summaries: dict[str, object] = {"n_scored": len(rows)}
    if df.empty:
        return summaries
    summaries["overall"] = (
        df.groupby("model_name")[ ["brier", "log_score"] ].mean().reset_index().to_dict(orient="records")
    )
    summaries["by_hazard"] = (
        df.groupby(["hazard_code", "model_name"])[["brier", "log_score"]]
        .mean()
        .reset_index()
        .sort_values(["hazard_code", "model_name"])
        .to_dict(orient="records")
    )
    summaries["by_metric"] = (
        df.groupby(["metric", "model_name"])[["brier", "log_score"]]
        .mean()
        .reset_index()
        .sort_values(["metric", "model_name"])
        .to_dict(orient="records")
    )
    counts = df.groupby("model_name").size().reset_index(name="n")
    summaries["counts"] = counts.to_dict(orient="records")
    if set(DEFAULT_MODELS).issubset(counts["model_name"].tolist()):
        mean_scores = summaries["overall"]
        mean_lookup = {row["model_name"]: row for row in mean_scores}
        deltas = {}
        for metric in ("brier", "log_score"):
            deltas[metric] = (
                mean_lookup.get(DEFAULT_MODELS[1], {}).get(metric)
                - mean_lookup.get(DEFAULT_MODELS[0], {}).get(metric)
            )
        summaries["delta"] = deltas
    return summaries


def _print_summary(rows: list[ScoreRow]) -> None:
    if not rows:
        print("No resolved forecasts found for evaluation. 0 scored points.")
        return
    df = pd.DataFrame(rows)
    print("### SPD aggregation evaluation")
    print("| model_name | mean_brier | mean_log_score | n |")
    print("| --- | --- | --- | --- |")
    overall = (
        df.groupby("model_name")[ ["brier", "log_score"] ]
        .mean()
        .reset_index()
        .sort_values("model_name")
    )
    counts = df.groupby("model_name").size().reset_index(name="n")
    overall = overall.merge(counts, on="model_name")
    for _, row in overall.iterrows():
        print(
            f"| {row['model_name']} | {row['brier']:.4f} | {row['log_score']:.4f} | {int(row['n'])} |"
        )

    print("\n| hazard_code | model_name | mean_brier | mean_log_score |")
    print("| --- | --- | --- | --- |")
    hazard = (
        df.groupby(["hazard_code", "model_name"])[["brier", "log_score"]]
        .mean()
        .reset_index()
        .sort_values(["hazard_code", "model_name"])
    )
    for _, row in hazard.iterrows():
        hazard_code = row["hazard_code"] if row["hazard_code"] is not None else ""
        print(
            f"| {hazard_code} | {row['model_name']} | {row['brier']:.4f} | {row['log_score']:.4f} |"
        )

    print("\n| metric | model_name | mean_brier | mean_log_score |")
    print("| --- | --- | --- | --- |")
    metric = (
        df.groupby(["metric", "model_name"])[["brier", "log_score"]]
        .mean()
        .reset_index()
        .sort_values(["metric", "model_name"])
    )
    for _, row in metric.iterrows():
        print(
            f"| {row['metric']} | {row['model_name']} | {row['brier']:.4f} | {row['log_score']:.4f} |"
        )

    models_present = sorted(df["model_name"].unique())
    if all(m in models_present for m in DEFAULT_MODELS):
        means = overall.set_index("model_name").loc[list(DEFAULT_MODELS)]
        delta_brier = means.iloc[1]["brier"] - means.iloc[0]["brier"]
        delta_log = means.iloc[1]["log_score"] - means.iloc[0]["log_score"]
        print("\n| metric | bayesmc_minus_mean |")
        print("| --- | --- |")
        print(f"| brier | {delta_brier:.4f} |")
        print(f"| log_score | {delta_log:.4f} |")


def main() -> None:
    args = _parse_args()
    model_names = [m.strip() for m in args.model_names.split(",") if m.strip()]
    conn = duckdb_io.get_db(args.db_url)
    try:
        run_id = _resolve_run_id(conn, args.run_id)
        forecasts = _load_forecasts(
            conn,
            run_id,
            model_names,
            args.metric,
            args.hazard,
            args.iso3,
        )
        truth_map = _load_truth(conn, forecasts)
    finally:
        duckdb_io.close_db(conn)

    scored_rows = _score(forecasts, truth_map)
    _print_summary(scored_rows)

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"spd_eval__{run_id}.csv")
    json_path = os.path.join(out_dir, f"spd_eval_summary__{run_id}.json")

    pd.DataFrame(scored_rows, columns=ScoreRow._fields).to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_summaries(scored_rows), f, indent=2)

    print(f"Wrote scores to {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
