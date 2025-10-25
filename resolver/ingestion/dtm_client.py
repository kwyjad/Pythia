"""Compact DTM connector used by CI smoke tests and production runs."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Iterable, List, Sequence

import pandas as pd
import yaml

from resolver.ingestion.dtm_auth import get_dtm_api_key
from resolver.ingestion.utils import month_start, stable_digest
from resolver.ingestion.utils.io import resolve_ingestion_window, resolve_output_path

LOG = logging.getLogger("resolver.ingestion.dtm_client")

INGESTION_ROOT = Path(__file__).resolve().parent
RESOLVER_ROOT = INGESTION_ROOT.parent
REPO_ROOT = RESOLVER_ROOT.parent
CONFIG_PATH = INGESTION_ROOT / "config" / "dtm.yml"
DEFAULT_OUTPUT = RESOLVER_ROOT / "staging" / "dtm_displacement.csv"
DIAGNOSTICS_ROOT = REPO_ROOT / "diagnostics"
INGESTION_DIAG_ROOT = DIAGNOSTICS_ROOT / "ingestion"
DTM_DIAG_DIR = INGESTION_DIAG_ROOT / "dtm"
REQUEST_LOG_PATH = DTM_DIAG_DIR / "request_log.jsonl"
SUMMARY_PATH = DTM_DIAG_DIR / "summary.json"
SAMPLE_PATH = DIAGNOSTICS_ROOT / "sample_dtm_displacement.csv"
CONNECTORS_REPORT_PATH = DIAGNOSTICS_ROOT / "connectors_report.jsonl"
SCHEMA_NAME = "dtm_displacement"
SCHEMA_VERSION = 1

CANONICAL_COLUMNS: Sequence[str] = (
    "source",
    "country_iso3",
    "admin1",
    "event_id",
    "as_of",
    "month_start",
    "value_type",
    "value",
    "unit",
    "method",
    "confidence",
    "raw_event_id",
    "raw_fields_json",
)


class SkipConnector(RuntimeError):
    """Exception raised when the connector should exit with status=skipped."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass
class ConnectorResult:
    rows: list[dict[str, Any]]
    summary: dict[str, Any]
    request_log: list[dict[str, Any]]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch monthly displacement flows from DTM")
    parser.add_argument("--debug", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--no-date-filter",
        action="store_true",
        help="Disable date filtering when querying the API",
    )
    parser.add_argument(
        "--strict-empty",
        action="store_true",
        help="Exit with a non-zero code if no rows are produced in online mode",
    )
    parser.add_argument(
        "--offline-smoke",
        action="store_true",
        help="Generate synthetic rows without contacting the API",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        help="Override the CSV output location",
    )
    return parser.parse_args(argv)


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    if not path.exists():
        return {
            "enabled": True,
            "api": {"countries": [], "admin_levels": ["admin0", "admin1"]},
        }
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    enabled = bool(data.get("enabled", True))
    api = data.get("api") or {}
    countries = list(api.get("countries") or [])
    admin_levels = list(api.get("admin_levels") or ["admin0", "admin1"])
    allowed_levels = {"admin0", "admin1", "admin2"}
    admin_levels = [level for level in admin_levels if level in allowed_levels]
    if not admin_levels:
        admin_levels = ["admin0", "admin1"]
    cfg = {"enabled": enabled, "api": {"countries": countries, "admin_levels": admin_levels}}
    return cfg


def compute_monthly_flows(frame: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Convert a stock time-series into month-on-month flows."""

    if frame.empty:
        return frame.iloc[0:0], False

    working = frame.copy()
    working["admin1"] = working["admin1"].fillna("")
    working["month_start"] = working["month_start"].apply(month_start)
    working = working.dropna(subset=["month_start"])  # type: ignore[arg-type]
    working.sort_values(["country_iso3", "admin1", "month_start", "as_of"], inplace=True)
    working = working.drop_duplicates(
        ["country_iso3", "admin1", "month_start"], keep="last"
    )

    flow_rows: list[dict[str, Any]] = []
    has_negative = False
    for (iso3, admin1), group in working.groupby(["country_iso3", "admin1"], sort=False):
        previous_value: float | None = None
        for _, row in group.sort_values("month_start").iterrows():
            value = float(row.get("value") or 0)
            bucket = row["month_start"]
            if isinstance(bucket, str):
                bucket = month_start(bucket)
            if bucket is None:
                continue
            as_of = row.get("as_of")
            if isinstance(as_of, str):
                text = as_of.replace("Z", "+00:00")
                try:
                    as_of_dt = datetime.fromisoformat(text)
                except ValueError:
                    as_of_dt = datetime.combine(bucket, datetime.min.time(), tzinfo=UTC)
            elif isinstance(as_of, datetime):
                as_of_dt = as_of
            else:
                as_of_dt = datetime.combine(bucket, datetime.min.time(), tzinfo=UTC)
            if previous_value is None:
                previous_value = value
                continue
            flow = value - previous_value
            if flow < 0:
                has_negative = True
            flow_rows.append(
                {
                    "country_iso3": iso3,
                    "admin1": admin1,
                    "month_start": bucket,
                    "as_of": as_of_dt,
                    "value": int(round(flow)),
                    "raw_event_id": row.get("raw_event_id"),
                    "raw_fields": row.get("raw_fields") or {},
                }
            )
            previous_value = value

    flows = pd.DataFrame(flow_rows)
    if flows.empty:
        return flows, has_negative
    flows["value"] = flows["value"].astype(int)
    return flows, has_negative


def _canonicalise_rows(flow_rows: pd.DataFrame) -> list[dict[str, Any]]:
    if flow_rows.empty:
        return []

    output: list[dict[str, Any]] = []
    flow_rows = flow_rows.copy()
    flow_rows["admin1"] = flow_rows["admin1"].fillna("")

    for _, row in flow_rows.iterrows():
        iso3 = str(row.get("country_iso3") or "").upper()
        admin1 = str(row.get("admin1") or "")
        bucket = row.get("month_start")
        if isinstance(bucket, str):
            bucket_date = month_start(bucket)
        else:
            bucket_date = bucket
        if not isinstance(bucket_date, date):
            continue
        month_iso = bucket_date.isoformat()
        as_of = row.get("as_of")
        if isinstance(as_of, datetime):
            as_of_dt = as_of.astimezone(UTC)
        elif isinstance(as_of, str):
            text = as_of.replace("Z", "+00:00")
            try:
                parsed = datetime.fromisoformat(text)
            except ValueError:
                parsed = datetime.combine(bucket_date, datetime.min.time(), tzinfo=UTC)
            as_of_dt = parsed.astimezone(UTC)
        else:
            as_of_dt = datetime.combine(bucket_date, datetime.min.time(), tzinfo=UTC)
        value = int(row.get("value") or 0)
        raw_fields = dict(row.get("raw_fields") or {})
        raw_fields.setdefault("month_start", month_iso)
        raw_fields.setdefault("value_used", value)
        raw_event_id = str(row.get("raw_event_id") or "")
        event_id = stable_digest((iso3, admin1, month_iso, "dtm_stock_to_flow"), length=16)
        if not raw_event_id:
            raw_event_id = event_id
        canonical = {
            "source": "dtm",
            "country_iso3": iso3,
            "admin1": admin1,
            "event_id": event_id,
            "as_of": as_of_dt.isoformat().replace("+00:00", "Z"),
            "month_start": month_iso,
            "value_type": "new_displaced",
            "value": value,
            "unit": "people",
            "method": "dtm_stock_to_flow",
            "confidence": "medium",
            "raw_event_id": raw_event_id,
            "raw_fields_json": json.dumps(raw_fields, sort_keys=True, separators=(",", ":")),
        }
        output.append(canonical)

    # Build admin0 rollups from admin1 flows.
    grouped = flow_rows.groupby(["country_iso3", "month_start"], sort=False)
    for (iso3, bucket), group in grouped:
        if group.empty:
            continue
        bucket_date = bucket if isinstance(bucket, date) else month_start(bucket)
        if not isinstance(bucket_date, date):
            continue
        month_iso = bucket_date.isoformat()
        value_sum = int(group["value"].sum())
        def _coerce_as_of(value: Any) -> datetime:
            if isinstance(value, datetime):
                return value
            text = str(value or "").replace("Z", "+00:00")
            try:
                parsed = datetime.fromisoformat(text)
            except ValueError:
                parsed = datetime.combine(bucket_date, datetime.min.time(), tzinfo=UTC)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            return parsed

        as_of_dt = max(
            (_coerce_as_of(row) for row in group["as_of"]),
            default=datetime.combine(bucket_date, datetime.min.time(), tzinfo=UTC),
        )
        component_ids = [str(val or "") for val in group.get("raw_event_id", []) if val]
        event_id = stable_digest((iso3, "", month_iso, "dtm_stock_to_flow"), length=16)
        raw_fields = {
            "aggregation": "admin1_sum",
            "component_raw_event_ids": component_ids,
            "value_used": value_sum,
            "month_start": month_iso,
        }
        canonical = {
            "source": "dtm",
            "country_iso3": str(iso3).upper(),
            "admin1": "",
            "event_id": event_id,
            "as_of": as_of_dt.astimezone(UTC).isoformat().replace("+00:00", "Z"),
            "month_start": month_iso,
            "value_type": "new_displaced",
            "value": value_sum,
            "unit": "people",
            "method": "dtm_stock_to_flow",
            "confidence": "medium",
            "raw_event_id": event_id,
            "raw_fields_json": json.dumps(raw_fields, sort_keys=True, separators=(",", ":")),
        }
        output.append(canonical)

    output.sort(key=lambda row: (row["country_iso3"], row["admin1"], row["month_start"]))
    return output


def _offline_smoke_frame() -> pd.DataFrame:
    now = datetime.now(tz=UTC)
    records: list[dict[str, Any]] = []
    samples = [
        ("COL", "Antioquia", date(2024, 1, 15), 1200),
        ("COL", "Antioquia", date(2024, 2, 15), 1600),
        ("COL", "Antioquia", date(2024, 3, 15), 1400),
        ("COL", "Chocó", date(2024, 1, 20), 800),
        ("COL", "Chocó", date(2024, 2, 20), 950),
        ("NGA", "Borno", date(2024, 1, 12), 5000),
        ("NGA", "Borno", date(2024, 2, 12), 5200),
        ("NGA", "Borno", date(2024, 3, 12), 5150),
        ("NGA", "Adamawa", date(2024, 1, 10), 2300),
        ("NGA", "Adamawa", date(2024, 2, 10), 2600),
    ]
    for iso3, admin1, reporting_date, value in samples:
        bucket = month_start(reporting_date)
        as_of = datetime.combine(reporting_date, datetime.min.time(), tzinfo=UTC)
        records.append(
            {
                "country_iso3": iso3,
                "admin1": admin1,
                "month_start": bucket,
                "as_of": as_of,
                "value": value,
                "raw_event_id": f"offline:{iso3}:{admin1}:{bucket.isoformat()}",
                "raw_fields": {
                    "reportingDate": reporting_date.isoformat(),
                    "admin1": admin1,
                    "value": value,
                    "generated_at": now.isoformat().replace("+00:00", "Z"),
                },
            }
        )
    return pd.DataFrame.from_records(records)


def build_rows(
    cfg: dict[str, Any],
    *,
    window_start: date | None,
    window_end: date | None,
    no_date_filter: bool = False,
    offline_smoke: bool = False,
) -> ConnectorResult:
    summary: dict[str, Any] = {
        "window_start": window_start.isoformat() if window_start else None,
        "window_end": window_end.isoformat() if window_end else None,
        "has_negative_flows": False,
        "rows_in": 0,
        "rows_out": 0,
        "planned_countries": 0,
        "fetched_countries": 0,
        "mode": "offline_smoke" if offline_smoke else "online",
        "status": "ok",
        "reason": None,
    }

    if offline_smoke:
        frame = _offline_smoke_frame()
        flows, has_negative = compute_monthly_flows(frame)
        rows = _canonicalise_rows(flows)
        summary.update(
            {
                "rows_in": len(frame),
                "rows_out": len(rows),
                "planned_countries": 2,
                "fetched_countries": 2,
                "has_negative_flows": has_negative,
            }
        )
        return ConnectorResult(rows=rows, summary=summary, request_log=[])

    if not cfg.get("enabled", True):
        raise SkipConnector("disabled", "DTM connector disabled via configuration")

    api_cfg = cfg.get("api") or {}
    admin_levels = list(api_cfg.get("admin_levels") or ["admin0", "admin1"])

    try:
        import dtmapi  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise SkipConnector("sdk_missing", "dtmapi SDK is not installed") from exc

    api_key = get_dtm_api_key()
    if not api_key:
        raise SkipConnector("auth_missing", "DTM API key not configured")

    countries: list[str] = []
    configured_countries = list(api_cfg.get("countries") or [])
    if configured_countries:
        countries = [str(code).strip().upper() for code in configured_countries if code]
    else:  # Discover from the SDK if possible.
        discovery = getattr(dtmapi, "get_all_countries", None)
        if callable(discovery):
            try:
                discovered = discovery()
            except Exception as exc:  # pragma: no cover - depends on SDK
                raise SkipConnector("discovery_failed", f"Discovery failed: {exc}") from exc
            if isinstance(discovered, pd.DataFrame):
                countries = [str(code).strip().upper() for code in discovered.get("iso3", []) if code]
            elif isinstance(discovered, Iterable):
                for item in discovered:
                    if isinstance(item, str):
                        code = item.strip().upper()
                        if len(code) == 3:
                            countries.append(code)
                    elif isinstance(item, dict):
                        iso_candidate = item.get("iso3") or item.get("iso_code") or item.get("iso")
                        if iso_candidate:
                            code = str(iso_candidate).strip().upper()
                            if len(code) == 3:
                                countries.append(code)
        countries = sorted(set(countries))
        if not countries:
            raise SkipConnector("discovery_empty", "DTM discovery returned no countries")

    summary["planned_countries"] = len(countries)

    client_ctor = getattr(dtmapi, "Client", None) or getattr(dtmapi, "DTMClient", None)
    if not callable(client_ctor):  # pragma: no cover - depends on SDK
        raise SkipConnector("sdk_unsupported", "dtmapi module missing Client constructor")

    client = client_ctor(api_key=api_key)
    LOG.info(
        "Initialised dtmapi client for %d countries (admin levels: %s)",
        len(countries),
        ",".join(admin_levels),
    )

    request_log: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    fetched_countries: set[str] = set()

    for iso3 in countries:  # pragma: no cover - requires real SDK or heavy mocks
        params: dict[str, Any] = {"country_iso3": iso3, "admin_levels": admin_levels}
        if not no_date_filter:
            if window_start:
                params["from_reporting_date"] = window_start.isoformat()
            if window_end:
                params["to_reporting_date"] = window_end.isoformat()
        start_time = datetime.now(tz=UTC)
        rows_in_country: list[dict[str, Any]] = []
        for level in admin_levels:
            fetcher = getattr(client, f"fetch_{level}", None)
            if not callable(fetcher):
                continue
            response = fetcher(iso3, params)
            if isinstance(response, pd.DataFrame):
                df = response
            elif isinstance(response, Iterable):
                df = pd.DataFrame(list(response))
            else:
                df = pd.DataFrame()
            if not df.empty:
                df["country_iso3"] = iso3
                df["admin1"] = df.get("admin1") or df.get("adminName") or ""
                df["month_start"] = df.get("reportingDate")
                df["as_of"] = df.get("reportingDate")
                df["value"] = df.get("numPresentIdpInd") or df.get("value") or 0
                df["raw_event_id"] = df.get("id") or df.get("event_id")
                df["raw_fields"] = [
                    {
                        "reportingDate": row.get("reportingDate"),
                        "roundNumber": row.get("roundNumber"),
                        "operation": row.get("operation"),
                        "admin1": row.get("admin1") or row.get("adminName"),
                        "value": row.get("numPresentIdpInd") or row.get("value"),
                    }
                    for _, row in df.iterrows()
                ]
                frames.append(df)
                rows_in_country.append({"level": level, "rows": int(len(df))})
        elapsed = (datetime.now(tz=UTC) - start_time).total_seconds() * 1000
        request_log.append(
            {
                "country_iso3": iso3,
                "admin_levels": admin_levels,
                "params": params,
                "rows_per_level": rows_in_country,
                "elapsed_ms": round(elapsed, 2),
            }
        )
        if rows_in_country:
            fetched_countries.add(iso3)

    if not frames:
        summary.update({"rows_in": 0, "rows_out": 0, "fetched_countries": 0})
        return ConnectorResult(rows=[], summary=summary, request_log=request_log)

    combined = pd.concat(frames, ignore_index=True)
    flows, has_negative = compute_monthly_flows(combined)
    rows = _canonicalise_rows(flows)
    summary.update(
        {
            "rows_in": int(len(combined)),
            "rows_out": len(rows),
            "fetched_countries": len(fetched_countries),
            "has_negative_flows": has_negative,
        }
    )
    return ConnectorResult(rows=rows, summary=summary, request_log=request_log)


def _ensure_directories(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    DTM_DIAG_DIR.mkdir(parents=True, exist_ok=True)
    INGESTION_DIAG_ROOT.mkdir(parents=True, exist_ok=True)
    DIAGNOSTICS_ROOT.mkdir(parents=True, exist_ok=True)


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANONICAL_COLUMNS)
        writer.writeheader()
        for row in rows:
            serialised = row.copy()
            serialised["value"] = int(serialised["value"])
            writer.writerow(serialised)


def _write_meta(path: Path, row_count: int) -> None:
    meta = {
        "schema": SCHEMA_NAME,
        "version": SCHEMA_VERSION,
        "generated_at": datetime.now(tz=UTC).isoformat().replace("+00:00", "Z"),
        "row_count": row_count,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)


def _write_summary(summary: dict[str, Any]) -> None:
    with SUMMARY_PATH.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


def _write_request_log(entries: list[dict[str, Any]]) -> None:
    if not entries:
        return
    with REQUEST_LOG_PATH.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, sort_keys=True))
            handle.write("\n")


def _write_sample(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SAMPLE_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANONICAL_COLUMNS)
        writer.writeheader()
        for row in rows[:20]:
            writer.writerow(row)


def _append_connectors_report(
    summary: dict[str, Any],
    output_path: Path,
    started_at: datetime,
    ended_at: datetime,
) -> None:
    CONNECTORS_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "connector": "dtm",
        "status": summary.get("status", "error"),
        "reason": summary.get("reason"),
        "rows_out": summary.get("rows_out", 0),
        "output_path": str(output_path),
        "started_at": started_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "ended_at": ended_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "elapsed_s": round((ended_at - started_at).total_seconds(), 3),
    }
    with CONNECTORS_REPORT_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True))
        handle.write("\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    started_at = datetime.now(tz=UTC)
    output_path = Path(args.output).expanduser() if args.output else resolve_output_path(DEFAULT_OUTPUT)
    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    _ensure_directories(output_path)

    window_start, window_end = resolve_ingestion_window()

    summary: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []
    request_log: list[dict[str, Any]] = []
    exit_code = 0

    try:
        cfg = load_config()
        result = build_rows(
            cfg,
            window_start=window_start,
            window_end=window_end,
            no_date_filter=args.no_date_filter,
            offline_smoke=args.offline_smoke,
        )
        rows = result.rows
        summary = result.summary
        request_log = result.request_log
    except SkipConnector as exc:
        LOG.warning("DTM connector skipped: %s", exc)
        summary = {
            "status": "skipped",
            "reason": exc.reason,
            "rows_out": 0,
            "rows_in": 0,
            "has_negative_flows": False,
            "mode": "offline_smoke" if args.offline_smoke else "online",
            "window_start": window_start.isoformat() if window_start else None,
            "window_end": window_end.isoformat() if window_end else None,
            "planned_countries": 0,
            "fetched_countries": 0,
        }
    except Exception as exc:  # pragma: no cover - defensive guard
        LOG.exception("DTM connector failed: %s", exc)
        summary = {
            "status": "error",
            "reason": str(exc),
            "rows_out": 0,
            "rows_in": 0,
            "has_negative_flows": False,
            "mode": "offline_smoke" if args.offline_smoke else "online",
            "window_start": window_start.isoformat() if window_start else None,
            "window_end": window_end.isoformat() if window_end else None,
            "planned_countries": 0,
            "fetched_countries": 0,
        }
        exit_code = 1

    if summary is None:
        summary = {
            "status": "error",
            "reason": "unexpected_state",
            "rows_out": 0,
        }

    summary.setdefault("rows_out", len(rows))
    summary.setdefault("rows_in", 0)
    summary.setdefault("has_negative_flows", False)
    summary.setdefault("mode", "offline_smoke" if args.offline_smoke else "online")

    if rows:
        _write_csv(rows, output_path)
        _write_meta(meta_path, len(rows))
        _write_sample(rows)
    elif args.strict_empty and summary.get("status") == "ok" and not args.offline_smoke:
        exit_code = 1

    _write_request_log(request_log)
    _write_summary(summary)

    ended_at = datetime.now(tz=UTC)
    _append_connectors_report(summary, output_path, started_at, ended_at)

    return exit_code


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
