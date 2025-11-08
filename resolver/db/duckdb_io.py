"""Helpers for connecting to and writing into the DuckDB backend."""

from __future__ import annotations

import json
import numbers
import os
import platform
import re
import sys
import uuid
import datetime as dt
import logging
from dataclasses import dataclass
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import pandas as pd

from resolver.db._duckdb_available import (
    DUCKDB_AVAILABLE,
    duckdb_unavailable_reason,
    get_duckdb,
)

if DUCKDB_AVAILABLE:  # pragma: no branch - cache the imported module once
    duckdb = get_duckdb()
else:  # pragma: no cover - exercised in environments without DuckDB

    class _DuckDBStub:
        """Minimal stub that exposes DuckDB exception attributes for typing."""

        __slots__ = ()
        __version__ = "unavailable"
        Error = Exception
        CatalogException = Exception
        DependencyException = Exception
        ConnectionException = Exception
        NotImplementedException = Exception
        ParserException = Exception
        TransactionException = Exception

        def __getattr__(self, name: str) -> object:
            raise RuntimeError(
                "DuckDB is required for this operation but is not installed. "
                f"Attempted to access attribute '{name}'."
            )

    duckdb = _DuckDBStub()

_DUCKDB_ERROR = getattr(duckdb, "Error", Exception)
_CATALOG_EXC = getattr(duckdb, "CatalogException", _DUCKDB_ERROR)
_DEPEND_EXC = getattr(duckdb, "DependencyException", _DUCKDB_ERROR)
_CONN_EXC = getattr(duckdb, "ConnectionException", _DUCKDB_ERROR)
_NOTIMPL_EXC = getattr(duckdb, "NotImplementedException", _DUCKDB_ERROR)
_PARSER_EXC = getattr(duckdb, "ParserException", _DUCKDB_ERROR)
_TXN_EXC = getattr(duckdb, "TransactionException", _DUCKDB_ERROR)

_SAVEPOINT_PROBED = False
_SAVEPOINT_WORKS = False
_SCHEMA_EXC_TUPLE = (
    _DUCKDB_ERROR,
    _CATALOG_EXC,
    _DEPEND_EXC,
    _CONN_EXC,
    _NOTIMPL_EXC,
)

from resolver.common import compute_series_semantics, dict_counts, df_schema
from resolver.db.conn_shared import (
    canonicalize_duckdb_target as _shared_canonicalize_duckdb_target,
    get_shared_duckdb_conn,
)
from resolver.helpers.series_semantics import normalize_series_semantics
from resolver.diag.diagnostics import (
    diag_enabled,
    dump_counts,
    dump_table_meta,
    get_logger as get_diag_logger,
    log_json,
)

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "db" / "schema.sql"

FACTS_RESOLVED_KEY_COLUMNS = [
    "event_id",
    "iso3",
    "hazard_code",
    "metric",
    "as_of_date",
    "publication_date",
    "source_id",
    "series_semantics",
    "ym",
]
FACTS_DELTAS_KEY_COLUMNS = [
    "iso3",
    "hazard_code",
    "metric",
    "as_of",
    "ym",
    "series_semantics",
]
FACTS_RESOLVED_KEY = FACTS_RESOLVED_KEY_COLUMNS  # Backwards compatibility
FACTS_DELTAS_KEY = FACTS_DELTAS_KEY_COLUMNS
TABLE_KEY_SPECS: dict[str, dict[str, object]] = {
    "facts_resolved": {
        "columns": FACTS_RESOLVED_KEY_COLUMNS,
        "primary": "pk_facts_resolved_series",
        "unique": "ux_facts_resolved_series",
    },
    "facts_deltas": {
        "columns": FACTS_DELTAS_KEY_COLUMNS,
        "primary": "pk_facts_deltas_series",
        "unique": "ux_facts_deltas_series",
    },
}
_COERCE_NUMERIC_COLUMNS: dict[str, list[str]] = {
    "facts_resolved": ["value"],
    "facts_deltas": ["value_new", "value_stock"],
}
DATE_STRING_COLUMNS: dict[str, tuple[str, ...]] = {
    "facts_resolved": ("as_of_date", "publication_date"),
    "facts_deltas": ("as_of",),
}
DEFAULT_DB_URL = os.environ.get(
    "RESOLVER_DB_URL", f"duckdb:///{ROOT / 'db' / 'resolver.duckdb'}"
)

_DB_CACHE: dict[str, "duckdb.DuckDBPyConnection"] = {}


@dataclass
class UpsertResult:
    """Structured counts returned after an upsert operation."""

    table: str
    rows_in: int
    rows_written: int
    rows_before: int
    rows_after: int
    rows_delta: int
    matched_existing: int | None = None

    def to_dict(self) -> dict[str, int | None | str]:
        return {
            "table": self.table,
            "rows_in": int(self.rows_in),
            "rows_written": int(self.rows_written),
            "rows_before": int(self.rows_before),
            "rows_after": int(self.rows_after),
            "rows_delta": int(self.rows_delta),
            "matched_existing": None if self.matched_existing is None else int(self.matched_existing),
        }

    def __int__(self) -> int:  # pragma: no cover - compatibility shim
        return int(self.rows_written)

    def __repr__(self) -> str:  # pragma: no cover - logging helper
        return (
            "UpsertResult(table={table!r}, rows_in={rows_in}, rows_written={rows_written}, "
            "rows_before={rows_before}, rows_after={rows_after}, rows_delta={rows_delta}, "
            "matched_existing={matched_existing})"
        ).format(
            table=self.table,
            rows_in=self.rows_in,
            rows_written=self.rows_written,
            rows_before=self.rows_before,
            rows_after=self.rows_after,
            rows_delta=self.rows_delta,
            matched_existing=self.matched_existing,
        )


LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:  # pragma: no cover - avoid "No handler" warnings in tests
    LOGGER.addHandler(logging.NullHandler())
DEBUG_ENABLED = os.getenv("RESOLVER_DEBUG") == "1"
if DEBUG_ENABLED:
    LOGGER.setLevel(logging.DEBUG)

DIAG_LOGGER = get_diag_logger(f"{__name__}.diag")
if diag_enabled():
    try:
        log_json(
            DIAG_LOGGER,
            "module_import",
            file=__file__,
            python=sys.version.split()[0],
            platform=platform.platform(),
            duckdb_version=getattr(duckdb, "__version__", "unknown"),
            duckdb_available=DUCKDB_AVAILABLE,
            duckdb_error=duckdb_unavailable_reason() if not DUCKDB_AVAILABLE else None,
        )
    except Exception as exc:  # pragma: no cover - diagnostics only
        log_json(
            DIAG_LOGGER,
            "module_import_error",
            file=__file__,
            error=repr(exc),
        )

_HEALING_WARNED: set[tuple[str, tuple[str, ...]]] = set()
_WARNED_EXPLICIT_OVERRIDE: set[tuple[str, str]] = set()


def _connection_in_transaction(conn: "duckdb.DuckDBPyConnection") -> bool:
    """Return True if ``conn`` is currently inside an explicit transaction."""

    try:
        first = conn.execute("SELECT current_transaction_id()").fetchone()[0]
        second = conn.execute("SELECT current_transaction_id()").fetchone()[0]
    except Exception:  # pragma: no cover - defensive probe
        LOGGER.debug("duckdb.ddl.txn_probe_failed", exc_info=True)
        return False
    return first == second


def _is_savepoint_syntax_error(exc: Exception) -> bool:
    s = f"{type(exc).__name__}: {exc}".lower()
    return "savepoint" in s and ("syntax error" in s or "parser error" in s or "not implemented" in s)


def _record_savepoint_support(supported: bool) -> None:
    global _SAVEPOINT_PROBED, _SAVEPOINT_WORKS
    _SAVEPOINT_PROBED = True
    _SAVEPOINT_WORKS = supported


def _savepoints_supported() -> bool:
    return _SAVEPOINT_PROBED and _SAVEPOINT_WORKS


def _savepoints_certainly_unsupported() -> bool:
    return _SAVEPOINT_PROBED and not _SAVEPOINT_WORKS


def _series_has_non_empty(frame: pd.DataFrame, column: str) -> bool:
    try:
        series = frame[column]
    except KeyError:
        return False
    if series.empty:
        return False
    if series.dtype == object:
        return series.astype(str).str.strip().ne("").any()
    return series.notna().any()


def resolve_upsert_keys(table: str, frame: pd.DataFrame | None) -> list[str]:
    """Return the natural key columns to use for ``table`` writes."""

    spec = TABLE_KEY_SPECS.get(table, {})
    if frame is None or frame.empty:
        return list(spec.get("columns", [])) if spec else []

    columns = set(frame.columns)
    keys: list[str] = []

    def _extend(candidates: Iterable[str]) -> None:
        for column in candidates:
            if column in columns and column not in keys:
                keys.append(column)

    if table == "facts_resolved":
        if "event_id" in columns and _series_has_non_empty(frame, "event_id"):
            _extend(["event_id"])
        fallback: list[str] = ["iso3", "hazard_code", "metric"]
        for candidate in ("as_of_date", "as_of"):
            if candidate in columns:
                fallback.append(candidate)
                break
        if "publication_date" in columns:
            fallback.append("publication_date")
        for candidate in ("source_id", "source"):
            if candidate in columns:
                fallback.append(candidate)
                break
        fallback.extend(
            [column for column in ("series_semantics", "ym") if column in columns]
        )
        _extend(fallback)
    elif table == "facts_deltas":
        if "event_id" in columns and _series_has_non_empty(frame, "event_id"):
            _extend(["event_id"])
        fallback = ["iso3", "hazard_code", "metric"]
        for candidate in ("as_of", "as_of_date"):
            if candidate in columns:
                fallback.append(candidate)
                break
        if "publication_date" in columns:
            fallback.append("publication_date")
        fallback.extend(
            [column for column in ("series_semantics", "ym") if column in columns]
        )
        _extend(fallback)

    if not keys and spec.get("columns"):
        _extend(spec.get("columns", []))

    return keys


@contextmanager
def _ddl_transaction(conn: "duckdb.DuckDBPyConnection", label: str) -> Iterator[None]:
    """Wrap DuckDB DDL in a savepoint when available or fall back to passthrough."""

    if os.environ.get("RESOLVER_DUCKDB_DDL_TRANSACTIONS", "1") == "0":
        _record_savepoint_support(False)
        LOGGER.debug("duckdb.ddl.transactions_disabled | label=%s", label)
        yield
        return

    savepoint = f"sp_{uuid.uuid4().hex[:8]}"
    mode = "savepoint"
    try:
        try:
            conn.execute(f"SAVEPOINT {savepoint}")
            _record_savepoint_support(True)
            LOGGER.debug(
                "duckdb.ddl.savepoint_started | label=%s | savepoint=%s",
                label,
                savepoint,
            )
        except Exception as exc:
            LOGGER.debug(
                "duckdb.ddl.savepoint_begin_failed | label=%s", label, exc_info=True
            )
            if _is_savepoint_syntax_error(exc):
                _record_savepoint_support(False)
                mode = "passthrough"
                LOGGER.debug("duckdb.ddl.passthrough_begin | label=%s", label)
            else:
                raise

        yield

        if mode == "savepoint":
            conn.execute(f"RELEASE SAVEPOINT {savepoint}")
            LOGGER.debug(
                "duckdb.ddl.savepoint_released | label=%s | savepoint=%s",
                label,
                savepoint,
            )
        else:
            LOGGER.debug("duckdb.ddl.passthrough_completed | label=%s", label)
    except Exception:
        if mode == "savepoint":
            try:
                conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                LOGGER.debug(
                    "duckdb.ddl.savepoint_rolled_back | label=%s | savepoint=%s",
                    label,
                    savepoint,
                )
            except Exception:
                LOGGER.debug(
                    "duckdb.ddl.savepoint_rollback_failed | label=%s | savepoint=%s",
                    label,
                    savepoint,
                    exc_info=True,
                )
            finally:
                try:
                    conn.execute(f"RELEASE SAVEPOINT {savepoint}")
                except Exception:
                    LOGGER.debug(
                        "duckdb.ddl.savepoint_release_failed | label=%s | savepoint=%s",
                        label,
                        savepoint,
                        exc_info=True,
                    )
        else:
            LOGGER.debug("duckdb.ddl.passthrough_error | label=%s", label)
        raise


def _run_ddl_batch(
    conn: "duckdb.DuckDBPyConnection", statements: Sequence[str], label: str
) -> None:
    if not statements:
        return

    executed: list[str] = []

    def _strip(statement: str) -> str:
        content_lines = []
        for line in statement.splitlines():
            line_stripped = line.strip()
            if not line_stripped or line_stripped.startswith("--"):
                continue
            content_lines.append(line_stripped)
        stripped = "\n".join(content_lines).rstrip(";").strip()
        if not stripped:
            return ""
        normalized = re.sub(r"\s+", " ", stripped).upper()
        if normalized in {
            "BEGIN",
            "BEGIN TRANSACTION",
            "BEGIN TRANSACTION;",
            "END",
            "END TRANSACTION",
            "COMMIT",
        }:
            return ""
        return stripped

    cleaned = [_strip(stmt) for stmt in statements]
    cleaned = [stmt for stmt in cleaned if stmt]
    if not cleaned:
        return

    with _ddl_transaction(conn, label):
        for statement in cleaned:
            conn.execute(statement)
            executed.append(statement)

    if diag_enabled() and executed:
        try:
            log_json(
                DIAG_LOGGER,
                "ddl_statements_executed",
                label=label,
                statements=executed,
            )
        except Exception:  # pragma: no cover - diagnostics only
            LOGGER.debug("ddl_statements_executed logging failed", exc_info=True)


def _merge_enabled() -> bool:
    """Return ``True`` when DuckDB MERGE statements should be attempted."""

    flag = os.getenv("RESOLVER_DUCKDB_DISABLE_MERGE", "").strip()
    return flag not in {"1", "true", "True"}


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _quote_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


_YM_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")

_ALLOWED_SERIES_SEMANTICS = {"", "new", "stock", "stock_estimate"}


def _normalize_keys_df(frame: pd.DataFrame | None, table_name: str) -> pd.DataFrame | None:
    """Normalize key columns prior to persistence."""

    if frame is None or frame.empty:
        return frame
    normalised = frame.copy()
    if "ym" in normalised.columns:
        normalised["ym"] = normalised["ym"].astype(str).str.strip()
        bad = ~normalised["ym"].fillna("").str.match(_YM_RE)
        if bad.any():
            raise ValueError(
                f"{table_name}: invalid ym format; expected YYYY-MM, got {sorted(normalised.loc[bad, 'ym'].unique())[:3]}"
            )
    if "iso3" in normalised.columns:
        normalised["iso3"] = normalised["iso3"].astype(str).str.strip().str.upper()
    if "hazard_code" in normalised.columns:
        normalised["hazard_code"] = (
            normalised["hazard_code"].astype(str).str.strip().str.upper()
        )
    return normalised


def _canonicalize_semantics(
    frame: pd.DataFrame | None, table_name: str, default_target: str
) -> tuple[pd.DataFrame | None, dict[str, dict[str, int]]]:
    """Canonicalize and enforce table-specific ``series_semantics`` values."""

    if frame is None or frame.empty or "series_semantics" not in frame.columns:
        return frame, {}
    raw = frame["series_semantics"].astype(str).fillna("").str.strip()
    before_counts = raw.str.lower().value_counts(dropna=False).to_dict()
    canonical = normalize_series_semantics(frame).copy()
    semantics = canonical["series_semantics"].astype("string").fillna("")
    semantics = semantics.str.strip().str.lower()
    semantics = semantics.mask(~semantics.isin(_ALLOWED_SERIES_SEMANTICS), "")
    semantics = semantics.mask(semantics.eq(""), default_target)
    if table_name == "facts_resolved":
        semantics = semantics.mask(semantics != "", "stock")
    elif table_name == "facts_deltas":
        semantics = pd.Series("new", index=semantics.index, dtype="string")
    canonical["series_semantics"] = semantics.astype("string")
    after_counts = (
        canonical["series_semantics"]
        .astype(str)
        .str.strip()
        .str.lower()
        .value_counts(dropna=False)
        .to_dict()
    )
    if DEBUG_ENABLED and LOGGER.isEnabledFor(logging.DEBUG):
        LOGGER.debug(
            "canonicalized semantics (%s): %s -> %s",
            table_name,
            before_counts,
            after_counts,
        )
    return canonical, {"before": before_counts, "after": after_counts}


def _canonicalise_series_semantics(series: pd.Series) -> pd.Series:
    """Return ``series`` mapped into the canonical {"", "new", "stock", "stock_estimate"}."""

    frame = pd.DataFrame({"series_semantics": series})
    normalised = normalize_series_semantics(frame)["series_semantics"]
    semantics = normalised.astype(str).str.strip()
    semantics = semantics.mask(~semantics.isin(_ALLOWED_SERIES_SEMANTICS), "")
    return semantics.astype(str)


def _assert_semantics_required(frame: pd.DataFrame, table: str) -> None:
    if frame is None or frame.empty or "series_semantics" not in frame.columns:
        return
    values = (
        frame["series_semantics"].astype(str).str.strip().str.lower().unique().tolist()
    )
    if table == "facts_resolved":
        allowed = {"stock"}
    elif table == "facts_deltas":
        allowed = {"new"}
    else:
        allowed = {"stock", "new"}
    if not set(values).issubset(allowed):
        raise ValueError(
            f"{table}: series_semantics must be subset of {sorted(allowed)}, got {sorted(set(values))}"
        )


def _coerce_numeric_cols(
    frame: pd.DataFrame | None, cols: Sequence[str], table_name: str
) -> pd.DataFrame | None:
    """Coerce known numeric columns to floats, mapping placeholder strings to NULL."""

    if frame is None or frame.empty:
        return frame

    coerced = frame.copy()
    replaced_info: dict[str, dict[str, int]] = {}
    for col in cols:
        if col not in coerced.columns:
            continue
        series = coerced[col]
        stringified = series.astype(str).str.strip()
        lowered = stringified.str.lower()
        placeholder_values = {"", "none", "null", "nan", "<na>"}
        placeholder_mask = lowered.isin(placeholder_values)
        numeric = pd.to_numeric(
            stringified.mask(placeholder_mask, pd.NA), errors="coerce"
        )
        try:
            coerced[col] = pd.Series(numeric, dtype="Float64")
        except TypeError:  # pragma: no cover - pandas compatibility shim
            coerced[col] = numeric
        replaced_info[col] = {
            "total": int(len(series)),
            "placeholders_to_null": int(placeholder_mask.sum()),
            "non_null_after": int(coerced[col].notna().sum()),
        }
    if DEBUG_ENABLED and LOGGER.isEnabledFor(logging.DEBUG):
        present = [col for col in cols if col in coerced.columns]
        LOGGER.debug(
            "duckdb.numeric_coercion | table=%s columns=%s", table_name, present
        )
    if diag_enabled() and replaced_info:
        log_json(
            DIAG_LOGGER,
            "numeric_coercion",
            table=table_name,
            columns=replaced_info,
        )
    return coerced


def _coerce_numeric(
    frame: pd.DataFrame | None, table_name: str
) -> pd.DataFrame | None:
    columns = _COERCE_NUMERIC_COLUMNS.get(table_name)
    if not columns:
        return frame
    return _coerce_numeric_cols(frame, columns, table_name)


def _normalise_iso_date_strings(frame: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    """Cast ``columns`` to ISO ``YYYY-MM-DD`` strings when present in ``frame``."""

    normalised: list[str] = []
    for column in columns:
        if column not in frame.columns:
            continue
        series = frame[column]
        formatted: pd.Series
        if pd.api.types.is_datetime64_any_dtype(series):
            formatted = series.dt.strftime("%Y-%m-%d")
        else:
            parsed = pd.to_datetime(series, errors="coerce")
            formatted = parsed.dt.strftime("%Y-%m-%d")
            # Preserve original strings where parsing failed but coerce missing to empty
            fallback = series.fillna("").astype(str)
            fallback = fallback.replace({"NaT": "", "<NA>": "", "nan": "", "NaN": ""})
            mask = parsed.notna()
            formatted = formatted.where(mask, fallback)
        formatted = formatted.fillna("").replace({"NaT": "", "<NA>": "", "nan": "", "NaN": ""})
        frame[column] = formatted.astype(str)
        normalised.append(column)
    return normalised


def _constraint_column_sets(
    conn: "duckdb.DuckDBPyConnection", table: str
) -> list[list[str]]:
    """Return column name sets for PK and UNIQUE constraints on ``table``."""

    constraints: list[list[str]] = []

    try:
        table_info = conn.execute(
            f"PRAGMA table_info({_quote_literal(table)})"
        ).fetchall()
    except Exception:  # pragma: no cover - defensive logging helper
        LOGGER.debug("Constraint inspection failed via PRAGMA table_info", exc_info=True)
        table_info = []

    pk_columns: list[str] = []
    for row in table_info:
        try:
            is_pk = bool(row[5])
        except IndexError:
            is_pk = False
        if is_pk:
            pk_columns.append(str(row[1]))
    if pk_columns:
        constraints.append(pk_columns)

    try:
        unique_rows = conn.execute(
            """
            SELECT LIST(column_name ORDER BY ordinal_position) AS column_names
            FROM information_schema.key_column_usage
            WHERE table_schema = current_schema()
              AND table_name = ?
              AND constraint_name IN (
                    SELECT constraint_name
                    FROM information_schema.table_constraints
                    WHERE table_schema = current_schema()
                      AND table_name = ?
                      AND constraint_type = 'UNIQUE'
              )
            GROUP BY constraint_name
            """,
            [table, table],
        ).fetchall()
    except Exception:  # pragma: no cover - diagnostics only
        LOGGER.debug("Constraint inspection failed via information_schema", exc_info=True)
        unique_rows = []

    for (columns,) in unique_rows:
        if not columns:
            continue
        constraint_columns = [str(column) for column in columns]
        if constraint_columns:
            constraints.append(constraint_columns)
    return constraints


def _unique_index_columns(
    conn: "duckdb.DuckDBPyConnection", table: str
) -> dict[str, list[str]]:
    """Return mapping of unique index name to ordered column list for ``table``."""

    indexes: dict[str, list[str]] = {}
    try:
        rows = conn.execute(
            "SELECT index_name, expressions, sql FROM duckdb_indexes() "
            "WHERE table_name = ? AND is_unique",
            [table],
        ).fetchall()
    except Exception:  # pragma: no cover - optional diagnostic path
        LOGGER.debug(
            "duckdb.index_inspection_failed | table=%s", table, exc_info=True
        )
        rows = []

    for name, expressions, sql in rows or []:
        if not name:
            continue
        columns: list[str] = []
        try:
            info_rows = conn.execute(
                f"PRAGMA index_info({_quote_literal(str(name))})"
            ).fetchall()
        except Exception:  # pragma: no cover - diagnostics only
            LOGGER.debug(
                "duckdb.index_info_failed | table=%s index=%s", table, name, exc_info=True
            )
            info_rows = []
        for row in info_rows:
            column = None
            if len(row) >= 3:
                column = row[2]
            elif len(row) >= 2:
                column = row[1]
            if column is not None:
                columns.append(str(column))
        if not columns:
            expr_text = str(expressions or "").strip()
            if expr_text.startswith("[") and expr_text.endswith("]"):
                expr_text = expr_text[1:-1]
            columns = [
                part.strip().strip('"')
                for part in expr_text.split(",")
                if part.strip()
            ]
        if columns:
            indexes[str(name)] = columns
    return indexes


def _canonicalize_columns(columns: Sequence[str] | None) -> list[str]:
    return [str(column).strip().lower() for column in columns or []]


def _has_constraint_key(
    conn: "duckdb.DuckDBPyConnection", table: str, canon_keys: Sequence[str]
) -> bool:
    for constraint_columns in _constraint_column_sets(conn, table):
        if _canonicalize_columns(constraint_columns) == list(canon_keys):
            return True
    return False


def _has_declared_key(
    conn: "duckdb.DuckDBPyConnection", table: str, keys: Sequence[str] | None
) -> bool:
    if not keys:
        return False

    canonical = _canonicalize_columns(keys)
    if not canonical:
        return False

    existing_columns, _ = _table_columns(conn, table)
    existing_canonical = {col.lower() for col in existing_columns}
    available = [key for key in canonical if key in existing_canonical]
    if not available:
        LOGGER.debug(
            "duckdb.schema.declared_key.unavailable | table=%s requested=%s", table, keys
        )
        return False

    candidate_sets: list[list[str]] = [canonical]
    if available != canonical:
        candidate_sets.append(available)

    def _matches(columns: Sequence[str]) -> bool:
        normalized = _canonicalize_columns(columns)
        return any(normalized == candidate for candidate in candidate_sets)

    constraint_sets: list[list[str]] = []
    unique_indexes: dict[str, list[str]] = {}

    try:
        constraint_sets = _constraint_column_sets(conn, table)
        if diag_enabled():
            log_json(
                DIAG_LOGGER,
                "constraint_inventory",
                table=table,
                constraints=[list(cols) for cols in constraint_sets],
            )
        for constraint_columns in constraint_sets:
            if _matches(constraint_columns):
                if diag_enabled():
                    log_json(
                        DIAG_LOGGER,
                        "declared_key_detected",
                        table=table,
                        keys=list(keys),
                        via="constraint",
                        columns=constraint_columns,
                    )
                return True
    except Exception:  # pragma: no cover - diagnostic aid only
        LOGGER.debug(
            "duckdb.constraint_detection_failed | table=%s", table, exc_info=True
        )

    try:
        unique_indexes = _unique_index_columns(conn, table)
        if diag_enabled():
            log_json(
                DIAG_LOGGER,
                "unique_index_inventory",
                table=table,
                indexes=unique_indexes,
            )
        for index_name, columns in unique_indexes.items():
            if _matches(columns):
                if diag_enabled():
                    log_json(
                        DIAG_LOGGER,
                        "declared_key_detected",
                        table=table,
                        keys=list(keys),
                        via="unique_index",
                        index=index_name,
                        columns=columns,
                    )
                return True
    except Exception:  # pragma: no cover - diagnostic aid only
        LOGGER.debug(
            "duckdb.unique_index_detection_failed | table=%s", table, exc_info=True
        )

    if diag_enabled():
        log_json(
            DIAG_LOGGER,
            "declared_key_missing",
            table=table,
            keys=list(keys),
            canonical=canonical,
            constraint_sets=constraint_sets,
            unique_indexes=unique_indexes,
        )

    return False


def _table_columns(
    conn: "duckdb.DuckDBPyConnection", table: str
) -> tuple[list[str], dict[str, str]]:
    try:
        rows = conn.execute(
            f"PRAGMA table_info({_quote_literal(table)})"
        ).fetchall()
    except Exception:  # pragma: no cover - diagnostics only
        LOGGER.debug("duckdb.schema.table_info_failed | table=%s", table, exc_info=True)
        return [], {}
    existing = [row[1] for row in rows]
    mapping = {name.lower(): name for name in existing}
    return existing, mapping


def _split_available_columns(
    conn: "duckdb.DuckDBPyConnection", table: str, columns: Sequence[str]
) -> tuple[list[str], list[str]]:
    if not columns:
        return [], []
    _, mapping = _table_columns(conn, table)
    available: list[str] = []
    missing: list[str] = []
    for column in columns:
        lookup = mapping.get(column.lower())
        if lookup is None:
            missing.append(column)
        else:
            available.append(lookup)
    return available, missing


def _ensure_unique_index(
    conn: "duckdb.DuckDBPyConnection", table: str, columns: Sequence[str], index_name: str
) -> None:
    if not columns:
        LOGGER.debug(
            "duckdb.schema.unique_index_skipped_empty | table=%s index=%s",
            table,
            index_name,
        )
        return

    available, missing = _split_available_columns(conn, table, columns)
    if not available:
        LOGGER.debug(
            "duckdb.schema.unique_index_skipped_missing | table=%s index=%s requested=%s",
            table,
            index_name,
            list(columns),
        )
        return
    if missing:
        LOGGER.warning(
            "duckdb.schema.unique_index_degraded | table=%s index=%s missing=%s using=%s",
            table,
            index_name,
            missing,
            available,
        )
    column_list = ", ".join(_quote_identifier(col) for col in available)
    table_ident = _quote_identifier(table)
    index_ident = _quote_identifier(index_name)
    statement = (
        f"CREATE UNIQUE INDEX IF NOT EXISTS {index_ident} ON {table_ident} ({column_list})"
    )
    _run_ddl_batch(
        conn,
        [statement],
        label=f"unique_index:{table}:{index_name}",
    )
    LOGGER.debug(
        "duckdb.schema.unique_index_ensured | table=%s index=%s columns=%s",
        table,
        index_name,
        available,
    )


def _ensure_primary_key_or_unique(
    conn: "duckdb.DuckDBPyConnection",
    table: str,
    columns: Sequence[str],
    constraint_name: str,
    index_name: str,
) -> None:
    if not columns:
        LOGGER.debug(
            "duckdb.schema.primary_key_skipped_empty | table=%s constraint=%s",
            table,
            constraint_name,
        )
        return

    available, missing = _split_available_columns(conn, table, columns)
    if not available:
        LOGGER.warning(
            "duckdb.schema.primary_key_skipped_missing | table=%s constraint=%s requested=%s",
            table,
            constraint_name,
            list(columns),
        )
        return
    if missing:
        LOGGER.warning(
            "duckdb.schema.primary_key_degraded | table=%s constraint=%s missing=%s using=%s",
            table,
            constraint_name,
            missing,
            available,
        )
        _ensure_unique_index(conn, table, available, index_name)
        return

    column_list = ", ".join(_quote_identifier(col) for col in available)
    table_ident = _quote_identifier(table)
    constraint_ident = _quote_identifier(constraint_name)

    if _savepoints_certainly_unsupported() and _connection_in_transaction(conn):
        LOGGER.debug(
            "duckdb.schema.primary_key_skipped_savepoint | table=%s constraint=%s",
            table,
            constraint_name,
        )
        _ensure_unique_index(conn, table, available, index_name)
        return

    if _has_declared_key(conn, table, columns):
        LOGGER.debug(
            "duckdb.schema.primary_key_exists | table=%s constraint=%s",
            table,
            constraint_name,
        )
        return
    try:
        statement = (
            f"ALTER TABLE {table_ident} ADD CONSTRAINT {constraint_ident} PRIMARY KEY ({column_list})"
        )
        _run_ddl_batch(
            conn,
            [statement],
            label=f"primary_key:{table}:{constraint_name}",
        )
        LOGGER.debug(
            "duckdb.schema.primary_key_added | table=%s constraint=%s columns=%s",
            table,
            constraint_name,
            available,
        )
    except _NOTIMPL_EXC as exc:  # pragma: no cover - version-specific fallback
        LOGGER.debug(
            "duckdb.schema.primary_key_not_supported | table=%s constraint=%s error=%s",
            table,
            constraint_name,
            exc,
        )
        _ensure_unique_index(conn, table, available, index_name)
    except _SCHEMA_EXC_TUPLE as exc:  # pragma: no cover - idempotent path
        message = str(exc)
        if "already has a primary key" in message or "Constraint with name" in message:
            LOGGER.debug(
                "duckdb.schema.primary_key_exists | table=%s constraint=%s",
                table,
                constraint_name,
            )
        elif "Cannot alter entry" in message:
            LOGGER.debug(
                "duckdb.schema.primary_key_conflict | table=%s constraint=%s message=%s",
                table,
                constraint_name,
                message,
            )
        else:
            LOGGER.debug(
                "duckdb.schema.primary_key_failed | table=%s constraint=%s error=%s",
                table,
                constraint_name,
                message,
            )
        _ensure_unique_index(conn, table, available, index_name)


def _attempt_heal_missing_key(
    conn: "duckdb.DuckDBPyConnection", table: str, keys: Sequence[str] | None
) -> bool:
    if not keys:
        return False
    spec = TABLE_KEY_SPECS.get(table)
    if not spec:
        return False
    expected_columns = [col.lower() for col in spec["columns"]]
    provided = [key.lower() for key in keys]
    if expected_columns != provided:
        return False
    healing_key = (table, tuple(provided))
    if healing_key not in _HEALING_WARNED:
        LOGGER.warning(
            "duckdb.upsert.heal_missing_index | table=%s keys=%s -- attempting to create indexes",
            table,
            list(keys),
        )
        _HEALING_WARNED.add(healing_key)
    _ensure_primary_key_or_unique(
        conn,
        table,
        spec["columns"],
        str(spec["primary"]),
        str(spec["unique"]),
    )
    _ensure_unique_index(
        conn,
        table,
        spec["columns"],
        str(spec["unique"]),
    )
    healed = _has_declared_key(conn, table, keys)
    if healed:
        LOGGER.debug(
            "duckdb.upsert.heal_missing_index.success | table=%s keys=%s", table, list(keys)
        )
    else:
        LOGGER.debug(
            "duckdb.upsert.heal_missing_index.failed | table=%s keys=%s", table, list(keys)
        )
    return healed


def _delete_where(
    conn: "duckdb.DuckDBPyConnection",
    table: str,
    where_sql: str,
    params: Sequence[object],
) -> int:
    """Delete rows from ``table`` matching ``where_sql`` and return the count."""

    table_ident = _quote_identifier(table)
    base_delete = f"DELETE FROM {table_ident} WHERE {where_sql}"
    try:
        rows = conn.execute(f"{base_delete} RETURNING 1", params).fetchall()
        return len(rows)
    except duckdb.Error:
        count = conn.execute(
            f"SELECT COUNT(*) FROM {table_ident} WHERE {where_sql}", params
        ).fetchone()[0]
        conn.execute(base_delete, params)
        return int(count or 0)


def _normalize_duckdb_target(path_or_url: str | None) -> tuple[str, str]:
    """Return a canonical DuckDB URL and filesystem path for ``path_or_url``."""

    env_candidate = os.environ.get("RESOLVER_DB_URL", "").strip()
    explicit = (path_or_url or "").strip()
    if explicit and env_candidate and explicit != env_candidate:
        key = (explicit, env_candidate)
        if key not in _WARNED_EXPLICIT_OVERRIDE:
            LOGGER.warning(
                "RESOLVER_DB_URL overridden by explicit argument; using provided path",
            )
            _WARNED_EXPLICIT_OVERRIDE.add(key)
    raw = explicit or env_candidate or DEFAULT_DB_URL
    path, url = _shared_canonicalize_duckdb_target(raw)
    LOGGER.info("duckdb.target | raw=%s | url=%s | path=%s", raw, url, path)
    if path == ":memory:":
        return "duckdb:///:memory:", path
    return url, path


def get_db(path_or_url: str | None = None) -> "duckdb.DuckDBPyConnection":
    """Return a DuckDB connection for the given path or URL."""

    url, normalized_path = _normalize_duckdb_target(path_or_url)
    cached = _DB_CACHE.get(url)
    cache_disabled = os.getenv("RESOLVER_DISABLE_CONN_CACHE") == "1"

    force_reopen = False

    if cached is not None:
        if getattr(cached, "_closed", False):
            _DB_CACHE.pop(url, None)
            cached = None
            force_reopen = True
        else:
            healthcheck = getattr(cached, "_healthcheck", None)
            if callable(healthcheck) and not healthcheck():
                cached = None
                _DB_CACHE.pop(url, None)
                force_reopen = True
            else:
                resolved_path = (
                    getattr(cached, "_path", None)
                    or getattr(cached, "database", None)
                    or normalized_path
                )
                log_json(
                    DIAG_LOGGER,
                    "db_open",
                    db_url=url,
                    resolved_path=resolved_path,
                    cache_disabled=cache_disabled,
                    cache_mode=os.getenv("RESOLVER_CONN_CACHE_MODE", "process"),
                    cache_event="hit",
                )
                if os.getenv("RESOLVER_DEBUG") == "1" and LOGGER.isEnabledFor(
                    logging.DEBUG
                ):
                    LOGGER.debug(
                        "DuckDB connection cache hit: path=%s from=%s cache_disabled=%s",
                        resolved_path,
                        url,
                        cache_disabled,
                    )
                return cached

    if cache_disabled:
        force_reopen = True

    shared_target = ":memory:" if url.endswith("duckdb:///:memory:") else url
    conn, resolved_path = get_shared_duckdb_conn(
        shared_target, force_reopen=force_reopen
    )
    cache_event = getattr(conn, "_last_event", None)
    if cache_disabled:
        cache_event = "miss"
    else:
        _DB_CACHE[url] = conn
    log_json(
        DIAG_LOGGER,
        "db_open",
        db_url=url,
        resolved_path=resolved_path,
        cache_disabled=cache_disabled,
        cache_mode=os.getenv("RESOLVER_CONN_CACHE_MODE", "process"),
        cache_event=cache_event,
        forced=force_reopen if force_reopen else None,
    )
    if LOGGER.isEnabledFor(logging.INFO):
        LOGGER.info(
            "duckdb.connect | url=%s path=%s cache_event=%s", url, resolved_path, cache_event
        )
    if os.getenv("RESOLVER_DEBUG") == "1" and LOGGER.isEnabledFor(logging.DEBUG):
        LOGGER.debug(
            "DuckDB connection resolved: path=%s from=%s cache_disabled=%s",  # pragma: no cover - logging only
            resolved_path,
            url,
            cache_disabled,
        )
    conn.execute("PRAGMA threads=4")
    conn.execute("PRAGMA enable_progress_bar=false")
    return conn


def init_schema(
    conn: "duckdb.DuckDBPyConnection", schema_sql_path: Path | None = None
) -> None:
    """Initialise database schema if it does not already exist."""

    schema_path = schema_sql_path or SCHEMA_PATH
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema SQL not found at {schema_path}")

    try:
        conn.execute("CREATE SCHEMA IF NOT EXISTS main")
    except Exception:  # pragma: no cover - DuckDB pre-0.9 fallback
        LOGGER.debug("duckdb.schema.ensure_main_failed", exc_info=True)

    expected_tables = {
        "facts_resolved",
        "facts_deltas",
        "manifests",
        "meta_runs",
        "snapshots",
    }
    existing_tables = {
        row[0]
        for row in conn.execute("PRAGMA show_tables").fetchall()
    }
    if LOGGER.isEnabledFor(logging.INFO):
        LOGGER.info(
            "duckdb.schema.inspect | existing_tables=%s",
            ", ".join(sorted(existing_tables)) or "<none>",
        )

    core_tables = {"facts_resolved", "facts_deltas"}
    if core_tables.issubset(existing_tables):
        LOGGER.debug("DuckDB schema already initialised; skipping DDL execution")
        for table_name, spec in TABLE_KEY_SPECS.items():
            columns = spec["columns"]
            primary = spec["primary"]
            unique = spec["unique"]
            _ensure_primary_key_or_unique(
                conn, table_name, columns, str(primary), str(unique)
            )
            _ensure_unique_index(conn, table_name, columns, str(unique))
        return

    core_statements: list[tuple[str, str]] = []
    missing_core = sorted(core_tables - existing_tables)
    if missing_core:
        LOGGER.info(
            "duckdb.schema.create | ensuring core tables=%s", ", ".join(missing_core)
        )
    if "facts_resolved" not in existing_tables:
        core_statements.append(
            (
                "facts_resolved",
                """
                CREATE TABLE IF NOT EXISTS facts_resolved (
                    ym TEXT NOT NULL,
                    iso3 TEXT NOT NULL,
                    hazard_code TEXT NOT NULL,
                    hazard_label TEXT,
                    hazard_class TEXT,
                    metric TEXT NOT NULL,
                    series_semantics TEXT NOT NULL DEFAULT '',
                    value DOUBLE,
                    unit TEXT,
                    as_of DATE,
                    as_of_date VARCHAR,
                    publication_date VARCHAR,
                    publisher TEXT,
                    source_id TEXT,
                    source_type TEXT,
                    source_url TEXT,
                    doc_title TEXT,
                    definition_text TEXT,
                    precedence_tier TEXT,
                    event_id TEXT,
                    proxy_for TEXT,
                    confidence TEXT,
                    provenance_source TEXT,
                    provenance_rank INTEGER,
                    series TEXT,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """,
            ),
        )
    if "facts_deltas" not in existing_tables:
        core_statements.append(
            (
                "facts_deltas",
                """
                CREATE TABLE IF NOT EXISTS facts_deltas (
                    ym TEXT NOT NULL,
                    iso3 TEXT NOT NULL,
                    hazard_code TEXT NOT NULL,
                    metric TEXT NOT NULL,
                    value_new DOUBLE,
                    value_stock DOUBLE,
                    series_semantics TEXT NOT NULL DEFAULT 'new',
                    as_of VARCHAR,
                    source_id TEXT,
                    series TEXT,
                    rebase_flag INTEGER,
                    first_observation INTEGER,
                    delta_negative_clamped INTEGER,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """,
            ),
        )
    if core_statements:
        _run_ddl_batch(
            conn,
            [statement for _, statement in core_statements],
            label="schema:core_tables",
        )
        for table_name, _ in core_statements:
            existing_tables.add(table_name)

    if not expected_tables.issubset(existing_tables):
        sql = schema_path.read_text(encoding="utf-8")
        statements = [s.strip() for s in sql.split(";") if s.strip()]
        _run_ddl_batch(
            conn,
            statements,
            label=f"schema:{schema_path.name}",
        )
        LOGGER.debug("Ensured DuckDB schema from %s", schema_path)

    try:
        _run_ddl_batch(
            conn,
            [
                "ALTER TABLE facts_resolved DROP CONSTRAINT IF EXISTS facts_resolved_unique",
                "ALTER TABLE facts_deltas DROP CONSTRAINT IF EXISTS facts_deltas_unique",
            ],
            label="schema:drop_legacy_unique_constraints",
        )
    except _SCHEMA_EXC_TUPLE as exc:  # pragma: no cover - migration guard only
        LOGGER.debug(
            "duckdb.schema.drop_legacy_unique.failed | error=%s",
            exc,
            exc_info=LOGGER.isEnabledFor(logging.DEBUG),
        )

    try:
        _run_ddl_batch(
            conn,
            [
                "DROP INDEX IF EXISTS ux_facts_resolved_series",
                "DROP INDEX IF EXISTS ux_facts_deltas_series",
            ],
            label="schema:drop_legacy_unique_indexes",
        )
    except _SCHEMA_EXC_TUPLE as exc:  # pragma: no cover - migration guard only
        LOGGER.debug(
            "duckdb.schema.drop_legacy_unique_index.failed | error=%s",
            exc,
            exc_info=LOGGER.isEnabledFor(logging.DEBUG),
        )

    for table_name, spec in TABLE_KEY_SPECS.items():
        columns = spec["columns"]
        primary = spec["primary"]
        unique = spec["unique"]
        _ensure_primary_key_or_unique(conn, table_name, columns, str(primary), str(unique))
        _ensure_unique_index(conn, table_name, columns, str(unique))

    try:
        _run_ddl_batch(
            conn,
            [
                """
                CREATE UNIQUE INDEX IF NOT EXISTS ux_facts_resolved_series
                ON facts_resolved (
                    event_id,
                    iso3,
                    hazard_code,
                    metric,
                    as_of_date,
                    publication_date,
                    source_id,
                    series_semantics,
                    ym
                )
                """,
                """
                CREATE UNIQUE INDEX IF NOT EXISTS ux_facts_deltas_series
                ON facts_deltas (
                    iso3,
                    hazard_code,
                    metric,
                    as_of,
                    ym,
                    series_semantics
                )
                """,
            ],
            label="schema:unique_indexes",
        )
    except _SCHEMA_EXC_TUPLE as exc:  # pragma: no cover - idempotent path
        LOGGER.debug(
            "duckdb.schema.named_unique_index_failed | error=%s", exc, exc_info=False
        )

    if diag_enabled():
        try:
            inventory: dict[str, dict[str, object]] = {}
            for table_name, spec in TABLE_KEY_SPECS.items():
                inventory[table_name] = {
                    "expected_key": list(spec["columns"]),
                    "unique_indexes": _unique_index_columns(conn, table_name),
                    "constraints": _constraint_column_sets(conn, table_name),
                }
            log_json(
                DIAG_LOGGER,
                "schema_index_inventory",
                tables=inventory,
            )
        except Exception:  # pragma: no cover - diagnostics only
            LOGGER.debug("schema_index_inventory logging failed", exc_info=True)

    if LOGGER.isEnabledFor(logging.DEBUG):
        table_details = conn.execute(
            """
            SELECT table_name, column_name, ordinal_position
            FROM information_schema.columns
            WHERE table_schema = 'main'
              AND table_name IN ('facts_resolved','facts_deltas','manifests','meta_runs','snapshots')
            ORDER BY table_name, ordinal_position
            """
        ).fetchall()
        current_table = None
        columns: list[str] = []
        for table_name, column_name, _ in table_details:
            if table_name != current_table:
                if current_table is not None:
                    LOGGER.debug(
                        "Table %s columns: %s", current_table, ", ".join(columns)
                    )
                current_table = table_name
                columns = []
            columns.append(column_name)
        if current_table is not None:
            LOGGER.debug("Table %s columns: %s", current_table, ", ".join(columns))


def _table_row_count(
    conn: "duckdb.DuckDBPyConnection", table: str
) -> int:
    try:
        return int(
            conn.execute(
                f"SELECT COUNT(*) FROM {_quote_identifier(table)}"
            ).fetchone()[0]
        )
    except Exception:  # pragma: no cover - diagnostics only
        LOGGER.debug("duckdb.upsert.count_failed | table=%s", table, exc_info=True)
        return 0


def upsert_dataframe(
    conn: "duckdb.DuckDBPyConnection",
    table: str,
    df: pd.DataFrame,
    keys: Sequence[str] | None = None,
) -> UpsertResult:
    """Upsert rows into ``table`` using ``keys`` as the natural key.

    Preferred call signature::

        upsert_dataframe(conn, table: str, df: pd.DataFrame, keys=...)

    For backwards compatibility, the legacy order with the dataframe and
    table name swapped is also accepted::

        upsert_dataframe(conn, df: pd.DataFrame, table: str, keys=...)

    Any other argument ordering will raise ``TypeError``.
    """

    if isinstance(table, pd.DataFrame) and isinstance(df, str):
        LOGGER.debug(
            "duckdb.upsert.argorder_swapped | accepted call signature (conn, df, table, keys)"
        )
        table, df = df, table

    if not isinstance(table, str) or not isinstance(df, pd.DataFrame):
        raise TypeError(
            "upsert_dataframe expects (conn, table: str, df: pandas.DataFrame, keys=...) or "
            "(conn, df: pandas.DataFrame, table: str, keys=...)"
        )

    init_schema(conn)

    merge_sql = ""
    insert_sql = ""
    delete_sql = ""
    diag_join_count_sql = ""

    try:
        _source_frame = df
        _sample = (
            _source_frame.iloc[0].to_dict() if _source_frame is not None and len(_source_frame) else {}
        )
        _insert_cols_preview = (
            list(_source_frame.columns) if _source_frame is not None else None
        )
        LOGGER.debug(
            "UPSERT starting | table=%s | keys=%s | insert_columns=%s | sample_row=%s",
            table,
            keys,
            _insert_cols_preview,
            json.dumps(_sample, default=str),
        )
    except Exception as _e:  # pragma: no cover - logging helper only
        LOGGER.debug("UPSERT debug prelude failed: %s", _e)

    if df is None or df.empty:
        rows_before = _table_row_count(conn, table)
        LOGGER.debug(
            "duckdb.upsert.no_rows | table=%s | rows_before=%s", table, rows_before
        )
        return UpsertResult(
            table=table,
            rows_in=0,
            rows_written=0,
            rows_before=rows_before,
            rows_after=rows_before,
            rows_delta=0,
        )

    duckdb = get_duckdb()

    frame = df.copy()
    coerced = _coerce_numeric(frame, table)
    if coerced is not None:
        frame = coerced
    rows_incoming = len(frame)
    rows_before = _table_row_count(conn, table)
    LOGGER.info(
        "Upserting %s rows into %s (rows_before=%s)",
        rows_incoming,
        table,
        rows_before,
    )
    LOGGER.debug("Incoming frame schema: %s", df_schema(frame))

    if "series_semantics_out" in frame.columns:
        semantics_out = frame["series_semantics_out"].where(
            frame["series_semantics_out"].notna(), ""
        ).astype(str)
        if "series_semantics" in frame.columns:
            semantics_current_raw = frame["series_semantics"]
            semantics_current = semantics_current_raw.astype(str)
            prefer_out = semantics_current_raw.isna() | semantics_current.str.strip().eq("")
            frame.loc[prefer_out, "series_semantics"] = semantics_out.loc[prefer_out]
        else:
            frame["series_semantics"] = semantics_out
        frame = frame.drop(columns=["series_semantics_out"])

    if "series_semantics" not in frame.columns:
        frame["series_semantics"] = ""

    frame["series_semantics"] = _canonicalise_series_semantics(
        frame["series_semantics"]
    )

    table_info = conn.execute(f"PRAGMA table_info({_quote_literal(table)})").fetchall()
    if not table_info:
        init_schema(conn)
        table_info = conn.execute(
            f"PRAGMA table_info({_quote_literal(table)})"
        ).fetchall()
        if not table_info:
            raise ValueError(f"Table '{table}' does not exist in DuckDB database")
    table_columns = [row[1] for row in table_info]
    has_declared_key = _has_declared_key(conn, table, keys) if keys else False
    fallback_missing_key = False
    if keys and not has_declared_key:
        init_schema(conn)
        table_info = conn.execute(
            f"PRAGMA table_info({_quote_literal(table)})"
        ).fetchall()
        if table_info:
            table_columns = [row[1] for row in table_info]
        has_declared_key = _has_declared_key(conn, table, keys)
        if not has_declared_key:
            if DEBUG_ENABLED and LOGGER.isEnabledFor(logging.DEBUG):
                try:
                    LOGGER.debug(
                        "duckdb.upsert.key_mismatch | table=%s keys=%s table_info=%s",
                        table,
                        list(keys),
                        [
                            (row[1], row[2], row[3], row[4], row[5])
                            for row in table_info
                        ],
                    )
                    tables = conn.execute("PRAGMA show_tables").fetchall()
                    LOGGER.debug(
                        "duckdb.upsert.key_mismatch.tables | entries=%s", tables
                    )
                    constraints = _constraint_column_sets(conn, table)
                    LOGGER.debug(
                        "duckdb.upsert.key_mismatch.constraints | table=%s sets=%s",
                        table,
                        constraints,
                    )
                    index_rows = conn.execute(
                        "SELECT index_name, sql FROM duckdb_indexes() WHERE table_name = ?",
                        [table],
                    ).fetchall()
                    LOGGER.debug(
                        "duckdb.upsert.key_mismatch.indexes | table=%s indexes=%s",
                        table,
                        index_rows,
                    )
                except Exception:  # pragma: no cover - diagnostics only
                    LOGGER.debug(
                        "duckdb.upsert.key_mismatch.diag_failed | table=%s",
                        table,
                        exc_info=True,
                    )
            if _attempt_heal_missing_key(conn, table, keys):
                has_declared_key = True
            else:
                fallback_missing_key = True
                if os.getenv("RESOLVER_DIAG") == "1":
                    try:
                        LOGGER.debug(
                            "duckdb.upsert.key_mismatch.indexes | table=%s discovered=%s",
                            table,
                            json.dumps(_unique_index_columns(conn, table)),
                        )
                    except Exception:  # pragma: no cover - diagnostics only
                        LOGGER.debug(
                            "duckdb.upsert.key_mismatch.indexes_failed | table=%s",
                            table,
                            exc_info=True,
                        )

    insert_columns = [col for col in table_columns if col in frame.columns]
    dropped = [col for col in frame.columns if col not in table_columns]
    if LOGGER.isEnabledFor(logging.DEBUG):
        LOGGER.debug("Table %s columns: %s", table, ", ".join(table_columns))
        LOGGER.debug("Insert columns: %s", ", ".join(insert_columns))
        if dropped:
            LOGGER.debug(
                "Dropping columns not present in %s: %s (expected for staging-only fields)",
                table,
                ", ".join(dropped),
            )
    if not insert_columns:
        raise ValueError(f"No matching columns to insert into '{table}'")

    frame = frame.loc[:, insert_columns].copy()

    normalised_dates = []
    if table in DATE_STRING_COLUMNS:
        normalised_dates = _normalise_iso_date_strings(
            frame, DATE_STRING_COLUMNS[table]
        )
        if normalised_dates and LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug(
                "Normalized date columns to ISO strings: %s",
                ", ".join(normalised_dates),
            )

    effective_keys: Sequence[str] | None = None
    merge_keys: list[str] = []
    if keys:
        missing_keys = [k for k in keys if k not in frame.columns]
        effective_keys = [k for k in keys if k in frame.columns]
        if missing_keys:
            LOGGER.debug(
                "duckdb.upsert.keys.missing | table=%s missing=%s available=%s",
                table,
                missing_keys,
                effective_keys,
            )
            if not effective_keys:
                raise KeyError(
                    f"Upsert keys {missing_keys} are missing from dataframe for table '{table}'"
                )
        else:
            effective_keys = list(keys)
        for key in effective_keys:
            frame[key] = frame[key].where(frame[key].notna(), "").astype(str).str.strip()
        before = len(frame)
        frame = frame.drop_duplicates(subset=list(effective_keys), keep="last").reset_index(
            drop=True
        )
        if LOGGER.isEnabledFor(logging.DEBUG) and before != len(frame):
            LOGGER.debug(
                "Dropped %s duplicate rows for %s based on keys %s",
                before - len(frame),
                table,
                effective_keys,
            )
        merge_keys = list(effective_keys)
        if set(effective_keys) != set(keys):
            LOGGER.debug(
                "duckdb.upsert.keys.degraded | table=%s requested=%s used=%s",
                table,
                list(keys),
                merge_keys,
            )
            has_declared_key = False

    object_columns = frame.select_dtypes(include=["object"]).columns
    for column in object_columns:
        frame[column] = frame[column].astype(str)

    if frame.empty:
        LOGGER.debug(
            "duckdb.upsert.no_rows_after_prepare | table=%s | keys=%s",
            table,
            merge_keys or list(keys or []),
        )
        return UpsertResult(
            table=table,
            rows_in=len(df) if df is not None else 0,
            rows_written=0,
            rows_before=rows_before,
            rows_after=rows_before,
            rows_delta=0,
        )

    temp_name = f"tmp_{uuid.uuid4().hex}"
    conn.register(temp_name, frame)
    upsert_completed = False
    match_rows: int | None = None
    try:
        table_ident = _quote_identifier(table)
        temp_ident = _quote_identifier(temp_name)
        if merge_keys and LOGGER.isEnabledFor(logging.DEBUG):
            diag_join_pred = " AND ".join(
                f"t.{_quote_identifier(k)} = s.{_quote_identifier(k)}" for k in merge_keys
            )
            diag_join_count_sql = (
                "\n".join(
                    [
                        "SELECT COUNT(*) AS match_rows",
                        f"FROM {table_ident} t",
                        f"JOIN {temp_ident} s",
                        f"  ON {diag_join_pred}",
                    ]
                )
            )
            LOGGER.debug("DIAG join-count SQL:\n%s", diag_join_count_sql)
            try:
                match_rows = conn.execute(diag_join_count_sql).fetchone()[0]
            except Exception:
                match_rows = None
                LOGGER.debug("DIAG join-count execution failed", exc_info=True)
            else:
                LOGGER.debug(
                    "DIAG matched %s existing rows in %s using keys %s",
                    match_rows,
                    table,
                    "[" + ", ".join(merge_keys) + "]",
                )

        log_keys = merge_keys or list(keys or [])
        use_legacy_path = True
        if merge_keys and has_declared_key:
            if not _merge_enabled():
                LOGGER.debug(
                    "MERGE disabled via RESOLVER_DUCKDB_DISABLE_MERGE for table %s",
                    table,
                )
            else:
                all_columns = insert_columns
                non_key_columns = [c for c in all_columns if c not in merge_keys]
                on_predicate = " AND ".join(
                    f"t.{_quote_identifier(k)} = s.{_quote_identifier(k)}" for k in merge_keys
                )
                update_assignments = (
                    ", ".join(
                        f"{_quote_identifier(c)} = s.{_quote_identifier(c)}"
                        for c in non_key_columns
                    )
                    if non_key_columns
                    else ""
                )
                insert_cols_sql = ", ".join(_quote_identifier(c) for c in all_columns)
                select_cols_sql = ", ".join(
                    f"s.{_quote_identifier(c)}" for c in all_columns
                )

                merge_parts = [
                    f"MERGE INTO {table_ident} AS t",
                    f"USING {temp_ident} AS s",
                    f"ON {on_predicate}",
                ]
                if update_assignments:
                    merge_parts.append(
                        f"WHEN MATCHED THEN UPDATE SET {update_assignments}"
                    )
                merge_parts.append(
                    f"WHEN NOT MATCHED THEN INSERT ({insert_cols_sql}) VALUES ({select_cols_sql})"
                )
                merge_sql = "\n".join(merge_parts) + ";"

                LOGGER.debug("MERGE SQL:\n%s", merge_sql)
                try:
                    conn.execute(merge_sql)
                except duckdb.Error as exc:
                    use_legacy_path = True
                    LOGGER.warning(
                        "duckdb.merge_failed | table=%s | rows=%s | error=%s",
                        table,
                        len(frame),
                        exc,
                        exc_info=LOGGER.isEnabledFor(logging.DEBUG),
                    )
                else:
                    LOGGER.info("Upserted %s rows into %s via MERGE", len(frame), table)
                    use_legacy_path = False
                    upsert_completed = True

        if use_legacy_path:
            transaction_started = False
            try:
                if fallback_missing_key:
                    try:
                        conn.execute("BEGIN TRANSACTION")
                    except duckdb.Error:
                        LOGGER.debug(
                            "duckdb.upsert.fallback.begin_failed | table=%s",
                            table,
                            exc_info=LOGGER.isEnabledFor(logging.DEBUG),
                        )
                    else:
                        transaction_started = True
                        if os.getenv("RESOLVER_DIAG") == "1":
                            LOGGER.debug(
                                "duckdb.upsert.fallback.start | table=%s keys=%s rows=%s",
                                table,
                                log_keys,
                                len(frame),
                            )
                if merge_keys and has_declared_key and merge_sql:
                    LOGGER.debug(
                        "Falling back to legacy delete+insert after MERGE failure for table %s",
                        table,
                    )
                if merge_keys:
                    delete_predicate = " AND ".join(
                        f"t.{_quote_identifier(k)} = s.{_quote_identifier(k)}" for k in merge_keys
                    )
                    delete_sql = (
                        "\n".join(
                            [
                                f"DELETE FROM {table_ident} AS t",
                                "WHERE EXISTS (",
                                f"    SELECT 1 FROM {temp_ident} AS s",
                                f"    WHERE {delete_predicate}",
                                ")",
                            ]
                        )
                        + ";"
                    )
                    LOGGER.debug("DELETE SQL:\n%s", delete_sql)
                    try:
                        deleted_rows = conn.execute(
                            f"{delete_sql[:-1]} RETURNING 1"
                        ).fetchall()
                        deleted_count = len(deleted_rows)
                    except duckdb.Error:
                        LOGGER.debug(
                            "DELETE ... RETURNING failed; retrying without RETURNING",
                            exc_info=True,
                        )
                        count_sql = (
                            "\n".join(
                                [
                                    "SELECT COUNT(*)",
                                    f"FROM {table_ident} AS t",
                                    f"JOIN {temp_ident} AS s",
                                    f"  ON {delete_predicate}",
                                ]
                            )
                        )
                        deleted_count = conn.execute(count_sql).fetchone()[0]
                        conn.execute(delete_sql)
                    else:
                        deleted_count = int(deleted_count)
                    LOGGER.info(
                        "duckdb.upsert.legacy_delete | Deleted %s existing rows from %s using keys %s",
                        int(deleted_count or 0),
                        table,
                        "[" + ", ".join(log_keys) + "]",
                    )

                cols_csv = ", ".join(_quote_identifier(col) for col in insert_columns)
                insert_sql = (
                    f"INSERT INTO {table_ident} ({cols_csv}) SELECT {cols_csv} FROM {temp_ident}"
                )
                LOGGER.debug("INSERT SQL:\n%s", insert_sql)
                conn.execute(insert_sql)
                LOGGER.info(
                    "duckdb.upsert.legacy_insert | Inserted %s rows into %s",
                    len(frame),
                    table,
                )
                if transaction_started:
                    conn.execute("COMMIT")
                if fallback_missing_key and os.getenv("RESOLVER_DIAG") == "1":
                    LOGGER.debug(
                        "duckdb.upsert.fallback.complete | table=%s keys=%s rows=%s",
                        table,
                        log_keys,
                        len(frame),
                    )
            except Exception as exc:
                if transaction_started:
                    try:
                        conn.execute("ROLLBACK")
                    except Exception:  # pragma: no cover - diagnostics only
                        LOGGER.debug(
                            "duckdb.upsert.fallback.rollback_failed | table=%s",
                            table,
                            exc_info=True,
                        )
                if fallback_missing_key:
                    raise ValueError(
                        f"Fallback upsert failed for table '{table}' with keys {log_keys}"
                    ) from exc
                raise
            else:
                upsert_completed = True
    except Exception as e:
        if upsert_completed:
            LOGGER.debug(
                "duckdb.upsert.completed_with_followup_error | table=%s", table, exc_info=True
            )
        else:
            try:
                LOGGER.error(
                    (
                        "DuckDB upsert failed for table %s.\n"
                        "MERGE SQL:\n%s\nDELETE SQL:\n%s\nINSERT SQL:\n%s\n"
                        "JOIN-COUNT SQL:\n%s\nschema=%s | first_row=%s"
                    ),
                    table,
                    merge_sql or "<unset>",
                    delete_sql or "<unset>",
                    insert_sql or "<unset>",
                    diag_join_count_sql or "<unset>",
                    table_columns,
                    frame.head(1).to_dict(orient="records"),
                    exc_info=True,
                )
            except Exception:  # pragma: no cover - logging failure should not mask error
                LOGGER.exception("DIAG: assembling failure log failed")
            raise e
    finally:
        conn.unregister(temp_name)

    rows_after = _table_row_count(conn, table)
    rows_delta = rows_after - rows_before
    if LOGGER.isEnabledFor(logging.INFO):
        LOGGER.info(
            "duckdb.upsert.counts | table=%s rows_before=%s rows_after=%s delta=%s",
            table,
            rows_before,
            rows_after,
            rows_delta,
        )
    return UpsertResult(
        table=table,
        rows_in=len(df) if df is not None else 0,
        rows_written=len(frame),
        rows_before=rows_before,
        rows_after=rows_after,
        rows_delta=rows_delta,
        matched_existing=int(match_rows)
        if isinstance(match_rows, numbers.Integral)
        else None,
    )


def _default_created_at(value: str | None = None) -> str:
    if value:
        return value
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _ensure_columns(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    df = frame.copy()
    for column in columns:
        if column not in df.columns:
            df[column] = ""
    return df


def write_snapshot(
    conn: "duckdb.DuckDBPyConnection",
    *,
    ym: str,
    facts_resolved: pd.DataFrame | None,
    facts_deltas: pd.DataFrame | None,
    manifests: Iterable[Mapping[str, object]] | None,
    meta: Mapping[str, object] | None,
) -> None:
    """Write a snapshot bundle transactionally into the database."""

    init_schema(conn)

    facts_resolved = facts_resolved.copy() if facts_resolved is not None else None
    facts_deltas = facts_deltas.copy() if facts_deltas is not None else None

    if diag_enabled():
        if facts_resolved is not None:
            log_json(
                DIAG_LOGGER,
                "write_inputs_resolved",
                columns=list(facts_resolved.columns),
                n=int(len(facts_resolved)),
                sample=facts_resolved.head(2).to_dict(orient="records"),
            )
            log_json(
                DIAG_LOGGER,
                "table_meta_resolved",
                **dump_table_meta(conn, "facts_resolved"),
            )
        if facts_deltas is not None:
            log_json(
                DIAG_LOGGER,
                "write_inputs_deltas",
                columns=list(facts_deltas.columns),
                n=int(len(facts_deltas)),
                has_as_of="as_of" in facts_deltas.columns,
                sample=facts_deltas.head(2).to_dict(orient="records"),
            )
            log_json(
                DIAG_LOGGER,
                "table_meta_deltas",
                **dump_table_meta(conn, "facts_deltas"),
            )

    facts_resolved = _normalize_keys_df(facts_resolved, "facts_resolved")
    facts_deltas = _normalize_keys_df(facts_deltas, "facts_deltas")

    try:
        with _ddl_transaction(conn, "snapshot_write"):
            facts_rows = 0
            deltas_rows = 0

            deleted_resolved = _delete_where(conn, "facts_resolved", "ym = ?", [ym])
            LOGGER.debug(
                "Deleted %s facts_resolved rows for ym=%s", deleted_resolved, ym
            )

            if facts_resolved is not None and not facts_resolved.empty:
                facts_resolved = _ensure_columns(
                    facts_resolved,
                    FACTS_RESOLVED_KEY_COLUMNS + ["value"],
                )
                for key in FACTS_RESOLVED_KEY_COLUMNS:
                    series = (
                        facts_resolved[key]
                        .where(facts_resolved[key].notna(), "")
                        .astype(str)
                        .str.strip()
                    )
                    if key == "ym":
                        series = series.replace("", ym)
                    facts_resolved[key] = series
                computed_semantics = facts_resolved.apply(
                    lambda row: compute_series_semantics(
                        metric=row.get("metric"), existing=row.get("series_semantics")
                    ),
                    axis=1,
                )
                facts_resolved["series_semantics"] = computed_semantics
                facts_resolved, _ = _canonicalize_semantics(
                    facts_resolved,
                    "facts_resolved",
                    default_target="stock",
                )
                _assert_semantics_required(facts_resolved, "facts_resolved")
                facts_resolved = _coerce_numeric(
                    facts_resolved, "facts_resolved"
                )
                facts_resolved = facts_resolved.drop_duplicates(
                    subset=FACTS_RESOLVED_KEY_COLUMNS,
                    keep="last",
                ).reset_index(drop=True)
                facts_rows = upsert_dataframe(
                    conn,
                    "facts_resolved",
                    facts_resolved,
                    keys=FACTS_RESOLVED_KEY_COLUMNS,
                )
                LOGGER.info("facts_resolved rows upserted: %s", facts_rows)
                LOGGER.debug(
                    "facts_resolved series_semantics distribution: %s",
                    dict_counts(facts_resolved["series_semantics"]),
                )

            deleted_deltas = 0
            if facts_deltas is not None and not facts_deltas.empty:
                deleted_deltas = _delete_where(conn, "facts_deltas", "ym = ?", [ym])
                LOGGER.debug(
                    "Deleted %s facts_deltas rows for ym=%s", deleted_deltas, ym
                )
                facts_deltas = _ensure_columns(
                    facts_deltas,
                    FACTS_DELTAS_KEY_COLUMNS
                    + ["series_semantics", "value_new", "value_stock"],
                )
                for key in FACTS_DELTAS_KEY_COLUMNS:
                    series = (
                        facts_deltas[key]
                        .where(facts_deltas[key].notna(), "")
                        .astype(str)
                        .str.strip()
                    )
                    if key == "ym":
                        series = series.replace("", ym)
                    facts_deltas[key] = series
                if "series_semantics" not in facts_deltas.columns:
                    facts_deltas["series_semantics"] = "new"
                facts_deltas, _ = _canonicalize_semantics(
                    facts_deltas,
                    "facts_deltas",
                    default_target="new",
                )
                _assert_semantics_required(facts_deltas, "facts_deltas")
                facts_deltas = _coerce_numeric(
                    facts_deltas, "facts_deltas"
                )
                facts_deltas = facts_deltas.drop_duplicates(
                    subset=FACTS_DELTAS_KEY_COLUMNS,
                    keep="last",
                ).reset_index(drop=True)
                deltas_rows = upsert_dataframe(
                    conn,
                    "facts_deltas",
                    facts_deltas,
                    keys=FACTS_DELTAS_KEY_COLUMNS,
                )
                LOGGER.info("facts_deltas rows upserted: %s", deltas_rows)
                LOGGER.debug(
                    "facts_deltas series_semantics distribution: %s",
                    dict_counts(facts_deltas["series_semantics"]),
                )
            else:
                deleted_deltas = _delete_where(conn, "facts_deltas", "ym = ?", [ym])
                if deleted_deltas:
                    LOGGER.debug(
                        "Deleted %s facts_deltas rows for ym=%s (no deltas frame provided)",
                        deleted_deltas,
                        ym,
                    )
            manifest_rows: list[dict] = []
            manifest_created_at = _default_created_at(
                meta.get("created_at_utc") if meta else None
            )
            meta_schema_version = str(meta.get("schema_version", "")) if meta else ""
            meta_source_id = str(meta.get("source_id", "")) if meta else ""
            meta_sha256 = str(meta.get("sha256", "")) if meta else ""
            if manifests:
                for entry in manifests:
                    payload = dict(entry)
                    name = str(payload.get("name") or payload.get("path") or "artifact")
                    raw_path = str(payload.get("path") or "").strip()
                    if not raw_path:
                        raw_path = f"{ym}/{name}"
                    manifest_rows.append(
                        {
                            "path": raw_path,
                            "sha256": str(
                                payload.get("checksum")
                                or payload.get("sha256")
                                or meta_sha256
                            ),
                            "row_count": int(
                                payload.get("rows") or payload.get("row_count") or 0
                            ),
                            "schema_version": str(
                                payload.get("schema_version")
                                or meta_schema_version
                            ),
                            "source_id": str(payload.get("source_id") or meta_source_id),
                            "created_at": manifest_created_at,
                        }
                    )
            patterns = {
                f"%/{ym}/%",
                f"%\\{ym}\\%",
                f"{ym}/%",
                f"{ym}\\%",
            }
            deleted_manifests = 0
            for pattern in patterns:
                deleted_manifests += _delete_where(
                    conn, "manifests", "path LIKE ?", [pattern]
                )
            if manifest_rows:
                upsert_dataframe(
                    conn,
                    "manifests",
                    pd.DataFrame(manifest_rows),
                    keys=["path"],
                )
            LOGGER.debug("Deleted %s manifest rows for ym=%s", deleted_manifests, ym)
            snapshot_payload = {
                "ym": ym,
                "created_at": _default_created_at(
                    meta.get("created_at_utc") if meta else None
                ),
                "git_sha": str(meta.get("source_commit_sha", "")) if meta else "",
                "export_version": str(meta.get("export_version", "")) if meta else "",
                "facts_rows": facts_rows,
                "deltas_rows": deltas_rows,
                "meta": json.dumps(dict(meta or {}), sort_keys=True),
            }
            _delete_where(conn, "snapshots", "ym = ?", [ym])
            conn.execute(
                """
                INSERT INTO snapshots
                (ym, created_at, git_sha, export_version, facts_rows, deltas_rows, meta)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    snapshot_payload["ym"],
                    snapshot_payload["created_at"],
                    snapshot_payload["git_sha"],
                    snapshot_payload["export_version"],
                    int(snapshot_payload["facts_rows"] or 0),
                    int(snapshot_payload["deltas_rows"] or 0),
                    snapshot_payload["meta"],
                ],
            )
            LOGGER.debug(
                "Snapshot summary: %s",
                snapshot_payload,
            )
            LOGGER.info(
                (
                    "DuckDB snapshot write complete: ym=%s facts_resolved=%s deltas=%s "
                    "deleted_resolved=%s deleted_deltas=%s"
                ),
                ym,
                facts_rows,
                deltas_rows,
                deleted_resolved,
                deleted_deltas,
            )
            if diag_enabled():
                counts = dump_counts(conn, ym=ym)
                log_json(
                    DIAG_LOGGER,
                    "post_upsert_counts",
                    ym=ym,
                    **counts,
                )
    except Exception:
        if diag_enabled():
            log_json(
                DIAG_LOGGER,
                "write_snapshot_error",
                ym=ym,
                error=repr(sys.exc_info()[1]),
            )
        raise
