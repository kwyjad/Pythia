from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple, Union

import pandas as pd
import yaml

KIND_PREFIXES = {"staging", "facts", "resolved", "diagnostics", "review"}
NULL_LITERALS = {"", "na", "nan", "none", "null"}


def load_schema(schema_path: Union[str, Path]) -> Dict[str, Any]:
    """Load ``schema.yml`` and normalise entities for validation."""

    path = Path(schema_path)
    if not path.exists():
        return {"entities": {}}

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, Mapping):
        return {"entities": {}}

    raw_entities: Dict[str, Dict[str, Any]] = {}

    def register_entity(name: str, payload: Optional[Mapping[str, Any]], default_kind: Optional[str] = None) -> None:
        if not payload:
            payload = {}
        if not isinstance(payload, Mapping):
            payload = {}
        entity_name = str(name)
        if entity_name in raw_entities:
            # Merge duplicate declarations by layering additional info on top.
            existing = raw_entities[entity_name]
            combined_payload = {**existing.get("_raw_payload", {}), **payload}
            payload = combined_payload
        kind = str(payload.get("kind")) if payload.get("kind") else _infer_kind(entity_name) or (default_kind or "")
        raw_entities[entity_name] = {
            "name": entity_name,
            "description": str(payload.get("description", "") or "").strip(),
            "kind": kind,
            "file_glob": payload.get("file_glob"),
            "allow_extra_columns": payload.get("allow_extra_columns"),
            "keys": _ensure_list(payload.get("keys", [])),
            "extends": payload.get("extends"),
            "column_map": _build_column_map(payload, data),
            "_raw_payload": dict(payload),
        }

    entities_section = data.get("entities")
    if isinstance(entities_section, Mapping):
        for entity_name, payload in entities_section.items():
            register_entity(str(entity_name), payload, None)

    # Legacy sections such as ``staging: {foo: {...}}``
    for prefix in KIND_PREFIXES:
        section = data.get(prefix)
        if isinstance(section, Mapping):
            for short_name, payload in section.items():
                full_name = f"{prefix}.{short_name}"
                register_entity(full_name, payload, prefix)

    # Bare schema (legacy ``required``/``optional`` at top level)
    if not raw_entities and _looks_like_entity(data):
        register_entity("facts", data, "facts")

    resolved_entities: Dict[str, Dict[str, Any]] = {}
    resolved_columns: Dict[str, MutableMapping[str, Dict[str, Any]]] = {}

    def resolve_entity(name: str, stack: Optional[List[str]] = None) -> Dict[str, Any]:
        if name in resolved_entities:
            return resolved_entities[name]
        if name not in raw_entities:
            raise KeyError(f"Unknown entity '{name}' referenced in schema")
        stack = (stack or []) + [name]
        raw = raw_entities[name]
        extends = raw.get("extends")
        parent_entity: Optional[Dict[str, Any]] = None
        parent_columns: Optional[MutableMapping[str, Dict[str, Any]]] = None
        if extends:
            parent_name = str(extends)
            if parent_name in stack[:-1]:
                raise ValueError(f"Circular schema inheritance detected: {' -> '.join(stack + [parent_name])}")
            parent_entity = resolve_entity(parent_name, stack)
            parent_columns = resolved_columns[parent_name]

        if parent_columns is None:
            merged_columns: MutableMapping[str, Dict[str, Any]] = {}
        else:
            merged_columns = {col_name: dict(spec) for col_name, spec in parent_columns.items()}

        for col_name, spec in raw["column_map"].items():
            base = dict(merged_columns.get(col_name, {"name": col_name}))
            base.update(spec)
            base["name"] = col_name
            merged_columns[col_name] = base

        allow_extra = raw.get("allow_extra_columns")
        if allow_extra is None and parent_entity is not None:
            allow_extra = parent_entity.get("allow_extra_columns")
        allow_extra = bool(allow_extra)

        description = raw.get("description") or (parent_entity.get("description") if parent_entity else "")
        kind = raw.get("kind") or (parent_entity.get("kind") if parent_entity else "")
        file_glob = raw.get("file_glob") or (parent_entity.get("file_glob") if parent_entity else None)

        parent_keys = parent_entity.get("keys", []) if parent_entity else []
        keys = list(parent_keys)
        for key in _ensure_list(raw.get("keys", [])):
            if key not in keys:
                keys.append(key)

        columns_list = [_finalise_column(spec) for spec in merged_columns.values()]

        entity = {
            "name": name,
            "description": description,
            "kind": kind,
            "file_glob": file_glob,
            "keys": keys,
            "allow_extra_columns": allow_extra,
            "columns": columns_list,
        }
        resolved_entities[name] = entity
        resolved_columns[name] = merged_columns
        return entity

    for entity_name in list(raw_entities.keys()):
        resolve_entity(entity_name)

    return {"entities": resolved_entities}


def infer_entity_name_from_filename(fname: Union[str, Path]) -> str:
    path = Path(fname)
    stem = path.stem
    return f"staging.{stem}"


def validate_headers(df: pd.DataFrame, entity: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    specified_columns = [col.get("name") for col in entity.get("columns", [])]
    expected = {name for name in specified_columns if name}
    required = {col.get("name") for col in entity.get("columns", []) if col.get("required")}
    present = set(df.columns)

    missing = sorted(c for c in required if c not in present)
    if missing:
        errors.append(f"Missing required columns: {', '.join(missing)}")

    unknown = sorted(c for c in present - expected)
    if unknown and not entity.get("allow_extra_columns"):
        errors.append(f"Unexpected columns: {', '.join(unknown)}")

    return errors


def coerce_and_validate_types(df: pd.DataFrame, entity: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    for column in entity.get("columns", []):
        name = column.get("name")
        if not name:
            continue
        required = bool(column.get("required"))
        nullable = column.get("nullable")
        if nullable is None:
            nullable = not required
        col_type = (column.get("type") or "string").lower()

        if name not in df.columns:
            if required:
                errors.append(f"{name}: missing required column")
            continue

        series = df[name]
        clean_series = series.astype(str).str.strip()
        null_mask = clean_series.map(_is_null_like)
        non_null = ~null_mask

        if not nullable and bool(null_mask.any()):
            errors.append(f"{name}: {int(null_mask.sum())} null/empty values but column is not nullable")

        if col_type in {"integer", "int"}:
            invalid_msg = _validate_integer(clean_series, null_mask)
            if invalid_msg:
                errors.append(f"{name}: {invalid_msg}")
        elif col_type in {"float", "number", "numeric"}:
            invalid_msg = _validate_number(clean_series, null_mask)
            if invalid_msg:
                errors.append(f"{name}: {invalid_msg}")
        elif col_type == "enum":
            allowed = [str(v) for v in column.get("enum", [])]
            if allowed:
                invalid_values = clean_series[non_null & ~clean_series.isin(allowed)]
                if not invalid_values.empty:
                    counts = invalid_values.value_counts()
                    values = ", ".join(f"{val} ({counts[val]})" for val in counts.index)
                    errors.append(f"invalid enum values: {values}")
        elif col_type == "date":
            fmt = column.get("format")
            invalid_msg = _validate_date(clean_series, null_mask, fmt)
            if invalid_msg:
                errors.append(f"{name}: {invalid_msg}")
        else:
            # For strings, ensure dtype is object/string, which is guaranteed when reading as str.
            pass

    return list(dict.fromkeys(errors))


def validate_staging_csv(csv_path: Path, schema: Mapping[str, Any]) -> Tuple[bool, List[str]]:
    entities = schema.get("entities", {}) or {}
    entity_name = infer_entity_name_from_filename(csv_path)
    entity = entities.get(entity_name)
    if entity is None and entity_name.startswith("staging."):
        entity = entities.get("staging.common")
    if entity is None:
        return False, [f"No entity in schema for {csv_path}"]

    try:
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    except Exception as exc:  # pragma: no cover - IO errors surfaced in test assertions
        return False, [f"Failed to read {csv_path}: {exc}"]

    errors = []
    errors.extend(validate_headers(df, entity))
    errors.extend(coerce_and_validate_types(df, entity))
    deduped = list(dict.fromkeys(errors))
    return (len(deduped) == 0, deduped)


def _infer_kind(name: str) -> str:
    if not name:
        return ""
    prefix = name.split(".", 1)[0]
    if prefix in KIND_PREFIXES:
        return prefix
    return ""


def _looks_like_entity(payload: Mapping[str, Any]) -> bool:
    if not isinstance(payload, Mapping):
        return False
    return any(key in payload for key in ("columns", "required", "optional"))


def _ensure_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value if str(v).strip()]
    return [str(value)] if str(value).strip() else []


def _build_column_map(payload: Mapping[str, Any], context: Mapping[str, Any]) -> MutableMapping[str, Dict[str, Any]]:
    columns: MutableMapping[str, Dict[str, Any]] = {}

    def get_or_create(name: str) -> Dict[str, Any]:
        column = columns.get(name)
        if column is None:
            column = {
                "name": name,
                "type": None,
                "required": False,
                "enum": None,
                "format": None,
                "nullable": None,
            }
            columns[name] = column
        return column

    columns_data = payload.get("columns")
    if isinstance(columns_data, Mapping):
        iterator: Iterable[Tuple[str, Any]] = columns_data.items()
    elif isinstance(columns_data, list):
        iterator = []
        for entry in columns_data:
            if isinstance(entry, Mapping):
                name = str(entry.get("name")) if entry.get("name") else ""
                if name:
                    iterator.append((name, entry))
            elif entry:
                iterator.append((str(entry), {"name": entry}))
    else:
        iterator = []

    for name, info in iterator:
        if not name:
            continue
        column = get_or_create(str(name))
        if isinstance(info, Mapping):
            for key, value in info.items():
                if key == "name":
                    continue
                if key == "enum" and value is not None:
                    column["enum"] = [str(v) for v in _ensure_list(value)]
                else:
                    column[key] = value

    for required_name in _ensure_list(payload.get("required")):
        column = get_or_create(required_name)
        column["required"] = True

    optional = payload.get("optional")
    if isinstance(optional, Mapping):
        for opt_name, opt_info in optional.items():
            column = get_or_create(str(opt_name))
            if isinstance(opt_info, Mapping):
                for key, value in opt_info.items():
                    if key == "name":
                        continue
                    if key == "enum" and value is not None:
                        column["enum"] = [str(v) for v in _ensure_list(value)]
                    else:
                        column[key] = value
            column.setdefault("required", False)
    elif isinstance(optional, list):
        for opt_entry in optional:
            if isinstance(opt_entry, Mapping):
                name = opt_entry.get("name")
                if not name:
                    continue
                column = get_or_create(str(name))
                for key, value in opt_entry.items():
                    if key == "name":
                        continue
                    if key == "enum" and value is not None:
                        column["enum"] = [str(v) for v in _ensure_list(value)]
                    else:
                        column[key] = value
                column.setdefault("required", False)
            elif opt_entry:
                column = get_or_create(str(opt_entry))
                column.setdefault("required", False)

    context_enums = context.get("enums") if isinstance(context.get("enums"), Mapping) else {}
    if isinstance(context_enums, Mapping):
        for col_name, values in context_enums.items():
            column = columns.get(str(col_name))
            if column is None:
                continue
            if not column.get("enum"):
                column["enum"] = [str(v) for v in _ensure_list(values)]
                if not column.get("type"):
                    column["type"] = "enum"

    for col in columns.values():
        col_type = col.get("type")
        if isinstance(col_type, str):
            col["type"] = col_type.strip().lower()
        elif col_type is None:
            col["type"] = "string"
        else:
            col["type"] = str(col_type).strip().lower()

        col["required"] = bool(col.get("required"))
        if col.get("enum") and col["type"] not in {"enum", "string"}:
            col["type"] = "enum"
        if col.get("enum"):
            col["enum"] = [str(v) for v in col.get("enum") if str(v)]
        nullable = col.get("nullable")
        if nullable is None:
            col["nullable"] = not col["required"]
        else:
            col["nullable"] = bool(nullable)

    return columns


def _finalise_column(spec: Mapping[str, Any]) -> Dict[str, Any]:
    column = {
        "name": spec.get("name"),
        "type": (spec.get("type") or "string"),
        "required": bool(spec.get("required")),
        "enum": spec.get("enum"),
        "format": spec.get("format"),
        "nullable": bool(spec.get("nullable")),
    }
    return column


def _is_null_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str):
        return value.strip().lower() in NULL_LITERALS
    return False


def _validate_integer(series: pd.Series, null_mask: pd.Series) -> Optional[str]:
    coerced = pd.to_numeric(series.where(~null_mask, None), errors="coerce")
    invalid_mask = (~null_mask) & coerced.isna()
    if bool(invalid_mask.any()):
        return f"invalid integers: {int(invalid_mask.sum())} rows"

    def _is_integer(value: Any) -> bool:
        if pd.isna(value):
            return True
        try:
            return float(value).is_integer()
        except Exception:
            return False

    non_integer_mask = (~null_mask) & ~coerced.apply(_is_integer)
    if bool(non_integer_mask.any()):
        return f"non-integer values: {int(non_integer_mask.sum())} rows"
    return None


def _validate_number(series: pd.Series, null_mask: pd.Series) -> Optional[str]:
    coerced = pd.to_numeric(series.where(~null_mask, None), errors="coerce")
    invalid_mask = (~null_mask) & coerced.isna()
    if bool(invalid_mask.any()):
        return f"invalid numbers: {int(invalid_mask.sum())} rows"

    non_finite_mask = (~null_mask) & coerced.apply(
        lambda value: False if pd.isna(value) else not math.isfinite(float(value))
    )
    if bool(non_finite_mask.any()):
        return f"non-finite numbers: {int(non_finite_mask.sum())} rows"
    return None


def _validate_date(series: pd.Series, null_mask: pd.Series, fmt: Optional[str]) -> Optional[str]:
    if fmt == "YYYY-MM":
        pattern = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
        invalid = series[~null_mask & ~series.str.match(pattern)]
        if not invalid.empty:
            return f"invalid YYYY-MM values: {len(invalid)} rows"
        return None

    invalid: List[str] = []
    for value in series[~null_mask]:
        try:
            pd.to_datetime(value, utc=False, errors="raise")
        except Exception:
            invalid.append(value)
    if invalid:
        return f"invalid dates: {len(invalid)} rows"
    return None
