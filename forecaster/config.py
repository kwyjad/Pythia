# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

# ANCHOR: config 
from __future__ import annotations
import os, json, math
from datetime import datetime, timezone, timedelta
try:
    from zoneinfo import ZoneInfo
    IST_TZ = ZoneInfo("Europe/Istanbul")
except Exception:
    IST_TZ = timezone(timedelta(hours=3))

# Load .env early if present
try:
    import dotenv; dotenv.load_dotenv()
except Exception:
    pass

# --- Cache toggles ---
DISABLE_RESEARCH_CACHE = os.getenv("PYTHIA_DISABLE_RESEARCH_CACHE", "0").lower() in ("1","true","yes")

# --- Markets ---
ENABLE_MARKET_SNAPSHOT = os.getenv("ENABLE_MARKET_SNAPSHOT", "1").lower() in ("1","true","yes")
MARKET_SNAPSHOT_MAX_MATCHES = int(os.getenv("MARKET_SNAPSHOT_MAX_MATCHES", 3))

TEST_POSTS_FILE = os.getenv("TEST_POSTS_FILE", "data/test_questions.json")
TEST_POST_IDS_ENV = os.getenv("TEST_POST_IDS", "").strip()

# --- Files & dirs ---
FORECASTS_CSV      = "forecasts.csv"
FORECASTS_BY_MODEL = "forecasts_by_model.csv"
FORECAST_LOG_DIR   = "forecast_logs"
RUN_LOG_DIR        = "logs"
CACHE_DIR          = "cache"
MCQ_WIDE_CSV       = "forecasts_mcq_wide.csv"
MAX_MCQ_OPTIONS    = 20

# --- Calibration note path ---
CALIBRATION_PATH = os.getenv("CALIBRATION_PATH", "")

# --- Time helpers ---
def ist_stamp(fmt: str = "%Y%m%d-%H%M%S") -> str:
    from datetime import datetime
    return datetime.now(IST_TZ).strftime(fmt)

def ist_iso(fmt: str = "%Y-%m-%d %H:%M:%S %z") -> str:
    from datetime import datetime
    return datetime.now(IST_TZ).strftime(fmt)

def ist_date(fmt: str = "%Y-%m-%d") -> str:
    from datetime import datetime
    return datetime.now(IST_TZ).strftime(fmt)

# --- Cache helpers (JSON) ---
def cache_path(kind: str, slug: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{kind}__{slug}.json")

def read_cache(kind: str, slug: str) -> dict | None:
    p = cache_path(kind, slug)
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def write_cache(kind: str, slug: str, data: dict) -> None:
    try:
        with open(cache_path(kind, slug), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

# --- Misc small utils ---
def clip01(x: float) -> float:
    return max(0.01, min(0.99, float(x)))

def fmt_float_or_blank(x) -> str:
    try:
        xf = float(x)
        if math.isnan(xf) or math.isinf(xf):
            return ""
        return f"{xf:.6f}"
    except Exception:
        return ""
