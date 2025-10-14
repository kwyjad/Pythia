from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from datetime import date
import uuid
from pythia.db.init import init as init_db

_pool = ThreadPoolExecutor(max_workers=1)


def _pipeline(countries: list[str]):
    # TODO: call HS (with likely_window_month), write questions (Card 1)
    # TODO: call Researcher (Card 3) and Forecaster from-DB (Card 4)
    # TODO: aggregate + write ensemble; optionally trigger Resolver join + scores
    pass


def enqueue_run(countries: list[str]) -> str:
    run_id = f"ui_run_{date.today().isoformat()}_{uuid.uuid4().hex[:8]}"
    _pool.submit(_pipeline, countries)
    return run_id
