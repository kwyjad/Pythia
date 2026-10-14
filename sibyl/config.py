# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Sibyl configuration.

Every knob is env-overridable (``SIBYL_*``) following the repo's
``_env_float`` convention, with the spec defaults baked in.
"""

from __future__ import annotations

import os


def _env_float(name: str, default: float) -> float:
    try:
        raw = os.getenv(name)
        return float(raw) if raw not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        raw = os.getenv(name)
        return int(raw) if raw not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes")


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    return raw if raw not in (None, "") else default


# --- Model -----------------------------------------------------------------
# Opus-tier deep-research model. This default is INDEPENDENT of the `claude`
# registry alias in pythia/config.yaml — swapping the ensemble's Anthropic
# member does not move Sibyl, so both must be edited together.
# NOTE: claude-opus-5-5 rejects sampling params (temperature/top_p/top_k ->
# HTTP 400, see _ANTHROPIC_NO_TEMPERATURE_PREFIXES); trial diversity comes from
# the research lane each trial takes (sibyl/agent.py::TRIAL_LANES).
MODEL = _env_str("SIBYL_MODEL", "claude-opus-5-5")

# Thinking depth sent as output_config.effort on every step. Explicit because
# Opus 5.5 defaults to "medium" where Opus 5 defaulted to "high": a step that
# sent nothing would think less after the swap, and nothing would say so.
# Opus 5.5 cannot disable thinking at all, so effort is the only lever. It is
# emitted only for models in providers._ANTHROPIC_EFFORT_PREFIXES; "off" or ""
# sends nothing and leaves the model's own default in force.
EFFORT = _env_str("SIBYL_EFFORT", "high").strip().lower()

# Stable model_name under which Sibyl SPDs are written to forecasts_raw /
# forecasts_ensemble (and therefore scored). Analogous to `track2_flash`:
# a track marker, independent of the backing model id above.
SIBYL_MODEL_NAME = "sibyl"

# --- Trials ----------------------------------------------------------------
# Independent agentic trials per question. Since 2026-10-14 two: the first
# (lane A) is shown the structured-data pack and the second (lane B) is not,
# so the pack is compared within each question (sibyl/pack.py). On 13 October
# 2026 three trials barely differed (inter-trial month-1 JSD 0.000 to 0.005
# on every question) while costing $5.04 a question.
K = _env_int("SIBYL_K", 2)
# Extra trials (sibyl/run.py): up to K_MAX - K more trials when the production
# trials disagree (largest pairwise month-1 JSD above EXTRA_TRIALS_JSD) or
# their pool departs far from the reference (EXTRA_TRIALS_DEPARTURE_JSD). At
# K_MAX = K there are none, except the reconciler (lane R), which still runs
# on disagreement. On 13 October's numbers it would never have fired.
K_MAX = _env_int("SIBYL_K_MAX", 2)
EXTRA_TRIALS_JSD = _env_float("SIBYL_EXTRA_TRIALS_JSD", 0.10)
EXTRA_TRIALS_DEPARTURE_JSD = _env_float("SIBYL_EXTRA_TRIALS_DEPARTURE_JSD", 0.25)
# Trials of one question run on this many worker threads. Only the main
# thread writes DuckDB: a trial's llm_calls rows are buffered and written
# after its batch.
TRIAL_WORKERS = _env_int("SIBYL_TRIAL_WORKERS", 3)
# The reconciler (lane R, review Part 4, sibyl/reconcile.py): its brief of
# the earlier trials is capped at this many characters; whole ledger items
# are dropped beyond it (tier 1-2 and dated figures kept first).
RECONCILE_BRIEF_MAX_CHARS = _env_int("SIBYL_RECONCILE_BRIEF_MAX_CHARS", 12_000)
# Outlier guard: a trial whose month-1 median is more than this many orders
# of magnitude (log10 of 1 + value) from the median of the others' medians
# is left out of the pool, if two trials remain. It stays in trials_json.
# It needs three valid trials to act, so at K=2 it is inert (unless the
# reconciler ran); the code is kept.
OUTLIER_LOG10 = _env_float("SIBYL_OUTLIER_LOG10", 1.5)

# Agent steps per trial. Since Oct 2026 a step may carry up to
# MAX_ACTIONS_PER_STEP tool calls and a submit must pass the research gate
# (sibyl/agent.py::submit_gate_missing); at the step limit the trial ends
# with what it has.
MAX_STEPS = _env_int("SIBYL_MAX_STEPS", 12)
MAX_ACTIONS_PER_STEP = _env_int("SIBYL_MAX_ACTIONS_PER_STEP", 3)
# Documents a trial must have read before a submit is accepted (the step
# limit overrides it; the evidence gate below does not). Raised from 3 to 5
# in Oct 2026: FutureSearch's typical run reads 5 to 20 pages. A document
# counts once (one URL, one text hash) and a failed fetch never counts.
SUBMIT_MIN_DOCS = _env_int("SIBYL_SUBMIT_MIN_DOCS", 5)
# A run whose median documents per trial is below this, or whose share of
# documents read from wikipedia.org is above WIKIPEDIA_WARN_SHARE, is warned
# about (never failed) by scripts/ci/stage_health.py.
DEPTH_WARN_MEDIAN_DOCS = _env_float("SIBYL_DEPTH_WARN_MEDIAN_DOCS", 5.0)
WIKIPEDIA_WARN_SHARE = _env_float("SIBYL_WIKIPEDIA_WARN_SHARE", 0.5)
# Above this many characters the oldest tool results in a trial's transcript
# are replaced by a one-line stub that keeps the URL (sibyl/transcript.py).
# 400,000 characters is about 100,000 tokens, well inside Opus 5.5's window.
TRANSCRIPT_MAX_CHARS = _env_int("SIBYL_TRANSCRIPT_MAX_CHARS", 400_000)

# Quantile levels each trial must report (discretized CDF).
QUANTILE_LEVELS = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]

# --- Aggregation -----------------------------------------------------------
# "linear_pool" (mean of CDFs = mixture; widens on disagreement) or
# "vincent" (per-level quantile averaging).
AGGREGATION = _env_str("SIBYL_AGGREGATION", "linear_pool")

# --- Calibration hook (deferred) --------------------------------------------
CALIBRATION_ENABLED = _env_bool("SIBYL_CALIBRATION_ENABLED", False)

# --- Calibration advice (sibyl/advice.py) -----------------------------------
# A (hazard, metric) class gets its own advice at this many distinct scored
# questions; below it the pooled row (all four classes) stands in at the same
# threshold; below that no advice is written and the prompt carries none.
ADVICE_MIN_QUESTIONS = _env_int("SIBYL_ADVICE_MIN_QUESTIONS", 20)
# Share of questions held out WITHOUT advice (hash of "sibyl:" + question_id,
# so a rerun keeps its arm), so the advice's effect can be measured.
ADVICE_EXPERIMENT_SHARE = _env_float("SIBYL_ADVICE_EXPERIMENT_SHARE", 0.5)

# --- Budget ------------------------------------------------------------------
# Hard run cut-off: stop STARTING new questions/trials once cumulative run
# cost reaches this. The in-flight unit runs to completion, so realized
# spend can exceed the cap by roughly one question's cost.
#
# Measured cost (2026-10-14, sibyl/cost_projection.py over one recorded
# eight-step trial of the 13 October run, priced from model_costs.json):
# about $0.60 a trial and $1.30 a picked question at K=2 (a step costs about
# $0.07, of which ~$0.045 is output: Opus 5.5 thinks ~1,200 tokens and
# answers ~900 every step). A control runs one trial, about $0.65. So 13
# picks and 2 controls come to about $18.50 a run. The same steps in the
# transcript shape used until 2026-10-14 project to $4.11 a question at K=3
# (13 October measured $5.04 with extra trials). The owner's estimate of
# $0.85 a question assumed far fewer output tokens than Opus 5.5 spends.
RUN_HARD_CAP_USD = _env_float("SIBYL_RUN_HARD_CAP_USD", 25.0)

# Secondary per-question guard (checked before each batch of trials).
_bpq = _env_float("SIBYL_BUDGET_USD_PER_QUESTION", 2.50)
BUDGET_USD_PER_QUESTION: float | None = _bpq if _bpq > 0 else None

# The cost per picked question the run is held to (2026-10-14). The CI cost
# gate (tests/test_sibyl_cost_gate.py) fails when the offline projection
# exceeds it by more than 50%. At run time, from the third question on, a
# realised cost per question (a one-trial control counted as 1/K) above
# COST_GATE_WARN_RATIO x this raises a workflow warning and a bundle
# anomaly, and above COST_GATE_STOP_RATIO x this no new question starts
# (skip reason "cost_per_question_gate"). Re-measure with
# ``python -m sibyl.cost_projection`` whenever the agent or the prompt changes.
STATED_COST_PER_QUESTION_USD = _env_float("SIBYL_STATED_COST_PER_QUESTION_USD", 1.40)
COST_GATE_AFTER_QUESTIONS = _env_int("SIBYL_COST_GATE_AFTER_QUESTIONS", 3)
COST_GATE_WARN_RATIO = _env_float("SIBYL_COST_GATE_WARN_RATIO", 1.5)
COST_GATE_STOP_RATIO = _env_float("SIBYL_COST_GATE_STOP_RATIO", 2.5)

# Wall-clock limit on the trial loop, in minutes. Same behaviour as the budget
# cap: past it no new question starts, the question in flight finishes, and
# the rest are stored as skipped ("run time cap"). The Sibyl job is the forecast
# chain's ONLY release trigger and times out at 330 minutes; the interpreter,
# bundles and canonical upload still have to run after the loop. 0 = no limit.
MAX_RUNTIME_MIN = _env_float("SIBYL_MAX_RUNTIME_MIN", 180.0)

# --- Question selection ------------------------------------------------------
# Since 2026-10-14 (sibyl/select_questions.py::select_on_rc): N_QUESTIONS is
# the whole run, N_QUESTIONS - N_CONTROL picks plus N_CONTROL controls. A
# pick must carry a regime-change flag (hs_triage.regime_change_level at
# least MIN_RC_LEVEL, and at least HAZARD_MIN_RC_LEVEL for a hazard named
# there). Picks are ranked RC level 2 and 3 first, then level 1, each by
# falling volatility. Each hazard first takes its FLOOR, then the rest go by
# rank to hazards under their CAP; slots no hazard could fill go to the
# hazards in OVERFLOW_ORDER, beyond their cap. Run order: controls, floor
# picks, fill picks, so a budget cut removes fill picks first and never the
# controls (owner decision).
N_QUESTIONS = _env_int("SIBYL_N_QUESTIONS", 15)
MIN_RC_LEVEL = _env_int("SIBYL_MIN_RC_LEVEL", 1)


def _parse_overrides(raw: str) -> dict:
    out = {}
    for part in (raw or "").split(","):
        if ":" not in part:
            continue
        hz, _, n = part.partition(":")
        try:
            out[hz.strip().upper()] = int(n.strip())
        except ValueError:
            continue
    return out


FLOOR_PER_HAZARD = _parse_overrides(_env_str("SIBYL_FLOOR_PER_HAZARD", "ACE:4,DR:2,FL:1,TC:0"))
CAP_PER_HAZARD = _parse_overrides(_env_str("SIBYL_CAP_PER_HAZARD", "ACE:7,DR:3,FL:2,TC:1"))
HAZARD_MIN_RC_LEVEL = _parse_overrides(_env_str("SIBYL_HAZARD_MIN_RC_LEVEL", "TC:2"))
OVERFLOW_ORDER = tuple(
    h.strip().upper() for h in _env_str("SIBYL_OVERFLOW_ORDER", "ACE,DR").split(",") if h.strip()
)
# Controls: questions with no RC flag (level 0, or no triage row), drawn by a
# hash of the run and the question id, from these classes. A control runs
# one trial on the no-pack lane, needs one valid trial and gets no extras.
N_CONTROL = _env_int("SIBYL_N_CONTROL", 2)
CONTROL_CLASSES = tuple(
    tuple(c.strip().upper().split("/", 1))
    for c in _env_str("SIBYL_CONTROL_CLASSES", "ACE/FATALITIES").split(",")
    if "/" in c
)

# The floor-then-fill rule of 2 to 13 October 2026 (25 questions, a floor of
# 3 per hazard, FL and TC held at 3). Kept as a pure function for replay and
# for SIBYL_SELECTION_RULE=floor_then_fill; not the default.
SELECTION_RULE = _env_str("SIBYL_SELECTION_RULE", "rc_focus").strip().lower()
MIN_PER_HAZARD = _env_int("SIBYL_MIN_PER_HAZARD", 3)
MAX_PER_HAZARD = _env_int("SIBYL_MAX_PER_HAZARD", 10)
MAX_PER_HAZARD_OVERRIDES = _parse_overrides(
    _env_str("SIBYL_MAX_PER_HAZARD_OVERRIDES", "FL:3,TC:3")
)

# Numeric affected/fatalities magnitude questions only (spec scope:
# "ACE fatalities; DR/FL/TC affected"). DR "affected" is represented as
# PHASE3PLUS_IN_NEED in this codebase (there are no DR/PA questions).
# EVENT_OCCURRENCE (binary) is excluded by construction.
ELIGIBLE_HAZARD_METRICS = frozenset(
    {
        ("ACE", "FATALITIES"),
        ("FL", "PA"),
        ("TC", "PA"),
        ("DR", "PHASE3PLUS_IN_NEED"),
    }
)

# Which standard-track aggregate a Sibyl SPD is compared against (and
# attached to), in preference order. Single-sourced here — the API route
# (pythia/api/routes/sibyl.py) and sibyl/spd.py both import it. This module
# must stay import-light (os only) so the API process can import it without
# pulling in the sibyl agent/provider tree.
STANDARD_MODEL_PREFERENCE = ("ensemble_bayesmc_v2", "ensemble_mean_v2", "track2_flash")

# --- Time / backtest ---------------------------------------------------------
# Live mode: asOf = now. Backtest mode: asOf = the question's window anchor
# (window_start_date), and the leakage controls in sibyl/leakage.py become
# active filters instead of no-ops.
BACKTEST_MODE = _env_bool("SIBYL_BACKTEST_MODE", False)

# --- The resolving source's latest reading (Oct 2026, review Part 5) --------
# One reading per question, taken on the main thread before its trials and
# shown to every lane (sibyl/resolver_reading.py): the newest Phase 3+ rows
# for drought, the last six months of resolving rows and GDACS alerts for
# flood and cyclone. Nothing in backtest. Off -> the prompt is unchanged.
RESOLVER_READING = _env_bool("SIBYL_RESOLVER_READING", True)
# The live month-to-date ACLED read for conflict deaths. Off in code; on in
# run_sibyl.yml, which holds the ACLED credentials at step scope. With it off
# a conflict question shows no reading.
LIVE_LOOKUPS_ENABLED = _env_bool("SIBYL_LIVE_LOOKUPS_ENABLED", False)

# --- The structured-data starting pack (Oct 2026, review Part 6) ------------
# Since 2026-10-14 the pack is compared WITHIN each question: at K=2 the first
# trial (lane A) is shown the pipeline's structured feeds (sibyl/pack.py) and
# the second (lane B) is not. The question-level hashed arm (SIBYL_PACK_SHARE,
# salt "sibyl_pack:") is retired; its rows stay readable as the earlier
# design. A control runs one trial, on the no-pack lane. PACK_ENABLED=0 runs
# every trial without the pack (tests use it: the loaders reach the network).
# The conflict forecasts (VIEWS, conflictforecast.org, ACLED CAST) are left
# out unless PACK_INCLUDE_FORECASTS. Nothing in backtest.
PACK_ENABLED = _env_bool("SIBYL_PACK_ENABLED", True)
PACK_LANE = _env_str("SIBYL_PACK_LANE", "A").strip().upper()
NO_PACK_LANE = _env_str("SIBYL_NO_PACK_LANE", "B").strip().upper()
PACK_INCLUDE_FORECASTS = _env_bool("SIBYL_PACK_INCLUDE_FORECASTS", False)
PACK_MAX_CHARS = _env_int("SIBYL_PACK_MAX_CHARS", 24000)
# The earlier (question-level) design's thresholds, kept for its rows.
PACK_MIN_QUESTIONS_IMMEDIATE = _env_int("SIBYL_PACK_MIN_QUESTIONS_IMMEDIATE", 10)
PACK_MIN_QUESTIONS_SCORED = _env_int("SIBYL_PACK_MIN_QUESTIONS_SCORED", 20)
# The paired design compares two trials of ONE question, so question-to-
# question noise cancels and fewer questions are needed: immediate measures
# from 5 paired questions, the outcome from 10.
PACK_PAIRED_MIN_QUESTIONS_IMMEDIATE = _env_int("SIBYL_PACK_PAIRED_MIN_QUESTIONS_IMMEDIATE", 5)
PACK_PAIRED_MIN_QUESTIONS_SCORED = _env_int("SIBYL_PACK_PAIRED_MIN_QUESTIONS_SCORED", 10)

# --- Search ------------------------------------------------------------------
BRAVE_MAX_RESULTS = _env_int("SIBYL_BRAVE_MAX_RESULTS", 8)
BRAVE_TIMEOUT_SEC = _env_int("SIBYL_BRAVE_TIMEOUT_SEC", 20)
# Lookback windows for date-filtered searches (days before asOf): the
# "news" lane (default) and the "reference" lane (ten years, for base rates,
# past episodes and structural material). Both end at asOf.
SEARCH_WINDOW_DAYS = _env_int("SIBYL_SEARCH_WINDOW_DAYS", 120)
REFERENCE_WINDOW_DAYS = _env_int("SIBYL_REFERENCE_WINDOW_DAYS", 3650)
FETCH_URL_TIMEOUT_SEC = _env_int("SIBYL_FETCH_URL_TIMEOUT_SEC", 20)
FETCH_URL_MAX_CHARS = _env_int("SIBYL_FETCH_URL_MAX_CHARS", 6000)
# The page body is read up to this many bytes and no further: the model only
# ever sees FETCH_URL_MAX_CHARS of text, and an unbounded read lets a hostile
# page fill the runner's memory.
FETCH_URL_MAX_BYTES = _env_int("SIBYL_FETCH_URL_MAX_BYTES", 3_000_000)
FETCH_URL_MAX_REDIRECTS = _env_int("SIBYL_FETCH_URL_MAX_REDIRECTS", 5)

# --- Document reader (Oct 2026, sibyl/reader.py) ------------------------------
# PDFs may be larger than pages; their first two pages plus the pages that
# score highest on the country and the question's terms are kept, up to
# PDF_MAX_PAGES; every document's text is capped at DOC_MAX_CHARS.
FETCH_PDF_MAX_BYTES = _env_int("SIBYL_FETCH_PDF_MAX_BYTES", 15_000_000)
PDF_MAX_PAGES = _env_int("SIBYL_PDF_MAX_PAGES", 40)
DOC_MAX_CHARS = _env_int("SIBYL_DOC_MAX_CHARS", 80_000)
# Documents longer than this go to the cheaper extraction model (role
# sibyl_extraction) with the agent's extraction_request; shorter ones are
# shown whole. On extraction failure the first this-many characters are shown.
EXTRACTION_SKIP_CHARS = _env_int("SIBYL_EXTRACTION_SKIP_CHARS", 6000)
EXTRACTION_MAX_WORDS = _env_int("SIBYL_EXTRACTION_MAX_WORDS", 500)

# --- ReliefWeb (Oct 2026) -----------------------------------------------------
# reliefweb.int answers page fetches with HTTP 202 (45 of 45 failed in the
# Sept 2026 runs), so Sibyl reads reports through the API, with the
# RELIEFWEB_APPNAME the resolution machine already uses.
RELIEFWEB_API_BASE = _env_str("SIBYL_RELIEFWEB_API_BASE", "https://api.reliefweb.int/v2")
RELIEFWEB_MAX_RESULTS = _env_int("SIBYL_RELIEFWEB_MAX_RESULTS", 10)
RELIEFWEB_TIMEOUT_SEC = _env_float("SIBYL_RELIEFWEB_TIMEOUT_SEC", 30.0)

# --- LLM call limits ----------------------------------------------------------
ANTHROPIC_MAX_ATTEMPTS = _env_int("SIBYL_ANTHROPIC_MAX_ATTEMPTS", 3)

# --- Evidence gate (Oct 2026) -------------------------------------------------
# A trial has evidence when it ran at least MIN_SEARCH_OK searches that
# returned a source and read at least MIN_DOCS_READ documents. A question is
# pooled from trials that are ok AND have evidence; below MIN_VALID_TRIALS such
# trials it is stored failed ("no evidence") and nothing is written to the
# forecast tables. The July 2026 run stored ten "ok" forecasts with every
# search failed; these defaults catch a blind trial and no more (Opus 5.5
# averages three searches a trial, so a stricter gate would fail most
# questions until the agent is made to research in depth).
# Raised from 1 / 0 in Part 3 (Oct 2026), once the plan and the submit gate
# make the agent research in depth.
MIN_SEARCH_OK = _env_int("SIBYL_MIN_SEARCH_OK", 3)
MIN_DOCS_READ = _env_int("SIBYL_MIN_DOCS_READ", 2)
# One since 2026-10-14 (K=2): a question with one valid trial is stored with
# its single_trial flag set, rather than lost.
MIN_VALID_TRIALS = _env_int("SIBYL_MIN_VALID_TRIALS", 1)

# --- Brave circuit breaker (Oct 2026) -----------------------------------------
# The shared breaker trips after three consecutive Brave failures and then
# short-circuits every later call for the life of the process. Sibyl resets it
# at the start of a run, and when a search finds it tripped it waits this long,
# resets it and retries once, at most BREAKER_MAX_RESETS times a run (a dead
# key must not loop).
BREAKER_COOLDOWN_SEC = _env_float("SIBYL_BREAKER_COOLDOWN_SEC", 60.0)
BREAKER_MAX_RESETS = _env_int("SIBYL_BREAKER_MAX_RESETS", 5)

# A run whose share of failed searches exceeds this is reported degraded by
# scripts/ci/stage_health.py.
DEGRADED_SEARCH_FAIL_SHARE = _env_float("SIBYL_DEGRADED_SEARCH_FAIL_SHARE", 0.20)

# --- Written vectors (Oct 2026) ---------------------------------------------
# Every bucket of every vector Sibyl writes is floored here and renormalised.
# Interpolation left ten of 258 buckets at exactly zero; compute_scores floors
# at 1e-9, so such a bucket occurring costs about 20.7 nats of log loss.
BUCKET_FLOOR = _env_float("SIBYL_BUCKET_FLOOR", 0.005)

# --- Reference and pooling (Oct 2026, sibyl/reference.py) --------------------
# Sibyl publishes, per window month, REFERENCE_WEIGHT x its mechanical
# reference + the rest x its pooled trials (the trials alone when there is no
# reference). 0.5 is a starting value; Part 6 fits it once 20 questions have
# both series scored.
REFERENCE_WEIGHT = _env_float("SIBYL_REFERENCE_WEIGHT", 0.5)
# DR/PHASE3PLUS_IN_NEED reference: this weight on "the last figure persists",
# the rest on the 36-month history vector. A starting value, no backtest.
DR_PERSISTENCE_WEIGHT = _env_float("SIBYL_DR_PERSISTENCE_WEIGHT", 0.5)


def _dr_persistence_weights() -> tuple:
    """Six weights, one per window month (Oct 2026, review Part 7).

    ``SIBYL_DR_PERSISTENCE_WEIGHTS`` is six comma-separated numbers in [0, 1];
    unset, malformed or the wrong length, every month takes
    ``SIBYL_DR_PERSISTENCE_WEIGHT``, which leaves the reference byte-identical
    to before. sibyl/reference_backtest.py proposes a schedule; nothing sets
    one by itself.
    """
    raw = os.getenv("SIBYL_DR_PERSISTENCE_WEIGHTS", "").strip()
    if raw:
        try:
            vals = tuple(float(x) for x in raw.split(","))
            if len(vals) == 6 and all(0.0 <= v <= 1.0 for v in vals):
                return vals
        except ValueError:
            pass
    return (float(DR_PERSISTENCE_WEIGHT),) * 6


DR_PERSISTENCE_WEIGHTS = _dr_persistence_weights()

# --- Measurement, pool weight and post-mortems (Oct 2026, Part 6) ----------
# The pre-extraction text of a document read is stored in sibyl_evidence up
# to this many characters (the shown text is stored whole).
EVIDENCE_DOC_MAX_CHARS = _env_int("SIBYL_EVIDENCE_DOC_MAX_CHARS", 40_000)

# How the reference weight in the published pool is set. "fitted" reads the
# newest sibyl_pool_weights row (sibyl/score_variants.py chooses it from
# {0.25, 0.5, 0.75} once POOL_WEIGHT_MIN_QUESTIONS questions carry both a raw
# and a reference score, with a prior worth POOL_WEIGHT_PRIOR_QUESTIONS
# questions at 0.5); with no row it falls back to REFERENCE_WEIGHT. "fixed"
# always uses REFERENCE_WEIGHT. Backtest always uses REFERENCE_WEIGHT.
REFERENCE_WEIGHT_MODE = _env_str("SIBYL_REFERENCE_WEIGHT_MODE", "fitted").strip().lower()
POOL_WEIGHT_GRID = (0.25, 0.5, 0.75)
POOL_WEIGHT_MIN_QUESTIONS = _env_int("SIBYL_POOL_WEIGHT_MIN_QUESTIONS", 20)
POOL_WEIGHT_PRIOR_QUESTIONS = _env_int("SIBYL_POOL_WEIGHT_PRIOR_QUESTIONS", 20)
# Rearranged variants (review Part 3, sibyl/score_variants.py): the
# disagreement-weighted variant puts 0.75 on the reference when the largest
# pairwise month-1 JSD among the pooled trials is above DW_HIGH_JSD, 0.25
# below DW_LOW_JSD, else 0.5. Fixed: never fitted on outcomes. A variant's
# comparison with sibyl says "not yet" below VARIANT_MIN_QUESTIONS.
DW_HIGH_JSD = _env_float("SIBYL_DW_HIGH_JSD", 0.10)
DW_LOW_JSD = _env_float("SIBYL_DW_LOW_JSD", 0.03)
VARIANT_MIN_QUESTIONS = _env_int("SIBYL_VARIANT_MIN_QUESTIONS", 20)

# Post-mortems (sibyl/postmortem.py): a note on each newly resolved
# question, and per class, once POSTMORTEM_MIN_NOTES notes exist, lessons of
# at most LESSONS_MAX_CHARS, each resting on LESSON_MIN_CASES cases or more.
POSTMORTEM_EFFORT = _env_str("SIBYL_POSTMORTEM_EFFORT", "medium").strip().lower()
POSTMORTEM_CAP_USD = _env_float("SIBYL_POSTMORTEM_CAP_USD", 5.0)
POSTMORTEM_MIN_NOTES = _env_int("SIBYL_POSTMORTEM_MIN_NOTES", 8)
LESSONS_MAX_CHARS = _env_int("SIBYL_LESSONS_MAX_CHARS", 6000)
LESSON_MIN_CASES = _env_int("SIBYL_LESSON_MIN_CASES", 3)
MAX_ANALOGUES = _env_int("SIBYL_MAX_ANALOGUES", 4)
# Failure types (Oct 2026, review Part 2): the note prompt carries the
# trials' plan findings, reconciliations and ledgers inside this many
# characters (whole ledger items dropped from the end beyond it), and a
# label's share of questions is shown from this many labelled questions.
POSTMORTEM_PROMPT_MAX_CHARS = _env_int("SIBYL_POSTMORTEM_PROMPT_MAX_CHARS", 20_000)
FAILURE_RATE_MIN_QUESTIONS = _env_int("SIBYL_FAILURE_RATE_MIN_QUESTIONS", 10)

# --- Shadow arm (Oct 2026, Part 7) -------------------------------------------
# One extra lane C trial per selected question (controls excluded) on a second
# model family, run AFTER every question's production trials. Its result
# replaces the Claude lane C trial in a shadow copy of the pool, stored in
# sibyl_forecasts.shadow_json and scored as __ext_sibyl_shadow; it never
# reaches forecasts_raw or forecasts_ensemble and never changes what Sibyl
# publishes. A registry alias or provider:model_id; only OpenAI is wired.
SHADOW_MODEL = _env_str("SIBYL_SHADOW_MODEL", "gpt").strip()
SHADOW_EFFORT = _env_str("SIBYL_SHADOW_EFFORT", "high").strip().lower()
# The shadow arm runs in production runs up to and including this month,
# then stops by itself. "off" switches it off.
SHADOW_UNTIL = _env_str("SIBYL_SHADOW_UNTIL", "2027-04").strip()
# A shadow trial does not start once the run is within this many dollars of
# its hard cap, or this many minutes of its time cap: the shadow arm is
# skipped first, so production work always has the room.
SHADOW_HEADROOM_USD = _env_float("SIBYL_SHADOW_HEADROOM_USD", 2.0)
SHADOW_HEADROOM_MIN = _env_float("SIBYL_SHADOW_HEADROOM_MIN", 20.0)
# The paired difference shadow minus sibyl is reported only from this many
# distinct scored questions; below it the report says "not yet".
SHADOW_MIN_QUESTIONS = _env_int("SIBYL_SHADOW_MIN_QUESTIONS", 20)
