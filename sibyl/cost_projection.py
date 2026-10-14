# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""What a Sibyl question costs, projected offline (2026-10-14).

Runs one recorded trial through the REAL prompt builder and transcript code
(``sibyl.agent.run_trial``) with a stubbed model and stubbed tools, builds
each step's Anthropic request body with ``build_anthropic_body`` exactly as
the default path does, and prices it under Anthropic's prompt-cache rule:

* a cache entry is written for the prefix ending at each marked block;
* a later request reads the longest cached prefix that ends at a block
  boundary within ``CACHE_LOOKBACK_BLOCKS`` (20) blocks before one of its
  own marked blocks;
* what lies between the read point and the last marked block is written
  (at the 5-minute write rate), what follows the last mark is plain input.

Every price comes from ``pythia/model_costs.json`` through
``forecaster.providers.compute_cost_split_usd``, the one cost formula.
Tokens are characters / 4, the same estimate the provider layer uses.

``project_question`` gives the cost of one picked question at K trials (the
first with the structured-data pack); ``legacy=True`` prices the same steps
in the transcript shape used until 2026-10-14 (one growing block holding
every raw result), which is what made 13 October cost $5.04 a question.

Used by the CI cost gate (tests/test_sibyl_cost_gate.py) and runnable by
hand: ``python -m sibyl.cost_projection``.
"""

from __future__ import annotations

import json
import os
from contextlib import ExitStack
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from unittest import mock

CHARS_PER_TOKEN = 4
BRAVE_USD_PER_QUERY = 0.005
EXTRACTION_MODEL = "claude-haiku-4-5-20251001"
EXTRACTION_OUTPUT_TOKENS = 700
DEFAULT_FIXTURE = (
    Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "sibyl" / "cost_gate" / "trial.json"
)
_FILLER = (
    "Reported clashes and casualty figures in the district, with dates, "
    "sources and the agencies that published them. "
)


def _tok(chars: int) -> int:
    return int(round(chars / CHARS_PER_TOKEN))


@dataclass
class StepCost:
    step: int
    prompt_tokens: int
    cache_read: int
    cache_write: int
    uncached: int
    completion_tokens: int
    cost_usd: float


@dataclass
class TrialProjection:
    steps: List[StepCost] = field(default_factory=list)
    brave_usd: float = 0.0
    extraction_usd: float = 0.0

    @property
    def opus_usd(self) -> float:
        return sum(s.cost_usd for s in self.steps)

    @property
    def total_usd(self) -> float:
        return self.opus_usd + self.brave_usd + self.extraction_usd


def simulate_cache(requests: Sequence[Sequence[Dict[str, Any]]],
                   *, lookback: int = 20) -> List[Dict[str, int]]:
    """Token counts for a sequence of requests' content blocks (see module doc)."""
    store: set = set()
    out: List[Dict[str, int]] = []
    for blocks in requests:
        texts = [b.get("text", "") for b in blocks]
        cum = [0]
        for t in texts:
            cum.append(cum[-1] + len(t))
        keys = []
        key = ""
        for t in texts:
            key = key + "\x00" + t
            keys.append(hash(key))
        marked = [i for i, b in enumerate(blocks) if "cache_control" in b]
        read_end = -1
        for m in marked:
            for j in range(m, max(-1, m - lookback - 1), -1):
                if keys[j] in store:
                    read_end = max(read_end, j)
                    break
        last = marked[-1] if marked else -1
        read_chars = cum[read_end + 1] if read_end >= 0 else 0
        write_chars = (cum[last + 1] - read_chars) if last > read_end else 0
        total = cum[-1]
        for m in marked:
            store.add(keys[m])
        out.append({
            "prompt_tokens": _tok(total),
            "cache_read_input_tokens": _tok(read_chars),
            "cache_creation_input_tokens": _tok(write_chars),
        })
    return out


def _legacy_requests(requests: List[List[Dict[str, Any]]]) -> List[List[Dict[str, Any]]]:
    """The same steps in the pre-2026-10-14 shape: one marked transcript block
    holding every entry AND every raw result, then a short tail."""
    from sibyl.transcript import RESULTS_HEADING  # noqa: PLC0415

    head_marker = RESULTS_HEADING.split("{step}")[0]
    out: List[List[Dict[str, Any]]] = []
    for n, blocks in enumerate(requests):
        head = [dict(b) for b in blocks[:3]]
        entries = [b["text"] for b in blocks[3:-1]]
        tails = [requests[k][-1]["text"] for k in range(1, n + 1)]
        raws = [t[: t.rfind("=== STEP ")] if head_marker in t else "" for t in tails]
        joined = "".join(e + r for e, r in zip(entries, raws))
        tail = blocks[-1]["text"]
        step_tail = tail[tail.rfind("=== STEP "):]
        new = head + ([{"type": "text", "text": joined, "cache_control": {"type": "ephemeral"}}]
                      if joined else [])
        new.append({"type": "text", "text": step_tail})
        out.append(new)
    return out


def _price(model: str, usage: Dict[str, int]) -> float:
    from forecaster.providers import compute_cost_split_usd  # noqa: PLC0415

    return float(compute_cost_split_usd(model, usage)[2])


def project_trial(
    fixture: Dict[str, Any],
    *,
    with_pack: bool = False,
    pack_chars: int = 5000,
    legacy: bool = False,
    model: Optional[str] = None,
) -> TrialProjection:
    """Run the fixture's trial through ``run_trial`` and price every step."""
    import sibyl.agent as agent  # noqa: PLC0415
    from forecaster.providers import build_anthropic_body  # noqa: PLC0415
    from sibyl import config as cfg  # noqa: PLC0415
    from sibyl.cost import CostTracker  # noqa: PLC0415
    from sibyl.select_questions import SibylQuestion  # noqa: PLC0415
    from sibyl.tools import ToolResult  # noqa: PLC0415

    model = model or cfg.MODEL
    replies = list(fixture["replies"])
    sizes = [list(x) for x in fixture["tool_result_chars"]]
    completion = list(fixture["completion_tokens"])
    requests: List[List[Dict[str, Any]]] = []
    calls_done = {"step": 0, "i": 0}
    proj = TrialProjection()

    def filler(n: int) -> str:
        return (_FILLER * (n // len(_FILLER) + 1))[:max(n, 1)]

    def next_size() -> int:
        step = len(requests)  # tools of the step just answered
        row = sizes[step - 1] if 0 < step <= len(sizes) else []
        if calls_done["step"] != step:
            calls_done.update(step=step, i=0)
        i = calls_done["i"]
        calls_done["i"] += 1
        return row[i] if i < len(row) else 2000

    def fake_call(prompt: str, *, cache_segments=None):
        body = build_anthropic_body(prompt, model, 1.0, purpose="sibyl_step",
                                    cache_segments=cache_segments)
        content = body["messages"][0]["content"]
        requests.append(content if isinstance(content, list) else [{"type": "text", "text": content}])
        idx = len(requests) - 1
        reply = replies[min(idx, len(replies) - 1)]
        return reply, {"cost_usd": 0.0}, ""

    def fake_brave(query, as_of, **kw):
        proj.brave_usd += BRAVE_USD_PER_QUERY
        from pythia.web_research.types import EvidenceSource  # noqa: PLC0415

        return ToolResult(tool="brave_search", ok=True, text=filler(next_size()),
                          sources=[EvidenceSource(title="t", url=f"https://s/{len(requests)}")])

    def fake_fetch(url, as_of, **kw):
        n = next_size()
        doc = filler(max(n, cfg.EXTRACTION_SKIP_CHARS + 1)) + str(url)
        return ToolResult(tool="fetch_url", ok=True, text=filler(n), doc_text=doc, url=url)

    def fake_extract(doc, req, **kw):
        usage = {"prompt_tokens": _tok(len(doc) + 1500), "completion_tokens": EXTRACTION_OUTPUT_TOKENS}
        proj.extraction_usd += _price(EXTRACTION_MODEL, usage)
        return mock.Mock(text=filler(3000), extracted=True, cost_usd=0.0)

    def fake_rw(query, as_of, **kw):
        return ToolResult(tool="reliefweb_search", ok=True, text=filler(next_size()))

    question = SibylQuestion(
        question_id="PROJ_ACE_FATALITIES_2026-11", hs_run_id="hs_proj", iso3="YEM",
        hazard_code="ACE", metric="FATALITIES", target_month="2027-04",
        window_start_date=date(2026, 11, 1),
        wording="How many conflict fatalities will ACLED record in Yemen in each month?",
        volatility_score=0.6, triage_score=0.0, rc_level=2,
    )
    reference = mock.Mock(prompt_text=filler(2600), by_month=None)
    with ExitStack() as st:
        st.enter_context(mock.patch.dict(os.environ, {"PYTHIA_PROMPT_CACHE_ENABLED": "1"}))
        st.enter_context(mock.patch.object(agent, "_call_model", fake_call))
        st.enter_context(mock.patch.object(agent, "brave_search", fake_brave))
        st.enter_context(mock.patch.object(agent, "fetch_url", fake_fetch))
        st.enter_context(mock.patch.object(agent, "reliefweb_search", fake_rw))
        st.enter_context(mock.patch.object(agent.sibyl_extract, "extract", fake_extract))
        st.enter_context(mock.patch.object(agent, "submit_gate_missing", lambda *a, **k: []))
        st.enter_context(mock.patch.object(agent, "log_sibyl_call", lambda **kw: None))
        agent.run_trial(
            question, reference, as_of=date(2026, 10, 13), trial_index=0, run_id="proj",
            tracker=CostTracker(run_hard_cap_usd=1e9), forecast_months=[
                "2026-11", "2026-12", "2027-01", "2027-02", "2027-03", "2027-04"],
            country_name="Yemen", log_sink=[],
            resolver_reading=filler(1500),
            pack=(filler(pack_chars) if with_pack else ""),
        )
    reqs = _legacy_requests(requests) if legacy else requests
    for n, usage in enumerate(simulate_cache(reqs), start=1):
        comp = completion[min(n - 1, len(completion) - 1)]
        full = dict(usage, completion_tokens=comp)
        uncached = usage["prompt_tokens"] - usage["cache_read_input_tokens"] - usage["cache_creation_input_tokens"]
        proj.steps.append(StepCost(n, usage["prompt_tokens"], usage["cache_read_input_tokens"],
                                   usage["cache_creation_input_tokens"], uncached, comp,
                                   _price(model, full)))
    return proj


def load_fixture(path: Optional[Path] = None) -> Dict[str, Any]:
    return json.loads(Path(path or DEFAULT_FIXTURE).read_text())


def project_question(fixture: Optional[Dict[str, Any]] = None, *, k: Optional[int] = None,
                     legacy: bool = False) -> Tuple[float, List[TrialProjection]]:
    """Cost of one picked question: K trials, the first with the pack."""
    from sibyl import config as cfg  # noqa: PLC0415

    fixture = fixture or load_fixture()
    k = int(cfg.K if k is None else k)
    trials = [project_trial(fixture, with_pack=(i == 0), legacy=legacy) for i in range(k)]
    return sum(t.total_usd for t in trials), trials


def main() -> None:  # pragma: no cover - a report for people
    new, trials = project_question()
    old, old_trials = project_question(legacy=True, k=3)
    print(f"projected cost per question: ${new:.3f} at K=2 "
          f"(the pre-2026-10-14 shape at K=3: ${old:.3f})")
    for s in trials[1].steps:
        print(f"  step {s.step:2d}: prompt {s.prompt_tokens:6d} read {s.cache_read:6d} "
              f"write {s.cache_write:6d} plain {s.uncached:6d} out {s.completion_tokens:5d} "
              f"${s.cost_usd:.4f}")
    print("  legacy steps: " + ", ".join(f"${s.cost_usd:.3f}" for s in old_trials[1].steps))


if __name__ == "__main__":  # pragma: no cover
    main()
