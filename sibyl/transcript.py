# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The append-only transcript of a trial (Oct 2026; reshaped 2026-10-14).

Each earlier step stays in the prompt as one ENTRY: the model's JSON as it
returned it, the ledger items the code assigned, and one line per tool call
naming the query or URL and whether it worked. An entry is rendered once and
reused byte for byte, and it is sent as its own content block, so the prompt
of step n+1 begins with the blocks of step n less its tail and a provider
cache can serve that prefix.

A tool's raw result is NOT kept in the entry. It is shown once, in the
uncached tail of the next step (``pending_results``), and what the model
needs from it lives on in the ledger. Until 2026-10-14 every raw result
stayed in the transcript and the whole transcript was one growing text
block, so it never matched the cache and input grew with the square of the
step count: on 13 October a step-10 call cost $0.43 against $0.05 at step 1.

The size guard (``SIBYL_TRANSCRIPT_MAX_CHARS``) still exists: above it the
oldest entries' responses are replaced by a stub. That rewrites earlier text
and breaks the cached prefix, so it is logged; with results out of the
entries it should almost never fire.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from sibyl import config as _cfg

logger = logging.getLogger(__name__)

RESULTS_HEADING = (
    "=== RESULTS OF YOUR STEP {step} TOOL CALLS (shown this once; record in "
    "ledger_add every figure, date and source you may need later) ==="
)


@dataclass
class ToolOutput:
    action: str
    target: str  # the query or the URL
    text: str
    url: str = ""
    ok: bool = True
    stubbed: bool = False

    def render(self) -> str:
        return f"--- {self.action}: {self.target}\n{self.text}"

    def call_line(self) -> str:
        """The permanent one-line record of the call."""
        status = "ok" if self.ok else "failed"
        return f"- {self.action}: {self.target} ({status})"


@dataclass
class TranscriptEntry:
    step: int
    response: str
    ledger_block: str
    outputs: List[ToolOutput] = field(default_factory=list)
    note: str = ""  # a refused submit, a dropped submit
    text: str = ""
    stubbed: bool = False

    def render(self) -> str:
        response = (
            "[response removed to keep the transcript within its size limit; "
            "its ledger items are kept]"
            if self.stubbed else self.response.strip()
        )
        calls = "\n".join(o.call_line() for o in self.outputs) or "(no tool calls)"
        parts = [
            f"=== STEP {self.step}: YOUR RESPONSE ===",
            response,
            f"=== STEP {self.step}: LEDGER ITEMS ADDED ===",
            self.ledger_block,
            f"=== STEP {self.step}: TOOL CALLS ===",
            calls,
        ]
        if self.note:
            parts.append(self.note)
        self.text = "\n".join(parts) + "\n\n"
        return self.text

    def results_text(self) -> str:
        """The raw results, for the next step's tail only."""
        if not self.outputs:
            return ""
        body = "\n\n".join(o.render() for o in self.outputs)
        return RESULTS_HEADING.format(step=self.step) + "\n" + body + "\n\n"


@dataclass
class Transcript:
    entries: List[TranscriptEntry] = field(default_factory=list)
    n_stubbed: int = 0

    def append(self, entry: TranscriptEntry) -> None:
        entry.render()
        self.entries.append(entry)
        self._guard()

    def text(self) -> str:
        return "".join(e.text for e in self.entries)

    def blocks(self) -> List[str]:
        """One text per entry, in order: each is sent as its own block."""
        return [e.text for e in self.entries]

    def pending_results(self) -> str:
        """The newest entry's raw results (shown once, in the next tail)."""
        return self.entries[-1].results_text() if self.entries else ""

    def size(self) -> int:
        return sum(len(e.text) for e in self.entries)

    def _guard(self, max_chars: Optional[int] = None) -> None:
        limit = int(max_chars if max_chars is not None else _cfg.TRANSCRIPT_MAX_CHARS)
        if limit <= 0 or self.size() <= limit:
            return
        # Never the newest entry: the model has not seen its results yet.
        for entry in self.entries[:-1]:
            if entry.stubbed:
                continue
            entry.stubbed = True
            entry.render()
            self.n_stubbed += 1
            logger.warning(
                "sibyl.transcript: size guard stubbed step %d's response "
                "(transcript over %d chars); the cached prefix is lost for one step",
                entry.step, limit,
            )
            if self.size() <= limit:
                return
