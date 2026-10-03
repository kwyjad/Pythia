# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The append-only transcript of a trial (Oct 2026).

Until October 2026 each step's prompt carried only the belief state and the
LAST tool result, so a figure read at step 2 was gone by step 4 unless the
model had copied it into a list of strings. Now every earlier step is kept:
the model's JSON as it returned it, the ledger ids the code assigned, and
each tool result. An entry is rendered ONCE when its step ends and reused
byte for byte afterwards, so the prompt of step n+1 begins with the prompt
of step n less its short closing instruction, and a provider cache can
serve that prefix.

The one exception is the size guard: above ``SIBYL_TRANSCRIPT_MAX_CHARS``
the oldest tool results are replaced by a one-line stub that keeps the URL.
That rewrites earlier text, so the cached prefix is lost for one step; it
fires rarely and only to keep a trial inside the context window.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from sibyl import config as _cfg


@dataclass
class ToolOutput:
    action: str
    target: str  # the query or the URL
    text: str
    url: str = ""
    stubbed: bool = False

    def render(self) -> str:
        return f"--- {self.action}: {self.target}\n{self.text}"

    def stub(self) -> None:
        keep = self.url or self.target
        self.text = (
            f"[result removed to keep the transcript within its size limit; "
            f"source: {keep}. The ledger items it gave are kept.]"
        )
        self.stubbed = True


@dataclass
class TranscriptEntry:
    step: int
    response: str
    ledger_block: str
    outputs: List[ToolOutput] = field(default_factory=list)
    note: str = ""  # a refused submit, a dropped submit
    text: str = ""

    def render(self) -> str:
        parts = [
            f"=== STEP {self.step}: YOUR RESPONSE ===",
            self.response.strip(),
            f"=== STEP {self.step}: LEDGER ITEMS ADDED ===",
            self.ledger_block,
            f"=== STEP {self.step}: RESULTS ===",
        ]
        body = "\n\n".join(o.render() for o in self.outputs)
        if self.note:
            body = (body + "\n\n" + self.note) if body else self.note
        parts.append(body or "(no tool calls)")
        self.text = "\n".join(parts) + "\n\n"
        return self.text


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

    def size(self) -> int:
        return sum(len(e.text) for e in self.entries)

    def _guard(self, max_chars: Optional[int] = None) -> None:
        limit = int(max_chars if max_chars is not None else _cfg.TRANSCRIPT_MAX_CHARS)
        if limit <= 0 or self.size() <= limit:
            return
        for entry in self.entries:
            changed = False
            for out in entry.outputs:
                if out.stubbed:
                    continue
                out.stub()
                self.n_stubbed += 1
                changed = True
                entry.render()
                if self.size() <= limit:
                    return
            if changed:
                entry.render()
