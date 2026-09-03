# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Store each shared prompt prefix once instead of a few hundred times.

``llm_calls_detail.jsonl.gz`` was 9 MB of a 9.2 MB bundle, and nearly all
of it was the same static prefix repeated: the V3 prompt order exists
precisely to put the role, the buckets, the hazard guidance and the schema
first, identical across every question in a family. That is what makes the
provider cache work, and it is also what made the artifact large enough to
crowd out everything else worth adding.

So each distinct prefix is written once, keyed by the sha256 of its text,
and every call record carries the key plus its own variable tail. Nothing
is lost: joining the prefix to the tail reproduces the prompt byte for
byte, and the record says how. Responses are never touched — they are the
part a reader has come to read.

The prefix is found by longest common prefix within a group of prompts
that shared a builder, not by pattern matching on the text. A group is
(phase, call_type, provider, model, hazard, metric): two prompts from
different builders can share an opening paragraph, and cutting there would
put question-specific text into a "static" block.
"""

from __future__ import annotations

import hashlib
from typing import Any, Sequence

# Below this many characters a shared prefix is not worth a lookup: the
# record grows by the key and the reader loses the ability to read the
# prompt in one place.
MIN_PREFIX_CHARS = 400


def _common_prefix_len(a: str, b: str) -> int:
    limit = min(len(a), len(b))
    i = 0
    while i < limit and a[i] == b[i]:
        i += 1
    return i


def group_key(row: dict[str, Any]) -> tuple[str, ...]:
    """Prompts that came from one builder, and could legitimately share text."""

    return (
        str(row.get("phase") or ""),
        str(row.get("call_type") or ""),
        str(row.get("provider") or ""),
        str(row.get("model_id") or ""),
        str(row.get("hazard_code") or ""),
        str(row.get("metric") or ""),
    )


def _cut_at_boundary(text: str, length: int) -> int:
    """Cut on a line boundary so both halves stay readable."""

    if length <= 0:
        return 0
    newline = text.rfind("\n", 0, length)
    return newline + 1 if newline > 0 else length


def build_prefix_index(
    rows: Sequence[dict[str, Any]],
    *,
    prompt_key: str = "prompt_text",
    min_chars: int = MIN_PREFIX_CHARS,
) -> tuple[dict[str, dict[str, Any]], dict[Any, tuple[str, int]]]:
    """Return (prefixes_by_hash, {row id: (hash, prefix_len)}).

    Rows are identified by ``id()`` so the caller can map back without the
    rows needing a stable key of their own.
    """

    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        text = row.get(prompt_key) or ""
        if len(text) < min_chars:
            continue
        groups.setdefault(group_key(row), []).append(row)

    prefixes: dict[str, dict[str, Any]] = {}
    assignment: dict[Any, tuple[str, int]] = {}
    for key, members in groups.items():
        if len(members) < 2:
            continue
        texts = [str(m.get(prompt_key) or "") for m in members]
        shared = texts[0]
        for text in texts[1:]:
            shared = shared[: _common_prefix_len(shared, text)]
            if len(shared) < min_chars:
                break
        if len(shared) < min_chars:
            continue
        cut = _cut_at_boundary(shared, len(shared))
        if cut < min_chars:
            continue
        prefix_text = texts[0][:cut]
        digest = hashlib.sha256(prefix_text.encode("utf-8")).hexdigest()[:16]
        prefixes.setdefault(
            digest,
            {
                "prefix_sha256_16": digest,
                "chars": cut,
                "group": {
                    "phase": key[0], "call_type": key[1], "provider": key[2],
                    "model_id": key[3], "hazard_code": key[4], "metric": key[5],
                },
                "n_calls": 0,
                "text": prefix_text,
            },
        )
        prefixes[digest]["n_calls"] += len(members)
        for member in members:
            assignment[id(member)] = (digest, cut)
    return prefixes, assignment


def apply(
    record: dict[str, Any],
    row: dict[str, Any],
    assignment: dict[Any, tuple[str, int]],
    *,
    prompt_key: str = "prompt_text",
) -> dict[str, Any]:
    """Replace ``prompt_text`` with (prefix hash + tail) where one applies.

    A record that keeps its full prompt says so explicitly, so a consumer
    never has to guess whether an absent key means "no prefix" or "prompt
    lost".
    """

    hit = assignment.get(id(row))
    text = record.get(prompt_key) or ""
    if not hit:
        record["prompt_prefix_sha256"] = None
        record["prompt_is_complete"] = True
        return record
    digest, length = hit
    record["prompt_prefix_sha256"] = digest
    record["prompt_prefix_chars"] = length
    record[prompt_key] = text[length:]
    record["prompt_is_complete"] = False
    record["prompt_reconstruction"] = (
        "prompt_prefixes.json[prompt_prefix_sha256].text + prompt_text"
    )
    return record


def savings(prefixes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    written = sum(int(p["chars"]) for p in prefixes.values())
    would_have_been = sum(int(p["chars"]) * int(p["n_calls"]) for p in prefixes.values())
    return {
        "n_prefixes": len(prefixes),
        "prefix_chars_written_once": written,
        "prefix_chars_avoided": max(0, would_have_been - written),
    }
