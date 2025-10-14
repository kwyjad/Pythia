from __future__ import annotations
import hashlib, json
from dataclasses import dataclass

@dataclass(frozen=True)
class PromptSpec:
    key: str          # e.g., "hs.scenario.v1", "research.v1", "forecast.v1"
    version: str      # "1.0.0"
    sha256: str       # content hash
    path: str         # file path

def hash_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def load_prompt_spec(key: str, version: str, text: str, path: str) -> PromptSpec:
    return PromptSpec(key=key, version=version, sha256=hash_text(text), path=path)

def snapshot_for_run(component: str, spec: PromptSpec) -> dict:
    return {"component": component, "prompt_key": spec.key, "version": spec.version, "sha256": spec.sha256, "path": spec.path}
