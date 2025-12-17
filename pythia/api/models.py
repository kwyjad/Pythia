from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ExtraFriendlyModel(BaseModel):
    model_config = ConfigDict(extra="allow")


class HsBundle(ExtraFriendlyModel):
    hs_run: Optional[Dict[str, Any]] = None
    triage: Optional[Dict[str, Any]] = None
    scenario_ids: List[str] = Field(default_factory=list)
    scenarios: List[Dict[str, Any]] = Field(default_factory=list)
    country_report: Optional[Dict[str, Any]] = None


class ForecastBundle(ExtraFriendlyModel):
    forecaster_run_id: Optional[str] = None
    research: Optional[Dict[str, Any]] = None
    ensemble_spd: List[Dict[str, Any]] = Field(default_factory=list)
    raw_spd: List[Dict[str, Any]] = Field(default_factory=list)
    scenario_writer: List[Dict[str, Any]] = Field(default_factory=list)


class ContextBundle(ExtraFriendlyModel):
    question_context: Optional[Dict[str, Any]] = None
    resolutions: List[Dict[str, Any]] = Field(default_factory=list)


class LlmCallsBundle(ExtraFriendlyModel):
    included: bool = False
    transcripts_included: bool = False
    rows: List[Dict[str, Any]] = Field(default_factory=list)
    by_phase: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)


class QuestionBundleResponse(ExtraFriendlyModel):
    question: Dict[str, Any]
    hs: HsBundle
    forecast: ForecastBundle
    context: ContextBundle
    llm_calls: LlmCallsBundle
