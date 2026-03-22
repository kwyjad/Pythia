"use client";

import { useMemo, useState } from "react";
import type { ReactNode } from "react";

import CollapsiblePanel from "../../../components/CollapsiblePanel";
import BinaryPanel from "./BinaryPanel";
import SpdPanel from "./SpdPanel";
import { asArray, asString } from "../../../lib/excerpts";
import { extractForecastRationale } from "../../../lib/forecast_rationale";
import { formatModelList } from "../../../lib/model_names";
import { formatScenario } from "../../../lib/scenario_format";
import type { QuestionBundleResponse } from "../../../lib/types";

type QuestionDetailViewProps = {
  bundle: QuestionBundleResponse;
};

type ModelRow = Record<string, unknown>;

const getTextList = (value: unknown): string[] => {
  if (Array.isArray(value)) {
    return value.map((item) => String(item));
  }
  if (value && typeof value === "object") {
    const obj = value as Record<string, unknown>;
    const values = Object.values(obj).flatMap((item) =>
      Array.isArray(item) ? item : []
    );
    if (values.length) {
      return values.map((item) => String(item));
    }
  }
  if (typeof value === "string" && value.trim().length > 0) {
    return [value];
  }
  return [];
};

const addMonths = (ym: string | null | undefined, months: number): string | null => {
  if (!ym) return null;
  const [year, month] = ym.split("-").map(Number);
  if (!year || !month) return null;
  const total = year * 12 + (month - 1) + months;
  const nextYear = Math.floor(total / 12);
  const nextMonth = (total % 12) + 1;
  return `${nextYear}-${String(nextMonth).padStart(2, "0")}`;
};

const resolveLatestDate = (rows: ModelRow[]): string | null => {
  const dates = rows
    .map((row) => row.created_at)
    .filter((value): value is string => typeof value === "string")
    .map((value) => new Date(value))
    .filter((value) => !Number.isNaN(value.getTime()));
  if (!dates.length) return null;
  const latest = dates.reduce((max, current) => (current > max ? current : max));
  return latest.toISOString().slice(0, 10);
};

const getModelNamesForPhase = (
  byPhase: Record<string, ModelRow[]>,
  phase: string
): string => {
  const rows = byPhase?.[phase] ?? [];
  const names: string[] = [];
  rows.forEach((row) => {
    const name =
      (row.model_id as string | undefined) ??
      (row.model_name as string | undefined) ??
      (row.model as string | undefined);
    if (name) {
      names.push(name);
    }
  });
  return formatModelList(names);
};

const getModelNamesForPhases = (
  byPhase: Record<string, ModelRow[]>,
  phases: string[]
): string => {
  const names: string[] = [];
  phases.forEach((phase) => {
    const rows = byPhase?.[phase] ?? [];
    rows.forEach((row) => {
      const name =
        (row.model_id as string | undefined) ??
        (row.model_name as string | undefined) ??
        (row.model as string | undefined);
      if (name) {
        names.push(name);
      }
    });
  });
  return formatModelList(names);
};

const renderScenarioBlocks = (text?: string | null) => {
  if (!text) {
    return <p className="text-sm text-fred-text">No scenario narrative available.</p>;
  }
  const blocks = formatScenario(text);
  if (!blocks.length) {
    return <p className="text-sm text-fred-text">No scenario narrative available.</p>;
  }
  const elements: ReactNode[] = [];
  const bulletBuffer: string[] = [];
  let listIndex = 0;

  const flushBullets = () => {
    if (!bulletBuffer.length) return;
    elements.push(
      <ul
        key={`scenario-list-${listIndex}`}
        className="list-disc space-y-1 pl-5 text-sm text-fred-text"
      >
        {bulletBuffer.map((item, index) => (
          <li key={`scenario-li-${listIndex}-${index}`}>{item}</li>
        ))}
      </ul>
    );
    bulletBuffer.length = 0;
    listIndex += 1;
  };

  blocks.forEach((block, index) => {
    if (block.type === "li") {
      bulletBuffer.push(block.text);
      return;
    }
    flushBullets();
    if (block.type === "h2") {
      elements.push(
        <h3 key={`scenario-h2-${index}`} className="text-base font-semibold text-fred-primary">
          {block.text}
        </h3>
      );
      return;
    }
    if (block.type === "h3") {
      elements.push(
        <h4 key={`scenario-h3-${index}`} className="text-sm font-semibold text-fred-text">
          {block.text}
        </h4>
      );
      return;
    }
    elements.push(
      <p
        key={`scenario-p-${index}`}
        className="text-sm text-fred-text leading-relaxed"
      >
        {block.text}
      </p>
    );
  });

  flushBullets();

  return <div className="space-y-2">{elements}</div>;
};

const formatTimeframe = (value: unknown): string | null => {
  if (typeof value === "number" && Number.isFinite(value)) {
    return `${value} months`;
  }
  if (typeof value === "string" && value.trim().length > 0) {
    return value;
  }
  return null;
};

const renderParagraphs = (text: string) => {
  const blocks = text
    .split(/\n\s*\n/)
    .map((block) => block.trim())
    .filter(Boolean);
  return blocks.map((block, index) => (
    <p key={`paragraph-${index}`} className="text-sm text-fred-text leading-relaxed">
      {block.replace(/\s+/g, " ")}
    </p>
  ));
};

type PromptStage = {
  key: string;
  label: string;
  promptOnly?: boolean;
  dedup?: boolean;
};

const PROMPT_STAGES: PromptStage[] = [
  { key: "regime_change", label: "Regime Change", dedup: true },
  { key: "rc_grounding", label: "RC Grounding" },
  { key: "triage", label: "Triage", dedup: true },
  { key: "triage_grounding", label: "Triage Grounding" },
  { key: "adversarial_check", label: "Adversarial Check" },
  { key: "forecast", label: "Forecast (SPD)", promptOnly: true, dedup: true },
  { key: "scenario", label: "Scenario", promptOnly: true },
];

const renderPromptBlock = (label: string, text: string) => (
  <div>
    <div className="text-[11px] font-semibold uppercase tracking-wide text-fred-muted">{label}</div>
    <pre className="mt-1 max-h-96 overflow-auto whitespace-pre-wrap break-words rounded border border-fred-secondary bg-fred-surface p-2 text-xs text-fred-text">
      {text}
    </pre>
  </div>
);

const PromptViewerSection = ({
  groupedLlmCalls,
}: {
  groupedLlmCalls: Record<string, ModelRow[]>;
}) => (
  <section className="space-y-3">
    <h2 className="text-lg font-semibold text-fred-text">Model Prompts & Responses</h2>
    {PROMPT_STAGES.map((stage) => {
      const rows = groupedLlmCalls[stage.key] ?? [];
      if (!rows.length) {
        return (
          <CollapsiblePanel key={stage.key} title={stage.label}>
            <p className="text-sm text-fred-text">No data available for this stage.</p>
          </CollapsiblePanel>
        );
      }
      const firstPrompt = asString(rows[0]?.prompt_text) ?? "";
      const shouldDedup = stage.dedup && rows.length > 1;

      return (
        <CollapsiblePanel key={stage.key} title={stage.label}>
          <div className="space-y-4">
            {/* Shared prompt (for deduplicated stages) */}
            {shouldDedup && firstPrompt ? (
              <div className="rounded border border-fred-secondary bg-fred-surface p-3">
                {renderPromptBlock("Prompt", firstPrompt)}
              </div>
            ) : null}
            {/* Per-row content */}
            {rows.map((row, index) => {
              const modelName = asString(row.model_name) ?? asString(row.model_id) ?? "Unknown model";
              const timestamp = asString(row.timestamp) ?? asString(row.created_at) ?? "";
              const promptText = asString(row.prompt_text) ?? "";
              const responseText = asString(row.response_text) ?? "";
              return (
                <div key={`${stage.key}-${index}`} className="space-y-2 rounded border border-fred-secondary bg-fred-surface p-3">
                  <div className="flex flex-wrap items-center gap-2 text-xs text-fred-muted">
                    <span className="font-semibold">{modelName}</span>
                    {timestamp ? <span>{timestamp}</span> : null}
                  </div>
                  {/* Show prompt inline only if NOT deduplicated */}
                  {!shouldDedup && promptText ? renderPromptBlock("Prompt", promptText) : null}
                  {/* Show response unless promptOnly */}
                  {!stage.promptOnly && responseText
                    ? renderPromptBlock(
                        shouldDedup ? `Response ${index + 1}` : "Response",
                        responseText
                      )
                    : null}
                  {!promptText && !responseText ? (
                    <p className="text-xs text-fred-muted">No transcript data available for this call.</p>
                  ) : null}
                </div>
              );
            })}
          </div>
        </CollapsiblePanel>
      );
    })}
  </section>
);

const QuestionDetailView = ({ bundle }: QuestionDetailViewProps) => {
  const question = (bundle.question ?? {}) as Record<string, unknown>;
  const hs = (bundle.hs ?? {}) as Record<string, unknown>;
  const forecast = (bundle.forecast ?? {}) as Record<string, unknown>;
  const context = (bundle.context ?? {}) as Record<string, unknown>;
  const llm = (bundle.llm_calls ?? {}) as Record<string, unknown>;

  const ensemble = (forecast.ensemble_spd ?? []) as ModelRow[];
  const rawSpd = (forecast.raw_spd ?? []) as ModelRow[];
  const scenarioWriter = (forecast.scenario_writer ?? []) as ModelRow[];
  const triage = (hs.triage ?? {}) as Record<string, unknown>;
  const resolutions = (context.resolutions ?? []) as ModelRow[];
  const scores = (context.scores ?? []) as ModelRow[];
  const byPhase = (llm.by_phase ?? {}) as Record<string, ModelRow[]>;
  const forecastRationale = extractForecastRationale(bundle);

  const [scenarioIndex, setScenarioIndex] = useState(0);
  const selectedScenario = scenarioWriter[scenarioIndex] ?? scenarioWriter[0];

  const forecastDate =
    resolveLatestDate(ensemble) ?? resolveLatestDate(rawSpd) ?? "—";
  const targetMonth = (question.target_month as string | undefined) ?? null;
  const endMonth = addMonths(targetMonth, 5);
  const status = (question.status as string | undefined) ?? "—";

  const triageScore =
    typeof triage.triage_score === "number"
      ? triage.triage_score.toFixed(2)
      : asString(triage.triage_score) ?? "—";
  const triageTier = asString(triage.tier) ?? "—";
  const rcProbability =
    typeof triage.regime_change_likelihood === "number"
      ? triage.regime_change_likelihood.toFixed(2)
      : asString(triage.regime_change_likelihood) ?? "—";
  const rcDirection = asString(triage.regime_change_direction) ?? "—";
  const rcMagnitude =
    typeof triage.regime_change_magnitude === "number"
      ? triage.regime_change_magnitude.toFixed(2)
      : asString(triage.regime_change_magnitude) ?? "—";
  const rcScore =
    typeof triage.regime_change_score === "number"
      ? triage.regime_change_score.toFixed(2)
      : asString(triage.regime_change_score) ?? "—";
  const webSearchModels = getModelNamesForPhases(byPhase, [
    "hs_web_research",
    "research_web_research",
    "forecast_web_research",
  ]);
  const hsTriageModels = getModelNamesForPhase(byPhase, "hs_triage");
  const hasLlmPhases = Object.keys(byPhase).length > 0;
  const hsTriageDisplay =
    hsTriageModels === "—" && hasLlmPhases
      ? "— (no hs_triage llm_calls found)"
      : hsTriageModels;
  const debugEnabled = useMemo(() => {
    if (typeof window === "undefined") return false;
    const params = new URLSearchParams(window.location.search);
    if (params.get("debug_question") === "1") return true;
    try {
      return window.localStorage.getItem("pythia_debug_question") === "1";
    } catch {
      return false;
    }
  }, []);

  const resolutionForTarget = resolutions.find(
    (row) => row.observed_month === targetMonth
  );
  const resolutionSummary = resolutions.length
    ? `${resolutions.length} row(s)${
        resolutionForTarget?.value !== undefined
          ? ` • target month: ${resolutionForTarget.value}`
          : ""
      }`
    : "No resolutions yet";

  const requestedHsRunId = asString(question.requested_hs_run_id);
  const requestedHsRunIdMatched = question.requested_hs_run_id_matched;
  const showRequestedRunBanner =
    Boolean(requestedHsRunId) && requestedHsRunIdMatched === false;

  const scenarioStubText = asString(triage.scenario_stub);
  const dataQuality = (triage.data_quality_json ?? null) as Record<string, unknown> | null;
  const dataQualityEntries = [
    { label: "Source", value: asString(dataQuality?.resolution_source) },
    { label: "Reliability", value: asString(dataQuality?.reliability) },
    { label: "Notes", value: asString(dataQuality?.notes) },
  ].filter((entry) => entry.value);
  const selectedScenarioText = asString(selectedScenario?.text);

  // Triage narrative
  const triageConfidenceNote = asString(triage.confidence_note);

  // Regime change detail from regime_change_json
  const rcJson = (triage.regime_change_json ?? null) as Record<string, unknown> | null;
  const rcRationaleBullets = getTextList(rcJson?.rationale_bullets);
  const rcTriggerSignals = getTextList(rcJson?.trigger_signals);
  const rcConfidenceNote = asString(rcJson?.confidence_note);

  // Prompt viewer: group LLM call rows by pipeline stage
  const llmRows = (llm.rows ?? []) as ModelRow[];

  const groupedLlmCalls = useMemo(() => {
    const groups: Record<string, ModelRow[]> = {};
    PROMPT_STAGES.forEach((stage) => {
      groups[stage.key] = [];
    });
    llmRows.forEach((row) => {
      const hz = ((row.hazard_code as string) ?? "").toLowerCase();
      const phase = ((row.phase as string) ?? "").toLowerCase();
      if (hz.startsWith("rc_") && hz.includes("_pass_")) {
        groups.regime_change.push(row);
      } else if (hz.startsWith("grounding_")) {
        groups.rc_grounding.push(row);
      } else if (hz.startsWith("triage_") && hz.includes("_pass_")) {
        groups.triage.push(row);
      } else if (hz.startsWith("triage_grounding_")) {
        groups.triage_grounding.push(row);
      } else if (hz.includes("adversarial") || phase.includes("adversarial")) {
        groups.adversarial_check.push(row);
      } else if (phase === "spd_v2") {
        groups.forecast.push(row);
      } else if (phase === "scenario_v2") {
        groups.scenario.push(row);
      }
    });
    return groups;
  }, [llmRows]);

  const brierScores = useMemo(() => {
    return scores.filter((row) => {
      const scoreType = row.score_type as string | undefined;
      const modelName = row.model_name as string | undefined;
      return scoreType === "brier" && !modelName;
    });
  }, [scores]);

  const brierAverage = useMemo(() => {
    if (!brierScores.length) return null;
    const values = brierScores
      .map((row) => row.value)
      .filter((value): value is number => typeof value === "number");
    if (!values.length) return null;
    const total = values.reduce((sum, value) => sum + value, 0);
    return total / values.length;
  }, [brierScores]);

  const brierByHorizon = useMemo(() => {
    const map = new Map<number, number>();
    brierScores.forEach((row) => {
      const horizon = row.horizon_m as number | undefined;
      const value = row.value as number | undefined;
      if (typeof horizon === "number" && typeof value === "number") {
        map.set(horizon, value);
      }
    });
    return map;
  }, [brierScores]);

  return (
    <div className="space-y-8">
      <section className="space-y-2">
        <h1 className="text-2xl font-semibold">
          {(question.wording as string | undefined) ?? "Question detail"}
        </h1>
        <p className="text-sm text-fred-text">
          {(question.iso3 as string | undefined) ?? ""} •{" "}
          {(question.hazard_code as string | undefined) ?? ""} •{" "}
          {(question.metric as string | undefined) ?? ""} •{" "}
          {(question.target_month as string | undefined) ?? ""}
        </p>
        {showRequestedRunBanner ? (
          <div className="rounded border border-fred-secondary bg-fred-surface px-3 py-2 text-sm text-fred-text">
            Requested HS run not available; showing latest run instead.
          </div>
        ) : null}
      </section>

      <section className="rounded-lg border border-fred-secondary bg-fred-surface p-4 text-fred-text">
        <h2 className="text-lg font-semibold">Summary</h2>
        <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-6">
          {[
            { label: "Status", value: status },
            { label: "Forecast date", value: forecastDate },
            {
              label: "Forecast window",
              value: `${targetMonth ?? "—"} → ${endMonth ?? "—"}`,
            },
            { label: "Triage score", value: triageScore },
            { label: "Triage tier", value: triageTier },
            { label: "RC probability", value: rcProbability },
            { label: "RC direction", value: rcDirection },
            { label: "RC magnitude", value: rcMagnitude },
            { label: "RC score", value: rcScore },
            { label: "Web search", value: webSearchModels },
            { label: "HS triage", value: hsTriageDisplay },
            { label: "Research", value: getModelNamesForPhase(byPhase, "research_v2") },
            { label: "SPD", value: getModelNamesForPhase(byPhase, "spd_v2") },
            { label: "Scenario", value: getModelNamesForPhase(byPhase, "scenario_v2") },
          ].map((item) => (
            <div
              key={item.label}
              className="rounded border border-fred-secondary bg-fred-surface px-3 py-2"
            >
              <div className="text-[11px] font-semibold uppercase tracking-wide text-fred-muted">
                {item.label}
              </div>
              <div className="mt-1 text-sm text-fred-text">{item.value}</div>
            </div>
          ))}
        </div>
        {debugEnabled ? (
          <details className="mt-4 rounded border border-fred-secondary bg-fred-surface px-3 py-2 text-xs text-fred-text">
            <summary className="cursor-pointer text-fred-primary">
              Question debug
            </summary>
            <div className="mt-2 space-y-2">
              <div>
                <div className="text-[11px] uppercase tracking-wide text-fred-muted">
                  by_phase keys
                </div>
                <div className="text-sm text-fred-text">
                  {Object.keys(byPhase).length ? Object.keys(byPhase).join(", ") : "—"}
                </div>
              </div>
              <div>
                <div className="text-[11px] uppercase tracking-wide text-fred-muted">
                  llm_calls.debug
                </div>
                <pre className="mt-1 whitespace-pre-wrap break-words text-xs text-fred-text">
                  {JSON.stringify(llm.debug ?? null, null, 2)}
                </pre>
              </div>
            </div>
          </details>
        ) : null}
        <div className="mt-4 grid gap-4 md:grid-cols-2">
          <div className="rounded border border-fred-secondary bg-fred-surface px-3 py-3">
            <div className="text-xs font-semibold uppercase tracking-wide text-fred-muted">
              Resolutions
            </div>
            <div className="mt-2 text-sm text-fred-text">{resolutionSummary}</div>
          </div>
          <div className="rounded border border-fred-secondary bg-fred-surface px-3 py-3">
            <div className="text-xs font-semibold uppercase tracking-wide text-fred-muted">
              Brier
            </div>
            <div className="mt-2 space-y-2">
              {brierAverage !== null ? (
                <div className="text-sm text-fred-text">
                  Avg Brier (resolved months): {brierAverage.toFixed(3)}
                </div>
              ) : (
                <div className="text-sm text-fred-text">No Brier yet</div>
              )}
              <div className="grid grid-cols-3 gap-2 text-xs text-fred-text md:grid-cols-6">
                {[1, 2, 3, 4, 5, 6].map((horizon) => (
                  <div
                    key={`brier-${horizon}`}
                    className="rounded border border-fred-secondary bg-fred-surface px-2 py-1"
                  >
                    <div className="text-[11px] text-fred-muted">
                      Month {horizon}
                    </div>
                    <div>
                      {brierByHorizon.has(horizon)
                        ? brierByHorizon.get(horizon)?.toFixed(3)
                        : "—"}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>

      {((question.metric as string) ?? "").toUpperCase() === "EVENT_OCCURRENCE" ? (
        <BinaryPanel bundle={bundle} />
      ) : (
        <SpdPanel bundle={bundle} />
      )}

      <section className="space-y-4">
        <div className="rounded-lg border border-fred-secondary bg-fred-surface p-4 text-fred-text">
          <h2 className="text-lg font-semibold">Forecast Rationale</h2>
          <div className="mt-3 space-y-2">
            {forecastRationale ? (
              renderParagraphs(forecastRationale)
            ) : (
              <p className="text-sm text-fred-text">
                No forecast rationale available for this forecast.
              </p>
            )}
          </div>
        </div>

        <div className="rounded-lg border border-fred-secondary bg-fred-surface p-4 text-fred-text">
          <h2 className="text-lg font-semibold">Scenario</h2>
          {scenarioWriter.length > 1 ? (
            <div className="mt-3 flex flex-col gap-2 md:flex-row md:items-center">
              <label className="text-sm text-fred-text">Scenario:</label>
              <select
                className="rounded border border-fred-secondary bg-fred-surface px-3 py-2 text-sm text-fred-text"
                value={scenarioIndex}
                onChange={(event) => setScenarioIndex(Number(event.target.value))}
              >
                {scenarioWriter.map((row, index) => (
                  <option key={`scenario-${index}`} value={index}>
                    {(row.scenario_type as string | undefined) ??
                      `Scenario ${index + 1}`}
                  </option>
                ))}
              </select>
            </div>
          ) : null}
          <div className="mt-4">{renderScenarioBlocks(selectedScenarioText)}</div>
        </div>

        <div className="rounded-lg border border-fred-secondary bg-fred-surface p-4 text-fred-text">
          <h2 className="text-lg font-semibold">Horizon Scan Triage</h2>
          <div className="mt-3 space-y-3">
            {triageConfidenceNote ? (
              <div>
                <h3 className="text-sm font-semibold">Assessment</h3>
                <p className="mt-2 text-sm text-fred-text leading-relaxed">
                  {triageConfidenceNote}
                </p>
              </div>
            ) : null}
            {scenarioStubText ? (
              <div>
                <h3 className="text-sm font-semibold">Scenario stub</h3>
                <div className="mt-2">{renderScenarioBlocks(scenarioStubText)}</div>
              </div>
            ) : null}
            <div>
              <h3 className="text-sm font-semibold">Drivers</h3>
              {getTextList(triage.drivers_json).length ? (
                <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-fred-text">
                  {getTextList(triage.drivers_json).map((item, index) => (
                    <li key={`driver-${index}`}>{item}</li>
                  ))}
                </ul>
              ) : (
                <p className="mt-2 text-sm text-fred-text">No drivers captured.</p>
              )}
            </div>
            <div>
              <h3 className="text-sm font-semibold">Data quality</h3>
              {dataQualityEntries.length ? (
                <div className="mt-2 rounded border border-fred-secondary bg-fred-surface p-3 text-sm text-fred-text">
                  {dataQualityEntries.map((entry) => (
                    <p key={entry.label}>
                      <span className="text-fred-muted">{entry.label}:</span>{" "}
                      {entry.value}
                    </p>
                  ))}
                </div>
              ) : (
                <p className="mt-2 text-sm text-fred-text">No data quality notes.</p>
              )}
            </div>
          </div>
        </div>

        <div className="rounded-lg border border-fred-secondary bg-fred-surface p-4 text-fred-text">
          <h2 className="text-lg font-semibold">Regime Change</h2>
          <div className="mt-3 space-y-3">
            <div className="grid gap-3 sm:grid-cols-3 lg:grid-cols-6">
              {[
                { label: "Likelihood", value: rcProbability },
                { label: "Magnitude", value: rcMagnitude },
                { label: "Score", value: rcScore },
                { label: "Direction", value: rcDirection },
                { label: "Level", value: asString(triage.regime_change_level) ?? "—" },
                { label: "Window", value: asString(triage.regime_change_window) ?? "—" },
              ].map((item) => (
                <div
                  key={item.label}
                  className="rounded border border-fred-secondary bg-fred-surface px-3 py-2"
                >
                  <div className="text-[11px] font-semibold uppercase tracking-wide text-fred-muted">
                    {item.label}
                  </div>
                  <div className="mt-1 text-sm text-fred-text">{item.value}</div>
                </div>
              ))}
            </div>
            {rcRationaleBullets.length ? (
              <div>
                <h3 className="text-sm font-semibold">Rationale</h3>
                <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-fred-text">
                  {rcRationaleBullets.map((bullet, index) => (
                    <li key={`rc-rationale-${index}`}>{bullet}</li>
                  ))}
                </ul>
              </div>
            ) : null}
            {rcTriggerSignals.length ? (
              <div>
                <h3 className="text-sm font-semibold">Trigger signals</h3>
                <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-fred-text">
                  {rcTriggerSignals.map((signal, index) => (
                    <li key={`rc-trigger-${index}`}>{signal}</li>
                  ))}
                </ul>
              </div>
            ) : null}
            {rcConfidenceNote ? (
              <div>
                <h3 className="text-sm font-semibold">Confidence</h3>
                <p className="mt-2 text-sm text-fred-text leading-relaxed">
                  {rcConfidenceNote}
                </p>
              </div>
            ) : null}
            {!rcRationaleBullets.length && !rcTriggerSignals.length && !rcConfidenceNote ? (
              <p className="text-sm text-fred-text">No detailed regime change data available.</p>
            ) : null}
          </div>
        </div>
      </section>

      <PromptViewerSection groupedLlmCalls={groupedLlmCalls} />
    </div>
  );
};

export default QuestionDetailView;
