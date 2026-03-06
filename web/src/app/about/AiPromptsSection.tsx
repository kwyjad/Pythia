"use client";

import React, { useState } from "react";

import type { VersionEntry } from "../../lib/prompt_extractor";

const GITHUB_BASE = "https://github.com/kwyjad/Pythia/blob/main";

const linkClass = "text-fred-primary underline hover:text-fred-secondary";

const panelClass =
  "group rounded-md border border-fred-secondary bg-fred-surface";

const summaryClass =
  "cursor-pointer px-4 py-3 font-semibold text-fred-secondary group-open:bg-fred-bg";

const bodyClass = "px-4 pb-4 pt-2 text-fred-text";

const preClass =
  "mt-3 max-h-[32rem] whitespace-pre-wrap rounded-md bg-fred-bg p-4 text-xs text-fred-text border border-fred-secondary/40 overflow-auto";

interface SourceLink {
  href: string;
  label: string;
}

interface PanelConfig {
  id: string;
  title: string;
  description: React.ReactNode;
  promptKeys: string[];
  /** Optional labels for each promptKey (keyed by promptKey). */
  promptLabels?: Record<string, string>;
  sourceLinks: SourceLink[];
  staticFallback?: string;
  notes?: React.ReactNode;
}

const HAZARD_LABELS: Record<string, string> = {
  ace: "ACE — Armed Conflict Events",
  dr: "DR — Drought",
  fl: "FL — Flood",
  hw: "HW — Heatwave",
  tc: "TC — Tropical Cyclone",
};

function hazardPromptLabels(prefix: string): Record<string, string> {
  const labels: Record<string, string> = {};
  for (const [code, label] of Object.entries(HAZARD_LABELS)) {
    labels[`${prefix}_${code}`] = label;
  }
  return labels;
}

const PANELS: PanelConfig[] = [
  {
    id: "web_search",
    title: "Web search / evidence pack",
    description: (
      <p>
        Used by the Gemini grounding retriever to produce a structured evidence
        pack (structural context + recent signals). It is a JSON-only response
        with no URLs in the text.
      </p>
    ),
    promptKeys: ["web_search"],
    sourceLinks: [
      {
        href: `${GITHUB_BASE}/pythia/web_research/backends/gemini_grounding.py`,
        label: "Source",
      },
      {
        href: `${GITHUB_BASE}/horizon_scanner/horizon_scanner.py`,
        label: "Query builder",
      },
    ],
  },
  {
    id: "rc_grounding",
    title: "RC grounding prompts (per-hazard)",
    description: (
      <p>
        Per-hazard Google Grounding queries for Regime Change evidence gathering.
        Each hazard gets a dedicated grounding call that surfaces the right
        signals for its RC assessment. RC grounding asks: &ldquo;What is{" "}
        <strong>changing</strong>?&rdquo;
      </p>
    ),
    promptKeys: [
      "rc_grounding_ace",
      "rc_grounding_dr",
      "rc_grounding_fl",
      "rc_grounding_hw",
      "rc_grounding_tc",
    ],
    promptLabels: hazardPromptLabels("rc_grounding"),
    sourceLinks: [
      {
        href: `${GITHUB_BASE}/horizon_scanner/rc_grounding_prompts.py`,
        label: "Source",
      },
    ],
  },
  {
    id: "rc",
    title: "Regime Change assessment (per-hazard)",
    description: (
      <p>
        Per-hazard RC prompt builders. RC is the <strong>first step</strong> in
        the HS pipeline, answering: &ldquo;Is this hazard breaking from its
        historical base rate in this country?&rdquo; Each hazard gets tailored
        calibration anchors, examples, and guidance. The shared preamble is shown
        first, followed by each hazard variant.
      </p>
    ),
    promptKeys: [
      "rc_preamble",
      "rc_ace",
      "rc_dr",
      "rc_fl",
      "rc_hw",
      "rc_tc",
    ],
    promptLabels: {
      rc_preamble: "Shared calibration preamble",
      ...hazardPromptLabels("rc"),
    },
    sourceLinks: [
      {
        href: `${GITHUB_BASE}/horizon_scanner/rc_prompts.py`,
        label: "Source",
      },
    ],
  },
  {
    id: "triage_grounding",
    title: "Triage grounding prompts (per-hazard)",
    description: (
      <p>
        Per-hazard Google Grounding queries for triage evidence gathering.
        Separate from RC grounding. Triage grounding asks: &ldquo;What is the{" "}
        <strong>current situation</strong>?&rdquo; It builds the operational
        picture: severity, response coverage, and seasonal outlook.
      </p>
    ),
    promptKeys: [
      "triage_grounding_ace",
      "triage_grounding_dr",
      "triage_grounding_fl",
      "triage_grounding_hw",
      "triage_grounding_tc",
    ],
    promptLabels: hazardPromptLabels("triage_grounding"),
    sourceLinks: [
      {
        href: `${GITHUB_BASE}/horizon_scanner/hs_triage_grounding_prompts.py`,
        label: "Source",
      },
    ],
  },
  {
    id: "hs_triage",
    title: "Horizon scan triage (HS v2)",
    description: (
      <p>
        The horizon scan prompt requires a single JSON object with a hazard
        entry for every code (ACE, DI, DR, FL, HW, TC). The model must output a
        numeric triage score and optional tier per hazard. It includes detailed
        regime change calibration guidance.
      </p>
    ),
    promptKeys: ["hs_triage"],
    sourceLinks: [
      { href: `${GITHUB_BASE}/horizon_scanner/prompts.py`, label: "Source" },
    ],
  },
  {
    id: "spd",
    title: "Forecast (SPD)",
    description: (
      <p>
        The v2 forecaster prompt produces a six-month SPD (Subjective
        Probability Distribution) over five impact buckets using a structured
        7-step Bayesian updating method. It injects calibration guidance,
        regime change signals, hazard tail pack data, prediction market
        signals, seasonal outlook, and structured data connectors
        (ReliefWeb, ACLED political events, IPC, ACAPS) at runtime.
      </p>
    ),
    promptKeys: ["spd_template", "spd_buckets"],
    promptLabels: {
      spd_template: "SPD v2 prompt (build_spd_prompt_v2)",
      spd_buckets: "Bucket definitions (PA + fatalities)",
    },
    sourceLinks: [
      {
        href: `${GITHUB_BASE}/forecaster/prompts.py`,
        label: "Source: forecaster/prompts.py",
      },
      {
        href: `${GITHUB_BASE}/forecaster/hazard_prompts.py`,
        label: "Hazard reasoning blocks",
      },
    ],
    notes: (
      <div className="mt-3 text-sm">
        <p className="font-medium text-fred-secondary">
          Runtime context injected per question
        </p>
        <ul className="mt-2 list-disc space-y-1 pl-5">
          <li>
            Calibration guidance auto-loaded from scoring history (per-hazard
            when available).
          </li>
          <li>
            Bucket set depends on metric: PA uses PA buckets; fatalities uses
            fatalities buckets.
          </li>
          <li>
            Regime change guidance, tail pack signals, and self-search escape
            hatch injected based on HS triage output.
          </li>
        </ul>
      </div>
    ),
  },
  {
    id: "scenario",
    title: "Scenario generation (priority-only)",
    description: (
      <p>
        Scenario writing only runs when triage tier is{" "}
        <strong>priority</strong> and an ensemble SPD exists. The writer returns
        structured JSON with context, needs by sector, and operational impacts.
      </p>
    ),
    promptKeys: ["scenario"],
    sourceLinks: [
      {
        href: `${GITHUB_BASE}/forecaster/prompts.py`,
        label: "Scenario prompt",
      },
      {
        href: `${GITHUB_BASE}/forecaster/scenario_writer.py`,
        label: "Priority-only rule",
      },
    ],
  },
  {
    id: "question_construction",
    title: "How forecast questions are constructed",
    description: (
      <p>
        Horizon Scanner outputs are converted into question rows during HS
        upsert. The pipeline computes a target month and forecast window, then
        formats question text using hazard templates.
      </p>
    ),
    promptKeys: [],
    sourceLinks: [
      {
        href: `${GITHUB_BASE}/horizon_scanner/db_writer.py`,
        label: "Source",
      },
    ],
    staticFallback: `- upsert_hs_payload computes target_month + [opening_date, closing_date]
- _build_questions_for_scenario formats wording from templates:
  * Conflict fatalities (ACLED), displacement (IDMC/DTM)
  * Natural hazards (IFRC Montandon "people affected")
- Persisted fields include:
  question_id, iso3, hazard_code, metric, target_month,
  window_start_date, window_end_date, wording, status`,
  },
  {
    id: "self_search",
    title: "Self-search escape hatch",
    description: (
      <p>
        If the model needs more evidence, it can respond with a single line
        containing <code>NEED_WEB_EVIDENCE:</code> and a query. The system then
        performs retrieval and retries the prompt with appended evidence.
      </p>
    ),
    promptKeys: [],
    sourceLinks: [
      {
        href: `${GITHUB_BASE}/forecaster/prompts.py`,
        label: "Escape hatch in SPD prompt",
      },
      {
        href: `${GITHUB_BASE}/forecaster/self_search.py`,
        label: "Self-search retry",
      },
    ],
    staticFallback: `NEED_WEB_EVIDENCE: {country} {hazard} {metric} outlook — include recent signals and structural drivers.`,
  },
];

const CURRENT_KEY = "__current__";

interface AiPromptsSectionProps {
  currentPrompts: Record<string, string | null>;
  versions: VersionEntry[];
  versionedPrompts: Record<string, Record<string, string | null>>;
}

export default function AiPromptsSection({
  currentPrompts,
  versions,
  versionedPrompts,
}: AiPromptsSectionProps) {
  const [selectedVersion, setSelectedVersion] = useState(CURRENT_KEY);

  const activePrompts =
    selectedVersion === CURRENT_KEY
      ? currentPrompts
      : (versionedPrompts[selectedVersion] ?? currentPrompts);

  const hasVersions = versions.length > 0;

  return (
    <section className="mt-10 space-y-4">
      <div className="flex flex-wrap items-center gap-4">
        <h2 className="text-2xl font-semibold">AI Prompts</h2>
        {hasVersions && (
          <div className="flex items-center gap-2">
            <label
              htmlFor="prompt-version"
              className="text-sm font-medium text-fred-secondary"
            >
              Version:
            </label>
            <select
              id="prompt-version"
              value={selectedVersion}
              onChange={(e) => setSelectedVersion(e.target.value)}
              className="rounded-md border border-fred-secondary bg-fred-surface px-3 py-1.5 text-sm text-fred-text"
            >
              <option value={CURRENT_KEY}>Current (live)</option>
              {versions.map((v) => (
                <option key={v.date} value={v.date}>
                  {v.date} — {v.label}
                </option>
              ))}
            </select>
          </div>
        )}
      </div>
      <p className="text-sm text-fred-text">
        {selectedVersion === CURRENT_KEY
          ? "These excerpts are extracted directly from the Python source files at build time."
          : `Showing archived prompts from ${selectedVersion}.`}{" "}
        Each panel links to the source file in GitHub so you can inspect the
        exact code.
      </p>

      {PANELS.map((panel) => (
        <details key={panel.id} className={panelClass}>
          <summary className={summaryClass}>{panel.title}</summary>
          <div className={bodyClass}>
            {panel.description}

            {panel.promptKeys.length > 0
              ? panel.promptKeys.map((key) => {
                  const content = activePrompts[key];
                  const label = panel.promptLabels?.[key];
                  if (!content) {
                    return (
                      <p
                        key={key}
                        className="mt-3 text-xs italic text-fred-text/60"
                      >
                        {label ? `${label}: ` : ""}Prompt excerpt unavailable
                        — see source link below.
                      </p>
                    );
                  }
                  return (
                    <React.Fragment key={key}>
                      {label && (
                        <p className="mt-4 mb-1 text-sm font-medium text-fred-secondary">
                          {label}
                        </p>
                      )}
                      <pre className={preClass}>{content}</pre>
                    </React.Fragment>
                  );
                })
              : panel.staticFallback && (
                  <pre className={preClass}>{panel.staticFallback}</pre>
                )}

            {panel.notes}

            <div className="mt-2 flex flex-wrap gap-3">
              {panel.sourceLinks.map((link) => (
                <a
                  key={link.href}
                  className={linkClass}
                  href={link.href}
                  rel="noreferrer"
                  target="_blank"
                >
                  {link.label}
                </a>
              ))}
            </div>
          </div>
        </details>
      ))}
    </section>
  );
}
