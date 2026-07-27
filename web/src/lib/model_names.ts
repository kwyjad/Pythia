// Aggregate / track slugs that are not base models. Kept here so every surface
// (Costs, Performance) renders them the same way.
const AGGREGATE_LABELS: Record<string, string> = {
  sibyl: "Sibyl (deep research)",
  ensemble_bayesmc_v2: "Ensemble (BayesMC)",
  ensemble_mean_v2: "Ensemble (mean)",
  ensemble: "Ensemble",
  track2_flash: "Track 2 (flash)",
};

// Not models: sentinels written by the HS grounding log sites when no backend produced
// a model id, so a grounding failure reads as a failure instead of a mystery model.
// Keep in sync with the GROUNDING_*_MODEL_ID constants in horizon_scanner/llm_logging.py.
const GROUNDING_SENTINEL_LABELS: Record<string, string> = {
  "grounding-failed": "Grounding failed (no backend)",
  "grounding-breaker-tripped": "Grounding unavailable (breaker)",
  "grounding-unavailable": "Grounding unavailable (no API key)",
};

export function formatModelName(raw: string): string {
  const trimmed = raw.trim();
  if (!trimmed) return raw;
  const lowered = trimmed.toLowerCase();

  if (AGGREGATE_LABELS[lowered]) {
    return AGGREGATE_LABELS[lowered];
  }

  if (GROUNDING_SENTINEL_LABELS[lowered]) {
    return GROUNDING_SENTINEL_LABELS[lowered];
  }

  if (lowered.startsWith("gemini-")) {
    const parts = lowered.replace(/^gemini-/, "").split("-");
    const version = parts.shift();
    if (version) {
      const tokens = parts.filter((token) => token !== "preview");
      const rest = tokens
        .map((token) => token.charAt(0).toUpperCase() + token.slice(1))
        .join(" ");
      return `Gemini ${version}${rest ? ` ${rest}` : ""}`;
    }
  }

  // GPT models: gpt-5.6-sol, gpt-5.6-luna, gpt-5.4-mini, etc.
  const gptMatch = lowered.match(/^gpt-([\d.]+)(?:-(.+))?$/);
  if (gptMatch) {
    const version = gptMatch[1];
    const suffix = gptMatch[2];
    if (suffix) {
      const suffixLabel = suffix.charAt(0).toUpperCase() + suffix.slice(1);
      return `GPT-${version} ${suffixLabel}`;
    }
    return `GPT-${version}`;
  }

  const claudeMatch = lowered.match(/^claude-([a-z]+)-(\d+)(?:-(\d+))?/);
  if (claudeMatch) {
    const tier = claudeMatch[1];
    const major = claudeMatch[2];
    const minor = claudeMatch[3];
    const version = minor ? `${major}.${minor}` : major;
    const tierLabel = tier.charAt(0).toUpperCase() + tier.slice(1);
    return `Claude ${tierLabel} ${version}`;
  }

  return raw;
}

export function formatModelList(values: string[]): string {
  const deduped = Array.from(new Set(values.map((value) => value.trim()).filter(Boolean)));
  if (!deduped.length) return "—";
  return deduped.map(formatModelName).sort((a, b) => a.localeCompare(b)).join(", ");
}
