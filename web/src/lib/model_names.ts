// Aggregate / track slugs that are not base models. Kept here so every surface
// (Costs, Performance) renders them the same way.
const AGGREGATE_LABELS: Record<string, string> = {
  sibyl: "Sibyl (deep research)",
  ensemble_bayesmc_v2: "Ensemble (BayesMC)",
  ensemble_mean_v2: "Ensemble (mean)",
  ensemble: "Ensemble",
  track2_flash: "Track 2 (flash)",
};

export function formatModelName(raw: string): string {
  const trimmed = raw.trim();
  if (!trimmed) return raw;
  const lowered = trimmed.toLowerCase();

  if (AGGREGATE_LABELS[lowered]) {
    return AGGREGATE_LABELS[lowered];
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

  // GPT models: gpt-5.4, gpt-5.4-mini, etc.
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
