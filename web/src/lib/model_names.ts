export function formatModelName(raw: string): string {
  const trimmed = raw.trim();
  if (!trimmed) return raw;
  const lowered = trimmed.toLowerCase();

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

  // GPT models: gpt-5.2, gpt-5-mini, gpt-5-nano, etc.
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

  const grokMatch = lowered.match(/^grok-(\d+)/);
  if (grokMatch) {
    return `Grok ${grokMatch[1]}`;
  }

  // Kimi models: kimi-k2.5 → Kimi K2.5
  const kimiMatch = lowered.match(/^kimi-(.+)$/);
  if (kimiMatch) {
    return `Kimi ${kimiMatch[1].toUpperCase()}`;
  }

  // DeepSeek models: deepseek-reasoner → DeepSeek Reasoner
  const deepseekMatch = lowered.match(/^deepseek-(.+)$/);
  if (deepseekMatch) {
    const suffix = deepseekMatch[1]
      .split("-")
      .map((token) => token.charAt(0).toUpperCase() + token.slice(1))
      .join(" ");
    return `DeepSeek ${suffix}`;
  }

  return raw;
}

export function formatModelList(values: string[]): string {
  const deduped = Array.from(new Set(values.map((value) => value.trim()).filter(Boolean)));
  if (!deduped.length) return "—";
  return deduped.map(formatModelName).sort((a, b) => a.localeCompare(b)).join(", ");
}
