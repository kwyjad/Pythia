const TARGET_KEYS = [
  "rationale",
  "short_rationale",
  "rationale_text",
  "analysis_summary",
  "summary",
  "explanation",
  "human_explanation",
];

const isUsefulRationale = (value: string): boolean => {
  const trimmed = value.trim();
  if (trimmed.length < 40) return false;
  if (trimmed.toLowerCase().startsWith("ensemble_meta:")) return false;
  return true;
};

export const extractForecastRationale = (bundle: any): string | null => {
  if (!bundle || typeof bundle !== "object") return null;
  const forecast = (bundle as { forecast?: any }).forecast;
  if (!forecast || typeof forecast !== "object") return null;

  const directCandidates = [
    forecast?.forecast_json?.rationale,
    forecast?.forecast_json?.short_rationale,
    forecast?.forecast_output?.rationale,
    forecast?.raw_forecast?.rationale,
  ];

  for (const candidate of directCandidates) {
    if (typeof candidate === "string" && isUsefulRationale(candidate)) {
      return candidate.trim();
    }
  }

  // Check human_explanation directly in raw_spd and ensemble_spd rows
  // (the DFS below can hit the seen-limit before reaching these large arrays)
  for (const arr of [forecast?.raw_spd, forecast?.ensemble_spd]) {
    if (Array.isArray(arr)) {
      for (const row of arr) {
        const he = row?.human_explanation;
        if (typeof he === "string" && isUsefulRationale(he)) {
          return he.trim();
        }
      }
    }
  }

  const stack: Array<{ node: unknown; depth: number }> = [
    { node: forecast, depth: 0 },
  ];
  const visited = new Set<unknown>();
  let seen = 0;

  while (stack.length) {
    const current = stack.pop();
    if (!current) break;
    const { node, depth } = current;
    if (node && typeof node === "object") {
      if (visited.has(node)) continue;
      visited.add(node);
    }

    if (depth > 6) continue;
    if (seen > 2000) break;

    if (node && typeof node === "object") {
      const entries = Object.entries(node as Record<string, unknown>);
      for (const [key, value] of entries) {
        seen += 1;
        if (TARGET_KEYS.includes(key)) {
          if (typeof value === "string" && isUsefulRationale(value)) {
            return value.trim();
          }
        }
        if (value && typeof value === "object") {
          stack.push({ node: value, depth: depth + 1 });
        }
      }
    }
  }

  return null;
};
