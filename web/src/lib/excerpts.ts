export const pickResearchJson = (bundle: any): any | null => {
  if (!bundle || typeof bundle !== "object") return null;
  const forecast = (bundle as { forecast?: any }).forecast;
  const candidate =
    forecast?.research?.research_json ??
    forecast?.research_json ??
    forecast?.research?.[0]?.research_json ??
    forecast?.research?.row?.research_json ??
    null;
  return candidate ?? null;
};

export const asArray = <T>(value: any): T[] => {
  if (Array.isArray(value)) return value as T[];
  return [];
};

export const asString = (value: any): string | null => {
  if (typeof value === "string") {
    const trimmed = value.trim();
    return trimmed.length ? trimmed : null;
  }
  if (typeof value === "number" && Number.isFinite(value)) {
    return String(value);
  }
  return null;
};
