// Pythia / Copyright (c) 2025 Kevin Wyjad
// Conflict displacement (ACE/PA) is forecast only for countries IDMC reports
// regularly (pythia/tools/ace_pa_eligibility.py). Where a country has a
// conflict-deaths question and no displacement question, say so: a missing
// question must never read as no displacement risk.

export const CONFLICT_DISPLACEMENT_NOT_FORECAST =
  "Conflict displacement not forecast (IDMC does not report this country regularly)";

type QuestionLike = {
  iso3?: string | null;
  hazard_code?: string | null;
  metric?: string | null;
  target_month?: string | null;
};

/** ISO3s from a risk-index response's not-forecast list. */
export function notForecastSet(
  list: { iso3: string }[] | null | undefined
): Set<string> {
  return new Set((list ?? []).map((c) => c.iso3.toUpperCase()));
}

/** A map tooltip value with the note appended where it applies. */
export function withDisplacementNote(
  label: string,
  iso3: string,
  notForecast: Set<string> | null | undefined
): string {
  if (!notForecast || !notForecast.has(iso3.toUpperCase())) return label;
  return `${label} · ${CONFLICT_DISPLACEMENT_NOT_FORECAST}`;
}

/** Windows (target months) for which a country's questions include conflict
 * deaths but not conflict displacement. */
export function displacementNotForecastWindows(rows: QuestionLike[]): string[] {
  const fat = new Set<string>();
  const pa = new Set<string>();
  for (const r of rows) {
    if ((r.hazard_code ?? "").toUpperCase() !== "ACE") continue;
    const tm = r.target_month ?? "";
    const metric = (r.metric ?? "").toUpperCase();
    if (metric === "FATALITIES") fat.add(tm);
    if (metric === "PA") pa.add(tm);
  }
  return Array.from(fat).filter((tm) => !pa.has(tm)).sort();
}
