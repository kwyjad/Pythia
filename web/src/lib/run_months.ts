export type RunMonth = {
  year_month: string;
  label?: string;
  is_latest?: boolean;
};

export const parseYearMonth = (value: string | null | undefined) => {
  if (!value) return null;
  const match = value.trim().match(/^(\d{4})-(\d{2})$/);
  if (!match) return null;
  const year = Number(match[1]);
  const month = Number(match[2]);
  if (!Number.isFinite(year) || !Number.isFinite(month)) return null;
  if (month < 1 || month > 12) return null;
  return { year, month };
};

export const formatYearMonthLabel = (value: string | null | undefined) => {
  const parsed = parseYearMonth(value ?? "");
  if (!parsed) return value ?? "";
  const date = new Date(parsed.year, parsed.month - 1, 1);
  return date.toLocaleString(undefined, { month: "long", year: "numeric" });
};

export const sortMonthsDesc = (months: RunMonth[]) => {
  return [...months].sort((a, b) => {
    const aParsed = parseYearMonth(a.year_month);
    const bParsed = parseYearMonth(b.year_month);
    if (!aParsed || !bParsed) return 0;
    if (aParsed.year !== bParsed.year) return bParsed.year - aParsed.year;
    return bParsed.month - aParsed.month;
  });
};
