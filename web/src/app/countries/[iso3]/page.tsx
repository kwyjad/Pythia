import { readFile } from "node:fs/promises";
import path from "node:path";

import Link from "next/link";

import { apiGet } from "../../../lib/api";
import type { QuestionsResponse } from "../../../lib/types";
import CountryQuestionsTable from "./CountryQuestionsTable";

type CountryPageProps = {
  params: { iso3: string };
};

type CountryQuestionRow = QuestionsResponse["rows"][number] & {
  first_forecast_month?: string | null;
  last_forecast_month?: string | null;
};

function parseCsvLine(line: string): string[] {
  const out: string[] = [];
  let cur = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === '"') {
      if (inQuotes && line[i + 1] === '"') {
        cur += '"';
        i++;
      } else {
        inQuotes = !inQuotes;
      }
      continue;
    }
    if (ch === "," && !inQuotes) {
      out.push(cur);
      cur = "";
      continue;
    }
    cur += ch;
  }
  out.push(cur);
  return out.map((v) => v.trim());
}

async function loadCountryNameMap(): Promise<Map<string, string>> {
  const candidates = [
    path.join(process.cwd(), "..", "resolver", "data", "countries.csv"),
    path.join(process.cwd(), "resolver", "data", "countries.csv"),
  ];

  let csvText: string | null = null;
  for (const p of candidates) {
    try {
      csvText = await readFile(p, "utf-8");
      break;
    } catch {
      // keep trying
    }
  }

  const map = new Map<string, string>();
  if (!csvText) return map;

  const lines = csvText.split(/\r?\n/).filter((l) => l.trim().length > 0);
  if (lines.length < 2) return map;

  const header = parseCsvLine(lines[0]).map((h) => h.toLowerCase());
  const isoIdx = header.indexOf("iso3");
  const nameIdx = header.indexOf("country_name");
  if (isoIdx === -1 || nameIdx === -1) return map;

  for (let i = 1; i < lines.length; i++) {
    const cols = parseCsvLine(lines[i]);
    const iso3 = (cols[isoIdx] ?? "").trim().toUpperCase();
    const name = (cols[nameIdx] ?? "").trim();
    if (iso3) map.set(iso3, name);
  }
  return map;
}

function addMonthsYYYYMM(value: string, months: number): string | null {
  const parts = value.split("-");
  if (parts.length < 2) return null;
  const year = Number(parts[0]);
  const month = Number(parts[1]);
  if (!Number.isFinite(year) || !Number.isFinite(month)) return null;
  if (month < 1 || month > 12) return null;
  const total = year * 12 + (month - 1) + months;
  if (!Number.isFinite(total)) return null;
  const outYear = Math.floor(total / 12);
  const outMonth = (total % 12) + 1;
  return `${String(outYear).padStart(4, "0")}-${String(outMonth).padStart(2, "0")}`;
}

const CountryPage = async ({ params }: CountryPageProps) => {
  let questions: CountryQuestionRow[] = [];
  let loadError: string | null = null;
  const iso3 = params.iso3.toUpperCase();
  try {
    const response = await apiGet<QuestionsResponse>("/questions", {
      iso3,
      latest_only: true,
    });
    questions = response.rows.map((row) => {
      const firstForecastMonth = row.target_month ?? null;
      const horizonMonths = row.forecast_horizon_max ?? 6;
      const lastForecastMonth =
        firstForecastMonth && horizonMonths > 0
          ? addMonthsYYYYMM(firstForecastMonth, horizonMonths - 1)
          : null;
      return {
        ...row,
        first_forecast_month: firstForecastMonth,
        last_forecast_month: lastForecastMonth,
      };
    });
  } catch (error) {
    loadError = "Unable to load questions right now.";
    console.warn("Failed to load questions:", error);
  }

  const nameMap = await loadCountryNameMap();
  const countryName = nameMap.get(iso3) ?? iso3;

  return (
    <div className="space-y-6">
      <section>
        <Link
          className="text-sm text-fred-primary underline underline-offset-2 hover:text-fred-secondary"
          href="/countries"
        >
          ← Back to Countries
        </Link>
        <h1 className="text-3xl font-semibold text-fred-primary">{countryName}</h1>
        <p className="text-sm text-fred-text">
          Latest questions for {countryName} ({iso3})
        </p>
      </section>

      {loadError || questions.length === 0 ? (
        <div className="rounded-lg border border-slate-800 bg-slate-900/40 px-4 py-6 text-center text-slate-300">
          {loadError ?? `No questions available for ${iso3} in this DB snapshot.`}
        </div>
      ) : (
        <div className="overflow-x-auto rounded-lg border border-slate-800">
          <CountryQuestionsTable rows={questions} />
        </div>
      )}
    </div>
  );
};

export default CountryPage;
