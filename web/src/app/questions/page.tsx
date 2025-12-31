import { apiGet } from "../../lib/api";
import QuestionsTable from "./QuestionsTable";
import { readFile } from "node:fs/promises";
import path from "node:path";

type QuestionRow = {
  question_id: string;
  hs_run_id?: string | null;
  iso3: string;
  hazard_code: string;
  metric: string;
  target_month: string;
  forecast_date?: string | null;
  forecast_horizon_max?: number | null;
  eiv_total?: number | null;
  status?: string;
  wording?: string;
  country_name?: string | null;
  first_forecast_month?: string | null;
  last_forecast_month?: string | null;
  triage_score?: number | null;
  triage_tier?: string | null;
  triage_need_full_spd?: boolean | null;
};

type QuestionsResponse = {
  rows: QuestionRow[];
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

const QuestionsPage = async () => {
  let rows: QuestionRow[] = [];
  try {
    const response = await apiGet<QuestionsResponse>("/questions", {
      latest_only: true
    });
    rows = response.rows;
  } catch (error) {
    console.warn("Failed to load questions:", error);
  }

  const nameMap = await loadCountryNameMap();

  rows = rows.map((row) => {
    const iso3 = (row.iso3 ?? "").toUpperCase();
    const firstForecastMonth = row.target_month ?? null;
    const monthsTotal = row.forecast_horizon_max ?? 6;
    const lastForecastMonth =
      firstForecastMonth && monthsTotal > 0
        ? addMonthsYYYYMM(firstForecastMonth, monthsTotal - 1)
        : null;
    return {
      ...row,
      iso3,
      country_name: nameMap.get(iso3) ?? null,
      first_forecast_month: firstForecastMonth,
      last_forecast_month: lastForecastMonth,
    };
  });

  return (
    <div className="space-y-6">
      <section>
        <h1 className="text-3xl font-semibold">Forecasts</h1>
        <p className="text-sm text-fred-text">
          Browse the latest forecasts by concept.
        </p>
      </section>

      <div className="overflow-x-auto rounded-lg border border-fred-secondary">
        <QuestionsTable rows={rows} />
      </div>
    </div>
  );
};

export default QuestionsPage;
