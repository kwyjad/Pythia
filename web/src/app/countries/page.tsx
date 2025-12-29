import { apiGet } from "../../lib/api";
import CountriesTable from "./CountriesTable";
import { readFile } from "node:fs/promises";
import path from "node:path";

type CountriesRow = {
  iso3: string;
  n_questions: number;
  n_forecasted: number;
  country_name?: string | null;
  last_triaged?: string | null; // YYYY-MM-DD
  last_forecasted?: string | null; // YYYY-MM-DD
};

type CountriesResponse = {
  rows: CountriesRow[];
};

type QuestionsResponse = {
  rows: Array<{
    iso3?: string;
    hs_run_created_at?: string | null;
  }>;
};

type ForecastsEnsembleResponse = {
  rows: Array<{
    iso3?: string;
    created_at?: string | null;
    timestamp?: string | null;
    status?: string | null;
  }>;
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

function updateMax(map: Map<string, number>, key: string, dateValue?: string | null) {
  const ts = dateValue ? Date.parse(dateValue) : NaN;
  if (!key || Number.isNaN(ts)) return;
  const prev = map.get(key);
  if (prev == null || ts > prev) map.set(key, ts);
}

const CountriesPage = async () => {
  let rows: CountriesRow[] = [];
  try {
    const response = await apiGet<CountriesResponse>("/countries");
    rows = response.rows;
  } catch (error) {
    console.warn("Failed to load countries:", error);
  }

  const nameMap = await loadCountryNameMap();

  const lastTriagedByIso3 = new Map<string, number>();
  try {
    const qResp = await apiGet<QuestionsResponse>("/questions", { latest_only: true });
    for (const q of qResp.rows ?? []) {
      const iso3 = (q.iso3 ?? "").toUpperCase();
      updateMax(lastTriagedByIso3, iso3, q.hs_run_created_at ?? null);
    }
  } catch (error) {
    console.warn("Failed to derive last triaged dates:", error);
  }

  const lastForecastedByIso3 = new Map<string, number>();
  try {
    const fResp = await apiGet<ForecastsEnsembleResponse>("/forecasts_ensemble", {
      latest_only: true,
    });
    for (const f of fResp.rows ?? []) {
      const iso3 = (f.iso3 ?? "").toUpperCase();
      const t = f.created_at ?? f.timestamp ?? null;
      if (typeof f.status === "string" && f.status.length > 0) {
        if (f.status !== "ok") continue;
      }
      updateMax(lastForecastedByIso3, iso3, t);
    }
  } catch (error) {
    console.warn("Failed to derive last forecasted dates:", error);
  }

  rows = rows.map((r) => {
    const iso3 = (r.iso3 ?? "").toUpperCase();
    const triagedTs = lastTriagedByIso3.get(iso3);
    const forecastedTs = lastForecastedByIso3.get(iso3);
    return {
      ...r,
      iso3,
      country_name: nameMap.get(iso3) ?? null,
      last_triaged: triagedTs ? new Date(triagedTs).toISOString().slice(0, 10) : null,
      last_forecasted: forecastedTs
        ? new Date(forecastedTs).toISOString().slice(0, 10)
        : null,
    };
  });

  return (
    <div className="space-y-6">
      <section>
        <h1 className="text-3xl font-semibold text-white">Countries</h1>
        <p className="text-sm text-slate-400">
          Browse available countries in the database.
        </p>
      </section>

      <div className="overflow-x-auto rounded-lg border border-slate-800">
        <CountriesTable rows={rows} />
      </div>
    </div>
  );
};

export default CountriesPage;
