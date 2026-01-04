import { readFile } from "node:fs/promises";
import path from "node:path";

import { apiGet } from "../../lib/api";
import ResolverClient, {
  ConnectorStatusRow,
  ResolverCountryOption,
} from "./ResolverClient";

export const dynamic = "force-dynamic";
export const revalidate = 0;

type CountriesResponse = {
  rows: Array<{
    iso3: string;
  }>;
};

type ConnectorStatusResponse = {
  rows: ConnectorStatusRow[];
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

const safeGet = async <T,>(path: string, fallback: T): Promise<T> => {
  try {
    return await apiGet<T>(path);
  } catch (error) {
    console.warn(`Failed to load ${path}:`, error);
    return fallback;
  }
};

const ResolverPage = async () => {
  if (process.env.NODE_ENV !== "production") {
    console.log("[page] dynamic=force-dynamic", { route: "/resolver" });
  }
  const [countriesResponse, connectorStatusResponse] = await Promise.all([
    safeGet<CountriesResponse>("/countries", { rows: [] }),
    safeGet<ConnectorStatusResponse>("/resolver/connector_status", { rows: [] }),
  ]);

  const nameMap = await loadCountryNameMap();
  const countries: ResolverCountryOption[] = countriesResponse.rows.map((row) => {
    const iso3 = (row.iso3 ?? "").toUpperCase();
    return {
      iso3,
      country_name: nameMap.get(iso3) ?? null,
    };
  });

  return (
    <ResolverClient
      countries={countries}
      connectorStatus={connectorStatusResponse.rows}
    />
  );
};

export default ResolverPage;
