import { readFile } from "node:fs/promises";
import path from "node:path";

/**
 * Read a source file from the monorepo, trying dual-path candidates
 * (web/ is one level below the repo root).
 */
async function readSourceFile(relativePath: string): Promise<string | null> {
  const candidates = [
    path.join(process.cwd(), "..", relativePath),
    path.join(process.cwd(), relativePath),
  ];
  for (const p of candidates) {
    try {
      return await readFile(p, "utf-8");
    } catch {
      // keep trying
    }
  }
  return null;
}

/**
 * Extract a named triple-quoted string constant from Python source.
 * Handles patterns like:
 *   NAME = """..."""
 *   NAME = _CAL_PREFIX + """..."""
 */
function extractNamedConstant(source: string, name: string): string | null {
  // Match NAME = (optional prefix +) """content"""
  const re = new RegExp(
    `^${name}\\s*=\\s*(?:[^"]*?\\+\\s*)?"""([\\s\\S]*?)"""`,
    "m",
  );
  const match = source.match(re);
  return match ? match[1] : null;
}

/**
 * Extract content between PROMPT_EXCERPT markers.
 */
function extractBetweenMarkers(
  source: string,
  startId: string,
  endId: string,
): string | null {
  const startMarker = `# --- PROMPT_EXCERPT: ${startId} ---`;
  const endMarker = `# --- PROMPT_EXCERPT: ${endId} ---`;
  const startIdx = source.indexOf(startMarker);
  const endIdx = source.indexOf(endMarker);
  if (startIdx === -1 || endIdx === -1 || endIdx <= startIdx) return null;
  return source.slice(startIdx + startMarker.length, endIdx);
}

/**
 * Clean up extracted Python prompt text for human-readable display.
 *
 * - Strips Python string syntax (return, f""", triple quotes, parens)
 * - Replaces json.dumps() calls with [data]
 * - Replaces f-string {variable_name} with [variable name]
 * - Unescapes doubled braces and \n literals
 * - Collapses excess blank lines
 */
function sanitizeForDisplay(text: string): string {
  let s = text;

  // Strip leading Python syntax:
  //   'return f"""...'  or  'return """...'
  //   'return (\n  ...'  or  'variable = (\n  ...'
  s = s.replace(/^\s*return\s+f?"""/, "");
  s = s.replace(/^\s*return\s*\(\s*\n?/, "");
  s = s.replace(/^\s*\w+\s*=\s*\(\s*\n?/, "");

  // Strip trailing triple-quote closure and parens
  s = s.replace(/"""\s*$/, "");
  s = s.replace(/\)\s*$/, "");

  // Join Python string concatenation:
  // end-of-string + start-of-next-string on next line
  // e.g.: '...\n"\n        "...'  or  '...\n"\n        f"...'
  s = s.replace(/"\s*\n\s*f?"/g, "");

  // Replace json.dumps / _json_dumps_for_prompt calls with [data]
  s = s.replace(/\{json\.dumps\([^)]+\)\}/g, "[data]");
  s = s.replace(/\{_json_dumps_for_prompt\([^)]+\)\}/g, "[data]");

  // Replace remaining f-string variables {var_name} with [var name]
  // But skip JSON-like patterns {{ and }}
  s = s.replace(/\{([a-z_][a-z0-9_]*)\}/gi, (_match, varName: string) => {
    return `[${varName.replace(/_/g, " ")}]`;
  });

  // Unescape doubled braces (Python f-string escaping for literal braces)
  s = s.replace(/\{\{/g, "{");
  s = s.replace(/\}\}/g, "}");

  // Unescape literal \n sequences (common in single-line f-strings)
  s = s.replace(/\\n/g, "\n");

  // Unescape escaped quotes
  s = s.replace(/\\"/g, '"');

  // Remove stray leading/trailing quotes from concatenation artifacts
  s = s.replace(/^"\s*/gm, "");
  s = s.replace(/\s*"$/gm, "");

  // Collapse 3+ consecutive blank lines to 2
  s = s.replace(/\n{4,}/g, "\n\n\n");

  return s.trim();
}

/**
 * Core extraction logic shared between live and versioned prompts.
 */
function extractPromptsFromSources(
  forecasterSrc: string | null,
  hsSrc: string | null,
  geminiSrc: string | null,
): Record<string, string | null> {
  const result: Record<string, string | null> = {};

  // --- Web search (marker extraction from gemini_grounding.py) ---
  if (geminiSrc) {
    const raw = extractBetweenMarkers(
      geminiSrc,
      "web_search_start",
      "web_search_end",
    );
    result.web_search = raw ? sanitizeForDisplay(raw) : null;
  } else {
    result.web_search = null;
  }

  // --- HS triage (marker extraction from horizon_scanner/prompts.py) ---
  if (hsSrc) {
    const raw = extractBetweenMarkers(
      hsSrc,
      "hs_triage_start",
      "hs_triage_end",
    );
    result.hs_triage = raw ? sanitizeForDisplay(raw) : null;
  } else {
    result.hs_triage = null;
  }

  // --- Researcher (named constant from forecaster/prompts.py) ---
  if (forecasterSrc) {
    const raw = extractNamedConstant(forecasterSrc, "RESEARCHER_PROMPT");
    result.researcher = raw ? sanitizeForDisplay(raw) : null;
  } else {
    result.researcher = null;
  }

  // --- Research v2 output schema ---
  if (forecasterSrc) {
    const raw = extractNamedConstant(
      forecasterSrc,
      "RESEARCH_V2_REQUIRED_OUTPUT_SCHEMA",
    );
    result.research_v2_schema = raw ? raw.trim() : null;
  } else {
    result.research_v2_schema = null;
  }

  // --- SPD template (named constant) ---
  if (forecasterSrc) {
    const raw = extractNamedConstant(forecasterSrc, "SPD_PROMPT_TEMPLATE");
    result.spd_template = raw ? sanitizeForDisplay(raw) : null;
  } else {
    result.spd_template = null;
  }

  // --- SPD bucket texts (named constants) ---
  if (forecasterSrc) {
    const pa = extractNamedConstant(forecasterSrc, "SPD_BUCKET_TEXT_PA");
    const fat = extractNamedConstant(
      forecasterSrc,
      "SPD_BUCKET_TEXT_FATALITIES",
    );
    const parts = [pa?.trim(), fat?.trim()].filter(Boolean);
    result.spd_buckets = parts.length > 0 ? parts.join("\n\n") : null;
  } else {
    result.spd_buckets = null;
  }

  // --- Scenario (marker extraction from forecaster/prompts.py) ---
  if (forecasterSrc) {
    const raw = extractBetweenMarkers(
      forecasterSrc,
      "scenario_start",
      "scenario_end",
    );
    result.scenario = raw ? sanitizeForDisplay(raw) : null;
  } else {
    result.scenario = null;
  }

  return result;
}

/**
 * Extract all prompt excerpts from live source files.
 * Returns a map of panel ID to extracted text (or null on failure).
 */
export async function extractAllPrompts(): Promise<
  Record<string, string | null>
> {
  const [forecasterSrc, hsSrc, geminiSrc] = await Promise.all([
    readSourceFile("forecaster/prompts.py"),
    readSourceFile("horizon_scanner/prompts.py"),
    readSourceFile("pythia/web_research/backends/gemini_grounding.py"),
  ]);
  return extractPromptsFromSources(forecasterSrc, hsSrc, geminiSrc);
}

/* ------------------------------------------------------------------ */
/*  Versioned prompt support                                          */
/* ------------------------------------------------------------------ */

export interface VersionEntry {
  date: string;
  label: string;
}

export interface VersionedPromptData {
  versions: VersionEntry[];
  prompts: Record<string, Record<string, string | null>>;
}

/**
 * Read the versions manifest from docs/prompts/versions.json.
 */
async function listPromptVersions(): Promise<VersionEntry[]> {
  const candidates = [
    path.join(process.cwd(), "..", "docs", "prompts", "versions.json"),
    path.join(process.cwd(), "docs", "prompts", "versions.json"),
  ];
  for (const p of candidates) {
    try {
      const raw = await readFile(p, "utf-8");
      return JSON.parse(raw) as VersionEntry[];
    } catch {
      // keep trying
    }
  }
  return [];
}

/**
 * Read the 3 archived Python files for a specific version date,
 * then extract prompts using the same logic as live extraction.
 */
async function extractPromptsForVersion(
  date: string,
): Promise<Record<string, string | null>> {
  const relBase = path.join("docs", "prompts", date);
  const [forecasterSrc, hsSrc, geminiSrc] = await Promise.all([
    readSourceFile(path.join(relBase, "forecaster_prompts.py")),
    readSourceFile(path.join(relBase, "hs_prompts.py")),
    readSourceFile(path.join(relBase, "gemini_grounding.py")),
  ]);
  return extractPromptsFromSources(forecasterSrc, hsSrc, geminiSrc);
}

/**
 * Load all versioned prompt snapshots.
 * Returns the version list and a map of date -> extracted prompts.
 */
export async function extractAllVersionedPrompts(): Promise<VersionedPromptData> {
  const versions = await listPromptVersions();
  const prompts: Record<string, Record<string, string | null>> = {};
  await Promise.all(
    versions.map(async (v) => {
      prompts[v.date] = await extractPromptsForVersion(v.date);
    }),
  );
  return { versions, prompts };
}
