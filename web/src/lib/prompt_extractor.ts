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
 * - Replaces f-string {expr.method(args)} with [expr]
 * - Unescapes doubled braces and \n literals
 * - Collapses excess blank lines
 */
function sanitizeForDisplay(text: string): string {
  let s = text;

  // Strip leading Python syntax:
  //   'return f"""...'  or  'return """...'
  //   'return (\n  ...'  or  'variable = (\n  ...'
  s = s.replace(/^\s*return\s+f?"""\s*\\?\s*\n?/, "");
  s = s.replace(/^\s*return\s*\(\s*\n?/, "");
  s = s.replace(/^\s*\w+\s*=\s*\(\s*\n?/, "");

  // Strip trailing triple-quote closure and parens
  s = s.replace(/"""\s*$/, "");
  s = s.replace(/\)\s*$/, "");

  // Join Python string concatenation:
  // end-of-string + start-of-next-string on next line
  // Handles both double-quoted and single-quoted Python strings:
  //   '...\n"\n        "...'  or  '...\n"\n        f"...'
  //   "...\n'\n        '..."  or  "...\n'\n        f'..."
  s = s.replace(/"\s*\n\s*f?"/g, "");
  s = s.replace(/'\s*\n\s*f?'/g, "");
  // Cross-quote joins (double→single and single→double)
  s = s.replace(/"\s*\n\s*f?'/g, "");
  s = s.replace(/'\s*\n\s*f?"/g, "");

  // Replace json.dumps / _json_dumps_for_prompt calls with [data]
  s = s.replace(/\{json\.dumps\([^)]+\)\}/g, "[data]");
  s = s.replace(/\{_json_dumps_for_prompt\([^)]+\)\}/g, "[data]");

  // Replace f-string expressions with method calls: {expr.method(args)} -> [expr]
  s = s.replace(
    /\{([a-z_][a-z0-9_]*)\.[a-z_][a-z0-9_]*\([^)]*\)\}/gi,
    (_match, varName: string) => `[${varName.replace(/_/g, " ")}]`,
  );

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
  s = s.replace(/^["']\s*/gm, "");
  s = s.replace(/\s*["']$/gm, "");

  // Collapse 3+ consecutive blank lines to 2
  s = s.replace(/\n{4,}/g, "\n\n\n");

  return s.trim();
}

/**
 * Helper: extract multiple marker-based prompts from one source file.
 */
function extractMarkerSet(
  source: string | null,
  prefix: string,
  hazards: string[],
  result: Record<string, string | null>,
): void {
  for (const h of hazards) {
    const key = `${prefix}_${h}`;
    if (source) {
      const raw = extractBetweenMarkers(
        source,
        `${prefix}_${h}_start`,
        `${prefix}_${h}_end`,
      );
      result[key] = raw ? sanitizeForDisplay(raw) : null;
    } else {
      result[key] = null;
    }
  }
}

const HAZARDS = ["ace", "dr", "fl", "hw", "tc"];

/**
 * Core extraction logic shared between live and versioned prompts.
 */
function extractPromptsFromSources(
  forecasterSrc: string | null,
  hsSrc: string | null,
  geminiSrc: string | null,
  rcPromptsSrc: string | null,
  rcGroundingSrc: string | null,
  triageGroundingSrc: string | null,
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

  // --- SPD v2 prompt (marker extraction from build_spd_prompt_v2) ---
  if (forecasterSrc) {
    const raw = extractBetweenMarkers(
      forecasterSrc,
      "spd_v2_start",
      "spd_v2_end",
    );
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

  // --- RC prompts (per-hazard, from horizon_scanner/rc_prompts.py) ---
  if (rcPromptsSrc) {
    const raw = extractNamedConstant(rcPromptsSrc, "_RC_CALIBRATION_PREAMBLE");
    result.rc_preamble = raw ? sanitizeForDisplay(raw) : null;
  } else {
    result.rc_preamble = null;
  }
  extractMarkerSet(rcPromptsSrc, "rc", HAZARDS, result);

  // --- RC grounding prompts (per-hazard) ---
  extractMarkerSet(rcGroundingSrc, "rc_grounding", HAZARDS, result);

  // --- Triage grounding prompts (per-hazard) ---
  extractMarkerSet(triageGroundingSrc, "triage_grounding", HAZARDS, result);

  return result;
}

/**
 * Extract all prompt excerpts from live source files.
 * Returns a map of panel ID to extracted text (or null on failure).
 */
export async function extractAllPrompts(): Promise<
  Record<string, string | null>
> {
  const [forecasterSrc, hsSrc, geminiSrc, rcPromptsSrc, rcGroundingSrc, triageGroundingSrc] =
    await Promise.all([
      readSourceFile("forecaster/prompts.py"),
      readSourceFile("horizon_scanner/prompts.py"),
      readSourceFile("pythia/web_research/backends/gemini_grounding.py"),
      readSourceFile("horizon_scanner/rc_prompts.py"),
      readSourceFile("horizon_scanner/rc_grounding_prompts.py"),
      readSourceFile("horizon_scanner/hs_triage_grounding_prompts.py"),
    ]);
  return extractPromptsFromSources(
    forecasterSrc, hsSrc, geminiSrc,
    rcPromptsSrc, rcGroundingSrc, triageGroundingSrc,
  );
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
 * Read archived Python files for a specific version date,
 * then extract prompts using the same logic as live extraction.
 */
async function extractPromptsForVersion(
  date: string,
): Promise<Record<string, string | null>> {
  const relBase = path.join("docs", "prompts", date);
  const [forecasterSrc, hsSrc, geminiSrc, rcPromptsSrc, rcGroundingSrc, triageGroundingSrc] =
    await Promise.all([
      readSourceFile(path.join(relBase, "forecaster_prompts.py")),
      readSourceFile(path.join(relBase, "hs_prompts.py")),
      readSourceFile(path.join(relBase, "gemini_grounding.py")),
      readSourceFile(path.join(relBase, "rc_prompts.py")),
      readSourceFile(path.join(relBase, "rc_grounding_prompts.py")),
      readSourceFile(path.join(relBase, "hs_triage_grounding_prompts.py")),
    ]);
  return extractPromptsFromSources(
    forecasterSrc, hsSrc, geminiSrc,
    rcPromptsSrc, rcGroundingSrc, triageGroundingSrc,
  );
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
