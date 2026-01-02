"use client";

import { useEffect, useMemo, useState } from "react";

type HsRunRow = {
  run_id: string;
  triage_date?: string | null;
  triage_year?: number | null;
  triage_month?: number | null;
  countries_triaged?: number | null;
  hazards_triaged?: number | null;
};

type HsTriageRow = {
  run_id: string;
  iso3?: string | null;
  hazard_code?: string | null;
  triage_score?: number | null;
  tier?: string | null;
  created_at?: string | null;
};

type HsLlmCallRow = {
  created_at?: string | null;
  iso3?: string | null;
  hazard_code?: string | null;
  model_id?: string | null;
  provider?: string | null;
  parse_error?: string | null;
  response_preview?: string | null;
};

type SummaryRow = {
  run_id: string;
  iso3: string;
  hazards_triaged: number | null;
  questions_generated: number | null;
  questions_forecasted: number | null;
  notes: string[];
};

type DebugRowsResponse<T> = {
  rows: T[];
  schema_debug?: Record<string, unknown>;
};

type DebugRowResponse<T> = {
  row: T;
};

const API_BASE =
  process.env.NEXT_PUBLIC_PYTHIA_API_BASE ?? "http://localhost:8000/v1";

const buildUrl = (
  path: string,
  params?: Record<string, string | number | boolean | null | undefined>
) => {
  const endpoint = path.startsWith("http") ? path : `${API_BASE}${path}`;
  const url = new URL(endpoint);
  if (params) {
    Object.entries(params).forEach(([key, value]) => {
      if (value === null || value === undefined || value === "") {
        return;
      }
      url.searchParams.set(key, String(value));
    });
  }
  return url.toString();
};

const hazardOptions = ["", "ACE", "DI", "DR", "FL", "HW", "TC"];

const DebugClient = () => {
  const [token, setToken] = useState("");
  const [runs, setRuns] = useState<HsRunRow[]>([]);
  const [runId, setRunId] = useState("");
  const [iso3, setIso3] = useState("");
  const [hazard, setHazard] = useState("");
  const [limitTriage, setLimitTriage] = useState(200);
  const [limitCalls, setLimitCalls] = useState(200);
  const [onlyAnomalies, setOnlyAnomalies] = useState(false);
  const [activeTab, setActiveTab] = useState("hs_triage");

  const [triageRows, setTriageRows] = useState<HsTriageRow[]>([]);
  const [triageLoading, setTriageLoading] = useState(false);
  const [triageError, setTriageError] = useState<string | null>(null);

  const [llmRows, setLlmRows] = useState<HsLlmCallRow[]>([]);
  const [llmLoading, setLlmLoading] = useState(false);
  const [llmError, setLlmError] = useState<string | null>(null);

  const [summary, setSummary] = useState<SummaryRow | null>(null);
  const [summaryError, setSummaryError] = useState<string | null>(null);

  const [runsError, setRunsError] = useState<string | null>(null);

  useEffect(() => {
    const stored = window.localStorage.getItem("fred_debug_token");
    if (stored) {
      setToken(stored);
    }
  }, []);

  useEffect(() => {
    if (token) {
      window.localStorage.setItem("fred_debug_token", token);
    } else {
      window.localStorage.removeItem("fred_debug_token");
    }
  }, [token]);

  const fetchDebug = async <T,>(
    path: string,
    params?: Record<string, string | number | boolean | null | undefined>
  ): Promise<T> => {
    const url = buildUrl(path, params);
    const response = await fetch(url, {
      cache: "no-store",
      headers: token ? { "X-Fred-Debug-Token": token } : {},
    });
    if (!response.ok) {
      const bodyText = await response.text().catch(() => "");
      throw new Error(
        `Debug request failed (${response.status}) for ${url}: ${bodyText.slice(0, 200)}`
      );
    }
    return (await response.json()) as T;
  };

  useEffect(() => {
    if (!token) {
      setRuns([]);
      setRunsError(null);
      setRunId("");
      return;
    }

    let cancelled = false;
    const loadRuns = async () => {
      try {
        const response = await fetchDebug<DebugRowsResponse<HsRunRow>>(
          "/debug/hs_runs",
          { limit: 50 }
        );
        if (cancelled) return;
        setRuns(response.rows);
        setRunsError(null);
        if (response.rows.length > 0) {
          setRunId((prev) => prev || response.rows[0].run_id);
        }
      } catch (error) {
        if (cancelled) return;
        setRuns([]);
        setRunsError((error as Error).message);
      }
    };
    loadRuns();
    return () => {
      cancelled = true;
    };
  }, [token]);

  useEffect(() => {
    if (!token || !runId) {
      setTriageRows([]);
      setLlmRows([]);
      setSummary(null);
      return;
    }

    let cancelled = false;

    const loadTriage = async () => {
      setTriageLoading(true);
      try {
        const response = await fetchDebug<DebugRowsResponse<HsTriageRow>>(
          "/debug/hs_triage",
          { run_id: runId, iso3: iso3 || null, hazard_code: hazard || null, limit: limitTriage }
        );
        if (cancelled) return;
        setTriageRows(response.rows);
        setTriageError(null);
      } catch (error) {
        if (cancelled) return;
        setTriageRows([]);
        setTriageError((error as Error).message);
      } finally {
        if (!cancelled) setTriageLoading(false);
      }
    };

    const loadLlmCalls = async () => {
      setLlmLoading(true);
      try {
        const response = await fetchDebug<DebugRowsResponse<HsLlmCallRow>>(
          "/debug/hs_triage_llm_calls",
          {
            run_id: runId,
            iso3: iso3 || null,
            hazard_code: hazard || null,
            limit: limitCalls,
            preview_chars: 800,
          }
        );
        if (cancelled) return;
        setLlmRows(response.rows);
        setLlmError(null);
      } catch (error) {
        if (cancelled) return;
        setLlmRows([]);
        setLlmError((error as Error).message);
      } finally {
        if (!cancelled) setLlmLoading(false);
      }
    };

    const loadSummary = async () => {
      try {
        const response = await fetchDebug<DebugRowResponse<SummaryRow>>(
          "/debug/hs_country_summary",
          { run_id: runId, iso3: iso3 || null }
        );
        if (cancelled) return;
        setSummary(response.row);
        setSummaryError(null);
      } catch (error) {
        if (cancelled) return;
        setSummary(null);
        setSummaryError((error as Error).message);
      }
    };

    loadTriage();
    loadLlmCalls();
    if (iso3) {
      loadSummary();
    } else {
      setSummary(null);
    }

    return () => {
      cancelled = true;
    };
  }, [token, runId, iso3, hazard, limitTriage, limitCalls]);

  const filteredTriageRows = useMemo(() => {
    if (!onlyAnomalies) return triageRows;
    return triageRows.filter((row) => {
      const score = row.triage_score ?? null;
      const tier = row.tier?.toLowerCase();
      return score === 0 || tier === "quiet";
    });
  }, [onlyAnomalies, triageRows]);

  const filteredLlmRows = useMemo(() => {
    if (!onlyAnomalies) return llmRows;
    return llmRows.filter((row) => Boolean(row.parse_error));
  }, [onlyAnomalies, llmRows]);

  const tabs = [
    { id: "hs_triage", label: "HS Triage" },
    { id: "research", label: "Research" },
    { id: "forecast", label: "Forecast" },
    { id: "scenario", label: "Scenario" },
    { id: "web_search", label: "Web search" },
  ];

  return (
    <div className="space-y-6">
      <section className="rounded-lg border border-fred-secondary bg-white p-4">
        <h2 className="text-lg font-semibold">Access</h2>
        <p className="text-sm text-fred-muted">
          Enter the debug token to unlock admin-only endpoints.
        </p>
        <div className="mt-3 flex flex-col gap-2 sm:flex-row sm:items-center">
          <label className="text-sm font-medium">Debug token</label>
          <input
            type="password"
            value={token}
            onChange={(event) => setToken(event.target.value)}
            className="w-full rounded-md border border-fred-secondary px-3 py-2 text-sm sm:max-w-md"
            placeholder="X-Fred-Debug-Token"
          />
        </div>
        {runsError ? (
          <p className="mt-2 text-sm text-red-600">{runsError}</p>
        ) : null}
      </section>

      <section className="rounded-lg border border-fred-secondary bg-white p-4">
        <div className="flex flex-wrap gap-2">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              type="button"
              onClick={() => setActiveTab(tab.id)}
              className={`rounded-full px-3 py-1 text-sm font-semibold ${
                activeTab === tab.id
                  ? "bg-fred-primary text-white"
                  : "border border-fred-secondary text-fred-primary"
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
        {activeTab !== "hs_triage" ? (
          <div className="mt-4 rounded-md border border-dashed border-fred-secondary p-4 text-sm text-fred-muted">
            {tabs.find((tab) => tab.id === activeTab)?.label} diagnostics are coming soon.
          </div>
        ) : null}
      </section>

      {activeTab === "hs_triage" ? (
        <section className="space-y-6">
          <div className="rounded-lg border border-fred-secondary bg-white p-4">
            <h2 className="text-lg font-semibold">Filters</h2>
            <div className="mt-4 grid gap-4 md:grid-cols-2">
              <div className="flex flex-col gap-2">
                <label className="text-sm font-medium">Run</label>
                <select
                  className="rounded-md border border-fred-secondary px-3 py-2 text-sm"
                  value={runId}
                  onChange={(event) => setRunId(event.target.value)}
                >
                  <option value="">Select a run</option>
                  {runs.map((run) => (
                    <option key={run.run_id} value={run.run_id}>
                      {run.run_id} {run.triage_date ? `(${run.triage_date})` : ""}
                    </option>
                  ))}
                </select>
              </div>
              <div className="flex flex-col gap-2">
                <label className="text-sm font-medium">ISO3</label>
                <input
                  className="rounded-md border border-fred-secondary px-3 py-2 text-sm"
                  value={iso3}
                  onChange={(event) => setIso3(event.target.value.toUpperCase())}
                  placeholder="UKR"
                />
              </div>
              <div className="flex flex-col gap-2">
                <label className="text-sm font-medium">Hazard code</label>
                <select
                  className="rounded-md border border-fred-secondary px-3 py-2 text-sm"
                  value={hazard}
                  onChange={(event) => setHazard(event.target.value)}
                >
                  {hazardOptions.map((code) => (
                    <option key={code || "all"} value={code}>
                      {code || "All hazards"}
                    </option>
                  ))}
                </select>
              </div>
              <div className="flex flex-col gap-2">
                <label className="text-sm font-medium">Only anomalies</label>
                <label className="flex items-center gap-2 text-sm">
                  <input
                    type="checkbox"
                    checked={onlyAnomalies}
                    onChange={(event) => setOnlyAnomalies(event.target.checked)}
                  />
                  Score = 0, tier = quiet, or parse errors
                </label>
              </div>
              <div className="flex flex-col gap-2">
                <label className="text-sm font-medium">Triage row limit</label>
                <select
                  className="rounded-md border border-fred-secondary px-3 py-2 text-sm"
                  value={limitTriage}
                  onChange={(event) => setLimitTriage(Number(event.target.value))}
                >
                  {[50, 100, 200, 500, 1000].map((value) => (
                    <option key={value} value={value}>
                      {value}
                    </option>
                  ))}
                </select>
              </div>
              <div className="flex flex-col gap-2">
                <label className="text-sm font-medium">LLM call limit</label>
                <select
                  className="rounded-md border border-fred-secondary px-3 py-2 text-sm"
                  value={limitCalls}
                  onChange={(event) => setLimitCalls(Number(event.target.value))}
                >
                  {[50, 100, 200, 500, 1000].map((value) => (
                    <option key={value} value={value}>
                      {value}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          </div>

          <div className="rounded-lg border border-fred-secondary bg-white p-4">
            <h2 className="text-lg font-semibold">Country/run summary</h2>
            {!iso3 ? (
              <p className="mt-2 text-sm text-fred-muted">
                Enter an ISO3 code to load summary counts.
              </p>
            ) : summaryError ? (
              <p className="mt-2 text-sm text-red-600">{summaryError}</p>
            ) : summary ? (
              <div className="mt-4 grid gap-4 md:grid-cols-3">
                <div>
                  <p className="text-sm text-fred-muted">Hazards triaged</p>
                  <p className="text-xl font-semibold">
                    {summary.hazards_triaged ?? "—"}
                  </p>
                </div>
                <div>
                  <p className="text-sm text-fred-muted">Questions generated</p>
                  <p className="text-xl font-semibold">
                    {summary.questions_generated ?? "—"}
                  </p>
                </div>
                <div>
                  <p className="text-sm text-fred-muted">Questions forecasted</p>
                  <p className="text-xl font-semibold">
                    {summary.questions_forecasted ?? "—"}
                  </p>
                </div>
                {summary.notes.length > 0 ? (
                  <div className="md:col-span-3">
                    <p className="text-sm font-semibold">Notes</p>
                    <ul className="list-disc pl-5 text-sm text-fred-muted">
                      {summary.notes.map((note) => (
                        <li key={note}>{note}</li>
                      ))}
                    </ul>
                  </div>
                ) : null}
              </div>
            ) : (
              <p className="mt-2 text-sm text-fred-muted">Summary not loaded.</p>
            )}
          </div>

          <div className="rounded-lg border border-fred-secondary bg-white p-4">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold">HS triage rows</h2>
              <span className="text-sm text-fred-muted">
                {filteredTriageRows.length} rows
              </span>
            </div>
            {triageLoading ? (
              <p className="mt-2 text-sm text-fred-muted">Loading triage rows…</p>
            ) : triageError ? (
              <p className="mt-2 text-sm text-red-600">{triageError}</p>
            ) : (
              <div className="mt-4 overflow-x-auto">
                <table className="min-w-full text-sm">
                  <thead className="text-left text-fred-muted">
                    <tr>
                      <th className="pb-2">ISO3</th>
                      <th className="pb-2">Hazard</th>
                      <th className="pb-2">Score</th>
                      <th className="pb-2">Tier</th>
                      <th className="pb-2">Created</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredTriageRows.map((row, index) => (
                      <tr key={`${row.run_id}-${row.iso3}-${row.hazard_code}-${index}`}>
                        <td className="py-1 pr-4">{row.iso3 ?? "—"}</td>
                        <td className="py-1 pr-4">{row.hazard_code ?? "—"}</td>
                        <td className="py-1 pr-4">{row.triage_score ?? "—"}</td>
                        <td className="py-1 pr-4">{row.tier ?? "—"}</td>
                        <td className="py-1">{row.created_at ?? "—"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          <div className="rounded-lg border border-fred-secondary bg-white p-4">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold">HS triage LLM calls</h2>
              <span className="text-sm text-fred-muted">
                {filteredLlmRows.length} rows
              </span>
            </div>
            {llmLoading ? (
              <p className="mt-2 text-sm text-fred-muted">Loading LLM calls…</p>
            ) : llmError ? (
              <p className="mt-2 text-sm text-red-600">{llmError}</p>
            ) : (
              <div className="mt-4 overflow-x-auto">
                <table className="min-w-full text-sm">
                  <thead className="text-left text-fred-muted">
                    <tr>
                      <th className="pb-2">ISO3</th>
                      <th className="pb-2">Hazard</th>
                      <th className="pb-2">Model</th>
                      <th className="pb-2">Provider</th>
                      <th className="pb-2">Parse error</th>
                      <th className="pb-2">Created</th>
                      <th className="pb-2">Response preview</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredLlmRows.map((row, index) => (
                      <tr key={`${row.iso3}-${row.hazard_code}-${index}`}>
                        <td className="py-1 pr-4">{row.iso3 ?? "—"}</td>
                        <td className="py-1 pr-4">{row.hazard_code ?? "—"}</td>
                        <td className="py-1 pr-4">{row.model_id ?? "—"}</td>
                        <td className="py-1 pr-4">{row.provider ?? "—"}</td>
                        <td className="py-1 pr-4">
                          {row.parse_error ? "Yes" : "—"}
                        </td>
                        <td className="py-1 pr-4">{row.created_at ?? "—"}</td>
                        <td className="py-1">
                          <pre className="max-w-xl whitespace-pre-wrap text-xs text-fred-muted">
                            {row.response_preview ?? ""}
                          </pre>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </section>
      ) : null}
    </div>
  );
};

export default DebugClient;
