"use client";

import { useEffect, useMemo, useState } from "react";

import type { DiagnosticsKpiScopesResponse } from "../lib/types";
import KpiCard from "./KpiCard";

const STORAGE_KEY = "fred_kpi_scope";

type RiskIndexKpiScopeSelectorProps = {
  kpiScopes: DiagnosticsKpiScopesResponse;
};

type ScopeKey = "latest_run" | "total_active" | "total_all";

const scopeOrder: ScopeKey[] = ["latest_run", "total_active", "total_all"];

const defaultScopeLabels: Record<ScopeKey, string> = {
  latest_run: "Most recent run",
  total_active: "Total active",
  total_all: "Total active + inactive",
};

const kpiLabels: Record<ScopeKey, { questions: string; forecasts: string }> = {
  latest_run: {
    questions: "Questions (most recent run)",
    forecasts: "Questions with forecasts (most recent run)",
  },
  total_active: {
    questions: "Active questions",
    forecasts: "Active questions with forecasts",
  },
  total_all: {
    questions: "All questions",
    forecasts: "All questions with forecasts",
  },
};

const RiskIndexKpiScopeSelector = ({
  kpiScopes,
}: RiskIndexKpiScopeSelectorProps) => {
  const [selectedScope, setSelectedScope] = useState<string>(
    kpiScopes.default_scope ?? "latest_run"
  );
  const [storedScope, setStoredScope] = useState<string | null>(null);
  const [debugEnabled, setDebugEnabled] = useState(false);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    const params = new URLSearchParams(window.location.search);
    setDebugEnabled(params.get("debug_kpi") === "1");
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    const stored = window.localStorage.getItem(STORAGE_KEY);
    setStoredScope(stored);
    if (stored && kpiScopes.scopes?.[stored]) {
      setSelectedScope(stored);
    } else if (kpiScopes.default_scope) {
      setSelectedScope(kpiScopes.default_scope);
    }
  }, [kpiScopes.default_scope, kpiScopes.scopes]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    window.localStorage.setItem(STORAGE_KEY, selectedScope);
    setStoredScope(selectedScope);
  }, [selectedScope]);

  const scopeData = useMemo(() => {
    const fallbackScope =
      kpiScopes.scopes?.[kpiScopes.default_scope] ??
      kpiScopes.scopes?.latest_run ??
      null;
    return kpiScopes.scopes?.[selectedScope] ?? fallbackScope;
  }, [kpiScopes.default_scope, kpiScopes.scopes, selectedScope]);

  const labelSet = useMemo(() => {
    const key = scopeOrder.includes(selectedScope as ScopeKey)
      ? (selectedScope as ScopeKey)
      : (kpiScopes.default_scope as ScopeKey) ?? "latest_run";
    return kpiLabels[key] ?? kpiLabels.latest_run;
  }, [kpiScopes.default_scope, selectedScope]);

  return (
    <div className="space-y-4" data-testid="risk-index-kpi-panel">
      <div className="space-y-2">
        <label
          className="text-xs font-semibold uppercase tracking-wide text-fred-muted"
          htmlFor="risk-index-kpi-scope"
        >
          KPI scope
        </label>
        <select
          className="w-full rounded-md border border-fred-secondary bg-fred-surface px-3 py-2 text-sm text-fred-primary"
          id="risk-index-kpi-scope"
          onChange={(event) => setSelectedScope(event.target.value)}
          value={selectedScope}
        >
          {scopeOrder.map((scopeKey) => (
            <option key={scopeKey} value={scopeKey}>
              {kpiScopes.scopes?.[scopeKey]?.label ??
                defaultScopeLabels[scopeKey]}
            </option>
          ))}
        </select>
      </div>

      <KpiCard label={labelSet.questions} value={scopeData?.questions ?? 0} />
      <KpiCard
        label={labelSet.forecasts}
        value={scopeData?.questions_with_forecasts ?? 0}
      />

      {debugEnabled ? (
        <details className="rounded-lg border border-fred-secondary/60 bg-fred-surface px-4 py-3 text-xs text-fred-muted">
          <summary className="cursor-pointer font-semibold text-fred-primary">
            KPI diagnostics
          </summary>
          <pre className="mt-2 whitespace-pre-wrap">
            {JSON.stringify(
              {
                selected_scope: selectedScope,
                stored_scope: storedScope,
                diagnostics: kpiScopes.diagnostics ?? null,
              },
              null,
              2
            )}
          </pre>
        </details>
      ) : null}
    </div>
  );
};

export default RiskIndexKpiScopeSelector;
