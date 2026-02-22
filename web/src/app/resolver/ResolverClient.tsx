"use client";

import { useEffect, useMemo, useState } from "react";

import SortableTable, { SortableColumn } from "../../components/SortableTable";
import { apiGet } from "../../lib/api";

export type ConnectorStatusRow = {
  source: string;
  last_updated: string | null;
  rows_scanned: number;
};

export type ResolverCountryOption = {
  iso3: string;
  country_name?: string | null;
};

type ResolverFactRow = {
  iso3: string;
  hazard: string | null;
  hazard_code: string | null;
  source_id: string | null;
  year: number | null;
  month: number | null;
  metric: string | null;
  value: number | null;
};

type ResolverFactsResponse = {
  rows: ResolverFactRow[];
  iso3: string;
};

type ResolverRowView = ResolverFactRow & {
  country_name: string | null;
  hazard_display: string;
  hazard_key: string;
  source_key: string;
  metric_key: string;
  year_key: number | null;
  month_key: number | null;
};

type ResolverClientProps = {
  countries: ResolverCountryOption[];
  connectorStatus: ConnectorStatusRow[];
};

const formatValue = (value: number | null) => {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "";
  }
  return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
};

const normalizeFilterValue = (value: string | null | undefined) =>
  (value ?? "").trim();

const resolveIso3 = (
  input: string,
  countryByIso3: Map<string, ResolverCountryOption>
) => {
  const trimmed = input.trim();
  if (!trimmed) return null;
  const upper = trimmed.toUpperCase();
  if (countryByIso3.has(upper)) return upper;
  const match = trimmed.match(/\(([A-Za-z]{3})\)/);
  if (match) {
    const iso3 = match[1].toUpperCase();
    if (countryByIso3.has(iso3)) return iso3;
  }
  const lower = trimmed.toLowerCase();
  for (const option of countryByIso3.values()) {
    if ((option.country_name ?? "").toLowerCase() === lower) {
      return option.iso3;
    }
  }
  return null;
};

const buildHazardDisplay = (row: ResolverFactRow) => {
  const code = (row.hazard_code ?? "").trim();
  const label = (row.hazard ?? "").trim();
  if (label && code && label.toUpperCase() !== code.toUpperCase()) {
    return `${label} (${code})`;
  }
  return label || code || "Unknown";
};

export default function ResolverClient({
  countries,
  connectorStatus,
}: ResolverClientProps) {
  const [countryInput, setCountryInput] = useState("");
  const [selectedIso3, setSelectedIso3] = useState<string | null>(null);
  const [rows, setRows] = useState<ResolverFactRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [sourceFilter, setSourceFilter] = useState("");
  const [hazardFilter, setHazardFilter] = useState("");
  const [yearFilter, setYearFilter] = useState("");
  const [monthFilter, setMonthFilter] = useState("");
  const [metricFilter, setMetricFilter] = useState("");

  const countryByIso3 = useMemo(() => {
    const map = new Map<string, ResolverCountryOption>();
    countries.forEach((country) => {
      if (country.iso3) {
        map.set(country.iso3.toUpperCase(), {
          ...country,
          iso3: country.iso3.toUpperCase(),
        });
      }
    });
    return map;
  }, [countries]);

  useEffect(() => {
    if (!selectedIso3) {
      setRows([]);
      setError(null);
      return;
    }
    let cancelled = false;
    const fetchRows = async () => {
      setLoading(true);
      setError(null);
      try {
        const response = await apiGet<ResolverFactsResponse>(
          "/resolver/country_facts",
          { iso3: selectedIso3 }
        );
        if (cancelled) return;
        setRows(response.rows ?? []);
      } catch (fetchError) {
        if (cancelled) return;
        console.warn("Failed to load resolver facts:", fetchError);
        setError("Failed to load resolver facts.");
        setRows([]);
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    };
    fetchRows();
    return () => {
      cancelled = true;
    };
  }, [selectedIso3]);

  const viewRows = useMemo<ResolverRowView[]>(() => {
    return rows.map((row) => {
      const iso3 = (row.iso3 ?? "").toUpperCase();
      const country = countryByIso3.get(iso3);
      const hazardDisplay = buildHazardDisplay(row);
      return {
        ...row,
        iso3,
        country_name: country?.country_name ?? null,
        hazard_display: hazardDisplay,
        hazard_key: row.hazard_code ?? row.hazard ?? "",
        source_key: row.source_id ?? "",
        metric_key: row.metric ?? "",
        year_key: row.year ?? null,
        month_key: row.month ?? null,
      };
    });
  }, [countryByIso3, rows]);

  const sourceOptions = useMemo(() => {
    const set = new Set<string>();
    viewRows.forEach((row) => {
      const value = normalizeFilterValue(row.source_key);
      if (value) set.add(value);
    });
    return Array.from(set).sort();
  }, [viewRows]);

  const hazardOptions = useMemo(() => {
    const map = new Map<string, string>();
    viewRows.forEach((row) => {
      const key = normalizeFilterValue(row.hazard_key);
      if (!key) return;
      map.set(key, row.hazard_display);
    });
    return Array.from(map.entries())
      .map(([key, label]) => ({ key, label }))
      .sort((a, b) => a.label.localeCompare(b.label));
  }, [viewRows]);

  const yearOptions = useMemo(() => {
    const set = new Set<number>();
    viewRows.forEach((row) => {
      if (row.year_key) set.add(row.year_key);
    });
    return Array.from(set).sort((a, b) => a - b);
  }, [viewRows]);

  const monthOptions = useMemo(() => {
    const set = new Set<number>();
    viewRows.forEach((row) => {
      if (row.month_key) set.add(row.month_key);
    });
    return Array.from(set).sort((a, b) => a - b);
  }, [viewRows]);

  const metricOptions = useMemo(() => {
    const set = new Set<string>();
    viewRows.forEach((row) => {
      const value = normalizeFilterValue(row.metric_key);
      if (value) set.add(value);
    });
    return Array.from(set).sort();
  }, [viewRows]);

  const filteredRows = useMemo(() => {
    return viewRows.filter((row) => {
      if (sourceFilter && row.source_key !== sourceFilter) return false;
      if (hazardFilter && row.hazard_key !== hazardFilter) return false;
      if (yearFilter && String(row.year_key ?? "") !== yearFilter) return false;
      if (monthFilter && String(row.month_key ?? "") !== monthFilter) return false;
      if (metricFilter && row.metric_key !== metricFilter) return false;
      return true;
    });
  }, [hazardFilter, metricFilter, monthFilter, sourceFilter, viewRows, yearFilter]);

  const columns = useMemo<Array<SortableColumn<ResolverRowView>>>(
    () => [
      {
        key: "ym_sort",
        label: "Year-Month",
        isVisible: false,
        sortValue: (row) =>
          row.year_key && row.month_key
            ? row.year_key * 100 + row.month_key
            : null,
        defaultSortDirection: "desc",
      },
      {
        key: "iso3",
        label: "ISO3",
        headerClassName: "text-left",
        cellClassName: "text-left",
        sortValue: (row) => row.iso3,
        render: (row) => row.iso3,
        defaultSortDirection: "asc",
      },
      {
        key: "country_name",
        label: "Country Name",
        headerClassName: "text-left",
        cellClassName: "text-left",
        sortValue: (row) => row.country_name ?? "",
        render: (row) => row.country_name ?? "",
        defaultSortDirection: "asc",
      },
      {
        key: "hazard_display",
        label: "Hazard",
        headerClassName: "text-left",
        cellClassName: "text-left",
        sortValue: (row) => row.hazard_display,
        render: (row) => row.hazard_display,
        defaultSortDirection: "asc",
      },
      {
        key: "source_id",
        label: "Data Source",
        headerClassName: "text-left",
        cellClassName: "text-left",
        sortValue: (row) => row.source_id ?? "",
        render: (row) => row.source_id ?? "",
        defaultSortDirection: "asc",
      },
      {
        key: "year",
        label: "Year",
        headerClassName: "text-right",
        cellClassName: "text-right tabular-nums",
        sortValue: (row) => row.year_key ?? null,
        render: (row) => (row.year_key ? row.year_key.toString() : ""),
        defaultSortDirection: "desc",
      },
      {
        key: "month",
        label: "Month",
        headerClassName: "text-right",
        cellClassName: "text-right tabular-nums",
        sortValue: (row) => row.month_key ?? null,
        render: (row) => (row.month_key ? row.month_key.toString() : ""),
        defaultSortDirection: "desc",
      },
      {
        key: "metric",
        label: "PA/Fatalities",
        headerClassName: "text-left",
        cellClassName: "text-left",
        sortValue: (row) => row.metric_key,
        render: (row) => row.metric_key,
        defaultSortDirection: "asc",
      },
      {
        key: "value",
        label: "Value",
        headerClassName: "text-right",
        cellClassName: "text-right tabular-nums",
        sortValue: (row) => row.value ?? null,
        render: (row) => formatValue(row.value),
        defaultSortDirection: "desc",
      },
    ],
    []
  );

  return (
    <div className="space-y-6">
      <section>
        <h1 className="text-3xl font-semibold">Resolver</h1>
        <p className="text-sm text-fred-text">
          Inspect resolved facts that feed forecasting outputs.
        </p>
      </section>

      <section className="space-y-3">
        <h2 className="text-lg font-semibold">Connector last updated</h2>
        <div className="grid gap-4 md:grid-cols-4">
          {["ACLED", "IDMC", "IFRC"].map((source) => {
            const row = connectorStatus.find((item) => item.source === source);
            return (
              <div
                key={source}
                className="rounded-lg border border-fred-secondary bg-fred-surface px-4 py-3"
              >
                <div className="text-sm font-semibold text-fred-primary">
                  {source}
                </div>
                <div className="text-xl font-semibold text-fred-text">
                  {row?.last_updated ?? "Unknown"}
                </div>
                <div className="text-xs text-fred-muted">
                  Rows scanned: {row?.rows_scanned?.toLocaleString() ?? "0"}
                </div>
              </div>
            );
          })}
          {(() => {
            const row = connectorStatus.find((item) => item.source === "EM-DAT");
            return (
              <div className="rounded-lg border border-fred-secondary/50 bg-fred-surface/50 px-4 py-3 opacity-60">
                <div className="text-sm font-semibold text-fred-muted">
                  EM-DAT <span className="font-normal">(historical)</span>
                </div>
                <div className="text-xl font-semibold text-fred-muted">
                  {row?.last_updated ?? "Discontinued"}
                </div>
                <div className="text-xs text-fred-muted">
                  Rows scanned: {row?.rows_scanned?.toLocaleString() ?? "0"}
                </div>
              </div>
            );
          })()}
        </div>
      </section>

      <section className="space-y-4">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-end">
          <div className="flex flex-col gap-2">
            <label className="text-sm font-semibold text-fred-primary">
              Country selector
            </label>
            <input
              list="country-options"
              className="w-full rounded-md border border-fred-secondary bg-fred-surface px-3 py-2 text-sm text-fred-text"
              placeholder="Type to search (ISO3 or country name)"
              value={countryInput}
              onChange={(event) => {
                const value = event.target.value;
                setCountryInput(value);
                setSelectedIso3(resolveIso3(value, countryByIso3));
              }}
            />
            <datalist id="country-options">
              {countries.map((country) => (
                <option key={country.iso3} value={country.iso3}>
                  {country.country_name
                    ? `${country.country_name} (${country.iso3})`
                    : country.iso3}
                </option>
              ))}
            </datalist>
            {selectedIso3 ? (
              <span className="text-xs text-fred-muted">
                Selected:{" "}
                {countryByIso3.get(selectedIso3)?.country_name ??
                  selectedIso3}
              </span>
            ) : null}
          </div>

          <div className="flex flex-1 flex-wrap gap-3">
            <div className="flex flex-col gap-1">
              <label className="text-xs font-semibold text-fred-muted">
                Data Source
              </label>
              <select
                className="rounded-md border border-fred-secondary bg-fred-surface px-2 py-1 text-sm text-fred-text"
                value={sourceFilter}
                onChange={(event) => setSourceFilter(event.target.value)}
              >
                <option value="">All</option>
                {sourceOptions.map((source) => (
                  <option key={source} value={source}>
                    {source}
                  </option>
                ))}
              </select>
            </div>
            <div className="flex flex-col gap-1">
              <label className="text-xs font-semibold text-fred-muted">
                Hazard
              </label>
              <select
                className="rounded-md border border-fred-secondary bg-fred-surface px-2 py-1 text-sm text-fred-text"
                value={hazardFilter}
                onChange={(event) => setHazardFilter(event.target.value)}
              >
                <option value="">All</option>
                {hazardOptions.map((hazard) => (
                  <option key={hazard.key} value={hazard.key}>
                    {hazard.label}
                  </option>
                ))}
              </select>
            </div>
            <div className="flex flex-col gap-1">
              <label className="text-xs font-semibold text-fred-muted">
                Year
              </label>
              <select
                className="rounded-md border border-fred-secondary bg-fred-surface px-2 py-1 text-sm text-fred-text"
                value={yearFilter}
                onChange={(event) => setYearFilter(event.target.value)}
              >
                <option value="">All</option>
                {yearOptions.map((year) => (
                  <option key={year} value={year}>
                    {year}
                  </option>
                ))}
              </select>
            </div>
            <div className="flex flex-col gap-1">
              <label className="text-xs font-semibold text-fred-muted">
                Month
              </label>
              <select
                className="rounded-md border border-fred-secondary bg-fred-surface px-2 py-1 text-sm text-fred-text"
                value={monthFilter}
                onChange={(event) => setMonthFilter(event.target.value)}
              >
                <option value="">All</option>
                {monthOptions.map((month) => (
                  <option key={month} value={month}>
                    {month}
                  </option>
                ))}
              </select>
            </div>
            <div className="flex flex-col gap-1">
              <label className="text-xs font-semibold text-fred-muted">
                PA/Fatalities
              </label>
              <select
                className="rounded-md border border-fred-secondary bg-fred-surface px-2 py-1 text-sm text-fred-text"
                value={metricFilter}
                onChange={(event) => setMetricFilter(event.target.value)}
              >
                <option value="">All</option>
                {metricOptions.map((metric) => (
                  <option key={metric} value={metric}>
                    {metric}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>

        <div className="flex items-center justify-between text-sm text-fred-muted">
          <span>Rows: {filteredRows.length.toLocaleString()} (after filters)</span>
          {loading ? <span>Loading…</span> : null}
        </div>

        {error ? <div className="text-sm text-red-400">{error}</div> : null}

        <div className="overflow-x-auto rounded-lg border border-fred-secondary">
          <SortableTable
            columns={columns}
            emptyMessage={
              selectedIso3
                ? "No resolver facts available for this country."
                : "Select a country to view resolver facts."
            }
            initialSortKey="ym_sort"
            initialSortDirection="desc"
            rowKey={(row) =>
              `${row.iso3}-${row.hazard_key}-${row.source_key}-${row.metric_key}-${row.year_key}-${row.month_key}`
            }
            rows={filteredRows}
            tableLayout="auto"
            dense
          />
        </div>
      </section>
    </div>
  );
}
