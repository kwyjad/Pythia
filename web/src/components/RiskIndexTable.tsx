"use client";

import Link from "next/link";
import { useMemo, useState } from "react";

import type { RiskIndexRow } from "../lib/types";
import SortableTable, { SortableColumn } from "./SortableTable";

const EIV_TOOLTIP =
  "EIV is computed as: for each question and month, sum over buckets of (P(bucket) × centroid(bucket)), then summed across all forecasted questions in the country. Centroids come from the SPD bucket definitions (or bucket_centroids table).";
const POPULATION_TOOLTIP = "population table not populated.";

const formatNumber = (value: number | null | undefined) =>
  typeof value === "number" ? value.toLocaleString() : null;

const formatPerCapita = (value: number | null | undefined) =>
  typeof value === "number"
    ? value.toLocaleString(undefined, { maximumFractionDigits: 6 })
    : null;

type RiskIndexTableProps = {
  rows: RiskIndexRow[];
  targetMonth?: string | null;
};

const renderEivHeader = (label: string) => (
  <span className="inline-flex items-center gap-1">
    {label}
    <span
      className="rounded-full border border-slate-500 px-1 text-xs text-slate-300"
      title={EIV_TOOLTIP}
    >
      ?
    </span>
  </span>
);

export default function RiskIndexTable({ rows, targetMonth }: RiskIndexTableProps) {
  const [showPerCapita, setShowPerCapita] = useState(false);
  const populationAvailable = rows.some(
    (row) => typeof row.population === "number"
  );

  const perCapitaCell = (value: number | null | undefined) => {
    const formatted = formatPerCapita(value);
    if (formatted) {
      return formatted;
    }
    return (
      <span title={populationAvailable ? undefined : POPULATION_TOOLTIP}>-</span>
    );
  };

  const columns = useMemo<Array<SortableColumn<RiskIndexRow>>>(() => {
    const eivColumns: Array<SortableColumn<RiskIndexRow>> = [
      {
        key: "m1",
        label: renderEivHeader("M1 EIV"),
        headerClassName: "text-right",
        cellClassName: "text-right",
        sortValue: (row) => row.m1 ?? null,
        render: (row) => formatNumber(row.m1) ?? "-",
      },
      {
        key: "m2",
        label: renderEivHeader("M2 EIV"),
        headerClassName: "text-right",
        cellClassName: "text-right",
        sortValue: (row) => row.m2 ?? null,
        render: (row) => formatNumber(row.m2) ?? "-",
      },
      {
        key: "m3",
        label: renderEivHeader("M3 EIV"),
        headerClassName: "text-right",
        cellClassName: "text-right",
        sortValue: (row) => row.m3 ?? null,
        render: (row) => formatNumber(row.m3) ?? "-",
      },
      {
        key: "m4",
        label: renderEivHeader("M4 EIV"),
        headerClassName: "text-right",
        cellClassName: "text-right",
        sortValue: (row) => row.m4 ?? null,
        render: (row) => formatNumber(row.m4) ?? "-",
      },
      {
        key: "m5",
        label: renderEivHeader("M5 EIV"),
        headerClassName: "text-right",
        cellClassName: "text-right",
        sortValue: (row) => row.m5 ?? null,
        render: (row) => formatNumber(row.m5) ?? "-",
      },
      {
        key: "m6",
        label: renderEivHeader("M6 EIV"),
        headerClassName: "text-right",
        cellClassName: "text-right",
        sortValue: (row) => row.m6 ?? null,
        render: (row) => formatNumber(row.m6) ?? "-",
      },
      {
        key: "total",
        label: renderEivHeader("Total EIV"),
        headerClassName: "text-right",
        cellClassName: "text-right",
        sortValue: (row) => row.total ?? null,
        render: (row) => formatNumber(row.total) ?? "-",
      },
    ];

    const perCapitaColumns: Array<SortableColumn<RiskIndexRow>> = [
      {
        key: "m1_pc",
        label: "M1 per capita",
        headerClassName: "text-right",
        cellClassName: "text-right",
        sortValue: (row) => row.m1_pc ?? null,
        render: (row) => perCapitaCell(row.m1_pc),
        isVisible: showPerCapita,
      },
      {
        key: "m2_pc",
        label: "M2 per capita",
        headerClassName: "text-right",
        cellClassName: "text-right",
        sortValue: (row) => row.m2_pc ?? null,
        render: (row) => perCapitaCell(row.m2_pc),
        isVisible: showPerCapita,
      },
      {
        key: "m3_pc",
        label: "M3 per capita",
        headerClassName: "text-right",
        cellClassName: "text-right",
        sortValue: (row) => row.m3_pc ?? null,
        render: (row) => perCapitaCell(row.m3_pc),
        isVisible: showPerCapita,
      },
      {
        key: "m4_pc",
        label: "M4 per capita",
        headerClassName: "text-right",
        cellClassName: "text-right",
        sortValue: (row) => row.m4_pc ?? null,
        render: (row) => perCapitaCell(row.m4_pc),
        isVisible: showPerCapita,
      },
      {
        key: "m5_pc",
        label: "M5 per capita",
        headerClassName: "text-right",
        cellClassName: "text-right",
        sortValue: (row) => row.m5_pc ?? null,
        render: (row) => perCapitaCell(row.m5_pc),
        isVisible: showPerCapita,
      },
      {
        key: "m6_pc",
        label: "M6 per capita",
        headerClassName: "text-right",
        cellClassName: "text-right",
        sortValue: (row) => row.m6_pc ?? null,
        render: (row) => perCapitaCell(row.m6_pc),
        isVisible: showPerCapita,
      },
      {
        key: "total_pc",
        label: "Total per capita",
        headerClassName: "text-right",
        cellClassName: "text-right",
        sortValue: (row) => row.total_pc ?? null,
        render: (row) => perCapitaCell(row.total_pc),
        isVisible: showPerCapita,
      },
    ];

    return [
      {
        key: "iso3",
        label: "ISO3",
        sortValue: (row) => row.iso3,
        render: (row) => (
          <Link className="underline underline-offset-2" href={`/countries/${row.iso3}`}>
            {row.iso3}
          </Link>
        ),
      },
      {
        key: "country_name",
        label: "Country",
        sortValue: (row) => row.country_name ?? "",
        render: (row) => row.country_name || "-",
      },
      {
        key: "n_hazards_forecasted",
        label: "Hazards forecasted",
        headerClassName: "text-right",
        cellClassName: "text-right",
        sortValue: (row) => row.n_hazards_forecasted ?? null,
        render: (row) =>
          row.n_hazards_forecasted != null
            ? row.n_hazards_forecasted.toLocaleString()
            : "-",
      },
      ...eivColumns,
      ...perCapitaColumns,
    ];
  }, [showPerCapita, populationAvailable]);

  const emptyMessage = targetMonth
    ? `No rows returned for ${targetMonth}.`
    : "No rows returned. (no month)";

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-4 text-sm text-slate-300">
        <label className="inline-flex items-center gap-2">
          <input
            checked={showPerCapita}
            className="h-4 w-4 rounded border-slate-600 bg-slate-900"
            onChange={(event) => setShowPerCapita(event.target.checked)}
            type="checkbox"
          />
          <span>Show per-capita columns</span>
        </label>
        {!populationAvailable ? (
          <span className="text-xs text-slate-500">
            Per-capita values unavailable: {POPULATION_TOOLTIP}
          </span>
        ) : null}
      </div>

      <SortableTable
        columns={columns}
        emptyMessage={emptyMessage}
        initialSortKey="total"
        initialSortDirection="desc"
        rowKey={(row) => row.iso3}
        rows={rows}
      />
    </div>
  );
}
