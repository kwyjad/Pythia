"use client";

import Link from "next/link";
import { useMemo } from "react";

import type { RiskIndexRow } from "../lib/types";
import InfoTooltip from "./InfoTooltip";
import SortableTable, { SortableColumn } from "./SortableTable";

const EIV_TOOLTIP =
  "EIV = sum over buckets of p(bucket) × centroid(bucket), aggregated across all forecasted hazards for the country, shown per month and summed over months 1–6.";
const POPULATION_TOOLTIP =
  "Per-capita requires population data. The populations table is empty in this snapshot.";

const formatNumber = (value: number | null | undefined) =>
  typeof value === "number" ? Math.round(value).toLocaleString() : null;

const formatPerCapita = (value: number | null | undefined) =>
  typeof value === "number"
    ? value.toLocaleString(undefined, { maximumFractionDigits: 6 })
    : null;

const parseTargetMonth = (targetMonth?: string | null): Date | null => {
  if (!targetMonth) {
    return null;
  }
  const [year, month] = targetMonth.split("-").map((value) => Number(value));
  if (!year || !month) {
    return null;
  }
  return new Date(Date.UTC(year, month - 1, 1));
};

const addMonthsUTC = (date: Date, months: number): Date =>
  new Date(Date.UTC(date.getUTCFullYear(), date.getUTCMonth() + months, 1));

const formatMonthLabel = (date: Date): string =>
  date.toLocaleString("en-US", {
    month: "short",
    year: "numeric",
    timeZone: "UTC",
  });

type RiskIndexTableProps = {
  rows: RiskIndexRow[];
  targetMonth?: string | null;
  mode: "raw" | "percap";
};

const renderEivHeader = (label: string) => (
  <span className="inline-flex items-center gap-1">
    {label}
    <InfoTooltip text={EIV_TOOLTIP} />
  </span>
);

export default function RiskIndexTable({
  rows,
  targetMonth,
  mode,
}: RiskIndexTableProps) {
  const populationAvailable = rows.some(
    (row) => typeof row.population === "number"
  );
  const baseMonth = parseTargetMonth(targetMonth);
  const monthLabel = (index: number) =>
    baseMonth
      ? formatMonthLabel(addMonthsUTC(baseMonth, index - 1))
      : `M${index}`;

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
        label: renderEivHeader(`${monthLabel(1)} EIV`),
        headerClassName: "w-32 text-right",
        cellClassName: "w-32 text-right tabular-nums",
        sortValue: (row) => row.m1 ?? null,
        render: (row) => formatNumber(row.m1) ?? "-",
        defaultSortDirection: "desc",
      },
      {
        key: "m2",
        label: renderEivHeader(`${monthLabel(2)} EIV`),
        headerClassName: "w-32 text-right",
        cellClassName: "w-32 text-right tabular-nums",
        sortValue: (row) => row.m2 ?? null,
        render: (row) => formatNumber(row.m2) ?? "-",
        defaultSortDirection: "desc",
      },
      {
        key: "m3",
        label: renderEivHeader(`${monthLabel(3)} EIV`),
        headerClassName: "w-32 text-right",
        cellClassName: "w-32 text-right tabular-nums",
        sortValue: (row) => row.m3 ?? null,
        render: (row) => formatNumber(row.m3) ?? "-",
        defaultSortDirection: "desc",
      },
      {
        key: "m4",
        label: renderEivHeader(`${monthLabel(4)} EIV`),
        headerClassName: "w-32 text-right",
        cellClassName: "w-32 text-right tabular-nums",
        sortValue: (row) => row.m4 ?? null,
        render: (row) => formatNumber(row.m4) ?? "-",
        defaultSortDirection: "desc",
      },
      {
        key: "m5",
        label: renderEivHeader(`${monthLabel(5)} EIV`),
        headerClassName: "w-32 text-right",
        cellClassName: "w-32 text-right tabular-nums",
        sortValue: (row) => row.m5 ?? null,
        render: (row) => formatNumber(row.m5) ?? "-",
        defaultSortDirection: "desc",
      },
      {
        key: "m6",
        label: renderEivHeader(`${monthLabel(6)} EIV`),
        headerClassName: "w-32 text-right",
        cellClassName: "w-32 text-right tabular-nums",
        sortValue: (row) => row.m6 ?? null,
        render: (row) => formatNumber(row.m6) ?? "-",
        defaultSortDirection: "desc",
      },
      {
        key: "total",
        label: renderEivHeader("Total EIV"),
        headerClassName: "w-32 text-right",
        cellClassName: "w-32 text-right tabular-nums",
        sortValue: (row) => row.total ?? null,
        render: (row) => formatNumber(row.total) ?? "-",
        defaultSortDirection: "desc",
      },
    ];

    const perCapitaColumns: Array<SortableColumn<RiskIndexRow>> = [
      {
        key: "m1_pc",
        label: `${monthLabel(1)} per capita`,
        headerClassName: "w-32 text-right",
        cellClassName: "w-32 text-right tabular-nums",
        sortValue: (row) => row.m1_pc ?? null,
        render: (row) => perCapitaCell(row.m1_pc),
        defaultSortDirection: "desc",
      },
      {
        key: "m2_pc",
        label: `${monthLabel(2)} per capita`,
        headerClassName: "w-32 text-right",
        cellClassName: "w-32 text-right tabular-nums",
        sortValue: (row) => row.m2_pc ?? null,
        render: (row) => perCapitaCell(row.m2_pc),
        defaultSortDirection: "desc",
      },
      {
        key: "m3_pc",
        label: `${monthLabel(3)} per capita`,
        headerClassName: "w-32 text-right",
        cellClassName: "w-32 text-right tabular-nums",
        sortValue: (row) => row.m3_pc ?? null,
        render: (row) => perCapitaCell(row.m3_pc),
        defaultSortDirection: "desc",
      },
      {
        key: "m4_pc",
        label: `${monthLabel(4)} per capita`,
        headerClassName: "w-32 text-right",
        cellClassName: "w-32 text-right tabular-nums",
        sortValue: (row) => row.m4_pc ?? null,
        render: (row) => perCapitaCell(row.m4_pc),
        defaultSortDirection: "desc",
      },
      {
        key: "m5_pc",
        label: `${monthLabel(5)} per capita`,
        headerClassName: "w-32 text-right",
        cellClassName: "w-32 text-right tabular-nums",
        sortValue: (row) => row.m5_pc ?? null,
        render: (row) => perCapitaCell(row.m5_pc),
        defaultSortDirection: "desc",
      },
      {
        key: "m6_pc",
        label: `${monthLabel(6)} per capita`,
        headerClassName: "w-32 text-right",
        cellClassName: "w-32 text-right tabular-nums",
        sortValue: (row) => row.m6_pc ?? null,
        render: (row) => perCapitaCell(row.m6_pc),
        defaultSortDirection: "desc",
      },
      {
        key: "total_pc",
        label: "Total per capita",
        headerClassName: "w-32 text-right",
        cellClassName: "w-32 text-right tabular-nums",
        sortValue: (row) => row.total_pc ?? null,
        render: (row) => perCapitaCell(row.total_pc),
        defaultSortDirection: "desc",
      },
    ];

    const metricColumns = mode === "percap" ? perCapitaColumns : eivColumns;

    return [
      {
        key: "iso3",
        label: "ISO3",
        headerClassName: "w-16",
        cellClassName: "w-16",
        sortValue: (row) => row.iso3,
        render: (row) => (
          <Link className="underline underline-offset-2" href={`/countries/${row.iso3}`}>
            {row.iso3}
          </Link>
        ),
        defaultSortDirection: "asc",
      },
      {
        key: "country_name",
        label: "Country",
        headerClassName: "w-56",
        cellClassName: "w-56",
        sortValue: (row) => row.country_name ?? "",
        render: (row) => row.country_name || "-",
        defaultSortDirection: "asc",
      },
      {
        key: "n_hazards_forecasted",
        label: "Hazards forecasted",
        headerClassName: "w-24 text-right",
        cellClassName: "w-24 text-right tabular-nums",
        sortValue: (row) => row.n_hazards_forecasted ?? null,
        render: (row) =>
          row.n_hazards_forecasted != null
            ? row.n_hazards_forecasted.toLocaleString()
            : "-",
        defaultSortDirection: "desc",
      },
      ...metricColumns,
    ];
  }, [mode, monthLabel, populationAvailable]);

  const emptyMessage = targetMonth
    ? `No rows returned for ${targetMonth}.`
    : "No rows returned. (no month)";

  return (
    <SortableTable
      columns={columns}
      emptyMessage={emptyMessage}
      initialSortKey={mode === "percap" ? "total_pc" : "total"}
      initialSortDirection="desc"
      rowKey={(row) => row.iso3}
      rows={rows}
    />
  );
}
