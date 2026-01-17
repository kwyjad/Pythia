"use client";

import Link from "next/link";
import { useMemo } from "react";

import InfoTooltip from "../../components/InfoTooltip";
import SortableTable, { SortableColumn } from "../../components/SortableTable";

type CountriesRow = {
  iso3: string;
  country_name?: string | null;
  last_forecasted?: string | null; // YYYY-MM-DD
  last_triaged?: string | null; // YYYY-MM-DD
  n_questions: number;
  n_forecasted: number;
  highest_rc_level?: number | null;
  highest_rc_score?: number | null;
};

type CountriesTableProps = {
  rows: CountriesRow[];
};

export default function CountriesTable({ rows }: CountriesTableProps) {
  const rcTooltip =
    "Highest RC score (probability × magnitude) across hazards for the latest HS run.";
  const columns = useMemo<Array<SortableColumn<CountriesRow>>>(
    () => [
      {
        key: "iso3",
        label: "ISO3",
        headerClassName: "text-left",
        cellClassName: "text-left",
        sortValue: (row) => row.iso3,
        render: (row) => (
          <Link
            href={`/countries/${row.iso3}`}
            className="block w-full px-0 py-0 text-sky-300 underline underline-offset-2 hover:text-sky-200"
          >
            {row.iso3}
          </Link>
        ),
        defaultSortDirection: "asc",
      },
      {
        key: "country_name",
        label: "Name",
        headerClassName: "text-left",
        cellClassName: "text-left",
        sortValue: (row) => row.country_name ?? "",
        render: (row) => row.country_name ?? "",
        defaultSortDirection: "asc",
      },
      {
        key: "highest_rc_score",
        label: (
          <span className="inline-flex items-center gap-1">
            Highest RC Score <InfoTooltip text={rcTooltip} />
          </span>
        ),
        headerClassName: "text-right",
        cellClassName: "text-right tabular-nums",
        sortValue: (row) => {
          const score = row.highest_rc_score;
          if (typeof score !== "number") {
            return null;
          }
          const level = row.highest_rc_level ?? 0;
          return score + level / 100;
        },
        render: (row) => {
          const score = row.highest_rc_score;
          if (typeof score !== "number") {
            return "—";
          }
          const level = row.highest_rc_level;
          const levelLabel = typeof level === "number" ? `L${level} • ` : "";
          return `${levelLabel}${score.toFixed(2)}`;
        },
        defaultSortDirection: "desc",
      },
      {
        key: "n_questions",
        label: "Questions",
        headerClassName: "text-right",
        cellClassName: "text-right tabular-nums",
        sortValue: (row) => row.n_questions,
        render: (row) => row.n_questions.toLocaleString(),
        defaultSortDirection: "desc",
      },
      {
        key: "n_forecasted",
        label: "Forecasted",
        headerClassName: "text-right",
        cellClassName: "text-right tabular-nums",
        sortValue: (row) => row.n_forecasted,
        render: (row) => row.n_forecasted.toLocaleString(),
        defaultSortDirection: "desc",
      },
      {
        key: "last_forecasted",
        label: "Last Forecasted",
        headerClassName: "text-right",
        cellClassName: "text-right tabular-nums",
        sortValue: (row) =>
          row.last_forecasted ? Date.parse(row.last_forecasted) : null,
        render: (row) => row.last_forecasted ?? "",
        defaultSortDirection: "desc",
      },
      {
        key: "last_triaged",
        label: "Last Triaged",
        headerClassName: "text-right",
        cellClassName: "text-right tabular-nums",
        sortValue: (row) => (row.last_triaged ? Date.parse(row.last_triaged) : null),
        render: (row) => row.last_triaged ?? "",
        defaultSortDirection: "desc",
      },
    ],
    []
  );

  return (
    <SortableTable
      columns={columns}
      emptyMessage="No countries available. No data in DB snapshot."
      rowKey={(row) => row.iso3}
      rows={rows}
    />
  );
}
