"use client";

import Link from "next/link";
import { useMemo } from "react";

import SortableTable, { SortableColumn } from "../../components/SortableTable";

type CountriesRow = {
  iso3: string;
  n_questions: number;
  n_forecasted: number;
};

type CountriesTableProps = {
  rows: CountriesRow[];
};

export default function CountriesTable({ rows }: CountriesTableProps) {
  const columns = useMemo<Array<SortableColumn<CountriesRow>>>(
    () => [
      {
        key: "iso3",
        label: "Country",
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
