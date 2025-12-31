"use client";

import { useMemo, useState } from "react";

export type SortDirection = "asc" | "desc";

export type SortableColumn<T> = {
  key: string;
  label: React.ReactNode;
  headerClassName?: string;
  cellClassName?: string;
  sortValue?: (row: T) => number | string | null | undefined;
  render?: (row: T) => React.ReactNode;
  sortable?: boolean;
  isVisible?: boolean;
  defaultSortDirection?: SortDirection;
};

type SortableTableProps<T> = {
  columns: Array<SortableColumn<T>>;
  rows: Array<T>;
  rowKey: (row: T) => string;
  initialSortKey?: string;
  initialSortDirection?: SortDirection;
  emptyMessage?: string;
  tableLayout?: "fixed" | "auto";
  dense?: boolean;
};

const compareValues = (
  a: number | string | null | undefined,
  b: number | string | null | undefined,
  direction: SortDirection
) => {
  if (a == null && b == null) {
    return 0;
  }
  if (a == null) {
    return 1;
  }
  if (b == null) {
    return -1;
  }
  if (typeof a === "number" && typeof b === "number") {
    const base = a - b;
    return direction === "asc" ? base : -base;
  }
  const base = String(a).localeCompare(String(b));
  return direction === "asc" ? base : -base;
};

export default function SortableTable<T>({
  columns,
  rows,
  rowKey,
  initialSortKey,
  initialSortDirection = "desc",
  emptyMessage = "No rows returned.",
  tableLayout = "fixed",
  dense = false,
}: SortableTableProps<T>) {
  const [sortKey, setSortKey] = useState<string | undefined>(initialSortKey);
  const [sortDirection, setSortDirection] =
    useState<SortDirection>(initialSortDirection);

  const visibleColumns = useMemo(
    () => columns.filter((column) => column.isVisible !== false),
    [columns]
  );

  const sortedRows = useMemo(() => {
    if (!sortKey) {
      return rows;
    }
    const column = columns.find((col) => col.key === sortKey);
    if (!column || column.sortable === false) {
      return rows;
    }
    const valueFor = (row: T) =>
      column.sortValue ? column.sortValue(row) : (row as Record<string, unknown>)[sortKey];
    return [...rows].sort((a, b) =>
      compareValues(
        valueFor(a) as number | string | null | undefined,
        valueFor(b) as number | string | null | undefined,
        sortDirection
      )
    );
  }, [columns, rows, sortDirection, sortKey]);

  const handleSort = (column: SortableColumn<T>) => {
    if (column.sortable === false) {
      return;
    }
    setSortKey((prevKey) => {
      if (prevKey === column.key) {
        setSortDirection((prevDirection) =>
          prevDirection === "asc" ? "desc" : "asc"
        );
        return prevKey;
      }
      setSortDirection(column.defaultSortDirection ?? "asc");
      return column.key;
    });
  };

  const paddingClass = dense ? "py-1" : "py-2";
  const layoutClass = tableLayout === "auto" ? "table-auto" : "table-fixed";

  return (
    <table className={`w-full ${layoutClass} border-collapse text-sm`}>
      <thead className="bg-fredBg text-fredPrimary">
        <tr>
          {visibleColumns.map((column) => (
            <th
              key={column.key}
              className={`px-2 ${paddingClass} text-left ${
                column.headerClassName ?? ""
              }`}
            >
              <button
                className={`inline-flex items-center gap-1 ${
                  column.sortable === false ? "cursor-default" : "cursor-pointer"
                }`}
                type="button"
                onClick={() => handleSort(column)}
              >
                <span>{column.label}</span>
                {sortKey === column.key ? (
                  <span className="text-xs">
                    {sortDirection === "asc" ? "▲" : "▼"}
                  </span>
                ) : null}
              </button>
            </th>
          ))}
        </tr>
      </thead>
      <tbody className="divide-y divide-fredBorder text-fredText">
        {sortedRows.map((row) => (
          <tr key={rowKey(row)} className="hover:bg-fredBg/60">
            {visibleColumns.map((column) => (
              <td
                key={column.key}
                className={`px-2 ${paddingClass} ${column.cellClassName ?? ""}`}
              >
                {column.render ? column.render(row) : null}
              </td>
            ))}
          </tr>
        ))}
        {sortedRows.length === 0 ? (
          <tr>
            <td className="px-3 py-3 text-fredMuted" colSpan={visibleColumns.length}>
              {emptyMessage}
            </td>
          </tr>
        ) : null}
      </tbody>
    </table>
  );
}
