"use client";

import { useEffect, useMemo, useRef } from "react";

import type { RunMonth } from "../lib/run_months";
import { formatYearMonthLabel, sortMonthsDesc } from "../lib/run_months";

const STORAGE_KEY = "fred_selected_run_month";

type RunMonthSelectorProps = {
  availableMonths: RunMonth[];
  selectedMonth: string | null;
  onChange: (yearMonth: string) => void;
};

const RunMonthSelector = ({
  availableMonths,
  selectedMonth,
  onChange,
}: RunMonthSelectorProps) => {
  const syncedRef = useRef(false);
  const sortedMonths = useMemo(
    () => sortMonthsDesc(availableMonths ?? []),
    [availableMonths]
  );

  useEffect(() => {
    if (typeof window === "undefined") return;
    if (syncedRef.current) return;
    syncedRef.current = true;
    const stored = window.localStorage.getItem(STORAGE_KEY);
    if (!stored) return;
    if (!sortedMonths.some((month) => month.year_month === stored)) return;
    if (stored !== selectedMonth) {
      onChange(stored);
    }
  }, [onChange, selectedMonth, sortedMonths]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    if (!selectedMonth) return;
    window.localStorage.setItem(STORAGE_KEY, selectedMonth);
  }, [selectedMonth]);

  return (
    <label className="flex items-center gap-2 text-sm text-fred-text">
      <span>Run month</span>
      <select
        className="rounded-md border border-fred-secondary bg-fred-surface px-3 py-2 text-sm text-fred-text focus:outline-none focus:ring-2 focus:ring-fred-primary/30"
        onChange={(event) => onChange(event.target.value)}
        value={selectedMonth ?? ""}
      >
        {sortedMonths.map((month) => (
          <option key={month.year_month} value={month.year_month}>
            {month.label ?? formatYearMonthLabel(month.year_month)}
          </option>
        ))}
      </select>
    </label>
  );
};

export default RunMonthSelector;
