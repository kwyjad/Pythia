"use client";
import { useEffect, useMemo, useRef, useState } from "react";

import { classifyJenks, jenksBreaks } from "../lib/jenks";
import type { CountriesRow, RiskIndexRow, RiskView } from "../lib/types";

type RiskIndexMapProps = {
  riskRows: RiskIndexRow[];
  countriesRows: CountriesRow[];
  view: RiskView;
};

const eivFormatter = new Intl.NumberFormat(undefined, {
  maximumFractionDigits: 0,
});

const perCapitaFormatter = new Intl.NumberFormat(undefined, {
  style: "percent",
  minimumFractionDigits: 3,
  maximumFractionDigits: 8,
});

const formatValueLabel = (value: number, isPerCapita: boolean) =>
  isPerCapita ? perCapitaFormatter.format(value) : eivFormatter.format(value);

export default function RiskIndexMap({
  riskRows,
  countriesRows,
  view,
}: RiskIndexMapProps) {
  const [svgText, setSvgText] = useState<string>("");
  const [svgWarnings, setSvgWarnings] = useState<string[]>([]);
  const [tooltip, setTooltip] = useState<{
    x: number;
    y: number;
    title: string;
    value: string;
  } | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    let active = true;
    fetch("/maps/world.svg")
      .then((res) => res.text())
      .then((data) => {
        if (!active) return;
        setSvgText(data);
      })
      .catch((error) => {
        console.warn("Failed to load world map:", error);
      });
    return () => {
      active = false;
    };
  }, []);

  const isPerCapita = view === "PA_PC" || view === "FATALITIES_PC";

  const hasQuestionsIso3 = useMemo(() => {
    const set = new Set<string>();
    countriesRows.forEach((row) => {
      const iso3 = (row.iso3 ?? "").toUpperCase();
      if (iso3) {
        set.add(iso3);
      }
    });
    return set;
  }, [countriesRows]);

  const valueByIso3 = useMemo(() => {
    const map = new Map<string, number>();
    riskRows.forEach((row) => {
      const iso3 = (row.iso3 ?? "").toUpperCase();
      const value = isPerCapita ? row.total_pc : row.total;
      if (iso3 && typeof value === "number" && Number.isFinite(value)) {
        map.set(iso3, value);
      }
    });
    return map;
  }, [riskRows, isPerCapita]);

  const values = useMemo(
    () => Array.from(valueByIso3.values()).filter((value) => Number.isFinite(value)),
    [valueByIso3]
  );

  const breaks = useMemo(() => jenksBreaks(values, 5), [values]);

  const colorScale = [
    "var(--risk-map-c1)",
    "var(--risk-map-c2)",
    "var(--risk-map-c3)",
    "var(--risk-map-c4)",
    "var(--risk-map-c5)",
  ];

  useEffect(() => {
    if (!svgText) {
      setTooltip(null);
      setSvgWarnings([]);
      return;
    }
    const container = containerRef.current;
    if (!container) return;
    const paths = Array.from(
      container.querySelectorAll<SVGPathElement>("[data-iso3]")
    );
    const warnings: string[] = [];
    if (paths.length < 150) {
      warnings.push(
        "World map asset invalid: expected 150+ country paths with data-iso3. Check web/public/maps/world.svg."
      );
    }
    const sampledLengths = paths
      .slice(0, 20)
      .map((path) => (path.getAttribute("d") || "").length)
      .filter((length) => length > 0)
      .sort((a, b) => a - b);
    if (sampledLengths.length) {
      const mid = Math.floor(sampledLengths.length / 2);
      const median =
        sampledLengths.length % 2 === 0
          ? (sampledLengths[mid - 1] + sampledLengths[mid]) / 2
          : sampledLengths[mid];
      if (median < 150) {
        warnings.push(
          "World map asset looks non-polygonal (paths too small). Replace web/public/maps/world.svg with real country outlines."
        );
      }
    }
    setSvgWarnings(warnings);
    const listeners: Array<() => void> = [];
    paths.forEach((path) => {
      const iso3 = (path.getAttribute("data-iso3") || "").toUpperCase();
      if (!iso3) {
        return;
      }
      const value = valueByIso3.get(iso3);
      let fill = "var(--risk-map-no-questions)";
      if (typeof value === "number" && Number.isFinite(value)) {
        const classIndex = classifyJenks(value, breaks);
        fill = colorScale[classIndex] ?? "var(--risk-map-c1)";
      } else if (hasQuestionsIso3.has(iso3)) {
        fill = "var(--risk-map-no-eiv)";
      }
      path.setAttribute("fill", fill);
      path.setAttribute("stroke", "var(--risk-map-stroke)");
      path.setAttribute("stroke-width", "0.5");
      path.setAttribute("vector-effect", "non-scaling-stroke");

      const handleMouseMove = (event: MouseEvent) => {
        const name =
          path.getAttribute("data-name")?.trim() || iso3.toUpperCase();
        let valueLabel = "No forecasts";
        if (typeof value === "number") {
          valueLabel = formatValueLabel(value, isPerCapita);
        } else if (!hasQuestionsIso3.has(iso3)) {
          valueLabel = "No questions";
        }
        const rect = container.getBoundingClientRect();
        setTooltip({
          x: event.clientX - rect.left + 12,
          y: event.clientY - rect.top + 12,
          title: name,
          value: valueLabel,
        });
      };

      const handleMouseLeave = () => setTooltip(null);

      path.addEventListener("mousemove", handleMouseMove);
      path.addEventListener("mouseleave", handleMouseLeave);

      listeners.push(() => {
        path.removeEventListener("mousemove", handleMouseMove);
        path.removeEventListener("mouseleave", handleMouseLeave);
      });
    });

    return () => {
      listeners.forEach((cleanup) => cleanup());
    };
  }, [svgText, valueByIso3, breaks, hasQuestionsIso3, isPerCapita]);

  return (
    <div className="relative w-full rounded-lg border border-slate-800 bg-slate-950/30 p-4">
      <div className="text-sm font-semibold text-slate-200">World overview</div>
      <p className="mt-1 text-xs text-slate-400">
        Jenks breaks calculated from the selected risk values.
      </p>
      {svgWarnings.length ? (
        <div className="mt-3 rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-xs text-amber-200">
          {svgWarnings.map((warning) => (
            <div key={warning}>{warning}</div>
          ))}
        </div>
      ) : null}
      <div ref={containerRef} className="relative mt-3 w-full">
        <div
          aria-label="Risk index world map"
          className="h-auto w-full [&_svg]:h-auto [&_svg]:w-full"
          dangerouslySetInnerHTML={{ __html: svgText }}
        />
        {tooltip ? (
          <div
            className="pointer-events-none absolute z-10 rounded-md border border-slate-700 bg-slate-950 px-3 py-2 text-xs text-slate-100 shadow-lg"
            style={{ left: tooltip.x, top: tooltip.y }}
          >
            <div className="font-semibold">{tooltip.title}</div>
            <div className="text-slate-300">{tooltip.value}</div>
          </div>
        ) : null}
      </div>
    </div>
  );
}
