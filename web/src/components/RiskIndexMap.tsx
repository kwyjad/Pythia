"use client";

import { useEffect, useMemo, useRef, useState, type MouseEvent } from "react";

import { classifyJenks, jenksBreaks } from "../lib/jenks";
import type { CountriesRow, RiskIndexRow, RiskView } from "../lib/types";

type GeoFeature = {
  type: "Feature";
  properties?: Record<string, unknown>;
  geometry: {
    type: string;
    coordinates: unknown;
  };
};

type GeoCollection = {
  type: "FeatureCollection";
  features: GeoFeature[];
};

type RiskIndexMapProps = {
  riskRows: RiskIndexRow[];
  countriesRows: CountriesRow[];
  view: RiskView;
};

const MAP_WIDTH = 960;
const MAP_HEIGHT = 520;

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
  const [features, setFeatures] = useState<GeoFeature[]>([]);
  const [tooltip, setTooltip] = useState<{
    x: number;
    y: number;
    title: string;
    value: string;
  } | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    let active = true;
    fetch("/maps/world-countries-iso3.geojson")
      .then((res) => res.json())
      .then((data: GeoCollection) => {
        if (!active) return;
        setFeatures(data.features ?? []);
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

  const projectPoint = (lon: number, lat: number) => {
    const x = ((lon + 180) / 360) * MAP_WIDTH;
    const y = ((90 - lat) / 180) * MAP_HEIGHT;
    return [x, y];
  };

  const renderRing = (ring: number[][]) => {
    if (!ring.length) return "";
    const [firstLon, firstLat] = ring[0];
    const [x0, y0] = projectPoint(firstLon, firstLat);
    const segments = [`M${x0},${y0}`];
    for (let i = 1; i < ring.length; i++) {
      const [lon, lat] = ring[i];
      const [x, y] = projectPoint(lon, lat);
      segments.push(`L${x},${y}`);
    }
    segments.push("Z");
    return segments.join(" ");
  };

  const pathForFeature = (feature: GeoFeature) => {
    const geometry = feature.geometry;
    if (!geometry) return "";
    if (geometry.type === "Polygon") {
      const coords = geometry.coordinates as number[][][];
      return coords.map(renderRing).join(" ");
    }
    if (geometry.type === "MultiPolygon") {
      const coords = geometry.coordinates as number[][][][];
      return coords.map((polygon) => polygon.map(renderRing).join(" ")).join(" ");
    }
    return "";
  };

  const getFeatureIso3 = (feature: GeoFeature) => {
    const props = feature.properties ?? {};
    const iso3 = (props.iso3 ?? props.ISO_A3 ?? props["ISO_A3"] ?? "") as string;
    return iso3 ? iso3.toUpperCase() : "";
  };

  const getFeatureName = (feature: GeoFeature) => {
    const props = feature.properties ?? {};
    return (props.name ?? props.NAME ?? props["NAME"] ?? "") as string;
  };

  const colorScale = [
    "var(--risk-map-c1)",
    "var(--risk-map-c2)",
    "var(--risk-map-c3)",
    "var(--risk-map-c4)",
    "var(--risk-map-c5)",
  ];

  const handleMouseMove = (
    event: MouseEvent<SVGPathElement>,
    feature: GeoFeature
  ) => {
    const iso3 = getFeatureIso3(feature);
    if (!iso3) {
      setTooltip(null);
      return;
    }
    const name = getFeatureName(feature) || iso3;
    const value = valueByIso3.get(iso3);
    let valueLabel = "No forecasts";
    if (typeof value === "number") {
      valueLabel = formatValueLabel(value, isPerCapita);
    } else if (!hasQuestionsIso3.has(iso3)) {
      valueLabel = "No questions";
    }
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) {
      return;
    }
    setTooltip({
      x: event.clientX - rect.left + 12,
      y: event.clientY - rect.top + 12,
      title: name,
      value: valueLabel,
    });
  };

  const handleMouseLeave = () => setTooltip(null);

  return (
    <div className="relative w-full rounded-lg border border-slate-800 bg-slate-950/30 p-4">
      <div className="text-sm font-semibold text-slate-200">World overview</div>
      <p className="mt-1 text-xs text-slate-400">
        Jenks breaks calculated from the selected risk values.
      </p>
      <div ref={containerRef} className="relative mt-3 w-full">
        <svg
          aria-label="Risk index world map"
          className="h-auto w-full"
          viewBox={`0 0 ${MAP_WIDTH} ${MAP_HEIGHT}`}
        >
          <rect
            x={0}
            y={0}
            width={MAP_WIDTH}
            height={MAP_HEIGHT}
            fill="transparent"
          />
          {features.map((feature, index) => {
            const iso3 = getFeatureIso3(feature);
            const value = iso3 ? valueByIso3.get(iso3) : undefined;
            let fill = "var(--risk-map-no-questions)";
            if (typeof value === "number" && Number.isFinite(value)) {
              const classIndex = classifyJenks(value, breaks);
              fill = colorScale[classIndex] ?? "var(--risk-map-c1)";
            } else if (iso3 && hasQuestionsIso3.has(iso3)) {
              fill = "var(--risk-map-no-eiv)";
            }
            return (
              <path
                key={`${iso3 || "country"}-${index}`}
                d={pathForFeature(feature) || undefined}
                fill={fill}
                stroke="var(--risk-map-stroke)"
                strokeWidth={0.6}
                onMouseLeave={handleMouseLeave}
                onMouseMove={(event) => handleMouseMove(event, feature)}
              />
            );
          })}
        </svg>
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
