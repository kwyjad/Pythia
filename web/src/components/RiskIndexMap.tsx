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

const roundLatitude = (lat: number) => Math.round(lat * 10000) / 10000;

const collectOuterRingLatitudes = (
  geometry: GeoFeature["geometry"],
  latitudes: Set<number>
) => {
  if (!geometry) return;
  if (geometry.type === "Polygon") {
    const coords = geometry.coordinates as number[][][] | undefined;
    const ring = coords?.[0];
    ring?.forEach(([, lat]) => {
      if (typeof lat === "number" && Number.isFinite(lat)) {
        latitudes.add(roundLatitude(lat));
      }
    });
    return;
  }
  if (geometry.type === "MultiPolygon") {
    const coords = geometry.coordinates as number[][][][] | undefined;
    coords?.forEach((polygon) => {
      const ring = polygon?.[0];
      ring?.forEach(([, lat]) => {
        if (typeof lat === "number" && Number.isFinite(lat)) {
          latitudes.add(roundLatitude(lat));
        }
      });
    });
  }
};

const collectOuterRingPositions = (geometry: GeoFeature["geometry"]) => {
  const positions: Array<[number, number]> = [];
  if (!geometry) return positions;
  if (geometry.type === "Polygon") {
    const coords = geometry.coordinates as number[][][] | undefined;
    const ring = coords?.[0];
    ring?.forEach(([lon, lat]) => {
      if (
        typeof lon === "number" &&
        Number.isFinite(lon) &&
        typeof lat === "number" &&
        Number.isFinite(lat)
      ) {
        positions.push([lon, lat]);
      }
    });
    return positions;
  }
  if (geometry.type === "MultiPolygon") {
    const coords = geometry.coordinates as number[][][][] | undefined;
    coords?.forEach((polygon) => {
      const ring = polygon?.[0];
      ring?.forEach(([lon, lat]) => {
        if (
          typeof lon === "number" &&
          Number.isFinite(lon) &&
          typeof lat === "number" &&
          Number.isFinite(lat)
        ) {
          positions.push([lon, lat]);
        }
      });
    });
  }
  return positions;
};

const normalizeCoord = ([a, b]: [number, number]) => {
  let lon = a;
  let lat = b;
  if (Math.abs(a) <= 90 && Math.abs(b) > 90) {
    lon = b;
    lat = a;
  }
  if (lon > 180) {
    lon -= 360;
  }
  return [lon, lat] as [number, number];
};

const mapCoordinates = (coords: unknown): unknown => {
  if (!Array.isArray(coords)) {
    return coords;
  }
  if (coords.length >= 2 && coords.every((value) => typeof value === "number")) {
    const [lon, lat] = coords as number[];
    const [nextLon, nextLat] = normalizeCoord([lon, lat]);
    return [nextLon, nextLat, ...(coords.slice(2) as number[])];
  }
  return coords.map((child) => mapCoordinates(child));
};

const normalizeFeatures = (rawFeatures: GeoFeature[]) => {
  const normalizedFeatures = rawFeatures.map((feature) => {
    const geometry = feature.geometry;
    if (!geometry) return feature;
    return {
      ...feature,
      geometry: {
        ...geometry,
        coordinates: mapCoordinates(geometry.coordinates),
      },
    };
  });

  const normalizedPositions: Array<[number, number]> = [];
  normalizedFeatures.slice(0, 50).forEach((feature) => {
    normalizedPositions.push(...collectOuterRingPositions(feature.geometry));
  });

  let normalizedAbsLonMax = 0;
  let normalizedAbsLatMax = 0;
  normalizedPositions.forEach(([lon, lat]) => {
    normalizedAbsLonMax = Math.max(normalizedAbsLonMax, Math.abs(lon));
    normalizedAbsLatMax = Math.max(normalizedAbsLatMax, Math.abs(lat));
  });

  const invalid =
    normalizedAbsLonMax > 200 ||
    normalizedAbsLatMax > 100 ||
    Number.isNaN(normalizedAbsLonMax) ||
    Number.isNaN(normalizedAbsLatMax);

  return { features: normalizedFeatures, invalid };
};

const projectCoordinate = (lon: number, lat: number) => {
  const x = ((lon + 180) / 360) * MAP_WIDTH;
  const y = ((90 - lat) / 180) * MAP_HEIGHT;
  return [x, y] as [number, number];
};

const toPathForRing = (ring: unknown) => {
  if (!Array.isArray(ring)) return null;
  const points: Array<[number, number]> = [];
  ring.forEach((coord) => {
    if (!Array.isArray(coord) || coord.length < 2) return;
    const [lon, lat] = coord as number[];
    if (
      typeof lon === "number" &&
      Number.isFinite(lon) &&
      typeof lat === "number" &&
      Number.isFinite(lat)
    ) {
      points.push(projectCoordinate(lon, lat));
    }
  });
  if (!points.length) return null;
  const [startX, startY] = points[0];
  const segments = points
    .slice(1)
    .map(([x, y]) => `L ${x} ${y}`)
    .join(" ");
  return `M ${startX} ${startY} ${segments} Z`;
};

const toPathForPolygon = (coordinates: unknown) => {
  if (!Array.isArray(coordinates)) return null;
  const ringPaths = coordinates
    .map((ring) => toPathForRing(ring))
    .filter((path): path is string => Boolean(path));
  if (!ringPaths.length) return null;
  return ringPaths.join(" ");
};

const toPathForMultiPolygon = (coordinates: unknown) => {
  if (!Array.isArray(coordinates)) return null;
  const polygonPaths = coordinates
    .map((polygon) => toPathForPolygon(polygon))
    .filter((path): path is string => Boolean(path));
  if (!polygonPaths.length) return null;
  return polygonPaths.join(" ");
};

const toPathForFeature = (feature: GeoFeature) => {
  const geometry = feature.geometry;
  if (!geometry) return null;
  if (geometry.type === "Polygon") {
    return toPathForPolygon(geometry.coordinates);
  }
  if (geometry.type === "MultiPolygon") {
    return toPathForMultiPolygon(geometry.coordinates);
  }
  return null;
};

export default function RiskIndexMap({
  riskRows,
  countriesRows,
  view,
}: RiskIndexMapProps) {
  const [features, setFeatures] = useState<GeoFeature[]>([]);
  const [assetInvalid, setAssetInvalid] = useState(false);
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
        const normalized = normalizeFeatures(data.features ?? []);
        setAssetInvalid(normalized.invalid);
        setFeatures(normalized.features);
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

  const assetLooksSynthetic = useMemo(() => {
    if (!features.length) return false;
    const latitudes = new Set<number>();
    features.slice(0, 50).forEach((feature) => {
      collectOuterRingLatitudes(feature.geometry, latitudes);
    });
    return latitudes.size <= 50;
  }, [features]);

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

  const getFeatureIso3 = (feature: GeoFeature) => {
    const props = feature.properties ?? {};
    const iso3 = (props.iso3 ??
      props.ISO_A3 ??
      props["ISO_A3"] ??
      props.ADM0_A3 ??
      props["ADM0_A3"] ??
      props.adm0_a3 ??
      props["adm0_a3"] ??
      "") as string;
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
      {assetInvalid ? (
        <div className="mt-3 rounded-md border border-rose-500/60 bg-rose-500/10 px-3 py-2 text-xs text-rose-200">
          World map asset coordinates do not look like lon/lat degrees. Expected
          lon [-180..180], lat [-90..90].
        </div>
      ) : assetLooksSynthetic ? (
        <div className="mt-3 rounded-md border border-amber-500/50 bg-amber-500/10 px-3 py-2 text-xs text-amber-200">
          World map asset looks synthetic (grid-like). Replace
          web/public/maps/world-countries-iso3.geojson with real country polygons.
        </div>
      ) : null}
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
          {assetInvalid
            ? null
            : features.map((feature, index) => {
                const iso3 = getFeatureIso3(feature);
                const value = iso3 ? valueByIso3.get(iso3) : undefined;
                let fill = "var(--risk-map-no-questions)";
                if (typeof value === "number" && Number.isFinite(value)) {
                  const classIndex = classifyJenks(value, breaks);
                  fill = colorScale[classIndex] ?? "var(--risk-map-c1)";
                } else if (iso3 && hasQuestionsIso3.has(iso3)) {
                  fill = "var(--risk-map-no-eiv)";
                }
                const d = toPathForFeature(feature);
                if (!d) {
                  return null;
                }
                return (
                  <path
                    key={`${iso3 || "country"}-${index}`}
                    d={d}
                    fill={fill}
                    stroke="var(--risk-map-stroke)"
                    strokeWidth={0.6}
                    vectorEffect="non-scaling-stroke"
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
