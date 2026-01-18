# Pythia Web

Minimal Next.js dashboard scaffold for the Pythia API.

## Pages
- **Forecast Index (Overview)**: Humanitarian Impact Forecast Index with RC KPI counts and RC map markers (highest RC level per country).
- **Forecasts** (`/questions`): Latest forecast table, including RC score/triage fields from `/v1/questions?latest_only=true`.
- **HS Triage** (`/hs-triage`): Run-level triage table with RC likelihood/direction/magnitude/score columns.
- **Countries** (`/countries`): Highest RC level/score per country from `/v1/countries`.
- **Downloads** (`/downloads`): Forecast SPD/EIV export with RC columns plus triage/cost exports.

## Setup

```bash
cd web
npm install
cp .env.example .env.local
```

Edit `.env.local` with your API base URL.

## Development

```bash
npm run dev
```

Open http://localhost:3000.

## Build & Lint

```bash
npm run build
npm run lint
```

## Maps

The overview map is generated from Natural Earth admin-0 country boundaries
(110m recommended). To refresh the assets:

1. Replace `web/public/maps/world-countries-iso3.geojson` with a Natural Earth
   admin-0 GeoJSON that includes ISO3 properties (`iso3`, `ISO_A3`, or
   `ADM0_A3`).
2. Run `python scripts/maps/build_world_svg.py`.
3. Commit both `world-countries-iso3.geojson` and the regenerated
   `world.svg`.

Web UI guardrails and resolver sanity tests will fail if the map asset is not
real geography.
